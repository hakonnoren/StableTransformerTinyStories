"""Distributed (DDP) helpers for train.py.

Design rule: **nothing here changes single-process behaviour.** ``init_distributed``
returns a disabled state unless the process was launched by ``torchrun`` (i.e. RANK /
WORLD_SIZE / LOCAL_RANK are in the environment), and every other helper degrades to a
no-op when disabled. Running ``python train.py ...`` exactly as before takes the same
code path it always did.

Launch:  torchrun --standalone --nproc_per_node=8 train.py <the usual args>

Convention for batch arguments (this is the part worth internalising):
``--batch_size`` and ``--grad_accum_steps`` describe the **global** optimizer step,
not the per-rank one. Under N ranks each rank runs ``grad_accum_steps // N``
micro-steps of ``batch_size`` sequences, so tokens/step is independent of how many
GPUs you use, and the token budget printed in the job files stays valid. This also
makes the data order *identical* to a single-GPU run -- see BlockEpochIterator in
data.py, which hands rank r the blocks that a 1-GPU run would have consumed at
micro-step ``m * N + r``.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass
class DDPState:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cuda"
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device_arg: str) -> DDPState:
    """Join the process group if launched under torchrun; otherwise stay single-process."""
    env = os.environ
    if not all(k in env for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return DDPState(enabled=False, device=device_arg)

    rank = int(env["RANK"])
    local_rank = int(env["LOCAL_RANK"])
    world_size = int(env["WORLD_SIZE"])

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available (distributed launch).")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = device_arg
        backend = "gloo"

    # device_id pins this rank to its GPU up front. Without it NCCL warns
    # "Guessing device ID based on global rank. This can cause a hang if rank to
    # GPU mapping is heterogeneous", and collectives that carry no tensor (our
    # barrier) have to infer a device -- which is a real hang risk on an 8-GPU
    # node, not a cosmetic warning. Passing it also silences the barrier() warning.
    kwargs = {}
    if device.startswith("cuda"):
        kwargs["device_id"] = torch.device(device)
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **kwargs)
    except TypeError:                     # torch too old for device_id
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DDPState(enabled=True, rank=rank, local_rank=local_rank,
                    world_size=world_size, device=device, backend=backend)


def wrap(model: torch.nn.Module, ddp: DDPState, find_unused_parameters: bool = False):
    """Wrap for training. Returns the model unchanged when distribution is off.

    broadcast_buffers=False: the only buffers here are causal masks and frozen
    gamma/alpha, all identical across ranks by construction and never mutated in
    forward. Leaving it on would re-broadcast them at every forward for nothing.
    """
    if not ddp.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[ddp.local_rank] if ddp.device.startswith("cuda") else None,
        output_device=ddp.local_rank if ddp.device.startswith("cuda") else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        find_unused_parameters=find_unused_parameters,
    )


def all_reduce_mean(value: float, ddp: DDPState) -> float:
    """Average a python scalar across ranks. Identity when distribution is off."""
    if not ddp.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64,
                     device=ddp.device if ddp.device.startswith("cuda") else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / ddp.world_size)


def barrier(ddp: DDPState) -> None:
    if ddp.enabled:
        dist.barrier()


def shutdown(ddp: DDPState) -> None:
    if ddp.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def redirect_nonmain_stdout(ddp: DDPState, run_dir: str) -> None:
    """Send non-main ranks' stdout to a per-rank file.

    With 8 ranks the interleaved training log is unreadable and the CSV/W&B writes
    are main-only anyway. stderr is deliberately left alone so tracebacks from any
    rank still reach the SLURM .err file -- a silent worker is far worse than a
    noisy one when a run dies at hour 40.
    """
    if not ddp.enabled or ddp.is_main:
        return
    path = os.path.join(run_dir, f"rank{ddp.rank}.log")
    sys.stdout = open(path, "w", buffering=1)


def report_missing_grads(model: torch.nn.Module, ddp: DDPState, device: str,
                         block_size: int, vocab_size: int, amp_dtype) -> list:
    """One synthetic fwd/bwd to find parameters that receive no gradient.

    This exists because of a specific DDP failure mode that is otherwise invisible
    until it costs you a multi-day job. DDP installs a hook on every parameter's
    AccumulateGrad node and waits for all of them before it finishes the backward
    all-reduce. A parameter that is *reachable in the graph* but ends up with no
    gradient never fires its hook, and the job does not crash -- it **hangs** until
    the NCCL watchdog kills it. ``find_unused_parameters=True`` does not help,
    because that only covers parameters unreachable from the loss.

    The reversible stack is the realistic source of this: revformer/rev_backprop.py
    calls ``torch.autograd.grad(..., allow_unused=True)`` and leaves a ``None`` in
    the returned gradient tuple for any parameter that produced nothing.

    Returns the list of parameter names with no gradient. Restores RNG state and
    zeroes gradients afterwards so training starts from an untouched state.
    """
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None

    was_training = model.training
    model.train()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, vocab_size, (2, block_size), generator=g).to(device)
    tgt = torch.randint(0, vocab_size, (2, block_size), generator=g).to(device)

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=(amp_dtype != torch.float32)):
        _, loss = model(idx, tgt, global_step=0)
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return missing
