"""Memory-efficient backpropagation for the reversible block stack.

The point of a reversible architecture: the activations entering block *l* can be
*recomputed* from the activations leaving it, so they never have to be stored.
This module runs the whole stack under ``torch.no_grad()`` -- keeping only the
final output -- and reconstructs everything during the backward pass:

    for l = n-1 ... 0:
        y_l  = block_l.inverse(y_{l+1})          # recover the input, no storage
        recompute block_l(y_l) with grad enabled # one block's graph at a time
        dL/dy_l, dL/dparams_l = autograd.grad(...)

Activation memory for the stack becomes **independent of n_layer** (one block's
intermediates at a time) at the cost of roughly 1.6x the stack's forward compute:
per block the backward pass does one inverse plus one recomputed forward on top of
the gradient computation itself. So this is a lever for fitting models that
otherwise would not fit (more depth, wider micro-batch, longer context), not a
speedup at a size that already fits.

Enable it with ``model.mem_efficient = True`` (see RevFormerModel). Supported for
``ReversibleBlock`` only; any other block type falls back to ordinary autograd.

Four things make this correct rather than merely plausible:

* **The inverse is algebraic, not iterative** -- see ``ReversibleBlock.inverse``.
* **``avg`` is a differentiable input, not a constant.** Under vpm_scaling every
  block's effective gamma/alpha is offset by the mean over *all* blocks
  (``RevFormerModel._avg_corr``), so dL/d(avg) must be accumulated across the
  whole stack and returned; autograd then routes it back to every gamma_l,
  alpha_l. A per-block custom Function cannot see this coupling and gets it wrong.
* **Autocast state is captured and replayed.** The forward runs under whatever
  autocast context the caller established (bf16 in training); a backward pass runs
  with autocast OFF. Recomputing in fp32 what was computed in bf16 would silently
  differentiate a slightly different function, so the dtype is recorded in forward
  and re-entered in backward.
* **Dropout is refused rather than mishandled.** The forward evaluates attn then
  mlp; the inverse evaluates mlp then attn. Replaying a stateful RNG in reversed
  order would hand each call site the other's mask -- silently wrong gradients
  with no error. Rather than build call-site-keyed masks for a feature no config
  here uses (dropout is 0.0 everywhere), active dropout inside the stack is a hard
  error. See _check_dropout_replayable.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Eligibility + guards
# ---------------------------------------------------------------------------
def stack_supported(blocks) -> bool:
    """True when every block implements the two-stream coupling we can invert.

    Deliberately excludes LinearMixedReversibleBlock (and hence the lowrank-Cayley
    map): it does define an algebraic inverse, but the conditioning of that
    inverse is a separate question from the diagonal case's and has not been
    validated here. Excludes the damped_mem regime for a subtler reason -- see
    _block_params.
    """
    from revformer.revformer import ReversibleBlock                  # noqa: PLC0415
    blocks = list(blocks)
    return bool(blocks) and all(
        type(b) is ReversibleBlock and getattr(b, "inverse", None) is not None
        for b in blocks
    )


def _check_dropout_replayable(blocks) -> None:
    """Fail loudly if any dropout inside the stack is active.

    Getting this wrong does not crash -- it silently corrupts gradients, because
    the inverse would subtract a differently-masked sub-layer output than the
    forward added. So it is checked rather than assumed.
    """
    for b in blocks:
        for m in b.modules():
            if isinstance(m, nn.Dropout) and m.training and m.p > 0.0:
                raise RuntimeError(
                    f"{type(m).__name__}(p={m.p}) is active inside the reversible "
                    "stack; its mask cannot be reproduced during the backward "
                    "recomputation, which would corrupt gradients silently. Train "
                    "with --dropout 0 or set model.mem_efficient = False.")


def _block_params(block):
    """Trainable parameters that block's forward actually depends on.

    NOT simply block.parameters(): under the damped_mem regime the memory gate is
    owned by RevFormerModel and handed to each block in a list wrapper precisely so
    it is registered once, which means it is absent from block.parameters(). It
    still enters the block's forward, so leaving it out here would drop its
    gradient with no error at all. stack_supported currently excludes damped_mem,
    but this stays correct if that changes.
    """
    ps = [p for p in block.parameters() if p.requires_grad]
    gate = getattr(block, "_mem_gate", [None])[0]
    if gate is not None and gate.requires_grad and not any(p is gate for p in ps):
        ps.append(gate)
    return ps


def _collect_params(blocks):
    """Flat list of the stack's trainable parameters plus per-block index slices.

    Deduplicated by identity so a parameter shared between blocks (the damped_mem
    gate) gets a single gradient slot that every block accumulates into.
    """
    params, index, seen = [], [], {}
    for b in blocks:
        idx = []
        for p in _block_params(b):
            slot = seen.get(id(p))
            if slot is None:
                slot = seen[id(p)] = len(params)
                params.append(p)
            idx.append(slot)
        index.append(tuple(idx))
    return params, tuple(index)


def _autocast_state(device_type: str):
    """(enabled, dtype) for the ambient autocast context on this device type."""
    try:                                      # torch >= 2.4 takes device_type
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:                         # older signature: CUDA only
        enabled = torch.is_autocast_enabled() and device_type == "cuda"
    dtype = torch.get_autocast_dtype(device_type) if enabled else None
    return bool(enabled), dtype


# ---------------------------------------------------------------------------
# The stack itself
# ---------------------------------------------------------------------------
def run_reversible_stack(blocks, y0, avg, segment: int = 0):
    """Apply ``blocks`` to ``y0`` storing no intermediate activations.

    Args:
        blocks: the reversible block stack (model.blocks).
        y0: stack input, (B, T, 2*n_embd).
        avg: volume correction passed to every block -- a tensor under
            vpm_scaling (and then differentiated through), or a float (0.0) in the
            frozen / per-block / free regimes.
        segment: if > 0, store the stack activation every ``segment`` blocks and
            invert only within a segment. Caps reconstruction-error growth at
            ``segment`` blocks; memory becomes ceil(n_layer/segment) tensors of
            size (B, T, 2*n_embd) -- still far below the per-block internals. 0
            (default) is fully reversible: nothing stored.
    """
    blocks = list(blocks)
    if not stack_supported(blocks):
        raise TypeError(
            "mem-efficient reversible backprop supports only ReversibleBlock; got "
            f"{sorted({type(b).__name__ for b in blocks})}")
    _check_dropout_replayable(blocks)

    avg_t = avg if torch.is_tensor(avg) else None
    avg_c = 0.0 if avg_t is not None else float(avg)
    params, index = _collect_params(blocks)
    return _RevStack.apply(y0, avg_t, blocks, avg_c, int(segment), index, *params)


class _RevStack(torch.autograd.Function):
    """One Function for the *whole* stack, not one per block.

    Per-block Functions each have to save their own output for backward, so all
    n_layer outputs stay alive and nothing is actually saved -- the usual
    workaround is to hand-deallocate with tensor.storage().resize_(0). A single
    Function over the stack saves exactly one tensor (the stack output, which ln_f
    retains anyway => zero marginal cost) and holds one block's intermediates at a
    time in backward.
    """

    @staticmethod
    def forward(ctx, y0, avg_t, blocks, avg_c, segment, index, *params):
        ctx.blocks, ctx.segment, ctx.index = blocks, segment, index
        ctx.avg_c, ctx.n_params = avg_c, len(params)
        ctx.has_avg = avg_t is not None
        # Captured here and re-entered in backward: a backward pass runs with
        # autocast off, and recomputing in fp32 what the forward computed in bf16
        # would differentiate a subtly different function.
        ctx.device_type = y0.device.type
        ctx.ac_enabled, ctx.ac_dtype = _autocast_state(ctx.device_type)

        avg = avg_t if ctx.has_avg else avg_c
        ckpts = []
        with torch.no_grad():
            y = y0
            for i, b in enumerate(blocks):
                if segment > 0 and i % segment == 0:
                    ckpts.append(y)
                y = b(y, avg)
        # Saved: the stack output, the (differentiable) correction, and any segment
        # boundaries. Note the per-block internals -- attention matrices, 4d MLP
        # hiddens -- were freed as the no_grad loop went.
        ctx.save_for_backward(*([y] + ([avg_t] if ctx.has_avg else []) + ckpts))
        return y

    @staticmethod
    def backward(ctx, dy):
        # A normal backward runs with grad mode off; it is on only when the caller
        # asked for create_graph=True, which we cannot honour (the recomputation
        # builds a fresh graph per block and discards it).
        if torch.is_grad_enabled():
            raise RuntimeError(
                "mem-efficient reversible backprop does not support "
                "create_graph=True / double backward; set model.mem_efficient = "
                "False for higher-order gradients.")

        saved = list(ctx.saved_tensors)
        y = saved.pop(0)
        if ctx.has_avg:
            # A leaf, so autograd.grad can target it; its accumulated gradient is
            # returned below and routed back to every block's gamma/alpha.
            avg = saved.pop(0).detach().requires_grad_(True)
        else:
            avg = ctx.avg_c
        ckpts, blocks, index, seg = saved, ctx.blocks, ctx.index, ctx.segment

        grads = [None] * ctx.n_params
        d_avg = None

        # cache_enabled=False is REQUIRED, not a tuning knob. Autocast caches the
        # bf16 cast of each weight; b.inverse() below runs under no_grad, so its
        # casts are graph-free, and the recomputed forward would then reuse those
        # cached tensors and lose the edge back to the fp32 parameter. The symptom
        # is silent: autograd.grad returns None for exactly the Linear weights
        # (c_attn, c_proj, fc1, fc2) while LayerNorm and gamma/alpha look fine.
        with torch.autocast(device_type=ctx.device_type, dtype=ctx.ac_dtype,
                            enabled=ctx.ac_enabled, cache_enabled=False):
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                if seg > 0 and i % seg == 0:
                    y_in = ckpts[i // seg]      # exact: resets accumulated drift
                else:
                    with torch.no_grad():
                        y_in = b.inverse(y, avg)
                with torch.enable_grad():
                    y_in = y_in.detach().requires_grad_(True)
                    # b(...) not b.forward(...): goes through __call__ so module
                    # forward/backward hooks still fire (cheap_metrics'
                    # GradNormTracker registers full backward hooks on the blocks).
                    y_out = b(y_in, avg)
                    bparams = _block_params(b)
                    assert len(bparams) == len(index[i]), \
                        "block parameter set changed between forward and backward"
                    ins = (y_in,) + ((avg,) if ctx.has_avg else ()) + tuple(bparams)
                    g = torch.autograd.grad(y_out, ins, dy, allow_unused=True)
                dy = g[0]
                off = 1
                if ctx.has_avg:
                    if g[1] is not None:
                        d_avg = g[1] if d_avg is None else d_avg + g[1]
                    off = 2
                for slot, gp in zip(index[i], g[off:]):
                    if gp is None:
                        continue
                    grads[slot] = gp if grads[slot] is None else grads[slot] + gp
                y = y_in.detach()

        # (y0, avg_t, blocks, avg_c, segment, index, *params)
        return (dy, d_avg, None, None, None, None, *grads)


# ---------------------------------------------------------------------------
# Diagnostic: how exact is the reconstruction, really?
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_drift(model, idx):
    """Forward the stack keeping every activation, invert from the top, compare.

    Reconstruction error compounds down the stack at a rate set by the
    conditioning of the inverse map -- and, critically, by the precision it runs
    in. bf16 carries 8 mantissa bits and the inverse is subtractive, so this is
    the measurement that decides whether the fully-reversible path (segment=0) is
    usable under autocast, or whether a small segment is required.

    Runs under whatever autocast context the caller established, so call it inside
    the same one training uses to get a representative number.

    Returns {'recon_drift': (n_layer+1,) ndarray, 'recon_drift_max': float} where
    entry l is max|y_hat_l - y_l| / max|y_l| for the activation entering block l;
    entry n_layer is the stack output, exact by construction (0.0).
    """
    blocks = list(model.blocks)
    if not stack_supported(blocks):
        raise TypeError("reconstruction_drift needs a ReversibleBlock stack")
    _check_dropout_replayable(blocks)

    y0, avg = model.stack_input(idx)
    acts = [y0]
    for b in blocks:
        acts.append(b(acts[-1], avg))
    y = acts[-1]
    errs = [0.0]                                   # the top is exact
    for i in range(len(blocks) - 1, -1, -1):
        y = blocks[i].inverse(y, avg)
        ref = acts[i]
        scale = float(ref.float().abs().max()) + 1e-12
        errs.append(float((y - ref).float().abs().max()) / scale)
    errs.reverse()                                 # index l -> input of block l
    return {"recon_drift": np.array(errs, dtype=np.float64),
            "recon_drift_max": float(max(errs))}
