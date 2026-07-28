"""
Tiny smoke test: can we actually train the reversible model?

Runs one epoch of real training (real DataConfig/BlockEpochIterator data
iterator + train.py's build_optimizer) on a tiny, *learnable* synthetic token
stream, with tiny dims (2 layers), for:

  * the vanilla baseline GPTModel  -- the reference arm, and
  * all four volume regimes x both embedding widths.

Every arm shares the task, optimizer, seed and data order, so the printed
"vs base" params ratio and loss delta are directly comparable. Checks that the
loss drops, params update, and generation runs.

Also checks that --rev_embed narrow really is parameter-matched to the baseline
GPTModel (to within the 2*n_embd*n_layer gamma/alpha vectors), which is the
whole point of that knob.

The loss numbers here are a smoke signal, not evidence about the architecture:
one epoch on a 17-token cycle at n_embd=16 says nothing about TinyStories.

Run as a module:   python -m revformer.test_train_reversible
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from model import ModelConfig, GPTModel
from data import DataConfig, BlockEpochIterator
from train import build_optimizer
from revformer import RevFormerModel, RevConfig

REGIMES = ["vpb_baseline", "vpb_scaling", "vpm_scaling", "vf_scaling"]
EMBEDS = ["wide", "narrow"]

# Tiny, deterministic, *learnable* task: token k is always followed by (k+1)%P,
# so next-token prediction is fully determined and the loss should fall sharply
# if training works at all.
PERIOD = 17
VOCAB = 64
BLOCK_SIZE = 16
N_TOKENS = 4096


def make_tokens() -> np.ndarray:
    return (np.arange(N_TOKENS, dtype=np.int64) % PERIOD).astype(np.uint16)


N_LAYER = 2
N_EMBD = 16


def make_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=2,
        n_embd=N_EMBD,
        dropout=0.0,
        bias=False,
    )


def n_params(model) -> int:
    """Total params, counting tied tensors (lm_head/tok_emb) once."""
    seen = set()
    tot = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        tot += p.numel()
    return tot


def train_one_epoch(regime: str, embed: str = "wide", arch: str = "reversible",
                    device: str = "cpu", seed: int = 0):
    """One epoch on the toy task. arch='baseline' trains the vanilla GPTModel
    (regime/embed ignored) so the reversible numbers have a reference trained
    under the identical data order, optimizer and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = make_cfg()
    if arch == "baseline":
        model = GPTModel(cfg).to(device)
    else:
        rev_cfg = RevConfig(regime=regime, embed=embed,
                            lambd=(0.0 if regime == "vpm_scaling" else 0.0))
        model = RevFormerModel(cfg, rev_cfg=rev_cfg).to(device)
    model.train()

    opt = build_optimizer(model, peak_lr=1e-2, betas=(0.9, 0.95), scalar_lr_mult=10.0)

    tokens = make_tokens()
    dcfg = DataConfig(block_size=BLOCK_SIZE, batch_size=8, grad_accum_steps=1, seed=seed, device=device)
    it = BlockEpochIterator(tokens, dcfg, split="train")

    # snapshot a trainable param to confirm it actually updates
    watch_name, watch_before = None, None
    for n, p in model.named_parameters():
        if p.requires_grad:
            watch_name, watch_before = n, p.detach().clone()
            break

    start_epoch = it.epoch          # 1 right after construction
    first_loss, last_loss, n_steps = None, None, 0
    while it.epoch == start_epoch:
        xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss
        n_steps += 1
        if n_steps > 1000:           # safety guard
            break

    # checks
    assert n_steps > 0, "no training steps ran"
    assert np.isfinite(first_loss) and np.isfinite(last_loss), "non-finite loss"
    watch_after = dict(model.named_parameters())[watch_name].detach()
    moved = float((watch_after - watch_before).abs().max())
    assert moved > 0, f"param {watch_name} did not update"

    # gamma/alpha should carry gradients for the scaling regimes (frozen otherwise)
    blk = model.blocks[0]
    if arch == "baseline":
        assert not hasattr(blk, "gamma_bias"), "baseline block should have no gamma/alpha"
    elif regime == "vpb_baseline":
        assert blk.frozen, "vpb_baseline block should be frozen"
        gamma_trainable = isinstance(blk.gamma_bias, torch.nn.Parameter)
        assert not gamma_trainable, "vpb_baseline gamma should be a buffer, not trainable"
    else:
        assert blk.gamma_bias.grad is not None, f"{regime}: gamma got no grad"

    # generation runs
    model.eval()
    prompt = torch.tensor([[1, 2, 3]], device=device)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, do_sample=False)
    assert out.shape[1] == prompt.shape[1] + 5

    return dict(arch=arch, regime=regime, embed=embed, steps=n_steps, first=first_loss,
                last=last_loss, param_moved=moved, n_params=n_params(model))


def check_param_parity() -> bool:
    """embed='narrow' must match the baseline GPTModel's total params, up to the
    per-block gamma/alpha vectors (2*n_embd each); embed='wide' must not."""
    cfg = make_cfg()
    base = n_params(GPTModel(cfg))
    ok = True
    print(f"\nparam parity vs. baseline GPTModel ({base:,} params, "
          f"n_layer={N_LAYER} n_embd={N_EMBD} vocab={VOCAB} block={BLOCK_SIZE}):")
    for regime in REGIMES:
        # frozen regime keeps gamma/alpha as buffers -> exact match expected
        slack = 0 if regime == "vpb_baseline" else 2 * N_EMBD * N_LAYER
        narrow = n_params(RevFormerModel(cfg, rev_cfg=RevConfig(regime=regime, embed="narrow")))
        wide = n_params(RevFormerModel(cfg, rev_cfg=RevConfig(regime=regime, embed="wide")))
        good = (narrow - base == slack) and (wide > base)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL':4s}] {regime:13s} "
              f"narrow {narrow:,} (+{narrow - base}, allowed +{slack}) | "
              f"wide {wide:,} ({wide / base:.2f}x)")
    return ok


def _row(flag: str, label: str, r: dict, base: dict = None) -> str:
    """One result line; `base` (the trained baseline) adds a params/loss delta."""
    rel = ""
    if base is not None:
        rel = (f"  | vs base: {r['n_params'] / base['n_params']:.3f}x params, "
               f"loss {r['last'] - base['last']:+.3f}")
    return (f"[{flag:7s}] {label:31s} steps={r['steps']:3d}  "
            f"loss {r['first']:.3f} -> {r['last']:.3f}  "
            f"({r['n_params']:,} params){rel}")


def main():
    print(f"learnable task: token k -> (k+1)%{PERIOD}; chance loss ~= ln({PERIOD}) = {np.log(PERIOD):.3f}")
    print(f"dims: n_layer={N_LAYER} n_embd={N_EMBD} vocab={VOCAB} block={BLOCK_SIZE}\n")
    all_ok = True

    # Reference: the vanilla baseline, same task/optimizer/seed/data order.
    base = None
    try:
        base = train_one_epoch(regime=None, arch="baseline")
        ok = base["last"] < base["first"]
        all_ok = all_ok and ok
        print(_row("OK" if ok else "NO-DROP", "baseline (GPTModel)", base))
    except Exception as e:  # noqa: BLE001
        all_ok = False
        print(f"[FAIL   ] {'baseline (GPTModel)':31s} {type(e).__name__}: {e}")

    for embed in EMBEDS:
        for regime in REGIMES:
            label = f"rev {regime} / {embed}"
            try:
                r = train_one_epoch(regime, embed=embed)
                ok = r["last"] < r["first"]
                all_ok = all_ok and ok
                print(_row("OK" if ok else "NO-DROP", label, r, base))
            except Exception as e:  # noqa: BLE001
                all_ok = False
                print(f"[FAIL   ] {label:31s} {type(e).__name__}: {e}")

    all_ok = check_param_parity() and all_ok
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
