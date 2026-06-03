"""
Tiny smoke test: can we actually train the reversible model?

Runs one epoch of real training (real DataConfig/BlockEpochIterator data
iterator + train.py's build_optimizer) on a tiny, *learnable* synthetic token
stream, for all four volume regimes, with tiny dims (2 layers). Checks that the
loss drops, params update, and generation runs.

Run directly:   python revformer/test_train_reversible.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from model import ModelConfig
from data import DataConfig, BlockEpochIterator
from train import build_optimizer
from revformer import RevFormerModel, RevConfig

REGIMES = ["vpb_baseline", "vpb_scaling", "vpm_scaling", "vf_scaling"]

# Tiny, deterministic, *learnable* task: token k is always followed by (k+1)%P,
# so next-token prediction is fully determined and the loss should fall sharply
# if training works at all.
PERIOD = 17
VOCAB = 64
BLOCK_SIZE = 16
N_TOKENS = 4096


def make_tokens() -> np.ndarray:
    return (np.arange(N_TOKENS, dtype=np.int64) % PERIOD).astype(np.uint16)


def train_one_epoch(regime: str, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ModelConfig(
        vocab_size=VOCAB,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
    )
    rev_cfg = RevConfig(regime=regime, lambd=(0.0 if regime == "vpm_scaling" else 0.0))
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
    if regime == "vpb_baseline":
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

    return dict(regime=regime, steps=n_steps, first=first_loss, last=last_loss, param_moved=moved)


def main():
    print(f"learnable task: token k -> (k+1)%{PERIOD}; chance loss ~= ln({PERIOD}) = {np.log(PERIOD):.3f}")
    all_ok = True
    for regime in REGIMES:
        try:
            r = train_one_epoch(regime)
            dropped = r["last"] < r["first"]
            ok = dropped
            flag = "OK" if ok else "NO-DROP"
            all_ok = all_ok and ok
            print(f"[{flag:7s}] {regime:13s} steps={r['steps']:3d}  "
                  f"loss {r['first']:.3f} -> {r['last']:.3f}  (|Δparam|={r['param_moved']:.2e})")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"[FAIL   ] {regime:13s} {type(e).__name__}: {e}")
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
