"""
The learned damping of each reversible checkpoint, per block and per channel, to
a small file next to the Jacobian records. CPU only, no GPU, seconds per arm.

    python -m analysis.extract_damping                                  # the paper's arms
    python -m analysis.extract_damping --ckpt_dir revparityregrevback_25000297 --out_dir results/x

The Jacobian records keep only the extremes of exp(-gamma), exp(-alpha) after
centering, which is all the bounds need; the per-channel values live only in the
checkpoints. Writes, per arm, <out_dir>/damping_<arm>.npz with (n_layer, n_embd)
arrays

    gamma_hat, alpha_hat   the learned damping BEFORE the regime's centering:
                           epsilon * tanh(c) with tanh scaling, else c. What the
                           paper's damping figure plots (gamma-hat, alpha-hat).
    gamma_eff, alpha_eff   AFTER centering, i.e. what the forward pass applies
                           (exp(-gamma_eff) = G-hat); at T=--seq_len, which only
                           matters for vpm_scaling with lambda != 0.

and the arm's regime, scaling and checkpoint path. vpb_baseline is written too,
with frozen=True and all zeros, so a missing file always means "not extracted".

Checkpoints are read with mmap: only the model tensors are paged in, never the
optimizer state that makes up most of each file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.loader import _model_cfg, build_model        # noqa: E402
from analysis.jacobian_spectrum import find_arms, volume_correction  # noqa: E402

PAPER_CKPTS = ["revparityreg_24979051/baseline_baseline", "revparityregrevback_25000297",
               "revparityregrevback_25001769"]


def damping(path: str, seq_len: int) -> dict | None:
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    a = ck.get("args", {})
    if a.get("arch") != "reversible":
        return None
    model = build_model(_model_cfg(ck["cfg"]), a)
    model.load_state_dict(ck["model"])
    model.eval()
    avg = volume_correction(model, seq_len)
    out = {k: [] for k in ("gamma_hat", "alpha_hat", "gamma_eff", "alpha_eff")}
    with torch.no_grad():
        for blk in model.blocks:
            g_eff, a_eff = blk._effective_gamma_alpha(avg)
            for k, v in (("gamma_hat", blk.get_gamma()), ("alpha_hat", blk.get_alpha()),
                         ("gamma_eff", g_eff), ("alpha_eff", a_eff)):
                out[k].append(v.reshape(-1).double().numpy())
    out = {k: np.stack(v) for k, v in out.items()}
    out["meta"] = {"ckpt": path, "regime": a.get("rev_regime"), "frozen": bool(model.blocks[0].frozen),
                   "tanh_scale": bool(a.get("rev_tanh", False)), "epsilon": a.get("rev_epsilon", 1.0),
                   "lambd": a.get("rev_lambda", 0.0), "seq_len": seq_len, "step": ck.get("step")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", nargs="+", default=PAPER_CKPTS,
                    help="sweep or run dirs, as for the Jacobian job (default: the paper's)")
    ap.add_argument("--out_dir", default="results/jacobian_spectrum_T16_paper")
    ap.add_argument("--seq_len", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name, path in find_arms(args.ckpt_dir).items():
        d = damping(path, args.seq_len)
        if d is None:
            print(f"  {name:32s} not reversible, skipped")
            continue
        f = os.path.join(args.out_dir, f"damping_{name}.npz")
        np.savez(f, meta=json.dumps(d.pop("meta") | {"arm": name}), **d)
        print(f"  {name:32s} -> {f}  gamma_hat in [{d['gamma_hat'].min():+.3f}, "
              f"{d['gamma_hat'].max():+.3f}], alpha_hat in [{d['alpha_hat'].min():+.3f}, "
              f"{d['alpha_hat'].max():+.3f}]")


if __name__ == "__main__":
    main()
