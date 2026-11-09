"""Where training puts the block Jacobian's spectrum, checkpoint by checkpoint, relative to the flip of the twin pair.

Test of the candidate mechanism for the twin's late-training decline (looped/dynlab/paper, sec. 5): the twins L, L'
of a twin_l model obey, at merged copies and fixed H, the recurrence
    w_{j+1} = (1-a)(c + mu) w_j + a w_{j-1}          (a = e^-gamma, c = e^-delta, mu an eigenvalue of dB)
whose copy-difference mode is neutral at mu = -1 - c (dynlab paper Prop. 3; Sudoku note Prop. 8). The read-out
cannot see that mode (H reads the twins' mean) and EqR's truncated gradient does not differentiate its evolution.
Prediction: over training the Ritz values of dB at the twins' force argument u = m + H + x (m = (L + L')/2)
approach mu = -1 - c, together with the copy split and the loss of test-time scaling. EqR itself is the control: its
L update z_L <- (1-lam) z_L + lam B(z_L + z_H + x) flips at mu = -(2 - lam)/lam = -1.105, and its read-out sees the
flip.

Per checkpoint (EMA weights, noise-free, the first --n test puzzles from EqR's random reset, --r-max iterations):
exact accuracy at r_max; copy separation |L - L'| / (|L| + |L'|) (twin); for --n-spec right and --n-spec wrong end
states the Ritz values of dB(u) (batched Arnoldi with forward-mode JVPs, float32, m vectors) and from them
    left      the leftmost real part
    dflip     the distance of the nearest Ritz value to mu_flip
    pair_rho  twin: the largest |lambda| of the pair recurrence over the Ritz values (per half-step; > 1 = split);
              EqR: the largest |1 - lam + lam mu| (its L self-map, per L update)
reported as medians over right / wrong end states, plus the share of states with a Ritz value within 0.15 of mu_flip.

    python looped/eqr/twin_spectrum.py --eqr-dir $EQR_DIR --run-root $RUN_ROOT --run-id twinL-leak2-s0-25939526 \
        [--steps 20000,25000,...] [--out twin_spectrum.json]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.attention  # noqa: F401  (sdpa_kernel for the JVPs)

HERE = os.path.dirname(os.path.abspath(__file__))


def checkpoints(run_root, run_id):
    paths = glob.glob(os.path.join(run_root, "outputs", run_id, "*", "*", "checkpoints", "step_*_*.pth"))
    by_step = {}
    for p in paths:
        by_step.setdefault(int(re.search(r"step_(\d+)_", os.path.basename(p)).group(1)), p)
    return [by_step[k] for k in sorted(by_step)]


def block_fn(inner, seq):
    """B(u): the shared layers alone (EqR's NoisyReasoningModule and the overlay's _Force both keep them in .layers)."""
    layers = inner.L_level.layers

    def B(u):
        for layer in layers:
            u = layer(hidden_states=u, **seq)
        return u
    return B


@torch.no_grad()
def ritz(B, u, m):
    """Ritz values [b, m] of v -> dB(u) v (forward-mode JVPs; modified Gram-Schmidt twice)."""
    b, N = u.shape[0], u[0].numel()
    V = torch.zeros(b, m + 1, N, device=u.device)
    H = torch.zeros(b, m + 1, m, dtype=torch.float64)
    v = torch.randn(b, N, device=u.device)
    V[:, 0] = v / v.norm(dim=1, keepdim=True)
    for j in range(m):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            _, w = torch.func.jvp(B, (u,), (V[:, j].view_as(u),))
        w = w.reshape(b, N).float()
        for _ in range(2):
            c = torch.einsum("bin,bn->bi", V[:, :j + 1], w)
            w = w - torch.einsum("bi,bin->bn", c, V[:, :j + 1])
            H[:, :j + 1, j] += c.double().cpu()
        nrm = w.norm(dim=1)
        H[:, j + 1, j] = nrm.double().cpu()
        V[:, j + 1] = w / nrm.clamp_min(1e-30).unsqueeze(1)
    return torch.linalg.eigvals(H[:, :m, :m]).numpy()


def pair_radius(mu, a, c):
    """max |lambda| of lambda^2 - (1-a)(c+mu) lambda - a = 0 over the Ritz values mu (one puzzle)."""
    p = (1 - a) * (c + mu)
    disc = np.sqrt(p * p + 4 * a + 0j)
    return float(np.max(np.maximum(np.abs((p + disc) / 2), np.abs((p - disc) / 2))))


def analyse_checkpoint(eqr_dir, ckpt, n, r_max, n_spec, m, device):
    sys.path.insert(0, HERE)
    from dynamics_eqr import _load, _set_noise
    from models.losses import IGNORE_LABEL_ID
    model, batch, config, ck = _load(eqr_dir, ckpt, n)
    _set_noise(model, 0.0)
    inner = model.model.inner
    twin = bool(getattr(inner, "twin_l", False))
    coupled = hasattr(inner, "rates")
    if coupled and not twin:
        return {"step": int(ck.get("step", -1)), "skipped": "coupled model without twin_l"}
    batch = {k: v.to(device) for k, v in batch.items()}
    b = batch["inputs"].shape[0]
    carry = inner.empty_carry(b, device=device)
    carry = inner.reset_carry(torch.ones(b, dtype=torch.bool, device=device), carry)
    with torch.no_grad():
        for _ in range(r_max):
            carry, logits, _ = inner(carry, batch)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    right = ((logits.argmax(-1) == labels) | ~mask).all(-1).cpu()
    out = {"step": int(ck.get("step", -1)), "exact": float(right.float().mean()), "n": b}
    zH, zL = carry.z_H.float(), carry.z_L.float()
    if twin:
        L, M = zL.chunk(2, dim=1)
        sep = ((L - M).flatten(1).norm(dim=1) / (L.flatten(1).norm(dim=1) + M.flatten(1).norm(dim=1)).clamp_min(1e-30)).cpu()
        out["copy_sep_right"] = float(sep[right].median()) if right.any() else None
        out["copy_sep_wrong"] = float(sep[~right].median()) if (~right).any() else None
        with torch.no_grad():
            a, c = float(torch.exp(-inner.rates.gamma()).mean()), inner.rates.c()
        c = 1.0 if c is None else float(c.mean())
        mu_flip = -1.0 - c
        out.update(a=a, c=c, read=getattr(inner, "twin_read", "mean"))
    else:
        lam = float(inner.L_level.lambda_)
        mu_flip = -(2.0 - lam) / lam
        out["lambda"] = lam
    out["mu_flip"] = mu_flip
    old_fd = inner.forward_dtype
    inner.forward_dtype = torch.float32
    try:
        x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]).float()
        seq = {"cos_sin": inner._cos_sin()}
        B = block_fn(inner, seq)
        u_all = (0.5 * (L + M) + zH + x) if twin else (zL + zH + x)
        for kind, sel in (("right", right), ("wrong", ~right)):
            idx = torch.nonzero(sel).flatten()[:n_spec]
            if len(idx) == 0:
                continue
            mu = ritz(B, u_all[idx.to(device)], m)
            left = mu.real.min(1)
            dflip = np.abs(mu - mu_flip).min(1)
            if twin:
                pr = np.array([pair_radius(mu[i], a, c) for i in range(len(mu))])
            else:
                pr = np.abs(1 - lam + lam * mu).max(1)
            out[kind] = {"n": int(len(idx)), "left": float(np.median(left)), "dflip": float(np.median(dflip)),
                         "near_flip": float((dflip < 0.15).mean()), "pair_rho": float(np.median(pr)),
                         "pair_rho_max": float(pr.max()), "absmax": float(np.median(np.abs(mu).max(1))),
                         "ritz_re": mu.real[:8].tolist(), "ritz_im": mu.imag[:8].tolist()}
    finally:
        inner.forward_dtype = old_fd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqr-dir", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", default=None, help="comma list of checkpoint steps (default: all)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--r-max", type=int, default=64)
    ap.add_argument("--n-spec", type=int, default=16)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_root = os.path.abspath(a.run_root)
    cks = checkpoints(run_root, a.run_id)
    if a.steps:
        keep = {int(s) for s in a.steps.split(",")}
        cks = [p for p in cks if int(re.search(r"step_(\d+)_", os.path.basename(p)).group(1)) in keep]
    if not cks:
        sys.exit(f"no checkpoints under {run_root}/outputs/{a.run_id}")
    out_path = a.out or os.path.join(run_root, f"twin_spectrum_{a.run_id}.json")
    done = {}
    if os.path.exists(out_path):
        done = {r["step"]: r for r in json.load(open(out_path))["rows"]}
    os.chdir(a.eqr_dir)                                   # EqR resolves data paths relative to its checkout
    sys.path.insert(0, os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = dict(done)
    for p in cks:
        step = int(re.search(r"step_(\d+)_", os.path.basename(p)).group(1))
        if step in rows:
            continue
        torch.manual_seed(0)
        r = analyse_checkpoint(a.eqr_dir, os.path.abspath(p), a.n, a.r_max, a.n_spec, a.m, device)
        r["step"] = step
        rows[step] = r
        f = lambda d, k: "--" if not d or d.get(k) is None else f"{d[k]:.3f}"
        print(f"[twin_spectrum] {a.run_id} step {step:6d}: exact {r.get('exact', float('nan')):.3f}  "
              f"copy_sep R/W {f(r, 'copy_sep_right')}/{f(r, 'copy_sep_wrong')}  mu_flip {r.get('mu_flip', float('nan')):.3f}  "
              f"left R/W {f(r.get('right'), 'left')}/{f(r.get('wrong'), 'left')}  "
              f"dflip R/W {f(r.get('right'), 'dflip')}/{f(r.get('wrong'), 'dflip')}  "
              f"near_flip R/W {f(r.get('right'), 'near_flip')}/{f(r.get('wrong'), 'near_flip')}  "
              f"pair_rho R/W {f(r.get('right'), 'pair_rho')}/{f(r.get('wrong'), 'pair_rho')}", flush=True)
        with open(out_path, "w") as fh:
            json.dump({"run_id": a.run_id, "rows": [rows[k] for k in sorted(rows)]}, fh, indent=1)
    print(f"[twin_spectrum] wrote {out_path}")


if __name__ == "__main__":
    main()
