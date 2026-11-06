"""Frozen-checkpoint dynamics of trained looped models (Stage B of the theory handoff): where does the state go
beyond the training horizon, and why do answers change?

For each checkpoint, on 256 test sequences, at r in R (up to 256 iterations):
  acc       accuracy on the answer positions
  flips     fraction of answers that differ from those at the previous recorded r
  res       ||s_r - s_{r-1}|| (full state, per sequence)
  res_q     same in the quotient by per-token stream shifts (P = model.center; LayerNorm makes the shifts invisible
            to every branch and to the readout, so for relax and the standard block the common shift is an exact
            neutral direction)
  norm_q    ||P s_r||;   shift  ||(I-P) s_r||
  inj       mean over tokens of ||P e|| / ||P(H + e)||: the weight of the injected input inside the read-stream
            force argument LN(H + e) (washout hypothesis: falls like 1/r if the shift-free state grows)
  rho_q     median spectral radius of P J P at s_r (power iteration, 64 sequences), J = d s_{r+1} / d s_r

    python looped/dynamics.py results/looped/khop_K3 results/looped/listops   # -> <dir>/dynamics.json, .md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "looped"))
from looped.model import LoopedConfig, LoopedGPT     # noqa: E402
import run_looped                                                          # noqa: E402

R = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RHO_AT = (8, 128)


def load(pt):
    ck = torch.load(pt, map_location="cpu")
    cfg = LoopedConfig(**ck["config"])
    m = LoopedGPT(cfg)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["args"]


def task_data(args, cache={}):
    ns = argparse.Namespace(**args)
    key = (args["task"], json.dumps({k: args.get(k) for k in sorted(args) if k not in ("arch", "out", "tag", "seed")}))
    if key not in cache:
        cache[key] = run_looped.TASKS[args["task"]](ns)
    return cache[key]


@torch.no_grad()
def analyse(model, x, y, n=256):
    x = torch.as_tensor(x[:n], dtype=torch.long); y = torch.as_tensor(y[:n], dtype=torch.long)
    mask = y != -1
    s, e = model.embed(x)
    nrm = lambda t: t.flatten(1).norm(dim=1)
    d = model.d
    out, prev_pred = {}, None
    for r in range(1, max(R) + 1):
        s_new = model.step(s, e)
        if r in R:
            logits = model.readout(s_new)
            pred = logits.argmax(-1)
            ok = (pred == y) & torch.isfinite(logits).all(-1)
            cq, cq_prev = model.center(s_new), model.center(s)
            Hs = s_new[..., d:] if model.two_stream else s_new
            Pe = e - e.mean(-1, keepdim=True)
            PHe = (Hs + e) - (Hs + e).mean(-1, keepdim=True)
            inj = (Pe.norm(dim=-1) / PHe.norm(dim=-1).clamp_min(1e-12)).mean()
            out[r] = dict(acc=float(ok[mask].float().mean()),
                          flips=float((pred != prev_pred)[mask].float().mean()) if prev_pred is not None else float("nan"),
                          res=float(nrm(s_new - s).mean()), res_q=float(nrm(cq - cq_prev).mean()),
                          norm_q=float(nrm(cq).mean()), shift=float(nrm(s_new - cq).mean()), inj=float(inj))
            prev_pred = pred
        s = s_new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["results/looped/khop_K3", "results/looped/listops"])
    ap.add_argument("--only", nargs="*", default=None, help="checkpoint stems (name_seedS); merged into dynamics.json")
    ap.add_argument("--threads", type=int, default=4)
    cli = ap.parse_args()
    torch.set_num_threads(cli.threads)
    for d in cli.dirs:
        d = ROOT / d
        old = d / "dynamics.json"
        res = json.loads(old.read_text()) if (cli.only and old.exists()) else {}
        for pt in sorted(d.glob("*_seed*.pt")):
            if cli.only and pt.stem not in cli.only:
                continue
            name = pt.stem.rsplit("_seed", 1)[0] if pt.stem.endswith("_seed1238") else pt.stem
            model, args = load(pt)
            data = task_data(args)
            x, y, _ = data["tests"]["test"]
            dyn = analyse(model, x, y)
            xs = torch.as_tensor(x[:64], dtype=torch.long)
            for r in RHO_AT:
                rq = model.step_spectral_radius(xs, r, iters=40, quotient=True).numpy()
                rf = model.step_spectral_radius(xs, r, iters=40, quotient=False).numpy()
                dyn[r]["rho_q"] = float(np.nanmedian(rq)); dyn[r]["rho"] = float(np.nanmedian(rf))
            res[name] = {str(k): v for k, v in dyn.items()}
            print(f"{d.name}/{name}: acc@128 {dyn[128]['acc']:.3f}  res {dyn[128]['res']:.3g} res_q {dyn[128]['res_q']:.3g}"
                  f"  norm_q {dyn[8]['norm_q']:.3g}->{dyn[256]['norm_q']:.3g}  shift {dyn[8]['shift']:.3g}->{dyn[256]['shift']:.3g}"
                  f"  inj {dyn[8]['inj']:.3f}->{dyn[256]['inj']:.3f}  rho_q@8 {dyn[8]['rho_q']:.3f} @128 {dyn[128]['rho_q']:.3f}"
                  f" (full {dyn[128]['rho']:.3f})", flush=True)
        (d / "dynamics.json").write_text(json.dumps(res, indent=1))
        lines = [f"# Frozen-checkpoint dynamics: {d.name}\n", "Columns at r = 8 → 32 → 128 → 256 (training r ends at 8).\n",
                 "| model | acc | flips (vs previous recorded r) | res_q (shift-free step) | res (full step) | norm_q | shift norm | inj | rho_q @8, @128 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        f = lambda m, k, fmt: " → ".join(format(res[m][str(r)][k], fmt) for r in (8, 32, 128, 256))
        for m in res:
            lines.append(f"| {m} | {f(m, 'acc', '.3f')} | {f(m, 'flips', '.3f')} | {f(m, 'res_q', '.2g')} | {f(m, 'res', '.2g')} | "
                         f"{f(m, 'norm_q', '.3g')} | {f(m, 'shift', '.3g')} | {f(m, 'inj', '.3f')} | "
                         f"{res[m]['8']['rho_q']:.3f}, {res[m]['128']['rho_q']:.3f} |")
        (d / "dynamics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
