"""Post-hoc dynamics of a trained EqR / coupled-EqR Sudoku checkpoint (looped/HANDOFF.md §5 metrics).

    python looped/eqr/dynamics_eqr.py --eqr-dir $EQR_DIR --ckpt <run>/checkpoints/step_20000_<id>.pth \
        [--out <run>/dynamics.json] [--n 768] [--r-max 256] [--n-rho 32]

Writes one JSON with three groups (all noise-free: the Langevin noise is switched off, so the
iteration is a deterministic map and "fixed point" is well defined):

1. rates_by_step  (coupled models) the damping actually learned, read from every checkpoint of the
   run (EMA weights when present, as evaluate.py uses): gamma / alpha (mean, min, max over channels),
   leak delta, and the radius of one outer iteration's LINEAR part (forces set to zero; product of
   the half-step maps on (L, H[, L']) over the schedule and H_cycles; < 1 means bounded forces give
   a bounded state). EqR's own model has no rates.

2. trajectory  the first --n test puzzles from EqR's random state reset, iterated to --r-max outer
   iterations (one iteration = one EqR ACT step). Per puzzle and iteration r:
       step_r  = |s_r - s_{r-1}| / |s_r|      s = concat(z_H, z_L)
       step2_r = |s_r - s_{r-2}| / |s_r|      (period-2 check)
   Reported: exact / cell accuracy at r in R_GRID; the fixed-point fraction (step_r < tol) and
   2-cycle fraction (step_r >= tol but step2_r < tol) at r = 64 and r_max for tol 1e-3 and 1e-2;
   step quantiles; state RMS; overthinking O(16->r) = P(wrong at r | exact at 16) and changed
   predictions dy(16->r); the halting head's "stop" rate (q > 0); for twin_l models the mean
   per-token cosine between the twins L and L' (1 = merged).

3. rho  spectral radius of one outer iteration's Jacobian at the final state s_{r_max}, for the first
   --n-rho puzzles: power iteration with Jacobian-vector products (torch.func.jvp), geometric mean of
   the last 20 of 60 growth factors, per puzzle (LoopedGPT.step_spectral_radius uses 10 of 30). Checked
   against dense-Jacobian eigenvalues on a tiny model: within 1% in 8 of 9 cases, 11% low in one where
   the top eigenvalues are close in modulus -- treat it as an estimate. Full radius,
   not the shift-free quotient: EqR's blocks are post-RMSNorm without mean subtraction, so they are
   not invariant to per-token shifts and the quotient is not an invariant subspace here. Computed
   with the forces in float32 at the trained weights.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from types import SimpleNamespace

import torch

R_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _load(eqr_dir, ckpt_path, n):
    """Model (EMA weights if present) and the first batch of n test puzzles, via evaluate.py's own loaders."""
    from evaluate import _base_config, _init_state, _load_checkpoint
    from evaluators.eval_config import EvalConfig
    from utils.checkpoint import load_ema, load_model_state_dict
    ckpt, path = _load_checkpoint(ckpt_path)
    config = _base_config(EvalConfig(global_batch_size=n), ckpt, path, rank=0)
    train_state, eval_loader, _ = _init_state(config, 0, 1)
    load_model_state_dict(train_state.model, ckpt.get("model", ckpt), strict=False, assign=True)
    if config.ema and "ema" in ckpt:
        train_state = load_ema(ckpt, train_state, config, 0)
    model = train_state.model
    model.eval()
    _, batch, _ = next(iter(eval_loader))
    return model, batch, config, ckpt


# ---------------------------------------------------------------------------------------- rates
def _arch_ns(arch):
    d = dict(coupling="relax", rate=3.0, rate_h=None, learn_rates=False, per_channel=False,
             rate_constraint="default", rate_min=0.0, rate_max=10.0, leak=0.0, learn_leak=False)
    d.update({k: v for k, v in arch.items() if k in d and v is not None})
    d["hidden_size"] = arch["hidden_size"]
    return SimpleNamespace(**d)


def _schedule(arch):
    """Half-steps of one outer iteration (the model's own schedule_of, repeated H_cycles times)."""
    from models.coupled_eqr import schedule_of
    return list(schedule_of(SimpleNamespace(**arch))) * int(arch["H_cycles"])


def linear_radius(g, a, c, kick, schedule, twin=False):
    """Spectral radius of one outer iteration with the forces set to zero, max over channels.
    State (L, H) per channel, plus the twin L' when twin; relax half-steps: L <- aL + (1-a)c*H
    (twin: c*L'), L' <- aL' + (1-a)c*L, H <- bH + (1-b)c*L (twin: c*(L + L')/2); kick: diagonal."""
    g, a = torch.as_tensor(g, dtype=torch.float64).flatten(), torch.as_tensor(a, dtype=torch.float64).flatten()
    d = max(g.numel(), a.numel())
    g, a = g.expand(d), a.expand(d)
    c = torch.ones(d, dtype=torch.float64) if c is None else torch.as_tensor(c, dtype=torch.float64).flatten().expand(d)
    kl, kh = torch.exp(-g), torch.exp(-a)
    n = 3 if twin else 2                                  # indices: L = 0, H = 1, L' = 2
    eye = torch.eye(n, dtype=torch.float64).expand(d, n, n).clone()
    ML, MH, MM = eye.clone(), eye.clone(), eye.clone()
    ML[:, 0, 0] = kl
    MH[:, 1, 1] = kh
    if twin:
        MM[:, 2, 2] = kl
    if not kick:
        if twin:
            ML[:, 0, 2] = (1 - kl) * c
            MM[:, 2, 0] = (1 - kl) * c
            MH[:, 1, 0] = MH[:, 1, 2] = 0.5 * (1 - kh) * c
        else:
            ML[:, 0, 1] = (1 - kl) * c
            MH[:, 1, 0] = (1 - kh) * c
    P = eye.clone()
    for s in schedule:
        P = {"L": ML, "H": MH, "M": MM}[s] @ P
    return float(torch.linalg.eigvals(P).abs().max())


def rates_of_checkpoint(path):
    from models.coupled_eqr import CouplingRates
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = (ck.get("config") or {}).get("arch", {})
    if "coupled_eqr" not in str(arch.get("name", "")):
        return None
    rates = CouplingRates(_arch_ns(arch))
    sources = [((ck.get("ema") or {}).get("shadow") or {}), ck.get("model", {})]
    for name in ("gamma_raw", "alpha_raw", "leak_raw"):
        if not hasattr(rates, name):
            continue
        for src in sources:
            hit = [v for k, v in src.items() if k.replace("_orig_mod.", "").endswith(f"inner.rates.{name}")]
            if hit:
                getattr(rates, name).data.copy_(hit[0].float())
                break
    with torch.no_grad():
        g, a, c = rates.gamma(), rates.alpha(), rates.c()
    stat = lambda t: {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}
    return {"step": int(ck.get("step", -1)), "gamma": stat(g), "alpha": stat(a),
            "delta": None if c is None else stat(-torch.log(c)),
            "keep_L": float(torch.exp(-g).mean()), "keep_H": float(torch.exp(-a).mean()),
            "linear_radius": linear_radius(g, a, c, arch.get("coupling") == "kick", _schedule(arch),
                                           twin=bool(arch.get("twin_l")))}


# ---------------------------------------------------------------------------------------- dynamics
def _set_noise(model, value):
    for m in model.modules():
        if hasattr(m, "noise_scale"):
            m.noise_scale = value


@torch.no_grad()
def trajectory(model, batch, r_max, device):
    from models.losses import IGNORE_LABEL_ID
    core = model.model
    inner = core.inner
    batch = {k: v.to(device) for k, v in batch.items()}
    b = batch["inputs"].shape[0]
    carry = inner.empty_carry(b, device=device)
    carry = inner.reset_carry(torch.ones(b, dtype=torch.bool, device=device), carry)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    flat = lambda c: torch.cat([c.z_H, c.z_L], dim=1).flatten(1).float()
    hist = [flat(carry)]
    steps, steps2, preds, out = {}, {}, {}, {"acc": {}, "exact": {}, "state_rms": {}, "q_stop": {}}
    twin = bool(getattr(inner, "twin_l", False))
    if twin:
        out["twin_cos"] = {}
    for r in range(1, r_max + 1):
        carry, logits, q = inner(carry, batch)
        s = flat(carry)
        nrm = s.norm(dim=1).clamp_min(1e-30)
        st = (s - hist[-1]).norm(dim=1) / nrm
        st2 = (s - hist[-2]).norm(dim=1) / nrm if len(hist) >= 2 else torch.full_like(st, float("nan"))
        hist = (hist + [s])[-2:]
        if r in R_GRID or r == r_max or r == 64:
            p = logits.argmax(-1)
            ok = (p == labels) | ~mask
            preds[r] = p
            steps[r], steps2[r] = st, st2
            out["acc"][r] = float(((p == labels) & mask).sum() / mask.sum())
            out["exact"][r] = float(ok.all(-1).float().mean())
            out["state_rms"][r] = float((s.pow(2).mean(1)).sqrt().mean())
            out["q_stop"][r] = float((q > 0).float().mean())
            if twin:
                l1, l2 = carry.z_L.float().chunk(2, dim=1)
                out["twin_cos"][r] = float(torch.nn.functional.cosine_similarity(l1, l2, dim=-1).mean())
    ok16 = ((preds[16] == labels) | ~mask).all(-1)
    out["overthink"], out["changed"] = {}, {}
    for r in (64, 128, 256, r_max):
        if r in preds and r > 16:
            okr = ((preds[r] == labels) | ~mask).all(-1)
            out["overthink"][r] = float((ok16 & ~okr).sum() / ok16.sum().clamp_min(1))
            out["changed"][r] = float((preds[r] != preds[16]).any(-1).float().mean())
    out["fixed_point"], out["cycle2"], out["step_quantiles"] = {}, {}, {}
    for r in sorted({64, r_max} & set(steps)):
        for tol in (1e-3, 1e-2):
            fp = steps[r] < tol
            out["fixed_point"][f"{r}@{tol:g}"] = float(fp.float().mean())
            out["cycle2"][f"{r}@{tol:g}"] = float(((~fp) & (steps2[r] < tol)).float().mean())
        q = torch.quantile(steps[r], torch.tensor([0.1, 0.5, 0.9], device=steps[r].device))
        out["step_quantiles"][r] = [float(v) for v in q]
    return out, carry


def spectral_radius(model, batch, carry, n_rho, device, iters=60, keep=20):
    from models.eqr import LatentCarry
    inner = model.model.inner
    batch = {k: v[:n_rho].to(device) for k, v in batch.items()}
    zH, zL = carry.z_H[:n_rho].float(), carry.z_L[:n_rho].float()
    old_fd = inner.forward_dtype
    old_sd = getattr(inner, "state_dtype", None)
    inner.forward_dtype = torch.float32                 # forces in float32 for the JVP
    if old_sd is not None:
        inner.state_dtype = torch.float32
    try:
        x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]).float()
        seq = {"cos_sin": inner._cos_sin()}
        shape = zH.shape

        n_h = zH.shape[1]                                 # z_L may be longer (twin_l: [L, L'])

        def f(s):
            h, l = s[:, :n_h], s[:, n_h:]
            for _ in range(inner.config.H_cycles):
                h, l = inner.latent_recursion(h, l, x, seq)
            return torch.cat([h, l], dim=1)

        s = torch.cat([zH, zL], dim=1)
        bcast = lambda t: t.view(-1, *([1] * (s.dim() - 1)))
        v = torch.randn_like(s)
        v = v / bcast(v.flatten(1).norm(dim=1))
        logs = []
        for k in range(iters):
            _, jv = torch.func.jvp(f, (s,), (v,))
            n = jv.flatten(1).norm(dim=1).clamp_min(1e-30)
            if k >= iters - keep:
                logs.append(torch.log(n))
            v = jv / bcast(n)
        rho = torch.exp(torch.stack(logs).mean(0)).detach().cpu()
    finally:
        inner.forward_dtype = old_fd
        if old_sd is not None:
            inner.state_dtype = old_sd
    return {"median": float(rho.median()), "max": float(rho.max()), "min": float(rho.min()),
            "frac_lt_1": float((rho < 1).float().mean()), "n": int(rho.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqr-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--r-max", type=int, default=256)
    ap.add_argument("--n-rho", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rates-only", action="store_true")
    a = ap.parse_args()

    ckpt_path = os.path.abspath(a.ckpt)
    os.chdir(a.eqr_dir)                                   # EqR resolves data paths relative to its checkout
    sys.path.insert(0, os.getcwd())
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    res = {"ckpt": ckpt_path}

    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pth")),
                 key=lambda p: int(re.search(r"step_(\d+)_", os.path.basename(p)).group(1)))
    rates = [x for x in (rates_of_checkpoint(p) for p in cks) if x is not None]
    res["rates_by_step"] = rates
    for x in rates:
        dl = "" if x["delta"] is None else f" delta {x['delta']['mean']:.3f}"
        print(f"[rates] step {x['step']:6d}: gamma {x['gamma']['mean']:.4f} [{x['gamma']['min']:.4f}, "
              f"{x['gamma']['max']:.4f}]  alpha {x['alpha']['mean']:.4f} [{x['alpha']['min']:.4f}, "
              f"{x['alpha']['max']:.4f}]{dl}  keep L/H {x['keep_L']:.3f}/{x['keep_H']:.3f}  "
              f"linear radius / iteration {x['linear_radius']:.4f}")

    if not a.rates_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(a.seed)
        model, batch, config, ck = _load(a.eqr_dir, ckpt_path, a.n)
        res["run_name"], res["step"] = config.run_name, int(ck.get("step", -1))
        _set_noise(model, 0.0)
        traj, carry = trajectory(model, batch, a.r_max, device)
        res["trajectory"] = traj
        res["rho"] = spectral_radius(model, batch, carry, a.n_rho, device)
        print(f"[dynamics] {config.run_name} step {res['step']}: exact " +
              " ".join(f"{r}:{traj['exact'][r]:.3f}" for r in R_GRID if r in traj["exact"]))
        print(f"[dynamics] fixed point {traj['fixed_point']}  2-cycle {traj['cycle2']}")
        print(f"[dynamics] step quantiles (10/50/90%) {traj['step_quantiles']}  state rms {traj['state_rms']}")
        print(f"[dynamics] overthink {traj['overthink']}  changed {traj['changed']}  stop-signal {traj['q_stop']}")
        if "twin_cos" in traj:
            print(f"[dynamics] twin cos(L, L') {traj['twin_cos']}")
        print(f"[dynamics] rho {res['rho']}")

    out = a.out or os.path.join(run_dir, "dynamics.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[dynamics] wrote {out}")


if __name__ == "__main__":
    main()
