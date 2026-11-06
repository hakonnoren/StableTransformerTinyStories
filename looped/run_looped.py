"""Train one looped model (looped/model.py) on a repo task with a randomised
number of iterations, then sweep the number of iterations at test time.

Tasks (the repo's own generators):
    khop     k-hop pointer chasing (looped/khop.py): S pair tokens a -> f(a)
             of a random single-cycle permutation, then n_query queries q with target f^K(q).
    listops  nested MIN/MAX (looped/listops.py), answer at the last position. The test
             set is split by actual nesting depth; an extra out-of-distribution set has deeper nesting
             (max_depth_ood) than training, within the same block size.

Archs (presets of LoopedConfig):
    vanilla         looped standard pre-LN block (baseline)
    kick_vp         Håkon's coupling, γ = α = 0 (volume preserving)
    kick_fixed      Håkon's coupling, γ = α = 0.005 fixed
    kick_learned    Håkon's coupling, γ = α learned from 0.005
    relax15         relax, rate 15 fixed (≈ baseline)
    relax3          relax, rate 3 fixed
    relax3_learned  relax, rate 3 learned
    relax1          relax, rate 1 fixed

Metrics (test set, after training): accuracy at r iterations for r in R_GRID; relative / absolute state
residual ||s_r - s_{r-1}||; overthinking = % of answers correct at r_hi (the largest training r) that are
wrong at r = 32 / 128.

    python looped/run_looped.py --task khop --arch relax3 --out results/looped/khop_K3
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]          # repo root (the folder containing looped/)
sys.path.insert(0, str(ROOT))
from looped.model import LoopedConfig, LoopedGPT          # noqa: E402

R_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]

ARCHS = {
    "vanilla":        dict(coupling="vanilla"),
    "kick_vp":        dict(coupling="kick", rate=0.0),
    "kick_fixed":     dict(coupling="kick", rate=0.005),
    "kick_learned":   dict(coupling="kick", rate=0.005, learn_rates=True),
    "relax15":        dict(coupling="relax", rate=15.0),
    "relax3":         dict(coupling="relax", rate=3.0),
    "relax3_learned": dict(coupling="relax", rate=3.0, learn_rates=True),
    "relax1":         dict(coupling="relax", rate=1.0),
    # --- constrained damping (looped/model.py, "Constrained damping") ---
    "relax3_leak01":  dict(coupling="relax", rate=3.0, leak=0.1),            # common-mode damping, c = e^-0.1
    "relax3_leak05":  dict(coupling="relax", rate=3.0, leak=0.5),
    "relax3_leakL":   dict(coupling="relax", rate=3.0, leak=0.1, learn_leak=True, learn_rates=True),
    "kick_pos":       dict(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.05,
                           learn_rates=True),                               # kick with rates kept > 0
    "relax_bounded":  dict(coupling="relax", rate=3.0, rate_constraint="bounded", rate_min=0.1, rate_max=5.0,
                           learn_rates=True, per_channel=True),
    "relax_budget":   dict(coupling="relax", rate=3.0, rate_constraint="budget", learn_rates=True,
                           per_channel=True),                               # γ+α = 6 per channel, learned split
}


# ---------------------------------------------------------------------------------------------- tasks
def task_khop(a):
    from looped.khop import make_khop
    np.random.seed(a.data_seed)
    xtr, ytr = make_khop(a.n_train, a.S, a.K, a.n_query)
    xte, yte = make_khop(a.n_test, a.S, a.K, a.n_query)
    return dict(x_train=xtr, y_train=ytr, tests={"test": (xte, yte, None)},
                vocab_size=a.S + a.S * a.S, block_size=a.S + a.n_query,
                params=dict(S=a.S, K=a.K, n_query=a.n_query))


def _nesting_depth(x):
    from looped.listops import OP_TOKENS, CLOSE_ID
    opens = np.isin(x, list(OP_TOKENS.values())).astype(int)
    closes = (x == CLOSE_ID).astype(int)
    return np.cumsum(opens - closes, axis=1).max(axis=1)


def task_listops(a):
    from looped.listops import get_xy_listops, LISTOPS_VOCAB_SIZE
    kw = dict(min_length=a.min_length, max_args=a.max_args, value_p=a.value_p, ops=["MIN", "MAX"])

    def gen(n, max_depth, min_depth=0):
        xs, ys = [], []
        while len(xs) < n:
            x, y = get_xy_listops(a.max_length, max_depth=max_depth, **kw)
            if min_depth and _nesting_depth(x[None])[0] < min_depth:
                continue
            xs.append(x); ys.append(y)
        return np.stack(xs), np.stack(ys)

    np.random.seed(a.data_seed)
    xtr, ytr = gen(a.n_train, a.max_depth)
    xte, yte = gen(a.n_test, a.max_depth)
    tests = {"test": (xte, yte, _nesting_depth(xte))}
    if a.max_depth_ood > a.max_depth:
        xo, yo = gen(a.n_test, a.max_depth_ood, min_depth=a.max_depth)     # depth >= train max + 1
        tests["ood"] = (xo, yo, _nesting_depth(xo))
    return dict(x_train=xtr, y_train=ytr, tests=tests, vocab_size=LISTOPS_VOCAB_SIZE,
                block_size=a.max_length,
                params=dict(max_length=a.max_length, max_depth=a.max_depth, max_depth_ood=a.max_depth_ood,
                            min_length=a.min_length, max_args=a.max_args, value_p=a.value_p, ops=["MIN", "MAX"]))


def task_fsc(a):
    """Finite-state composition (Liang et al. 2026): F fixed random permutations of S states. Input
    [PAD.. s0 o1 .. oH EQ] (left-padded to the longest test chain), answer f_{oH}(..f_{o1}(s0)) at EQ.
    Train H ~ U[h_lo, h_hi]; 'test' = the training range, 'ood' = longer chains up to h_test; groups = H."""
    S, Fn = a.fsc_states, a.fsc_funcs
    rng = np.random.default_rng(a.data_seed)
    perms = np.stack([rng.permutation(S) for _ in range(Fn)])
    EQ, PAD = S + Fn, S + Fn + 1
    T = a.h_test + 2

    def gen(n, h_lo, h_hi):
        x = np.full((n, T), PAD, dtype=np.int64); y = -np.ones((n, T), dtype=np.int64)
        H = rng.integers(h_lo, h_hi + 1, n)
        for i in range(n):
            s0 = rng.integers(S); ops = rng.integers(Fn, size=H[i])
            st = s0
            for o in ops:
                st = perms[o, st]
            seq = [s0] + list(S + ops) + [EQ]
            x[i, T - len(seq):] = seq
            y[i, -1] = st
        return x, y, H
    xtr, ytr, _ = gen(a.n_train, a.h_lo, a.h_hi)
    xte, yte, gte = gen(a.n_test, a.h_lo, a.h_hi)
    xo, yo, go = gen(a.n_test, a.h_hi + 1, a.h_test)
    return dict(x_train=xtr, y_train=ytr, tests={"test": (xte, yte, gte), "ood": (xo, yo, go)},
                vocab_size=S + Fn + 2, block_size=T,
                params=dict(S=S, F=Fn, h_lo=a.h_lo, h_hi=a.h_hi, h_test=a.h_test))


TASKS = {"khop": task_khop, "listops": task_listops, "fsc": task_fsc}


# ---------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, x, y, groups=None, r_hi=8, bs=512, r_max=128):
    """Per-r accuracy over supervised positions, residuals, overthinking, per-group accuracy."""
    model.eval()
    dev = next(model.parameters()).device
    grid = [r for r in R_GRID if r <= r_max]
    correct = {r: [] for r in grid}
    res_rel = {r: [] for r in grid}
    res_abs = {r: [] for r in grid}
    res_q = {r: [] for r in grid}
    norm_q = {r: [] for r in grid}
    norm_shift = {r: [] for r in grid}
    seq_group = []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i:i + bs], dtype=torch.long, device=dev)
        yb = torch.as_tensor(y[i:i + bs], dtype=torch.long, device=dev)
        mask = yb != -1
        traj = model.trajectory(xb, max(grid), grid)
        for r in grid:
            logits, rel, ab, ex = traj[r]
            res_q[r].append(ex["res_q"] * len(xb)); norm_q[r].append(ex["norm_q"] * len(xb))
            norm_shift[r].append(ex["norm_shift"] * len(xb))
            ok = (logits.argmax(-1) == yb) & torch.isfinite(logits).all(-1)
            correct[r].append(ok[mask].cpu().numpy())
            res_rel[r].append(rel * len(xb)); res_abs[r].append(ab * len(xb))
        if groups is not None:
            seq_group.append(np.repeat(groups[i:i + bs], mask.sum(1).cpu().numpy()))
    n = len(x)
    correct = {r: np.concatenate(v) for r, v in correct.items()}
    out = {"acc": {r: float(c.mean()) for r, c in correct.items()},
           "res_rel": {r: float(np.sum(v) / n) for r, v in res_rel.items()},
           "res_abs": {r: float(np.sum(v) / n) for r, v in res_abs.items()},
           "res_q": {r: float(np.sum(v) / n) for r, v in res_q.items()},
           "norm_q": {r: float(np.sum(v) / n) for r, v in norm_q.items()},
           "norm_shift": {r: float(np.sum(v) / n) for r, v in norm_shift.items()},
           "n_answers": int(len(correct[grid[0]]))}
    base = correct[r_hi]
    for r_late in (32, 128):
        if r_late in correct:
            out[f"overthink_{r_hi}_to_{r_late}"] = float(100 * (base & ~correct[r_late]).sum() / max(base.sum(), 1))
            out[f"recovered_{r_hi}_to_{r_late}"] = float(100 * (~base & correct[r_late]).sum() / max((~base).sum(), 1))
    if groups is not None:
        g = np.concatenate(seq_group)
        out["acc_by_group"] = {int(k): {r: float(correct[r][g == k].mean()) for r in grid}
                               for k in np.unique(g)}
        out["count_by_group"] = {int(k): int((g == k).sum()) for k in np.unique(g)}
    model.train()
    return out


# ---------------------------------------------------------------------------------------------- train
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--arch", choices=list(ARCHS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1238)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rate-lr-mult", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--r-lo", type=int, default=3)
    p.add_argument("--r-hi", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=32)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-core", type=int, default=1)
    p.add_argument("--no-inject", action="store_true")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--rate", type=float, default=None, help="override the preset's initial rate")
    p.add_argument("--rate-h", type=float, default=None, help="initial rate α of the read-out stream H (default: = rate)")
    p.add_argument("--leak", type=float, default=None, help="override: common-mode damping δ (relax)")
    p.add_argument("--learn-leak", action="store_true")
    p.add_argument("--inject-mode", choices=["pre", "post"], default="pre",
                   help="pre: Attn(LN(H+e)); post: Attn(LN(H)+e) (input weight independent of the state norm)")
    p.add_argument("--learn-rates", action="store_true", help="override: learn the rates")
    p.add_argument("--carry-max", type=int, default=0,
                   help="with prob. --carry-prob, first run U{1..carry_max} iterations without gradient "
                        "(detached carry, EqR-style), then the usual r ~ U[r_lo, r_hi] with gradient")
    p.add_argument("--carry-prob", type=float, default=0.5)
    p.add_argument("--zero-init", action="store_true",
                   help="zero the output projections of the core's attention and MLP (identity iteration at init)")
    p.add_argument("--res-penalty", type=float, default=0.0,
                   help="weight of the fixed-point penalty ||s_{r+1}-s_r||^2/||s_r||^2 after the last iteration")
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    # khop
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--n-query", type=int, default=8)
    # fsc
    p.add_argument("--fsc-states", type=int, default=16)
    p.add_argument("--fsc-funcs", type=int, default=8)
    p.add_argument("--h-lo", type=int, default=1)
    p.add_argument("--h-hi", type=int, default=6)
    p.add_argument("--h-test", type=int, default=16)
    p.add_argument("--no-pos", action="store_true", help="NoPE: no positional embeddings")
    # listops
    p.add_argument("--max-length", type=int, default=40)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-depth-ood", type=int, default=7)
    p.add_argument("--min-length", type=int, default=12)
    p.add_argument("--max-args", type=int, default=6)
    p.add_argument("--value-p", type=float, default=0.4)
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.arch}{a.tag}_seed{a.seed}"
    t0 = time.time()
    data = TASKS[a.task](a)
    t_data = time.time() - t0

    preset = dict(ARCHS[a.arch])
    if a.rate is not None:
        preset["rate"] = a.rate
    if a.rate_h is not None:
        preset["rate_h"] = a.rate_h
    if a.leak is not None:
        preset["leak"] = a.leak
    if a.learn_leak:
        preset["learn_leak"] = True
    if a.learn_rates:
        preset["learn_rates"] = True
    if a.per_channel:
        preset["per_channel"] = True
    preset["inject_mode"] = a.inject_mode
    if a.no_pos:
        preset["pos_emb"] = False
    cfg = LoopedConfig(vocab_size=data["vocab_size"], block_size=data["block_size"], n_embd=a.n_embd,
                       n_head=a.n_head, n_core=a.n_core, inject=not a.no_inject, **preset)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = LoopedGPT(cfg).to(a.device)
    if a.zero_init:
        with torch.no_grad():
            for layer in model.core:
                for proj in (layer.attn.c_proj, layer.mlp.c_proj):
                    proj.weight.zero_(); proj.bias.zero_()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = model.configure_optimizers(a.lr, rate_lr_mult=a.rate_lr_mult)

    def lr_lambda(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = min((step - a.warmup) / max(1, a.steps - a.warmup), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    xtr = torch.as_tensor(data["x_train"], dtype=torch.long, device=a.device)
    ytr = torch.as_tensor(data["y_train"], dtype=torch.long, device=a.device)
    xte, yte, _ = data["tests"]["test"]
    xq, yq = xte[:512], yte[:512]
    log = {"step": [], "loss": [], "r": []}
    curve = []
    print(f"[{stem}] task={a.task} params={n_params} data {t_data:.1f}s", flush=True)
    model.train()
    t_train = time.time()
    run_loss = []
    for step in range(1, a.steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(xtr), a.batch), device=a.device)
        r = int(rng.integers(a.r_lo, a.r_hi + 1))
        r0 = int(rng.integers(1, a.carry_max + 1)) if a.carry_max > 0 and rng.random() < a.carry_prob else 0
        if a.res_penalty > 0 or r0 > 0:
            s, e = model.embed(xtr[idx])
            if r0 > 0:              # detached carry: start the graded iterations from s_{r0}
                with torch.no_grad():
                    sd, ed = s.detach(), e.detach()
                    for _ in range(r0):
                        sd = model.step(sd, ed)
                s = sd
            for _ in range(r):
                s = model.step(s, e)
            logits = model.readout(s)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ytr[idx].reshape(-1), ignore_index=-1)
            if a.res_penalty > 0:
                s_next = model.step(s, e)
                res = ((s_next - s).flatten(1).pow(2).sum(1) / s.flatten(1).pow(2).sum(1).clamp_min(1e-12)).mean()
                loss = loss + a.res_penalty * res
        else:
            _, loss = model(xtr[idx], ytr[idx], n_iter=r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss.append(loss.detach())
        if step % 50 == 0:
            log["step"].append(step); log["loss"].append(float(torch.stack(run_loss).mean())); log["r"].append(r)
            run_loss = []
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, xq, yq, r_hi=a.r_hi, r_max=128)
            curve.append({"step": step, "acc": ev["acc"], "res_rel": ev["res_rel"], "rates": model.rates()})
            acc = ev["acc"]
            cur = log["loss"][-1] if log["loss"] else float(torch.stack(run_loss).mean())
            print(f"[{stem}] step {step:5d} loss {cur:.4f} acc@{a.r_lo} {acc[a.r_lo]:.3f} "
                  f"@{a.r_hi} {acc[a.r_hi]:.3f} @32 {acc[32]:.3f} @128 {acc[128]:.3f} "
                  f"res@128 {ev['res_rel'][128]:.2e} ({time.time() - t_train:.0f}s)", flush=True)
    t_train = time.time() - t_train

    final = {name: evaluate(model, x, y, groups=g, r_hi=a.r_hi) for name, (x, y, g) in data["tests"].items()}
    # spectral radius of the iteration map's Jacobian at s_r (64 test sequences, power iteration)
    xs = torch.as_tensor(xte[:64], dtype=torch.long, device=a.device)
    rho = {}
    for r_ in (a.r_hi, 32, 128):
        for q in (False, True):
            est = model.step_spectral_radius(xs, r_, quotient=q).cpu().numpy()
            est = est[np.isfinite(est)]
            rho[f"{r_}{'q' if q else ''}"] = {
                "median": float(np.median(est)) if len(est) else float("nan"),
                "p90": float(np.quantile(est, 0.9)) if len(est) else float("nan"),
                "frac_lt1": float((est < 1).mean()) if len(est) else float("nan")}
    final["test"]["spectral_radius"] = rho
    res = {"args": vars(a), "config": cfg.to_dict(), "n_params": n_params, "task_params": data["params"],
           "train_log": log, "curve": curve, "final": final, "rates": model.rates(),
           "time_train_s": t_train, "time_data_s": t_data}
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg.to_dict(), "args": vars(a)}, out / f"{stem}.pt")
    f = final["test"]
    print(f"[{stem}] DONE {t_train:.0f}s  acc " + " ".join(f"{r}:{f['acc'][r]:.3f}" for r in (a.r_lo, a.r_hi, 16, 32, 128))
          + f"  overthink {f.get(f'overthink_{a.r_hi}_to_128', float('nan')):.1f}%  rho@128 "
          f"{f['spectral_radius']['128']['median']:.3f} (quotient {f['spectral_radius']['128q']['median']:.3f})  "
          f"res_abs@128 {f['res_abs'][128]:.3g} (quotient {f['res_q'][128]:.3g})  rates {model.rates()}",
          flush=True)


if __name__ == "__main__":
    main()
