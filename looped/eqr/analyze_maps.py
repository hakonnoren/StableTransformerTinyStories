"""Which properties of the iteration map make depth, breadth and test noise work? Measurements on trained
EqR / coupled-EqR Sudoku checkpoints (exported by export_for_analysis.py, or full ones).

    python looped/eqr/analyze_maps.py --eqr-dir $EQR_DIR --data <export>/data/sudoku-extreme-1k-aug-1000 \
        --ckpt <export>/ckpts/eqr_s0__step20000.pth [--out <file>.json] [--preset gpu|local]

One outer iteration = one EqR ACT step = H_cycles latent recursions (`inner(carry, batch)`), run without
halting or resets, in float32. Per checkpoint:

  rates      (coupled) learned gamma / alpha / delta, and the volume bound: every coupled update is a shear
             times a constant scaling, so one outer iteration has |det J| = prod a^nL b^nH per channel and the
             geometric mean of |eigenvalues| is exp(-(nL gamma + nH alpha)/2) ("conformal floor"; rho >= it)
  multistart N puzzles x B random starts (EqR's reset), R iterations, without noise and at each --noise.
             At each recorded r: single-run exact/cell; per puzzle: any, plurality vote, distinct answers,
             mass of the correct answer, EqR's Delta-PI; selection by EqR's convergence score (logit change over
             the last 3 iterations), by relative state step, by the halting logit, by the Sudoku verifier
             (no conflicts + agrees with the givens; label-free); AUROC of each score for correctness;
             endpoint path independence (mean pairwise cosine of the B final states per puzzle)
  escape     from the noise-free endpoints, K iterations at each --escape-noise: P(keep answer | correct),
             P(keep | wrong), P(becomes correct | wrong); state RMS per stream (to read noise relative to scale)
  jacobian   at noise-free endpoints of up to --n-jac runs (half correct, half wrong when available): spectral
             radius of one outer iteration (power iteration on JVPs) and the leading Ritz values (Arnoldi):
             modulus, share of complex pairs (oscillatory modes)
  trace      per update inside a recursion (L, L', H), the relative change, at outer iterations 1, 2, 4, ...:
             does L settle within a recursion (HRM's hierarchical convergence)? And the inner fixed point:
             L-type updates repeated with H frozen -> contraction factor and the distance of the scheduled
             L from that fixed point
  pr         participation ratio (sum l)^2 / sum l^2 of the per-token state covariance, per stream
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.attention  # noqa: F401  (sdpa_kernel for the JVPs)
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maps_common as mc  # noqa: E402

PRESETS = {
    "gpu": dict(n=256, B=16, R=256, rec="4,16,64,256", chunk=1024, noise="0.25,0.5,0.75", escape_noise="0.1,0.25,0.5,0.75,1.0",
                escape_k=32, n_jac=32, power=60, arnoldi=30, n_trace=64, trace_iters=64, n_pr=128),
    "local": dict(n=8, B=4, R=32, rec="4,16,32", chunk=32, noise="0.5", escape_noise="0.25,0.5",
                  escape_k=8, n_jac=4, power=20, arnoldi=10, n_trace=8, trace_iters=8, n_pr=8),
}


# ------------------------------------------------------------------------------------------- sudoku
def sudoku_units(device):
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)] for br in range(3) for bc in range(3)]
    return torch.tensor(rows + cols + boxes, device=device)


def verify(pred, inputs, units):
    """Conflict count E (Ren & Liu Eq. 7: sum over units/digits of ReLU(count - 1), plus non-digit cells) and the
    number of changed givens. Tokens: 1 = blank, 2..10 = digits. Valid solution iff both are 0.
    units=None: Skyscrapers-7 (verify_skyscrapers)."""
    if units is None:
        return verify_skyscrapers(pred, inputs)
    digit = (pred >= 2) & (pred <= 10)
    oh = F.one_hot((pred - 2).clamp(0, 8), 9) * digit[..., None]
    cnt = oh[:, units].sum(2)
    E = (cnt - 1).clamp(min=0).sum((1, 2)) + (~digit).sum(1)
    clash = ((pred != inputs) & (inputs != 1)).sum(1)
    return E, clash


def verify_skyscrapers(pred, inputs, n=7):
    """Skyscrapers-7 on the 9x9 board (looped/eqr/skyscrapers.py). E = Latin conflicts of the inner grid (as for
    Sudoku, rows and columns) + non-height cells + clues whose visible count differs; clash = changed givens.
    Tokens: 1 = blank / no clue, 2..8 = 1..7. The border of `pred` is ignored (its labels are)."""
    b = n + 2
    p = pred.view(-1, b, b)[:, 1:-1, 1:-1].long() - 1                       # heights 1..7 where valid
    x = inputs.view(-1, b, b).long() - 1                                     # 0 = blank / no clue
    ok = (p >= 1) & (p <= n)
    h = torch.where(ok, p, torch.zeros_like(p))
    oh = F.one_hot(h, n + 1)[..., 1:]                                        # [m, 7, 7, 7]; invalid cells all zero
    E = (oh.sum(1) - 1).clamp(min=0).sum((1, 2)) + (oh.sum(2) - 1).clamp(min=0).sum((1, 2)) + (~ok).sum((1, 2))

    def seen(lines):                                                         # [m, k, 7] -> buildings seen from index 0
        prev = torch.cummax(lines, dim=-1).values
        prev = torch.cat([torch.zeros_like(prev[..., :1]), prev[..., :-1]], dim=-1)
        return (lines > prev).sum(-1)
    for vis, clue in ((seen(h), x[:, 1:-1, 0]), (seen(h.flip(-1)), x[:, 1:-1, -1]),
                      (seen(h.transpose(1, 2)), x[:, 0, 1:-1]), (seen(h.transpose(1, 2).flip(-1)), x[:, -1, 1:-1])):
        E = E + ((clue > 0) & (vis != clue)).sum(1)
    given = x[:, 1:-1, 1:-1]
    clash = ((given > 0) & (p != given)).sum((1, 2))
    return E, clash


def auroc(score, label):
    """P(score of a correct run < score of a wrong run), i.e. AUROC of -score for correctness (low = confident).
    None when one class is empty."""
    score, label = score.cpu().double(), label.cpu().bool()
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(allv)
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float64)
    # average ranks for ties
    uniq, inv, cnt = torch.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, ranks)
        ranks = (sums / cnt.double())[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(1 - u / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------- maps
def is_coupled(inner):
    return hasattr(inner, "rates")


def outer_map(inner, x, seq, n_h):
    """s = cat(z_H, z_L) along seq -> the same after one outer iteration (H_cycles recursions), noise off."""
    def f(s):
        h, l = s[:, :n_h], s[:, n_h:]
        for _ in range(inner.config.H_cycles):
            h, l = inner.latent_recursion(h, l, x, seq)
        return torch.cat([h, l], dim=1)
    return f


def jvp(f, s, v):
    try:
        # the attention block (RNA) needs SDPA's math kernel: the fused kernels have no forward-mode AD
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return torch.func.jvp(f, (s,), (v,))[1]
    except Exception:                                    # e.g. an op without forward-mode AD on this device
        eps = 1e-3 * s.flatten(1).norm(dim=1) / v.flatten(1).norm(dim=1).clamp_min(1e-30)
        eps = eps.view(-1, *([1] * (s.dim() - 1)))
        return (f(s + eps * v) - f(s - eps * v)) / (2 * eps)


def spectral(inner, carry, batch, power_iters, m, keep=20):
    """Spectral radius (power iteration, geometric mean of the last `keep` growth factors) and Ritz values of an
    m-step Arnoldi process, per sample, for one outer iteration's Jacobian at the given state."""
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]).float()
    seq = {"cos_sin": inner._cos_sin()}
    n_h = carry.z_H.shape[1]
    f = outer_map(inner, x, seq, n_h)
    s = torch.cat([carry.z_H, carry.z_L], dim=1).float()
    b = s.shape[0]
    bc = lambda t: t.view(-1, *([1] * (s.dim() - 1)))
    nrm = lambda t: t.flatten(1).norm(dim=1)
    v = torch.randn_like(s)
    v = v / bc(nrm(v))
    logs = []
    for k in range(power_iters):
        w = jvp(f, s, v)
        g = nrm(w).clamp_min(1e-30)
        if k >= power_iters - keep:
            logs.append(torch.log(g))
        v = w / bc(g)
    rho = torch.exp(torch.stack(logs).mean(0))
    # Arnoldi (modified Gram-Schmidt, one re-orthogonalisation)
    V = [torch.randn_like(s)]
    V[0] = V[0] / bc(nrm(V[0]))
    Hm = torch.zeros(b, m + 1, m, dtype=torch.float64)
    dot = lambda a, c: (a.flatten(1) * c.flatten(1)).sum(1)
    for j in range(m):
        w = jvp(f, s, V[j])
        for _ in range(2):
            for i in range(j + 1):
                h = dot(V[i], w)
                Hm[:, i, j] += h.cpu().double()
                w = w - bc(h) * V[i]
        hn = nrm(w)
        Hm[:, j + 1, j] = hn.cpu().double()
        V.append(w / bc(hn.clamp_min(1e-30)))
    ritz = torch.linalg.eigvals(Hm[:, :m, :m])
    order = ritz.abs().argsort(dim=1, descending=True)
    ritz = torch.gather(ritz, 1, order)
    return rho.cpu(), ritz


def trace_recursion(inner, zH, zL, x, seq):
    """One latent recursion step by step; returns the new (zH, zL) and [(stream, relative change)]."""
    rel = lambda new, old: ((new - old).flatten(1).norm(dim=1) / new.flatten(1).norm(dim=1).clamp_min(1e-30))
    out = []
    if not is_coupled(inner):                                             # EqR: L x L_cycles, then H
        for _ in range(inner.config.L_cycles):
            new = inner.L_level(zL, zH + x, **seq)
            out.append(("L", rel(new, zL)))
            zL = new
        new = inner.L_level(zH, zL, **seq)
        out.append(("H", rel(new, zH)))
        return new, zL, out
    if getattr(inner, "twin_l", False):
        L, M = zL.chunk(2, dim=1)
        for st in inner.schedule:
            if st == "L":
                new = inner.half_step_twin(L, M, zH, x, seq); out.append(("L", rel(new, L))); L = new
            elif st == "M":
                new = inner.half_step_twin(M, L, zH, x, seq); out.append(("L'", rel(new, M))); M = new
            else:
                new = inner.half_step_H(zH, 0.5 * (L + M), x, seq); out.append(("H", rel(new, zH))); zH = new
        return zH, torch.cat([L, M], dim=1), out
    for st in inner.schedule:
        if st == "L":
            new = inner.half_step_L(zH, zL, x, seq); out.append(("L", rel(new, zL))); zL = new
        else:
            new = inner.half_step_H(zH, zL, x, seq); out.append(("H", rel(new, zH))); zH = new
    return zH, zL, out


def inner_fixed_point(inner, zH, zL, x, seq, iters=64):
    """L-type updates repeated with H frozen (EqR: L <- L_level(L, H + x); coupled: its L half-step(s)).
    Returns per-iteration relative changes (tensor [iters, b]) and the final L."""
    rel = lambda new, old: ((new - old).flatten(1).norm(dim=1) / new.flatten(1).norm(dim=1).clamp_min(1e-30))
    steps, steps2, prev2 = [], [], None
    for _ in range(iters):
        if not is_coupled(inner):
            new = inner.L_level(zL, zH + x, **seq)
        elif getattr(inner, "twin_l", False):
            L, M = zL.chunk(2, dim=1)
            L = inner.half_step_twin(L, M, zH, x, seq)
            M = inner.half_step_twin(M, L, zH, x, seq)
            new = torch.cat([L, M], dim=1)
        else:
            new = inner.half_step_L(zH, zL, x, seq)
        steps.append(rel(new, zL))
        steps2.append(rel(new, prev2) if prev2 is not None else torch.full_like(steps[-1], float("nan")))
        prev2, zL = zL, new
    return torch.stack(steps), zL, torch.stack(steps2)


def participation_ratio(x):
    """(sum l)^2 / sum l^2 of the covariance of the rows of x."""
    x = x.double()
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-300))


# ------------------------------------------------------------------------------------------- runs
TWIN_SPLIT_TOKEN = 0.1         # a run's copies count as split if some token has sep > 0.1 (sep defined below)


def twin_split(views):
    """Twin models: per run, how far the copies L, L' are apart (paper Prop. 7c/8: merged copies are the
    self-map's fixed points; split copies, L != L', its 2-cycles). sep = |L - L'| / (|L| + |L'|) in [0, 1]:
    0 for merged copies, 1 for opposite ones; over the whole state (twin_sep) and the worst token (twin_tokmax).
    Empty for models without twins."""
    if "L2" not in views:
        return {}
    Lf, Mf = views["L"].float(), views["L2"].float()                    # [runs, seq, d]
    sep = lambda a, b, dim: (a - b).norm(dim=dim) / (a.norm(dim=dim) + b.norm(dim=dim)).clamp_min(1e-30)
    return {"twin_cos": F.cosine_similarity(Lf.flatten(1), Mf.flatten(1), dim=1).cpu(),
            "twin_sep": sep(Lf.flatten(1), Mf.flatten(1), 1).cpu(),
            "twin_tokmax": sep(Lf, Mf, -1).amax(dim=1).cpu()}


def twin_summary(rec_out):
    out = {}
    for cls, m in (("all", torch.ones_like(rec_out["exact"])), ("correct", rec_out["exact"]), ("wrong", ~rec_out["exact"])):
        if not m.any():
            out[cls] = None
            continue
        sep, tok = rec_out["twin_sep"][m], rec_out["twin_tokmax"][m]
        out[cls] = {"n": int(m.sum()), "cos_median": float(rec_out["twin_cos"][m].median()),
                    "sep_median": float(sep.median()), "sep_q90": float(torch.quantile(sep, 0.9)),
                    "tokmax_median": float(tok.median()), "frac_split": float((tok > TWIN_SPLIT_TOKEN).float().mean())}
    return out


@torch.no_grad()
def multistart_chunk(inner, batch, B, R, rec, noise, units, device):
    """B random starts per puzzle, R iterations at `noise`; per recorded r, per-run tensors (CPU)."""
    n = batch["inputs"].shape[0]
    big = mc.repeat_batch(batch, B)
    mc.set_noise(inner, noise)
    carry = mc.random_carry(inner, n * B, device)
    prev = mc.flat_state(carry)
    prev2 = None
    sres_buf, logit_buf, out = [], [], {}
    for t in range(1, R + 1):
        carry, logits, q = inner(carry, big)
        s = mc.flat_state(carry)
        sres_buf = (sres_buf + [(s - prev).norm(dim=1) / s.norm(dim=1).clamp_min(1e-30)])[-3:]
        sres2 = (s - prev2).norm(dim=1) / s.norm(dim=1).clamp_min(1e-30) if prev2 is not None else None
        prev2, prev = prev, s
        logit_buf = (logit_buf + [logits.float()])[-4:]
        if t in rec:
            lg = torch.stack(logit_buf)
            lscore = (lg[1:] - lg[:-1]).norm(dim=-1).mean(dim=(0, -1)) if len(lg) > 1 else torch.zeros(n * B, device=device)
            ex, cell, pred = mc.exact_and_cell(logits, big["labels"])
            E, clash = verify(pred, big["inputs"], units)
            views = mc.stream_views(inner, carry)
            out[t] = {"exact": ex.cpu(), "cell": cell.cpu(), "pred": pred.to(torch.uint8).cpu(),
                      "E": E.cpu(), "clash": clash.cpu(), "lscore": lscore.cpu(),
                      "sres": torch.stack(sres_buf).mean(0).cpu(), "q": q.float().cpu(),
                      "sres_last": sres_buf[-1].cpu(),
                      "sres2": (sres2 if sres2 is not None else torch.full_like(sres_buf[-1], float("nan"))).cpu(),
                      "rms": {k: float(v.float().pow(2).mean().sqrt()) for k, v in views.items()},
                      **twin_split(views)}
    # endpoint path independence: mean pairwise cosine of the B final states per puzzle
    sn = F.normalize(mc.flat_state(carry).view(n, B, -1), dim=-1)
    gram = sn @ sn.transpose(1, 2)
    pi_cos = ((gram.sum((1, 2)) - B) / max(B * (B - 1), 1)).cpu()
    return out, carry, big, pi_cos


def per_puzzle(rec_out, n, B):
    """Selection and diversity metrics from one recorded step's per-run tensors."""
    ex = rec_out["exact"].view(n, B)
    valid = ((rec_out["E"] == 0) & (rec_out["clash"] == 0)).view(n, B)
    pred = rec_out["pred"].view(n, B, -1)
    pick = lambda score, low=True: torch.gather(ex, 1, (score.view(n, B).argmin(1) if low else score.view(n, B).argmax(1))[:, None])[:, 0]
    vote, distinct, mass = [], [], []
    for p in range(n):
        uniq, inv, cnt = torch.unique(pred[p], dim=0, return_inverse=True, return_counts=True)
        win = cnt.argmax()
        vote.append(bool(ex[p][inv == win].any()))
        distinct.append(len(uniq))
        mass.append(float(ex[p].float().mean()))
    verifier = torch.where(valid.any(1), torch.gather(ex, 1, valid.float().argmax(1)[:, None])[:, 0], pick(rec_out["lscore"]))
    mass_t = torch.tensor(mass)
    return {
        "single": float(ex.float().mean()), "cell": float(rec_out["cell"].mean()),
        "any": float(ex.any(1).float().mean()), "vote": float(torch.tensor(vote).float().mean()),
        "conv1": float(pick(rec_out["lscore"]).float().mean()), "sres1": float(pick(rec_out["sres"]).float().mean()),
        "q1": float(pick(rec_out["q"], low=False).float().mean()), "verifier": float(verifier.float().mean()),
        "valid_but_wrong": float((valid & ~ex).float().mean()), "valid": float(valid.float().mean()),
        "distinct_mean": float(torch.tensor(distinct).float().mean()),
        "distinct_hist": {str(k): int(v) for k, v in zip(*torch.unique(torch.tensor(distinct), return_counts=True))},
        "mass_quantiles": [float(v) for v in torch.quantile(mass_t, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))],
        "frac_puzzles_all_correct": float((mass_t == 1).float().mean()),
        "frac_puzzles_mixed": float(((mass_t > 0) & (mass_t < 1)).float().mean()),
        "delta_pi": float((mass_t - ex[:, 0].float()).abs().mean()),
        "auroc": {"lscore": auroc(rec_out["lscore"], rec_out["exact"]), "sres": auroc(rec_out["sres"], rec_out["exact"]),
                  "neg_q": auroc(-rec_out["q"], rec_out["exact"]), "E": auroc(rec_out["E"].float(), rec_out["exact"])},
        # outer-iteration dynamics at this r: fixed point (step < 1e-4), 2-cycle (step >= 1e-3, 2-step < step/10), else moving
        "outer_fixed": float((rec_out["sres_last"] < 1e-4).float().mean()),
        "outer_2cycle": float(((rec_out["sres_last"] >= 1e-3) & (rec_out["sres2"] < 0.1 * rec_out["sres_last"])).float().mean()),
        "outer_dyn_by_class": {cls: ({"fixed": float((rec_out["sres_last"][m] < 1e-4).float().mean()),
                                      "step_median": float(rec_out["sres_last"][m].median())} if m.any() else None)
                               for cls, m in (("correct", rec_out["exact"]), ("wrong", ~rec_out["exact"]))},
        "by_class": {cls: {k: (float(rec_out[k][m].float().median()) if m.any() else None) for k in ("lscore", "sres", "q", "E")}
                     for cls, m in (("correct", rec_out["exact"]), ("wrong", ~rec_out["exact"]))},
        "rms": rec_out["rms"],
        **({"twin": twin_summary(rec_out)} if "twin_cos" in rec_out else {}),
    }


@torch.no_grad()
def escape(inner, carry, big, base_pred, base_exact, sigmas, K, units):
    out = {}
    for sg in sigmas:
        c = type(carry)(**{k: v.clone() for k, v in vars(carry).items()})
        mc.set_noise(inner, sg)
        for _ in range(K):
            c, logits, _ = inner(c, big)
        ex, _, pred = mc.exact_and_cell(logits, big["labels"])
        same = (pred.cpu().to(torch.uint8) == base_pred).all(-1)
        ex = ex.cpu()
        cor, wr = base_exact, ~base_exact
        out[str(sg)] = {"keep_if_correct": float(same[cor].float().mean()) if cor.any() else None,
                        "keep_if_wrong": float(same[wr].float().mean()) if wr.any() else None,
                        "rescued_if_wrong": float(ex[wr].float().mean()) if wr.any() else None,
                        "lost_if_correct": float((~ex[cor]).float().mean()) if cor.any() else None,
                        "n_correct": int(cor.sum()), "n_wrong": int(wr.sum())}
    mc.set_noise(inner, 0.0)
    return out


def rates_info(ck_path, inner):
    if not is_coupled(inner):
        return None
    with torch.no_grad():
        g, a = inner.rates.gamma().cpu().double(), inner.rates.alpha().cpu().double()
        c = inner.rates.c()
        c = None if c is None else c.cpu().double()
    sched = list(inner.schedule) * int(inner.config.H_cycles)
    n_l = sum(s in ("L", "M") for s in sched)
    n_h = sum(s == "H" for s in sched)
    n_streams = 3 if getattr(inner, "twin_l", False) else 2
    # per channel and token: log|det| of one outer iteration = -(n_l*gamma + n_h*alpha); geometric-mean |eig|
    logdet = -(n_l * g.mean() + n_h * a.mean())
    return {"gamma": float(g.mean()), "alpha": float(a.mean()), "delta": None if c is None else float(-torch.log(c).mean()),
            "keep_L": float(torch.exp(-g).mean()), "keep_H": float(torch.exp(-a).mean()),
            "schedule": "".join(inner.schedule), "n_streams": n_streams,
            "floor_geomean_abs_eig": float(torch.exp(logdet / n_streams)),
            "merged_retention": None if c is None else float(torch.exp(-g).mean() + (1 - torch.exp(-g).mean()) * c.mean())}


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqr-dir", required=True)
    ap.add_argument("--data", required=True, help="dataset dir with test/ and train/dataset.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="local", choices=sorted(PRESETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of sections to skip: multistart,escape,jacobian,trace,pr")
    for k, v in PRESETS["gpu"].items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset])
    cfg.update({k: v for k, v in vars(a).items() if k in cfg and v is not None})
    skip = {s for s in a.skip.split(",") if s}
    ckpt, data = os.path.abspath(a.ckpt), os.path.abspath(a.data)
    out_path = os.path.abspath(a.out) if a.out else os.path.splitext(ckpt)[0] + f".maps_{a.preset}.json"
    mc.setup_eqr(a.eqr_dir)
    device = mc.pick_device(a.device)
    torch.manual_seed(a.seed)
    inner, config, step = mc.load_model(ckpt, data, device)
    vocab = json.load(open(os.path.join(data, "test", "dataset.json"))).get("vocab_size")
    units = None if vocab == 9 else sudoku_units(device)          # 9 tokens: Skyscrapers-7 (verify_skyscrapers)
    rec = sorted({int(r) for r in cfg["rec"].split(",")} | {cfg["R"]})
    noises = [0.0] + [float(s) for s in str(cfg["noise"]).split(",") if s]
    res = {"ckpt": ckpt, "run_name": config.run_name, "step": step, "device": str(device), "cfg": cfg,
           "arch": {k: v for k, v in config.arch.__dict__.items() if isinstance(v, (int, float, str, bool, type(None)))},
           "rates": rates_info(ckpt, inner)}
    t0 = time.time()
    n, B = cfg["n"], cfg["B"]
    batch_all = mc.load_puzzles(data, n, device=device)
    per_chunk = max(1, cfg["chunk"] // B)

    jac_pool = []                                       # (carry rows, batch rows, exact) from the noise-free endpoints
    if "multistart" not in skip:
        res["multistart"] = {}
        for noise in noises:
            acc = {r: [] for r in rec}
            pis, esc_parts = [], []
            for p0 in range(0, n, per_chunk):
                sub = {k: v[p0:p0 + per_chunk] for k, v in batch_all.items()}
                out, carry, big, pi_cos = multistart_chunk(inner, sub, B, cfg["R"], rec, noise, units, device)
                for r in rec:
                    acc[r].append(out[r])
                pis.append(pi_cos)
                if noise == 0.0:
                    last = out[cfg["R"]]
                    if "escape" not in skip:
                        esc_parts.append(escape(inner, carry, big, last["pred"], last["exact"],
                                                [float(s) for s in cfg["escape_noise"].split(",")], cfg["escape_k"], units))
                    if len(jac_pool) < 4:
                        jac_pool.append((type(carry)(**{k: v.detach().clone() for k, v in vars(carry).items()}),
                                         {k: v.clone() for k, v in big.items()}, last["exact"].clone()))
            key = f"noise_{noise:g}"
            res["multistart"][key] = {}
            for r in rec:
                cat = {k: (torch.cat([o[k] for o in acc[r]]) if k != "rms" else acc[r][0][k]) for k in acc[r][0]}
                res["multistart"][key][str(r)] = per_puzzle(cat, n, B)
            pc = torch.cat(pis)
            ms = res["multistart"][key][str(cfg["R"])]
            res["multistart"][key]["path_independence"] = {
                "cos_mean": float(pc.mean()), "cos_quantiles": [float(v) for v in torch.quantile(pc, torch.tensor([0.1, 0.5, 0.9]))]}
            if esc_parts:
                agg = {}
                for sg in esc_parts[0]:
                    tot_c = sum(e[sg]["n_correct"] for e in esc_parts)
                    tot_w = sum(e[sg]["n_wrong"] for e in esc_parts)
                    wavg = lambda key, w: (sum((e[sg][key] or 0) * e[sg][w] for e in esc_parts) / max(sum(e[sg][w] for e in esc_parts), 1))
                    agg[sg] = {"keep_if_correct": wavg("keep_if_correct", "n_correct"), "keep_if_wrong": wavg("keep_if_wrong", "n_wrong"),
                               "rescued_if_wrong": wavg("rescued_if_wrong", "n_wrong"), "lost_if_correct": wavg("lost_if_correct", "n_correct"),
                               "n_correct": tot_c, "n_wrong": tot_w}
                res["escape"] = {"K": cfg["escape_k"], "endpoint_rms": ms["rms"], "by_sigma": agg}
            print(f"[multistart] noise {noise:g}: " + "  ".join(
                f"r{r}: single {res['multistart'][key][str(r)]['single']:.3f} conv1 {res['multistart'][key][str(r)]['conv1']:.3f} "
                f"vote {res['multistart'][key][str(r)]['vote']:.3f} any {res['multistart'][key][str(r)]['any']:.3f}" for r in rec[-2:])
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            tw = ms.get("twin")
            if tw:
                print(f"[multistart] noise {noise:g} twins at r{cfg['R']}: " + "  ".join(
                    f"{cls}: cos {v['cos_median']:.4f} sep {v['sep_median']:.2e} split {v['frac_split']:.2f}"
                    for cls, v in tw.items() if v), flush=True)

    if "jacobian" not in skip and jac_pool:
        carry = type(jac_pool[0][0])(**{k: torch.cat([getattr(j[0], k) for j in jac_pool]) for k in vars(jac_pool[0][0])})
        big = {k: torch.cat([j[1][k] for j in jac_pool]) for k in jac_pool[0][1]}
        exact = torch.cat([j[2] for j in jac_pool])
        idx_c, idx_w = torch.nonzero(exact)[:, 0], torch.nonzero(~exact)[:, 0]
        half = cfg["n_jac"] // 2
        idx = torch.cat([idx_c[: max(half, cfg["n_jac"] - len(idx_w))], idx_w[: max(half, cfg["n_jac"] - len(idx_c))]])[: cfg["n_jac"]]
        sel = idx.to(device)
        c_sel = type(carry)(**{k: v[sel] for k, v in vars(carry).items()})
        b_sel = {k: v[sel] for k, v in big.items()}
        mc.set_noise(inner, 0.0)
        rho, ritz = spectral(inner, c_sel, b_sel, cfg["power"], cfg["arnoldi"])
        ex_sel = exact[idx]
        top = ritz[:, :10]
        cplx = (top.imag.abs() > 1e-3 * top.abs().clamp_min(1e-30))
        def stats(mask):
            if not mask.any():
                return None
            return {"n": int(mask.sum()), "rho_median": float(rho[mask].median()), "rho_q10_q90": [float(rho[mask].quantile(0.1)), float(rho[mask].quantile(0.9))],
                    "ritz1_abs_median": float(top[mask, 0].abs().median()),
                    "top10_abs_median": [float(v) for v in top[mask].abs().median(0).values],
                    "frac_complex_top10": float(cplx[mask].float().mean()),
                    "top1_complex": float(cplx[mask, 0].float().mean())}
        res["jacobian"] = {"correct": stats(ex_sel), "wrong": stats(~ex_sel),
                           "ritz_top10": [[[float(z.real), float(z.imag)] for z in row] for row in top]}
        print(f"[jacobian] correct {res['jacobian']['correct']}\n[jacobian] wrong {res['jacobian']['wrong']}  ({time.time() - t0:.0f}s)", flush=True)

    if "trace" not in skip:
        mc.set_noise(inner, 0.0)
        nt = cfg["n_trace"]
        tb = mc.load_puzzles(data, nt, device=device)
        x = inner._input_embeddings(tb["inputs"], tb["puzzle_identifiers"]).float()
        seq = {"cos_sin": inner._cos_sin()}
        carry = mc.random_carry(inner, nt, device)
        zH, zL = carry.z_H.float(), carry.z_L.float()
        marks = [t for t in (1, 2, 4, 8, 16, 32, 64, 128, 256) if t <= cfg["trace_iters"]]
        prof = {}
        with torch.no_grad():
            for t in range(1, cfg["trace_iters"] + 1):
                steps = []
                for _ in range(inner.config.H_cycles):
                    zH, zL, st = trace_recursion(inner, zH, zL, x, seq)
                    steps += st
                if t in marks:
                    prof[str(t)] = [[name, float(v.median())] for name, v in steps]
            fp_steps, l_star, fp_steps2 = inner_fixed_point(inner, zH, zL, x, seq, iters=64)
            # distance of the scheduled L (after one recursion from the endpoint) from the frozen-H fixed point
            zH1, zL1, _ = trace_recursion(inner, zH, zL, x, seq)
        ok = (fp_steps[1:] > 1e-6) & (fp_steps[:-1] > 1e-6)                # below ~1e-6 it is float32 round-off
        ratios = torch.where(ok, fp_steps[1:] / fp_steps[:-1].clamp_min(1e-30), torch.nan)
        res["trace"] = {"profile_median": prof,
                        "inner_fp_rel_step_median": [float(v) for v in fp_steps.median(1).values[[0, 1, 2, 5, 10, 20, 40, 63]]],
                        "inner_contraction_median": float(ratios[5:40].nanmedian()) if ok[5:40].any() else 0.0,
                        # period-2 check: relative change over two inner updates (small while 1-step stays large = 2-cycle)
                        "inner_fp_rel_step2_median": [float(v) for v in fp_steps2[1:].nanmedian(1).values[[0, 1, 4, 9, 19, 39, 62]]],
                        "inner_2cycle_frac": float(((fp_steps[-1] > 1e-3) & (fp_steps2[-1] < 0.1 * fp_steps[-1])).float().mean()),
                        "inner_fixed_frac": float((fp_steps[-1] < 1e-4).float().mean()),
                        "scheduled_L_vs_inner_fp": float(((zL - l_star).flatten(1).norm(dim=1) / l_star.flatten(1).norm(dim=1)).median())}
        print(f"[trace] last profile {prof[str(marks[-1])][:8]} ... inner contraction {res['trace']['inner_contraction_median']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    if "pr" not in skip:
        mc.set_noise(inner, 0.0)
        npr = cfg["n_pr"]
        pb = mc.load_puzzles(data, npr, device=device)
        carry = mc.random_carry(inner, npr, device)
        with torch.no_grad():
            for _ in range(min(cfg["R"], 64)):
                carry, _, _ = inner(carry, pb)
        res["pr"] = {k: participation_ratio(v.float().reshape(-1, v.shape[-1]).cpu()) for k, v in mc.stream_views(inner, carry).items()}
        print(f"[pr] {res['pr']}", flush=True)

    res["seconds"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[done] {out_path} in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
