"""Collect the Sudoku-Extreme runs of jobs/looped/sudoku_eqr.slurm into one table.

    python looped/eqr/summarize_sudoku.py <RUN_ROOT> [<RUN_ROOT> ...] [--out table.md] [--scaling-only]

RUN_ROOT is the job's OUTPUT_ROOT (contains outputs/<run id>/<date>/<time>/). For each run
directory it reads the evaluate.py results written under eval_preds/. A run extended past its
first budget (sudoku_eqr.slurm resume) has one directory per segment; rows are keyed by
(arm, checkpoint step), so the 20k and 50k evaluations of one run are separate rows.

  full_D16       whole test set (423k puzzles), 16 iterations: the headline cell/exact accuracy
  sub_D{16,64,256}  the first SUB_BATCHES test batches at 16/64/256 iterations, with predictions

and from the subset predictions computes, per puzzle (exact = every cell right):
  O(16->r)   = P(not exact at r | exact at 16)       (overthinking: correct answers lost)
  dy(16->r)  = P(prediction at r != prediction at 16) (any cell changed)
Aggregates mean ± sd over seeds per arm (the run name is <arm>_s<seed>).

Scaling table (jobs/looped/sudoku_scaling.slurm: EqR's depth x breadth eval, tags
scale_D<depth>_B<breadth>; one row per depth, breadth and test-time noise, the noise read from
evaluate.py's directory label ..._arch_noise_scale-<noise>_...): single-run exact, the
most-converged run (conv1), the mean of the top-k most-converged runs (conv4, EqR's headline),
majority vote and any-run-correct.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch

ARM_ORDER = ["eqr-released", "eqr", "kick_fixed", "kick_learned", "relax3", "relax3_leak4", "relax3_leak1",
             "relax3_leak2", "relax3_leak3", "kick_pos", "relax3_leak1_injL", "relax3_leak1_injL_self",
             "kick_pos_injL", "twinL_leak1", "twinL_leak2", "innerL_self_leak1",
             "twinL_leak3", "twinL_leak10", "twinL_leak2_noise25", "eqr_noise25",
             "relax3_learned", "kick_learned_ch"]


def _metrics(run_dir, tag, step=None):
    s = "*" if step is None else str(step)
    hits = glob.glob(os.path.join(run_dir, "eval_preds", f"step_{s}_{tag}_*", "eval_metrics_step_*.json"))
    if not hits:
        return None, None
    path = max(hits, key=os.path.getmtime)
    return json.load(open(path))["metrics"], os.path.dirname(path)


def _run_step(run_dir):
    """Checkpoint step this run directory was evaluated at (the highest, if several)."""
    steps = [int(m.group(1)) for d in glob.glob(os.path.join(run_dir, "eval_preds", "step_*"))
             if (m := re.match(r"step_(\d+)_", os.path.basename(d)))]
    if steps:
        return max(steps)
    f = os.path.join(run_dir, "dynamics.json")
    if os.path.exists(f):
        return json.load(open(f)).get("step")
    return None


def _scaling(run_dir, step):
    """EqR depth x breadth evals (sudoku_scaling.slurm) of one run directory at one step."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "eval_preds", f"step_{step}_scale_D*_B*_*", "eval_metrics_step_*.json")):
        m = re.search(r"scale_D(\d+)_B(\d+)_", path)
        if not m:
            continue
        x = json.load(open(path))["metrics"]
        b = int(m.group(2))
        nm = re.search(r"noise_scale-([0-9.]+(?:e-?[0-9]+)?)(?:_|/|$)", path)
        out.append({"D": int(m.group(1)), "B": b, "N": float(nm.group(1)) if nm else None,
                    "single": x.get("all/exact_accuracy"), "cell": x.get("all/accuracy"),
                    "conv1": x.get("convergence_top_k/cumulative_exact_acc_top1") if b > 1 else None,
                    "convk": x.get("convergence_top_k/exact_accuracy") if b > 1 else None,
                    "vote": x.get("majority_vote/exact_accuracy") if b > 1 else None,
                    "any": x.get("different_init/any_correct") if b > 1 else None,
                    "k": x.get("convergence_top_k/k"), "n": x.get("majority_vote/num_samples")})
    return out


def _preds(eval_dir):
    f = os.path.join(eval_dir, "all_preds.0")
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu", weights_only=False)
    return d.get("preds"), d.get("labels")


def _run_name(run_dir):
    cfg = os.path.join(run_dir, "all_config.yaml")
    if os.path.exists(cfg):
        m = re.search(r"^run_name:\s*(\S+)", open(cfg).read(), re.M)
        if m:
            return m.group(1)
    return os.path.basename(os.path.dirname(os.path.dirname(run_dir)))


def _dynamics(run_dir, row):
    f = os.path.join(run_dir, "dynamics.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    t = d.get("trajectory") or {}
    get = lambda grp, k: (t.get(grp) or {}).get(str(k))
    row["nf_exact@16"], row["nf_exact@256"] = get("exact", 16), get("exact", 256)
    row["FP@256"], row["cyc2@256"] = get("fixed_point", "256@0.001"), get("cycle2", "256@0.001")
    row["FP@256(1e-2)"] = get("fixed_point", "256@0.01")
    q = (t.get("step_quantiles") or {}).get("256")
    row["step50@256"] = q[1] if q else None
    row["O16->256nf"] = get("overthink", 256)
    row["stop@256"] = get("q_stop", 256)
    row["twin_cos@256"] = get("twin_cos", 256)
    row["rho"] = (d.get("rho") or {}).get("median")
    rates = d.get("rates_by_step") or []
    if rates:
        last = rates[-1]
        row["gamma"], row["alpha"] = last["gamma"]["mean"], last["alpha"]["mean"]
        row["delta"] = (last.get("delta") or {}).get("mean")
        row["lin_radius"] = last["linear_radius"]
        row["rates_by_step"] = [(x["step"], round(x["gamma"]["mean"], 4), round(x["alpha"]["mean"], 4)) for x in rates]


def summarize_run(run_dir):
    step = _run_step(run_dir)
    row = {"run": _run_name(run_dir), "step": step, "dir": run_dir}
    _dynamics(run_dir, row)
    row["scaling"] = _scaling(run_dir, step) if step is not None else []
    m, _ = _metrics(run_dir, "full_D16", step)
    if m:
        row["cell@16"], row["exact@16"] = m.get("all/accuracy"), m.get("all/exact_accuracy")
    sub = {}
    for r in (16, 64, 256):
        m, d = _metrics(run_dir, f"sub_D{r}", step)
        if m is None:
            continue
        row[f"cell@{r}s"], row[f"exact@{r}s"] = m.get("all/accuracy"), m.get("all/exact_accuracy")
        row[f"resid@{r}"] = m.get(f"all/residual_of_{r}_steps")
        sub[r] = _preds(d)
    if 16 in sub and sub[16] and sub[16][0] is not None:
        p16, lab = sub[16]
        ok16 = (p16 == lab).all(-1)
        for r in (64, 256):
            if r in sub and sub[r] and sub[r][0] is not None:
                pr = sub[r][0]
                n = min(len(p16), len(pr))
                okr = (pr[:n] == lab[:n]).all(-1)
                k = ok16[:n]
                row[f"O16->{r}"] = float((k & ~okr).sum()) / max(int(k.sum()), 1)
                row[f"dy16->{r}"] = float((pr[:n] != p16[:n]).any(-1).float().mean())
    return row


DCOLS = ["nf_exact@16", "nf_exact@256", "FP@256", "FP@256(1e-2)", "cyc2@256", "step50@256", "O16->256nf",
         "stop@256", "rho", "gamma", "alpha", "delta", "lin_radius", "twin_cos@256"]

COLS = ["cell@16", "exact@16", "cell@64s", "exact@64s", "cell@256s", "exact@256s",
        "resid@64", "resid@256", "O16->64", "O16->256", "dy16->256"]


def _fmt(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "–"
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mu:.3g}"
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{mu:.3g}±{sd:.2g}"


SCOLS = ["single", "conv1", "convk", "vote", "any", "cell"]


def _scaling_table(arms, by_key):
    rows = []
    for key in arms:
        cells = defaultdict(list)
        for r in by_key[key]:
            for x in r.get("scaling", []):
                cells[(x["D"], x["B"], x.get("N"))].append(x)
        for (d, b, nz) in sorted(cells, key=lambda t: (t[0], t[1], -1.0 if t[2] is None else t[2])):
            xs = cells[(d, b, nz)]
            rows.append(f"| {key[0]} | {_step_str(key[1])} | {d} | {b} | {'?' if nz is None else f'{nz:g}'} | {len(xs)} | "
                        + " | ".join(_fmt([x.get(c) for x in xs]) for c in SCOLS) + " |")
    if not rows:
        return ""
    head = ["", "Depth x breadth (jobs/looped/sudoku_scaling.slurm: EqR's evaluate.py, first 2048 test puzzles by "
            "default; noise = test-time Langevin noise std, added after every update):", "",
            "| arm | step | depth | breadth | noise | n | " + " | ".join(SCOLS) + " |", "|" + "---|" * (len(SCOLS) + 6)]
    tail = ["", "single: exact accuracy of one noisy run (mean over the breadth). conv1: the most-converged run "
            "(smallest logit change over the last 3 iterations). convk: mean of the top-k most-converged (k = 4; "
            "EqR's headline, 98.60 for its released checkpoint at depth 64, breadth 128). vote: majority vote. "
            "any: some run correct. cell: per-cell accuracy of one run.", ""]
    return "\n".join(head + rows + tail)


def _step_str(step):
    return "–" if step is None else (f"{step // 1000}k" if step % 1000 == 0 else str(step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scaling-only", action="store_true")
    a = ap.parse_args()
    runs = set()
    for root in a.roots:
        runs |= {os.path.dirname(os.path.dirname(c)) for c in
                 glob.glob(os.path.join(root, "outputs", "*", "*", "*", "eval_preds", "*"))}
        runs |= {os.path.dirname(c) for c in glob.glob(os.path.join(root, "outputs", "*", "*", "*", "dynamics.json"))}
    rows = [summarize_run(r) for r in sorted(runs)]
    by_key = defaultdict(list)                      # (arm, checkpoint step) -> runs (seeds)
    for r in rows:
        by_key[(re.sub(r"_s\d+$", "", r["run"]), r["step"])].append(r)
    rank = lambda arm: ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
    keys = sorted(by_key, key=lambda k: (rank(k[0]), k[0], k[1] if k[1] is not None else -1))
    scaling = _scaling_table(keys, by_key)
    if a.scaling_only:
        print(scaling or "no scaling evals found")
        return
    main_keys = [k for k in keys if any(any(c in r for c in COLS) for r in by_key[k])]
    lines = ["| arm | step | n | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 3)]
    for key in main_keys:
        rs = by_key[key]
        lines.append(f"| {key[0]} | {_step_str(key[1])} | {len(rs)} | "
                     + " | ".join(_fmt([r.get(c) for r in rs]) for c in COLS) + " |")
    text = "\n".join(lines)
    text += ("\n\ncell/exact@16: whole test set; columns ending in s, resid and O/dy: the first SUB_BATCHES test "
             "batches. O16->r: share of puzzles exact at 16 iterations that are wrong at r. dy16->r: share "
             "whose prediction changed. resid@r: EqR's residual_of_r_steps (norm of the state change at "
             "iteration r; not scale-normalised).\n")
    if any("rho" in r or "FP@256" in r for r in rows):
        dkeys = [k for k in keys if any("rho" in r or "FP@256" in r for r in by_key[k])]
        dl = ["", "Dynamics (looped/eqr/dynamics_eqr.py; noise off, first 768 test puzzles; rho on 32):", "",
              "| arm | step | n | " + " | ".join(DCOLS) + " |", "|" + "---|" * (len(DCOLS) + 3)]
        for key in dkeys:
            rs = by_key[key]
            dl.append(f"| {key[0]} | {_step_str(key[1])} | {len(rs)} | "
                      + " | ".join(_fmt([r.get(c) for r in rs]) for c in DCOLS) + " |")
        dl += ["", "nf_exact: exact accuracy without noise. FP@256: share of puzzles with relative step "
               "|s_r - s_{r-1}|/|s_r| < 1e-3 at r = 256 (also at 1e-2); cyc2: period-2 instead. step50: median "
               "relative step at 256. O16->256nf: overthinking without noise. stop: share where the halting head "
               "says stop (q > 0). rho: median spectral radius of one iteration at r = 256 (full, not shift-free). "
               "gamma/alpha/delta: final learned or fixed rates; lin_radius: radius of one iteration's linear part "
               "(forces zero). twin_cos: twin_l arms, mean cosine between the twins L and L' at r = 256 "
               "(1 = merged).", ""]
        text += "\n".join(dl)
    text += scaling
    print(text)
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k not in ("scaling", "dir")}))
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
