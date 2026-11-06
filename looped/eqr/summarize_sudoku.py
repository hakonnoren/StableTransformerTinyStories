"""Collect the Sudoku-Extreme runs of jobs/looped/sudoku_eqr.slurm into one table.

    python looped/eqr/summarize_sudoku.py <RUN_ROOT> [--out table.md]

RUN_ROOT is the job's OUTPUT_ROOT (contains outputs/<run id>/<date>/<time>/). For each run
it reads the evaluate.py results written under eval_preds/:

  full_D16       whole test set (423k puzzles), 16 iterations: the headline cell/exact accuracy
  sub_D{16,64,256}  the first SUB_BATCHES test batches at 16/64/256 iterations, with predictions

and from the subset predictions computes, per puzzle (exact = every cell right):
  O(16->r)   = P(not exact at r | exact at 16)       (overthinking: correct answers lost)
  dy(16->r)  = P(prediction at r != prediction at 16) (any cell changed)
Aggregates mean ± sd over seeds per arm (the run name is <arm>_s<seed>).
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch

ARM_ORDER = ["eqr", "kick_fixed", "kick_learned", "relax3", "relax3_leak1", "relax3_leak4", "kick_pos",
             "relax3_learned", "kick_learned_ch"]


def _metrics(run_dir, tag):
    hits = glob.glob(os.path.join(run_dir, "eval_preds", f"step_*_{tag}_*", "eval_metrics_step_*.json"))
    if not hits:
        return None, None
    path = max(hits, key=os.path.getmtime)
    return json.load(open(path))["metrics"], os.path.dirname(path)


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
    row["rho"] = (d.get("rho") or {}).get("median")
    rates = d.get("rates_by_step") or []
    if rates:
        last = rates[-1]
        row["gamma"], row["alpha"] = last["gamma"]["mean"], last["alpha"]["mean"]
        row["delta"] = (last.get("delta") or {}).get("mean")
        row["lin_radius"] = last["linear_radius"]
        row["rates_by_step"] = [(x["step"], round(x["gamma"]["mean"], 4), round(x["alpha"]["mean"], 4)) for x in rates]


def summarize_run(run_dir):
    row = {"run": _run_name(run_dir)}
    _dynamics(run_dir, row)
    m, _ = _metrics(run_dir, "full_D16")
    if m:
        row["cell@16"], row["exact@16"] = m.get("all/accuracy"), m.get("all/exact_accuracy")
    sub = {}
    for r in (16, 64, 256):
        m, d = _metrics(run_dir, f"sub_D{r}")
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
         "stop@256", "rho", "gamma", "alpha", "delta", "lin_radius"]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = sorted({os.path.dirname(os.path.dirname(c)) for c in
                   glob.glob(os.path.join(a.root, "outputs", "*", "*", "*", "eval_preds", "*"))}
                  | {os.path.dirname(c) for c in
                     glob.glob(os.path.join(a.root, "outputs", "*", "*", "*", "dynamics.json"))})
    rows = [summarize_run(r) for r in runs]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[re.sub(r"_s\d+$", "", r["run"])].append(r)
    arms = [x for x in ARM_ORDER if x in by_arm] + sorted(set(by_arm) - set(ARM_ORDER))
    lines = ["| arm | n | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for arm in arms:
        rs = by_arm[arm]
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(_fmt([r.get(c) for r in rs]) for c in COLS) + " |")
    text = "\n".join(lines)
    text += ("\n\ncell/exact@16: whole test set; columns ending in s, resid and O/dy: the first SUB_BATCHES test "
             "batches. O16->r: share of puzzles exact at 16 iterations that are wrong at r. dy16->r: share "
             "whose prediction changed. resid@r: EqR's residual_of_r_steps (norm of the state change at "
             "iteration r; not scale-normalised).\n")
    if any("rho" in r or "FP@256" in r for r in rows):
        dl = ["", "Dynamics (looped/eqr/dynamics_eqr.py; noise off, first 768 test puzzles; rho on 32):", "",
              "| arm | n | " + " | ".join(DCOLS) + " |", "|" + "---|" * (len(DCOLS) + 2)]
        for arm in arms:
            rs = by_arm[arm]
            dl.append(f"| {arm} | {len(rs)} | " + " | ".join(_fmt([r.get(c) for r in rs]) for c in DCOLS) + " |")
        dl += ["", "nf_exact: exact accuracy without noise. FP@256: share of puzzles with relative step "
               "|s_r - s_{r-1}|/|s_r| < 1e-3 at r = 256 (also at 1e-2); cyc2: period-2 instead. step50: median "
               "relative step at 256. O16->256nf: overthinking without noise. stop: share where the halting head "
               "says stop (q > 0). rho: median spectral radius of one iteration at r = 256 (full, not shift-free). "
               "gamma/alpha/delta: final learned or fixed rates; lin_radius: radius of one iteration's linear part "
               "(forces zero).", ""]
        text += "\n".join(dl)
    print(text)
    for r in rows:
        print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
