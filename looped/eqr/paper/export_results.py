"""Export seed-resolved results of pulled run roots into data/*.csv for this note.

    python export_results.py ~/sudoku_results                                     # Sudoku: data/runs_*.csv
    python export_results.py ~/skyscrapers_results/skyscrapers*_* --prefix sky_runs --puzzles-from-root
                                                                                  # Skyscrapers-7: data/sky_runs_*.csv

Uses summarize_sudoku.py's readers, one row per (run directory, checkpoint step):
  data/<prefix>_main.csv      evaluate.py at the training noise 0.01, breadth 1 (full_D16, sub_D{16,64,256})
  data/<prefix>_scaling.csv   sudoku_scaling / eqr_scaling.sh: depth x breadth x test noise, first 2048 test puzzles
  data/<prefix>_dynamics.csv  dynamics_eqr.py (noise off, 768 puzzles, spectral radius on 32)
With --puzzles-from-root, a column `puzzles` holds the training-set size read from the run root's name
(skyscrapers100k_<job> -> 100000), since Skyscrapers runs of different sizes share arm names.
"""
import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # looped/eqr
import summarize_sudoku as S                       # noqa: E402

MAIN = ["cell@16", "exact@16", "exact@64s", "exact@256s", "resid@64", "resid@256"]
SCALE = ["D", "B", "N", "single", "conv1", "convk", "vote", "any", "cell"]
DYN = ["nf_exact@16", "nf_exact@256", "FP@256", "FP@256(1e-2)", "cyc2@256", "step50@256", "stop@256", "rho", "twin_cos@256"]


def arm_seed(name):
    m = re.match(r"(.*)_s(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, "")


def puzzles_of(root):
    m = re.search(r"(\d+)k_", os.path.basename(os.path.normpath(root)) + "_")
    if not m:
        raise SystemExit(f"cannot read a training-set size from {root}")
    return int(m.group(1)) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--prefix", default="runs")
    ap.add_argument("--puzzles-from-root", action="store_true")
    a = ap.parse_args()
    lead = ["arm", "seed", "step"] + (["puzzles"] if a.puzzles_from_root else [])
    main_rows, scale_rows, dyn_rows = [], [], []
    for root in a.roots:
        extra = [puzzles_of(root)] if a.puzzles_from_root else []
        runs = {os.path.dirname(os.path.dirname(c)) for c in glob.glob(os.path.join(root, "outputs", "*", "*", "*", "eval_preds", "*"))}
        runs |= {os.path.dirname(c) for c in glob.glob(os.path.join(root, "outputs", "*", "*", "*", "dynamics.json"))}
        for r in sorted(runs):
            steps = S._run_steps(r)
            for s in steps:
                row = S.summarize_run(r, s, steps[-1])
                arm, seed = arm_seed(row["run"])
                key = [arm, seed, row["step"]] + extra
                if any(row.get(c) is not None for c in MAIN):
                    main_rows.append(key + [row.get(c) for c in MAIN])
                for x in row.get("scaling", []):
                    scale_rows.append(key + [x.get(c) for c in SCALE])
                if any(row.get(c) is not None for c in DYN):
                    dyn_rows.append(key + [row.get(c) for c in DYN])
    out = os.path.join(HERE, "data")
    nk = len(lead)
    for name, cols, rows in (("main", MAIN, main_rows), ("scaling", SCALE, scale_rows), ("dynamics", DYN, dyn_rows)):
        name = f"{a.prefix}_{name}"
        with open(os.path.join(out, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(lead + cols)
            for row in sorted(rows, key=lambda t: [str(v) for v in t[:3]] + [str(v) for v in t[nk:nk + 3]] + [str(v) for v in t[3:nk]]):
                w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in row])
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
