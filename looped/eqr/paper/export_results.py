"""Export seed-resolved results of a pulled Sudoku run root into data/*.csv for this note.

    python export_results.py ~/sudoku_results

Uses summarize_sudoku.py's readers, one row per (run directory, checkpoint step):
  data/runs_main.csv      evaluate.py at the training noise 0.01, breadth 1 (full_D16, sub_D{16,64,256})
  data/runs_scaling.csv   sudoku_scaling / eqr_scaling.sh: depth x breadth x test noise, first 2048 test puzzles
  data/runs_dynamics.csv  dynamics_eqr.py (noise off, 768 puzzles, spectral radius on 32)
"""
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


def main():
    root = sys.argv[1]
    runs = {os.path.dirname(os.path.dirname(c)) for c in glob.glob(os.path.join(root, "outputs", "*", "*", "*", "eval_preds", "*"))}
    runs |= {os.path.dirname(c) for c in glob.glob(os.path.join(root, "outputs", "*", "*", "*", "dynamics.json"))}
    main_rows, scale_rows, dyn_rows = [], [], []
    for r in sorted(runs):
        steps = S._run_steps(r)
        for s in steps:
            row = S.summarize_run(r, s, steps[-1])
            arm, seed = arm_seed(row["run"])
            key = [arm, seed, row["step"]]
            if any(row.get(c) is not None for c in MAIN):
                main_rows.append(key + [row.get(c) for c in MAIN])
            for x in row.get("scaling", []):
                scale_rows.append(key + [x.get(c) for c in SCALE])
            if any(row.get(c) is not None for c in DYN):
                dyn_rows.append(key + [row.get(c) for c in DYN])
    out = os.path.join(HERE, "data")
    for name, cols, rows in (("runs_main", MAIN, main_rows), ("runs_scaling", SCALE, scale_rows), ("runs_dynamics", DYN, dyn_rows)):
        with open(os.path.join(out, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "seed", "step"] + cols)
            for row in sorted(rows, key=lambda t: [str(v) for v in t[:3]] + [str(v) for v in t[3:6]]):
                w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in row])
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
