"""Seed-aggregated table for run_looped.py results (summarize.py shows one seed per configuration).

    python looped/seeds_table.py results/looped/listops results/looped/khop_K3 [--only name1 name2 ...]

For every configuration (file stem without the seed): number of seeds, accuracy at the largest training r and at
r = 128, the share of answers correct at r_hi that are wrong at 128, the step ||s_128 - s_127|| and the
shift-free spectral radius at r = 128 (newer runs only); mean ± s.d. over seeds, with per-seed values.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import summarize   # labels


def collect(folder):
    runs = defaultdict(dict)
    for f in sorted(Path(folder).glob("*_seed*.json")):
        name, seed = f.stem.rsplit("_seed", 1)
        r = json.loads(f.read_text())
        t, hi = r["final"]["test"], r["args"]["r_hi"]
        sr = t.get("spectral_radius", {})
        runs[name][int(seed)] = dict(acc_hi=t["acc"][str(hi)], acc128=t["acc"]["128"],
                                     over=t[f"overthink_{hi}_to_128"], step=t["res_abs"]["128"],
                                     rho=sr.get("128q", {}).get("median", np.nan))
    return runs


def table(runs, only=None):
    cols = [("acc_hi", "acc @r_hi", ".3f"), ("acc128", "acc @128", ".3f"), ("over", "correct @r_hi → wrong @128 (%)", ".1f"),
            ("step", "step @128", ".2g"), ("rho", "ρ shift-free @128", ".3f")]
    lines = ["| model | seeds | " + " | ".join(c for _, c, _ in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name in sorted(runs, key=lambda n: (summarize.ORDER.index(n) if n in summarize.ORDER else 99, n)):
        if only and name not in only:
            continue
        seeds = sorted(runs[name])
        cells = []
        for k, _, fmt in cols:
            v = np.array([runs[name][s][k] for s in seeds], float)
            ok = v[np.isfinite(v)]
            if not len(ok):
                cells.append("–"); continue
            cell = format(ok.mean(), fmt) + (f" ± {format(ok.std(), fmt)}" if len(ok) > 1 else "")
            if len(seeds) > 1:
                cell += " (" + " / ".join(format(x, fmt) for x in v) + ")"
            cells.append(cell)
        lines.append(f"| {summarize.label(name)} | {','.join(map(str, seeds))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    for d in a.folders:
        print(f"## {Path(d).name}\n\n{table(collect(d), a.only)}\n")


if __name__ == "__main__":
    main()
