"""One table for a multihop.slurm run root: curriculum progress, depth scaling, noise x breadth, halting, overthinking.

    python looped/multihop/summarize.py /cluster/work/$USER/multihop_runs/<job> [--out table.md]

Per run directory (arm_sS) and per evaluated checkpoint (eval_*.json):
  hop        the curriculum's hop level at the end (41 = all 40 reached) and the steps at which 30 / 35 / 40 were reached
  acc a-b@d  mean exact accuracy over the hop levels a..b at d iterations (noise-free), with the best depth in brackets
  noise      at the longest breadth depth, the best (sigma, selection) over the noise section's grid, per hop group
  halt       the authors' adaptive halting, mean accuracy over hops 21-40 and 41-60, and mean iterations
  lost       of the answers right at 8 iterations, the share wrong at 128 (hops <= 20; the October note's column)
  gap        training-set samples minus test facts, mean over k <= 40, at 16 iterations
"""
import argparse
import glob
import json
import os

import numpy as np

GROUPS = [(2, 20), (21, 30), (31, 40), (41, 50), (51, 70), (71, 100)]


def curriculum(d):
    f = os.path.join(d, "curriculum.jsonl")
    if not os.path.exists(f):
        return {}
    recs = [json.loads(x) for x in open(f) if x.strip()]
    reached = {r["hop"]: r["step"] for r in recs if r.get("advance")}
    return {"hop": recs[-1]["hop"] + (1 if recs[-1].get("advance") else 0), "reached": reached, "step": recs[-1]["step"]}


def group_acc(res, lo, hi):
    D = res["depths"]
    acc = res["depth"]["acc"]
    ks = [k for k in acc if lo <= int(k) <= hi]
    if not ks:
        return None, None
    per_d = [np.mean([acc[k][i] for k in ks]) for i in range(len(D))]
    i_best = int(np.argmax(per_d))
    return per_d, D[i_best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["| run | ckpt | hop (30/35/40 at step) | " + " | ".join(f"acc {lo}-{hi} @16 / @64 / best" for lo, hi in GROUPS)
             + " | noise best, 31-40 / 41-50 / 51-70 | halt 21-40 / 41-60 (it.) | lost 8->128 | gap |",
             "|" + "---|" * (len(GROUPS) + 7)]
    for d in sorted(glob.glob(os.path.join(a.root, "*_s*"))):
        cur = curriculum(d)
        hop = "–" if not cur else (f"{cur['hop']} ({'/'.join(str(cur['reached'].get(k, '–')) for k in (30, 35, 40))})")
        for ev in sorted(glob.glob(os.path.join(d, "eval_*.json"))):
            res = json.load(open(ev))
            cells = []
            for lo, hi in GROUPS:
                per_d, best_d = group_acc(res, lo, hi) if "depth" in res else (None, None)
                if per_d is None:
                    cells.append("–"); continue
                D = res["depths"]
                g = lambda dd: f"{per_d[D.index(dd)]:.2f}" if dd in D else "–"
                cells.append(f"{g(16)} / {g(64)} / {max(per_d):.2f} ({best_d})")
            nz = "–"
            if "noise" in res:
                dd = str(max(res["noise"]["depths"]))
                parts = []
                for lo, hi in ((31, 40), (41, 50), (51, 70)):
                    best = None
                    for s, per in res["noise"]["per_sigma"].items():
                        ks = [k for k in per if lo <= int(k) <= hi]
                        if not ks:
                            continue
                        for sel in ("single", "vote", "conv", "any"):
                            v = np.mean([per[k][dd][sel] for k in ks])
                            if sel != "any" and (best is None or v > best[0]):
                                best = (v, s, sel)
                    parts.append("–" if best is None else f"{best[0]:.2f} ({best[1]} {best[2]})")
                nz = " / ".join(parts)
            hl = "–"
            if "halting" in res:
                h = res["halting"]["per_hop"]
                f = lambda lo, hi: ([h[k]["acc"] for k in h if lo <= int(k) <= hi], [h[k]["iters"] for k in h if lo <= int(k) <= hi])
                (a1, i1), (a2, i2) = f(21, 40), f(41, 60)
                hl = f"{np.mean(a1):.2f} ({np.mean(i1):.0f}) / " + (f"{np.mean(a2):.2f} ({np.mean(i2):.0f})" if a2 else "–")
            lost = res.get("overthink", {}).get("8->128 k<=20")
            lost = "–" if not lost else f"{100 * lost['lost']:.1f}%"
            gap = "–"
            if "seen" in res and "depth" in res and 16 in res["seen"]["depths"] and 16 in res["depths"]:
                i_s, i_t = res["seen"]["depths"].index(16), res["depths"].index(16)
                ks = [k for k in res["seen"]["acc"] if k in res["depth"]["acc"]]
                gap = f"{np.mean([res['seen']['acc'][k][i_s] - res['depth']['acc'][k][i_t] for k in ks]):+.3f}"
            lines.append(f"| {os.path.basename(d)} | {os.path.basename(ev)[5:-5]} | {hop} | " + " | ".join(cells)
                         + f" | {nz} | {hl} | {lost} | {gap} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
