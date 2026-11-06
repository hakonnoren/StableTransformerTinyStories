"""Tables from analyze_maps.py JSON files (one per checkpoint).

    python looped/eqr/summarize_maps.py <dir with *.json> [--r 256] [--noises 0,0.5]
"""
import argparse
import glob
import json
import os

ORDER = ["eqr-released", "eqr", "relax3_leak1", "relax3_leak4", "relax3", "kick_pos", "kick_learned", "kick_fixed",
         "relax3_leak1_injL", "relax3_leak1_injL_self", "kick_pos_injL", "relax3_leak2", "relax3_leak3",
         "twinL_leak1", "twinL_leak2", "innerL_self_leak1"]


def f(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else str(v)
    return str(v)


def name_of(path):
    base = os.path.basename(path).split(".")[0]
    arm, _, step = base.partition("__step")
    arm = arm[:-3] if arm.endswith("_s0") else arm
    return arm, int(step) if step.isdigit() else -1


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--r", default=None, help="recorded iteration for the multistart tables (default: the largest)")
    ap.add_argument("--noises", default=None)
    a = ap.parse_args()
    res = []
    for p in glob.glob(os.path.join(a.dir, "*.json")):
        arm, step = name_of(p)
        res.append((arm, step, json.load(open(p))))
    rank = lambda arm: ORDER.index(arm) if arm in ORDER else len(ORDER)
    res.sort(key=lambda t: (rank(t[0]), t[0], t[1]))
    lab = lambda arm, step: f"{arm} {step // 1000}k"

    # 1. rates, scale, dimensionality
    rows = []
    for arm, step, r in res:
        rt = r.get("rates") or {}
        ms0 = (r.get("multistart") or {}).get("noise_0", {})
        rlast = max((int(k) for k in ms0 if k.isdigit()), default=None)
        rms = ms0.get(str(rlast), {}).get("rms", {}) if rlast else {}
        rows.append([lab(arm, step), f(rt.get("gamma")), f(rt.get("alpha")), f(rt.get("delta")), f(rt.get("merged_retention")),
                     f(rt.get("floor_geomean_abs_eig")), " / ".join(f(v) for v in rms.values()),
                     " / ".join(f(v, 1) for v in (r.get("pr") or {}).values())])
    print("## Rates, state size, dimensionality\n")
    print(table(["arm", "gamma", "alpha", "delta", "merged retention", "volume floor / it", "RMS H / L (/L')", "PR H / L"], rows))

    # 2. multistart
    noises = None if a.noises is None else [f"noise_{float(x):g}" for x in a.noises.split(",")]
    keys = sorted({k for _, _, r in res for k in (r.get("multistart") or {})}, key=lambda k: float(k.split("_")[1]))
    for nk in (noises or keys):
        rows = []
        for arm, step, r in res:
            ms = (r.get("multistart") or {}).get(nk)
            if not ms:
                continue
            rl = a.r or str(max(int(k) for k in ms if k.isdigit()))
            m = ms.get(rl)
            if not m:
                continue
            au = m["auroc"]
            pi = ms.get("path_independence", {})
            rows.append([lab(arm, step), f(m["single"]), f(m["any"]), f(m["vote"]), f(m["conv1"]), f(m["sres1"]), f(m["q1"]),
                         f(m["verifier"]), f(m["distinct_mean"], 2), f(m["frac_puzzles_mixed"]), f(m["delta_pi"]),
                         f(pi.get("cos_mean")), f(au["lscore"]), f(au["sres"]), f(au["neg_q"]),
                         f(m.get("outer_fixed")), f(m.get("outer_2cycle"))])
        print(f"\n## Multistart, {nk.replace('_', ' ')}, iteration {a.r or 'last'} (B random starts per puzzle)\n")
        print(table(["arm", "single", "any", "vote", "conv1", "step1", "q1", "verifier", "distinct", "mixed", "ΔPI",
                     "endpoint cos", "AUROC conv", "AUROC step", "AUROC q", "fixed pt", "2-cycle"], rows))
    print("\nsingle: mean exact of one start. any / vote: some start right / plurality right. conv1: EqR's most-converged "
          "(logit change); step1: smallest relative state step; q1: largest halting logit; verifier: a start that is a valid "
          "Sudoku agreeing with the givens (label-free). distinct: mean number of different answers over the starts; mixed: "
          "share of puzzles with some but not all starts right; ΔPI: EqR's path-independence gap; endpoint cos: mean pairwise "
          "cosine of the final states of one puzzle's starts. AUROC: how well the score separates right from wrong runs.")

    # 3. jacobian
    rows = []
    for arm, step, r in res:
        j = r.get("jacobian")
        if not j:
            continue
        cell = lambda d, k: f(d.get(k)) if d else "–"
        c, w = j.get("correct"), j.get("wrong")
        rows.append([lab(arm, step), cell(c, "n"), cell(c, "rho_median"), cell(c, "frac_complex_top10"), cell(c, "top1_complex"),
                     cell(w, "n"), cell(w, "rho_median"), cell(w, "frac_complex_top10"), cell(w, "top1_complex")])
    print("\n## Jacobian of one outer iteration at noise-free endpoints\n")
    print(table(["arm", "n right", "rho right", "complex top10", "top1 complex", "n wrong", "rho wrong", "complex top10", "top1 complex"], rows))

    # 4. escape
    sig = sorted({s for _, _, r in res for s in ((r.get("escape") or {}).get("by_sigma") or {})}, key=float)
    rows = []
    for arm, step, r in res:
        e = (r.get("escape") or {}).get("by_sigma")
        if not e:
            continue
        rows.append([lab(arm, step)] + [f"{f(e[s]['keep_if_correct'], 2)} / {f(e[s]['rescued_if_wrong'], 2)}" if s in e else "–" for s in sig])
    print("\n## Escape: K noisy iterations from the noise-free endpoints, P(keep | right) / P(rescued | wrong)\n")
    print(table(["arm"] + [f"σ={s}" for s in sig], rows))

    # 5. trace
    rows = []
    for arm, step, r in res:
        t = r.get("trace")
        if not t:
            continue
        prof = t["profile_median"]
        first, last = prof[min(prof, key=int)], prof[max(prof, key=int)]
        short = lambda p: " ".join(f"{n}{v:.0e}" for n, v in p[: len(p) // max(1, (r['arch'].get('H_cycles') or 3))])
        rows.append([lab(arm, step), f(t.get("inner_contraction_median")), f(t.get("inner_fixed_frac")), f(t.get("inner_2cycle_frac")),
                     f(t.get("scheduled_L_vs_inner_fp")), short(first), short(last)])
    print("\n## Within one recursion (median relative change per update; first and last traced outer iteration)\n")
    print(table(["arm", "inner contraction", "inner fixed", "inner 2-cycle", "L vs inner fixed point", "first iteration", "last iteration"], rows))


if __name__ == "__main__":
    main()
