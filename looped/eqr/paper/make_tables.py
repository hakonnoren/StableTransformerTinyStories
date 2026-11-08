"""LaTeX tables for the Sudoku map-analysis note (writes tables/*.tex from data/)."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))
import tables as T  # noqa: E402

OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

NAME = {"eqr": "EqR", "eqr-released": "EqR (released)", "relax3_leak1": r"relax, $\delta=1$", "kick_pos": "kick, positive rates",
        "relax3_leak4": r"relax, $\delta=\tfrac14$", "relax3": r"relax, $\delta=0$", "kick_learned": "kick, learned rates",
        "kick_fixed": "kick, fixed rates", "relax3_leak1_injL": r"relax, $\delta=1$, $x\to L$ only",
        "relax3_leak1_injL_self": r"relax, $\delta=1$, $x\to L$, self-read", "kick_pos_injL": r"kick, positive, $x\to L$ only",
        "relax3_leak2": r"relax, $\delta=2$", "relax3_leak3": r"relax, $\delta=3$", "twinL_leak1": r"twin $L$, $\delta=1$",
        "twinL_leak2": r"twin $L$, $\delta=2$", "innerL_self_leak1": r"inner loop, self-read, $\delta=1$",
        "twinL_leak3": r"twin $L$, $\delta=3$", "twinL_leak10": r"twin $L$, $\delta=10$",
        "twinL_leak2_noise25": r"twin $L$, $\delta=2$, train $\sigma=0.25$", "eqr_noise25": r"EqR, train $\sigma=0.25$"}


def pct(v, nd=1):
    return "--" if v is None else f"{100 * v:.{nd}f}"


def num(v, nd=2):
    if v is None:
        return "--"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def k(step):
    return f"{step // 1000}k"


def write(name, text):
    open(os.path.join(OUT, name + ".tex"), "w").write(text)


def maps():
    out = {}
    d = os.path.join(HERE, "data", "maps")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            arm, _, step = f[:-5].partition("__step")
            arm = arm[:-3] if arm.endswith("_s0") else arm
            out[(arm, int(step))] = json.load(open(os.path.join(d, f)))
    return out


M = maps()


def last(ms):
    return ms[str(max(int(x) for x in ms if x.isdigit()))]


def tab_main():
    order = ["eqr", "relax3_leak1", "kick_pos", "relax3_leak4", "kick_learned", "relax3", "kick_fixed",
             "relax3_leak1_injL", "relax3_leak1_injL_self", "kick_pos_injL",
             "relax3_leak2", "relax3_leak3", "twinL_leak1", "twinL_leak2", "innerL_self_leak1"]
    rows = sorted(T.MAIN, key=lambda r: (order.index(r[0]), r[1]))
    lines = []
    prev = None
    for arm, step, cell16, e16, e64, e256, o256, dy in rows:
        if prev is not None and arm in ("relax3_leak4", "relax3_leak1_injL", "relax3_leak2") and prev != arm:
            lines.append(r"\midrule")
        lines.append(f"{NAME[arm]} & {k(step)} & {pct(e16)} & {pct(cell16)} & {pct(e64)} & {pct(e256)} & {pct(o256)} & {pct(dy)} \\\\")
        prev = arm
    write("tab_main", "\n".join(lines))


def tab_scaling():
    rows = T.scaling_rows()
    keys = [("eqr", 20000), ("eqr", 50000), ("eqr-released", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000),
            ("kick_pos", 20000), ("kick_pos", 50000), None,
            ("relax3_leak2", 20000), ("relax3_leak2", 50000), ("relax3_leak3", 20000), ("twinL_leak1", 20000),
            ("twinL_leak2", 20000), ("twinL_leak2", 50000), ("innerL_self_leak1", 20000), ("innerL_self_leak1", 50000)]
    get = lambda arm, step, D, B, key: next((r[key] for r in rows if r["arm"] == arm and r["step"] == step and r["D"] == D
                                           and r["B"] == B and r["noise"] == 0.5), None)
    lines = []
    for key in keys:
        if key is None:
            lines.append(r"\midrule")
            continue
        arm, step = key
        single = [get(arm, step, D, 1, "single") for D in (16, 64, 256, 1024)]
        sel = [get(arm, step, 64, 128, kk) for kk in ("conv1", "convk", "vote", "any")]
        best = max((v for v in sel[:3] if v is not None), default=None)     # best non-oracle rule
        cells = [pct(v) for v in single] + [(r"\textbf{" + pct(v) + "}") if i < 3 and v == best and v is not None else pct(v)
                                    for i, v in enumerate(sel)]
        lines.append(f"{NAME[arm]} & {k(step)} & " + " & ".join(cells) + r" \\")
    write("tab_scaling", "\n".join(lines))


def tab_noisedep():
    """Depth 16 / 256 at the training noise (MAIN, evaluate.py sub/full sets) vs. sigma = 0.5 (scaling, 2048 puzzles)."""
    rows = T.scaling_rows()
    main = {(r[0], r[1]): r for r in T.MAIN}
    get = lambda arm, step, D: next((r["single"] for r in rows if r["arm"] == arm and r["step"] == step and r["D"] == D
                                     and r["B"] == 1 and r["noise"] == 0.5), None)
    keys = [("eqr", 20000), ("eqr", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000), ("kick_pos", 20000),
            ("kick_pos", 50000), None, ("relax3_leak2", 20000), ("relax3_leak2", 50000), ("relax3_leak3", 20000),
            ("twinL_leak1", 20000), ("twinL_leak2", 20000), ("twinL_leak2", 50000), ("innerL_self_leak1", 20000),
            ("innerL_self_leak1", 50000)]
    lines = []
    for key in keys:
        if key is None:
            lines.append(r"\midrule")
            continue
        arm, step = key
        m = main[(arm, step)]
        lo16, lo256 = m[3], m[5]
        hi16, hi256, hi1024 = get(arm, step, 16), get(arm, step, 256), get(arm, step, 1024)
        gain = None if hi256 is None else hi256 - lo256
        g = "--" if gain is None else f"${100 * gain:+.1f}$"
        lines.append(f"{NAME[arm]} & {k(step)} & {pct(lo16)} & {pct(hi16)} & {pct(lo256)} & {pct(hi256)} & {g} & {pct(hi1024)} \\\\")
    write("tab_noisedep", "\n".join(lines))


def tab_ext50():
    lines = []
    for arm, ev, (evmax, stmax), tr, act, cos in T.EXT50:
        peak = pct(evmax) + ("" if stmax == 49000 else f" ({stmax // 1000}k)")
        lines.append(f"{NAME[arm]} & {pct(ev)} & {peak} & {pct(tr)} & ${100 * (tr - ev):+.1f}$ & {num(act, 1)} & {num(cos)} \\\\")
    write("tab_ext50", "\n".join(lines))


def tab_rates():
    order = [("eqr", 20000), ("eqr", 50000), ("eqr-released", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000),
             ("relax3_leak4", 20000), ("relax3", 20000), ("kick_pos", 20000), ("kick_pos", 50000), ("kick_learned", 20000),
             ("kick_fixed", 20000), ("kick_pos_injL", 20000)]
    lines = []
    for arm, step in order:
        r = M[(arm, step)]
        rt = r.get("rates")
        rms = last(r["multistart"]["noise_0"])["rms"]
        pr = r.get("pr") or {}
        if rt is None:
            pred_h, pred_l = 1.0, 1.0
            rate = r"\multicolumn{3}{c}{$1-\lambda=0.05$ (self-read)}"
            floor = "--"
        else:
            a, b = rt["keep_L"], rt["keep_H"]
            if arm.startswith("kick"):
                pred_h, pred_l = 1 / (1 - b), a / (1 - a)
            elif rt["delta"] is not None:
                c = math.exp(-rt["delta"])
                pred_h = pred_l = 1 / (1 - c)
            else:
                pred_h = pred_l = math.inf
            delta = "--" if rt["delta"] is None else num(rt["delta"])
            r3 = lambda v: f"{v:.3f}" if v < 0.1 else num(v)
            rate = f"{r3(rt['gamma'])} & {r3(rt['alpha'])} & {delta}"
            floor = f"{rt['floor_geomean_abs_eig']:.1e}".replace("e-0", "e-").replace("e-", r"\!\times\!10^{-") + "}"
            floor = f"${floor}$" if "times" in floor else floor
        ph = r"$\infty$" if math.isinf(pred_h) else num(pred_h)
        pl = r"$\infty$" if math.isinf(pred_l) else num(pred_l)
        lines.append(f"{NAME[arm]} & {k(step)} & {rate} & {floor} & {ph} / {pl} & {num(rms['H'])} / {num(rms['L'])} & "
                     f"{num(pr.get('H'), 1)} / {num(pr.get('L'), 1)} \\\\")
    write("tab_rates", "\n".join(lines))


def tab_select():
    order = [("eqr", 20000), ("eqr", 50000), ("eqr-released", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000),
             ("kick_pos", 20000), ("kick_pos", 50000), ("relax3_leak4", 20000)]
    blocks = []
    for noise in ("0", "0.5"):
        lines = [r"\midrule" if noise == "0.5" else "", rf"\multicolumn{{11}}{{l}}{{\emph{{test noise $\sigma={noise}$}}}} \\"]
        for arm, step in order:
            m = last(M[(arm, step)]["multistart"][f"noise_{noise}"])
            lines.append(f"{NAME[arm]} & {k(step)} & {pct(m['single'])} & {pct(m['any'])} & {pct(m['vote'])} & {pct(m['conv1'])} & "
                         f"{pct(m['q1'])} & {pct(m['verifier'])} & {num(m['auroc']['lscore'], 3)} & {num(m['distinct_mean'], 1)} & "
                         f"{num(m['delta_pi'], 3)} \\\\")
        blocks.append("\n".join(x for x in lines if x))
    write("tab_select", "\n".join(blocks))


def tab_spectral():
    order = [("eqr", 20000), ("eqr", 50000), ("eqr-released", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000),
             ("relax3_leak1_injL_self", 20000), ("relax3_leak4", 20000), ("relax3", 20000), ("kick_pos", 20000), ("kick_pos", 50000),
             ("kick_learned", 20000), ("kick_fixed", 20000)]
    lines = []
    for arm, step in order:
        j = M[(arm, step)]["jacobian"]
        c, w = j["correct"], j["wrong"]
        lines.append(f"{NAME[arm]} & {k(step)} & {num(c['rho_median'])} & [{num(c['rho_q10_q90'][0])}, {num(c['rho_q10_q90'][1])}] & "
                     f"{pct(c['top1_complex'], 0)} & {pct(c['frac_complex_top10'], 0)} & {num(w['rho_median'])} & "
                     f"{pct(w['top1_complex'], 0)} \\\\")
    write("tab_spectral", "\n".join(lines))


def tab_escape():
    order = [("eqr", 20000), ("eqr", 50000), ("eqr-released", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000),
             ("kick_pos", 20000), ("kick_pos", 50000), ("relax3_leak4", 20000), ("relax3", 20000), ("kick_fixed", 20000)]
    sig = ["0.1", "0.25", "0.5", "0.75", "1.0"]
    lines = []
    for arm, step in order:
        e = M[(arm, step)]["escape"]["by_sigma"]
        cells = [f"{num(e[s]['keep_if_correct'])} / {num(e[s]['rescued_if_wrong'])}" for s in sig]
        lines.append(f"{NAME[arm]} & {k(step)} & " + " & ".join(cells) + r" \\")
    write("tab_escape", "\n".join(lines))


def tab_lcycles():
    lines = []
    for nm, d in T.LCYCLES.items():
        lines.append(nm.replace("eqr", "EqR") + " & " + " & ".join(f"{pct(d[L][0])} / {pct(d[L][1])}" for L in (4, 5, 6, 7, 8)) + r" \\")
    write("tab_lcycles", "\n".join(lines))


def tab_round3():
    lines = []
    for arm, e14, tr, cos, steps in T.ROUND3:
        lines.append(f"{NAME[arm]} & {pct(e14)} & {pct(tr)} & {num(cos)} & {num(steps, 1)} \\\\")
    write("tab_round3", "\n".join(lines))


# ------------------------------------------------------------------------- seed-resolved tables (2026-11-08)
def ms(st, nd=1, scale=100):
    """mean +- sd (sd only for n > 1); '--' if missing."""
    if st is None:
        return "--"
    m, sd, n = st
    return f"{scale * m:.{nd}f}" if sd is None else f"${scale * m:.{nd}f}\\pm{scale * sd:.{nd}f}$"


def tab_seeds():
    """20k, three seeds: training-noise accuracy and test-time scaling at sigma = 0.5."""
    lines = []
    for arm in ("eqr", "twinL_leak2", "twinL_leak3", "twinL_leak10"):
        sc = lambda D, B, key: T.stat(T.runs(T.RUNS_SCALING, arm, 20000, D=D, B=B, N=0.5), key)
        n = len(T.runs(T.RUNS_SCALING, arm, 20000, D=1024, B=1, N=0.5))
        cells = [ms(T.stat(T.runs(T.RUNS_MAIN, arm, 20000), "exact@16")), ms(sc(256, 1, "single")), ms(sc(1024, 1, "single")),
                 ms(sc(64, 128, "conv1")), ms(sc(64, 128, "convk")), ms(sc(64, 128, "vote")), ms(sc(64, 128, "any"))]
        lines.append(f"{NAME[arm]} & {n} & " + " & ".join(cells) + r" \\")
    rel = {r["D"] if r["B"] == 1 else (r["D"], r["B"]): r for r in T.scaling_rows() if r["arm"] == "eqr-released" and r["noise"] == 0.5}
    r = rel[(64, 128)]
    lines += [r"\midrule", f"EqR (released, 50k) & 1 & -- & {pct(rel[256]['single'])} & {pct(rel[1024]['single'])} & "
              f"{pct(r['conv1'])} & {pct(r['convk'])} & {pct(r['vote'])} & {pct(r['any'])} \\\\"]
    write("tab_seeds", "\n".join(lines))


def tab_round4():
    """50k, seed 0: training-noise accuracy and test-time scaling at sigma = 0.25 and 0.5."""
    lines = []
    for arm in ("eqr", "twinL_leak2", "twinL_leak3", "twinL_leak10", "innerL_self_leak1", None, "twinL_leak2_noise25", "eqr_noise25"):
        if arm is None:
            lines.append(r"\midrule")
            continue
        m = T.runs(T.RUNS_MAIN, arm, 50000, seeds=[0])
        g = lambda key: m[0].get(key) if m else None
        sc = lambda D, B, N, key: (lambda x: x[0][key] if x else None)(T.runs(T.RUNS_SCALING, arm, 50000, seeds=[0], D=D, B=B, N=N))
        d20 = T.runs(T.RUNS_SCALING, arm, 20000, seeds=[0], D=1024, B=1, N=0.5)
        cells = [pct(g("exact@16")), pct(g("exact@256s")), num(g("resid@256"), 1), pct(d20[0]["single"] if d20 else None),
                 pct(sc(1024, 1, 0.25, "single")), pct(sc(64, 16, 0.25, "convk")), pct(sc(1024, 1, 0.5, "single")),
                 pct(sc(64, 128, 0.5, "convk"))]
        lines.append(f"{NAME[arm]} & " + " & ".join(cells) + r" \\")
    write("tab_round4", "\n".join(lines))


def tab_dyn():
    """dynamics_eqr.py: noise-free trajectories of 768 test puzzles to 256 iterations; spectral radius on 32."""
    keys = [("eqr", 20000), ("eqr", 50000), ("relax3_leak1", 50000), ("innerL_self_leak1", 50000), None,
            ("twinL_leak2", 20000), ("twinL_leak2", 50000), ("twinL_leak3", 20000), ("twinL_leak3", 50000),
            ("twinL_leak10", 20000), ("twinL_leak10", 50000), None, ("twinL_leak2_noise25", 50000), ("eqr_noise25", 50000)]
    lines = []
    for key in keys:
        if key is None:
            lines.append(r"\midrule")
            continue
        arm, step = key
        rows = T.runs(T.RUNS_DYN, arm, step)
        cos = T.stat(rows, "twin_cos@256")
        cells = [str(len(rows)), ms(T.stat(rows, "nf_exact@256")), ms(T.stat(rows, "FP@256(1e-2)")),
                 ms(T.stat(rows, "step50@256"), 1), ms(T.stat(rows, "rho"), 2, 1), "--" if cos is None else ms(cos, 3, 1)]
        lines.append(f"{NAME[arm]} & {k(step)} & " + " & ".join(cells) + r" \\")
    write("tab_dyn", "\n".join(lines))


def tab_sky():
    """Skyscrapers-7 pilot (W&B): test exact at 16 iterations along training, training-batch exact and ACT steps."""
    lines = []
    for key, arm, n in (("eqr_s0_1k", "eqr", "1k"), ("eqr_s0_10k", "eqr", "10k"), ("eqr_s0_100k", "eqr", "100k"),
                        ("twinL_leak3_s0_100k", "twinL_leak3", "100k")):
        d = T.SKY[key]
        ev = {s: (e, c) for s, e, c in d["eval"]}
        best = max(d["eval"], key=lambda t: t[1])
        cells = [pct(ev[s][0]) for s in (8000, 10000, 14000, 20000)] + [f"{pct(best[1])} ({k(best[0])})", pct(ev[20000][1]),
                 pct(d["train_18_20k"]), f"{d['act_18_20k']:.1f}"]
        lines.append(f"{n} & {NAME[arm]} & " + " & ".join(cells) + r" \\")
    write("tab_sky", "\n".join(lines))


if __name__ == "__main__":
    for f in (tab_main, tab_scaling, tab_noisedep, tab_ext50, tab_rates, tab_select, tab_spectral, tab_escape,
              tab_lcycles, tab_round3, tab_seeds, tab_round4, tab_dyn, tab_sky):
        f()
        print("ok", f.__name__)
