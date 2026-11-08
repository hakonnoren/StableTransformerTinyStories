"""Figures for the Sudoku map-analysis note.  python make_figures.py  (writes figures/*.pdf and figures/*.png)

Inputs (all in data/): tables.py (transcribed summaries), maps/*.json (analyze_maps.py, GPU preset),
wandb_curves.json (W&B history), eqr_inner_loop.json (inner-loop eigen-analysis of the EqR checkpoints).
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))
import tables as T  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 4.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})
W = 6.3          # text width (in)

C = {"eqr": "C0", "eqr-released": "C4", "relax3_leak1": "C1", "kick_pos": "C2", "relax3_leak4": "C8",
     "relax3": "C7", "kick_learned": "C9", "kick_fixed": "C5", "relax3_leak1_injL": "C3",
     "relax3_leak1_injL_self": "C6", "kick_pos_injL": "C3", "relax3_leak2": "C9", "relax3_leak3": "C7",
     "twinL_leak1": "C5", "twinL_leak2": "C3", "innerL_self_leak1": "C6", "twinL_leak3": "C8", "twinL_leak10": "C9"}
LABEL = {"eqr": "EqR", "eqr-released": "EqR (released)", "relax3_leak1": r"relax $\delta{=}1$",
         "kick_pos": "kick (pos.)", "relax3_leak4": r"relax $\delta{=}1/4$", "relax3": r"relax $\delta{=}0$",
         "kick_learned": "kick (learned)", "kick_fixed": "kick (fixed)", "relax3_leak1_injL": r"relax $\delta{=}1$, $x\!\to\!L$",
         "relax3_leak1_injL_self": r"relax $\delta{=}1$, $x\!\to\!L$, self", "kick_pos_injL": r"kick, $x\!\to\!L$",
         "relax3_leak2": r"relax $\delta{=}2$", "relax3_leak3": r"relax $\delta{=}3$", "twinL_leak1": r"twin $\delta{=}1$",
         "twinL_leak2": r"twin $\delta{=}2$", "innerL_self_leak1": r"inner loop, self", "twinL_leak3": r"twin $\delta{=}3$",
         "twinL_leak10": r"twin $\delta{=}10$"}
STEP_STYLE = {20000: dict(ls="--", mfc="white"), 50000: dict(ls="-")}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def maps():
    out = {}
    for f in sorted(os.listdir(os.path.join(HERE, "data", "maps"))):
        if f.endswith(".json"):
            base = f[:-5]
            arm, _, step = base.partition("__step")
            arm = arm[:-3] if arm.endswith("_s0") else arm
            out[(arm, int(step))] = json.load(open(os.path.join(HERE, "data", "maps", f)))
    return out


M = maps()
SC = T.scaling_rows()
WB = json.load(open(os.path.join(HERE, "data", "wandb_curves.json")))
IL = json.load(open(os.path.join(HERE, "data", "eqr_inner_loop.json")))
NOISES = ["0", "0.25", "0.5", "0.75"]


def lab(arm, step):
    return f"{LABEL.get(arm, arm)}, {step // 1000}k" if arm != "eqr-released" else LABEL[arm]


# --------------------------------------------------------------------------------------------- training
def fig_training():
    ext = ["eqr", "twinL_leak2", "innerL_self_leak1", "relax3_leak1", "kick_pos", "relax3_leak2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.2, 1]})
    for arm in ext:
        ev = np.array(WB[arm]["eval"])
        a.plot(ev[:, 0] / 1000, ev[:, 1], "o-", color=C[arm], label=LABEL[arm], ms=3.2)
    a.axvline(20, color="0.6", lw=0.6, ls=":")
    a.set_xlabel("training step (thousands)")
    a.set_ylabel("exact accuracy (16 it., test)")
    a.set_ylim(0, 0.92)
    a.legend(loc="lower right", fontsize=6.6)
    a.set_title("(a) in-training evaluation, arms trained to 50k")
    rows = sorted(T.EXT50, key=lambda r: r[1])
    for i, (arm, ev, _, tr, act, _) in enumerate(rows):
        b.plot([ev, tr], [i, i], color=C[arm], lw=1.0)
        b.plot([ev], [i], "o", color=C[arm], ms=5)
        b.plot([tr], [i], "D", color=C[arm], mfc="white", ms=4)
        b.text(1.005, i, f"{act:.1f}", va="center", fontsize=7, transform=b.get_yaxis_transform())
    b.text(1.005, len(rows) - 0.55, "ACT", va="bottom", fontsize=7, transform=b.get_yaxis_transform())
    b.set_yticks(range(len(rows)), [LABEL[r[0]] for r in rows])
    b.set_ylim(-0.6, len(rows) + 0.75)
    b.set_xlabel("exact accuracy")
    b.set_xlim(0.65, 1.0)
    b.plot([], [], "o", color="0.3", label="test (eval at 49k)")
    b.plot([], [], "D", color="0.3", mfc="white", label="train batches, 48k to 50k")
    b.legend(loc="upper left", handletextpad=0.2, fontsize=6.6)
    b.set_title("(b) train vs. test at 50k; ACT steps")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_training")


def fig_retention():
    fig, a = plt.subplots(figsize=(3.1, 2.3))
    a_keep = math.exp(-3)
    xs, ys = [], []
    for d, acc in T.RELAX_DELTA:
        r = a_keep + (1 - a_keep) * math.exp(-d)
        xs.append(r); ys.append(acc)
        a.annotate(rf"$\delta{{=}}{d:g}$", (r, acc), textcoords="offset points",
                   xytext=(-6, -12) if d == 3.0 else (4, 3), fontsize=7)
    a.plot(xs, ys, "o-", color="C1", label=r"relax, rate 3")
    a.axvline(0.05, color="C0", lw=0.8, ls="--")
    a.text(0.06, 0.08, r"EqR: $1-\lambda=0.05$", color="C0", fontsize=7, rotation=90, va="bottom")
    a.set_xlabel(r"merged retention $r=a+(1-a)e^{-\delta}$")
    a.set_ylabel("exact accuracy (eval, 14k)")
    a.set_xlim(0, 1.05)
    a.set_ylim(0, 0.65)
    fig.tight_layout()
    save(fig, "fig_retention")


# --------------------------------------------------------------------------------------------- test-time scaling
def _depth_curve(arm, step):
    rows = sorted([r for r in SC if r["arm"] == arm and r["step"] == step and r["B"] == 1 and r["noise"] == 0.5], key=lambda r: r["D"])
    return np.array([r["D"] for r in rows]), np.array([r["single"] for r in rows])


def fig_depth():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45))
    for arm, step in (("eqr-released", 50000), ("eqr", 50000), ("eqr", 20000), ("relax3_leak1", 50000), ("relax3_leak1", 20000),
                      ("kick_pos", 50000), ("kick_pos", 20000)):
        d, s = _depth_curve(arm, step)
        st = dict(STEP_STYLE[step]) if arm != "eqr-released" else dict(ls=":")
        a.plot(d, s, marker="o", color=C[arm], label=lab(arm, step), **st)
        b.plot(d, 1 - s, marker="o", color=C[arm], **st)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024], ["16", "64", "256", "1024"])
        ax.set_xlabel("depth $D$ (outer iterations)")
    a.set_ylabel("exact accuracy, one run")
    a.set_title(r"(a) depth scaling, noise $\sigma=0.5$")
    b.set_yscale("log")
    b.set_ylabel("error rate, one run")
    b.set_title("(b) error, log-log")
    # power-law guides: slope estimated on relax 50k
    d, s = _depth_curve("relax3_leak1", 50000)
    k = np.polyfit(np.log(d), np.log(1 - s), 1)[0]
    b.text(0.04, 0.06, rf"relax $\delta{{=}}1$, 50k: error $\propto D^{{{k:.2f}}}$", transform=b.transAxes, fontsize=7)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14), fontsize=6.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_depth")


def fig_breadth():
    rules = [("single", "one run"), ("conv1", "conv."), ("convk", "top-4"), ("vote", "vote"), ("any", "any")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, step in zip(axes, (20000, 50000)):
        arms = ["eqr", "relax3_leak1", "kick_pos"]
        arms += (["relax3_leak2", "twinL_leak1", "twinL_leak2", "innerL_self_leak1"] if step == 20000
                 else ["twinL_leak2", "innerL_self_leak1", "eqr-released"])
        off = np.linspace(-0.3, 0.3, len(arms))
        for j, arm in enumerate(arms):
            st = 50000 if arm == "eqr-released" else step
            r = [x for x in SC if x["arm"] == arm and x["step"] == st and x["D"] == 64 and x["B"] == 128 and x["noise"] == 0.5][0]
            vals = [r[k] for k, _ in rules]
            ax.plot(np.arange(len(rules)) + off[j], vals, "o", color=C[arm], label=LABEL[arm], ms=4.2,
                    mfc="white" if arm in ("eqr", "relax3_leak1", "kick_pos") and step == 20000 else None)
        ax.set_xticks(range(len(rules)), [n for _, n in rules])
        ax.set_title(f"({'ab'[step == 50000]}) {step // 1000}k steps, depth 64, 128 starts, " + r"$\sigma=0.5$")
        ax.set_ylim(0.3, 1.01)
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.legend(loc="lower right", fontsize=6.3, handletextpad=0.1)
    axes[0].set_ylabel("exact accuracy")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_breadth")


def _err_axes(ax):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 64, 256, 1024], ["16", "64", "256", "1024"])
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_ylim(0.012, 0.75)
    ax.set_xlabel("depth $D$ (outer iterations)")
    ax.grid(color="0.92", lw=0.5, which="major")


def fig_r3_depth():
    """Round three: one-run error against depth at sigma = 0.5, 20k and 50k."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
    sets = {
        a: (20000, ["eqr", "relax3_leak1", "relax3_leak2", "relax3_leak3", "twinL_leak1", "twinL_leak2", "innerL_self_leak1"]),
        b: (50000, ["eqr", "relax3_leak1", "kick_pos", "twinL_leak2", "innerL_self_leak1"]),
    }
    for ax, (step, arms) in sets.items():
        d, s_ = _depth_curve("eqr-released", 50000)
        ax.plot(d, 1 - s_, ":", marker="D", ms=3, color=C["eqr-released"], label=LABEL["eqr-released"])
        for arm in arms:
            d, s_ = _depth_curve(arm, step)
            ax.plot(d, 1 - s_, marker="o", color=C[arm], label=LABEL[arm], lw=1.6 if arm in ("eqr", "twinL_leak2") else 1.0,
                    mfc="white" if step == 20000 else None, ms=3.5)
        _err_axes(ax)
        ax.legend(loc="lower left", fontsize=6.3, handletextpad=0.3)
    a.set_ylabel(r"error rate, one run ($\sigma=0.5$)")
    a.set_title("(a) 20k steps")
    b.set_title("(b) 50k steps")
    b.text(0.97, 0.03, "relax $\\delta{=}2$, 50k: error 97% at every depth", transform=b.transAxes, ha="right", va="bottom", fontsize=6.6,
           color=C["relax3_leak2"])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_r3_depth")


def fig_noisedep():
    """Depth-256 accuracy at the training noise against sigma = 0.5; arrows 20k -> 50k."""
    main = {(r[0], r[1]): r[5] for r in T.MAIN}
    hi = {(r["arm"], r["step"]): r["single"] for r in SC if r["D"] == 256 and r["B"] == 1 and r["noise"] == 0.5}
    arms = ["eqr", "relax3_leak1", "kick_pos", "relax3_leak2", "relax3_leak3", "twinL_leak1", "twinL_leak2", "innerL_self_leak1"]
    fig, a = plt.subplots(figsize=(3.6, 3.1))
    lo_, hi_ = 0.5, 1.0
    a.plot([lo_, hi_], [lo_, hi_], color="0.5", lw=0.6, ls="--")
    floor = 0.515
    for arm in arms:
        pts = [(st, main.get((arm, st)), hi.get((arm, st))) for st in (20000, 50000)]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        for st, x, y in pts:
            yy = max(y, floor)
            a.plot([x], [yy], "o" if y >= floor else "v", color=C[arm], ms=5.5 if st == 50000 else 5,
                   mfc="white" if st == 20000 else None, zorder=3)
        if len(pts) == 2:
            (_, x0, y0), (_, x1, y1) = pts
            a.annotate("", xy=(x1, max(y1, floor)), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle="-|>", color=C[arm], lw=0.8, shrinkA=3, shrinkB=3, mutation_scale=7))
        a.plot([], [], "o", color=C[arm], label=LABEL[arm])
    a.text(0.765, 0.517, "relax $\\delta{=}2$, 50k:\n2.6% (off scale)", fontsize=6.3, color=C["relax3_leak2"], ha="left")
    a.text(0.53, 0.60, "noise helps", fontsize=7, color="0.4", ha="left")
    a.text(0.80, 0.70, "noise hurts", fontsize=7, color="0.4")
    a.set_xlim(lo_, hi_)
    a.set_ylim(lo_, hi_)
    a.set_xlabel(r"exact, depth 256, training noise $\sigma=0.01$")
    a.set_ylabel(r"exact, depth 256, test noise $\sigma=0.5$")
    a.plot([], [], "o", color="0.3", mfc="white", label="20k")
    a.plot([], [], "o", color="0.3", label="50k")
    a.legend(loc="upper left", fontsize=6.0, handletextpad=0.1, ncol=1)
    fig.tight_layout()
    save(fig, "fig_noisedep")


# --------------------------------------------------------------------------------------------- multistart (maps)
MAIN_ARMS = [("eqr", 20000), ("eqr", 50000), ("relax3_leak1", 20000), ("relax3_leak1", 50000), ("kick_pos", 20000), ("kick_pos", 50000)]


def _ms(arm, step, noise, key):
    m = M[(arm, step)]["multistart"][f"noise_{noise}"]
    rl = str(max(int(k) for k in m if k.isdigit()))
    return m[rl][key] if key != "auroc_conv" else m[rl]["auroc"]["lscore"]


def fig_noise():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.2))
    xs = [float(n) for n in NOISES]
    for arm, step in MAIN_ARMS + [("eqr-released", 50000)]:
        st = dict(STEP_STYLE[step]) if arm != "eqr-released" else dict(ls=":")
        for ax, key in zip(axes, ("single", "any", "distinct_mean")):
            ax.plot(xs, [_ms(arm, step, n, key) for n in NOISES], marker="o", color=C[arm], label=lab(arm, step), **st)
    axes[0].set_ylabel("exact accuracy")
    axes[0].set_title("(a) one run")
    axes[1].set_title("(b) some start right")
    axes[2].set_title("(c) distinct answers / puzzle")
    for ax in axes:
        ax.set_xlabel(r"test noise $\sigma$")
        ax.set_xticks(xs)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 16.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=6.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_noise")


def fig_escape():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3))
    arms = MAIN_ARMS + [("eqr-released", 50000), ("relax3_leak4", 20000)]
    for arm, step in arms:
        e = M[(arm, step)]["escape"]["by_sigma"]
        sg = sorted(e, key=float)
        st = dict(STEP_STYLE[step]) if arm != "eqr-released" else dict(ls=":")
        a.plot([float(s) for s in sg], [e[s]["keep_if_correct"] for s in sg], marker="o", color=C[arm], label=lab(arm, step), **st)
        b.plot([float(s) for s in sg], [e[s]["rescued_if_wrong"] for s in sg], marker="o", color=C[arm], **st)
    a.set_title(r"(a) right answer kept, $P(\mathrm{keep}\mid\mathrm{right})$")
    b.set_title(r"(b) wrong answer rescued, $P(\mathrm{right}\mid\mathrm{wrong})$")
    for ax in (a, b):
        ax.set_xlabel(r"noise $\sigma$ for $K=32$ iterations")
    a.set_ylim(-0.03, 1.05)
    a.legend(loc="lower left", fontsize=6.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_escape")


def fig_auroc():
    fig, a = plt.subplots(figsize=(3.1, 2.2))
    xs = [float(n) for n in NOISES]
    for arm, step in MAIN_ARMS:
        a.plot(xs, [_ms(arm, step, n, "auroc_conv") for n in NOISES], marker="o", color=C[arm], label=lab(arm, step), **STEP_STYLE[step])
    a.set_xlabel(r"test noise $\sigma$")
    a.set_ylabel("AUROC, convergence score")
    a.set_xticks(xs)
    a.set_ylim(0.8, 1.01)
    a.legend(loc="lower left", fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig_auroc")


# --------------------------------------------------------------------------------------------- Jacobians
SPEC_ARMS = [("eqr-released", 50000), ("eqr", 50000), ("eqr", 20000), ("relax3_leak1", 50000), ("relax3_leak1", 20000),
             ("relax3_leak1_injL_self", 20000), ("relax3_leak1_injL", 20000), ("relax3_leak4", 20000), ("relax3", 20000),
             ("kick_pos", 50000), ("kick_pos", 20000), ("kick_pos_injL", 20000), ("kick_learned", 20000), ("kick_fixed", 20000)]


def fig_spectral():
    fig, a = plt.subplots(figsize=(W, 2.5))
    for i, (arm, step) in enumerate(SPEC_ARMS):
        j = M[(arm, step)]["jacobian"]
        a.plot([i], [j["correct"]["rho_median"]], "o", color=C[arm], ms=6)
        a.plot([i], [j["wrong"]["rho_median"]], "s", color=C[arm], mfc="white", ms=6)
        a.plot([i, i], [j["correct"]["rho_median"], j["wrong"]["rho_median"]], color=C[arm], lw=0.6)
    a.axhline(1.0, color="0.4", lw=0.7, ls="--")
    a.set_yscale("log")
    a.set_yticks([0.3, 0.5, 1, 2, 5, 10], ["0.3", "0.5", "1", "2", "5", "10"])
    a.set_ylim(0.3, 11)
    a.set_ylabel(r"$\rho(J)$, one outer iteration")
    a.set_xticks(range(len(SPEC_ARMS)), [lab(a_, s) for a_, s in SPEC_ARMS], rotation=35, ha="right", fontsize=7)
    a.plot([], [], "o", color="0.3", label="right endpoints (median)")
    a.plot([], [], "s", color="0.3", mfc="white", label="wrong endpoints (median)")
    a.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_spectral")


def fig_ritz():
    sel = [("eqr", 50000), ("relax3_leak1", 50000), ("kick_pos", 50000), ("eqr", 20000), ("relax3_leak1", 20000), ("kick_pos", 20000)]
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2), sharex=True, sharey=True)
    th = np.linspace(0, 2 * np.pi, 300)
    for ax, (arm, step) in zip(axes.flat, sel):
        j = M[(arm, step)]["jacobian"]
        rz = np.array(j["ritz_top10"])       # [n, 10, 2]; rows: right endpoints first
        nr = j["correct"]["n"]
        ax.plot(np.cos(th), np.sin(th), color="0.6", lw=0.6)
        ax.plot(rz[nr:, :, 0].ravel(), rz[nr:, :, 1].ravel(), "x", color="0.55", ms=3, mew=0.6, label="wrong")
        ax.plot(rz[:nr, :, 0].ravel(), rz[:nr, :, 1].ravel(), "o", color=C[arm], ms=2.6, alpha=0.8, label="right")
        ax.set_title(lab(arm, step), fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\mathrm{Re}\,\mu$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathrm{Im}\,\mu$")
    axes[0, 0].legend(loc="upper left", fontsize=6.5, handletextpad=0.1)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "fig_ritz")


# --------------------------------------------------------------------------------------------- recursion / inner loop
def fig_recursion():
    sel = [("eqr", 20000), ("eqr", 50000), ("relax3_leak1", 50000)]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (arm, step) in zip(axes, sel):
        prof = M[(arm, step)]["trace"]["profile_median"]
        marks = sorted(prof, key=int)
        for k, t in enumerate(marks):
            p = prof[t]
            v = np.array([x[1] for x in p])
            names = [x[0] for x in p]
            col = cmap(k / max(1, len(marks) - 1))
            ax.plot(np.arange(1, len(v) + 1), np.maximum(v, 1e-9), "-", color=col, lw=1.0, label=f"iteration {t}")
            hidx = [i + 1 for i, n in enumerate(names) if n == "H"]
            ax.plot(hidx, np.maximum(v[[i - 1 for i in hidx]], 1e-9), "v", color=col, ms=3)
        ax.set_yscale("log")
        ax.set_title(lab(arm, step))
        ax.set_xticks([1, 7, 14, 21])
    axes[1].set_xlabel("update index within one outer iteration (21 updates; $\\blacktriangledown$ = $H$ update)")
    axes[0].set_ylabel(r"relative change $\|s'-s\|/\|s'\|$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.1), fontsize=6.8)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_recursion")


def fig_inner():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    names = [("eqr_s0__step20000", "EqR 20k", "C0", "--"), ("eqr_s0__step50000", "EqR 50k", "C0", "-"),
             ("eqr-released__step50000", "EqR (released)", "C4", ":")]
    for key, nm, col, ls in names:
        d = IL[key]
        s1 = np.median(np.array(d["inner_step1"]), axis=1)
        s2 = np.median(np.array(d["inner_step2"]), axis=1)
        a.plot(np.arange(1, len(s1) + 1), np.maximum(s1, 1e-8), color=col, ls=ls, label=f"{nm}: one step")
        a.plot(np.arange(2, len(s2) + 2), np.maximum(s2, 1e-8), color=col, ls=ls, lw=2.2, alpha=0.45, label=f"{nm}: two steps")
    a.set_yscale("log")
    a.set_xlabel(r"inner update $k$ ($z_H$ frozen)")
    a.set_ylabel(r"relative change of $z_L$")
    a.set_title(r"(a) $\|z^{k+1}-z^k\|$ (thin) and $\|z^{k+1}-z^{k-1}\|$ (thick)")
    a.legend(fontsize=6.0, loc="center right")
    th = np.linspace(0, 2 * np.pi, 300)
    b.plot(np.cos(th), np.sin(th), color="0.6", lw=0.6)
    for key, nm, col, ls in names:
        d = IL[key]
        ev = np.array(d["eig_DF_fixed_point"])        # [16, 6, 2]
        right = np.array(d["right"], dtype=bool)
        mk = {"EqR 20k": "o", "EqR 50k": "s", "EqR (released)": "D"}[nm]
        b.plot(ev[right, :, 0].ravel(), ev[right, :, 1].ravel(), mk, color=col, ms=2.8, alpha=0.75, label=nm,
               mfc="white" if nm == "EqR 20k" else None)
        if (~right).any():
            b.plot(ev[~right, :, 0].ravel(), ev[~right, :, 1].ravel(), "x", color=col, ms=3, mew=0.6)
    b.axvline(-1, color="0.4", lw=0.6, ls="--")
    b.set_aspect("equal")
    b.set_xlim(-2.2, 1.2)
    b.set_ylim(-1.3, 1.3)
    b.set_xlabel(r"$\mathrm{Re}\,\mu$")
    b.set_ylabel(r"$\mathrm{Im}\,\mu$")
    b.set_title(r"(b) eigenvalues of $DF$ at its fixed point")
    b.legend(fontsize=6.2, loc="upper left", handletextpad=0.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_inner")


def fig_flip():
    """Stability of z <- (1-lam) z + lam B(z) along a real Jacobian eigen-direction mu of B: |1 - lam + lam mu| < 1."""
    fig, a = plt.subplots(figsize=(3.1, 2.5))
    lam = np.linspace(0.02, 1.0, 300)
    a.fill_between(lam, 1 - 2 / lam, 1, color="C2", alpha=0.12, lw=0)
    a.plot(lam, 1 - 2 / lam, color="C3", lw=1.0)
    a.axhline(1, color="C3", lw=1.0)
    a.text(0.55, -0.3, "stable", fontsize=8, color="C2")
    a.text(0.25, -4.3, "period doubling\n(flip)", fontsize=7, color="C3")
    a.axvline(0.95, color="C0", lw=0.7, ls="--")
    for key, nm, mk in (("eqr_s0__step20000", "20k", "o"), ("eqr_s0__step50000", "50k", "s"), ("eqr-released__step50000", "rel.", "D")):
        ev = np.array(IL[key]["eig_DF_fixed_point"])
        right = np.array(IL[key]["right"], dtype=bool)
        real = ev[right, :, 0]
        most_neg = real.min(axis=1)                      # most negative real part among the leading Ritz values
        mu = (np.median(most_neg) - 0.05) / 0.95
        a.plot([0.95], [mu], mk, color="C0" if nm != "rel." else "C4", ms=5, mfc="white" if nm == "20k" else None)
        a.annotate(f"EqR {nm}", (0.95, mu), textcoords="offset points", xytext=(-36, -2), fontsize=6.5)
    a.set_xlim(0, 1.02)
    a.set_ylim(-6, 1.5)
    a.set_xlabel(r"step size $\lambda$")
    a.set_ylabel(r"eigenvalue $\mu$ of $J_B$")
    fig.tight_layout()
    save(fig, "fig_flip")


def fig_scale():
    fig, a = plt.subplots(figsize=(3.1, 2.6))
    pts = []
    for (arm, step), r in M.items():
        rt = r.get("rates")
        ms = r["multistart"]["noise_0"]
        rms = ms[str(max(int(k) for k in ms if k.isdigit()))]["rms"]
        if rt is None:
            pts.append((1.0, rms["H"], arm, step, "H")); pts.append((1.0, rms["L"], arm, step, "L"))
            continue
        a_, b_ = rt["keep_L"], rt["keep_H"]
        if arm.startswith("kick"):
            pts.append((1 / (1 - b_), rms["H"], arm, step, "H")); pts.append((a_ / (1 - a_), rms["L"], arm, step, "L"))
        elif rt["delta"] is not None and arm in ("relax3_leak1", "relax3_leak4"):
            c = math.exp(-rt["delta"])
            pts.append((1 / (1 - c), rms["H"], arm, step, "H"))
    for p, m, arm, step, s in pts:
        a.plot([p], [m], "o" if s == "H" else "s", color=C[arm], ms=4.5, mfc=None if s == "H" else "white")
    lo, hi = 0.3, 400
    a.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, ls="--")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("predicted RMS (Prop. 2)")
    a.set_ylabel("measured RMS at endpoints")
    for arm in ("eqr", "relax3_leak1", "relax3_leak4", "kick_pos", "kick_pos_injL", "kick_learned", "kick_fixed"):
        a.plot([], [], "o", color=C[arm], label=LABEL[arm])
    a.plot([], [], "o", color="0.3", label="H"); a.plot([], [], "s", color="0.3", mfc="white", label="L")
    a.legend(fontsize=6.2, loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "fig_scale")


def fig_trace():
    """One-run error at depth 1024 against training step (seed 0, every 5k), at test noise 0.5 and 0.25;
    20k seed means +- sd as error bars."""
    arms = ["eqr", "twinL_leak2", "twinL_leak3", "twinL_leak10"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, N in zip(axes, (0.5, 0.25)):
        for j, arm in enumerate(arms):
            pts = sorted((r["step"], r["single"]) for r in T.RUNS_SCALING if r["arm"] == arm and r["seed"] == 0 and r["D"] == 1024
                         and r["B"] == 1 and abs(r["N"] - N) < 1e-9)
            x, y = zip(*pts)
            ax.plot(np.array(x) / 1000, 1 - np.array(y), marker="o", color=C[arm], label=LABEL[arm] + " (seed 0)", ms=3.2,
                    lw=1.6 if arm == "eqr" else 1.1)
            st = T.stat(T.runs(T.RUNS_SCALING, arm, 20000, D=1024, B=1, N=N), "single")
            if st and st[2] > 1:
                ax.errorbar(20 + 0.6 * (j - 1.5), 1 - st[0], yerr=st[1], fmt="s", color=C[arm], ms=3.5, mfc="white", capsize=2,
                            lw=0.9)
        if N == 0.5:
            rel = next(r["single"] for r in T.scaling_rows() if r["arm"] == "eqr-released" and r["D"] == 1024 and r["B"] == 1)
            ax.axhline(1 - rel, ls=":", color=C["eqr-released"], lw=1.0)
            ax.text(50.5, (1 - rel) * 0.93, "EqR released (50k)", color=C["eqr-released"], fontsize=6.6, ha="right", va="top")
        ax.set_yscale("log")
        ax.set_xlabel("training step (k)")
        ax.set_title(f"({'ab'[N == 0.25]}) depth 1024, one run, test noise $\\sigma={N:g}$")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xlim(8, 52)
        ax.grid(axis="y", which="both", color="0.92", lw=0.5)
    axes[0].set_ylabel("error rate")
    axes[0].set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["1%", "2%", "5%", "10%", "20%", "50%"])
    axes[0].set_ylim(0.009, 0.65)
    axes[1].legend(loc="upper right", fontsize=6.4, handletextpad=0.3)
    axes[1].text(0.97, 0.04, "open squares: mean $\\pm$ sd of seeds 0-2 at 20k", transform=axes[1].transAxes, ha="right",
                 fontsize=6.4, color="0.3")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_trace")


def fig_sky():
    """Skyscrapers-7: in-training test exact accuracy (16 iterations, noise 0.01) against training step."""
    fig, ax = plt.subplots(figsize=(0.55 * W, 2.3))
    spec = [("eqr_s0_1k", "EqR, 1k puzzles", C["eqr"], ":"), ("eqr_s0_10k", "EqR, 10k puzzles", C["eqr"], "--"),
            ("eqr_s0_100k", "EqR, 100k puzzles", C["eqr"], "-"), ("twinL_leak3_s0_100k", r"twin $\delta{=}3$, 100k puzzles", C["twinL_leak3"], "-")]
    for key, lab_, col, ls in spec:
        ev = T.SKY[key]["eval"]
        ax.plot([s / 1000 for s, _, _ in ev], [e for _, e, _ in ev], ls=ls, marker="o", ms=2.8, color=col, label=lab_)
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("test exact accuracy")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(-0.02, 1.0)
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.legend(loc="upper left", fontsize=6.4, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig_sky")


if __name__ == "__main__":
    for f in (fig_training, fig_retention, fig_depth, fig_breadth, fig_r3_depth, fig_noisedep, fig_noise, fig_escape, fig_auroc, fig_spectral, fig_ritz,
              fig_recursion, fig_inner, fig_flip, fig_scale, fig_trace, fig_sky):
        f()
        print("ok", f.__name__)
