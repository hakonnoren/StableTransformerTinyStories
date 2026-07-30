"""
Figures from the v2 per-item records.

Reads the same JSONL as analysis/report.py and reuses its compare() so every
figure and every table come from one aggregation path — a plot that disagrees with
the report would otherwise be impossible to diagnose.

Every figure shows uncertainty. A point estimate with no interval is what made the
v1 results unreadable, and a plot hides that even better than a table does.

Design notes (dataviz skill):
  - Categorical palette is the reference instance's first five slots IN FIXED
    ORDER, one per arm, assigned by arm identity and never cycled or re-assigned
    when a filter drops an arm. Validated: all hard gates pass in light mode
    (worst adjacent CVD ΔE 9.1, normal-vision 19.6).
  - That validation raises one WARN: aqua/yellow/magenta sit below 3:1 contrast on
    a white surface, which obligates *relief* — so line charts carry direct
    end-labels, and report.md is the table view. Do not remove either.
  - No dual axes anywhere. Where two measures share a figure they share one scale
    (both probabilities) or become separate panels.
  - Δ-vs-reference panels use the diverging job (below/above a baseline): blue for
    "better than reference", red for "worse", neutral gray zero line.
  - Light mode only, deliberately: these are PDFs bound for LaTeX, not a themed
    web page.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .records import read_records  # noqa: E402
from .report import compare  # noqa: E402
from .stats import bootstrap_ci, proportion  # noqa: E402

# --- reference palette, categorical slots 1..8, light mode (fixed order) -------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
DIVERGE_GOOD = "#2a78d6"     # better than reference
DIVERGE_BAD = "#e34948"      # worse than reference
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK_2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "lines.markersize": 5.5,
})


def arm_colors(arms: list[str]) -> dict[str, str]:
    """Assign palette slots by arm IDENTITY in a stable sort order.

    Stable so the same arm keeps its hue across every figure, and across runs where
    an arm is missing — color must follow the entity, never its rank.
    """
    return {a: SLOTS[i % len(SLOTS)] for i, a in enumerate(sorted(arms))}


def short(name: str) -> str:
    """Compact arm label for legends and direct labels."""
    return (name.replace("reversible_", "rev:").replace("baseline_baseline", "baseline")
            .replace("_narrow", "").replace("_scaling", "-scl").replace("_baseline", "-base"))


def _style(ax, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _save(fig, out_dir, name, fmt):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def _end_labels(ax, xs, series: dict[str, np.ndarray], colors, dx=0.01,
                min_gap_frac=0.045):
    """Direct-label each line at its right end (the relief the validator requires).

    Labels are pushed apart vertically when the curves converge — which they do
    here, since the arms are near-identical at high corruption rates and in the
    deep layers. Without this the five labels stack into an unreadable blob.
    Anchors stay on the true y value; only the text is displaced, with a leader
    line drawn when the offset is visible.
    """
    ends = [(name, float(ys[-1])) for name, ys in series.items()
            if np.isfinite(ys[-1])]
    if not ends:
        return
    x1 = float(xs[-1])
    xspan = (float(max(xs)) - float(min(xs))) or 1.0

    # De-collide in the AXIS'S OWN SCALE space. On a log axis, equal data gaps are
    # wildly unequal visual gaps, so a linear min_gap either overlaps labels at the
    # compressed end or flings them off the plot at the other. Doing it in log space
    # for log axes fixes that without needing display coordinates — matplotlib's
    # transforms are not finalised until the figure is drawn, and querying them
    # early silently returns the identity.
    log_y = ax.get_yscale() == "log"
    fwd = (lambda v: np.log10(max(v, 1e-12))) if log_y else (lambda v: v)
    inv = (lambda v: 10.0 ** v) if log_y else (lambda v: v)

    y0, y1 = ax.get_ylim()
    lo, hi = fwd(min(y0, y1)), fwd(max(y0, y1))
    min_gap = min_gap_frac * (hi - lo)

    ends.sort(key=lambda t: fwd(t[1]))
    placed: list[float] = []
    for _, y in ends:
        t = min(max(fwd(y), lo), hi)
        if placed and t - placed[-1] < min_gap:
            t = placed[-1] + min_gap
        placed.append(t)
    over = placed[-1] - hi
    if over > 0:
        placed = [p - over for p in placed]

    x_text = x1 + dx * xspan
    for (name, y_true), t in zip(ends, placed):
        y_text = inv(t)
        ax.annotate(short(name), (x_text, y_text), color=colors[name],
                    fontsize=7, va="center", ha="left", weight="bold",
                    annotation_clip=False)
        if abs(t - fwd(y_true)) > 0.2 * min_gap:
            ax.plot([x1, x_text - 0.02 * xspan], [y_true, y_text],
                    color=colors[name], lw=0.5, alpha=0.5, zorder=2, clip_on=False)


# ------------------------------------------------------------------ P1.4 strata
STRATA = [
    ("mean_loss", "all tokens"),
    ("loss_first", "first occurrence in story"),
    ("loss_repeat", "repeated token"),
    ("loss_word_init", "word-initial"),
    ("loss_subword", "continuation subword"),
    ("loss_punct", "punctuation"),
    ("loss_freq_top16", "freq rank <16"),
    ("loss_freq_top128", "freq rank 16–128"),
    ("loss_freq_top1k", "freq rank 128–1k"),
    ("loss_freq_top4k", "freq rank 1k–4k"),
    ("loss_freq_gt4k", "freq rank >4k"),
]


def plot_strata_delta(recs, ref, out_dir, fmt, n_boot=2000):
    """Δ mean loss vs the reference arm, per stratum, with paired-bootstrap CIs.

    This is the figure the 0.01-nat aggregate gap cannot give you: it shows WHERE
    an arm's loss difference lives. Diverging color job — left of zero is better
    than the reference, right is worse.
    """
    rows = []
    for key, label in STRATA:
        cmp = compare(recs, key, ref, n_boot=n_boot)
        if not cmp:
            continue
        for arm, d in cmp["arms"].items():
            if arm == cmp["ref"] or "paired" not in d:
                continue
            p = d["paired"]
            rows.append((label, arm, p["delta"], p["ci_lo"], p["ci_hi"],
                         p.get("q_value", 1.0)))
    if not rows:
        return None

    arms = sorted({r[1] for r in rows})
    labels = [lab for _, lab in STRATA if any(r[0] == lab for r in rows)]
    colors = arm_colors(arms + [ref])

    fig, ax = plt.subplots(figsize=(6.4, 0.40 * len(labels) + 1.5))
    dodge = np.linspace(-0.26, 0.26, len(arms)) if len(arms) > 1 else [0.0]

    for j, arm in enumerate(arms):
        ys, xs, los, his, sig = [], [], [], [], []
        for i, lab in enumerate(labels):
            m = [r for r in rows if r[0] == lab and r[1] == arm]
            if not m:
                continue
            _, _, d, lo, hi, q = m[0]
            ys.append(i + dodge[j]); xs.append(d); los.append(d - lo); his.append(hi - d)
            sig.append(q < 0.05)
        ax.errorbar(xs, ys, xerr=[los, his], fmt="o", color=colors[arm],
                    ecolor=colors[arm], elinewidth=1.2, capsize=0, markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3,
                    label=short(arm))
        # hollow out non-significant points so the eye lands on the real ones
        ns = [(x, y) for x, y, s in zip(xs, ys, sig) if not s]
        if ns:
            ax.scatter([p[0] for p in ns], [p[1] for p in ns], s=20,
                       facecolors=SURFACE, edgecolors=colors[arm], linewidths=1.0,
                       zorder=4)

    ax.axvline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.4, -1.0)          # headroom so the caption clears data
    # ASCII only in axis text: the sans stack used for PDF export has no arrow glyphs,
    # and a missing glyph renders as a tofu box rather than failing loudly.
    _style(ax, xlabel=f"delta mean loss vs {short(ref)}  (nats; left = better)",
           title="P1.4  where the loss difference lives", grid_axis="x")
    ax.annotate("filled = significant after BH-FDR;  hollow = not",
                xy=(0.0, -0.75), xycoords=("axes fraction", "data"),
                fontsize=7, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 - 0.6 / len(labels)),
              ncol=min(4, len(arms)))
    return _save(fig, out_dir, "p14_strata_delta", fmt)


def plot_loss_by_position(recs, out_dir, fmt, n_boot=2000):
    """Mean loss vs position-in-story, per arm — the P1.5 curve in full."""
    buckets = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)
    xs = np.arange(len(buckets), dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    series = {}
    for arm in arms:
        mid, lo, hi = [], [], []
        for b in buckets:
            vals = [r[f"loss_pos_{b[0]}_{b[1]}"] for r in recs
                    if r["arm"] == arm and f"loss_pos_{b[0]}_{b[1]}" in r]
            if not vals:
                mid.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            ci = bootstrap_ci(vals, n_boot=n_boot)
            mid.append(ci.point); lo.append(ci.lo); hi.append(ci.hi)
        mid = np.array(mid)
        ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, mid, color=colors[arm], zorder=3, marker="o",
                markeredgecolor=SURFACE, markeredgewidth=0.7)
        series[arm] = mid
    _end_labels(ax, xs, series, colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}–{b}" for a, b in buckets])
    _style(ax, xlabel="token position within story", ylabel="mean loss (nats)",
           title="P1.5  loss vs position — does the arm exploit context?")
    ax.set_xlim(-0.2, len(buckets) - 0.3)
    return _save(fig, out_dir, "p15_loss_by_position", fmt)


def plot_corruption(recs, out_dir, fmt, n_boot=2000):
    """P1.1 corruption robustness: TV vs corruption rate, with CI bands."""
    rates = sorted({float(k.replace("tv_rate_", "")) for r in recs for k in r
                    if k.startswith("tv_rate_")})
    if not rates:
        return None
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)
    xs = np.array(rates)

    # wide gutter: the left panel's direct end-labels sit in the space between the
    # panels, and would otherwise land on the right panel's y tick labels
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4),
                             gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    series = {}
    for arm in arms:
        mid, lo, hi = [], [], []
        for rt in rates:
            vals = [r[f"tv_rate_{rt}"] for r in recs if r["arm"] == arm]
            ci = bootstrap_ci(vals, n_boot=n_boot)
            mid.append(ci.point); lo.append(ci.lo); hi.append(ci.hi)
        mid = np.array(mid)
        ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, mid, color=colors[arm], marker="o", zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.7)
        series[arm] = mid
    _end_labels(ax, xs, series, colors)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
    ax.set_xlim(rates[0] - 0.01, rates[-1] + 0.02)
    _style(ax, xlabel="fraction of prefix tokens replaced",
           ylabel="mean TV vs clean next-token dist",
           title="P1.1  corruption robustness (lower = more robust)")

    # context sensitivity: one number per arm -> dot plot with CI, not a bar chart
    ax2 = axes[1]
    ys = np.arange(len(arms), dtype=float)
    for i, arm in enumerate(arms):
        vals = [r["ctx_sens"] for r in recs if r["arm"] == arm and "ctx_sens" in r]
        ci = bootstrap_ci(vals, n_boot=n_boot)
        ax2.errorbar([ci.point], [i], xerr=[[ci.point - ci.lo], [ci.hi - ci.point]],
                     fmt="o", color=colors[arm], ecolor=colors[arm], elinewidth=1.3,
                     markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([short(a) for a in arms])
    ax2.invert_yaxis()
    _style(ax2, xlabel="TV after flipping one early token",
           title="P1.1  context sensitivity", grid_axis="x")
    return _save(fig, out_dir, "p11_perturbation", fmt)


def plot_icl_and_closure(icl, cps, out_dir, fmt, n_boot=2000):
    """P1.5 in-context score and P1.6 closure calibration, as dot plots with CIs.

    Two panels rather than two y-axes on one plot. The closure panel puts both
    probabilities on ONE shared axis (they are the same unit), which is what makes
    the comparison legal.
    """
    arms = sorted({r["arm"] for r in (icl or cps)})
    colors = arm_colors(arms)
    have_icl, have_cps = bool(icl), bool(cps)
    ncol = int(have_icl) + int(have_cps)
    if ncol == 0:
        return None
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.0), squeeze=False)
    axes = axes[0]
    k = 0

    if have_icl:
        ax = axes[k]; k += 1
        for i, arm in enumerate(arms):
            vals = [r["icl_score"] for r in icl if r["arm"] == arm]
            if not vals:
                continue
            ci = bootstrap_ci(vals, n_boot=n_boot)
            ax.errorbar([ci.point], [i], xerr=[[ci.point - ci.lo], [ci.hi - ci.point]],
                        fmt="o", color=colors[arm], ecolor=colors[arm], elinewidth=1.3,
                        markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.7)
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in arms])
        ax.invert_yaxis()
        _style(ax, xlabel="loss@32–64 − loss@256–512 (nats)",
               title="P1.5  in-context-use score (higher = uses context more)",
               grid_axis="x")

    if have_cps:
        ax = axes[k]
        for i, arm in enumerate(arms):
            for key, mark, lab in (("p_eot_at_end", "o", "at true story end"),
                                   ("p_eot_midstory_max", "D", "max mid-story")):
                vals = [r[key] for r in cps if r["arm"] == arm and key in r]
                if not vals:
                    continue
                ci = bootstrap_ci(vals, n_boot=n_boot)
                off = -0.16 if mark == "o" else 0.16
                ax.errorbar([ci.point], [i + off],
                            xerr=[[ci.point - ci.lo], [ci.hi - ci.point]],
                            fmt=mark, color=colors[arm], ecolor=colors[arm],
                            elinewidth=1.3, markersize=5.0,
                            markerfacecolor=colors[arm] if mark == "o" else SURFACE,
                            markeredgecolor=colors[arm], markeredgewidth=1.0)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in arms])
        ax.invert_yaxis()
        _style(ax, xlabel="P(EOT)  — one shared probability scale",
               title="P1.6  story-closure calibration", grid_axis="x")
        ax.legend(handles=[
            Line2D([], [], marker="o", color=INK_2, ls="", label="at true story end"),
            Line2D([], [], marker="D", color=INK_2, ls="", markerfacecolor=SURFACE,
                   label="max mid-story (false-stop pressure)")],
            loc="lower right")
    return _save(fig, out_dir, "p15_p16_context_closure", fmt)


# ------------------------------------------------------------------ P1.3 depth
def plot_depth_curves(recs, out_dir, fmt, n_boot=1000):
    """Per-layer early-exit loss with CI bands; one panel per (lens, stream)."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        groups[(r["lens"], r["policy"])][r["arm"]][r["layer"]].append(r["loss"])
    if not groups:
        return None
    keys = sorted(groups, key=lambda k: (k[0] != "raw", k[1]))
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.52, "wspace": 0.30})
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)

    for idx, key in enumerate(keys):
        ax = axes[idx // ncol][idx % ncol]
        per_arm = groups[key]
        layers = sorted({l for d in per_arm.values() for l in d})
        xs = np.array(layers, dtype=float)
        series = {}
        for arm in sorted(per_arm):
            mid, lo, hi = [], [], []
            for l in layers:
                v = per_arm[arm].get(l, [])
                if not v:
                    mid.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                ci = bootstrap_ci(v, n_boot=n_boot)
                mid.append(ci.point); lo.append(ci.lo); hi.append(ci.hi)
            mid = np.array(mid)
            ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, mid, color=colors[arm], zorder=3)
            series[arm] = mid
        _end_labels(ax, xs, series, colors)
        lens, policy = key
        ax.set_xticks(layers)
        _style(ax, xlabel="layer", ylabel="early-exit loss (nats)",
               title=f"P1.3  {lens} lens · stream = {policy}")
        ax.set_xlim(-0.3, layers[-1] + 0.9)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return _save(fig, out_dir, "p13_depth_curves", fmt)


def plot_lens_quality(recs, out_dir, fmt):
    """Basis change across depth: raw vs tuned lens KL, and translator size.

    Left  — per-layer KL(p_final || p_lens) for the raw lens (dashed) and the tuned
            lens (solid). The GAP between them is basis change: an affine translator
            can only undo a change of basis, so anything it removes was never
            missing information. What survives is the layer genuinely not encoding
            the final prediction.
    Right — per-layer ||A_l||_F / ||I||_F, the size of the correction the lens had to
            learn. Needs no data; it is a property of the fitted map.
    """
    kl = [r for r in recs if r.get("lens") in ("raw", "tuned")]
    tr = [r for r in recs if r.get("lens") == "translator"]
    if not kl and not tr:
        return None
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5),
                             gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    layers = sorted({r["layer"] for r in kl})
    xs = np.array(layers, dtype=float)
    tuned_series = {}
    n_exact_zero = 0
    for arm in arms:
        for lens_name, ls, lw in (("raw", (0, (3, 2)), 1.2), ("tuned", "-", 1.9)):
            ys = np.array([next((r["kl"] for r in kl if r["arm"] == arm
                                 and r["layer"] == l and r["lens"] == lens_name),
                                np.nan) for l in layers], dtype=float)
            if lens_name == "raw":
                # The raw lens at the LAST layer is the model's own forward pass, so
                # its KL is exactly 0 — a log axis would drop the point silently and
                # make the curve look like it merely got small. Mask it and say so.
                n_exact_zero += int(np.sum(ys == 0.0))
                ys = np.where(ys == 0.0, np.nan, ys)
            ax.plot(xs, ys, ls=ls, lw=lw, color=colors[arm], zorder=3,
                    alpha=0.75 if lens_name == "raw" else 1.0)
            if lens_name == "tuned":
                tuned_series[arm] = ys
    # scale must be set before de-colliding labels: _end_labels reads get_yscale()
    ax.set_yscale("log")
    _end_labels(ax, xs, tuned_series, colors)
    ax.set_xticks(layers)
    _style(ax, xlabel="layer", ylabel="KL(final || lens)   [log scale]",
           title="Tuned lens · dashed = raw, solid = tuned")
    ax.set_xlim(-0.3, layers[-1] + 0.9)
    handles = [Line2D([], [], color=INK_2, ls=(0, (3, 2)), lw=1.2, label="raw logit lens"),
               Line2D([], [], color=INK_2, ls="-", lw=1.9, label="tuned lens")]
    ax.legend(handles=handles, loc="lower left")
    if n_exact_zero:
        ax.annotate(f"raw KL is exactly 0 at L{layers[-1]} for all {n_exact_zero} arms\n"
                    f"(the model's own readout) — omitted on a log axis",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6.5,
                    color=MUTED, va="bottom")
        ax.legend(handles=handles, loc="lower left",
                  bbox_to_anchor=(0.0, 0.10))

    ax2 = axes[1]
    if tr:
        tlayers = sorted({r["layer"] for r in tr})
        xs2 = np.array(tlayers, dtype=float)
        s2 = {}
        for arm in arms:
            ys = [next((r["rel_A"] for r in tr if r["arm"] == arm and r["layer"] == l),
                       np.nan) for l in tlayers]
            ax2.plot(xs2, ys, color=colors[arm], marker="o", markersize=4,
                     markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
            s2[arm] = np.array(ys)
        _end_labels(ax2, xs2, s2, colors)
        ax2.set_xticks(tlayers)
        _style(ax2, xlabel="layer",
               ylabel=r"$\|A_\ell\|_F / \|I\|_F$",
               title="Size of the affine correction the lens had to learn")
        ax2.set_xlim(-0.3, tlayers[-1] + 0.9)
        ax2.set_ylim(bottom=0)
    else:
        ax2.axis("off")
    return _save(fig, out_dir, "tuned_lens_quality", fmt)


# ------------------------------------------------------------------ P2.5 agreement
def plot_agreement(recs, out_dir, fmt, n_boot=2000):
    """Accuracy vs #attractors, split by whether attractors match subject number."""
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)

    for k, match in enumerate((True, False)):
        ax = axes[k]
        sub = [r for r in recs if bool(r["attractor_match"]) == match]
        ns = sorted({r["n_attractors"] for r in sub})
        if not ns:
            ax.axis("off"); continue
        xs = np.array(ns, dtype=float)
        series = {}
        for arm in arms:
            mid, lo, hi = [], [], []
            for na in ns:
                v = [r["correct"] for r in sub
                     if r["arm"] == arm and r["n_attractors"] == na]
                if not v:
                    mid.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                ci = proportion(v)
                mid.append(ci.point); lo.append(ci.lo); hi.append(ci.hi)
            mid = np.array(mid)
            ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, mid, color=colors[arm], marker="o", zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
            series[arm] = mid
        _end_labels(ax, xs, series, colors)
        ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(ns)
        ax.set_xlim(min(ns) - 0.15, max(ns) + 0.6)
        _style(ax, xlabel="number of attractors",
               ylabel="P(target) > P(foil)" if k == 0 else "",
               title=f"P2.5  attractors {'MATCH' if match else 'MISMATCH'} "
                     f"subject number")
        if k == 0:
            ax.annotate("chance", (min(ns), 0.5), xytext=(2, 3),
                        textcoords="offset points", fontsize=6.5, color=MUTED)
    return _save(fig, out_dir, "p25_agreement", fmt)


def plot_resolution_depth(recs, out_dir, fmt, n_boot=2000):
    """E1: where in depth the agreement decision resolves, per arm and lens."""
    lenses = sorted({r["lens"] for r in recs})
    arms = sorted({r["arm"] for r in recs})
    colors = arm_colors(arms)
    fig, axes = plt.subplots(1, len(lenses), figsize=(4.6 * len(lenses), 3.2),
                             squeeze=False, sharex=True)
    for k, ln in enumerate(lenses):
        ax = axes[0][k]
        sub = [r for r in recs if r["lens"] == ln]
        for i, arm in enumerate(arms):
            v = [r["resolution_depth"] for r in sub if r["arm"] == arm]
            if not v:
                continue
            ci = bootstrap_ci(v, n_boot=n_boot)
            ax.errorbar([ci.point], [i], xerr=[[ci.point - ci.lo], [ci.hi - ci.point]],
                        fmt="o", color=colors[arm], ecolor=colors[arm], elinewidth=1.3,
                        markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.7)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in arms])
        ax.invert_yaxis()
        _style(ax, xlabel="first layer with a stable positive margin",
               title=f"E1  resolution depth · {ln} lens", grid_axis="x")
    return _save(fig, out_dir, "e1_resolution_depth", fmt)


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out_dir", default=None, help="default: <dir>/figures")
    ap.add_argument("--ref", default=None, help="reference arm for Δ panels")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.dir, "figures")

    def load(name):
        p = os.path.join(args.dir, f"{name}.jsonl")
        return read_records(p) if os.path.exists(p) else []

    cps, icl, pert = load("corpus"), load("icl"), load("perturbation")
    lens, agree, rd = load("lens_depth"), load("agreement"), load("resolution_depth")
    lq = load("lens_quality")

    ref = args.ref
    if ref is None and cps:
        arms = sorted({r["arm"] for r in cps})
        ref = next((a for a in arms if "baseline_baseline" in a), arms[0])

    print(f"Figures -> {out_dir}  (format={args.format})")
    made, skipped = [], []
    for name, fn in (
        ("p14_strata_delta", lambda: plot_strata_delta(cps, ref, out_dir, args.format, args.n_boot) if cps else None),
        ("p15_loss_by_position", lambda: plot_loss_by_position(cps, out_dir, args.format, args.n_boot) if cps else None),
        ("p11_perturbation", lambda: plot_corruption(pert, out_dir, args.format, args.n_boot) if pert else None),
        ("p15_p16_context_closure", lambda: plot_icl_and_closure(icl, cps, out_dir, args.format, args.n_boot) if (icl or cps) else None),
        ("tuned_lens_quality", lambda: plot_lens_quality(lq, out_dir, args.format) if lq else None),
        ("p13_depth_curves", lambda: plot_depth_curves(lens, out_dir, args.format, max(500, args.n_boot // 2)) if lens else None),
        ("p25_agreement", lambda: plot_agreement(agree, out_dir, args.format, args.n_boot) if agree else None),
        ("e1_resolution_depth", lambda: plot_resolution_depth(rd, out_dir, args.format, args.n_boot) if rd else None),
    ):
        if fn() is None:
            skipped.append(name)
        else:
            made.append(name)
    if skipped:
        print(f"  skipped (no records yet): {', '.join(skipped)}")
    print(f"  {len(made)} figure(s) written")


if __name__ == "__main__":
    main()
