"""
Per-layer, per-dimension alpha/gamma distributions for reversible checkpoints.

Regimes with a per-dimension gamma/alpha: vf_scaling, vpb_scaling, vpm_scaling.
vpb_baseline is frozen at zero and lowrank_cayley blocks have no gamma/alpha at
all (a linear map instead) -- extract_alpha_gamma_full() returns {} for both, so
they are skipped here.

Unlike analysis/plots.py this reads checkpoints directly rather than JSONL
records: alpha/gamma live only in the model's parameters. cheap_metrics.py logs
their per-layer MEAN to wandb during training, but never the per-dimension
values, so there is no record file this could be computed from after the fact.

    python -m analysis.plot_alpha_gamma --ckpt_dir fetched/revparityreg_24979051 \\
        --out_dir results/revparityreg_24979051/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cheap_metrics import extract_alpha_gamma_full  # noqa: E402

from .loader import load_arms  # noqa: E402

REGIMES = ("vf_scaling", "vpb_scaling", "vpm_scaling")
COLOR_ALPHA = "steelblue"
COLOR_GAMMA = "indianred"


def _save(fig, out_dir, name, fmt):
    # Not analysis.plots._save: importing that module applies its rcParams (cream
    # background, custom fonts, spines off) globally, which this deliberately
    # plain-matplotlib module avoids.
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def load_results(ckpt_dir: str, only: list[str] | None = None) -> list[dict]:
    """Load reversible arms and extract their per-layer, per-dimension alpha/gamma.

    Returns a list of {"name", "alpha": (n_layer, n_embd), "gamma": (n_layer,
    n_embd)} dicts, one per arm that has per-dimension alpha/gamma (others are
    silently skipped -- see module docstring).
    """
    out = []
    for arm in load_arms(ckpt_dir, only=only):
        ag = extract_alpha_gamma_full(arm.model)
        if ag:
            out.append({"name": arm.name, **ag})
    return out


def plot_alpha_gamma_violin(result, ax=None, color_a=COLOR_ALPHA, color_g=COLOR_GAMMA):
    """Violin of alpha and gamma per layer for one arm.

    One violin per (param, layer) -- alpha violins colored `color_a`, gamma
    violins `color_g`. Median dots overlaid; 0 = volume-preserving reference."""
    A, G = result["alpha"], result["gamma"]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    series, labels, colors = [], [], []
    for L in range(A.shape[0]):
        series.append(A[L]); labels.append(rf"$\alpha$ L{L}"); colors.append(color_a)
    for L in range(G.shape[0]):
        series.append(G[L]); labels.append(rf"$\gamma$ L{L}"); colors.append(color_g)
    positions = np.arange(len(series))
    parts = ax.violinplot(series, positions=positions, showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_edgecolor("0.3"); pc.set_alpha(0.75); pc.set_linewidth(0.5)
    ax.scatter(positions, [np.median(d) for d in series], color="k", s=10, zorder=3)
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(rf"{result['name']}: $\alpha/\gamma$ distribution per layer")
    return fig


def plot_alpha_gamma_lines(result, ax=None, color_a=COLOR_ALPHA, color_g=COLOR_GAMMA,
                           trace_alpha=0.12):
    """Alpha/gamma across depth, one thin low-opacity line per dimension plus
    the bold per-layer mean -- layers on the x-axis."""
    A, G = result["alpha"], result["gamma"]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = np.arange(A.shape[0])
    for label, M, c in ((r"$\alpha$", A, color_a), (r"$\gamma$", G, color_g)):
        for d in range(M.shape[1]):
            ax.plot(layers, M[:, d], color=c, alpha=trace_alpha, lw=0.8, zorder=1)
        ax.plot(layers, M.mean(axis=1), color=c, lw=2.2, zorder=3, label=f"{label} mean")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8, zorder=2)
    ax.legend(loc="best")
    ax.set_xlabel("layer")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.set_title(rf"{result['name']}: $\alpha/\gamma$ per channel across depth")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="fetched/revparityreg_24979051")
    ap.add_argument("--out_dir", default=None, help="default: <ckpt_dir>/figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"arm names to include (default: reversible_{{{','.join(REGIMES)}}}_narrow)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    only = args.only or [f"reversible_{r}_narrow" for r in REGIMES]
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "figures")

    results = load_results(args.ckpt_dir, only=only)
    if not results:
        raise SystemExit(f"no arms with per-dimension alpha/gamma under {args.ckpt_dir} (only={only})")
    for result in results:
        _save(plot_alpha_gamma_violin(result), out_dir,
              f"alpha_gamma_violin_{result['name']}", args.format)
        _save(plot_alpha_gamma_lines(result), out_dir,
              f"alpha_gamma_lines_{result['name']}", args.format)


if __name__ == "__main__":
    main()
