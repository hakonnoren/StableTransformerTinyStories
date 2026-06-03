"""
Pull selected W&B runs and generate comparison plots (loss/accuracy + the cheap
diagnostics logged by cheap_metrics.py).

Auth: reads the API key from --api_key_file (default api_key_wnb.txt) into
WANDB_API_KEY if that env var isn't already set. The key is never printed.

Examples
--------
  # all full runs in the default project
  python analyze_wandb.py --min_step 1000 --out_dir wandb_plots

  # only specific runs
  python analyze_wandb.py --runs baseline reversible_vpm_scaling reversible_vpb_baseline

Run it in an env that has wandb + matplotlib (e.g. conda env `torch`).
"""
import argparse
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="hakon-noren-ntnu")
    ap.add_argument("--project", default="sympformer-reversible")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run names to include (default: all). Repeats are disambiguated by id.")
    ap.add_argument("--min_step", type=int, default=0,
                    help="only keep runs whose last _step >= this (filters smoke tests)")
    ap.add_argument("--state", default="finished",
                    help="filter by run state (e.g. finished); empty string = any")
    ap.add_argument("--out_dir", default="wandb_plots")
    ap.add_argument("--api_key_file", default="api_key_wnb.txt")
    ap.add_argument("--samples", type=int, default=5000, help="max history rows per run")
    ap.add_argument("--csv", action="store_true", help="also dump each run's history to CSV")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY") and os.path.exists(args.api_key_file):
        with open(args.api_key_file) as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    os.makedirs(args.out_dir, exist_ok=True)
    api = wandb.Api()

    # ---- select runs ----------------------------------------------------------
    runs = []
    for r in api.runs(f"{args.entity}/{args.project}"):
        if args.state and r.state != args.state:
            continue
        if int(r.summary.get("_step", 0)) < args.min_step:
            continue
        if args.runs and r.name not in args.runs:
            continue
        runs.append(r)
    if not runs:
        raise SystemExit("No runs matched the filters.")

    # stable order + disambiguate duplicate names
    runs.sort(key=lambda r: (r.name, -int(r.summary.get("_step", 0))))
    name_counts = {}
    labels = {}
    for r in runs:
        name_counts[r.name] = name_counts.get(r.name, 0) + 1
    seen = {}
    for r in runs:
        if name_counts[r.name] > 1:
            seen[r.name] = seen.get(r.name, 0) + 1
            labels[r.id] = f"{r.name}#{r.id[:6]}"
        else:
            labels[r.id] = r.name

    print(f"Selected {len(runs)} runs:")
    for r in runs:
        print(f"  {labels[r.id]:32s} steps={r.summary.get('_step')} best_val={r.summary.get('best_val')}")

    # ---- fetch history ---------------------------------------------------------
    hist = {}
    for r in runs:
        df = r.history(samples=args.samples)
        hist[r.id] = df
        if args.csv:
            df.to_csv(os.path.join(args.out_dir, f"history_{labels[r.id].replace('/', '_')}.csv"), index=False)

    def series(rid, key):
        df = hist[rid]
        if key not in df.columns:
            return None, None
        s = df[["_step", key]].dropna()
        return s["_step"].to_numpy(), s[key].to_numpy()

    def line_plot(keys, title, fname, ylabel=None, logy=False):
        """Overlay one curve per run for each present key (separate figure if
        multiple keys, else one)."""
        present = [k for k in keys if any(k in hist[r.id].columns for r in runs)]
        if not present:
            return
        n = len(present)
        ncol = min(3, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
        for ax, key in zip(axes.flat, present):
            for r in runs:
                x, y = series(r.id, key)
                if x is not None and len(x):
                    ax.plot(x, y, label=labels[r.id], lw=1.6)
            ax.set_title(key.replace("cheap/", ""))
            ax.set_xlabel("step")
            if ylabel:
                ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
        for ax in axes.flat[len(present):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    def depth_profile(metric, fname):
        """Final-step per-layer profile: x=layer index, one line per run."""
        # discover layer indices from columns
        pat = re.compile(rf"^cheap/{re.escape(metric)}/L(\d+)$")
        any_cols = False
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in runs:
            cols = []
            for c in hist[r.id].columns:
                mobj = pat.match(c)
                if mobj:
                    cols.append((int(mobj.group(1)), c))
            if not cols:
                continue
            any_cols = True
            cols.sort()
            xs, ys = [], []
            for li, c in cols:
                s = hist[r.id][["_step", c]].dropna()
                if len(s):
                    xs.append(li)
                    ys.append(s[c].iloc[-1])
            ax.plot(xs, ys, marker="o", label=labels[r.id])
        if not any_cols:
            plt.close(fig)
            return
        ax.set_title(f"{metric} — final per-layer profile")
        ax.set_xlabel("layer index")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    # ---- figures ---------------------------------------------------------------
    line_plot(["val_loss", "train_loss"], "Loss", "loss.png", ylabel="loss")
    line_plot(["cheap/token_accuracy"], "Token accuracy", "accuracy.png", ylabel="acc")
    line_plot(
        ["cheap/gamma_mean_mean", "cheap/alpha_mean_mean", "cheap/block_logdet_mean"],
        "Reversible scaling (alpha/gamma & log|det|)", "scaling.png",
    )
    line_plot(
        ["cheap/actgrad_mean", "cheap/wgrad_mean"],
        "Gradient norms (mean over layers)", "grad_norms.png", logy=True,
    )
    line_plot(
        ["cheap/act_rms_mean", "cheap/act_erank_mean", "cheap/act_token_sim_mean"],
        "Representation geometry (mean over layers)", "representation.png",
    )
    line_plot(
        ["cheap/attn_entropy_norm_mean", "cheap/attn_top1_mean",
         "cheap/attn_k90_mean", "cheap/attn_key_mass_max_mean"],
        "Attention distribution (mean over layers/heads)", "attention.png",
    )
    for metric in ("act_erank", "actgrad", "act_token_sim", "attn_entropy_norm", "attn_top1"):
        depth_profile(metric, f"depth_{metric}.png")

    print(f"\nDone. Plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
