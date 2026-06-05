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
    ap.add_argument("--out_dir", default="plots/wandb_plots")
    ap.add_argument("--api_key_file", default="api_key_wnb.txt")
    ap.add_argument("--samples", type=int, default=5000, help="max history rows per run")
    ap.add_argument("--csv", action="store_true", help="also dump each run's history to CSV")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="output image format")
    ap.add_argument("--filter_config", nargs="*", default=None, metavar="KEY=VAL",
                    help="keep only runs whose config matches all KEY=VAL (e.g. max_steps=10000 n_layer=6)")
    ap.add_argument("--loss_xmin", type=int, default=None,
                    help="trim the loss plot to steps >= this and rescale y to that window "
                         "(default: ~12%% of max step, to drop the initial hockey-stick). Use 0 for full range.")
    ap.add_argument("--loss_linear", action="store_true", help="use linear y for the loss plot (default: log)")
    args = ap.parse_args()

    cfg_filters = {}
    for kv in (args.filter_config or []):
        k, _, v = kv.partition("=")
        cfg_filters[k] = v

    def cfg_match(run):
        for k, v in cfg_filters.items():
            cv = run.config.get(k)
            if str(cv) == v:
                continue
            try:
                if float(cv) == float(v):
                    continue
            except (TypeError, ValueError):
                pass
            return False
        return True

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
        if not cfg_match(r):
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

    def line_plot(keys, title, fname, ylabel=None, logy=False, xmin=None):
        """Overlay one curve per run for each present key (separate figure if
        multiple keys, else one). If xmin is given, the x-axis is trimmed to
        [xmin, ...] AND the y-axis is rescaled to just the data in that window
        (so an initial 'hockey-stick' transient doesn't flatten everything)."""
        present = [k for k in keys if any(k in hist[r.id].columns for r in runs)]
        if not present:
            return
        n = len(present)
        ncol = min(3, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
        for ax, key in zip(axes.flat, present):
            vis = []  # y-values within [xmin, ...] for autoscaling
            for r in runs:
                x, y = series(r.id, key)
                if x is not None and len(x):
                    ax.plot(x, y, label=labels[r.id], lw=1.6)
                    if xmin is not None:
                        vis.extend(y[x >= xmin].tolist())
            ax.set_title(key.replace("cheap/", ""))
            ax.set_xlabel("step")
            if ylabel:
                ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale("log")
            if xmin is not None:
                ax.set_xlim(left=xmin)
                vis = [v for v in vis if v == v]  # drop NaN
                if vis:
                    lo, hi = min(vis), max(vis)
                    if logy and lo > 0:
                        ax.set_ylim(lo * 0.97, hi * 1.03)
                    else:
                        pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
                        ax.set_ylim(lo - pad, hi + pad)
            ax.grid(alpha=0.3, which="both")
        for ax in axes.flat[len(present):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{fname}.{args.format}")
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
        out = os.path.join(args.out_dir, f"{fname}.{args.format}")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    def evolution_heatmap(metric, fname, cmap="viridis"):
        """One (layer x step) heatmap per run, sharing a colour scale, showing
        how the per-layer profile of `metric` evolves over training."""
        # build per-run matrices M[layer, step]
        mats, steps_ref = {}, None
        for r in runs:
            cols = []
            li = 0
            while f"cheap/{metric}/L{li}" in hist[r.id].columns:
                cols.append(f"cheap/{metric}/L{li}")
                li += 1
            if not cols:
                continue
            rows, steps = [], None
            for c in cols:
                s = hist[r.id][["_step", c]].dropna().set_index("_step")[c]
                rows.append(s)
                steps = s.index if steps is None else steps.union(s.index)
            steps = sorted(steps)
            M = np.full((len(cols), len(steps)), np.nan)
            sidx = {st: j for j, st in enumerate(steps)}
            for i, s in enumerate(rows):
                for st, v in s.items():
                    M[i, sidx[st]] = v
            mats[r.id] = (M, steps)
            steps_ref = steps if steps_ref is None else steps_ref
        if not mats:
            return
        allv = np.concatenate([M[np.isfinite(M)].ravel() for M, _ in mats.values()])
        vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
        nrun = len(mats)
        fig, axes = plt.subplots(1, nrun, figsize=(4.2 * nrun, 3.6), squeeze=False)
        im = None
        for ax, r in zip(axes.flat, [r for r in runs if r.id in mats]):
            M, steps = mats[r.id]
            im = ax.imshow(M, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[steps[0], steps[-1], M.shape[0] - 0.5, -0.5])
            ax.set_title(labels[r.id], fontsize=9)
            ax.set_xlabel("step")
            ax.set_ylabel("layer")
            ax.set_yticks(range(M.shape[0]))
        fig.colorbar(im, ax=list(axes.flat), shrink=0.85, label=metric)
        fig.suptitle(f"{metric} — layer x step evolution")
        out = os.path.join(args.out_dir, f"{fname}.{args.format}")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    # ---- figures ---------------------------------------------------------------
    # Loss: log-y by default, x trimmed past the initial hockey-stick (y rescaled
    # to the visible window) so models separate where the curves flatten.
    loss_xmin = args.loss_xmin
    if loss_xmin is None:
        maxstep = max((int(hist[r.id]["_step"].max()) for r in runs), default=0)
        loss_xmin = int(0.12 * maxstep)
    line_plot(["val_loss", "train_loss"], "Loss", "loss", ylabel="loss",
              logy=(not args.loss_linear), xmin=(loss_xmin or None))
    line_plot(["cheap/token_accuracy"], "Token accuracy", "accuracy", ylabel="acc")
    line_plot(["gen/distinct_1", "gen/distinct_2", "gen/rep_4"],
              "Generation quality (distinct-n up = diverse; rep-4 up = repetitive)", "generation")
    line_plot(
        ["cheap/gamma_mean_mean", "cheap/alpha_mean_mean", "cheap/block_logdet_mean"],
        "Reversible scaling (alpha/gamma & log|det|)", "scaling",
    )
    line_plot(
        ["cheap/actgrad_mean", "cheap/wgrad_mean"],
        "Gradient norms (mean over layers)", "grad_norms", logy=True,
    )
    line_plot(
        ["cheap/act_rms_mean", "cheap/act_erank_mean", "cheap/act_token_sim_mean"],
        "Representation geometry (mean over layers)", "representation",
    )
    line_plot(
        ["cheap/attn_entropy_norm_mean", "cheap/attn_top1_mean",
         "cheap/attn_k90_mean", "cheap/attn_key_mass_max_mean"],
        "Attention distribution (mean over layers/heads)", "attention",
    )
    for metric in ("act_erank", "actgrad", "act_token_sim", "attn_entropy_norm", "attn_top1"):
        depth_profile(metric, f"depth_{metric}")

    # training-time evolution of the per-layer structure (layer x step heatmaps)
    for metric in ("act_erank", "act_token_sim", "actgrad", "attn_entropy_norm", "gamma_mean"):
        evolution_heatmap(metric, f"evo_{metric}")

    print(f"\nDone. Plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
