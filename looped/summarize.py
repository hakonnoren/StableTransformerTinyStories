"""Table + figures for one results folder of looped/run_looped.py.

    python looped/summarize.py results/looped/khop_K3 [more folders ...]

Writes <folder>/summary.md and <folder>/acc_vs_iters.png (accuracy and state residual vs iterations).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["vanilla", "kick_vp", "kick_fixed", "kick_learned", "relax15", "relax3", "relax3_learned", "relax1",
         "kick_pos", "relax3_leakL", "relax_bounded", "relax_budget"]
LABEL = {"vanilla": "Looped standard block", "kick_vp": "Kick, γ=α=0 (VP)", "kick_fixed": "Kick, fixed 0.005",
         "kick_learned": "Kick, learned", "relax15": "Relax, rate 15 (≈baseline)", "relax3": "Relax, rate 3, fixed",
         "relax3_learned": "Relax, rate 3, learned", "relax1": "Relax, rate 1, fixed",
         "kick_pos": "Kick, positive learned rates", "relax3_leakL": "Relax, learned rates + learned δ",
         "relax_bounded": "Relax, bounded learned rates", "relax_budget": "Relax, fixed budget γ+α"}


COLORS = {k: f"C{i}" for i, k in enumerate(ORDER)}
EXTRA_COLORS = ["C8", "C9", "k", "olive", "teal", "navy", "maroon", "gold"]


def color(name, extra=[]):
    if name in COLORS:
        return COLORS[name]
    if name not in extra:
        extra.append(name)
    return EXTRA_COLORS[extra.index(name) % len(EXTRA_COLORS)]


TAGS = {"_noinj": "no input injection", "_h10": "H keeps 10%", "_h25": "H keeps 25%", "_h50": "H keeps 50%",
        "_r12": "trained r 2-12", "_leak64": "δ = 1/64", "_leak16": "δ = 1/16", "_leak4": "δ = 1/4",
        "_post": "post-LN injection", "_leak16_post": "δ = 1/16 + post-LN injection",
        "_carry": "carry ≤64", "_leak16_carry": "δ = 1/16, carry ≤64", "_leak4_carry": "δ = 1/4, carry ≤64",
        "_zinit": "zero-init outputs"}


def label(name):
    if name in LABEL:
        return LABEL[name]
    base = max((k for k in ORDER if name.startswith(k)), key=len, default=None)
    if base is None:
        return name
    tag = name[len(base):]
    return f"{LABEL[base]}, {TAGS.get(tag, tag)}"


def load(folder):
    runs = {}
    for f in sorted(Path(folder).glob("*_seed*.json")):
        r = json.loads(f.read_text())
        runs[f.stem.rsplit("_seed", 1)[0]] = r
    key = lambda k: (ORDER.index(k.split("+")[0]) if k.split("+")[0] in ORDER else 99, k)
    return dict(sorted(runs.items(), key=lambda kv: key(kv[0])))


def fmt_rates(r):
    """γ, α (+ δ when present) of the first core layer; newer runs store (γ, α[, δ], linear-part radius)."""
    if not r["rates"]:
        return "–"
    v = list(r["rates"].values())[0]
    g, a = v[0], v[1]
    k = lambda x: np.exp(-np.mean(x))
    out = f"γ={np.mean(g):.3g} α={np.mean(a):.3g} (keep L {k(g):.2f}, H {k(a):.2f})"
    if len(v) == 4:
        out += f", δ={np.mean(v[2]):.3g}"
    if len(v) >= 3:
        out += f", linear radius {v[-1]:.3f}"
    return out


def table(runs, split="test"):
    any_r = next(iter(runs.values()))
    lo, hi = any_r["args"]["r_lo"], any_r["args"]["r_hi"]
    cols = sorted({lo, hi, 16, 32, 128})
    head = (f"| model | " + " | ".join(f"acc @{c}" + (" (train)" if lo <= c <= hi else "") for c in cols)
            + f" | correct@{hi} → wrong@128 (%) | step @128 | shift-free step @128 | shift-free norm @{hi} → @128 "
              f"| ρ shift-free @{hi} / @128 | rates | train s |")
    lines = [head, "|" + "---|" * (len(cols) + 7)]
    for name, r in runs.items():
        f = r["final"][split]
        acc = {int(k): v for k, v in f["acc"].items()}
        rr = {int(k): v for k, v in f["res_rel"].items()}
        ra = {int(k): v for k, v in f["res_abs"].items()}
        rq = {int(k): v for k, v in f.get("res_q", {}).items()}
        nq = {int(k): v for k, v in f.get("norm_q", {}).items()}
        sr = f.get("spectral_radius", {})
        rho = (f"{sr[f'{hi}q']['median']:.3f} / {sr['128q']['median']:.3f}" if f"{hi}q" in sr else "–")
        lines.append(f"| {label(name)} | " + " | ".join(f"{acc[c]:.3f}" for c in cols)
                     + f" | {f.get(f'overthink_{hi}_to_128', float('nan')):.1f} | {ra[128]:.2g} | "
                     + (f"{rq[128]:.2g} | {nq[hi]:.3g} → {nq[128]:.3g}" if rq else "– | –")
                     + f" | {rho} | {fmt_rates(r)} | {r['time_train_s']:.0f} |")
    return "\n".join(lines)


def depth_table(runs, split):
    lines = []
    for name, r in runs.items():
        f = r["final"][split]
        if "acc_by_group" not in f:
            return ""
        for g, accs in f["acc_by_group"].items():
            accs = {int(k): v for k, v in accs.items()}
            lines.append((name, int(g), f["count_by_group"][g], accs))
    if not lines:
        return ""
    cols = [4, 8, 16, 32, 128]
    out = ["| model | depth | n | " + " | ".join(f"acc @{c}" for c in cols) + " |", "|" + "---|" * (len(cols) + 3)]
    for name, g, n, accs in lines:
        out.append(f"| {label(name)} | {g} | {n} | " + " | ".join(f"{accs[c]:.3f}" for c in cols) + " |")
    return "\n".join(out)


def figure(runs, path, split="test", title=""):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    any_r = next(iter(runs.values()))
    lo, hi = any_r["args"]["r_lo"], any_r["args"]["r_hi"]
    for i, (name, r) in enumerate(runs.items()):
        f = r["final"][split]
        e_hi = r["args"]["r_hi"]
        rs = sorted(int(k) for k in f["acc"])
        acc = [f["acc"][str(k)] for k in rs]
        ab = [f["res_abs"][str(k)] for k in rs]
        st = "--" if name.startswith("kick") else ("-" if name.startswith("relax") else ":")
        ax[0].plot([max(k, 0.5) for k in rs], acc, st, marker="o", ms=3, label=label(name), color=color(name))
        ax[1].plot(rs[1:], ab[1:], st, marker="o", ms=3, color=color(name))
        c = r["curve"]
        ax[2].plot([e["step"] for e in c], [e["acc"][str(e_hi)] for e in c], st, color=color(name))
    for a_ in ax[:2]:
        a_.set_xscale("log", base=2); a_.axvspan(lo, hi, color="0.9", zorder=0); a_.set_xlabel("iterations r")
    ax[0].set_ylabel(f"{split} accuracy"); ax[0].set_ylim(-0.02, 1.02)
    ax[1].set_yscale("log"); ax[1].set_ylabel("‖s_r − s_{r−1}‖ (per sequence)")
    ax[2].set_xlabel("training step"); ax[2].set_ylabel(f"accuracy @ r={hi} (512 test seqs)")
    ax[0].set_title(f"{title} accuracy vs iterations (grey: training r)")
    ax[1].set_title("absolute state residual (flat = constant drift)"); ax[2].set_title("training curve")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def main(folders):
    for folder in folders:
        folder = Path(folder)
        runs = load(folder)
        if not runs:
            print(f"{folder}: no runs"); continue
        md = [f"# {folder.name}\n", f"single seed; args of first run: `{json.dumps(next(iter(runs.values()))['args'])}`\n"]
        for split in next(iter(runs.values()))["final"]:
            md += [f"## {split}\n", table(runs, split), ""]
            dt = depth_table(runs, split)
            if dt:
                md += [f"### {split} by nesting depth\n", dt, ""]
            figure(runs, folder / f"acc_vs_iters_{split}.png", split, title=f"{folder.name} {split}:")
        (folder / "summary.md").write_text("\n".join(md))
        print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
