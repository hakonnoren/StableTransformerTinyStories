"""
Aggregate per-item records into tables with intervals and paired tests.

Reads the JSONL written by analysis/run_suite.py and produces, for every metric:
  - each arm's point estimate with a bootstrap/Wilson interval over items
  - the paired comparison against the reference arm (McNemar for binary, paired
    bootstrap for continuous), with BH-FDR across the comparison family

Runs entirely on saved records, so re-slicing or re-testing needs no GPU.

    python -m analysis.report --dir results/revparityreg_24957997
    python -m analysis.report --dir ... --ref baseline_baseline --md report.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .records import read_records
from .stats import (benjamini_hochberg, bootstrap_ci, mcnemar, paired_bootstrap,
                    proportion)

BINARY_METRICS = {"correct", "resolved", "passed"}


def _by_arm(recs: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """{arm: {item_id: value}} for one metric, dropping rows that lack it."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in recs:
        if metric in r and r[metric] is not None:
            out[r["arm"]][r["item_id"]] = r[metric]
    return out


def compare(recs: list[dict], metric: str, ref: str | None = None,
            filt=None, n_boot: int = 10000) -> dict:
    """Per-arm interval + paired test vs `ref` on the shared item set."""
    rows = [r for r in recs if (filt is None or filt(r))]
    per = _by_arm(rows, metric)
    if not per:
        return {}
    arms = sorted(per)
    ref = ref if ref in per else arms[0]
    binary = metric in BINARY_METRICS

    out = {"metric": metric, "ref": ref, "binary": binary, "arms": {}}
    for a in arms:
        vals = np.array(list(per[a].values()), dtype=float)
        ci = proportion(vals) if binary else bootstrap_ci(vals, n_boot=n_boot)
        out["arms"][a] = {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n}

    # paired tests on the intersection of item ids
    pvals, keys = [], []
    for a in arms:
        if a == ref:
            continue
        shared = sorted(set(per[a]) & set(per[ref]))
        if not shared:
            continue
        av = np.array([per[a][i] for i in shared], dtype=float)
        bv = np.array([per[ref][i] for i in shared], dtype=float)
        res = mcnemar(av, bv) if binary else paired_bootstrap(av, bv, n_boot=n_boot)
        out["arms"][a]["paired"] = res.to_dict()
        pvals.append(res.p_value)
        keys.append(a)
    if pvals:
        _, q = benjamini_hochberg(pvals)
        for a, qq in zip(keys, q):
            out["arms"][a]["paired"]["q_value"] = float(qq)
    return out


def fmt_table(cmp: dict, title: str, lines: list[str]) -> None:
    if not cmp:
        return
    ref = cmp["ref"]
    lines.append(f"\n### {title}")
    lines.append(f"reference arm: `{ref}`  ({'binary → McNemar' if cmp['binary'] else 'continuous → paired bootstrap'})\n")
    lines.append("| arm | estimate | 95% CI | n | Δ vs ref | p | q |")
    lines.append("|---|---|---|---|---|---|---|")
    for a, d in sorted(cmp["arms"].items(), key=lambda kv: -kv[1]["point"]):
        ci = f"[{d['lo']:.3f}, {d['hi']:.3f}]" if np.isfinite(d["lo"]) else "—"
        if a == ref:
            lines.append(f"| **{a}** (ref) | {d['point']:.3f} | {ci} | {d['n']} | — | — | — |")
        else:
            p = d.get("paired", {})
            star = "*" if p.get("q_value", 1) < 0.05 else ""
            lines.append(
                f"| {a} | {d['point']:.3f} | {ci} | {d['n']} | "
                f"{p.get('delta', float('nan')):+.3f} | {p.get('p_value', float('nan')):.4f} | "
                f"{p.get('q_value', float('nan')):.4f}{star} |")


def depth_curves(recs: list[dict], lines: list[str], n_boot: int = 2000) -> None:
    """Per-layer loss with CIs, one block per (lens, policy)."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        groups[(r["lens"], r["policy"])][r["arm"]][r["layer"]].append(r["loss"])
    for (lens, policy), per_arm in sorted(groups.items()):
        lines.append(f"\n### P1.3 depth curve — lens={lens}, stream={policy}")
        layers = sorted({l for d in per_arm.values() for l in d})
        lines.append("| arm | " + " | ".join(f"L{l}" for l in layers) + " | argmin |")
        lines.append("|---" * (len(layers) + 2) + "|")
        for arm in sorted(per_arm):
            cells, means = [], []
            for l in layers:
                v = np.array(per_arm[arm].get(l, []), dtype=float)
                if v.size == 0:
                    cells.append("—"); means.append(np.inf); continue
                ci = bootstrap_ci(v, n_boot=n_boot)
                cells.append(f"{ci.point:.2f}<br><sub>±{(ci.hi-ci.lo)/2:.2f}</sub>")
                means.append(ci.point)
            lines.append(f"| {arm} | " + " | ".join(cells) +
                         f" | L{layers[int(np.argmin(means))]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default=None, help="reference arm (default: first alphabetically)")
    ap.add_argument("--md", default=None, help="also write markdown here")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    d = args.dir
    lines: list[str] = [f"# Behavior suite v2 — {d}"]

    mpath = os.path.join(d, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        lines.append(f"\ngit `{man.get('git_sha','?')[:10]}`"
                     f"{' (dirty)' if man.get('git_dirty') else ''} · "
                     f"torch {man.get('torch')} · device {man.get('device')}")
        lines.append("\n| arm | arch | regime | embed | attn | step | best_val | params |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a, info in sorted(man.get("arms", {}).items()):
            bv = info.get("best_val")
            lines.append(f"| {a} | {info.get('arch')} | {info.get('regime') or '—'} | "
                         f"{info.get('embed') or '—'} | {info.get('attn_impl')} | "
                         f"{info.get('step')} | {bv:.4f} | "
                         f"{(info.get('n_params') or 0)/1e6:.1f}M |")
        # re-derive warnings from the saved manifest
        from .records import RunManifest
        rm = RunManifest(**{k: v for k, v in man.items() if k in RunManifest.__dataclass_fields__})
        w = rm.warnings()
        if w:
            lines.append("\n**Comparability warnings**\n")
            for x in w:
                lines.append(f"- {x}")

    def load(name):
        p = os.path.join(d, f"{name}.jsonl")
        return read_records(p) if os.path.exists(p) else []

    # ---- P1.2
    leak = load("leak")
    if leak:
        lines.append("\n### P1.2 causality\n")
        lines.append("| arm | max abs Δ at i<j | trials | verdict |")
        lines.append("|---|---|---|---|")
        for r in sorted(leak, key=lambda r: r["arm"]):
            lines.append(f"| {r['arm']} | {r['max_abs_delta']:.2e} | {r['n_trials']} | "
                         f"{'PASS' if r['passed'] else '**LEAK**'} |")

    # ---- P1.4 / P1.5 / P1.6
    cps = load("corpus")
    if cps:
        fmt_table(compare(cps, "mean_loss", args.ref, n_boot=args.n_boot),
                  "P1.4 mean per-token loss (held-out stories)", lines)
        for key, title in (("loss_first", "P1.4 loss on FIRST occurrence of a token in the story"),
                           ("loss_repeat", "P1.4 loss on REPEATED tokens (copying/memory)"),
                           ("loss_word_init", "P1.4 loss on word-initial tokens"),
                           ("loss_freq_top16", "P1.4 loss on the 16 most frequent tokens"),
                           ("loss_freq_gt4k", "P1.4 loss on rare tokens (corpus rank>4k)"),
                           ("p_eot_at_end", "P1.6 P(EOT) at the true story end"),
                           ("p_eot_midstory_max", "P1.6 max mid-story P(EOT) (false-stop pressure)")):
            fmt_table(compare(cps, key, args.ref, n_boot=args.n_boot), title, lines)

    icl = load("icl")
    if icl:
        fmt_table(compare(icl, "icl_score", args.ref, n_boot=args.n_boot),
                  "P1.5 in-context-use score (loss@32-64 − loss@256-512; higher = uses context more)",
                  lines)

    # ---- P1.1
    pert = load("perturbation")
    if pert:
        fmt_table(compare(pert, "ctx_sens", args.ref, n_boot=args.n_boot),
                  "P1.1 context sensitivity (TV after one early-token flip)", lines)
        rates = sorted({k for r in pert for k in r if k.startswith("tv_rate_")})
        for k in rates:
            fmt_table(compare(pert, k, args.ref, n_boot=args.n_boot),
                      f"P1.1 corruption robustness — {k.replace('tv_rate_','rate ')} "
                      f"(lower = more robust)", lines)

    # ---- P1.3
    lens = load("lens_depth")
    if lens:
        depth_curves(lens, lines)

    # ---- P2.5
    agree = load("agreement")
    if agree:
        fmt_table(compare(agree, "correct", args.ref, n_boot=args.n_boot),
                  "P2.5 subject-verb agreement — overall accuracy", lines)
        fmt_table(compare(agree, "margin", args.ref, n_boot=args.n_boot),
                  "P2.5 agreement margin logP(target)−logP(foil)", lines)
        cells = sorted({r["cell"] for r in agree})
        for c in cells:
            fmt_table(compare(agree, "correct", args.ref,
                              filt=lambda r, c=c: r["cell"] == c, n_boot=args.n_boot),
                      f"P2.5 agreement accuracy — cell `{c}`", lines)

    rd = load("resolution_depth")
    if rd:
        for ln in sorted({r["lens"] for r in rd}):
            fmt_table(compare(rd, "resolution_depth", args.ref,
                              filt=lambda r, ln=ln: r["lens"] == ln, n_boot=args.n_boot),
                      f"E1 resolution depth (lens={ln}, lower = resolves earlier)", lines)
            fmt_table(compare(rd, "resolved", args.ref,
                              filt=lambda r, ln=ln: r["lens"] == ln, n_boot=args.n_boot),
                      f"E1 fraction of items ever resolved (lens={ln})", lines)

    text = "\n".join(lines)
    print(text)
    out = args.md or os.path.join(d, "report.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
