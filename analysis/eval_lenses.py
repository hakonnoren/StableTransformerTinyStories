"""
Evaluate fitted tuned lenses: per-layer raw vs tuned KL, plus translator size.

Separate from run_suite so a finished set of lenses can be characterised without
re-running the whole suite. Writes lens_quality.jsonl, which analysis/plots.py
turns into the basis-change figure.

    python -m analysis.eval_lenses --dir results/revparityreg_24957997 \
        --ckpt_dir fetched/revparityreg_24957997
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import corpus
from analysis.lenses import eval_lens_kl, load_lens, translator_magnitudes
from analysis.loader import load_arms
from analysis.records import RecordWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results dir holding tuned_lens_*.pt")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--val_bin", default="data/tinystories_val.bin")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_stories", type=int, default=24)
    ap.add_argument("--n_batches", type=int, default=6)
    ap.add_argument("--n_pos", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_offset", type=int, default=2600,
                    help="skip this many stories so the eval text is disjoint from "
                         "both the suite's evaluation set and the lens training pool")
    args = ap.parse_args()

    lens_paths = {os.path.basename(p).replace("tuned_lens_", "").replace(".pt", ""): p
                  for p in sorted(glob.glob(os.path.join(args.dir, "tuned_lens_*.pt")))}
    if not lens_paths:
        raise SystemExit(f"no tuned_lens_*.pt in {args.dir}")
    print(f"Found {len(lens_paths)} fitted lenses")

    stories = corpus.load_stories(args.val_bin, n=args.eval_offset + args.n_stories,
                                  seed=args.seed)[args.eval_offset:]
    print(f"  eval text: {len(stories)} held-out stories (offset {args.eval_offset})")

    arms = load_arms(args.ckpt_dir, device=args.device, only=list(lens_paths))
    with RecordWriter(args.dir, "lens_quality") as w:
        for a in arms:
            lens = load_lens(lens_paths[a.name], args.device)
            rows = eval_lens_kl(a.model, lens, stories, device=args.device,
                                n_batches=args.n_batches, n_pos=args.n_pos,
                                seed=args.seed)
            for row in rows:
                w.write(arm=a.name, **row)
            for row in translator_magnitudes(lens):
                w.write(arm=a.name, lens="translator", **row)
            mean_raw = sum(r["kl"] for r in rows if r["lens"] == "raw") / max(
                1, sum(1 for r in rows if r["lens"] == "raw"))
            mean_tuned = sum(r["kl"] for r in rows if r["lens"] == "tuned") / max(
                1, sum(1 for r in rows if r["lens"] == "tuned"))
            print(f"  {a.name:32s} raw={mean_raw:.3f} tuned={mean_tuned:.3f} "
                  f"({100*(1-mean_tuned/mean_raw):.0f}% removed by an affine map)")
    print(f"\nWrote {os.path.join(args.dir, 'lens_quality.jsonl')}")


if __name__ == "__main__":
    main()
