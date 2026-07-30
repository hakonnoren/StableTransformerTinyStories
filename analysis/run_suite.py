"""
Driver for the v2 behavior suite (plan steps 1-5).

Writes per-item JSONL + a provenance manifest to --out_dir; run analysis/report.py
over that directory to get tables with intervals and paired tests. Nothing here
prints a mean without the report pass, on purpose.

Typical run (5 arms, ~2000 stories, CPU):
    python -m analysis.run_suite \
        --ckpt_dir fetched/revparityreg_24979051 \
        --out_dir results/revparityreg_24979051 \
        --n_stories 2000 --device cpu

Cheap smoke run:
    python -m analysis.run_suite --n_stories 40 --lens_stories 20 \
        --agreement_items 400 --depth_items 100 --tuned_steps 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import corpus, lenses
from analysis.loader import LensPolicy, load_arms
from analysis.probes import agreement as agr
from analysis.probes import scale
from analysis.records import RecordWriter, RunManifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="fetched/revparityreg_24957997")
    ap.add_argument("--out_dir", default="results/behavior_v2")
    ap.add_argument("--val_bin", default="data/tinystories_val.bin")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these arm names")
    # corpus probes
    ap.add_argument("--n_stories", type=int, default=2000, help="P1.4-P1.6 stories")
    ap.add_argument("--lens_stories", type=int, default=200, help="P1.3 stories (12x cost)")
    ap.add_argument("--pert_stories", type=int, default=300, help="P1.1 stories")
    ap.add_argument("--leak_trials", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    # tuned lens
    ap.add_argument("--tuned_steps", type=int, default=300, help="0 disables the tuned lens")
    ap.add_argument("--tuned_pos", type=int, default=128)
    ap.add_argument("--lens_policies", nargs="*", default=["sum", "x", "z"])
    ap.add_argument("--lens_max_pos", type=int, default=64,
                    help="token positions scored per story in P1.3 (cost is one "
                         "50304-way projection per layer per lens per policy)")
    # agreement
    ap.add_argument("--stimuli", default="analysis/stimuli/agreement.jsonl")
    ap.add_argument("--agreement_items", type=int, default=0, help="0 = all")
    ap.add_argument("--depth_items", type=int, default=800,
                    help="subsample for per-layer resolution depth")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="probe names to skip: perturb leak lens corpus agreement depth")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    print(f"Loading arms from {args.ckpt_dir} onto {args.device} ...")
    arms = load_arms(args.ckpt_dir, device=args.device, only=args.only)

    manifest = RunManifest(
        torch=torch.__version__, numpy=np.__version__, device=args.device, seed=args.seed,
        ckpt_dir=args.ckpt_dir, config=vars(args))
    for a in arms:
        manifest.add_arm(a.name, a.path, a.meta)
        print(f"  {a.name:32s} arch={a.meta['arch']:11s} regime={str(a.meta['regime']):14s} "
              f"embed={str(a.meta['embed']):7s} attn={a.meta['attn_impl']:6s} "
              f"step={a.meta['step']} best_val={a.meta['best_val']:.4f}")

    warn = manifest.warnings()
    if warn:
        print("\n!! comparability warnings (these bound what the numbers can support):")
        for w in warn:
            print(f"   - {w}")

    print(f"\nLoading corpus from {args.val_bin} ...")
    stories = corpus.load_stories(args.val_bin, n=args.n_stories, seed=args.seed)
    print("  " + json.dumps(corpus.length_stats(stories)))
    lens_stories = stories[:args.lens_stories]
    pert_stories = stories[:args.pert_stories]
    # tuned-lens training text must be disjoint from anything it is evaluated on
    train_stories = corpus.load_stories(args.val_bin, n=args.n_stories + 600,
                                        seed=args.seed)[args.n_stories:]
    print(f"  tuned-lens train pool: {len(train_stories)} held-out stories")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)

    # ---------------------------------------------------------------- P1.2 leak
    if "leak" not in args.skip:
        print("\n== P1.2 causality (test) ==")
        with RecordWriter(args.out_dir, "leak", manifest) as w:
            for a in arms:
                r = scale.probe_leak(a, stories, args.device, args.leak_trials, seed=args.seed)
                w.write(**r)
                flag = "PASS" if r["passed"] else "*** LEAK ***"
                print(f"  {a.name:32s} max|Δ|={r['max_abs_delta']:.3e}  {flag}")

    # ---------------------------------------------------------------- P1.1 perturbation
    if "perturb" not in args.skip:
        print(f"\n== P1.1 perturbation / corruption ({len(pert_stories)} stories) ==")
        with RecordWriter(args.out_dir, "perturbation") as w:
            for a in arms:
                w.write_many(scale.probe_perturbation(
                    a, pert_stories, args.device, seed=args.seed,
                    batch_size=args.batch_size))
                print(f"  {a.name}: done")

    # ---------------------------------------------------------------- P1.4-1.6 corpus
    if "corpus" not in args.skip:
        print(f"\n== P1.4-P1.6 stratified corpus loss ({len(stories)} stories) ==")
        # frequency ranks from the FULL val split, not the evaluation sample
        strata = scale.Strata(corpus.load_bin(args.val_bin), enc)
        with RecordWriter(args.out_dir, "corpus") as w, \
             RecordWriter(args.out_dir, "icl") as w2:
            for a in arms:
                recs = scale.score_corpus(
                    a, stories, strata, args.device, batch_size=args.batch_size,
                    dump_npz=os.path.join(args.out_dir, f"tokens_{a.name}.npz"))
                w.write_many(recs)
                w2.write_many(scale.in_context_score(recs))
                print(f"  {a.name}: {len(recs)} stories scored")

    # ---------------------------------------------------------------- tuned lens
    tuned = {}
    if args.tuned_steps > 0 and ("lens" not in args.skip or "depth" not in args.skip):
        print(f"\n== Tuned lens ({args.tuned_steps} steps/arm) ==")
        for a in arms:
            print(f"  {a.name}:")
            lens = lenses.train_tuned_lens(
                a.model, train_stories, device=args.device, steps=args.tuned_steps,
                n_pos=args.tuned_pos, seed=args.seed)
            path = os.path.join(args.out_dir, f"tuned_lens_{a.name}.pt")
            lenses.save_lens(lens, path)
            tuned[a.name] = lens
            print(f"    max|A|={lens.identity_check():.4f} -> {path}")

    # ---------------------------------------------------------------- P1.3 depth curves
    if "lens" not in args.skip:
        pols = [p for p in args.lens_policies if p in LensPolicy.ALL]
        print(f"\n== P1.3 depth curves ({len(lens_stories)} stories, policies={pols}) ==")
        with RecordWriter(args.out_dir, "lens_depth") as w:
            for a in arms:
                w.write_many(scale.probe_lens_depth(
                    a, lens_stories, args.device, batch_size=max(2, args.batch_size // 2),
                    tuned=tuned.get(a.name), policies=pols,
                    max_pos=args.lens_max_pos, seed=args.seed))
                print(f"  {a.name}: done")

    # ---------------------------------------------------------------- P2.5 agreement
    if "agreement" not in args.skip:
        items = agr.load_stimuli(args.stimuli)
        if args.agreement_items:
            rng = np.random.default_rng(args.seed)
            idx = sorted(rng.permutation(len(items))[:args.agreement_items])
            items = [items[i] for i in idx]
        prompts = agr.encode_prompts(items, enc)
        print(f"\n== P2.5 agreement ({len(items)} items) ==")
        with RecordWriter(args.out_dir, "agreement") as w:
            for a in arms:
                w.write_many(agr.score_items(a, items, prompts, args.device))
                print(f"  {a.name}: done")

        if "depth" not in args.skip and args.depth_items:
            rng = np.random.default_rng(args.seed + 1)
            idx = sorted(rng.permutation(len(items))[:min(args.depth_items, len(items))])
            ditems = [items[i] for i in idx]
            dprompts = [prompts[i] for i in idx]
            print(f"\n== E1 resolution depth ({len(ditems)} items) ==")
            with RecordWriter(args.out_dir, "resolution_depth") as w:
                for a in arms:
                    for lname, ln in (("raw", None), ("tuned", tuned.get(a.name))):
                        if lname == "tuned" and ln is None:
                            continue
                        w.write_many(agr.resolution_depth(
                            a, ditems, dprompts, args.device, tuned=ln))
                    print(f"  {a.name}: done")

    print(f"\nDone. Records in {args.out_dir}/")
    print(f"Aggregate with:  python -m analysis.report --dir {args.out_dir}")


if __name__ == "__main__":
    main()
