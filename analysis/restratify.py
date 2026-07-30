"""
Recompute stratified loss fields from the per-token .npz dumps — no model needed.

This is the reason score_corpus() saves raw per-token log-probabilities: strata are
a post-hoc choice, and getting them wrong should cost a CPU minute, not a full
re-run of every arm.

Used when the frequency-band edges changed: the original run ranked tokens within
the 2000-story evaluation sample, which capped the top band at the sample's
distinct-token count (~6.4k) and left it permanently empty. Ranks now come from the
whole val split.

Only the loss_freq_* / n_freq_* fields are rewritten. Position buckets,
first-vs-repeat, word-initial and the closure probabilities do not depend on
frequency and are carried through untouched.

    python -m analysis.restratify --dir results/revparityreg_24957997
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import corpus
from .probes.scale import FREQ_BANDS, Strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--val_bin", default="data/tinystories_val.bin")
    ap.add_argument("--records", default="corpus.jsonl")
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    print("Building strata from the full val split ...")
    strata = Strata(corpus.load_bin(args.val_bin), enc)
    present = np.unique(strata.rank[np.unique(corpus.load_bin(args.val_bin))])
    print(f"  observed token ranks span 0..{int(present.max())}")

    rec_path = os.path.join(args.dir, args.records)
    with open(rec_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    by_arm_story = {(r["arm"], r["story_idx"]): r for r in recs}
    print(f"  {len(recs)} existing records in {rec_path}")

    arms = sorted({r["arm"] for r in recs})
    old_keys = [k for r in recs[:1] for k in r
                if k.startswith("loss_freq_") or k.startswith("n_freq_")]

    updated = 0
    for arm in arms:
        npz = os.path.join(args.dir, f"tokens_{arm}.npz")
        if not os.path.exists(npz):
            print(f"  !! no {npz}; skipping {arm}")
            continue
        d = np.load(npz)
        lp, tok, story = d["logprob"], d["token"].astype(np.int64), d["story"]
        band = strata.freq_band(tok)
        loss = -lp

        # group by story without assuming the dump is contiguous or sorted
        order = np.argsort(story, kind="stable")
        story_s, loss_s, band_s = story[order], loss[order], band[order]
        bounds = np.flatnonzero(np.diff(story_s)) + 1
        for seg_start, seg_end in zip(np.r_[0, bounds], np.r_[bounds, story_s.size]):
            sidx = int(story_s[seg_start])
            rec = by_arm_story.get((arm, sidx))
            if rec is None:
                continue
            for k in old_keys:
                rec.pop(k, None)
            b = band_s[seg_start:seg_end]
            l = loss_s[seg_start:seg_end]
            for i, (name, _, _) in enumerate(FREQ_BANDS):
                m = b == i
                if m.any():
                    rec[f"loss_freq_{name}"] = float(l[m].mean())
                    rec[f"n_freq_{name}"] = int(m.sum())
            updated += 1
        print(f"  {arm}: restratified {int(np.unique(story).size)} stories")

    bak = rec_path + ".bak"
    if not os.path.exists(bak):
        os.rename(rec_path, bak)
        print(f"  original saved as {bak}")
    with open(rec_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"  rewrote {updated} records -> {rec_path}")

    # report band coverage so an empty band can never pass unnoticed again
    print("\n  band coverage (records containing each band):")
    for name, _, _ in FREQ_BANDS:
        n = sum(1 for r in recs if f"loss_freq_{name}" in r)
        flag = "   << EMPTY" if n == 0 else ""
        print(f"    loss_freq_{name:8s} {n:6d}/{len(recs)}{flag}")


if __name__ == "__main__":
    main()
