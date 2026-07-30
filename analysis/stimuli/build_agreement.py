"""
Generate the subject-verb agreement stimulus set (plan P2.5).

v1 had 20 items per distractor level, giving a measurement resolution of 0.05 and
95% CIs ~0.31-0.40 wide — the entire observed 0.65-0.85 spread across arms was
inside the noise. Target here is >=1000 items per cell, which puts a paired
McNemar comparison in the "detects a 3-5 point difference" regime.

Two design rules, both about not fooling ourselves:

1. Lexical items are FILTERED AGAINST THE CORPUS, not just typed out. A candidate
   noun/verb form is kept only if it is a single GPT-2 token with a leading space
   AND occurs at least --min_count times in the TinyStories val bin. Scoring a
   minimal pair whose target the model has essentially never seen measures
   vocabulary coverage, not agreement.
2. Target and foil are both single tokens, so the contrast is a single next-token
   decision. That keeps it leak-safe (nothing after the decision point is fed in)
   and avoids the length-normalisation ambiguity of full-sentence scoring.

Attractors take either the OPPOSITE number to the head subject (the hard
agreement-attractor case, Linzen et al. 2016) or the SAME number (control). The
match/mismatch contrast is what isolates attractor interference from mere distance.

Usage:
    python -m analysis.stimuli.build_agreement --out analysis/stimuli/agreement.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# TinyStories-register candidates. Deliberately over-inclusive: the corpus filter
# below decides what actually survives, and the manifest records what was dropped.
NOUN_CANDIDATES = [
    ("cat", "cats"), ("dog", "dogs"), ("boy", "boys"), ("girl", "girls"),
    ("bird", "birds"), ("mouse", "mice"), ("bear", "bears"), ("duck", "ducks"),
    ("frog", "frogs"), ("fish", "fish"), ("bunny", "bunnies"), ("puppy", "puppies"),
    ("kitten", "kittens"), ("friend", "friends"), ("child", "children"),
    ("kid", "kids"), ("baby", "babies"), ("man", "men"), ("woman", "women"),
    ("toy", "toys"), ("ball", "balls"), ("box", "boxes"), ("tree", "trees"),
    ("flower", "flowers"), ("house", "houses"), ("car", "cars"), ("book", "books"),
    ("bed", "beds"), ("cake", "cakes"), ("apple", "apples"), ("star", "stars"),
    ("cloud", "clouds"), ("rock", "rocks"), ("stick", "sticks"), ("hat", "hats"),
    ("shoe", "shoes"), ("door", "doors"), ("window", "windows"), ("bag", "bags"),
    ("cup", "cups"), ("hill", "hills"), ("lake", "lakes"), ("road", "roads"),
    ("park", "parks"), ("garden", "gardens"), ("room", "rooms"), ("chair", "chairs"),
    ("table", "tables"), ("plant", "plants"), ("leaf", "leaves"), ("bug", "bugs"),
    ("horse", "horses"), ("sheep", "sheep"), ("cow", "cows"), ("pig", "pigs"),
]

# (singular verb form, plural verb form)
VERB_CANDIDATES = [
    ("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"),
    ("likes", "like"), ("wants", "want"), ("plays", "play"), ("runs", "run"),
    ("sees", "see"), ("goes", "go"), ("says", "say"), ("looks", "look"),
    ("jumps", "jump"), ("walks", "walk"), ("sits", "sit"), ("helps", "help"),
    ("finds", "find"), ("knows", "know"), ("feels", "feel"), ("makes", "make"),
]

PREPS = ["near", "by", "beside", "behind", "under"]


def single_token_id(enc, word: str) -> int | None:
    """Token id of ' word' if it is exactly one GPT-2 token, else None."""
    ids = enc.encode_ordinary(" " + word)
    return ids[0] if len(ids) == 1 else None


def build(out_path: str, val_bin: str = "data/tinystories_val.bin",
          min_count: int = 50, max_per_cell: int = 4000, seed: int = 0) -> dict:
    import tiktoken
    from analysis.corpus import load_bin

    enc = tiktoken.get_encoding("gpt2")
    counts = np.bincount(load_bin(val_bin).astype(np.int64), minlength=50304)

    def keep(word: str):
        tid = single_token_id(enc, word)
        if tid is None:
            return None, "multi_token"
        if counts[tid] < min_count:
            return None, f"rare({int(counts[tid])})"
        return tid, None

    nouns, dropped_nouns = [], {}
    for sg, pl in NOUN_CANDIDATES:
        if sg == pl:
            dropped_nouns[sg] = "no_number_contrast"   # fish, sheep
            continue
        t_sg, r_sg = keep(sg)
        t_pl, r_pl = keep(pl)
        if t_sg is None or t_pl is None:
            dropped_nouns[f"{sg}/{pl}"] = r_sg or r_pl
            continue
        nouns.append((sg, pl, t_sg, t_pl))

    verbs, dropped_verbs = [], {}
    for vs, vp in VERB_CANDIDATES:
        t_s, r_s = keep(vs)
        t_p, r_p = keep(vp)
        if t_s is None or t_p is None:
            dropped_verbs[f"{vs}/{vp}"] = r_s or r_p
            continue
        verbs.append((vs, vp, t_s, t_p))

    if not nouns or not verbs:
        raise SystemExit("corpus filter removed everything — check --val_bin/--min_count")

    rng = np.random.default_rng(seed)
    rows = []
    # attractor nouns are drawn from the same filtered pool, excluding the subject
    for n_attr, attr_match in itertools.product((0, 1, 2, 3), (True, False)):
        if n_attr == 0 and not attr_match:
            continue                      # no attractor -> match flag meaningless
        cell = []
        for (sg, pl, _t_sg, _t_pl) in nouns:
            for number in ("sing", "plur"):
                subj = sg if number == "sing" else pl
                for (vs, vp, t_s, t_p) in verbs:
                    target, foil = (vs, vp) if number == "sing" else (vp, vs)
                    tgt_id, foil_id = (t_s, t_p) if number == "sing" else (t_p, t_s)
                    # attractor number: opposite (hard) or same (control)
                    attr_plural = (number == "sing") if not attr_match else (number == "plur")
                    parts = [f"The {subj}"]
                    used = []
                    for k in range(n_attr):
                        cand = [n for n in nouns if n[0] != sg]
                        pick = cand[int(rng.integers(0, len(cand)))]
                        while pick[0] in used and len(used) < len(cand):
                            pick = cand[int(rng.integers(0, len(cand)))]
                        used.append(pick[0])
                        a_word = pick[1] if attr_plural else pick[0]
                        parts.append(f"{PREPS[k % len(PREPS)]} the {a_word}")
                    prompt = " ".join(parts)
                    cell.append({
                        "paradigm": "sv_agreement",
                        "prompt": prompt, "target": " " + target, "foil": " " + foil,
                        "target_id": int(tgt_id), "foil_id": int(foil_id),
                        "n_attractors": n_attr,
                        "attractor_match": bool(attr_match),
                        "cell": f"attr{n_attr}_{'match' if attr_match else 'mismatch'}",
                        "subject": subj, "number": number, "verb_pair": f"{vs}/{vp}",
                    })
        if len(cell) > max_per_cell:
            idx = rng.permutation(len(cell))[:max_per_cell]
            cell = [cell[i] for i in sorted(idx)]
        rows.extend(cell)

    for i, r in enumerate(rows):
        r["item_id"] = f"agr{i:06d}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]
    cells: dict[str, int] = {}
    for r in rows:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1
    manifest = {
        "path": out_path, "sha256_16": digest, "n_items": len(rows),
        "cells": cells, "n_nouns": len(nouns), "n_verbs": len(verbs),
        "min_count": min_count, "max_per_cell": max_per_cell, "seed": seed,
        "val_bin": val_bin,
        "dropped_nouns": dropped_nouns, "dropped_verbs": dropped_verbs,
        "nouns_kept": [n[0] for n in nouns], "verbs_kept": [v[0] for v in verbs],
    }
    with open(out_path.replace(".jsonl", "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/stimuli/agreement.jsonl")
    ap.add_argument("--val_bin", default="data/tinystories_val.bin")
    ap.add_argument("--min_count", type=int, default=50,
                    help="minimum corpus occurrences for a lexical item to be used")
    ap.add_argument("--max_per_cell", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = build(args.out, args.val_bin, args.min_count, args.max_per_cell, args.seed)
    print(f"wrote {m['n_items']} items -> {m['path']}  (sha {m['sha256_16']})")
    print(f"  {m['n_nouns']} noun pairs x {m['n_verbs']} verb pairs kept")
    print("  per-cell counts:")
    for c, n in sorted(m["cells"].items()):
        flag = "" if n >= 1000 else "   << under 1000, power target missed"
        print(f"    {c:22s} {n:6d}{flag}")
    if m["dropped_nouns"]:
        print(f"  dropped nouns: {m['dropped_nouns']}")
    if m["dropped_verbs"]:
        print(f"  dropped verbs: {m['dropped_verbs']}")


if __name__ == "__main__":
    main()
