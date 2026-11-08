"""Build RNA inverse-folding datasets in EqR's format (looped/eqr/rna/rna_design.py has the task definition).

    python looped/eqr/rna/build_rna_dataset.py --out-root $EQR_DIR/data --n-train 10000,100000 --n-test 5000 \
        --len-max 128 --workers 32 [--eterna-tsv eterna100_puzzles.tsv]

Writes, for each training size T in --n-train (nested: the smaller set is a prefix of the larger),
    <out-root>/rna-L<len-max>-<T>/{train,test}/all__{inputs,labels,puzzle_identifiers,puzzle_indices,
                                               group_indices}.npy + dataset.json
    <out-root>/rna-L<len-max>-<T>/{identifiers.json, examples.csv}
and with --eterna-tsv (data/eterna100_puzzles.tsv of github.com/eternagame/eterna100-benchmarking)
    <out-root>/rna-eterna100v2-L<len-max>/{train,test}/...   Eterna100-V2 targets of length <= len-max
                                                            (train = test; only the test split is used)
Every size shares one test set. Targets are unique across all splits and exclude the Eterna100 targets.
One example per puzzle and one puzzle per group (no augmentation: the energy model has no symmetry).
Each example depends only on --seed and its index, so the output does not depend on --workers.

Tokens: inputs are the target structure ('.', '(', ')' = 1, 2, 3; PAD 0 past its length), labels the
sequence (A, C, G, U = 4..7; 0 = ignored past its length). The label is one solution among many: score
predictions by refolding (rna_design.verify), not by agreement with the label.
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rna_design as rd  # noqa: E402

FEATURES = ("pairs", "helices", "min_helix", "short_helices", "hairpins", "interior_loops", "bulges", "multiloops")


def _one(args):
    seed, i, len_min, len_max, random_frac, walk_steps = args
    rng = np.random.default_rng([seed, i])
    length = int(rng.integers(len_min, len_max + 1))
    mode = "random" if rng.random() < random_frac else "design"
    ex = rd.make_example(rng, length, mode, walk_steps)
    ex["draw"] = i
    return ex


def _write_split(save_dir, targets, sequences, seq_len):
    os.makedirs(save_dir, exist_ok=True)
    n = len(targets)
    inputs = np.stack([rd.encode_structure(t, seq_len) for t in targets]) if n else np.zeros((0, seq_len), np.uint8)
    labels = np.stack([rd.encode_sequence(s, seq_len) if s else np.zeros(seq_len, np.uint8) for s in sequences]) \
        if n else np.zeros((0, seq_len), np.uint8)
    arrays = {"inputs": inputs.astype(np.uint8), "labels": labels.astype(np.uint8),
              "puzzle_identifiers": np.zeros(n, dtype=np.int32),
              "puzzle_indices": np.arange(n + 1, dtype=np.int32),
              "group_indices": np.arange(n + 1, dtype=np.int32)}
    for k, v in arrays.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    meta = {"pad_id": rd.PAD, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": rd.VOCAB_SIZE,
            "seq_len": seq_len, "num_puzzle_identifiers": 1, "total_groups": n, "mean_puzzle_examples": 1,
            "sets": ["all"], "total_samples": n}
    json.dump(meta, open(os.path.join(save_dir, "dataset.json"), "w"))
    return n


def read_eterna(path, len_max):
    """Eterna100-V2 puzzles of length <= len_max: (number, name, target, solution or '' if it does not fold)."""
    out = []
    for r in csv.DictReader(open(path), delimiter="\t"):
        t = r["Secondary Structure V2"].strip()
        if not t or len(t) > len_max:
            continue
        sol = r["Sample Solution (V2/Vienna2)"].strip().upper().replace("T", "U")
        if sol and (len(sol) != len(t) or (rd.RNA is not None and rd.fold(sol)[0] != t)):
            sol = ""
        out.append((int(r["Puzzle #"]), r["Puzzle Name"].strip(), t, sol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-train", default="10000,100000", help="comma list of training-set sizes (nested)")
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--len-min", type=int, default=16)
    ap.add_argument("--len-max", type=int, default=128, help="also the sequence length of the dataset")
    ap.add_argument("--walk-steps", type=int, default=100, help="adaptive-walk folds per designed example")
    ap.add_argument("--random-frac", type=float, default=0.2, help="share of folds of random sequences")
    ap.add_argument("--eterna-tsv", default=None, help="eterna100_puzzles.tsv: also write the Eterna100-V2 test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    rd._need_rna()
    sizes = sorted(int(s) for s in a.n_train.split(","))
    need = a.n_test + sizes[-1]
    os.makedirs(a.out_root, exist_ok=True)                 # fail now, not after the generation
    if not os.access(a.out_root, os.W_OK):
        raise SystemExit(f"cannot write to {a.out_root}")
    print(f"[out] {os.path.abspath(a.out_root)}  (ViennaRNA {rd.RNA.__version__})", flush=True)

    eterna = read_eterna(a.eterna_tsv, a.len_max) if a.eterna_tsv else []
    seen = {t for _, _, t, _ in eterna}
    t0 = time.time()
    kept, i0 = [], 0
    with Pool(a.workers) as pool:
        while len(kept) < need:
            batch = range(i0, i0 + int(1.02 * (need - len(kept))) + 64)
            i0 = batch.stop
            jobs = [(a.seed, i, a.len_min, a.len_max, a.random_frac, a.walk_steps) for i in batch]
            for ex in sorted(pool.imap_unordered(_one, jobs, chunksize=8), key=lambda e: e["draw"]):
                if ex["target"] in seen or len(kept) >= need:
                    continue
                seen.add(ex["target"])
                kept.append(ex)
            print(f"[gen] {len(kept)}/{need} examples from {i0} draws ({time.time() - t0:.0f}s)", flush=True)

    # sanity: every label folds into its target (by construction)
    for ex in kept[:: max(1, len(kept) // 200)]:
        assert rd.fold(ex["sequence"])[0] == ex["target"], ex
    des = [e for e in kept if e["mode"] == "design"]
    lens = np.array([len(e["target"]) for e in kept])
    print(f"[gen] length median {np.median(lens):.0f} (min {lens.min()}, max {lens.max()}); designed {len(des)}, "
          f"of which the fold equals the sampled structure: {np.mean([e['matched'] for e in des]):.2f}")

    test, train = kept[:a.n_test], kept[a.n_test:]
    for size in sizes:
        out = os.path.join(a.out_root, f"rna-L{a.len_max}-{size // 1000}k" if size % 1000 == 0
                           else f"rna-L{a.len_max}-{size}")
        n_tr = _write_split(os.path.join(out, "train"), [e["target"] for e in train[:size]],
                            [e["sequence"] for e in train[:size]], a.len_max)
        n_te = _write_split(os.path.join(out, "test"), [e["target"] for e in test],
                            [e["sequence"] for e in test], a.len_max)
        json.dump(["<blank>"], open(os.path.join(out, "identifiers.json"), "w"))
        with open(os.path.join(out, "examples.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "draw", "length", "mode", "matched", "walk_folds", "mfe", *FEATURES,
                        "target", "sequence"])
            for split, rows in (("train", train[:size]), ("test", test)):
                for e in rows:
                    ft = rd.structure_features(e["target"])
                    w.writerow([split, e["draw"], len(e["target"]), e["mode"], e["matched"], e["walk_folds"],
                                f"{e['mfe']:.2f}", *[ft[k] for k in FEATURES], e["target"], e["sequence"]])
        print(f"[write] {out}: train {n_tr}, test {n_te}")

    if eterna:
        out = os.path.join(a.out_root, f"rna-eterna100v2-L{a.len_max}")
        for split in ("train", "test"):
            _write_split(os.path.join(out, split), [t for _, _, t, _ in eterna], [s for *_, s in eterna], a.len_max)
        json.dump(["<blank>"], open(os.path.join(out, "identifiers.json"), "w"))
        with open(os.path.join(out, "puzzles.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["puzzle", "name", "length", *FEATURES, "target", "sample_solution"])
            for num, name, t, s in eterna:
                ft = rd.structure_features(t)
                w.writerow([num, name, len(t), *[ft[k] for k in FEATURES], t, s])
        print(f"[write] {out}: {len(eterna)} Eterna100-V2 puzzles of length <= {a.len_max} "
              f"({sum(bool(s) for *_, s in eterna)} with a sample solution that folds)")


if __name__ == "__main__":
    main()
