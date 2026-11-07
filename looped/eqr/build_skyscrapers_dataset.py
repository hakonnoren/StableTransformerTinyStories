"""Build Skyscrapers-7 datasets in EqR's format (looped/eqr/skyscrapers.py has the puzzle definition).

    python looped/eqr/build_skyscrapers_dataset.py --out-root $EQR_DIR/data --n-train 1000,10000 \
        --n-test 10000 --workers 8

Writes, for each training size T in --n-train,
    <out-root>/skyscrapers7-<T/1000>k-aug-8/{train,test}/all__{inputs,labels,puzzle_identifiers,puzzle_indices,
                                                       group_indices}.npy + dataset.json
    <out-root>/skyscrapers7-<T/1000>k-aug-8/{identifiers.json, puzzles.csv}
Every size shares one test set; the training sets are nested (the 1k set is the first 1000 of the 10k set).
Training puzzles come with their 8 rotations / reflections as one group (EqR samples one per group per
epoch, as for Sudoku's digit/band augmentations); test puzzles are not augmented. Puzzles are unique and
minimal (skyscrapers.make_puzzle), de-duplicated up to symmetry across both splits, and depend only on
--seed and their index, so the output does not depend on --workers.
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
import skyscrapers as sk  # noqa: E402


def _one(args):
    seed, i = args
    rng = np.random.default_rng([seed, i])
    puzzle, solution, rating = sk.make_puzzle(rng)
    return i, puzzle, solution, rating


def _write_split(save_dir, boards, sols, augment):
    os.makedirs(save_dir, exist_ok=True)
    inputs, labels, puzzle_indices, group_indices = [], [], [0], [0]
    for p, s in zip(boards, sols):
        lab = s.astype(np.int64) + 1
        lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1] = 0, 0, 0, 0          # border and corners: ignore
        for t in (range(8) if augment else [0]):
            inputs.append(np.ascontiguousarray(sk.dihedral(p.astype(np.int64) + 1, t)).reshape(-1))
            labels.append(np.ascontiguousarray(sk.dihedral(lab, t)).reshape(-1))
            puzzle_indices.append(len(inputs))
        group_indices.append(len(inputs))
    arrays = {"inputs": np.array(inputs, dtype=np.uint8), "labels": np.array(labels, dtype=np.uint8),
              "puzzle_identifiers": np.zeros(len(inputs), dtype=np.int32),
              "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
              "group_indices": np.array(group_indices, dtype=np.int32)}
    for k, v in arrays.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": sk.N + 2,   # PAD, blank, 1..7
            "seq_len": sk.B * sk.B, "num_puzzle_identifiers": 1, "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": 1, "sets": ["all"], "total_samples": len(inputs)}
    json.dump(meta, open(os.path.join(save_dir, "dataset.json"), "w"))
    return len(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-train", default="1000,10000", help="comma list of training-set sizes (nested)")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--min-rating", type=int, default=0, help="keep puzzles needing at least this many search branches")
    a = ap.parse_args()
    sizes = sorted(int(s) for s in a.n_train.split(","))
    need = a.n_test + sizes[-1]

    t0 = time.time()
    kept, seen, i0 = [], set(), 0
    with Pool(a.workers) as pool:
        while len(kept) < need:
            batch = range(i0, i0 + int(1.05 * (need - len(kept))) + 64)
            i0 = batch.stop
            for i, p, s, r in sorted(pool.imap_unordered(_one, [(a.seed, i) for i in batch], chunksize=16)):
                key = sk.canonical(p)
                if r < a.min_rating or key in seen or len(kept) >= need:
                    continue
                seen.add(key)
                kept.append((i, p, s, r))
            print(f"[gen] {len(kept)}/{need} puzzles from {i0} draws ({time.time() - t0:.0f}s)", flush=True)

    test, train = kept[:a.n_test], kept[a.n_test:]
    ratings = np.array([r for *_, r in kept])
    print(f"[gen] rating (search branches) median {np.median(ratings):.0f}, 90% {np.quantile(ratings, 0.9):.0f}, "
          f"max {ratings.max()}; share solved by propagation alone {np.mean(ratings == 0):.2f}")
    for size in sizes:
        out = os.path.join(a.out_root, f"skyscrapers7-{size // 1000}k-aug-8" if size % 1000 == 0 else f"skyscrapers7-{size}-aug-8")
        n_tr = _write_split(os.path.join(out, "train"), [p for _, p, _, _ in train[:size]], [s for _, _, s, _ in train[:size]], True)
        n_te = _write_split(os.path.join(out, "test"), [p for _, p, _, _ in test], [s for _, _, s, _ in test], False)
        json.dump(["<blank>"], open(os.path.join(out, "identifiers.json"), "w"))
        with open(os.path.join(out, "puzzles.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "draw", "rating", "clues", "givens", "puzzle", "solution"])
            for split, rows in (("train", train[:size]), ("test", test)):
                for i, p, s, r in rows:
                    g, cl = sk.from_board(p)
                    w.writerow([split, i, r, sum(int((v > 0).sum()) for v in cl.values()), int((g > 0).sum()),
                                "".join(map(str, p.reshape(-1))), "".join(map(str, s[1:-1, 1:-1].reshape(-1)))])
        print(f"[write] {out}: train {size} puzzles x 8 = {n_tr} examples, test {n_te}")
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
