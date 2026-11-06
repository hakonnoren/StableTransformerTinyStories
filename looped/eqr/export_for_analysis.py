"""Copy what the map analysis needs out of a sudoku_eqr.slurm run root, small enough to rsync.

    python looped/eqr/export_for_analysis.py --eqr-dir $EQR_DIR --run-root $R --out <dir> \
        [--n-test 4096] [--steps 5000,10000] [--arms eqr,kick_pos] [--no-released]

Writes
  <out>/ckpts/<run_name>__step<S>.pth   per run directory, its last checkpoint (so 20k, and 50k for
                                        extended runs), plus --steps if given; stripped to model + EMA
                                        weights + config (no optimiser, training carry or RNG state)
  <out>/ckpts/eqr-released__step50000.pth   EqR's released checkpoint (unless --no-released)
  <out>/data/<dataset>/test/            the first --n-test test puzzles, in EqR's .npy format
  <out>/data/<dataset>/train/dataset.json   training-set metadata only (vocab, seq_len, identifiers)

Run on the login node from this repo, with EqR's env (unpickling a checkpoint needs EqR's classes).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

KEEP = ("step", "total_steps", "config", "model", "ema", "wandb_run_id")


def _step_of(path):
    m = re.search(r"step_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _run_name(run_dir):
    cfg = os.path.join(run_dir, "all_config.yaml")
    if os.path.exists(cfg):
        m = re.search(r"^run_name:\s*(\S+)", open(cfg).read(), re.M)
        if m:
            return m.group(1)
    return os.path.basename(os.path.dirname(os.path.dirname(run_dir.rstrip("/"))))


def _strip(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: ck[k] for k in KEEP if k in ck}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(out, dst)
    return out


def _slice_test(src_dir, dst_dir, n):
    """First groups of the test split holding at most n examples (EqR's PuzzleDataset layout)."""
    os.makedirs(dst_dir, exist_ok=True)
    meta = json.load(open(os.path.join(src_dir, "dataset.json")))
    for s in meta.get("sets", ["all"]):
        gi = np.load(os.path.join(src_dir, f"{s}__group_indices.npy"))
        pi = np.load(os.path.join(src_dir, f"{s}__puzzle_indices.npy"))
        n_groups = int(np.searchsorted(pi[gi], n, side="right") - 1)       # groups whose examples fit
        n_groups = max(1, min(n_groups, len(gi) - 1))
        n_puz = int(gi[n_groups])
        n_ex = int(pi[n_puz])
        for name, arr in (("inputs", None), ("labels", None)):
            a = np.load(os.path.join(src_dir, f"{s}__{name}.npy"), mmap_mode="r")
            np.save(os.path.join(dst_dir, f"{s}__{name}.npy"), np.ascontiguousarray(a[:n_ex]))
        ids = np.load(os.path.join(src_dir, f"{s}__puzzle_identifiers.npy"), mmap_mode="r")
        np.save(os.path.join(dst_dir, f"{s}__puzzle_identifiers.npy"), np.ascontiguousarray(ids[:n_puz]))
        np.save(os.path.join(dst_dir, f"{s}__puzzle_indices.npy"), np.ascontiguousarray(pi[:n_puz + 1]))
        np.save(os.path.join(dst_dir, f"{s}__group_indices.npy"), np.ascontiguousarray(gi[:n_groups + 1]))
        print(f"[data] {s}: {n_groups} groups, {n_puz} puzzles, {n_ex} examples")
    meta["total_groups"] = n_groups
    if "total_samples" in meta:
        meta["total_samples"] = n_ex
    meta["mean_puzzle_examples"] = n_ex / max(n_puz, 1)
    json.dump(meta, open(os.path.join(dst_dir, "dataset.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqr-dir", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--steps", default="", help="extra checkpoint steps to export, e.g. 5000,10000")
    ap.add_argument("--arms", default="", help="only these run names' arms (default: all)")
    ap.add_argument("--no-released", action="store_true")
    a = ap.parse_args()
    eqr_dir = os.path.abspath(a.eqr_dir)
    sys.path.insert(0, eqr_dir)                       # EqR's classes, for unpickling the carry
    extra = {int(s) for s in a.steps.split(",") if s.strip()}
    arms = {s.strip() for s in a.arms.split(",") if s.strip()}

    data_path = None
    n = 0
    for run_dir in sorted(glob.glob(os.path.join(a.run_root, "outputs", "*", "*", "*"))):
        cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*_*.pth")), key=_step_of)
        if not cks:
            continue
        name = _run_name(run_dir)
        if arms and re.sub(r"_s\d+$", "", name) not in arms:
            continue
        chosen = {cks[-1]} | {p for p in cks if _step_of(p) in extra}
        for p in sorted(chosen, key=_step_of):
            dst = os.path.join(a.out, "ckpts", f"{name}__step{_step_of(p)}.pth")
            if os.path.exists(dst):
                print(f"[skip] {dst} exists")
                continue
            ck = _strip(p, dst)
            data_path = data_path or ((ck.get("config") or {}).get("dataset") or {}).get("data_path")
            n += 1
            print(f"[ckpt] {name} step {_step_of(p)} -> {dst} ({os.path.getsize(dst) / 2**20:.0f} MB)")

    if not a.no_released:
        src = os.path.join(eqr_dir, "downloaded_checkpoints", "sudoku-extreme", "eqr.pth")
        dst = os.path.join(a.out, "ckpts", "eqr-released__step50000.pth")
        if os.path.exists(src) and not os.path.exists(dst):
            ck = _strip(src, dst)
            data_path = data_path or ((ck.get("config") or {}).get("dataset") or {}).get("data_path")
            print(f"[ckpt] released -> {dst} ({os.path.getsize(dst) / 2**20:.0f} MB)")

    if data_path:
        src = data_path if os.path.isabs(data_path) else os.path.join(eqr_dir, data_path)
        name = os.path.basename(src.rstrip("/"))
        _slice_test(os.path.join(src, "test"), os.path.join(a.out, "data", name, "test"), a.n_test)
        os.makedirs(os.path.join(a.out, "data", name, "train"), exist_ok=True)
        shutil.copy(os.path.join(src, "train", "dataset.json"), os.path.join(a.out, "data", name, "train", "dataset.json"))
    print(f"[done] {n} checkpoints; data from {data_path}")


if __name__ == "__main__":
    main()
