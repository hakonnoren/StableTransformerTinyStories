"""Verify the OpenWebText bins are complete and usable before committing GPU time.

Run after job_owt_preprocess.slurm:

    python verify_owt.py --data_dir data

The check that actually matters is the end-of-text count. process_openwebtext.py
appends exactly one EOT (50256) per document, so counting EOTs in the bin recovers
the number of documents that were written. If the batched/threaded encoder had
dropped, duplicated or reordered a batch, that count would not land on the
expected value. Everything else here is cheap corroboration.

Exit status is 0 only if every check passes, so this is safe to chain:
    python verify_owt.py && sbatch job_owt_medium_ddp.slurm
"""

import argparse
import os
import sys

import numpy as np

EOT = 50256
GPT2_MAX_TOKEN = 50256          # ids run 0..50256 inclusive
OWT_DOCS = 8_013_769            # Skylion007/openwebtext, matches the paper


def scan(path, chunk=1 << 26):
    """Single streaming pass: token count, EOT count, max/min id."""
    size = os.path.getsize(path)
    if size % 2:
        raise SystemExit(f"{path}: odd byte count {size} -- not a uint16 array")
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n = arr.shape[0]
    eots = 0
    hi, lo = 0, 65535
    done = 0
    while done < n:
        blk = np.asarray(arr[done:done + chunk])
        eots += int(np.count_nonzero(blk == EOT))
        if blk.size:
            hi = max(hi, int(blk.max()))
            lo = min(lo, int(blk.min()))
        done += blk.shape[0]
        pct = 100.0 * done / n
        # Redraw in place only on a terminal; piped/redirected output would
        # otherwise accumulate a wall of half-cleared progress lines.
        if sys.stdout.isatty():
            print(f"\r  scanning {os.path.basename(path)}: {pct:5.1f}%", end="", flush=True)
    if sys.stdout.isatty():
        print("\r" + " " * 70 + "\r", end="", flush=True)
    return n, eots, hi, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--total_docs", type=int, default=OWT_DOCS)
    ap.add_argument("--steps", type=int, default=30_000,
                    help="planned training steps, for the epoch-budget check")
    ap.add_argument("--tokens_per_step", type=int, default=491_520)
    ap.add_argument("--decode", action="store_true",
                    help="decode a window from each split (needs tiktoken)")
    args = ap.parse_args()

    n_val_docs = int(args.total_docs * args.val_fraction)
    n_train_docs = args.total_docs - n_val_docs
    expect = {"train": n_train_docs, "val": n_val_docs}

    print(f"expecting a {100*(1-args.val_fraction):.0f}/{100*args.val_fraction:.0f} split of "
          f"{args.total_docs:,} docs -> train {n_train_docs:,} / val {n_val_docs:,}\n")

    ok = True
    totals = {}
    for split in ("train", "val"):
        path = os.path.join(args.data_dir, f"openwebtext_{split}.bin")
        if not os.path.exists(path):
            print(f"[{split}] MISSING {path}")
            ok = False
            continue

        n, eots, hi, lo = scan(path)
        totals[split] = n
        gib = os.path.getsize(path) / 2**30
        print(f"[{split}] {path}")
        print(f"  tokens      {n:>15,}   ({gib:.2f} GiB)")

        # --- the real check ---------------------------------------------------
        want = expect[split]
        good = (eots == want)
        ok &= good
        print(f"  documents   {eots:>15,}   (one EOT each; expected {want:,})"
              f"  {'OK' if good else 'MISMATCH <-- documents lost or duplicated'}")

        # --- corroboration ----------------------------------------------------
        good = hi <= GPT2_MAX_TOKEN
        ok &= good
        print(f"  max token   {hi:>15,}   (GPT-2 vocab tops out at {GPT2_MAX_TOKEN})"
              f"  {'OK' if good else 'OUT OF RANGE'}")
        print(f"  min token   {lo:>15,}")

        tpd = n / eots if eots else 0
        # OWT averages ~1130 tokens/doc; a wildly different value means the wrong
        # corpus or a truncated encode, not just noise.
        good = 800 < tpd < 1600
        ok &= good
        print(f"  tokens/doc  {tpd:>15,.1f}   (OpenWebText is ~1130)"
              f"  {'OK' if good else 'SUSPICIOUS'}")
        print()

    if len(totals) == 2:
        tot = totals["train"] + totals["val"]
        print(f"total tokens  {tot:>15,}   (paper reports 9.05B for this corpus)")
        good = abs(tot - 9.05e9) / 9.05e9 < 0.05
        ok &= good
        print(f"              {'OK' if good else 'off by more than 5% <-- different snapshot?'}\n")

        budget = args.steps * args.tokens_per_step
        epochs = budget / totals["train"]
        print(f"planned run   {args.steps:,} steps x {args.tokens_per_step:,} tok "
              f"= {budget/1e9:.2f}B tokens = {epochs:.2f} epochs "
              f"(paper: 14.75B / ~1.7 epochs)")

    # --- does the training loader actually accept them? -----------------------
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data import DataConfig, BlockEpochIterator
        arr = np.memmap(os.path.join(args.data_dir, "openwebtext_val.bin"), dtype=np.uint16, mode="r")
        it = BlockEpochIterator(np.asarray(arr[:4_000_000]),
                                DataConfig(block_size=1024, batch_size=2), split="val")
        x, y = next(it)
        good = tuple(x.shape) == (2, 1024) and bool((x[:, 1:] == y[:, :-1]).all())
        ok &= good
        print(f"\nloader        x{tuple(x.shape)} y{tuple(y.shape)}, y is x shifted by one: "
              f"{'OK' if good else 'BROKEN'}")
    except Exception as e:                                   # noqa: BLE001
        print(f"\nloader        could not test: {e}")

    if args.decode:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            for split in ("train", "val"):
                arr = np.memmap(os.path.join(args.data_dir, f"openwebtext_{split}.bin"),
                                dtype=np.uint16, mode="r")
                mid = len(arr) // 2
                txt = enc.decode([int(t) for t in arr[mid:mid + 60]])
                print(f"\n[{split}] sample @{mid:,}: {txt!r}")
        except Exception as e:                               # noqa: BLE001
            print(f"decode failed: {e}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE -- do not train on these bins"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
