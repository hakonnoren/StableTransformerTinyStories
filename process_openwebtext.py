import argparse
import os
import time
from itertools import chain

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--val_fraction", type=float, default=0.005, help="fraction of docs for validation (OpenWebText has only train split)")
    ap.add_argument("--max_docs_train", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_docs_val", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_threads", type=int, default=0,
                    help="tiktoken encode threads. 0 = SLURM_CPUS_PER_TASK, else os.cpu_count(). "
                         "The BPE runs in Rust with the GIL released, so these are real threads.")
    ap.add_argument("--batch_docs", type=int, default=2000,
                    help="documents fetched from Arrow and handed to the tokenizer per batch")
    args = ap.parse_args()

    if args.num_threads <= 0:
        args.num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 8)

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import tiktoken
    except ImportError as e:
        raise SystemExit("Please install tiktoken to run preprocessing.") from e

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Please install datasets (huggingface) to run preprocessing.") from e

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    ds = load_dataset("Skylion007/openwebtext", split="train")  # single split
    n_total = len(ds)
    n_val = int(n_total * args.val_fraction)

    # Deterministic split: first n_val docs for val, rest for train. Expressed as
    # an OFFSET into ds rather than ds.select(): .select() attaches an indices
    # mapping, so every subsequent read goes through an extra indirection. That is
    # cheap locally but expensive on a network filesystem like /cluster, where the
    # original per-row loop was the dominant cost of this script. Same documents,
    # same order, no indirection.
    def dump(split, offset, n_docs, max_docs, out_path):
        # Batched + threaded encoding. Two things made the original loop slow, and
        # both are fixed here without changing a single output byte:
        #
        #  1. `dset[i]["text"]` is one Arrow row lookup per document. Slicing
        #     `dset[j:j+B]["text"]` returns B strings in one go.
        #  2. `enc.encode_ordinary` encodes one document per GIL-holding call.
        #     tiktoken's `encode_ordinary_batch` runs the BPE in its Rust core with
        #     the GIL released, so plain threads genuinely parallelise it -- no
        #     multiprocessing, no temp shards, no extra disk.
        #
        # Documents are still consumed in strict order and their token lists are
        # still concatenated in that order, so the emitted .bin is byte-identical
        # to what the sequential version produced (verified against it directly).
        n = n_docs if max_docs == 0 else min(n_docs, max_docs)
        print(f"[{split}] docs={n} -> {out_path} "
              f"(batch={args.batch_docs}, threads={args.num_threads})", flush=True)
        t0 = time.time()
        with open(out_path, "wb") as f:
            for j in range(0, n, args.batch_docs):
                hi = min(j + args.batch_docs, n)
                texts = ds[offset + j:offset + hi]["text"]
                batch = enc.encode_ordinary_batch(texts, num_threads=args.num_threads)
                total = 0
                for toks in batch:
                    if toks and max(toks) >= 65535:
                        raise ValueError("Token id exceeds uint16 range")
                    total += len(toks) + 1          # +1 for the eot appended below
                # Flatten the whole batch in one C-level pass. The obvious version
                # -- buf.extend(toks) once per document -- costs a Python-level call
                # per doc and measured ~1.7x slower here. That matters more than it
                # looks: it is the serial part of an otherwise threaded loop, so it
                # sets the Amdahl ceiling on how much --num_threads can ever buy.
                np.fromiter(
                    chain.from_iterable(chain(toks, (eot,)) for toks in batch),
                    dtype=np.uint16, count=total,
                ).tofile(f)
                dt = time.time() - t0
                rate = hi / dt if dt > 0 else 0.0
                eta = (n - hi) / rate if rate > 0 else 0.0
                print(f"  processed {hi}/{n}  ({rate:,.0f} docs/s, "
                      f"ETA {eta/3600:.1f}h)", flush=True)
            # No trailing flush needed: each batch is written whole above.
        arr = np.memmap(out_path, dtype=np.uint16, mode="r")
        print(f"[{split}] tokens={len(arr)} in {(time.time()-t0)/60:.1f} min", flush=True)

    dump("train", n_val, n_total - n_val, args.max_docs_train,
         os.path.join(args.out_dir, "openwebtext_train.bin"))
    dump("val", 0, n_val, args.max_docs_val,
         os.path.join(args.out_dir, "openwebtext_val.bin"))

if __name__ == "__main__":
    main()