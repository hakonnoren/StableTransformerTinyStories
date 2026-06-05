"""
Preprocess TinyStories with a ~10K-vocab byte-level BPE, to match the original
TinyStories paper's setup (GPT-Neo-style decoder; vocab truncated to ~10K).

This trains a fresh 10K ByteLevel BPE on the TinyStories train split (a faithful
approximation of "GPT-Neo tokenizer truncated to top 10K" — same byte-level BPE
family, vocab specialised to TinyStories' simple English), writes:

    data/tinystories10k_train.bin   (uint16 token ids)
    data/tinystories10k_val.bin
    data/tinystories10k_tokenizer.json

The <|endoftext|> separator is the FIRST special token => id 0 (verify from the
printed report; pass it to train.py as --sample_eos_token_id).

Deps: `datasets`, `tokenizers`. Run on a node with internet (login node):
    python preprocess_tinystories_10k.py --vocab_size 10000
"""
import argparse
import os
from array import array

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--prefix", type=str, default="tinystories10k")
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--train_tokenizer_docs", type=int, default=200000,
                    help="# train docs to fit the BPE on (0 = all)")
    ap.add_argument("--max_docs_train", type=int, default=0, help="0 = all")
    ap.add_argument("--max_docs_val", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError as e:
        raise SystemExit("Need `datasets` and `tokenizers` installed.") from e

    ds = load_dataset("roneneldan/TinyStories")
    train, val = ds["train"], ds["validation"]

    # ---- train the BPE -------------------------------------------------------
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],          # -> id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    n_fit = len(train) if args.train_tokenizer_docs == 0 else min(len(train), args.train_tokenizer_docs)
    print(f"[tokenizer] training {args.vocab_size}-BPE on {n_fit} docs ...")

    def text_iter():
        for i in range(n_fit):
            yield train[i]["text"]

    tok.train_from_iterator(text_iter(), trainer)
    tok_path = os.path.join(args.out_dir, f"{args.prefix}_tokenizer.json")
    tok.save(tok_path)
    eot = tok.token_to_id("<|endoftext|>")
    real_vocab = tok.get_vocab_size()
    print(f"[tokenizer] saved {tok_path}  vocab_size={real_vocab}  <|endoftext|> id={eot}")
    print(f"            -> train.py: --dataset {args.prefix} --vocab_size {real_vocab} "
          f"--sample_eos_token_id {eot} --tokenizer_json {tok_path}")

    if real_vocab - 1 > 65535:
        raise ValueError("vocab too large for uint16")

    # ---- encode splits to .bin ----------------------------------------------
    def dump(split, dset, max_docs, out_path):
        n = len(dset) if max_docs == 0 else min(len(dset), max_docs)
        print(f"[{split}] encoding {n} docs -> {out_path}")
        with open(out_path, "wb") as f:
            buf = array("H")
            for i in range(n):
                ids = tok.encode(dset[i]["text"]).ids
                ids.append(eot)
                buf.extend(ids)
                if len(buf) > 1_000_000:
                    buf.tofile(f); buf = array("H")
                if (i + 1) % 50000 == 0:
                    print(f"  {i+1}/{n}")
            if len(buf):
                buf.tofile(f)
        arr = np.memmap(out_path, dtype=np.uint16, mode="r")
        print(f"[{split}] tokens={len(arr)}")

    dump("train", train, args.max_docs_train, os.path.join(args.out_dir, f"{args.prefix}_train.bin"))
    dump("val", val, args.max_docs_val, os.path.join(args.out_dir, f"{args.prefix}_val.bin"))


if __name__ == "__main__":
    main()
