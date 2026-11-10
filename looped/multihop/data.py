"""Data of the multi-hop composition task (Kohli et al., data/multi_hop from the authors' Drive folder).

Files: vocab.json (entity tokens <e_i>, relation tokens <r_j>, ...), train.json (2k atomic facts and 15k k-hop facts
for each k = 2..40; input "<e_h><r_1>...<r_k>", target "...<e_t></a>"), test.json (750 held-out k-hop facts for each
k = 2..100, plus 2000 training-set samples per hop level and 2000 atomic facts, as types "{k}hop_train" and
"atomic_facts"). valid.json is a copy of test.json, which the authors' curriculum used for its stage decisions.

Here the held-out facts of each hop level are split deterministically: every third one (index % 3 == 0, 250 per k) is
the dev split that the curriculum decides on, the other 500 the test split that evaluate.py reports.

Batches are right-padded to the longest input + 1 and the answer is read at the last position, which is then always a
<pad> position (as in the authors' setup, which padded to 50 tokens and read position -1). With NoPE and pad keys
masked, the prediction does not depend on how much padding there is.
"""
import json
import os
import re

import numpy as np
import torch

TOKEN = re.compile(r"<[^>]+>")


def load_vocab(path):
    with open(path) as f:
        toks = json.load(f)
    if "<pad>" not in toks:
        toks = ["<pad>"] + toks
    return {t: i for i, t in enumerate(toks)}


def hops_of(text):
    return len(TOKEN.findall(text)) - 1


def encode(items, vocab):
    """-> list of (input ids, target id, hops)."""
    out = []
    for it in items:
        x = [vocab[t] for t in TOKEN.findall(it["input_text"])]
        y = [vocab[t] for t in TOKEN.findall(it["target_text"]) if t != "</a>"][-1]
        out.append((np.asarray(x, dtype=np.int64), int(y), len(x) - 1))
    return out


def collate(rows, device="cpu"):
    n = max(len(r[0]) for r in rows) + 1
    ids = np.zeros((len(rows), n), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, :len(r[0])] = r[0]
    ids = torch.as_tensor(ids, device=device)
    y = torch.as_tensor([r[1] for r in rows], device=device)
    return ids, (ids != 0).long(), y


class MultiHop:
    def __init__(self, data_dir, vocab=None, need_train=True):
        self.vocab = vocab or load_vocab(os.path.join(data_dir, "vocab.json"))
        self.train = []
        if need_train:
            with open(os.path.join(data_dir, "train.json")) as f:
                self.train = encode(json.load(f), self.vocab)
            self.train_hops = np.array([r[2] for r in self.train])
        test_file = os.path.join(data_dir, "test.json")
        if not os.path.exists(test_file):                         # local smoke tests: the 50-per-hop subset
            test_file = os.path.join(data_dir, "test_small.json")
        with open(test_file) as f:
            items = json.load(f)
        self.dev, self.test, self.seen = {}, {}, {}
        by_type = {}
        for it in items:
            by_type.setdefault(it["type"], []).append(it)
        for typ, its in by_type.items():
            m = re.fullmatch(r"(\d+)hop_(test|train)", typ)
            if not m:
                continue
            k, kind = int(m.group(1)), m.group(2)
            rows = encode(its, self.vocab)
            if kind == "train":
                self.seen[k] = rows                                     # training-set samples (train accuracy)
            else:
                self.dev[k] = rows[0::3]
                self.test[k] = [r for i, r in enumerate(rows) if i % 3]

    def train_upto(self, k):
        return np.flatnonzero(self.train_hops <= k)

    def batch(self, idx, device):
        return collate([self.train[i] for i in idx], device)
