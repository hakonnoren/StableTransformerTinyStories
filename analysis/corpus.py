"""
TinyStories val corpus: story segmentation and batching.

The v1 suite measured context sensitivity from ONE 24-token prompt and the depth
curves from ONE 127-token sequence. Everything here exists so those become
N=2000 held-out stories instead (plan §4, P1.1-P1.6).

Build the bin locally — no cluster fetch needed:
    python preprocess_tinystories.py --out_dir data --splits val
That writes data/tinystories_val.bin: 21,990 stories / 4.77M GPT-2 tokens / 9.1MB.

Padding: stories are right-padded within a batch. That is safe for a causal model
— position i attends only to <= i, so trailing pad tokens cannot influence any
real position — but the loss must still be masked to real positions, which
`token_mask` does. Never switch this to left-padding without adding position-id
handling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

EOT = 50256          # gpt2 end-of-text; the story delimiter written by preprocessing
DEFAULT_BIN = "data/tinystories_val.bin"


@dataclass
class Story:
    """One TinyStories document. `tokens` includes the trailing EOT.

    Keeping EOT is deliberate: P(EOT) is what the story-closure probe (P1.6)
    measures, so dropping it would remove the signal.
    """
    idx: int
    tokens: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.size)


def load_bin(path: str = DEFAULT_BIN) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Build it locally with:\n"
            f"    python preprocess_tinystories.py --out_dir data --splits val")
    return np.array(np.memmap(path, dtype=np.uint16, mode="r"), dtype=np.uint16)


def split_stories(tokens: np.ndarray, eot: int = EOT) -> list[Story]:
    """Segment a flat token array into stories on the EOT delimiter."""
    ends = np.flatnonzero(tokens == eot)
    out, start = [], 0
    for i, e in enumerate(ends):
        out.append(Story(i, tokens[start:e + 1]))     # inclusive of EOT
        start = e + 1
    if start < tokens.size:                            # trailing partial story
        out.append(Story(len(out), tokens[start:]))
    return out


def load_stories(path: str = DEFAULT_BIN, n: int | None = None, seed: int = 0,
                 min_len: int = 32, max_len: int = 512,
                 eot: int = EOT) -> list[Story]:
    """Load, filter and subsample stories.

    n=None keeps all of them. Sampling is a fixed permutation under `seed` so the
    same item set is reused across arms and across runs — comparisons must be on
    identical items for the paired tests in analysis/stats.py to be valid.
    """
    stories = split_stories(load_bin(path), eot=eot)
    keep = [s for s in stories if min_len <= len(s) <= max_len]
    if n is not None and n < len(keep):
        rng = np.random.default_rng(seed)
        pick = rng.permutation(len(keep))[:n]
        keep = [keep[i] for i in sorted(pick)]
    return keep


def length_stats(stories: list[Story]) -> dict:
    L = np.array([len(s) for s in stories])
    return {"n_stories": int(L.size), "tokens": int(L.sum()),
            "mean_len": float(L.mean()), "median_len": float(np.median(L)),
            "p5_len": float(np.percentile(L, 5)), "p95_len": float(np.percentile(L, 95)),
            "min_len": int(L.min()), "max_len": int(L.max())}


def pad_batch(stories: list[Story], pad_id: int = EOT,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """(ids, lengths) for a list of stories, right-padded to the batch max."""
    T = max(len(s) for s in stories)
    ids = np.full((len(stories), T), pad_id, dtype=np.int64)
    lens = np.zeros(len(stories), dtype=np.int64)
    for i, s in enumerate(stories):
        ids[i, :len(s)] = s.tokens
        lens[i] = len(s)
    return (torch.from_numpy(ids).to(device), torch.from_numpy(lens).to(device))


def token_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T) bool mask of real (non-pad) positions."""
    ar = torch.arange(T, device=lengths.device)[None, :]
    return ar < lengths[:, None]


def sorted_batches(stories: list[Story], batch_size: int):
    """Yield batches of similar-length stories to minimise padding waste.

    Sorting by length keeps padding under a few percent, which matters when the
    length distribution is wide (p5~130, p95~340 tokens).
    """
    order = sorted(range(len(stories)), key=lambda i: len(stories[i]))
    for i in range(0, len(order), batch_size):
        yield [stories[j] for j in order[i:i + batch_size]]
