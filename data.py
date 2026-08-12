
import contextlib
import os
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import torch


@dataclass
class DataConfig:
    block_size: int = 1024
    batch_size: int = 4          # microbatch per step (per GPU)
    grad_accum_steps: int = 1
    seed: int = 1337
    device: str = "cuda"
    # Data-parallel sharding. Defaults (0, 1) reproduce the historical
    # single-process behaviour exactly -- see BlockEpochIterator.__next__.
    rank: int = 0
    world_size: int = 1


class BlockEpochIterator:
    '''
    Implements the deterministic epoch protocol described in YuriiFormer Appendix A.2:
      - tokenize into a long sequence
      - train in epochs over non-overlapping T-token blocks
      - each epoch visits every block exactly once
      - between epochs: shift block boundaries by a seeded offset and reshuffle block order
    '''
    def __init__(self, tokens: np.ndarray, cfg: DataConfig, split: str):
        assert tokens.ndim == 1
        self.tokens = tokens
        self.cfg = cfg
        self.split = split
        self.rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))

        self.T = cfg.block_size
        # number of full blocks
        self.n_blocks = (len(tokens) - 1) // self.T  # -1 because we need y = x shifted by 1
        if self.n_blocks <= 0:
            raise ValueError("Not enough tokens for one block")

        self.epoch = 0
        self._starts = None
        self._pos = 0
        self._prepare_epoch()

    def _prepare_epoch(self):
        # seeded offset in [0, T-1]
        offset = int(self.rng.integers(low=0, high=self.T))
        starts = offset + np.arange(self.n_blocks, dtype=np.int64) * self.T
        # ensure within bounds for x of length T and y shifted by 1
        max_start = len(self.tokens) - (self.T + 1)
        starts = starts[starts <= max_start]
        self.rng.shuffle(starts)
        self._starts = starts
        self._pos = 0
        self.epoch += 1

    def __iter__(self) -> "BlockEpochIterator":
        return self

    @contextlib.contextmanager
    def unsharded(self):
        """Draw rank-independent batches for the duration of the block.

        For one-off *probe* batches (cheap_metrics' fixed batch, the reconstruction-
        drift batch) that are pulled from the val iterator at startup. Those draws
        must advance the stream by the same amount no matter how many ranks are
        running, otherwise the position at which evaluation subsequently begins
        depends on the world size and val loss stops being comparable between a
        1-GPU and an N-GPU run of the same config.

        Inside the block every rank takes the identical batch and _pos advances by
        batch_size (not batch_size*world_size), exactly as a single-process run
        does -- so single-process behaviour is untouched and the N-rank stream stays
        aligned across ranks.
        """
        rank, world = self.cfg.rank, self.cfg.world_size
        self.cfg.rank, self.cfg.world_size = 0, 1
        try:
            yield self
        finally:
            self.cfg.rank, self.cfg.world_size = rank, world

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.cfg.batch_size
        ws = max(1, int(self.cfg.world_size))
        # Blocks consumed *globally* per micro-step. Rank r takes the r-th slice of
        # that span, so the union over ranks at micro-step m is exactly the span a
        # 1-GPU run would have consumed at micro-steps m*ws .. m*ws+ws-1, in the same
        # order. Data order is therefore identical to single-GPU for any world size,
        # which is what lets DDP and non-DDP runs of the same config be compared
        # directly (the paper controls for data order across arms).
        # With ws=1 this reduces term-for-term to the original two lines.
        span = bs * ws
        if self._pos + span > len(self._starts):
            # start new epoch
            self._prepare_epoch()

        lo = self._pos + self.cfg.rank * bs
        batch_starts = self._starts[lo:lo + bs]
        self._pos += span

        T = self.T
        x = np.stack([self.tokens[s:s+T] for s in batch_starts], axis=0)
        y = np.stack([self.tokens[s+1:s+T+1] for s in batch_starts], axis=0)

        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x, y


def load_bin(path: str) -> np.ndarray:
    # uint16 tokens as in nanoGPT
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    return np.array(arr, dtype=np.uint16)
