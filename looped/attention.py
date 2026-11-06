"""Attention and GELU used by looped.model (copied from StableTransformers_claude/stabletransformers/models/gpt.py).

Differences from the source: plain nn.Dropout instead of SeededDropout (no reversible-backprop replay needed here),
and a ``causal`` switch read from the config (default True). Sudoku-style tasks need bidirectional attention:
set ``LoopedConfig(causal=False)``. Parameter/buffer names (c_attn, c_proj, bias) are unchanged, so checkpoints
from the source repo load.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """GELU, tanh approximation (as in GPT-2)."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with output projection; causal unless ``config.causal`` is False."""

    def __init__(self, config, n_embd):
        super().__init__()
        assert n_embd % config.n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = n_embd
        self.causal = getattr(config, "causal", True)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
