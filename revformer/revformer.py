"""
RevFormer: a reversible-coupling transformer for an as-equal-as-possible
comparison against the vanilla GPT baseline defined in ../model.py.

Design goal
-----------
The ONLY architectural difference between RevFormer and the baseline GPTModel
should be (a) the reversible coupling and (b) the optional volume scaling. To
guarantee that, the attention / MLP / LayerNorm primitives are imported
*verbatim* from ../model.py and merely instantiated at half width.

The reversible block
--------------------
Each block partitions the internal state y (width 2*n_embd) into two streams
(x, z), each of width d = n_embd, and applies

    x_new = exp(-gamma_used) * (x + Attn(LN_1(z)))
    z_new = exp(-alpha_used) * z + MLP(LN_2(x_new))

with per-dimension scale vectors gamma, alpha of length d = n_embd. The block's
log|det| at one sequence position is -sum_i (gamma_used_i + alpha_used_i).

Width matching (no manual doubling needed)
------------------------------------------
Attn and MLP run at width d = n_embd, i.e. the *same* width as the baseline. So
passing the same --n_embd / --n_head to both gives matching per-block attn+MLP
parameter counts *naively* -- no need to double anything:

    baseline:   --arch baseline   --n_embd D --n_head H
    revformer:  --arch reversible --n_embd D --n_head H

The per-block attn+MLP parameter counts then match the baseline *exactly*. The
internal state is 2*D wide (two D-wide streams), so the embedding / lm_head /
final-LayerNorm params are ~2x the baseline's -- inherent to the reversible
coupling, which carries two streams.

Volume regimes (RevConfig.regime / --rev_regime)
------------------------------------------------
The only thing that differs between the four regimes is how gamma/alpha are
centered, which sets each block's log-volume change:

  vpb_baseline : gamma = alpha = 0, frozen. Plain identity-residual reversible
                 block, no scaling. (trivial per-block volume preservation)
  vpb_scaling  : each block self-centers its own gamma/alpha so that
                 sum_i (gamma_used + alpha_used) = 0 within every block.
                 log|det| = 0 per block, gamma/alpha trainable.
  vpm_scaling  : global centering across all blocks. Total log|det| of the full
                 stack = -lambd (set lambd=0 for volume preservation across the
                 whole depth, with individual blocks free to expand/contract).
                 This is the default.
  vf_scaling   : no centering; gamma/alpha apply directly. Volume is free to
                 change however the model likes (deliberately not VP).

Increasing freedom: vpb_baseline -> vpb_scaling -> vpm_scaling -> vf_scaling.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the baseline's primitives so the comparison is apples-to-apples.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from model import (  # noqa: E402
    ModelConfig,
    CausalSelfAttention,
    MLP,
    LayerNorm,
    _generic_generate,
)


_VALID_REGIMES = ("vpb_baseline", "vpb_scaling", "vpm_scaling", "vf_scaling",
                  "damped", "damped_mem")
_DAMPED_REGIMES = ("damped", "damped_mem")


@dataclass
class RevConfig:
    """Volume-preservation configuration for the reversible block."""

    regime: str = "vpm_scaling"   # one of _VALID_REGIMES
    lambd: float = 0.0            # vpm only: total log|det| of the full stack = -lambd
    epsilon: float = 1.0          # scale of gamma/alpha init (and tanh range if tanh_scale)
    randn_init: bool = False      # init gamma/alpha ~ N(0,1)*epsilon instead of zeros
    tanh_scale: bool = False      # squash gamma/alpha through tanh(.)*epsilon
    # ---- damped regimes (per-layer contraction budget; see theory/damped_reversible_plan.md) ----
    # gamma = 0.5*kappa*(1+tanh u), alpha = 0.5*kappa*(1-tanh u), kappa in [kappa_min,kappa_max]>0
    # => gamma+alpha = kappa > 0: every layer & coord contracts; u routes damping X(+)/Z(-).
    kappa_min: float = 0.005
    kappa_max: float = 0.08
    kappa_mem: float = 0.001      # damped_mem: contraction floor for protected "memory" channels

    def __post_init__(self):
        if self.regime not in _VALID_REGIMES:
            raise ValueError(
                f"RevConfig.regime must be one of {_VALID_REGIMES}; got {self.regime!r}"
            )
        if self.regime != "vpm_scaling" and self.lambd != 0.0:
            raise ValueError(
                f"lambd only applies to regime='vpm_scaling'; got lambd={self.lambd} "
                f"with regime={self.regime!r}"
            )
        if self.regime == "vpb_baseline" and (self.randn_init or self.tanh_scale):
            offenders = [n for n, v in (("randn_init", self.randn_init),
                                        ("tanh_scale", self.tanh_scale)) if v]
            raise ValueError(
                "regime='vpb_baseline' freezes gamma/alpha to zero, so these flags "
                f"have no effect: {offenders}"
            )


class ReversibleBlock(nn.Module):
    """One reversible coupling block; see module docstring for the update rule."""

    def __init__(self, cfg: ModelConfig, rev_cfg: RevConfig):
        super().__init__()
        # Each stream is n_embd wide; Attn/MLP run at the same width as the
        # baseline, so per-block params match it without any width doubling.
        d = cfg.n_embd

        self.ln_1 = LayerNorm(d, bias=cfg.bias)
        self.ln_2 = LayerNorm(d, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

        self.regime = rev_cfg.regime
        self.epsilon = float(rev_cfg.epsilon)
        self.use_tanh = bool(rev_cfg.tanh_scale)
        self.frozen = self.regime == "vpb_baseline"
        self.damped = self.regime in _DAMPED_REGIMES
        self.use_mem = self.regime == "damped_mem"

        if self.damped:
            # Per-layer contraction budget. rho -> kappa in [kappa_min,kappa_max];
            # u -> split. Init rho=-2 (sigmoid~0.12 => light kappa), u=0 (symmetric).
            self.kappa_min = float(rev_cfg.kappa_min)
            self.kappa_max = float(rev_cfg.kappa_max)
            self.kappa_mem = float(rev_cfg.kappa_mem)
            self.rho = nn.Parameter(torch.full((1, 1, d), -2.0))
            self.u = nn.Parameter(torch.zeros(1, 1, d))
            # shared memory gate (set by the model for damped_mem); list wrapper so
            # the shared Parameter is NOT re-registered on every block.
            self._mem_gate = [None]
        elif self.frozen:
            # Buffers (follow .to(device) but stay frozen at zero).
            self.register_buffer("gamma_bias", torch.zeros(1, 1, d))
            self.register_buffer("alpha_bias", torch.zeros(1, 1, d))
        else:
            init = torch.randn if rev_cfg.randn_init else torch.zeros
            self.gamma_bias = nn.Parameter(init(1, 1, d) * self.epsilon)
            self.alpha_bias = nn.Parameter(init(1, 1, d) * self.epsilon)

    def _kappa(self) -> torch.Tensor:
        """Per-coordinate contraction budget kappa = gamma + alpha > 0."""
        kd = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.rho)
        if self.use_mem and self._mem_gate[0] is not None:
            m = torch.sigmoid(self._mem_gate[0])          # protected memory channels
            kd = (1.0 - m) * kd + m * self.kappa_mem
        return kd

    def get_gamma(self) -> torch.Tensor:
        if self.damped:
            return 0.5 * self._kappa() * (1.0 + torch.tanh(self.u))   # X-stream log-damp >= 0
        if self.use_tanh:
            return torch.tanh(self.gamma_bias) * self.epsilon
        return self.gamma_bias

    def get_alpha(self) -> torch.Tensor:
        if self.damped:
            return 0.5 * self._kappa() * (1.0 - torch.tanh(self.u))   # Z-stream log-damp >= 0
        if self.use_tanh:
            return torch.tanh(self.alpha_bias) * self.epsilon
        return self.alpha_bias

    def _effective_gamma_alpha(self, avg):
        """gamma/alpha actually used as exponents after the configured centering."""
        gamma = self.get_gamma()
        alpha = self.get_alpha()
        if self.regime == "vpb_scaling":
            # Self-center so sum_i (gamma_used + alpha_used) = 0 within this block.
            c = (gamma.mean() + alpha.mean()) / 2
            return gamma - c, alpha - c
        # vpm_scaling: avg is the global correction passed in from the model.
        # vf_scaling: avg is 0.0, so this is the identity.
        return gamma - avg, alpha - avg

    def forward(self, y: torch.Tensor, avg=0.0) -> torch.Tensor:
        x, z = torch.split(y, y.shape[-1] // 2, dim=-1)
        if self.frozen:
            x = x + self.attn(self.ln_1(z))
            z = z + self.mlp(self.ln_2(x))
        else:
            gamma, alpha = self._effective_gamma_alpha(avg)
            x = torch.exp(-gamma) * (x + self.attn(self.ln_1(z)))
            z = torch.exp(-alpha) * z + self.mlp(self.ln_2(x))
        return torch.cat([x, z], dim=-1)


class RevFormerModel(nn.Module):
    """Reversible-coupling GPT. Drop-in interface match with GPTModel:
    forward(idx, targets, global_step) -> (logits, loss), plus .generate()."""

    def __init__(self, cfg: ModelConfig, rev_cfg: Optional[RevConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.rev_cfg = rev_cfg if rev_cfg is not None else RevConfig()

        # Internal state carries two n_embd-wide streams.
        state_dim = 2 * cfg.n_embd
        self.tok_emb = nn.Embedding(cfg.vocab_size, state_dim)
        self.pos_emb = nn.Embedding(cfg.block_size, state_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [ReversibleBlock(cfg, self.rev_cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(state_dim, bias=cfg.bias)
        self.lm_head = nn.Linear(state_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Match GPTModel's initialization exactly for a fair comparison.
        self.apply(self._init_weights)

        # damped_mem: one memory gate shared across all layers (a channel is
        # "memory" consistently through depth). Owned here; handed to each block
        # via a list wrapper so it is registered (and optimized) exactly once.
        if self.rev_cfg.regime == "damped_mem":
            self.mem_gate = nn.Parameter(torch.full((1, 1, cfg.n_embd), -2.0))  # m~0.12: mostly damping
            for b in self.blocks:
                b._mem_gate = [self.mem_gate]

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _avg_corr(self, T: int) -> torch.Tensor:
        """Global-centering correction for the vpm_scaling regime. Recomputed
        from current params each forward so it stays differentiable in
        gamma/alpha. Total log|det| of the full stack then equals -lambd
        (T-independent)."""
        n = len(self.blocks)
        d = self.cfg.n_embd
        gamma_avg = torch.mean(torch.stack([b.get_gamma().mean() for b in self.blocks]))
        alpha_avg = torch.mean(torch.stack([b.get_alpha().mean() for b in self.blocks]))
        avg = (gamma_avg + alpha_avg) / 2
        return avg - self.rev_cfg.lambd / (2 * n * d * T)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                global_step: Optional[int] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        # Only the global (vpm) regime needs a cross-block correction; the others
        # either self-center (vpb_scaling), are frozen (vpb_baseline), or apply
        # gamma/alpha directly (vf_scaling).
        avg = self._avg_corr(T) if self.rev_cfg.regime == "vpm_scaling" else 0.0

        for blk in self.blocks:
            x = blk(x, avg)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# Reuse the baseline's exact sampling logic.
RevFormerModel.generate = _generic_generate
