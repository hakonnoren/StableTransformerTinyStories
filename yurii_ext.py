"""Determinant-preserving extensions of Polyak + Lie-Trotter YuriiFormer.

Kept out of `model.py` on purpose: everything here is one architecture family and
the point of the family is that all four members share *exactly* the same layer
determinant, so they belong together and nowhere else.

The base block
--------------
Polyak (not Nesterov) momentum, Lie-Trotter split into an attention substep and
an MLP substep, written in the **reversible shear form**::

    Z_A = ln_x_attn(x)
    v_half = beta_A * v + gamma_A * Attn(Z_A)
    x_half = x + eta_A * ln_v(v_half)

    Z_M = ln_x_mlp(x_half)
    v_next = beta_M * v_half + gamma_M * MLP(Z_M)
    x_next = x_half + eta_M * ln_v(v_next)

Two things differ from `model.YuriiFormerLieTrotterBlock`, both deliberate:

* **No lookahead** (`mu`). Nesterov's `x_in = x + mu * v` feeds `v` into the
  drift's argument, so `d v_half / d v` picks up `mu * dAttn/dx` and the substep
  Jacobian stops being triangular. Polyak keeps it triangular, which is the
  whole basis of the exact-determinant claim below.
* **`ln_v` is read-only.** The existing block does `v <- ln_v(v)`, which
  overwrites the velocity and destroys both reversibility and any closed-form
  determinant. Here `ln_v` is applied *inside the drift* only; the retained `v`
  is the raw one, so `x <- x + eta * ln_v(v)` is a plain shear (unit Jacobian in
  `x`) and `ln_v`'s affine parameters stay learnable without affecting anything
  volumetric. Same for `ln_x_attn` / `ln_x_mlp`.

Because each substep factors as (shear in v) o (shear in x),

    det d(x_half, v_half) / d(x, v) = det diag(retention)

and for the whole block, per token,

    log|det J| = n_embd * (log beta_A + log beta_M).

The two extensions
------------------
Both replace the scalar retention `beta * v` with something richer whose
determinant is *still* `beta^d` per token, so the four arms below are comparable
at fixed volume contraction:

1. **Adaptive damping** (`adaptive=True`). Channel-wise, input-dependent
   retention with the geometric mean pinned to `beta`::

       q = tanh(Z),  q~ = q - mean_channel(q),  S = beta * exp(-eps * q~)

   `sum_i q~_i = 0` exactly, so `prod_i S_i = beta^d` exactly. `Z` is the
   pre-norm activation the substep already computed, so the extra work is
   O(n*d): a channel mean, a tanh, an exp, a multiply. Cost: **1 scalar per
   substep** (`eps`).

2. **Gyro-Polyak** (`gyro=True`). A fixed-pair reversible shear mixer applied to
   the velocity before retention::

       p' = p + w q
       q' = q - w p'

   `det = 1` exactly (two unit-triangular shears), inverse is two more axpys, and
   the retained eigenvalues move from the real pole `beta` to the conjugate pair
   `beta * exp(+-i theta)` with `cos theta = 1 - w^2/2` (for `|w| < 2`), i.e. the
   momentum gains a *phase* on top of its decay rate. Channel pairing alternates
   across depth -- `(0,1),(2,3),...` in even layers, `(1,2),(3,4),...,(d-1,0)`
   in odd layers -- so no two coordinates stay an isolated pair through the
   stack. Cost: **1 scalar per substep** (`omega`), or `gyro_groups` of them.

Both are bounded (`eps = eps_max * tanh(rho)`, `omega = omega_max * tanh(rho)`)
and initialized at `rho = 0`, so **at initialization every arm is bit-identical
to the plain Polyak block** and any difference is the mechanism, not extra
capacity. The bounds matter: determinant one does not imply good conditioning,
and shear singular values come in reciprocal pairs whose spread grows with the
shear magnitude.

The four arms
-------------
=================  ==================  =============  ==================
arm                new params / block  extra Attn/MLP  layer determinant
=================  ==================  =============  ==================
polyak (baseline)  0                   0              (beta_A beta_M)^nd
adaptive           2 scalars           0              same
gyro               2 scalars           0              same
adaptive+gyro      4 scalars           0              same
=================  ==================  =============  ==================

Run the self-test with ``python yurii_ext.py`` (checks the determinant against
an autograd Jacobian, the exact inverse, init-time equivalence to the baseline,
and causality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    ModelConfig,
    LayerNorm,
    CausalSelfAttention,
    MLP,
    ConstrainedScalar,
    _generic_generate,
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class YuriiExtConfig:
    """Which extension(s) to switch on, and how far they may travel.

    Defaults give the plain Polyak + Lie-Trotter baseline.
    """

    adaptive: bool = False          # channel-wise state-dependent damping
    gyro: bool = False              # reversible shear mixer on the velocity

    eps_max: float = 0.5            # |eps| bound; S in [beta*e^-2eps_max, beta*e^+2eps_max]
    omega_max: float = 0.5          # |omega| bound; keep < 2 for the rotation reading
    gyro_groups: int = 1            # channel-pair groups, each with its own omega
    gyro_alternate: bool = True     # shift the pairing by one channel in odd layers

    beta_init: float = 0.9          # momentum retention (unit-constrained)
    gamma_init: float = 1.0         # drift gain (positive-constrained)
    eta_init: float = 1.0           # shear step size (positive-constrained)

    use_v0_init: bool = True        # separate token+pos tables for v_0 (YuriiFormer A.1)
    no_mlp: bool = False            # drop the MLP substep entirely

    def arm_name(self) -> str:
        if self.adaptive and self.gyro:
            return "adaptive+gyro"
        if self.adaptive:
            return "adaptive"
        if self.gyro:
            return "gyro"
        return "polyak"


# --------------------------------------------------------------------------- #
# bounded scalar
# --------------------------------------------------------------------------- #
class BoundedScalar(nn.Module):
    """`bound * tanh(raw)`, initialized at `init` (default 0, i.e. raw = 0 exactly).

    The parameter is called `raw` so `train.build_optimizer` sweeps it into the
    "learned scalar update-rule params" group (weight decay 0, 5x lr) alongside
    the `ConstrainedScalar` raws.

    `init != 0` exists for coefficients that would otherwise start at a
    zero-gradient point -- see `yurii_state.ThermostatBlock`, where eps and rho
    multiply each other and both are dead if either starts at 0.
    """

    def __init__(self, bound: float, numel: int = 1, init: float = 0.0):
        super().__init__()
        self.bound = float(bound)
        frac = float(init) / self.bound
        if not -1.0 < frac < 1.0:
            raise ValueError(f"init={init} must lie strictly inside (-{bound}, {bound})")
        raw0 = math.atanh(frac)
        self.raw = nn.Parameter(torch.full((numel,), raw0, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.bound * torch.tanh(self.raw)

    def extra_repr(self) -> str:
        return f"bound={self.bound}, numel={self.raw.numel()}"


# --------------------------------------------------------------------------- #
# extension 1: adaptive damping
# --------------------------------------------------------------------------- #
def _hi(t: torch.Tensor) -> torch.Tensor:
    """Promote low-precision tensors to fp32, leave fp32/fp64 alone.

    `Tensor.float()` would *down*cast fp64, which silently costs the fp64
    determinant test six orders of magnitude.
    """
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


class AdaptiveDamping(nn.Module):
    """S(Z) = beta * exp(-eps * center(tanh(Z))), with prod_i S_i = beta^d exactly.

    `Z` is the substep's own pre-norm activation, so this adds no projection, no
    gate MLP and no new normalization -- just O(n*d) elementwise work.

    Precision, measured rather than assumed (`_check_realized_determinant`):

    * Under `train.py`'s bf16 autocast the residual and velocity streams stay
      **fp32** -- embeddings are fp32, LayerNorm runs in fp32 under autocast, and
      only the Linear/matmul oracles emit bf16, which promotes back on the add.
      So `Z` arrives in fp32 and the identity holds to fp32 rounding.
    * `_hi` covers the case where someone runs the whole model in bf16, where a
      768-term mean in bf16 would be the dominant error. Even then the retention
      that actually multiplies `v` is the bf16 round of `S`, and the realized
      log-det wanders from nominal by ~0.06 nats/token/substep at d=768 -- about
      20x *less* than the scalar baseline's own ~1.39, because `bf16(beta)` is a
      coherent -0.17% shift applied to all d channels while `S`'s per-channel
      roundings scatter and cancel. An all-bf16 Polyak baseline is the arm whose
      volume budget is least trustworthy, not this one.
    """

    def __init__(self, eps_max: float):
        super().__init__()
        self.eps = BoundedScalar(eps_max)

    def forward(self, z: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        q = torch.tanh(_hi(z))
        q = q - q.mean(dim=-1, keepdim=True)
        s = _hi(beta) * torch.exp(-_hi(self.eps()) * q)
        return s.to(z.dtype)


# --------------------------------------------------------------------------- #
# extension 2: gyroscopic shear mixer
# --------------------------------------------------------------------------- #
class GyroMixer(nn.Module):
    """Fixed-pair volume-preserving mixer: p' = p + w q; q' = q - w p'.

    Two unit-triangular shears, so `det = 1` exactly and the inverse is two more
    axpys -- no solve, no matrix inverse, no matrix exponential. Eigenvalues are
    `exp(+-i theta)` with `cos theta = 1 - w^2/2` for `|w| < 2`.

    `offset=1` rolls the channel axis by one before pairing, giving
    `(1,2),(3,4),...,(d-1,0)`; the roll is undone afterwards, so the composite is
    exactly the shifted pairing (and still `det = 1`).
    """

    def __init__(self, n_embd: int, omega_max: float, groups: int = 1, offset: int = 0):
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError(f"GyroMixer needs an even n_embd; got {n_embd}")
        self.n_pairs = n_embd // 2
        if self.n_pairs % int(groups) != 0:
            raise ValueError(
                f"gyro_groups={groups} must divide n_embd/2={self.n_pairs}"
            )
        self.groups = int(groups)
        self.pairs_per_group = self.n_pairs // self.groups
        self.offset = int(offset) % 2
        self.omega = BoundedScalar(omega_max, numel=self.groups)

    def _w(self, dtype: torch.dtype) -> torch.Tensor:
        w = self.omega()
        if self.groups != self.n_pairs:
            w = w.repeat_interleave(self.pairs_per_group)
        return w.to(dtype)

    def _split(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.offset:
            v = torch.roll(v, shifts=-1, dims=-1)
        return v[..., 0::2], v[..., 1::2]

    def _join(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        out = torch.stack((p, q), dim=-1).flatten(-2)
        if self.offset:
            out = torch.roll(out, shifts=1, dims=-1)
        return out

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        w = self._w(v.dtype)
        p, q = self._split(v)
        p = p + w * q
        q = q - w * p
        return self._join(p, q)

    def inverse(self, v: torch.Tensor) -> torch.Tensor:
        w = self._w(v.dtype)
        p, q = self._split(v)
        q = q + w * p
        p = p - w * q
        return self._join(p, q)

    def theta(self) -> torch.Tensor:
        """Rotation angle per pair group (nan where |omega| >= 2)."""
        w = self.omega()
        return torch.acos(torch.clamp(1.0 - w * w / 2.0, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# the block
# --------------------------------------------------------------------------- #
class PolyakLTBlock(nn.Module):
    """One Polyak + Lie-Trotter layer, optionally with adaptive damping / gyro.

    `forward` and `inverse` are exact algebraic inverses of each other (up to
    floating point) whenever the retention is nonzero, which the unit-constrained
    `beta` guarantees.
    """

    def __init__(self, cfg: ModelConfig, ext: YuriiExtConfig, layer_idx: int = 0):
        super().__init__()
        self.ext = ext
        self.no_mlp = bool(ext.no_mlp)

        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        # One velocity norm per block, shared by both substeps -- matches
        # YuriiFormerLieTrotterBlock, and it is read-only here so sharing it
        # costs nothing structurally.
        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias)

        self.beta_attn = ConstrainedScalar(ext.beta_init, "unit")
        self.gamma_attn = ConstrainedScalar(ext.gamma_init, "pos")
        self.eta_attn = ConstrainedScalar(ext.eta_init, "pos")

        self.beta_mlp = ConstrainedScalar(ext.beta_init, "unit")
        self.gamma_mlp = ConstrainedScalar(ext.gamma_init, "pos")
        self.eta_mlp = ConstrainedScalar(ext.eta_init, "pos")

        self.damp_attn = AdaptiveDamping(ext.eps_max) if ext.adaptive else None
        self.damp_mlp = (
            AdaptiveDamping(ext.eps_max) if (ext.adaptive and not self.no_mlp) else None
        )

        offset = (layer_idx % 2) if ext.gyro_alternate else 0
        self.gyro_attn = (
            GyroMixer(cfg.n_embd, ext.omega_max, ext.gyro_groups, offset)
            if ext.gyro
            else None
        )
        self.gyro_mlp = (
            GyroMixer(cfg.n_embd, ext.omega_max, ext.gyro_groups, offset)
            if (ext.gyro and not self.no_mlp)
            else None
        )

    # -- substeps ----------------------------------------------------------- #
    def _substep(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        ln_x: nn.Module,
        oracle: nn.Module,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        eta: torch.Tensor,
        damp: Optional[AdaptiveDamping],
        gyro: Optional[GyroMixer],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = ln_x(x)
        v_in = gyro(v) if gyro is not None else v
        retention = damp(z, beta) if damp is not None else beta
        v_new = retention * v_in + gamma * oracle(z)
        x_new = x + eta * self.ln_v(v_new)
        return x_new, v_new

    def _substep_inverse(
        self,
        x_new: torch.Tensor,
        v_new: torch.Tensor,
        ln_x: nn.Module,
        oracle: nn.Module,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        eta: torch.Tensor,
        damp: Optional[AdaptiveDamping],
        gyro: Optional[GyroMixer],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_new - eta * self.ln_v(v_new)
        z = ln_x(x)
        retention = damp(z, beta) if damp is not None else beta
        v_in = (v_new - gamma * oracle(z)) / retention
        v = gyro.inverse(v_in) if gyro is not None else v_in
        return x, v

    # -- forward / inverse -------------------------------------------------- #
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, v = self._substep(
            x, v, self.ln_x_attn, self.attn,
            self.beta_attn(), self.gamma_attn(), self.eta_attn(),
            self.damp_attn, self.gyro_attn,
        )
        if self.no_mlp:
            return x, v
        return self._substep(
            x, v, self.ln_x_mlp, self.mlp,
            self.beta_mlp(), self.gamma_mlp(), self.eta_mlp(),
            self.damp_mlp, self.gyro_mlp,
        )

    @torch.no_grad()
    def inverse(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover `(x, v)` from the block's output. Verification tool, not a
        training path -- nothing here backprops through it."""
        if not self.no_mlp:
            x, v = self._substep_inverse(
                x, v, self.ln_x_mlp, self.mlp,
                self.beta_mlp(), self.gamma_mlp(), self.eta_mlp(),
                self.damp_mlp, self.gyro_mlp,
            )
        return self._substep_inverse(
            x, v, self.ln_x_attn, self.attn,
            self.beta_attn(), self.gamma_attn(), self.eta_attn(),
            self.damp_attn, self.gyro_attn,
        )

    # -- diagnostics -------------------------------------------------------- #
    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        """Exact per-token log|det J| of the block. Independent of the input --
        that is the point of both extensions."""
        ld = torch.log(self.beta_attn())
        if not self.no_mlp:
            ld = ld + torch.log(self.beta_mlp())
        return float(n_embd) * ld

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        rep = {
            "beta_attn": float(self.beta_attn()),
            "gamma_attn": float(self.gamma_attn()),
            "eta_attn": float(self.eta_attn()),
        }
        if not self.no_mlp:
            rep.update(
                beta_mlp=float(self.beta_mlp()),
                gamma_mlp=float(self.gamma_mlp()),
                eta_mlp=float(self.eta_mlp()),
            )
        if self.damp_attn is not None:
            rep["eps_attn"] = float(self.damp_attn.eps())
        if self.damp_mlp is not None:
            rep["eps_mlp"] = float(self.damp_mlp.eps())
        if self.gyro_attn is not None:
            rep["omega_attn"] = float(self.gyro_attn.omega().abs().max())
            rep["theta_attn"] = float(self.gyro_attn.theta().abs().max())
        if self.gyro_mlp is not None:
            rep["omega_mlp"] = float(self.gyro_mlp.omega().abs().max())
            rep["theta_mlp"] = float(self.gyro_mlp.theta().abs().max())
        return rep


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class YuriiExtModel(nn.Module):
    """Same embedding / readout scaffolding as `model.YuriiFormerModel`, so the
    optimizer grouping, checkpointing and cheap-metric hooks in `train.py` apply
    unchanged. The noise and restart machinery is deliberately absent: this
    family exists to isolate one mechanism at fixed volume, and both of those
    knobs are confounds."""

    def __init__(self, cfg: ModelConfig, ext: Optional[YuriiExtConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.ext = ext if ext is not None else YuriiExtConfig()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        # Separate v0 tables (YuriiFormer Appendix A.1).
        self.use_v0_init = bool(self.ext.use_v0_init)
        self.tok_v0_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_v0_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [PolyakLTBlock(cfg, self.ext, layer_idx=i) for i in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        if self.use_v0_init:
            v = self.drop(self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :])
        else:
            v = torch.zeros_like(x)

        for blk in self.blocks:
            x, v = blk(x, v)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    # -- diagnostics -------------------------------------------------------- #
    @torch.no_grad()
    def log_det_per_token(self) -> float:
        """Total per-token log|det| of the stack. Should be identical across the
        four arms at equal `beta` -- check it, do not assume it."""
        total = 0.0
        for blk in self.blocks:
            total += float(blk.log_det_per_token(self.cfg.n_embd))
        return total

    @torch.no_grad()
    def scalar_report(self) -> List[Dict[str, float]]:
        return [blk.scalar_report() for blk in self.blocks]

    @torch.no_grad()
    def reconstruction_drift(
        self, idx: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> Dict[str, float]:
        """Run the stack forward, then invert it, and report how far the
        round-trip lands from the true `(x_0, v_0)`. Mirrors the reversible
        stack's `reconstruction_drift` so the two families are comparable."""
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x0 = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        v0 = (
            self.tok_v0_emb(idx) + self.pos_v0_emb(pos)[None, :, :]
            if self.use_v0_init
            else torch.zeros_like(x0)
        )
        if dtype is not None:
            x0, v0 = x0.to(dtype), v0.to(dtype)

        x, v = x0, v0
        for blk in self.blocks:
            x, v = blk(x, v)
        for blk in reversed(self.blocks):
            x, v = blk.inverse(x, v)

        scale = max(float(x0.abs().max()), float(v0.abs().max()), 1e-12)
        return {
            "recon_drift_x": float((x - x0).abs().max()),
            "recon_drift_v": float((v - v0).abs().max()),
            "recon_drift_rel": float(
                max(float((x - x0).abs().max()), float((v - v0).abs().max())) / scale
            ),
        }


# Reuse the baseline's exact sampling logic, so train.py's print_sample works.
YuriiExtModel.generate = _generic_generate


def build_yurii_ext(cfg: ModelConfig, ext: Optional[YuriiExtConfig] = None) -> YuriiExtModel:
    return YuriiExtModel(cfg, ext)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _tiny(n_layer: int = 2, n_embd: int = 8, n_head: int = 2, vocab: int = 32):
    return ModelConfig(
        vocab_size=vocab,
        block_size=16,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        attn_impl="manual",
    )


def _check_determinant(ext: YuriiExtConfig, tol: float = 1e-8) -> float:
    """Compare the analytic per-token log|det| against an autograd Jacobian of
    the full (x, v) -> (x', v') block map."""
    torch.manual_seed(0)
    cfg = _tiny(n_layer=1)
    blk = PolyakLTBlock(cfg, ext, layer_idx=0).double().eval()
    # Non-trivial norm affine params: they must not touch the determinant.
    for ln in (blk.ln_x_attn, blk.ln_x_mlp, blk.ln_v):
        ln.weight.data.normal_(1.0, 0.1)
    if ext.adaptive:
        blk.damp_attn.eps.raw.data.fill_(0.7)
        blk.damp_mlp.eps.raw.data.fill_(-0.4)
    if ext.gyro:
        blk.gyro_attn.omega.raw.data.fill_(0.6)
        blk.gyro_mlp.omega.raw.data.fill_(-0.9)

    T, d = 3, cfg.n_embd
    x = torch.randn(1, T, d, dtype=torch.float64)
    v = torch.randn(1, T, d, dtype=torch.float64)

    def f(flat):
        xx, vv = flat[: T * d].view(1, T, d), flat[T * d:].view(1, T, d)
        ox, ov = blk(xx, vv)
        return torch.cat([ox.reshape(-1), ov.reshape(-1)])

    flat = torch.cat([x.reshape(-1), v.reshape(-1)])
    J = torch.autograd.functional.jacobian(f, flat, vectorize=True)
    measured = float(torch.linalg.slogdet(J)[1])
    analytic = float(blk.log_det_per_token(d).detach()) * T
    err = abs(measured - analytic)
    assert err < tol, f"{ext.arm_name()}: logdet {measured:.12f} vs {analytic:.12f}"
    return analytic


def _check_inverse(ext: YuriiExtConfig, tol: float = 1e-9) -> float:
    torch.manual_seed(1)
    cfg = _tiny(n_layer=3)
    model = YuriiExtModel(cfg, ext).double().eval()
    for blk in model.blocks:
        if ext.adaptive:
            blk.damp_attn.eps.raw.data.normal_(0.0, 0.8)
            blk.damp_mlp.eps.raw.data.normal_(0.0, 0.8)
        if ext.gyro:
            blk.gyro_attn.omega.raw.data.normal_(0.0, 0.8)
            blk.gyro_mlp.omega.raw.data.normal_(0.0, 0.8)
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    d = model.reconstruction_drift(idx)
    assert d["recon_drift_rel"] < tol, f"{ext.arm_name()}: {d}"
    return d["recon_drift_rel"]


def _check_init_equivalence() -> None:
    """At rho = 0 every arm must be *bit*-identical to the plain Polyak block:
    exp(0) = 1 and omega = 0 are exact in floating point, so this is not an
    approximate claim."""
    idx = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(3))
    base_logits = None
    for ext in (
        YuriiExtConfig(),
        YuriiExtConfig(adaptive=True),
        YuriiExtConfig(gyro=True),
        YuriiExtConfig(adaptive=True, gyro=True),
    ):
        torch.manual_seed(7)
        model = YuriiExtModel(_tiny(n_layer=3), ext).eval()
        with torch.no_grad():
            logits, _ = model(idx)
        if base_logits is None:
            base_logits = logits
        else:
            assert torch.equal(logits, base_logits), (
                f"{ext.arm_name()} differs from polyak at init "
                f"(max |d| = {float((logits - base_logits).abs().max()):.3e})"
            )


def _check_realized_determinant(d: int = 768) -> None:
    """The determinant is exact in exact arithmetic; this measures what the
    *realized* retention does per token per substep at each precision.

    Reported, not asserted, for bf16: the point is that the adaptive arm is no
    worse than the scalar baseline there (it is markedly better), so the
    fixed-volume premise of the four-arm comparison survives low precision.
    """
    torch.manual_seed(0)
    beta = torch.tensor(0.9)
    damp = AdaptiveDamping(0.5)
    damp.eps.raw.data.fill_(1.0)
    nominal = float(d * beta.log())

    print(f"\nrealized log|det| error per token per substep (d={d}, nominal {nominal:.2f} nats)")
    print(f"{'stream dtype':>13}  {'polyak beta*v':>15}  {'adaptive S*v':>14}")
    with torch.no_grad():
        for dt in (torch.float32, torch.bfloat16):
            z = torch.randn(2, 64, d, dtype=dt)
            v = torch.randn(2, 64, d, dtype=dt)
            s = damp(z, beta)

            def err(eff):
                return float((eff.float().log().sum(-1) - nominal).abs().mean())

            e_pk = err((beta * v) / v.float())
            e_ad = err((s * v) / v.float())
            print(f"{str(dt).replace('torch.', ''):>13}  {e_pk:>15.2e}  {e_ad:>14.2e}")
            if dt is torch.float32:
                assert e_ad < 1e-4 and e_pk < 1e-4, (e_pk, e_ad)


def _check_causality(ext: YuriiExtConfig) -> None:
    torch.manual_seed(4)
    cfg = _tiny(n_layer=2)
    model = YuriiExtModel(cfg, ext).eval()
    if ext.adaptive:
        for blk in model.blocks:
            blk.damp_attn.eps.raw.data.fill_(0.9)
            blk.damp_mlp.eps.raw.data.fill_(0.9)
    T = 6
    pos = torch.arange(T)
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    x = (model.tok_emb(idx) + model.pos_emb(pos)[None]).detach().requires_grad_(True)
    v = (model.tok_v0_emb(idx) + model.pos_v0_emb(pos)[None]).detach().requires_grad_(True)
    h, w = x, v
    for blk in model.blocks:
        h, w = blk(h, w)
    t = 2
    h[0, t].sum().backward()
    future = max(
        float(x.grad[0, t + 1:].abs().max()), float(v.grad[0, t + 1:].abs().max())
    )
    assert future == 0.0, f"{ext.arm_name()}: leak from the future, max |g| = {future:.3e}"


def _self_test() -> None:
    torch.set_printoptions(precision=6)
    arms = [
        YuriiExtConfig(),
        YuriiExtConfig(adaptive=True),
        YuriiExtConfig(gyro=True),
        YuriiExtConfig(adaptive=True, gyro=True),
        YuriiExtConfig(adaptive=True, gyro=True, gyro_groups=4, gyro_alternate=True),
    ]
    print(f"{'arm':<24} {'logdet/token':>14} {'recon drift':>12}")
    dets = []
    for ext in arms:
        ld = _check_determinant(ext)
        dr = _check_inverse(ext)
        _check_causality(ext)
        dets.append(ld)
        name = ext.arm_name() + (f" x{ext.gyro_groups}" if ext.gyro_groups > 1 else "")
        print(f"{name:<24} {ld:>14.9f} {dr:>12.2e}")
    assert max(dets) - min(dets) < 1e-12, f"arms disagree on the determinant: {dets}"
    _check_init_equivalence()
    _check_realized_determinant()

    # Parameter accounting: the extensions must be scalars, not capacity.
    cfg = _tiny(n_layer=6, n_embd=64, n_head=4, vocab=256)
    counts = {}
    for ext in arms[:4]:
        m = YuriiExtModel(cfg, ext)
        counts[ext.arm_name()] = sum(p.numel() for p in m.parameters())
    base = counts["polyak"]
    print()
    for name, n in counts.items():
        print(f"{name:<24} {n:>10,} params  (+{n - base} vs polyak, 6 layers)")

    print("\nall checks passed: determinant matches autograd, inverse is exact,")
    print("all arms are bit-identical to polyak at init, no leakage from the future.")


if __name__ == "__main__":
    _self_test()
