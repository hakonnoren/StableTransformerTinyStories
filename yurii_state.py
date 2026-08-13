"""Four state-structure variants of the YuriiFormer block.

`yurii_ext.py` perturbs the Polyak recurrence while holding the state `(X, V)`
and the layer determinant fixed. This file changes the *state structure*
itself. Every variant keeps: one attention + one MLP call per oracle stage,
O(L) new trainable scalars, an explicit inverse, and determinant control
through scalar or per-token diagonal factors only.

    variant          state          A/M per block   new params    det
    ---------------  -------------  --------------  ------------  --------------
    thermostat       (X, V, r)      1 / 1           O(L) scalars  (b_A b_M)^nd *
    dual_momentum    (X, V_A, V_M)  1 / 1           O(L) scalars  (b_A b_M)^nd
    two_stream       (X, Z)         1 / 1           O(L*d)        (prod a_i b_i)^n
    multirate        (X, V)         1 / k           O(L) scalars  (b_A prod b_j)^nd

    * in the default channel mode; input-dependent in scalar mode.

With the default `beta_init = 0.9` all three of thermostat, dual_momentum and
two_stream have *exactly* the same per-layer log|det| as `yurii_ext`'s polyak
baseline (`n_embd * L * log 0.81`), so they slot straight into the same
fixed-volume comparison. `multirate` matches too once you count its k substeps.

Five deliberate departures from the proposals as written. The first three are
because the proposal as stated does not survive contact with a causal LM or
with autograd; the last two are improvements. All are load-bearing -- read them
before interpreting a run. Each has a `StateConfig` flag that reverts it.

1. THE THERMOSTAT KEEPS A BASELINE beta. The proposal has `V <- e^{-eps
   zeta(r)} V`, i.e. no damping at all when `r = 0`. Undamped momentum
   accumulates every oracle output down the stack. Here the retention is
   `beta * e^{-eps zeta(r)}` with `beta` learned and initialized at 0.9, so the
   thermostat *modulates* a Polyak block rather than replacing its contraction,
   the r = 0 state reduces exactly to Polyak, and there is a guaranteed
   contraction floor. Set `beta_init` near 1 to recover the original.

2. NEVER TOKEN-MEAN CENTERING OF zeta. The proposal suggests subtracting the
   *token* mean of `zeta(r)` to pin the layer determinant. In a causal LM that
   is a **future leak**: token t's scaling would depend on token t+1's r. There
   is no causal-safe variant either -- the only other axis is the batch, which
   leaks across sequences and changes behavior at inference. Channel centering
   is the fix (see 4). In scalar mode, where there is no channel axis, the
   determinant simply stays input-dependent and is instead made exactly
   *observable*: `block.last_logdet_per_token` after every forward, averaged
   into `last_logdet_delta` for logging or a volume penalty.

3. THE THERMOSTAT STARTS AT eps, rho != 0. `eps` and `rho` multiply: `s`
   depends on `rho` only through `eps * zeta(r)`, so at `eps = 0` the gradient
   w.r.t. `rho` is exactly zero, and at `r = 0` the gradient w.r.t. `eps` is
   exactly zero (`zeta(0) = 0`). Initialize both at 0 -- the natural choice,
   and the one the other file uses -- and the thermostat is provably dead for
   all of training. Both therefore start nonzero (`eps_init`, `rho_init`).
   Since `r` is still initialized to 0, the block is *numerically* within
   O(rho) of Polyak at step 0 while both gradients are live.

4. THE THERMOSTAT IS CHANNEL-WISE BY DEFAULT (`thermostat_channels`). A scalar
   r per token cannot see anisotropy, and per-channel scale drift is exactly
   what velocity LayerNorm exists to fix -- so the scalar version is not a
   like-for-like replacement for the thing it is meant to replace. Going d-wide
   also makes the determinant exactly prescribable *causally*, because the
   channel mean is a reduction over one token's own features. Read it as
   `yurii_ext.AdaptiveDamping` with an integrator state. What it regulates is
   each channel's energy *relative to its token's mean* rather than raw `v^2`;
   `_energy` explains why the raw version is 4 orders of magnitude too weak to
   train.

5. TWO_STREAM SCALES PER CHANNEL (`channel_scaling`). This repo's own reversible
   experiments found the network learns nontrivial channel-dependent contraction
   when allowed to, and that free scaling beat scalar scaling -- so scalar a/b
   would omit the axis those runs said matters. d params per layer instead of 1,
   and identical determinant at uniform init.

Also `multirate` gained an optional invertible attention memory
(`attn_memory`); see `MultirateBlock` for why the obvious cached-extrapolation
version cannot be inverted.

`drift_norm` (read-only `LN(V)` inside the drift shear, as in
`yurii_ext.PolyakLTBlock`) defaults per variant: OFF for the thermostat, whose
entire hypothesis is that dynamical regulation replaces velocity LayerNorm; ON
for dual_momentum and multirate, so the only difference from the polyak
baseline is the state structure. two_stream has no velocity and ignores it.

Run `python yurii_state.py` to check, for every variant: the analytic
determinant against an fp64 autograd Jacobian of the real block map, the exact
inverse, no leakage from the future, and the oracle-call accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from yurii_ext import BoundedScalar

VARIANTS = ("thermostat", "dual_momentum", "two_stream", "multirate")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class StateConfig:
    variant: str = "thermostat"

    # shared update-rule inits
    beta_init: float = 0.9          # retention / stream scaling, unit-constrained
    gamma_init: float = 1.0         # oracle gain, positive-constrained
    eta_init: float = 1.0           # drift step, positive-constrained
    drift_norm: Optional[bool] = None   # None => per-variant default (see module docstring)
    use_v0_init: bool = True        # separate token+pos tables for the auxiliary streams
    share_v0: bool = True           # one v0 table for ALL auxiliary streams, not one each

    # thermostat
    thermostat_channels: bool = True  # d-wide r with channel-centered zeta (see below)
    zeta_max: float = 1.0           # |zeta| bound; retention in beta*[e^-eps*zeta_max, e^+eps*zeta_max]
    eps_max: float = 0.5
    eps_init: float = 0.25          # MUST be nonzero, else rho has zero gradient forever
    rho_max: float = 0.1
    rho_init: float = 0.01          # MUST be nonzero, else eps has zero gradient forever
    tau_init: float = 1.0           # target energy, learned

    # two_stream
    swap: bool = True               # exchange the streams' roles every layer
    readout: str = "x"              # "x" (stream 0) or "sum" (both streams)
    channel_scaling: bool = True    # per-channel a, b vectors instead of scalars

    # multirate
    n_mlp: int = 2                  # MLP substeps per attention call
    share_mlp: bool = False         # reuse one MLP across the k substeps
    attn_memory: bool = False       # carry a decaying attention field A into every substep
    attn_mem_init: float = 0.9      # decay c of that field, unit-constrained
    attn_mem_delta_init: float = 0.1  # how strongly each MLP substep draws on A

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}; got {self.variant!r}")
        if self.readout not in ("x", "sum"):
            raise ValueError(f"readout must be 'x' or 'sum'; got {self.readout!r}")
        if self.variant == "multirate" and self.n_mlp < 1:
            raise ValueError(f"n_mlp must be >= 1; got {self.n_mlp}")

    def uses_drift_norm(self) -> bool:
        if self.drift_norm is not None:
            return bool(self.drift_norm)
        return self.variant in ("dual_momentum", "multirate")

    def name(self) -> str:
        if self.variant == "multirate":
            return (f"multirate_1a{self.n_mlp}m"
                    + ("_shared" if self.share_mlp else "")
                    + ("_amem" if self.attn_memory else ""))
        if self.variant == "two_stream":
            return ("two_stream"
                    + ("" if self.swap else "_noswap")
                    + ("" if self.channel_scaling else "_scalarab"))
        if self.variant == "thermostat":
            return "thermostat" + ("_ch" if self.thermostat_channels else "_scalar")
        return self.variant


# --------------------------------------------------------------------------- #
# stream plumbing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stream:
    """One slot of the block's state.

    `init` is how the model seeds it: "main" = the token+pos embedding sum,
    "embed" = its own separate v0 token+pos tables (YuriiFormer App. A.1),
    "zeros" = a zero tensor (used for the thermostat's per-token scalar).
    `width` is "d" (n_embd) or "1" (one scalar per token).
    """
    name: str
    width: str
    init: str


State = Tuple[torch.Tensor, ...]


class ConstrainedVector(nn.Module):
    """Per-channel version of `ConstrainedScalar`: a length-d vector mapped to
    (0,1) by sigmoid or to (0,inf) by softplus, initialized uniformly.

    Uniform init means a channel-wise arm has *exactly* the same determinant at
    step 0 as its scalar counterpart; the channels only differentiate once
    gradients pull them apart. The parameter is `raw`, so it joins the
    update-rule scalar group (wd 0, 5x lr) rather than the reversible
    gamma/alpha group -- keeping the optimizer treatment identical across arms,
    which would otherwise be a confound.
    """

    def __init__(self, n: int, init: float, kind: str = "unit"):
        super().__init__()
        self.kind = kind
        if kind == "unit":
            init = min(max(init, 1e-4), 1 - 1e-4)
            p = math.log(init / (1 - init))
        elif kind == "pos":
            p = math.log(math.expm1(max(init, 1e-8)))
        else:
            raise ValueError("kind must be 'unit' or 'pos'")
        self.raw = nn.Parameter(torch.full((n,), p, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) if self.kind == "unit" else F.softplus(self.raw)


class StateBlock(nn.Module):
    """Base class: a block is an invertible map on a tuple of streams."""

    SPEC: Tuple[Stream, ...] = ()

    @classmethod
    def spec_for(cls, scfg: "StateConfig") -> Tuple[Stream, ...]:
        """State layout, which for some variants depends on the config."""
        return cls.SPEC

    def forward(self, state: State) -> State:      # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def inverse(self, state: State) -> State:      # pragma: no cover - interface
        raise NotImplementedError

    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        """Static (input-independent) part of the per-token log|det|."""
        raise NotImplementedError

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        return {}

    # -- shared helpers ---------------------------------------------------- #
    def _drift(self, x: torch.Tensor, eta: torch.Tensor, v: torch.Tensor,
               ln: Optional[nn.Module]) -> torch.Tensor:
        """x + eta * N(v). A shear in x: unit Jacobian in x whatever N is, so
        `ln`'s affine parameters stay learnable without touching the volume."""
        return x + eta * (ln(v) if ln is not None else v)


# --------------------------------------------------------------------------- #
# variant 1: thermostat
# --------------------------------------------------------------------------- #
class ThermostatBlock(StateBlock):
    """State (X, V, r): a retained thermostat state regulates momentum energy.

    Attention half (channel mode, the default)::

        V1 = V + gamma_A Attn(LN X)
        r1 = r + rho_A (V1^2 - tau_A)                    elementwise, r is d-wide
        zt = center_channel(zeta(r1))                    zeta(r) = zeta_max tanh r
        V2 = beta_A exp(-eps_A zt) V1
        X1 = X + eta_A N(V2)

    then the same around the MLP. Hot coordinates push `r` up and get more
    friction on the *next* substep; cold ones get less, and `zeta` may go
    negative, which pumps energy back in. Nothing is projected away, which is
    what makes it invertible where velocity LayerNorm is not.

    CHANNEL vs SCALAR. `thermostat_channels=False` is the original proposal: one
    scalar r per token driven by `|V|^2/d`. Two reasons the default is the
    d-wide version instead.

    * What LayerNorm actually fixes is *per-channel* scale drift. A single
      scalar per token cannot see anisotropy, let alone correct it, so a scalar
      thermostat is not a like-for-like replacement for velocity LN.
    * Going d-wide makes the determinant exactly prescribable, causally. The
      channel mean of `zeta` is a reduction over the *same token's* features, so
      subtracting it leaks nothing (the token mean would leak the future -- see
      note 2 in the module docstring). With `sum_i zt_i = 0` exactly,
      `prod_i s_i = beta^d` exactly, and the block's log|det| is
      `d(log beta_A + log beta_M)` independent of the input -- the same value as
      the polyak baseline, so the arm is volume-matched by construction rather
      than only observably.

    Read the channel mode as `yurii_ext.AdaptiveDamping` with an integrator
    state: same channel-centered exponential retention, but driven by an
    accumulated energy error instead of an instantaneous function of the
    activations. Scalar mode keeps an input-dependent determinant, exposed via
    `last_logdet_per_token`.

    Jacobian, per substep, in the order the code applies it: the `V` kick is a
    shear (det 1), the `r` update is a shear (det 1), and the scaling is block
    triangular in `(V1, r1)` -- `r1` does not depend on `V1` *after* the r
    update, so the lower-left block vanishes and only `diag(s)` survives.
    """

    SPEC = (Stream("x", "d", "main"), Stream("v", "d", "embed"), Stream("r", "1", "zeros"))

    @classmethod
    def spec_for(cls, scfg: "StateConfig") -> Tuple[Stream, ...]:
        w = "d" if scfg.thermostat_channels else "1"
        return (Stream("x", "d", "main"), Stream("v", "d", "embed"), Stream("r", w, "zeros"))

    def __init__(self, cfg: ModelConfig, scfg: StateConfig, layer_idx: int = 0):
        super().__init__()
        self.scfg = scfg
        self.n_embd = cfg.n_embd
        self.zeta_max = float(scfg.zeta_max)
        self.channels = bool(scfg.thermostat_channels)

        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias) if scfg.uses_drift_norm() else None

        self.beta_attn = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_attn = ConstrainedScalar(scfg.gamma_init, "pos")
        self.eta_attn = ConstrainedScalar(scfg.eta_init, "pos")
        self.beta_mlp = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_mlp = ConstrainedScalar(scfg.gamma_init, "pos")
        self.eta_mlp = ConstrainedScalar(scfg.eta_init, "pos")

        self.eps_attn = BoundedScalar(scfg.eps_max, init=scfg.eps_init)
        self.eps_mlp = BoundedScalar(scfg.eps_max, init=scfg.eps_init)
        self.rho_attn = BoundedScalar(scfg.rho_max, init=scfg.rho_init)
        self.rho_mlp = BoundedScalar(scfg.rho_max, init=scfg.rho_init)
        self.tau_attn = ConstrainedScalar(scfg.tau_init, "pos")
        self.tau_mlp = ConstrainedScalar(scfg.tau_init, "pos")

        # Filled by forward: exact per-token log|det| of this block, detached.
        self.last_logdet_per_token: Optional[torch.Tensor] = None

    def _zeta(self, r: torch.Tensor) -> torch.Tensor:
        return self.zeta_max * torch.tanh(r)

    def _energy(self, v: torch.Tensor) -> torch.Tensor:
        """The quantity the thermostat regulates. Either way the reduction is
        within a single token, which is what keeps the thermostat causal.

        Scalar mode: `|v|^2/d`, the proposal as written.

        Channel mode: the energy of each channel *relative to the token's mean*,
        `v_i^2 / mean_j v_j^2`. Raw `v_i^2` is wrong here for a reason worth
        recording. Channel-centering the resulting zeta deletes any component
        that is common to all channels -- including the whole `-tau` term, which
        makes tau inert -- so the only surviving drive is the channel-to-channel
        *spread* of `v^2`. At init that is ~rho * var(v^2) ~ 1e-6 in absolute
        units, and the measured eps/rho gradients came out 4 orders of magnitude
        below the scalar mode's: alive in principle, dead in 10k steps. The
        ratio is scale-invariant, so the drive is O(1) whatever the velocity
        scale, and `tau` recovers its meaning as "each channel should carry the
        token's average energy".
        """
        if not self.channels:
            return v.pow(2).mean(dim=-1, keepdim=True)
        e = v.pow(2)
        return e / (e.mean(dim=-1, keepdim=True) + 1e-8)

    def _scale(self, r1: torch.Tensor, beta: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        z = self._zeta(r1)
        if self.channels:
            # Channel mean, not token mean: a reduction over one token's own
            # features leaks nothing. sum_i z_i = 0 => prod_i s_i = beta^d exactly.
            z = z - z.mean(dim=-1, keepdim=True)
        return beta * torch.exp(-eps * z)

    def _half(self, x, v, r, ln_x, oracle, beta, gamma, eta, eps, rho, tau):
        v1 = v + gamma * oracle(ln_x(x))
        r1 = r + rho * (self._energy(v1) - tau)
        scale = self._scale(r1, beta, eps)
        v2 = scale * v1
        x1 = self._drift(x, eta, v2, self.ln_v)
        return x1, v2, r1, scale

    def _half_inverse(self, x1, v2, r1, ln_x, oracle, beta, gamma, eta, eps, rho, tau):
        x = x1 - eta * (self.ln_v(v2) if self.ln_v is not None else v2)
        v1 = v2 / self._scale(r1, beta, eps)
        r = r1 - rho * (self._energy(v1) - tau)
        v = v1 - gamma * oracle(ln_x(x))
        return x, v, r

    def forward(self, state: State) -> State:
        x, v, r = state
        x, v, r, s_a = self._half(
            x, v, r, self.ln_x_attn, self.attn,
            self.beta_attn(), self.gamma_attn(), self.eta_attn(),
            self.eps_attn(), self.rho_attn(), self.tau_attn(),
        )
        x, v, r, s_m = self._half(
            x, v, r, self.ln_x_mlp, self.mlp,
            self.beta_mlp(), self.gamma_mlp(), self.eta_mlp(),
            self.eps_mlp(), self.rho_mlp(), self.tau_mlp(),
        )
        # Exact per-token log|det| of this block. In channel mode the two terms
        # sum to d*(log beta_A + log beta_M) up to fp error, by construction; in
        # scalar mode this is genuinely input-dependent.
        ld = torch.log(s_a) + torch.log(s_m)
        self.last_logdet_per_token = (
            ld.sum(dim=-1) if self.channels else float(self.n_embd) * ld.squeeze(-1)
        ).detach()
        return x, v, r

    @torch.no_grad()
    def inverse(self, state: State) -> State:
        x, v, r = state
        x, v, r = self._half_inverse(
            x, v, r, self.ln_x_mlp, self.mlp,
            self.beta_mlp(), self.gamma_mlp(), self.eta_mlp(),
            self.eps_mlp(), self.rho_mlp(), self.tau_mlp(),
        )
        return self._half_inverse(
            x, v, r, self.ln_x_attn, self.attn,
            self.beta_attn(), self.gamma_attn(), self.eta_attn(),
            self.eps_attn(), self.rho_attn(), self.tau_attn(),
        )

    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        """In channel mode this is the *exact* per-token log|det|: channel
        centering pins the eps*zeta term to zero. In scalar mode it is only the
        beta part, and the rest lives in `last_logdet_per_token`."""
        return float(n_embd) * (torch.log(self.beta_attn()) + torch.log(self.beta_mlp()))

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        rep = {
            "beta_attn": float(self.beta_attn()), "beta_mlp": float(self.beta_mlp()),
            "gamma_attn": float(self.gamma_attn()), "gamma_mlp": float(self.gamma_mlp()),
            "eta_attn": float(self.eta_attn()), "eta_mlp": float(self.eta_mlp()),
            "eps_attn": float(self.eps_attn()), "eps_mlp": float(self.eps_mlp()),
            "rho_attn": float(self.rho_attn()), "rho_mlp": float(self.rho_mlp()),
            "tau_attn": float(self.tau_attn()), "tau_mlp": float(self.tau_mlp()),
        }
        if self.last_logdet_per_token is not None:
            rep["logdet_tok"] = float(self.last_logdet_per_token.mean())
        return rep


# --------------------------------------------------------------------------- #
# variant 2: separate attention / MLP momentum
# --------------------------------------------------------------------------- #
class DualMomentumBlock(StateBlock):
    """State (X, V_A, V_M): attention and MLP get their own memory.

        V_A' = beta_A V_A + gamma_A Attn(LN X)
        X'   = X + eta_A N(V_A')
        V_M' = beta_M V_M + gamma_M MLP(LN X')
        X''  = X' + eta_M N(V_M')

    One attention, one MLP, no new dense operations. The point is that with a
    shared velocity the two oracle histories superpose --
    `V ~ sum_j beta^{l-j}(A_j + M_j)` -- and can cancel; with split memories
    they decay at independently learned rates `beta_A != beta_M`, so global
    token-mixing information and local featurewise corrections can have
    different lifetimes in depth.

    Determinant is the same `(beta_A beta_M)^{nd}` as the polyak baseline, so
    this is directly comparable to it at equal volume; the state grows 2d -> 3d,
    which is the real cost.
    """

    SPEC = (Stream("x", "d", "main"), Stream("v_a", "d", "embed"), Stream("v_m", "d", "embed"))

    def __init__(self, cfg: ModelConfig, scfg: StateConfig, layer_idx: int = 0):
        super().__init__()
        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x_mlp = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        # Separate velocity norms: the two streams are different objects, unlike
        # the two substeps of a single-velocity block.
        use_ln = scfg.uses_drift_norm()
        self.ln_v_a = LayerNorm(cfg.n_embd, bias=cfg.bias) if use_ln else None
        self.ln_v_m = LayerNorm(cfg.n_embd, bias=cfg.bias) if use_ln else None

        self.beta_attn = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_attn = ConstrainedScalar(scfg.gamma_init, "pos")
        self.eta_attn = ConstrainedScalar(scfg.eta_init, "pos")
        self.beta_mlp = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_mlp = ConstrainedScalar(scfg.gamma_init, "pos")
        self.eta_mlp = ConstrainedScalar(scfg.eta_init, "pos")

    def forward(self, state: State) -> State:
        x, v_a, v_m = state
        v_a = self.beta_attn() * v_a + self.gamma_attn() * self.attn(self.ln_x_attn(x))
        x = self._drift(x, self.eta_attn(), v_a, self.ln_v_a)
        v_m = self.beta_mlp() * v_m + self.gamma_mlp() * self.mlp(self.ln_x_mlp(x))
        x = self._drift(x, self.eta_mlp(), v_m, self.ln_v_m)
        return x, v_a, v_m

    @torch.no_grad()
    def inverse(self, state: State) -> State:
        x, v_a, v_m = state
        x = x - self.eta_mlp() * (self.ln_v_m(v_m) if self.ln_v_m is not None else v_m)
        v_m = (v_m - self.gamma_mlp() * self.mlp(self.ln_x_mlp(x))) / self.beta_mlp()
        x = x - self.eta_attn() * (self.ln_v_a(v_a) if self.ln_v_a is not None else v_a)
        v_a = (v_a - self.gamma_attn() * self.attn(self.ln_x_attn(x))) / self.beta_attn()
        return x, v_a, v_m

    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        return float(n_embd) * (torch.log(self.beta_attn()) + torch.log(self.beta_mlp()))

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        return {
            "beta_attn": float(self.beta_attn()), "beta_mlp": float(self.beta_mlp()),
            "gamma_attn": float(self.gamma_attn()), "gamma_mlp": float(self.gamma_mlp()),
            "eta_attn": float(self.eta_attn()), "eta_mlp": float(self.eta_mlp()),
        }


# --------------------------------------------------------------------------- #
# variant 3: alternating two-stream
# --------------------------------------------------------------------------- #
class TwoStreamBlock(StateBlock):
    """State (X, Z): two equally representational streams, no velocity.

        X' = a X + gamma_A Attn(LN Z)
        Z' = b Z + gamma_M MLP(LN X')
        (X, Z) <- (Z', X')                       # free swap, |det| = 1

    Without the swap each stream is permanently specialized (Z always feeds
    attention, X always feeds the MLP). With it, both streams see both oracles
    on alternating layers, so the depth pattern is still A -> M -> A -> M but
    information rides two coupled trajectories instead of one residual stream.

    This is the ablation that asks whether YuriiFormer's gain is *momentum* or
    merely *a structured, invertible two-stream state*: same oracle budget, same
    2d state, same `(a b)^nd = (beta_A beta_M)^nd` determinant at equal init --
    but neither stream is a velocity.
    """

    SPEC = (Stream("x", "d", "main"), Stream("z", "d", "embed"))

    def __init__(self, cfg: ModelConfig, scfg: StateConfig, layer_idx: int = 0):
        super().__init__()
        self.swap = bool(scfg.swap)
        self.ln_z = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_x = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

        # Per-channel by default. The reversible experiments in this repo found
        # the network learns nontrivial channel-dependent contraction when given
        # the freedom, and that the free-scaling regime beat the scalar one --
        # so a scalar a/b here would be leaving the interesting axis out. Costs
        # d params per layer instead of 1, and at uniform init the determinant
        # is identical to the scalar version's.
        if scfg.channel_scaling:
            self.a = ConstrainedVector(cfg.n_embd, scfg.beta_init, "unit")
            self.b = ConstrainedVector(cfg.n_embd, scfg.beta_init, "unit")
        else:
            self.a = ConstrainedScalar(scfg.beta_init, "unit")
            self.b = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_attn = ConstrainedScalar(scfg.gamma_init, "pos")
        self.gamma_mlp = ConstrainedScalar(scfg.gamma_init, "pos")

    def forward(self, state: State) -> State:
        x, z = state
        x = self.a() * x + self.gamma_attn() * self.attn(self.ln_z(z))
        z = self.b() * z + self.gamma_mlp() * self.mlp(self.ln_x(x))
        return (z, x) if self.swap else (x, z)

    @torch.no_grad()
    def inverse(self, state: State) -> State:
        x, z = (state[1], state[0]) if self.swap else state
        z = (z - self.gamma_mlp() * self.mlp(self.ln_x(x))) / self.b()
        x = (x - self.gamma_attn() * self.attn(self.ln_z(z))) / self.a()
        return x, z

    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        # sum over channels when a/b are vectors; n_embd * value when scalar.
        la, lb = torch.log(self.a()), torch.log(self.b())
        if la.ndim == 0:
            return float(n_embd) * (la + lb)
        return la.sum() + lb.sum()

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        a, b = self.a(), self.b()
        return {
            "a_mean": float(a.mean()), "b_mean": float(b.mean()),
            "a_std": float(a.std()) if a.ndim else 0.0,
            "b_std": float(b.std()) if b.ndim else 0.0,
            "gamma_attn": float(self.gamma_attn()), "gamma_mlp": float(self.gamma_mlp()),
        }


# --------------------------------------------------------------------------- #
# variant 4: multirate
# --------------------------------------------------------------------------- #
class MultirateBlock(StateBlock):
    """State (X, V), one attention call per k MLP substeps.

        V <- beta_A V + gamma_A Attn(LN X);   X <- X + eta_A N(V)
        for j in 1..k:
            V <- beta_j V + gamma_j MLP_j(LN X);  X <- X + eta_j N(V)

    A macroblock costs `A + k M` where k ordinary blocks cost `k A + k M`, so at
    matched MLP count the model holds k times fewer attention modules and makes
    k times fewer attention calls -- parameters *fall*, and at long context the
    saving is the quadratic term. The hypothesis: token mixing is the slow field
    and does not need re-sampling at every tokenwise refinement.

    Note `n_layer` counts MACROBLOCKS here. To match a 12-layer baseline's MLP
    count with k=2, use `--n_layer 6`: 6 attention calls, 12 MLP calls.

    ATTENTION MEMORY (`attn_memory=True`). Plain skipping leaves the MLP
    substeps with no token mixing at all. The numerical-integration move for a
    slow field is not "sample it less" but "reuse the last sample", so this adds
    a third stream `A` carrying a decaying attention field that every substep
    can draw on::

        A <- c A + Attn(LN X)                     once per macroblock
        V <- beta_A V + gamma_A A;  X <- X + eta_A N(V)
        for j: V <- beta_j V + gamma_j MLP_j(LN X) + delta_j A;  X <- X + eta_j N(V)

    The obvious implementation -- cache `Attn(LN X)` in a local and add it into
    each substep -- is NOT invertible, and it is worth being precise about why:
    inverting substep j needs the cached value, but that value is only
    recomputable from the macroblock's *input* X, which is exactly what you are
    still solving for. The dependency is circular. Making `A` a first-class
    stream with its own invertible update `A <- cA + Attn(LN X)` breaks the
    cycle: during the reverse pass `A` is simply carried, the substeps invert
    against it, and only once `X` is recovered is `A_old = (A - Attn(LN X))/c`
    undone. Determinant becomes `(c beta_A prod_j beta_j)^{nd}` -- still exact.

    Cost: a third d-wide stream and one extra scalar per substep, no extra
    oracle calls.
    """

    SPEC = (Stream("x", "d", "main"), Stream("v", "d", "embed"))

    @classmethod
    def spec_for(cls, scfg: "StateConfig") -> Tuple[Stream, ...]:
        base = (Stream("x", "d", "main"), Stream("v", "d", "embed"))
        return base + (Stream("a", "d", "zeros"),) if scfg.attn_memory else base

    def __init__(self, cfg: ModelConfig, scfg: StateConfig, layer_idx: int = 0):
        super().__init__()
        self.k = int(scfg.n_mlp)
        self.share_mlp = bool(scfg.share_mlp)
        self.attn_memory = bool(scfg.attn_memory)

        self.ln_x_attn = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_v = LayerNorm(cfg.n_embd, bias=cfg.bias) if scfg.uses_drift_norm() else None

        if self.attn_memory:
            self.c_mem = ConstrainedScalar(scfg.attn_mem_init, "unit")
            self.delta_mem = nn.ModuleList(
                [ConstrainedScalar(scfg.attn_mem_delta_init, "pos") for _ in range(self.k)]
            )

        n_mlp_modules = 1 if self.share_mlp else self.k
        self.mlp = nn.ModuleList([MLP(cfg) for _ in range(n_mlp_modules)])
        self.ln_x_mlp = nn.ModuleList([LayerNorm(cfg.n_embd, bias=cfg.bias) for _ in range(self.k)])

        self.beta_attn = ConstrainedScalar(scfg.beta_init, "unit")
        self.gamma_attn = ConstrainedScalar(scfg.gamma_init, "pos")
        self.eta_attn = ConstrainedScalar(scfg.eta_init, "pos")
        self.beta_mlp = nn.ModuleList([ConstrainedScalar(scfg.beta_init, "unit") for _ in range(self.k)])
        self.gamma_mlp = nn.ModuleList([ConstrainedScalar(scfg.gamma_init, "pos") for _ in range(self.k)])
        self.eta_mlp = nn.ModuleList([ConstrainedScalar(scfg.eta_init, "pos") for _ in range(self.k)])

    def _mlp(self, j: int) -> nn.Module:
        return self.mlp[0] if self.share_mlp else self.mlp[j]

    def forward(self, state: State) -> State:
        if self.attn_memory:
            x, v, a = state
            a = self.c_mem() * a + self.attn(self.ln_x_attn(x))
            v = self.beta_attn() * v + self.gamma_attn() * a
        else:
            x, v = state
            v = self.beta_attn() * v + self.gamma_attn() * self.attn(self.ln_x_attn(x))
        x = self._drift(x, self.eta_attn(), v, self.ln_v)
        for j in range(self.k):
            v = self.beta_mlp[j]() * v + self.gamma_mlp[j]() * self._mlp(j)(self.ln_x_mlp[j](x))
            if self.attn_memory:
                v = v + self.delta_mem[j]() * a
            x = self._drift(x, self.eta_mlp[j](), v, self.ln_v)
        return (x, v, a) if self.attn_memory else (x, v)

    @torch.no_grad()
    def inverse(self, state: State) -> State:
        a = state[2] if self.attn_memory else None
        x, v = state[0], state[1]
        for j in reversed(range(self.k)):
            x = x - self.eta_mlp[j]() * (self.ln_v(v) if self.ln_v is not None else v)
            if self.attn_memory:
                v = v - self.delta_mem[j]() * a
            v = (v - self.gamma_mlp[j]() * self._mlp(j)(self.ln_x_mlp[j](x))) / self.beta_mlp[j]()
        x = x - self.eta_attn() * (self.ln_v(v) if self.ln_v is not None else v)
        if self.attn_memory:
            v = (v - self.gamma_attn() * a) / self.beta_attn()
            # Only now, with x recovered, can the memory update be undone.
            a = (a - self.attn(self.ln_x_attn(x))) / self.c_mem()
            return x, v, a
        v = (v - self.gamma_attn() * self.attn(self.ln_x_attn(x))) / self.beta_attn()
        return x, v

    def log_det_per_token(self, n_embd: int) -> torch.Tensor:
        ld = torch.log(self.beta_attn())
        for j in range(self.k):
            ld = ld + torch.log(self.beta_mlp[j]())
        if self.attn_memory:
            ld = ld + torch.log(self.c_mem())
        return float(n_embd) * ld

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, float]:
        rep = {"beta_attn": float(self.beta_attn()), "gamma_attn": float(self.gamma_attn()),
               "eta_attn": float(self.eta_attn())}
        for j in range(self.k):
            rep[f"beta_mlp{j}"] = float(self.beta_mlp[j]())
            rep[f"gamma_mlp{j}"] = float(self.gamma_mlp[j]())
        return rep


_BLOCKS = {
    "thermostat": ThermostatBlock,
    "dual_momentum": DualMomentumBlock,
    "two_stream": TwoStreamBlock,
    "multirate": MultirateBlock,
}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class YuriiStateModel(nn.Module):
    """Shared scaffolding for all four variants.

    Embeddings, tying, init and the optimizer-visible parameter names match
    `model.YuriiFormerModel` / `yurii_ext.YuriiExtModel`, so `train.py`'s
    parameter grouping, checkpointing and cheap-metric hooks apply unchanged.
    Auxiliary streams get their own `tok_v0_emb_*` / `pos_v0_emb_*` tables
    (App. A.1), whose names keep them in the weight-decay-0.1 embedding group.
    """

    def __init__(self, cfg: ModelConfig, scfg: Optional[StateConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.scfg = scfg if scfg is not None else StateConfig()
        block_cls = _BLOCKS[self.scfg.variant]
        self.spec: Tuple[Stream, ...] = block_cls.spec_for(self.scfg)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.use_v0_init = bool(self.scfg.use_v0_init)
        # A v0 token table is vocab*n_embd = 38.6M parameters at d=768. Giving
        # dual_momentum one per velocity stream would put it +31% over every
        # other arm and wreck parameter parity, so by default all auxiliary
        # streams share one table: V_A and V_M start identical and diverge
        # through the dynamics, which is the cleaner control anyway.
        aux = [s.name for s in self.spec if s.init == "embed"]
        self.share_v0 = bool(self.scfg.share_v0) and len(aux) > 1
        keys = ["shared"] if self.share_v0 else aux
        self.tok_v0_emb = nn.ModuleDict({n: nn.Embedding(cfg.vocab_size, cfg.n_embd) for n in keys})
        self.pos_v0_emb = nn.ModuleDict({n: nn.Embedding(cfg.block_size, cfg.n_embd) for n in keys})
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [block_cls(cfg, self.scfg, layer_idx=i) for i in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.last_logdet_delta = 0.0     # thermostat only; mean input-dependent term
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # -- state construction ------------------------------------------------- #
    def init_state(self, idx: torch.Tensor, drop: bool = True) -> State:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        main = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        if drop:
            main = self.drop(main)
        out: List[torch.Tensor] = []
        for s in self.spec:
            if s.init == "main":
                out.append(main)
            elif s.init == "embed":
                if self.use_v0_init:
                    k = "shared" if self.share_v0 else s.name
                    aux = self.tok_v0_emb[k](idx) + self.pos_v0_emb[k](pos)[None, :, :]
                    out.append(self.drop(aux) if drop else aux)
                else:
                    out.append(torch.zeros_like(main))
            else:  # zeros, at this stream's own width
                w = self.cfg.n_embd if s.width == "d" else 1
                out.append(torch.zeros(B, T, w, dtype=main.dtype, device=main.device))
        return tuple(out)

    def _readout(self, state: State) -> torch.Tensor:
        if self.scfg.variant == "two_stream" and self.scfg.readout == "sum":
            return state[0] + state[1]
        return state[0]

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ):
        assert idx.shape[1] <= self.cfg.block_size
        state = self.init_state(idx)
        for blk in self.blocks:
            state = blk(state)

        if self.scfg.variant == "thermostat":
            vals = [b.last_logdet_per_token for b in self.blocks if b.last_logdet_per_token is not None]
            if vals:
                # mean over tokens of the stack's realized per-token log|det|,
                # minus the beta-only part => the thermostat's contribution.
                realized = float(torch.stack(vals).sum(dim=0).mean())
                self.last_logdet_delta = realized - self.log_det_per_token()

        h = self.ln_f(self._readout(state))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # -- diagnostics -------------------------------------------------------- #
    @torch.no_grad()
    def log_det_per_token(self) -> float:
        """Static per-token log|det| of the whole stack. For the thermostat this
        is the beta-only part; add `last_logdet_delta` for the realized value."""
        return float(sum(float(b.log_det_per_token(self.cfg.n_embd).detach()) for b in self.blocks))

    @torch.no_grad()
    def scalar_report(self) -> List[Dict[str, float]]:
        return [b.scalar_report() for b in self.blocks]

    def oracle_calls(self) -> Dict[str, int]:
        k = self.scfg.n_mlp if self.scfg.variant == "multirate" else 1
        return {"attn": self.cfg.n_layer, "mlp": self.cfg.n_layer * k}

    @torch.no_grad()
    def reconstruction_drift(self, idx: torch.Tensor) -> Dict[str, float]:
        """Forward the stack, invert it, and report the round-trip error against
        the true initial state."""
        s0 = self.init_state(idx, drop=False)
        state = s0
        for blk in self.blocks:
            state = blk(state)
        for blk in reversed(self.blocks):
            state = blk.inverse(state)
        worst, scale = 0.0, 1e-12
        for a, b in zip(state, s0):
            worst = max(worst, float((a - b).abs().max()))
            scale = max(scale, float(b.abs().max()))
        return {"recon_drift": worst, "recon_drift_rel": worst / scale}


# `train.print_sample` calls model.generate; model.py only attaches it to the
# classes defined there, so every model outside it must opt in explicitly.
YuriiStateModel.generate = _generic_generate


def build_yurii_state(cfg: ModelConfig, scfg: Optional[StateConfig] = None) -> YuriiStateModel:
    return YuriiStateModel(cfg, scfg)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _tiny(n_layer: int = 2, n_embd: int = 8, n_head: int = 2, vocab: int = 32):
    return ModelConfig(vocab_size=vocab, block_size=16, n_layer=n_layer, n_head=n_head,
                       n_embd=n_embd, dropout=0.0, bias=False, attn_impl="manual")


def _perturb(blk: nn.Module) -> None:
    """Move every learned scalar and norm off its init, so the tests exercise a
    generic point in parameter space rather than the special init point."""
    for name, p in blk.named_parameters():
        if name.endswith(".raw") or "ln" in name:
            p.data = p.data + 0.3 * torch.randn_like(p.data)


def _check_determinant(scfg: StateConfig, tol: float = 1e-8) -> float:
    """Analytic per-token log|det| vs an fp64 autograd Jacobian of the real
    block map on the full (all-streams) state."""
    torch.manual_seed(0)
    cfg = _tiny(n_layer=1)
    blk = _BLOCKS[scfg.variant](cfg, scfg, layer_idx=0).double().eval()
    _perturb(blk)

    T, d = 3, cfg.n_embd
    widths = [d if s.width == "d" else 1 for s in blk.spec_for(scfg)]
    parts = [torch.randn(1, T, w, dtype=torch.float64) for w in widths]
    sizes = [T * w for w in widths]

    def unflatten(flat):
        out, o = [], 0
        for w, n in zip(widths, sizes):
            out.append(flat[o:o + n].view(1, T, w))
            o += n
        return tuple(out)

    def f(flat):
        return torch.cat([t.reshape(-1) for t in blk(unflatten(flat))])

    flat = torch.cat([t.reshape(-1) for t in parts])
    J = torch.autograd.functional.jacobian(f, flat, vectorize=True)
    measured = float(torch.linalg.slogdet(J)[1])

    with torch.no_grad():
        blk(unflatten(flat))
        static = float(blk.log_det_per_token(d).detach())
        if scfg.variant == "thermostat":
            analytic = float(blk.last_logdet_per_token.sum())   # the block's own exact value
            delta = abs(analytic - static * T)
            if scfg.thermostat_channels:
                # THE claim of channel mode: channel-centering pins the eps*zeta
                # term to zero, so the realized determinant IS the beta-only one.
                assert delta < 1e-9, (
                    f"channel thermostat is not volume-pinned: eps*zeta contributes "
                    f"{delta:.2e}, so `sum_i zeta~_i = 0` is not holding"
                )
            else:
                # Guard against a vacuous pass: if the eps*zeta term were ~0 here,
                # this test would confirm the beta-only formula and nothing else.
                assert delta > 1e-3, (
                    f"scalar thermostat determinant test is vacuous: input-dependent "
                    f"term is {delta:.2e}, so it does not probe the zeta path"
                )
        else:
            analytic = static * T

    err = abs(measured - analytic)
    assert err < tol, f"{scfg.name()}: logdet {measured:.12f} vs {analytic:.12f} (err {err:.2e})"
    return static


def _check_inverse(scfg: StateConfig, tol: float = 1e-9) -> float:
    torch.manual_seed(1)
    model = YuriiStateModel(_tiny(n_layer=3), scfg).double().eval()
    for blk in model.blocks:
        _perturb(blk)
    idx = torch.randint(0, 32, (2, 5))
    d = model.reconstruction_drift(idx)
    assert d["recon_drift_rel"] < tol, f"{scfg.name()}: {d}"
    return d["recon_drift_rel"]


def _check_causality(scfg: StateConfig) -> None:
    """No stream, at any depth, may depend on a future token. This is the test
    that rejects token-mean centering of the thermostat's zeta -- see note 2."""
    torch.manual_seed(4)
    cfg = _tiny(n_layer=2)
    model = YuriiStateModel(cfg, scfg).eval()
    for blk in model.blocks:
        _perturb(blk)
    T, t = 6, 2
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    state = tuple(s.detach().requires_grad_(True) for s in model.init_state(idx, drop=False))
    h = state
    for blk in model.blocks:
        h = blk(h)
    model._readout(h)[0, t].sum().backward()
    future = max(float(s.grad[0, t + 1:].abs().max()) for s in state)
    assert future == 0.0, f"{scfg.name()}: leak from the future, max |g| = {future:.3e}"


def _check_oracle_calls(scfg: StateConfig) -> Tuple[int, int]:
    """Count actual module invocations rather than trusting the arithmetic."""
    cfg = _tiny(n_layer=3)
    model = YuriiStateModel(cfg, scfg).eval()
    counts = {"attn": 0, "mlp": 0}
    handles = []
    for m in model.modules():
        if isinstance(m, CausalSelfAttention):
            handles.append(m.register_forward_hook(lambda *a: counts.__setitem__("attn", counts["attn"] + 1)))
        elif isinstance(m, MLP):
            handles.append(m.register_forward_hook(lambda *a: counts.__setitem__("mlp", counts["mlp"] + 1)))
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 4)))
    for h in handles:
        h.remove()
    claimed = model.oracle_calls()
    assert counts == claimed, f"{scfg.name()}: counted {counts}, claimed {claimed}"
    return counts["attn"], counts["mlp"]


def _check_thermostat_gradients() -> None:
    """eps and rho multiply, so a zero init for either kills both gradients for
    good. Assert the shipped defaults do not do that -- and that the natural
    all-zero init provably would."""
    cfg = _tiny(n_layer=2)

    def grads(scfg):
        torch.manual_seed(5)
        m = YuriiStateModel(cfg, scfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 6))
        _, loss = m(idx, idx)
        loss.backward()
        g = {"eps": 0.0, "rho": 0.0}
        for n, p in m.named_parameters():
            for key in g:
                if key in n:
                    g[key] = max(g[key], float(p.grad.abs().max()))
        return g

    for channels in (True, False):
        mode = "channel" if channels else "scalar "
        live = grads(StateConfig(variant="thermostat", thermostat_channels=channels))
        assert live["eps"] > 0 and live["rho"] > 0, f"{mode}: shipped defaults are dead: {live}"
        dead = grads(StateConfig(variant="thermostat", thermostat_channels=channels,
                                 eps_init=0.0, rho_init=0.0))
        assert dead["eps"] == 0.0 and dead["rho"] == 0.0, f"{mode}: expected a dead start, got {dead}"
        print(f"thermostat {mode} gradients: shipped eps {live['eps']:.2e} rho {live['rho']:.2e}"
              f"  |  zero-init {dead['eps']:.0e}/{dead['rho']:.0e} (dead, as predicted)")


def _self_test() -> None:
    arms = [
        StateConfig(variant="thermostat"),                          # channel (default)
        StateConfig(variant="thermostat", thermostat_channels=False),
        StateConfig(variant="dual_momentum"),
        StateConfig(variant="two_stream"),                          # per-channel a,b (default)
        StateConfig(variant="two_stream", channel_scaling=False),
        StateConfig(variant="two_stream", swap=False),
        StateConfig(variant="multirate", n_mlp=2),
        StateConfig(variant="multirate", n_mlp=2, attn_memory=True),
        StateConfig(variant="multirate", n_mlp=3, share_mlp=True),
    ]
    print(f"{'variant':<26} {'streams':>8} {'A/M':>7} {'logdet/token':>14} {'recon':>10}")
    for scfg in arms:
        static = _check_determinant(scfg)
        drift = _check_inverse(scfg)
        _check_causality(scfg)
        a, m = _check_oracle_calls(scfg)
        n_streams = len(_BLOCKS[scfg.variant].spec_for(scfg))
        note = ("" if (scfg.variant != "thermostat" or scfg.thermostat_channels)
                else " (beta only)")
        print(f"{scfg.name():<26} {n_streams:>8} {f'{a}/{m}':>7} {static:>14.9f} {drift:>10.1e}{note}")

    print()
    _check_thermostat_gradients()

    # Volume comparability against yurii_ext's polyak baseline at equal init.
    from yurii_ext import YuriiExtModel, YuriiExtConfig
    cfg = _tiny(n_layer=6, n_embd=64, n_head=4, vocab=256)
    ref = YuriiExtModel(cfg, YuriiExtConfig()).log_det_per_token()
    print(f"\n{'model':<24} {'params':>12} {'log|det|/token':>15}")
    print(f"{'yurii_ext polyak':<24} {sum(p.numel() for p in YuriiExtModel(cfg, YuriiExtConfig()).parameters()):>12,} {ref:>15.6f}")
    for scfg in arms:
        m = YuriiStateModel(cfg, scfg)
        ld = m.log_det_per_token()
        tag = "  <-- matches polyak" if abs(ld - ref) < 1e-9 else ""
        print(f"{scfg.name():<24} {sum(p.numel() for p in m.parameters()):>12,} {ld:>15.6f}{tag}")

    print("\nall checks passed: determinants match autograd (thermostat's exactly, "
          "input-dependent term included),\ninverses are exact, no leakage from the "
          "future, oracle counts are as claimed.")


if __name__ == "__main__":
    _self_test()
