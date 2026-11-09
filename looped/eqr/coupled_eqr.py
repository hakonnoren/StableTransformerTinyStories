"""Kick / relax two-stream couplings as the recurrent update inside EqR.

EqR (Huang, Geng & Kolter, "Equilibrium Reasoners", arXiv:2605.21488,
github.com/locuslab/EqR) iterates one shared reasoning module B on two states
(z_H, z_L). Its own update for a state h with injection u is

    h <- (1 - lambda) h + lambda * B(h + u) + noise,      lambda = 0.95,

i.e. a relaxation (keep 5% = e^-3) whose force reads the state itself. Here the
same B -- EqR's L-level layers, with the same parameters and initialisation --
is used as the force of the two-stream couplings from looped/model.py, which
read only the OTHER stream plus the input injection x (so every half-step is a
shear followed by a diagonal scaling, exactly invertible, with a
state-independent Jacobian determinant):

    kick :  L <- a (L + B(H + x))                     H <- b H + B(L + x)
    relax:  L <- a L + (1-a)(c H + B(H + x))           H <- b H + (1-b)(c L + B(L + x))

with a = e^-gamma, b = e^-alpha, c = e^-delta (delta = common-mode damping
"leak", relax only; delta = 0 is plain relax). The readout stream is H = z_H,
the second stream L = z_L. EqR's Langevin noise is added after every
half-step, exactly as EqR adds it after every module call.

`inject_h=False` drops x from H's force (H <- ... B(L) ...), as in EqR, whose
z_H update is B(z_H + z_L) without the input. With x in both forces and one
shared B, the two half-steps evaluate the same function of the other stream,
so the relax update is symmetric in (H, L) and its streams merge (Sudoku round
1: cos(z_H, z_L) = 1.00 for every relax arm). Without x the H force differs
from the L force and the two-state structure can survive. The update stays an
exact shear + scaling either way.

`self_read=True` is a DIAGNOSTIC, not an invertible coupling: each force also
reads the stream it updates (L <- ... B(L + H + x) ..., H <- ... B(H + L) ...),
as EqR's force reads its own state. Invertibility requires a stream's force not to
depend on that stream, so for fast rates L becomes an explicit function of H and
EqR/HRM's inner L iteration given H disappears; comparing an arm with and without
self_read measures what that constraint costs. inverse_half_step and
log_volume_per_recursion refuse to run when it is set.

`schedule="eqr"` runs EqR's own order inside a recursion: L_cycles updates of L, then one of H
(LLLLLLH for L_cycles = 6), instead of the alternation below. Repeated L updates only do work
when L's force changes between them, i.e. with self_read (L reads itself) or twin_l.

`twin_l=True` (needs schedule="eqr") doubles L into L and its twin L' and lets each read the
other where EqR's L reads itself; H reads their mean m = (L + L')/2:

    relax:  L  <- a L  + (1-a)(c L' + B(L' + H + x))
            L' <- a L' + (1-a)(c L  + B(L  + H + x))
            H  <- b H  + (1-b)(c m  + B(m [+ x if inject_h]))
    schedule per recursion: L, L', L, L', L, L', H      (L_cycles + 1 = 7 evaluations, as EqR)

(kick likewise: L <- a(L + B(L' + H + x)), H <- bH + B(m ...)). The L pair now changes between
its updates, so EqR's inner L iteration given H comes back; if the twins merge (L = L') the
pair acts like one L that reads itself with retention a + (1-a)c. Every update still reads only
streams it does not change, but no inverse is implemented for it. The twins live in the carry's
z_L, stacked along the sequence axis (z_L is [batch, 2 seq, hidden]), so EqR's carry, random
reset, evaluator and z_H/z_L metrics (which compare the first seq positions, i.e. H with L)
need no changes.

`twin_read="first"` (twin_l only) lets H read L alone instead of the mean m. With the mean, the
copy difference L - L' does not reach H, hence not the loss, at first order, and EqR's truncated
gradient never differentiates its evolution either; together with its neutrality at the flip
(dB eigenvalue -1 - c) this makes the copy mode invisible to training (looped/dynlab/paper,
Prop. 7). Reading one copy makes it visible.

`neumann_k=k > 0` changes only the gradient. Per outer iteration EqR differentiates the last of
H_cycles recursions psi from a detached state s (the truncated gradient: cotangent g on psi(s)).
For carried states whose recursion moved them by less than `neumann_tol` (relative), the
cotangent becomes v = g + J^T g + ... + (J^T)^k g, J = d psi / d s at s: the Neumann series of the
implicit fixed-point gradient (I - J^T)^-1 g, which the truncated gradient weights by 1 - lambda
per mode of J (looped/dynlab/paper, Prop. 5). Only near fixed points: at mid-search states the
series diverges and training fails (dynlab E8). The forward pass is unchanged. The series is
computed by an autograd function that, like reentrant checkpointing, recomputes the recursion
(noise-free; EqR's noise is additive, so J is unchanged) inside the backward pass and runs the
k vector-Jacobian products on that graph, so train with DISABLE_COMPILE=1.

Schedule (the one free choice of the port). EqR's latent recursion updates
z_L L_cycles times and then z_H once: L_cycles + 1 evaluations of B. Under a
coupling the force on a stream does not depend on that stream, so repeated
same-stream updates would recompute the same force. Each latent recursion
therefore runs `half_steps` (default L_cycles + 1 = 7) half-steps ALTERNATING
between the streams and ending on H (H, L, H, L, H, L, H). Everything around
it is EqR's: H_cycles - 1 recursions without gradient and one with, the
detached ACT carry across outer iterations, the halting head, the random
state reset, the noise. Per outer iteration this is H_cycles * half_steps
evaluations of B visited (21) and half_steps differentiated (7) -- the same
as EqR.

Other differences from EqR's own model, all deliberate:
  * the state is kept in float32 (`state_dtype`); the forces are computed in
    EqR's forward dtype (bf16). bf16 resolves 1 - 2^-8 near 1, so a rate of
    0.005 (e^-0.005 = 0.995) would not survive a bf16 state;
  * the readout (lm_head, q_head) sees rms_norm(H) (`readout_norm`; no
    parameters). EqR's z_H is a convex mix of post-normed outputs and stays at
    unit scale; a coupled H need not;
  * learned rates are excluded from weight decay unless `decay_rates`, as in
    looped/model.py's optimiser (EqR's recipe decays everything at 1.0, which
    pulls raw rates towards 0 at lr * wd per step). pretrain.py honours this
    through `rate_parameters()` (see install_overlay.sh).

Rate parametrisations follow looped/model.py ("Constrained damping"):
  default : relax gamma = softplus(raw); kick gamma = raw (may turn negative)
  positive: gamma = rate_min + softplus(raw)
  bounded : gamma in [rate_min, rate_max] via a sigmoid
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import model_validator
from torch import nn

from models.eqr import EqRConfig, EqRModel, InnerNetwork, LatentCarry
from models.layers import CosSin, rms_norm

COUPLINGS = ("kick", "relax")
SCHEDULES = ("alternate", "eqr")
CONSTRAINTS = ("default", "positive", "bounded")


def _inv_softplus(r: float) -> float:
    return r if r > 20 else math.log(math.expm1(r))


class CoupledEqRConfig(EqRConfig):
    coupling: str = "relax"            # 'kick' | 'relax'
    rate: float = 3.0                  # gamma at init (L stream)
    rate_h: Optional[float] = None     # alpha at init (H stream); None -> rate
    learn_rates: bool = False
    per_channel: bool = False          # one rate per channel instead of one scalar per stream
    rate_constraint: str = "default"   # 'default' | 'positive' | 'bounded'
    rate_min: float = 0.0
    rate_max: float = 10.0
    leak: float = 0.0                  # relax only: common-mode damping delta >= 0
    learn_leak: bool = False
    half_steps: Optional[int] = None   # half-steps per latent recursion; None -> L_cycles + 1
    inject_h: bool = True              # False: H's force is B(L) without the input x (EqR's own choice)
    self_read: bool = False            # diagnostic: forces also read their own stream (NOT invertible)
    schedule: str = "alternate"        # 'alternate' (H,L,...,H) | 'eqr' (L x L_cycles, then H)
    twin_l: bool = False               # L doubled into L, L' that read each other (needs schedule='eqr')
    twin_read: str = "mean"            # twin_l: H reads the twins' mean ('mean') or L alone ('first')
    neumann_k: int = 0                 # > 0: Neumann-series adjoint of k terms at converged carried states
    neumann_tol: float = 0.05          # relative change of the differentiated recursion below which it applies
    readout_norm: bool = True
    state_dtype: str = "float32"
    decay_rates: bool = False

    @model_validator(mode="after")
    def _check(self) -> "CoupledEqRConfig":
        if self.coupling not in COUPLINGS:
            raise ValueError(f"coupling={self.coupling!r}; choices {COUPLINGS}")
        if self.rate_constraint not in CONSTRAINTS:
            raise ValueError(f"rate_constraint={self.rate_constraint!r}; choices {CONSTRAINTS}")
        rh = self.rate if self.rate_h is None else self.rate_h
        if self.coupling == "relax" and self.rate_constraint == "default" and min(self.rate, rh) <= 0:
            raise ValueError("relax rates must be > 0 (softplus parametrisation)")
        if self.rate_constraint == "positive" and min(self.rate, rh) <= self.rate_min:
            raise ValueError("positive: initial rates must exceed rate_min")
        if self.rate_constraint == "bounded" and not (self.rate_min < min(self.rate, rh) <= max(self.rate, rh) < self.rate_max):
            raise ValueError("bounded: need rate_min < initial rates < rate_max")
        if self.leak < 0 or ((self.leak > 0 or self.learn_leak) and self.coupling != "relax"):
            raise ValueError("leak (common-mode damping) must be >= 0 and needs coupling='relax'")
        if self.half_steps is not None and self.half_steps < 1:
            raise ValueError("half_steps must be >= 1")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule={self.schedule!r}; choices {SCHEDULES}")
        if self.schedule == "eqr" and self.half_steps is not None:
            raise ValueError("half_steps applies to schedule='alternate'; 'eqr' runs L_cycles L updates + 1 H")
        if self.twin_l and self.schedule != "eqr":
            raise ValueError("twin_l needs schedule='eqr' (the twins' inner loop)")
        if self.twin_read not in ("mean", "first"):
            raise ValueError(f"twin_read={self.twin_read!r}; choices ('mean', 'first')")
        if self.twin_read != "mean" and not self.twin_l:
            raise ValueError("twin_read applies to twin_l=True")
        if self.neumann_k < 0 or self.neumann_tol <= 0:
            raise ValueError("neumann_k must be >= 0 and neumann_tol > 0")
        return self


def schedule_of(cfg: Any) -> Tuple[str, ...]:
    """Half-step order of one latent recursion: 'L' / 'H' (and 'M' = the twin L' when twin_l).
    alternate: half_steps (default L_cycles + 1) alternating, ending on the readout stream H.
    eqr      : L_cycles L-type updates (alternating L, L' with twin_l), then H -- EqR's own order."""
    if getattr(cfg, "schedule", "alternate") == "eqr":
        n_l = int(cfg.L_cycles)
        twin = bool(getattr(cfg, "twin_l", False))
        return tuple("M" if twin and i % 2 else "L" for i in range(n_l)) + ("H",)
    n = getattr(cfg, "half_steps", None) or (int(cfg.L_cycles) + 1)
    return tuple("H" if (n - 1 - i) % 2 == 0 else "L" for i in range(n))


class _NeumannCotangent(torch.autograd.Function):
    """Identity in the forward pass. In the backward pass the cotangent g of the recursion's output becomes
    v = g + sum_{i=1..k} (J^T)^i (mask g), J = d psi / d s at the saved input s; psi is recomputed on a fresh
    graph (the reentrant-checkpointing pattern; autograd inside a tensor hook is not supported)."""

    @staticmethod
    def forward(ctx, s_out, s_in, mask, psi, k):
        ctx.save_for_backward(s_in, mask)
        ctx.psi, ctx.k = psi, k
        return s_out.view_as(s_out)

    @staticmethod
    def backward(ctx, g):
        s_in, mask = ctx.saved_tensors
        with torch.enable_grad():
            s = s_in.detach().requires_grad_(True)
            out = ctx.psi(s)
            v, w = g, g * mask
            for _ in range(ctx.k):
                (w,) = torch.autograd.grad(out, s, grad_outputs=w, retain_graph=True)
                w = w * mask                 # samples do not interact (B mixes tokens within a puzzle)
                v = v + w
        return v, None, None, None, None


class CouplingRates(nn.Module):
    """gamma (L stream), alpha (H stream) and the relax leak delta."""

    def __init__(self, cfg: CoupledEqRConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        shape = (d,) if cfg.per_channel else (1,)
        c = cfg.rate_constraint
        if c == "default":
            c = "softplus" if cfg.coupling == "relax" else "raw"
        self.constraint, self.rmin, self.rmax = c, float(cfg.rate_min), float(cfg.rate_max)
        to_raw = {
            "raw": float,
            "softplus": _inv_softplus,
            "positive": lambda r: _inv_softplus(r - self.rmin),
            "bounded": lambda r: math.log((r - self.rmin) / (self.rmax - r)),
        }[c]
        g0 = float(cfg.rate)
        a0 = float(cfg.rate if cfg.rate_h is None else cfg.rate_h)
        for name, r in (("gamma_raw", g0), ("alpha_raw", a0)):
            t = torch.full(shape, to_raw(r), dtype=torch.float32)
            if cfg.learn_rates:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        self.has_leak = cfg.coupling == "relax" and (cfg.leak > 0 or cfg.learn_leak)
        self.learn_leak = bool(cfg.learn_leak)
        if self.has_leak:
            if self.learn_leak:
                self.leak_raw = nn.Parameter(torch.full(shape, _inv_softplus(cfg.leak) if cfg.leak > 0 else -5.0))
            else:
                self.register_buffer("leak_raw", torch.full(shape, float(cfg.leak)))

    def _rate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.constraint == "raw":
            return raw
        if self.constraint == "softplus":
            return F.softplus(raw)
        if self.constraint == "positive":
            return self.rmin + F.softplus(raw)
        return self.rmin + (self.rmax - self.rmin) * torch.sigmoid(raw)

    def gamma(self) -> torch.Tensor:
        return self._rate(self.gamma_raw)

    def alpha(self) -> torch.Tensor:
        return self._rate(self.alpha_raw)

    def c(self) -> Optional[torch.Tensor]:
        """Cross-stream weight e^-delta (None when there is no leak, i.e. c = 1)."""
        if not self.has_leak:
            return None
        delta = F.softplus(self.leak_raw) if self.learn_leak else self.leak_raw
        return torch.exp(-delta)


class _Force(nn.Module):
    """B(u): EqR's L-level layers applied to u, without EqR's lambda-mix and noise.

    Kept under the attribute path L_level.layers so EqR's parameter names, its
    initialisation order and pretrain.py's gradient-checkpointing hook are unchanged."""

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, u: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        for layer in self.layers:
            u = layer(hidden_states=u, **kwargs)
        return u


class CoupledInnerNetwork(InnerNetwork):
    def __init__(self, config: CoupledEqRConfig) -> None:
        super().__init__(config)                       # identical parameter init to EqR's InnerNetwork
        self.L_level = _Force(self.L_level.layers)     # same ReasoningBlocks, rule replaced
        self.rates = CouplingRates(config)
        self.kick = config.coupling == "kick"
        self.inject_h = bool(config.inject_h)
        self.self_read = bool(config.self_read)
        self.twin_l = bool(config.twin_l)
        self.twin_read = str(config.twin_read)
        self.neumann_k, self.neumann_tol = int(config.neumann_k), float(config.neumann_tol)
        self._neumann_seen, self._neumann_conv = 0, 0.0
        self.noise_scale = float(config.noise_scale)   # evaluate.py's runtime override sets this attribute
        self.state_dtype = getattr(torch, config.state_dtype)
        self.schedule: Tuple[str, ...] = schedule_of(config)

    # bookkeeping --------------------------------------------------------------------
    def rate_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.rates.parameters() if p.requires_grad]

    def depth_per_outer_step(self) -> Dict[str, int]:
        n = len(self.schedule)
        return {"visited_half_steps": self.config.H_cycles * n, "differentiated_half_steps": n}

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> LatentCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        shape_l = (batch_size, 2 * self.config.seq_len, self.config.hidden_size) if self.twin_l else shape
        return LatentCarry(
            z_H=torch.empty(shape, dtype=self.state_dtype, device=device),
            z_L=torch.empty(shape_l, dtype=self.state_dtype, device=device),   # twin_l: [L, L'] along seq
        )

    # maps ---------------------------------------------------------------------------
    def _force(self, other: torch.Tensor, x: Optional[torch.Tensor], seq: Dict[str, CosSin],
               own: Optional[torch.Tensor] = None) -> torch.Tensor:
        u = other if x is None else other + x
        if self.self_read:
            u = own + u                                 # diagnostic: EqR-style self-reading force
        u = u.to(self.forward_dtype)                    # EqR: (hidden + injection) in forward dtype
        return self.L_level(u, **seq).to(self.state_dtype)

    def _force_H(self, L: torch.Tensor, x: torch.Tensor, seq: Dict[str, CosSin],
                 H: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._force(L, x if self.inject_h else None, seq, own=H)

    def _check_invertible(self) -> None:
        if self.self_read:
            raise RuntimeError("self_read=True: the force reads the updated stream, so the update is not invertible "
                               "and its Jacobian determinant is state-dependent")
        if self.twin_l:
            raise RuntimeError("twin_l=True: no inverse / log-volume implemented for the twin-stream update")

    def _step(self, stream: str, own: torch.Tensor, other: torch.Tensor, rate: torch.Tensor,
              f: torch.Tensor) -> torch.Tensor:
        rate = rate.to(own.dtype)
        k = torch.exp(-rate)
        if self.kick:   # Hakon's coupling: L <- a (L + B), H <- b H + B
            return k * (own + f) if stream == "L" else k * own + f
        c = self.rates.c()
        cross = other if c is None else c.to(own.dtype) * other
        return k * own + (-torch.expm1(-rate)) * (cross + f)

    def _noisy(self, s: torch.Tensor) -> torch.Tensor:
        return s + torch.randn_like(s) * self.noise_scale if self.noise_scale > 0 else s

    def half_step_L(self, H, L, x, seq):
        return self._noisy(self._step("L", L, H, self.rates.gamma(), self._force(H, x, seq, own=L)))

    def half_step_H(self, H, L, x, seq):
        """H's update reading L (for twin_l, pass the twins' mean)."""
        return self._noisy(self._step("H", H, L, self.rates.alpha(), self._force_H(L, x, seq, H=H)))

    def half_step_twin(self, own, twin, H, x, seq):
        """twin_l: update of L (or L') reading its twin where EqR's L reads itself:
        relax own <- a own + (1-a)(c twin + B(twin + H + x)); kick own <- a (own + B(twin + H + x))."""
        f = self._force(twin + H, x, seq, own=own)
        return self._noisy(self._step("L", own, twin, self.rates.gamma(), f))

    def _unstep(self, stream: str, new: torch.Tensor, other: torch.Tensor, rate: torch.Tensor,
                f: torch.Tensor) -> torch.Tensor:
        """Exact inverse of _step (noise-free) given the unchanged other stream."""
        rate = rate.to(new.dtype)
        if self.kick:
            return torch.exp(rate) * new - f if stream == "L" else torch.exp(rate) * (new - f)
        c = self.rates.c()
        cross = other if c is None else c.to(new.dtype) * other
        return torch.exp(rate) * (new + torch.expm1(-rate) * (cross + f))

    def inverse_half_step(self, stream: str, H, L, x, seq):
        self._check_invertible()
        if stream == "L":
            return H, self._unstep("L", L, H, self.rates.gamma(), self._force(H, x, seq))
        return self._unstep("H", H, L, self.rates.alpha(), self._force_H(L, x, seq)), L

    def latent_recursion(self, z_H, z_L, x, seq):
        if self.twin_l:
            L, M = z_L.chunk(2, dim=1)                   # L and its twin L'
            for stream in self.schedule:
                if stream == "L":
                    L = self.half_step_twin(L, M, z_H, x, seq)
                elif stream == "M":
                    M = self.half_step_twin(M, L, z_H, x, seq)
                else:
                    z_H = self.half_step_H(z_H, 0.5 * (L + M) if self.twin_read == "mean" else L, x, seq)
            return z_H, torch.cat([L, M], dim=1)
        for stream in self.schedule:
            if stream == "L":
                z_L = self.half_step_L(z_H, z_L, x, seq)
            else:
                z_H = self.half_step_H(z_H, z_L, x, seq)
        return z_H, z_L

    def deep_recursion(self, z_H, z_L, x, seq):
        """EqR's deep recursion; with neumann_k > 0 (training only) the differentiated recursion's cotangent
        is replaced by its k-term Neumann series at converged carried states (see the module docstring)."""
        if self.neumann_k <= 0 or not torch.is_grad_enabled():
            return super().deep_recursion(z_H, z_L, x, seq)
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                z_H, z_L = self.latent_recursion(z_H, z_L, x, seq)
        n_h = z_H.shape[1]
        s_in = torch.cat([z_H, z_L], dim=1).detach().requires_grad_(True)
        h, l = self.latent_recursion(s_in[:, :n_h], s_in[:, n_h:], x, seq)
        s_out = torch.cat([h, l], dim=1)
        with torch.no_grad():
            moved = (s_out - s_in).flatten(1).norm(dim=1) / s_in.flatten(1).norm(dim=1).clamp_min(1e-30)
            conv = moved < self.neumann_tol
        self._neumann_log(conv)
        if bool(conv.any()):
            mask = conv.view(-1, *([1] * (s_out.dim() - 1))).to(s_out.dtype)
            xd = x.detach()

            def psi(s: torch.Tensor) -> torch.Tensor:          # the differentiated recursion, noise-free
                noise, self.noise_scale = self.noise_scale, 0.0
                try:
                    return torch.cat(self.latent_recursion(s[:, :n_h], s[:, n_h:], xd, seq), dim=1)
                finally:
                    self.noise_scale = noise

            s_out = _NeumannCotangent.apply(s_out, s_in.detach(), mask, psi, self.neumann_k)
        return s_out[:, :n_h], s_out[:, n_h:]

    def _neumann_log(self, conv: torch.Tensor) -> None:
        self._neumann_seen += 1
        self._neumann_conv += float(conv.float().mean())
        if self._neumann_seen % 500 == 0:
            print(f"[coupled_eqr] neumann_k={self.neumann_k}: {self._neumann_conv / 500:.3f} of carried states "
                  f"within tol {self.neumann_tol:g} (mean of the last 500 steps)", flush=True)
            self._neumann_conv = 0.0

    def log_volume_per_recursion(self, n_tokens: int) -> torch.Tensor:
        """log|det| of one latent recursion on n_tokens positions (state-independent)."""
        self._check_invertible()
        d = self.config.hidden_size
        g = self.rates.gamma().double().expand(d).sum()
        a = self.rates.alpha().double().expand(d).sum()
        nL = sum(s == "L" for s in self.schedule)
        nH = len(self.schedule) - nL
        return -n_tokens * (nL * g + nH * a)

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]):
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H, z_L = self.deep_recursion(
            carry.z_H.to(self.state_dtype), carry.z_L.to(self.state_dtype), x, {"cos_sin": self._cos_sin()}
        )
        h = rms_norm(z_H, variance_epsilon=self.config.rms_norm_eps) if self.config.readout_norm else z_H
        h = h.to(self.forward_dtype)
        logits = self.lm_head(h[:, self.config.puzzle_emb_len:]).contiguous()
        q = self.q_head(h[:, 0]).to(torch.float32)
        return LatentCarry(z_H=z_H.detach(), z_L=z_L.detach()), logits, q[..., 0]


class CoupledEqRModel(EqRModel):
    """EqRModel (ACT carry, halting, resets) with the coupled inner network."""

    def __init__(self, config_dict: dict) -> None:
        nn.Module.__init__(self)
        config_dict = dict(config_dict)
        # same puzzle-embedding bookkeeping as EqRModel.__init__
        puzzle_emb_ndim = int(config_dict.get("puzzle_emb_ndim", 0) or 0)
        puzzle_emb_len = int(config_dict.get("puzzle_emb_len", 0) or 0)
        if puzzle_emb_ndim > 0 and puzzle_emb_len <= 0:
            hidden_size = int(config_dict["hidden_size"])
            puzzle_emb_len = -(puzzle_emb_ndim // -hidden_size)
            config_dict["puzzle_emb_len"] = puzzle_emb_len
        if puzzle_emb_len > 0 and config_dict.get("input_seq_len") is None:
            config_dict["input_seq_len"] = config_dict["seq_len"]
            config_dict["seq_len"] = int(config_dict["seq_len"]) + puzzle_emb_len
        self.config = CoupledEqRConfig(**config_dict)
        self.inner = CoupledInnerNetwork(self.config)
        depth = self.inner.depth_per_outer_step()
        print(f"[coupled_eqr] coupling={self.config.coupling} inject_h={self.config.inject_h} "
              f"self_read={self.config.self_read} twin_l={self.config.twin_l} twin_read={self.config.twin_read} "
              f"neumann_k={self.config.neumann_k} "
              f"schedule={''.join(self.inner.schedule)} "
              f"visited={depth['visited_half_steps']} differentiated={depth['differentiated_half_steps']} "
              f"half-steps per outer iteration; learned rate params={len(self.inner.rate_parameters())}")

    def rate_parameters(self) -> List[nn.Parameter]:
        return [] if self.config.decay_rates else self.inner.rate_parameters()
