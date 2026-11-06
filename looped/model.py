"""Looped (recurrent-depth) transformers with two-stream conformal couplings.

One weight-tied core is applied ``n_iter`` times to a state; the number of
iterations is chosen per forward call (randomised during training, swept at test
time). Three iteration maps ("couplings"), cf. the conformal-relaxation draft:

``vanilla``  (looped baseline; single stream x of width d)
    x <- x + Attn(LN(x + e));   x <- x + MLP(LN(x + e))

``kick``     (Håkon's coupling = the ``ReversibleBlock`` update, iterated)
    L <- e^{-γ} (L + Attn(LN(H + e)))
    H <- e^{-α} H + MLP(LN(L + e))

``relax``    (the relaxation form)
    L <- e^{-γ} L + (1 - e^{-γ}) (H + Attn(LN(H + e)))
    H <- e^{-α} H + (1 - e^{-α}) (L + MLP(LN(L + e)))

``e`` is the token+position embedding, re-injected into every force argument
(``inject=True``, the usual input injection of looped / recurrent-depth models;
``inject=False`` sets e = 0 after the initial state). The forces only see the
*other* stream plus the constant e, so both two-stream couplings stay exactly
invertible (:meth:`CouplingLayer.inverse`) and every iteration has the
state-independent Jacobian determinant e^{-T·Σ_i(γ_i+α_i)} (T tokens; Σ over the
d channels, d·(γ+α) for scalar rates): a conformal, constant volume contraction.

What separates them is the fixed point. Relax stops changing when
L = H + Attn(LN(H+e)) and H = L + MLP(LN(L+e)), i.e. Attn + MLP = 0 — the same
condition as the looped standard block, and as γ, α -> ∞ relax *is* that block
(with state H). Kick needs Attn = (e^{γ}-1) L, MLP = (1-e^{-α}) H: the force has
to balance the leak.

Constrained damping (``rate_constraint``, ``leak``):
* ``positive``: γ = rate_min + softplus(raw) (also for kick, whose default rates are unconstrained and
  can turn into expansion);  ``bounded``: γ ∈ [rate_min, rate_max] via a sigmoid;  ``budget``: γ + α fixed
  (constant volume per iteration), learned split.
* ``leak`` δ (relax): the cross-stream term of both targets is scaled by c = e^{-δ},
      L <- e^{-γ}L + (1-e^{-γ})(c·H + Attn(LN(H+e))),  H <- e^{-α}H + (1-e^{-α})(c·L + MLP(LN(L+e))).
  The Jacobian determinant is unchanged (c only enters the shears), but the linear part becomes a strict
  contraction: at c = 1 it keeps the common mode L = H with eigenvalue 1, which is what lets the state
  drift; for c < 1 its spectral radius is < 1, so bounded forces give a bounded state and fixed points
  L = c·H + Attn, H = c·L + MLP. ODE view: Ẋ = G(cZ + H(Z) - X), Ż = A(cX + F(X) - Z).

Rates: relax uses γ = softplus(raw) > 0 (so e^{-γ} ∈ (0, 1) and each update is
a convex combination); kick uses γ = raw directly, as ``ReversibleBlock`` does
(it may go negative = expansion). ``learn_rates=False`` registers them as
buffers. Readout is LN_f(H) -> lm_head (H is the read-out stream; x for vanilla).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from looped.attention import CausalSelfAttention, NewGELU

COUPLINGS = ("vanilla", "kick", "relax")


@dataclass
class LoopedConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 32            # width d of each stream (vanilla: of the single state)
    n_head: int = 4
    n_core: int = 1             # coupling layers per iteration (weights shared across iterations)
    coupling: str = "relax"     # 'vanilla' | 'kick' | 'relax'
    rate: float = 3.0           # γ at init (L stream)
    rate_h: float = None        # α at init (H stream); None -> rate
    learn_rates: bool = False
    per_channel: bool = False   # one rate per channel instead of one scalar per stream
    inject: bool = True         # re-inject the embedding into every force argument
    inject_mode: str = "pre"    # 'pre': Attn(LN(H + e)) (the input's weight falls like 1/||H||);
                                # 'post': Attn(LN(H) + e) (constant weight, whatever the state norm)
    # --- damping constraints (see "Constrained damping" in the module docstring) ---
    rate_constraint: str = "default"   # 'default' | 'positive' | 'bounded' | 'budget'
    rate_min: float = 0.0       # positive: γ ≥ rate_min;  bounded: γ ∈ [rate_min, rate_max]
    rate_max: float = 10.0
    leak: float = 0.0           # relax only: common-mode damping δ ≥ 0 (cross-stream term scaled by e^{-δ})
    learn_leak: bool = False    # learn δ = softplus(raw) ≥ 0
    normalize: bool = True
    pos_emb: bool = True        # learned absolute positions; False = NoPE (causal attention only)
    causal: bool = True         # False = bidirectional attention (e.g. Sudoku); see looped/attention.py
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0

    def __post_init__(self):
        if self.coupling not in COUPLINGS:
            raise ValueError(f"coupling={self.coupling!r}; choices {COUPLINGS}")
        if self.inject_mode not in ("pre", "post"):
            raise ValueError(f"inject_mode={self.inject_mode!r}")
        if self.rate_constraint not in ("default", "positive", "bounded", "budget"):
            raise ValueError(f"rate_constraint={self.rate_constraint!r}")
        rh = self.rate if self.rate_h is None else self.rate_h
        if self.coupling == "relax" and min(self.rate, rh) <= 0:
            raise ValueError("relax rates must be > 0 (softplus parameterisation)")
        if self.rate_constraint == "positive" and min(self.rate, rh) <= self.rate_min:
            raise ValueError("positive: initial rates must exceed rate_min")
        if self.rate_constraint == "bounded" and not (self.rate_min < min(self.rate, rh) <= max(self.rate, rh) < self.rate_max):
            raise ValueError("bounded: need rate_min < initial rates < rate_max")
        if self.leak < 0 or (self.leak and self.coupling != "relax"):
            raise ValueError("leak (common-mode damping) must be >= 0 and needs coupling='relax'")

    def to_dict(self):
        return asdict(self)


def _inv_softplus(r):
    return r if r > 20 else math.log(math.expm1(r))


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d)
        self.act = NewGELU()
        self.c_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class CouplingLayer(nn.Module):
    """One attention + MLP update of the looped state (see module docstring)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.n_embd
        self.d, self.kind = d, cfg.coupling
        self.post = cfg.inject_mode == "post"
        ln = (lambda: nn.LayerNorm(d)) if cfg.normalize else nn.Identity
        self.ln_1, self.ln_2 = ln(), ln()
        self.attn = CausalSelfAttention(cfg, d)
        self.mlp = _MLP(d)
        if self.kind == "vanilla":
            return
        shape = (d,) if cfg.per_channel else (1,)
        self.constraint = cfg.rate_constraint
        if self.constraint == "default":            # previous behaviour: relax softplus, kick raw
            self.constraint = "softplus" if self.kind == "relax" else "raw"
        self.rmin, self.rmax = cfg.rate_min, cfg.rate_max
        g0 = cfg.rate
        a0 = cfg.rate if cfg.rate_h is None else cfg.rate_h
        reg = (lambda n, t: self.register_parameter(n, nn.Parameter(t))) if cfg.learn_rates else self.register_buffer
        if self.constraint == "budget":
            # γ + α = β fixed (constant volume per iteration), learned split γ = β·sigmoid(raw)
            self.register_buffer("budget", torch.full(shape, g0 + a0))
            reg("split_raw", torch.full(shape, math.log(g0 / a0)))
        else:
            to_raw = {"raw": float,
                      "softplus": _inv_softplus,
                      "positive": lambda r: _inv_softplus(r - self.rmin),
                      "bounded": lambda r: math.log((r - self.rmin) / (self.rmax - r))}[self.constraint]
            reg("gamma_raw", torch.full(shape, to_raw(g0)))
            reg("alpha_raw", torch.full(shape, to_raw(a0)))
        # common-mode damping δ (relax only): the cross-stream term of the targets is scaled by e^{-δ}
        self.has_leak = self.kind == "relax" and (cfg.leak > 0 or cfg.learn_leak)
        if self.has_leak:
            if cfg.learn_leak:
                self.leak_raw = nn.Parameter(torch.full(shape, _inv_softplus(cfg.leak) if cfg.leak > 0 else -5.0))
            else:
                self.register_buffer("leak_raw", torch.full(shape, float(cfg.leak)))
            self.learn_leak = cfg.learn_leak

    # forces -----------------------------------------------------------------
    def _fa(self, H, e):
        return self.attn(self.ln_1(H) + e) if self.post else self.attn(self.ln_1(H + e))

    def _fm(self, L, e):
        return self.mlp(self.ln_2(L) + e) if self.post else self.mlp(self.ln_2(L + e))

    # rates ------------------------------------------------------------------
    def _rate(self, raw):
        c = self.constraint
        if c == "raw":
            return raw
        if c == "softplus":
            return F.softplus(raw)
        if c == "positive":
            return self.rmin + F.softplus(raw)
        return self.rmin + (self.rmax - self.rmin) * torch.sigmoid(raw)      # bounded

    def gamma(self):
        if self.constraint == "budget":
            return self.budget * torch.sigmoid(self.split_raw)
        return self._rate(self.gamma_raw)

    def alpha(self):
        if self.constraint == "budget":
            return self.budget * torch.sigmoid(-self.split_raw)
        return self._rate(self.alpha_raw)

    def delta(self):
        """Common-mode damping δ ≥ 0 (0 when absent)."""
        if not self.has_leak:
            return None
        return F.softplus(self.leak_raw) if self.learn_leak else self.leak_raw

    def log_volume(self, T):
        """log|det| of this layer's map on a length-T sequence (state-independent)."""
        if self.kind == "vanilla":
            return None
        g = self.gamma().expand(self.d)
        a = self.alpha().expand(self.d)
        return -T * (g.sum() + a.sum())

    # maps -------------------------------------------------------------------
    def forward(self, s, e):
        if self.kind == "vanilla":
            s = s + self._fa(s, e)
            return s + self._fm(s, e)
        L, H = s.split(self.d, dim=-1)
        g, a = self.gamma(), self.alpha()
        kL, kH = torch.exp(-g), torch.exp(-a)
        if self.kind == "kick":
            L = kL * (L + self._fa(H, e))
            H = kH * H + self._fm(L, e)
        else:   # relax (c = e^{-δ} < 1 damps the common mode; c = 1 without leak)
            c = 1.0 if not self.has_leak else torch.exp(-self.delta())
            L = kL * L + (-torch.expm1(-g)) * (c * H + self._fa(H, e))
            H = kH * H + (-torch.expm1(-a)) * (c * L + self._fm(L, e))
        return torch.cat([L, H], dim=-1)

    def inverse(self, s, e):
        """Exact algebraic inverse of :meth:`forward` (two-stream couplings only)."""
        if self.kind == "vanilla":
            raise NotImplementedError("the looped standard block is not invertible")
        L, H = s.split(self.d, dim=-1)          # L', H'
        g, a = self.gamma(), self.alpha()
        if self.kind == "kick":
            H = torch.exp(a) * (H - self._fm(L, e))
            L = torch.exp(g) * L - self._fa(H, e)
        else:
            c = 1.0 if not self.has_leak else torch.exp(-self.delta())
            H = (H - (-torch.expm1(-a)) * (c * L + self._fm(L, e))) * torch.exp(a)
            L = (L - (-torch.expm1(-g)) * (c * H + self._fa(H, e))) * torch.exp(g)
        return torch.cat([L, H], dim=-1)

    def linear_part_radius(self):
        """Spectral radius of the iteration's linear part (forces set to zero), per channel, max over channels.

        relax: [[kL, (1-kL)c], [(1-kH)kL c, kH + (1-kH)(1-kL)c^2]], eigenvalues 1 and kL kH at c = 1;
        kick: diag(kL, kH). < 1 means the state stays bounded for bounded forces."""
        if self.kind == "vanilla":
            return None
        g, a = self.gamma().detach(), self.alpha().detach()
        kL, kH = torch.exp(-g), torch.exp(-a)
        if self.kind == "kick":
            return float(torch.maximum(kL, kH).max())
        c = torch.ones_like(kL) if not self.has_leak else torch.exp(-self.delta().detach()).expand_as(kL)
        t = kL + kH + (1 - kH) * (1 - kL) * c ** 2
        disc = (t ** 2 - 4 * kL * kH).clamp_min(0)
        return float(((t + disc.sqrt()) / 2).max())


class LoopedGPT(nn.Module):
    """Embedding -> weight-tied core iterated ``n_iter`` times -> LN_f(H) -> head."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        d = cfg.n_embd
        self.d = d
        self.two_stream = cfg.coupling != "vanilla"
        self.wte = nn.Embedding(cfg.vocab_size, d)
        self.wpe = nn.Embedding(cfg.block_size, d) if cfg.pos_emb else None
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.core = nn.ModuleList([CouplingLayer(cfg) for _ in range(cfg.n_core)])
        self.ln_f = nn.LayerNorm(d) if cfg.normalize else nn.Identity()
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_core))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)

    # pieces -------------------------------------------------------------------
    def embed(self, idx):
        """Returns (initial state s0, injection e). Two-stream: L0 = H0 = e."""
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        e = self.wte(idx) + (self.wpe(pos)[None] if self.wpe is not None else 0.0)
        e = self.drop(e)
        s0 = torch.cat([e, e], dim=-1) if self.two_stream else e
        inj = e if self.config.inject else torch.zeros_like(e)
        return s0, inj

    def step(self, s, e):
        for layer in self.core:
            s = layer(s, e)
        return s

    def center(self, s):
        """Remove each stream's per-token channel mean. Pre-LN branches and the final LN are invariant to
        per-token shifts of the streams, so the shifts form an invariant subspace of the iteration (for relax
        and the standard block the common shift is exactly neutral, eigenvalue 1); this projects onto its
        orthogonal complement, the "quotient" in which convergence is meaningful."""
        if not self.two_stream:
            return s - s.mean(-1, keepdim=True)
        L, H = s.split(self.d, dim=-1)
        return torch.cat([L - L.mean(-1, keepdim=True), H - H.mean(-1, keepdim=True)], dim=-1)

    def readout_stream(self, s):
        return s[..., self.d:] if self.two_stream else s

    def readout(self, s):
        return self.lm_head(self.ln_f(self.readout_stream(s)))

    # forward ------------------------------------------------------------------
    def forward(self, idx, targets=None, n_iter=8):
        s, e = self.embed(idx)
        for _ in range(n_iter):
            s = self.step(s, e)
        logits = self.readout(s)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def trajectory(self, idx, r_max, record):
        """Iterate to ``r_max`` and return {r: (logits, rel_residual, abs_residual, extra)} for r in ``record``.

        rel_residual = mean over sequences of ||s_r - s_{r-1}|| / ||s_r|| (Frobenius over T x width);
        abs_residual = mean over sequences of ||s_r - s_{r-1}||. Both on the full state.
        extra = dict with the same quantities in the quotient by per-token stream shifts (see ``center``):
        res_q = ||P(s_r - s_{r-1})||, norm_q = ||P s_r||, and the shift part norm_shift = ||(I-P) s_r||.
        Non-finite states are kept (their logits produce wrong answers downstream).
        """
        record = set(record)
        out = {}
        s, e = self.embed(idx)
        nrm = lambda x: x.flatten(1).norm(dim=1)
        if 0 in record:
            out[0] = (self.readout(s), float("nan"), float("nan"),
                      dict(res_q=float("nan"), norm_q=float(nrm(self.center(s)).mean()),
                           norm_shift=float(nrm(s - self.center(s)).mean())))
        for r in range(1, r_max + 1):
            s_new = self.step(s, e)
            if r in record:
                diff = nrm(s_new - s)
                rel = diff / nrm(s_new).clamp_min(1e-12)
                cq = self.center(s_new)
                extra = dict(res_q=float(nrm(cq - self.center(s)).mean()), norm_q=float(nrm(cq).mean()),
                             norm_shift=float(nrm(s_new - cq).mean()))
                out[r] = (self.readout(s_new), float(rel.mean()), float(diff.mean()), extra)
            s = s_new
        return out

    def rates(self):
        """{layer: (γ, α[, δ, linear-part radius])} as python lists, two-stream couplings only."""
        if not self.two_stream:
            return {}
        out = {}
        for i, l in enumerate(self.core):
            r = [l.gamma().detach().cpu().tolist(), l.alpha().detach().cpu().tolist()]
            if l.has_leak:
                r.append(l.delta().detach().cpu().tolist())
            r.append(l.linear_part_radius())
            out[i] = tuple(r)
        return out

    def step_spectral_radius(self, idx, r, iters=30, quotient=False):
        """Per-sequence estimate of the spectral radius of the iteration map's Jacobian at s_r (power iteration
        with Jacobian-vector products; dominant |eigenvalue|, geometric mean of the last 10 growth factors).
        ``quotient=True`` iterates P J P with P = ``center`` (the map on the quotient by per-token stream
        shifts; exact when the shift subspace is invariant, as for relax, kick and the standard block).
        Returns a 1-D tensor over the sequences of ``idx``."""
        with torch.no_grad():
            s, e = self.embed(idx)
            for _ in range(r):
                s = self.step(s, e)
        f = lambda x: self.step(x, e)
        P = self.center if quotient else (lambda x: x)
        v = P(torch.randn_like(s))
        v = v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))
        logs = []
        for k in range(iters):
            _, jv = torch.func.jvp(f, (s,), (v,))
            jv = P(jv)
            n = jv.flatten(1).norm(dim=1).clamp_min(1e-30)
            if k >= iters - 10:
                logs.append(torch.log(n))
            v = jv / n.view(-1, *[1] * (v.dim() - 1))
        return torch.exp(torch.stack(logs).mean(0)).detach()

    def configure_optimizers(self, lr, weight_decay=0.1, betas=(0.9, 0.95), rate_lr_mult=1.0):
        decay, no_decay, rates = [], [], []
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if pn in ("gamma_raw", "alpha_raw", "split_raw", "leak_raw"):
                    rates.append(p)
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    decay.append(p)
                else:
                    no_decay.append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        if rates:
            groups.append({"params": rates, "weight_decay": 0.0, "lr": lr * rate_lr_mult})
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
