"""Invertible linear maps with controlled log-determinant, for reversible blocks.

A reversible block is factored as  Y_{l+1} = S_2 ∘ L_l ∘ S_1(Y_l)  where S_1, S_2
are volume-preserving shears (det = 1) and L_l is an invertible linear map acting
per-token on the feature vector. All volume change lives in L_l, with

    log|det J_block| = T · log|det L_l|        (T = sequence length).

Each map exposes a common interface:

    forward(y, c=0.0)   apply L           (..., m) -> (..., m)
    inverse(y, c=0.0)   apply L^{-1}
    logdet(c=0.0)       scalar log|det L|  (per token)
    logscales()         raw per-direction log-singular values  (m,)

``c`` is an external volume correction subtracted from the log-scales (the global
volume-preservation centering, supplied by GPT.forward); maps with
``vol_pres_per_block=True`` instead self-center so their per-block logdet is 0.

Convention: L has log-singular values ℓ (``logscales``); ℓ = 0 ⇒ L is orthogonal
(volume preserving). This generalizes the diagonal exp(-γ)/exp(-α) scaling of the
original ReversibleBlock to contraction/expansion of *learned* directions.
"""
import math

import torch
import torch.nn as nn

_EPS = 1e-12


class LinearMap(nn.Module):
    """Base class. Subclasses implement ``logscales`` and the apply/inverse of
    the orthogonal factors; centering, logdet and ``materialize`` are shared."""

    def __init__(self, m, vol_pres_per_block=False):
        super().__init__()
        self.m = m
        self.vol_pres_per_block = vol_pres_per_block

    # --- to implement in subclasses ---
    def logscales(self):
        """Raw per-direction log-singular values, shape (m,)."""
        raise NotImplementedError

    def _apply_core(self, y, eff):
        """Apply L with effective log-scales ``eff`` (shape (m,))."""
        raise NotImplementedError

    def _apply_core_inv(self, y, eff):
        """Apply L^{-1} with effective log-scales ``eff``."""
        raise NotImplementedError

    # --- shared ---
    def effective_logscales(self, c=0.0):
        s = self.logscales()
        if self.vol_pres_per_block:
            return s - s.mean()
        return s - c

    def forward(self, y, c=0.0):
        return self._apply_core(y, self.effective_logscales(c))

    def inverse(self, y, c=0.0):
        return self._apply_core_inv(y, self.effective_logscales(c))

    def logdet(self, c=0.0):
        return self.effective_logscales(c).sum()

    def mean_logscale(self):
        """Mean raw log-scale — what GPT.forward averages across blocks to build
        the global volume correction ``c`` (analogue of mean(γ)+mean(α))."""
        return self.logscales().mean()

    @torch.no_grad()
    def materialize(self, c=0.0):
        """Dense operator (m, m) such that forward(y) = y @ A.T — for testing."""
        return self.forward(torch.eye(self.m, device=self.logscales().device), c)


class HouseholderStack(nn.Module):
    """Orthogonal Q = H_k ⋯ H_1, each H_i = I - 2 v̂_i v̂_iᵀ a reflection.

    apply(y) = Q y ; applyT(y) = Qᵀ y (reflections in reverse order). det Q =
    (-1)^k, constant — so it carries no volume (all volume is in the diagonal)."""

    def __init__(self, m, k=None):
        super().__init__()
        k = m if k is None else min(k, m)
        self.V = nn.Parameter(torch.randn(k, m))

    def _units(self):
        return self.V / (self.V.norm(dim=1, keepdim=True) + _EPS)

    def mv(self, y):                                  # Q y
        for v in self._units():                       # H_1 first ... H_k last  => Q
            y = y - 2.0 * (y @ v).unsqueeze(-1) * v
        return y

    def mvT(self, y):                                 # Qᵀ y
        for v in reversed(self._units()):             # reverse order => Qᵀ
            y = y - 2.0 * (y @ v).unsqueeze(-1) * v
        return y


class WYHouseholder(nn.Module):
    """Orthogonal Q = H_1 H_2 ⋯ H_k (k unit Householder reflections) in the
    compact WY representation  Q = I - V T Vᵀ  (Schreiber & Van Loan 1989), with
    V = [v̂_1 … v̂_k] (m×k, unit columns) and T a k×k upper-triangular factor
    (diag 2). Applying Q is then THREE batched matmuls — no per-reflection loop:

        Q y  = y - (y @ V) @ Tᵀ @ Vᵀ
        Qᵀ y = y - (y @ V) @ T  @ Vᵀ        (Qᵀ = Q⁻¹, since Q is orthogonal)

    Cost O(m·k) per token (vs O(m·k) sequential for the naive stack, but here the
    batch never enters a Python loop), k·m parameters, exactly orthogonal so it
    carries no volume (det = ±1). The k-step T recursion is batch-free and tiny.
    """

    def __init__(self, m, k=4):
        super().__init__()
        k = max(1, min(k, m))
        self.m, self.k = m, k
        self.V = nn.Parameter(torch.randn(m, k))

    def _factors(self):
        V = self.V / (self.V.norm(dim=0, keepdim=True) + _EPS)   # unit columns
        # Build T functionally (no in-place) so it stays differentiable in V.
        # Recursion: T_1=[2]; t_j = -2 · T_{<j} (V_{<j}ᵀ v_j); T_j = [[T_{<j}, t_j],[0,2]].
        T = V.new_tensor([[2.0]])
        for j in range(1, self.k):
            tj = -2.0 * (T @ (V[:, :j].t() @ V[:, j]))            # (j,)
            top = torch.cat([T, tj.unsqueeze(1)], dim=1)          # (j, j+1)
            row = torch.cat([V.new_zeros(j), V.new_tensor([2.0])]).unsqueeze(0)
            T = torch.cat([top, row], dim=0)                      # (j+1, j+1)
        return V, T

    def mv(self, y):                                  # Q y
        V, T = self._factors()
        return y - ((y @ V) @ T.t()) @ V.t()

    def mvT(self, y):                                 # Qᵀ y = Q⁻¹ y
        V, T = self._factors()
        return y - ((y @ V) @ T) @ V.t()


class DiagMap(LinearMap):
    """L = diag(exp(ℓ)). Recovers the original diagonal contraction (with ℓ
    playing the role of -γ on the x-half and -α on the z-half)."""

    def __init__(self, m, vol_pres_per_block=False, frozen=False, init=None):
        super().__init__(m, vol_pres_per_block)
        s0 = torch.zeros(m) if init is None else torch.as_tensor(init, dtype=torch.float32)
        if frozen:                                    # volume_pres: ℓ ≡ 0 -> L = I
            self.register_buffer("ell", s0)
        else:
            self.ell = nn.Parameter(s0)

    def logscales(self):
        return self.ell

    def _apply_core(self, y, eff):
        return y * torch.exp(eff)

    def _apply_core_inv(self, y, eff):
        return y * torch.exp(-eff)


class SVDMap(LinearMap):
    """L = Q diag(exp(η)) R with Q, R fast orthogonal (Householder products).

    Contracts/expands *learned* directions (the columns of Q / rows of R) by
    e^{η_i}. log|det L| = Σ η_i exactly (Q, R have det = ±1). ``frozen`` (η ≡ 0)
    gives an orthogonal, volume-preserving mixing L = QR."""

    def __init__(self, m, k=None, vol_pres_per_block=False, frozen=False):
        super().__init__(m, vol_pres_per_block)
        self.Q = HouseholderStack(m, k)
        self.R = HouseholderStack(m, k)
        eta0 = torch.zeros(m)
        if frozen:
            self.register_buffer("eta", eta0)
        else:
            self.eta = nn.Parameter(eta0)

    def logscales(self):
        return self.eta

    def _apply_core(self, y, eff):
        return self.Q.mv(torch.exp(eff) * self.R.mv(y))

    def _apply_core_inv(self, y, eff):
        return self.R.mvT(torch.exp(-eff) * self.Q.mvT(y))


class LowRankCayleyMap(LinearMap):
    """L = e^{-c}·Q(K)·[I + U(diag(e^{-ρ})-I)Uᵀ] — rotation × low-rank bottleneck.

    Q(K) = (I - h/2 K)^{-1}(I + h/2 K) is an orthogonal Cayley transform of a
    skew K (det 1, pure mixing); the projector contracts r learned directions
    (columns of U, UᵀU=I) by e^{-ρ}. Separates *mixing* (rotation) from
    *forgetting* (a learned r-dim information bottleneck).

        log|det L| = -Σ_j ρ_j - m·c.

    The global volume correction ``c`` enters as an isotropic e^{-c} factor, so
    the same centering machinery works as for Diag/SVD. ``vol_pres_per_block``
    self-centers (per-block logdet 0); ``frozen`` (ρ≡0) gives a pure rotation.

    Cayley is applied with a dense solve (O(m^3)); fine for the small widths here
    — a structured / low-rank K with a Woodbury solve is the scaling path later.
    """

    def __init__(self, m, r=4, h=1.0, vol_pres_per_block=False, frozen=False,
                 rotation="cayley", n_householder=4):
        super().__init__(m, vol_pres_per_block)
        r = min(r, m)
        self.r = r
        self.h = h
        if rotation not in ("none", "cayley", "householder"):
            raise ValueError(f"rotation must be 'none'|'cayley'|'householder', got {rotation!r}")
        self.rotation = rotation
        self.U_param = nn.Parameter(torch.randn(m, r))
        if rotation == "cayley":
            self.K_param = nn.Parameter(torch.zeros(m, m))   # K = K_param - K_paramᵀ; 0 => Q = I; O(m^3) solve
        elif rotation == "householder":
            self.hh = WYHouseholder(m, n_householder)        # cheap O(m·k) orthogonal mixing
        rho0 = torch.zeros(r)
        if frozen:
            self.register_buffer("rho", rho0)
        else:
            self.rho = nn.Parameter(rho0)

    def logscales(self):
        return -self.rho                                 # the r active log-scales

    def mean_logscale(self):
        return (-self.rho.sum()) / self.m                # mean over all m dims (others = 0)

    def _c_eff(self, c):
        return self.mean_logscale() if self.vol_pres_per_block else c

    def _Uorth(self):
        return torch.linalg.qr(self.U_param)[0]          # (m, r), orthonormal columns

    def _proj(self, y, inverse=False):
        U = self._Uorth()
        coeff = torch.exp(self.rho if inverse else -self.rho) - 1.0   # e^{±ρ}-1
        return y + (y @ U) * coeff @ U.T

    def _cayley(self, y, inverse=False):
        K = self.K_param - self.K_param.T
        if inverse:
            K = -K
        eye = torch.eye(self.m, device=y.device, dtype=y.dtype)
        A = eye - (self.h / 2) * K
        Bm = eye + (self.h / 2) * K
        shp = y.shape
        w = (y.reshape(-1, self.m) @ Bm.T)               # (N, m)
        x = torch.linalg.solve(A, w.T).T                 # A x = w  (columns)
        return x.reshape(shp)

    def _iso(self, ce, sign):
        """Isotropic factor e^{sign·ce}, handling float or tensor ce."""
        x = sign * ce
        return torch.exp(x) if torch.is_tensor(x) else math.exp(x)

    def _rotate(self, y, inverse=False):
        """Apply Q (or Q⁻¹) — Cayley solve, WY-Householder, or identity."""
        if self.rotation == "cayley":
            return self._cayley(y, inverse=inverse)
        if self.rotation == "householder":
            return self.hh.mvT(y) if inverse else self.hh.mv(y)
        return y                                          # rotation == "none"

    def forward(self, y, c=0.0):
        ce = self._c_eff(c)
        out = self._rotate(self._proj(y, inverse=False), inverse=False)
        return self._iso(ce, -1.0) * out

    def inverse(self, y, c=0.0):
        ce = self._c_eff(c)
        out = self._proj(self._rotate(y, inverse=True), inverse=True)
        return self._iso(ce, +1.0) * out

    def logdet(self, c=0.0):
        return -self.rho.sum() - self.m * self._c_eff(c)
