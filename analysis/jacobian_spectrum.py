"""
Jacobian spectra of every block and of the whole block stack, the Corollary 4.5
coarse bounds on their extremes, the center (mean log singular value), and what
each of those costs to compute.

What is computed
----------------
Per block l, at a linearization point y_l on the model's own trajectory:

  1. Phi'_l = d block(y)/d y, dense (N x N), and ALL of its singular values.
  2. The center  lambda_bar(Phi'_l) = log|det Phi'_l| / N, by LU (slogdet) for
     every arch, and in closed form for reversible blocks, -sum_i(gamma_i +
     alpha_i) / (2d) (Proposition 4.1(iii)).
  3. Reversible / conformal blocks only: the coarse bounds of Corollary 4.5,

         sigma_min(Phi')^2 >= mu_-(s*, g_-^2, a_-^2 / (1 + sigma_F^2))
         sigma_max(Phi')^2 <= mu_+(s*, g_+^2 (1 + sigma_F^2), a_+^2)
         s* = sigma_H + (a_+ / g_-) beta(sigma_F)

     g_+-, a_+- are the extreme entries of G^ = exp(-gamma), A^ = exp(-alpha) after
     the regime's centering; sigma_H = ||H'||_2 and sigma_F = ||F'||_2 are the norms
     of the attention sublayer H = attn o ln_1 at z_l and of the MLP sublayer
     F = mlp o ln_2 at x_{l+1}, as T*d x T*d Jacobians over all tokens. The two
     norms are obtained two ways:
       power  matrix-free power iteration on J^T J, one JVP + one VJP per step.
              The cheap route the paper proposes, and the one that is TIMED as
              "coarse bounds". A Rayleigh quotient, so it approaches the true norm
              from below.
       exact  dense H' and F', built like Phi'. F' is block diagonal over tokens
              (MLP and LN act per token; the off-diagonal blocks are checked to be
              zero, not assumed), so ||F'|| is the largest of T small d x d SVDs;
              ||H'|| is sqrt of the top eigenvalue of H'^T H'. The reference the
              power estimates are scored against.
     Both versions of the bound are then checked against the SVD's extremes.
  4. sigma_max(Phi'_l) by the same power iteration, for every arch: the top of the
     spectrum is cheap; the bottom is what needs either the SVD or the bound.

Per stack depth k in --stack_depths, at the same points:

  5. Psi_{0,k} = d y_k / d y_0, dense, built by VJPs straight through blocks
     0..k-1 -- NOT as a product of the Phi'_l, so it is an independent object --
     and all of its singular values.
  6. Its center by LU, against the sum of the per-block LU log|dets|
     (Lemma 3.1(i), every arch) and against -T sum_l Tr(G_l + A_l) (reversible).
  7. The Lemma 3.1(iii) product bounds, from the exact per-block extremes and from
     the per-block coarse bounds:
         sum_l log sigma_min(Phi'_l) <= log sigma_min(Psi) <= log sigma_max(Psi)
                                     <= sum_l log sigma_max(Phi'_l).

Bounds only, at longer T (--bound_seq_lens):

  8. Steps 3-4 and the stack's power sigma_max with no dense matrix at all, at
     sequence lengths where Phi' cannot even be stored (T=1024: 19.8 TB in fp64).
     Linearization points are contiguous windows of the val split, i.e. what a
     training step sees, since few TinyStories stories are that long.

Timing
------
Every timer brackets a device synchronize, so it measures finished work rather
than kernel launches. Per record:

    jac_sec               dense Jacobian build
    svd_sec               full singular spectrum of it
    logdet_lu_sec         center by LU
    bound_sec             coarse bounds: both power iterations + the closed-form mu
    exact_norm_sec        dense sublayer norms (the reference, not a method)
    sigma_max_power_sec   power iteration for sigma_max of the block / stack

"Full spectrum" costs jac_sec + svd_sec, "coarse bounds" cost bound_sec. The power
iterations store their per-iteration estimates (sigma_*_trace), so accuracy against
iteration count -- hence against time -- can be read off against the exact norms.

Geometry
--------
    arch         per-token width C      N at T tokens
    baseline     n_embd     (768)       T *  768
    reversible   2*n_embd  (1536)       T * 1536   (the (x, z) state)

The two archs live on spaces of different dimension at the same T. Only
per-dimension quantities (mean_log_sigma, center_*, frac_sigma_gt1, quantiles, the
shape of the log-spectrum) compare directly across them; sum_log_sigma / frob /
erank scale with N and do not. `n_dim` is on every record so this stays checkable.

Why the sequence length must be short
-------------------------------------
J has N^2 entries and an SVD needs it resident:

    T     N (base)   N (rev)    J fp64 base   J fp64 rev
     8      6,144     12,288       0.30 GB       1.21 GB
    16     12,288     24,576       1.21 GB       4.83 GB
    32     24,576     49,152       4.83 GB      19.3  GB
   64+     49,152     98,304      19.3  GB      77.3  GB   <- infeasible
  1024    786,432  1,572,864       4.9  TB      19.8  TB   <- hopeless

At the training block_size of 1024 this object cannot be formed at any precision,
which is why --seq_len defaults to 16. That is an in-distribution measurement, not
a truncation hack: positions 0..T-1 with a real story prefix are exactly what the
model sees at the start of every training window. It is still a SHORT-CONTEXT
Jacobian, and nothing here licenses extrapolating the spectrum to T=1024. The
bounds of step 8 are the part that does reach T=1024.

How J is built (this is the part that would otherwise blow up)
--------------------------------------------------------------
Never `torch.autograd.functional.jacobian(..., vectorize=True)`: that vmaps over
all N cotangents at once, i.e. N-fold activation memory. Instead one forward graph
is built once and reused across `--chunk`-sized batched backward passes
(`is_grads_batched=True`), so peak activation memory is chunk-fold, not N-fold,
and the forward is not recomputed. Each pass yields `chunk` ROWS of J. Rows vs
columns is irrelevant here — J and J^T have identical singular values.

If a vmap batching rule is missing for some op, the loop degrades automatically to
one cotangent at a time (slower, same result). The manual attention kernel is the
default for that reason (`--attn_impl`); it is the same function as sdpa, and fp64
would fall out of the fused kernels anyway.

Correctness checks per Jacobian
-------------------------------
1. `fd_rel_err` - central finite differences along random unit directions vs J@v.
   Catches any AD/reshape/transposition error. Expect ~1e-7 or better in fp64.

2. `causal_resid` - max |dy_t/dx_s| over s > t. Both blocks are causal, so this is
   0 up to rounding. Catches a mask error or an off-by-one in the (T,C,T,C) view.

3. `logdet_err` (reversible only) - the reversible block factors as
   S2 o Lambda o S1 with both shears block-unit-triangular (x' = x + Attn(LN(z))
   depends on z alone; z' = ... + MLP(LN(x')) on x' alone), so det = det Lambda
   exactly and

       log|det J| = -T * (sum_i gamma_i + sum_i alpha_i)      (post-centering)

   which must equal the LU log|det|. For the stack the same holds summed over
   blocks, and for every arch the stack's log|det| must equal the sum of the
   blocks' (`logdet_composition_err`).

4. `bound_*_gap_exact` (reversible only) - log sigma_min - log(lower bound) and
   log(upper bound) - log sigma_max. A theorem, so both must be >= 0; `bound_ok`
   says whether they are.

Usage
-----
    # one arm, all 12 layers, 1 linearization point, plus the full stack and the
    # T=1024 bounds (what the SLURM job runs)
    python -m analysis.jacobian_spectrum \
        --ckpt_dir revparityreg_24979051/baseline_baseline revparityregrevback_25000297 \
        --out_dir results/jacobian_spectrum_T16_paper --arm reversible_vf_scaling_narrow \
        --seq_len 16 --n_points 1 --stack_depths L --bound_seq_lens 1024 --device cuda

    # cheap CPU smoke test
    python -m analysis.jacobian_spectrum --seq_len 4 --n_points 1 --layers 0 \
        --device cpu --out_dir /tmp/jac

Output, all under --out_dir:
    spectrum_<arm>.jsonl   one record per (layer, point)
    stack_<arm>.jsonl      one record per (depth, point)
    bounds_<arm>.jsonl     one record per (T, layer | stack, point), bounds only
    sv/<arm>_L<l>_p<p>_T<T>.npy         per-block spectrum (float64, length N)
    sv_stack/<arm>_K<k>_p<p>_T<T>.npy   stack spectrum
    manifest_<arm>.json    token ids and settings
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import corpus                                     # noqa: E402
from analysis.loader import discover, load_model                 # noqa: E402
from model import CausalSelfAttention                           # noqa: E402
from revformer.revformer import ReversibleBlock                 # noqa: E402

DTYPES = {"float64": torch.float64, "float32": torch.float32}
_REV_BLOCKS = (ReversibleBlock,)

# Record schema. 1: per-block spectra + checks (array 25002324). 2: adds ckpt,
# the coarse bounds, power iterations, the center by LU for every arch, and the
# stack / bounds-only files. --resume only reuses records of the current schema
# from the same checkpoint.
SCHEMA = 2

# A bound gap below this is a violation, not rounding. The SVD's sigma_min carries
# relative error ~eps * cond, i.e. ~1e-13 at the conditioning seen at T=16.
BOUND_TOL = 1e-8


def _clock(device) -> float:
    """time.perf_counter() after the device has drained its queue. Without the
    synchronize a CUDA timer measures kernel LAUNCHES, which for the Jacobian build
    and the power iterations is most of what there is to measure."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


# ------------------------------------------------------------------ model setup
def set_attn_impl(model, impl: str) -> int:
    """Force every attention module onto one kernel. The two kernels compute the
    same function (see the job header's A/B); manual is preferred here because it
    is plain matmul/softmax, hence fp64-safe and vmap-safe under the batched
    backward. Returns how many modules were switched."""
    n = 0
    for m in model.modules():
        if isinstance(m, CausalSelfAttention):
            m.attn_impl = impl
            n += 1
    return n


def block_fn(model, layer: int, avg):
    """The block as a pure function of its input state.

    Reversible blocks take the volume-centering correction as a second argument;
    it is a function of the parameters and T only, and is held fixed (it is not
    part of the state), so the Jacobian below is the Jacobian of the map the
    forward pass actually applies.
    """
    blk = model.blocks[layer]
    if isinstance(blk, _REV_BLOCKS):
        return lambda x: blk(x, avg)
    return lambda x: blk(x)


def stack_fn(model, depth: int, avg):
    """Blocks 0..depth-1 composed: y_0 -> y_depth, the map whose Jacobian is
    Psi_{0,depth}. Starts at the first block's input, i.e. after embedding (and
    the narrow models' lift), and stops before readout / ln_f -- the block stack
    on the state space, as in the paper's Appendix C."""
    fns = [block_fn(model, l, avg) for l in range(depth)]

    def f(y):
        for g in fns:
            y = g(y)
        return y
    return f


def volume_correction(model, T: int):
    """`avg` as RevFormerModel.stack_input would compute it (0.0 for baselines)."""
    if not hasattr(model, "rev_cfg"):
        return 0.0
    if model.rev_cfg.regime != "vpm_scaling":
        return 0.0
    with torch.no_grad():
        return model._avg_corr(T)


def linearization_points(model, ids: torch.Tensor, layers: list[int]) -> dict:
    """Block INPUT states for a real batch, via forward pre-hooks.

    Pre- rather than post-hooks on purpose: the Jacobian is wanted at the state a
    block actually receives in situ, so the linearization point is on the model's
    own trajectory rather than at some synthetic input.
    """
    caps: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(_mod, inp):
            caps[i] = inp[0].detach()
        return hook

    for i in layers:
        handles.append(model.blocks[i].register_forward_pre_hook(mk(i)))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    missing = [i for i in layers if i not in caps]
    if missing:
        raise RuntimeError(f"no state captured for layers {missing}")
    return caps


# ------------------------------------------------------------------ jacobian
def build_jacobian(fn, x0: torch.Tensor, chunk: int = 256, verbose: bool = False):
    """Dense J[i, j] = d fn(x)_i / d x_j at x0, one forward graph + batched VJPs.

    x0 is (1, T, C). Returns (J, n_fallback_rows): the second value is 0 when the
    batched path carried the whole build, and > 0 when a missing vmap batching
    rule forced the per-row fallback (same numbers, more time).
    """
    x = x0.detach().clone().requires_grad_(True)
    y = fn(x)
    n_out, n_in = y.numel(), x.numel()
    J = torch.empty((n_out, n_in), dtype=x.dtype, device=x.device)

    batched, fallback_rows = True, 0
    ar = torch.arange(chunk, device=y.device)
    for s in range(0, n_out, chunk):
        k = min(chunk, n_out - s)
        if batched:
            cot = torch.zeros(k, n_out, dtype=y.dtype, device=y.device)
            cot[ar[:k], torch.arange(s, s + k, device=y.device)] = 1
            try:
                g, = torch.autograd.grad(
                    y, x, grad_outputs=cot.view(k, *y.shape),
                    retain_graph=True, is_grads_batched=True)
                J[s:s + k] = g.reshape(k, n_in)
                del cot, g
                continue
            except RuntimeError as e:
                print(f"  [jac] batched VJP unavailable ({type(e).__name__}: "
                      f"{str(e)[:120]}); falling back to per-row", flush=True)
                batched = False
                del cot
        basis = torch.zeros(n_out, dtype=y.dtype, device=y.device)
        for r in range(s, s + k):
            basis.zero_()
            basis[r] = 1
            g, = torch.autograd.grad(y, x, grad_outputs=basis.view(*y.shape),
                                     retain_graph=True)
            J[r] = g.reshape(n_in)
            fallback_rows += 1
        del basis
        if verbose:
            print(f"  [jac] {s + k}/{n_out} rows", flush=True)

    del y, x
    return J, fallback_rows


@torch.no_grad()
def fd_check(fn, x0: torch.Tensor, J: torch.Tensor, n_dirs: int = 3,
             rel_eps: float | None = None, seed: int = 0):
    """max relative error between J@v and a central difference along v.

    eps is scaled to the RMS coordinate of x0 so it is meaningful whatever the
    residual stream's scale at this depth.
    """
    if n_dirs <= 0:
        return None, None
    n = x0.numel()
    if rel_eps is None:
        rel_eps = 1e-6 if x0.dtype == torch.float64 else 1e-3
    eps = float(rel_eps * (x0.norm() / math.sqrt(n)).clamp_min(1e-12))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    worst = 0.0
    for _ in range(n_dirs):
        v = torch.randn(n, generator=gen, dtype=torch.float64)
        v = (v / v.norm()).to(device=x0.device, dtype=x0.dtype)
        vv = v.view_as(x0)
        num = (fn(x0 + eps * vv).reshape(-1) - fn(x0 - eps * vv).reshape(-1)) / (2 * eps)
        ana = J @ v
        worst = max(worst, float((num - ana).norm() / ana.norm().clamp_min(1e-30)))
    return worst, eps


def causal_resid(J: torch.Tensor, T: int, C: int) -> float:
    """max |dy_t/dx_s| over strictly future inputs s > t. Zero for a causal block."""
    if T < 2:
        return 0.0
    m = J.view(T, C, T, C).abs().amax(dim=(1, 3))          # [t_out, s_in]
    fut = torch.triu(torch.ones(T, T, dtype=torch.bool, device=J.device), diagonal=1)
    return float(m.masked_select(fut).max())


def analytic_logdet(model, layer: int, avg, T: int):
    """log|det J| in closed form for the reversible blocks; None for the baseline.

    The coupling is shear o diagonal scale o shear with det-1 shears, so the whole
    block's log|det| is T times the per-token log-scale -sum(gamma + alpha).
    """
    blk = model.blocks[layer]
    if isinstance(blk, ReversibleBlock):
        if blk.frozen:                     # vpb_baseline: gamma = alpha = 0
            return 0.0
        with torch.no_grad():
            gamma, alpha = blk._effective_gamma_alpha(avg)
            return float(-(gamma.sum() + alpha.sum())) * T
    return None


# ------------------------------------------------------------------ spectrum
def svdvals(J: torch.Tensor, driver: str | None = None, cpu_fallback: bool = True):
    """All singular values (no vectors). Falls back to LAPACK on the host if
    cuSOLVER runs out of workspace -- the D2H copy needs no new device memory, so
    this recovers rather than dies at the most expensive point of the run."""
    try:
        if driver and J.is_cuda:
            return torch.linalg.svdvals(J, driver=driver)
        return torch.linalg.svdvals(J)
    except RuntimeError as e:
        if not (cpu_fallback and J.is_cuda):
            raise
        print(f"  [svd] device SVD failed ({str(e)[:140]}); retrying on CPU", flush=True)
        host = J.cpu()                 # D2H copy: needs host RAM, no new device RAM
        # J itself stays alive here (the caller owns it and frees it on return);
        # this only returns cuSOLVER's freed workspace to the allocator.
        torch.cuda.empty_cache()
        return torch.linalg.svdvals(host)


def spectrum_stats(sv: torch.Tensor) -> dict:
    sv = sv.detach().double().cpu()
    sv, _ = torch.sort(sv, descending=True)
    n = int(sv.numel())
    logs = torch.log(sv.clamp_min(1e-300))
    s2 = sv * sv
    tot = s2.sum()
    p = s2 / tot.clamp_min(1e-300)
    out = {
        "n_dim": n,
        "sigma_max": float(sv[0]),
        "sigma_min": float(sv[-1]),
        "cond": float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf"),
        # sum log sigma == log|det J|; the reversible check above pins this exactly
        "sum_log_sigma": float(logs.sum()),
        "mean_log_sigma": float(logs.mean()),
        # L1 magnitude of the summands. This is the SCALE of the log|det| sum, and
        # therefore the only honest denominator for its error: the sum is a
        # cancellation of N terms of order |log sigma|, so when it cancels to ~0
        # (vpb_baseline, vpb_scaling) the size of the result says nothing about how
        # well it was computed. See the logdet_rel_err note in run_arm.
        "sum_abs_log_sigma": float(logs.abs().sum()),
        "frob": float(torch.sqrt(tot)),
        "stable_rank": float(tot / s2[0].clamp_min(1e-300)),
        "entropy_erank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum())),
        "n_sigma_gt1": int((sv > 1).sum()),
        "frac_sigma_gt1": float((sv > 1).double().mean()),
    }
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        out[f"sigma_q{int(q * 100):02d}"] = float(torch.quantile(sv, q))
    return out


def noise_floor(sv: torch.Tensor, dtype) -> dict:
    """Singular values below ~N * eps * sigma_max are rounding, not signal: a
    backward-stable SVD perturbs every sigma by up to that much in absolute terms.
    Irrelevant for a single block at T=16 (cond ~1e6), but a product of 12 baseline
    blocks can plausibly push its bottom singular values under it, and then those
    values say only "smaller than this"."""
    floor = float(sv.max()) * sv.numel() * torch.finfo(dtype).eps
    return {"sv_floor": floor, "n_sigma_below_floor": int((sv < floor).sum())}


# ------------------------------------------------------------------ bounds
def sublayers(blk, avg, y0: torch.Tensor):
    """H and F of one reversible block, with the points their Jacobians are taken at.

    Mirrors ReversibleBlock.forward exactly,

        x_{l+1} = G^ (x_l + H(z_l)),      z_{l+1} = A^ z_l + F(x_{l+1}),

    with H = attn o ln_1, F = mlp o ln_2, G^ = exp(-gamma), A^ = exp(-alpha), so the
    block Jacobian is [I 0; F' I] diag(G^, A^) [I H'; 0 I] (Proposition 4.1(ii))
    with H' taken at z_l and F' at x_{l+1}. Returns (H, z_l, F, x_{l+1}).
    """
    x, z = torch.split(y0, y0.shape[-1] // 2, dim=-1)

    def H(u):
        return blk.attn(blk.ln_1(u))

    def F(u):
        return blk.mlp(blk.ln_2(u))

    with torch.no_grad():
        x1 = x + H(z)
        if not blk.frozen:
            gamma, _ = blk._effective_gamma_alpha(avg)
            x1 = torch.exp(-gamma) * x1
    return H, z.contiguous(), F, x1.contiguous()


def damping_extremes(blk, avg) -> tuple[float, float, float, float]:
    """(g_-, g_+, a_-, a_+): smallest and largest entries of G^ and A^."""
    if blk.frozen:
        return 1.0, 1.0, 1.0, 1.0
    with torch.no_grad():
        gamma, alpha = blk._effective_gamma_alpha(avg)
        g, a = torch.exp(-gamma), torch.exp(-alpha)
        return float(g.min()), float(g.max()), float(a.min()), float(a.max())


def power_sigma_max(fn, x0: torch.Tensor, max_iters: int, rtol: float, seed: int):
    """Largest singular value of d fn / dx at x0, never forming the matrix.

    Power iteration on J^T J: per step one JVP (u = J v) and one VJP (J^T u). The
    estimate ||J v|| with ||v|| = 1 is a Rayleigh quotient, so it never exceeds the
    true sigma_max and increases toward it; it stops once the relative change per
    step falls to rtol. Returns (sigma, iters, last_rel_change, trace).
    """
    _, vjp_fn = torch.func.vjp(fn, x0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(x0.shape, generator=gen, dtype=torch.float64)
    v = (v / v.norm()).to(device=x0.device, dtype=x0.dtype)
    trace, rel = [], float("inf")
    for _ in range(max_iters):
        _, u = torch.func.jvp(fn, (x0,), (v,))
        s = float(u.norm())
        if trace:
            rel = abs(s - trace[-1]) / max(s, 1e-300)
        trace.append(s)
        if rel <= rtol:
            break
        w, = vjp_fn(u)
        v = w / w.norm().clamp_min(1e-300)
    return trace[-1], len(trace), rel, trace


def mu_plus(s: float, g: float, a: float) -> float:
    """mu_+(s, g, a) of eq. (15)."""
    p = g * (1.0 + s * s)
    return 0.5 * (p + a + math.sqrt((p - a) ** 2 + 4.0 * g * a * s * s))


def mu_minus(s: float, g: float, a: float) -> float:
    """mu_-(s, g, a) of eq. (15), as g*a / mu_+ (Lemma 4.3(2)). The textbook form
    (b - sqrt(...))/2 cancels catastrophically exactly when the lower bound is
    small, which is the case it exists for."""
    return g * a / mu_plus(s, g, a)


def coarse_bounds(sigma_H: float, sigma_F: float, g_lo: float, g_hi: float,
                  a_lo: float, a_hi: float) -> tuple[float, float, float]:
    """Corollary 4.5: (lower bound on sigma_min, upper bound on sigma_max, s*)."""
    beta = 0.5 if sigma_F >= 1.0 else sigma_F / (1.0 + sigma_F ** 2)
    s = sigma_H + a_hi / g_lo * beta
    lo2 = mu_minus(s, g_lo ** 2, a_lo ** 2 / (1.0 + sigma_F ** 2))
    hi2 = mu_plus(s, g_hi ** 2 * (1.0 + sigma_F ** 2), a_hi ** 2)
    return math.sqrt(lo2), math.sqrt(hi2), s


def layer_bounds(blk, avg, y0: torch.Tensor, args, seed: int) -> dict:
    """The coarse bounds by power iteration -- the cheap route, timed end to end
    (the sublayer forward that locates x_{l+1}, both power iterations, the damping
    extremes and mu)."""
    dev = y0.device
    t0 = _clock(dev)
    H, z0, F, x1 = sublayers(blk, avg, y0)
    sH, itH, relH, trH = power_sigma_max(H, z0, args.power_iters, args.power_rtol, seed)
    t1 = _clock(dev)
    sF, itF, relF, trF = power_sigma_max(F, x1, args.power_iters, args.power_rtol, seed + 1)
    t2 = _clock(dev)
    g_lo, g_hi, a_lo, a_hi = damping_extremes(blk, avg)
    lo, hi, s = coarse_bounds(sH, sF, g_lo, g_hi, a_lo, a_hi)
    t3 = _clock(dev)
    return {
        "g_min": g_lo, "g_max": g_hi, "a_min": a_lo, "a_max": a_hi,
        "sigma_H_power": sH, "sigma_H_iters": itH, "sigma_H_relchg": relH,
        "sigma_F_power": sF, "sigma_F_iters": itF, "sigma_F_relchg": relF,
        "sigma_H_sec": round(t1 - t0, 4), "sigma_F_sec": round(t2 - t1, 4),
        "s_star_power": s, "bound_lo_power": lo, "bound_hi_power": hi,
        "bound_sec": round(t3 - t0, 4),
        "sigma_H_trace": trH, "sigma_F_trace": trF,
    }


def exact_sublayer_norms(blk, avg, y0: torch.Tensor, chunk: int) -> dict:
    """||H'||_2 and ||F'||_2 from the dense sublayer Jacobians (T*d square each).

    F' is block diagonal over tokens because LN and the MLP act on each position
    alone; its norm is then the largest of T d x d spectral norms. The off-diagonal
    blocks are measured (F_offdiag_resid, must be 0) rather than trusted. H' mixes
    positions, so its norm is sqrt(lambda_max(H'^T H')): a symmetric eigensolve,
    several times cheaper than an SVD of H' and exact to rounding for the TOP
    singular value (squaring only hurts the bottom of the spectrum).
    """
    dev = y0.device
    t0 = _clock(dev)
    H, z0, F, x1 = sublayers(blk, avg, y0)
    T, d = z0.shape[1], z0.shape[2]
    Hd, _ = build_jacobian(H, z0, chunk)
    sH = float(torch.linalg.eigvalsh(Hd.T @ Hd)[-1].clamp_min(0).sqrt())
    del Hd
    Fd, _ = build_jacobian(F, x1, chunk)
    Fv = Fd.view(T, d, T, d)
    tok = torch.arange(T, device=dev)
    diag = Fv[tok, :, tok, :]                                 # (T, d, d)
    blockmax = Fv.abs().amax(dim=(1, 3))                      # (T, T)
    off = blockmax.masked_select(~torch.eye(T, dtype=torch.bool, device=dev))
    sF = float(torch.linalg.matrix_norm(diag, ord=2).max())
    off = float(off.max()) if off.numel() else 0.0
    del Fd, Fv, diag
    t1 = _clock(dev)
    return {"sigma_H_exact": sH, "sigma_F_exact": sF, "F_offdiag_resid": off,
            "exact_norm_sec": round(t1 - t0, 3)}


def power_record(fn, x0: torch.Tensor, args, seed: int) -> dict:
    """sigma_max of a block or a stack by power iteration, timed."""
    t0 = _clock(x0.device)
    s, it, rel, tr = power_sigma_max(fn, x0, args.power_iters, args.power_rtol, seed)
    t1 = _clock(x0.device)
    return {"sigma_max_power": s, "sigma_max_power_iters": it,
            "sigma_max_power_relchg": rel, "sigma_max_power_sec": round(t1 - t0, 4),
            "sigma_max_power_trace": tr}


def bound_gaps(rec: dict) -> None:
    """Log-space slack of each available bound against the SVD's extremes, in place.
    >= 0 means the bound holds; bound_ok is judged on the exact-norm version, the
    power version being an estimate of it."""
    for kind in ("power", "exact"):
        lo, hi = rec.get(f"bound_lo_{kind}"), rec.get(f"bound_hi_{kind}")
        if lo is not None and lo > 0 and rec.get("sigma_min", 0) > 0:
            rec[f"bound_lo_gap_{kind}"] = math.log(rec["sigma_min"]) - math.log(lo)
        if hi is not None and rec.get("sigma_max") is not None:
            rec[f"bound_hi_gap_{kind}"] = math.log(hi) - math.log(rec["sigma_max"])
    if "bound_lo_gap_exact" in rec and "bound_hi_gap_exact" in rec:
        rec["bound_ok"] = (rec["bound_lo_gap_exact"] >= -BOUND_TOL
                           and rec["bound_hi_gap_exact"] >= -BOUND_TOL)


# ------------------------------------------------------------------ driver
def _gb(n: int, dtype) -> float:
    return n * n * torch.finfo(dtype).bits / 8 / 1024 ** 3


def parse_depths(tokens, n_layer: int) -> list[int]:
    """--stack_depths: integers k in 1..n_layer, 'L' for the full stack, 'all'."""
    out = set()
    for t in tokens or []:
        t = str(t).strip().lower()
        if t in ("l", "full"):
            out.add(n_layer)
        elif t == "all":
            out.update(range(1, n_layer + 1))
        else:
            k = int(t)
            if not 1 <= k <= n_layer:
                raise SystemExit(f"--stack_depths {k} outside 1..{n_layer}")
            out.add(k)
    return sorted(out)


def val_windows(val_bin: str, T: int, n: int, seed: int) -> torch.Tensor:
    """n contiguous T-token windows of the val split (story boundaries included,
    as in a training batch), at seeded positions so every arm gets the same ones."""
    toks = corpus.load_bin(val_bin)
    if toks.size <= T:
        raise SystemExit(f"{val_bin} has {toks.size} tokens, need > {T}")
    starts = np.random.default_rng(seed * 100003 + T).integers(0, toks.size - T, size=n)
    return torch.from_numpy(np.stack([toks[s:s + T].astype(np.int64) for s in starts]))


def open_records(path: str, ckpt: str, resume: bool, keep) -> tuple[list[dict], object]:
    """Existing reusable records of one output file, and a handle to append to it.

    Never truncates a file it did not write for this checkpoint. The same arm name
    exists in several sweeps (revparityreg_24979051, revparityregrevback_25000297,
    ...), so with a shared --out_dir the old behaviour -- resume from whatever
    records carry the arm's name, or open with "w" -- would either reuse another
    model's spectra or overwrite them. A file holding records of another checkpoint
    or of an older schema is therefore refused outright.
    """
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return [], open(path, "w")
    recs, foreign = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"  resume: ignoring a truncated record in {path}", flush=True)
            continue
        if r.get("schema") != SCHEMA or r.get("ckpt") != ckpt:
            foreign += 1
        elif keep(r):
            recs.append(r)
    if foreign:
        raise SystemExit(
            f"{path} holds {foreign} record(s) from another checkpoint or an older "
            f"record schema (this run: schema {SCHEMA}, ckpt {ckpt}). Refusing to "
            f"append to or overwrite it -- pass a different --out_dir.")
    if not resume:
        raise SystemExit(f"{path} already has records for this checkpoint; pass "
                         f"--resume to continue it, or a different --out_dir.")
    return recs, open(path, "a")


def _emit(fh, rec: dict) -> None:
    # Flush per record, not per arm: this line is the difference between a killed
    # run being readable and being a directory of orphaned .npy files.
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def _free(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _peak(rec: dict, device) -> None:
    if str(device).startswith("cuda"):
        rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def dense_spectrum(rec: dict, fn, x0: torch.Tensor, args, dtype, ana_ld, sv_path,
                   seed: int) -> None:
    """Build J, check it, take its LU log|det| and full spectrum; fill rec in place.
    Shared by the per-block and the stack records."""
    T, C = x0.shape[1], x0.shape[2]
    n = T * C
    dev = x0.device
    t0 = _clock(dev)
    J, fb = build_jacobian(fn, x0, chunk=args.chunk, verbose=args.verbose)
    t_jac = _clock(dev) - t0

    t0 = _clock(dev)
    fd_err, fd_eps = fd_check(fn, x0, J, n_dirs=args.fd_dirs, seed=seed)
    cz = causal_resid(J, T, C)
    t_chk = _clock(dev) - t0
    rec.update({"fd_rel_err": fd_err, "fd_eps": fd_eps, "causal_resid": cz,
                "fallback_rows": fb, "jac_sec": round(t_jac, 3),
                "check_sec": round(t_chk, 3), "n_dim": n})
    if args.skip_svd:
        del J
        return

    # log|det J| by LU, BEFORE the SVD. This is the accurate route to the
    # determinant and it is nearly free next to the SVD: measured on one
    # N=24,576 fp64 Jacobian, slogdet 48 s vs svdvals 6447 s, and it
    # recovers the analytic 0 to 7.7e-14 where LAPACK's own svdvals manages
    # 1.2e-11 and cuSOLVER's manages 2.6e-07. Summing log sigma was always
    # the worst way to get a determinant; it only looked natural because the
    # spectrum was already being computed. Done for every arch: it is the center.
    t0 = _clock(dev)
    rec["logdet_lu"] = float(torch.linalg.slogdet(J)[1])
    rec["logdet_lu_sec"] = round(_clock(dev) - t0, 3)
    rec["center_lu"] = rec["logdet_lu"] / n

    t0 = _clock(dev)
    sv = svdvals(J, driver=args.svd_driver)
    del J
    rec["svd_sec"] = round(_clock(dev) - t0, 3)
    rec.update(spectrum_stats(sv))
    rec.update(noise_floor(sv, dtype))
    rec["logdet_analytic"] = ana_ld
    if ana_ld is not None:
        rec["center_analytic"] = ana_ld / n
        # The CORRECTNESS gate: analytic identity vs the accurate (LU)
        # determinant. Independent of whatever the SVD backend does.
        rec["logdet_err"] = abs(rec["logdet_lu"] - ana_ld)
        # And separately, a PRECISION diagnostic for the SVD backend:
        # how far its sum log sigma drifts from the same LU value. This
        # is not a correctness signal -- on the T=16 sweep it sits at
        # 2.6e-07 for cuSOLVER and ~1e-11 for LAPACK on identical
        # matrices -- so it is reported, never gated on.
        rec["svd_logdet_drift"] = abs(rec["sum_log_sigma"] - rec["logdet_lu"])
        # Normalize by the SCALE OF THE SUMMANDS, not by the result.
        #
        # The earlier denominator max(|analytic|, 1.0) silently became an
        # ABSOLUTE tolerance for exactly the two regimes whose analytic
        # log|det| is 0 by construction (vpb_baseline frozen at
        # gamma=alpha=0; vpb_scaling self-centered to 0 per block). Those
        # two then "failed" a 1e-9 gate at 2.5e-07 while vf/vpm "passed"
        # with the SAME 2.6e-07 absolute error, purely because dividing by
        # |analytic| ~ 6000 shrank it. Measured on the T=16 sweep: the
        # absolute error is 2.60e-07 +/- 1.4e-08 on all four regimes alike.
        #
        # sum_abs_log_sigma is the L1 norm of the N terms being summed, so
        # this ratio is the relative accuracy of a cancelling sum -- the
        # standard yardstick, finite and meaningful whether the result is
        # 0 or 30000. It lands at 3-5e-11 uniformly across the four
        # regimes, which is what a consistent metric should look like.
        scale = max(abs(ana_ld), rec["sum_abs_log_sigma"], 1.0)
        rec["logdet_rel_err"] = rec["logdet_err"] / scale
        rec["logdet_err_scale"] = scale
    if args.save_sv:
        os.makedirs(os.path.dirname(sv_path), exist_ok=True)
        np.save(sv_path, sv.detach().double().cpu().numpy())
        rec["sv_file"] = os.path.relpath(sv_path, args.out_dir)
    del sv


def run_arm(name: str, path: str, ids: torch.Tensor, args, dtype,
            ids_long: dict[int, torch.Tensor]) -> tuple[list[dict], dict]:
    """Every Jacobian, bound and timing for one arm. Records are STREAMED, not
    returned-then-written.

    A single reversible arm is hours of SVD, so a run that hits the wall clock or
    gets killed partway is a normal outcome, not an exception. Writing the JSONL
    only at the end made those runs produce a directory full of .npy spectra and no
    records at all — everything computed, nothing readable. The record files and the
    manifest are therefore opened and flushed as we go, so whatever finished is
    usable immediately.
    """
    t_load = time.time()
    model, meta = load_model(path, device="cpu")
    n_switched = set_attn_impl(model, args.attn_impl)
    # Nothing here differentiates w.r.t. parameters; without this every Jacobian
    # graph would also carry the weights' autograd history.
    model.requires_grad_(False)
    model = model.to(dtype).to(args.device).eval()
    T = ids.shape[1]
    P = ids.shape[0]
    n_layer = len(model.blocks)
    layers = args.layers if args.layers else list(range(n_layer))
    bad = [l for l in layers if not 0 <= l < n_layer]
    if bad:
        raise SystemExit(f"{name}: layers {bad} out of range (n_layer={n_layer})")
    depths = parse_depths(args.stack_depths, n_layer)
    rev = isinstance(model.blocks[0], _REV_BLOCKS)
    bounds_on = rev and not args.no_bounds

    ids_dev = ids.to(args.device)
    avg = volume_correction(model, T)
    caps = linearization_points(model, ids_dev, sorted(set(layers) | ({0} if depths else set())))
    C = caps[0 if depths else layers[0]].shape[-1]
    n = T * C
    print(f"[{name}] arch={meta['arch']} regime={meta['regime']} n_layer={n_layer} "
          f"T={T} C={C} -> N={n}  (J = {_gb(n, dtype):.2f} GB in {args.dtype}, "
          f"attn_impl={args.attn_impl} on {n_switched} modules, "
          f"loaded in {time.time() - t_load:.1f}s)", flush=True)
    print(f"[{name}] per-block spectra for layers {layers}; stack depths {depths or 'none'}; "
          f"coarse bounds {'on' if bounds_on else 'off (' + ('baseline' if not rev else '--no_bounds') + ')'}; "
          f"bounds-only at T={sorted(ids_long) or 'none'}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    man_path = os.path.join(args.out_dir, f"manifest_{name}.json")
    rec_path = os.path.join(args.out_dir, f"spectrum_{name}.jsonl")
    stk_path = os.path.join(args.out_dir, f"stack_{name}.jsonl")
    bnd_path = os.path.join(args.out_dir, f"bounds_{name}.jsonl")
    sv_dir = os.path.join(args.out_dir, "sv")
    stk_dir = os.path.join(args.out_dir, "sv_stack")

    # The manifest is overwritten below, so it gets the same protection as the
    # record files: a manifest describing another checkpoint is not ours to replace.
    if os.path.exists(man_path):
        try:
            old = json.load(open(man_path))
        except (OSError, json.JSONDecodeError):
            old = {}
        if old.get("ckpt") not in (None, path) or old.get("schema", 1) != SCHEMA:
            raise SystemExit(
                f"{man_path} describes ckpt {old.get('ckpt')} (schema "
                f"{old.get('schema', 1)}); this run is {path} (schema {SCHEMA}). "
                f"Pass a different --out_dir.")

    # --resume: one reversible arm is 24 SVDs at ~8.5 min each, so a task that dies
    # at hour 5 has real work in it. A record only counts as done if its .npy is
    # present too (unless --save_sv is off), so a record flushed just before the
    # process died without its spectrum is recomputed rather than trusted.
    need_sv = args.save_sv and not args.skip_svd

    def keep_layer(r):
        return (r.get("seq_len") == T and r.get("dtype") == args.dtype and
                (not need_sv or os.path.exists(os.path.join(args.out_dir, r.get("sv_file", "")))))

    def keep_stack(r):
        return keep_layer(r)

    def keep_bound(r):
        return r.get("dtype") == args.dtype and r.get("seq_len") in ids_long

    recs, rec_fh = open_records(rec_path, path, args.resume, keep_layer)
    srecs, stk_fh = open_records(stk_path, path, args.resume, keep_stack)
    brecs, bnd_fh = open_records(bnd_path, path, args.resume, keep_bound)
    done = {(r["layer"], r["point"]) for r in recs}
    sdone = {(r["depth"], r["point"]) for r in srecs}
    bdone = {(r["seq_len"], r["kind"], r.get("layer", r.get("depth")), r["point"])
             for r in brecs}
    if recs or srecs or brecs:
        print(f"[{name}] resume: {len(done)} block, {len(sdone)} stack and "
              f"{len(bdone)} bounds-only records already complete", flush=True)

    # Manifest before the first Jacobian: provenance is what makes a partial run
    # interpretable, so it must not depend on the run completing.
    manifest = {
        "arm": name, "ckpt": path, "meta": meta, "schema": SCHEMA,
        "seq_len": T, "width": C, "n_dim": n, "layers": layers,
        "stack_depths": depths, "bound_seq_lens": sorted(ids_long),
        "n_points": P, "dtype": args.dtype,
        "attn_impl": args.attn_impl, "chunk": args.chunk,
        "svd_driver": args.svd_driver, "device": args.device,
        "power_iters": args.power_iters, "power_rtol": args.power_rtol,
        "coarse_bounds": bounds_on, "exact_norms": bounds_on and args.exact_norms,
        "seed": args.seed, "val_bin": args.val_bin,
        "token_ids": ids.tolist(),
        "bound_token_ids": {str(t): v.tolist() for t, v in ids_long.items()},
        "torch": torch.__version__,
        "device_name": (torch.cuda.get_device_name(0)
                        if str(args.device).startswith("cuda") else "cpu"),
        "complete": False,
    }
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    base = {"arm": name, "ckpt": path, "schema": SCHEMA, "arch": meta["arch"],
            "regime": meta["regime"], "seq_len": T, "width": C, "dtype": args.dtype,
            "attn_impl": args.attn_impl}
    nan = float("nan")

    # ---------------------------------------------------------------- per block
    for layer in layers:
        ana_ld = analytic_logdet(model, layer, avg, T)
        fn = block_fn(model, layer, avg)
        blk = model.blocks[layer]
        for p in range(P):
            if (layer, p) in done:
                print(f"  L{layer:02d} p{p} | skipped (already on disk)", flush=True)
                continue
            x0 = caps[layer][p:p + 1].contiguous()
            seed = args.seed + 1000 * layer + p
            rec = dict(base, layer=layer, point=p, logdet_analytic=ana_ld)

            # The cheap quantities first, on an otherwise idle device, so their
            # timers are not sharing it with anything.
            if bounds_on:
                rec.update(layer_bounds(blk, avg, x0, args, seed))
            rec.update(power_record(fn, x0, args, seed + 7))
            if bounds_on and args.exact_norms:
                rec.update(exact_sublayer_norms(blk, avg, x0, args.chunk))
                lo, hi, s = coarse_bounds(rec["sigma_H_exact"], rec["sigma_F_exact"],
                                          rec["g_min"], rec["g_max"],
                                          rec["a_min"], rec["a_max"])
                rec.update({"s_star_exact": s, "bound_lo_exact": lo, "bound_hi_exact": hi})
                _free(args.device)

            sv_path = os.path.join(sv_dir, f"{name}_L{layer:02d}_p{p}_T{T}.npy")
            dense_spectrum(rec, fn, x0, args, dtype, ana_ld, sv_path, args.seed + p)
            bound_gaps(rec)
            _peak(rec, args.device)
            recs.append(rec)
            _emit(rec_fh, rec)

            ld_txt = "n/a" if ana_ld is None else \
                f"{ana_ld:+.4f} err {rec.get('logdet_err', nan):.2e}"
            bnd_txt = "" if not bounds_on else (
                f" | bound [{rec['bound_lo_power']:.3e}, {rec['bound_hi_power']:.3e}] "
                f"in {rec['bound_sec']:.2f}s (sH {rec['sigma_H_power']:.3e}/{rec['sigma_H_iters']}it, "
                f"sF {rec['sigma_F_power']:.3e}/{rec['sigma_F_iters']}it)"
                + (f" ok={rec.get('bound_ok')}" if "bound_ok" in rec else ""))
            print(f"  L{layer:02d} p{p} | "
                  f"jac {rec['jac_sec']:6.1f}s svd {rec.get('svd_sec', nan):7.1f}s | "
                  f"smax {rec.get('sigma_max', nan):9.3e} (power {rec['sigma_max_power']:9.3e}) "
                  f"smin {rec.get('sigma_min', nan):9.3e} | "
                  f"logdet {rec.get('logdet_lu', nan):+.4f} (analytic {ld_txt}) | "
                  f"fd {nan if rec['fd_rel_err'] is None else rec['fd_rel_err']:.2e} "
                  f"causal {rec['causal_resid']:.2e}{bnd_txt}", flush=True)
    rec_fh.close()

    # ---------------------------------------------------------------- stack
    by_lp = {(r["layer"], r["point"]): r for r in recs}
    for k in depths:
        sfn = stack_fn(model, k, avg)
        ana_k = None
        if rev:
            ana_k = sum(analytic_logdet(model, l, avg, T) for l in range(k))
        for p in range(P):
            if (k, p) in sdone:
                print(f"  stack K{k:02d} p{p} | skipped (already on disk)", flush=True)
                continue
            y0 = caps[0][p:p + 1].contiguous()
            srec = dict(base, depth=k, point=p)
            srec.update(power_record(sfn, y0, args, args.seed + 500 + p))

            # Lemma 3.1 from the per-block records, when every block 0..k-1 is there.
            lr = [by_lp.get((l, p)) for l in range(k)]
            if all(r is not None for r in lr):
                def total(key, f=lambda v: v):
                    vals = [r.get(key) for r in lr]
                    return None if any(v is None for v in vals) else sum(f(v) for v in vals)
                srec["logdet_layers_sum"] = total("logdet_lu")
                srec["log_prod_layer_smin"] = total("sigma_min", math.log)
                srec["log_prod_layer_smax"] = total("sigma_max", math.log)
                for kind in ("power", "exact"):
                    srec[f"log_prod_bound_lo_{kind}"] = total(f"bound_lo_{kind}", math.log)
                    srec[f"log_prod_bound_hi_{kind}"] = total(f"bound_hi_{kind}", math.log)
                srec["layers_bound_sec_sum"] = total("bound_sec")
                svd_sum = total("svd_sec")
                srec["layers_jac_svd_sec_sum"] = (None if svd_sum is None
                                                  else svd_sum + total("jac_sec"))

            sv_path = os.path.join(stk_dir, f"{name}_K{k:02d}_p{p}_T{T}.npy")
            dense_spectrum(srec, sfn, y0, args, dtype, ana_k, sv_path, args.seed + 500 + p)
            if srec.get("logdet_layers_sum") is not None and "logdet_lu" in srec:
                srec["logdet_composition_err"] = abs(srec["logdet_lu"] - srec["logdet_layers_sum"])
            if "sigma_min" in srec and srec.get("log_prod_layer_smin") is not None:
                # Lemma 3.1(iii) with the exact per-block extremes; both must be >= 0.
                srec["prod_lo_gap"] = math.log(srec["sigma_min"]) - srec["log_prod_layer_smin"]
                srec["prod_hi_gap"] = srec["log_prod_layer_smax"] - math.log(srec["sigma_max"])
            _peak(srec, args.device)
            srecs.append(srec)
            _emit(stk_fh, srec)
            print(f"  stack K{k:02d} p{p} | jac {srec['jac_sec']:6.1f}s "
                  f"svd {srec.get('svd_sec', nan):7.1f}s | smax {srec.get('sigma_max', nan):9.3e} "
                  f"(power {srec['sigma_max_power']:9.3e}) smin {srec.get('sigma_min', nan):9.3e} "
                  f"below floor {srec.get('n_sigma_below_floor', 'n/a')} | "
                  f"center {srec.get('center_lu', nan):+.5f} "
                  f"composition err {srec.get('logdet_composition_err', nan):.2e} "
                  f"analytic err {srec.get('logdet_err', nan):.2e} | "
                  f"fd {nan if srec['fd_rel_err'] is None else srec['fd_rel_err']:.2e}",
                  flush=True)
    stk_fh.close()

    # ---------------------------------------------------------------- bounds only
    # No dense matrix anywhere below: these are the quantities that survive at the
    # training context length.
    for T2, ids2 in sorted(ids_long.items()):
        avg2 = volume_correction(model, T2)
        caps2 = linearization_points(model, ids2.to(args.device),
                                     sorted(set(layers) | ({0} if depths else set())))
        b2 = dict(base, seq_len=T2, n_dim=T2 * C)
        for layer in layers:
            blk = model.blocks[layer]
            fn = block_fn(model, layer, avg2)
            for p in range(ids2.shape[0]):
                if (T2, "layer", layer, p) in bdone:
                    continue
                x0 = caps2[layer][p:p + 1].contiguous()
                seed = args.seed + 1000 * layer + p
                brec = dict(b2, kind="layer", layer=layer, point=p,
                            logdet_analytic=analytic_logdet(model, layer, avg2, T2))
                if bounds_on:
                    brec.update(layer_bounds(blk, avg2, x0, args, seed))
                brec.update(power_record(fn, x0, args, seed + 7))
                _peak(brec, args.device)
                _emit(bnd_fh, brec)
                print(f"  T={T2} L{layer:02d} p{p} | smax(power) {brec['sigma_max_power']:9.3e} "
                      f"in {brec['sigma_max_power_sec']:.2f}s"
                      + ("" if not bounds_on else
                         f" | bound [{brec['bound_lo_power']:.3e}, {brec['bound_hi_power']:.3e}] "
                         f"in {brec['bound_sec']:.2f}s"), flush=True)
        if depths:
            for p in range(ids2.shape[0]):
                if (T2, "stack", n_layer, p) in bdone:
                    continue
                brec = dict(b2, kind="stack", depth=n_layer, point=p)
                brec.update(power_record(stack_fn(model, n_layer, avg2),
                                         caps2[0][p:p + 1].contiguous(), args,
                                         args.seed + 500 + p))
                _peak(brec, args.device)
                _emit(bnd_fh, brec)
                print(f"  T={T2} stack p{p} | smax(power) {brec['sigma_max_power']:9.3e} "
                      f"in {brec['sigma_max_power_sec']:.2f}s", flush=True)
        del caps2
        _free(args.device)
    bnd_fh.close()

    timing_summary(name, args.out_dir)

    # Only now is the run known to have finished every (layer, point). Anything
    # reading these files can distinguish "still going / died" from "done".
    manifest["complete"] = True
    manifest["n_records"] = len(recs)
    manifest["n_stack_records"] = len(srecs)
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return recs, manifest


def timing_summary(name: str, out_dir: str) -> None:
    """Mean seconds per Jacobian, full spectrum vs coarse bound, from the files on
    disk (so resumed records count too)."""
    def load(f):
        p = os.path.join(out_dir, f"{f}_{name}.jsonl")
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

    def mean(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return sum(v) / len(v) if v else float("nan")

    L, S, B = load("spectrum"), load("stack"), load("bounds")
    print(f"[{name}] timing, mean seconds per Jacobian:", flush=True)
    if L:
        print(f"  block  T={L[0]['seq_len']:<5d} full spectrum {mean(L, 'jac_sec') + mean(L, 'svd_sec'):9.2f}"
              f" (build {mean(L, 'jac_sec'):.2f} + svd {mean(L, 'svd_sec'):.2f}) | "
              f"coarse bounds {mean(L, 'bound_sec'):7.3f} | sigma_max power "
              f"{mean(L, 'sigma_max_power_sec'):7.3f} | exact sublayer norms "
              f"{mean(L, 'exact_norm_sec'):7.2f}", flush=True)
        for key in ("sigma_H", "sigma_F"):
            err = [abs(r[f"{key}_power"] / r[f"{key}_exact"] - 1) for r in L
                   if r.get(f"{key}_exact")]
            if err:
                print(f"         {key}: power vs exact, max rel err {max(err):.2e}", flush=True)
        ok = [r["bound_ok"] for r in L if "bound_ok" in r]
        if ok:
            print(f"         coarse bound holds in {sum(ok)}/{len(ok)} records", flush=True)
    for k in sorted({r["depth"] for r in S}):
        rows = [r for r in S if r["depth"] == k]
        print(f"  stack  K={k:<5d} full spectrum {mean(rows, 'jac_sec') + mean(rows, 'svd_sec'):9.2f}"
              f" (build {mean(rows, 'jac_sec'):.2f} + svd {mean(rows, 'svd_sec'):.2f}) | "
              f"sum of per-block bounds {mean(rows, 'layers_bound_sec_sum'):7.3f} | "
              f"sigma_max power {mean(rows, 'sigma_max_power_sec'):7.3f}", flush=True)
    for T2 in sorted({r["seq_len"] for r in B}):
        rows = [r for r in B if r["seq_len"] == T2 and r["kind"] == "layer"]
        srows = [r for r in B if r["seq_len"] == T2 and r["kind"] == "stack"]
        print(f"  block  T={T2:<5d} (no dense J possible) coarse bounds "
              f"{mean(rows, 'bound_sec'):7.3f} | sigma_max power "
              f"{mean(rows, 'sigma_max_power_sec'):7.3f}"
              + (f" | stack sigma_max power {mean(srows, 'sigma_max_power_sec'):7.3f}"
                 if srows else ""), flush=True)


def find_arms(ckpt_dirs: list[str]) -> dict:
    """Arms (run-dir name -> best_*.pt) across one or more directories. A directory
    may be a sweep (revparityregrevback_25000297) or a single run dir
    (revparityreg_24979051/baseline_baseline); both resolve the same way. Two dirs
    holding the same arm name is an error, not a silent pick."""
    found = {}
    for d in ckpt_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"--ckpt_dir {d!r} does not exist (cwd={os.getcwd()})")
        for nm, p in discover(d).items():
            if nm in found:
                raise SystemExit(f"arm {nm!r} found twice: {found[nm]} and {p}. Pass "
                                 f"the run dirs you want, not both sweeps.")
            found[nm] = p
    return dict(sorted(found.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", nargs="+", default=["fetched/revparityreg_24979051"],
                    help="one or more sweep dirs or run dirs; arms are their run-dir names")
    ap.add_argument("--out_dir", default="results/jacobian_spectrum")
    ap.add_argument("--val_bin", default="data/tinystories_val.bin")
    ap.add_argument("--arm", default=None,
                    help="run one arm by run-dir name (default: every arm found)")
    ap.add_argument("--arm_index", type=int, default=None,
                    help="run one arm by index into the sorted arm list (SLURM arrays)")
    ap.add_argument("--list_arms", action="store_true",
                    help="print the sorted arm list (index + name) and exit")

    ap.add_argument("--seq_len", type=int, default=16,
                    help="tokens per linearization point; N = seq_len*width, J is N^2")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="block indices (default: all)")
    ap.add_argument("--n_points", type=int, default=2,
                    help="distinct held-out story prefixes to linearize at")
    ap.add_argument("--stack_depths", nargs="*", default=[],
                    help="stack Jacobians Psi_{0,k} to take in full: integers k, 'L' "
                         "for the whole stack, 'all' for every k. Each costs one SVD "
                         "the size of a block's.")
    ap.add_argument("--bound_seq_lens", nargs="*", type=int, default=[],
                    help="extra sequence lengths for the bounds-only pass (no dense "
                         "Jacobians), e.g. 64 256 1024")
    ap.add_argument("--power_iters", type=int, default=500,
                    help="max power iterations per norm estimate")
    ap.add_argument("--power_rtol", type=float, default=1e-6,
                    help="stop a power iteration once its relative change per step "
                         "falls to this")
    ap.add_argument("--no_bounds", action="store_true",
                    help="skip the Corollary 4.5 bounds (reversible arms)")
    ap.add_argument("--exact_norms", action="store_true", default=True)
    ap.add_argument("--no_exact_norms", dest="exact_norms", action="store_false",
                    help="skip the dense reference norms of H' and F'")
    ap.add_argument("--dtype", default="float64", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256,
                    help="rows of J per batched backward; peak activation memory "
                         "scales with this, wall-clock inversely")
    ap.add_argument("--svd_driver", default=None,
                    choices=["gesvd", "gesvdj", "gesvda"],
                    help="cuSOLVER driver; default lets PyTorch choose")
    ap.add_argument("--attn_impl", default="manual", choices=["manual", "sdpa"])
    ap.add_argument("--fd_dirs", type=int, default=3,
                    help="random directions for the finite-difference check (0 skips)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_sv", action="store_true", default=True)
    ap.add_argument("--no_save_sv", dest="save_sv", action="store_false")
    ap.add_argument("--skip_svd", action="store_true",
                    help="build J and run the checks only (timing/memory probe)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep records already present for this checkpoint (with their "
                         ".npy) and compute only the rest. Safe to pass always; "
                         "without existing record files it is a no-op.")
    args = ap.parse_args()

    found = find_arms(args.ckpt_dir)
    if not found:
        raise SystemExit(f"no best_*.pt under {args.ckpt_dir}")
    names = sorted(found)
    if args.list_arms:
        for i, nm in enumerate(names):
            print(f"{i}\t{nm}\t{found[nm]}")
        return
    if args.arm_index is not None:
        if not 0 <= args.arm_index < len(names):
            raise SystemExit(f"--arm_index {args.arm_index} outside 0..{len(names) - 1} "
                             f"({names})")
        names = [names[args.arm_index]]
    elif args.arm is not None:
        if args.arm not in found:
            raise SystemExit(f"arm {args.arm!r} not found; have {sorted(found)}")
        names = [args.arm]

    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)

    # Identical prefixes for every arm -- the spectra are only comparable if the
    # linearization points come from the same tokens.
    T = args.seq_len
    stories = corpus.load_stories(args.val_bin, n=args.n_points, seed=args.seed,
                                  min_len=max(T, 32), max_len=512)
    if len(stories) < args.n_points:
        raise SystemExit(f"only {len(stories)} stories >= {T} tokens in {args.val_bin}")
    ids = torch.from_numpy(
        np.stack([s.tokens[:T].astype(np.int64) for s in stories[:args.n_points]]))
    ids_long = {T2: val_windows(args.val_bin, T2, args.n_points, args.seed)
                for T2 in sorted(set(args.bound_seq_lens))}

    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        # run_arm owns its output files and streams to them; nothing to write here.
        recs, _ = run_arm(name, found[name], ids, args, dtype, ids_long)
        print(f"[{name}] wrote {len(recs)} block records -> "
              f"{args.out_dir}/spectrum_{name}.jsonl", flush=True)


if __name__ == "__main__":
    main()
