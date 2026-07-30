"""
All singular values of one transformer block's input-output Jacobian.

What is computed
----------------
A block maps a state to a state of the same shape. Flattening (T, C) -> N = T*C,
that map's Jacobian at a linearization point x0 is a dense N x N matrix

    J[i, j] = d block(x)_i / d x_j   at  x = x0

and this script builds J explicitly, then takes its full singular-value spectrum.

    arch         per-token width C      N at T tokens
    baseline     n_embd     (768)       T *  768
    reversible   2*n_embd  (1536)       T * 1536   (the (x, z) state)

The two archs therefore live on spaces of different dimension at the same T. Only
per-dimension quantities (mean_log_sigma, frac_sigma_gt1, quantiles, the shape of
the log-spectrum) compare directly across them; sum_log_sigma / frob / erank scale
with N and do not. `n_dim` is on every record so this stays checkable.

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
Jacobian, and nothing here licenses extrapolating the spectrum to T=1024.

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

Three independent correctness checks per Jacobian
-------------------------------------------------
1. `fd_rel_err` - central finite differences along random unit directions vs J@v.
   Catches any AD/reshape/transposition error. Expect ~1e-7 or better in fp64.

2. `causal_resid` - max |dy_t/dx_s| over s > t. Both blocks are causal, so this is
   0 up to rounding. Catches a mask error or an off-by-one in the (T,C,T,C) view.

3. `logdet_err` (reversible only) - the reversible block factors as
   S2 o Lambda o S1 with both shears block-unit-triangular (x' = x + Attn(LN(z))
   depends on z alone; z' = ... + MLP(LN(x')) on x' alone), so det = det Lambda
   exactly and

       log|det J| = -T * (sum_i gamma_i + sum_i alpha_i)      (post-centering)

   which must equal sum_i log sigma_i. This is an exact analytic identity on the
   full N x N spectrum, and it is the very quantity the regime sweep varies
   (vpb_baseline: 0; vpb_scaling: 0 per block; vpm_scaling: -lambda over the
   stack; vf_scaling: free). If it holds to ~1e-9 relative, the whole pipeline
   is validated. The baseline block has no closed form, so sum_log_sigma is
   reported without a reference.

Usage
-----
    # one arm, all 12 layers, 2 linearization points (what the SLURM job runs)
    python -m analysis.jacobian_spectrum \
        --ckpt_dir fetched/revparityreg_24979051 \
        --out_dir results/jacobian_spectrum --arm reversible_vf_scaling_narrow \
        --seq_len 16 --n_points 2 --device cuda

    # cheap CPU smoke test
    python -m analysis.jacobian_spectrum --seq_len 4 --n_points 1 --layers 0 \
        --device cpu --out_dir /tmp/jac

Output: one JSONL record per (arm, layer, point) in <out_dir>/spectrum_<arm>.jsonl,
the full spectrum in <out_dir>/sv/<arm>_L<l>_p<p>_T<T>.npy (float64, length N),
and <out_dir>/manifest_<arm>.json with the token ids and settings used.
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
from revformer.revformer import (                                # noqa: E402
    LinearMixedReversibleBlock,
    ReversibleBlock,
)

DTYPES = {"float64": torch.float64, "float32": torch.float32}
_REV_BLOCKS = (ReversibleBlock, LinearMixedReversibleBlock)


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

    Both couplings are shear o scale o shear with det-1 shears, so the whole
    block's log|det| is T times the per-token log-scale of the middle map.
    """
    blk = model.blocks[layer]
    if isinstance(blk, LinearMixedReversibleBlock):
        return float(blk.linear_map.logdet(avg)) * T
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
        "frob": float(torch.sqrt(tot)),
        "stable_rank": float(tot / s2[0].clamp_min(1e-300)),
        "entropy_erank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum())),
        "n_sigma_gt1": int((sv > 1).sum()),
        "frac_sigma_gt1": float((sv > 1).double().mean()),
    }
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        out[f"sigma_q{int(q * 100):02d}"] = float(torch.quantile(sv, q))
    return out


# ------------------------------------------------------------------ driver
def _gb(n: int, dtype) -> float:
    return n * n * torch.finfo(dtype).bits / 8 / 1024 ** 3


def run_arm(name: str, path: str, ids: torch.Tensor, args, dtype) -> tuple[list[dict], dict]:
    t_load = time.time()
    model, meta = load_model(path, device="cpu")
    n_switched = set_attn_impl(model, args.attn_impl)
    model = model.to(dtype).to(args.device).eval()
    T = ids.shape[1]
    n_layer = len(model.blocks)
    layers = args.layers if args.layers else list(range(n_layer))
    bad = [l for l in layers if not 0 <= l < n_layer]
    if bad:
        raise SystemExit(f"{name}: layers {bad} out of range (n_layer={n_layer})")

    ids_dev = ids.to(args.device)
    avg = volume_correction(model, T)
    caps = linearization_points(model, ids_dev, layers)
    C = caps[layers[0]].shape[-1]
    n = T * C
    print(f"[{name}] arch={meta['arch']} regime={meta['regime']} n_layer={n_layer} "
          f"T={T} C={C} -> N={n}  (J = {_gb(n, dtype):.2f} GB in {args.dtype}, "
          f"attn_impl={args.attn_impl} on {n_switched} modules, "
          f"loaded in {time.time() - t_load:.1f}s)", flush=True)

    recs = []
    for layer in layers:
        ana_ld = analytic_logdet(model, layer, avg, T)
        fn = block_fn(model, layer, avg)
        for p in range(ids.shape[0]):
            x0 = caps[layer][p:p + 1].contiguous()
            t0 = time.time()
            J, fb = build_jacobian(fn, x0, chunk=args.chunk, verbose=args.verbose)
            t_jac = time.time() - t0

            t0 = time.time()
            fd_err, fd_eps = fd_check(fn, x0, J, n_dirs=args.fd_dirs, seed=args.seed + p)
            cz = causal_resid(J, T, C)
            t_chk = time.time() - t0

            rec = {
                "arm": name, "arch": meta["arch"], "regime": meta["regime"],
                "layer": layer, "point": p, "seq_len": T, "width": C,
                "dtype": args.dtype, "attn_impl": args.attn_impl,
                "fd_rel_err": fd_err, "fd_eps": fd_eps, "causal_resid": cz,
                "fallback_rows": fb,
                "jac_sec": round(t_jac, 2), "check_sec": round(t_chk, 2),
            }

            if args.skip_svd:
                rec["n_dim"] = n
                del J
            else:
                t0 = time.time()
                sv = svdvals(J, driver=args.svd_driver)
                del J
                t_svd = time.time() - t0
                rec.update(spectrum_stats(sv))
                rec["svd_sec"] = round(t_svd, 2)
                rec["logdet_analytic"] = ana_ld
                if ana_ld is not None:
                    rec["logdet_err"] = abs(rec["sum_log_sigma"] - ana_ld)
                    rec["logdet_rel_err"] = rec["logdet_err"] / max(abs(ana_ld), 1.0)
                if args.save_sv:
                    sv_dir = os.path.join(args.out_dir, "sv")
                    os.makedirs(sv_dir, exist_ok=True)
                    f = os.path.join(sv_dir, f"{name}_L{layer:02d}_p{p}_T{T}.npy")
                    np.save(f, sv.detach().double().cpu().numpy())
                    rec["sv_file"] = os.path.relpath(f, args.out_dir)
                del sv

            if args.device.startswith("cuda"):
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            recs.append(rec)

            nan = float("nan")
            ld = rec.get("sum_log_sigma", nan)
            ld_txt = "n/a" if ana_ld is None else \
                f"{ana_ld:+.4f} err {rec.get('logdet_err', nan):.2e}"
            print(f"  L{layer:02d} p{p} | "
                  f"jac {rec['jac_sec']:6.1f}s svd {rec.get('svd_sec', nan):7.1f}s | "
                  f"smax {rec.get('sigma_max', nan):9.3e} "
                  f"smin {rec.get('sigma_min', nan):9.3e} | "
                  f"logdet {ld:+.4f} (analytic {ld_txt}) | "
                  f"fd {nan if fd_err is None else fd_err:.2e} "
                  f"causal {cz:.2e}", flush=True)

    manifest = {
        "arm": name, "ckpt": path, "meta": meta,
        "seq_len": T, "width": C, "n_dim": n, "layers": layers,
        "n_points": int(ids.shape[0]), "dtype": args.dtype,
        "attn_impl": args.attn_impl, "chunk": args.chunk,
        "svd_driver": args.svd_driver, "device": args.device,
        "seed": args.seed, "val_bin": args.val_bin,
        "token_ids": ids.tolist(),
        "torch": torch.__version__,
    }
    return recs, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default="fetched/revparityreg_24979051")
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
    args = ap.parse_args()

    found = discover(args.ckpt_dir)
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

    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        recs, manifest = run_arm(name, found[name], ids, args, dtype)
        with open(os.path.join(args.out_dir, f"spectrum_{name}.jsonl"), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, f"manifest_{name}.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[{name}] wrote {len(recs)} records -> "
              f"{args.out_dir}/spectrum_{name}.jsonl", flush=True)


if __name__ == "__main__":
    main()
