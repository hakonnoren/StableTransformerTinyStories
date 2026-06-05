# Plan: per-layer damped (contractive) reversible blocks

Goal: replace the "free / globally-centered γ,α" with a **per-layer, per-coordinate
contraction budget** and a learned **split** between the X- and Z-stream damping —
so no layer can be a free amplifier, while the model chooses *where* to damp.
Motivated by the corruption-robustness result (reversible uses context strongly,
so corrupted context moves the output a lot) and the shear/reciprocal-singular-
value structure (volume preservation ⇒ some directions must expand).

## Sign convention (resolved against our code)
Our block ([revformer.py](../revformer/revformer.py)):
`x ← exp(-γ)(x+H(z))`, `z ← exp(-α)z+F(x)`  ⇒  `det J_ℓ = exp(-Σ_i(γ_{ℓi}+α_{ℓi}))`.
So **contraction ⟺ γ+α > 0**, with γ,α the values returned by `get_gamma/get_alpha`.
We are in the "Cambridge γ = −g" convention ⇒ we want **γ,α ≥ 0**.
(The existing `cheap/block_logdet = -(Σγ+Σα)` will therefore read **negative** = contracting — diagnostics work unchanged.)

## v1 — split-budget parametrization (the whole scheme, kept minimal)
Two learnable per-(layer,dim) tensors `rho, u` of shape `(1,1,d)` replace
`gamma_bias/alpha_bias` for this regime. Two scalar hyperparameters `kappa_min,
kappa_max`. Then:

```
kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(rho)   # > 0  (per-coord budget)
s     = tanh(u)                                              # split in [-1,1]
gamma = 0.5 * kappa * (1 + s)                                # X-stream log-damp, >= 0
alpha = 0.5 * kappa * (1 - s)                                # Z-stream log-damp, >= 0
# guarantee: gamma + alpha = kappa > 0  (every layer & coord contracts)
```
`u=0` → symmetric; `u>0` → damp X more; `u<0` → damp Z more. This is exactly your
proposal mapped to our `exp(-γ),exp(-α)` form (your `g=-γ`, `a=-α`).

## Code changes (small, localized)
1. **`RevConfig`** (revformer.py): add `regime="damped"`; fields
   `kappa_min: float = 0.005`, `kappa_max: float = 0.08`. (`__post_init__`: reject
   `lambd != 0` for damped, like the other non-vpm regimes.)
2. **`ReversibleBlock.__init__`**: `if regime == "damped"`: register `self.rho`,
   `self.u` `(1,1,d)` instead of `gamma_bias/alpha_bias`. Init `rho ≈ -2`
   (σ≈0.12 → κ light, ~0.014) and `u = 0` (symmetric). Store `kappa_min/max`.
3. **`get_gamma/get_alpha`**: `if damped`: compute `kappa,s` and return
   `0.5κ(1±s)`; else existing path.
4. **`_effective_gamma_alpha(avg)`**: `if damped`: return `(get_gamma(),
   get_alpha())` **with no centering and no avg** (it is already self-contained,
   like `vf_scaling`). The model-level global-avg branch is skipped (treat
   `damped` like the non-centered regimes → `avg = 0`).
5. **`RevFormerModel.forward` / `_avg_corr`**: include `"damped"` in the set that
   uses `avg = 0` (no global correction).
6. **`train.py`**: add `"damped"` to `--rev_regime` choices; add
   `--rev_kappa_min` / `--rev_kappa_max`; pass into `RevConfig`.
7. **Diagnostics**: `cheap_metrics.extract_alpha_gamma` already reads
   `_effective_gamma_alpha` → `gamma_mean>0`, `alpha_mean>0`, `block_logdet<0`.
   *Optional* nicety: also log mean `kappa` and mean `|s|` (split usage).

That's the entire v1: one new regime, two param tensors, two hyperparameters,
no centering logic. ~30 lines in revformer.py + ~5 in train.py.

## Init / hyperparameters
- `rho_init ≈ -2`, `u_init = 0` (start light + symmetric).
- Start `kappa_min=0.005, kappa_max=0.08`. Sweep `kappa_max ∈ {0.02,0.05,0.08,0.12}`,
  `kappa_min ∈ {0.0, 0.005, 0.01}`.
- Expectation: too small → little robustness gain; moderate → robustness ↑ with
  small loss hit; too large → worse long-range agreement / copy.

## v2 (optional, after v1 works) — memory channels
Protect a low-contraction subspace so syntactic state survives:
```
m       = sigmoid(mem_gate)          # per-channel gate in [0,1], shape (d,)
kappa   = (1 - m) * kappa_damp + m * kappa_mem      # kappa_mem ~ 0.001
```
`mem_gate` is **shared across layers** (one `(1,1,d)` param owned by the model and
passed to blocks, or a fixed mask) so a channel is "memory" consistently through
depth. Sharper hypothesis: robust models put subject-number / entity state in the
near-volume-preserving channels and damp nuisance directions elsewhere — testable
with activation patching (E3).

## Relaxation (later) — η
v1 is already η=1: `δ ≡ (γ−α)/2 = ½κ·s`, and `|δ| ≤ ½κ` ⇒ both streams stay ≥ 0.
To allow one stream to locally expand while the pair still contracts, scale the
split: `s → η·tanh(u)` with `η>1`. Start η=1; only try η=1.5 after the strict
version works.

## Experiment this enables (4 variants)
1. volume-preserving (`vpb_baseline`),
2. globally contractive (`vpm_scaling` with `--rev_lambda > 0`),
3. per-layer split-budget (`damped`, v1),
4. per-layer + memory channels (`damped_mem`, v2).

Compare (reuse `analysis/behavior_suite.py`): corruption TV (Probe A), clean val
loss, agreement accuracy vs attractors (E4), agreement resolution depth (E1),
subject-number activation-patching persistence (E3, once built). Hoped-for result
is **not** "contractive is best" but: *per-layer contraction lowers random-
corruption sensitivity while memory channels keep syntactic dependencies alive* —
turning γ,α from stability knobs into an interpretable nuisance-vs-signal split.

## Build order (simplest first)
v1 split-budget regime → smoke-test (loss sane, `block_logdet<0`) → small
κ_max sweep on the bench config → add v2 memory channels → only then η>1.
