# Computational architectures: comparison, practical differences, and testable hypotheses

This note synthesizes what we learned from the TinyStories experiments (baseline
transformer vs. RevFormer vs. YuriiFormer vs. presymplectic SympFormer), the
internal diagnostics (`cheap_metrics.py`), and the leak investigation. It is a
set of **hypotheses with supporting hints**, not settled results — see the
caveats at the end.

## Empirical anchors (what we observed)

- **TinyStories LM, L=6 / n_embd=128 / block 256, 10k steps:** reversible beats
  baseline by ~0.02 val loss (~3% ppl); the four volume regimes are tied — the
  volume scaling is functionally inert even after engaging it with 10× LR.
- **L=20 (depth):** the gap **reverses** — baseline catches up / slightly wins.
  Baseline trains cleanly at depth (no pathology for reversible to fix).
- **L=8 / n_embd=64 / block 512:** reversible beats baseline by ~0.14;
  YuriiFormer sits in between; baseline worst (all mildly undertrained).
- **Presymplectic SympFormer leaks** future tokens in autoregressive LM
  (ppl ~2.5, 74% next-token acc) — proven causal (see leak section); excluded
  from LM comparisons.
- **Internal mechanisms (per-layer diagnostics):** the architecture-specific
  structure is *learned*, crystallizing around step ~1000–3000 (as the loss
  leaves the hockey-stick), then frozen.

## What kind of dynamical system is each?

- **Baseline (pre-LN transformer):** one residual stream, each block adds a
  correction. Learns a **"compress-in-the-middle"** strategy (mid-network rank
  bottleneck → re-expand). Relies on LayerNorm for conditioning. Simplest,
  fastest, cheapest.
- **RevFormer (reversible):** two coupled streams, an **invertible** map.
  Empirically **information-preserving** (monotone rank expansion, no
  bottleneck), **front-loaded** computation, strong gradient to early layers.
  Structurally supports **O(1)-in-depth memory** (recompute activations) —
  currently unexploited. Volume scaling (γ/α) is a learnable but **functionally
  inert** knob. Costs ~18% more/step + 2× embedding params.
- **YuriiFormer (Lie–Trotter momentum):** depth-as-**accelerated-gradient-flow**
  (heavy-ball / Nesterov), with learned per-layer step/damping scalars and
  optional **restart**. Runs in a **high-magnitude, low-activation-gradient
  regime** yet converges like baseline-or-better. Mechanically lands on the
  *same* compress-in-the-middle geometry as baseline.
- **SympFormer (presymplectic):** energy-conserving, **bidirectional** dynamics.
  Proven **non-causal by construction** (force = gradient of a global symmetric
  Hamiltonian → leaks future tokens), so it is invalid for autoregressive LM.
  It is a *non-autoregressive* architecture (home turf: LRA / encoder tasks).

## Practical differences expected

| axis | baseline | RevFormer | YuriiFormer |
|---|---|---|---|
| memory vs depth | O(depth) | **O(1) possible** | O(depth) |
| speed / step | fastest | ~+18% | ~mid (extra scalars, big norms) |
| params | fewest | +2× embeddings | +v0 embeddings |
| gradient flow | mid-bump | **strong, monotone (no vanishing)** | tiny-but-trainable |
| needs LayerNorm? | yes | maybe not | maybe not |
| built-in robustness mechanism | — | invertibility | **restart** |

## Are some more robust?

- **RevFormer looks more robust to *gradient pathology*** — no vanishing,
  front-loaded gradient, no rank collapse. Predicts robustness to **removing
  normalization** and to **higher LR / less warmup**.
- **Not automatically more robust to *depth*** — we saw a **crossover**
  (reversible won at L=6/L=8, baseline caught up/won at L=20). "Reversible is
  better deep" is *false* as stated for loss; its depth value is **memory**, not
  accuracy.
- **YuriiFormer's restart is an explicit robustness device** — should help on
  harder optimization landscapes (high LR, deeper, harder data); its
  acceleration should show as faster early convergence.

## Testable hypotheses (prioritized; most reuse existing infra)

**H1 — Input-sensitivity / perturbation robustness (cheap, run today).**
*Claim:* invertible / energy-structured maps have more bounded input→output
sensitivity, so RevFormer (and the symplectic encoder) degrade more gracefully
under token corruption than baseline.
*Test:* reuse the perturbation harness from `test_causality.py` — corrupt k% of
input tokens, measure Δ val-loss and mean ‖Δlogit‖ vs k, per model. Predicts a
flatter degradation curve for reversible.

**H2 — Depth crossover (cheap, reuses array-job + `analyze_wandb.py`).**
*Claim:* the reversible loss advantage shrinks monotonically with depth and
crosses zero around some L\*.
*Test:* depth sweep L ∈ {4,8,12,16,20,32} at fixed width, **multiple seeds**,
train to convergence; plot (rev − base) gap vs L. Two existing points (gap>0 at
8, gap<0 at 20) already suggest this. Bonus: does the rank-bottleneck depth scale
with L for baseline but stay absent for reversible? (`evo_act_erank` heatmaps).

**H3 — Stability envelope: max stable LR, no-warmup, no-LayerNorm (cheap).**
*Claim:* reversible / yurii train where baseline diverges.
*Test:* (a) LR sweep → largest non-diverging LR per model; (b) warmup=0;
(c) `normalize=False` at depth. Predicts reversible (and yurii-with-restart) have
a wider stable envelope. The most direct "which is more robust" experiment.

**H4 — Memory / throughput at scale — reversible's real win (higher effort, highest payoff).**
*Claim:* with the O(1)-memory reversible backward implemented, RevFormer trains
at depths/contexts where baseline **OOMs** on the same GPU.
*Test:* implement the reversible backward; measure peak VRAM vs depth (should
flatten) and find the depth where baseline OOMs but reversible trains. Converts
"≈ same loss" into "can train models baseline can't."

**H5 — Bottleneck-avoidance → retrieval/copy tasks (moderate effort).**
*Claim:* baseline's mid-network rank bottleneck discards information that
high-fidelity tasks need; reversible's monotone expansion preserves it.
*Test:* synthetic **copy / associative-recall / needle-in-haystack** tasks.
Predicts the reversible gap is *larger* there than on TinyStories, and that
baseline failure correlates with where its `act_erank` collapses.

**H6 — Acceleration & restart (cheap).**
*Claim:* YuriiFormer converges faster early (momentum) and `restart` improves
stability/final loss on harder settings.
*Test:* loss-vs-steps in the first ~1k steps (acceleration); ablate
`--yurii_restart {none,speed,loss}` at high LR / depth.

## The SympFormer leak (why it can't be used for AR-LM as-is)

The momentum force is `G = -∂H/∂x_ln`, the gradient of a sequence-global
**symmetric** Hamiltonian `H = Σ_i ½(B·Pᵢ·Pᵢ)/zᵢ − ½ Σ_{ij} E_{ij}` with
`zᵢ = (1/T)Σ_j E_{ij}` (causal). Even though the attention kernel `E` is
correctly causal-masked, the **gradient w.r.t. a key position m collects
contributions from every (future) query that attends to m** (`i ≥ m`). So info
flows forward through `E` but **backward through its gradient** (the `Eᵀ`
direction), injecting future-token information into `P_m → X_m → logits_m`. This
compounds across layers and is *not* the lookahead (it persists with lookahead
off) and is *not* caught by the built-in `_check_future_attention_mass` (which
only inspects `E`, which is clean). It is fine for non-autoregressive tasks where
bidirectional flow is intended. The causality test (`test_causality.py`)
confirms: baseline / RevFormer / YuriiFormer are exactly causal; presymp leaks.

## Caveats

Evidence is **single-seed, tiny (n_embd=64), mildly undertrained, one dataset**,
and cross-model absolutes are confounded by width and by compute (reversible is
~18% slower — fair claims need a **compute-matched** control, not just
param-matched). Effective rank is not directly comparable across models
(reversible state is 2× wide — compare *shapes*, not absolutes). Treat the above
as hypotheses with supporting hints.

## Suggested order of attack

`H1` and `H3` first (≈ an afternoon each with existing code; directly address
robustness), then `H2` (formalizes the depth crossover), with `H4` as the
high-value structural project that demonstrates reversible's actual advantage
(memory, not loss).
