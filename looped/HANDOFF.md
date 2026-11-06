# Handoff: looped conformal reasoners with constrained damping

From: the agent working in `StableTransformers_claude` (ideas/looped/), 2026-11-06.
To: the agent working in this repo (SympFormer), which knows the server, SLURM, W&B and the environments.

**Goal.** Run the looped kick/relax experiments at larger scale on the server:
1. **Sudoku-Extreme in the Equilibrium Reasoners (EqR) setting**, with the constrained-damping variants (§5).
2. **Larger versions of the toy experiments**: more width, depth, seeds, and the length-generalisation task (§6).

Background reading, in this folder:
- `docs/looped_summary.pdf`: 5 pages, results and takeaways;
- `docs/looped_conformal.pdf`: 8 pages, the maths (exact inverse, the drift lemma, the linear-part spectrum) and all
  ablations.

## 1. What is in `looped/` (self-contained, nothing else in this repo was touched)

| file | role |
|---|---|
| `model.py` | `LoopedConfig`, `LoopedGPT`, `CouplingLayer`: couplings `vanilla` / `kick` / `relax`, constraints, inverse, diagnostics |
| `attention.py` | attention + GELU; `LoopedConfig(causal=False)` gives **bidirectional** attention (needed for Sudoku) |
| `run_looped.py` | trains one model on `khop` / `listops` / `fsc` with a random iteration count, optional detached carry; evaluates r up to 128 |
| `dynamics.py` | frozen-checkpoint analysis to r = 256: shift-free state norm, step, input weight, spectral radius |
| `drift_check.py` | long horizon to r = 2048, relax's running-average force, prediction persistence (lose/gain/changed) |
| `summarize.py`, `seeds_table.py` | per-folder tables/plots; seed-aggregated table |
| `khop.py`, `listops.py` | task generators (data generated on the fly; nothing to download) |
| `tests/test_looped.py` | 20 tests: exact inverse, log-det, constraints, spectral-radius estimators |

The port was checked locally before handing over:
- `python -m pytest -q looped/tests/test_looped.py` passes (20 tests).
- Checkpoints from the source repo load here and give bit-identical logits after 32 iterations.
- A 100-step smoke run trains.

The source repo has more, and is the reference if something looks off: `StableTransformers_claude/ideas/looped/`, with
`constrained_damping.md` as the running status note and the `queue_phase*.sh` launch scripts.

## 2. The model

The state is s = (L, H), two streams of width d. The token + position embedding `e` is re-injected into every force
argument. Write a = e^{−γ}, b = e^{−α}, c = e^{−δ}. Each iteration applies:

```
A(H) = Attn(LN(H + e))        M(L) = MLP(LN(L + e))

kick:        L ← a (L + A(H))                  H ← b H + M(L)               (L already updated)
relax_δ:     L ← a L + (1−a)(c H + A(H))       H ← b H + (1−b)(c L + M(L))  (δ = 0: plain relax)
vanilla:     x ← x + A(x);  x ← x + M(x)       (looped standard block, single stream)
readout:     logits = W · LN_f(H_r)
```

**Structural facts** (all tested):
- **Exact inverse:** `CouplingLayer.inverse`.
- **Constant Jacobian determinant:** log|det| = −T·Σ(γ + α), independent of δ.
- **Plain relax (δ = 0) has a neutral mode.** Its linear part keeps L = H with eigenvalue 1, so the
  attention/MLP outputs add up along it.
- **δ > 0 or kick with γ, α > 0 contracts.** The linear-part spectral radius is then < 1, so the state stays bounded.

**Configurations that matter** (presets in `run_looped.py`):

| name in results | flags | meaning |
|---|---|---|
| `vanilla` | `--arch vanilla` | looped standard block |
| `relax3` | `--arch relax3` | relax, γ = α = 3 fixed, δ = 0 |
| `relax3_leak4` | `--arch relax3 --leak 0.25 --tag _leak4` | relax with common-mode damping δ = 1/4 |
| `kick_fixed` | `--arch kick_fixed` | kick, γ = α = 0.005 fixed (the October failure case) |
| `kick_pos` | `--arch kick_pos` | kick, γ, α = 0.05 + softplus(θ), learned, init 1 |
| `*_carry` | `--carry-max 64` | detached carry: with prob. 0.5 first run U{1..64} iterations without gradient |

**Recipe** (all toy results): d = 32, 4 heads, one coupling layer per iteration (about 18k parameters), AdamW at
learning rate 1e-3 with cosine decay, batch 64, 2000 test sequences.
- ListOps: `--steps 6000 --r-lo 2 --r-hi 8`.
- 3-hop: `--steps 8000 --r-lo 3 --r-hi 8`.

## 3. Results so far (toy scale; mean over 3 seeds unless noted)

| model | ListOps acc @8 → @128 | ListOps: correct @8, wrong @128 | 3-hop acc @8 → @128 | 3-hop: correct @8, wrong @128 |
|---|---|---|---|---|
| looped standard block | 0.918 → 0.143 | 85% | 1.000 → 0.990 (1 seed) | 1% |
| relax | 0.917 → 0.262 | 72% | 0.990 → 0.935 | 6% (one seed 16%) |
| kick, rates 0.005 (1 seed) | 0.843 → 0.193 | 79% | 0.999 → 0.155 | 85% |
| kick, positive rates (1 seed) | 0.893 → 0.892 | 0.7% | 0.913 → 0.567 | 39% |
| **kick, positive rates + carry** | 0.918 → 0.916 | 0.5% | **0.962 → 0.946** | 2.4% (one seed 6.6%) |
| **relax δ = 1/4 + carry** | **0.944 → 0.946** | 0.4% | 0.998 → 0.909 | 9.0% |

**Findings to carry over:**
1. **Without constraints, everything overthinks on ListOps.** The standard block and plain relax drift linearly: the
   running-average force converges to a nonzero vector, and the input's weight inside LN(H + e) falls like 1/r.
2. **Kick with rates 0.005 is not drift but a long transient.** It is bounded (retention 0.995, e-folding time 200
   iterations), but the state grows for about 1000 iterations before it saturates, on wrong answers. **The damping
   must be strong relative to the iteration horizon, not just positive.** Free learned kick rates go negative and
   diverge.
3. **A constraint alone gives attractors, but not necessarily the right one.** Relax with δ = 1/16 converges to
   all-wrong ListOps answers. At r = 1024, 79–100% of sequences sit at a fixed point; the rest are on short cycles,
   period 2 for kick.
4. **Constraint + carry is the working recipe.** Kick with positive rates + carry is the only configuration that is
   accurate and persistent on both tasks. Damped relax + carry is best on ListOps, but on 3-hop it loses accuracy
   gradually *inside* the carry range (r = 16–64).
5. **The 3-hop errors are hop-count errors.** The damped models' wrong 3-hop answers are almost all f²(q) or f⁴(q),
   off by one hop in either direction. Weakly damped kick's errors spread over all hop counts instead.
6. **The carry without a strong contraction hurts:** 12–33 points lost at r = 8.

**Update (phase 4, local), 3-hop, seeds 1239 and 1240:** both of the following fix damped relax + carry.
- **δ = 1 + carry:** 0.996 → 0.972, overthinking 2.5%.
- **δ = 1/4 with carry on every step (`--carry-prob 1.0`):** accuracy *rises* with iterations, 0.933 → 0.987,
  overthinking 1.2%.

A ListOps check of both is running. The latest numbers are always in
`StableTransformers_claude/ideas/looped/constrained_damping.md` (§5b).

## 4. Pitfalls (each of these bit us once)

- **Don't train the loop with reversible / memory-efficient backprop.** Contraction and exact inversion pull against
  each other: ‖(DΦ^r)^{-1}‖ ≥ e^{r·Σ(γ+α)/(2d)}. This repo's `jobs/relax/relaxL12.slurm` already measured the same
  effect for stacked relax. Use ordinary autograd over r ≤ r_hi graded iterations plus the detached carry.
- **Keep rates scalar per stream** (the default; not `--per-channel`). With channelwise rates, removing the per-token
  stream mean is no longer an exact quotient, so the shift-free spectral radius ρ_q can understate the true one.
  If you use per-channel rates, also report the full radius: `step_spectral_radius(..., quotient=False)`.
- **Equal accuracy at r_hi and at a later r does not mean unchanged answers.** Report the overthinking rate
  O = P(wrong late | correct at r_hi) and the changed-prediction share Δŷ = P(ŷ_late ≠ ŷ_{r_hi}).
  `drift_check.py --persist-only` computes both.
- **Settling does not mean correct.** Report the fixed-point fraction (step < 1e-3 at large r), not only the mean
  step, which a few cycling sequences dominate.
- **Kick's rates must stay bounded away from zero:** use `rate_constraint="positive"` with `rate_min` ≥ 0.05.
- **Pre-LN injection washes the input out when the state grows.** It is fine once the state is bounded. Post-LN
  injection (`--inject-mode post`) only roughly halved overthinking, and cost accuracy at r = 8.
- **Attention is causal by default.** Sudoku needs `causal=False`.

## 5. Sudoku-Extreme (EqR setting)

Reference: Huang, Geng & Kolter, *Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning*
(arXiv:2605.21488; code at github.com/locuslab/EqR).
- Use EqR's own data pipeline, model, training recipe and evaluation, and check them against the paper first: cell
  and exact accuracy of the EqR baseline at 16 and 64 iterations.
- Add our couplings as an alternative recurrent update inside EqR, rather than porting EqR into `looped/`.

**Context from the October summary** (seen as a PDF; the code is not in either repo). Trained from scratch, 20k
steps, EqR's recipe:

| model | cell @16 | cell @64 | exact @16 | residual @64 |
|---|---|---|---|---|
| EqR | 0.66 | 0.66 | 0.07 | 31 |
| kick, learned scalar rates | 0.64 | 0.63 | 0.03 | 201 |
| relax, rate 3 fixed | 0.65 | 0.65 | 0.05 | 24 |

EqR's Sudoku model mixes tokens with an MLP rather than attention. So there, each update applied EqR's whole
reasoning block B to one stream, alternating between the streams.

**Port:**
```
kick_pos:   L ← a (L + B(H, x))                    H ← b H + B(L, x)
relax_δ:    L ← a L + (1−a)(c H + B(H, x))          H ← b H + (1−b)(c L + B(L, x))
```
- x is EqR's input injection, and the readout comes from H.
- Use the **same B for both half-steps**, so the parameter count matches EqR. Say so if you deviate.
- EqR already trains with detached state carry. Keep its schedule as is, and record the visited depth and the
  differentiated depth.

**Arms**, 3 seeds each if the budget allows:
1. EqR, original;
2. relax, rate 3 fixed (δ = 0; reproduces the October row);
3. **relax, δ = 1/4**;
4. relax, δ = 1;
5. **kick, positive learned rates** (rate_min 0.05, init 1);
6. kick, learned rates without constraint (reproduces the October failure).

**Metrics:**
- cell and exact accuracy at r = 16, 64 and 256;
- O and Δŷ between the training horizon and 4× / 16× it;
- the per-iteration state residual at 64 and 256;
- the fixed-point fraction;
- for scalar rates, ρ_q via JVP power iteration (`LoopedGPT.step_spectral_radius` in `model.py`; port it to EqR's
  state shape).

**Hypothesis to test:** the constrained arms (3–5) keep EqR-level accuracy while settling much harder than kick's
residual of 201. Arm 3 may still show the "attractor not tied to the answer" failure, as on 3-hop.

## 6. Larger toy experiments (server)

`run_looped.py` runs on CPU with one thread per job (`--threads`), or on `--device cuda`. At d = 32 one job takes
5–20 minutes on one CPU core, so these are cheap SLURM arrays over (arch, seed).

**In priority order:**
1. **More seeds of the main table:** 5 seeds of `kick_pos_carry`, `relax3_leak4_carry`, `relax3` and `vanilla` on
   both tasks, plus `kick_pos` and `relax3_leak4` without carry.
2. **Width / depth:**
   - `--n-embd 64` and `128`, and `--n-core 2`, for the two carry winners and `relax3`;
   - scale `--steps` with width (for example 2×);
   - question: does the overthinking gap persist, and does damped relax + carry recover on 3-hop at larger width?
3. **Harder 3-hop:** `--K 4`, `--K 5` (and `--S 16`).
4. **Length generalisation** (FSC, finite-state composition):
   ```
   python looped/run_looped.py --task fsc --steps 6000 --r-lo 3 --r-hi 10 --no-pos --arch kick_pos --carry-max 64 --tag _carry --out results/looped/fsc
   ```
   - The task trains on chains of length H ≤ 6 and tests H up to 16; the `ood` test set, grouped by H, is in the
     result JSON.
   - Question: does accuracy on longer chains rise with more test iterations?
   - **Learnability first.** A local pilot at d = 32 / 6000 steps does *not* learn it:
     - 16 states: near chance;
     - 8 states / 4 functions (`--fsc-states 8 --fsc-funcs 4`): H = 1, 2 solved, H = 3 at 0.49, H ≥ 4 at chance.
     Use larger width (`--n-embd 128`+), more steps, possibly a curriculum in H, and confirm the training range is
     solved before comparing couplings.
   - Without positional embeddings, order must come from the causal mask alone. Check that the standard block learns
     the in-distribution range before comparing couplings.
5. **Carry horizon and compute:** `--carry-max 256`, and `--carry-prob 1.0`. Report compute: the carry adds on average
   carry_prob · E[r₀] forward-only iterations per step.

**Example array line**, mirroring the source repo's `queue_phase3.sh`:
```
python looped/run_looped.py --task listops --out results/looped/listops --steps 6000 --r-lo 2 --r-hi 8 \
    --seed $SEED --arch relax3 --leak 0.25 --carry-max 64 --tag _leak4_carry
```

**Analysis after a batch:**
```
python looped/seeds_table.py results/looped/listops results/looped/khop_K3
python looped/dynamics.py results/looped/listops --only <stem_seedS> ...
python looped/drift_check.py --tasks listops khop_K3
```
`drift_check.py` uses seed 1238 checkpoints for the long-horizon part, and all seeds for persistence.

## 7. Conventions and open decisions for you

- **Result layout:** `results/looped/<task>/<arch><tag>_seed<S>.{json,pt}`. The JSON holds args, config, curves,
  the final per-r accuracy, overthinking, residuals and spectral radii.
- **Data:** nothing to download for the toy tasks. Sudoku-Extreme data comes via EqR's pipeline.
- **Repo housekeeping:** `looped/` is uncommitted and new in this repo. Commit it, and decide on W&B logging,
  following this repo's conventions.
- **Write-ups:** the source PDFs (`docs/`) were built from `StableTransformers_claude/ideas/looped/paper/`. Larger
  results should go into a new table, not overwrite those.
