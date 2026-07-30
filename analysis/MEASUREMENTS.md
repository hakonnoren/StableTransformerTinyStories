# Behavior suite v2 — what every measurement is

Reference for each probe: the question it answers, the exact computation, the
record fields it writes, a worked example on real data, how to read it, and what
it cannot tell you.

Companion documents: `behavior_suite_v2_plan.md` (why these probes, what is still
unbuilt), and the generated `report.md` / `figures/` in a results directory.

**Naming.** `P1.x` are corpus probes — the v1 measurements re-run over held-out
TinyStories at scale. `P2.x` are controlled stimuli. `E1` is the
resolution-depth probe inherited from `theory/syntactic_probing_extension.md`.
Numbers below come from `results/revparityreg_24957997` (5 arms, 12L/768d) and are
illustrative, not results to cite — that run was one seed per arm on crashed
checkpoints.

---

## How to run

```bash
# 1. build the corpus (once) — no cluster fetch needed
python preprocess_tinystories.py --out_dir data --splits val

# 2. build the stimulus set (once, versioned + hashed)
python -m analysis.stimuli.build_agreement --out analysis/stimuli/agreement.jsonl

# 3. run the probes -> per-item JSONL + manifest
python -m analysis.run_suite --ckpt_dir fetched/<run> --out_dir results/<run>

# 4. aggregate: tables with intervals and paired tests, and figures
python -m analysis.report --dir results/<run> --ref baseline_baseline
python -m analysis.plots  --dir results/<run> --format pdf

# extras
python -m analysis.eval_lenses --dir results/<run> --ckpt_dir fetched/<run>
python -m analysis.restratify  --dir results/<run>     # redefine strata, no GPU
```

Every probe writes **one JSONL row per (probe, arm, item)**. Aggregation is a
separate pass, so intervals, paired tests and re-slicing never require re-running a
model. `manifest.json` records git SHA, per-arm checkpoint digest, arch/rev args,
`attn_impl`, torch version, device and seed.

---

## The corpus

`data/tinystories_val.bin` — uint16 GPT-2 tokens, stories delimited by EOT
(`50256`). Whole split: **21,990 stories / 4,765,918 tokens**. The default
evaluation sample (`--n_stories 2000`, `seed 0`, length 32–512) is 2000 stories /
410,620 tokens, mean length 205, median 188, p5 129, p95 362.

Two corpus facts that shape several probes:

- **Effective vocabulary is small.** Only **12,615 of 50,304** embedding rows ever
  occur; rank<1000 covers 90.5% of all occurrences, rank<4096 covers 98.7%. This is
  why the frequency bands cut at 4k rather than 8k, and why external suites like
  BLiMP need a coverage statistic before their scores mean anything.
- **Stories are short.** Median 188 tokens, so any probe needing a late position
  (P1.5) loses most of the sample.

Sampling is a fixed permutation under `--seed`, so **all arms see identical items** —
that is what makes the paired tests valid. Batches are length-sorted and
right-padded; right padding is safe for a causal model (position *i* cannot attend
to *i+1*), and the loss is masked to real positions.

---

## P1.1 — Perturbation sensitivity and corruption robustness

**Question.** How much does the next-token distribution move when the context is
disturbed? Two disturbances: one flipped token (sensitivity) and a fraction of the
prefix randomised (robustness).

**Computation.** Take the first `prefix_len=64` tokens of each story (stories
longer than 65 tokens; default 300 stories). Let `p_clean` be the softmax at the
last prefix position.

- `ctx_sens` = TV(`p_clean`, `p_flip`) where `p_flip` replaces the token at
  **position 2** with `(t + 11) mod vocab_size`.
- `tv_rate_R` = mean over `n_rep=3` repetitions of TV(`p_clean`, `p_corrupt`), where
  `max(1, floor(R·64))` uniformly chosen positions are replaced by uniform random
  token ids. R ∈ {0.05, 0.1, 0.2, 0.4}.

TV is total variation, `0.5·Σ|p−q|`, so it is bounded in [0,1] and 0 means "the
disturbance changed nothing".

**Worked example.** Story 22, 64-token prefix:

> `Once upon a time, there was a lively little boy named Tim. He loved to play and run all day. One day, Tim found a big bag of oats. He believed that if he ate the oats, he would be very strong.\n\nTim at`

`ctx_sens` flips position 2, `' a'` → `'en'`. At 5% corruption (3 tokens) the prefix
becomes:

> `Once upon a time, there was a lively little boy named Tim. He loved to play and run all day. One day, Tim found a big bag of->. He believed that if he ateWhat oats, he would be very strong.\n\nTimhen a `

and at 20% (12 tokens):

> `Once upon a time, there was a lively historians boy named Tim. He loved to play and run all day. One day, Tim foundí big bag of oats longtime HeMJ that ifels ate the oats,mor would be very strong.\n\n c`

**Records.** One row per (arm, story): `ctx_sens`, `tv_rate_0.05`, `tv_rate_0.1`,
`tv_rate_0.2`, `tv_rate_0.4`.

**Reading it.** For robustness, **lower is more robust**. The curve must be monotone
in R; v1 reported baseline 0.288 at 5% but 0.151 at 10%, which was a 4-prompt
sampling artifact — at n=300 it is cleanly monotone (0.17 → 0.30 → 0.47 → 0.74) and
the arms are indistinguishable. For `ctx_sens` there is **no preferred direction**:
higher means one early token moves the prediction more, which could be better
context use or greater brittleness. Do not report it as good or bad without a task
attached.

**Limitations.** Corruption tokens are drawn uniformly from the full 50,304-row
vocabulary, so most are tokens the model never saw in training (see effective
vocabulary above) — this measures robustness to *out-of-distribution* noise, not to
plausible-but-wrong text. A corpus-frequency-matched corruption would be a
different and arguably more interesting measurement.

---

## P1.2 — Causality (a test, not a measurement)

**Question.** Does any position attend to the future? For a causal LM, logits at
position *i* must depend only on tokens 0..*i*.

**Computation.** 200 trials. Each draws a random story, takes a 64-token prefix,
picks `j ∈ [16, 64)`, replaces token *j* with `(t + 7) mod vocab_size`, and records
`max |logits[:, :j] − logits'[:, :j]|`. The reported figure is the max over trials.
Passes if `< 1e-4` for architectures declared causal; `presymp_*` is exempt because
its sequence-global Hamiltonian genuinely leaks (see `test_causality.py`).

**Records.** One row per arm: `max_abs_delta`, `n_trials`, `is_causal`, `passed`.

**Reading it.** This is pass/fail, not a quantity to compare. On the reference run
all five arms give **exactly 0.00e+00** across 200 trials each. Exact zero rather
than ~1e-7 is expected: changing a future token cannot alter any float in the
earlier positions' computation, so the result is bit-identical.

**Limitations.** Run this on **CPU**. It reads differences at the 1e-6 level, and
MPS/CUDA reduction-order nondeterminism can manufacture a nonzero max that looks
like a leak.

---

## P1.3 — Depth curves (raw and tuned lens, per stream)

**Question.** Where in depth does the prediction form, and does that differ by
architecture?

**Computation.** 150 stories, batch 4. For each block output `h_l`:

```
logits_l = lm_head( ln_f( T_l( bridge(h_l, policy) ) ) )
loss_l   = cross_entropy(logits_l, next_token)      # mean over kept positions
```

- `bridge` maps a block state to the head's width. For a narrow RevFormer the blocks
  emit the full `2·n_embd` state while `ln_f`/`lm_head` sit at `n_embd`, so a policy
  is required: **`sum`** = the model's own readout `x+z`, **`x`** or **`z`** = one
  stream. Baseline and wide-embed models already match and pass through.
- `T_l` is the identity for `lens="raw"` and the tuned-lens translator for
  `lens="tuned"`. The tuned lens is applied **only under `policy="sum"`**, because
  its translators are fit to sum-bridged states.
- `max_pos=64` token positions are sampled per story, **once per batch and shared
  across every layer/lens/policy**, so all curves are compared on identical tokens.
  A per-story mean over a random subsample is unbiased for the story mean; the
  item-level bootstrap absorbs the extra noise.

**Records.** One row per (arm, story, policy, lens, layer): `loss`, `n_pos`.

**The invariant that validates it.** At the last layer, `sum` + raw lens *is* the
model's forward pass, so its loss must equal the model's true loss. Verified to
<2e-4 for all five arms in `test_analysis_v2.py`. Any bridge bug breaks this — v1's
did.

**Worked example** (stream = sum, mean loss in nats):

| | L0 | L5 | L11 |
|---|---|---|---|
| raw — baseline | 8.21 | 5.82 | 1.24 |
| raw — vpb_base | 8.82 | 4.40 | 1.26 |
| raw — vf_scl | 8.04 | 4.57 | 1.26 |
| raw — vpb_scl | **16.99** | 4.99 | 1.27 |
| raw — vpm_scl | **26.58** | 6.92 | 1.26 |
| tuned — all five | 3.73–4.08 | 2.58–2.74 | 1.33–1.37 |

**Reading it.** The raw lens spans 3.3× at L0; the tuned lens collapses that to 9%.
The raw-lens spread was therefore **a change of basis, not a difference in
information content**. Always report both — the gap is the diagnostic, and the raw
curve alone will produce a false mechanistic story.

**Limitations.** The tuned lens here is **undertrained at deep layers**: at L11 the
optimal translator is provably `A=0` (raw is exact there), yet the fit sits at
rel‖A‖ ≈ 0.20–0.23 and scores 1.33–1.37, *worse* than raw's 1.24–1.27 — impossible
at convergence. Treat tuned deep-layer values as an upper bound; raise
`--tuned_steps` from 300 to ~2000. The `x`/`z` curves are raw-lens only, and at L11
the head has never seen a single stream, so some rise is expected by construction.

---

## The tuned lens itself

**Question.** How much of a layer's apparent unreadability is a rotated basis
rather than missing information?

**Computation.** Per layer, an affine translator `T_l(h) = h + A_l h + b_l` with
`A_l, b_l` initialised to **zero**, so the lens starts exactly at the raw logit lens
and can only improve on it (Belrose et al. 2023, arXiv 2303.08112). Trained on
held-out stories disjoint from the evaluation set, minimising
`KL(p_final ‖ p_lens)` — matching the model's *own* final distribution, which is
what makes it a lens on the model rather than a probe fit to the labels. The model
is frozen; only translators receive gradient. `--tuned_pos` positions are subsampled
per step because the 50,304-way projection per layer dominates cost.

**Records** (`analysis.eval_lenses` → `lens_quality.jsonl`): per (arm, layer,
lens∈{raw,tuned}) a `kl`; plus per (arm, layer) `lens="translator"` rows carrying
`rel_A` = ‖A_l‖_F / ‖I‖_F and `bias_norm`. `rel_A = 0` means "already in the final
basis"; `1` means the correction is as large as the identity map.

**Worked example.** Mean over layers, on held-out text:

| arm | raw KL | tuned KL | removed by an affine map |
|---|---|---|---|
| baseline | 4.01 | 1.16 | 71% |
| vf_scaling | 3.01 | 1.24 | 59% |
| vpb_baseline | 2.89 | 1.19 | 59% |
| vpb_scaling | 5.91 | 1.38 | 77% |
| vpm_scaling | 7.84 | 1.30 | 83% |

Per-layer `rel_A` at L0–L3 is ~0.47–0.56 for `vpb_scaling`/`vpm_scaling` versus
~0.31–0.37 for the other three — the two arms that looked worst under the raw lens
are exactly the two whose early basis is furthest from their final basis.

**Reading it.** Raw KL at the final layer is **exactly 0** by construction (the
model's own readout). On a log axis that point vanishes, so the plot annotates it;
if it is *not* zero, the bridge is wrong.

---

## P1.4 — Per-token loss, stratified

**Question.** *Where* does a loss difference live? A 0.01-nat aggregate gap says
nothing about which tokens it comes from.

**Computation.** One forward pass per arm over 2000 stories. Per-token loss is
`−log p(true next token)`. Each token is assigned to strata computed from token ids
alone — no parser needed:

| stratum | definition |
|---|---|
| `pos_A_B` | position within the story, buckets 0–16, 16–32, 32–64, 64–128, 128–256, 256–512 |
| `first` / `repeat` | has this token id appeared **earlier in the same story**? Proxy for novel vs established material |
| `freq_{top16,top128,top1k,top4k,gt4k}` | corpus frequency rank, ranked over the **whole val split** (not the sample) |
| `word_init` / `subword` | does the decoded token begin with a space? |
| `punct` | decoded token is entirely punctuation |

**Worked example.** Story 22, first 22 target tokens:

| i | token | text | first? | word-init | punct | freq band | rank |
|---|---|---|---|---|---|---|---|
| 1 | 2402 | `' upon'` | yes | yes | no | top128 | 58 |
| 2 | 257 | `' a'` | yes | yes | no | top16 | 6 |
| 4 | 11 | `','` | yes | no | **yes** | top16 | 4 |
| 7 | 257 | `' a'` | **no** | yes | no | top16 | 6 |
| 8 | 29696 | `' lively'` | yes | yes | no | **top4k** | 1967 |
| 19 | 1057 | `' run'` | yes | yes | no | top1k | 302 |
| 22 | 13 | `'.'` | **no** | no | yes | top16 | 0 |

**Records.** One row per (arm, story) carrying `mean_loss`, `n_tokens`,
`story_len`, and for every stratum a `loss_<stratum>` plus its token count
`n_<stratum>`. Per-story rows are exactly what a paired bootstrap over items needs.
Raw per-token log-probabilities also go to `tokens_<arm>.npz`
(`logprob`, `token`, `pos`, `story`).

**Reading it.** Deltas vs a reference arm, per stratum, with paired-bootstrap CIs —
that is `figures/p14_strata_delta.pdf`. Example: `vpb_baseline` sits at +0.007 on
the 16 most frequent tokens and +0.009 on repeated tokens but +0.027 on first
occurrences, i.e. its deficit is concentrated in *novel* material; `vf_scaling`
instead pays a flat ~+0.02 everywhere.

**Limitations.** `first`/`repeat` keys on **token id**, not on entity or lemma, so
`' a'` counts as "established" the second time it appears. It is a cheap proxy for
first-mention, not a coreference annotation. POS and dependency strata need spaCy
and are additive (plan P1.4). Because the strata are computed at write time,
changing them normally means re-running — which is what `analysis/restratify.py`
avoids by recomputing from the `.npz`.

---

## P1.5 — In-context-use score

**Question.** Does the model predict better later in a story, i.e. does it exploit
accumulated context? Adapted from the in-context-learning score of Olsson et al.
2022 to TinyStories' ~200-token documents.

**Computation.** Per story, `icl_score = loss_pos_32_64 − loss_pos_256_512`. Only
stories having **both** buckets contribute.

**Records.** One row per (arm, story): `icl_score`, `early`, `late`.

**Reading it.** Positive means later positions are predicted better. On the
reference run every arm sits at ≈0.000 with CIs spanning zero (q ≥ 0.87) — a clean
null: none of these models measurably predicts better at token 256–512 than at
32–64, and the arms do not differ.

**Limitations.** **n collapses to 307 of 2000** stories, because the median story is
188 tokens and few reach position 256. That is the dominant limitation of this
probe as configured. Move the late bucket to 128–256 to recover most of the sample
at the cost of a shorter lever arm.

---

## P1.6 — Story-closure calibration

**Question.** Does the model know when a story ends? TinyStories' endings are
formulaic, so this needs no annotation.

**Computation.** From the same forward pass, `P(EOT)` is read at every position:

- `p_eot_at_end` — P(EOT) at the position whose true target is the final EOT.
- `p_eot_midstory_mean`, `p_eot_midstory_max` — over all earlier positions
  ("false-stop pressure").

**Worked example.** One baseline record for story 6291 (78 tokens):
`p_eot_at_end = 0.488`, `p_eot_midstory_max = 0.0255`,
`p_eot_midstory_mean = 0.000346`. So it puts ~49% on ending exactly where the story
ends, and never exceeds 2.6% anywhere before that.

**Reading it.** Higher `p_eot_at_end` with low mid-story mass is better calibration.
Both quantities are probabilities, so they share **one axis** — never plot them on
twin axes. Notably this dissociates from loss: three reversible arms score 0.755–0.767
against the baseline's 0.737 despite *worse* overall loss.

---

## P2.5 — Subject–verb agreement

**Question.** Does the model prefer the correctly inflected verb, and how far does
an intervening attractor of the opposite number degrade that?

**Stimulus construction** (`analysis/stimuli/build_agreement.py`). Templates
`The <subject> [<prep> the <attractor>]×k` with k ∈ {0,1,2,3}. Two rules that matter:

1. **Lexical items are filtered against the corpus.** A noun or verb form is kept
   only if it is a **single GPT-2 token with a leading space** *and* occurs ≥
   `--min_count` (50) times in the val split. 39 of 55 noun pairs and 20 of 20 verb
   pairs survived; the manifest records every rejection with a reason
   (`woman/women: rare(0)`, `bunny/bunnies: multi_token`,
   `fish: no_number_contrast`). Scoring a pair whose target the model never saw
   measures vocabulary coverage, not agreement.
2. **Target and foil are both single tokens**, so the contrast is one next-token
   decision — leak-safe (nothing after the decision point is in the input) and free
   of full-sentence length-normalisation ambiguity.

Attractors take either the **opposite** number to the head subject (the hard
attractor case, Linzen et al. 2016) or the **same** number (control). Result:
**10,920 items, 1560 per cell** across 7 cells, versus v1's 20 per cell.

**Worked examples.**

| cell | prompt | target | foil |
|---|---|---|---|
| `attr0_match` | `The cat` | `' is'` | `' are'` |
| `attr0_match` | `The cars` | `' were'` | `' was'` |
| `attr2_mismatch` | `The dogs near the plant by the child` | `' go'` | `' goes'` |
| `attr2_mismatch` | `The cake near the children by the hats` | `' says'` | `' say'` |
| `attr3_mismatch` | `The toy near the cows by the stars beside the doors` | `' sits'` | `' sit'` |

**Computation.** Batched, right-padded, logits gathered at the last real prompt
position. `correct = 1[logP(target) > logP(foil)]`; `margin = logP(target) −
logP(foil)`. The margin keeps magnitude and so needs fewer items than accuracy for
the same power.

**Records.** One row per (arm, item): `correct`, `margin`, `logp_target`,
`logp_foil`, `cell`, `n_attractors`, `attractor_match`, `number`, `paradigm`.

**Reading it.** Accuracy with Wilson intervals, paired arm-vs-arm by exact McNemar,
split by cell. Reference run: `vpb_baseline` 0.847 [0.840, 0.854] overall against
the baseline's 0.689, and 0.671 vs 0.484 under mismatched attractors — the largest
and best-powered effect in the whole suite, belonging to the arm that ranks *worst*
on language-modeling loss.

**Limitations.** Templates are synthetic and short, so they sit at the edge of the
TinyStories register even with corpus-filtered vocabulary; a prompt like
`The toy near the cows by the stars beside the doors` is grammatical but not
story-like. Treat these as *targeted* stimuli, and let the external suites
(Zorro/BLiMP, plan §5.2) and the corpus-mined probes (plan P2.1–P2.3) carry
naturalness. A test asserts the prompt never contains the target or foil verb, so
the answer cannot leak lexically.

---

## E1 — Resolution depth

**Question.** At which layer does the agreement decision get made and *stay* made?

**Computation.** On a 600-item subsample, read the margin `logP(target) −
logP(foil)` at the decision position through every layer's lens.
`resolution_depth` = the first layer *l* whose margin is positive **and remains
positive through the last layer**; `n_layer` if that never happens.
`resolved = depth < n_layer`. Computed for both the raw and tuned lens.

The "and stays" clause is the point: a margin that flickers positive mid-stack and
then goes negative has not resolved anything.

**Worked example.** Item `agr000015`, cell `attr0_match`, tuned lens — per-layer
margins:

```
baseline      -1.78 +0.22 +0.44 -0.22 +0.07 +0.16 +0.88 +0.02 +0.43 +1.72 +3.96 +3.52   depth=4
vf_scaling    -0.70 +0.60 +0.67 +0.95 +0.42 +1.23 +0.65 +0.63 +0.05 +2.03 +0.17 +3.33   depth=1
vpb_baseline  -0.05 +0.01 -0.01 -0.43 -0.05 -0.09 -0.16 -0.89 -0.61 +1.21 +2.84 +2.01   depth=9
vpb_scaling   -0.19 -0.11 -0.07 -0.11 +1.82 +0.57 -0.73 +0.05 +0.13 +1.18 +1.23 +2.53   depth=7
vpm_scaling   +0.20 +0.68 +0.87 +0.67 +1.76 +1.48 +0.33 -0.86 -0.46 +1.60 +0.69 +3.08   depth=9
```

Note `vpm_scaling`: positive from L0 but it dips negative at L7–L8, so its
resolution depth is 9, not 0. `baseline` is positive at L1–L2, dips at L3, and only
holds from L4.

**Records.** One row per (arm, item, lens): `resolution_depth`, `resolved`,
`final_margin`, and the full `margins` array.

**Reading it.** Lower = resolves earlier. **Use the tuned lens**: raw-lens depths
span 6.73–8.07 across arms, but under the tuned lens four of five converge to
5.86–6.35 — most of the apparent architectural spread was basis change, the same
correction as in P1.3.

**Limitations.** Inherits the tuned lens's deep-layer undertraining. `resolved` is
also a censored quantity — items that never resolve are assigned `depth = n_layer`,
which pulls the mean toward 12 by an amount that depends on the unresolved fraction
(0.64–0.78 resolved here), so report `resolved` alongside `resolution_depth` and
never the depth mean alone.

---

## Statistics

All arms see identical items, so every comparison is **paired**:

| outcome | per-arm interval | arm vs arm |
|---|---|---|
| binary (`correct`, `resolved`) | Wilson score | exact **McNemar** (binomial on discordant pairs) |
| continuous (loss, margin, TV, KL, depth) | bootstrap over items | **paired bootstrap** over items |

Pairing is what buys the power: only items where two arms *disagree* carry
information, so a paired design needs ~500 items where an unpaired one needs ~2400
for the same 5-point difference. Multiplicity across arms within a metric is handled
by **Benjamini–Hochberg FDR**; the reported `q` is what to threshold, not `p`.

Two reading rules:

- **The bootstrap p-value floors at `1/n_boot`.** A reported `q = 0.0005` at
  `n_boot = 2000` means "below this bootstrap's resolution", not literally 0.0005.
- **`cluster_bootstrap_ci` returns no interval for a single cluster.** With one seed
  per arm that is the honest output, not a bug.

Power targets used to size the stimulus sets (2-sided α=0.05):

| design | target | n |
|---|---|---|
| unpaired proportion | ±0.10 half-width | ~100 |
| unpaired proportion | ±0.05 | ~385 |
| unpaired proportion | ±0.02 | ~2400 |
| paired McNemar, 10% discordant, 70/30 | 80% power | ~500 |
| paired McNemar, 5% discordant, 65/35 | 80% power | ~1760 |

---

## Provenance and the warnings that bound every number

`manifest.json` is checked automatically and its warnings are printed before any
table. Three fire on the reference run, and each limits what the numbers support:

- **One seed per arm.** Architecture and seed are perfectly confounded, so all
  between-arm differences are *descriptive*, not inferential. Fixing this needs
  training runs, not analysis.
- **Truncated checkpoints.** Every arm crashed at ~87% of `max_steps` with val loss
  still descending. Worse, the arms stopped at *different* steps — and val loss was
  falling at 0.026–0.030 nats/1000 steps, so `vpb_baseline`'s 750 extra steps are
  worth ~0.02 nats, **larger than every between-arm difference measured**. Compare
  arms at equal steps or not at all.
- **Kernel.** `attn_impl` manual vs sdpa changes reduction order; never compare
  across it. The manifest makes the mix visible.

---

## Probe index

| ID | Measures | Unit of observation | Default n | File |
|---|---|---|---|---|
| P1.1 | context sensitivity, corruption robustness | story | 300 | `probes/scale.py` |
| P1.2 | causality leak (test) | arm | 200 trials | `probes/scale.py` |
| P1.3 | per-layer early-exit loss | story × layer × lens × stream | 150 | `probes/scale.py` |
| P1.4 | stratified per-token loss | story | 2000 | `probes/scale.py` |
| P1.5 | in-context-use score | story | 307 usable | `probes/scale.py` |
| P1.6 | story-closure calibration | story | 2000 | `probes/scale.py` |
| P2.5 | subject–verb agreement | item | 10,920 | `probes/agreement.py` |
| E1 | resolution depth | item × lens | 600 | `probes/agreement.py` |
| — | tuned-lens quality | layer × lens | 12 × 2 | `eval_lenses.py` |

Not yet built (plan §6, steps 6–9): Zorro/BLiMP with sequence scoring and coverage;
corpus-mined entity tracking, attribute binding and TinyStories-LAMBADA; induction
heads, activation patching, CKA; discourse-feature slices and generation quality.
