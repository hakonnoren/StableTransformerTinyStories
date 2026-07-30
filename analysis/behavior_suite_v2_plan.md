# Behavior suite v2 — plan

Goal: turn `analysis/behavior_suite.py` from a set of hand-written spot-checks into
a benchmark that can actually **resolve behavioral differences between the
architectures** — with enough trials to put error bars on every claim, stimuli
grounded in the targeted-evaluation literature, and evaluation data derived from
the TinyStories corpus itself rather than from 17 sentences typed by hand.

Supersedes and absorbs `theory/syntactic_probing_extension.md` (its E1–E5 survive
as §5 here); read that doc first for the mechanistic→syntax hypothesis this is
built to test.

---

## 1. Diagnosis: why the current suite cannot answer the question

Run on `revparityreg_24957997` (5 arms, 12L/768d), every probe reported
differences that are inside their own noise. Exact per-cell 95% Wilson intervals:

| probe | cell | n | observed | 95% CI | CI width |
|---|---|---|---|---|---|
| E agreement d=1 | vpb_baseline | 20 | 0.85 | [0.64, 0.95] | 0.31 |
| E agreement d=1 | vf_scaling | 20 | 0.65 | [0.43, 0.82] | 0.39 |
| E agreement d=2 | baseline | 20 | 0.50 | [0.30, 0.70] | 0.40 |
| D overall | baseline | 17 | 0.82 | [0.59, 0.94] | 0.35 |
| D overall | vf_scaling | 17 | 0.71 | [0.47, 0.87] | 0.40 |
| D induction | baseline | 4 | 0.25 | [0.05, 0.70] | 0.65 |

Every interval spans every other arm's point estimate. The `0.85 vs 0.65`
agreement "effect" is **four items**. The induction family is **four items**, so
its resolution is 0.25 — the entire observed spread is one item either way.

Three separate problems, in order of severity:

1. **Item count.** Probe A's leak/context-sensitivity numbers come from *one*
   24-token prompt; probe B's depth curves from *one* 127-token sequence; probe D
   from 17 items across 5 families (2–4 items each); probe E from 20 items per
   distractor level. Required n for a ±0.02 half-width at p=0.5 is ~2400 unpaired;
   a paired design over minimal pairs needs ~250–1800 items depending on the
   disagreement rate (table in §3).
2. **Seed confound — the dominant one.** There is **one seed per arm**. The val-loss
   spread across arms is 0.010–0.019 nats; published pretraining-seed variance at
   this scale is comparable or larger (Sellam et al. 2022; Dodge et al. 2019). No
   amount of item-level power fixes this: with n=1 seed, "architecture" and "seed"
   are perfectly confounded, and *every* between-arm claim in the current report is
   uninterpretable as an architecture effect. This must be fixed in the training
   design, not the analysis.
3. **Construct validity.** Probe C measures only distinct-n / rep-4, which came out
   near-identical for all five arms — it is not measuring generation *quality*.
   Probe B's raw logit lens reported 25.17 nats at L0 for `vpm_scaling`, which is
   almost certainly a readout-basis artifact rather than a fact about the model
   (§5.1). The hand-written stimuli are also unvalidated: no check that the target
   and foil are matched for frequency, or that the target is actually the
   corpus-preferred continuation.

A fourth, run-level issue: this analysis ran on **crashed, non-converged**
checkpoints (all 5 arms died at ~17.4–18.5k of 20001 steps, val loss still
descending) at a quarter of the paper's per-step token budget, with the pre-sdpa
manual attention kernel. Whatever v2 becomes, it should be re-run on a completed
run; `revparityreg_24979051` was the live candidate at time of writing.

---

## 2. Design principles

- **Every number ships with an interval.** No point estimate in a table or figure
  without a CI, and no arm-vs-arm claim without a paired test.
- **Per-item records are the artifact.** Probes write JSONL (one row per item per
  arm per seed), and all aggregation/statistics happen in a separate pass. Stats
  can then be recomputed, re-sliced, and re-tested without re-running any model.
- **In-distribution first, OOD second.** These are 124M models trained only on
  TinyStories. BLiMP's adult register is largely OOD; it stays as a stress set,
  reported separately from the in-register suites, never pooled.
- **Prefer stimuli mined from the corpus over stimuli typed by hand.** TinyStories
  is unusually regular — stereotyped openings, 1–3 named characters, explicit
  adjective–noun bindings, formulaic closings. That structure yields thousands of
  controlled items automatically (§4). This is both more items and less
  experimenter bias.
- **Fix the item sets once, version them.** Stimulus files get a hash and a
  generation script; item selection must never depend on the models under test.
- **One primary metric per phenomenon, declared before looking.** With 67 BLiMP
  paradigms plus everything below, garden-of-forking-paths is a live threat; see
  the FDR rule in §3.

---

## 3. Phase 0 — statistical and engineering foundation

Nothing else is worth building until this exists.

**Statistics module** (`analysis/stats.py`):
- Wilson (default) and Clopper–Pearson binomial CIs.
- **Paired McNemar** for arm-vs-baseline on shared binary items — the right test
  here, since all arms see identical stimuli and only discordant pairs carry
  information.
- **Paired bootstrap over items** (10k resamples) for continuous metrics: lens
  margins, per-token loss, resolution depth.
- **Cluster bootstrap over (seed, item)** once multi-seed runs exist, so
  architecture CIs include seed variance. Report between-seed SD next to
  between-arm difference; if the former exceeds the latter, say so in the table.
- **Benjamini–Hochberg FDR** across paradigm families, with the per-family
  primary metric declared in the stimulus spec.

Power targets to size every new stimulus set (computed for 2-sided α=0.05):

| design | target | required n |
|---|---|---|
| unpaired proportion | ±0.10 half-width | ~100 |
| unpaired proportion | ±0.05 | ~385 |
| unpaired proportion | ±0.02 | ~2400 |
| paired McNemar, 10% of items discordant, 70/30 split | 80% power | ~500 items |
| paired McNemar, 10% discordant, 65/35 split | 80% power | ~880 items |
| paired McNemar, 5% discordant, 65/35 split | 80% power | ~1760 items |

**Rule of thumb adopted: ≥1000 items per phenomenon**, which lands every
in-register suite in the "detects a 3–5 point paired difference" regime.

**Seeds.** Re-run the sweep with **≥3 seeds per arm** (5 preferred). At `tiny`
(3.4M params, ~2h/arm) this is cheap and is the honest way to get an
architecture-level claim; the `small` sweep can then confirm the two or three
effects that survive at `tiny`. Recommendation: make the tiny multi-seed sweep the
primary inference vehicle and `small` the confirmation, rather than the reverse.

**Engineering** (`analysis/` becomes a package):

```
analysis/
  __init__.py
  loader.py        # load_model/discover + the 2*D bridge policy (§5.3)
  data.py          # val-bin streaming, story segmentation, POS/parse cache
  lenses.py        # raw logit lens + tuned lens
  stimuli/
    build_*.py     # generators; each writes a versioned .jsonl + manifest
    *.jsonl
  probes/
    causality.py  scaling.py  lens_depth.py  agreement.py
    entity.py     binding.py  lambada.py     induction.py  generation.py
  stats.py
  report.py        # results.jsonl -> tables + figures + report.md
```

Three engineering fixes that gate throughput:
- **Batch everything.** The current suite scores one prompt per forward. Batched
  scoring plus a KV cache for generation is a 10–50× speedup, and 1000-item
  suites are impossible without it.
- **MPS support with an equivalence gate.** MPS measured 3.5× faster than CPU on
  this hardware (23 vs 81 ms/forward at T=135). Add a `--device mps` path plus a
  test asserting cpu/mps agreement to ~1e-4 on logits and exact agreement on
  argmax decisions; run the leak probe on CPU regardless, since it reads
  differences at 1e-6.
- **Provenance manifest** per results file: git SHA, checkpoint path + hash,
  `arch`/`rev_*` args, `attn_impl`, torch version, device, seed. The
  manual-vs-sdpa kernel difference is exactly the kind of thing that silently
  invalidates a comparison, and it is currently invisible in the output.

---

## 4. Phase 1–2 — TinyStories itself as the evaluation set

**The val bin can be regenerated locally.** `preprocess_tinystories.py` pulls
`roneneldan/TinyStories` from HF and tokenizes with tiktoken; `datasets` 5.0.0 and
tiktoken are both already installed, and HF is reachable. No cluster fetch needed:

```bash
python preprocess_tinystories.py --out_dir data          # writes tinystories_{train,val}.bin
```

That unlocks everything below. Stories are EOT-delimited (`50256`), so
segmentation into individual stories is a one-liner over the uint16 array.

### Phase 1 — the existing probes, at scale on real text

Same measurements, but over **N=2000 held-out stories** instead of one prompt:

- **P1.1 Context sensitivity / corruption robustness** (replaces probe A's single
  prompt): per-story TV between clean and corrupted next-token distributions,
  swept over corruption rate, with paired bootstrap CIs. The current corruption
  curve is non-monotone (baseline: 0.288 at 5% but 0.151 at 10%), which is a
  4-prompt sampling artifact — at N=2000 it should be monotone, and if it isn't,
  that's a real finding.
- **P1.2 Leak check** stays a *test*, not a measurement: assert exactly 0.0 across
  1000 random (story, position) pairs, and fail the run if any causal arm leaks.
  Belongs in CI next to `test_causality.py`.
- **P1.3 Depth curves with CIs**: the logit/tuned lens per-layer loss, averaged
  over ~500k real tokens, with per-layer bootstrap intervals. This alone turns the
  single most interesting current result (baseline's flat-then-collapse vs the
  reversible arms' monotone descent) from one sequence into a claim.
- **P1.4 Loss stratified by linguistic type.** Per-token loss broken out by POS
  and dependency relation (spaCy over the decoded stories, cached), by
  position-in-story, and by first-mention vs repeated-mention. Where an arm's
  advantage lives is far more informative than the 0.01-nat aggregate gap.
- **P1.5 In-context-use score** (Olsson et al. 2022, adapted): mean loss at
  story-token 50 minus at token 250. Directly measures how much each arm exploits
  accumulated context, and is the natural aggregate companion to the induction
  probe in §5.4.
- **P1.6 Story-closure calibration**: P(EOT) as a function of position, and the
  KL between each arm's implied story-length distribution and the empirical one.
  TinyStories' formulaic endings make this a clean discourse-level signal that
  needs no annotation at all.

### Phase 2 — controlled stimuli mined from the corpus

Each of these is auto-generated, yields ≥1000 items, and is *in-register by
construction*. This is the core of the "rely on TinyStories structure" ask.

- **P2.1 Entity tracking / name resolution.** Stories introduce 1–3 named
  characters. Mine (first-mention, later-mention-slot) pairs; at each later slot
  score `logP(correct name) − logP(the other name introduced in the same story)`.
  Because both names appear in the same story, the contrast is controlled for name
  frequency and register — a genuine within-item minimal pair. Expect
  ~5–20k items from 2000 stories. Grounded in the entity-tracking literature
  (Kim & Schuster 2023).
- **P2.2 Attribute binding.** Mine `a/the ADJ NOUN` introductions, then later bare
  `the NOUN` references; score the bound adjective against a foil adjective
  matched on corpus frequency. This is the scaled, controlled version of the four
  hand-written "recall" items, at ~2–5k items.
- **P2.3 TinyStories-LAMBADA.** Following Paperno et al. 2016: select story-final
  content words that are recoverable from the whole story but *not* from the final
  sentence. Selection must use a **reference model independent of the arms under
  test** (an n-gram LM over TinyStories train, or off-the-shelf GPT-2) so the item
  set isn't tuned to any arm. Fix and version the resulting set. This is the
  sharpest available test of long-range context use.
- **P2.4 Discourse-feature slices** via `roneneldan/TinyStoriesInstruct`, whose
  stories carry `Summary` / `Words` / `Features` annotations (Dialogue, Twist,
  MoralValue, Foreshadowing, BadEnding, Conflict). These models were trained on
  plain TinyStories, so this is **not** an instruction-following test — the
  annotations are used as *free labels* to slice held-out loss and the P2.1–P2.3
  metrics by discourse phenomenon. "Which arm is better at dialogue vs at
  foreshadowing" is a behavioral difference the aggregate loss cannot show, and it
  costs nothing beyond the download.
- **P2.5 Template grammar, properly sized.** Keep the existing
  TinyStories-vocabulary agreement templates but expand the generator: ≥30
  subjects × both numbers × ≥6 verbs × 0/1/2/3 attractors × matched/mismatched
  attractor number, with lexical items sampled from the corpus's actual frequency
  distribution and every cell ≥1000 items. Add determiner–noun agreement,
  pronoun–antecedent gender, and reflexive binding (the Marvin & Linzen
  phenomenon inventory) using the same vocabulary.

---

## 5. Phase 3 — established datasets and methods

### 5.1 Tuned lens (do this before trusting any depth claim)

The raw logit lens reads intermediate states through the *final* unembedding,
which is only valid if the residual basis is stable across depth. It reported
25.17 and 15.46 nats at L0 for `vpm_scaling`/`vpb_scaling` — values that likely
say "this arm's early residual basis differs from its final one", not "this arm
predicts terribly at L0". The **tuned lens** (Belrose et al. 2023) fits a
per-layer affine probe on frozen activations and removes exactly this confound.
Cheap to train (one linear map per layer on val activations). Report raw and tuned
side by side; **the gap is itself the diagnostic**, and it is plausibly largest for
the volume-scaling regimes.

### 5.2 External minimal-pair suites

| suite | what it is | role here |
|---|---|---|
| **Zorro** (Huebner et al. 2021) | BLiMP-style paradigms restricted to a child-directed vocabulary | **Primary external suite** — closest register to TinyStories |
| **BLiMP** (Warstadt et al. 2020) | 67 paradigms × 1000 pairs, adult register | OOD stress set; report coverage and the in-vocab subset separately |
| **BabyLM evaluation pipeline** | BLiMP + BLiMP-supplement + EWoK etc. | Makes numbers comparable to published small-data baselines |
| **Linzen et al. 2016 / Marvin & Linzen 2018** | agreement with attractors; reflexives, NPIs | The literature version of probe E, with real counts |
| **Gulordava et al. 2018** | nonce ("colorless green") sentences | Control: separates syntax from semantic plausibility |
| **SyntaxGym / Hu et al. 2020** | surprisal-based region-wise predictions | Tests *shape* of surprisal, not just pairwise accuracy |

Scoring notes that matter: BLiMP/Zorro are scored by **full-sentence log-probability**
comparison, not the single-next-token trick the current probe D uses — so the
scorer needs a proper sequence-scoring path (and should report both length-normalized
and unnormalized, since they can disagree). Always publish **coverage**: the
fraction of pairs whose tokens fall inside the model's effective vocabulary, since
a TinyStories model's effective vocabulary is far smaller than its 50304-row
embedding table. Absolute scores will be low; only *relative* arm ordering is
meaningful, and that must be stated in the report.

`nyu-mll/blimp` resolves on HF and `datasets` is installed, so BLiMP is a
one-liner. Zorro ships as a GitHub repo of text files.

### 5.3 The 2×D state — a decision to make explicitly

The reversible arms carry a `2*n_embd` state (`x`, `z`) bridged to a `n_embd` head
by a parameter-free readout. Every depth-resolved probe therefore has a choice:
lens `x`, lens `z`, or lens `readout(x,z) = x+z`. v1 silently got this wrong (it
fed the raw 1536-d state to a 768-d head and crashed), and I fixed it to use the
model's own `_readout` — but the *scientific* question is untouched:
**which stream carries the prediction, and does that change with depth?**
Lensing `x` and `z` separately, per layer, is a finding this architecture makes
uniquely available and is arguably the most interesting single addition here.
Centralize the policy in `loader.py` and make it a flag, not an implicit default.

### 5.4 Mechanistic probes

- **Induction heads** (Olsson et al. 2022): per-head prefix-matching score plus
  copying behavior on repeated random-token sequences, alongside the P1.5
  in-context score. This replaces a 4-item family with a real measurement and
  directly addresses the one place the arms visibly differed (`vf_scaling` scored
  0.00 on induction).
- **Activation patching** (E3 in the old doc; Zhang & Nanda 2023 for practice):
  clean/corrupted pairs differing only in subject number, patch the residual
  stream at (layer, position), measure recovery. Localizes where the number
  feature lives and how long it survives — and tests whether the baseline's
  mid-network compression coincides with feature relocation.
- **Representational similarity across arms** (CKA, Kornblith et al. 2019): is the
  reversible stack computing the *same* functions as the baseline on a different
  depth schedule, or different functions? Layer×layer CKA between arms answers
  this and is the natural formalization of the "front-loading" story.
- **Structural probe** (Hewitt & Manning 2019), stretch goal: at which depth do
  parse-tree distances become linearly decodable per arm.

### 5.5 Generation quality (replaces probe C)

distinct-n/rep-4 cannot distinguish these models. Two literature options:
- **MAUVE** (Pillutla et al. 2021) — divergence between generated and human text
  distributions; needs an embedding model, but is the standard automatic
  open-ended-generation metric.
- **LLM-judge rubric** — the TinyStories paper's own protocol (Eldan & Li 2023)
  grades completions on grammar / creativity / consistency-with-the-prompt, which
  would also make results comparable to their published figures. Requires an API
  key and per-call cost, and sends generations to a third party, so it stays
  **opt-in behind a flag** and off by default.

---

## 6. Sequencing

| # | Step | Effort | Unblocks | Status |
|---|---|---|---|---|
| 1 | `stats.py` + per-item JSONL + provenance manifest | S | everything | **done** |
| 2 | Regenerate val bin locally; batched scorer; story segmentation | S | Phase 1 | **done** |
| 3 | P1.1–P1.6 (existing probes at N=2000, with CIs) | S–M | first real error bars | **done** |
| 4 | Tuned lens (§5.1) + `x`/`z`/`x+z` lens policy (§5.3) | M | any depth claim | **done** |
| 5 | P2.5 template expansion to ≥1000/cell | S | powered agreement result | **done** |
| 6 | Zorro + BLiMP with sequence scoring and coverage | M | external comparability | todo |
| 7 | P2.1–P2.3 mined stimuli (entity, binding, LAMBADA) | M | the in-register payoff | todo |
| 8 | Induction heads + activation patching + CKA | M–L | mechanism | todo |
| 9 | P2.4 discourse slices; MAUVE; optional LLM judge | M | generation quality | todo |

### Implemented (steps 1–5)

```
analysis/stats.py                      Wilson/CP intervals, exact McNemar, paired +
                                       cluster bootstrap, BH-FDR, power tables
analysis/records.py                    per-item JSONL sink; RunManifest with
                                       auto-detected comparability warnings
analysis/corpus.py                     val-bin loading, EOT story segmentation,
                                       length-sorted batching, pad masking
analysis/loader.py                     canonical checkpoint loader (mirrors train.py),
                                       LensPolicy sum/x/z, batched scoring
analysis/lenses.py                     TunedLens (Belrose-style affine translators,
                                       zero-init = raw lens), KL-to-final objective
analysis/probes/scale.py               P1.1–P1.6, token-only strata
analysis/probes/agreement.py           minimal-pair scoring + E1 resolution depth
analysis/stimuli/build_agreement.py    corpus-filtered generator -> agreement.jsonl
analysis/run_suite.py                  driver
analysis/report.py                     records -> tables with CIs + paired tests
test_analysis_v2.py                    36 checks incl. lens faithfulness
```

Notes on what was deliberately narrowed:

- **P1.4 strata are token-only** (position bucket, first-vs-repeat occurrence,
  corpus frequency band, word-initial vs continuation subword, punctuation). POS
  and dependency strata need spaCy and are additive; the token-only cut already
  separates "predicting a newly introduced token" from "predicting one the story
  established", which is where the arms turn out to differ.
- **The tuned lens is applied only under `policy='sum'`.** Its translators are fit
  to sum-bridged states, so using them on a single stream would read an `x`-state
  through a map estimated for `x+z`. Per-stream tuned lenses need their own fits;
  the `x`/`z` curves are therefore raw-lens only.
- **P1.3 subsamples token positions** (`--lens_max_pos`, default 64/story). The
  50304-way projection runs once per layer per lens per policy, so scoring every
  token dominates the cost; a per-story mean over a random subsample is unbiased
  and the item-level bootstrap absorbs the extra noise.
- **`test_analysis_v2.py` asserts the invariant that v1 violated**: reading the
  last layer through the model's own readout must reproduce the model's true loss.
  It holds to <2e-4 for all five arms — that is what certifies the 2*D bridge.
- v1's `behavior_suite.py` now imports its loader from `analysis/loader.py` rather
  than keeping a second copy of the `RevConfig` construction. A test checks the
  loader passes every field `train.py` does, so the shapes-vs-function class of bug
  cannot silently return.

**In parallel and independent of all of it: launch a ≥3-seed sweep.** Steps 1–9
raise item-level precision, but only seeds make an *architecture* claim
defensible, and that needs cluster time, not code.

**First experiment to run** (one figure, high information per unit effort):
steps 1–4 on a completed run, producing tuned-lens depth curves with bootstrap CIs
over ~500k real val tokens, per arm, with `x` and `z` lensed separately. That
tests the front-loading hypothesis properly, and the raw-vs-tuned gap tells us
whether the dramatic `vpm/vpb_scaling` early-layer numbers were ever real.

---

## 7. Dependencies

```bash
# env: rev_torch (py3.11, torch 2.12) — datasets 5.0.0, tiktoken, pandas already present
pip install scipy scikit-learn spacy
python -m spacy download en_core_web_sm      # POS/dependency strata for P1.4, P2.1-2.3
# optional: mauve-text (needs transformers) for §5.5
```

Absent from `rev_torch` today: `scipy`, `sklearn`, `spacy`, `nltk`, `transformers`.
Note `wandb` is broken in `rev_torch` (imports but exposes no `Api`); the working
one is in env `torch`, which is why `analyze_wandb.py` must run there.

---

## 8. Threats to validity, recorded up front

- **Seed confound** (§1.2) — the big one. Until multi-seed runs exist, every
  between-arm number is descriptive, not inferential. Label it as such in the
  report rather than hedging in prose.
- **Non-converged checkpoints.** The current arms crashed mid-descent; depth and
  behavior profiles may not be stable properties of the trained architecture.
  Re-run on completed runs before drawing conclusions.
- **Kernel mismatch.** `attn_impl` manual vs sdpa changes reduction order. Never
  compare across it; the manifest makes it visible.
- **Register mismatch on external suites.** Low BLiMP scores are expected and are
  not evidence about architecture; only within-suite arm ordering is.
- **Multiplicity.** ~67 BLiMP paradigms + ~10 in-register phenomena × 5 arms is a
  large comparison surface. Declare the primary metric per family, apply BH-FDR,
  and report the full grid as exploratory.
- **Item-set independence.** P2.3 selection and any filtering must use a reference
  model outside the comparison set, or the benchmark is circular.

---

## 9. References

- Tuned Lens — Belrose et al. 2023. https://arxiv.org/abs/2303.08112
- Logit lens — nostalgebraist (LessWrong, 2020).
- Stages of inference / Iterative Inference — https://arxiv.org/abs/2406.19384
- Derivational probing — https://arxiv.org/abs/2506.21861
- BLiMP — Warstadt et al. 2020. https://arxiv.org/abs/1912.00582
- Zorro / BabyBERTa — Huebner et al. 2021 (CoNLL).
- Targeted syntactic evaluation — Marvin & Linzen 2018; agreement attractors, Linzen et al. 2016; Newman et al. 2021 (https://arxiv.org/abs/2104.09635)
- Nonce-sentence control — Gulordava et al. 2018. https://arxiv.org/abs/1803.11138
- SyntaxGym / syntactic generalization — Hu et al. 2020. https://arxiv.org/abs/2005.03692
- Structural probe — Hewitt & Manning 2019. https://aclanthology.org/N19-1419/
- Activation patching practice — Zhang & Nanda 2023. https://arxiv.org/abs/2309.16042
- Induction heads — Olsson et al. 2022. https://arxiv.org/abs/2209.11895
- CKA — Kornblith et al. 2019. https://arxiv.org/abs/1905.00414
- LAMBADA — Paperno et al. 2016. https://arxiv.org/abs/1606.06031
- Entity tracking — Kim & Schuster 2023. https://arxiv.org/abs/2305.02363
- MAUVE — Pillutla et al. 2021. https://arxiv.org/abs/2102.01454
- TinyStories — Eldan & Li 2023. https://arxiv.org/abs/2305.07759
- Seed variance — Dodge et al. 2019 (https://arxiv.org/abs/1909.03004); Sellam et al. 2022, The MultiBERTs (https://arxiv.org/abs/2106.16163)
