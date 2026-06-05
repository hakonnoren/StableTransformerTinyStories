# Extension plan: connecting internal dynamics to syntax

Goal: turn the two probes we like — **(A) causality/perturbation** and
**(B) logit-lens depth readout** — into a benchmark that links each
architecture's *computational style* (reversible = incremental/monotone;
baseline = compress-then-resolve-late; YuriiFormer = hold-until-the-end) to
**linguistic behavior** (when/where syntactic decisions are made, and how
robustly). This is the missing bridge: right now we have mechanistic shapes and
separate aggregate loss; we want shapes that *explain* a syntactic competence.

## Why these two tests map onto established methods

- **B (logit-lens) is a special case of the *lens* line.** The "logit lens"
  (nostalgebraist) reads each layer's hidden state through the model's own
  unembedding; the **Tuned Lens** (Belrose et al. 2023) trains a per-layer affine
  probe so the readout is faithful across the layer-wise basis change. Both
  operationalize the **Iterative Inference Hypothesis**: each residual block
  nudges the prediction toward lower loss, and "stages of inference"
  (detokenize → feature-engineer → ensemble → sharpen) unfold with depth. Our
  per-layer readout *is* this, and the architecture differences we saw are
  differences in *how prediction is built across depth*.
- **A (perturbation) is an informal *activation patch*.** Flipping an input token
  and measuring downstream change is a coarse causal intervention; the principled
  version is **activation patching / causal tracing** (interchange interventions:
  clean vs corrupted minimal pair, patch one component, measure recovery), the
  standard localization tool in mechanistic interpretability.
- **The link to syntax already exists in the literature.** Targeted syntactic
  evaluation (Marvin & Linzen 2018; agreement-attractor stimuli, Linzen et al.
  2016), the **BLiMP** minimal-pair benchmark (Warstadt et al. 2020), and
  **structural probes** (Hewitt & Manning 2019) measure *what* syntax a model
  knows. **Derivational Probing** (2025, arXiv 2506.21861) shows syntactic
  structure is built **bottom-up across layers** and — crucially — that *the
  layer at which global ("macro-syntactic") structure is integrated affects
  downstream performance*. That is exactly the variable our logit-lens depth
  curves measure, differing sharply by architecture.

## Two methodological upgrades we need first

1. **Tuned lens, not raw logit lens.** Our raw logit-lens showed YuriiFormer's
   readout loss *rising* to ~12 mid-network. That could be real suppression OR a
   basis-change artifact the logit lens can't see through (yurii runs at huge
   activation norms — a different basis). A **per-layer affine tuned lens**
   (cheap: linear maps trained on frozen activations) distinguishes the two. This
   is essential before claiming "yurii holds the prediction until the end."
2. **Principled minimal-pair stimuli, in-distribution.** Replace the ad-hoc 8
   contrasts with a **TinyStories-grammar targeted set** (simple-English
   templates), because BLiMP sentences are largely OOD for n_embd=64 TinyStories
   models. Keep BLiMP as an OOD stress set, separately.

## The unifying hypothesis

> The architecture's computational style sets *when and how robustly* syntactic
> decisions are formed across depth, and this "syntactic resolution profile"
> predicts agreement competence (especially under distractors) and perturbation
> robustness — beyond what aggregate eval loss shows.

Concretely, our findings predict: reversible resolves agreement **earlier and
more gradually**; baseline/yurii resolve **late** (yurii latest). Per derivational
probing, *earlier/cleaner integration may generalize better* — testable.

## Extension, prioritized

**E1 — Syntactic resolution-depth probe (core).**
For each targeted minimal pair (e.g. "the keys ___" → are/is), apply the lens at
every layer to the decision position and record the **margin**
`logP(correct) − logP(wrong)` vs depth. Define **resolution depth** = first layer
where the margin turns (and stays) positive. Output per phenomenon per arch:
margin-vs-layer curves + resolution-depth distribution.
*Predicts:* reversible curve rises monotonically & early; yurii flat-then-jump at
the last layer; baseline mid-late. *Tests:* does resolution depth correlate with
final agreement accuracy / loss across archs? (the derivational-probing claim).
*Effort:* small — extends `behavior_suite.py` probe B with per-pair margins.

**E2 — Tuned-lens upgrade (methodological, do alongside E1).**
Train per-layer affine probes (frozen model, val text) and recompute B/E1 with
them. Report logit-lens vs tuned-lens side by side; the gap itself is diagnostic
(large gap = strong basis change, expected for yurii). *Effort:* small–medium.

**E3 — Causal patching of the agreement signal.**
Clean/corrupted pairs differing only in subject number ("The key" vs "The keys").
Patch the residual stream at (layer ℓ, subject position) clean→corrupted and
measure recovery of correct verb-number probability → a **causal layer profile**.
Localizes *where the number feature lives* and *how far through depth it
survives*. *Predicts:* reversible — patchable across all layers (info preserved);
baseline/yurii — signal concentrated/relocated around the mid bottleneck.
*Tests:* does the mid-network rank bottleneck (baseline/yurii) coincide with where
the syntactic feature is (de)localized? *Effort:* medium (residual-stream hooks;
handle reversible 2D state and yurii (x,v)).

**E4 — Distractor / agreement-attractor battery (the payoff test).**
Classic Linzen-style stimuli: "the key(s) **to the cabinet(s)** ___" with 0/1/2
distractors of (mis)matching number. Measure agreement accuracy vs #distractors,
AND resolution depth (E1) as distractors increase. *Predicts:* reversible's
information-preservation → more robust to distractors / shallower degradation.
This is where "computational style → real linguistic competence" would show.
*Effort:* small (template generation) + reuses E1.

**E5 — Structural probe across depth (stretch).**
Hewitt–Manning structural probe per layer on dependency-parsed simple text:
where do parse-tree distances become linearly decodable, and does reversible's
monotone rank expansion encode them earlier/more cleanly? *Effort:* heavier
(needs a parser + per-layer probe training).

## Datasets / stimuli
- **In-distribution targeted set:** generate subject–verb number agreement,
  determiner–noun, and pronoun-gender minimal pairs from a TinyStories-style
  template grammar (simple vocabulary the models actually saw).
- **OOD stress:** BLiMP subset (agreement, ellipsis) — expect low absolute scores
  at this scale, use only for *relative* arch comparison.
- All single-next-token where possible (leak-safe; lets us include presymp in
  behavior even though its eval loss is invalid).

## First experiment to run
E1 + E4 on the `cmp_24693039` checkpoints (and later the scale checkpoints):
"agreement resolution depth and distractor robustness across architectures."
One figure: margin-vs-layer per arch for agreement, plus accuracy-vs-#distractors.
If reversible resolves earlier and degrades less with distractors, we have a
clean mechanistic→linguistic story; if not, that's an equally publishable null
that constrains the "info-preservation helps syntax" claim.

## References
- Tuned Lens — Belrose et al. 2023. https://arxiv.org/abs/2303.08112
- Logit lens — nostalgebraist (LessWrong, 2020).
- Stages of inference / IIH — https://arxiv.org/abs/2406.19384
- Derivational probing (layer-wise syntactic derivation; timing matters) — https://arxiv.org/abs/2506.21861
- BLiMP — Warstadt et al. 2020. https://arxiv.org/abs/1912.00582
- Targeted syntactic evaluation — Marvin & Linzen 2018; Newman et al. 2021 (https://arxiv.org/abs/2104.09635); agreement attractors, Linzen et al. 2016.
- Structural probe — Hewitt & Manning 2019. https://aclanthology.org/N19-1419/
- Activation patching best practices — Zhang & Nanda 2023. https://arxiv.org/abs/2309.16042
