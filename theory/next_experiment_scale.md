# Next experiment: scale sweep toward the TinyStories paper

## Question
How does the architecture comparison (baseline vs RevFormer vs YuriiFormer) evolve
as we scale **width** toward the TinyStories paper's regime? We already saw the
reversible edge depends on shape (wins at L=6/L=8, reverses at L=20). Width is the
untested axis, and it gives us **paper-comparable eval losses**.

## Design (`job_scale.slurm`)
3 scales × 4 archs = 12 array tasks, **8 layers, context 512, 10K vocab**, fixed
token budget (same batch/steps across scales so only model size varies):

| scale | d_model | heads | ffn | ~params (10K vocab) | paper eval-loss |
|---|---|---|---|---|---|
| small | 128 | 2 | 512 | ~3M | 1.65 |
| medium | 256 | 4 | 1024 | ~8–9M | 1.38 |
| large | 512 | 8 | 2048 | ~28–30M | 1.20 |

(head_dim = 64 and ffn = 4·d_model everywhere.) Archs: `baseline`,
`reversible vpb_baseline`, `reversible vf_scaling`, `yurii_lt` (presymp excluded —
it leaks for AR-LM). W&B project `sympformer-scale`.

## Prerequisite: 10K-vocab data (`preprocess_tinystories_10k.py`)
The paper's losses are only comparable under its ~10K vocab; our current bins are
GPT-2 (50257), which also makes small models embedding-dominated. The script
trains a fresh 10K byte-level BPE on TinyStories and writes
`tinystories10k_{train,val}.bin` + `tinystories10k_tokenizer.json`. Run on the
login node, then set `VOCAB`/`EOT` in `job_scale.slurm` from its printout.

## New metrics added
- **`tokens_per_sec`** and **`peak_mem_gb`** logged to W&B (train.py) — quantifies
  the ~18% reversible compute cost and sets up the **memory hypothesis (H4)**:
  peak-mem vs scale per arch. (Reversible's O(1)-memory backward is still NOT
  implemented, so at these sizes it pays the cost without the memory benefit —
  this metric will show that explicitly.)
- **`--tokenizer_json`** so samples decode correctly under the 10K BPE.
- `--cheap_metrics` stays on → tests whether the internal signatures (mid-network
  bottleneck for baseline/yurii vs monotone expansion for reversible) persist or
  change with width.

## What to measure / predict
- **Eval loss vs scale**, per arch — does the reversible gap grow, hold, or shrink
  with width? Plot (arch − baseline) vs scale. Compare absolute losses to the
  paper's 1.65 / 1.38 / 1.20.
- **Cost frontier:** loss vs `tokens_per_sec` and vs `peak_mem_gb` — the
  compute/memory-matched view, not just param-matched.
- **Mechanism vs scale:** do the `evo_act_erank` / depth profiles keep their
  per-arch shape as width grows?

## Open decisions / caveats (flag before launching)
1. **Attention window.** Paper uses a 256 sliding-window; we use **full causal
   attention** at context 512. Fine (cleaner) for the architecture comparison,
   but NOT an exact paper replication. Implementing local/windowed attention is a
   separate, modest code change if exact replication matters.
2. **Tokenizer.** A fresh 10K BPE ≈ "GPT-Neo truncated to top 10K" but isn't
   literally it; eval losses are comparable in spirit, not to 3 decimals.
3. **Resources.** The array is sized for the large model (`--mem=24G`,
   `--time=18:00:00`); small/medium finish much sooner. Split into per-scale jobs
   if you want tighter scheduling. Verify the large model + batch 12 fits VRAM
   (reversible state is 2× wide); drop `--batch_size`/raise `--grad_accum_steps`
   if needed.
4. **Fairness.** Reversible's 2× embeddings matter less at 10K vocab but still
   exist; report compute-matched alongside param-matched.

## Not yet implemented (proposed)
- **Generation-quality proxies** (judge-free): distinct-1/2, n-gram repetition
  rate on the generated samples — cheap proxies for the paper's creativity/
  degeneration axes. Easy to add to the sample logging.
- **GPT-Eval scaffolding:** dump a fixed set of prompted continuations per run to
  a file, to later batch through an LLM judge and reproduce the paper's
  GPT-Eval table (grammar/consistency/creativity/...).
- **Windowed attention** (see decision 1).
