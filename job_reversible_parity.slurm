#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity.slurm — is the reversible win real, or just 2x embeddings?
#
# Cloned from job_reversible_bench.slurm, with the lowrank_cayley arm dropped:
# this job is only about the *simple diagonal* scaling block, run at parameter
# parity with the baseline.
#
# The problem this answers
# -----------------------
# Attn/MLP in the reversible block run at width n_embd, matching the baseline
# per block. But with --rev_embed wide (the historical default) the token/pos
# tables, ln_f and the tied lm_head sit at the full 2*n_embd state width, so the
# embedding params double. At n_embd=64 with the GPT-2 50304 vocab the embedding
# is ~94% of the baseline's params, so "wide" is 1.94x the baseline in total
# (3.43M -> 6.67M) — any win there is confounded by 2x embedding capacity.
#
# --rev_embed narrow puts those tables at n_embd (the baseline's exact shapes)
# and bridges to the 2*n_embd state parameter-free: lift dup (x = z = emb) in,
# readout sum (x + z) out. Total params then match the baseline to within the
# gamma/alpha vectors (+0 for vpb_baseline, +512 for vf_scaling — 0.015%).
#
#   ID  arch                      embed    params    optimizer
#    0  baseline                  —        3.43M     muon      (reference)
#    1  reversible_vpb_baseline   narrow   3.43M     muon      (exact parity)
#    2  reversible_vf_scaling     narrow   3.43M     muon      (parity, +512)
#    3  reversible_vpb_baseline   wide     6.67M     muon      (1.94x, historical)
#    4  reversible_vf_scaling     wide     6.67M     muon      (1.94x, historical)
#
# Read it as: 1,2 vs 0 is the architecture at equal params. 3,4 vs 1,2 is what
# the doubled embedding was worth on its own. Every run logs n_params /
# n_params_emb / n_params_blocks to its W&B summary, so parity is checkable.
#
# Muon optimizes the 2D hidden weight matrices (attn/MLP); AdamW handles
# embeddings, lm_head, norms, biases and the reversible scaling vectors. Recipe
# from the TinyStories 12L table: muon_lr 0.02, AdamW peak 6e-4, warmup 1000.
#
# Usage:    sbatch job_reversible_parity.slurm
# Analyze:  python analyze_wandb.py --filter_config n_embd=64 block_size=256 max_steps=10001 \
#               --format pdf --out_dir plots/revparity
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_parity
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-4                        # baseline + {vpb,vf} x {narrow,wide}
#SBATCH --output=output_revparity_%A_%a.txt
#SBATCH --error=output_revparity_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revparity_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- config (small/fast; identical to job_reversible_bench.slurm) ----
N_LAYER=4
N_HEAD=4
N_EMBD=64
BLOCK_SIZE=256
BATCH_SIZE=6
GRAD_ACCUM_STEPS=12
MAX_STEPS=10001
EVAL_INTERVAL=100
EVAL_BATCHES=50
LOG_INTERVAL=50
WARMUP_STEPS=1000           # TinyStories 12L recipe (matters more for Muon)
MIN_LR_RATIO=0.1
PEAK_LR=6e-4
DATASET=tinystories          # existing GPT-2 bins

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

COMMON_ARGS=(
  --data_dir data --dataset "${DATASET}"
  --out_dir "${OUTDIR}"
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult 10
  --optimizer muon
  --wandb --wandb_project sympformer-reversible
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ID -> per-task flags (run dir = OUTDIR/<arch>[_<run_name>]). Set --array=0-4.
# --rev_lift/--rev_readout only apply to --rev_embed narrow (passing them with
# wide is a hard error, since a wide table needs no bridge). Both are given
# explicitly on the narrow arms so the run record is unambiguous; dup/sum are
# also the defaults, and are the combo that leaves no dead sublayer:
#   lift=zeros  -> LN(0)=0, so block 1's attention contributes nothing at init
#   readout=x   -> z_final can't reach the logits, so the last MLP gets no grad
NAMES=(baseline_muon rev_vpb_narrow_muon rev_vf_narrow_muon rev_vpb_wide_muon rev_vf_wide_muon)

TASK_ARGS=(
  "--arch baseline --run_name baseline_muon"
  "--arch reversible --rev_regime vpb_baseline --rev_embed narrow --rev_lift dup --rev_readout sum --run_name vpb_narrow_muon"
  "--arch reversible --rev_regime vf_scaling   --rev_embed narrow --rev_lift dup --rev_readout sum --run_name vf_narrow_muon"
  "--arch reversible --rev_regime vpb_baseline --rev_embed wide --run_name vpb_wide_muon"
  "--arch reversible --rev_regime vf_scaling   --rev_embed wide --run_name vf_wide_muon"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "OUTDIR=${OUTDIR}"

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
