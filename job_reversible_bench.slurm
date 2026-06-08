#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_bench.slurm — small/fast Muon-vs-AdamW A/B across architectures
# on the existing GPT-2 TinyStories bins (no preprocessing needed). Three archs
# x two optimizers, 10001 steps, d=64.
#
#   ID  arch                      optimizer
#    0  baseline                  adamw     (vanilla transformer, reference)
#    1  baseline                  muon      (Muon on hidden matrices + AdamW)
#    2  reversible_vpb_baseline   adamw     (volume-preserving, gamma=alpha=0)
#    3  reversible_vpb_baseline   muon
#    4  reversible_vf_scaling     adamw     (free volume)
#    5  reversible_vf_scaling     muon
#
# Muon optimizes the 2D hidden weight matrices (attn/MLP); AdamW handles
# embeddings, lm_head, norms, biases and the reversible scaling vectors. Recipe
# from the TinyStories 12L table: muon_lr 0.02, AdamW peak 6e-4, warmup 1000.
#
# In our convention reversible does NOT double n_embd (state is 2*n_embd
# internally); all runs use n_embd=64, so per-block attn+MLP params match.
#
# Usage:    sbatch job_reversible_bench.slurm
# Analyze:  python analyze_wandb.py --filter_config n_embd=64 block_size=256 max_steps=10001 \
#               --format pdf --out_dir plots/revbench
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_bench
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-5                        # 3 archs x {adamw, muon}: baseline, rev_vpb, rev_vf
#SBATCH --output=output_revbench_%A_%a.txt
#SBATCH --error=output_revbench_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revbench_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- config (small/fast) ----
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
  --wandb --wandb_project sympformer-reversible
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ID -> per-task flags (run dir = OUTDIR/<arch>[_<run_name>]). Set --array=0-5.
#   0 baseline                + adamw   (vanilla transformer, reference)
#   1 baseline                + muon
#   2 reversible_vpb_baseline + adamw   (volume-preserving)
#   3 reversible_vpb_baseline + muon
#   4 reversible_vf_scaling   + adamw   (free volume)
#   5 reversible_vf_scaling   + muon
NAMES=(baseline_adamw baseline_muon rev_vpb_adamw rev_vpb_muon rev_vf_adamw rev_vf_muon)
TASK_ARGS=(
  "--arch baseline --run_name baseline_adamw --optimizer adamw"
  "--arch baseline --run_name baseline_muon --optimizer muon"
  "--arch reversible --rev_regime vpb_baseline --run_name vpb_adamw --optimizer adamw"
  "--arch reversible --rev_regime vpb_baseline --run_name vpb_muon --optimizer muon"
  "--arch reversible --rev_regime vf_scaling --run_name vf_adamw --optimizer adamw"
  "--arch reversible --rev_regime vf_scaling --run_name vf_muon --optimizer muon"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_bench.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "OUTDIR=${OUTDIR}"

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
