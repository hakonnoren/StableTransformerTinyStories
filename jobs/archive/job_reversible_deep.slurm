#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_deep.slurm — DEEP (L=20) array job: vanilla baseline vs
# reversible on TinyStories. Tests whether the reversible advantage WIDENS with
# depth (where reversible architectures are expected to win). One task per model.
#
# Differences vs job_reversible.slurm: N_LAYER=20, only 3 models (baseline +
# vpb_baseline + vf_scaling — the regimes were tied at L=6, so no need for all 5),
# and run names carry an _L20 suffix so they don't collide with the L=6 runs in
# W&B (filter by config n_layer==20, or by the _L20 name suffix).
#
# Usage:        sbatch job_reversible_deep.slurm
# Monitor:      squeue -u $USER ; tail -f output_L20_<ARRAYJOBID>_<TASKID>.txt
# Analyze:      python analyze_wandb.py --runs baseline_L20 reversible_vpb_baseline_L20 reversible_vf_scaling_L20
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_deep_L20
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-2                        # 3 tasks: 0=baseline 1=vpb_baseline 2=vf_scaling (all L=20)
#SBATCH --output=output_L20_%A_%a.txt      # per-task stdout (%A=array job id, %a=task id)
#SBATCH --error=output_L20_%A_%a.err       # per-task stderr

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
# Shared across all array tasks: $SLURM_ARRAY_JOB_ID is the SAME for every task
# in this array, so all three runs land in one directory.
OUTDIR=revL20_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- shared hyperparameters (identical to the L=6 run except N_LAYER) ----
# Width matching: RevFormer Attn/MLP run at N_EMBD (same as baseline); internal
# reversible state is 2*N_EMBD. Pass the SAME N_EMBD/N_HEAD to both. See revformer/README.md.
N_LAYER=20
N_HEAD=4
N_EMBD=128
BLOCK_SIZE=256
BATCH_SIZE=6
GRAD_ACCUM_STEPS=12
MAX_STEPS=10000
EVAL_INTERVAL=100
EVAL_BATCHES=50
LOG_INTERVAL=50
WARMUP_STEPS=250
MIN_LR_RATIO=0.1
PEAK_LR=6e-4
DATASET=tinystories

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
  # 10x LR on the reversible alpha/gamma scaling params (no-op for baseline / vpb_baseline)
  --rev_scale_lr_mult 10
  --wandb --wandb_project sympformer-reversible
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
#   ID  run dir (under OUTDIR)              per-task flags
#    0  baseline_L20                        --arch baseline
#    1  reversible_vpb_baseline_L20         --rev_regime vpb_baseline (gamma=alpha=0, frozen)
#    2  reversible_vf_scaling_L20           --rev_regime vf_scaling   (no centering, volume free)
# Keep --array (top) in sync with the number of entries here.
NAMES=(
  baseline_L20
  reversible_vpb_baseline_L20
  reversible_vf_scaling_L20
)
TASK_ARGS=(
  "--arch baseline --run_name L20"
  "--arch reversible --rev_regime vpb_baseline --run_name vpb_baseline_L20"
  "--arch reversible --rev_regime vf_scaling --run_name vf_scaling_L20"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_deep.slurm"
  exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} host=$(hostname) pwd=$(pwd)"
log "Device: ${DEVICE}   OUTDIR: ${OUTDIR}   experiment: ${NAME}   (L=${N_LAYER})"

# ── Launch this task's training ───────────────────────────────────────────────
# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
