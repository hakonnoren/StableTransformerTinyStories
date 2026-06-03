#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible.slurm — SLURM array job: vanilla baseline vs reversible variants
# on TinyStories. One array task per model (run in parallel by the scheduler).
#
# Usage (from repo root, login node):
#   sbatch job_reversible.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f output_<ARRAYJOBID>_<TASKID>.txt
#
# After ALL tasks finish, make the comparison plots (find the dir with
# `ls -dt rev1_*/ | head`):
#   python3 plot_compare.py \
#     --runs rev1_<ID>/baseline rev1_<ID>/reversible_vpb_baseline rev1_<ID>/reversible_vpm_scaling rev1_<ID>/reversible_vpb_scaling rev1_<ID>/reversible_vf_scaling \
#     --labels "Vanilla baseline" "VPB Baseline" "VPM Scaling" "VPB Scaling" "VF Scaling" \
#     --out rev1_<ID>/compare_loss.png --latex_table
# Or pull straight from W&B once they're logged:  python analyze_wandb.py --min_step 1000
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_tiny_stories
#SBATCH --account=share-ie-imf           # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-4                       # 5 tasks: 0=baseline 1=vpb_baseline 2=vpm_scaling 3=vpb_scaling 4=vf_scaling
#SBATCH --output=output_%A_%a.txt         # per-task stdout (%A=array job id, %a=task id)
#SBATCH --error=output_%A_%a.err          # per-task stderr

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
# Shared across all array tasks: $SLURM_ARRAY_JOB_ID is the SAME for every task
# in this array (unlike $SLURM_JOB_ID, which differs per task), so all three
# runs land in one directory and can be compared/plotted together.
OUTDIR=rev1_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- shared hyperparameters ----
# NOTE on width matching: RevFormer's Attn/MLP run at width N_EMBD (same as the
# baseline), and the internal reversible state is 2*N_EMBD wide (two streams).
# So passing the SAME N_EMBD/N_HEAD to baseline and reversible already gives
# matching per-block attn+MLP parameter counts -- no manual doubling. See
# revformer/README.md.
N_LAYER=6
N_HEAD=4
N_EMBD=128
BLOCK_SIZE=256
BATCH_SIZE=6
GRAD_ACCUM_STEPS=12
MAX_STEPS=10000
EVAL_INTERVAL=100 #100
EVAL_BATCHES=50
LOG_INTERVAL=50
WARMUP_STEPS=250 #150
MIN_LR_RATIO=0.1
PEAK_LR=6e-4
DATASET=tinystories

# vpm_scaling target across-depth log-volume change: total log|det| = -REV_LAMBDA
REV_LAMBDA=0.0

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
  # 10x LR on the reversible alpha/gamma scaling params (no-op for the baseline)
  --rev_scale_lr_mult 10
  --wandb --wandb_project sympformer-reversible
  # generate a text sample from an eval batch at every eval (-> stdout, samples.txt, W&B table)
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  # cheap diagnostics (grad norms, alpha/gamma, attention & representation stats) -> W&B at each eval
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
#   ID  run dir (under OUTDIR)        per-task flags
#    0  baseline                      --arch baseline
#    1  reversible_vpb_baseline       --rev_regime vpb_baseline (gamma=alpha=0, frozen)
#    2  reversible_vpm_scaling        --rev_regime vpm_scaling  (global centering, log|det|=-lambda)
#    3  reversible_vpb_scaling        --rev_regime vpb_scaling  (per-block centering, log|det|=0/block)
#    4  reversible_vf_scaling         --rev_regime vf_scaling   (no centering, volume free)
# Keep --array (top) in sync with the number of entries here.
NAMES=(
  baseline
  reversible_vpb_baseline
  reversible_vpm_scaling
  reversible_vpb_scaling
  reversible_vf_scaling
)
TASK_ARGS=(
  "--arch baseline"
  "--arch reversible --rev_regime vpb_baseline --run_name vpb_baseline"
  "--arch reversible --rev_regime vpm_scaling --run_name vpm_scaling --rev_lambda ${REV_LAMBDA}"
  "--arch reversible --rev_regime vpb_scaling --run_name vpb_scaling"
  "--arch reversible --rev_regime vf_scaling --run_name vf_scaling"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible.slurm"
  exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} host=$(hostname) pwd=$(pwd)"
log "Device: ${DEVICE}   OUTDIR: ${OUTDIR}   experiment: ${NAME}"

# ── Launch this task's training ───────────────────────────────────────────────
# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
