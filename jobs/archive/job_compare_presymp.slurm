#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_compare_presymp.slurm — array job comparing, on the ORIGINAL small setup
# (L=4, n_embd=64, block 128), five architectures:
#   baseline | reversible vpb_baseline | reversible vf_scaling | presymp ETD-AB2 | YuriiFormer
# One array task per model (run in parallel by the scheduler).
#
# Usage:    sbatch job_compare_presymp.slurm
# Monitor:  squeue -u $USER ; tail -f output_cmp_<ARRAYJOBID>_<TASKID>.txt
# Analyze:  python analyze_wandb.py --filter_config n_embd=64 max_steps=10001 --format pdf --out_dir wandb_plots_cmp
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=compare_presymp
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-4                        # 5 tasks: 0=baseline 1=vpb_baseline 2=vf_scaling 3=presymp_etd_ab2 4=yurii_lt
#SBATCH --output=output_cmp_%A_%a.txt      # per-task stdout (%A=array job id, %a=task id)
#SBATCH --error=output_cmp_%A_%a.err       # per-task stderr

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
# $SLURM_ARRAY_JOB_ID is the SAME for every task in the array -> shared OUTDIR.
OUTDIR=cmp_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- shared hyperparameters (original small setup) ----
# Width matching: RevFormer Attn/MLP run at N_EMBD (same as baseline); internal
# reversible state is 2*N_EMBD. Pass the SAME N_EMBD/N_HEAD to all. See revformer/README.md.
N_LAYER=8
N_HEAD=8
N_EMBD=64
BLOCK_SIZE=512
BATCH_SIZE=6
GRAD_ACCUM_STEPS=16
MAX_STEPS=10001
EVAL_INTERVAL=100
EVAL_BATCHES=20
LOG_INTERVAL=200
WARMUP_STEPS=150
MIN_LR_RATIO=0.4
PEAK_LR=3e-4
DATASET=tinystories

# Presymp ETD-AB2 settings (from the original job.job): learned damping schedule
# (loglin eta) + small integrator step h0. Learned scalars get a higher LR via
# --scalar_lr_mult (no-op for baseline/reversible, which have no such params).
PRESYMP_DAMPING="--eta_learnable --eta_mode loglin --eta_log_init 3 --eta_lin_init 1e-4 --eta_clip 12"
PRESYMP_H=0.01
SCALAR_LR_MULT=100

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
  --scalar_lr_mult "${SCALAR_LR_MULT}"
  # 10x LR on the reversible alpha/gamma scaling params (no-op for baseline / vpb_baseline / presymp)
  --rev_scale_lr_mult 10
  --wandb --wandb_project sympformer-reversible
  # text sample from an eval batch at every eval (-> stdout, samples.txt, W&B table)
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  # cheap diagnostics (full for baseline/reversible; token-accuracy only for presymp, which has no .blocks)
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
#   ID  run dir (under OUTDIR)        per-task flags
#    0  baseline                      --arch baseline
#    1  reversible_vpb_baseline       --rev_regime vpb_baseline (gamma=alpha=0, frozen)
#    2  reversible_vf_scaling         --rev_regime vf_scaling   (no centering, volume free)
#    3  presymp_etd_ab2               --arch presymp_etd_ab2 + damping schedule
#    4  yurii_lt                      --arch yurii_lt (YuriiFormer Lie-Trotter)
# Keep --array (top) in sync with the number of entries here.
NAMES=(
  baseline
  reversible_vpb_baseline
  reversible_vf_scaling
  presymp_etd_ab2
  yurii_lt
)
TASK_ARGS=(
  "--arch baseline"
  "--arch reversible --rev_regime vpb_baseline --run_name vpb_baseline"
  "--arch reversible --rev_regime vf_scaling --run_name vf_scaling"
  "--arch presymp_etd_ab2 ${PRESYMP_DAMPING} --presymp_h ${PRESYMP_H} --presymp_mlp_use_attn_vel --presymp_lookahead"
  "--arch yurii_lt"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_compare_presymp.slurm"
  exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} host=$(hostname) pwd=$(pwd)"
log "Device: ${DEVICE}   OUTDIR: ${OUTDIR}   experiment: ${NAME}"

# ── Launch this task's training ───────────────────────────────────────────────
# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
