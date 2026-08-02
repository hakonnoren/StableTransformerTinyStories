#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL12.slurm — 30000-step run at 12L ('small's own depth, the 12L anchor for
# the 12L/24L/48L/96L depth ladder), all 4 arms (baseline, vpb_baseline+segL,
# vf_scaling+segL, vpm_scaling+segL) at matched geometry (12H/768d/1024block).
# segL kept throughout. Logs to sympformer-reversible-revback, the SAME
# project job_reversible_parity_regimes_revback.slurm already uses for its
# single-regime (vpb_baseline) segL validation at this depth -- this job adds
# baseline + the other two regimes into that same project rather than
# starting a fourth one, so the 12L anchor point lives in one place.
#
# Batch/accum: 24x20 for EVERY arm (491,520 tokens/step) -- this is exactly
# job_reversible_parity_regimes_revback.slurm's 'small' case, and its own
# vf_scaling+segL run at these exact settings just measured 58,000s for
# MAX_STEPS=10001 = 5.8 s/step. That's a REAL data point, not extrapolated.
#
# Time budget -- scaled from that real number, not a depth-scaling estimate:
#   5.8 s/step x 30000 steps = 174,000s = ~48.3h (proportional scaling of a
#   measured end-to-end wall time, which already bakes in that run's own eval
#   overhead at the same EVAL_INTERVAL=200 -- 50 evals there, 150 here, same
#   ratio as steps, so scaling the whole wall-time number is more reliable here
#   than separately estimating step-compute and eval-compute).
#   baseline has no segL invert/recompute overhead, so budget somewhat under
#   this (~40h), but --time is sized for the worse (reversible+segL) arm.
#   ~48.3h + ~30% headroom =~ 66h.
#
# Resource requests otherwise MATCH job_reversible_parity_regimes_revback.slurm's
# 'small' case directly (--mem=32G, --gres=gpu:a100:1, no --nodelist): 12L's
# memory footprint is small enough that the 80GB-node targeting used for
# revL24.slurm/revL48.slurm is unnecessary here (segL+batch=24 measured
# ~9.3-9.4GB at batch=12 in the original small-scale validation; batch=24
# should be well under 20GB). --exclude=idun-07-06 kept regardless, since that
# node's problem was external contention at model.to(device), unrelated to
# how much memory this job itself needs.
#
# Parameter parity (narrow embed, exact -- see revformer/test_train_reversible.py):
#   baseline/vpb_baseline: 124,373,760
#   vf_scaling/vpm_scaling: 124,392,192 (+18,432 = 2*n_embd*n_layer, +0.015%)
#
# Usage:   sbatch revL12.slurm
# Analyze: python analyze_wandb.py --project sympformer-reversible-revback \
#              --filter_config n_embd=768 block_size=1024 n_layer=12 \
#              attn_impl=sdpa --format pdf --out_dir plots/revL12
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL12
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # known external-contention node, see header
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=66:00:00
#SBATCH --array=0-3                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL 3=vpm_scaling+segL
#SBATCH --output=output_revL12_%A_%a.txt
#SBATCH --error=output_revL12_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL12_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small', 12 layers ────────────────────────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- matches job_reversible_parity_regimes_revback.slurm's 'small' ─
BATCH_SIZE_BASE=24; GRAD_ACCUM_BASE=20   # baseline:            491,520 tokens/step
BATCH_SIZE_REV=24;  GRAD_ACCUM_REV=20    # reversible + segL:   491,520 tokens/step

# ── 30k-step recipe, warmup fraction matched to 'small' (~10%) ───────────────
MAX_STEPS=30000
WARMUP_STEPS=3000
EVAL_INTERVAL=200
EVAL_BATCHES=50
LOG_INTERVAL=50

PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
DATASET=tinystories          # existing GPT-2 bins

# vpm_scaling target across-depth log-volume change: total log|det| = -REV_LAMBDA
REV_LAMBDA=0.0

log() { echo "[$(date -Is)] $*"; }
run() {
  log "RUN: $(printf '%q ' "$@")"
  "$@"; local rc=$?
  if (( rc != 0 )); then
    log "FAILED: exit ${rc}"
    log "GPU state at failure (host=$(hostname), SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true
  fi
  return $rc
}

COMMON_ARGS=(
  --data_dir data --dataset "${DATASET}"
  --out_dir "${OUTDIR}"
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-reversible-revback
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  baseline_L12
  vpb_segL_L12
  vf_segL_L12
  vpm_segL_L12
)
TASK_ARGS=(
  "--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L12"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_segL_L12"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_L12"
  "--arch reversible --rev_regime vpm_scaling  ${NARROW} --rev_lambda ${REV_LAMBDA} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpm_segL_L12"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL12.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM after many hours.
# NEED=12000 is a floor well below the ~16-20GB expected at this geometry/batch
# (the original small-scale validation measured ~9.3-9.4GB at batch=12; batch=24
# roughly doubles the batch-dependent portion of that).
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [[ -n "${GPU_TOTAL}" && "${GPU_TOTAL}" =~ ^[0-9]+$ ]]; then
  case "${GPU_NAME}" in
    *P100*|*V100*|*P40*|*K80*|*T4*)
      log "FATAL: got '${GPU_NAME}', which predates Ampere -- no native bf16 and no flash"
      log "       SDPA, and only ${GPU_TOTAL}MiB. Resubmit with --gres=gpu:a100:1 (or h100)."
      exit 1 ;;
  esac
  NEED=12000
  if (( GPU_TOTAL < NEED )); then
    log "FATAL: this job needs ~${NEED}MiB but '${GPU_NAME}' has ${GPU_TOTAL}MiB total."
    exit 1
  fi
  if (( GPU_FREE < NEED )); then
    log "FATAL: only ${GPU_FREE}MiB free of ${GPU_TOTAL}MiB on '${GPU_NAME}' -- another"
    log "       process holds the card. See the compute-apps list above."
    exit 1
  fi
  log "GPU preflight OK: ${GPU_NAME}, ${GPU_FREE}MiB free of ${GPU_TOTAL}MiB (need ~${NEED}MiB)"
else
  log "WARNING: could not query GPU memory; skipping preflight"
fi

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}
RC=$?

if (( RC != 0 )); then
  log "Task ${ID} (${NAME}) FAILED after $(( SECONDS ))s with exit ${RC}."
  exit "${RC}"
fi

log "Task ${ID} (${NAME}) done in $(( SECONDS ))s."
