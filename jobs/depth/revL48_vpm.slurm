#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL48_vpm.slurm — adds the vpm_scaling arm to the revL48.slurm sweep.
# Triggered by vf_scaling currently trailing vpb_baseline in the running
# revL48.slurm comparison -- vpm_scaling was the second-best regime (after
# vf_scaling) in the original 'small'-geometry 5-regime sweep, so it's the
# natural next arm to check at this depth.
#
# Everything else identical to revL48.slurm's reversible+segL arms -- same
# geometry, same CONFIRMED batch=60/accum=8 (42.93GB measured, 15.97-16.88
# s/step), same segL (mem_efficient_segment=48), same full 10001-step 'small'
# schedule, same --nodelist/allocator-fragmentation fix, same
# sympformer-reversible project so it's directly filterable alongside the
# other three arms (baseline, vpb_baseline+segL, vf_scaling+segL).
#
# Only new thing: --rev_lambda, vpm_scaling-specific (targets total log|det| =
# -REV_LAMBDA across the whole stack, vs vpb_scaling's per-block log|det|=0 --
# see job_reversible_parity_regimes.slurm's header for the regime table).
# REV_LAMBDA=0.0 matches every prior vpm_scaling run in this investigation.
#
# Kept as a single-task array (not folded into revL48.slurm's array, which is
# already running) so this can be submitted independently without touching
# the in-flight job. Structured the same NAMES/TASK_ARGS way as the rest of
# this family in case vpb_scaling (the one regime never run at 48L) gets
# added later -- just extend the arrays and --array range.
#
# Usage:   sbatch revL48_vpm.slurm
# Analyze: python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#              n_layer=48 attn_impl=sdpa --format pdf --out_dir plots/revL48
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL48_vpm
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --nodelist=idun-06-[01-05,07],idun-07-[08-10]   # confirmed >=80GB A100 nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=66:00:00
#SBATCH --array=0                          # 0=vpm_scaling+segL
#SBATCH --output=output_revL48_vpm_%A_%a.txt
#SBATCH --error=output_revL48_vpm_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see revL48.slurm's header
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL48_vpm_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: identical to revL48.slurm ─────────────────────────────────
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- CONFIRMED for reversible+segL at this depth (revL48.slurm) ─
BATCH_SIZE_REV=60; GRAD_ACCUM_REV=8      # 491,520 tokens/step, 42.93GB measured

# ── Full 'small' recipe: 10001 steps, same schedule/LRs as revL48.slurm ──────
MAX_STEPS=10001
WARMUP_STEPS=1000
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
  --batch_size "${BATCH_SIZE_REV}" --grad_accum_steps "${GRAD_ACCUM_REV}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-reversible
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
  --arch reversible --mem_efficient --mem_efficient_segment "${N_LAYER}"
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  vpm_segL_L48
)
TASK_ARGS=(
  "--rev_regime vpm_scaling ${NARROW} --rev_lambda ${REV_LAMBDA} --run_name vpm_segL_L48"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL48_vpm.slurm"; exit 1
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
# NEED=40000 matches revL48.slurm's confirmed batch=60 floor (see there).
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
  NEED=40000
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
