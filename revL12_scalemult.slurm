#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL12_scalemult.slurm — one reversible regime (vpm_scaling+segL, currently
# the strongest reversible performer at 48L) at 12L, sweeping
# --rev_scale_lr_mult -- the LR multiplier applied to gamma/alpha, fixed at 10
# everywhere else in this investigation with no derived justification (see
# conversation: it's a house convention shared with --scalar_lr_mult's own
# default, not something tuned per-experiment; one old file used 100 instead).
#
# Companion to revL12_lrsweep.slurm (same horizon/project/geometry), but tests
# a reversible-specific hyperparameter that baseline has no equivalent of, so
# it gets its own file/regime rather than another baseline arm.
#
# Arms -- rev_scale_lr_mult in {1, 5, 10, 20, 50}, everything else FIXED at the
# current standing recipe (cosine, warmup=300 [10% of 3000], peak_lr=6e-4,
# muon_lr=0.02) so this isolates rev_scale_lr_mult alone, not warmup/LR too:
#   mult=1   no boost at all -- gamma/alpha train at the same LR as other AdamW
#            params. The null hypothesis: is the boost doing anything?
#   mult=5   half the current default
#   mult=10  CURRENT DEFAULT (used in every other job in this investigation)
#   mult=20  double the current default
#   mult=50  the aggressive end -- still well short of job_reversible.slurm's
#            historical 100, kept in reserve if 50 shows a monotonic trend
#
# segL kept (mem_efficient_segment=12), narrow embed (parameter parity with
# baseline), batch=24/accum=20 -- all matching the rest of the 12L work.
# reversible+segL@12L/batch=24 measured 5.8 s/step (the "58k seconds" reference
# run) -- 3000 steps =~ 4.8h/arm, --time below adds margin.
#
# --gres=gpu:1 (no A100 pin) -- same relaxation as revL12_lrsweep.slurm: IDUN
# has ~268 GPUs total (P100/V100/A100/H100/H200/B300), this job's ~10-17GB
# footprint fits any of the non-ancient ones, and the preflight below still
# rejects P100/V100/P40/K80/T4 by name if one slips through.
#
# This is a screening sweep like revL12_lrsweep.slurm -- whichever mult wins
# should get a confirmation run at the full 10001-step budget (and ideally on
# more than one regime) before changing the standing default anywhere else.
#
# Usage:   sbatch revL12_scalemult.slurm
# Analyze: python analyze_wandb.py --project sympformer-reversible-lrsweep \
#              --filter_config rev_regime=vpm_scaling --format pdf \
#              --out_dir plots/revL12_scalemult
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL12_scalemult
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # known external-contention node
#SBATCH --gres=gpu:1                       # ANY GPU, not just A100 -- see header
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=output_revL12_scalemult_%A_%a.txt
#SBATCH --error=output_revL12_scalemult_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL12_scalemult_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small', reversible/vpm_scaling+segL ─────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
BATCH_SIZE=24
GRAD_ACCUM_STEPS=20          # 491,520 tokens/step

# ── Screening horizon -- same as revL12_lrsweep.slurm ────────────────────────
MAX_STEPS=3000
WARMUP_STEPS=300              # 10%, current standing recipe -- fixed, not swept here
EVAL_INTERVAL=150
EVAL_BATCHES=50
LOG_INTERVAL=25

PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
DATASET=tinystories          # existing GPT-2 bins

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
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-reversible-lrsweep
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
  --arch reversible --rev_regime vpm_scaling
  --rev_embed narrow --rev_lift dup --rev_readout sum
  --mem_efficient --mem_efficient_segment "${N_LAYER}"
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NAMES=(
  scalemult_1
  scalemult_5
  scalemult_10
  scalemult_20
  scalemult_50
)
TASK_ARGS=(
  "--rev_scale_lr_mult 1  --run_name scalemult_1"
  "--rev_scale_lr_mult 5  --run_name scalemult_5"
  "--rev_scale_lr_mult 10 --run_name scalemult_10"
  "--rev_scale_lr_mult 20 --run_name scalemult_20"
  "--rev_scale_lr_mult 50 --run_name scalemult_50"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL12_scalemult.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM. NEED is a
# generous floor -- reversible+segL@12L at this batch measured ~9-17GB
# previously, comfortably under this on any A100/H100/H200/B300.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [[ -n "${GPU_TOTAL}" && "${GPU_TOTAL}" =~ ^[0-9]+$ ]]; then
  case "${GPU_NAME}" in
    *P100*|*V100*|*P40*|*K80*|*T4*)
      log "FATAL: got '${GPU_NAME}', which predates Ampere -- no native bf16 and no flash"
      log "       SDPA, and only ${GPU_TOTAL}MiB. Resubmit requesting a newer GPU."
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
