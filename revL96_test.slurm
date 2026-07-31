#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL96_test.slurm — first probe at 96L (2x the deep job's 48L). 101 steps,
# 3 arms: baseline, vpb_baseline+segL, vf_scaling+segL. Own wandb project
# (sympformer-reversible-L96) since nothing here is comparable to the L48 runs.
#
# New naming convention (files were getting long): revL<depth>[_<suffix>].slurm.
# This replaces the job_reversible_parity_regimes_deep* family name-wise; happy
# to rename those to match (revL48.slurm, revL48_test.slurm, revL48_test_base.slurm)
# if wanted, not done here since it wasn't asked for.
#
# Batch/accum: NOT copied from L48 as-is, because depth changes the two arms'
# memory very differently:
#   - reversible+segL: segL stores exactly ONE (B,T,2*n_embd) tensor regardless
#     of n_layer, so its batch-dependent slope shouldn't change with depth --
#     only the fixed params+optimizer term grows (params scale ~with n_layer
#     for the block matrices: 379.2M @ 48L -> ~718.9M @ 96L, ~1.9x, so fixed
#     ~4.81GB @ 48L -> ~9.1GB @ 96L). Kept at the SAME batch=12,accum=40 the
#     original L48 deep job started at (safe, ~16.7GB predicted, 42%).
#   - baseline (eager, no checkpointing): its batch-dependent term carries the
#     FULL n_layer-deep stack, so doubling depth should roughly DOUBLE its
#     per-batch-unit cost (L48's one-point slope ~2.11GB/unit -> ~4.22GB/unit
#     estimated @ 96L). Batch halved from L48's 6 to 3 (accum 80->160) to land
#     back in roughly the same absolute territory (~21.8GB predicted, 55%) that
#     batch=6 occupied at 48L, rather than reusing 6 and risking ~34GB+ (86%+)
#     on a totally uncalibrated depth.
# Both estimates compound two extrapolations (the L48 models themselves were
# thin on data -- see revL48_test*.slurm) and have no direct 96L measurement
# behind them yet -- this run IS that first measurement. Read peak_mem_gb off
# the actual logs before trusting either number for a squeeze at this depth.
#
# tokens/step held at 491,520 for both arms (12x1024x40 and 3x1024x160), same
# as every other job in this investigation -- memory/speed tradeoff only.
#
# --exclude=idun-07-06: that node has failed 3/3 submissions at model.to(device)
# across this whole investigation (see revL48_test_base.slurm's header) --
# routed around here too.
#
# Usage:  sbatch revL96_test.slurm
# Read:   peak_mem_gb / tokens_per_sec from the W&B log points, project
#         sympformer-reversible-L96.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL96_test
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # see header: 3/3 failures at model.to(device) on this node
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-2                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL
#SBATCH --output=output_revL96_test_%A_%a.txt
#SBATCH --error=output_revL96_test_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL96_test_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small' width, 96 layers ──────────────────────────────────
N_LAYER=96
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- see header for why these differ per arm type ─────────────
BATCH_SIZE_REV=12; GRAD_ACCUM_REV=40      # reversible+segL: 491,520 tokens/step, ~16.7GB predicted
BATCH_SIZE_BASE=3; GRAD_ACCUM_BASE=160    # baseline:        491,520 tokens/step, ~21.8GB predicted

# ── Steps -- just enough for a handful of throughput/memory readings ─────────
MAX_STEPS=101
WARMUP_STEPS=10
EVAL_INTERVAL=100
EVAL_BATCHES=10
LOG_INTERVAL=20                 # 5 readings: steps 20/40/60/80/100

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
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-reversible-L96
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  baseline_L96
  vpb_segL_L96
  vf_segL_L96
)
TASK_ARGS=(
  "--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L96"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_segL_L96"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_L96"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL96_test.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM. NEED is a
# rough floor (both arms predicted well above it, ~16.7-21.8GB) -- catches an
# obviously-wrong allocation, not a tight one, given neither prediction is
# measured yet at this depth.
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
  NEED=16000
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
