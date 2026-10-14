#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL48_test.slurm — round 2. Round 1 (BATCH_SIZE_REV=96) FAILED both
# reversible arms: OOM during loss.backward(), on a genuine 80GB card
# (idun-07-08-class), needing to allocate 18.42GiB more with only 10.58GiB
# free and 68.66GiB already resident -- i.e. this process actually needed
# ~87GiB+, far past the 80GB card, and far past the linear model's ~65.8GB
# prediction. baseline (B=24) SUCCEEDED and is left unchanged.
#
# Why the model missed by this much: the segL memory model (4.81 + 0.635*B)
# was fit on THREE confirmed points, all <=40 (12.43/20.06/30.23GB) -- a
# 2.4x extrapolation to 96 was too far. The failing 18.42GiB allocation lines
# up with the fp32 cross-entropy upcast of the logits (B*T*vocab elements,
# fp32) -- a term that DOES scale linearly with B in principle, but at this
# batch it also means one single ~18GB contiguous allocation, which is exactly
# where PyTorch's caching allocator fragments hardest (note the error's own
# suggestion: "reserved but unallocated" was 26.85GiB -- memory nominally
# free but not in one usable block). Two changes this round:
#   1. BATCH_SIZE_REV backed off 96 -> 60 -- a 1.5x step past the last
#      CONFIRMED point (40) instead of a 2.4x leap past it.
#   2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True added (the fix
#      PyTorch's own OOM message suggests) to reduce fragmentation on these
#      large single allocations regardless of batch size.
# baseline dropped from this array (B=24 already confirmed working -- see
# revL48_test_%A_0.txt from round 1) so this only re-tests what actually failed.
#
# Lands in sympformer-reversible-deep-test (same project as the earlier 48L
# squeeze tests), NOT the main sympformer-reversible project revL48.slurm uses.
#
# Usage:  sbatch revL48_test.slurm
# Read:   peak_mem_gb / tokens_per_sec from the W&B log points. If B=60 still
#         gets close to 80GB or fails, the next step down is 48 (480/48=10),
#         not another large jump.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL48_test
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --nodelist=idun-06-[01-05,07],idun-07-[08-10]   # confirmed >=80GB A100 nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-1                        # 0=vpb_baseline+segL 1=vf_scaling+segL (baseline already confirmed)
#SBATCH --output=output_revL48_test_%A_%a.txt
#SBATCH --error=output_revL48_test_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see header: round 1's OOM suggested this

DEVICE=cuda
OUTDIR=revL48_test_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: identical to revL48.slurm ─────────────────────────────────
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- baseline confirmed working (round 1), reversible backed off
BATCH_SIZE_BASE=24; GRAD_ACCUM_BASE=20   # baseline:            491,520 tokens/step, CONFIRMED ~round 1
BATCH_SIZE_REV=60;  GRAD_ACCUM_REV=8     # reversible + segL:   491,520 tokens/step (round 1's 96 OOM'd)

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
  --wandb --wandb_project sympformer-reversible-deep-test
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  vpb_segL_L48_test60
  vf_segL_L48_test60
)
TASK_ARGS=(
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_segL_L48_test60"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_L48_test60"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL48_test.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM. NEED=40000 is
# a floor for B=60 -- well below what round 1's failure implies B=96 needed
# (~87GB+), but this is a genuinely different (much smaller) batch, not the
# same request retried.
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
