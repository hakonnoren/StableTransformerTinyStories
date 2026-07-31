#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes_deep_test.slurm — 101-step smoke test: does a
# bigger micro-batch (better memory utilization) actually lower s/step at 48L,
# with segL UNCHANGED. Companion to job_reversible_parity_regimes_deep.slurm.
#
# Round 1 (batch=12 -> 24, this job's first submission) confirmed the
# hypothesis for the reversible+segL arms -- measured, not estimated:
#
#   arm                  batch  peak_mem_gb   tokens/s   s/step
#   vf_scaling + segL     12       12.43       25,270     19.4   (real deep job)
#   vf_scaling + segL     24       20.06       30,499     16.1   (round 1, this job)
#   vpb_baseline + segL   24       20.06       26,847     18.3   (round 1, this job;
#                                                                  no 12x40 reference --
#                                                                  that arm OOM'd on
#                                                                  external contention
#                                                                  in the real job)
#
# ~17% faster at batch=24, not just "a few percent" -- segL's per-microstep
# invert+recompute cost is apparently a bigger share of step time than eager's,
# so halving the microstep count paid off more than the original conservative
# estimate assumed. (baseline's round-1 arm failed independently of this --
# ignore it for now, not investigated here.)
#
# Round 1's two peak_mem_gb readings (12.43 @ B=12, 20.06 @ B=24) pin down an
# almost exactly linear memory model for the reversible+segL arms:
#   peak_mem_gb(B) ~= 4.81 + 0.635*B   (fixed ~4.81GB params/opt + ~0.635GB/unit batch)
# which predicts:
#   B=32 -> ~25.1 GB (63%)   B=40 -> ~30.2 GB (75%)   B=48 -> ~35.3 GB (88%)
#   B=60 -> ~42.9 GB -- EXCEEDS the 40GB card, so treat 48 as roughly the ceiling.
# Round 2 (this version) pushes to BATCH_SIZE_REV=40 (~30GB predicted, comfortable
# margin below the ~48 ceiling) to squeeze further while staying safely clear of
# the model's own extrapolation error and node-to-node variance.
#
# segL (--mem_efficient_segment ${N_LAYER}) is UNCHANGED on purpose: the real
# job exists to test reversibility at depth, and segL's O(1)-in-depth stack
# memory is the property under test, so this smoke test does not touch it. The
# ONLY variable here vs the real job is the batch/accum split (baseline's split
# is untouched from round 1 -- its arm is being ignored/debugged separately):
#
#   arm                  real job        round 1        round 2 (this file)
#   reversible + segL    12x40           24x20          40x12
#   baseline              6x80           12x40          12x40 (unchanged)
#
# tokens/step is held fixed at 491,520 throughout, so this only ever moves the
# memory-vs-speed tradeoff, never the gradient.
#
# 101 steps is enough for LOG_INTERVAL=20 to give 5 tokens_per_sec/peak_mem_gb
# readings (steps 20/40/60/80/100) and one EVAL_INTERVAL=100 pass near the end
# to sanity-check cheap_metrics/track_recon_drift/sampling still run at the new
# batch -- it is NOT enough to say anything about loss quality (WARMUP_STEPS is
# scaled down to match, not a real schedule). Separate --wandb_project so these
# don't land in the real 4000-step sweep's analysis filter (both are n_layer=48,
# so filtering on that alone would not separate them).
#
# Usage:   sbatch job_reversible_parity_regimes_deep_test.slurm
# Read:    tokens_per_sec / peak_mem_gb from the first few W&B log points (or
#          the run's stdout) against the reference numbers in the table above.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_deep_L48_test
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-2                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL
#SBATCH --output=output_revdeeptest_%A_%a.txt
#SBATCH --error=output_revdeeptest_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revdeeptestL48_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: identical to the real deep job ────────────────────────────
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum split -- round 2: reversible pushed 24->40, same tokens/step ─
BATCH_SIZE_REV=40;  GRAD_ACCUM_REV=12    # reversible + segL: 491,520 tokens/step (round 1 was 24x20, real job 12x40)
BATCH_SIZE_BASE=12; GRAD_ACCUM_BASE=40   # baseline: unchanged from round 1 (ignored/debugged separately)

# ── Steps -- just enough for a handful of throughput/memory readings ─────────
MAX_STEPS=101
WARMUP_STEPS=10                # not a real schedule, just keeps things sane
EVAL_INTERVAL=100               # one eval near the end (+ the post-loop final eval)
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
# Run dir = OUTDIR/<arch>[_<run_name>]. Keep --array (top) in sync with this.
# segL (--mem_efficient_segment ${N_LAYER}) is UNCHANGED from the real job --
# only batch/accum differ (see header).
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  baseline_L48_test
  reversible_vpb_baseline_segL_test_r2
  reversible_vf_scaling_segL_test_r2
)
TASK_ARGS=(
  "--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L48_test"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_baseline_segL_L48_test_r2"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_scaling_segL_L48_test_r2"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes_deep_test.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM. Round 2 raises
# reversible to batch=40 -- the linear model in the header predicts ~30.2GB;
# baseline is unchanged at ~30GB (round 1 estimate, untested since it failed).
# NEED set below both with margin, still well clear of the ~35GB (batch=48)
# ceiling the header's model flags.
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
  NEED=24000
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
