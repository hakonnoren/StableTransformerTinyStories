#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes_deep_test_baseline.slurm — 101-step smoke test,
# BASELINE ONLY: squeeze baseline's micro-batch as far as it plausibly goes at
# 48L. Companion to job_reversible_parity_regimes_deep_test.slurm (which did
# this for the reversible+segL arms across two rounds: 12->24->40, ~17%/~12%
# faster each time, memory tracking an almost-exact linear model).
#
# Why baseline-only, and why TWO identical arms
# ----------------------------------------------
# baseline's arm has now failed THREE separate times, always with the same
# signature -- CUDA OOM at model.to(device), i.e. before any batch-dependent
# memory is even touched -- and always on the SAME node:
#
#   run                                   host         gpu               batch
#   vpb_baseline+segL (real deep job)     idun-07-06   A100-PCIe-40GB    12x40
#   baseline round 1 (deep_test)          idun-07-06   A100-PCIe-40GB    12x40
#   baseline round 2 (deep_test)          idun-07-06   A100-PCIe-40GB    12x40
#
# model.to(device) only moves the ~5GB of params -- it is BATCH-INDEPENDENT, so
# these failures have nothing to do with batch_size/grad_accum_steps; they mean
# idun-07-06 specifically keeps handing out a card without ~5GB free even for
# that first small allocation. Three-for-three on the exact same node is not
# random cluster-wide contention, it is that ONE node. --exclude=idun-07-06 is
# added below to route around it; the two identical arms are a cheap belt-and-
# suspenders on top of that, in case some other node is having a bad day too.
# Hacky, but fine for a 101-step probe -- not something to carry into a real run
# without understanding why that node behaves this way (a stuck process from
# another user's job that never released memory is the most likely explanation,
# not a card-capacity issue, since the SAME model.to(device) call succeeds fine
# on every other node observed so far).
#
# Squeezing the batch: what we actually know vs what's extrapolated
# --------------------------------------------------------------------
# There is exactly ONE real baseline measurement so far: B=6 -> peak_mem_gb
# 17.48 (the real deep job, host idun-11-18, uncontended). Unlike the
# reversible+segL model (calibrated on TWO points, 12 and 24, and validated
# again at 40 -- see job_reversible_parity_regimes_deep_test.slurm's header),
# baseline's slope below is a ONE-POINT extrapolation and should be trusted
# less. Borrowing the reversible model's ~4.81GB fixed (params+optimizer, a
# similar total parameter count) and solving for the batch-dependent slope from
# the single point:
#   17.48 = 4.81 + k*6   =>   k ~= 2.11 GB per unit batch
#   (~3.3x the reversible+segL slope of 0.635 -- expected, since baseline has
#   no checkpointing at all: its batch-dependent term carries the FULL 48-layer
#   stack, not segL's flat one-tensor cost.)
# Predicted peak_mem_gb(B) ~= 4.81 + 2.11*B:
#   B=12 -> ~30.1 GB (75%, the size already in job_reversible_parity_regimes_deep_test.slurm,
#                      never actually run due to the node issue above)
#   B=15 -> ~36.5 GB (91%) <- BATCH_SIZE_BASE below: the next valid divisor of
#                      480 past 12, and about as far as a ONE-POINT model
#                      should be pushed on a 40GB card
#   B=16 -> ~38.5 GB (96%) -- too tight given the extrapolation uncertainty
# This run is also, therefore, this model's second real calibration point --
# if it fails on a clean (non-idun-07-06) node, that means the true slope is
# steeper than 2.11 and B=12 is closer to the real ceiling than B=15.
#
# tokens/step held at 491,520 (15x1024x32), identical to every other arm in
# this whole investigation, so this only ever trades memory for speed, never
# changes the gradient.
#
# Usage:   sbatch job_reversible_parity_regimes_deep_test_baseline.slurm
# Read:    tokens_per_sec / peak_mem_gb from whichever of the two arms
#          succeeds, against the B=6 reference (17.48GB, 13.1 s/step) above.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_deep_L48_baseline_test
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # see header: 3/3 failures at model.to(device) on this node
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0                        # two IDENTICAL baseline arms, redundancy against a bad node
#SBATCH --output=output_revdeeptestbase_%A_%a.txt
#SBATCH --error=output_revdeeptestbase_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revdeeptestbaseL48_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: identical to the real deep job ────────────────────────────
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- squeezed past the deep_test job's 12x40 (see header) ──────
BATCH_SIZE_BASE=15; GRAD_ACCUM_BASE=32    # 491,520 tokens/step (15x1024x32), ~36.5GB predicted

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
  --batch_size "${BATCH_SIZE_BASE}" --grad_accum_steps "${GRAD_ACCUM_BASE}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --wandb --wandb_project sympformer-reversible-deep-test
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
  --arch baseline
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Two IDENTICAL baseline arms (same config, different run_name) -- redundancy
# against a bad node, not a real A/B. Run dir = OUTDIR/baseline_<run_name>.
NAMES=(
  baseline_L48_squeeze
  #baseline_L48_squeeze_B
)
TASK_ARGS=(
  "--run_name baseline_L48_squeeze"
  #"--run_name baseline_L48_squeeze_B"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes_deep_test_baseline.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE_BASE}x${GRAD_ACCUM_BASE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM. NEED=30000 is a
# floor sized off the ~36.5GB prediction above -- rejects a card that obviously
# can't fit this, but (per the header) the last three failures happened well
# ABOVE this floor's threshold (model.to(device) needs ~5GB, not 30GB), so this
# preflight would NOT have caught the idun-07-06 failures -- --exclude is what
# actually fixes that; this check is only a backstop for a genuinely small card.
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
  NEED=30000
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
