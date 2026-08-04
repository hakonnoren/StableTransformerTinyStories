#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL12_warmup_confirm.slurm — full-budget (10001-step) confirmation of the
# revL12_lrsweep.slurm screening result: "wradam" (warmup_steps=40, the
# RAdam/Ma&Yarats variance-warmup rule 2/(1-beta2), independent of max_steps)
# beat the standing "ref" convention (warmup_steps=300, ~10% of max_steps) at
# 3000 steps -- best_val 1.0917 vs 1.1013, baseline arch, 12L. That screen was
# short/cheap by design; this run checks the result holds at the horizon this
# investigation actually cares about before adopting wradam anywhere else.
#
# 2 arms, baseline arch ONLY (no reversible arm here -- this isolates the
# warmup-scheme question from arch/regime, which revL12_vf_sched.slurm already
# checked separately at screening scale and found no reversible-specific
# warmup sensitivity worth chasing):
#   ref_w10pct_10k  warmup_steps=1000  (10% of 10001, the historical default,
#                   e.g. revL12.slurm/job_reversible_parity_regimes_revback.slurm)
#   wradam_10k      warmup_steps=40    (RAdam rule, screening-sweep winner)
# Everything else identical and matched to the rest of the 12L work: peak_lr
# 6e-4, muon_lr 0.02, cosine schedule (wsd already ruled out at screening
# scale -- wsd_wradam scored 1.1635, far worse than every cosine variant, in
# BOTH baseline and vf_scaling arms -- not retested here).
#
# Batch/accum: 24x20 (491,520 tokens/step), matching revL12.slurm and every
# other 12L job in this investigation, for direct comparability.
#
# Time budget -- from REAL measured s/step, not extrapolated: pulling wandb
# runtimes off the 3000-step screening sweep's own baseline arms (same
# geometry/batch, just fewer steps) gives 2.96-3.65 s/step across the 9 runs
# (spread reflects landing on different GPU types in IDUN's heterogeneous
# fleet). The wradam arm itself measured 3.65 s/step; ref measured 3.28.
# Using the slower of the two as the per-arm estimate:
#   3.65 s/step x 10001 steps = 36,504s = ~10.1h
#   +~50% headroom (generous on purpose -- this is a cheap job to over-
#   provision and the fleet is heterogeneous enough that a slower-than-
#   observed card is plausible) =~ 15.2h -> --time=16:00:00
#
# Resources: --mem=32G, --gres=gpu:1 (any GPU, not pinned to A100 -- see
# revL12_lrsweep.slurm's header for the IDUN-fleet-relaxation rationale),
# --exclude carries forward the same P100/V100/bad-node denylist fixed in
# revL12_lrsweep.slurm/revL12_scalemult.slurm/revL12_vf_sched.slurm after a
# P100 slipped through the relaxed gres and (correctly, but wastefully) hit
# the preflight check.
#
# Fresh project (sympformer-reversible-warmup-confirm) rather than reusing
# sympformer-reversible-lrsweep -- this is a full-budget confirmation run, not
# another screening arm, and keeping it separate avoids mixing 3000-step and
# 10001-step best_val numbers in the same analyze_wandb.py pull.
#
# Usage:   sbatch revL12_warmup_confirm.slurm
# Analyze: python analyze_wandb.py --project sympformer-reversible-warmup-confirm \
#              --filter_config n_embd=768 block_size=1024 n_layer=12 \
#              attn_impl=sdpa --format pdf --out_dir plots/revL12_warmup_confirm
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL12_warmup_confirm
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]   # P100/V100 pools + the bad-node A100
#SBATCH --gres=gpu:1                       # ANY GPU, not just A100 -- see header
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-1                        # 0=ref_w10pct_10k 1=wradam_10k
#SBATCH --output=output_revL12_warmup_confirm_%A_%a.txt
#SBATCH --error=output_revL12_warmup_confirm_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL12_warmup_confirm_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small', 12 layers, baseline arch only ───────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
BATCH_SIZE=24
GRAD_ACCUM_STEPS=20          # 491,520 tokens/step, matches all other 12L jobs

# ── Full confirmation horizon ─────────────────────────────────────────────────
MAX_STEPS=10001
EVAL_INTERVAL=200
EVAL_BATCHES=50
LOG_INTERVAL=50

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
  --arch baseline
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --wandb --wandb_project sympformer-reversible-warmup-confirm
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NAMES=(
  ref_w10pct_10k
  wradam_10k
)
TASK_ARGS=(
  "--warmup_steps 1000 --run_name ref_w10pct_10k"
  "--warmup_steps 40   --run_name wradam_10k"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL12_warmup_confirm.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM after many hours.
# NEED=12000 is a generous floor -- baseline@12L/batch=24 is well under this on
# any Ampere-or-newer card; the case that matters here is catching a
# pre-Ampere card that slipped through the relaxed --gres=gpu:1 despite the
# --exclude list above (see revL12_lrsweep.slurm's header for the incident
# this preflight is guarding against).
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [[ -n "${GPU_TOTAL}" && "${GPU_TOTAL}" =~ ^[0-9]+$ ]]; then
  case "${GPU_NAME}" in
    *P100*|*V100*|*P40*|*K80*|*T4*)
      log "FATAL: got '${GPU_NAME}', which predates Ampere -- no native bf16 and no flash"
      log "       SDPA, and only ${GPU_TOTAL}MiB. Should not happen given --exclude above --"
      log "       check for a P100/V100 node not covered by it."
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
