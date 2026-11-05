#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# relaxL12.slurm — the relaxed reversible coupling (--rev_regime relax) on the
# paper's 12L TinyStories recipe, as a non-reversible and a reversible variant.
#
#   x' = e^-g x + (1 - e^-g)(z + Attn(LN_1 z))
#   z' = e^-a z + (1 - e^-a)(x' + MLP(LN_2 x'))       g, a = softplus(raw) > 0
#
#   ID  run name               rate          backprop                  W&B project
#    0  relax_eager_L12        3.0, learned  ordinary autograd         sympformer-reversible
#    1  relax_segL_r0.1_L12    0.1, fixed    reconstructed (segL)      sympformer-reversible-revback
#    2  relax_eager_r0.1_L12   0.1, fixed    ordinary autograd         sympformer-reversible
#
# Default submission is arms 0-1. Arm 2 is the clean control for arm 1 (same
# model, eager backprop): `sbatch --array=2 jobs/relax/relaxL12.slurm`.
#
# Why the two variants use different rates
# ----------------------------------------
# The relax inverse re-expands the x-z difference mode by e^(g+a) per block, so
# reconstructing activations (--mem_efficient) is only numerically usable at small
# rates. Measured at THIS geometry (12L/768d, narrow, random init, B=2 T=128, CPU),
# relative gradient error of segL vs eager autograd:
#
#   arm            fp32       bf16
#   vf_scaling     1.9e-6     5.2e-3     <- conformal reference (paper's segL runs)
#   relax r=0.1    4.9e-6     9.9e-3
#   relax r=0.3    2.4e-4     2.3e-1
#   relax r=1.0    1.0e0      5.3e2
#   relax r=3.0    NaN        NaN
#
# So the note's default rate 3 cannot run reversibly at all, and arm 0 uses ordinary
# autograd. Arm 1 is the reversible variant at rate 0.1, where segL's gradient error
# is ~2x the conformal arms'. Its rates are FIXED (--rev_relax_fixed): learned rates
# that drift to ~0.2 would already cost ~10x in gradient error (it grows like
# e^(2*rate*11)), silently. --track_recon_drift logs the inverse's error at every
# eval, so check recon_drift_max on arm 1 against the conformal segL runs (~1e-2).
#
# Note rate 0.1 is a different model, not just a different backprop: each block
# moves only ~10% of the way to its target (relax -> identity as rate -> 0). Arm 2
# separates those two effects.
#
# Recipe: identical to the paper's TinyStories runs (Table 3; main sweep
# job_reversible_parity_regimes.slurm and its segL twin ..._revback.slurm, 'small'):
# 10001 steps, 1000 warmup, 24x20 = 491,520 tokens/step, Muon 0.02 / AdamW 6e-4,
# damping LR x10, min_lr_ratio 0.1, sdpa, narrow embedding (dup/sum), one seed.
# The rates are named gamma_bias/alpha_bias, so they land in the same no-decay,
# x10-LR optimizer group as the conformal damping.
#
# Params: 124,392,192 (the conformal arms' count; the learned rates are 2*n_embd*n_layer
# = 18,432 scalars). With --rev_relax_fixed they are buffers -> 124,373,760, the
# baseline's count.
#
# W&B projects follow the existing split (see the revback header): eager runs next to
# the main sweep, segL runs next to the other segL runs. Compare against
#   sympformer-reversible         baseline 1.0522, eager conformal arms
#   sympformer-reversible-revback baseline 1.0528, segL arms (rev 1.0636, vf 1.0551 ...)
#
# Time: eager arms ran 3.05-3.47 s/step (8.5-9.7 h), segL 5.74-6.20 s/step
# (16.0-17.2 h) + ~1 h evals/startup. 24h covers the segL arm with the same headroom
# the revback job uses.
#
# Memory: eager reversible vf measured 14.4 GB at B=12 (sdpa, cheap_metrics), so
# <= ~28.8 GB at B=24 (pessimistic, linear in B); segL ~16-19 GB.
#
# Usage:   sbatch jobs/relax/relaxL12.slurm              # arms 0-1
#          sbatch --array=2 jobs/relax/relaxL12.slurm    # optional eager control
# Override a rate without editing: sbatch --export=ALL,RELAX_RATE_REV=0.05 ...
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=relaxL12
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # known bad (node, GPU) pair, see revback header
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-1                        # 0=relax eager (rate 3)  1=relax segL (rate 0.1)  [2=eager control]
#SBATCH --output=output_relaxL12_%A_%a.txt
#SBATCH --error=output_relaxL12_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=relaxL12_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: GPT-2 small ───────────────────────────────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Paper TinyStories recipe (Table 3) ────────────────────────────────────────
BATCH_SIZE=24; GRAD_ACCUM_STEPS=20     # 480 seq = 491,520 tokens/step, every arm
MAX_STEPS=10001
WARMUP_STEPS=1000
EVAL_INTERVAL=200
EVAL_BATCHES=50
LOG_INTERVAL=50
PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
DATASET=tinystories          # existing GPT-2 bins

# ── Relax rates (overridable with sbatch --export=ALL,...) ────────────────────
RELAX_RATE_EAGER=${RELAX_RATE_EAGER:-3.0}   # the note's default: keep e^-3 = 5% per block
RELAX_RATE_REV=${RELAX_RATE_REV:-0.1}       # largest measured rate segL survives in bf16

log() { echo "[$(date -Is)] $*"; }
run() {
  log "RUN: $(printf '%q ' "$@")"
  "$@"; local rc=$?
  if (( rc != 0 )); then
    log "FAILED: exit ${rc}"
    log "GPU state at failure (host=$(hostname), SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,compute_mode --format=csv 2>&1 || true
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
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --arch reversible --rev_regime relax
  --rev_embed narrow --rev_lift dup --rev_readout sum
  # Inverse error per block at every eval. On the eager arms it is a free diagnostic
  # of how far from invertible the trained model is; on arm 1 it is the health check.
  --track_recon_drift
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# segL: checkpoint the stack input only (i % 12 == 0), reconstruct blocks 11..1.
SEGL="--mem_efficient --mem_efficient_segment ${N_LAYER}"

NAMES=(
  relax_eager_L12
  relax_segL_r${RELAX_RATE_REV}_L12
  relax_eager_r${RELAX_RATE_REV}_L12
)
PROJECTS=(
  sympformer-reversible
  sympformer-reversible-revback
  sympformer-reversible
)
TASK_ARGS=(
  "--rev_relax_rate ${RELAX_RATE_EAGER}"
  "--rev_relax_rate ${RELAX_RATE_REV} --rev_relax_fixed ${SEGL}"
  "--rev_relax_rate ${RELAX_RATE_REV} --rev_relax_fixed"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch jobs/relax/relaxL12.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "Arm args: ${TASK_ARGS[$ID]}  -> W&B ${PROJECTS[$ID]}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Bail out now rather than on a CUDA OOM hours in. The eager arms are the larger
# (<= ~28.8 GB pessimistic at B=24), so that sets the floor for all.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [[ -n "${GPU_TOTAL}" && "${GPU_TOTAL}" =~ ^[0-9]+$ ]]; then
  case "${GPU_NAME}" in
    *P100*|*V100*|*P40*|*K80*|*T4*)
      log "FATAL: got '${GPU_NAME}', which predates Ampere -- no native bf16 and no flash"
      log "       SDPA. Resubmit with --gres=gpu:a100:1 (or h100)."
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
run python3 -u train.py "${COMMON_ARGS[@]}" \
  --wandb --wandb_project "${PROJECTS[$ID]}" --run_name "${NAME}" \
  ${TASK_ARGS[$ID]}
RC=$?

if (( RC != 0 )); then
  log "Task ${ID} (${NAME}) FAILED after $(( SECONDS ))s with exit ${RC}."
  exit "${RC}"
fi

log "Task ${ID} (${NAME}) done in $(( SECONDS ))s."
