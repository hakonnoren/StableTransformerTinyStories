#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL48.slurm — the FULL 10001-step run at 48L (4x 'small's depth), matching
# 'small's exact recipe (peak_lr/muon_lr/warmup fraction/schedule) so arm 0
# (baseline) is directly comparable to the paper's 12L numbers. segL is kept
# throughout -- the whole point of this line of work is testing reversibility
# at depth, not just squeezing memory.
#
# Supersedes job_reversible_parity_regimes_deep.slurm (which ran this at 4000
# steps and small batches: baseline 6x80, reversible+segL 12x40). This version
# uses everything learned since from the 101-step smoke tests, INCLUDING a
# failed first attempt at pushing too far:
#   - baseline's memory model (B=6->17.48GB, B=15->35.40GB) predicted B=24
#     -> ~53.3GB, and revL48_test.slurm CONFIRMED this: batch=24 succeeded.
#   - reversible+segL's memory model, fit on THREE confirmed points
#     (B=12->12.43GB, B=24->20.06GB, B=40->30.23GB, peak_mem_gb(B) ~=
#     4.81+0.635*B), predicted B=96 -> ~65.8GB. This was WRONG: B=96 actually
#     OOM'd during loss.backward() on a real 80GB card, needing ~87GB+ (the
#     failing single allocation, 18.42GiB, lines up with the fp32 cross-entropy
#     upcast of the logits -- a genuinely batch-linear term, but one huge
#     enough at B=96 to fragment the allocator badly on top of the
#     extrapolation error). A 2.4x jump past the last confirmed point (40) was
#     too far; BATCH_SIZE_REV is backed off to 60 (1.5x past 40), confirmed
#     separately via revL48_test.slurm before this file was finalized.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True added below -- the
#     mitigation PyTorch's own OOM message suggested for exactly this
#     large-single-allocation fragmentation pattern.
#   - --nodelist targets the specific nodes confirmed >=80GB by W&B metadata
#     (see job_reversible_parity_regimes_deep_test_baseline.slurm's header for
#     how these were identified -- sinfo exposes no GRES distinction between
#     A100 memory variants, so this list is built from direct observation, not
#     a documented cluster feature).
#
# tokens/step held at 491,520 for every arm (24x1024x20 baseline, 96x1024x5
# reversible), identical to 'small' and to every prior arm in this
# investigation -- this only ever trades memory for speed, never the gradient.
#
# Time budget -- CONFIRMED, both settings actually run now (not extrapolated):
#   baseline @ B=24: measured peak_mem_gb=53.19GB (predicted 53.3), 9.83 s/step.
#   reversible+segL @ B=60: measured peak_mem_gb=42.93GB (predicted 42.91 --
#     the memory model held exactly here), 15.97 s/step (vpb) / 16.88 s/step
#     (vf) -- confirming B=40->60 barely helped (16.08/17.33 at B=40, same
#     80GB PCIe hardware): batching's overhead-amortization payoff had already
#     bottomed out by B=40, this is compute-bound now, not overhead-bound.
#   At MAX_STEPS=10001, worst arm (vf): 10001*16.88s = 46.9h + ~1.4h evals
#     (deeper model, more expensive drift/cheap_metrics/sampling per eval than
#     12L) + ~0.3h startup =~ 48.6h. --time below adds ~30% headroom over that.
#
# New naming convention (see revL96_test.slurm): revL<depth>[_<suffix>].slurm.
# This is the first non-suffixed file in the family -- the "main" run at its
# depth, same role job_reversible_parity_regimes_deep.slurm played before.
#
# Lands in the MAIN sympformer-reversible project (not -deep-test), same as
# 'small' -- this is the real experiment now, filterable by n_layer=48.
#
# Usage:   sbatch revL48.slurm
# Analyze: python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#              n_layer=48 attn_impl=sdpa --format pdf --out_dir plots/revL48
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL48
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --nodelist=idun-06-[01-05,07],idun-07-[08-10]   # confirmed >=80GB A100 nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=66:00:00
#SBATCH --array=0-2                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL
#SBATCH --output=output_revL48_%A_%a.txt
#SBATCH --error=output_revL48_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see header: mitigates large-allocation fragmentation
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL48_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small' width, 48 layers ──────────────────────────────────
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- see header for the memory models behind these ────────────
BATCH_SIZE_BASE=24; GRAD_ACCUM_BASE=20   # baseline:            491,520 tokens/step, ~53.3GB predicted
BATCH_SIZE_REV=60;  GRAD_ACCUM_REV=8     # reversible + segL:   491,520 tokens/step (96 OOM'd, see header)

# ── Full 'small' recipe: 10001 steps, same schedule/LRs ───────────────────────
MAX_STEPS=10001
WARMUP_STEPS=1000               # ~10% fraction, matching 'small'
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
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
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
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Run dir = OUTDIR/<arch>[_<run_name>]. Keep --array (top) in sync with this.
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  baseline_L48
  vpb_segL_L48
  vf_segL_L48
)
TASK_ARGS=(
  "--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L48"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_segL_L48"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_L48"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL48.slurm"; exit 1
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
# NEED=40000 is a floor for the now-confirmed B=60/B=24 settings (see header --
# B=96 needed ~87GB+ and is NOT what this file runs), comfortably below the
# 80GB --nodelist targets, but high enough to refuse a card that clearly can't
# fit this rather than fail deep inside train.py hours into the run.
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
