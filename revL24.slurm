#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL24.slurm — 30000-step run at 24L (2x 'small's depth, half of revL48's),
# same 4 arms as the current 48L sweep (baseline, vpb_baseline+segL,
# vf_scaling+segL, vpm_scaling+segL) so 12L/24L/48L/96L form one depth ladder
# at matched geometry (12H/768d/1024block throughout -- only n_layer varies).
# segL kept for every reversible arm, same as everywhere else in this line of
# work. Lands in sympformer-reversible, filterable by n_layer=24 alongside
# the rest of the ladder.
#
# Batch/accum: REUSED directly from revL48.slurm's CONFIRMED settings
# (baseline 24x20, reversible+segL 60x8) rather than re-calibrated for 24L.
# This is deliberately conservative, not optimal -- 24L's memory footprint
# will be LOWER than 48L's at the same batch (fixed params+optimizer term
# scales with n_layer: 24L has 209.3M non-embedding-adjacent params vs 48L's
# 379.2M, a 0.55x ratio, so the ~4.81GB/~5.53GB fixed terms at 48L become
# roughly ~2.65GB/~3.05GB at 24L; segL's batch-dependent slope is depth-
# independent by design, and baseline's slope roughly halves with depth). So
# expect peak_mem_gb well under revL48's measured 42.93GB/53.19GB -- reusing
# the same batch just means comfortable headroom rather than a fresh squeeze,
# which isn't the point of this run. NOT yet measured at 24L; watch the first
# few peak_mem_gb log lines.
#
# Time budget (ESTIMATE -- 24L has never been run in this investigation):
# per-step time is assumed to scale linearly with n_layer (the same
# deliberately-pessimistic assumption used for every other cross-depth budget
# in this investigation -- true scaling is likely somewhat better, since
# embedding/loss-head compute doesn't shrink with depth). Halving revL48's
# CONFIRMED numbers (baseline 9.83s/step @ B=24, worst reversible arm (vf)
# 16.88s/step @ B=60):
#   baseline:            ~4.9 s/step  x 30000 = ~41h
#   reversible+segL (vf): ~8.4 s/step x 30000 = ~70h
# Plus eval overhead (50 evals at EVAL_INTERVAL=600, see below -- same COUNT
# as revL48's 50 evals, just spread over 3x the steps) and startup, worst arm
# =~72h. --time gives ~30% headroom over that given the estimate is
# unverified at this depth: 96h.
#
# CHECK THE PARTITION ALLOWS 96H: sinfo -p GPUQ -o "%P %l". If it doesn't,
# this is a case for actually splitting the array into separate submissions
# with per-arm --time (baseline needs far less than reversible+segL), same
# idea as revL48.slurm's header describes.
#
# vpm_scaling: same regime added to revL48_vpm.slurm, --rev_lambda 0.0
# (targets total log|det|=0 across the stack). Parameter parity (narrow
# embed): baseline/vpb_baseline 209,289,216; vf/vpm_scaling 209,326,080
# (+36,864 = 2*n_embd*n_layer gamma/alpha vectors, +0.018%).
#
# Usage:   sbatch revL24.slurm
# Analyze: python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#              n_layer=24 attn_impl=sdpa --format pdf --out_dir plots/revL24
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL24
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --nodelist=idun-06-[01-05,07],idun-07-[08-10]   # confirmed >=80GB A100 nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --array=0-3                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL 3=vpm_scaling+segL
#SBATCH --output=output_revL24_%A_%a.txt
#SBATCH --error=output_revL24_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see revL48.slurm's header
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL24_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small' width, 24 layers ──────────────────────────────────
N_LAYER=24
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum -- reused from revL48.slurm's confirmed settings (see header) ─
BATCH_SIZE_BASE=24; GRAD_ACCUM_BASE=20   # baseline:            491,520 tokens/step
BATCH_SIZE_REV=60;  GRAD_ACCUM_REV=8     # reversible + segL:   491,520 tokens/step

# ── 30k-step recipe, warmup fraction matched to 'small'/revL48 (~10%) ────────
MAX_STEPS=30000
WARMUP_STEPS=3000
EVAL_INTERVAL=600                # 50 evals total, same count as revL48 (10001/200)
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
  --wandb --wandb_project sympformer-reversible
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  baseline_L24
  vpb_segL_L24
  vf_segL_L24
  vpm_segL_L24
)
TASK_ARGS=(
  "--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L24"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_segL_L24"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_L24"
  "--arch reversible --rev_regime vpm_scaling  ${NARROW} --rev_lambda ${REV_LAMBDA} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpm_segL_L24"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL24.slurm"; exit 1
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
# NEED=40000 matches revL48.slurm's floor for the same batch settings -- 24L's
# actual footprint should be LOWER (see header), so this is generous, not tight.
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
