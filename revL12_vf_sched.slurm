#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL12_vf_sched.slurm — does shortening gamma/alpha's warmup destabilize
# vf_scaling specifically? Companion to revL12_lrsweep.slurm (which tested
# schedule variants on baseline only) and revL12_scalemult.slurm (which fixed
# the schedule and swept rev_scale_lr_mult on vpm_scaling). This is the piece
# neither of those covers: schedule variants on a LIVE-gamma/alpha regime.
#
# Why this needs its own test, not just reusing baseline's winner
# ------------------------------------------------------------------
# gamma/alpha enter the forward pass as exp(-gamma)/exp(-alpha) scaling the
# ENTIRE residual branch, initialized at zero so the model starts identical to
# vpb_baseline. A large early step there isn't like perturbing a normal weight
# matrix -- it directly perturbs an exponential. The 48L full run already
# showed what happens when gamma/alpha wander too far: vf_scaling's
# recon_drift_max grew ~200x over training as gamma/alpha kept evolving with
# no bound, and under segL that reconstruction error IS the training gradient,
# not just a diagnostic -- so a bad choice here could plausibly feed back on
# itself (faster drift -> worse reconstructed gradients -> more drift).
# vf_scaling (fully unconstrained) is the regime most exposed to this; it's
# the one used here rather than vpm_scaling or vpb_scaling.
#
# This is specifically an AdamW-side concern: gamma/alpha are AdamW params
# (rev_scale_params), not Muon params. Muon only touches attn/mlp matrices,
# which are architecturally identical to baseline's, so "Muon doesn't need
# warmup" (arm 3 here) is expected to be safe regardless -- included as the
# control that ISN'T expected to move gamma/alpha's dynamics at all.
#
# Arms (rev_scale_lr_mult fixed at 10, the standing default -- that axis is
# revL12_scalemult.slurm's job, not this one):
#   0 ref_vf            cosine, warmup=300 (10% of 3000) -- current recipe,
#                        same-horizon reference (the real 10001-step vf_scaling
#                        runs aren't comparable to a 3000-step screen).
#   1 wradam_vf          warmup=40 (RAdam-equivalence rule) -- the risky one:
#                        shortens gamma/alpha's ramp along with AdamW's.
#   2 w20pct_vf          warmup=600 (20%) -- opposite direction, does MORE
#                        warmup help vf_scaling specifically more than baseline?
#   3 muon_nowarmup_vf   AdamW/gamma-alpha keep 300, Muon warmup=1 -- the
#                        control: doesn't touch gamma/alpha's schedule at all.
#   4 wsd_wradam_vf      wsd schedule + warmup=40 -- the most aggressive
#                        combination (different shape AND short ramp).
#
# READ recon_drift_max OVER TRAINING FOR EACH ARM, not just best_val -- a
# destabilizing schedule choice should show up there first (the 48L run's
# drift swings were ~200x; val_loss gaps between regimes have been ~0.5-1%).
# --track_recon_drift is on for all arms.
#
# Same geometry/batch/horizon/project as the rest of the 12L screening work:
# n_layer=12, batch=24/accum=20, segL (mem_efficient_segment=12), narrow embed,
# MAX_STEPS=3000, sympformer-reversible-lrsweep. --gres=gpu:1 (no A100 pin,
# same relaxation as the other lrsweep-family files -- see revL12_lrsweep.slurm).
#
# RESUBMIT STATUS: none of these 5 arms have registered in W&B yet. Arms 3
# (muon_nowarmup_vf) and 4 (wsd_wradam_vf) need the same new train.py flags as
# revL12_lrsweep.slurm's two missing arms -- sync train.py first. Arms 0-2
# (ref_vf, wradam_vf, w20pct_vf) use only pre-existing flags, so if they're
# also missing after that, it's queue/scheduling, not a code issue -- check
# `squeue -u $USER`. --array left at 0-4 since nothing here has run yet.
#
# Usage:   sbatch revL12_vf_sched.slurm
# Analyze: python analyze_wandb.py --project sympformer-reversible-lrsweep \
#              --filter_config rev_regime=vf_scaling --format pdf \
#              --out_dir plots/revL12_vf_sched
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL12_vf_sched
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06               # known external-contention node
#SBATCH --gres=gpu:1                       # ANY GPU, not just A100 -- see revL12_lrsweep.slurm
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=output_revL12_vf_sched_%A_%a.txt
#SBATCH --error=output_revL12_vf_sched_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL12_vf_sched_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small', reversible/vf_scaling+segL ──────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
BATCH_SIZE=24
GRAD_ACCUM_STEPS=20          # 491,520 tokens/step

# ── Screening horizon -- same as revL12_lrsweep.slurm/revL12_scalemult.slurm ─
MAX_STEPS=3000
EVAL_INTERVAL=150
EVAL_BATCHES=50
LOG_INTERVAL=25

PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
REV_SCALE_LR_MULT=10          # fixed at the standing default -- see header
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
  --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --rev_scale_lr_mult "${REV_SCALE_LR_MULT}"
  --optimizer muon --muon_lr "${MUON_LR}"
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-reversible-lrsweep
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
  --arch reversible --rev_regime vf_scaling
  --rev_embed narrow --rev_lift dup --rev_readout sum
  --mem_efficient --mem_efficient_segment "${N_LAYER}"
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Only warmup_steps/muon_warmup_steps/lr_schedule/decay_frac vary -- see header.
NAMES=(
  ref_vf
  wradam_vf
  w20pct_vf
  muon_nowarmup_vf
  wsd_wradam_vf
)
TASK_ARGS=(
  "--warmup_steps 300 --run_name ref_vf"
  "--warmup_steps 40  --run_name wradam_vf"
  "--warmup_steps 600 --run_name w20pct_vf"
  "--warmup_steps 300 --muon_warmup_steps 1 --run_name muon_nowarmup_vf"
  "--lr_schedule wsd --decay_frac 0.2 --warmup_steps 40 --run_name wsd_wradam_vf"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL12_vf_sched.slurm"; exit 1
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
