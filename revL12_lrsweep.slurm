#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# revL12_lrsweep.slurm — cheap LR/warmup screening sweep at 12L ('small'),
# baseline arch only (no segL/rev_scale_lr_mult confounds), 3000 steps
# (<10k, per the ask). 9 arms, each isolating ONE change from arm 0 (the
# current standing recipe) except arm 8 which combines two literature-backed
# ideas. Requires the train.py changes made alongside this file:
#   --muon_warmup_steps  (default -1 = same as --warmup_steps): lets Muon's
#     schedule be decoupled from AdamW's.
#   --lr_schedule {cosine,wsd}, --decay_frac: adds warmup-stable-decay
#     (trapezoidal) as an alternative to the historical cosine shape.
#
# RESUBMIT STATUS: 7/9 arms finished (0,1,2,3,4,6,7). Arms 5 (muon_nowarmup)
# and 8 (wsd_wradam) never registered in W&B at all -- the two arms that need
# the NEW flags above, everything else only uses the pre-existing
# --warmup_steps. Consistent with the cluster's train.py not having these
# changes yet: an unrecognized-argument error exits before wandb.init() ever
# runs, so nothing gets logged (not a "failed" run, just absent). --array
# below is trimmed to just these two -- sync the updated train.py first.
#
# --gres=gpu:1 CAVEAT: first resubmission attempt (array_job=25074093, task 5)
# landed on a Tesla P100 (idun-11-14) and correctly bailed at the preflight
# below rather than crashing mid-training -- but it still burned that task's
# turnaround for nothing. SLURM's --gres has no "not this type" option, so
# --exclude now explicitly lists the known P100/V100 node ranges (P100:
# idun-11-[01-14], V100: idun-06-[08-12] + idun-07-[01-03] + idun-11-[15-17])
# alongside the original bad-A100-node exclude. If a P100/V100 still gets
# through, IDUN has added a node to one of those pools outside this range.
#
# Finished results (best_val, for reference): ref_w10pct 1.1013, w2pct 1.0933,
# w5pct 1.0951, w20pct 1.1107, wradam 1.0917 (best so far), lr_2x 1.1077,
# lr_0p5x 1.1198 (worst). Shorter warmup won every comparison run so far;
# arm 5/8 fill in the two pieces (Muon-only, and a different schedule shape)
# that result doesn't yet cover.
#
# Rationale per arm (see conversation for full literature discussion):
#   0 ref_w10pct       current recipe: cosine, warmup=300 (10% of 3000), 6e-4/0.02
#   1 w2pct            warmup=60 (2%) -- is 10% actually needed?
#   2 w5pct            warmup=150 (5%)
#   3 w20pct           warmup=600 (20%) -- does MORE warmup help instead?
#   4 wradam           warmup=40 for BOTH optimizers -- Ma & Yarats' RAdam-
#                       equivalence rule of thumb, 2/(1-beta2) with beta2=0.95.
#                       Independent of max_steps entirely, unlike the 10%
#                       convention -- if this matches arm 0, the whole
#                       scale-warmup-with-max_steps habit may be unnecessary.
#   5 muon_nowarmup    AdamW keeps 300 (10%), Muon warmup=1 -- tests Keller
#                       Jordan's finding that Muon's orthogonalized update
#                       doesn't need warmup (he removed it entirely in the
#                       nanoGPT speedrun), in isolation from AdamW's schedule.
#   6 lr_2x            peak_lr=1.2e-3, muon_lr=0.04 (2x), warmup unchanged
#   7 lr_0p5x          peak_lr=3e-4, muon_lr=0.01 (0.5x), warmup unchanged
#   8 wsd_wradam       --lr_schedule wsd (modded-nanogpt's default: linear
#                       warmup, then CONSTANT at peak, then linear decay over
#                       the final --decay_frac), warmup=40 (both optimizers),
#                       decay_frac=0.2 -- tests a genuinely different SCHEDULE
#                       SHAPE, not just a different warmup length within cosine.
#
# All arms otherwise identical: n_layer=12, batch=24/accum=20 (491,520
# tokens/step, matching the rest of this investigation), arch=baseline (fast,
# no segL overhead, no rev_scale_lr_mult confound), min_lr_ratio=0.1.
#
# Time: baseline@12L is unmeasured directly, but scaling from the 48L ratio
# (baseline 9.83 s/step vs reversible+segL 16.88 s/step, 0.58x) against the
# CONFIRMED 12L reversible+segL rate (5.8 s/step) gives ~3.4 s/step estimated.
# 3000 steps -> ~2.8h/arm. --time below adds real margin since this is
# extrapolated, not measured.
#
# This is a SCREENING sweep, not a final answer: the paper's own 12L recipe
# bottoms out near step 7.6k, so whichever arm(s) look best here should get
# ONE confirmation run at the full 10001-step budget before replacing the
# standing default anywhere else in this investigation.
#
# Separate wandb project (sympformer-reversible-lrsweep) -- this is
# hyperparameter search, not an architecture-comparison result, and shouldn't
# mix with sympformer-reversible-revback's real 12L segL runs.
#
# Usage:   sbatch revL12_lrsweep.slurm
# Analyze: python analyze_wandb.py --project sympformer-reversible-lrsweep \
#              --format pdf --out_dir plots/revL12_lrsweep
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revL12_lrsweep
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]   # P100/V100 pools + the bad-node A100, see header
#SBATCH --gres=gpu:1                       # ANY GPU, not just A100 -- see header
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=5,8                        # only the arms that never registered in wandb -- see header
#SBATCH --output=output_revL12_lrsweep_%A_%a.txt
#SBATCH --error=output_revL12_lrsweep_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revL12_lrsweep_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small', baseline only ────────────────────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
BATCH_SIZE=24
GRAD_ACCUM_STEPS=20          # 491,520 tokens/step

# ── Screening horizon (<10k steps, per the ask) ──────────────────────────────
MAX_STEPS=3000
EVAL_INTERVAL=150             # 20 evals
EVAL_BATCHES=50
LOG_INTERVAL=25

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
  --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}"
  --optimizer muon
  --attn_impl sdpa
  --wandb --wandb_project sympformer-reversible-lrsweep
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
  --arch baseline
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Each entry is the FULL set of schedule/LR args for that arm (see header for
# the reasoning behind each one). warmup_steps/muon_warmup_steps/peak_lr/
# muon_lr/lr_schedule/decay_frac are the only things that vary across arms.
NAMES=(
  ref_w10pct
  w2pct
  w5pct
  w20pct
  wradam
  muon_nowarmup
  lr_2x
  lr_0p5x
  wsd_wradam
)
TASK_ARGS=(
  "--warmup_steps 300 --peak_lr 6e-4  --muon_lr 0.02 --run_name ref_w10pct"
  "--warmup_steps 60  --peak_lr 6e-4  --muon_lr 0.02 --run_name w2pct"
  "--warmup_steps 150 --peak_lr 6e-4  --muon_lr 0.02 --run_name w5pct"
  "--warmup_steps 600 --peak_lr 6e-4  --muon_lr 0.02 --run_name w20pct"
  "--warmup_steps 40  --peak_lr 6e-4  --muon_lr 0.02 --run_name wradam"
  "--warmup_steps 300 --muon_warmup_steps 1 --peak_lr 6e-4  --muon_lr 0.02 --run_name muon_nowarmup"
  "--warmup_steps 300 --peak_lr 1.2e-3 --muon_lr 0.04 --run_name lr_2x"
  "--warmup_steps 300 --peak_lr 3e-4  --muon_lr 0.01 --run_name lr_0p5x"
  "--lr_schedule wsd --decay_frac 0.2 --warmup_steps 40 --peak_lr 6e-4 --muon_lr 0.02 --run_name wsd_wradam"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch revL12_lrsweep.slurm"; exit 1
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
# generous floor -- baseline@12L at this batch should need well under 20GB.
# --gres=gpu:1 (no type pin) lets this land on ANY of IDUN's ~268 GPUs, not
# just the 104 A100s -- this job's memory footprint is small enough that
# H100/H200/B300 (all newer/larger than A100) are trivially fine, and this
# preflight's P100/V100/P40/K80/T4 name-match below still catches the older
# cards that genuinely lack bf16/flash SDPA, so relaxing the gres type is safe.
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
  NEED=10000
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
