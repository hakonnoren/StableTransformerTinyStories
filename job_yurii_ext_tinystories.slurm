#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_yurii_ext_tinystories.slurm — TinyStories 'small' (12L/12H/768d), the
# paper's own TinyStories recipe, 5 arms, no reversible backprop anywhere.
#
# The question: do the two determinant-preserving extensions in yurii_ext.py
# (adaptive damping, gyro mixer) buy anything over the momentum block they
# perturb, at the SAME layer determinant and the SAME oracle budget?
#
#   task 0  baseline    GD + Lie-Trotter (nanoGPT)          arch=baseline
#   task 1  yurii_lt    Nesterov + Lie-Trotter (the paper)  arch=yurii_lt
#   task 2  polyak      Polyak + Lie-Trotter, shear form    arch=yurii_ext
#   task 3  adaptive    polyak + channel-wise damping       arch=yurii_ext --yx_adaptive
#   task 4  gyro        polyak + velocity phase mixer       arch=yurii_ext --yx_gyro
#
# WHY TASK 2 EXISTS (it is not padding). Tasks 3/4 are perturbations of
# *Polyak* in yurii_ext.py's read-only-LNv shear form. `yurii_lt` is *Nesterov*
# and it OVERWRITES the velocity with LNv(v). So "adaptive vs yurii_lt" would
# mix three changes at once: Nesterov->Polyak, LNv placement, and the mechanism
# under test. Task 2 is the arm that isolates the mechanism -- tasks 3 and 4 are
# bit-identical to it at initialization (verified: `python yurii_ext.py`), so
# any gap over task 2 is the mechanism and nothing else. Tasks 0/1 are the
# external anchors. Dropping task 2 makes tasks 3/4 uninterpretable; if GPU
# budget forces a cut, cut task 0 instead (its number is already known, below).
#
# Extensions cost 2 scalars per block and zero extra Attn/MLP calls, and all
# three yurii_ext arms have identical per-token log|det| = n_embd*sum_l
# log(beta_A beta_M) -- printed at startup by every arm, so it can be checked
# rather than trusted.
#
# ── Parameter parity at this geometry (measured, not estimated) ─────────────
#   baseline                123,587,328 non-positional  (paper: 123.6M, exact)
#   yurii_lt                162,230,088     "           (+38.6M = v0 token table)
#   polyak                  162,230,088     "           (identical to yurii_lt)
#   adaptive / gyro         162,230,112     "           (+24 = 2 scalars x 12L)
# So tasks 2/3/4 differ by 24 parameters out of 162M. Whatever separates them
# is the update rule, not capacity.
#
# ── Paper recipe (Tables 1, 2; App. A.1/A.3) — reproduced exactly ────────────
#   12L/12H/768d, block 1024, dropout 0, no bias, GPT-2 BPE padded to 50304
#   10k steps, 1k warmup, cosine to 0.1x peak, bf16, grad clip 1.0
#   batch 30 x grad_accum 16 = 491,520 tokens/step  => 4.92B tokens (~10.4 epochs)
#   Muon LR 0.02 (matrices) + AdamW 6e-4 (rest), AdamW betas (0.9, 0.95),
#   Muon momentum 0.95, embeddings wd 0.1, norms/scalars wd 0,
#   scalar update-rule group at 5x LR  <-- NOTE: repo default is 10x, the paper
#   says 5x (App. A.3), so --scalar_lr_mult 5 is passed explicitly below. This
#   group carries beta/gamma/eta AND the new eps/omega, so it is the one knob
#   that directly sets how fast the extensions can move away from zero.
#   Eval every 100 steps over 160 batches; v0 token+pos tables on for all
#   momentum arms (App. A.1), i.e. --no_v0_init is deliberately NOT passed.
#
# ── Reference numbers ───────────────────────────────────────────────────────
# Paper, TinyStories small, best/final val (Table 2):
#   GD+Lie-Trotter 1.106/1.114 | Polyak+LT 1.083/1.096 | Nesterov+LT 1.078/1.090
# Do NOT expect those absolute values here. This repo's own baseline at this
# exact geometry reached best_val 1.05216 (W&B 5rfufypa, 12L/768d TinyStories,
# 24x20, 10001 steps) -- lower than the paper's 1.106 because the TinyStories
# preprocessing and val split differ. The ORDERING is the thing to check:
# baseline > polyak > nesterov is the paper's result, and ~1.05 (not ~1.11) is
# the number this repo's baseline should land near.
#
# ── Eval note ───────────────────────────────────────────────────────────────
# 160 x 30 x 1024 = 4,915,200 tokens vs this repo's tinystories_val.bin at
# 4,765,918 tokens -- so one eval is ~1.03 complete passes over the validation
# set. Every arm therefore sees the identical eval data at the identical step
# (shared seed 1337), which is what makes the PAIRED per-eval comparison in
# analyze_wandb.py far tighter than any single final number.
#
# ── Time budget ─────────────────────────────────────────────────────────────
# Measured anchor: this repo's baseline, same geometry, same 491,520
# tokens/step, 10001 steps = 8.1h on an A100 (170,030 tokens/s). Momentum arms
# carry a second stream (~1.25x), eval at 160/100 rather than 50/200 adds ~3%,
# cheap_metrics a few % more => ~11h worst arm on A100, less on H100.
# --time=20:00:00 is ~1.8x that.
#
# ── Memory ──────────────────────────────────────────────────────────────────
# 30x16 at 12L/768d with two streams: job_owt_medium.slurm already runs the
# paper's 30x16 with arch=yurii_lt at 24L/1024d on gpu:h100:1, and this is
# ~3.5x smaller in activations. If you must fall back to gpu:a100:1 (40GB) and
# it OOMs, switch to BATCH_SIZE=24 / GRAD_ACCUM=20 -- identical 491,520
# tokens/step, measured to fit at this geometry in the runs cited above.
#
# Usage:   sbatch job_yurii_ext_tinystories.slurm
#          sbatch --array=2-4 job_yurii_ext_tinystories.slurm   # mechanism only
# Analyze: python analyze_wandb.py --project sympformer-yurii-ext \
#              --filter_config n_embd=768 block_size=1024 n_layer=12 \
#              --format pdf --out_dir plots/yurii_ext
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=yurii_ext_ts
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                 # verify name: sinfo -o "%N %G" | sort -u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-4                       # 0=baseline 1=yurii_lt 2=polyak 3=adaptive 4=gyro
#SBATCH --output=output_yurii_ext_ts_%A_%a.txt
#SBATCH --error=output_yurii_ext_ts_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=yurii_ext_ts_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: paper's TinyStories 'small' ───────────────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum: the paper's own 30x16 = 491,520 tokens/step ─────────────────
BATCH_SIZE=30
GRAD_ACCUM=16

# ── 10k-step TinyStories recipe (paper Table 1) ──────────────────────────────
MAX_STEPS=10000
WARMUP_STEPS=1000
EVAL_INTERVAL=100
EVAL_BATCHES=160
LOG_INTERVAL=50
SAMPLE_INTERVAL=1000        # qualitative only; 100 would cost wall-clock for nothing

PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
SCALAR_LR_MULT=5            # paper App. A.3; repo default is 10 -- see header
DATASET=tinystories         # existing GPT-2 BPE bins

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
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM}"
  --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}"
  --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --device "${DEVICE}" --plot
  --lr_schedule cosine --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --optimizer muon --muon_lr "${MUON_LR}" --muon_momentum 0.95
  --betas 0.9 0.95 --grad_clip 1.0
  --scalar_lr_mult "${SCALAR_LR_MULT}"
  --amp_dtype bf16
  --attn_impl sdpa
  --seed 1337
  --wandb --wandb_project sympformer-yurii-ext
  --sample_interval "${SAMPLE_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Every arm: 1 Attn + 1 MLP oracle call per block, no reversible backprop,
# no --mem_efficient, no noise, no restarts.
NAMES=(
  baseline_TS
  yuriilt_TS
  polyak_TS
  adaptive_TS
  gyro_TS
)
TASK_ARGS=(
  "--arch baseline                            --run_name baseline_TS"
  "--arch yurii_lt                            --run_name yuriilt_TS"
  "--arch yurii_ext                           --run_name polyak_TS"
  "--arch yurii_ext --yx_adaptive             --run_name adaptive_TS"
  "--arch yurii_ext --yx_gyro                 --run_name gyro_TS"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_yurii_ext_tinystories.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "Tokens/step: $(( BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE )) (paper: 491,520)"
log "OUTDIR=${OUTDIR}"

if [[ ! -f "data/${DATASET}_train.bin" || ! -f "data/${DATASET}_val.bin" ]]; then
  log "FATAL: missing data/${DATASET}_{train,val}.bin -- run preprocess_tinystories.py first"
  exit 1
fi
log "data OK: $(ls -lh data/${DATASET}_*.bin | awk '{print $9"="$5}' | tr '\n' ' ')"

# ── yurii_ext preflight ───────────────────────────────────────────────────────
# Seconds of CPU. Re-verifies on THIS box's torch build that the block
# determinant matches an autograd Jacobian, the inverse round-trips, and every
# arm is bit-identical to polyak at init -- i.e. the premise the whole
# comparison rests on. Cheap insurance against burning 11 GPU-hours on a stack
# whose fixed-volume claim silently stopped holding.
if [[ "${ID}" -ge 2 ]]; then
  log "yurii_ext preflight:"
  run python3 -u yurii_ext.py || { log "FATAL: yurii_ext self-test failed"; exit 1; }
fi

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW rather than dying on a CUDA OOM after many hours.
# NEED=20000 is a floor for 30x16 at this geometry with two streams; see header
# for the 24x20 fallback if you land on a 40GB card and it will not fit.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [[ -n "${GPU_TOTAL}" && "${GPU_TOTAL}" =~ ^[0-9]+$ ]]; then
  case "${GPU_NAME}" in
    *P100*|*V100*|*P40*|*K80*|*T4*)
      log "FATAL: got '${GPU_NAME}', which predates Ampere -- no native bf16 and no flash"
      log "       SDPA. Resubmit with --gres=gpu:h100:1 (or a100)."
      exit 1 ;;
  esac
  NEED=20000
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
