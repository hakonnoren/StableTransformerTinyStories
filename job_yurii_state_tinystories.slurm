#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_yurii_state_tinystories.slurm — three state-structure variants
# (yurii_state.py) x TWO SEEDS, on the paper's TinyStories 'small' recipe.
# No reversible backprop anywhere.
#
#   task 0  thermostat     seed 1337      task 3  thermostat     seed 1338
#   task 1  dual_momentum  seed 1337      task 4  dual_momentum  seed 1338
#   task 2  two_stream     seed 1337      task 5  two_stream     seed 1338
#
# ── WHY TWO SEEDS AND ONLY THREE ARMS ───────────────────────────────────────
# The effects in play are 0.001-0.004 nats. theory/volume_regime_findings.md
# already says the quiet part out loud about the last comparison of this kind:
# "there is one training seed per arm, so seed-to-seed variance is unmeasured
# and the ordering is not yet established as a property of the regimes rather
# than of these five runs. That is the single highest-value next experiment."
# Six single-seed arms would repeat exactly that. Three arms x two seeds costs
# the same six GPU-slots and actually licenses the conclusion. multirate and
# the swap/LNv ablations are deferred, not cancelled -- they are one edit away
# (see the ABLATIONS block at the bottom of this header).
#
# With two seeds you can separate within-run eval noise (already ~0.99
# correlated across arms at a shared seed, so paired per-eval differences are
# tight) from between-seed variance, which is the term that actually decides
# whether an ordering is real. Two seeds gives a crude but honest estimate of
# it; it does not give a confidence interval. If a gap is smaller than the
# seed-to-seed spread, report it as "not resolved at n=2", not as a win.
#
# ── ANCHORS COME FROM THE OTHER JOB ─────────────────────────────────────────
# Logs to the SAME W&B project as job_yurii_ext_tinystories.slurm on a
# byte-identical recipe, so baseline / yurii_lt / polyak are NOT repeated here
# -- run that job too. Seed 1337 arms there pair exactly with the seed 1337
# arms here (same eval batches at the same step). NOTE that job is single-seed;
# to compare against a seed-1338 arm here you need a seed-1338 polyak, which is
# one extra task there (`--seed 1338`).
#
# The reference arm is `polyak` (arch yurii_ext, no flags): at beta_init 0.9 all
# three variants here have *exactly* the same per-token log|det| as it,
# -n_embd * 12 * log 0.81, verified by `python yurii_state.py`. The thermostat
# only achieves that in its default channel mode -- see below.
#
# ── WHAT THESE THREE ARMS ARE FOR ───────────────────────────────────────────
# thermostat     Can a retained, invertible feedback state replace velocity
#                LayerNorm? Runs channel-wise by default: d-wide r, zeta
#                centered over channels (causal, unlike token-centering), which
#                pins the determinant exactly and lets it see the anisotropy
#                that LN actually fixes. `--ys_thermostat_scalar` reverts to the
#                one-scalar-per-token version.
# dual_momentum  Do attention and MLP want separate memories with independent
#                decay rates, rather than superposing in one velocity?
# two_stream     THE FALSIFIER. Same oracle budget, same 2d state, same
#                determinant, but neither stream is a velocity and their roles
#                swap every layer. If this matches yurii_lt, the momentum
#                interpretation is not what is doing the work -- two coupled
#                invertible streams are. Per-channel a/b by default, since this
#                repo's reversible runs found channel-dependent contraction is
#                what the network actually learns.
#
# ── Parameter parity (measured at 12L/12H/768d, non-positional) ─────────────
#   polyak (other job)      162,230,088          +0    log|det|/tok -1942.005615
#   thermostat              162,220,944      -9,144                 -1942.005615
#   dual_momentum           162,239,304      +9,216                 -1942.005615
#   two_stream              162,239,256      +9,168                 -1942.005432
# Auxiliary streams share one v0 table by default; giving dual_momentum its own
# second table would add 38.6M (+31%) and wreck parity (--ys_no_share_v0).
# two_stream's log|det| differs in the 7th significant figure only because 768
# fp32 logs are summed instead of one log scaled by 768 -- summation rounding,
# not a design difference. Do not go looking for a cause.
#
# ── Time / memory ───────────────────────────────────────────────────────────
# Anchor: this repo's baseline, same geometry and tokens/step, 10k steps = 8.1h
# on an A100. thermostat and dual_momentum carry a third stream (~1.3x), so
# ~12h worst arm; --time=20:00:00. NEED=24000 MiB below is sized for those.
# On a 40GB card use BATCH_SIZE=24/GRAD_ACCUM=20 (identical tokens/step).
#
# ── ABLATIONS, deferred ─────────────────────────────────────────────────────
# Append to NAMES/TASK_ARGS and widen --array when there is budget:
#   multirate 1a2m  "--ys_variant multirate --ys_n_mlp 2 --n_layer 6 ..."   (6 attn / 12 mlp)
#   + attn memory   "... --ys_attn_memory"                (invertible decaying A field)
#   two_stream      "--ys_variant two_stream --ys_no_swap ..."   (does swapping matter?)
#   thermostat      "--ys_variant thermostat --ys_drift_norm on ..." (thermostat AND LNv)
#   thermostat      "--ys_variant thermostat --ys_thermostat_scalar ..." (original proposal)
#
# Usage:   sbatch job_yurii_state_tinystories.slurm
#          sbatch --array=0-2 job_yurii_state_tinystories.slurm   # seed 1337 only
# Analyze: python analyze_wandb.py --project sympformer-yurii-ext \
#              --filter_config n_embd=768 block_size=1024 n_layer=12 \
#              --format pdf --out_dir plots/yurii_state
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=yurii_state_ts
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                 # verify name: sinfo -o "%N %G" | sort -u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-5                       # 0-2 = seed 1337, 3-5 = seed 1338
#SBATCH --output=output_yurii_state_ts_%A_%a.txt
#SBATCH --error=output_yurii_state_ts_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=yurii_state_ts_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: paper's TinyStories 'small' ───────────────────────────────
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Recipe: identical to job_yurii_ext_tinystories.slurm. Do not drift. ──────
BATCH_SIZE=30
GRAD_ACCUM=16           # 30 x 16 x 1024 = 491,520 tokens/step
MAX_STEPS=10000
WARMUP_STEPS=1000
EVAL_INTERVAL=100
EVAL_BATCHES=160
LOG_INTERVAL=50
SAMPLE_INTERVAL=1000

PEAK_LR=6e-4
MUON_LR=0.02
MIN_LR_RATIO=0.1
SCALAR_LR_MULT=5        # paper App. A.3; carries beta/gamma/eta AND eps/rho/tau AND a/b
DATASET=tinystories

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
  --arch yurii_state
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
  --wandb --wandb_project sympformer-yurii-ext
  --sample_interval "${SAMPLE_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Three variants x two seeds. Run names carry the seed so W&B keeps them apart.
NAMES=(
  thermostat_s1337  dual_momentum_s1337  two_stream_s1337
  thermostat_s1338  dual_momentum_s1338  two_stream_s1338
)
TASK_ARGS=(
  "--ys_variant thermostat    --seed 1337 --run_name thermostat_s1337"
  "--ys_variant dual_momentum --seed 1337 --run_name dual_momentum_s1337"
  "--ys_variant two_stream    --seed 1337 --run_name two_stream_s1337"
  "--ys_variant thermostat    --seed 1338 --run_name thermostat_s1338"
  "--ys_variant dual_momentum --seed 1338 --run_name dual_momentum_s1338"
  "--ys_variant two_stream    --seed 1338 --run_name two_stream_s1338"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_yurii_state_tinystories.slurm"; exit 1
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

# ── yurii_state preflight ─────────────────────────────────────────────────────
# Seconds of CPU. Re-checks on THIS box's torch build that every variant's
# determinant matches an fp64 autograd Jacobian, that the channel thermostat is
# volume-pinned to fp64 (its whole claim), that the inverses are exact, that
# nothing leaks from the future, and that the thermostat's eps/rho gradients are
# live -- a zero init for either makes it provably untrainable, and the test
# asserts both that the shipped defaults avoid that and that the zero init
# would not.
log "yurii_state preflight:"
run python3 -u yurii_state.py || { log "FATAL: yurii_state self-test failed"; exit 1; }

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

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
