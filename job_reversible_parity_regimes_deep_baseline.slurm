#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes_deep.slurm — the volume-regime question at 4x
# the depth: does the reversible/baseline gap seen at 'small' (12L) hold, widen,
# or vanish at 48L? And does --mem_efficient_segment n_layer ("segL"), validated
# at 12L in job_reversible_parity_regimes_test.slurm, still hold up when the
# inverse chain it walks is 4x as long?
#
# Same geometry as MODEL_SIZE=small (job_reversible_parity_regimes.slurm) except
# N_LAYER: 12 -> 48. N_HEAD=12, N_EMBD=768, BLOCK_SIZE=1024 unchanged. This file
# has no MODEL_SIZE switch -- unlike the regimes job it is ONE fixed geometry,
# so there is nothing to switch between.
#
#   ID  run dir                        regime        backprop
#    0  baseline_L48                   —             eager (no checkpointing exists for it)
#    1  reversible_vpb_baseline_segL   vpb_baseline  mem_efficient, segment=48 (=n_layer)
#    2  reversible_vf_scaling_segL     vf_scaling    mem_efficient, segment=48 (=n_layer)
#
# Trimmed from the regimes job's 5 arms to these 3 -- vpb_baseline (frozen
# gamma=alpha=0, purely additive) and vf_scaling (volume free, unconstrained)
# are the two ends of the conditioning question that job_reversible_parity_regimes_test.slurm
# already established at 12L: vpb_baseline sat at the run-to-run noise floor
# under every backprop mode, vf_scaling was the one that could develop a
# monotone bias. vpm_scaling/vpb_scaling sit strictly between them, so this arm
# selection covers the extremes at the depth where a real effect is most likely
# to show. All three still run --rev_embed narrow (see "Parameter parity" below)
# and --optimizer muon at the SAME peak_lr/muon_lr/warmup fraction as 'small' --
# only depth, batch/accum split and steps are touched.
#
# Why baseline can't use segL, and why that changes the array design
# --------------------------------------------------------------------
# --mem_efficient hard-errors unless --arch reversible (train.py: "needs --arch
# reversible"). Baseline has no reversible structure to invert and no gradient
# checkpointing wired in at all, so its activation memory scales LINEARLY with
# n_layer with no mitigation -- exactly the 4x that reversible+segL is built to
# avoid. That asymmetry is why baseline gets a smaller micro-batch than the
# reversible arms below (same tokens/step, see "Batch/accum split" below), and
# why this is 3 arms sharing one --time directive rather than a size-switchable
# job: baseline and the segL arms have genuinely different resource profiles at
# this depth, and over-requesting for the cheaper arm costs only queue
# priority, not correctness.
#
# Parameter parity
# ----------------
# Measured directly (revformer/test_train_reversible.py's n_params helper, same
# ModelConfig used here):
#
#   n_layer   baseline / vpb_baseline (narrow)   vf_scaling (narrow)
#   12        124,373,760                        124,392,192  (+18,432, +0.015%)
#   48        379,233,024                        379,306,752  (+73,728, +0.019%)
#
# baseline and vpb_baseline are IDENTICAL at every depth (vpb_baseline's
# gamma/alpha are frozen buffers, not parameters); vf_scaling's gamma/alpha
# vectors are the only residual, 2*n_embd*n_layer of them, and they stay at
# ~0.02% of total params regardless of depth. So going deep does not change the
# parity story: a difference between arms 0 vs 1 vs 2 is still a difference in
# architecture/coupling, not capacity. 379.2M at 48L is ~3.05x the 12L count
# (not 4x), because the token/pos tables and tied lm_head (~39.4M, narrow) do
# not scale with depth -- only the 48 blocks' attn/mlp matrices do.
#
# Batch/accum split -- same tokens/step, different micro-batch per arm
# ---------------------------------------------------------------------
# batch_size x grad_accum_steps x block_size = tokens/step is held at 491,520
# for ALL THREE arms (identical to 'small'), so accumulation is a pure
# memory-vs-speed choice and does not change the gradient (loss-scaled, as
# elsewhere in this repo) -- arms 0 vs 1 vs 2 differ only in architecture, not
# in effective batch.
#
#   reversible (segL):  BATCH_SIZE=12  GRAD_ACCUM_STEPS=40   (12x1024x40 = 491,520)
#   baseline (eager):   BATCH_SIZE=6   GRAD_ACCUM_STEPS=80   (6x1024x80  = 491,520)
#
# 12x40 for the reversible arms is exactly the split job_reversible_parity_regimes_test.slurm
# measured at 12L (--mem_efficient_segment ${N_LAYER}), so the GPU-memory
# estimate below has real numbers under it rather than a guess from scratch.
# Baseline gets HALF that micro-batch because it has no way to keep its stack
# activation memory flat across 4x the depth (see above) -- this is the same
# "halve batch, double accum, gradient unchanged" move the other jobs in this
# repo use for reversible arms that OOM where baseline fits, just pointed the
# other way, since here baseline is the arm with no memory-saving option.
#
# GPU memory -- ESTIMATE, ANCHORED to job_reversible_parity_regimes_test.slurm's
# measurements, NOT measured at this depth. Verify against train.py's logged
# peak_mem_gb on the first few log lines before trusting this for a bigger run.
# ------------------------------------------------------------------------------
# That job measured, at 12L / BATCH=12x40 / --attn_impl sdpa / --cheap_metrics:
#   eager peak_mem_gb   13.93 GB   (stack term ~3.9GB of it; loss head ~6-8GB;
#                                    params+optimizer ~1.65GB at 124.4M params)
#   segL  peak_mem_gb   ~9.34-9.40 GB predicted (measured seg=0 at 9.30 GB,
#                                    segL adds exactly one more (B,T,2*n_embd)
#                                    tensor, ~38MB at this size)
# The stack term is the one that is FLAT IN DEPTH under segL (memory is
# ceil(n_layer/k) tensors of (B,T,2*n_embd); k=n_layer gives exactly 1,
# regardless of n_layer) -- that flatness is the entire reason 48L is viable at
# all. What does NOT stay flat is params+optimizer: 48L has ~3.05x the params
# (see table above), so at the same batch that term goes ~1.65GB -> ~5.0GB.
# Loss head (logits at B*T*vocab) depends only on batch/block_size/vocab, not
# depth, so it is unchanged by going deeper, but IS halved by baseline's
# smaller micro-batch below.
#
#   reversible segL, 48L, BATCH=12:  stack(~3.9, flat) + loss_head(~6-8, same
#                                    batch as the 12L measurement) +
#                                    params+opt(~5.0, 3.05x) =~ 12.7-14.7 GB
#   baseline, 48L, BATCH=6 (half):  stack(~3.3->~13.2 at 4x depth, THEN halved
#                                    for batch =~6.6) + loss_head(~3-4, halved
#                                    batch) + params+opt(~5.0, same as above,
#                                    batch-independent) =~ 14.6-15.6 GB
#
# Both land comfortably under a 40GB A100 with this batch split; do NOT skip
# the GPU preflight below or the first-log-line peak_mem_gb check, since every
# number above is compounded from two different jobs' measurements at a
# geometry this was never run at.
#
# Compute time -- ESTIMATE, same anchoring, NOT measured at this depth
# ---------------------------------------------------------------------
# job_reversible_parity_regimes_test.slurm measured, at 12L / BATCH=12x40 / sdpa:
#   eager   vpb 125,418 tok/s (3.92 s/step)   vf 116,398 tok/s (4.22 s/step)
#   segL   ~vpb  85,701 tok/s (5.74 s/step)  ~vf  79,291 tok/s (6.20 s/step)
#     (segL times taken from that job's measured seg=0 column -- its own words:
#     "segL inverts as much as seg=0" -- since segL was not itself re-timed there)
# job_reversible_parity_regimes.slurm separately measured baseline eager at
# ~3.05 s/step (12L, sdpa smoke test).
#
# Scaling assumption: per-step time scales linearly with n_layer (attn/mlp
# FLOPs dominate at this width; token/pos embedding and the lm_head are
# depth-independent, so the true scaling is somewhat BELOW 4x -- treating it as
# exactly 4x is the deliberately pessimistic choice, consistent with how this
# repo already books GPU-memory estimates). The test job's own eager-replicate
# pair (identical config and seed) differed 22% in tokens_per_sec node-to-node;
# applying that as a worst-case margin on top of the 4x:
#
#   arm                  12L s/step   x4 (depth)   x1.22 (node variance, worst)
#   baseline (eager)      3.05          12.2          14.9 s/step
#   vpb segL              5.74          23.0          28.1 s/step
#   vf  segL              6.20          24.8          30.3 s/step
#
# At MAX_STEPS=4000 (see "Why 4000 steps" below), plus ~0.75h for evals (drift
# probe + cheap_metrics + sampling, 20 evals) and startup/data loading:
#
#   baseline:   4000 x 14.9s = 16.6h  + 0.75h =~ 17.3h
#   vpb segL:   4000 x 28.1s = 31.2h  + 0.75h =~ 32.0h
#   vf  segL:   4000 x 30.3s = 33.7h  + 0.75h =~ 34.5h
#
# --time below is sized for the WORST arm (vf segL, ~34.5h) with ~1.4x headroom
# over that already-pessimistic estimate, rounded to 48h. baseline needs far
# less (~17h) but shares the same array directive -- over-requesting costs only
# queue priority (see the "Batch/accum split" note above for why this is the
# same asymmetry as the memory budget).
#
# CHECK THE PARTITION ALLOWS 48H: sinfo -p GPUQ -o "%P %l". If it does not,
# split this into two submissions instead of shrinking MAX_STEPS blind:
#   sbatch --array=0-0   --time=20:00:00 job_reversible_parity_regimes_deep.slurm   # baseline
#   sbatch --array=1-2   --time=48:00:00 job_reversible_parity_regimes_deep.slurm   # segL arms
# (both read the same NAMES/TASK_ARGS below; --array on the command line
# overrides the #SBATCH directive.)
#
# Why 4000 steps, not the 'small' recipe's 10001
# -----------------------------------------------
# At 10001 steps the worst arm's estimate above scales to ~34.5h x
# (10001/4000) =~ 86h -- multiple days, and likely outside most partitions'
# walltime cap. 4000 steps matches the horizon job_reversible_parity_regimes_test.slurm
# already validated segL at (20x its original 200-step smoke test), so this
# reuses a horizon with a known interpretation rather than picking a new one.
# WARMUP_STEPS is scaled to keep the same ~10% fraction as 'small' (1000/10001)
# for the same reason that job scaled it for its own 4000-step test: the run
# should walk through warmup AND into cosine decay, not sit in a linear ramp
# the whole time. As with that job, losses here are comparable BETWEEN THESE
# THREE ARMS but not directly against the full 10001-step 'small'/'medium' sweeps.
#
# Recon drift at 4x the depth
# ---------------------------
# --track_recon_drift is on for all arms (silently ignored for baseline, which
# has no invertible stack -- train.py reports this at startup rather than
# erroring). This is the metric most likely to reveal something new at 48L:
# error amplification through the inverse chain goes like exp(sum_l gamma_l),
# and vf_scaling's gamma/alpha are unconstrained and now have 4x as many terms
# to accumulate over. At 12L/4000 steps this had not yet materialized even for
# vf_scaling; 48L is a real stress test of that bound, not a repeat of it.
#
# Usage:    sbatch job_reversible_parity_regimes_deep.slurm
# Analyze:  python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#               n_layer=48 attn_impl=sdpa --format pdf --out_dir plots/revparity_deep
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_deep_L48
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0                        # 0=baseline 1=vpb_baseline+segL 2=vf_scaling+segL
#SBATCH --output=output_revdeep_%A_%a.txt
#SBATCH --error=output_revdeep_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revdeepL48_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: 'small' with N_LAYER 12 -> 48, everything else unchanged ──
N_LAYER=48
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024

# ── Batch/accum split (see header) -- same tokens/step, different micro-batch
BATCH_SIZE_REV=12;  GRAD_ACCUM_REV=40    # reversible + segL: 491,520 tokens/step
BATCH_SIZE_BASE=6;  GRAD_ACCUM_BASE=80   # baseline, no checkpointing: 491,520 tokens/step

# ── Steps (see "Why 4000 steps" above) ────────────────────────────────────────
MAX_STEPS=4000
WARMUP_STEPS=1000              # ~10% fraction, matching 'small' (1000/10001)
EVAL_INTERVAL=200              # 19 evals at 200..3800 (+ the post-loop final eval)
EVAL_BATCHES=20
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
#
# --rev_lift/--rev_readout only apply to --rev_embed narrow (passing them with
# wide is a hard error). dup/sum are also the defaults, and the only combo that
# leaves no dead sublayer -- see job_reversible_parity_regimes.slurm for why.
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

NAMES=(
  #baseline_L48
  reversible_vpb_baseline_segL
  #reversible_vf_scaling_segL
)
TASK_ARGS=(
  #"--arch baseline --batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --run_name baseline_L48"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpb_baseline_segL_L48"
  #"--arch reversible --rev_regime vf_scaling   ${NARROW} --batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_scaling_segL_L48"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes_deep.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Geometry: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW with a readable message rather than dying on a
# CUDA OOM after ~17-35h. train.py forces bf16 autocast on any CUDA device,
# which needs Ampere (sm80+): P100 is sm60, V100 is sm70, neither has native
# bf16 or flash SDPA. NEED is a rough floor (see the memory estimate above,
# ~13-16GB expected), set well below the 40GB an a100:1 provides so it only
# catches a clearly-wrong allocation, not a tight one.
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
