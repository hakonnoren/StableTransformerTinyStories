#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes_test.slurm — 200-step A/B of memory-efficient
# reversible backprop, for job_reversible_parity_regimes.slurm. Six arms:
# {vpb_baseline, vf_scaling} x {eager autograd, --mem_efficient_segment 1, 0}.
#
# THIS IS NOT AN EXPERIMENT ABOUT ARCHITECTURE. 200 steps at a 20-step warmup is
# nowhere near convergence and the LR schedule is compressed 50x, so do not
# compare vpb_baseline vs vf_scaling here. Comparing BACKPROP MODES within one
# arch is exactly what it is for.
#
#   ID  run dir                      regime        backprop
#    0  reversible_vpb_eager         vpb_baseline  ordinary autograd (reference)
#    1  reversible_vpb_seg1          vpb_baseline  mem_efficient, segment=1
#    2  reversible_vpb_seg0          vpb_baseline  mem_efficient, segment=0
#    3  reversible_vf_eager          vf_scaling    ordinary autograd (reference)
#    4  reversible_vf_seg1           vf_scaling    mem_efficient, segment=1
#    5  reversible_vf_seg0           vf_scaling    mem_efficient, segment=0
#
# The question
# -----------
# segment=0 runs the block stack under no_grad and reconstructs every activation
# in backward by algebraically inverting each block -- O(1) activation memory in
# depth, the actual payoff of reversibility. segment=1 stores every block's input
# and never inverts (i.e. plain gradient checkpointing), which is bit-exact but
# saves less.
#
# Measured offline on CPU: segment=1 gives bit-identical gradients in any
# precision; segment=0 gives ~2e-5 relative gradient error in fp32 but 1.2-2.3e-2
# in bf16 -- the bf16 sublayer recompute is a quantization floor, unaffected by
# gamma/alpha init or by segmenting. That looked disqualifying.
#
# It may not be. Reformer (Kitaev et al. 2020) ships exactly this reconstruction,
# additive coupling, with no precision handling whatsoever, and validates it on
# LEARNING CURVES rather than gradient error -- the paper never measures
# reconstruction error at all. And 1.6e-2 relative gradient error should be judged
# against gradient NOISE: at 480 sequences / 491,520 tokens per step, minibatch
# sampling noise is order 100% relative, ~2 orders of magnitude larger. So this
# job asks the question the way Reformer did: does the loss curve care?
#
# What to read off, per arch (0 vs 1 vs 2, and 3 vs 4 vs 5)
# ---------------------------------------------------------
#   * step-0 train_loss   MUST BE IDENTICAL to the last digit across all three
#                         modes of one arch. The forward is the same arithmetic in
#                         the same order -- only the BACKWARD differs -- and this
#                         was verified bit-identical offline. If step 0 differs at
#                         all, something is wrong; stop and investigate. This is
#                         the tripwire that makes the rest interpretable: every
#                         later divergence is attributable to gradients alone.
#   * loss curve          segment=1 should track eager essentially exactly
#                         (bit-exact gradients; expect ~0 divergence, growing only
#                         through optimizer chaos). segment=0 WILL drift. The
#                         question is the CHARACTER of the drift: if it plateaus
#                         and changes sign it is trajectory noise and bf16
#                         inversion is fine; if it grows monotonically it is bias
#                         and segment=1 is the answer. For calibration, the
#                         manual-vs-sdpa A/B plateaued at ~0.4% relative and
#                         changed sign repeatedly.
#   * peak_mem_gb         Expect a MODEST drop, roughly 14.4 -> ~11.4 (seg=1) ->
#                         ~10.6 GB (seg=0) at 'small'. Do NOT expect the 74x from
#                         the offline saved-for-backward measurement: that counted
#                         bytes stashed inside the STACK, and after --attn_impl
#                         sdpa the stack is only ~3.9GB of the 14.4GB peak. The
#                         rest is the loss head (logits at B*T*V = 618M elements,
#                         bf16 plus the fp32 cross_entropy upcast plus the saved
#                         log_softmax, ~6-8GB) and params+optimizer (~1.65GB),
#                         neither of which this touches. The real claim is that
#                         the stack term is FLAT IN DEPTH -- so the gap widens at
#                         24L/48L, which is what makes 'medium' viable.
#   * tokens_per_sec      seg=0 and seg=1 are SLOWER: one inverse plus one
#                         recomputed forward per block on top of the gradient,
#                         ~1.6x on the stack. This trade only makes sense when
#                         memory is the binding constraint.
#   * recon_drift_max     logged at every eval by --track_recon_drift. This is the
#                         interesting scientific output: vpb_baseline freezes
#                         gamma=alpha=0, so its drift should be CONSTANT, while
#                         vf_scaling's gamma/alpha learn, so its drift should
#                         EVOLVE -- error amplification through the inverse chain
#                         goes like exp(sum_l gamma_l), and vf_scaling constrains
#                         that sum not at all. A one-off measurement at init is
#                         not representative; this is why it is logged over time.
#
# NOTE: --cheap_metrics forces the manual attention path for one forward per eval
# on the small cheap_batch (the fused kernel never builds the attention matrix
# those diagnostics need), so peak_mem_gb includes one such forward at
# cheap_batch_size=8. All six arms use --attn_impl sdpa otherwise.
#
# Parameter parity
# ----------------
# Both arms share one geometry (whatever MODEL_SIZE selects). Attn/MLP
# already match per block; --rev_embed narrow additionally puts the token/pos
# tables, ln_f and the tied lm_head at n_embd (the baseline's exact shapes)
# instead of the full 2*n_embd state width, bridging to the state
# parameter-free (lift dup: x = z = emb; readout sum: x + z). The residual is
# only the gamma/alpha vectors, 2*n_embd*n_layer of them:
#
#   size    baseline       vpb_baseline/narrow    vpm|vpb|vf / narrow
#   tiny     3,433,024      3,433,024   (+0)       3,433,536  (+512,    +0.015%)
#   small  124,373,760    124,373,760   (+0)     124,392,192  (+18,432, +0.015%)
#   medium 354,599,936    354,599,936   (+0)     354,649,088  (+49,152, +0.014%)
#
# (totals as train.py logs them, i.e. including the positional table; +0 for
# vpb_baseline because its frozen gamma/alpha are buffers, not parameters.)
# So a difference between arms is a difference in the coupling, not in capacity.
# Every run logs n_params / n_params_emb / n_params_blocks to its W&B summary.
#
# Under the paper's non-positional convention (exclude the T x d positional
# table, include the tied V x d token embedding), the baseline geometries are
# 123.6M for small (12L/12H/768d) and 353.6M for medium (24L/16H/1024d).
#
# Muon optimizes the 2D hidden weight matrices (attn/MLP); AdamW handles
# embeddings, lm_head, norms, biases and the reversible scaling vectors.
#
# Geometry, batching and LRs come from the same MODEL_SIZE switch as the real
# job, so the memory and throughput measured here transfer directly. Only the
# step counts are overridden (see TEST OVERRIDES after the switch).
#
# Usage:    sbatch job_reversible_parity_regimes_test.slurm        # default: small
#           sbatch --export=ALL,MODEL_SIZE=tiny --mem=8G --gres=gpu:1 \
#                  job_reversible_parity_regimes_test.slurm
#
# Worth running on the node first (seconds, no queue) -- the gradient-equivalence
# suites, whose CUDA-only checks cannot run on a laptop:
#           python -m test_attn_impl          # asserts a FUSED sdpa backend exists
#
# Then compare within each arch (0/1/2 and 3/4/5): step-0 loss identical, then
# loss curve, peak_mem_gb, tokens_per_sec, recon_drift_max. See the checklist at
# the top.
#
# Outcome -> action:
#   seg=0 tracks eager  ->  use --mem_efficient --mem_efficient_segment 0 in the
#                           real job for O(1)-in-depth stack memory; this is what
#                           makes 'medium' (24L) and deeper studies viable.
#   seg=0 drifts, seg=1 clean -> use segment=1 (bit-exact, smaller win).
#   step-0 loss differs at all -> a bug; do not use either.
# Either way, do not MIX modes within one comparison: train.py logs mem_efficient
# and mem_efficient_segment to the W&B config, so filter on them.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revparityreg_test
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# These three must match the DEFAULT MODEL_SIZE below ('small'), since sbatch reads
# them from the file before the script runs. To submit at another size, pass the
# matching --mem/--gres on the sbatch command line (they override these).
# --time is per TASK, not for the whole array (the six tasks are independent jobs).
# With --attn_impl sdpa the real job measured 3.05-3.47 s/step, so 200 steps is
# ~12 min eager. The mem_efficient arms add one inverse plus one recomputed forward
# per block (~1.6x on the stack) and one drift pass per eval, so budget ~25 min.
# 1h leaves room for startup, data loading and a slower node.
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5                        # {vpb_baseline, vf_scaling} x {eager, seg1, seg0}
#SBATCH --output=output_revparityregtest_%A_%a.txt
#SBATCH --error=output_revparityregtest_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revparityreg_test_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model size switch ─────────────────────────────────────────────────────────
# Edit MODEL_SIZE here, or override at submit time:
#   sbatch --export=ALL,MODEL_SIZE=tiny job_reversible_parity_regimes.slurm
#
# The #SBATCH --mem/--time/--gres above are static (sbatch reads them from the
# file before this script runs), so they are sized for the DEFAULT below. When
# submitting at another size, pass the matching resources on the command line --
# sbatch CLI args override the in-file directives:
#   tiny    --mem=8G  --time=02:00:00 --gres=gpu:1
#   small   --mem=32G --time=24:00:00 --gres=gpu:a100:1     (the default)
#   medium  --mem=48G --time=24:00:00 --gres=gpu:a100:1
#
# --mem is HOST RAM, not GPU, and is dataset-driven rather than model-driven:
# data.load_bin() copies the whole token file into RAM for train and val both
# (~950MB for TinyStories). Batch size barely touches it. For OpenWebText (~17GB
# of bins) raise it to 48G+ regardless of model size.
#
# GPU memory instead scales with BATCH_SIZE (the micro-batch), NOT with
# GRAD_ACCUM_STEPS -- each micro-step's activations are freed after its backward,
# so raising accumulation costs wall-clock, not memory. Measured ~25 GB peak at
# small/BATCH_SIZE=12 with --cheap_metrics (the attention-capture hook
# materializes the full B x n_head x T x T tensor and gives up flash attention;
# drop --cheap_metrics if you need the headroom). train.py logs peak_mem_gb to
# W&B each log interval, so a short probe measures this rather than guessing.
#
# If a reversible arm OOMs where the baseline fits (its residual stream is
# 2*n_embd wide, ~15-20% more block activations), halve BATCH_SIZE and double
# GRAD_ACCUM_STEPS -- the gradient is unchanged.
MODEL_SIZE=${MODEL_SIZE:-small}

case "${MODEL_SIZE}" in
  tiny)     # 3.4M non-positional params -- fast probe, matches job_reversible_parity.slurm
    N_LAYER=4;  N_HEAD=4;  N_EMBD=64;   BLOCK_SIZE=256
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=12          # 18,432 tokens/step
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=100; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  small)    # 123.6M non-positional params -- paper TS 12L recipe, matched exactly
    N_LAYER=12; N_HEAD=12; N_EMBD=768;  BLOCK_SIZE=1024
    # 12x40 = 480 sequences = 491,520 tokens/step, identical to the paper's 30x16:
    # accumulation is loss-scaled (train.py), so the micro-batch/accum split is a
    # pure memory-vs-speed choice and does not change the gradient. Micro-batch 12
    # measures ~25 GB peak; micro-batch 30 would not fit a 40GB card.
    BATCH_SIZE=12; GRAD_ACCUM_STEPS=40          # 491,520 tokens/step
    # 10001 steps x 491,520 = 4.92B tokens = 10.4 TinyStories epochs (~960 steps
    # per epoch). Paper's val loss bottoms out near step 7.6k and overfits after,
    # so report BOTH best and final val -- train.py tracks best_val already.
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=200; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  medium)   # 353.6M non-positional params
    # WARNING: off-paper. The paper trains 24L/16H/1024d on OpenWebText only, at
    # muon_lr 0.006 / adamw 6e-4 / 30k steps / 3k warmup -- never on TinyStories
    # (354M params over 471.6M tokens overfits hard). The values below are a
    # scaled-down guess and are NOT the paper's recipe; fix before relying on them.
    N_LAYER=24; N_HEAD=16; N_EMBD=1024; BLOCK_SIZE=1024
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=20          # 122,880 tokens/step
    MAX_STEPS=20001; WARMUP_STEPS=2000; PEAK_LR=3e-4; MUON_LR=0.02
    EVAL_INTERVAL=250; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  *)
    echo "Unknown MODEL_SIZE='${MODEL_SIZE}' (expected tiny, small or medium)"; exit 1
    ;;
esac

# ── TEST OVERRIDES ────────────────────────────────────────────────────────────
# Applied AFTER the switch so they hold for every size. Only the step counts are
# touched: geometry, BATCH_SIZE, GRAD_ACCUM_STEPS, PEAK_LR and MUON_LR are left
# exactly as the real job has them, so peak_mem_gb and tokens_per_sec measured
# here carry over unchanged.
MAX_STEPS=200
# Warmup scaled to keep the same 10% fraction as the real run, so the test walks
# through warmup AND into cosine decay rather than sitting in a linear ramp the
# whole time. Note this compresses the schedule 50x -- another reason the losses
# here are not comparable to the real run's.
WARMUP_STEPS=20
EVAL_INTERVAL=50             # evals at 50/100/150 (+ the post-loop final eval)
EVAL_BATCHES=20              # cheaper than the real job's 50; still ~245k val tokens
LOG_INTERVAL=10              # 20 log points, so tokens_per_sec/peak_mem_gb settle

MIN_LR_RATIO=0.1
DATASET=tinystories          # existing GPT-2 bins

# vpm_scaling target across-depth log-volume change: total log|det| = -REV_LAMBDA
REV_LAMBDA=0.0

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

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
  # Separate project: 200-step runs must not land in the analysis filter of the
  # real sweep, and they share n_embd/block_size with it so a config filter alone
  # would not exclude them.
  --wandb --wandb_project sympformer-reversible-test
  # sdpa for every arm: this A/B is about the BACKWARD pass, so the attention
  # kernel must be held fixed (it was settled by the previous version of this job).
  --attn_impl sdpa
  # Logs recon_drift_max + per-layer drift at each eval. Costs one extra forward
  # and one full inverse pass over the stack per eval, on a 4-sequence batch.
  --track_recon_drift
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Run dir = OUTDIR/<arch>[_<run_name>]. Keep --array (top) in sync with this.
#
# --rev_lift/--rev_readout only apply to --rev_embed narrow (passing them with
# wide is a hard error, since a wide table needs no bridge). Given explicitly so
# the run record is unambiguous; dup/sum are also the defaults, and are the only
# combo that leaves no dead sublayer:
#   lift=zeros  -> LN(0)=0, so block 1's attention contributes nothing at init
#   readout=x   -> z_final can't reach the logits, so the last MLP gets no grad
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

# Each regime runs three times -- eager / segment=1 / segment=0 -- on the same
# init, seed and batch order, so the A/B is strictly within-arch. --attn_impl sdpa
# and --track_recon_drift are in COMMON_ARGS, so every arm shares them (drift is a
# property of the weights, measurable on the eager arms too).
#
# vpb_baseline and vf_scaling are the two ends of the conditioning question:
# vpb_baseline freezes gamma=alpha=0, i.e. a purely ADDITIVE coupling -- exactly
# Reformer's block, and the case with a global bound on the inverse's Lipschitz
# constant. vf_scaling is AFFINE with the volume change unconstrained, which
# Behrmann et al. (arXiv:2006.09347) show has only local bounds and can develop
# arbitrarily large inverse-Jacobian singular values. If bf16 inversion is going
# to fail anywhere, it fails on vf_scaling first.
#
# Keep --array (top) in sync with this.
NAMES=(
  reversible_vpb_eager
  reversible_vpb_seg1
  reversible_vpb_seg0
  reversible_vf_eager
  reversible_vf_seg1
  reversible_vf_seg0
)
VPB="--rev_regime vpb_baseline ${NARROW}"
VF="--rev_regime vf_scaling ${NARROW}"
TASK_ARGS=(
  "--arch reversible ${VPB} --run_name vpb_eager"
  "--arch reversible ${VPB} --mem_efficient --mem_efficient_segment 1 --run_name vpb_seg1"
  "--arch reversible ${VPB} --mem_efficient --mem_efficient_segment 0 --run_name vpb_seg0"
  "--arch reversible ${VF}  --run_name vf_eager"
  "--arch reversible ${VF}  --mem_efficient --mem_efficient_segment 1 --run_name vf_seg1"
  "--arch reversible ${VF}  --mem_efficient --mem_efficient_segment 0 --run_name vf_seg0"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes_test.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Model size '${MODEL_SIZE}': ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
# A CUDA OOM at model.to(device) means the card was already full BEFORE we
# allocated anything (a 124M model needs only ~0.5GB), i.e. GPU contention, not a
# batch-size problem. Log which device we were given and what is already resident
# so the task output answers that question instead of leaving it to guesswork.
# memory.free here should be within a few hundred MB of memory.total; if it is
# not, another process owns the card and this task will fail regardless of config.
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
# Processes already holding memory on the visible device(s), if any:
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW with a readable message rather than dying on a
# CUDA OOM minutes later. Only checks the first visible device (we request 1).
#   - 'small' needs ~25GB peak with --cheap_metrics; a 16GB P100/V100 cannot run it.
#   - train.py forces bf16 autocast on any CUDA device, which needs Ampere (sm80+):
#     P100 is sm60, V100 is sm70. Neither has native bf16 or flash SDPA.
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
  NEED=24000; [[ "${MODEL_SIZE}" == "tiny" ]] && NEED=4000
  if (( GPU_TOTAL < NEED )); then
    log "FATAL: '${MODEL_SIZE}' needs ~${NEED}MiB but '${GPU_NAME}' has ${GPU_TOTAL}MiB total."
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

log "Task ${ID} (${NAME}) done."
