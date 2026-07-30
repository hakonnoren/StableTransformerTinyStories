#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes_revback.slurm — the full volume-regime sweep at
# parameter parity, run through MEMORY-EFFICIENT REVERSIBLE BACKPROP
# (--mem_efficient --mem_efficient_segment n_layer, "segL") instead of ordinary
# autograd. The four reversible arms of job_reversible_parity_regimes.slurm, same
# recipe, reconstructing their activations in backward instead of storing them. The
# baseline arm is omitted -- it cannot use segL and the reference already exists.
#
#   ID  run dir                   regime        centering / volume behaviour
#    0  reversible_vpb_baseline   vpb_baseline  gamma=alpha=0, frozen (no scaling)
#    1  reversible_vpm_scaling    vpm_scaling   global centering, total log|det| = -REV_LAMBDA
#    2  reversible_vpb_scaling    vpb_scaling   per-block centering, log|det| = 0 per block
#    3  reversible_vf_scaling     vf_scaling    no centering, volume free
#
# Increasing freedom: vpb_baseline -> vpb_scaling -> vpm_scaling -> vf_scaling.
# Every arm runs segL. There is no baseline arm: a vanilla transformer cannot use
# --mem_efficient (it is a hard error on --arch baseline, and there is nothing to
# invert), and the reference already exists -- see "The baseline reference" below.
#
# ⚠ PREREQUISITE -- DO NOT SUBMIT THIS BLIND ⚠
# ----------------------------------------------
# segL is not yet validated at a long horizon. What IS established (200-step A/B,
# array 24979558, plus offline gradient measurements):
#   * segment=0 is DISQUALIFIED in bf16. It is the only setting that reconstructs
#     the stack input y0, and that inverse cancels catastrophically against the
#     small-magnitude embedding: 7.6e-1 relative gradient error on block 0, 4.3e-1
#     overall. On vf_scaling it produced a monotone +3.18% train-loss bias and a
#     best_val gap of 0.058, i.e. 5x the run-to-run noise.
#   * segment=n_layer (segL) checkpoints at i % segment == 0, hence at i=0 only, so
#     it stores exactly ONE (B,T,2*n_embd) tensor -- O(1) in depth, like segment=0 --
#     and never performs that bad inverse. Total bf16 gradient error drops 77x,
#     4.3e-1 -> 5.6e-3, which is the interior chain's own error.
#   * segment=1 is bit-exact on CPU but NOT on GPU, and its memory grows with depth.
# What is NOT established is whether segL's residual 5.6e-3 stays harmless over
# thousands of steps. The failure mode is cumulative -- vf_seg0's bias was invisible
# for ~100 steps and only obvious in the last third -- so run
# job_reversible_parity_regimes_test.slurm (4000 steps, eager vs segL) FIRST and
# confirm segL tracks its eager reference for BOTH vpb_baseline and vf_scaling.
# Submitting this before that is betting ~4 x 20 GPU-hours on an unvalidated setting.
#
# Why bother, i.e. what segL buys
# -------------------------------
# Not speed -- it costs ~1.46x wall-clock (one inverse plus one recomputed forward
# per block on top of the gradient). It buys MEMORY, and specifically memory that is
# FLAT IN DEPTH. Measured at 'small', BATCH_SIZE=12, --cheap_metrics --attn_impl sdpa:
#
#   eager reversible   14.39 GB        segment=1   10.21 GB        segL  ~9.34 GB
#
# The saving is modest at 12 layers because after sdpa the stack is only ~3.9 GB of
# that peak (the rest is the loss head at B*T*V logits plus params/optimizer, neither
# of which this touches). The point is that the stack term no longer grows with
# n_layer at all, which is what makes 'medium' (24L) and deeper studies viable. At
# 'small' the freed memory is simply left as headroom -- see "Batch size" for why it
# is not spent on a larger micro-batch.
#
# Batch size: 24x20, and why not larger now that the baseline is gone
# -------------------------------------------------------------------
# Accumulation is loss-scaled in train.py, so BATCH_SIZE x GRAD_ACCUM_STEPS is the
# only thing the gradient sees -- the split between them is a pure memory-vs-speed
# choice. Every arm keeps 480 sequences = 491,520 tokens/step (the paper's TinyStories
# budget, its own 30x16) at a 24x20 split: ~18.7 GB pessimistic, ~16 GB real.
#
# "Pessimistic" = assume ALL of the measured B=12 peak (9.34 GB for segL) scales
# linearly in B, which it does not (params, optimizer state and the fixed
# --cheap_metrics forward do not).
#
# segL at 24x20 uses well under half a 40 GB card, so 40x12 (~31.1 GB pessimistic)
# would now fit -- the baseline was what capped this at 24, and it is gone. It stays
# at 24x20 anyway:
#
# 1. DATA-ORDER ALIGNMENT WITH THE EXISTING BASELINE. This is the reason that
#    survives dropping the arm. data.py's BlockEpochIterator is a deterministic
#    block-epoch protocol: per epoch it draws one seeded offset and one shuffle of the
#    block-start list, and NEITHER call depends on batch_size -- so the block list is
#    identical for any split. Within one epoch, any split consumes the same 480 blocks
#    per optimizer step. At the epoch boundary they diverge: the reset discards
#    whatever tail does not fill a micro-batch (464,000 blocks % 40 = 0 vs % 24 = 8)
#    and the boundary lands at a different point inside an optimizer step. Simulated
#    over 2000 steps at 'small', only the first 966 -- exactly one epoch -- have
#    byte-identical 480-block sets; everything after drifts. Over 10001 steps
#    (10.4 epochs) that is ~90% of the run. The baseline this sweep is measured
#    against ran at 24x20, so matching it keeps step-to-step curves comparable
#    instead of merely statistically equivalent.
# 2. It buys almost nothing. 20 -> 12 micro-steps amortizes per-micro-step host
#    overhead and nothing else; the FLOPs per optimizer step are identical.
# 3. Every memory figure quoted here was measured at 24x20, and it is the main
#    sweep's split, so it is the known-good configuration.
#
# The lower peak is also a mild hedge against the unresolved OOM below: two segL arms
# co-scheduled on one 40 GB card need ~32 GB at 24x20 (real) and might both survive,
# where 2 x 40x12 could not. Do not mistake that for a fix.
#
# train.py logs peak_mem_gb every log interval -- read it off the first few log lines
# rather than trusting these estimates. If you do want the larger micro-batch, change
# BATCH_SIZE/GRAD_ACCUM_STEPS in the size switch; the SEQ_PER_STEP guard checks the
# product for you.
#
# Post-mortem: array 24994972 (UNRESOLVED -- read this before rerunning)
# ----------------------------------------------------------------------
# NOTE: arm numbers in this section are the PREVIOUS submission's, which had the
# baseline at ID 0. In the current table vpb_baseline is ID 0, so old arms 1-4 are
# new arms 0-3.
# Arms 2, 3 and 4 died with `torch.AcceleratorError: CUDA error: out of memory` at
# train.py's `model.to(device)`, ~9s after launch. This is NOT a batch-size problem
# and lowering BATCH_SIZE is not known to fix it. The evidence:
#   * model.to(device) allocates the PARAMETERS -- ~0.5 GB for a 124M model. No
#     activation, no batch, no optimizer state exists yet, so nothing at that point
#     is sensitive to BATCH_SIZE.
#   * arm 1 (vpb_baseline) SURVIVED on the identical 40x12 split, and is
#     memory-indistinguishable from arms 2-4 (its frozen gamma/alpha are buffers, an
#     18,432-element difference). Same config, different outcome => environment.
#   * it is a raw driver-level cudaErrorMemoryAllocation, not PyTorch's
#     `torch.cuda.OutOfMemoryError`, which would have printed "Tried to allocate X
#     MiB (GPU 0; Y GiB total capacity; Z GiB already allocated)". PyTorch's caching
#     allocator never got to report, i.e. the driver refused the very first request.
#   * AND YET the preflight passed on a nearly empty card. Task 2 logged
#     "40442MiB free of 40960MiB", no compute-apps, host idun-07-06,
#     SLURM_JOB_GPUS=7, CUDA_VISIBLE_DEVICES=0 -- then OOM'd seconds later.
# So the simple "another job owned the card" story does NOT fit the preflight, and the
# cause is still open. Two hypotheses remain, and they are distinguishable:
#   (a) Two array tasks were placed on the SAME physical GPU. The preflight would then
#       pass for both (both see it free at the same instant) and whichever process
#       allocated its activations first would leave the other unable to get even
#       0.5 GB. DIAGNOSTIC: compare `host` and SLURM_JOB_GPUS across all task
#       logs. Any repeated (host, SLURM_JOB_GPUS) pair confirms it. FIX: throttle with
#       `sbatch --array=0-3%1`, or add #SBATCH --exclusive.
#   (b) Context creation failed for a non-memory reason the driver reports as OOM --
#       compute mode Exclusive_Process with another context already open, or a wedged
#       / ECC-degraded device. DIAGNOSTIC: the failure-time snapshot that run() now
#       prints, including compute_mode. FIX: none from here; report the node.
# Note SLURM's cgroup can hide other jobs' PIDs from --query-compute-apps while
# memory.used still reflects them, so a clean app list is not proof of an idle card --
# but memory.free coming back at 40442 MiB genuinely does mean it was idle at 22:17:33.
#
# Also fixed here: run() now checks train.py's exit status and the script propagates
# it. On array 24994972 a crashed arm still logged "Task 2 done" and exited 0, so
# sacct reported COMPLETED and the failure looked like a success. The same defect
# exists in job_reversible_parity_regimes.slurm and job_reversible_parity.slurm.
#
# Under the current 0-3 numbering the three that died are arms 1, 2 and 3, i.e.
# `sbatch --array=1-3` reruns exactly the failures.
#
# The baseline reference (why there is no baseline arm here)
# ----------------------------------------------------------
# It already exists, twice, at exactly this recipe -- sdpa, 10001 steps, 24x20,
# 12L/768d:
#
#   project                          best_val   note
#   sympformer-reversible            1.0522     the main sweep's arm 0
#   sympformer-reversible-revback    1.0528     arm 0 of array 24994972; it survived
#                                               the OOM that killed arms 2-4
#
# So the reference is already IN this project and no rerun is needed. Two further
# points worth having:
#
#   * Those two runs are an unintended REPLICATE: identical config and seed, differing
#     only by GPU nondeterminism, and their best_val agrees to 0.0006. That is the
#     long-horizon run-to-run floor, and it is the yardstick to judge regime
#     differences against -- far tighter than the 0.011 measured from a 200-step pair,
#     because best_val near convergence is a much more stable quantity than train_loss
#     at step 190. For scale, the regime spread in the earlier 20001-step manual-kernel
#     sweep was ~0.019 (baseline 1.1405 / vf 1.1534 / vpb_baseline 1.1591), i.e. about
#     30x the floor. Differences below ~0.001 are noise.
#   * It is a PIPELINE check, not a segL check. The baseline is eager, so it says
#     nothing about the reversible backprop. For a like-for-like segL control use the
#     eager arms of the 4000-step test job, which hold every other variable fixed.
#
# Do NOT compare against the baseline in results/revparityreg_24957997: that run is
# attn_impl=manual, max_steps=20001 and truncated at step 18500. Wrong kernel, wrong
# budget, incomplete -- see the attention-kernel note below for why the kernel alone
# disqualifies it.
#
# Parameter parity
# ----------------
# All four arms share one geometry (whatever MODEL_SIZE selects). Attn/MLP already
# match per block; --rev_embed narrow additionally puts the token/pos tables, ln_f
# and the tied lm_head at n_embd (the baseline's exact shapes) instead of the full
# 2*n_embd state width, bridging to the state parameter-free (lift dup: x = z = emb;
# readout sum: x + z). The residual is only the gamma/alpha vectors,
# 2*n_embd*n_layer of them:
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
# Under the paper's non-positional convention (exclude the T x d positional table,
# include the tied V x d token embedding), the baseline geometries are 123.6M for
# small (12L/12H/768d) and 353.6M for medium (24L/16H/1024d).
#
# Muon optimizes the 2D hidden weight matrices (attn/MLP); AdamW handles embeddings,
# lm_head, norms, biases and the reversible scaling vectors.
#
# All arms run --attn_impl sdpa (fused/flash). The kernel was A/B'd against the
# explicit 'manual' path and the two agree to trajectory noise (~0.4% relative,
# sign-changing), but that 0.4% is the same order as some regime differences this
# sweep exists to detect, so NEVER mix kernels within one comparison. train.py logs
# attn_impl to the W&B config; filter on it.
#
# Reconstruction drift
# --------------------
# --track_recon_drift is on for every arm (the baseline ignores it with a printed
# notice -- it needs an invertible stack). Over 10001 steps this is the mechanism
# data the short runs could not provide: whether the inverse map's conditioning
# degrades as gamma/alpha learn. Error amplification through the inverse chain goes
# like exp(sum_l gamma_l), and vf_scaling constrains that sum not at all, so watch
# arm 4 in particular. Two keys, and they are DIFFERENT quantities:
#   recon_drift_max    max over the interior layers l1..l_n. This is the one that
#                      corresponds to what segL's backward actually does.
#   recon_drift_input  l0, the stack input. Expect it to be LARGE (order 1-10) on
#                      every arm including the eager ones. The probe always inverts
#                      the whole stack regardless of backprop mode, so it keeps
#                      measuring the y0 cancellation even though segL never
#                      reconstructs y0 in training. That is the probe characterizing
#                      the inverse map, NOT a fault and NOT a regression.
# Cost is one extra forward plus one full inverse pass per eval on a 4-sequence
# batch -- negligible against a 50-batch eval.
#
# Usage:    sbatch job_reversible_parity_regimes_revback.slurm       # default: small
#           sbatch --export=ALL,MODEL_SIZE=tiny --mem=8G --time=02:00:00 \
#                  --gres=gpu:1 job_reversible_parity_regimes_revback.slurm
#           (see the MODEL_SIZE switch below for the per-size resources)
#           sbatch --array=0,3 job_reversible_parity_regimes_revback.slurm
#                  # just the two ends of the freedom ordering (vpb_baseline, vf_scaling)
#           sbatch --array=0-3%1 job_reversible_parity_regimes_revback.slurm
#                  # one arm at a time -- use this if the OOM below turns out to be
#                  # two array tasks sharing a GPU
#
# Resume:   train.py takes --resume <path>. Each arm writes best_<arch>.pt and
#           final_<arch>.pt into its own run dir, and best_val is tracked, so a
#           wall-clock timeout past ~step 8k costs little: the paper's val loss
#           bottoms out near step 7.6k and overfits after.
#
# Analyze:  python analyze_wandb.py --project sympformer-reversible-revback \
#               --filter_config n_embd=768 block_size=1024 attn_impl=sdpa \
#               --format pdf --out_dir plots/revparity_regimes_revback
#           The already-finished baseline_baseline (best_val 1.0528) lives in this
#           same project and this filter picks it up, so the reference plots itself
#           alongside the four regimes with no extra work.
#
# Separate W&B project, deliberately: these runs share n_embd/block_size/attn_impl
# with the main sweep, so a config filter alone would not separate them, and mixing
# eager and segL trajectories inside one regime comparison is exactly the mistake to
# avoid. mem_efficient / mem_efficient_segment are logged to the config if you do
# want to pool them later.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revparityreg_revback
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# These three must match the DEFAULT MODEL_SIZE below ('small'), since sbatch reads
# them from the file before the script runs. To submit at another size, pass the
# matching --mem/--time/--gres on the sbatch command line (they override these).
#SBATCH --gres=gpu:a100:1
# --mem is HOST RAM, not GPU. It is dataset-driven (data.load_bin copies the whole
# token file into RAM for train and val both, ~950MB each for TinyStories) and does
# NOT scale with batch size or step count, so 32G is sufficient here exactly as in
# the main sweep. The memory this job actually spends is GPU memory, via BATCH_SIZE.
# Raise to 48G+ only for OpenWebText (~17GB of bins).
#SBATCH --mem=32G
# --time is per TASK, not for the whole array (the four tasks are independent jobs).
# Budget for MAX_STEPS=10001 at 'small', from the first run's MEASURED tokens_per_sec
# (491,520 tokens/step):
# Every arm is segL, so every arm is on the slow path: ~5.74-6.20 s/step measured,
# i.e. 16.0-17.2 h for 10001 steps. Plus 50 evals (50 batches + drift +
# cheap_metrics + sampling, ~40s each) = 0.6 h, plus ~0.3 h startup and data loading
# => ~18.1 h. A slow node costs up to another 22% (the eager replicate pair differed
# that much on identical config), which would put it at ~21.6 h -- hence 24h rather
# than the main sweep's 16h. For reference the eager baseline ran 3.05-3.47 s/step
# (8.5-9.7 h); segL trades that ~1.46x for memory flat in depth.
# Check the partition allows it: sinfo -p GPUQ -o "%P %l"
#SBATCH --time=24:00:00
#SBATCH --array=0-3                        # 4 volume regimes, all narrow, all segL
#SBATCH --output=output_revparityregrevback_%A_%a.txt
#SBATCH --error=output_revparityregrevback_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revparityregrevback_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model size switch ─────────────────────────────────────────────────────────
# Edit MODEL_SIZE here, or override at submit time:
#   sbatch --export=ALL,MODEL_SIZE=tiny job_reversible_parity_regimes_revback.slurm
#
# The #SBATCH --mem/--time/--gres above are static (sbatch reads them from the file
# before this script runs), so they are sized for the DEFAULT below. When submitting
# at another size, pass the matching resources on the command line -- sbatch CLI args
# override the in-file directives:
#   tiny    --mem=8G  --time=02:00:00 --gres=gpu:1
#   small   --mem=32G --time=24:00:00 --gres=gpu:a100:1     (the default)
#   medium  --mem=48G --time=36:00:00 --gres=gpu:a100:1
#
# 'medium' (24L) is where segL actually matters -- the stack term is flat in depth,
# so it should permit a much larger micro-batch than the 6x20 inherited here. Nothing
# has been measured at that geometry, so the split is left alone and the time budget
# is a guess; probe it with a short run before trusting either.
#
# GPU memory scales with BATCH_SIZE (the micro-batch), NOT with GRAD_ACCUM_STEPS --
# each micro-step's activations are freed after its backward, so raising accumulation
# costs wall-clock, not memory. See the header for the per-arm split and its
# measured basis.
#
# If an arm OOMs, halve its BATCH_SIZE and double its GRAD_ACCUM_STEPS in the arrays
# below -- the gradient is unchanged, and the SEQ_PER_STEP guard will verify it.
MODEL_SIZE=${MODEL_SIZE:-small}

case "${MODEL_SIZE}" in
  tiny)     # 3.4M non-positional params -- fast probe
    N_LAYER=4;  N_HEAD=4;  N_EMBD=64;   BLOCK_SIZE=256
    SEQ_PER_STEP=72                             # 18,432 tokens/step
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=12          # too small for memory to bind
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=100; EVAL_BATCHES=50; LOG_INTERVAL=50
    GPU_NEED_MB=4000
    ;;
  small)    # 123.6M non-positional params -- paper TS 12L recipe, matched exactly
    N_LAYER=12; N_HEAD=12; N_EMBD=768;  BLOCK_SIZE=1024
    SEQ_PER_STEP=480                            # 491,520 tokens/step
    # 24x20 matches the existing baseline's split, so step-to-step curves stay
    # comparable to it -- see "Batch size" in the header for the epoch-boundary
    # drift that motivates it. segL leaves plenty of room; this is not a cap.
    BATCH_SIZE=24; GRAD_ACCUM_STEPS=20          # ~18.7 GB pessimistic, ~16 GB real
    # 10001 steps x 491,520 = 4.92B tokens = 10.4 TinyStories epochs (~960 steps per
    # epoch). Paper's val loss bottoms out near step 7.6k and overfits after, so
    # report BOTH best and final val -- train.py tracks best_val already.
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=200; EVAL_BATCHES=50; LOG_INTERVAL=50
    # segL at 24x20 needs ~18.7 GB pessimistic. Set above that but well below a clean
    # card's ~40.5 GB free, so the check still fires on a card another process owns.
    # Note it CANNOT catch two of our own tasks sharing a GPU -- both would see it
    # free at the same instant. See the post-mortem in the header.
    GPU_NEED_MB=24000
    ;;
  medium)   # 353.6M non-positional params
    # WARNING: off-paper. The paper trains 24L/16H/1024d on OpenWebText only, at
    # muon_lr 0.006 / adamw 6e-4 / 30k steps / 3k warmup -- never on TinyStories
    # (354M params over 471.6M tokens overfits hard). The values below are a
    # scaled-down guess and are NOT the paper's recipe; fix before relying on them.
    N_LAYER=24; N_HEAD=16; N_EMBD=1024; BLOCK_SIZE=1024
    SEQ_PER_STEP=120                            # 122,880 tokens/step
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=20          # unmeasured at 24L; segL allows more
    MAX_STEPS=20001; WARMUP_STEPS=2000; PEAK_LR=3e-4; MUON_LR=0.02
    EVAL_INTERVAL=250; EVAL_BATCHES=50; LOG_INTERVAL=50
    GPU_NEED_MB=30000
    ;;
  *)
    echo "Unknown MODEL_SIZE='${MODEL_SIZE}' (expected tiny, small or medium)"; exit 1
    ;;
esac

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
    # A CUDA OOM at model.to(device) means the card filled between the preflight
    # above and a ~0.5GB parameter allocation. Snapshot the device NOW, while the
    # failure is fresh: this is the evidence that separates "another process took
    # the card after our preflight" from "this device was never exclusively ours".
    # Note SLURM's cgroup can hide other jobs' PIDs from --query-compute-apps while
    # memory.used still reflects them, so read the totals, not just the app list.
    log "GPU state at failure (host=$(hostname), SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true
    nvidia-smi --query-gpu=index,compute_mode,persistence_mode --format=csv 2>&1 || true
  fi
  return $rc
}

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Run dir = OUTDIR/<arch>[_<run_name>]. Keep --array (top) in sync with this.
#
# --rev_lift/--rev_readout only apply to --rev_embed narrow (passing them with wide
# is a hard error, since a wide table needs no bridge). Given explicitly so the run
# record is unambiguous; dup/sum are also the defaults, and are the only combo that
# leaves no dead sublayer:
#   lift=zeros  -> LN(0)=0, so block 1's attention contributes nothing at init
#   readout=x   -> z_final can't reach the logits, so the last MLP gets no grad
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

# segL = --mem_efficient_segment ${N_LAYER}, i.e. the layer count of whatever
# MODEL_SIZE selected -- NOT a hardcoded 12. rev_backprop checkpoints at
# i % segment == 0, so segment=n_layer checkpoints at i=0 alone: exactly one stored
# tensor (the stack input y0), O(1) in depth like segment=0, but without the one
# catastrophically-cancelling inverse that disqualifies segment=0 in bf16.
SEGL="--mem_efficient --mem_efficient_segment ${N_LAYER}"

NAMES=(
  reversible_vpb_baseline
  reversible_vpm_scaling
  reversible_vpb_scaling
  reversible_vf_scaling
)
TASK_ARGS=(
  "--arch reversible --rev_regime vpb_baseline ${NARROW} ${SEGL} --run_name vpb_baseline_narrow"
  "--arch reversible --rev_regime vpm_scaling  ${NARROW} ${SEGL} --run_name vpm_scaling_narrow --rev_lambda ${REV_LAMBDA}"
  "--arch reversible --rev_regime vpb_scaling  ${NARROW} ${SEGL} --run_name vpb_scaling_narrow"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} ${SEGL} --run_name vf_scaling_narrow"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes_revback.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

# Hard guard: an arm that takes a different number of sequences per optimizer step is
# not comparable to the others, and the mistake is silent -- the loss curve just
# looks a bit off. Fail loudly at submit time instead.
if (( BATCH_SIZE * GRAD_ACCUM_STEPS != SEQ_PER_STEP )); then
  log "FATAL: arm ${ID} (${NAME}) has BATCH_SIZE x GRAD_ACCUM_STEPS ="
  log "       ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} = $(( BATCH_SIZE * GRAD_ACCUM_STEPS )),"
  log "       but every arm must take SEQ_PER_STEP=${SEQ_PER_STEP} sequences per step."
  log "       Fix BATCH_*/ACCUM_* for MODEL_SIZE=${MODEL_SIZE} so the product matches."
  exit 1
fi

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
  # Fused attention. See the header for the A/B that justifies this; do not flip it
  # for a subset of the arms.
  --attn_impl sdpa
  # Separate project from the main sweep -- see the header for why.
  --wandb --wandb_project sympformer-reversible-revback
  # Logs recon_drift_max (interior l1..l_n) + recon_drift_input (l0) + per-layer
  # drift at each eval. Every arm here is reversible, so every arm logs it.
  --track_recon_drift
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Model size '${MODEL_SIZE}': ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "Tokens/step: $(( BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE )) (${SEQ_PER_STEP} sequences)"
log "Backprop: mem_efficient reversible, segment=${N_LAYER} (segL)"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
# A CUDA OOM at model.to(device) means the card was already full BEFORE we allocated
# anything (a 124M model needs only ~0.5GB), i.e. GPU contention, not a batch-size
# problem. Log which device we were given and what is already resident so the task
# output answers that question instead of leaving it to guesswork. memory.free here
# should be within a few hundred MB of memory.total; if it is not, another process
# owns the card and this task will fail regardless of config.
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
# Processes already holding memory on the visible device(s), if any:
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# Hard preflight: bail out NOW with a readable message rather than dying on a CUDA
# OOM hours later. This job runs for ~18h, so a late OOM is expensive. Only checks
# the first visible device (we request 1).
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
  if (( GPU_TOTAL < GPU_NEED_MB )); then
    log "FATAL: '${MODEL_SIZE}' at batch ${BATCH_SIZE} needs ~${GPU_NEED_MB}MiB but"
    log "       '${GPU_NAME}' has ${GPU_TOTAL}MiB total. Either request a bigger card or"
    log "       halve BATCH_SIZE and double GRAD_ACCUM_STEPS (gradient unchanged)."
    exit 1
  fi
  if (( GPU_FREE < GPU_NEED_MB )); then
    log "FATAL: only ${GPU_FREE}MiB free of ${GPU_TOTAL}MiB on '${GPU_NAME}' -- another"
    log "       process holds the card. See the compute-apps list above."
    exit 1
  fi
  log "GPU preflight OK: ${GPU_NAME}, ${GPU_FREE}MiB free of ${GPU_TOTAL}MiB (need ~${GPU_NEED_MB}MiB)"
else
  log "WARNING: could not query GPU memory; skipping preflight"
fi

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}
RC=$?

# Propagate the failure. Without this the script exits 0 and logs "done" even when
# train.py died in the first ten seconds, so sacct reports COMPLETED and a broken arm
# reads as a successful one -- which is exactly what happened on array 24994972.
if (( RC != 0 )); then
  log "Task ${ID} (${NAME}) FAILED after $(( SECONDS ))s with exit ${RC}."
  exit "${RC}"
fi

log "Task ${ID} (${NAME}) done in $(( SECONDS ))s."
