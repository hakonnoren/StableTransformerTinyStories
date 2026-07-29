#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_parity_regimes.slurm — the full volume-regime sweep, at
# parameter parity with the baseline. Narrow embedding only.
#
# Companion to job_reversible_parity.slurm, which answers "how much of the
# reversible result was just the doubled embedding?" by running {vpb_baseline,
# vf_scaling} x {narrow, wide}. This one takes that as settled, drops the wide
# arm entirely, and instead sweeps all four volume regimes at parity — so the
# only thing varying across arms 1-4 is how gamma/alpha are centered.
#
#   ID  run dir                   regime        centering / volume behaviour
#    0  baseline                  —             vanilla transformer (reference)
#    1  reversible_vpb_baseline   vpb_baseline  gamma=alpha=0, frozen (no scaling)
#    2  reversible_vpm_scaling    vpm_scaling   global centering, total log|det| = -REV_LAMBDA
#    3  reversible_vpb_scaling    vpb_scaling   per-block centering, log|det| = 0 per block
#    4  reversible_vf_scaling     vf_scaling    no centering, volume free
#
# Increasing freedom: vpb_baseline -> vpb_scaling -> vpm_scaling -> vf_scaling.
#
# Parameter parity
# ----------------
# All five arms share one geometry (whatever MODEL_SIZE selects). Attn/MLP
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
# Attention kernel
# ----------------
# All five arms run --attn_impl sdpa (fused/flash, F.scaled_dot_product_attention).
# job_reversible_parity_regimes_test.slurm A/B'd it against the explicit 'manual'
# path at 'small' for both archs, 200 steps, same seed and batch order:
#
#   peak_mem_gb      24.4 -> 13.5 (baseline)      25.3 -> 14.4 (reversible vf)
#   tokens_per_sec   59.9k -> 160.9k (2.69x)      51.3k -> 141.8k (2.77x)
#
# The two kernels compute the same function: train_loss at step 0 agrees to
# 3.8e-05 absolute / 3.5e-06 relative, i.e. bf16 reduction order and nothing else.
# Divergence stays at ~1e-5 through step 10, jumps two orders of magnitude exactly
# at the end of warmup when the LR peaks (Muon's Newton-Schulz amplifies small
# perturbations), then PLATEAUS at ~1e-2 absolute / ~0.4% relative rather than
# growing. On the reversible pair the gap also changes sign repeatedly and shrinks
# back to 6e-05 at one point, so it is trajectory noise, not a kernel bias.
#
# That 0.4% is the same order as some of the regime differences this sweep exists
# to detect, so NEVER mix kernels within one comparison. train.py logs attn_impl
# to the W&B config; filter earlier manual runs out of any analysis with it.
#
# The 'small' size (the default) matches the paper's TinyStories 12L recipe
# exactly: 491,520 tokens/step (12x40x1024, same effective batch as the paper's
# 30x16), 10k steps = 4.92B tokens = 10.4 epochs, warmup 1k, Muon 0.02, AdamW
# 6e-4, min LR 0.1x peak, cosine + linear warmup, betas (0.9, 0.95), Muon
# momentum 0.95, grad clip 1.0, bf16 -- all of which are train.py's defaults
# except the ones set below.
#
# That match buys a free correctness check: arm 0 (baseline) IS the paper's
# GD+Euler / GD+Lie-Trotter, which reach best val 1.107 / 1.106 on this budget.
# If arm 0 lands near 1.11, the data order, optimizer grouping, schedule and eval
# are all validated before anyone reads the reversible arms. If it doesn't,
# something in the pipeline is off and the comparison is not yet trustworthy.
#
# Usage:    sbatch job_reversible_parity_regimes.slurm             # default: small
#           sbatch --export=ALL,MODEL_SIZE=tiny --mem=8G --time=02:00:00 \
#                  --gres=gpu:1 job_reversible_parity_regimes.slurm
#           (see the MODEL_SIZE switch below for the per-size resources)
# Analyze:  python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#               attn_impl=sdpa --format pdf --out_dir plots/revparity_regimes
#           (attn_impl=sdpa excludes earlier manual-kernel runs at this geometry --
#            see the "Attention kernel" note below for why that matters)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=reversible_parity_regimes
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# These three must match the DEFAULT MODEL_SIZE below ('small'), since sbatch reads
# them from the file before the script runs. To submit at another size, pass the
# matching --mem/--time/--gres on the sbatch command line (they override these).
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
# Was 48h when this ran the manual kernel at 7.95 s/step. With --attn_impl sdpa
# the smoke test measured 3.05 s/step (baseline) and 3.47 s/step (reversible vf)
# at 'small', evals included, so 10,001 steps is ~8.5h / ~9.7h -- and those rates
# amortize startup over only 200 steps, so they are pessimistic here. 16h leaves
# ~1.6x headroom over the slowest arm, and a shorter wall queues sooner on GPUQ.
# Check the partition allows it: sinfo -p GPUQ -o "%P %l"
#SBATCH --time=16:00:00
#SBATCH --array=0-4                        # baseline + 4 regimes, all narrow
#SBATCH --output=output_revparityreg_%A_%a.txt
#SBATCH --error=output_revparityreg_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revparityreg_${SLURM_ARRAY_JOB_ID}
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
#   small   --mem=32G --time=16:00:00 --gres=gpu:a100:1     (the default)
#   medium  --mem=48G --time=24:00:00 --gres=gpu:a100:1
#
# 'medium' keeps its 6x20 split: sdpa frees memory there too, but nothing has been
# measured at that geometry, and it is off-paper anyway (see its case below).
#
# --mem is HOST RAM, not GPU, and is dataset-driven rather than model-driven:
# data.load_bin() copies the whole token file into RAM for train and val both
# (~950MB for TinyStories). Batch size barely touches it. For OpenWebText (~17GB
# of bins) raise it to 48G+ regardless of model size.
#
# GPU memory instead scales with BATCH_SIZE (the micro-batch), NOT with
# GRAD_ACCUM_STEPS -- each micro-step's activations are freed after its backward,
# so raising accumulation costs wall-clock, not memory. Measured at 'small' with
# --cheap_metrics and --attn_impl sdpa: 13.5 GB peak (baseline) / 14.4 GB
# (reversible vf) at BATCH_SIZE=12. The manual kernel needed 24.4 / 25.3 GB for
# that same batch, because it materializes three live (B, n_head, T, T) bf16
# tensors per layer (~906 MB each at B=12/T=1024, ~10 GB over 12 layers).
#
# BATCH_SIZE below is 24, spending that freed memory on a larger micro-batch. Even
# under the deliberately pessimistic assumption that ALL of the 14.4 GB scales
# linearly with B, 24 lands at ~28.8 GB, inside a 40 GB card. It is not linear
# (params, optimizer state and the cheap_metrics forward are fixed), so the real
# figure is lower. train.py logs peak_mem_gb every log interval -- read it off the
# first few log lines rather than trusting this estimate.
#
# --cheap_metrics still forces ONE manual forward per eval on the small
# cheap_batch (the fused kernel never builds the attention matrix those
# diagnostics need). That puts a fixed floor under peak_mem_gb which no batch
# change removes; drop --cheap_metrics if you need the headroom.
#
# If a reversible arm OOMs where the baseline fits (its residual stream is
# 2*n_embd wide, ~15-20% more block activations), halve BATCH_SIZE and double
# GRAD_ACCUM_STEPS -- the gradient is unchanged. Back to 12x40 is the first thing
# to try, and it is the exact split the numbers above were measured at.
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
    # 24x20 = 480 sequences = 491,520 tokens/step, identical to the paper's 30x16:
    # accumulation is loss-scaled (train.py), so the micro-batch/accum split is a
    # pure memory-vs-speed choice and does not change the gradient. Raised from
    # 12x40 to spend the memory sdpa freed (see the block above) -- but expect only
    # a few percent from it, NOT another 2x: halving the micro-step count amortizes
    # per-micro-step host overhead and nothing else, since the FLOPs per optimizer
    # step are identical either way. If it measures no faster, revert to 12x40 for
    # the headroom; the gradient does not care.
    BATCH_SIZE=24; GRAD_ACCUM_STEPS=20          # 491,520 tokens/step
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
  # Fused attention. See the "Attention kernel" note in the header for the A/B
  # that justifies this; do not flip it for a subset of the arms.
  --attn_impl sdpa
  --wandb --wandb_project sympformer-reversible
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

NAMES=(
  baseline
  reversible_vpb_baseline
  reversible_vpm_scaling
  reversible_vpb_scaling
  reversible_vf_scaling
)
TASK_ARGS=(
  "--arch baseline --run_name baseline"
  "--arch reversible --rev_regime vpb_baseline ${NARROW} --run_name vpb_baseline_narrow"
  "--arch reversible --rev_regime vpm_scaling  ${NARROW} --run_name vpm_scaling_narrow --rev_lambda ${REV_LAMBDA}"
  "--arch reversible --rev_regime vpb_scaling  ${NARROW} --run_name vpb_scaling_narrow"
  "--arch reversible --rev_regime vf_scaling   ${NARROW} --run_name vf_scaling_narrow"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_parity_regimes.slurm"; exit 1
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

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
