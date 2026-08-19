#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_reversible_regimes_wide.slurm — the full volume-regime sweep with the WIDE
# embedding, i.e. the token/position tables, ln_f and the tied lm_head at the
# full 2*n_embd state width instead of n_embd.
#
# This is job_reversible_parity_regimes.slurm (array 24979051) with --rev_embed
# wide substituted for --rev_embed narrow on arms 1-4. Same geometry, same
# optimizer, same token budget, same seed, so the two arrays are directly
# comparable arm by arm.
#
#   ID  run dir                        regime        embedding
#    0  baseline                       —             vanilla transformer (reference)
#    1  reversible_vpb_baseline_wide   vpb_baseline  gamma=alpha=0, frozen (no scaling)
#    2  reversible_vpm_scaling_wide    vpm_scaling   global centering, total log|det| = -REV_LAMBDA
#    3  reversible_vpb_scaling_wide    vpb_scaling   per-block centering, log|det| = 0 per block
#    4  reversible_vf_scaling_wide     vf_scaling    no centering, volume free
#    5  yuriilt                        —             YuriiFormer (Nesterov + Lie-Trotter)
#
# NO MEMORY-EFFICIENT BACKPROP ANYWHERE IN THIS ARRAY
# ---------------------------------------------------
# No arm passes --mem_efficient (train.py's default is off), so every arm stores
# its activations and differentiates the function it actually computed. This is
# deliberate: the reversible inverse is exact algebraically but not numerically,
# and in bf16 its error grows with depth and with volume freedom -- interior
# relative error 1e-2 to 1e-1 at 12 layers, and up to 6.9 at 48. Whatever
# separates the arms of this array, it is not reconstruction drift.
#
# ⚠ THIS SWEEP IS NOT AT PARAMETER PARITY ⚠
# ------------------------------------------
# That is the entire point of it, but it means the arms are NOT interchangeable
# with the narrow ones when quoting a result. Counted from the models themselves
# at vocab=50304, block_size=1024:
#
#   size    baseline      narrow (vf)   wide (vf)      wide delta
#   small  124,373,760   124,392,192   163,812,864    +39,420,672  (+31.7%, 1.317x base)
#   medium 354,599,936   354,649,088   407,209,984    +52,560,896  (+14.8%, 1.148x base)
#
# All of the delta is embedding/head, none of it the blocks:
#   small : wte 50304x768  = 38,633,472 + wpe 1024x768  = 786,432 + ln_f   768
#   medium: wte 50304x1024 = 51,511,296 + wpe 1024x1024 = 1,048,576 + ln_f 1,024
#
# So a wide arm beating its narrow counterpart is not evidence about the coupling;
# it is 2x embedding capacity, and the comparison to read off this array is
# wide-vs-narrow WITHIN a regime, not wide-vs-baseline. job_reversible_parity.slurm
# already answers this question at 'tiny' (where wide is 1.94x the baseline and the
# confound is overwhelming) for vpb_baseline and vf_scaling only; this array
# extends it to all four regimes at the paper's 'small' geometry, where the
# embedding is a third of the model rather than nearly all of it.
#
# Why the yurii arm belongs in the WIDE array specifically
# -------------------------------------------------------
# yurii_lt carries its own second token+position table pair (the v0 tables,
# YuriiFormer App. A.1), which is exactly the extra capacity a wide embedding
# buys. Counted from the models at this geometry:
#
#   reversible wide (vf)   163,812,864   embedding group 78,839,808
#   yurii_lt               163,802,952   embedding group 78,839,808
#                                        (-9,912 total, -0.006%)
#
# The two embedding groups match to the parameter and the totals differ by 9,912
# out of 163.8M. So THIS array is the parameter-matched comparison against
# YuriiFormer, exactly as the narrow array is the parameter-matched comparison
# against the baseline -- putting yurii_lt in the narrow array would have handed
# it +31.7% over its opponents. Arm 0 is the same non-parity reference in both.
#
# yurii_lt has no narrow/wide switch of its own: --rev_embed feeds RevConfig only,
# and train.py builds YuriiFormerModel without it, so the flag is inert for
# --arch yurii_lt. Its capacity lives in the v0 tables (--no_v0_init drops them).
#
# --scalar_lr_mult on the yurii arm only
# --------------------------------------
# The scalar group (ConstrainedScalar '.raw' parameters) carries yurii_lt's
# learned beta/gamma/eta. YuriiFormer App. A.3 puts that group at 5x the AdamW
# rate where this repo defaults to 10x, so arm 5 passes --scalar_lr_mult 5 to
# match the paper, as job_yurii_ext_tinystories.slurm does. It sits in the arm's
# own flags rather than COMMON_ARGS, but that is cosmetic: the baseline and all
# four reversible models have zero '.raw' parameters (counted, not assumed), so
# the flag is inert for arms 0-4 either way. The reversible gamma/alpha are a
# different group, governed by --rev_scale_lr_mult 10 in COMMON_ARGS.
#
# Expected wall-clock for arm 5: yurii_lt measured 8.1 h at 491,520 tokens/step on
# an A100, well inside the 20 h below. Its state is x,v at n_embd each, i.e. the
# same 2*n_embd the reversible arms carry, so batch 24x20 fits the same card.
#
# Why no --rev_lift / --rev_readout here
# --------------------------------------
# Those two flags exist only to bridge a narrow (n_embd) table to the 2*n_embd
# state -- lift on the way in, readout on the way out. A wide table already spans
# the state, so neither has anything to do. RevConfig rejects a NON-DEFAULT value
# under embed='wide' (lift=zeros or readout=x is a hard error); the defaults
# dup/sum are accepted but are dead flags. They are omitted below because they
# would be no-ops, not because passing them would fail.
#
# Resources vs the narrow array
# -----------------------------
# GPU memory: the wide tables add ~39.4M parameters, all of them AdamW-owned
# (Muon takes the block matrices only), so ~0.63 GB of param+grad+moments on top
# of the narrow run's measured 26.9 GB peak at BATCH_SIZE=24. Activations barely
# move -- the residual stream was already 2*n_embd wide -- so ~27.6 GB, still
# inside a 40 GB card. train.py logs peak_mem_gb; read it off the first log lines.
# If an arm OOMs, halve BATCH_SIZE and double GRAD_ACCUM_STEPS (the gradient is
# unchanged, 12x40 is the fallback).
#
# Wall-clock: the tied head matmul is the one thing that really doubles, from
# B*T*n_embd*vocab to B*T*2*n_embd*vocab, and at 'small' the head is roughly a
# third of the forward FLOPs. The narrow arms took 9.5-11.2 h; budget ~1.3x on
# top, so 20h rather than the narrow array's 16h. Check the partition allows it:
# sinfo -p GPUQ -o "%P %l"
#
# Submit
# ------
#   sbatch job_reversible_regimes_wide.slurm                 # all 5 arms, 'small'
#   sbatch --array=1-5 job_reversible_regimes_wide.slurm     # skip the baseline
#                                                           # (24979051 arm 0 is the
#                                                           #  same run, seed included)
#   sbatch --export=ALL,MODEL_SIZE=tiny --mem=8G --time=02:00:00 --gres=gpu:1 \
#          job_reversible_regimes_wide.slurm                 # fast smoke probe
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revregwide
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# These three are sized for the DEFAULT MODEL_SIZE below ('small'); sbatch reads
# them before this script runs, so override on the command line for other sizes.
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --array=0-5                        # baseline + 4 wide regimes + yurii_lt
#SBATCH --output=output_revregwide_%A_%a.txt
#SBATCH --error=output_revregwide_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revregwide_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model size switch ─────────────────────────────────────────────────────────
# --mem is HOST RAM, dataset-driven: data.load_bin() copies the whole token file
# into RAM for train and val both (~950MB for TinyStories). GPU memory instead
# scales with BATCH_SIZE (the micro-batch), not GRAD_ACCUM_STEPS.
MODEL_SIZE=${MODEL_SIZE:-small}

case "${MODEL_SIZE}" in
  tiny)     # fast probe. NOTE: wide is 1.94x the baseline here -- see the header
    N_LAYER=4;  N_HEAD=4;  N_EMBD=64;   BLOCK_SIZE=256
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=12          # 18,432 tokens/step
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=100; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  small)    # paper TS 12L recipe, identical to job_reversible_parity_regimes.slurm
    N_LAYER=12; N_HEAD=12; N_EMBD=768;  BLOCK_SIZE=1024
    BATCH_SIZE=24; GRAD_ACCUM_STEPS=20          # 491,520 tokens/step
    # 10001 steps x 491,520 = 4.92B tokens = 10.4 TinyStories epochs. Val loss
    # bottoms out near step 7.6k and overfits after, so report BOTH best and
    # final val -- train.py tracks best_val already.
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=200; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  medium)   # WARNING: off-paper on TinyStories (354M params over 471.6M tokens
            # overfits hard). Scaled-down guess, not the paper's recipe.
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
  # Fused attention, as in the narrow array. Do not flip it for a subset of arms.
  --attn_impl sdpa
  --wandb --wandb_project sympformer-reversible-wide
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Run dir = OUTDIR/<arch>[_<run_name>]. Keep --array (top) in sync with this.
# No --rev_lift/--rev_readout: they are narrow-only, see the header.
WIDE="--rev_embed wide"

NAMES=(
  baseline
  reversible_vpb_baseline_wide
  reversible_vpm_scaling_wide
  reversible_vpb_scaling_wide
  reversible_vf_scaling_wide
  yuriilt
)
TASK_ARGS=(
  "--arch baseline --run_name baseline"
  "--arch reversible --rev_regime vpb_baseline ${WIDE} --run_name vpb_baseline_wide"
  "--arch reversible --rev_regime vpm_scaling  ${WIDE} --run_name vpm_scaling_wide --rev_lambda ${REV_LAMBDA}"
  "--arch reversible --rev_regime vpb_scaling  ${WIDE} --run_name vpb_scaling_wide"
  "--arch reversible --rev_regime vf_scaling   ${WIDE} --run_name vf_scaling_wide"
  "--arch yurii_lt --scalar_lr_mult 5 --run_name yuriilt"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_reversible_regimes_wide.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} experiment=${NAME} host=$(hostname)"
log "Model size '${MODEL_SIZE}': ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d block=${BLOCK_SIZE} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS}"
log "OUTDIR=${OUTDIR} (wide embedding: NOT parameter-matched to the baseline)"

# ── GPU sanity check ──────────────────────────────────────────────────────────
# A CUDA OOM at model.to(device) means the card was already full BEFORE we
# allocated anything, i.e. GPU contention, not a batch-size problem.
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
