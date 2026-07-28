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
# embeddings, lm_head, norms, biases and the reversible scaling vectors. Recipe
# from the TinyStories 12L table: muon_lr 0.02, AdamW peak 6e-4, warmup 1000 --
# which is exactly the 'small' size, the default here.
#
# Usage:    sbatch job_reversible_parity_regimes.slurm             # default: small
#           sbatch --export=ALL,MODEL_SIZE=tiny --mem=8G --time=02:00:00 \
#                  --gres=gpu:1 job_reversible_parity_regimes.slurm
#           (see the MODEL_SIZE switch below for the per-size resources)
# Analyze:  python analyze_wandb.py --filter_config n_embd=768 block_size=1024 \
#               --format pdf --out_dir plots/revparity_regimes
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
#SBATCH --time=12:00:00
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
#   small   --mem=32G --time=12:00:00 --gres=gpu:a100:1     (the default)
#   medium  --mem=48G --time=24:00:00 --gres=gpu:a100:1
#
# batch_size x grad_accum_steps x block_size = tokens/step is held at ~123k for
# small and medium so the two are comparable. If a reversible arm OOMs where the
# baseline fits (the residual stream is 2*n_embd wide), halve BATCH_SIZE and
# double GRAD_ACCUM_STEPS -- tokens/step is unchanged.
#
# --mem is HOST RAM, not GPU: data.load_bin() copies the whole token file into
# RAM for train and val both (~950MB for TinyStories). For OpenWebText (~17GB of
# bins) raise it to 48G+ regardless of model size.
MODEL_SIZE=${MODEL_SIZE:-small}

case "${MODEL_SIZE}" in
  tiny)     # 3.4M non-positional params -- fast probe, matches job_reversible_parity.slurm
    N_LAYER=4;  N_HEAD=4;  N_EMBD=64;   BLOCK_SIZE=256
    BATCH_SIZE=6;  GRAD_ACCUM_STEPS=12          # 18,432 tokens/step
    MAX_STEPS=10001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=100; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  small)    # 123.6M non-positional params -- the TinyStories 12L recipe
    N_LAYER=12; N_HEAD=12; N_EMBD=768;  BLOCK_SIZE=1024
    BATCH_SIZE=12; GRAD_ACCUM_STEPS=10          # 122,880 tokens/step
    MAX_STEPS=20001; WARMUP_STEPS=1000; PEAK_LR=6e-4; MUON_LR=0.02
    EVAL_INTERVAL=250; EVAL_BATCHES=50; LOG_INTERVAL=50
    ;;
  medium)   # 353.6M non-positional params -- LR/warmup scaled down but UNTUNED
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

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
