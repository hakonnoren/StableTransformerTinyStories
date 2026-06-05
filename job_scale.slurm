#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_scale.slurm — SCALE SWEEP toward the TinyStories paper. 3 scales x 4 archs
# = 12 array tasks. Tests how the architecture comparison evolves with model size
# (small 128 -> medium 256 -> large 512 hidden, 8 layers, context 512, 10K vocab).
#
#   scale  d_model  heads  ffn   ~params(10K vocab)   paper eval-loss target
#   small  128      2      512   ~3M                  1.65
#   medium 256      4      1024  ~8-9M                1.38
#   large  512      8      2048  ~28-30M              1.20
# (ffn = 4*d_model and head_dim = d_model/heads = 64 are set automatically.)
#
# Archs compared at every scale: baseline | reversible vpb_baseline |
# reversible vf_scaling | YuriiFormer.  (Presymp excluded: it leaks for AR-LM.)
#
# PREREQUISITE — build the 10K-vocab data first (login node, has internet):
#   python preprocess_tinystories_10k.py --vocab_size 10000
#   # note the printed <|endoftext|> id and real vocab size; set EOT/VOCAB below.
# To instead use the existing GPT-2 (50257) bins, set DATASET=tinystories,
# VOCAB=50304, EOT=50256, TOKJSON="" (numbers then NOT comparable to the paper).
#
# Usage:    sbatch job_scale.slurm
# Analyze:  python analyze_wandb.py --filter_config block_size=512 dataset=tinystories10k \
#               --format pdf --out_dir plots/scale
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=tinystories_scale
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=18:00:00                     # sized for the large model; small/medium finish sooner
#SBATCH --array=0-11                        # 12 tasks: scale_idx = ID/4, arch_idx = ID%4
#SBATCH --output=output_scale_%A_%a.txt
#SBATCH --error=output_scale_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=scale_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ---- dataset / tokenizer (see PREREQUISITE above) ----
DATASET=tinystories10k
VOCAB=10000              # set to the real vocab size printed by the preprocessor
EOT=0                    # <|endoftext|> id printed by the preprocessor
TOKJSON=data/tinystories10k_tokenizer.json

# ---- fixed training budget (same tokens across scales -> fair scale comparison) ----
N_LAYER=8
BLOCK_SIZE=512           # context length (paper). NOTE: full causal attention here,
                         # NOT the paper's 256 sliding-window (see README/plan).
BATCH_SIZE=12
GRAD_ACCUM_STEPS=8       # effective batch ~96 seqs x 512 tok ~= 49k tok/step
MAX_STEPS=20000
EVAL_INTERVAL=250
EVAL_BATCHES=50
LOG_INTERVAL=50
WARMUP_STEPS=1000
MIN_LR_RATIO=0.1
PEAK_LR=5e-4

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

# ---- scale table (d_model ; heads = d_model/64 ; ffn = 4*d_model auto in model) ----
SCALE_NAMES=(small medium large)
SCALE_DMODEL=(128 256 512)
SCALE_HEADS=(2 4 8)

# ---- arch table ----
ARCH_NAMES=(baseline vpb_baseline vf_scaling yurii_lt)
ARCH_ARGS=(
  "--arch baseline"
  "--arch reversible --rev_regime vpb_baseline --rev_scale_lr_mult 10"
  "--arch reversible --rev_regime vf_scaling --rev_scale_lr_mult 10"
  "--arch yurii_lt --scalar_lr_mult 100"
)

ID=${SLURM_ARRAY_TASK_ID}
NSC=${#SCALE_NAMES[@]}; NAR=${#ARCH_NAMES[@]}
if [[ -z "${ID}" || "${ID}" -ge $((NSC*NAR)) ]]; then
  echo "Bad SLURM_ARRAY_TASK_ID='${ID}' (expected 0..$((NSC*NAR-1)))"; exit 1
fi
SI=$(( ID / NAR ))           # scale index
AI=$(( ID % NAR ))           # arch index
DMODEL=${SCALE_DMODEL[$SI]}; HEADS=${SCALE_HEADS[$SI]}
SCALE=${SCALE_NAMES[$SI]};   ARCH=${ARCH_NAMES[$AI]}
RUN_NAME=${SCALE}_${ARCH}    # e.g. large_baseline ; run dir = OUTDIR/<arch>_<RUN_NAME>

COMMON_ARGS=(
  --data_dir data --dataset "${DATASET}" --vocab_size "${VOCAB}" --tokenizer_json "${TOKJSON}"
  --out_dir "${OUTDIR}"
  --n_layer "${N_LAYER}" --n_head "${HEADS}" --n_embd "${DMODEL}" --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --wandb --wandb_project sympformer-scale
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 200
  --sample_temperature 0.8 --sample_top_k 40 --sample_eos_token_id "${EOT}"
  --cheap_metrics
  --run_name "${RUN_NAME}"
)

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} scale=${SCALE}(d=${DMODEL},h=${HEADS}) arch=${ARCH} host=$(hostname)"
log "OUTDIR=${OUTDIR} run=${RUN_NAME}"

# shellcheck disable=SC2086  (intentional word-splitting of ARCH_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${ARCH_ARGS[$AI]}

log "Task ${ID} (${RUN_NAME}) done."
