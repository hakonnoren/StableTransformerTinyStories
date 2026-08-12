#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_owt_medium.slurm — OpenWebText at the paper's 'medium' scale
# (24L/16H/1024d, 353.6M non-positional params), 30,000 steps, single GPU per
# arm. Reproduces the OWT column of YuriiFormer (arXiv:2601.23236, Zimin,
# Polyanskiy & Rigollet) Table 1 / Table 3, and adds this repo's reversible
# regimes as extra arms on the same corpus.
#
# PREREQUISITE: sbatch job_owt_preprocess.slurm first. data/ has no OpenWebText
# bins yet; train.py exits at the "Missing dataset bins" check without them.
#
# ── Hyperparameters: paper Table 1, OWT 24L column ───────────────────────────
#   24L / 16H / 1024d, T=1024, vocab padded to 50304   (paper A.2; train.py default)
#   30,000 steps, 3,000 warmup, cosine to 0.1x peak
#   Muon LR 0.006  <-- NOT 0.02. The 0.02 in every existing job here is the
#                      TinyStories value; paper Table 1 gives 0.004 for OWT 12L
#                      and 0.006 for OWT 24L. This is the single most important
#                      difference from the revL*.slurm recipes.
#   AdamW LR 6e-4, betas (0.9,0.95), Muon momentum 0.95, grad clip 1.0, bf16
#   dropout 0, no bias                                 (train.py defaults)
#
# ── Token budget: 491,520 tokens/optimizer step ──────────────────────────────
# Paper: "global batch size of 30 sequences with 16-step gradient accumulation
# at context length T=1024 (491,520 tokens per optimizer step)". 30 x 16 x 1024
# = 491,520, and 30,000 steps x 491,520 = 14.75B tokens = ~1.7 epochs over the
# 8.62B-token train split. The paper's 17.5k steps/epoch checks out exactly
# against our bins: 8.62e9/1024 = 8.42M blocks / 480 blocks-per-step = 17.5k.
#
# The reversible+segL arms use 60 x 8 instead of 30 x 16 -- same 491,520 tokens,
# fewer sequential accumulation passes, which is the whole point of the segL
# memory saving (same trade revL48.slurm makes). This does NOT perturb data
# order: BlockEpochIterator (data.py) seeds its shuffle independently of
# batch_size and __next__ slices _starts consecutively, so 30x16 and 60x8 consume
# the SAME 480 blocks in the SAME order every optimizer step. The only divergence
# is the <=59 leftover blocks discarded at the single epoch boundary near step
# 17.5k; epoch 2's shuffle is identical again (same number of rng calls).
#
# ── Eval: matched to the paper AND matched across arms ───────────────────────
# Paper: val loss every 100 steps, averaged over 160 batches (4,915,200 tokens).
# estimate_loss (train.py:438) draws from val_it, which uses args.batch_size --
# so eval_batches must be halved on the 60-microbatch arms to average over the
# same 4,915,200 val tokens. Hence EVAL_BATCHES 160 vs 80 below. Getting this
# wrong would make the arms' val curves noisier/smoother than each other rather
# than measuring the same thing.
#
# ── GPU: 1x H100 80GB ────────────────────────────────────────────────────────
# The paper used 2x H200 and explicitly says it did not optimize throughput.
#
# NOTE: train.py now DOES support multi-GPU (see ddp_utils.py). For the same run
# on all 8 GPUs of one node -- same tokens/step, same data order, ~8x less wall
# time -- use job_owt_medium_ddp.slurm instead. This single-GPU job is kept as
# the reference/fallback: it is the simpler configuration, it needs no
# torchrun rendezvous, and it holds one copy of the dataset in RAM instead of
# eight. Both produce the same experiment; pick on queue availability.
#
# Confirmed gres types: a100, h100, h200, p100, v100.
# H100 80GB 8/node: idun-01-[01-06], idun-08-01
# H200 141GB 8/node: idun-10-[01-03], idun-08-02, idun-09-*
# A single-GPU job only needs one free card, and partly-used H100/A100 nodes
# almost always have one, so h100 queues fine here -- 80GB is sufficient (see
# memory below). Swap to gpu:h200:1 if the H100s are saturated.
#
# GPU memory, scaled from revL48.slurm's ~53.3GB prediction at 48L/768d/B=24:
# activations track n_layer x n_embd x microbatch, so 24x1024x30 = 737,280 vs
# 48x768x24 = 884,736 -> 0.83x, i.e. ~44GB, plus ~7GB for 354M params + grads +
# Muon/AdamW state = ~51GB. Comfortable on 80GB. The 60x8 reversible arms hold
# O(1) activations in depth under segL, so they sit lower still.
#
# ── Host RAM: 64G, up from the 32-48G the TinyStories jobs use ───────────────
# load_bin (data.py:83) does np.array(memmap), i.e. it COPIES the whole bin into
# RAM rather than memory-mapping it: 17.2GB train + 0.86GB val = ~18.1GB
# resident before the model is even built. 32G would be uncomfortably tight.
#
# ── Walltime: 110h ───────────────────────────────────────────────────────────
# Anchored on a measured number, not a guess. revL12.slurm:15 records 5.8 s/step
# at 12L/12H/768d, 491,520 tok/step, on an A100, for the vf_scaling+segL arm with
# cheap_metrics and track_recon_drift on. Scaling:
#   FLOPs/token, 12L/768d : 6*85.0M + 6*50304*768  + 12*12*1024*768  ~=  855 MFLOP
#   FLOPs/token, 24L/1024d: 6*302M  + 6*50304*1024 + 12*24*1024*1024 ~= 2423 MFLOP
#   -> 2.83x compute; A100->H100 bf16 in practice ~2.2x
#   reversible+segL: 5.8 * 2.83 / 2.2 ~= 7.5 s/step -> 30k steps ~= 62h
#   baseline (no segL recompute):                                  ~= 48h
# Worst arm ~70h with eval + cheap_metrics overhead; 110h leaves real headroom.
# GPUQ's cap is 14 days (noted in EXAMPLE.slurm:28), so this is well
# inside the limit and needs no resume-chaining. That matters, because --resume
# (train.py:1277) restores model/opt/step but NOT the data iterator position or
# RNG state -- a mid-run restart would silently break data-order parity between
# arms, which is exactly what the paper controls for. Let these run in one shot.
#
# ── Disk ─────────────────────────────────────────────────────────────────────
# Checkpoints are ~3-4GB each at this size (354M params + Muon/AdamW state), and
# train.py writes best_*.pt and final_*.pt per arm -> budget ~30GB for 4 arms.
#
# Usage:   sbatch job_owt_medium.slurm
# Analyze: python analyze_wandb.py --project sympformer-owt-medium \
#              --filter_config n_embd=1024 block_size=1024 n_layer=24 \
#              --format pdf --out_dir plots/owt_medium
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=owt_medium
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                 # verify name: sinfo -o "%N %G" | sort -u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=110:00:00
#SBATCH --array=0-3                       # 0=baseline 1=yurii_lt 2=vpm+segL 3=vf+segL
#SBATCH --output=output_owt_medium_%A_%a.txt
#SBATCH --error=output_owt_medium_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=owt_medium_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: paper 'medium' (24L/16H/1024d) ───────────────────────────
N_LAYER=24
N_HEAD=16
N_EMBD=1024
BLOCK_SIZE=1024

# ── Batch/accum -- both give 491,520 tokens/step (see header) ────────────────
BATCH_SIZE_BASE=30; GRAD_ACCUM_BASE=16   # baseline / yurii_lt : paper's own 30x16
BATCH_SIZE_REV=60;  GRAD_ACCUM_REV=8     # reversible + segL   : same tokens, half the passes

# ── Eval batches -- both average over 4,915,200 val tokens (see header) ──────
EVAL_BATCHES_BASE=160                    # 160 x 30 x 1024
EVAL_BATCHES_REV=80                      #  80 x 60 x 1024

# ── 30k-step OWT recipe, paper Table 1 (OWT 24L column) ─────────────────────
MAX_STEPS=30000
WARMUP_STEPS=3000
EVAL_INTERVAL=100
LOG_INTERVAL=50
SAMPLE_INTERVAL=2000                     # generations are cheap diagnostics; keep rare on OWT

PEAK_LR=6e-4                             # AdamW peak
MUON_LR=0.006                            # paper OWT 24L (12L would be 0.004)
MIN_LR_RATIO=0.1
DATASET=openwebtext

# vpm_scaling target across-depth log-volume change: total log|det| = -REV_LAMBDA
REV_LAMBDA=0.0

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

# ── Pre-flight: fail loudly and early if preprocessing has not been run ──────
for f in data/${DATASET}_train.bin data/${DATASET}_val.bin; do
  if [[ ! -s "$f" ]]; then
    log "FATAL: missing ${f}"
    log "       Run:  sbatch job_owt_preprocess.slurm   (and wait for it to finish)"
    exit 1
  fi
done
log "data OK: $(ls -lh data/${DATASET}_*.bin | awk '{print $9"="$5}' | tr '\n' ' ')"
log "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo NA)"

COMMON_ARGS=(
  --data_dir data --dataset "${DATASET}"
  --out_dir "${OUTDIR}"
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
  --max_steps "${MAX_STEPS}" --eval_interval "${EVAL_INTERVAL}" --log_interval "${LOG_INTERVAL}"
  --warmup_steps "${WARMUP_STEPS}" --device "${DEVICE}" --plot
  --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --lr_schedule cosine
  --rev_scale_lr_mult 10
  --optimizer muon --muon_lr "${MUON_LR}" --muon_momentum 0.95
  --betas 0.9 0.95 --grad_clip 1.0
  --amp_dtype bf16
  --attn_impl sdpa
  --track_recon_drift
  --wandb --wandb_project sympformer-owt-medium
  --sample_interval "${SAMPLE_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
NARROW="--rev_embed narrow --rev_lift dup --rev_readout sum"

BASE_BATCH="--batch_size ${BATCH_SIZE_BASE} --grad_accum_steps ${GRAD_ACCUM_BASE} --eval_batches ${EVAL_BATCHES_BASE}"
REV_BATCH="--batch_size ${BATCH_SIZE_REV} --grad_accum_steps ${GRAD_ACCUM_REV} --eval_batches ${EVAL_BATCHES_REV}"

NAMES=(
  baseline_owt_M
  yuriilt_owt_M
  vpm_segL_owt_M
  vf_segL_owt_M
)
TASK_ARGS=(
  "--arch baseline  ${BASE_BATCH} --run_name baseline_owt_M"
  "--arch yurii_lt  ${BASE_BATCH} --run_name yuriilt_owt_M"
  "--arch reversible --rev_regime vpm_scaling --rev_lambda ${REV_LAMBDA} ${NARROW} ${REV_BATCH} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vpm_segL_owt_M"
  "--arch reversible --rev_regime vf_scaling                             ${NARROW} ${REV_BATCH} --mem_efficient --mem_efficient_segment ${N_LAYER} --run_name vf_segL_owt_M"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$IDX]}
ARGS=${TASK_ARGS[$IDX]}

log "task ${IDX}: ${NAME}"
log "tokens/step = batch x accum x block_size (target 491,520)"

run python3 -u train.py "${COMMON_ARGS[@]}" ${ARGS} \
  --wandb_run_name "${NAME}"
rc=$?

log "task ${IDX} (${NAME}) finished with exit ${rc}"
exit $rc
