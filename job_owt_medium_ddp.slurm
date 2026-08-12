#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_owt_medium_ddp.slurm — the same OpenWebText 'medium' run as
# job_owt_medium.slurm (24L/16H/1024d, 30k steps, paper arXiv:2601.23236), but
# data-parallel across all 8 GPUs of one node via torchrun.
#
# Identical science, ~8x less wall time. The single-GPU job stays in the repo as
# the fallback / reference; this one is the same experiment, not a different one.
#
# PREREQUISITE: sbatch job_owt_preprocess.slurm first.
#
# ── Why the results are comparable to the 1-GPU job ──────────────────────────
# --batch_size and --grad_accum_steps describe the GLOBAL optimizer step. Under
# N ranks each rank runs grad_accum_steps//N micro-steps of batch_size sequences,
# so tokens/step is 491,520 whatever N is, and BlockEpochIterator hands rank r
# the blocks a 1-GPU run would have consumed at micro-step m*N+r -- the data
# order is the same sequence, merely dealt out across ranks. Verified on CPU/gloo
# at N = 1, 2, 4: identical best_val to 6 decimals, identical tokens/step.
# The gradient is also exactly the 1-GPU gradient: with AdamW the parameters after
# 6 steps agree to 6e-8 (float32 epsilon). With Muon the agreement is ~1e-4
# instead, because Newton-Schulz amplifies float reassociation noise -- that is a
# property of Muon, not of DDP, and the same wobble appears between two 1-GPU runs
# that differ only in accumulation order.
#
# ── Resources ────────────────────────────────────────────────────────────────
# One whole 8-GPU node per arm. Confirmed gres types on this cluster:
# a100, h100, h200, p100, v100 (sinfo -o "%N %G" -h | grep -o 'gpu:[a-z0-9]*').
#   H100 80GB  8/node: idun-01-[01-06], idun-08-01
#   H200 141GB 8/node: idun-10-[01-03], idun-08-02, idun-09-*
#
# h200 is the default here on availability grounds: at the time of writing two
# H200 nodes (idun-10-[01-02]) were fully idle versus one H100 node (idun-01-06),
# so twice as many arms can start immediately. 141GB also leaves generous room
# over the ~51GB that microbatch 30 needs at 24L/1024d. Re-check before
# submitting -- this changes hour to hour:
#   sinfo -p GPUQ -O "nodelist:22,gres:28,gresused:28,statecompact:12" | sort
#
# DO NOT relax this to a bare --gres=gpu:8. There are idle 8-GPU V100 nodes
# (idun-07-01) that would match, and Volta has no bf16 -- the run would fail or
# silently fall back. The type pin is load-bearing.
#
# --mem=256G, and this is NOT padding. load_bin (data.py) does np.array(memmap),
# i.e. it COPIES the bin into RAM rather than mapping it -- and it does so
# per-process, so 8 ranks hold 8 x 18.1GB = ~145GB of identical token data. The
# H100/H200 nodes have ~2TB so this is affordable, but it is the one place where
# the DDP run costs materially more than the single-GPU one. Switching load_bin
# to return the memmap directly would drop this to ~18GB shared via the page
# cache; it is a one-line change that was deliberately NOT made here to keep the
# single-GPU numerics untouched.
#
# ── Walltime ─────────────────────────────────────────────────────────────────
# Single-GPU estimate was ~48h (baseline) / ~62h (reversible+segL) on one H100.
# Communication is one 1.4GB all-reduce per optimizer step (no_sync suppresses it
# on the other micro-steps), which is milliseconds over NVLink inside an XE9680,
# so scaling should be close to linear: ~7h and ~9h respectively. --time=24:00:00
# is deliberately ~2.5x the estimate, since this is the first multi-GPU run of
# this code on real hardware and the estimate is extrapolated, not measured.
#
# ── First run: watch for these two things ────────────────────────────────────
# 1. "[ddp][grad-check] ..." near the top. If it lists parameters, read the note
#    it prints. Anything under blocks.* is already handled by dense-grad mode;
#    anything else can stall the all-reduce, in which case add
#    --ddp_find_unused_params.
# 2. If step 0 never completes, that is the classic DDP hang, not slowness.
#    Kill it and rerun with --ddp_find_unused_params before burning a day.
#
# Usage:   sbatch job_owt_medium_ddp.slurm
# Analyze: python analyze_wandb.py --project sympformer-owt-medium \
#              --filter_config n_embd=1024 block_size=1024 n_layer=24 \
#              --format pdf --out_dir plots/owt_medium
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=owt_med_ddp
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # ONE task; torchrun forks the 8 workers
#SBATCH --gres=gpu:h200:8                 # or gpu:h100:8 -- see header, check availability first
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-3                       # 0=baseline 1=yurii_lt 2=vpm+segL 3=vf+segL
#SBATCH --output=output_owt_med_ddp_%A_%a.txt
#SBATCH --error=output_owt_med_ddp_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1                  # 8 ranks on one node; do not oversubscribe
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN                    # quiet unless something is actually wrong
export TOKENIZERS_PARALLELISM=false

NPROC=8                                   # GPUs per node; must match --gres count
DEVICE=cuda
OUTDIR=owt_med_ddp_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Model geometry: paper 'medium' (24L/16H/1024d) ───────────────────────────
N_LAYER=24
N_HEAD=16
N_EMBD=1024
BLOCK_SIZE=1024

# ── GLOBAL batch/accum -- both give 491,520 tokens/step, split across NPROC ──
# Divisibility is enforced in train.py: grad_accum_steps and eval_batches must
# both divide by NPROC or the run refuses to start rather than silently changing
# the token budget. At NPROC=8: 16->2 and 8->1 micro-steps per rank.
BATCH_SIZE_BASE=30; GRAD_ACCUM_BASE=16   # baseline / yurii_lt
BATCH_SIZE_REV=60;  GRAD_ACCUM_REV=8     # reversible + segL

# ── GLOBAL eval batches -- both cover 4,915,200 val tokens (paper) ──────────
EVAL_BATCHES_BASE=160                    # 160 x 30 x 1024 ; /8 = 20 per rank
EVAL_BATCHES_REV=80                      #  80 x 60 x 1024 ; /8 = 10 per rank

# ── 30k-step OWT recipe, paper Table 1 (OWT 24L column) ─────────────────────
MAX_STEPS=30000
WARMUP_STEPS=3000
EVAL_INTERVAL=100
LOG_INTERVAL=50
SAMPLE_INTERVAL=2000

PEAK_LR=6e-4                             # AdamW peak
MUON_LR=0.006                            # paper OWT 24L (NOT the TinyStories 0.02)
MIN_LR_RATIO=0.1
DATASET=openwebtext
REV_LAMBDA=0.0

log() { echo "[$(date -Is)] $*"; }
run() {
  log "RUN: $(printf '%q ' "$@")"
  "$@"; local rc=$?
  if (( rc != 0 )); then
    log "FAILED: exit ${rc}"
    log "GPU state at failure (host=$(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
  fi
  return $rc
}

# ── Pre-flight ───────────────────────────────────────────────────────────────
for f in data/${DATASET}_train.bin data/${DATASET}_val.bin; do
  if [[ ! -s "$f" ]]; then
    log "FATAL: missing ${f}"
    log "       Run:  sbatch job_owt_preprocess.slurm   (and wait for it to finish)"
    exit 1
  fi
done
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if (( NGPU != NPROC )); then
  log "FATAL: NPROC=${NPROC} but the allocation has ${NGPU} visible GPU(s)."
  log "       Keep --gres=gpu:<type>:N and NPROC in sync, or the ranks will fight over devices."
  exit 1
fi
log "host=$(hostname) gpus=${NGPU}"
log "data OK: $(ls -lh data/${DATASET}_*.bin | awk '{print $9"="$5}' | tr '\n' ' ')"
log "NOTE: each rank copies the 17.2GB train bin into RAM at startup; expect a slow first minute."

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

log "task ${IDX}: ${NAME} on ${NPROC} GPUs"

# --standalone: single-node rendezvous, no master addr/port bookkeeping needed.
run torchrun --standalone --nproc_per_node="${NPROC}" train.py \
  "${COMMON_ARGS[@]}" ${ARGS} \
  --wandb_run_name "${NAME}"
rc=$?

log "task ${IDX} (${NAME}) finished with exit ${rc}"
exit $rc
