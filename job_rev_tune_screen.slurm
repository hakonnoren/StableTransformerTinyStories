#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_rev_tune_screen.slurm — Tier 1 of theory/yurii_transfer_experiments.md.
#
# Question: of the settings that differ between yurii_lt and the conformal
# reversible model, is any of them a HYPERPARAMETER -- changeable without
# altering the forward step -- and does moving it close the 0.020 gap?
#
# Short horizon (3000 steps) so a knob either shows up cheaply or gets dropped.
# Every arm is the same protocol with ONE thing changed. Tier 1 uses existing
# flags only; no patch, no code change.
#
#   ID  arm          flags added to the shared protocol      what it tests
#    0  ref          (none)                                  the reference
#    1  yurii        --arch yurii_lt --scalar_lr_mult 5      the target to close
#    2  muon01       --muon_lr 0.01                          never swept for this arch
#    3  muon04       --muon_lr 0.04                            "
#    4  randninit    --rev_randn_init --rev_epsilon 0.5      break the gamma=alpha=0 init
#    5  readout_x    --rev_readout x                         yurii reads x only (*)
#    6  lift_zeros   --rev_lift zeros                        yurii's --no_v0_init analogue
#    7  tanh         --rev_tanh                              bound gamma,alpha to (-1,1)
#    8  baseline     --arch baseline                         the non-reversible anchor
#
# Arm 8 is appended rather than grouped with the other two references so the ID of
# every arm above is unchanged from the first version of this file. The three
# reference points are therefore 0 (reversible), 1 (yurii) and 8 (baseline).
# It also doubles as a protocol check: the baseline at warmup 40 is the one
# configuration whose behaviour is already known at a longer horizon -- 1.0444 at
# 10k (yujpwafo) against 1.0529 at warmup 1000 -- so if arm 8 comes out level with
# or worse than arm 0, suspect the protocol before believing the knobs.
#
# (*) Verified by autograd: under --rev_readout x the LAST block's MLP receives
# exactly zero gradient (z_L never reaches the logits), so arm 5 trains 12
# attention sublayers and 11 MLP sublayers. That is inherent to the setting, not a
# bug -- but it means arm 5 is not purely a readout change, it also removes one
# MLP's worth of trainable depth. Read a loss regression on this arm accordingly.
#
# Arms 0 and 2-7 are --arch reversible --rev_regime vf_scaling --rev_embed narrow:
# vf_scaling is the best reversible arm at 10k (1.0553 narrow / 1.0569 wide) and
# narrow keeps parameter parity with the baseline. Arm 1 is yurii_lt, which has no
# narrow/wide switch of its own (--rev_embed is inert for it).
#
# WARMUP IS 40 STEPS FOR EVERY ARM, INCLUDING YURII
# -------------------------------------------------
# Not an arm -- part of the shared protocol, so all eight runs get it. Evidence:
# on the baseline at 10k, warmup 1000 -> 40 moved best val 1.0529 -> 1.0444
# (myvrtiq9 -> yujpwafo, both eager), the largest single tuning effect measured
# anywhere in this project; on the reversible model at 3000 steps it was worth
# 1.1123 -> 1.1099. Faster warmup should also let the arms separate sooner, which
# is the point of a 3000-step screen.
#
# --muon_warmup_steps stays at its default -1, which means "reuse
# --warmup_steps", so BOTH the AdamW groups and the Muon group warm up over 40
# steps. That is exactly what the baseline run cited above did.
#
# ⚠ Consequence for arm 1: yurii_lt's own recipe is warmup 1000 (App. A.3), so
# this arm is deliberately OFF-recipe and its number here is NOT comparable to
# the 1.0352 it scored in job 25257774. What arm 1 provides is the gap measured
# under the same protocol as arms 0 and 2-7. If yurii degrades at warmup 40 while
# the reversible arms improve, that is itself a result -- but do not read it as a
# yurii number.
#
# NO MEMORY-EFFICIENT BACKPROP
# ----------------------------
# No arm passes --mem_efficient. This matters more than usual here: the previous
# 3000-step tuning sweep (sympformer-reversible-lrsweep, revL12_*) ran entirely
# with --mem_efficient --mem_efficient_segment 12 and logged reconstruction drift
# of 0.014-0.021, so all of its numbers -- including the 1.1123 reference -- are
# confounded by reconstruction error. There is no eager reversible reference at
# this horizon. Arm 0 establishes one, and nothing here should be compared to the
# old segL sweep.
#
# ONE SEED, AND WHAT THAT COSTS
# -----------------------------
# Seed 1337 everywhere, no replicates. The noise estimate therefore comes from
# elsewhere: the yurii_state seed pairs in sympformer-yurii-ext differ by 0.0014,
# 0.0012 and 0.0004 at 10k (s1337 vs s1338), so sigma ~ 0.0015 conservatively.
# Treat an improvement over arm 0 of >= 0.004 (~2.7 sigma) as real and anything
# smaller as unresolved. The gap being chased is ~0.020, i.e. ~13 sigma, so the
# screen can still detect a knob that recovers a fifth of it. Promoted arms get
# 10k steps and a second seed; nothing here is a final number.
#
# READ THE MECHANISM, NOT JUST THE LOSS
# -------------------------------------
# --cheap_metrics is on, so every arm logs cheap/act_rms per layer. Record the
# L11/L0 ratio next to best_val. Measured at 10k: yurii x4.04, baseline x3.77,
# vpb_baseline x4.64, but vf_scaling x1.00 -- it is the only model whose
# activation scale does not grow with depth, because it learns gamma ~ +0.33 and
# contracts (~-400 nats of log|det| per block). The hypothesis is that this
# update-to-state ratio is what separates the architectures, which predicts arms
# that push the ratio from x1 toward x4 should improve. If the loss moves without
# the ratio moving, or the reverse, that hypothesis is wrong -- worth knowing
# before spending 10k-step runs.
#
# Resources
# ---------
# Measured eager at this geometry and batch split (job 25257774, 10001 steps):
# peak 26.8-27.7 GB for the reversible arms, 30.2 GB for yurii_lt and 25.1 GB for
# the baseline, all inside a 40 GB A100. Wall-clock there was 10.1 h (baseline) to
# 13.7 h (yurii) for 10001 steps, so 3000 steps is ~3.0-4.1 h plus startup;
# --time=08:00:00 leaves ~2x headroom and queues sooner than a long ask.
# Host --mem is dataset-driven (TinyStories bins ~950 MB into RAM twice).
#
# Submit
# ------
#   sbatch job_rev_tune_screen.slurm                  # all 9 arms
#   sbatch --array=0,1,8 job_rev_tune_screen.slurm    # the three references only
#   sbatch --array=2-7 job_rev_tune_screen.slurm      # knobs only
#
# Analyze
#   python fetch_wandb_runs.py --project sympformer-rev-tune-screen \
#          --out_dir_tag revtune_<JOBID> --dest results/wandb_revtune_<JOBID>
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=revtunescreen
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-8                        # ref + yurii + 6 knobs + baseline
#SBATCH --output=output_revtune_%A_%a.txt
#SBATCH --error=output_revtune_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DEVICE=cuda
OUTDIR=revtune_${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUTDIR}"

# ── Shared protocol ───────────────────────────────────────────────────────────
# 12L/12H/768d at the paper's TinyStories geometry, batch 24x20 = 491,520
# tokens/step (accumulation is loss-scaled in train.py, so the split is a pure
# memory-vs-speed choice), 3000 steps = 1.47B tokens ~ 3.1 epochs.
N_LAYER=12; N_HEAD=12; N_EMBD=768; BLOCK_SIZE=1024
BATCH_SIZE=24; GRAD_ACCUM_STEPS=20
MAX_STEPS=3000
WARMUP_STEPS=40              # shared by every arm, AdamW and Muon both -- see header
PEAK_LR=6e-4
MUON_LR=0.02                 # arms 2 and 3 override this
MIN_LR_RATIO=0.1
EVAL_INTERVAL=150; EVAL_BATCHES=50; LOG_INTERVAL=25
SEED=1337
DATASET=tinystories
REV_LAMBDA=0.0

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

COMMON_ARGS=(
  --data_dir data --dataset "${DATASET}"
  --out_dir "${OUTDIR}"
  --n_layer "${N_LAYER}" --n_head "${N_HEAD}" --n_embd "${N_EMBD}" --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}"
  --eval_interval "${EVAL_INTERVAL}" --eval_batches "${EVAL_BATCHES}" --log_interval "${LOG_INTERVAL}"
  --lr_schedule cosine --min_lr_ratio "${MIN_LR_RATIO}" --peak_lr "${PEAK_LR}"
  --optimizer muon --muon_lr "${MUON_LR}"
  --rev_scale_lr_mult 10
  --grad_clip 1.0 --amp_dtype bf16 --attn_impl sdpa
  --seed "${SEED}" --device "${DEVICE}" --plot
  --wandb --wandb_project sympformer-rev-tune-screen
  --sample_interval "${EVAL_INTERVAL}" --sample_prefix_tokens 32 --sample_max_new_tokens 128
  --sample_temperature 0.8 --sample_top_k 40
  --cheap_metrics
)

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Arms 2-7 are the reference plus exactly one changed flag. Keep --array in sync.
# NOTE --muon_lr appears twice on arms 2/3 (once from COMMON_ARGS, once here);
# argparse keeps the LAST occurrence, which is the intended override.
REV="--arch reversible --rev_regime vf_scaling --rev_embed narrow"

NAMES=(
  ref
  yurii
  muon01
  muon04
  randninit
  readout_x
  lift_zeros
  tanh
  baseline
)
TASK_ARGS=(
  "${REV} --rev_lift dup --rev_readout sum --run_name ref"
  "--arch yurii_lt --scalar_lr_mult 5 --run_name yurii"
  "${REV} --rev_lift dup --rev_readout sum --muon_lr 0.01 --run_name muon01"
  "${REV} --rev_lift dup --rev_readout sum --muon_lr 0.04 --run_name muon04"
  "${REV} --rev_lift dup --rev_readout sum --rev_randn_init --rev_epsilon 0.5 --run_name randninit"
  "${REV} --rev_lift dup --rev_readout x   --run_name readout_x"
  "${REV} --rev_lift zeros --rev_readout sum --run_name lift_zeros"
  "${REV} --rev_lift dup --rev_readout sum --rev_tanh --run_name tanh"
  "--arch baseline --run_name baseline"
)

ID=${SLURM_ARRAY_TASK_ID}
if [[ -z "${ID}" || -z "${NAMES[$ID]+x}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID='${ID}'. Submit with: sbatch job_rev_tune_screen.slurm"; exit 1
fi
NAME="${NAMES[$ID]}"

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} arm=${NAME} host=$(hostname)"
log "Protocol: ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS} steps=${MAX_STEPS} warmup=${WARMUP_STEPS} (AdamW+Muon) eager"
log "OUTDIR=${OUTDIR}"

# ── GPU sanity check ──────────────────────────────────────────────────────────
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"

# shellcheck disable=SC2086  (intentional word-splitting of TASK_ARGS)
run python3 -u train.py "${COMMON_ARGS[@]}" ${TASK_ARGS[$ID]}

log "Task ${ID} (${NAME}) done."
