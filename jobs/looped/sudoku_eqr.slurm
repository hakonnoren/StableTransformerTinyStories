#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sudoku_eqr.slurm — Sudoku-Extreme in the Equilibrium Reasoners (EqR) setting:
# EqR vs Håkon's kick coupling vs relax, plus the constrained-damping arms.
#
# Reproduces the Sudoku table of looped/docs/looped_summary.pdf ("Conformal
# relaxation coupling for looped reasoning models", October 2026) and adds the
# constrained arms of looped/HANDOFF.md §5. All arms are EqR's own code at a pinned
# commit (locuslab/EqR 134c754): its data, model, ACT carry/halting, noise, EMA,
# optimiser (AdamATan2) and evaluator. The coupled arms change only the state update
# (looped/eqr/coupled_eqr.py; see its docstring for the port and its one free choice,
# the alternating H,L,H,L,H,L,H schedule that matches EqR's 21 visited / 7
# differentiated block evaluations per iteration).
#
#   arm               update                                         PDF row / handoff arm
#   eqr               EqR, original                                  "EqR (original)" / 1
#   kick_fixed        kick, gamma = alpha = 0.005 fixed               "Kick, fixed per-stream rates"
#   kick_learned      kick, scalar rates learned from 0.005 (raw)     "Kick, learned scalar rates" / 6
#   kick_learned_ch   kick, per-channel rates learned from 0.005      "Kick, learned channel rates"
#   relax3            relax, rate 3 fixed                             "Relax, rate 3, fixed" / 2
#   relax3_learned    relax, scalar rates learned from 3              "Relax, learned rates"
#   relax3_leak4      relax, rate 3, common-mode damping delta = 1/4  NEW / 3
#   relax3_leak1      relax, rate 3, delta = 1                        NEW / 4
#   kick_pos          kick, gamma = 0.05 + softplus, learned, init 1  NEW / 5
#
# 9 arms x 3 seeds = 27 tasks; task ID = arm * 3 + seed index.
#   sbatch jobs/looped/sudoku_eqr.slurm                    # everything
#   sbatch --array=0-2,12-14 jobs/looped/sudoku_eqr.slurm  # EqR + relax3 only (3 seeds each)
#
# Recipe (the PDF's): config/train/eqr_sudoku.yaml unchanged except max_steps =
# MAX_STEPS (20000; EqR's paper budget is 50000 -- export MAX_STEPS=50000 for that).
# The October code was not kept, so the kick/relax rows are a re-implementation:
# rate values follow looped/run_looped.py's presets (kick 0.005, relax 3); the PDF
# does not say which "fixed per-stream" kick rate it used.
#
# Evaluation of the final EMA checkpoint (evaluate.py, breadth 1, noise EVAL_NOISE =
# EqR's training noise 0.01, i.e. what EqR's in-training eval uses):
#   full_D16          whole test set (423k puzzles) at 16 iterations -> "cell/exact, 16 it."
#   sub_D{16,64,256}  first SUB_BATCHES x 768 test puzzles (default 64 -> 49,152), with
#                     predictions saved -> "cell, 64 it.", "residual, 64 it.", r = 256,
#                     and the overthinking rate O / changed predictions dy (handoff §5)
# Then: python looped/eqr/summarize_sudoku.py ${RUN_ROOT} --out sudoku_table.md
#
# Not here yet (handoff §5): the fixed-point fraction and the shift-free spectral
# radius rho_q. Both need per-puzzle trajectories rather than EqR's mean residual.
#
# Prerequisites: bash looped/eqr/setup_eqr.sh (login node), then a passing
# jobs/looped/eqr_sanity.slurm, whose 200-step timings size --time below. The
# estimate here (~0.3-0.6 s/step on an A100 -> 2-3.5 h for 20k steps, plus ~2 h of
# evaluation) is from FLOP counts, not measured.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=sudoku_eqr
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-26
#SBATCH --output=output_sudoku_eqr_%A_%a.txt
#SBATCH --error=output_sudoku_eqr_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL

set -uo pipefail
WORK=${USERWORK:-$HOME/work}
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
# JOB_TAG names the run root and the W&B run ids. To redo one task of an earlier
# submission in place (training is skipped once its final checkpoint exists):
#   sbatch --export=ALL,JOB_TAG=<old array job id> --array=<task> jobs/looped/sudoku_eqr.slurm
JOB=${JOB_TAG:-${SLURM_ARRAY_JOB_ID:-local}}
RUN_ROOT=${RUN_ROOT:-${WORK}/eqr_runs/sudoku_${JOB}}

MAX_STEPS=${MAX_STEPS:-20000}
# In-training eval (whole test set, 16 it., ~30 min each) only feeds the W&B curves; 7000 gives
# points at 7k and 14k and none at the end, where evaluate.py's full_D16 would duplicate it.
EVAL_INTERVAL=${EVAL_INTERVAL:-7000}
SUB_BATCHES=${SUB_BATCHES:-64}
EVAL_NOISE=${EVAL_NOISE:-0.01}
SEEDS=(0 1 2)
WANDB_PROJECT=${WANDB_PROJECT:-looped-eqr-sudoku}
WANDB_ENTITY=${WANDB_ENTITY:-hakon-noren-ntnu}

COUPLED="arch.name=coupled_eqr@CoupledEqRModel arch.short_name=Coupled"
ARMS=(eqr kick_fixed kick_learned kick_learned_ch relax3 relax3_learned relax3_leak4 relax3_leak1 kick_pos)
ARM_ARGS=(
  ""
  "${COUPLED} ++arch.coupling=kick ++arch.rate=0.005"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=0.005 ++arch.learn_rates=true"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=0.005 ++arch.learn_rates=true ++arch.per_channel=true"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.learn_rates=true"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=0.25"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=1.0"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=1.0 ++arch.rate_constraint=positive ++arch.rate_min=0.05 ++arch.learn_rates=true"
)

ID=${SLURM_ARRAY_TASK_ID:?submit as an array: sbatch jobs/looped/sudoku_eqr.slurm}
NS=${#SEEDS[@]}
A=$(( ID / NS )); SEED=${SEEDS[$(( ID % NS ))]}
if (( A >= ${#ARMS[@]} )); then echo "no arm for task ${ID}"; exit 1; fi
ARM=${ARMS[$A]}
NAME="${ARM}_s${SEED}"
RUN_ID="${ARM//_/-}-s${SEED}-${JOB}"

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OUTPUT_ROOT=${RUN_ROOT}
mkdir -p "${RUN_ROOT}"

log() { echo "[$(date -Is)] $*"; }
log "task=${ID} arm=${ARM} seed=${SEED} run_id=${RUN_ID} host=$(hostname)"
log "EQR_DIR=${EQR_DIR} RUN_ROOT=${RUN_ROOT} MAX_STEPS=${MAX_STEPS}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "${EQR_DIR}" || { log "FATAL: no ${EQR_DIR}; run looped/eqr/setup_eqr.sh"; exit 1; }
# The overlay must match this repo checkout: refuse to train an out-of-date model.
if ! cmp -s "${REPO}/looped/eqr/coupled_eqr.py" models/coupled_eqr.py; then
  log "FATAL: ${EQR_DIR}/models/coupled_eqr.py differs from ${REPO}/looped/eqr/coupled_eqr.py."
  log "       Rerun: bash looped/eqr/install_overlay.sh ${EQR_DIR}  (not from inside a running array)"
  exit 1
fi

final_ckpt() { ls ${RUN_ROOT}/outputs/${RUN_ID}/*/*/checkpoints/step_${MAX_STEPS}_*.pth 2>/dev/null | head -1; }

# ── train (skipped if this run id already has its final checkpoint) ─────────────
if [[ -z "$(final_ckpt)" ]]; then
  # shellcheck disable=SC2086  (intentional word-splitting of ARM_ARGS)
  WANDB_RUN_ID=${RUN_ID} python pretrain.py --config-name train/eqr_sudoku \
    ++max_steps=${MAX_STEPS} ++seed=${SEED} eval_interval_steps=${EVAL_INTERVAL} \
    ++project_name=${WANDB_PROJECT} ++entity=${WANDB_ENTITY} ++run_name=${NAME} \
    ${ARM_ARGS[$A]} 2>&1 | tr '\r' '\n' | grep -v "it/s\]$\|s/it\]$"
  RC=${PIPESTATUS[0]}
  (( RC != 0 )) && { log "training FAILED (exit ${RC})"; exit "${RC}"; }
fi
CKPT=$(final_ckpt)
[[ -z "${CKPT}" ]] && { log "FATAL: no step_${MAX_STEPS} checkpoint under ${RUN_ROOT}/outputs/${RUN_ID}"; exit 1; }
log "checkpoint: ${CKPT}"

# ── evaluate ─────────────────────────────────────────────────────────────────
run_eval() {   # tag depth [max_eval_steps]
  local tag=$1 depth=$2 nb=${3:-}
  local extra=()
  [[ -n "${nb}" ]] && extra=(max_eval_steps=${nb} "save_outputs=[preds,labels]")
  log "eval ${tag}: ${depth} iterations${nb:+, ${nb} batches}"
  python evaluate.py eval_yaml=config/eval/depth_breadth.yaml checkpoint="${CKPT}" \
    halt_max_steps=${depth} noise_scale=${EVAL_NOISE} different_init=1 global_batch_size=768 \
    suffix=${tag} "${extra[@]}" 2>&1 | tr '\r' '\n' | grep -E "saved|Error|Traceback|coupled_eqr" | tail -3
}
run_eval full_D16 16
for D in 16 64 256; do run_eval sub_D${D} ${D} ${SUB_BATCHES}; done

log "task ${ID} (${NAME}) done in $(( SECONDS / 60 )) min. Summarize with:"
log "  python ${REPO}/looped/eqr/summarize_sudoku.py ${RUN_ROOT}"
