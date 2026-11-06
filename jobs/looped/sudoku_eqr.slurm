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
#   relax3            relax, rate 3 fixed                             "Relax, rate 3, fixed" / 2
#   relax3_leak1      relax, rate 3, common-mode damping delta = 1    NEW / 3 (best toy recipe)
#   relax3_leak4      relax, rate 3, delta = 1/4                      NEW / 4
#   kick_pos          kick, gamma = 0.05 + softplus, learned, init 1  NEW / 5
#   -- round two (after round 1 = array job 25939526) --
#   relax3_leak1_injL relax3_leak1, input x only in L's force          H <- ... B(L), as in EqR
#   kick_pos_injL     kick_pos, input x only in L's force
#   relax3_leak2      relax, rate 3, delta = 2 (c = 0.14)              more damping (submitted, then killed)
#   relax3_leak1_injL_self  relax3_leak1_injL whose forces also read their own stream, as EqR's do
#                     (NOT invertible; diagnostic): L <- ... B(L + H + x) ..., H <- ... B(H + L) ...
#                     Differs from relax3_leak1_injL only in self-reading, so the pair measures what
#                     the invertibility constraint (a force may not read the stream it updates) costs.
#   Why x only in L: with x in both forces and one shared B, relax's update is symmetric in
#   (H, L) and its streams merged in round 1 (cos 1.00). EqR's own H update has no x.
#
# Left out to keep the sweep small (both were in the first version of this job):
#   kick_learned_ch (per-channel learned kick; the PDF has it equal to the scalar one)
#   relax3_learned  (relax rates learned from 3; the PDF's single-run row, and on the toy
#                    tasks learned rates changed nothing -- HANDOFF.md §3, phase-4 update)
#
# 11 arms x len(SEEDS) tasks; task ID = arm * len(SEEDS) + seed index. One seed (0) by default:
#   sbatch jobs/looped/sudoku_eqr.slurm                                   # all 11 arms, seed 0
#   sbatch --export=ALL,SEEDS="0 1 2" --array=0-32 jobs/looped/sudoku_eqr.slurm   # 3 seeds
# Round two, into round 1's run root (one summary table; W&B ids <arm>-s0-25939526):
#   sbatch --export=ALL,JOB_TAG=25939526 --array=7-9 jobs/looped/sudoku_eqr.slurm
#   sbatch --export=ALL,JOB_TAG=25939526 --array=10 jobs/looped/sudoku_eqr.slurm   # self-read diagnostic
#
# Resume / extend: if this run id has checkpoints but none at MAX_STEPS, training resumes from the
# latest one (EqR restores weights, optimiser, EMA, carry, RNG and data position; the W&B run
# continues). EqR's LR is constant after warmup (lr_min_ratio 1.0), so extending 20k -> 50k
# equals a 50k run. Extend eqr, relax3_leak1 and kick_pos to EqR's paper budget:
#   sbatch --export=ALL,JOB_TAG=25939526,MAX_STEPS=50000 --array=0,4,6 jobs/looped/sudoku_eqr.slurm
# (The first submission of this, 2026-11-06, crashed at resume: IDUN's wandb 0.31 refuses EqR's
#  wandb.run._step assignment. pretrain_wandb_resume.patch fixes it; the job now checks both patches.)
# The resumed run writes a new outputs/<run id>/<date>/<time>/ dir; summarize_sudoku.py keys
# rows by checkpoint step, so the 20k and 50k evaluations appear as separate rows.
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
# Then looped/eqr/dynamics_eqr.py on the final checkpoint -> <run dir>/dynamics.json: learned
# rates at every checkpoint, the fixed-point / 2-cycle fraction out to 256 noise-free iterations,
# overthinking, and the spectral radius of one iteration (full radius, not the shift-free quotient:
# EqR's post-RMSNorm blocks are not shift-invariant). For runs that predate this, use
# jobs/looped/sudoku_dynamics.slurm.
#
# Prerequisites: bash looped/eqr/setup_eqr.sh (login node), then a passing
# jobs/looped/eqr_sanity.slurm, whose 200-step timings size --time below. The
# current estimate (~0.3-0.6 s/step on an A100 -> 2-3.5 h for 20k steps, plus ~2 h of
# evaluation) is from FLOP counts, not measured. The GPU type is not pinned (see the
# preflight), so a slower card than the sanity job's may need more of the 12 h.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=sudoku_eqr
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# Any GPU type; the P100/V100 pools are excluded by node (as in jobs/depth/revL12_lrsweep.slurm)
# together with the bad A100 node idun-07-06, and the preflight below refuses anything else unfit.
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=output_sudoku_eqr_%A_%a.txt
#SBATCH --error=output_sudoku_eqr_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
# Work area for the EqR checkout, the conda env, the HF cache and the runs. IDUN has no
# $USERWORK; its user work area is /cluster/work/$USER (not $HOME: quota).
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
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
read -r -a SEEDS <<< "${SEEDS:-0}"
WANDB_PROJECT=${WANDB_PROJECT:-looped-eqr-sudoku}
WANDB_ENTITY=${WANDB_ENTITY:-hakon-noren-ntnu}

COUPLED="arch.name=coupled_eqr@CoupledEqRModel arch.short_name=Coupled"
ARMS=(eqr kick_fixed kick_learned relax3 relax3_leak1 relax3_leak4 kick_pos
      relax3_leak1_injL kick_pos_injL relax3_leak2 relax3_leak1_injL_self)
ARM_ARGS=(
  ""
  "${COUPLED} ++arch.coupling=kick ++arch.rate=0.005"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=0.005 ++arch.learn_rates=true"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=1.0"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=0.25"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=1.0 ++arch.rate_constraint=positive ++arch.rate_min=0.05 ++arch.learn_rates=true"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=1.0 ++arch.inject_h=false"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=1.0 ++arch.rate_constraint=positive ++arch.rate_min=0.05 ++arch.learn_rates=true ++arch.inject_h=false"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=2.0"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=1.0 ++arch.inject_h=false ++arch.self_read=true"
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
# Any GPU type (faster to schedule than pinning A100s), as long as it can run this env:
# bf16 + torch 2.6/CUDA 12.4 + adam-atan2 built for sm_80/sm_90 (sm_80 kernels also run on
# 8.6/8.9 cards: A40, L40S, RTX A6000). So compute capability 8.x or 9.x; B300 (10.x) and
# P100/V100 are refused. Memory floor 40 GB until eqr_sanity.slurm has measured the need.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
case "${GPU_CC%%.*}" in
  8|9) ;;
  "") case "${GPU_NAME}" in *P100*|*V100*|*P40*|*K80*|*T4*|*B300*|*B200*)
        echo "FATAL: unsupported GPU '${GPU_NAME}'"; exit 1 ;; esac ;;
  *) echo "FATAL: '${GPU_NAME}' has compute capability ${GPU_CC}; this env needs 8.x or 9.x."
     echo "       Resubmit (P100/V100 nodes are excluded; add this node to --exclude)."; exit 1 ;;
esac
if [[ "${GPU_MEM}" =~ ^[0-9]+$ ]] && (( GPU_MEM < 40000 )); then
  echo "FATAL: '${GPU_NAME}' has ${GPU_MEM} MiB; EqR's batch of 768 is sized for >= 40 GB."; exit 1
fi
echo "GPU OK: ${GPU_NAME} (cc ${GPU_CC:-?}, ${GPU_MEM} MiB)"

cd "${EQR_DIR}" || { log "FATAL: no ${EQR_DIR}; run looped/eqr/setup_eqr.sh"; exit 1; }
# The overlay must match this repo checkout: refuse to train an out-of-date model.
if ! cmp -s "${REPO}/looped/eqr/coupled_eqr.py" models/coupled_eqr.py; then
  log "FATAL: ${EQR_DIR}/models/coupled_eqr.py differs from ${REPO}/looped/eqr/coupled_eqr.py."
  log "       Rerun: bash looped/eqr/install_overlay.sh ${EQR_DIR}  (not from inside a running array)"
  exit 1
fi
for P in pretrain_rate_groups.patch pretrain_wandb_resume.patch; do
  if ! git apply --reverse --check "${REPO}/looped/eqr/${P}" 2>/dev/null; then
    log "FATAL: ${EQR_DIR}/pretrain.py lacks ${P}; rerun bash looped/eqr/install_overlay.sh ${EQR_DIR}"
    exit 1
  fi
done

final_ckpt() { ls ${RUN_ROOT}/outputs/${RUN_ID}/*/*/checkpoints/step_${MAX_STEPS}_*.pth 2>/dev/null | head -1; }
# newest checkpoint of this run id by step number (the path itself has other underscores)
latest_ckpt() {
  ls ${RUN_ROOT}/outputs/${RUN_ID}/*/*/checkpoints/step_*_*.pth 2>/dev/null \
    | awk '{f=$0; sub(/.*\/step_/, "", f); split(f, a, "_"); print a[1], $0}' | sort -n | tail -1 | cut -d' ' -f2-
}

# ── train (skipped if this run id already has its final checkpoint; resumed if it has an earlier one) ──
if [[ -z "$(final_ckpt)" ]]; then
  RESUME_ARGS=()
  RESUME=$(latest_ckpt)
  if [[ -n "${RESUME}" ]]; then
    log "resuming from ${RESUME} (to step ${MAX_STEPS})"
    RESUME_ARGS=("++load_checkpoint='${RESUME}'")    # quoted for Hydra's override grammar
  fi
  # shellcheck disable=SC2086  (intentional word-splitting of ARM_ARGS)
  WANDB_RUN_ID=${RUN_ID} python pretrain.py --config-name train/eqr_sudoku \
    ++max_steps=${MAX_STEPS} ++seed=${SEED} eval_interval_steps=${EVAL_INTERVAL} \
    ++project_name=${WANDB_PROJECT} ++entity=${WANDB_ENTITY} ++run_name=${NAME} \
    ${ARM_ARGS[$A]} "${RESUME_ARGS[@]}" 2>&1 | tr '\r' '\n' | grep -v "it/s\]$\|s/it\]$"
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

# ── dynamics: learned rates per checkpoint, fixed points / 2-cycles to r=256, spectral radius ──────
log "dynamics (looped/eqr/dynamics_eqr.py)"
python "${REPO}/looped/eqr/dynamics_eqr.py" --eqr-dir "${EQR_DIR}" --ckpt "${CKPT}" 2>&1 \
  | grep -E "^\[rates\]|^\[dynamics\]|Error|Traceback"

log "task ${ID} (${NAME}) done in $(( SECONDS / 60 )) min. Summarize with:"
log "  python ${REPO}/looped/eqr/summarize_sudoku.py ${RUN_ROOT}"
