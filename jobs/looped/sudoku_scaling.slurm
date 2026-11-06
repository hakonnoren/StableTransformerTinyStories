#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sudoku_scaling.slurm — EqR's test-time scaling evaluation (depth x breadth) on
# already-trained Sudoku runs, so our arms are measured the way the EqR paper measures EqR.
#
# EqR's Sudoku numbers (README, "Reproduce Sudoku results") use depth 64, breadth 128
# (128 random initial states per puzzle), Langevin noise 0.5 during the iterations, and pick
# the answer by convergence (the top-4 runs whose logits changed least over the last 3
# iterations). sudoku_eqr.slurm only evaluated breadth 1 at the training noise 0.01. Here,
# per run, EqR's own evaluate.py (config/eval/depth_breadth.yaml) on the first NPUZ test
# puzzles (2048 = EqR's "lite" smoke set) at each depth:breadth in GRID, noise NOISE:
#
#   GRID default  16:16 64:16 256:16 16:128 64:128
#   64:128 with top-4 is EqR's paper setting; the README's released checkpoint (50k steps)
#   gives 98.60 (top-4 exact), 98.67 (majority vote), 99.20 (any of 128 correct).
#
# Reported per (depth, breadth) by summarize_sudoku.py (scaling table):
#   single   all/exact_accuracy: mean exact accuracy of one noisy run (averaged over the breadth)
#   conv1    exact accuracy of the single most-converged run
#   conv4    mean exact accuracy of the TOPK most-converged runs (EqR's headline metric)
#   vote     majority vote over the breadth
#   any      any run correct (oracle upper bound)
# Breadth only helps if runs from different starting states end in different answers;
# strongly contracting arms may all fall into one, which is what this measures.
#
# Usage (one task per arm; ARMS index = array index):
#   sbatch --export=ALL,RUN_ROOT=/cluster/work/$USER/eqr_runs/sudoku_25939526 jobs/looped/sudoku_scaling.slurm
#   ARMS default: eqr kick_pos relax3_leak1 relax3_leak4 released   (array 0-4)
#   "released" = EqR's released 50k-step checkpoint (reference for the paper's numbers).
#   Other arms / a specific checkpoint step:
#   sbatch --export=ALL,RUN_ROOT=...,ARMS="relax3_leak1_injL kick_pos_injL",CKPT_STEP=20000 \
#          --array=0-1 jobs/looped/sudoku_scaling.slurm
# Default checkpoint: the run's highest step (so after a 50k extension, the 50k one).
# Then: python looped/eqr/summarize_sudoku.py ${RUN_ROOT}
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=sudoku_scale
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=output_sudoku_scale_%A_%a.txt
#SBATCH --error=output_sudoku_scale_%A_%a.err

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
[[ -z "${RUN_ROOT:-}" ]] && { echo "set RUN_ROOT=<the sudoku_eqr run root>"; exit 1; }

read -r -a ARMS <<< "${ARMS:-eqr kick_pos relax3_leak1 relax3_leak4 released}"
SEED=${SEED:-0}
CKPT_STEP=${CKPT_STEP:-}
GRID=${GRID:-"16:16 64:16 256:16 16:128 64:128"}
NPUZ=${NPUZ:-2048}
SEQS=${SEQS:-4096}      # sequences per eval batch = puzzles per batch x breadth (EqR's lite: 32 x 128)
NOISE=${NOISE:-0.5}
TOPK=${TOPK:-4}

ID=${SLURM_ARRAY_TASK_ID:?submit as an array}
(( ID >= ${#ARMS[@]} )) && { echo "no arm for task ${ID} (ARMS=${ARMS[*]})"; exit 1; }
ARM=${ARMS[$ID]}

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
log() { echo "[$(date -Is)] $*"; }

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
case "${GPU_CC%%.*}" in
  8|9) ;;
  "") case "${GPU_NAME}" in *P100*|*V100*|*P40*|*K80*|*T4*|*B300*|*B200*)
        echo "FATAL: unsupported GPU '${GPU_NAME}'"; exit 1 ;; esac ;;
  *) echo "FATAL: '${GPU_NAME}' has compute capability ${GPU_CC}; this env needs 8.x or 9.x."; exit 1 ;;
esac
if [[ "${GPU_MEM}" =~ ^[0-9]+$ ]] && (( GPU_MEM < 40000 )); then
  echo "FATAL: '${GPU_NAME}' has ${GPU_MEM} MiB; need >= 40 GB."; exit 1
fi
cd "${EQR_DIR}" || { log "FATAL: no ${EQR_DIR}"; exit 1; }
if ! cmp -s "${REPO}/looped/eqr/coupled_eqr.py" models/coupled_eqr.py; then
  log "FATAL: ${EQR_DIR}/models/coupled_eqr.py differs from this repo's; run looped/eqr/install_overlay.sh ${EQR_DIR}"
  exit 1
fi

# ── checkpoint ──────────────────────────────────────────────────────────────────
if [[ "${ARM}" == "released" ]]; then
  # Link EqR's released checkpoint into the run root so its evals sit next to ours
  # (evaluate.py writes eval_preds/ next to the checkpoint's directory).
  SRC=${EQR_DIR}/downloaded_checkpoints/sudoku-extreme/eqr.pth
  [[ -f "${SRC}" ]] || { log "FATAL: no ${SRC}; run looped/eqr/setup_eqr.sh"; exit 1; }
  mkdir -p "${RUN_ROOT}/outputs/eqr-released/ref/ref/checkpoints"
  CKPT=${RUN_ROOT}/outputs/eqr-released/ref/ref/checkpoints/eqr.pth
  ln -sf "${SRC}" "${CKPT}"
else
  RID="${ARM//_/-}-s${SEED}-"
  # all checkpoints of this arm/seed (any job tag, any resumed segment), "step path" sorted by step
  LIST=$(ls "${RUN_ROOT}"/outputs/${RID}*/*/*/checkpoints/step_*_*.pth 2>/dev/null \
         | awk '{f=$0; sub(/.*\/step_/, "", f); split(f, a, "_"); print a[1], $0}' | sort -n)
  if [[ -n "${CKPT_STEP}" ]]; then
    CKPT=$(awk -v s="${CKPT_STEP}" '$1 == s {print $2}' <<< "${LIST}" | tail -1)
  else
    CKPT=$(tail -1 <<< "${LIST}" | cut -d' ' -f2-)
  fi
  [[ -z "${CKPT}" ]] && { log "FATAL: no checkpoint for ${ARM} seed ${SEED}${CKPT_STEP:+ at step ${CKPT_STEP}} under ${RUN_ROOT}"; exit 1; }
fi
log "arm=${ARM} checkpoint=${CKPT}"
log "grid=${GRID} puzzles=${NPUZ} noise=${NOISE} top-k=${TOPK}"

# ── depth x breadth ─────────────────────────────────────────────────────────────
for DB in ${GRID}; do
  D=${DB%%:*}; B=${DB##*:}
  GBS=$(( SEQS / B )); (( GBS < 1 )) && GBS=1
  NB=$(( (NPUZ + GBS - 1) / GBS ))
  log "eval depth ${D} breadth ${B}: ${GBS} puzzles x ${NB} batches"
  START=${SECONDS}
  python evaluate.py eval_yaml=config/eval/depth_breadth.yaml checkpoint="${CKPT}" \
    halt_max_steps=${D} noise_scale=${NOISE} different_init=${B} convergence_top_k=${TOPK} \
    global_batch_size=${GBS} max_eval_steps=${NB} suffix=scale_D${D}_B${B} \
    2>&1 | tr '\r' '\n' | grep -E "saved|Error|Traceback|coupled_eqr" | tail -3
  log "   done in $(( SECONDS - START ))s"
done

log "task ${ID} (${ARM}) done in $(( SECONDS / 60 )) min"
python "${REPO}/looped/eqr/summarize_sudoku.py" "${RUN_ROOT}" --scaling-only
