#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# multihop.slurm — long chains on multi-hop composition (Kohli et al. 2604.07822), from the released r_dyn
# model, in the couplings and with the training/test-time methods of the Sudoku-Extreme runs (looped/multihop).
# Each task: fine-tune with the authors' curriculum from 23 hops (where r_dyn stops) towards 40, then
# looped/multihop/evaluate.py on the final model: 2-100 hops, depth to 256, adaptive halting, test noise x
# breadth, overthinking, train-vs-test.
#
# Once, on the login node:   bash looped/multihop/setup_multihop.sh
#
#   ID  arm               coupling (looped/multihop/model.py)       training
#   0   released          the authors' r_dyn and R=8 models, evaluated only
#   1   vanilla           the released recurrence h <- f(h)         the authors' recipe: R in [2, 8], Poisson(4)
#   2   vanilla_carry     the same                                  + EqR's recipe: detached carry U{1..64} (p 1/2), noise 0.01
#   3   relax             relax, gamma 3, delta 0 (the October note's "rate 3, fixed")   the authors' recipe
#   4   relax_d1_carry    relax, gamma 3, delta 1 (damped flip)     + carry + noise
#   5   twin_d2           twin, gamma 3, delta 2 (mean read-out)    the authors' recipe
#   6   twin_d2_carry     the same                                  + carry + noise
#
#   sbatch jobs/looped/multihop.slurm                                    # all 7, seed 0
#   sbatch --export=ALL,SEEDS="0 1 2" --array=0-20 jobs/looped/multihop.slurm   # 3 seeds: task = arm * 3 + seed index
#   bash jobs/looped/multihop.slurm list                                 # print the task table
#
# Budget per task: training stops at 40 hops, 300k steps or 20 h; evaluation ~30-45 min. A task whose
# final.pt / eval json exists skips that part, so a task that hit the time limit can be resubmitted as is
# (training restarts from the init checkpoint, since only stage checkpoints are kept).
# Results: ${RUN_ROOT}/<arm>_s<seed>/ (config.json, train_log.jsonl, curriculum.jsonl, stage_k*.pt, final.pt,
# eval_final.json); W&B project looped-multihop. Summary:  python looped/multihop/summarize.py ${RUN_ROOT}
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=multihop
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=output_multihop_%A_%a.txt
#SBATCH --error=output_multihop_%A_%a.err

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
LTG_DIR=${LTG_DIR:-${WORK}/LTG}
LTG_ENV=${LTG_ENV:-${WORK}/conda_envs/ltg_venv}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
JOB=${JOB_TAG:-${SLURM_ARRAY_JOB_ID:-local}}
RUN_ROOT=${RUN_ROOT:-${WORK}/multihop_runs/${JOB}}
DATA=${LTG_DIR}/data/multi_hop
INIT=${LTG_DIR}/checkpoints/multi_hop/final_models/r_dyn/checkpoint_epoch_5388.pt
R8=${LTG_DIR}/checkpoints/multi_hop/final_models/r8/checkpoint_epoch_1259.pt
read -r -a SEEDS <<< "${SEEDS:-0}"
STEPS=${STEPS:-300000}
HOURS=${HOURS:-20}
WANDB_PROJECT=${WANDB_PROJECT:-looped-multihop}
WANDB_ENTITY=${WANDB_ENTITY:-hakon-noren-ntnu}

AUTHORS=""                                                   # the authors' recipe (train.py defaults)
EQR="--carry-max 64 --carry-prob 0.5 --sigma-train 0.01"     # + EqR's detached carry and training noise
NAMES=(released vanilla vanilla_carry relax relax_d1_carry twin_d2 twin_d2_carry)
ARGS=(
  "eval-only"
  "--coupling vanilla ${AUTHORS}"
  "--coupling vanilla ${EQR}"
  "--coupling relax --gamma 3 --delta 0 ${AUTHORS}"
  "--coupling relax --gamma 3 --delta 1 ${EQR}"
  "--coupling twin --gamma 3 --delta 2 ${AUTHORS}"
  "--coupling twin --gamma 3 --delta 2 ${EQR}"
)
NS=${#SEEDS[@]}
if [[ "${1:-}" == list ]]; then
  for ((i = 0; i < ${#NAMES[@]} * NS; i++)); do
    echo "$i  ${NAMES[i / NS]}_s${SEEDS[i % NS]}  ${ARGS[i / NS]}"; done
  exit 0
fi
ID=${SLURM_ARRAY_TASK_ID:-0}
ARM=${NAMES[ID / NS]}
SEED=${SEEDS[ID % NS]}
[[ -z "${ARM}" ]] && { echo "FATAL: no task ${ID} (bash $0 list)"; exit 1; }
NAME=${ARM}_s${SEED}
OUT=${RUN_ROOT}/${NAME}
log() { echo "[$(date -Is)] $*"; }
log "task=${ID} arm=${ARM} seed=${SEED} out=${OUT} host=$(hostname)"

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
PY=${LTG_ENV}/bin/python
export PYTHONPATH=${REPO} OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
[[ -x "${PY}" && -s "${INIT}" && -s "${DATA}/test.json" && -s "${DATA}/train.json" ]] || {
  log "FATAL: missing env or data; run: bash looped/multihop/setup_multihop.sh"; exit 1; }
"${PY}" -c "import transformers; assert transformers.__version__.startswith('4.'), transformers.__version__" || exit 1
mkdir -p "${OUT}"
COMMON=(--ltg-dir "${LTG_DIR}" --data-dir "${DATA}")

if [[ "${ARM}" == released ]]; then
  for CK in "${INIT}" "${R8}"; do
    TAG=$(basename "$(dirname "${CK}")")
    if [[ -s "${OUT}/eval_${TAG}.json" ]]; then log "eval ${TAG}: done already"; continue; fi
    log "eval ${TAG}: ${CK}"
    "${PY}" -m looped.multihop.evaluate "${COMMON[@]}" --ckpt "${CK}" --released --seed "${SEED}" \
      --out "${OUT}/eval_${TAG}.json" || exit 1
  done
  log "done"; exit 0
fi

if [[ ! -s "${OUT}/final.pt" ]]; then
  log "train: ${ARGS[ID / NS]}"
  # shellcheck disable=SC2086  (intentional word splitting of the arm's arguments)
  "${PY}" -m looped.multihop.train "${COMMON[@]}" --init-ckpt "${INIT}" --out "${OUT}" --seed "${SEED}" \
    --steps "${STEPS}" --hours "${HOURS}" ${ARGS[ID / NS]} \
    --wandb-project "${WANDB_PROJECT}" --wandb-entity "${WANDB_ENTITY}" --run-name "${NAME}-${JOB}" || {
    log "training FAILED"; exit 1; }
else
  log "train: final.pt exists, skipped"
fi
if [[ ! -s "${OUT}/eval_final.json" ]]; then
  log "eval final.pt"
  "${PY}" -m looped.multihop.evaluate "${COMMON[@]}" --ckpt "${OUT}/final.pt" --seed "${SEED}" || exit 1
fi
log "task ${ID} (${NAME}) done in $(( SECONDS / 60 )) min. Summarize with: python looped/multihop/summarize.py ${RUN_ROOT}"
