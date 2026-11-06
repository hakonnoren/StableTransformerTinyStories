#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sudoku_maps.slurm — looped/eqr/analyze_maps.py on every checkpoint of an export
# (looped/eqr/export_for_analysis.py), one array task per checkpoint, GPU preset:
# 256 puzzles x 16 random starts x 256 iterations at noise 0 / 0.25 / 0.5 / 0.75
# (selection, diversity, path independence), escape under noise from the noise-free
# endpoints, Jacobian spectra at endpoints, within-recursion traces, participation ratios.
#
#   E=/cluster/work/$USER/eqr_export
#   N=$(ls $E/ckpts/*.pth | wc -l)
#   sbatch --export=ALL,EXPORT=$E --array=0-$((N-1)) jobs/looped/sudoku_maps.slurm
#   # then: python looped/eqr/summarize_maps.py $E/maps   (or rsync $E/maps/ to the laptop)
# Single checkpoints: CKPTS="eqr_s0__step20000 kick_pos_s0__step20000" (names without .pth).
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=sudoku_maps
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=output_sudoku_maps_%A_%a.txt
#SBATCH --error=output_sudoku_maps_%A_%a.err

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
EXPORT=${EXPORT:-${WORK}/eqr_export}
PRESET=${PRESET:-gpu}
EXTRA=${EXTRA:-}                     # extra analyze_maps.py arguments, e.g. "--noise 0.5 --n 128"

if [[ -n "${CKPTS:-}" ]]; then read -r -a LIST <<< "${CKPTS}"; LIST=("${LIST[@]/%/.pth}"); LIST=("${LIST[@]/#/${EXPORT}/ckpts/}")
else mapfile -t LIST < <(ls "${EXPORT}"/ckpts/*.pth | sort); fi
ID=${SLURM_ARRAY_TASK_ID:-0}
(( ID >= ${#LIST[@]} )) && { echo "no checkpoint for task ${ID} (${#LIST[@]} in ${EXPORT}/ckpts)"; exit 1; }
CKPT=${LIST[$ID]}
NAME=$(basename "${CKPT}" .pth)
DATA=$(ls -d "${EXPORT}"/data/*/ | head -1)

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! cmp -s "${REPO}/looped/eqr/coupled_eqr.py" "${EQR_DIR}/models/coupled_eqr.py"; then
  echo "FATAL: ${EQR_DIR}/models/coupled_eqr.py differs from this repo's; run looped/eqr/install_overlay.sh ${EQR_DIR}"; exit 1
fi
mkdir -p "${EXPORT}/maps"
echo "[$(date -Is)] ${NAME} (${PRESET}) data ${DATA}"
python "${REPO}/looped/eqr/analyze_maps.py" --eqr-dir "${EQR_DIR}" --data "${DATA}" --ckpt "${CKPT}" \
  --preset "${PRESET}" --out "${EXPORT}/maps/${NAME}.json" ${EXTRA} 2>&1 \
  | grep -E "^\[(multistart|jacobian|trace|pr|done)\]|Error|Traceback|line [0-9]+"
echo "[$(date -Is)] done"
