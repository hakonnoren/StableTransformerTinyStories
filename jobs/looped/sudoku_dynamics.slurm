#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sudoku_dynamics.slurm — run looped/eqr/dynamics_eqr.py on every finished run of a
# sudoku_eqr.slurm submission (one GPU, runs one after another; a few minutes each),
# then print the summary table. For runs trained before sudoku_eqr.slurm did this itself.
#
#   sbatch --export=ALL,RUN_ROOT=/cluster/work/$USER/eqr_runs/sudoku_<array job id> \
#          jobs/looped/sudoku_dynamics.slurm
#
# Per run it writes <run dir>/dynamics.json: learned rates at every checkpoint, the
# noise-free trajectory of 768 test puzzles to 256 iterations (fixed-point and 2-cycle
# fractions, overthinking, changed predictions, stop-signal rate) and the spectral
# radius of one iteration at the final state. Uses each run's LAST checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=sudoku_dyn
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=output_sudoku_dyn_%j.txt
#SBATCH --error=output_sudoku_dyn_%j.err

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
[[ -z "${RUN_ROOT:-}" ]] && { echo "set RUN_ROOT=<the sudoku_eqr run root>"; exit 1; }

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

N=0
for RUN in "${RUN_ROOT}"/outputs/*/*/*/; do
  # last checkpoint by step number (the path itself has other underscores)
  CKPT=$(ls "${RUN}"checkpoints/step_*_*.pth 2>/dev/null \
         | awk '{f=$0; sub(/.*\/step_/, "", f); split(f, a, "_"); print a[1], $0}' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -z "${CKPT}" ]] && continue
  echo "[$(date -Is)] ${CKPT}"
  python "${REPO}/looped/eqr/dynamics_eqr.py" --eqr-dir "${EQR_DIR}" --ckpt "${CKPT}" 2>&1 \
    | grep -E "^\[rates\]|^\[dynamics\]|Error|Traceback"
  N=$(( N + 1 ))
done
echo "[$(date -Is)] analysed ${N} runs"
python "${REPO}/looped/eqr/summarize_sudoku.py" "${RUN_ROOT}"
