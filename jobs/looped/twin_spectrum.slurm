#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# twin_spectrum.slurm — looped/eqr/twin_spectrum.py on every checkpoint of a few runs: the block Jacobian's
# Ritz values at the twins' force argument against the twin pair's flip mu = -1 - c, copy separation and exact
# accuracy, checkpoint by checkpoint. Test of the candidate mechanism for the twin's decline between 20k and 50k
# (looped/dynlab/paper sec. 5): prediction, the right-answer spectrum of the delta = 2 twin drifts onto
# mu = -1 - c = -1.135 as its copies split; EqR's (flip at -1.105, visible to its read-out) does not.
#
#   sbatch jobs/looped/twin_spectrum.slurm                     # default: round-three/four runs of job 25939526
#   sbatch --export=ALL,RUN_IDS="twinL-leak2-br20k-s0-25939526 twinL-leak2-readL-br20k-s0-25939526" \
#          jobs/looped/twin_spectrum.slurm                     # the branch arms of sudoku_eqr.slurm (19-21)
#
# One JSON per run: ${RUN_ROOT}/twin_spectrum_<run id>.json (checkpoints already in it are skipped, so a job that
# hit its time limit can be resubmitted as is). ~2-3 min per checkpoint on an A100.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=twin_spec
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=output_twin_spec_%j.txt
#SBATCH --error=output_twin_spec_%j.err

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
JOB=${JOB_TAG:-25939526}
RUN_ROOT=${RUN_ROOT:-${WORK}/eqr_runs/sudoku_${JOB}}
RUN_IDS=${RUN_IDS:-"twinL-leak2-s0-${JOB} eqr-s0-${JOB} twinL-leak3-s0-${JOB} twinL-leak10-s0-${JOB}"}

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
cmp -s "${REPO}/looped/eqr/coupled_eqr.py" "${EQR_DIR}/models/coupled_eqr.py" || {
  echo "FATAL: ${EQR_DIR}/models/coupled_eqr.py differs from the repo's; run bash looped/eqr/install_overlay.sh ${EQR_DIR}"
  exit 1; }

for ID in ${RUN_IDS}; do
  echo "[$(date -Is)] ${ID}"
  python "${REPO}/looped/eqr/twin_spectrum.py" --eqr-dir "${EQR_DIR}" --run-root "${RUN_ROOT}" --run-id "${ID}" 2>&1 \
    | grep -E "^\[twin_spectrum\]|Error|Traceback|no checkpoints"
done
echo "[$(date -Is)] done"
