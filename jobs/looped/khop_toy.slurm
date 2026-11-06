#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# khop_toy.slurm — the looped k-hop toy task (looped/khop.py, run_looped.py) at
# scale: reproduce the 3-hop column of looped/HANDOFF.md §3 with more seeds, and
# extend it with the constrained-damping suggestions (HANDOFF.md §6).
#
# Each task is one run_looped.py job (d=32 takes ~5-20 min on one core); the task ID
# indexes the table built below. Print it with:   bash jobs/looped/khop_toy.slurm list
#
# One seed (1238, the one drift_check.py's long-horizon analysis uses) by default; IDs
# below are for one seed. With SEEDS="1238 1239 1240" every arm gets 3 tasks and the
# table is 3x longer -- print it with 'list' to get the ranges.
#
#   block  IDs      what
#   A      0-6      3-hop main table, d=32, 8000 steps, r in [3, 8]: vanilla, relax3,
#                   kick_fixed (rates 0.005: the October failure), kick_pos,
#                   kick_pos+carry, relax3 delta=1/4+carry, relax3 delta=1+carry
#                   (delta=1+carry: the best toy recipe, HANDOFF.md §3 phase-4 update)
#   B      7-18     width / depth: d=64 and d=128 (16000 steps), 2 core layers (8000
#                   steps) for vanilla, relax3, kick_pos+carry, relax3 delta=1+carry
#   C      19-30    harder: K=4, K=5, and S=16 at K=3 (16000 steps), same 4 arms
#   D      31-34    carry horizon 256, and carry on every step (prob 1), for the two
#                   carry recipes
#
#   sbatch jobs/looped/khop_toy.slurm                                         # all 35
#   sbatch --array=0-6 jobs/looped/khop_toy.slurm                             # block A only
#   sbatch --export=ALL,SEEDS="1238 1239 1240" --array=0-104 jobs/looped/khop_toy.slurm
#
# "carry" = --carry-max 64: with prob. 0.5 first run U{1..64} iterations without
# gradient (the detached carry). A task whose result JSON exists is skipped.
#
# Results: ${RESULTS}/<variant>/<arch><tag>_seed<S>.{json,pt} (RESULTS defaults to
# results/looped in this checkout). Then, per variant directory:
#   python looped/seeds_table.py results/looped/khop_K3 results/looped/khop_K4 ...
#   python looped/drift_check.py --tasks khop_K3        (long horizon, seed 1238)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=khop_toy
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-34
#SBATCH --output=output_khop_toy_%A_%a.txt
#SBATCH --error=output_khop_toy_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=FAIL

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
RESULTS=${RESULTS:-${REPO}/results/looped}
read -r -a SEEDS <<< "${SEEDS:-1238}"

# ── task table: "<variant dir>|<run_looped.py args>" ─────────────────────────────
K3="--task khop --K 3 --steps 8000 --r-lo 3 --r-hi 8"
LONG="--r-lo 3 --r-hi 8 --steps 16000"
ARMS_A=(
  "--arch vanilla"
  "--arch relax3"
  "--arch kick_fixed"
  "--arch kick_pos"
  "--arch kick_pos --carry-max 64 --tag _carry"
  "--arch relax3 --leak 0.25 --carry-max 64 --tag _leak4_carry"
  "--arch relax3 --leak 1.0 --carry-max 64 --tag _leak1_carry"
)
ARMS_4=(
  "--arch vanilla"
  "--arch relax3"
  "--arch kick_pos --carry-max 64 --tag _carry"
  "--arch relax3 --leak 1.0 --carry-max 64 --tag _leak1_carry"
)
TASKS=()
for arm in "${ARMS_A[@]}"; do for s in "${SEEDS[@]}"; do
  TASKS+=("khop_K3|${K3} ${arm} --seed ${s}"); done; done
for v in "khop_K3_d64|--task khop --K 3 ${LONG} --n-embd 64" \
         "khop_K3_d128|--task khop --K 3 ${LONG} --n-embd 128" \
         "khop_K3_core2|${K3} --n-core 2"; do
  for arm in "${ARMS_4[@]}"; do for s in "${SEEDS[@]}"; do TASKS+=("${v} ${arm} --seed ${s}"); done; done; done
for v in "khop_K4|--task khop --K 4 ${LONG}" \
         "khop_K5|--task khop --K 5 ${LONG}" \
         "khop_K3_S16|--task khop --K 3 --S 16 ${LONG}"; do
  for arm in "${ARMS_4[@]}"; do for s in "${SEEDS[@]}"; do TASKS+=("${v} ${arm} --seed ${s}"); done; done; done
for arm in "--arch kick_pos --carry-max 256 --tag _carry256" \
           "--arch relax3 --leak 1.0 --carry-max 256 --tag _leak1_carry256" \
           "--arch kick_pos --carry-max 64 --carry-prob 1.0 --tag _carryp1" \
           "--arch relax3 --leak 1.0 --carry-max 64 --carry-prob 1.0 --tag _leak1_carryp1"; do
  for s in "${SEEDS[@]}"; do TASKS+=("khop_K3|${K3} ${arm} --seed ${s}"); done; done

if [[ "${1:-}" == "list" ]]; then
  for i in "${!TASKS[@]}"; do printf '%4d  %s\n' "$i" "${TASKS[$i]}"; done
  exit 0
fi

ID=${SLURM_ARRAY_TASK_ID:?submit as an array, or run with 'list'}
[[ -z "${TASKS[$ID]+x}" ]] && { echo "no task ${ID} (there are ${#TASKS[@]})"; exit 1; }
VARIANT=${TASKS[$ID]%%|*}
ARGS=${TASKS[$ID]#*|}

# Output stem as run_looped.py names it: <arch><tag>_seed<S>
ARCH=$(sed -E 's/.*--arch ([^ ]+).*/\1/' <<< "${ARGS}")
TAG=$(grep -oE -- '--tag [^ ]+' <<< "${ARGS}" | cut -d' ' -f2)
SEED=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<< "${ARGS}")
STEM="${ARCH}${TAG}_seed${SEED}"
OUT="${RESULTS}/${VARIANT}"
mkdir -p "${OUT}/logs"
if [[ -f "${OUT}/${STEM}.json" ]]; then
  echo "[$(date -Is)] ${OUT}/${STEM}.json exists, skipping task ${ID}"; exit 0
fi

module load Anaconda3/2024.02-1
conda activate lra_torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "[$(date -Is)] task ${ID}: ${VARIANT} ${STEM} on $(hostname)"
# shellcheck disable=SC2086  (intentional word-splitting of ARGS)
python -u "${REPO}/looped/run_looped.py" ${ARGS} --out "${OUT}" --threads "${OMP_NUM_THREADS}" \
  > "${OUT}/logs/${STEM}.log" 2>&1
RC=$?
tail -3 "${OUT}/logs/${STEM}.log"
echo "[$(date -Is)] task ${ID} exit ${RC} after $(( SECONDS / 60 )) min"
exit ${RC}
