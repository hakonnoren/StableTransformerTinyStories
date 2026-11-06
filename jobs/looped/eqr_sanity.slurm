#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# eqr_sanity.slurm — check the EqR install before spending GPU-days on it.
# Prerequisite: bash looped/eqr/setup_eqr.sh on the login node.
#
#   1. overlay unit tests (looped/eqr/test_coupled_eqr.py) on the GPU node
#   2. adam-atan2's CUDA backend loads
#   3. EqR's own "quick 2048-sample smoke eval" of the RELEASED Sudoku checkpoint
#      (README, "Reproduce Sudoku results"): depth 64, breadth 128, convergence
#      top-4, Langevin noise 0.5. README over 5 seeds:
#        convergence_top_k/exact_accuracy  98.60 ± 0.04
#        majority_vote/exact_accuracy      98.67 ± 0.05
#        different_init/any_correct        99.20 ± 0.10
#      README runs global_batch_size=128 x 16 batches on 2x80GB; here 32 x 64
#      batches (the same 2048 puzzles, 4096 sequences per step) to fit one GPU.
#   4. 200 training steps of EqR and of relax (rate 3) at the real Sudoku config,
#      compiled, no eval: the seconds/step for sizing sudoku_eqr.slurm's --time.
#
# Usage:  sbatch jobs/looped/eqr_sanity.slurm
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=eqr_sanity
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --exclude=idun-07-06
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output_eqr_sanity_%j.txt
#SBATCH --error=output_eqr_sanity_%j.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL

set -uo pipefail
WORK=${USERWORK:-$HOME/work}
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

log() { echo "[$(date -Is)] $*"; }
FAIL=0
check() { local name=$1; shift; log "── ${name}"; if "$@"; then log "   OK ${name}"; else log "   FAILED ${name}"; FAIL=1; fi; }

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "${EQR_DIR}" || { log "FATAL: no ${EQR_DIR}; run looped/eqr/setup_eqr.sh first"; exit 1; }

# The overlay in the EqR checkout must be the one in this repo checkout.
check "overlay up to date" cmp -s "${REPO}/looped/eqr/coupled_eqr.py" models/coupled_eqr.py
check "unit tests" python -m pytest -q tests_coupled/test_coupled_eqr.py
check "adam_atan2 CUDA backend" python -c "
import torch, adam_atan2_backend
from adam_atan2 import AdamATan2
p = torch.nn.Parameter(torch.randn(8, device='cuda'))
o = AdamATan2([p], lr=1e-3); p.sum().backward(); o.step()
print(torch.cuda.get_device_name(), torch.__version__, 'ok')"

check "released checkpoint, lite eval" python evaluate.py \
  eval_yaml=config/eval/sudoku_lite_noise05.yaml \
  checkpoint=downloaded_checkpoints/sudoku-extreme/eqr.pth \
  global_batch_size=32 max_eval_steps=64 suffix=sanity_lite
# evaluate.py writes to <dir of the checkpoint's dir>/eval_preds/step_<S>_<suffix>_<overrides>/
J=$(ls -t downloaded_checkpoints/eval_preds/*sanity_lite*/eval_metrics_step_*.json 2>/dev/null | head -1)
[[ -n "${J}" ]] && python - "${J}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
for k in ("convergence_top_k/exact_accuracy", "majority_vote/exact_accuracy", "different_init/any_correct",
          "convergence_top_k/cumulative_exact_acc_top1"):
    v = m.get(k, m.get("all/" + k))
    print(f"{k:45s} {100 * v if v is not None and v <= 1 else v}")
print("README (5 seeds): 98.60 ± 0.04 / 98.67 ± 0.05 / 99.20 ± 0.10 / 99.19 ± 0.12")
PY

# Throughput at the real config (compiled; eval and checkpoints pushed past the end).
OUT=${WORK}/eqr_runs/sanity_${SLURM_JOB_ID:-local}
for ARM in "eqr" "relax3 arch.name=coupled_eqr@CoupledEqRModel ++arch.coupling=relax ++arch.rate=3.0"; do
  set -- ${ARM}; NAME=$1; shift
  log "── 200-step timing: ${NAME}"
  START=${SECONDS}
  OUTPUT_ROOT=${OUT} python pretrain.py --config-name train/eqr_sudoku ++max_steps=200 \
    eval_interval_steps=100000 checkpoint_interval_steps=100000 ++wandb_mode=disabled ++run_name=${NAME} "$@" \
    2>&1 | tr '\r' '\n' | grep -E "Model parameters|coupled_eqr|Error|Traceback|200/200" | tail -4
  log "   ${NAME}: 200 steps incl. compile in $(( SECONDS - START ))s"
done
nvidia-smi --query-gpu=memory.used --format=csv

(( FAIL == 0 )) && log "ALL CHECKS PASSED" || { log "SOME CHECKS FAILED"; exit 1; }
