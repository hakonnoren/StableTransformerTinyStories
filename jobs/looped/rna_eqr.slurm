#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# rna_eqr.slurm — RNA inverse folding in the Equilibrium Reasoners (EqR) setting (looped/eqr/rna/README.md):
# a target secondary structure in (dot-bracket, up to 128 nt), a sequence out, scored by refolding it with
# ViennaRNA 2, since a target has many solutions. EqR's code at its pinned commit with the overlay
# (looped/eqr/install_overlay.sh): the attention block with 1D RoPE (config/train/eqr_rna.yaml; pairing
# partners change per target, so not the Sudoku recipe's token-mixing MLP), the refold evaluator
# (evaluators/rna_refold.py -> rna/solved, rna/bp_dist, ...) and the SDPA fallback for attention without
# FlashAttention (layers_sdpa_fallback.patch).
#
#   arm            update (definitions as in sudoku_eqr.slurm)
#   eqr            EqR, original
#   twinL_leak2    twin inner loop, relax rate 3, delta = 2, no injection into H, EqR's order (best
#                  test-time scaling on Sudoku at 20k)
#   relax3_leak1   relax, rate 3, common-mode damping delta = 1
#   kick_pos       kick, learned positive rates from 1.0 (floor 0.05)
#
# Once, on the login node (EqR env active):
#   git pull && bash looped/eqr/install_overlay.sh /cluster/work/$USER/EqR
#   pip install ViennaRNA==2.8.0
#   build the data as in looped/eqr/rna/README.md (rna-L128-10k, rna-L128-100k, rna-eterna100v2-L128)
# Pilot (EqR and the twin on 100k targets, seed 0):
#   sbatch --array=0-1 jobs/looped/rna_eqr.slurm
#   sbatch --export=ALL,RNA_DATA=rna-L128-10k --array=0-1 jobs/looped/rna_eqr.slurm     # 10k targets
# Run root eqr_runs/<RNA_DATA>_<JOB>, run ids <arm>-s<seed>-<L128-100k>-<JOB>, W&B project looped-eqr-rna.
# Resume / extend as in sudoku_eqr.slurm: with checkpoints but none at MAX_STEPS, training resumes from the
# latest one; evaluations already done are skipped, so a task that hit its time limit can be resubmitted
# with the same JOB_TAG.
#
# Evaluation of the final EMA checkpoint (evaluate.py, breadth 1, noise EVAL_NOISE = the training noise):
#   test_D{16,64,256}    the whole test set (5k targets) at 16 / 64 / 256 iterations
#   eterna_D{16,64,256}  Eterna100-V2, the 66 puzzles of length <= 128 (RNA_ETERNA; skipped if absent)
# Predictions saved; one row per target in eval_preds/*/evaluator_RNARefold/rna_refold.csv. EqR's own
# accuracy / exact_accuracy compare with the training label, one solution among many: read rna/* instead.
# EqR's halting head learns "prediction == label", which rarely holds here, so runs mostly go to the step cap.
# Then looped/eqr/dynamics_eqr.py (fixed points, 2-cycles, spectral radius; its "exact" is label agreement).
# No test-time scaling (eqr_scaling.sh scores by label agreement); breadth with refolding is the next phase.
#
# GPU memory: attention over 144 positions (16 puzzle-embedding slots + 128) at batch 768 is not the Sudoku
# model's footprint and is not measured yet; the 40 GB floor below is Sudoku's.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=rna_eqr
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# Any GPU type; the P100/V100 pools are excluded by node (as in sudoku_eqr.slurm) together with the bad
# A100 node idun-07-06, and the preflight below refuses anything else unfit.
#SBATCH --exclude=idun-07-[01-03,06],idun-06-[08-12],idun-11-[01-17]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=output_rna_eqr_%A_%a.txt
#SBATCH --error=output_rna_eqr_%A_%a.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL

set -o pipefail    # no -u: Lmod's and conda's activation scripts read unset variables
# Work area for the EqR checkout, the conda env, the HF cache and the runs (/cluster/work/$USER on IDUN).
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
# JOB_TAG names the run root and the W&B run ids; reuse an old array job id to redo a task in place.
JOB=${JOB_TAG:-${SLURM_ARRAY_JOB_ID:-local}}
RNA_DATA=${RNA_DATA:-rna-L128-100k}                       # training set (EqR/data/<RNA_DATA>)
RNA_TAG=${RNA_DATA#rna-}                                  # L128-10k | L128-100k
RNA_ETERNA=${RNA_ETERNA:-rna-eterna100v2-L128}            # second test set (skipped if absent)
RUN_ROOT=${RUN_ROOT:-${WORK}/eqr_runs/${RNA_DATA}_${JOB}}
MAX_STEPS=${MAX_STEPS:-20000}
EVAL_INTERVAL=${EVAL_INTERVAL:-2000}                      # in-training eval of the 5k test targets (W&B curves)
EVAL_NOISE=${EVAL_NOISE:-0.01}
read -r -a SEEDS <<< "${SEEDS:-0}"
WANDB_PROJECT=${WANDB_PROJECT:-looped-eqr-rna}
WANDB_ENTITY=${WANDB_ENTITY:-hakon-noren-ntnu}

COUPLED="arch.name=coupled_eqr@CoupledEqRModel arch.short_name=Coupled"
ARMS=(eqr twinL_leak2 relax3_leak1 kick_pos)
ARM_ARGS=(
  ""
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=2.0 ++arch.inject_h=false ++arch.schedule=eqr ++arch.twin_l=true"
  "${COUPLED} ++arch.coupling=relax ++arch.rate=3.0 ++arch.leak=1.0"
  "${COUPLED} ++arch.coupling=kick ++arch.rate=1.0 ++arch.rate_constraint=positive ++arch.rate_min=0.05 ++arch.learn_rates=true"
)
(( ${#ARMS[@]} == ${#ARM_ARGS[@]} )) || { echo "FATAL: ${#ARMS[@]} arm names but ${#ARM_ARGS[@]} argument sets"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID:?submit as an array: sbatch jobs/looped/rna_eqr.slurm}
NS=${#SEEDS[@]}
A=$(( ID / NS )); SEED=${SEEDS[$(( ID % NS ))]}
if (( A >= ${#ARMS[@]} )); then echo "no arm for task ${ID}"; exit 1; fi
ARM=${ARMS[$A]}
NAME="${ARM}_s${SEED}"
RUN_ID="${ARM//_/-}-s${SEED}-${RNA_TAG}-${JOB}"

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OUTPUT_ROOT=${RUN_ROOT}
mkdir -p "${RUN_ROOT}"

log() { echo "[$(date -Is)] $*"; }
log "task=${ID} arm=${ARM} seed=${SEED} run_id=${RUN_ID} host=$(hostname)"
log "EQR_DIR=${EQR_DIR} RUN_ROOT=${RUN_ROOT} RNA_DATA=${RNA_DATA} MAX_STEPS=${MAX_STEPS}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
# Any GPU that can run this env: bf16 + torch 2.6/CUDA 12.4 + adam-atan2 built for sm_80/sm_90, so compute
# capability 8.x or 9.x (as in sudoku_eqr.slurm).
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
# The overlay must match this repo checkout: refuse to train with out-of-date files.
for P in "looped/eqr/coupled_eqr.py models/coupled_eqr.py" \
         "looped/eqr/configs/dataset/rna_design.yaml config/dataset/rna_design.yaml" \
         "looped/eqr/configs/train/eqr_rna.yaml config/train/eqr_rna.yaml" \
         "looped/eqr/rna/rna_refold.py evaluators/rna_refold.py" \
         "looped/eqr/rna/rna_design.py evaluators/rna_design.py"; do
  read -r SRC DST <<< "${P}"
  cmp -s "${REPO}/${SRC}" "${DST}" || {
    log "FATAL: ${EQR_DIR}/${DST} missing or out of date; rerun bash looped/eqr/install_overlay.sh ${EQR_DIR}"
    log "       (not from inside a running array)"; exit 1; }
done
for P in pretrain_rate_groups.patch pretrain_wandb_resume.patch layers_sdpa_fallback.patch; do
  git apply --reverse --check "${REPO}/looped/eqr/${P}" 2>/dev/null || {
    log "FATAL: ${EQR_DIR} lacks ${P}; rerun bash looped/eqr/install_overlay.sh ${EQR_DIR}"; exit 1; }
done
python -c "import RNA" 2>/dev/null || { log "FATAL: no ViennaRNA in ${EQR_ENV}: pip install ViennaRNA==2.8.0"; exit 1; }
[[ -f "data/${RNA_DATA}/train/dataset.json" && -f "data/${RNA_DATA}/test/dataset.json" ]] || {
  log "FATAL: no ${EQR_DIR}/data/${RNA_DATA}; build it as in looped/eqr/rna/README.md"; exit 1; }

ckpt_at() { ls ${RUN_ROOT}/outputs/${RUN_ID}/*/*/checkpoints/step_${1}_*.pth 2>/dev/null | head -1; }
final_ckpt() { ckpt_at "${MAX_STEPS}"; }
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
  WANDB_RUN_ID=${RUN_ID} python pretrain.py --config-name train/eqr_rna \
    ++max_steps=${MAX_STEPS} ++seed=${SEED} eval_interval_steps=${EVAL_INTERVAL} \
    ++project_name=${WANDB_PROJECT} ++entity=${WANDB_ENTITY} ++run_name=${NAME} \
    "++dataset.data_path=data/${RNA_DATA}" ${ARM_ARGS[$A]} "${RESUME_ARGS[@]}" 2>&1 \
    | tr '\r' '\n' | grep -v "it/s\]$\|s/it\]$"
  RC=${PIPESTATUS[0]}
  (( RC != 0 )) && { log "training FAILED (exit ${RC})"; exit "${RC}"; }
fi
CKPT=$(final_ckpt)
[[ -z "${CKPT}" ]] && { log "FATAL: no step_${MAX_STEPS} checkpoint under ${RUN_ROOT}/outputs/${RUN_ID}"; exit 1; }
log "checkpoint: ${CKPT}"
RUN_DIR=$(dirname "$(dirname "${CKPT}")")       # evaluate.py writes eval_preds/ here
EVAL_NL=$(python -c "print(float('${EVAL_NOISE}'))")

# ── evaluate (each step skipped if already done, so a resubmitted task continues where it stopped) ──
rna_metrics() {   # tag: print the refold metrics (evaluators/rna_refold.py) of that eval
  python - "${RUN_DIR}" "${MAX_STEPS}" "$1" <<'EOF'
import glob, json, sys
run_dir, step, tag = sys.argv[1:]
for f in sorted(glob.glob(f"{run_dir}/eval_preds/step_{step}_{tag}_*/eval_metrics_step_*.json")):
    m = json.load(open(f))["metrics"]
    g = lambda k: m.get(f"rna/{k}", float("nan"))
    print(f"  {tag}: solved {g('solved'):.3f}  bp_dist {g('bp_dist'):.3f}  pairs_ok {g('pairs_ok'):.3f}  n {g('n'):.0f}")
EOF
}
run_eval() {   # tag depth [further evaluate.py overrides]
  local tag=$1 depth=$2
  shift 2
  if compgen -G "${RUN_DIR}/eval_preds/step_${MAX_STEPS}_${tag}_*_noise_scale-${EVAL_NL}_*/eval_metrics_step_*.json" > /dev/null; then
    log "eval ${tag}: already done"
  else
    log "eval ${tag}: ${depth} iterations"
    python evaluate.py eval_yaml=config/eval/depth_breadth.yaml checkpoint="${CKPT}" \
      halt_max_steps=${depth} noise_scale=${EVAL_NOISE} different_init=1 global_batch_size=768 \
      suffix=${tag} "save_outputs=[preds,labels]" "$@" 2>&1 | tr '\r' '\n' | grep -E "saved|Error|Traceback|coupled_eqr" | tail -3
  fi
  rna_metrics "${tag}"
}
for D in 16 64 256; do run_eval test_D${D} ${D}; done
if [[ -f "data/${RNA_ETERNA}/test/dataset.json" ]]; then
  for D in 16 64 256; do run_eval eterna_D${D} ${D} "dataset_data_path=data/${RNA_ETERNA}"; done
else
  log "eval eterna: no ${EQR_DIR}/data/${RNA_ETERNA}, skipped (build_rna_dataset.py --eterna-tsv writes it)"
fi

# ── dynamics: learned rates per checkpoint, fixed points / 2-cycles to r=256, spectral radius ──────
if [[ -f "${RUN_DIR}/dynamics.json" ]]; then
  log "dynamics: already done"
else
  log "dynamics (looped/eqr/dynamics_eqr.py)"
  python "${REPO}/looped/eqr/dynamics_eqr.py" --eqr-dir "${EQR_DIR}" --ckpt "${CKPT}" 2>&1 \
    | grep -E "^\[rates\]|^\[dynamics\]|Error|Traceback"
fi

log "task ${ID} (${NAME}) done in $(( SECONDS / 60 )) min. Refold metrics: rna/* in"
log "  ${RUN_DIR}/eval_preds/*/eval_metrics_step_${MAX_STEPS}.json"
