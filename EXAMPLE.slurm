#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_a100.slurm — IDUN SLURM array job for stable-transformer LRA experiments.
#
# Submits 6 independent GPU jobs (one per experiment) that run in parallel.
#
# Usage (from repo root on IDUN login node):
#   sbatch submit_a100.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/slurm_<JOBID>_<ARRAYID>.out
#
# Rerun a single failed task (e.g. task 3):
#   sbatch --array=3 submit_a100.slurm
#
# After all tasks finish:
#   python scripts/plot_results.py runs/a100/ --save_dir figs/a100/
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=stable-lra
#SBATCH --account=share-ie-+        # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                 # 12 h per task; GPUQ max is 14 days
#SBATCH --array=0-5                    # 14 tasks: 2 tasks × 7 variants
#SBATCH --output=logs/slurm_%A_%a.out  # %A = job ID, %a = array index

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Repo and output roots ─────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$REPO/runs/a100"
mkdir -p "$RUNS" "$REPO/logs"

# ── Conda environment ─────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate lra_torch

# ── Scale parameters (shared across all tasks) ───────────────────────────────
N_EMBD=256
N_EMBDX2=512
N_LAYER=12
N_HEAD=8
BATCH=8
GRAD_ACCUM=16        # effective batch = 128
MAX_ITERS=20000
NUM_WORKERS=4

# ── Experiment table (indexed by SLURM_ARRAY_TASK_ID) ────────────────────────
#
#   ID  name                   task      extra flags
#    0  listops_vanilla        listops   --n_embd $N_EMBD
#    1  listops_rev_vpres      listops   --n_embd $N_EMBDX2 --reversible --volume_pres
#    2  listops_rev_lam_00     listops   --n_embd $N_EMBDX2 --reversible --lambd 0.0 --epsilon 1.0
#    3  text_vanilla           text      --n_embd $N_EMBD
#    4  text_rev_vpres         text      --n_embd $N_EMBDX2 --reversible --volume_pres
#    5  text_rev_lam_00        text      --n_embd $N_EMBDX2 --reversible --lambd 0.0 --epsilon 1.0

NAMES=(
    listops_vanilla
    listops_rev_vpres
    listops_rev_lam_00
    text_vanilla
    text_rev_vpres
    text_rev_lam_00
)

TASKS=(
    listops listops listops 
    text    text    text   
)

MAX_LENGTHS=(
    2048 2048 2048 
    1024 1024 1024 
)

# Extra flags per experiment (empty string = no extra flags)
EXTRA=(
    "--n_embd $N_EMBD"
    "--n_embd $N_EMBDX2 --reversible --volume_pres"
    "--n_embd $N_EMBDX2 --reversible --lambd 0.0 --epsilon 1.0"
    "--n_embd $N_EMBD"
    "--n_embd $N_EMBDX2 --reversible --volume_pres"
    "--n_embd $N_EMBDX2 --reversible --lambd 0.0 --epsilon 1.0"
)

# ── Resolve this task's parameters ───────────────────────────────────────────
ID=${SLURM_ARRAY_TASK_ID}
NAME="${NAMES[$ID]}"
TASK="${TASKS[$ID]}"
MAX_LEN="${MAX_LENGTHS[$ID]}"
OUT="$RUNS/$NAME"

echo "════════════════════════════════════════════════════════════════"
echo "  SLURM job   : $SLURM_JOB_ID  (array task $ID)"
echo "  Experiment  : $NAME"
echo "  Task        : $TASK  seq_len=$MAX_LEN"
echo "  Output dir  : $OUT"
echo "  Node        : $SLURMD_NODENAME"
echo "════════════════════════════════════════════════════════════════"

# Skip if already completed
if [[ -f "$OUT/summary.json" ]]; then
    echo "  [skip] summary.json exists — delete it to re-run"
    exit 0
fi

# Warmup scales with training length
if [[ "$TASK" == "listops" ]]; then
    WARMUP=2000
    TRAIN_ARGS="--max_train_examples 96000"
else
    WARMUP=2000
    TRAIN_ARGS=""   # text uses all 25k IMDb train examples
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
python "$REPO/lra_train.py" \
    --task "$TASK" \
    --data_dir "$REPO/data" \
    --n_layer  $N_LAYER  --n_head $N_HEAD   \
    --max_length $MAX_LEN \
    --dropout 0.1  --pooling last \
    --batch_size $BATCH  --max_iters $MAX_ITERS \
    --lr 3e-4  --min_lr 3e-5  --warmup_iters $WARMUP \
    --weight_decay 0.1  --grad_clip 1.0 \
    --eval_interval 1000  --eval_batches 100 \
    --amp  --num_workers $NUM_WORKERS  --grad_accum_steps $GRAD_ACCUM \
    --out_dir "$OUT" \
    --seed 42 \
    $TRAIN_ARGS \
    ${EXTRA[$ID]}

echo ""
echo "  Done: $NAME"
