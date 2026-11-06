#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup for the Sudoku-Extreme / EqR experiments on IDUN.
# Run on the LOGIN node (needs network: GitHub, PyPI, Hugging Face; no GPU):
#
#   bash looped/eqr/setup_eqr.sh
#
# then submit jobs/looped/eqr_sanity.slurm, then jobs/looped/sudoku_eqr.slurm.
#
# What it does (each step is skipped if already done):
#   1. clone github.com/locuslab/EqR at the pinned commit into $EQR_DIR
#   2. install the overlay (looped/eqr/install_overlay.sh)
#   3. create a conda env at $EQR_ENV with torch + EqR's requirements
#      (minus flash_attn_3: the Sudoku model mixes tokens with an MLP, no attention;
#       plus exceptiongroup, which EqR imports but does not list)
#   4. build adam-atan2 with its CUDA backend (EqR's optimiser; training refuses to
#      run without it). Needs nvcc: loads $CUDA_MODULE and builds for A100 + H100.
#   5. download EqR's data (locuslab/EqR-data) and the released Sudoku checkpoint
#      (locuslab/EqR-model) into $EQR_DIR, with the HF cache under $USERWORK
#   6. run the overlay unit tests on CPU
#
# Overridable: EQR_DIR, EQR_ENV, CUDA_MODULE, TORCH_VERSION, TORCH_INDEX, HF_HOME
# If `module load ${CUDA_MODULE}` fails, list the options with `module spider CUDA`
# and rerun with CUDA_MODULE=<one of them> (keep the major version = torch's, 12).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EQR_COMMIT=134c7541711125860954c78463023c583b2ef21b
WORK=${USERWORK:-$HOME/work}
EQR_DIR=${EQR_DIR:-${WORK}/EqR}
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
CUDA_MODULE=${CUDA_MODULE:-CUDA/12.4.0}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
export HF_HOME=${HF_HOME:-${WORK}/hf_cache}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

log() { echo "[$(date -Is)] $*"; }

# 1. EqR checkout ---------------------------------------------------------------
if [[ ! -d "${EQR_DIR}/.git" ]]; then
  log "cloning EqR into ${EQR_DIR}"
  git clone https://github.com/locuslab/EqR.git "${EQR_DIR}"
fi
cd "${EQR_DIR}"
if [[ "$(git rev-parse HEAD)" != "${EQR_COMMIT}" ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    log "FATAL: ${EQR_DIR} has local changes and is not at ${EQR_COMMIT}; inspect it by hand"
    exit 1
  fi
  git fetch --quiet origin
  git checkout --quiet "${EQR_COMMIT}"
fi
log "EqR at $(git rev-parse --short HEAD)"

# 2. overlay ---------------------------------------------------------------------
bash "${REPO}/looped/eqr/install_overlay.sh" "${EQR_DIR}"

# 3. environment -----------------------------------------------------------------
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
if [[ ! -d "${EQR_ENV}" ]]; then
  log "creating conda env ${EQR_ENV}"
  conda create -y -q -p "${EQR_ENV}" python=3.11
fi
conda activate "${EQR_ENV}"
if ! python -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}')" 2>/dev/null; then
  pip install -q "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
fi
pip install -q numpy pydantic omegaconf hydra-core wandb coolname psutil tqdm pyyaml argdantic colorama \
  huggingface_hub exceptiongroup matplotlib plotly pandas scipy scikit-learn pytest
log "torch $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"

# 4. adam-atan2 with CUDA backend ----------------------------------------------
if python -c "import adam_atan2_backend" 2>/dev/null; then
  log "adam_atan2_backend already built"
else
  module load "${CUDA_MODULE}"
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export TORCH_CUDA_ARCH_LIST="8.0;9.0"          # A100, H100 (build on a GPU-less login node)
  log "building adam-atan2 against ${CUDA_HOME} for ${TORCH_CUDA_ARCH_LIST}"
  # --no-deps: do not let pip swap the cu124 torch for a PyPI one
  pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps adam-atan2==0.0.3
  python -c "from adam_atan2 import AdamATan2; import adam_atan2_backend; print('adam_atan2_backend:', adam_atan2_backend.__file__)"
fi

# 5. data + released checkpoint -------------------------------------------------
mkdir -p "${HF_HOME}"
if [[ ! -f data/sudoku-extreme-1k-aug-1000/test/dataset.json ]]; then
  log "downloading locuslab/EqR-data"
  bash scripts/download_artifacts.sh
fi
if [[ ! -f downloaded_checkpoints/sudoku-extreme/eqr.pth ]]; then
  log "downloading the released Sudoku checkpoint"
  bash scripts/download_artifacts.sh --ckpts-only --ckpt-pattern "sudoku-extreme/*.pth"
fi
python - <<'PY'
import json
for split in ("train", "test"):
    m = json.load(open(f"data/sudoku-extreme-1k-aug-1000/{split}/dataset.json"))
    print(split, {k: m[k] for k in ("total_groups", "total_samples", "seq_len", "vocab_size")})
PY

# 6. overlay tests (CPU) --------------------------------------------------------
python -m pytest -q tests_coupled/test_coupled_eqr.py
log "setup done: EQR_DIR=${EQR_DIR} EQR_ENV=${EQR_ENV}. Next: sbatch jobs/looped/eqr_sanity.slurm"
