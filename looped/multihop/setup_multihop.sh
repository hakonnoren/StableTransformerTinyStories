#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_multihop.sh — once, on an IDUN login node (needs internet): the authors' code, data and released
# checkpoints of Kohli et al. (2604.07822, "Loop, Think & Generalize"), and a Python env for looped/multihop.
#
#   bash looped/multihop/setup_multihop.sh            # from this repo's root
#
# Creates
#   ${LTG_DIR}  (default /cluster/work/$USER/LTG)      OSU-NLP-Group/Loop-Think-Generalize at a pinned commit, with
#       data/multi_hop/{vocab,train,test,test_small}.json        (the authors' Google Drive folder, via gdown)
#       checkpoints/multi_hop/final_models/{r_dyn,r8}/...        (the released dynamic- and R=8 models)
#   ${LTG_ENV}  (default /cluster/work/$USER/conda_envs/ltg_venv) a venv on top of the EqR conda env (torch + CUDA)
#       with transformers 4.46.3 (the GPT-2 block API the authors' code uses), gdown, pytest
# Then runs looped/multihop/test_multihop.py there. Safe to rerun: existing files are kept.
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail
if [[ -n "${USERWORK:-}" ]]; then WORK=${USERWORK}
elif [[ -d /cluster/work/${USER} ]]; then WORK=/cluster/work/${USER}
else WORK=${HOME}/work; fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LTG_DIR=${LTG_DIR:-${WORK}/LTG}
LTG_COMMIT=bb22b192977b1b136fde985c8e7a2392d172d97a     # main on 2026-09-03, the version this port was checked against
EQR_ENV=${EQR_ENV:-${WORK}/conda_envs/eqr}
LTG_ENV=${LTG_ENV:-${WORK}/conda_envs/ltg_venv}

module load Anaconda3/2024.02-1 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${EQR_ENV}"
if [[ ! -x "${LTG_ENV}/bin/python" ]]; then
  python -m venv --system-site-packages "${LTG_ENV}"
fi
"${LTG_ENV}/bin/python" -m pip install -q "transformers==4.46.3" gdown pytest
"${LTG_ENV}/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

if [[ ! -d "${LTG_DIR}/.git" ]]; then
  git clone https://github.com/OSU-NLP-Group/Loop-Think-Generalize.git "${LTG_DIR}"
fi
git -C "${LTG_DIR}" fetch -q origin && git -C "${LTG_DIR}" checkout -q "${LTG_COMMIT}"
echo "Loop-Think-Generalize at $(git -C "${LTG_DIR}" rev-parse HEAD)"

# file ids in the authors' Drive folder (README: drive.google.com/drive/folders/1AQvp8erdtRRn9_ix7E1ssmyWarwFNLVU)
get() {   # id path
  if [[ -s "${LTG_DIR}/$2" ]]; then echo "have $2"; return; fi
  mkdir -p "$(dirname "${LTG_DIR}/$2")"
  "${LTG_ENV}/bin/python" -m gdown -q "$1" -O "${LTG_DIR}/$2"
  echo "got  $2 ($(du -h "${LTG_DIR}/$2" | cut -f1))"
}
get 1-ChSYN2b8EjJahxdD6a9LF1NJzV35PdV data/multi_hop/vocab.json
get 18AZXc1dFWHy3iEoSpbXqMybOqmxF75nb data/multi_hop/train.json
get 1CuoajnhvCXT5iEstqaKPaR5IeMi4JcC0 data/multi_hop/test.json
get 12TSeReTnXBSet5XzCD9wSs6mO8Us9i2G data/multi_hop/test_small.json
get 17AnqMH4Zzx99RUMKLxz8FQjiMLkUvpez checkpoints/multi_hop/final_models/r_dyn/checkpoint_epoch_5388.pt
get 1dywaxNsrVi7e6_f2N-R_2EBLfUtFWtFu checkpoints/multi_hop/final_models/r8/checkpoint_epoch_1259.pt

cd "${REPO}"
LTG_DIR="${LTG_DIR}" PYTHONPATH="${REPO}" "${LTG_ENV}/bin/python" -m pytest -q -p no:cacheprovider looped/multihop/test_multihop.py
echo "setup done: LTG_DIR=${LTG_DIR} LTG_ENV=${LTG_ENV}"
