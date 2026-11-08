#!/usr/bin/env bash
# Install the coupled-EqR overlay into an EqR checkout (idempotent):
#   models/coupled_eqr.py             <- looped/eqr/coupled_eqr.py
#   tests_coupled/test_coupled_eqr.py <- looped/eqr/test_coupled_eqr.py
#   pretrain.py                       <- + pretrain_rate_groups.patch (no weight decay on learned rates)
#                                        + pretrain_wandb_resume.patch (resume works with wandb >= 0.31)
#   models/layers.py                  <- + layers_sdpa_fallback.patch (attention without FlashAttention:
#                                        PyTorch SDPA; only the attention block, i.e. RNA, uses it)
#   config/{dataset,train}/*.yaml     <- looped/eqr/configs/ (Skyscrapers-7: dataset/skyscrapers.yaml,
#                                        train/eqr_skyscrapers.yaml; build the data with
#                                        looped/eqr/build_skyscrapers_dataset.py --out-root <EqR>/data.
#                                        RNA: dataset/rna_design.yaml, train/eqr_rna.yaml)
#   evaluators/rna_refold.py, evaluators/rna_design.py <- looped/eqr/rna/ (RNA evaluator: refolds predictions;
#                                        needs pip install ViennaRNA in the env)
#   tests_coupled/test_rna_eqr.py     <- looped/eqr/rna/test_rna_eqr.py
#
# Usage: bash looped/eqr/install_overlay.sh <EqR checkout>
set -euo pipefail
EQR_DIR=${1:?usage: install_overlay.sh <EqR checkout>}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cp "${HERE}/coupled_eqr.py" "${EQR_DIR}/models/coupled_eqr.py"
mkdir -p "${EQR_DIR}/tests_coupled"
cp "${HERE}/test_coupled_eqr.py" "${EQR_DIR}/tests_coupled/test_coupled_eqr.py"
cp "${HERE}/rna/test_rna_eqr.py" "${EQR_DIR}/tests_coupled/test_rna_eqr.py"
cp "${HERE}/rna/rna_refold.py" "${HERE}/rna/rna_design.py" "${EQR_DIR}/evaluators/"
for D in dataset train; do
  for F in "${HERE}/configs/${D}"/*.yaml; do cp "${F}" "${EQR_DIR}/config/${D}/"; done
done

cd "${EQR_DIR}"
for P in pretrain_rate_groups.patch pretrain_wandb_resume.patch layers_sdpa_fallback.patch; do   # in this order
  if git apply --reverse --check "${HERE}/${P}" 2>/dev/null; then
    echo "${P} already applied"
  else
    git apply "${HERE}/${P}"
    echo "applied ${P}"
  fi
done
echo "overlay installed in ${EQR_DIR} (EqR $(git rev-parse --short HEAD))"
