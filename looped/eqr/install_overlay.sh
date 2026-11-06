#!/usr/bin/env bash
# Install the coupled-EqR overlay into an EqR checkout (idempotent):
#   models/coupled_eqr.py             <- looped/eqr/coupled_eqr.py
#   tests_coupled/test_coupled_eqr.py <- looped/eqr/test_coupled_eqr.py
#   pretrain.py                       <- + pretrain_rate_groups.patch (no weight decay on learned rates)
#                                        + pretrain_wandb_resume.patch (resume works with wandb >= 0.31)
#
# Usage: bash looped/eqr/install_overlay.sh <EqR checkout>
set -euo pipefail
EQR_DIR=${1:?usage: install_overlay.sh <EqR checkout>}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cp "${HERE}/coupled_eqr.py" "${EQR_DIR}/models/coupled_eqr.py"
mkdir -p "${EQR_DIR}/tests_coupled"
cp "${HERE}/test_coupled_eqr.py" "${EQR_DIR}/tests_coupled/test_coupled_eqr.py"

cd "${EQR_DIR}"
for P in pretrain_rate_groups.patch pretrain_wandb_resume.patch; do   # in this order
  if git apply --reverse --check "${HERE}/${P}" 2>/dev/null; then
    echo "${P} already applied"
  else
    git apply "${HERE}/${P}"
    echo "applied ${P}"
  fi
done
echo "overlay installed in ${EQR_DIR} (EqR $(git rev-parse --short HEAD))"
