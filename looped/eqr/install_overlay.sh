#!/usr/bin/env bash
# Install the coupled-EqR overlay into an EqR checkout (idempotent):
#   models/coupled_eqr.py             <- looped/eqr/coupled_eqr.py
#   tests_coupled/test_coupled_eqr.py <- looped/eqr/test_coupled_eqr.py
#   pretrain.py                       <- + pretrain_rate_groups.patch (no weight decay on learned rates)
#
# Usage: bash looped/eqr/install_overlay.sh <EqR checkout>
set -euo pipefail
EQR_DIR=${1:?usage: install_overlay.sh <EqR checkout>}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cp "${HERE}/coupled_eqr.py" "${EQR_DIR}/models/coupled_eqr.py"
mkdir -p "${EQR_DIR}/tests_coupled"
cp "${HERE}/test_coupled_eqr.py" "${EQR_DIR}/tests_coupled/test_coupled_eqr.py"

cd "${EQR_DIR}"
if git apply --reverse --check "${HERE}/pretrain_rate_groups.patch" 2>/dev/null; then
  echo "pretrain_rate_groups.patch already applied"
else
  git apply "${HERE}/pretrain_rate_groups.patch"
  echo "applied pretrain_rate_groups.patch"
fi
echo "overlay installed in ${EQR_DIR} (EqR $(git rev-parse --short HEAD))"
