#!/bin/bash
# Fetch the best checkpoints of a cluster run dir to this machine.
# RUN THIS LOCALLY (on your Mac), not while ssh'd into the cluster.
#
# Each run lands in  <OUTDIR>/<run>/best_<arch>.pt  on the cluster; this pulls
# every best_*.pt while PRESERVING the per-run subdir structure (important — the
# two reversible runs both save "best_reversible.pt", so a flat copy would clash).
#
# Usage:
#   ./fetch_checkpoints.sh                      # defaults to cmp_24693039
#   ./fetch_checkpoints.sh scale_24700001       # another OUTDIR
#   ./fetch_checkpoints.sh cmp_24693039 --all   # also grab metrics.csv / loss.png / samples.txt / plots
set -euo pipefail

# ── edit these to match your cluster ─────────────────────────────────────────
USER_HOST="haakno@idun-login1.hpc.ntnu.no"
REMOTE_REPO="/cluster/home/haakno/research/tiny_stories/StableTransformerTinyStories"
LOCAL_DEST="./fetched"                  # local folder to download into
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR="${1:-cmp_24693039}"
MODE="${2:-best}"                       # "best" (default) or "--all"

SRC="${USER_HOST}:${REMOTE_REPO}/${OUTDIR}/"
DST="${LOCAL_DEST}/${OUTDIR}/"
mkdir -p "${DST}"

echo "Fetching from ${SRC}"
echo "          to ${DST}"

if [[ "${MODE}" == "--all" ]]; then
  # everything except the (large) optimizer-laden checkpoints? no — grab all .pt + logs/plots
  rsync -avhP \
    --include='*/' \
    --include='best_*.pt' --include='final_*.pt' \
    --include='metrics.csv' --include='loss.png' --include='samples.txt' \
    --include='compare_*.png' --include='*.png' \
    --exclude='*' \
    "${SRC}" "${DST}"
else
  # only the best checkpoints, structure preserved
  rsync -avhP \
    --include='*/' \
    --include='best_*.pt' \
    --exclude='*' \
    "${SRC}" "${DST}"
fi

# Also fetch the tokenizer(s) into ./data/ — 10K-vocab models need their
# tokenizer.json to encode/decode prompts (analysis & sampling). Harmless if the
# run used GPT-2 (no such file exists; rsync just copies nothing).
mkdir -p data
echo
echo "Fetching tokenizer(s) from ${REMOTE_REPO}/data/ -> ./data/"
rsync -avhP \
  --include='*_tokenizer.json' --exclude='*' \
  "${USER_HOST}:${REMOTE_REPO}/data/" "data/" || true

echo
echo "Done. Downloaded best checkpoints:"
find "${DST}" -name 'best_*.pt' -print 2>/dev/null || true
echo "Tokenizers in ./data/:"
ls -1 data/*_tokenizer.json 2>/dev/null || echo "  (none — GPT-2 run, use the default tokenizer)"
