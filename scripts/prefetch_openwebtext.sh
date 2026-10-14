#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# prefetch_openwebtext.sh — download OpenWebText into the HuggingFace cache.
#
# RUN THIS ON THE LOGIN NODE, NOT VIA SBATCH.
# IDUN compute nodes have no outbound network (see job_preprocess_10k.slurm:8),
# so job_owt_preprocess.slurm runs with HF_DATASETS_OFFLINE=1 and can only read a
# cache that already exists. This script is what creates it.
#
# It downloads ~40GB and can take a long while, so run it under tmux/screen:
#     tmux new -s owt
#     ./prefetch_openwebtext.sh
#     # detach with Ctrl-b d ; reattach later with: tmux attach -t owt
#
# It is resumable: HuggingFace skips shards already in the cache, so if it dies
# partway just run it again.
#
# Usage:  ./prefetch_openwebtext.sh [target_dir]
#         (default target: $USERWORK/hf_cache, falling back to $HOME/work/hf_cache)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:-${HF_HOME:-${USERWORK:-$HOME/work}/hf_cache}}"
export HF_HOME="$TARGET"
mkdir -p "$HF_HOME"

echo "=============================================================="
echo " HF_HOME : $HF_HOME"
echo " disk    : $(df -h "$HF_HOME" | tail -1 | awk '{print $4" free of "$2}')"
echo "=============================================================="
echo
echo "OpenWebText needs roughly 64GB here (24GB download + 40GB extracted arrow)."
echo "The encoded bins written later by job_owt_preprocess.slurm need a further"
echo "~18GB under data/ in the repo. Ctrl-C now if either is short."
echo
sleep 5

python3 -u - <<'PY'
import os, sys
print(f"HF_HOME = {os.environ.get('HF_HOME')}")
try:
    import datasets, huggingface_hub
    print(f"datasets {datasets.__version__} | huggingface_hub {huggingface_hub.__version__}")
except ImportError as e:
    sys.exit(f"FATAL: {e}\nActivate the environment first (e.g. conda activate lra_torch).")

from datasets import load_dataset

# Newer `datasets` refuses legacy loading scripts unless trust_remote_code is set;
# older ones do not accept the kwarg at all. Try plain first, then with the flag.
try:
    ds = load_dataset("Skylion007/openwebtext", split="train")
except TypeError:
    ds = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
except ValueError as e:
    if "trust_remote_code" not in str(e):
        raise
    print("retrying with trust_remote_code=True ...")
    ds = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)

n = len(ds)
print(f"\nOK: {n:,} documents cached.")
# The paper's corpus is 8,013,769 docs / 9.05B tokens (App. A.2). A materially
# different count here means a different snapshot, and the absolute losses will
# not line up with Table 3 -- worth knowing now rather than after 30k steps.
if abs(n - 8_013_769) > 50_000:
    print(f"NOTE: paper reports 8,013,769 docs; this snapshot has {n:,}.")
PY

echo
echo "=============================================================="
echo " Done. Submit the encoder with the SAME cache location:"
echo
echo "   export HF_HOME=$HF_HOME"
echo "   sbatch job_owt_preprocess.slurm"
echo
echo " (job_owt_preprocess.slurm defaults HF_HOME to this same path, so if you"
echo "  used the default target you do not need to export anything.)"
echo "=============================================================="
