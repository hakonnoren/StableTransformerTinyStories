#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_preprocess_10k.slurm — build the 10K-vocab TinyStories data for the scale
# sweep (job_scale.slurm). CPU-only: trains a 10K byte-level BPE and encodes
# train/val to uint16 bins + tokenizer.json under data/.
#
# !!! INTERNET: this downloads roneneldan/TinyStories from HuggingFace. IDUN
# compute nodes usually have NO internet. Two options:
#   (a) run on a node/partition with internet, OR
#   (b) populate the HF cache once on the login node, then this job reads cache:
#         python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories')"
#       and submit this job with HF_DATASETS_OFFLINE=1 (already exported below).
#
# Usage:    sbatch job_preprocess_10k.slurm
# Output:   data/tinystories10k_{train,val}.bin , data/tinystories10k_tokenizer.json
#           (note the printed <|endoftext|> id + real vocab size -> set in job_scale.slurm)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=preprocess_ts10k
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=CPUQ                  # CPU-only; adjust to your cluster's CPU partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=output_preprocess_10k.txt
#SBATCH --error=output_preprocess_10k.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

# tokenizers parallelism uses these; keep generous on a CPU node.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=true
# If the dataset is already cached on the login node, run fully offline:
# export HF_DATASETS_OFFLINE=1

log() { echo "[$(date -Is)] $*"; }
log "host=$(hostname) pwd=$(pwd) cpus=${SLURM_CPUS_PER_TASK:-NA}"

# Build the 10K BPE + encode train/val. Tune --train_tokenizer_docs / --max_docs_* for speed.
python3 -u preprocess_tinystories_10k.py \
  --out_dir data \
  --prefix tinystories10k \
  --vocab_size 10000

log "done. Set VOCAB / EOT / TOKJSON in job_scale.slurm from the printout above."
ls -lh data/tinystories10k_* 2>/dev/null
