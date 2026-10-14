#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_owt_preprocess.slurm — build the OpenWebText GPT-2 bins that
# job_owt_medium.slurm consumes. CPU-only: encodes the corpus with the GPT-2
# byte-level BPE (tiktoken) to uint16 and writes data/openwebtext_{train,val}.bin.
#
# WHY THIS JOB EXISTS: data/ holds only the TinyStories bins (tinystories* and
# tinystories10k*). There is no OpenWebText on disk, so the training jobs cannot
# start until this finishes -- train.py raises SystemExit at its "Missing dataset
# bins" check.
#
# ── SPLIT: 5% val, NOT the script default ────────────────────────────────────
# process_openwebtext.py defaults to --val_fraction 0.005 (0.5%). The paper
# (App. A.2) uses a 95/5 document split: 7,630,278 train docs / 8.62B tokens and
# 383,491 val docs / 432M tokens out of 8,013,769 docs / 9.05B tokens. We pass
# --val_fraction 0.05 to match. The paper assigns docs to splits by a seeded
# hash of the document index; process_openwebtext.py takes the FIRST 5% of docs
# as val and the rest as train. Same proportions, different membership -- the
# absolute val losses are therefore comparable to the paper only up to that
# difference in which documents landed in val. Arm-vs-arm comparison within this
# repo is unaffected, since all arms read the identical bins.
#
# ── INTERNET: works either way ───────────────────────────────────────────────
# This needs ~40GB from HuggingFace. The job runs in whichever mode fits:
#
#   * Compute node HAS outbound network (reported to be the case on IDUN, though
#     job_preprocess_10k.slurm:8 claims otherwise -- verify with the srun probe
#     below): just submit. It downloads straight into HF_HOME and encodes.
#   * Compute node has NO network: prefetch on the login node first with
#     ./prefetch_openwebtext.sh, then submit -- this job reads the cache it made.
#
# Verify which you have, before spending a queue slot finding out:
#   srun --account=share-ie-imf --partition=GPUQ --time=00:05:00 --pty \
#        bash -c 'curl -sI --max-time 20 https://huggingface.co | head -1'
#
# Set HF_DATASETS_OFFLINE=1 in the environment to force cache-only mode (fails
# fast rather than silently re-downloading if the cache is somehow incomplete).
#
# No trust_remote_code needed: Skylion007/openwebtext is auto-converted to
# Parquet on the Hub (8,013,769 rows -- exactly the paper's count), so it has no
# legacy loading script for `datasets` 4.x to reject. Checked against the
# installed datasets 4.4.1.
#
# ── DISK ─────────────────────────────────────────────────────────────────────
# HF cache after download:  ~64GB (24GB download + 40GB extracted arrow)
# Emitted bins:             ~18.1GB (9.05B tokens x 2 bytes uint16)
# Keep BOTH off $HOME -- HF_HOME is redirected to $USERWORK below. The bins land
# in data/ under the submit dir; check `dusage` before running if data/ is on a
# quota'd filesystem.
#
# ── RUNTIME ──────────────────────────────────────────────────────────────────
# The first version of this job measured ~137 docs/s on a real run, i.e. ~15h for
# the 7.6M train docs. process_openwebtext.py has since been reworked and now
# scales with --cpus-per-task. Two changes, neither of which alters a single
# output byte (verified by hashing both versions' bins on a synthetic corpus):
#
#   1. Documents are read in batches (ds[j:j+2000]["text"]) instead of one Arrow
#      row at a time, and addressed by OFFSET instead of through ds.select().
#      .select() attaches an indices mapping, so every read paid an extra
#      indirection -- cheap on a local disk, expensive on /cluster. Measured 8.8x
#      from batching and a further 1.5x from dropping the mapping.
#   2. Encoding goes through tiktoken's encode_ordinary_batch(num_threads=N),
#      whose Rust core releases the GIL, so these are real parallel threads. No
#      multiprocessing, no temp shards, no extra disk.
#
# ── WHY 16 CPUs AND NOT 32 ───────────────────────────────────────────────────
# Because it is measured, not assumed. The loop is threaded encode + a serial
# flatten/write, so Amdahl sets a hard ceiling. On the benchmark corpus the
# serial part is ~12% of single-threaded runtime (after the np.fromiter change
# above cut it 1.7x), which caps end-to-end at ~8x however many threads you add:
#
#     threads   8      16      32      inf
#     speedup  4.3x   5.7x    6.8x    8.3x
#
# So 16 -> 32 buys ~19% more for 2x the cores, and a 16-core request backfills
# far sooner on a busy CPUQ than a 32-core one. Since the job is short, spending
# less time queued almost certainly beats the 19%. Raise it if CPUQ is idle.
#
# --num_threads defaults to SLURM_CPUS_PER_TASK, so changing --cpus-per-task is
# enough. Estimate is ~1-2h; --time=6h is 3-6x that, kept as headroom because the
# figures above come from a laptop, not a CPUQ node -- and the script has no
# checkpointing, so being killed at the wall means starting over. The job prints
# live docs/s and an ETA every batch, so the real number shows up within a minute.
#
# Usage:   sbatch job_owt_preprocess.slurm
# Output:  data/openwebtext_train.bin , data/openwebtext_val.bin
# Then:    sbatch job_owt_medium.slurm
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=owt_preprocess
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16       # measured sweet spot -- see RUNTIME; also backfills fast
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=output_owt_preprocess.txt
#SBATCH --error=output_owt_preprocess.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

# Keep the HF cache off $HOME (quota) -- see header.
export HF_HOME=${HF_HOME:-${USERWORK:-$HOME/work}/hf_cache}
# NOT forced offline: if the compute node has network this job can download for
# itself. Export HF_DATASETS_OFFLINE=1 before sbatch to require the cache instead.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
# HF `tokenizers` is unused here (we encode with tiktoken); silence its fork warning.
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${HF_HOME}"

log() { echo "[$(date -Is)] $*"; }
log "host=$(hostname) pwd=$(pwd) cpus=${SLURM_CPUS_PER_TASK:-NA}"
log "HF_HOME=${HF_HOME} (HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE})"
log "cache present: $([[ -d "${HF_HOME}/hub" || -d "${HF_HOME}/datasets" ]] && echo yes || echo 'no -- will download (~40GB)')"
log "disk at HF_HOME: $(df -h "${HF_HOME}" | tail -1 | awk '{print $4" free"}')"

# Fail early and legibly rather than 20 minutes into a download that cannot work.
if [[ "${HF_DATASETS_OFFLINE}" != "1" ]]; then
  if ! curl -sI --max-time 25 https://huggingface.co >/dev/null 2>&1; then
    log "FATAL: no route to huggingface.co from $(hostname), and HF_DATASETS_OFFLINE is not set."
    log "       This compute node has no outbound network. Prefetch on the LOGIN node:"
    log "         ./prefetch_openwebtext.sh"
    log "       then resubmit (the cache at ${HF_HOME} will be used automatically)."
    exit 1
  fi
  log "network to huggingface.co: OK"
fi

mkdir -p data

# 95/5 document split to match the paper (see header).
log "encoding OpenWebText -> data/openwebtext_{train,val}.bin (${SLURM_CPUS_PER_TASK:-8} threads)"
python3 -u process_openwebtext.py \
  --out_dir data \
  --val_fraction 0.05 \
  --seed 1337 \
  --num_threads "${SLURM_CPUS_PER_TASK:-8}" \
  --batch_docs 2000
rc=$?

if (( rc != 0 )); then
  log "FAILED: process_openwebtext.py exit ${rc}"
  exit $rc
fi

log "done. Expect ~8.6B train tokens and ~0.43B val tokens in the printout above."
ls -lh data/openwebtext_*.bin 2>/dev/null
df -h data 2>/dev/null | tail -n 1
