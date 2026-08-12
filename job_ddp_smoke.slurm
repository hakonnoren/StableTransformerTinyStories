#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_ddp_smoke.slurm — 15-minute multi-GPU pre-flight. RUN THIS BEFORE
# job_owt_medium_ddp.slurm.
#
# The DDP implementation was verified on CPU/gloo (world sizes 1/2/4: identical
# best_val, single-process results bit-identical to pre-DDP HEAD). What that
# CANNOT test is the transport: NCCL init, GPU peer-to-peer, and the one failure
# mode that matters here -- a backward that never completes because a parameter's
# gradient hook did not fire. On CPU that path is exercised, but on real hardware
# a hang shows up as a job that burns its whole walltime doing nothing.
#
# So: 2 GPUs, a tiny model, 20 steps, all four arms in sequence, on TinyStories
# data you already have. If this prints SMOKE PASS you can submit the real job.
#
# It does NOT touch W&B (no --wandb), so it will not pollute your projects.
#
# Usage:  sbatch job_ddp_smoke.slurm
# Watch:  tail -f output_ddp_smoke_<jobid>.txt
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp_smoke
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2                 # A100 is plentiful and bf16-capable.
                                          # To validate the exact target hardware
                                          # instead, use gpu:h100:2.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=output_ddp_smoke_%j.txt
#SBATCH --error=output_ddp_smoke_%j.err

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=FAIL


module load Anaconda3/2024.02-1
conda activate lra_torch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1         # fail with an error instead of wedging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NPROC=2
DATASET=tinystories
OUTDIR=ddp_smoke_${SLURM_JOB_ID}
mkdir -p "${OUTDIR}"

log() { echo "[$(date -Is)] $*"; }

log "host=$(hostname)"
log "torch: $(python3 -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())' 2>&1)"
log "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

NGPU=$(nvidia-smi --list-gpus | wc -l)
if (( NGPU != NPROC )); then
  log "FATAL: NPROC=${NPROC} but ${NGPU} GPU(s) visible"; exit 1
fi
for f in data/${DATASET}_train.bin data/${DATASET}_val.bin; do
  if [[ ! -s "$f" ]]; then
    log "FATAL: missing ${f} -- point DATASET at bins you actually have"
    ls -lh data/*.bin 2>/dev/null || true
    exit 1
  fi
done

# Deliberately tiny: this measures "does it complete", not "does it learn".
# grad_accum(4) and eval_batches(4) both divide by NPROC=2, as train.py requires.
SMOKE=(
  --data_dir data --dataset "${DATASET}"
  --n_layer 4 --n_head 4 --n_embd 256 --block_size 256
  --batch_size 4 --grad_accum_steps 4 --eval_batches 4
  --max_steps 20 --warmup_steps 5 --eval_interval 10 --log_interval 5
  --sample_interval 0 --device cuda --amp_dtype bf16 --attn_impl sdpa
  --optimizer muon --muon_lr 0.006 --peak_lr 6e-4
  --cheap_metrics --track_recon_drift
)
NARROW=(--rev_embed narrow --rev_lift dup --rev_readout sum)
REV=(--mem_efficient --mem_efficient_segment 4)

declare -a NAMES=(baseline yurii_lt vpm_segL vf_segL)
declare -a ARMS=(
  "--arch baseline"
  "--arch yurii_lt"
  "--arch reversible --rev_regime vpm_scaling ${NARROW[*]} ${REV[*]}"
  "--arch reversible --rev_regime vf_scaling  ${NARROW[*]} ${REV[*]}"
)

# ── Is the DDP code even here? ───────────────────────────────────────────────
# Checked explicitly because the failure is silent: an older train.py has no
# torchrun handling at all, so `torchrun --nproc_per_node=2 train.py` happily
# launches TWO INDEPENDENT single-GPU runs. Every arm then "passes" while
# testing nothing, both processes write the same checkpoint path, and the losses
# look identical (same seed, same unsharded data) which reads as success.
if ! grep -q "^import ddp_utils" train.py; then
  log "FATAL: train.py does not import ddp_utils -- this checkout predates the DDP"
  log "       work. Sync train.py, data.py, ddp_utils.py and revformer/rev_backprop.py"
  log "       before running this test."
  exit 1
fi
python3 -c "import ddp_utils, sys; sys.stderr.write(f'ddp_utils: {ddp_utils.__file__}\n')" || {
  log "FATAL: cannot import ddp_utils.py from $(pwd)"; exit 1; }

fail=0
for i in "${!ARMS[@]}"; do
  name=${NAMES[$i]}
  armlog="${OUTDIR}/arm_${name}.log"
  log "---- arm ${i}: ${name} ----"
  # timeout: a DDP hang must look like a failure, not like a slow job.
  timeout 480 torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      "${SMOKE[@]}" ${ARMS[$i]} \
      --out_dir "${OUTDIR}" --run_name "smoke_${name}" 2>&1 | tee "${armlog}"
  rc=${PIPESTATUS[0]}
  if (( rc == 124 )); then
    log "FAIL [${name}]: TIMED OUT -- this is the DDP hang. Re-run this job with"
    log "      --ddp_find_unused_params added to SMOKE and see if it clears."
    fail=1
  elif (( rc != 0 )); then
    log "FAIL [${name}]: exit ${rc}"
    fail=1
  elif ! grep -q "\[ddp\] rank .*/${NPROC} " "${armlog}"; then
    # Exit 0 is NOT sufficient -- see the note above. The banner is printed by
    # train.py only after it has actually joined the process group.
    log "FAIL [${name}]: ran clean but printed no '[ddp] rank i/${NPROC}' banner."
    log "      DDP did NOT engage: torchrun started ${NPROC} independent single-GPU"
    log "      runs. Nothing about distributed training was exercised."
    fail=1
  else
    ranks=$(grep -c "\[ddp\] rank .*/${NPROC} " "${armlog}")
    log "ok   [${name}]  (${ranks}/${NPROC} ranks joined the process group)"
  fi
done

echo
if (( fail == 0 )); then
  log "SMOKE PASS -- NCCL works and all four arms complete under DDP."
  log "Next: sbatch job_owt_medium_ddp.slurm (after preprocessing finishes)."
else
  log "SMOKE FAIL -- do NOT submit the 24h job yet. See the arm(s) marked FAIL above."
fi
exit $fail
