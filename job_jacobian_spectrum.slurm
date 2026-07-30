#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_jacobian_spectrum.slurm — full singular-value spectrum of every block's
# input-output Jacobian, for all five arms of a revparityreg sweep.
#
# One array task per arm; each task loops over all 12 blocks x N_POINTS
# linearization points and writes the complete spectrum of each N x N Jacobian.
# See analysis/jacobian_spectrum.py for the method; this file is about the
# resource envelope and why it is what it is.
#
#   ID  arm (run dir under CKPT_DIR)      per-token width C   N at T=16
#    0  baseline_baseline                  768  (n_embd)      12,288
#    1  reversible_vf_scaling_narrow      1536  (2*n_embd)    24,576
#    2  reversible_vpb_baseline_narrow    1536                24,576
#    3  reversible_vpb_scaling_narrow     1536                24,576
#    4  reversible_vpm_scaling_narrow     1536                24,576
#
# The IDs are positions in the SORTED arm list, which is exactly what
# `--list_arms` prints. Verify before trusting the mapping above:
#   python -m analysis.jacobian_spectrum --ckpt_dir revparityreg_24979051 --list_arms
#
# Why the sequence length is 16 and not 1024
# ------------------------------------------
# The Jacobian of a block is dense N x N with N = T * C, and an SVD needs it
# resident. At the training block_size of 1024 that is 4.9 TB (baseline) /
# 19.8 TB (reversible) in float64 — the object cannot be formed at any precision
# on any machine. The whole measurement therefore only exists at short T:
#
#    T     N (base)   N (rev)    J fp64 base   J fp64 rev    rev SVD budget
#     8      6,144     12,288       0.30 GB       1.21 GB     ~1 min
#    16     12,288     24,576       1.21 GB       4.83 GB     ~5-15 min   <- default
#    32     24,576     49,152       4.83 GB      19.3  GB     ~1-2 h, tight on 40 GB
#    64     49,152     98,304      19.3  GB      77.3  GB     does not fit
#
# T=16 is in-distribution rather than a truncation hack: positions 0..15 of a
# real held-out story prefix are exactly what the model sees at the start of every
# training window. It is still a SHORT-CONTEXT Jacobian. Nothing measured here
# licenses extrapolating the spectrum to T=1024 — if the T-dependence matters,
# submit T=8 and T=16 and read the trend off the two (see "Other sizes" below).
#
# Precision: float64, not negotiable at the default settings
# ----------------------------------------------------------
# Measured on the trained 'small' baseline at T=4: sigma_max 3.6e+01,
# sigma_min 3.0e-05, i.e. cond(J) ~ 1.2e6. float32 has ~1.2e-7 relative
# precision, so the bottom of that spectrum is at the noise floor in fp32 and
# sum_log_sigma (= log|det J|, the quantity the regime sweep is *about*) would be
# garbage. fp64 costs ~2x the flops on an A100 and buys the whole tail. Use
# --dtype float32 only for a T=32+ probe where fp64 will not fit, and then do not
# report anything from the small singular values.
#
# Validation this job carries with it
# -----------------------------------
# Every record has three independent checks; a run whose checks fail is not a
# result. Verified locally against the real trained checkpoints at T=2 and T=4:
#
#   fd_rel_err     J@v vs a central difference along random unit directions.
#                  Observed 4e-08 .. 1.7e-07 at the trained geometry (it tracks
#                  cond(J), so ~1e-7 is healthy here, not a warning).
#   causal_resid   max |dy_t/dx_s| over s > t. Observed EXACTLY 0.0 everywhere.
#   logdet_err     |sum_i log sigma_i  -  (-T * sum_i (gamma_i + alpha_i))|.
#                  The reversible block is shear o scale o shear with det-1
#                  shears, so that identity is exact. Observed 2.3e-13 absolute
#                  on |log|det|| ~ 4e3, i.e. ~5e-17 relative — machine precision,
#                  on all four regimes including the frozen vpb_baseline (0) and
#                  the self-centering vpb_scaling (0 per block).
#
# The baseline block has no closed-form determinant, so it gets the first two
# checks only and sum_log_sigma is reported without a reference. This is also the
# one place the two archs are NOT comparable: N differs (T*768 vs T*1536), so
# sum_log_sigma / frob / entropy_erank scale with N. Compare mean_log_sigma,
# frac_sigma_gt1 and the quantiles across archs; every record carries n_dim.
#
# Attention kernel
# ----------------
# --attn_impl manual, deliberately, and this does NOT contradict the
# "never mix kernels" rule in job_reversible_parity_regimes.slurm. That rule is
# about comparing TRAINING trajectories, where reduction order compounds. Here
# every arm is read at manual, the weights are fixed, and the two kernels are the
# same function of them. Manual is required anyway: fused SDPA has no fp64 kernel
# and no vmap batching rule for the batched-VJP path the Jacobian build uses.
#
# Calibrate before committing 8 h x 5
# -----------------------------------
# The Jacobian-build cost is measured; the cuSOLVER SVD cost at N=24,576 fp64 is
# an ESTIMATE and cuSOLVER's SVD is the one routine that regularly comes in far
# off a flop-count guess. Spend 20 minutes buying the real number first — one
# reversible arm, one layer, one point, which is exactly one SVD:
#
#   sbatch --array=1 --time=00:40:00 \
#     --export=ALL,N_POINTS=1,LAYERS=0 job_jacobian_spectrum.slurm
#
# Read svd_sec and peak_mem_gb off that record, multiply by 12*N_POINTS, and set
# --time from the measurement. If svd_sec is unexpectedly large, try
# SVD_DRIVER=gesvdj before reducing scope.
#
# Usage:    sbatch job_jacobian_spectrum.slurm                    # T=16, all 5 arms
#           sbatch --export=ALL,SEQ_LEN=8 job_jacobian_spectrum.slurm
#           sbatch --export=ALL,CKPT_DIR=revparityreg_24957997 job_jacobian_spectrum.slurm
#           sbatch --array=0 job_jacobian_spectrum.slurm          # baseline only
# Fetch:    rsync -avhP idun:.../results/jacobian_spectrum_T16/ results/jacobian_spectrum_T16/
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=jacobian_spectrum
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# 40 GB is the binding constraint, not compute: the reversible J is 4.83 GB in
# fp64 at T=16 and cuSOLVER's SVD workspace sits on top of it. An a100 also has
# real fp64 throughput (9.7 TFLOPS, 19.5 with DMMA), which a consumer card does
# not — do not relax this to gpu:1 without dropping to --dtype float32.
#SBATCH --gres=gpu:a100:1
# HOST ram, and unusually small for this repo: no training bins are loaded, only
# data/tinystories_val.bin (9.5 MB). The 24 G is sized for the CPU SVD fallback
# in svdvals(), which needs the 4.83 GB matrix plus LAPACK workspace on the host
# if cuSOLVER runs out of device workspace.
#SBATCH --mem=24G
# Per task at the defaults: 12 layers x 2 points = 24 Jacobians.
# The Jacobian BUILD is cheap (~seconds each: one forward graph, reused across
# chunked batched backwards). The SVD dominates and is O(N^3):
#   baseline    N=12,288  -> ~0.5-2 min each  -> ~20-50 min total
#   reversible  N=24,576  -> ~5-15 min each   -> ~2-6 h total
# 8 h leaves headroom on the slowest arm even if cuSOLVER underperforms. Every
# record logs jac_sec / svd_sec / peak_mem_gb, so the second submission can be
# sized off measurements instead of this estimate.
#SBATCH --time=08:00:00
#SBATCH --array=0-4                        # one task per arm; see the table above
#SBATCH --output=output_jacspec_%A_%a.txt
#SBATCH --error=output_jacspec_%A_%a.err
# Allow the GPU-usability probe below to put a task back in the queue when it lands
# on a node whose card cannot create a CUDA context. Without this, `scontrol
# requeue` is refused and the task just dies.
#SBATCH --requeue
# idun-07-06 failed all three reversible tasks of array 25002324 this way (see the
# probe for the evidence). The probe now excludes a bad node dynamically, so this
# line is belt-and-braces for the node we already know about — drop it once 07-06 is
# fixed, and override per-submission with `sbatch --exclude=...` (the command line
# wins over this directive).
#SBATCH --exclude=idun-07-06

#SBATCH --mail-user=hakon.noren@ntnu.no
#SBATCH --mail-type=ALL


# ── Conda environment ─────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate lra_torch

export OMP_NUM_THREADS=8          # the CPU SVD fallback is the only threaded part
export MKL_NUM_THREADS=8

# ── Configuration (override any of these with --export=ALL,VAR=value) ─────────
# Checkpoint sweep to read. Relative to the repo root, which is $SLURM_SUBMIT_DIR.
CKPT_DIR=${CKPT_DIR:-revparityreg_24979051}

# Tokens per linearization point. THE memory/time dial: J is (T*C)^2 and the SVD
# is O((T*C)^3), so T=32 costs 8x the T=16 SVD and 4x its memory. See the table
# in the header before raising it.
SEQ_LEN=${SEQ_LEN:-16}

# Distinct held-out story prefixes per block. Multiple points are what makes a
# per-block claim about spectrum shape rather than about one input; 2 is the
# minimum that shows the spread, and cost is exactly linear in this.
N_POINTS=${N_POINTS:-2}

# Blank = all blocks. Set e.g. LAYERS="0 5 11" to shard a T=32 run.
LAYERS=${LAYERS:-}

DTYPE=${DTYPE:-float64}

# Rows of J per batched backward. Activation memory scales with it, wall-clock
# inversely. 256 keeps the batched-VJP intermediates at a few hundred MB at these
# geometries; the SVD dominates anyway, so there is little to win by raising it.
CHUNK=${CHUNK:-256}

# cuSOLVER driver. Empty = PyTorch chooses. If the SVD is the bottleneck, try
# gesvdj (Jacobi, usually fastest for square matrices with no vectors wanted).
SVD_DRIVER=${SVD_DRIVER:-}

OUT_DIR=${OUT_DIR:-results/jacobian_spectrum_T${SEQ_LEN}}

# Select the arm BY NAME instead of by array index. The index is a position in the
# sorted arm list, so a missing or renamed run dir silently shifts every later
# index; a name cannot shift. Use this for any single-arm rerun:
#   sbatch --array=0 --export=ALL,ARM=reversible_vf_scaling_narrow job_jacobian_spectrum.slurm
ARM=${ARM:-}

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p "${OUT_DIR}"

ID=${SLURM_ARRAY_TASK_ID:-0}

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} host=$(hostname)"
log "CKPT_DIR=${CKPT_DIR} OUT_DIR=${OUT_DIR}"
log "SEQ_LEN=${SEQ_LEN} N_POINTS=${N_POINTS} DTYPE=${DTYPE} CHUNK=${CHUNK} LAYERS='${LAYERS:-all}'"
log "Arm selector: ${ARM:+ARM=${ARM} (name overrides index)}${ARM:-arm_index=${ID}}"

# ── Preflight ─────────────────────────────────────────────────────────────────
# The arm list is derived, not hardcoded, so a renamed or missing run dir fails
# here with the actual list rather than silently shifting every array index.
if [[ ! -d "${CKPT_DIR}" ]]; then
  log "ERROR: CKPT_DIR '${CKPT_DIR}' does not exist (cwd=$(pwd))"; exit 1
fi
if [[ ! -f data/tinystories_val.bin ]]; then
  log "ERROR: data/tinystories_val.bin missing. Build it with:"
  log "       python preprocess_tinystories.py --out_dir data --splits val"
  exit 1
fi
log "Arms discovered under ${CKPT_DIR} (array index = first column):"
run python3 -u -m analysis.jacobian_spectrum --ckpt_dir "${CKPT_DIR}" --list_arms || exit 1

# ── GPU sanity check ──────────────────────────────────────────────────────────
# An OOM inside cuSOLVER is indistinguishable from an OOM caused by another
# process already owning the card, and the two have opposite fixes (lower SEQ_LEN
# vs resubmit). memory.free should be within a few hundred MB of memory.total
# here; if it is not, this task will fail regardless of SEQ_LEN.
log "GPU alloc: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || log "nvidia-smi unavailable"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true

# ── GPU USABILITY PROBE — do not remove ───────────────────────────────────────
# nvidia-smi reporting a healthy card is NOT sufficient. In array 25002324 tasks
# 1, 2 and 3 all died 6 s in, at `model.to('cuda')`, with
#
#   torch.AcceleratorError: CUDA error: out of memory
#
# while the preflight smi on the SAME line printed "40960 MiB total, 0 MiB used,
# 40442 MiB free" and an empty compute-apps table. The model is ~1 GB in fp64, so
# this was never a sizing problem: the card could not create a CUDA context at
# all. Tasks 0 and 4 ran to completion on other nodes with identical settings, and
# all three failures were on idun-07-06.
#
# A node whose GPU cannot create a context is indistinguishable from a healthy one
# until something tries to allocate, so allocate here — 3 s of probe instead of an
# array task that dies after the 6 s model load having written nothing.
GPU_PROBE=$(python3 -u -c "
import torch
torch.zeros(1, device='cuda'); torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f'{torch.cuda.get_device_name(0)} | allocatable {free/2**30:.1f} / {total/2**30:.1f} GiB')
" 2>&1)
PROBE_STATUS=$?
if [[ ${PROBE_STATUS} -ne 0 ]]; then
  log "ERROR: GPU on $(hostname -s) is unusable despite nvidia-smi looking healthy:"
  log "  ${GPU_PROBE}"
  RESTARTS=${SLURM_RESTART_COUNT:-0}
  if [[ -n "${SLURM_JOB_ID:-}" ]] && (( RESTARTS < 3 )); then
    # Exclude THIS node before requeueing, otherwise Slurm is free to hand the task
    # straight back to the same broken card and we burn all three restarts on it.
    log "Adding $(hostname -s) to this task's exclude list and requeueing (restart ${RESTARTS} -> $((RESTARTS+1)))."
    scontrol update jobid="${SLURM_JOB_ID}" ExcNodeList="$(hostname -s)" 2>&1 | while read -r l; do log "  scontrol: ${l}"; done
    if scontrol requeue "${SLURM_JOB_ID}"; then
      log "Requeued. Exiting; Slurm will restart this task elsewhere."
      exit 1
    fi
    log "requeue failed — giving up on this task."
  else
    log "Not requeueing (SLURM_RESTART_COUNT=${RESTARTS}). Resubmit with:"
    log "  sbatch --exclude=$(hostname -s) --array=${ID} job_jacobian_spectrum.slurm"
  fi
  exit 1
fi
log "GPU probe OK: ${GPU_PROBE}"

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
  --ckpt_dir "${CKPT_DIR}"
  --out_dir "${OUT_DIR}"
  --seq_len "${SEQ_LEN}"
  --n_points "${N_POINTS}"
  --dtype "${DTYPE}"
  --chunk "${CHUNK}"
  --device cuda
  --attn_impl manual
  --fd_dirs 3
  --seed 0
)
if [[ -n "${ARM}" ]]; then
  ARGS+=(--arm "${ARM}")            # name wins over the array index when given
else
  ARGS+=(--arm_index "${ID}")
fi
[[ -n "${LAYERS}" ]] && ARGS+=(--layers ${LAYERS})
[[ -n "${SVD_DRIVER}" ]] && ARGS+=(--svd_driver "${SVD_DRIVER}")

run python3 -u -m analysis.jacobian_spectrum "${ARGS[@]}"
STATUS=$?

log "Task ${ID} exited with status ${STATUS}. Records in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"/spectrum_*.jsonl 2>/dev/null || log "  (none written)"
du -sh "${OUT_DIR}/sv" 2>/dev/null || true
exit ${STATUS}

# ─────────────────────────────────────────────────────────────────────────────
# Other sizes
# -----------
# T=8 (cheap T-scaling companion; ~8x faster SVD, 4x less memory than T=16):
#   sbatch --export=ALL,SEQ_LEN=8 --time=02:00:00 job_jacobian_spectrum.slurm
#
# T=32, baseline only (N=24,576 — same cost as a T=16 reversible arm):
#   sbatch --array=0 --export=ALL,SEQ_LEN=32 --time=08:00:00 job_jacobian_spectrum.slurm
#
# T=32, reversible (N=49,152). fp64 is 19.3 GB for J alone plus cuSOLVER
# workspace, so this needs float32 AND fewer Jacobians — and in fp32 the tail of
# the spectrum is noise (see the precision note in the header), so read only
# sigma_max / the top quantiles off it:
#   sbatch --array=1-4 --export=ALL,SEQ_LEN=32,DTYPE=float32,N_POINTS=1,LAYERS="0 5 11" \
#          --time=08:00:00 job_jacobian_spectrum.slurm
#
# Reading the output
# ------------------
# Per-record scalars land in ${OUT_DIR}/spectrum_<arm>.jsonl; the full spectra in
# ${OUT_DIR}/sv/<arm>_L<layer>_p<point>_T<T>.npy (float64, length N, ~200 KB
# each, ~5 MB for the whole sweep). A quick sanity pass:
#
#   python - <<'PY'
#   import glob, json
#   for f in sorted(glob.glob("results/jacobian_spectrum_T16/spectrum_*.jsonl")):
#       for line in open(f):
#           r = json.loads(line)
#           print(f"{r['arm']:32s} L{r['layer']:02d} p{r['point']} N={r['n_dim']:6d} "
#                 f"smax={r['sigma_max']:9.3e} smin={r['sigma_min']:9.3e} "
#                 f"mean_log_sigma={r['mean_log_sigma']:+.4f} "
#                 f"frac>1={r['frac_sigma_gt1']:.3f} "
#                 f"fd={r['fd_rel_err']:.1e} causal={r['causal_resid']:.1e} "
#                 f"logdet_err={r.get('logdet_err', float('nan')):.1e}")
#   PY
#
# Before reading any spectrum, confirm ALL of: fd_rel_err < 1e-5,
# causal_resid < 1e-10, and (reversible arms) logdet_rel_err < 1e-9. Those three
# are the difference between a measurement and a plausible-looking array.
# ─────────────────────────────────────────────────────────────────────────────
