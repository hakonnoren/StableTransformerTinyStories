#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_jacobian_spectrum.slurm — Jacobian spectra of every block AND of the whole
# block stack, the Corollary 4.5 coarse bounds on their extremes, the center
# (mean log singular value), and the cost of each, for the five arms of the paper.
#
# One array task per arm. Each task, at N_POINTS linearization points:
#   per block    full SVD of the N x N Jacobian; center by LU (+ closed form for
#                reversible arms); coarse bounds by power iteration (reversible
#                arms), scored against dense sublayer norms and the SVD; sigma_max
#                by power iteration (every arm)
#   stack        full SVD of Psi_{0,k} for k in STACK_DEPTHS (default: L only),
#                its center by LU, Lemma 3.1 composition checks
#   bounds only  the same power-iteration bounds at BOUND_SEQ_LENS (default 1024),
#                where no dense Jacobian can exist
# Every step is timed with a device sync, which is the full-spectrum vs
# coarse-bound cost comparison. See analysis/jacobian_spectrum.py for the method;
# this file is about which checkpoints, the resource envelope, and why.
#
#   ID  arm                               checkpoint (default CKPT_DIR)                  N at T=16
#    0  baseline_baseline                 revparityreg_24979051/baseline_baseline         12,288
#    1  reversible_vf_scaling_narrow      revparityregrevback_25000297                    24,576
#    2  reversible_vpb_baseline_narrow    revparityregrevback_25001769                    24,576
#    3  reversible_vpb_scaling_narrow     revparityregrevback_25000297                    24,576
#    4  reversible_vpm_scaling_narrow     revparityregrevback_25000297                    24,576
#
# These are the runs of the paper's Table 3 / Figure 3 (ts_L12_memeff_loss_zoom):
# the reversible arms trained with memory-efficient reversible backprop, the
# baseline with ordinary backprop (best_val 1.0522; a baseline cannot use the
# reversible backward). The memeff sweep is split over three job dirs, and
# 24994972 also holds the discarded first vpb_scaling attempt (vrab6ylp), so the
# default CKPT_DIR names the four directories explicitly rather than a sweep.
#
# The IDs are positions in the SORTED arm list, which is exactly what
# `--list_arms` prints. Verify before trusting the mapping above:
#   python -m analysis.jacobian_spectrum --ckpt_dir ${CKPT_DIR} --list_arms
#
# What it costs (measured on array 25002324: A100 40 GB, fp64, T=16; fp32 should be no slower)
# ------------------------------------------------------------------
#   per-block SVD          baseline ~62 s, reversible ~470-570 s     x 12 x N_POINTS
#   stack SVD              same N as a block, so the same ~8.5 min   x |STACK_DEPTHS| x N_POINTS
#   dense build, LU        seconds
#   exact sublayer norms   two T*d = 12,288 builds + one 12,288 symmetric eigensolve,
#                          estimated well under a minute per block
#   power-iteration bounds seconds per block, also at T=1024
#   => at the defaults (N_POINTS=1, STACK_DEPTHS=L): 12 block + 1 stack SVD per arm,
#      reversible arm ~2-2.5 h, baseline ~15-20 min. STACK_DEPTHS=all adds 11 more
#      SVDs (~+1.4-1.8 h per reversible arm): give it --time=06:00:00. Each extra
#      point multiplies all of it.
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
# Precision: float32 by default, so any GPU will do
# -------------------------------------------------
# float32 halves memory and drops the A100 requirement. Checked against the fp64
# T=16 spectra of array 25002324: every reversible block has cond <= 5e3, well
# inside float32's ~1e7 range, and on a GPT-2-width block fp32 reproduced the fp64
# singular values to ~1e-6 relative, the LU center to 3e-9, the bounds to 1e-5.
# What float32 does lose: the baseline's smallest singular values (block 0 has
# cond 1.8e8; blocks 1-3 a few tail values) and the bottom of the baseline STACK.
# n_sigma_below_floor flags them. For those, rerun the baseline in fp64:
#   sbatch --array=0 --gres=gpu:a100:1 --export=ALL,DTYPE=float64,OUT_DIR=results/jacobian_spectrum_T16_paper_fp64 ...
# The center is taken by LU, not by summing log sigma, so it is fine in fp32.
# fd_rel_err is a central difference in the run's dtype and sits at ~1e-2 in fp32
# (cancellation), so in fp32 it only catches gross AD errors.
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
#                  shears, so that identity is exact.
#
#                  CALIBRATED ON THE REAL T=16 SWEEP (array 25002324), which is
#                  NOT what the earlier local T=2/T=4 CPU figure suggested. Do not
#                  expect machine precision here:
#
#                    absolute error   +2.60e-07 +/- 1.4e-08, POSITIVE in 96/96
#                                     records, and essentially IDENTICAL across all
#                                     four regimes (vf, vpm, vpb_baseline,
#                                     vpb_scaling) despite their analytic log|det|
#                                     spanning 0 .. 3.2e+04.
#                    relative error   3-5e-11 once normalized by sum_i|log sigma_i|
#                                     (the L1 scale of the cancelling sum).
#
#                  CONFIRMED as a cuSOLVER precision artifact, not an error in J
#                  or in the identity. The SAME Jacobian (vpb_baseline L0 p0 T=16,
#                  N=24,576, sigma_max 3.0907e+01 / sigma_min 5.4637e-02 /
#                  cond 5.657e+02 -- identical to the cluster record) evaluated
#                  three ways, against a true value of exactly 0:
#
#                    CPU/LAPACK slogdet (LU)   +7.7e-14      48 s
#                    CPU/LAPACK svdvals        -1.2e-11    6447 s
#                    A100/cuSOLVER svdvals     +2.6e-07     ~5 min
#
#                  So cuSOLVER's singular values carry ~2.1e4x the error of
#                  LAPACK's here (~1e-11 relative bias PER sigma, accumulated over
#                  24,576 of them). It is a bias in the SUM and does not perturb
#                  sigma_max/min, cond, the quantiles, the median or the skew at
#                  any digit those are read to.
#
#                  Consequence, now implemented: log|det J| is taken by LU
#                  (logdet_lu) rather than by summing log sigma. That is both the
#                  accurate route AND ~80-130x cheaper than the SVD it used to ride
#                  on, so logdet_err is now |logdet_lu - analytic| and lands at
#                  ~1e-14. The old quantity survives as svd_logdet_drift =
#                  |sum_log_sigma - logdet_lu|, a PRECISION DIAGNOSTIC for the SVD
#                  backend (expect ~2.6e-07 on cuSOLVER, ~1e-11 on LAPACK) that is
#                  reported and never gated on.
#
#                  logdet_rel_err is therefore normalized by
#                  max(|analytic|, sum_i|log sigma_i|), NOT by max(|analytic|, 1.0).
#                  The old denominator became an ABSOLUTE tolerance for exactly the
#                  two regimes whose analytic log|det| is 0 by construction, so
#                  vpb_baseline and vpb_scaling "failed" a 1e-9 gate at 2.5e-07
#                  while vf/vpm "passed" carrying the SAME absolute error. Records
#                  written before that fix are re-gated in memory from the stored
#                  spectra by analysis/plot_jacobian_spectrum.py:regate_logdet().
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
# Calibrate before committing 4 h x 5
# -----------------------------------
# The block SVD cost is now measured (array 25002324: ~8.5 min at N=24,576 fp64).
# What is NOT yet measured on the A100 is everything this job added: the dense
# sublayer norms (a 12,288 symmetric eigensolve; ~8 min on a laptop CPU, expected
# well under a minute here), the power iterations, and the T=1024 bounds. One
# reversible arm, one layer, one point, no stack buys all of those in ~15 min,
# into a scratch out_dir so the real run starts clean:
#
#   sbatch --array=1 --time=00:40:00 \
#     --export=ALL,N_POINTS=1,LAYERS=0,STACK_DEPTHS=,OUT_DIR=results/jacspec_calib \
#     jobs/analysis/job_jacobian_spectrum.slurm
#
# Read svd_sec, exact_norm_sec, bound_sec and peak_mem_gb off spectrum_*.jsonl and
# bound_sec off bounds_*.jsonl (the log's timing table has all of them). If
# svd_sec is unexpectedly large, try SVD_DRIVER=gesvdj before reducing scope.
#
# Usage:    sbatch jobs/analysis/job_jacobian_spectrum.slurm                    # T=16, the paper's 5 arms
#           sbatch --export=ALL,SEQ_LEN=8 jobs/analysis/job_jacobian_spectrum.slurm
#           sbatch --export=ALL,STACK_DEPTHS=all --time=06:00:00 jobs/analysis/job_jacobian_spectrum.slurm
#           sbatch --export=ALL,CKPT_DIR=revparityreg_24979051,OUT_DIR=results/jacobian_spectrum_T16_24979051 \
#                  jobs/analysis/job_jacobian_spectrum.slurm                    # the ordinary-backprop sweep
#           sbatch --array=0 jobs/analysis/job_jacobian_spectrum.slurm          # baseline only
# Fetch:    scripts/fetch_jacobian_spectrum.sh jacobian_spectrum_T16_paper
#
# --resume is always on, so resubmitting the same command after a task is killed
# keeps every (layer, point) and (depth, point) already on disk. It only ever
# resumes records of the SAME checkpoint and record schema: an out_dir holding
# another checkpoint's records for the same arm name is refused, never appended
# to or overwritten. That is why OUT_DIR is per checkpoint set.
#
# History: array 25002324 produced results/jacobian_spectrum_T16 (per-block
# spectra only, revparityreg_24979051 checkpoints, schema 1). Its tasks 1-3 died on
# idun-07-06; see the GPU USABILITY PROBE below.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=jacobian_spectrum
#SBATCH --account=share-ie-imf            # replace with your IDUN account
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
# Any GPU: in float32 the reversible J is 2.4 GB at T=16 and the measured fp64 peak
# of 28 GB halves to ~14 GB. A 16 GB card is tight; if cuSOLVER runs out of
# workspace, svdvals() falls back to the CPU (slow, not fatal). For DTYPE=float64
# pass --gres=gpu:a100:1 on the command line.
#SBATCH --gres=gpu:1
# HOST ram, and unusually small for this repo: no training bins are loaded, only
# data/tinystories_val.bin (9.5 MB). The 24 G is sized for the CPU SVD fallback
# in svdvals(), which needs the 4.83 GB matrix plus LAPACK workspace on the host
# if cuSOLVER runs out of device workspace.
#SBATCH --mem=24G
# Per task at the defaults: 12 layers x 1 point = 12 block SVDs + 1 stack SVD.
# The Jacobian BUILD is cheap (~seconds each: one forward graph, reused across
# chunked batched backwards). The SVD dominates and is O(N^3); measured on array
# 25002324 (slowest single SVD 584 s):
#   baseline    N=12,288  -> ~62 s each       -> ~15-20 min total
#   reversible  N=24,576  -> ~470-584 s each  -> ~2-2.5 h total, stack and bounds included
# 4 h is ~1.6x the worst case. STACK_DEPTHS=all needs --time=06:00:00, N_POINTS=2
# --time=08:00:00. A task killed at the limit loses nothing: resubmit, --resume
# keeps what is on disk.
# Every record logs jac_sec / svd_sec / bound_sec / exact_norm_sec / peak_mem_gb.
#SBATCH --time=04:00:00
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
# Checkpoints to read: one or more sweep dirs or run dirs, space separated,
# relative to the repo root ($SLURM_SUBMIT_DIR). Arms are the run-dir names, and
# an arm name found in two of them is an error. Default: the paper's five runs
# (see the table in the header).
CKPT_DIR=${CKPT_DIR:-"revparityreg_24979051/baseline_baseline revparityregrevback_25000297 revparityregrevback_25001769"}
read -r -a CKPT_DIRS <<< "${CKPT_DIR}"

# Tokens per linearization point. THE memory/time dial: J is (T*C)^2 and the SVD
# is O((T*C)^3), so T=32 costs 8x the T=16 SVD and 4x its memory. See the table
# in the header before raising it.
SEQ_LEN=${SEQ_LEN:-16}

# Distinct held-out story prefixes per block. One point, as for the paper's
# figures: every spectrum is then at a single input, so the run says nothing about
# the spread across inputs (N_POINTS=2 is the minimum that shows it, at exactly
# twice the cost).
N_POINTS=${N_POINTS:-1}

# Blank = all blocks. Set e.g. LAYERS="0 5 11" to shard a T=32 run.
LAYERS=${LAYERS:-}

DTYPE=${DTYPE:-float32}

# Rows of J per batched backward. Activation memory scales with it, wall-clock
# inversely. 256 keeps the batched-VJP intermediates at a few hundred MB at these
# geometries; the SVD dominates anyway, so there is little to win by raising it.
CHUNK=${CHUNK:-256}

# cuSOLVER driver. Empty = PyTorch chooses. If the SVD is the bottleneck, try
# gesvdj (Jacobi, usually fastest for square matrices with no vectors wanted).
SVD_DRIVER=${SVD_DRIVER:-}

# Stack Jacobians Psi_{0,k} to take in full: "L" (the whole stack), integers, or
# "all" (every depth k=1..L: the depth profile of the paper's Figures 1-2, at one
# extra block-sized SVD per k per point). Empty = no stack.
STACK_DEPTHS=${STACK_DEPTHS-L}

# Sequence lengths for the bounds-only pass: power-iteration bounds and sigma_max
# with no dense Jacobian, on contiguous val windows. 1024 is the training context,
# where J would be 19.8 TB. Empty = skip.
BOUND_SEQ_LENS=${BOUND_SEQ_LENS-1024}

# Power iteration: stop at relative change POWER_RTOL per step, or POWER_ITERS
# steps. Each estimate stores its per-iteration trace, so cheaper settings can be
# evaluated after the fact against the exact sublayer norms.
POWER_ITERS=${POWER_ITERS:-500}
POWER_RTOL=${POWER_RTOL:-1e-6}

# Per checkpoint set, never shared with another: the same arm names exist in every
# sweep, and the script refuses an out_dir holding another checkpoint's records.
# results/jacobian_spectrum_T16 holds array 25002324 (revparityreg_24979051).
OUT_DIR=${OUT_DIR:-results/jacobian_spectrum_T${SEQ_LEN}_paper}

# Select the arm BY NAME instead of by array index. The index is a position in the
# sorted arm list, so a missing or renamed run dir silently shifts every later
# index; a name cannot shift. Use this for any single-arm rerun:
#   sbatch --array=0 --export=ALL,ARM=reversible_vf_scaling_narrow jobs/analysis/job_jacobian_spectrum.slurm
ARM=${ARM:-}

log() { echo "[$(date -Is)] $*"; }
run() { log "RUN: $(printf '%q ' "$@")"; "$@"; }

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p "${OUT_DIR}"

ID=${SLURM_ARRAY_TASK_ID:-0}

log "Job meta: array_job=${SLURM_ARRAY_JOB_ID:-NA} task=${ID} host=$(hostname)"
log "CKPT_DIR=${CKPT_DIR} OUT_DIR=${OUT_DIR}"
log "SEQ_LEN=${SEQ_LEN} N_POINTS=${N_POINTS} DTYPE=${DTYPE} CHUNK=${CHUNK} LAYERS='${LAYERS:-all}'"
log "STACK_DEPTHS='${STACK_DEPTHS}' BOUND_SEQ_LENS='${BOUND_SEQ_LENS}' POWER_ITERS=${POWER_ITERS} POWER_RTOL=${POWER_RTOL}"
log "Arm selector: ${ARM:+ARM=${ARM} (name overrides index)}${ARM:-arm_index=${ID}}"

# ── Preflight ─────────────────────────────────────────────────────────────────
# The arm list is derived, not hardcoded, so a renamed or missing run dir fails
# here with the actual list rather than silently shifting every array index.
for d in "${CKPT_DIRS[@]}"; do
  if [[ ! -d "${d}" ]]; then
    log "ERROR: CKPT_DIR entry '${d}' does not exist (cwd=$(pwd))"; exit 1
  fi
done
if [[ ! -f data/tinystories_val.bin ]]; then
  log "ERROR: data/tinystories_val.bin missing. Build it with:"
  log "       python scripts/preprocess_tinystories.py --out_dir data --splits val"
  exit 1
fi
log "Arms discovered under ${CKPT_DIR} (array index = first column):"
run python3 -u -m analysis.jacobian_spectrum --ckpt_dir "${CKPT_DIRS[@]}" --list_arms || exit 1

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
    # Try to exclude THIS node before requeueing so Slurm cannot hand the task
    # straight back to the same broken card. Slurm normally only accepts
    # ExcNodeList on a PENDING job and this one is RUNNING, so expect this to be
    # refused more often than not — it is opportunistic. The real backstops are the
    # static --exclude directive above and the restart cap on this branch.
    log "Attempting to exclude $(hostname -s), then requeueing (restart ${RESTARTS} -> $((RESTARTS+1)))."
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
  --ckpt_dir "${CKPT_DIRS[@]}"
  --out_dir "${OUT_DIR}"
  --seq_len "${SEQ_LEN}"
  --n_points "${N_POINTS}"
  --dtype "${DTYPE}"
  --chunk "${CHUNK}"
  --device cuda
  --attn_impl manual
  --fd_dirs 3
  --seed 0
  --power_iters "${POWER_ITERS}"
  --power_rtol "${POWER_RTOL}"
  # Always on. With no existing record files it is a no-op; with them it keeps the
  # (layer, point) and (depth, point) pairs whose .npy is already on disk. That
  # makes both a wall-clock kill and the probe's requeue cheap to recover from,
  # instead of restarting 13 SVDs from zero.
  --resume
)
[[ -n "${STACK_DEPTHS}" ]] && ARGS+=(--stack_depths ${STACK_DEPTHS})
[[ -n "${BOUND_SEQ_LENS}" ]] && ARGS+=(--bound_seq_lens ${BOUND_SEQ_LENS})
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
ls -lh "${OUT_DIR}"/{spectrum,stack,bounds}_*.jsonl 2>/dev/null || log "  (none written)"
du -sh "${OUT_DIR}/sv" "${OUT_DIR}/sv_stack" 2>/dev/null || true
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
# sigma_max / the top quantiles off it. No stack at this size (STACK_DEPTHS=, empty):
#   sbatch --array=1-4 --export=ALL,SEQ_LEN=32,DTYPE=float32,N_POINTS=1,LAYERS="0 5 11",STACK_DEPTHS=,BOUND_SEQ_LENS= \
#          --time=08:00:00 job_jacobian_spectrum.slurm
#
# Bounds only, cheap, any T up to the block size of 1024 (no SVD anywhere):
#   python -m analysis.jacobian_spectrum --ckpt_dir ${CKPT_DIR} --skip_svd --no_exact_norms ...
# --skip_svd still BUILDS each T=16 Jacobian (seconds) for the fd / causal checks;
# only the SVD and the LU are skipped.
#
# Reading the output
# ------------------
# Under ${OUT_DIR}:
#   spectrum_<arm>.jsonl   per (layer, point): spectrum scalars, center_lu (and
#                          center_analytic), sigma_max_power, and for reversible arms
#                          g/a extremes, sigma_{H,F}_{power,exact}, bound_{lo,hi}_{power,exact},
#                          bound_*_gap_*, bound_ok, and the timers
#   stack_<arm>.jsonl      per (depth, point): the same for Psi_{0,k}, plus
#                          logdet_composition_err, the Lemma 3.1(iii) products
#                          (log_prod_layer_*, log_prod_bound_*), sv_floor
#   bounds_<arm>.jsonl     per (T, layer | stack, point) at BOUND_SEQ_LENS: bounds
#                          and power sigma_max only
#   sv/, sv_stack/         full spectra (float64, length N, ~200 KB each)
# The task log ends with a timing table (full spectrum vs coarse bounds, per block
# and per stack). A quick sanity pass:
#
#   python - <<'PY'
#   import glob, json
#   for f in sorted(glob.glob("results/jacobian_spectrum_T16_paper/spectrum_*.jsonl")):
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
# are the difference between a measurement and a plausible-looking array. For the
# bounds also: bound_ok true and F_offdiag_resid == 0 in every reversible record,
# and for the stack logdet_composition_err ~1e-9 relative to |logdet_lu| and
# n_sigma_below_floor == 0 (if not, the smallest stack singular values are
# rounding and only say "below sv_floor").
# ─────────────────────────────────────────────────────────────────────────────
