# eqr_scaling.sh — EqR's test-time depth x breadth evaluation of one checkpoint (evaluate.py,
# config/eval/depth_breadth.yaml), sourced by sudoku_eqr.slurm (right after training) and
# sudoku_scaling.slurm (on existing runs). Needs: cwd = the EqR checkout, the eqr env active, log().
#
#   scale_eval <checkpoint> [grid]
#
# grid: depth:breadth[:noise] entries, noise defaulting to SCALE_NOISE. Breadth b runs b random initial
# states per puzzle and picks the answer by convergence (top-k least-changed runs), majority vote or any.
# A point already evaluated for this checkpoint's step is skipped (SCALE_REDO=1 re-runs it), so a job that
# hit its time limit continues where it stopped when resubmitted.
# Env: SCALE_NOISE (0.5), SCALE_NPUZ (2048 = EqR's "lite" test subset), SCALE_SEQS (4096 sequences per
# batch = puzzles x breadth), SCALE_TOPK (4).

# The standard grid: EqR's depth curve and its headline 64 x 128 (top-4; 98.60 for its released checkpoint)
# at test noise 0.5, and the depth curve and 64 x 16 at 0.25, where the twin and inner-loop arms do best
# at 50k (paper sec. 5.1). About 1 h per checkpoint on an A100, 40 min of it the 64 x 128 point.
SCALE_GRID_DEFAULT="16:1 64:1 256:1 1024:1 64:128 64:16:0.25 256:1:0.25 1024:1:0.25"

scale_eval() {
  local ckpt=$1 grid=${2:-${SCALE_GRID_DEFAULT}}
  local noise=${SCALE_NOISE:-0.5} npuz=${SCALE_NPUZ:-2048} seqs=${SCALE_SEQS:-4096} topk=${SCALE_TOPK:-4}
  local run_dir step db d b n nl sq gbs nb t0
  run_dir=$(dirname "$(dirname "${ckpt}")")          # evaluate.py writes eval_preds/ next to checkpoints/
  step=$(basename "${ckpt}"); step=${step#step_}; step=${step%%_*}
  [[ "${step}" =~ ^[0-9]+$ ]] || step='*'            # e.g. EqR's released eqr.pth
  for db in ${grid}; do
    IFS=: read -r d b n <<< "${db}"; n=${n:-${noise}}
    [[ -z "${d}" || -z "${b}" ]] && { log "FATAL: bad grid entry '${db}' (want depth:breadth[:noise])"; return 1; }
    nl=$(python -c "print(float('${n}'))")             # evaluate.py's directory label writes the noise as a float
    if [[ "${SCALE_REDO:-0}" != 1 ]] && compgen -G \
        "${run_dir}/eval_preds/step_${step}_scale_D${d}_B${b}_*_noise_scale-${nl}_*/eval_metrics_step_*.json" > /dev/null; then
      log "   depth ${d} breadth ${b} noise ${n}: already evaluated"; continue
    fi
    # With breadth > 1, evaluate.py keeps every iteration's logits for the convergence selection, so
    # memory grows with depth: past depth 256 the batch shrinks to keep sequences x depth at seqs x 256.
    sq=$(( b > 1 && d > 256 ? seqs * 256 / d : seqs ))
    gbs=$(( sq / b )); (( gbs < 1 )) && gbs=1; (( gbs > npuz )) && gbs=${npuz}
    nb=$(( (npuz + gbs - 1) / gbs ))
    log "eval depth ${d} breadth ${b} noise ${n}: ${gbs} puzzles x ${nb} batches"
    t0=${SECONDS}
    python evaluate.py eval_yaml=config/eval/depth_breadth.yaml checkpoint="${ckpt}" \
      halt_max_steps=${d} noise_scale=${n} different_init=${b} convergence_top_k=${topk} \
      global_batch_size=${gbs} max_eval_steps=${nb} suffix=scale_D${d}_B${b} \
      2>&1 | tr '\r' '\n' | grep -E "saved|Error|Traceback|coupled_eqr" | tail -3
    log "   done in $(( SECONDS - t0 ))s"
  done
}
