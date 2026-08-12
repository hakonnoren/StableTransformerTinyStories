#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# check_owt.sh — status of the OpenWebText preprocessing job.
#
# The job has two phases that report in DIFFERENT places, which is the main
# reason this script exists:
#
#   Phase 1, download: HuggingFace draws tqdm progress bars on STDERR, so they
#       land in output_owt_preprocess.err. Meanwhile stdout shows nothing new
#       after "encoding OpenWebText ...". A .txt that looks frozen for an hour is
#       normal and does not mean the job is stuck.
#   Phase 2, encode:  process_openwebtext.py prints "processed N/M" to STDOUT
#       every 5000 docs, and data/openwebtext_train.bin starts growing.
#
# Usage:
#   ./check_owt.sh            one-shot status
#   ./check_owt.sh -w         refresh every 60s until the job leaves the queue
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

OUT=output_owt_preprocess.txt
ERR=output_owt_preprocess.err
TRAIN=data/openwebtext_train.bin
VAL=data/openwebtext_val.bin

# Expected finished sizes: 8.62B train + 432M val tokens at 2 bytes (uint16).
TRAIN_EXPECT=$((8620000000 * 2))
VAL_EXPECT=$((432000000 * 2))

hr() { printf '%.0s─' {1..70}; echo; }
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

status_once() {
  hr
  echo "OpenWebText preprocessing — $(date '+%Y-%m-%d %H:%M:%S')"
  hr

  # ---- queue state -----------------------------------------------------------
  local q
  q=$(squeue -u "$USER" -n owt_preprocess -h -o "%i %T %M %L %R" 2>/dev/null)
  if [[ -n "$q" ]]; then
    echo "QUEUE   : $q"
    echo "          (jobid state elapsed timeleft reason)"
  else
    echo "QUEUE   : not in queue — finished, failed, or never submitted"
    local jid
    jid=$(sacct -u "$USER" --name=owt_preprocess -n -o JobID,State,Elapsed,ExitCode -X 2>/dev/null | tail -3)
    [[ -n "$jid" ]] && echo "SACCT   :"$'\n'"$jid"
  fi

  # ---- HF cache (phase 1) ----------------------------------------------------
  local hf="${HF_HOME:-${USERWORK:-$HOME/work}/hf_cache}"
  if [[ -d "$hf" ]]; then
    echo "CACHE   : $(du -sh "$hf" 2>/dev/null | cut -f1) in $hf  (finished ≈ 64G)"
  else
    echo "CACHE   : $hf does not exist yet"
  fi

  # ---- bins (phase 2) --------------------------------------------------------
  if [[ -f "$TRAIN" ]]; then
    local sz pct
    sz=$(stat -c %s "$TRAIN" 2>/dev/null || stat -f %z "$TRAIN")
    pct=$(( sz * 100 / TRAIN_EXPECT ))
    echo "TRAIN   : $(human "$sz") / ~$(human "$TRAIN_EXPECT")  (~${pct}%)"
  else
    echo "TRAIN   : not started (still downloading, or job not running)"
  fi
  if [[ -f "$VAL" ]]; then
    local sz
    sz=$(stat -c %s "$VAL" 2>/dev/null || stat -f %z "$VAL")
    echo "VAL     : $(human "$sz") / ~$(human "$VAL_EXPECT")   (written AFTER train)"
  fi

  # ---- throughput + ETA ------------------------------------------------------
  # Sampled live rather than assumed, since the encode rate depends on the node.
  if [[ -f "$TRAIN" ]] && [[ -n "$q" ]]; then
    local a b rate left
    a=$(stat -c %s "$TRAIN" 2>/dev/null || stat -f %z "$TRAIN")
    sleep 20
    b=$(stat -c %s "$TRAIN" 2>/dev/null || stat -f %z "$TRAIN")
    rate=$(( (b - a) / 20 ))
    if (( rate > 0 )); then
      left=$(( (TRAIN_EXPECT - b) / rate ))
      printf "RATE    : %s/s  →  train ETA ~%dh%02dm\n" \
             "$(human "$rate")" $((left/3600)) $(((left%3600)/60))
    else
      echo "RATE    : 0 B/s over 20s — either downloading still, or stalled"
    fi
  fi

  # ---- log tails -------------------------------------------------------------
  echo
  echo "── stdout ($OUT) — slurm log + 'processed N/M' -----------------------"
  [[ -f "$OUT" ]] && tail -6 "$OUT" || echo "  (no file yet)"
  echo
  echo "── stderr ($ERR) — HF download bars live HERE ------------------------"
  if [[ -f "$ERR" ]]; then
    # tqdm rewrites one line with \r; translate to newlines to see the latest.
    tail -c 3000 "$ERR" | tr '\r' '\n' | grep -v '^$' | tail -5
  else
    echo "  (no file yet)"
  fi

  # ---- anything alarming -----------------------------------------------------
  if [[ -f "$OUT" ]] && grep -qE "FATAL|FAILED" "$OUT"; then
    echo
    echo "!! PROBLEM:"
    grep -E "FATAL|FAILED" "$OUT" | tail -5
  fi
  hr
}

if [[ "${1:-}" == "-w" ]]; then
  while true; do
    clear; status_once
    squeue -u "$USER" -n owt_preprocess -h -o "%i" | grep -q . || { echo "Job left the queue — stopping watch."; break; }
    sleep 60
  done
else
  status_once
fi
