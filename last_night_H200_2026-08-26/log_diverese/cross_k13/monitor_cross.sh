#!/usr/bin/env bash
set -u
ROOT=/home/ubuntu/mohammad/PoisonBase
LOGDIR="$ROOT/log_diverese/cross_k13"
while :; do
  now=$(date --iso-8601=seconds)
  pid=$(cat "$LOGDIR/launcher.pid" 2>/dev/null || true)
  {
    echo "===== $now ====="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | sed -n '2,6p'
    echo "cache_nets=$(find "$ROOT/cache_cross_k13_live" -type f -name 'net_*.pt' 2>/dev/null | wc -l)"
    echo "completed_experiments=$(awk -F '\t' '$3=="experiment" && $4=="complete"{n++} END{print n+0}' "$LOGDIR/status.tsv" 2>/dev/null)"
    tail -14 "$LOGDIR/status.tsv" 2>/dev/null
  } >> "$LOGDIR/monitor.log" 2>&1
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "===== $now launcher no longer running =====" >> "$LOGDIR/monitor.log"
    break
  fi
  sleep 300
done
