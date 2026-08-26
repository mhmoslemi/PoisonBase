#!/usr/bin/env bash
set -u
ROOT=/home/ubuntu/mohammad/PoisonBase
LOGDIR="$ROOT/log_diverese"
MON="$LOGDIR/monitor.log"
while :; do
  now=$(date --iso-8601=seconds)
  launcher=$(cat "$LOGDIR/launcher.pid" 2>/dev/null || true)
  {
    echo "===== $now ====="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
    echo "our_python_processes=$(pgrep -fc "final_update.py|defense.py" || true)"
    if [ -s "$LOGDIR/status.tsv" ]; then tail -12 "$LOGDIR/status.tsv"; fi
    echo "delta_files=$(find "$ROOT/ours_result_alpha025_live" -type f -name 'delta_*.pt' 2>/dev/null | wc -l)"
    echo "defense_rows=$(find "$ROOT/defense_result" -mindepth 2 -maxdepth 2 -name results.csv -path '*seldpp0.25*' -exec awk 'FNR>1{n++} END{print n+0}' {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  } >> "$MON" 2>&1
  if [ -n "$launcher" ] && ! kill -0 "$launcher" 2>/dev/null; then
    echo "===== $now launcher no longer running =====" >> "$MON"
    break
  fi
  sleep 300
done
