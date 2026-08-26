#!/usr/bin/env bash
set -u
pidfile=$1
deadline_iso=$2
status_file=$3
label=$4
guard_log=$5
pid=$(cat "$pidfile" 2>/dev/null || true)
if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
  printf '%s invalid launcher pidfile: %s\n' "$(date -Is)" "$pidfile" >> "$guard_log"
  exit 1
fi
pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
if [ "$pgid" != "$pid" ] || [[ "$cmd" != *launch*sh* ]]; then
  printf '%s refusing guard: pid=%s pgid=%s cmd=%s\n' "$(date -Is)" "$pid" "$pgid" "$cmd" >> "$guard_log"
  exit 1
fi
deadline_epoch=$(date -d "$deadline_iso" +%s)
printf '%s armed label=%s pid=%s pgid=%s deadline=%s\n' "$(date -Is)" "$label" "$pid" "$pgid" "$deadline_iso" >> "$guard_log"
while kill -0 "$pid" 2>/dev/null; do
  now_epoch=$(date +%s)
  if [ "$now_epoch" -ge "$deadline_epoch" ]; then
    now=$(date -Is)
    printf '%s\t%s\tscheduler\tdeadline_reached\t-\tauthorized window ended at %s; terminating only pgid=%s\n' "$now" "$label" "$deadline_iso" "$pgid" >> "$status_file"
    printf '%s deadline reached; TERM only process group %s\n' "$now" "$pgid" >> "$guard_log"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      printf '%s group still alive after 60s; KILL only process group %s\n' "$(date -Is)" "$pgid" >> "$guard_log"
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    exit 0
  fi
  sleep 45
done
printf '%s launcher exited before deadline label=%s pid=%s\n' "$(date -Is)" "$label" "$pid" >> "$guard_log"
