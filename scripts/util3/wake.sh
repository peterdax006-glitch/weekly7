#!/bin/bash
# Exit (= wake the agent) on an anomaly, or after MAXMIN minutes. Prints the reason.
MAXMIN=${1:-60}; M=C:/Users/peter/creator_runtime/util3/meter.log
start=$(date +%s); n0=$(wc -l < $M 2>/dev/null || echo 0)
while true; do
  sleep 30
  if ! powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'guardian_sync' }) { exit 0 } else { exit 1 }"; then echo "sync loop dead"; exit 0; fi
  n=$(wc -l < $M 2>/dev/null || echo 0)
  if [ "$n" -gt "$n0" ]; then
    last=$(tail -1 $M); n0=$n
    a=$(echo "$last" | grep -o -E "avg=[0-9.]+" | cut -d= -f2)
    if [ -z "$a" ] || awk "BEGIN{exit !($a < 85)}"; then echo "low: $last"; exit 0; fi
  fi
  [ $(( $(date +%s) - start )) -ge $((MAXMIN*60)) ] && { echo "timer: $(tail -1 $M)"; exit 0; }
done
