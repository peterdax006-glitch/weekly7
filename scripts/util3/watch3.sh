#!/bin/bash
# util3 GPU filler (h64): a trace-generation llama-server on pod port 18350 that only fills what the runner leaves.
#  - any runner 8B/14B/27B/30B/32B server up  -> filler OFF (instantly)
#  - runner small server up (or runner idle < IDLE_S) -> 4B Q4 filler (~4.5 GB, coexists with the largest small-tier server)
#  - runner idle >= IDLE_S and >= 14 GB free -> 14B Q4 filler (~12.5 GB); killed the moment any runner server appears
#  - free VRAM < 700 MiB while the filler is up -> filler OFF
# Mode/model is written to mode.txt for the PC driver. STOP file ends it. Polls every 0.5 s.
D=/root/util3; cd $D; touch watch.on
M4=/workspace/nupen/models/Qwen3-4B-Q4_K_M.gguf
M14=/workspace/nupen/models/Qwen3-14B-Q4_K_M.gguf
PORT=18350; IDLE_S=${IDLE_S:-240}
srv=""; mode=off; idle_since=$(date +%s)
log(){ echo "$(date +%T) $*" >> watch.log; }
stop(){ [ -n "$srv" ] && { kill $srv 2>/dev/null; for i in $(seq 1 20); do kill -0 $srv 2>/dev/null || break; sleep 0.1; done; kill -9 $srv 2>/dev/null; log "filler $mode off ($1)"; }; srv=""; mode=off; echo off > mode.txt; }
start(){ # $1 = 4b|14b
  if [ "$1" = 14b ]; then m=$M14; a="-np 32 -c 32768"; else m=$M4; a="-np 24 -c 24576"; fi
  [ -f "$m" ] || { log "missing $m"; return; }
  nohup nice -n 10 /opt/llama.cpp/llama-server -m $m --host 127.0.0.1 --port $PORT -ngl 99 -fa on -ctk q8_0 -ctv q8_0 $a -b 2048 -ub 512 -t 2 --no-webui --metrics > srv_$1.log 2>&1 &
  srv=$!; mode=$1; echo "starting $1" > mode.txt; log "filler $1 start pid $srv"
  for i in $(seq 1 120); do
    curl -sf -o /dev/null localhost:$PORT/health && { echo "$1 $(basename $m)" > mode.txt; log "filler $1 healthy"; return; }
    kill -0 $srv 2>/dev/null || { log "filler $1 died at start"; srv=""; mode=off; echo off > mode.txt; return; }
    sleep 0.5
  done
}
echo off > mode.txt
while [ ! -f STOP ]; do
  [ -n "$srv" ] && ! kill -0 $srv 2>/dev/null && { log "filler $mode exited"; srv=""; mode=off; echo off > mode.txt; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  others=$(pgrep -af "llama-server" | grep -v -- "--port $PORT " | grep -v -- "-ngl 0" | grep -v -E -- "--port 1830[0-9] " | grep -v pgrep)
  big=$(echo "$others" | grep -c -E "8B|14B|27B|30B|32B")
  nother=$(echo "$others" | grep -c "llama-server")
  now=$(date +%s); [ "$nother" -gt 0 ] && idle_since=$now
  if [ "$big" -gt 0 ]; then [ -n "$srv" ] && stop "runner big server"
  elif [ "$mode" = 14b ] && [ "$nother" -gt 0 ]; then stop "runner server appeared"
  elif [ -n "$srv" ] && [ "${free:-0}" -lt 700 ]; then stop "free ${free} MiB"
  elif [ -z "$srv" ]; then
    if [ "$nother" -eq 0 ] && [ $((now-idle_since)) -ge $IDLE_S ] && [ "${free:-0}" -ge 14000 ]; then start 14b
    elif [ "${free:-0}" -ge 6000 ]; then start 4b; fi
  elif [ "$mode" = 4b ] && [ "$nother" -eq 0 ] && [ $((now-idle_since)) -ge $IDLE_S ] && [ "${free:-0}" -ge 10000 ]; then stop "upgrade to 14b"
  fi
  sleep 0.5
done
stop "STOP file"; rm -f watch.on
