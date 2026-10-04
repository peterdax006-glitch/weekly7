#!/bin/bash
# util3 GPU filler (h64): trace-generation llama-servers on pod ports 18350-18352 that only fill what the runner leaves.
# One llama-server is bound by its single server thread (36 slots of a 4B: GPU 44%); a second instance took it to 75%, so the filler runs
# several small instances instead of one big one.
#  - a runner 8B/14B/27B/30B/32B llama-server up            -> all filler instances OFF (instantly)
#  - the set of runner GPU processes changes (server/fine-tune start or stop) -> all OFF, then wait STABLE_S before refilling
#  - runner active (small server or a non-llama GPU app)   -> up to 3 x 4B Q4 (-np 16, ~4.3 GB each) while free VRAM >= 7000 MiB per new one
#  - runner idle >= IDLE_S                                  -> up to 2 x 14B Q4 (-np 16, ~11 GB each) while free VRAM >= 14000 MiB per new one
#  - free VRAM < MINFREE while instances run                -> drop the newest instance
# STOP file ends it. Polls every 0.5 s. CPU servers (-ngl 0) and the teacher's ports 1830x are not runner activity.
D=/root/util3; cd $D; touch watch.on
M4=/workspace/nupen/models/Qwen3-4B-Q4_K_M.gguf
M14=/workspace/nupen/models/Qwen3-14B-Q4_K_M.gguf
BASE=18350; IDLE_S=${IDLE_S:-240}; STABLE_S=${STABLE_S:-20}; MINFREE=${MINFREE:-2000}
pids=(); mode=off; idle_since=$(date +%s); stable_since=$(date +%s); lastsig=""; last_start=0
log(){ echo "$(date +%T) $*" >> watch.log; }
writemode(){ { echo "$mode ${#pids[@]}"; } > mode.txt; }
killone(){ local p=${pids[-1]}; kill $p 2>/dev/null; for i in $(seq 1 20); do kill -0 $p 2>/dev/null || break; sleep 0.1; done; kill -9 $p 2>/dev/null; unset 'pids[-1]'; log "instance $((${#pids[@]})) ($mode) off ($1)"; }
killall_(){ [ ${#pids[@]} -gt 0 ] || return; while [ ${#pids[@]} -gt 0 ]; do killone "$1"; done; mode=off; writemode; }
start(){ # $1 = 4b|14b
  local i=${#pids[@]} port=$((BASE+${#pids[@]})) m a
  if [ "$1" = 14b ]; then m=$M14; else m=$M4; fi; a="-np 16 -c 16384"
  [ -f "$m" ] || { log "missing $m"; return; }
  nohup /opt/llama.cpp/llama-server -m $m --host 127.0.0.1 --port $port -ngl 99 -fa on -ctk q8_0 -ctv q8_0 $a -t ${THREADS:-4} --no-webui --metrics > srv_$port.log 2>&1 &
  local p=$!; pids+=($p); mode=$1; last_start=$(date +%s)
  for k in $(seq 1 120); do
    curl -sf -o /dev/null localhost:$port/health && { log "instance $i ($1) up on $port pid $p"; writemode; return; }
    kill -0 $p 2>/dev/null || { log "instance $i ($1) died at start"; unset 'pids[-1]'; writemode; return; }
    sleep 0.5
  done
}
writemode
while [ ! -f STOP ]; do
  # forget instances that exited on their own (only the newest can be dropped cleanly; restart all if an older one died)
  for p in "${pids[@]}"; do kill -0 $p 2>/dev/null || { log "an instance exited"; killall_ "instance exited"; break; }; done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mine="${pids[*]}"
  others=$(pgrep -af "llama-server" | grep -v -E -- "--port 1835[0-9] " | grep -v -- "-ngl 0" | grep -v -E -- "--port 1830[0-9] " | grep -v pgrep)
  big=$(echo "$others" | grep -c -E "8B|14B|27B|30B|32B")
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep . | while read q; do case " $mine " in *" $q "*) ;; *) echo $q;; esac; done | sort | tr '\n' ,)
  sig="$(echo "$others" | awk '{print $1}' | sort | tr '\n' ,)|$apps"
  now=$(date +%s)
  [ "$sig" != "|" ] && idle_since=$now
  if [ "$sig" != "$lastsig" ]; then
    [ -n "$lastsig" ] && killall_ "runner change"; lastsig=$sig; stable_since=$now
  fi
  if [ "$big" -gt 0 ]; then killall_ "runner big server"
  elif [ ${#pids[@]} -gt 0 ] && [ "${free:-0}" -lt $MINFREE ]; then killone "free ${free} MiB"; writemode
  elif [ $((now-stable_since)) -ge $STABLE_S ] && [ $((now-last_start)) -ge 5 ]; then
    want=4b; [ "$sig" = "|" ] && [ $((now-idle_since)) -ge $IDLE_S ] && want=14b
    if [ "$mode" != off ] && [ "$mode" != "$want" ]; then killall_ "switch to $want"
    elif [ "$want" = 4b ] && [ ${#pids[@]} -lt 3 ] && [ "${free:-0}" -ge 7000 ]; then start 4b
    elif [ "$want" = 14b ] && [ ${#pids[@]} -lt 2 ] && [ "${free:-0}" -ge 14000 ]; then start 14b
    fi
  fi
  sleep 0.5
done
killall_ "STOP file"; rm -f watch.on
