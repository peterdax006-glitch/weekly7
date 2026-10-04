#!/bin/bash
# GPU GUARDIAN (h64; owner 3 Oct: "average 90%+ GPU", "figure out why it keeps going down and patch it"). Lives entirely on the pod:
# cron (every minute, flock) -> run.sh (respawns this script in 3 s) -> guard.sh. No dependence on the PC, SSH or an agent.
#  - every second: GPU util + free VRAM + the runner's GPU processes
#  - a runner 8B/14B/27B/30B/32B llama-server up                 -> every filler instance OFF at once (the 27B / 14B tiers own the card)
#  - a NEW runner GPU process appears (server or fine-tune start) -> every filler OFF, refill after STABLE_S; one that only goes away
#    (a job gap: fetch, download, upload, PC-side merge/eval) keeps ours running and lets us scale UP
#  - util < 90% over the last 15 s and free VRAM >= need + margin -> one more instance (4B Q4 ~4.3 GB; a 1.7B Q4 ~2.4 GB where a 4B does
#    not fit; 2 x 14B Q4 ~11 GB when the runner has been idle IDLE_S); up to 4 (2 for 14B)
#  - free VRAM < margin                                          -> drop the newest instance (margin 1.5 GB beside llama-servers,
#    2.5 GB beside a fine-tune, which can grow)
#  - feed.py (question feeder) kept running; filler servers -t 2 (pod CPU: filler work stays <= ~6 cores)
#  - disk guard: keep / >= 6 GB free by deleting stale ladder variants / .part files / own regenerable chunks (never the HF caches - a fine-tune
#    re-downloading its base idles the GPU ~10 min -, never the
#    base Q4 GGUFs, /dev/shm, files any process has open, budget guard or runner files)
#  - one log line per minute to /root/guardian/log
# STOP file: everything off and the guardian stays down (run.sh honours it too).
D=/root/guardian; cd $D || exit 1
MODELS=/workspace/nupen/models
M4=$MODELS/Qwen3-4B-Q4_K_M.gguf; M17=$MODELS/Qwen3-1.7B-Q4_K_M.gguf; M14=$MODELS/Qwen3-14B-Q4_K_M.gguf
BASE=18350; IDLE_S=${IDLE_S:-240}; STABLE_S=${STABLE_S:-10}; UTIL_T=${UTIL_T:-90}
pids=(); kinds=(); mode=off; now=$(date +%s); idle_since=$now; stable_since=$now; lastsig=""; last_start=0; utils=(); lastmin=0; lastdisk=0
log(){ echo "$(date '+%F %T') $*" >> $D/events.log; }
# a restarted guardian adopts nothing: clear filler servers a dead predecessor left (only ports 18350-18353, only llama-server)
for p in $(pgrep -f "llama-server .*--port 1835[0-3] "); do kill $p 2>/dev/null; done; sleep 1
killone(){ local p=${pids[-1]}; kill $p 2>/dev/null; for i in $(seq 1 20); do kill -0 $p 2>/dev/null || break; sleep 0.1; done; kill -9 $p 2>/dev/null
  unset 'pids[-1]'; unset 'kinds[-1]'; log "instance ${#pids[@]} off ($1)"; [ ${#pids[@]} -eq 0 ] && mode=off; }
killall_(){ # all at once: TERM every instance, 1 s grace, then KILL (a runner server loading must not wait on us)
  [ ${#pids[@]} -gt 0 ] || return 0; kill ${pids[@]} 2>/dev/null
  for i in $(seq 1 10); do local alive=0; for p in ${pids[@]}; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive -eq 0 ] && break; sleep 0.1; done
  kill -9 ${pids[@]} 2>/dev/null; log "all ${#pids[@]} instances off ($1)"; pids=(); kinds=(); mode=off; }
start(){ # $1 = 4b|17b|14b
  local port=$((BASE+${#pids[@]})) m
  case $1 in 14b) m=$M14;; 17b) m=$M17;; *) m=$M4;; esac
  [ -f "$m" ] || { log "missing $m"; return; }
  nohup /opt/llama.cpp/llama-server -m $m --host 127.0.0.1 --port $port -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 16 -c 16384 -t 2 -tb 2 \
    --no-webui --metrics > $D/srv_$port.log 2>&1 &
  local p=$!; pids+=($p); kinds+=($1); [ $1 = 14b ] && mode=14b || mode=4b; last_start=$(date +%s)
  for k in $(seq 1 120); do
    curl -sf -o /dev/null localhost:$port/health && { log "instance $((${#pids[@]}-1)) ($1) up on $port"; return; }
    kill -0 $p 2>/dev/null || { log "instance ($1) died at start: $(tail -2 $D/srv_$port.log | tr '\n' ' ' | cut -c1-200)"; unset 'pids[-1]'; unset 'kinds[-1]'; return; }
    sleep 0.5
  done
}
diskguard(){
  local free=$(df --output=avail -m / | tail -1 | tr -d ' ')
  [ "$free" -ge 6144 ] && return
  local open=$(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -o '/[^ ]*' ; cat /proc/[0-9]*/maps 2>/dev/null | awk '{print $6}' | grep '^/' | sort -u)
  for f in $(find $MODELS -maxdepth 1 \( -name '*.part' -o -name 'Qwen3-*-Q8_0.gguf*' -o -name 'Qwen3-*-Q3_K_M.gguf*' -o -name 'Qwen3-*-Q5_K_M.gguf*' \) -mmin +60 2>/dev/null) \
\
           $(ls /root/pubembed*/chunks.jsonl 2>/dev/null); do
    case "$f" in $M4|$M17|$M14) continue;; esac
    echo "$open" | grep -q -F "$f" && continue
    log "disk guard: ${free} MiB free, deleting $f ($(du -sm "$f" | cut -f1) MiB)"; rm -rf -- "$f"
    free=$(df --output=avail -m / | tail -1 | tr -d ' '); [ "$free" -ge 6144 ] && break
  done
}
while [ ! -f $D/STOP ]; do
  for i in "${!pids[@]}"; do kill -0 ${pids[$i]} 2>/dev/null || { log "an instance exited"; killall_ "instance exited"; break; }; done
  pgrep -f "guardian/feed.py" > /dev/null || { nohup nice -n 5 python3 $D/feed.py >> $D/feed.out 2>&1 & log "feed started"; }
  read util free <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -1 | tr -d ',')"
  utils+=(${util:-0}); [ ${#utils[@]} -gt 60 ] && utils=("${utils[@]:1}")
  u15=$(printf '%s\n' "${utils[@]: -15}" | awk '{s+=$1;n++} END {print (n? int(s/n) : 0)}')
  mine=" ${pids[*]} "
  # ports 18350-18353 belong to the guardian: a filler server it does not track (a dead predecessor's) is killed, never counted as runner
  for q in $(pgrep -f "llama-server .*--port 1835[0-3] "); do case "$mine" in *" $q "*) ;; *) kill $q 2>/dev/null; log "killed untracked filler $q";; esac; done
  fill=" $(pgrep -f 'llama-server .*--port 1835[0-3] ' | tr '
' ' ') "
  others=$(pgrep -af "llama-server" | grep -v -E -- "--port 1835[0-9] " | grep -v -- "-ngl 0" | grep -v -E -- "--port 1830[0-9] " | grep -v pgrep)
  big=$(echo "$others" | grep -c -E "8B|14B|27B|30B|32B")
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep . | while read q; do case "$mine$fill" in *" $q "*) ;; *) echo $q;; esac; done | sort | tr '\n' ,)
  sig="$(echo "$others" | awk '{print $1}' | grep . | sort | tr '\n' ,)|$apps"
  allllama=",$(pgrep -f llama-server | tr '\n' ,)"; nonllama=0
  ftused=0
  while IFS=', ' read q mem; do [ -z "$q" ] && continue; case "$mine$fill" in *" $q "*) continue;; esac
    case "$allllama" in *",$q,"*) ;; *) nonllama=1; ftused=$((ftused + ${mem:-0}));; esac
  done <<< "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)"
  if [ $nonllama -eq 1 ]; then
    # a fine-tune grows while it loads and trains (the 1.7B trainmix run peaked ~20.7 GB): reserve FT_RESERVE for it, not what it holds now
    mf=2500; res=$(( ${FT_RESERVE:-22000} - ftused )); [ $res -lt 0 ] && res=0; free=$(( ${free:-0} - res ))
  else mf=1500; fi
  now=$(date +%s)
  [ "$sig" != "|" ] && idle_since=$now
  if [ "$sig" != "$lastsig" ]; then
    new=$(echo "${sig//|/,}" | tr ',' ' ' | xargs -n1 2>/dev/null | while read q; do case ",${lastsig//|/,}," in *",$q,"*) ;; *) echo $q;; esac; done | head -1)
    if [ -n "$lastsig" ] && [ -n "$new" ]; then
      killall_ "runner change"; [ $stable_since -lt $now ] && stable_since=$now   # never shortens a fine-tune hold
      # a new non-llama GPU process (a fine-tune) grows for a minute or two while it loads: wait longer before refilling
      case "$allllama" in *",$new,"*) ;; *) stable_since=$((now + ${FT_WAIT:-120})); log "new non-llama GPU process $new: refill held $((${FT_WAIT:-120} + STABLE_S)) s";; esac
    fi
    lastsig=$sig
  fi
  if [ "$big" -gt 0 ]; then killall_ "runner big server"
  elif [ ${#pids[@]} -gt 0 ] && [ "${free:-0}" -lt $mf ]; then killone "free ${free} MiB"
  elif [ $((now-stable_since)) -ge $STABLE_S ] && [ $((now-last_start)) -ge 4 ] && [ "$u15" -lt $UTIL_T ]; then
    want=4b; [ "$sig" = "|" ] && [ $((now-idle_since)) -ge $IDLE_S ] && want=14b
    if [ "$mode" != off ] && [ "$mode" != "$want" ]; then killall_ "switch to $want"
    elif [ $want = 14b ] && [ ${#pids[@]} -lt 2 ] && [ "${free:-0}" -ge $((11000 + mf + 400)) ]; then start 14b
    elif [ $want = 4b ] && [ ${#pids[@]} -lt 4 ] && [ "${free:-0}" -ge $((4300 + mf + 400)) ]; then start 4b
    elif [ $want = 4b ] && [ ${#pids[@]} -lt 4 ] && [ "${free:-0}" -ge $((2400 + mf + 400)) ]; then start 17b
    fi
  fi
  if [ $((now-lastmin)) -ge 60 ]; then
    lastmin=$now
    rows=$(cat $D/out/*.jsonl 2>/dev/null | wc -l)
    echo "$(date '+%F %T') util60=$(printf '%s\n' "${utils[@]}" | awk '{s+=$1;n++} END {print int(s/n)}')% free=${free}MiB filler=${mode}x${#pids[@]}[${kinds[*]}] runner=${sig} big=$big disk=$(df --output=avail -m / | tail -1 | tr -d ' ')MiB rows=$rows" >> $D/log
  fi
  if [ $((now-lastdisk)) -ge 60 ]; then lastdisk=$now; diskguard; fi
  sleep 1
done
killall_ "STOP file"
