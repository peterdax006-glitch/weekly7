#!/bin/bash
# GPU GUARDIAN (h64; owner 3 Oct: "average 90%+ GPU", "figure out why it keeps going down and patch it"). Lives entirely on the pod:
# cron (every minute, flock) -> run.sh (respawns this script in 3 s) -> guard.sh. No dependence on the PC, SSH or an agent.
#  - every second: GPU util + free VRAM + the runner's GPU processes
#  - a runner 8B/14B/27B/30B/32B llama-server up                 -> every filler instance OFF at once (the 27B / 14B tiers own the card)
#  - a NEW runner GPU process appears (server or fine-tune start) -> every filler OFF, refill after STABLE_S; one that only goes away
#    (a job gap: fetch, download, upload, PC-side merge/eval) keeps ours running and lets us scale UP
#  - util < 90% over the last 15 s and free VRAM >= need + margin -> one more instance: the best item of the WORK QUEUE (queue.txt, by
#    priority: efficiency jobs, coding, language, traces last) whose model fits; up to 4. A job outranks and replaces a traces instance.
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
declare -A ftpeak ftkey
[ -s $D/queue.txt ] || printf "8 Qwen3-4B-Q4_K_M.gguf traces
9 Qwen3-1.7B-Q4_K_M.gguf traces
" > $D/queue.txt   # traces fallback without a PC
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
# ---- the work queue (owner 3 Oct: run the unfinished work by priority, 1 efficiency, 2 coding, 3 language; traces last)
# queue.txt lines: '<prio> <model file> <tag> [<bytes> <sha256> <url>]'. A non-'traces' line is one job: served by at most one instance, driven
# from the PC (guardian_queue.py), removed from the file when done (its instance is then stopped). 'traces' lines may fill any number of
# instances (the pod feeder drives them). Missing model files are fetched in the background (sha256-checked) while disk stays >= 6 GB.
fetch_bg(){ # $1 file $2 bytes $3 sha $4 url
  local f=$MODELS/$1; [ -f $D/fetch_$1.on ] && return; [ -n "$4" ] || return
  local avail=$(df --output=avail -m / | tail -1 | tr -d ' '); [ $((avail - $2/1048576)) -lt 6144 ] && return
  touch $D/fetch_$1.on; log "fetching $1"
  ( nice -n 10 curl -fL --retry 5 -s -o $f.part "$4" && [ "$(sha256sum $f.part | cut -d' ' -f1)" = "$3" ] && mv -f $f.part $f && log "fetched $1" \
      || { rm -f $f.part; log "fetch of $1 failed"; }; rm -f $D/fetch_$1.on ) &
}
pick(){ # echo the model file of the best queue item that fits $1 MiB of VRAM (after margin), or nothing
  local room=$1
  sort -n $D/queue.txt 2>/dev/null | while read prio file tag bytes sha url; do
    [ -z "$file" ] && continue
    [ "$tag" != traces ] && ! pcalive && continue
    [ -f $MODELS/$file ] || { [ "$tag" != traces ] && fetch_bg "$file" "${bytes:-0}" "$sha" "$url"; continue; }
    [ "$tag" != traces ] && [[ " ${kinds[*]} " == *" $file "* ]] && continue
    local need=$(( $(stat -c %s $MODELS/$file) / 1048576 + 1900 ))
    [ $room -ge $need ] && { echo $file; break; }
  done
}
inqueue(){ awk -v f="$1" '$2==f {found=1} END {exit !found}' $D/queue.txt 2>/dev/null; }
pcalive(){ [ -n "$(find $D/pc_heartbeat -mmin -5 2>/dev/null)" ]; }   # queued JOBS are driven from the PC: serve them only while it is alive
istraces(){ awk -v f="$1" '$2==f && $3=="traces" {t=1} END {exit !t}' $D/queue.txt 2>/dev/null; }
start(){ # $1 = model file under $MODELS
  local port=$((BASE+${#pids[@]})) m=$MODELS/$1
  [ -f "$m" ] || { log "missing $m"; return; }
  nohup /opt/llama.cpp/llama-server -m $m --host 127.0.0.1 --port $port -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 16 -c 16384 -t 2 -tb 2 \
    --no-webui --metrics > $D/srv_$port.log 2>&1 &
  local p=$!; pids+=($p); kinds+=($1); mode=on; last_start=$(date +%s)
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
    inqueue "$(basename "$f" .part)" && continue                        # a queued job's model (or its download) stays
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
  # a fine-tune grows while it loads and trains: reserve its MEASURED peak + 2 GB (peaks learned per training data path in ftpeaks.txt;
  # an unseen kind gets 16 GB, the 1.7B LoRA runs measured 15-16 GB), minus what it already holds
  ftused=0; res=0
  while IFS=', ' read q mem; do [ -z "$q" ] && continue; case "$mine$fill" in *" $q "*) continue;; esac
    case "$allllama" in *",$q,"*) ;; *) nonllama=1; mem=${mem:-0}; ftused=$((ftused + mem))
      key=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null | grep -o -E -- '--data [^ ]+|^[^ ]+ [^ ]+' | head -1 | tr ' /' '__')
      [ -z "${ftkey[$q]}" ] && ftkey[$q]=${key:-unknown}
      [ "$mem" -gt "${ftpeak[$q]:-0}" ] && ftpeak[$q]=$mem
      pk=$(awk -v k="${ftkey[$q]}" '$1==k {m=$2} END {print m}' $D/ftpeaks.txt 2>/dev/null)      # peak of a COMPLETED run of this kind
      [ -z "$pk" ] && pk=${FT_DEFAULT:-16000}
      [ "${ftpeak[$q]}" -gt "$pk" ] && pk=${ftpeak[$q]}
      r=$((pk + 2000 - mem)); [ $r -gt 0 ] && res=$((res + r));;
    esac
  done <<< "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)"
  for q in "${!ftpeak[@]}"; do kill -0 $q 2>/dev/null && continue    # the run ended: remember its peak for the next run of that kind
    echo "${ftkey[$q]} ${ftpeak[$q]}" >> $D/ftpeaks.txt; log "fine-tune ${ftkey[$q]} ended, peak ${ftpeak[$q]} MiB"; unset "ftpeak[$q]" "ftkey[$q]"; done
  if [ $nonllama -eq 1 ]; then mf=1500; free=$(( ${free:-0} - res ))
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
  else
    # an instance whose queue job is finished (line removed) makes room for the next job
    for i in "${!kinds[@]}"; do { inqueue "${kinds[$i]}" && { istraces "${kinds[$i]}" || pcalive; }; } || { if [ $i -eq $((${#pids[@]}-1)) ]; then killone "job ${kinds[$i]} done"; else killall_ "job ${kinds[$i]} done"; fi; break; }; done
    # a job that outranks a running traces instance takes its place when it would fit in that instance's VRAM
    if [ ${#pids[@]} -gt 0 ] && istraces "${kinds[-1]}" && [ $((now-last_start)) -ge 4 ]; then
      j=$(pick $(( ${free:-0} - mf - 400 ))); if [ -z "$j" ] || istraces "$j" || [ ${#pids[@]} -ge 4 ]; then
        j2=$(pick $(( ${free:-0} + $(( $(stat -c %s $MODELS/${kinds[-1]}) / 1048576 + 1900 )) - mf - 400 )))
        [ -n "$j2" ] && ! istraces "$j2" && killone "make room for $j2"
      fi
    fi
    if [ $((now-stable_since)) -ge $STABLE_S ] && [ $((now-last_start)) -ge 4 ] && [ ${#pids[@]} -lt 4 ] && [ "$u15" -lt $UTIL_T ]; then
      j=$(pick $(( ${free:-0} - mf - 400 ))); [ -n "$j" ] && start $j
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
