#!/bin/bash
# util3: every 10 min one line - GPU average over the last 600 one-second samples, filler mode, banked util3 rows, runner phase.
S="ssh -i $HOME/.ssh/nupen_vast -p $POD_PORT -o BatchMode=yes -o LogLevel=ERROR root@$POD_HOST"
while true; do
  sleep 600
  r=$($S 'cd /root/util3; tail -600 gpu.csv | awk -F", " "{gsub(/ %/,\"\",\$2); s+=\$2; n++} END {printf \"%.1f n=%d\", s/n, n}"; echo -n " mode=$(cat mode.txt | tr " " _)"; echo -n " srv="; pgrep -af llama-server | grep -v -- "-ngl 0" | grep -o -E "[A-Za-z0-9.]+-[0-9.]+B[^/ ]*gguf" | tr "\n" ,' 2>/dev/null || echo "ssh-fail")
  rows=$(grep -c '"gpu_pulse": "util3-' C:/Users/peter/weekly7/state/creator/thinking/trace_bank.jsonl 2>/dev/null)
  echo "$(date +%H:%M) avg=$r util3_rows=$rows chain=$(tail -1 C:/Users/peter/creator_runtime/gpu/day2b_chain.out | cut -c1-60)"
done
