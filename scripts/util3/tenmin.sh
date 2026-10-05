#!/bin/bash
# util3: every 10 min one line - GPU average of the guardian's last 10 per-minute samples, its state, rows produced on the pod / banked on the PC.
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/nupen_vast -p $POD_PORT -o BatchMode=yes -o LogLevel=ERROR root@$POD_HOST"
while true; do
  sleep 600
  r=$($S 'tail -10 /root/guardian/log | awk "{for(i=1;i<=NF;i++) if (\$i ~ /^util60=/) {v=substr(\$i,8)+0; s+=v; n++}} END {printf \"%.1f n=%d\", s/n, n}"; echo -n " "; tail -1 /root/guardian/log | grep -o -E "filler=[^ ]*|rows=[0-9]*" | tr "\n" " "' 2>/dev/null || echo "ssh-fail")
  rows=$(grep -c -E '"gpu_pulse": "(util3|guardian)-' $HOME/weekly7/state/creator/thinking/trace_bank.jsonl 2>/dev/null)
  echo "$(date +%H:%M) avg=$r banked=$rows" >> $HOME/creator_runtime/util3/meter.log
done
