#!/bin/bash
# GPU guardian keeper: cron starts this every minute under flock (one copy); it respawns guard.sh 3 s after any exit. STOP file = stay down.
D=/root/guardian
while [ ! -f $D/STOP ]; do
  $D/guard.sh >> $D/guard.out 2>&1
  echo "$(date '+%F %T') guard.sh exited ($?) - respawning" >> $D/events.log
  sleep 3
done
