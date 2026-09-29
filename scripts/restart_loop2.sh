#!/bin/bash
# rebuild archived weekly snapshots (stable codes, no insider data, with big-mover probability), then relaunch Test
cd /c/Users/Peter/weekly7
export PYTHONIOENCODING=utf-8 SEC_CONTACT=peterdax006@gmail.com
for i in 0 1 2 3; do .venv/Scripts/python -u scripts/regen_weekly_snaps.py $i 4 > data/regen_$i.log 2>&1 & done
wait
echo "regen done $(date +%T)"
.venv/Scripts/python -u scripts/livesim_loop2.py 60 > data/loop2.log 2>&1
echo "loop2 exit=$?"
