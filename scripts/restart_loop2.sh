#!/bin/bash
# rebuild archived weekly snapshots WITHOUT insider data (C18 fail-safe), then relaunch the self-adjusting loop
cd /c/Users/Peter/weekly7
export PYTHONIOENCODING=utf-8 SEC_CONTACT=peterdax006@gmail.com
.venv/Scripts/python -u scripts/regen_weekly_snaps.py 0 2 > data/regen_0.log 2>&1 &
.venv/Scripts/python -u scripts/regen_weekly_snaps.py 1 2 > data/regen_1.log 2>&1 &
wait
echo "regen done $(date +%T)"
.venv/Scripts/python -u scripts/livesim_loop2.py 60 > data/loop2.log 2>&1
echo "loop2 exit=$?"
