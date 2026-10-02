#!/bin/bash
# wait for any running rebuild, rebuild what is missing, archive opens (C33), then relaunch Test
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 SEC_CONTACT=peterdax006@gmail.com
while powershell -NoProfile -Command "exit [int](-not (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'regen_weekly_snaps' }))"; do sleep 20; done
for i in 0 1 2 3; do .venv/Scripts/python -u scripts/regen_weekly_snaps.py $i 4 > data/regen_b$i.log 2>&1 & done
wait
.venv/Scripts/python -u scripts/archive_opens.py
echo "regen done $(date +%T)"
.venv/Scripts/python -u scripts/livesim_loop2.py 60 > data/loop2.log 2>&1
echo "loop2 exit=$?"
