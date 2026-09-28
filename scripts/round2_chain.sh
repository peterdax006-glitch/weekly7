#!/bin/bash
# Round 2 (canon C7/C8): extended history -> walk-forward from 2008 -> 150 configs -> fresh-era exam 2008-2016
cd /c/Users/Peter/weekly7
until grep -q "extend exit" data/extend.log; do sleep 30; done
grep -q "extend exit=0" data/extend.log || { echo "EXTEND FAILED"; exit 1; }
export SEC_CONTACT=peterdax006@gmail.com W7_SUFFIX=_ext
W7_PANEL_START=2002-01-01 W7_FIRST_YEAR=2008 .venv/Scripts/python -u scripts/research.py > data/research_ext.log 2>&1 || { echo "RESEARCH FAILED"; exit 1; }
echo "research done"
for i in 0 1 2; do W7_ROUND=2 .venv/Scripts/python -u scripts/tuning_lab.py $i 3 150 2026 > data/tune2_$i.log 2>&1 & done; wait
echo "lab done"
W7_LOCK_YEARS=2008-2016 W7_PRIOR_TRIALS=120 W7_CHAMPION=CHAMPION_v1.2 .venv/Scripts/python -u scripts/tuning_report.py 400 > data/round2_report.log 2>&1
echo "ROUND2 DONE"
