#!/bin/bash
cd /c/Users/Peter/weekly7
run() { .venv/Scripts/python -u scripts/experiment.py $1 > data/exp_$1.log 2>&1; echo "done $1 exit=$?"; }
.venv/Scripts/python -u scripts/frontier.py > data/frontier2.log 2>&1; echo "done frontier exit=$?"
run v9_goal_nostop & run v10_goal_stop3 & wait
run v11_growth_nostop & run v12_goal_stop2 & wait
run controls
echo ALLDONE
