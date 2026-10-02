#!/bin/bash
# run one movers variant over windows m01..m13; $1 = tag, $2 = variant json
cd "$(dirname "$0")/.."
for i in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  PYTHONIOENCODING=utf-8 .venv/Scripts/python -u scripts/movers.py run m$i "$2" > /dev/null 2>&1
done
echo "variant $1 done"
