#!/bin/bash
# After every simulation round: rebuild the Pattern Explorer data from revealed years and publish it.
cd "$(dirname "$0")/.."
last=""
while true; do
  cur=$(stat -c %Y state/livesim/cycles.json 2>/dev/null)
  if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
    last=$cur
    PYTHONIOENCODING=utf-8 .venv/Scripts/python -u scripts/patterns.py > data/patterns.log 2>&1 && \
      git add docs/patterns.json docs/patterns.html docs/index.html && \
      git commit -q -m "Pattern Explorer: refresh after simulation round" && \
      git pull -q --rebase --autostash && git push -q && echo "published $(date +%T)"
  fi
  grep -q "livesim exit\|TARGET REACHED\|stopping" data/livesim.log 2>/dev/null && [ "$cur" == "$last" ] && break
  sleep 60
done
