# R05_winner_loser_research  [model: sonnet]
C66 §5 (2,500-4,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/winners_losers.py, engine/research/loss_pipeline.py; tests/test_research_winners_losers.py.
Build on (import, extend - never copy): engine/learning/failure.py, separation.py, postmortem.py, credit.py, engine/lessons.py.

Build: Winner research (why moved, why detected, why ranked, which signals contributed/irrelevant/generalised, magnitude + timing predictability) and a DEDICATED loss research pipeline asking every §5 loser question with the full cause list (selection, direction, timing, exit, risk, regime, data quality, external event, pattern failure, unknown). Losses studied at least as hard as winners (test: symmetric coverage).
