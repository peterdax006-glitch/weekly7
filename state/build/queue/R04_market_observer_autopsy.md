# R04_market_observer_autopsy  [model: sonnet]
C66 §4 (1,500-2,500), §21 daily autopsy (1,500-2,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/observer.py, engine/research/autopsy.py; tests/test_research_observer.py.
Build on (import, extend - never copy): engine/learning/missed_winners.py, failure.py, surprise.py, engine/missed_winners.py, engine/fv_pipeline.py, engine/features.py.

Build: Every simulated day: a complete research record over the WHOLE eligible universe with every §4 category (considered high/medium, rejected, abstentions, low-confidence, winners, losers, extreme up/down, predictable vs unpredictable movers, near misses, false positives, false negatives). Vectorised: must handle ~3,000 names/day cheaply (measure it). The daily autopsy (market / model / research / risk / learning sections of §21) built from that record, feeding research questions.
