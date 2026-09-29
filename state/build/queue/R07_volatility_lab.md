# R07_volatility_lab  [model: sonnet]
C66 §9 (3,000-4,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/volatility_lab.py, engine/research/vol_hypotheses.py; tests/test_research_volatility.py.
Build on (import, extend - never copy): engine/fv_pipeline.py, scripts/movers.py logic, engine/features.py, engine/candles.py, engine/analogs.py, engine/edgar.py (insider/13D), engine/patterns.py, engine/learning/transfer.py, competition.py.

Build: Volatility as its own scientific programme: competing hypotheses H1-H9 (clustering, event anticipation, liquidity shock, cross-sectional movement, regime-dependence, compression->expansion, volume->movement, interactions, unknown) with room for discovered H10+; each §9 research question as a runnable study (walk-forward, past-only) with transfer across eras/sectors/stocks/regimes; calibrated P(volatility) and magnitude + timing; flags signals that find volatility without direction.
