# R16_scales_crosssection_regimes  [model: sonnet]
C66 §23 multi-scale (1,200-2,000), §24 cross-sectional (1,500-2,500), §25 regime-aware (1,500-2,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/multiscale.py, engine/research/cross_section.py, engine/research/regimes.py; tests/test_research_scales.py.
Build on (import, extend - never copy): engine/features.py, engine/learning/context.py, situation.py, hierarchy.py, the minute collector outputs (intraday data only where it exists).

Build: Explicit horizon on every pattern; no assumed transfer between timescales (tested); stock vs sector/industry/market/vol/liquidity/momentum/event cohort decomposition -> stock-specific / sector / market / regime-wide labels feeding both models; regime monitor for every §25 regime plus DISCOVERED regimes (unsupervised, past-only) and pattern x regime learning.
