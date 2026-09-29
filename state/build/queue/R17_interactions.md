# R17_interactions  [model: sonnet]
C66 §26 (2,000-3,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/interactions.py; tests/test_research_interactions.py.
Build on (import, extend - never copy): engine/patterns.py (pair rules), pattern_stats.py (BH, bonferroni), engine/antioverfit.py, engine/learning/complexity.py.

Build: Search every §26 interaction family with held-out validation, multiple-testing correction counting EVERY combination tried, fresh seeds, cross-year and cross-stock validation, shuffled controls and complexity penalties; planted real interaction found, planted noise-only search finds nothing.
