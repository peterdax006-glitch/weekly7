# R11_break_research  [model: sonnet]
C66 §14 (2,000-3,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/break_research.py; tests/test_research_breaks.py.
Build on (import, extend - never copy): engine/learning/break_detection.py, reliability.py, lifecycle.py, health.py, engine/pattern_reliability.py (B27) - EXTEND.

Build: Turn every pattern break into a scheduled investigation (a ResearchQuestion + hypotheses from the §14 cause list), with the six §14 questions as tests (pre-existing predictor? transfers OOS? predicts future breaks? gating improves performance? reduces losses? beats random explanations?) and UNKNOWN when not. Output feeds the research queue, not the trader directly.
