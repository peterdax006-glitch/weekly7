# R08_direction_lab  [model: sonnet]
C66 §10 (3,000-4,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/direction_lab.py; tests/test_research_direction.py.
Build on (import, extend - never copy): engine/direction.py, direction_features.py, direction_calib.py, direction_ablate.py, engine/fv_pipeline.py, engine/learning/calibration.py.

Build: Direction ONLY on the universe the volatility model would actually select (conditional pipeline universe -> P(vol) -> predicted movers -> P(up|mover)). Every §10 hypothesis family as a candidate; each must beat base rate, random, shuffled labels, no-learning, a simple baseline and the existing model; the honest 'no reliable signal' outcome is first-class.
