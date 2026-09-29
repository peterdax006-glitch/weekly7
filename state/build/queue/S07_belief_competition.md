# S07_belief_competition
Contract sections 32, 33, 39, 40, 41. Checklist C10, C12, C13. Minimum lines: 800+600+700+500+700 = 3,300.
Read state/build/CONTEXT.md fully (rules 1-18) and SELF_LEARNING_CONTRACT.md sections 0-4, 58-62, 83, 85 plus the sections below, in full. C63: foundation first - all code + unit tests now, no real-data runs or tuning.
You own: engine/learning/belief.py, questions.py, competition.py, complexity.py, boundary.py; tests/test_learning_belief.py, test_learning_boundary.py.

Build: Evidence-weighted belief updates (prior, evidence strength, posterior, uncertainty, n, contradictions; never overwrite). Three separate answers per relation: real / useful / useful-now. Competing explanations kept alive until evidence separates them. Complexity penalty that prefers the simpler rule at equal OOS value but lets complexity earn its place. Boundary learning: learn where a pattern stops working (thresholds, transitions, disagreement, shocks, concentration) as knowledge; conditions and anti-conditions.
