# S05_failure_learning
Contract sections 9, 22, 23, 24. Checklist D01-D10, D13, D14, C09. Minimum lines: 1,100+1,000+700+800 = 3,600.
Read state/build/CONTEXT.md fully (rules 1-18) and SELF_LEARNING_CONTRACT.md sections 0-4, 58-62, 83, 85 plus the sections below, in full. C63: foundation first - all code + unit tests now, no real-data runs or tuning.
You own: engine/learning/failure.py, missed_winners.py, separation.py, postmortem.py; tests/test_learning_failure.py, test_learning_missed.py.

Build: Loss classifier over all FailureCause values with evidence per cause and UNKNOWN when evidence is insufficient (never forced). Detectors for selection, timing, direction, risk, pattern, context, regime and measurement failure; subsystem attribution so a well-selected but badly timed trade teaches TIMING, not SELECTION. Structured postmortem with every section-24 field; it only emits HYPOTHESES, never changes production. Missed-winner learning: why rejected, and what distinction separates missed winners from false positives, with an out-of-sample check hook. Extend engine/missed_winners.py and engine/lessons.py rather than copying.
