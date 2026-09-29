# R15_priority_targets_questions  [model: sonnet]
C66 §2 priority engine (1,500-2,500), §22 daily target generator (1,500-2,500), §40 question generator (1,200-2,000), §41 hypothesis tree (1,000-1,800).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/priority.py, engine/research/targets.py, engine/research/questions.py, engine/research/hypothesis_tree.py; tests/test_research_priority.py.
Build on (import, extend - never copy): engine/learning/research_priority.py, research_policy.py (EXTEND - they exist), interpretation.py, questions.py, unknowns.py.

Build: A learned, validated research-priority model over the §2 value vector (loss-reduction and volatility value weighted per §34), not a fixed formula; the daily generator from every §22 source (with the dispersion example as a test); the question generator producing §40 research objects; the hypothesis tree (question -> hypotheses -> tests/counterexamples, branch death redirects research). Planted tests for §46 G (big-information experiment A beats tiny-gain B) and H (loss-avoidance question outranks many tiny winner gains).
