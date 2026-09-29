# R12_experiments_failed_learners  [model: sonnet]
C66 §15 experiment memory (1,500-2,500), §17 failed-learner lab (1,200-2,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/experiments.py, engine/research/failed_lab.py; tests/test_research_experiments.py.
Build on (import, extend - never copy): engine/learning/experiment_memory.py (the ONE facade - extend it), failed_learners.py, engine/registry.py, engine/repro.py.

Build: Every §15 field recorded; 'have we done this / did it answer / justified replication / what differs' before launch; the failed-learner lab records every §17 field and answers 'this class attempted N times: k overfit, ...; new attempt needs a materially different hypothesis' by learner CLASS similarity.
