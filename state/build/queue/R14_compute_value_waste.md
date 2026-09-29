# R14_compute_value_waste  [model: sonnet]
C66 §18 compute manager (1,500-2,500), §19 value accounting (1,200-2,000), §20 waste controller (800-1,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/compute_manager.py, engine/research/value_accounting.py, engine/research/waste.py; tests/test_research_compute.py.
Build on (import, extend - never copy): engine/learning/compute.py (workers, admission, OOM), checkpoints.py, engine/resources.py.

Build: Research states QUEUED..RETIRED and the 5-stage escalation ladder (no big compute on weak hypotheses; no abandoning a promising one after one underpowered cheap test); per-job realised value vector (§19) incl. 'this feature family is unreliable' as high value; progress-per-compute monitor moving branches to DORMANT with a reason and reviving on new data/representation/regime/evidence/hypothesis.
