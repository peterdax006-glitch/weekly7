# R20_brain_health_diversity  [model: sonnet]
C66 §37 research-brain health (1,200-2,000), §38 diversity controller (1,000-1,800).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/brain_health.py, engine/research/diversity.py; tests/test_research_health.py.
Build on (import, extend - never copy): engine/learning/research_policy.py (exploration/exploitation), meta_learning.py, health.py.

Build: Monitor research diversity, duplicate rate, success rate, FDR, replication rate, transfer rate, compute efficiency, knowledge churn, overfitting and memorisation rates, concentration; detect 90%-of-compute-on-a-tiny-family, abandoning hard questions, and easy-area bias; adaptive allocation across the §38 areas learned from where exploration pays.
