# R13_meta_learning  [model: sonnet]
C66 §16 (2,500-4,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/meta_research.py; tests/test_research_meta.py.
Build on (import, extend - never copy): engine/learning/meta_learning.py (EXTEND), experiment_memory.py, failed_learners.py, research_priority.py.

Build: Meta-learning about RESEARCH itself: every §16 question (durable-knowledge experiment types, overfitting types, transferring families, failing representations, datasets that create false discoveries, validation methods that catch most errors, decision-changing questions, informative failures, compute-wasting paths, era/stock/regime survival) evaluated out of sample; outputs feed the scheduler; never trains on its own future evaluation results (test it).
