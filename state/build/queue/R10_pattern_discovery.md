# R10_pattern_discovery  [model: sonnet]
C66 §13 (4,000-6,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/discovery.py, engine/research/discovery_sources.py; tests/test_research_discovery.py.
Build on (import, extend - never copy): engine/patterns.py PatternMiner (EXTEND via wrapper), pattern_stats.py, pattern_identity.py, pattern_memory.py, engine/learning/knowledge.py (every discovery becomes a KnowledgeObject), antioverfit.py.

Build: Continual discovery over every §13 source family, each discovered pattern a formal knowledge object with all §13 fields (definition, provenance, discovery timestamp, training/validation periods, evidence + independent evidence counts, truth probability, reliability, transfer confidence, context dependence, failure conditions, counterexamples, complexity, decision impact). Cumulative multiple-testing accounting across all discovery runs; nothing trusted because it was mined.
