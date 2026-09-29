# R18_graph_bridge_memory  [model: sonnet]
C66 §27 knowledge graph (1,500-2,500), §28 knowledge-to-decision bridge (1,000-1,800), §39 scientific long-term memory (1,500-2,500).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/research_graph.py, engine/research/decision_bridge.py, engine/research/science_memory.py; tests/test_research_graph.py.
Build on (import, extend - never copy): engine/learning/knowledge_graph.py, archive.py, decision_contract.py, knowledge.py (EXTEND, never a second graph/store).

Build: Graph nodes/edges for patterns, features, sectors, regimes, events, outcomes, failures, experiments, learners, questions on top of the existing graph; discovery of unanswered relationships; the bridge forcing every discovery to state what decision or research priority it changes (else recorded as informational); the scientific history answering 'why do you believe this?' and 'what evidence would stop you trusting it?' per item.
