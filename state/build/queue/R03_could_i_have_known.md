# R03_could_i_have_known  [model: sonnet]
C66 §8 (2,000-3,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/counterfactual.py; tests/test_research_counterfactual.py.
Build on (import, extend - never copy): engine/pit.py, engine/livesim.py Feed (hardened), engine/learning/curator.py, situation.py, archive.py, knowability (R02 - duck-type its outputs; it is being written now).

Build: Reconstruct the exact information state at a decision timestamp (features, patterns, memory, macro, market state, events, price, volume, technical structure, cross-section) and compare with what became known later; emit knowledge_state_at_decision, future_information_used_by_auditor, information_that_would_have_been_available, information_that_was_unavailable, confidence_in_classification. The classification must never become a hidden future signal (test: feeding it back is refused by the gate).
