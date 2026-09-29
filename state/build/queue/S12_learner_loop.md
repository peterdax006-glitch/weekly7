# S12_learner_loop
Contract sections 3, 4, 66, 86, 87; checklist C01-C17 (the loop stages), L02 (legitimate learner frozen). Target: engine/learning/learner.py 1,200-1,800 meaningful lines (ruler) - it is the conductor; mechanisms live in the modules it calls.
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 0-4, 58-62, 66, 83, 85-87. C63: code + unit tests now; no real-data runs.
You own: engine/learning/learner.py, tests/test_learning_learner.py. EXCEPTION to rule 15: you MAY import every engine/learning module (they are written now); read each module's public API before calling it. Do not edit them; list any API gap under INTEGRATION.

Build the complete section-4 loop as one class (LegitimateLearner) with one method per stage, each stage calling the real module:
OBSERVE (situation.py) -> DESCRIBE SITUATION (situation/context) -> RETRIEVE (retrieval.py, abstains without proven skill) -> ASSESS
RELIABILITY (calibration/temporal/retirement; break_detection/reliability/health from S15 when present, else pattern_reliability) ->
FORM EXPECTATIONS -> DECIDE (decision_contract: only production knowledge may act; unknowns -> ABSTAIN etc.) -> OBSERVE OUTCOME
(require_past: outcome strictly before the next `now`) -> MEASURE SURPRISE (surprise.py) -> ASSIGN CREDIT/BLAME (credit.py,
separation.py) -> UPDATE BELIEFS (belief.py) -> INVESTIGATE FAILURE (failure.py, postmortem.py, missed_winners.py) -> LEARN
CONDITIONS / ANTI-CONDITIONS (boundary.py, context.py, contradiction.py) -> UPDATE RELIABILITY -> TEST TRANSFER (transfer.py) ->
STORE KNOWLEDGE (knowledge.py + archive.py, versioned, provenance) -> UPDATE GRAPH (knowledge_graph.py) -> UPDATE META-KNOWLEDGE
(meta_learning.py) -> SELECT NEXT RESEARCH QUESTION (research_priority.py / research_policy.py).
Every stage passes through the firewall gate (firewalls.py) where it consumes data or memory; any FirewallBreach stops the step.
Also the section-3 protocol as a function: BEFORE (record decision, confidence, prediction, expected outcome, risk, selected patterns,
retrieved memories, abstentions, uncertainty, explanation) -> EXPERIENCE -> LEARNING -> AFTER (same disguised situation) ->
VALIDATION (did behaviour change only because of transferable information?). Freeze support: code+config hash, refuses to run
if changed (L02). Deterministic by seed. Emits a per-step trace that the reports can read.
Tests on engine/learning/planted_world.py: the full loop runs end to end on a small planted world; stage order is enforced; a
planted future-leak input is refused at the right stage; the section-87 acceptance experiment runs in miniature (encounter,
learn, disguise, re-encounter under new identity, decision changes, improvement measured) - report honestly whether it improves.
