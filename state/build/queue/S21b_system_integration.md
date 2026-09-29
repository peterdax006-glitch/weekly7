# S21b_system_integration (make the wiring hooks real outside the learner; honest reachability check)
Source of truth: state/build/INTEGRATION_AUDIT.md (29 Sep), section 1b especially: wiring.unwired_hooks() passes but only checks
call TEXT; 5 of its 11 hooks sit where production never reaches or do nothing. Contract sections 55, 60, 83, 85. C63: code + tests.
Read state/build/CONTEXT.md (rules 1-19), INTEGRATION_AUDIT.md in full, state/build/INTEGRATION.md.
You OWN: engine/learning/wiring.py, engine/adaptive.py, engine/improve.py, engine/registry.py, engine/missed_winners.py,
engine/lessons.py, engine/memory.py, engine/patterns.py (hook call sites only), engine/learning/promotion.py (evidence
registration only), engine/learning/champion.py (effective_champion / audit_decision_sources call sites only), scripts/ (except
livesim_loop2.py), .github/workflows/ci.yml, NEW scripts/reachability.py, NEW tests/test_learning_reachability_system.py.
S21a owns learner.py, test_path.py, loop_hooks.py - do not touch.

Do:
1. Missed winners: adaptive.py:358 calls MissedLedger.add, so the learning hook in observe() is dead - route the real call path
   through the learning ledger.
2. Registry memory hook: its host class is only built in tests - hook the class production actually uses.
3. Knowledge board: HUB.board is None outside tests, so KnowledgeBoard.weight is a no-op - configure the hub (root, board) from
   the production entry points (livesim_loop2 when --learner legit, weekly/improve runs), so weights actually apply there, and
   stay neutral with no board (documented).
4. Promotion gate: learning checks never run because nothing registers evidence or marks claims - make promotions of learned
   knowledge register their evidence and scorecard so the gate genuinely evaluates them; effective_champion and
   audit_decision_sources called after each decision run where champions are used.
5. Sinks: everything except the experiment ledger dies with the process - persist hypotheses, decisions, missed-winner and
   failure ledgers under the configured root (state/learning/...), append-only.
6. Lessons hooks run only from research scripts: connect them to the path that actually runs (adaptive session / loop).
7. scripts/reachability.py: an AST call-graph + import-reachability checker from the production entry points (scripts/*.py
   main functions, engine live path excluded per C65) that reports every engine/learning module and every INTEGRATION.md hook as
   REACHED / UNREACHED / RESEARCH-ONLY(with reason). Replace wiring.unwired_hooks() in CI with it (fail on UNREACHED).
Tests: behavioural proof that each fixed hook moves data in the production path (not just that text exists); the reachability
checker catches a planted unreachable hook. Report ruler counts and test results.
