# S17a_integration (part A: everything except the Test loop itself)
Contract section 60 row "Integration/refactoring/type safety" (4,000 lines total across S17a + S17b), sections 55, 61, 85; canon C62-C65.
Read state/build/CONTEXT.md fully (rules 1-19), SELF_LEARNING_CONTRACT.md sections 58-62, 83, 85, and state/build/INTEGRATION.md
section "C62 contract wave 1" (every hook listed there). C63: this is code; unit + integration tests; no real-data runs.
EXCEPTION to rule 1 - you OWN these existing files for this task (nobody else edits them now): engine/lessons.py, engine/memory.py,
engine/missed_winners.py, engine/improve.py, engine/registry.py, engine/patterns.py (only the prune_redundant call site),
engine/learning/archive.py (the _YEAR regex fix only), engine/learning/curator.py (the strength band only), .github/workflows/ci.yml,
pyproject.toml. NOT yours: scripts/livesim_loop2.py, engine/livesim.py, engine/adaptive.py (S17b, after the learner loop lands).
New files you own: engine/learning/wiring.py, tests/test_learning_integration.py, tests/test_learning_types.py.

Do:
1. Apply every queued hook from INTEGRATION.md that does not touch the Test loop: lessons/memory lessons -> failure hypotheses;
   MissedLedger.add -> learning.missed_winners; improve.log_experiment -> ExperimentLedger (one facade) + already_tested pre-launch;
   registry records carry ReproRecord and registry_findings; miner prune_redundant -> archive.record_redundancy / graph
   REDUNDANT_WITH; promotion calls scorecard.gate_improvement_claim + LearningFirewallGate + IdentityHarness; production readers use
   KnowledgeBoard.weight. Each hook is a small call into engine/learning/wiring.py (keep the edits to existing files one-line where
   possible) and each has an integration test proving the data actually flows end to end.
2. Fixes queued by builders: archive.py _YEAR misses "AAPL_2008" (use digit-only lookarounds like trader_view); curator release
   weights normalised to 1 hide absolute strength - add a coarse, era-free, calibrated strength band per item.
3. Retire duplicates behind adapters per CONTRACT_MAPPING.md "Duplicate / parallel systems" (the old paths call the new facade;
   no behaviour change where untested - mark such adapters clearly).
4. Type safety: add mypy (or pyright if already present) configuration for engine/learning in pyproject.toml, fix what it finds in
   engine/learning, and add CI steps: mypy on engine/learning, trader_view.assert_trader_path_clean(), contract_lines report.
5. Run the FULL test suite at the end (python -m pytest -q -p no:cacheprovider); report pass/fail counts honestly. Changing
   engine/ files makes the running Test loop's rounds stale by design (code_hash) - that is expected; do not touch the loop.
Report ruler counts for everything you wrote (wiring.py + tests + net lines added to existing files).
