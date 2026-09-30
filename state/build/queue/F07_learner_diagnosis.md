# F07_learner_diagnosis  [model: opus]
C69 ledger W-07 (priority 2, scientific defect); C69 sections 4, 12-14, 27, 28; C62 C03, E04, I16, I18 (FAILED rows).
Read state/build/CONTEXT.md (rules 1-28), state/build/EXECUTION_LEDGER.md (W-07; section 3 row 24; section 4 items 4 and 8),
scripts/acceptance_mini.py, state/research/acceptance_mini/summary.json, engine/learning/learner.py, loop_hooks.py, test_path.py,
curator.py, trader_view.py, planted_world.py.
You OWN: engine/learning/learner.py, engine/learning/loop_hooks.py, engine/learning/test_path.py, engine/learning/planted_world.py,
scripts/acceptance_mini.py, their tests. Never edit engine/research/loop.py/feeds.py/evidence.py (W02), error_loop.py or the C68
modules (P06), scripts/livesim_loop2.py or engine/leak_audit.py (F06). Planted data only (C63); no real run.

Facts: acceptance_mini improved on 1/3 seeds (seed 3: 722 changed decisions, edge +1.27pp, t 3.6); the other two seeds did not.
Do:
1. Diagnose WHY per seed, with evidence, not guesses: trace each planted truth through store -> retrieval (C03) -> knowledge ->
   decision change -> outcome. Name the stage where the planted signal is lost on the failing seeds (never found, found but not
   admitted, admitted but not retrieved, retrieved but not acted on, acted on but wrong side, etc.). Write the trace table to
   state/research/acceptance_mini/diagnosis.md.
2. Fix the root cause at the source (never lower a threshold or a test; never plant an easier world). Every fix gets a test that
   FAILS on the old behaviour.
3. Re-run acceptance_mini on >= 6 seeds (the old 3 plus 3 new; detached, RAM >= 2.5 GB) with the null (no-truth) control. State the
   before/after table: improved seeds, mean improvement vs none/control, and false improvement on the null world (must stay 0).
4. If the learner still fails, say so plainly and name the next hypothesis; do not mark anything VALIDATED.
Report ruler counts, test results, and the tables.
