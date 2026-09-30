# F09_panel_normalisation_leak  [model: opus]
C69 ledger section 5 "Future-leak concerns" item 2 and "Data-quality problems" item 6 (anti-cheating); C69 sections 21, 22; canon
C56. Read state/build/CONTEXT.md (rules 1-28), EXECUTION_LEDGER.md section 5, the journal entry "R08 caught two of its own features"
(Masterstock JOURNAL.md, 29 Sep ~15:40).
You OWN: engine/features.py and the feature builders under engine/ that compute panel-wide or cross-sectional statistics (list them),
engine/research/volatility_lab.py frame_from_panel ONLY, NEW scripts/feature_leak_audit.py, NEW tests/test_feature_leak_audit.py.
NEVER edit files owned by F06 (livesim_loop2.py, leak_audit.py), F07 (engine/learning/learner.py, loop_hooks.py, test_path.py,
planted_world.py, acceptance_mini.py), F08 (backtest.py, train.py, pit.py, experiment_memory.py), P06 (error_loop.py, C68 modules,
two_stage.py), W02's feeds.py/evidence.py/loop.py.
Do:
1. Build scripts/feature_leak_audit.py: for EVERY feature column any production frame builder emits, compute it on the full panel
   and on the panel truncated at a cut date T, and compare values dated <= T. Any difference = the feature reads the future
   (panel-wide mean/std/rank/quantile/winsor over all dates, full-sample fitted scalers, bfill, centred windows, etc.). Run it on
   a planted panel and on a small real-cache slice (RAM >= 2.5 GB, detached) and write state/research/feature_leak/report.md with
   one row per feature: CLEAN / LEAK (max abs diff, the offending operation file:line).
2. Fix every LEAK at the source to a past-only / as-of computation (expanding or rolling past-only stats, same-date cross-section
   only). Never delete a feature to hide a leak; if a fix changes a feature's meaning, say so.
3. Test: the audit FAILS on a planted leaking feature and passes on the fixed set; each fixed feature has a truncation test.
4. Add the audit as a CI step suggestion (exact ci.yml lines; do not edit ci.yml).
Report the before/after table, ruler counts, test results, and which past results used leaking features (they need re-running).
