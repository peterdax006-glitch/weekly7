# B12_control_reports
Bible: PHASES 0, 30, 35, 36 (lines 357-431, 1597-1630, 1730-1835). Estimated code: 2,500-5,000 lines.
You own: engine/registry.py, engine/checkpoint.py, engine/champion.py, engine/run_report.py, tests/test_registry.py, tests/test_checkpoint.py, tests/test_champion.py, tests/test_run_report.py

Experiment registry (query/filter/compare over state/experiments log written by engine.improve.log_experiment); checkpoint bundle per major run (config, logs, metrics, manifest, seeds, hashes; verify); immutable baseline snapshot; experiment memory (what was tried, don't repeat); champion/challenger promotion with fail-closed gates; the required per-run report (Phase 36 sections).
