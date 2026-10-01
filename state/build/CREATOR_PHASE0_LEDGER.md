# C77 Phase 0 - computed reconnaissance ledger

Generated 2026-10-01T06:36:53Z by scripts/phase0_recon.py (reading only). Self-model digest c6f5ce80e0216ef4.

## CR001-004_documents

```json
{
 "blueprints": [
  "ALGORITHM_BLUEPRINT.md",
  "BLUEPRINT.md",
  "MASTER_BLUEPRINT.md"
 ],
 "contracts": [
  "ALGORITHM_BLUEPRINT.md",
  "BIBLE.md",
  "BLUEPRINT.md",
  "CREATOR_MASTER_PROMPT.md",
  "MASTER_BLUEPRINT.md",
  "MASTER_EXECUTION_PROMPT.md",
  "PREDICTION_ERROR_ADDITION.md",
  "RESEARCH_BRAIN_CONTRACT.md",
  "SELF_LEARNING_CONTRACT.md",
  "TEN_HOUR_EXECUTION_CHECKLIST.md",
  "ULTIMATE_MASTER_PROMPT.md"
 ],
 "checklists": [
  "CREATOR_MASTER_CHECKLIST.json",
  "MASTER_EXECUTION_CHECKLIST.json",
  "PREDICTION_ERROR_CHECKLIST.json",
  "RESEARCH_BRAIN_CHECKLIST.json",
  "SELF_LEARNING_MASTER_CHECKLIST.json",
  "TEN_HOUR_CHECKLIST.json",
  "ULTIMATE_MASTER_CHECKLIST.json"
 ],
 "masterstock": "C:\\Users\\Peter\\Masterstock\\MASTERSTOCK.md",
 "masterstock_bytes": 1889468
}
```

## CR005_structure

```json
{
 "files": 567,
 "by_package_files": {
  "tests": 194,
  "scripts": 140,
  "engine": 85,
  "engine/research": 66,
  "engine/learning": 65,
  "creator": 16,
  "canon": 1
 },
 "by_package_meaningful_lines": {
  "engine/research": 76755,
  "tests": 61458,
  "engine/learning": 52970,
  "engine": 28791,
  "scripts": 16705,
  "creator": 4583,
  "canon": 180
 }
}
```

## CR006_branch_state

```json
{
 "head": "f7949a2e6293cd994f13c6963acd3eacb512d702",
 "branch": "main",
 "dirty_files": 63,
 "sandbox_branches": [],
 "ahead_behind_origin": "0\t0"
}
```

## CR007_builders

```json
{
 "running_python_processes": [
  "C:\\Users\\Peter\\weekly7\\.venv\\Scripts\\python.exe -u scripts/creator_kernel.py --cycles 1 --allow-agent-calls --call-usd 1.5",
  "C:\\Users\\Peter\\weekly7\\.venv\\Scripts\\python.exe scripts/phase0_recon.py",
  "C:\\Users\\Peter\\weekly7\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --rootdir C:\\Users\\Peter\\weekly7 --junitxml=C:\\Users\\Peter\\weekly7\\state\\creator\\evidence\\main\\junit\\tests__test_creato",
  "\"C:\\Users\\Peter\\AppData\\Local\\Programs\\Python\\Python311\\python.exe\" -m pytest -q -p no:cacheprovider --rootdir C:\\Users\\Peter\\weekly7 --junitxml=C:\\Users\\Peter\\weekly7\\state\\creator\\evidence\\main\\juni"
 ],
 "kernel_lock": true
}
```

## CR008_integrations

```json
{
 "ci_workflows": {
  "ci.yml": {
   "runs": [
    "pip install -r requirements.txt pytest mypy",
    "|",
    "|",
    "python scripts/arch_doc.py --check",
    "python scripts/quality_gate.py --baseline state/quality/baseline.json",
    "python -m mypy",
    "python -c \"from engine.learning import trader_view as T; print(T.assert_trader_path_clean(), 'modules inspected')\"",
    "python scripts/reachability.py --fail-on all --json state/research/reachability/report.json",
    "python scripts/contract_lines.py | tail -80",
    "python scripts/survivorship_gate.py",
    "python scripts/feature_leak_audit.py --source planted --no-report --fail-on leak",
    "python -m pytest -q -p no:cacheprovider --durations=15"
   ],
   "secrets": []
  },
  "engine.yml": {
   "runs": [
    "pip install -r requirements.txt",
    "|",
    "python -m engine.tick ${{ github.event.inputs.job }}",
    "|",
    "|"
   ],
   "secrets": [
    "ALPACA_KEY",
    "ALPACA_SECRET",
    "SEC_CONTACT"
   ]
  },
  "intraday.yml": {
   "runs": [
    "pip install -q pandas pyarrow yfinance",
    "|",
    "|",
    "|"
   ],
   "secrets": []
  }
 }
}
```

## CR009_tests

```json
{
 "test_files": 196,
 "test_functions": 5576,
 "collect": "",
 "collection_errors": [],
 "modules_without_any_test": 89,
 "examples": [
  "canon/build_canon.py",
  "engine/edgar.py",
  "engine/explain.py",
  "engine/model.py",
  "engine/options.py",
  "engine/portfolio.py",
  "engine/replay.py",
  "engine/scoring.py",
  "engine/shadows.py",
  "engine/site_data.py",
  "engine/tick.py",
  "scripts/_save_prediction_error_addition.py",
  "scripts/acceptance_mini.py",
  "scripts/agent_smoke.py",
  "scripts/antioverfit_real.py",
  "scripts/apply_corrections.py",
  "scripts/arch_doc.py",
  "scripts/audit_registry.py",
  "scripts/backtest_summary.py",
  "scripts/basis_offline.py",
  "scripts/bible_trace.py",
  "scripts/blind_gates_real.py",
  "scripts/c67_mover_sweep_real.py",
  "scripts/candidate_recall.py",
  "scripts/check_retester.py"
 ]
}
```

## CR010_validation_infra

```json
[
 "engine/ablation.py",
 "engine/experiment_memory.py",
 "engine/learning/experiment_memory.py",
 "engine/learning/reproducibility.py",
 "engine/repro.py",
 "engine/research/experiments.py",
 "engine/retester.py",
 "scripts/check_retester.py",
 "scripts/legacy/experiment.py",
 "scripts/repro_w01c.py",
 "creator/ledger.py",
 "engine/claims.py",
 "engine/provenance.py",
 "engine/registry.py",
 "scripts/audit_registry.py"
]
```

## CR011_provenance

```json
{
 "verify_integrity": "OK",
 "canon_verify_tail": "canon ok: 77 directives",
 "infra": [
  "creator/ledger.py",
  "engine/claims.py",
  "engine/provenance.py",
  "engine/registry.py",
  "scripts/audit_registry.py"
 ]
}
```

## CR012_rollback

```json
[
 "creator/sandbox.py",
 "engine/checkpoint.py",
 "engine/learning/checkpoints.py"
]
```

## CR013_deterministic_replay

```json
[
 "engine/learning/reproducibility.py",
 "engine/parity.py",
 "engine/parity_suite.py",
 "engine/replay.py",
 "engine/repro.py",
 "scripts/legacy/replay_year.py",
 "scripts/repro_w01c.py",
 "scripts/run_parity.py"
]
```

## CR014_reachability

```json
{
 "unreached_modules": 14,
 "examples": [
  "scripts/_save_prediction_error_addition.py",
  "scripts/backtest_summary.py",
  "scripts/legacy/experiment.py",
  "scripts/legacy/frontier.py",
  "scripts/legacy/loop_years.py",
  "scripts/legacy/replay_year.py",
  "scripts/legacy/research.py",
  "scripts/legacy/topk_rules.py",
  "scripts/legacy/tuning_report.py",
  "scripts/livesim_cycle.py",
  "scripts/miner_tune_real.py",
  "scripts/patterns.py",
  "scripts/sensitivity.py",
  "tests/integration/pipeline_world.py"
 ]
}
```

## CR015_data_flow

```json
{
 "package_edges": [
  [
   "creator.ledger",
   "engine"
  ],
  [
   "scripts.acceptance_mini",
   "engine"
  ],
  [
   "scripts.agent_smoke",
   "creator"
  ],
  [
   "scripts.algo_test",
   "engine"
  ],
  [
   "scripts.analog_test",
   "engine"
  ],
  [
   "scripts.antioverfit_real",
   "engine"
  ],
  [
   "scripts.archive_opens",
   "engine"
  ],
  [
   "scripts.audit_registry",
   "engine"
  ],
  [
   "scripts.backfill_market_snaps",
   "engine"
  ],
  [
   "scripts.backtest_summary",
   "engine"
  ],
  [
   "scripts.basis_offline",
   "engine"
  ],
  [
   "scripts.blind_gates_real",
   "engine"
  ],
  [
   "scripts.bootstrap_history",
   "engine"
  ],
  [
   "scripts.c67_mover_sweep_real",
   "engine"
  ],
  [
   "scripts.c68_fix_promote",
   "engine"
  ],
  [
   "scripts.candidate_recall",
   "engine"
  ],
  [
   "scripts.check_retester",
   "engine"
  ],
  [
   "scripts.collect_intraday",
   "engine"
  ],
  [
   "scripts.contract_checklist",
   "engine"
  ],
  [
   "scripts.creator_kernel",
   "creator"
  ],
  [
   "scripts.creator_status",
   "creator"
  ],
  [
   "scripts.dashboard_build",
   "engine"
  ],
  [
   "scripts.data_live_audit",
   "engine"
  ],
  [
   "scripts.devbench_validate",
   "creator"
  ],
  [
   "scripts.diag",
   "engine"
  ],
  [
   "scripts.dial_offline",
   "engine"
  ],
  [
   "scripts.direction_research",
   "engine"
  ],
  [
   "scripts.direction_study",
   "engine"
  ],
  [
   "scripts.download_1962",
   "engine"
  ],
  [
   "scripts.extend_history",
   "engine"
  ],
  [
   "scripts.f17_filing_regression",
   "engine"
  ],
  [
   "scripts.f23_planted_trades",
   "engine"
  ],
  [
   "scripts.feature_leak_audit",
   "engine"
  ],
  [
   "scripts.fetch_delisted",
   "engine"
  ],
  [
   "scripts.fetch_macro",
   "engine"
  ],
  [
   "scripts.fv_eval",
   "engine"
  ],
  [
   "scripts.grid_runner",
   "engine"
  ],
  [
   "scripts.heavy_algo_real",
   "engine"
  ],
  [
   "scripts.leak_audit",
   "engine"
  ],
  [
   "scripts.learner_search",
   "engine"
  ],
  [
   "scripts.learning_curve",
   "engine"
  ],
  [
   "scripts.learning_delta",
   "engine"
  ],
  [
   "scripts.learning_report",
   "engine"
  ],
  [
   "scripts.legacy",
   "engine"
  ],
  [
   "scripts.lessons_archive",
   "engine"
  ],
  [
   "scripts.lessons_real",
   "engine"
  ],
  [
   "scripts.livesim_cycle",
   "engine"
  ],
  [
   "scripts.livesim_loop2",
   "engine"
  ],
  [
   "scripts.miner_coverage",
   "engine"
  ],
  [
   "scripts.miner_tune_real",
   "engine"
  ],
  [
   "scripts.movers",
   "engine"
  ],
  [
   "scripts.pattern_benchmark",
   "engine"
  ],
  [
   "scripts.pattern_memory_demo",
   "engine"
  ],
  [
   "scripts.pattern_memory_real",
   "engine"
  ],
  [
   "scripts.pattern_reliability_study",
   "engine"
  ],
  [
   "scripts.patterns",
   "engine"
  ],
  [
   "scripts.phase0_recon",
   "creator"
  ],
  [
   "scripts.phase0_recon",
   "engine"
  ],
  [
   "scripts.pit_audit_real",
   "engine"
  ],
  [
   "scripts.planted_calibration",
   "engine"
  ],
  [
   "scripts.regen_weekly_snaps",
   "engine"
  ],
  [
   "scripts.repair_history",
   "engine"
  ],
  [
   "scripts.research_loop",
   "engine"
  ],
  [
   "scripts.run_13d_fix",
   "engine"
  ],
  [
   "scripts.run_analogs_ext",
   "engine"
  ],
  [
   "scripts.run_b15_memory_adapter",
   "engine"
  ],
  [
   "scripts.run_exits_stops",
   "engine"
  ],
  [
   "scripts.run_gaprisk",
   "engine"
  ],
  [
   "scripts.run_parity",
   "engine"
  ],
  [
   "scripts.run_pattern_bank",
   "engine"
  ],
  [
   "scripts.run_pattern_fail",
   "engine"
  ],
  [
   "scripts.sensitivity",
   "engine"
  ],
  [
   "scripts.site_build",
   "engine"
  ],
  [
   "scripts.site_sources",
   "engine"
  ],
  [
   "scripts.smoke",
   "engine"
  ],
  [
   "scripts.sync_intraday",
   "engine"
  ],
  [
   "scripts.three_way",
   "engine"
  ],
  [
   "scripts.timeline_basis_report",
   "engine"
  ],
  [
   "scripts.tuning_lab",
   "engine"
  ],
  [
   "scripts.voltarget_of
... (truncated; full in the .json)
```

## CR016_firewalls

```json
[
 "engine/isolation.py",
 "engine/leak_audit.py",
 "engine/learning/boundary.py",
 "engine/learning/firewalls.py",
 "engine/learning/future_firewall.py",
 "engine/learning/identity_firewall.py",
 "engine/learning/memory_firewall.py",
 "engine/learning/separation.py",
 "engine/research/firewall.py",
 "engine/scramble_audit.py",
 "scripts/feature_leak_audit.py",
 "scripts/leak_audit.py"
]
```

## CR017_research_boundary

```json
{
 "rule": "trading path never imports engine.research (statically, incl. function-level imports)",
 "violations": []
}
```

## CR018_learner_truth_boundary

```json
[
 "engine/blind_gates.py",
 "engine/leak_audit.py",
 "engine/learning/future_firewall.py",
 "engine/pit.py",
 "engine/timeline.py",
 "scripts/blind_gates_real.py",
 "scripts/feature_leak_audit.py",
 "scripts/leak_audit.py",
 "scripts/timeline_basis_report.py"
]
```

## CR019_known_failures

```json
{
 "note": "creator tests: see state/creator/test_evidence.json; engine/full-suite failures need a full run (not executed by recon)",
 "collection_errors": []
}
```

## CR020_known_limitations

```json
{
 "by_kind": {
  "BELOW_FLOOR": 15,
  "UNTESTED_MODULE": 89,
  "UNREACHED": 77,
  "STUB": 8
 },
 "stubs": [
  "engine/exits.py:Rule.run:336: raises NotImplementedError",
  "engine/learners.py:CrossYearLearner.propose:722: raises NotImplementedError",
  "engine/learning/controls.py:Control.decide:338: raises NotImplementedError",
  "engine/learning/credit.py:Combiner.describe:273: raises NotImplementedError",
  "engine/learning/firewalls.py:FirewallLayer.inspect:331: raises NotImplementedError",
  "engine/research/break_research.py:GateRule.mask:289: raises NotImplementedError",
  "engine/research/break_research.py:GateRule.describe:292: raises NotImplementedError",
  "engine/stops.py:StopRule.dist:193: raises NotImplementedError"
 ]
}
```

## CR021_stale_checklists

```json
{
 "CREATOR_MASTER_CHECKLIST.json": {
  "items": 304,
  "status": {
   "NOT_STARTED": 304
  },
  "done_without_evidence": 0,
  "examples": []
 },
 "MASTER_EXECUTION_CHECKLIST.json": {
  "items": 585,
  "status": {
   "NOT_STARTED": 585
  },
  "done_without_evidence": 0,
  "examples": []
 },
 "PREDICTION_ERROR_CHECKLIST.json": {
  "items": 64,
  "status": {
   "TESTING": 57,
   "NOT_STARTED": 4,
   "VALIDATED": 1,
   "FAILED": 1,
   "IN_PROGRESS": 1
  },
  "done_without_evidence": 0,
  "examples": []
 },
 "RESEARCH_BRAIN_CHECKLIST.json": {
  "items": 114,
  "status": {
   "TESTING": 79,
   "IN_PROGRESS": 18,
   "IMPLEMENTED": 10,
   "FAILED": 3,
   "NOT_STARTED": 4
  },
  "done_without_evidence": 2,
  "examples": [
   "RG03",
   "RG05"
  ]
 },
 "SELF_LEARNING_MASTER_CHECKLIST.json": {
  "items": 191,
  "status": {
   "TESTING": 154,
   "FAILED": 12,
   "IMPLEMENTED": 12,
   "IN_PROGRESS": 7,
   "NOT_STARTED": 6
  },
  "done_without_evidence": 1,
  "examples": [
   "I03"
  ]
 },
 "TEN_HOUR_CHECKLIST.json": {
  "items": 366,
  "status": {
   "VALIDATED": 18,
   "NOT_STARTED": 347,
   "IN_PROGRESS": 1
  },
  "done_without_evidence": 0,
  "examples": []
 },
 "ULTIMATE_MASTER_CHECKLIST.json": {
  "items": 302,
  "status": {
   "NOT_STARTED": 281,
   "VALIDATED": 20,
   "IN_PROGRESS": 1
  },
  "done_without_evidence": 0,
  "examples": []
 }
}
```

## CR022_duplicates

```json
[
 {
  "body": "a08d90d8ebe7c003",
  "sites": [
   "engine/direction_features.py:week_end_positions",
   "engine/fv_pipeline.py:week_end_sessions"
  ]
 },
 {
  "body": "b445c7a786fcafa2",
  "sites": [
   "engine/learning/calibration.py:dump",
   "engine/learning/surprise.py:dump"
  ]
 },
 {
  "body": "88ba58dcfa0961e3",
  "sites": [
   "engine/learning/similarity.py:hash_bin",
   "engine/learning/situation.py:hash_label"
  ]
 },
 {
  "body": "78c33431d09f3954",
  "sites": [
   "engine/research/counterfactual.py:_clean",
   "engine/research/winners_losers.py:_fin"
  ]
 }
]
```

## CR023_disconnected

```json
[
 "scripts/_save_prediction_error_addition.py",
 "scripts/backtest_summary.py",
 "scripts/legacy/experiment.py",
 "scripts/legacy/frontier.py",
 "scripts/legacy/loop_years.py",
 "scripts/legacy/replay_year.py",
 "scripts/legacy/research.py",
 "scripts/legacy/topk_rules.py",
 "scripts/legacy/tuning_report.py",
 "scripts/livesim_cycle.py",
 "scripts/miner_tune_real.py",
 "scripts/patterns.py",
 "scripts/sensitivity.py",
 "tests/integration/pipeline_world.py"
]
```

## CR024_implemented_not_validated

```json
{
 "CREATOR_MASTER_CHECKLIST.json": {
  "NOT_STARTED": 304
 },
 "MASTER_EXECUTION_CHECKLIST.json": {
  "NOT_STARTED": 585
 },
 "PREDICTION_ERROR_CHECKLIST.json": {
  "TESTING": 57,
  "NOT_STARTED": 4,
  "VALIDATED": 1,
  "FAILED": 1,
  "IN_PROGRESS": 1
 },
 "RESEARCH_BRAIN_CHECKLIST.json": {
  "TESTING": 79,
  "IN_PROGRESS": 18,
  "IMPLEMENTED": 10,
  "FAILED": 3,
  "NOT_STARTED": 4
 },
 "SELF_LEARNING_MASTER_CHECKLIST.json": {
  "TESTING": 154,
  "FAILED": 12,
  "IMPLEMENTED": 12,
  "IN_PROGRESS": 7,
  "NOT_STARTED": 6
 },
 "TEN_HOUR_CHECKLIST.json": {
  "VALIDATED": 18,
  "NOT_STARTED": 347,
  "IN_PROGRESS": 1
 },
 "ULTIMATE_MASTER_CHECKLIST.json": {
  "NOT_STARTED": 281,
  "VALIDATED": 20,
  "IN_PROGRESS": 1
 }
}
```

## CR025_shallow

```json
[]
```

## CR026_future_leak_risks

```json
[
 "engine/blind_gates.py",
 "engine/leak_audit.py",
 "engine/learning/future_firewall.py",
 "engine/pit.py",
 "engine/timeline.py",
 "scripts/blind_gates_real.py",
 "scripts/feature_leak_audit.py",
 "scripts/leak_audit.py",
 "scripts/timeline_basis_report.py"
]
```

## CR027_self_learning_infra

```json
[
 "engine/learning/__init__.py",
 "engine/learning/archive.py",
 "engine/learning/belief.py",
 "engine/learning/boundary.py",
 "engine/learning/break_detection.py",
 "engine/learning/calibration.py",
 "engine/learning/champion.py",
 "engine/learning/checkpoints.py",
 "engine/learning/competition.py",
 "engine/learning/complexity.py",
 "engine/learning/compute.py",
 "engine/learning/context.py",
 "engine/learning/contradiction.py",
 "engine/learning/contradiction_monitor.py",
 "engine/learning/controls.py",
 "engine/learning/core.py",
 "engine/learning/credit.py",
 "engine/learning/curator.py",
 "engine/learning/decision_contract.py",
 "engine/learning/disagreement.py",
 "engine/learning/epistemic.py",
 "engine/learning/experiment_memory.py",
 "engine/learning/failed_learners.py",
 "engine/learning/failure.py",
 "engine/learning/firewalls.py",
 "engine/learning/future_firewall.py",
 "engine/learning/health.py",
 "engine/learning/hierarchy.py",
 "engine/learning/identity_firewall.py",
 "engine/learning/interpretation.py",
 "engine/learning/knowledge.py",
 "engine/learning/knowledge_graph.py",
 "engine/learning/learner.py",
 "engine/learning/learning_curve.py",
 "engine/learning/lifecycle.py",
 "engine/learning/loop_hooks.py",
 "engine/learning/memory_firewall.py",
 "engine/learning/meta_learning.py",
 "engine/learning/missed_winners.py",
 "engine/learning/planted_world.py",
 "engine/learning/portfolio_value.py",
 "engine/learning/postmortem.py",
 "engine/learning/promotion.py",
 "engine/learning/questions.py",
 "engine/learning/redundancy.py",
 "engine/learning/reliability.py",
 "engine/learning/reports.py",
 "engine/learning/reproducibility.py",
 "engine/learning/research_policy.py",
 "engine/learning/research_priority.py",
 "engine/learning/retirement.py",
 "engine/learning/retrieval.py",
 "engine/learning/same_year.py",
 "engine/learning/scorecard.py",
 "engine/learning/separation.py",
 "engine/learning/similarity.py",
 "engine/learning/situation.py",
 "engine/learning/surprise.py",
 "engine/learning/temporal.py",
 "engine/learning/trader_view.py",
 "engine/learning/transfer.py",
 "engine/learning/transfer_score.py",
 "engine/learning/unknowns.py",
 "engine/learning/wiring.py"
]
```

## CR028_research_infra

```json
[
 "engine/research/__init__.py",
 "engine/research/autopsy.py",
 "engine/research/brain_health.py",
 "engine/research/break_research.py",
 "engine/research/calibration_target.py",
 "engine/research/candidate_forms.py",
 "engine/research/change_points.py",
 "engine/research/compute_manager.py",
 "engine/research/controller.py",
 "engine/research/core.py",
 "engine/research/counterfactual.py",
 "engine/research/cross_section.py",
 "engine/research/decision_bridge.py",
 "engine/research/direction_lab.py",
 "engine/research/discovery.py",
 "engine/research/discovery_sources.py",
 "engine/research/diversity.py",
 "engine/research/episode_paths.py",
 "engine/research/episodes.py",
 "engine/research/error_loop.py",
 "engine/research/error_research.py",
 "engine/research/evidence.py",
 "engine/research/exit_research.py",
 "engine/research/expectations.py",
 "engine/research/experiments.py",
 "engine/research/failed_lab.py",
 "engine/research/feeds.py",
 "engine/research/firewall.py",
 "engine/research/frontier.py",
 "engine/research/hypothesis_tree.py",
 "engine/research/interactions.py",
 "engine/research/knowability.py",
 "engine/research/loop.py",
 "engine/research/loss_pipeline.py",
 "engine/research/market_expectations.py",
 "engine/research/meta_research.py",
 "engine/research/missed.py",
 "engine/research/multiscale.py",
 "engine/research/namespaces.py",
 "engine/research/observer.py",
 "engine/research/outcomes.py",
 "engine/research/pattern_benchmark.py",
 "engine/research/pattern_change.py",
 "engine/research/precursors.py",
 "engine/research/prediction_error.py",
 "engine/research/priority.py",
 "engine/research/quality_gate.py",
 "engine/research/questions.py",
 "engine/research/regime_memory.py",
 "engine/research/regimes.py",
 "engine/research/replication.py",
 "engine/research/research_graph.py",
 "engine/research/science_memory.py",
 "engine/research/scorecard.py",
 "engine/research/selection_constraint.py",
 "engine/research/self_correct.py",
 "engine/research/symmetry.py",
 "engine/research/targets.py",
 "engine/research/two_stage.py",
 "engine/research/unknown_cause.py",
 "engine/research/value_accounting.py",
 "engine/research/vol_hypotheses.py",
 "engine/research/volatility_lab.py",
 "engine/research/waste.py",
 "engine/research/what_changed.py",
 "engine/research/winners_losers.py"
]
```

## CR029_experiment_infra

```json
[
 "engine/ablation.py",
 "engine/experiment_memory.py",
 "engine/learning/experiment_memory.py",
 "engine/learning/reproducibility.py",
 "engine/repro.py",
 "engine/research/experiments.py",
 "engine/retester.py",
 "scripts/check_retester.py",
 "scripts/legacy/experiment.py",
 "scripts/repro_w01c.py"
]
```

## CR030_health_infra

```json
[
 "engine/health.py",
 "engine/learning/contradiction_monitor.py",
 "engine/learning/health.py",
 "engine/research/brain_health.py",
 "engine/run_report.py"
]
```

## CR031_compute_controls

```json
[
 "engine/learning/compute.py",
 "engine/research/compute_manager.py",
 "engine/resources.py"
]
```
