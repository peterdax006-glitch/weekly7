# Foundation status (scripts/foundation_status.py)

Rule (C46/C50/C51): every unit in or above its Bible range with passing tests, before any rabbit hole.
One row per work unit; range = sum of the Bible ranges of the phases it covers; no file counted twice.

| unit | phases | range | lines | status | tests |
|---|---|---|---|---|---|
| B01_pit_firewall | 1 | 1,500-3,000 | 2,376 | IN RANGE | - |
| B02_parity | 2 | 1,000-2,000 | 1,454 | IN RANGE | - |
| B03_lifecycle_bank | 4,5 | 1,700-3,500 | 2,025 | IN RANGE | - |
| B04_heavy_algo | 6,7 | 2,200-4,500 | 2,331 | IN RANGE | - |
| B05_analogs_ext | 8 | 2,000-4,000 | 2,234 | IN RANGE | - |
| B06_lessons | 10,11 | 2,300-4,500 | 2,248 | BELOW RANGE | - |
| B07_trust_direction | 12,13 | 2,500-5,000 | 2,119 | BELOW RANGE | - |
| B08_exits_stops | 15,16 | 2,000-4,000 | 2,605 | IN RANGE | - |
| B09_timeline_basis | 17,19,20 | 3,200-6,500 | 3,258 | IN RANGE | - |
| B10_blind_gates | 21,22,23,24 | 3,700-7,500 | 4,094 | IN RANGE | - |
| B11_antioverfit | 26,33,34 | 1,000-2,000 | 2,040 | IN RANGE | - |
| B12_control_reports | 0,30,35,36 | 1,500-3,000 | 2,929 | IN RANGE | - |
| B13_data_live_safety | 27,28 | 1,500-3,500 | 2,570 | IN RANGE | - |
| B14_miner_hardening | 3 | 3,000-6,000 | 4,397 | IN RANGE | - |
| B15_memory_adapter | 9,14,18 | 4,200-8,500 | 4,486 | IN RANGE | - |
| B16_public_explanation | 29 | 1,500-3,000 | 4,534 | IN RANGE | - |
| B17_quality_pyramid | 31,32 | 0-0 | 2,196 | IN RANGE | - |
| B18_bible_trace | 37,38,39 | 0-0 | 1,776 | IN RANGE | - |
| B19_find_volatility_pipeline |  | 0-0 | 1,626 | IN RANGE | - |
| B20_trace_precision | 37,39 | 0-0 | 1,776 | IN RANGE | - |
| B21_resources_claims_dashboard | 30,36,42,43,44 | 500-1,000 | 3,457 | IN RANGE | - |
| B22_learning_delta |  | 0-0 | 2,300 | IN RANGE | - |
| B23_future_leak_audit |  | 0-0 | 2,598 | IN RANGE | - |
| B24_generalising_learner |  | 0-0 | 2,817 | IN RANGE | - |
| B25_direction_research |  | 0-0 | 3,401 | IN RANGE | - |
| pre:0 | 0 | 0-0 | 717 | IN RANGE | - |
| pre:1 | 1 | 0-0 | 120 | IN RANGE | - |
| pre:3 | 3 | 0-0 | 89 | IN RANGE | - |
| pre:25 | 25 | 500-1,000 | 610 | IN RANGE | - |
| pre:A7 |  | 0-0 | 312 | IN RANGE | - |
| pre:8 | 8 | 0-0 | 169 | IN RANGE | - |
| pre:A1 |  | 0-0 | 127 | IN RANGE | - |

Repository Python lines: engine 32,411, scripts 14,352, tests 20,021 = 66,784 (floor 25,000).

Units below range or without code: B06_lessons, B07_trust_direction
Bible phases with an estimate but no unit: none

## Files per unit

- **B01_pit_firewall**: `engine/pit.py` (1348), `scripts/pit_audit_real.py` (386), `tests/test_pit.py` (642)
- **B02_parity**: `engine/parity.py` (496), `engine/parity_suite.py` (422), `scripts/run_parity.py` (167), `tests/test_parity.py` (185), `tests/test_parity_suite.py` (184)
- **B03_lifecycle_bank**: `engine/pattern_bank.py` (495), `engine/pattern_lifecycle.py` (582), `scripts/run_pattern_bank.py` (176), `tests/test_pattern_bank.py` (386), `tests/test_pattern_lifecycle.py` (386)
- **B04_heavy_algo**: `engine/heavy_tests.py` (676), `engine/pattern_movers.py` (842), `scripts/heavy_algo_real.py` (190), `tests/test_heavy_tests.py` (331), `tests/test_pattern_movers.py` (292)
- **B05_analogs_ext**: `engine/analog_weighting.py` (898), `engine/analogs_sector.py` (211), `engine/analogs_stock.py` (209), `scripts/run_analogs_ext.py` (243), `tests/test_analogs_ext.py` (673)
- **B06_lessons**: `engine/antimemo.py` (577), `engine/lessons.py` (801), `scripts/lessons_real.py` (127), `tests/test_antimemo.py` (272), `tests/test_lessons.py` (471)
- **B07_trust_direction**: `engine/direction.py` (330), `engine/direction_ablate.py` (92), `engine/direction_calib.py` (186), `engine/trust.py` (319), `engine/trust_store.py` (219), `scripts/direction_study.py` (329), `tests/test_direction.py` (122), `tests/test_direction_ablate.py` (69), `tests/test_direction_calib.py` (125), `tests/test_direction_study.py` (106), `tests/test_trust.py` (106), `tests/test_trust_store.py` (116)
- **B08_exits_stops**: `engine/exits.py` (669), `engine/gaprisk.py` (390), `engine/stops.py` (400), `scripts/run_exits_stops.py` (179), `scripts/run_gaprisk.py` (95), `scripts/run_pattern_fail.py` (128), `tests/test_exits.py` (375), `tests/test_gaprisk.py` (145), `tests/test_stops.py` (224)
- **B09_timeline_basis**: `engine/basis_search.py` (318), `engine/objective.py` (270), `engine/timeline.py` (374), `scripts/basis_offline.py` (102), `scripts/dial_offline.py` (141), `scripts/livesim_loop2.py` (485), `scripts/timeline_basis_report.py` (157), `tests/test_basis_search.py` (327), `tests/test_livesim_loop2.py` (465), `tests/test_objective.py` (300), `tests/test_timeline.py` (319)
- **B10_blind_gates**: `engine/blind_gates.py` (643), `engine/health.py` (415), `engine/livesim.py` (597), `engine/retester.py` (492), `scripts/blind_gates_real.py` (227), `scripts/check_retester.py` (54), `scripts/livesim_cycle.py` (293), `tests/test_blind_gates.py` (406), `tests/test_health.py` (282), `tests/test_livesim_gates.py` (354), `tests/test_retester.py` (331)
- **B11_antioverfit**: `engine/ablation.py` (229), `engine/antioverfit.py` (644), `engine/repro.py` (360), `scripts/antioverfit_real.py` (247), `tests/test_ablation.py` (164), `tests/test_antioverfit.py` (247), `tests/test_repro.py` (149)
- **B12_control_reports**: `engine/baseline.py` (154), `engine/champion.py` (437), `engine/checkpoint.py` (184), `engine/experiment_memory.py` (388), `engine/registry.py` (286), `engine/run_report.py` (607), `scripts/audit_registry.py` (41), `tests/test_audit_registry.py` (26), `tests/test_baseline.py` (81), `tests/test_champion.py` (252), `tests/test_checkpoint.py` (90), `tests/test_experiment_memory.py` (97), `tests/test_registry.py` (100), `tests/test_run_report.py` (186)
- **B13_data_live_safety**: `engine/data_sources.py` (737), `engine/isolation.py` (521), `scripts/data_live_audit.py` (210), `scripts/fetch_delisted.py` (291), `tests/test_data_sources.py` (430), `tests/test_isolation.py` (192), `tests/test_live_safety.py` (189)
- **B14_miner_hardening**: `engine/candidates.py` (567), `engine/pattern_identity.py` (588), `engine/pattern_stats.py` (748), `engine/patterns.py` (530), `scripts/miner_coverage.py` (198), `tests/test_candidates.py` (421), `tests/test_pattern_identity.py` (367), `tests/test_pattern_stats.py` (708), `tests/test_patterns_integration.py` (270)
- **B15_memory_adapter**: `engine/adaptive.py` (776), `engine/memory.py` (636), `engine/memory_diagnostics.py` (384), `engine/missed_winners.py` (601), `scripts/run_b15_memory_adapter.py` (190), `tests/test_adapter.py` (580), `tests/test_memory.py` (109), `tests/test_memory_ext.py` (665), `tests/test_missed_winners.py` (393), `tests/test_session.py` (152)
- **B16_public_explanation**: `docs/checklist.html` (62), `docs/explorer.html` (62), `docs/runs.html` (2394), `docs/sensitivity2.html` (62), `scripts/site_audit.py` (136), `scripts/site_build.py` (217), `scripts/site_charts.py` (108), `scripts/site_pages.py` (355), `scripts/site_publish.py` (92), `scripts/site_render.py` (154), `scripts/site_safety.py` (128), `scripts/site_sources.py` (404), `tests/test_site_build.py` (360)
- **B17_quality_pyramid**: `scripts/quality_gate.py` (789), `scripts/test_inventory.py` (361), `tests/integration` (456), `tests/test_quality_gate.py` (590)
- **B18_bible_trace**: `scripts/bible_trace.py` (1272), `tests/test_bible_trace.py` (504)
- **B19_find_volatility_pipeline**: `engine/fv_pipeline.py` (1148), `scripts/fv_eval.py` (129), `tests/test_fv_pipeline.py` (349)
- **B20_trace_precision**: `scripts/bible_trace.py` (1272), `tests/test_bible_trace.py` (504)
- **B21_resources_claims_dashboard**: `docs/dashboard.html` (26), `engine/claims.py` (397), `engine/experiment_memory.py` (388), `engine/resources.py` (516), `engine/run_report.py` (607), `scripts/dashboard_build.py` (500), `tests/test_claims.py` (316), `tests/test_dashboard.py` (334), `tests/test_resources.py` (373)
- **B22_learning_delta**: `engine/learning_delta.py` (1180), `engine/livesim.py` (597), `scripts/learning_delta.py` (156), `tests/test_learning_delta.py` (367)
- **B23_future_leak_audit**: `engine/leak_audit.py` (1154), `scripts/leak_audit.py` (731), `tests/test_leak_audit.py` (713)
- **B24_generalising_learner**: `engine/learners.py` (1013), `engine/learning_delta.py` (1180), `scripts/learner_search.py` (262), `tests/test_learners.py` (362)
- **B25_direction_research**: `engine/candles.py` (50), `engine/direction.py` (330), `engine/direction_features.py` (485), `engine/features.py` (288), `engine/fv_pipeline.py` (1148), `engine/patterns.py.` (530), `scripts/direction_research.py` (268), `tests/test_direction_features.py` (302)
- **pre:0**: `engine/provenance.py` (126), `tests/test_provenance.py` (96), `engine/improve.py` (351), `scripts/foundation_status.py` (144)
- **pre:1**: `engine/fill_audit.py` (56), `tests/test_fill_audit.py` (64)
- **pre:3**: `tests/test_planted_patterns.py` (89)
- **pre:25**: `scripts/planted_calibration.py` (93), `engine/planted.py` (353), `tests/test_planted_library.py` (164)
- **pre:A7**: `engine/miner_tuning.py` (177), `tests/test_miner_tuning.py` (69), `scripts/miner_tune_real.py` (66)
- **pre:8**: `engine/analogs.py` (125), `scripts/analog_test.py` (44)
- **pre:A1**: `tests/test_candles.py` (127)
