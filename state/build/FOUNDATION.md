# Foundation status (scripts/foundation_status.py)

Rule (C46/C50/C51): every unit in or above its Bible range with passing tests, before any rabbit hole.
One row per work unit; range = sum of the Bible ranges of the phases it covers; no file counted twice.

| unit | phases | range | lines | status | tests |
|---|---|---|---|---|---|
| B01_pit_firewall | 1 | 1,500-3,000 | 1,978 | IN RANGE | - |
| B02_parity | 2 | 1,000-2,000 | 681 | BELOW RANGE | - |
| B03_lifecycle_bank | 4,5 | 1,700-3,500 | 1,602 | BELOW RANGE | - |
| B04_heavy_algo | 6,7 | 2,200-4,500 | 1,839 | BELOW RANGE | - |
| B05_analogs_ext | 8 | 2,000-4,000 | 1,989 | BELOW RANGE | - |
| B06_lessons | 10,11 | 2,300-4,500 | 1,231 | BELOW RANGE | - |
| B07_trust_direction | 12,13 | 2,500-5,000 | 877 | BELOW RANGE | - |
| B08_exits_stops | 15,16 | 2,000-4,000 | 1,606 | BELOW RANGE | - |
| B09_timeline_basis | 17,19,20 | 3,200-6,500 | 1,240 | BELOW RANGE | - |
| B10_blind_gates | 21,22,23,24 | 3,700-7,500 | 2,380 | BELOW RANGE | - |
| B11_antioverfit | 26,33,34 | 1,000-2,000 | 1,793 | IN RANGE | - |
| B12_control_reports | 0,30,35,36 | 1,500-3,000 | 1,944 | IN RANGE | - |
| B13_data_live_safety | 27,28 | 1,500-3,500 | 1,845 | IN RANGE | - |
| B14_miner_hardening | 3 | 3,000-6,000 | 1,938 | BELOW RANGE | - |
| B15_memory_adapter | 9,14,18 | 4,200-8,500 | 1,853 | BELOW RANGE | - |
| B16_public_explanation | 29 | 1,500-3,000 | 2,646 | IN RANGE | - |
| B17_quality_pyramid | 31,32 | 0-0 | 1,288 | IN RANGE | - |
| pre:0 | 0 | 0-0 | 671 | IN RANGE | - |
| pre:1 | 1 | 0-0 | 120 | IN RANGE | - |
| pre:3 | 3 | 0-0 | 441 | IN RANGE | - |
| pre:25 | 25 | 500-1,000 | 42 | BELOW RANGE | - |
| pre:8 | 8 | 0-0 | 166 | IN RANGE | - |
| pre:A1 |  | 0-0 | 177 | IN RANGE | - |

Repository Python lines: engine 22,494, scripts 8,261, tests 9,106 = 39,861 (floor 25,000).

Units below range or without code: B02_parity, B03_lifecycle_bank, B04_heavy_algo, B05_analogs_ext, B06_lessons, B07_trust_direction, B08_exits_stops, B09_timeline_basis, B10_blind_gates, B14_miner_hardening, B15_memory_adapter, pre:25
Bible phases with an estimate but no unit: none

## Files per unit

- **B01_pit_firewall**: `engine/pit.py` (1345), `tests/test_pit.py` (633)
- **B02_parity**: `engine/parity.py` (496), `tests/test_parity.py` (185)
- **B03_lifecycle_bank**: `engine/pattern_bank.py` (428), `engine/pattern_lifecycle.py` (533), `tests/test_pattern_bank.py` (328), `tests/test_pattern_lifecycle.py` (313)
- **B04_heavy_algo**: `engine/heavy_tests.py` (570), `engine/pattern_movers.py` (768), `tests/test_heavy_tests.py` (251), `tests/test_pattern_movers.py` (250)
- **B05_analogs_ext**: `engine/analog_weighting.py` (896), `engine/analogs_sector.py` (211), `engine/analogs_stock.py` (209), `tests/test_analogs_ext.py` (673)
- **B06_lessons**: `engine/antimemo.py` (284), `engine/lessons.py` (494), `tests/test_antimemo.py` (196), `tests/test_lessons.py` (257)
- **B07_trust_direction**: `engine/direction.py` (330), `engine/trust.py` (319), `tests/test_direction.py` (122), `tests/test_trust.py` (106)
- **B08_exits_stops**: `engine/exits.py` (669), `engine/stops.py` (363), `tests/test_exits.py` (375), `tests/test_stops.py` (199)
- **B09_timeline_basis**: `engine/basis_search.py` (201), `engine/objective.py` (198), `engine/timeline.py` (248), `tests/test_basis_search.py` (200), `tests/test_objective.py` (195), `tests/test_timeline.py` (198)
- **B10_blind_gates**: `engine/blind_gates.py` (643), `engine/health.py` (399), `engine/retester.py` (457), `tests/test_blind_gates.py` (406), `tests/test_health.py` (235), `tests/test_retester.py` (240)
- **B11_antioverfit**: `engine/ablation.py` (229), `engine/antioverfit.py` (644), `engine/repro.py` (360), `tests/test_ablation.py` (164), `tests/test_antioverfit.py` (247), `tests/test_repro.py` (149)
- **B12_control_reports**: `engine/champion.py` (437), `engine/checkpoint.py` (159), `engine/registry.py` (286), `engine/run_report.py` (448), `tests/test_champion.py` (252), `tests/test_checkpoint.py` (76), `tests/test_registry.py` (100), `tests/test_run_report.py` (186)
- **B13_data_live_safety**: `engine/data_sources.py` (596), `engine/isolation.py` (521), `tests/test_data_sources.py` (356), `tests/test_isolation.py` (192), `tests/test_live_safety.py` (180)
- **B14_miner_hardening**: `engine/candidates.py` (499), `engine/pattern_identity.py` (577), `engine/pattern_stats.py` (729), `scripts/miner_coverage.py` (133), `tests/test_candidates.py` (0), `tests/test_pattern_identity.py` (0), `tests/test_pattern_stats.py` (0)
- **B15_memory_adapter**: `engine/adaptive.py` (617), `engine/memory.py` (523), `engine/memory_diagnostics.py` (0), `engine/missed_winners.py` (452), `tests/test_adapter.py` (0), `tests/test_memory.py` (109), `tests/test_memory_ext.py` (0), `tests/test_missed_winners.py` (0), `tests/test_session.py` (152)
- **B16_public_explanation**: `docs/checklist.html` (62), `docs/explorer.html` (62), `docs/runs.html` (1111), `docs/sensitivity2.html` (62), `scripts/site_build.py` (205), `scripts/site_charts.py` (108), `scripts/site_pages.py` (355), `scripts/site_render.py` (154), `scripts/site_safety.py` (128), `scripts/site_sources.py` (399), `tests/test_site_build.py` (0)
- **B17_quality_pyramid**: `scripts/quality_gate.py` (770), `scripts/test_inventory.py` (362), `tests/integration` (156), `tests/test_quality_gate.py` (0)
- **pre:0**: `engine/provenance.py` (126), `tests/test_provenance.py` (96), `engine/improve.py` (333), `scripts/foundation_status.py` (116)
- **pre:1**: `engine/fill_audit.py` (56), `tests/test_fill_audit.py` (64)
- **pre:3**: `engine/patterns.py` (352), `tests/test_planted_patterns.py` (89)
- **pre:25**: `scripts/planted_calibration.py` (42)
- **pre:8**: `engine/analogs.py` (122), `scripts/analog_test.py` (44)
- **pre:A1**: `engine/candles.py` (50), `tests/test_candles.py` (127)
