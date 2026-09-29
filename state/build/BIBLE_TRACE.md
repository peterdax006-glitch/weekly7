# BIBLE TRACE

Generated 2026-09-29T04:48Z by scripts/bible_trace.py from BIBLE.md (sha deb456974707d57e). Indexed 142 code files, 67 test files, 413 evidence files.

States are computed, never typed. [x] means TRACEABLE (code + a test that names the requirement + PASS evidence where the Bible asks for real data); it is a term-overlap heuristic and does not replace reading the evidence.

## Rules

- no code -> [ ]
- failed verdict recorded -> [!]
- code without test -> [~]
- code+test, Bible asks real-data proof, none found -> [~]
- evidence exists but no PASS line -> [?]
- [x] only with code + speaking test + (where asked) PASS evidence cited by path

## Totals

667 requirements.

| state | meaning | count | share |
|---|---|---:|---:|
| [ ] | NOT STARTED | 52 | 7.8% |
| [~] | IMPLEMENTED / TESTING | 277 | 41.5% |
| [?] | UNPROVEN | 55 | 8.2% |
| [x] | VALIDATED | 237 | 35.5% |
| [!] | FAILED | 46 | 6.9% |

By requirement kind:

| kind | [ ] | [~] | [?] | [x] | [!] |
|---|---:|---:|---:|---:|---:|
| checkbox | 5 | 35 | 9 | 24 | 5 |
| checklist | 0 | 19 | 11 | 5 | 10 |
| heading | 6 | 26 | 7 | 11 | 4 |
| list | 36 | 176 | 25 | 182 | 26 |
| phase | 2 | 3 | 0 | 0 | 0 |
| report | 3 | 18 | 3 | 15 | 1 |

## Per phase

| phase | title | reqs | [ ] | [~] | [?] | [x] | [!] | linked code lines | Bible range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | BASELINE AND CONTROL SYSTEM | 43 | 0 | 37 | 1 | 5 | 0 | 1,644 | 1,000-2,000 |
| 1 | POINT-IN-TIME DATA FIREWALL | 16 | 0 | 4 | 0 | 12 | 0 | 1,790 | 1,500-3,000 |
| 2 | FEATURE/PARITY FIREWALL | 9 | 2 | 5 | 1 | 1 | 0 | 1,085 | 1,000-2,000 |
| 3 | PATTERN MINER HARDENING | 39 | 1 | 15 | 4 | 18 | 1 | 2,594 | 3,000-6,000 |
| 4 | PATTERN LIFECYCLE | 9 | 0 | 3 | 1 | 5 | 0 | 1,253 | 1,000-2,000 |
| 5 | LONG-TERM PATTERN BANK | 11 | 0 | 1 | 3 | 5 | 2 | 1,253 | 700-1,500 |
| 6 | PATTERN → FIND VOLATILITY INTEGRATION | 10 | 0 | 3 | 0 | 7 | 0 | 1,708 | 700-1,500 |
| 7 | HEAVY ALGORITHM TESTING | 22 | 0 | 14 | 1 | 7 | 0 | 1,708 | 1,500-3,000 |
| 8 | ANALOG ENGINE | 40 | 0 | 21 | 0 | 19 | 0 | 1,561 | 2,000-4,000 |
| 9 | MEMORY SYSTEM | 26 | 1 | 12 | 3 | 9 | 1 | 1,808 | 2,000-4,000 |
| 10 | LESSON MEMORY / LEARNING FROM MISTAKES | 14 | 2 | 3 | 0 | 7 | 2 | 1,487 | 1,500-3,000 |
| 11 | RERUN ANTI-MEMORIZATION EXPERIMENT | 13 | 1 | 3 | 2 | 5 | 2 | 1,574 | 800-1,500 |
| 12 | PER-STOCK-TYPE TRUST TABLES | 11 | 1 | 4 | 0 | 6 | 0 | 1,197 | 1,000-2,000 |
| 13 | DIRECTION ENGINE | 11 | 0 | 3 | 1 | 6 | 1 | 1,256 | 1,500-3,000 |
| 14 | MISSED-WINNER DETECTOR | 11 | 0 | 3 | 1 | 4 | 3 | 2,584 | 700-1,500 |
| 15 | EXIT LEARNER | 8 | 0 | 3 | 0 | 5 | 0 | 1,376 | 1,000-2,000 |
| 16 | STOP / LOSS ENGINE | 7 | 0 | 6 | 0 | 1 | 0 | 1,638 | 1,000-2,000 |
| 17 | TIMELINE DIAL | 12 | 1 | 3 | 0 | 4 | 4 | 962 | 1,000-2,000 |
| 18 | WEEKLY ADAPTER | 21 | 3 | 0 | 3 | 14 | 1 | 2,584 | 1,500-3,000 |
| 19 | TRAIN THE TRAINING BASIS | 20 | 6 | 3 | 1 | 9 | 1 | 1,322 | 1,500-3,000 |
| 20 | TIERED OBJECTIVE FIREWALL | 9 | 0 | 3 | 0 | 4 | 2 | 2,031 | 700-1,500 |
| 21 | BLIND SIMULATOR HARDENING | 7 | 0 | 4 | 1 | 2 | 0 | 2,678 | 1,500-3,000 |
| 22 | BLIND CLOCK | 1 | 0 | 1 | 0 | 0 | 0 | 2,678 | 1,000-2,000 |
| 23 | RE-TESTER | 7 | 0 | 2 | 0 | 3 | 2 | 2,207 | 700-1,500 |
| 24 | WORKER HEALTH | 10 | 0 | 0 | 1 | 9 | 0 | 1,777 | 500-1,000 |
| 25 | PLANTED-PATTERN CALIBRATION | 6 | 3 | 0 | 0 | 2 | 1 | 743 | 500-1,000 |
| 26 | ANTI-OVERFITTING BATTERY | 10 | 0 | 2 | 3 | 2 | 3 | 1,657 | 1,000-2,000 |
| 27 | DATA EXPANSION | 7 | 0 | 2 | 2 | 3 | 0 | 1,730 | 1,000-2,500 |
| 28 | LIVE/RESEARCH SEPARATION | 7 | 1 | 2 | 1 | 3 | 0 | 1,509 | 500-1,000 |
| 29 | PUBLIC EXPLANATION SYSTEM | 19 | 1 | 9 | 1 | 8 | 0 | 1,582 | 1,500-3,000 |
| 30 | EXPERIMENT MEMORY | 12 | 4 | 4 | 0 | 3 | 1 | 1,507 | 500-1,000 |
| 31 | CODE QUALITY FIREWALL | 11 | 5 | 6 | 0 | 0 | 0 | 1,150 | - |
| 32 | TEST PYRAMID | 7 | 0 | 6 | 1 | 0 | 0 | 1,150 | - |
| 33 | REQUIRED REPRODUCIBILITY | 7 | 0 | 1 | 0 | 6 | 0 | 1,495 | - |
| 34 | REQUIRED ABLATION FRAMEWORK | 6 | 0 | 2 | 0 | 3 | 1 | 3,365 | - |
| 35 | CHAMPION / CHALLENGER SYSTEM | 10 | 2 | 4 | 0 | 4 | 0 | 1,330 | - |
| 36 | REQUIRED REPORT AFTER EACH MAJOR RUN | 40 | 3 | 18 | 3 | 15 | 1 | 2,006 | - |
| 37 | CHECKLIST STATE MACHINE | 1 | 0 | 1 | 0 | 0 | 0 | 1,392 | - |
| 38 | MASTER CHECKLIST | 45 | 0 | 19 | 11 | 5 | 10 | 1,392 | - |
| 39 | DEFINITION OF DONE | 32 | 2 | 18 | 5 | 3 | 4 | 1,031 | - |
| 40 | WHAT "KEEP WORKING" MEANS | 1 | 0 | 1 | 0 | 0 | 0 | 0 | - |
| 41 | NEVER SETTLE FOR A COSMETIC IMPLEMENTAT… | 1 | 1 | 0 | 0 | 0 | 0 | 0 | - |
| 42 | THE STANDARD FOR EVERY CLAIM | 19 | 3 | 6 | 2 | 7 | 1 | 0 | - |
| 43 | PRIORITY WHEN TIME/COMPUTE IS LIMITED | 11 | 3 | 2 | 1 | 3 | 2 | 0 | - |
| 44 | RESOURCE MANAGEMENT | 9 | 5 | 2 | 1 | 1 | 0 | 0 | - |
| 45 | CONTINUOUS RESEARCH LOOP | 1 | 1 | 0 | 0 | 0 | 0 | 0 | - |
| 46 | FINAL REPORT REQUIREMENT | 18 | 0 | 16 | 0 | 2 | 0 | 266 | - |

## The 20 most important MISSING items

Ranked by Phase 43's priority order (anti-leak first), then leak/gate/fail-closed wording; at most 5 per phase.

1. `2.0.7` (phase 2, Bible line 506): maximum relative error
2. `2.0.8` (phase 2, Bible line 507): NaN mismatch detection
3. `3.5.5` (phase 3, Bible line 592): does not violate anti-leak rules
4. `25.0.2` (phase 25, Bible line 1435): false discovery rate
5. `25.0.5` (phase 25, Bible line 1438): recovery rate
6. `25.0.6` (phase 25, Bible line 1439): rescoping behavior
7. `10.0.3` (phase 10, Bible line 912): false positives
8. `10.0.4` (phase 10, Bible line 913): false negatives
9. `11.0.10` (phase 11, Bible line 954): degradation on B
10. `9.4.1` (phase 9, Bible line 886): 9.4 Market similarity: Implement Gaussian similarity
11. `12.0.5` (phase 12, Bible line 975): theme/industry momentum
12. `43.0.1` (phase 43, Bible line 2112): anti-leak correctness
13. `42.0.1` (phase 42, Bible line 2079): exact pattern definition
14. `42.0.12` (phase 42, Bible line 2090): cost-adjusted result
15. `42.0.17` (phase 42, Bible line 2102): statistical comparison
16. `43.0.10` (phase 43, Bible line 2121): dashboard
17. `43.0.9` (phase 43, Bible line 2120): outer-loop tuning
18. `44.0.3` (phase 44, Bible line 2134): 16 GB RAM
19. `44.0.4` (phase 44, Bible line 2138): 3–7 workers depending on memory
20. `44.0.5` (phase 44, Bible line 2139): memory-aware scheduling

## Failed (46)

- `3.1.7` random pair sampling -> failed verdict recorded in tests/test_candidates.py::test_generator_reserves_budget_for_context_pairs_and_random_pairs
- `5.0.8` context -> failed verdict recorded in tests/test_candidates.py::test_generator_reserves_budget_for_context_pairs_and_random_pairs
- `5.0.11` last validation -> failed verdict recorded in state/research/pattern_bank/bank_v3/bank_v000004.json
- `9.1.7` date -> failed verdict recorded in tests/test_antimemo.py::test_ticker_keyed_memory_is_attributed_to_names_not_dates
- `10.0.2` missed winners -> failed verdict recorded in state/research/memory_adapter/results.json
- `10.0.9` context -> failed verdict recorded in tests/test_candidates.py::test_generator_reserves_budget_for_context_pairs_and_random_pairs
- `11.0.7` improvement on A -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `11.0.9` improvement on B -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `13.0.4` missed-winner detector -> failed verdict recorded in state/research/memory_adapter/results.json
- `14.0.2` mu_raw -> failed verdict recorded in state/research/memory_adapter/results.json
- `14.0.5` log_dv -> failed verdict recorded in state/research/memory_adapter/results.json
- `14.0.6` r5 -> failed verdict recorded in state/research/memory_adapter/results.json
- `17.0.1` regime -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `17.0.6` k -> failed verdict recorded in state/research/timeline_basis/report_seed0.json
- `17.0.9` improvement -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `17.0.12` stability across eras -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `18.0.12` improvement -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `19.0.14` detector max -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `20.TIER-2.3` weeks outside band -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `20.TIER-3.1` positive in-band weeks -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `23.0.1` holdings -> failed verdict recorded in state/research/blind_gates/report.md
- `23.0.2` trades -> failed verdict recorded in state/research/blind_gates/report.md
- `25.0.3` P(real) calibration -> failed verdict recorded in state/research/algorithm/planted/summary.json
- `26.B.1` B. Label permutation: Random labels should destroy apparent predictive performance -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `26.C.1` C. Ticker permutation: Ticker identities should not create fake signal -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `26.F.1` F. Feature shuffle: Important features should lose signal when shuffled -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `30.WHAT-IMPROVED.1` What improved? -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `34.0.5` shuffled feature -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `36.GATES.2` GATES: retester -> failed verdict recorded in state/research/blind_gates/results.json
- `38.ALGORITHM.5` ALGORITHM: Long-term pattern bank -> derived from 11 requirements: 1 [~], 3 [?], 5 [x], 2 [!]
- `38.ALGORITHM.12` ALGORITHM: Timeline dial -> derived from 12 requirements: 1 [ ], 3 [~], 4 [x], 4 [!]
- `38.ALGORITHM.14` ALGORITHM: Rerun anti-memorization test -> derived from 13 requirements: 1 [ ], 3 [~], 2 [?], 5 [x], 2 [!]
- `38.ALGORITHM.17` ALGORITHM: Planted-pattern calibration -> derived from 6 requirements: 3 [ ], 2 [x], 1 [!]
- `38.TEST.3` TEST: Tiered 7% objective -> derived from 9 requirements: 3 [~], 4 [x], 2 [!]
- `38.TEST.9` TEST: Re-tester parity -> derived from 7 requirements: 2 [~], 3 [x], 2 [!]
- `38.TEST.14` TEST: Label permutation -> derived from 1 requirements: 1 [!]
- `38.TEST.15` TEST: Feature shuffle -> derived from 1 requirements: 1 [!]
- `38.TEST.16` TEST: Ticker permutation -> derived from 1 requirements: 1 [!]
- `38.TEST.17` TEST: Planted-pattern calibration -> derived from 6 requirements: 3 [ ], 2 [x], 1 [!]
- `39.0.6` re-tester works -> derived from 7 requirements: 2 [~], 3 [x], 2 [!]
- `39.0.18` anti-memorization test works -> derived from 13 requirements: 1 [ ], 3 [~], 2 [?], 5 [x], 2 [!]
- `39.0.23` timeline dial exists and is tested -> derived from 12 requirements: 1 [ ], 3 [~], 4 [x], 4 [!]
- `39.0.25` planted-pattern calibration works -> derived from 6 requirements: 3 [ ], 2 [x], 1 [!]
- `42.0.6` permutation evidence -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `43.0.3` Pattern validation -> failed verdict recorded in state/research/pattern_bank/bank_v3/bank_v000004.json
- `43.0.8` Exit/stop -> failed verdict recorded in tests/test_gaprisk.py::test_empty_and_unfitted_and_stop_rule_hook

## Bible claims the trace cannot support (34)

- `0.1.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.2`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.4`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.6`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `2.0.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.1`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.2`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.4`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.5`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.6`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.8`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.9`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.1.10`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `3.5.1`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `4.0.1`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `4.0.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `4.0.4`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `4.0.8`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `4.0.9`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `38.ALGORITHM.2` A2 claim [x/~] vs computed [~]: Bible marks this built/validated; the trace cannot support that
- `38.ALGORITHM.3` A2b claim [x/~] vs computed [~]: Bible marks this built/validated; the trace cannot support that
- `38.ALGORITHM.9` A8 claim [x/~] vs computed [~]: Bible marks this built/validated; the trace cannot support that
- `38.ALGORITHM.10` A8b claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
... and 9 more (all in bible_trace.json).

## Task-file output not on disk

- engine/fv_pipeline.py
- scripts/fv_eval.py
- tests/test_fv_pipeline.py

## Code no requirement links to

- engine/candles.py
- engine/config.py
- engine/edgar.py
- engine/explain.py
- engine/model.py
- engine/options.py
- engine/scoring.py
- engine/shadows.py
- engine/site_data.py
- engine/tick.py
- engine/train.py
- engine/universe.py
- scripts/algo_test.py
- scripts/analog_test.py
- scripts/archive_opens.py
- scripts/audit_registry.py
- scripts/backfill_market_snaps.py
- scripts/backtest_summary.py
- scripts/bootstrap_history.py
- scripts/diag.py
- scripts/download_1962.py
- scripts/experiment.py
- scripts/extend_history.py
- scripts/fetch_macro.py
- scripts/frontier.py
- scripts/grid_runner.py
- scripts/loop_years.py
- scripts/miner_tune_real.py
- scripts/movers.py
- scripts/movers_compare.py
- scripts/patterns.py
- scripts/planted_calibration.py
- scripts/regen_weekly_snaps.py
- scripts/repair_history.py
- scripts/replay_year.py
- scripts/repro_w01c.py
- scripts/research.py
- scripts/run_gaprisk.py
- scripts/sector_cap.py
- scripts/sensitivity.py
- scripts/site_build.py
- scripts/site_publish.py
- scripts/site_safety.py
- scripts/smoke.py
- scripts/sync_intraday.py
- scripts/three_way.py
- scripts/topk_rules.py
- scripts/tuning_lab.py
- scripts/tuning_report.py

## Integration hooks still unchecked (code exists, not wired)

- adaptive.py: 1 pending, e.g. adaptive.py: META_DEFAULT det_max/det_min_weeks; dial hook (sent to B15)
- antioverfit_real.py: 1 pending, e.g. mutable default scripts/antioverfit_real.py:43; tests/test_livesim_gates.py:172 has no assert
- backtest.py: 1 pending, e.g. backtest.py + livesim.py: fills via PITStore.executor(as_of).fill_next_open; run pit.audit_fills on…
- improve.py: 1 pending, e.g. major runs call checkpoint.write_checkpoint, run_report.build_report/write_report; Board.promote re…
- livesim.py: 1 pending, e.g. backtest.py + livesim.py: fills via PITStore.executor(as_of).fill_next_open; run pit.audit_fills on…
- quality_gate.py: 1 pending, e.g. CI: quality_gate.py --baseline state/quality/baseline.json
- repro.py: 1 pending, e.g. engine/repro.py seeds global RNGs; tests/test_repro.py draws from them
- site_build.py: 1 pending, e.g. CI: site_build.py --verify; optional site_publish.py --interval 300 (no --push)
- site_publish.py: 1 pending, e.g. CI: site_build.py --verify; optional site_publish.py --interval 300 (no --push)
- test_livesim_gates.py: 1 pending, e.g. mutable default scripts/antioverfit_real.py:43; tests/test_livesim_gates.py:172 has no assert
- test_repro.py: 1 pending, e.g. engine/repro.py seeds global RNGs; tests/test_repro.py draws from them
- tick.py: 1 pending, e.g. import boundaries: parity_suite -> engine.live; tick.py -> alpaca.trading; research scripts data_li…
- train.py: 2 pending, e.g. train.py: pit.purged_training_set(X, y, as_of, horizon, cal) instead of ad-hoc label cutting

## Trace invariants (0 violations)

All hold.

## Detail by phase


### Phase 0: BASELINE AND CONTROL SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 0.1.1 | [~] | Read CANON.md | engine/champion.py::_canon | 1 | 2 |
| 0.1.2 | [~] | Hash the canon | engine/champion.py::_canon<br>engine/provenance.py::code_hash | 3 | 21 |
| 0.1.3 | [x] | Verify current hash | engine/basis_search.py::_hash<br>engine/data_sources.py::frame_hash | 7 | 18 |
| 0.1.4 | [~] | Add a canon integrity check | engine/provenance.py::IntegrityError | 2 | 19 |
| 0.1.5 | [~] | Ensure experiments record the canon hash | engine/champion.py::_canon<br>engine/memory.py::record | 5 | 20 |
| 0.1.6 | [~] | Ensure experiments record blueprint version | engine/champion.py::_canon<br>engine/provenance.py | 3 | 21 |
| 0.1.7 | [~] | Ensure code commits are recorded | engine/provenance.py::code_hash | 2 | 19 |
| 0.2.1 | [x] | experiment ID | engine/registry.py::ExperimentMemory<br>engine/antimemo.py::archive_experiment | 9 | 7 |
| 0.2.2 | [~] | timestamp | engine/registry.py::ExperimentMemory | 4 | 2 |
| 0.2.3 | [~] | git commit | engine/provenance.py::git_commit<br>engine/registry.py::ExperimentMemory | 2 | 19 |
| 0.2.4 | [x] | canon hash | engine/data_sources.py::DelistedRegistry<br>engine/provenance.py::code_hash | 7 | 32 |
| 0.2.5 | [~] | blueprint version | engine/provenance.py<br>engine/improve.py::log_experiment | 3 | 19 |
| 0.2.6 | [~] | configuration hash | scripts/final_report.py::cfg_hash<br>engine/baseline.py | 7 | 4 |
| 0.2.7 | [~] | data snapshot | engine/antimemo.py::archive_experiment<br>engine/baseline.py | 3 | 3 |
| 0.2.8 | [~] | random seed | engine/antimemo.py::archive_experiment<br>engine/registry.py::ExperimentMemory | 3 | 3 |
| 0.2.9 | [?] | window IDs | engine/antimemo.py::archive_experiment<br>engine/registry.py::ExperimentMemory | 3 | 3 |
| 0.2.10 | [~] | model parameters | engine/experiment_memory.py::import_registry<br>engine/antimemo.py::archive_experiment | 7 | 24 |
| 0.2.11 | [~] | training range | engine/experiment_memory.py::import_registry<br>engine/registry.py::ExperimentMemory | 2 | 0 |
| 0.2.12 | [~] | validation range | engine/experiment_memory.py::import_registry<br>engine/exits.py::validate | 4 | 19 |
| 0.2.13 | [~] | test range | engine/experiment_memory.py::import_registry<br>engine/registry.py::ExperimentMemory | 2 | 0 |
| 0.2.14 | [~] | metrics | engine/registry.py::ExperimentMemory<br>engine/exits.py::band_metrics | 6 | 21 |
| 0.2.15 | [~] | gates | engine/registry.py::ExperimentMemory | 4 | 2 |
| 0.2.16 | [~] | outcome | engine/registry.py::ExperimentMemory<br>engine/antimemo.py::archive_experiment | 9 | 7 |
| 0.2.17 | [x] | reason for adoption/rejection | engine/experiment_memory.py::import_registry<br>engine/run_report.py | 2 | 0 |
| 0.3.1 | [~] | config | engine/checkpoint.py::CheckpointError | 2 | 0 |
| 0.3.2 | [~] | logs | engine/checkpoint.py::CheckpointError | 2 | 0 |
| 0.3.3 | [~] | metrics | engine/baseline.py::extract_metrics<br>engine/checkpoint.py::CheckpointError | 3 | 0 |
| 0.3.4 | [~] | artifact manifest | engine/checkpoint.py::CheckpointError | 2 | 0 |
| 0.3.5 | [~] | random seeds | engine/checkpoint.py::CheckpointError | 2 | 0 |
| 0.3.6 | [~] | hashes | engine/checkpoint.py::CheckpointError | 2 | 0 |
| 0.3.7 | [~] | output summary | engine/pit.py::summary<br>engine/checkpoint.py::CheckpointError | 1 | 1 |
| 0.4.1 | [~] | current blind-window results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::BlindGateError | 10 | 2 |
| 0.4.2 | [~] | current mover accuracy | engine/baseline.py::diff_vs_baseline<br>engine/champion.py::freeze_baseline | 2 | 0 |
| 0.4.3 | [~] | current model results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::snapshot_from | 10 | 2 |
| 0.4.4 | [~] | current pattern results | engine/baseline.py::diff_vs_baseline<br>engine/adaptive.py::from_snapshot | 13 | 4 |
| 0.4.5 | [~] | current analog results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::snapshot_from | 13 | 4 |
| 0.4.6 | [x] | current weekly distribution | scripts/final_report.py::distribution<br>engine/baseline.py::diff_vs_baseline | 1 | 0 |
| 0.4.7 | [~] | max drawdown | engine/run_report.py::max_drawdown<br>engine/analog_weighting.py::baseline | 2 | 0 |
| 0.4.8 | [~] | worst weeks | engine/live.py::week_state<br>engine/baseline.py::diff_vs_baseline | 2 | 0 |
| 0.4.9 | [~] | turnover | engine/run_report.py::baseline_metrics<br>engine/baseline.py::diff_vs_baseline | 3 | 2 |
| 0.4.10 | [~] | costs | engine/baseline.py::diff_vs_baseline<br>engine/champion.py::freeze_baseline | 3 | 2 |
| 0.4.11 | [~] | missed winners | engine/baseline.py::diff_vs_baseline<br>engine/champion.py::freeze_baseline | 3 | 2 |
| 0.4.12 | [~] | direction accuracy where available | engine/baseline.py::diff_vs_baseline<br>engine/run_report.py::baseline_metrics | 2 | 0 |

### Phase 1: POINT-IN-TIME DATA FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 1.1.1 | [~] | Every feature carries an effective date | engine/lessons.py::usable_features<br>engine/antioverfit.py::date_disguise | 3 | 11 |
| 1.1.2 | [x] | Every filing carries publication/availability date | engine/pit.py::IntegrityError<br>engine/data_sources.py::_rank_ic_by_date | 2 | 2 |
| 1.1.3 | [x] | Every macro series has publication lag | engine/pit.py::IntegrityError<br>scripts/pit_audit_real.py::macro_lags | 1 | 1 |
| 1.1.4 | [x] | Every label has a close date | engine/pit.py::label_close_dates<br>scripts/pit_audit_real.py::panel_integrity | 4 | 5 |
| 1.1.5 | [~] | Every training row proves its label closed before prediction | engine/blind_gates.py::check_label_alignment<br>engine/pit.py::label_close_dates | 3 | 4 |
| 1.1.6 | [x] | Future rows cannot be queried | engine/pit.py::IntegrityError<br>engine/analog_weighting.py::query | 2 | 1 |
| 1.2.1 | [~] | 1.2 Time-fence enforcement: Implement a hard object-level time fence | engine/pit.py::fence | 1 | 1 |
| 1.3.1 | [x] | predictions unchanged | engine/pit.py::future_scramble<br>engine/antioverfit.py::future_scramble | 7 | 9 |
| 1.3.2 | [x] | scores unchanged | engine/pit.py::future_scramble<br>engine/ablation.py::score | 5 | 10 |
| 1.3.3 | [x] | pattern selection unchanged | engine/pit.py::future_scramble | 1 | 1 |
| 1.3.4 | [x] | memory unchanged | engine/livesim.py::long_term_memory<br>engine/pit.py::future_scramble | 4 | 6 |
| 1.3.5 | [x] | adaptive decisions unchanged | engine/pit.py::future_scramble<br>engine/blind_gates.py::check_decisions_use_recorded_info | 6 | 7 |
| 1.4.1 | [x] | decision after close | engine/pit.py::audit_fills<br>scripts/pit_audit_real.py::live_decisions | 4 | 5 |
| 1.4.2 | [x] | fill next open | engine/pit.py::fill_next_open<br>scripts/pit_audit_real.py | 5 | 5 |
| 1.4.3 | [x] | no same-close execution | engine/pit.py::audit_fills<br>scripts/pit_audit_real.py | 4 | 5 |
| 1.4.4 | [~] | no next-day information leakage | engine/livesim.py::_information_sources<br>engine/blind_gates.py::InformationLedger | 2 | 2 |

### Phase 2: FEATURE/PARITY FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 2.0.1 | [~] | feature parity harness | engine/parity.py::FeatureContract<br>engine/parity_suite.py::analog_find_parity | 2 | 2 |
| 2.0.2 | [?] | random-date sampling | engine/parity.py::sample_dates<br>engine/antioverfit.py::date_disguise | 6 | 7 |
| 2.0.3 | [x] | strict recomputation | engine/parity.py::strict_row<br>engine/parity_suite.py::strict | 2 | 2 |
| 2.0.4 | [~] | fast recomputation | engine/parity.py<br>scripts/run_parity.py::fast_git_commit | 2 | 2 |
| 2.0.5 | [~] | exact comparison | engine/parity.py | 2 | 2 |
| 2.0.6 | [~] | maximum absolute error | engine/adaptive.py::GuardRailError<br>engine/parity.py::max_abs | 8 | 2 |
| 2.0.7 | [ ] | maximum relative error | - | 0 | 0 |
| 2.0.8 | [ ] | NaN mismatch detection | - | 0 | 0 |
| 2.0.9 | [~] | missing-data mismatch detection | scripts/bible_trace.py::top_missing<br>engine/data_sources.py::detect_splits | 2 | 1 |

### Phase 3: PATTERN MINER HARDENING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 3.1.1 | [x] | all 54 engine features | engine/candidates.py::Candidate<br>engine/pattern_identity.py::features | 2 | 4 |
| 3.1.2 | [x] | all 25 candle/micro signals | engine/candidates.py::Candidate<br>engine/patterns.py::with_candles | 2 | 4 |
| 3.1.3 | [x] | market context | engine/candidates.py::context_pair_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.4 | [x] | quintile transformation | engine/candidates.py::Candidate<br>engine/planted.py::generate | 4 | 10 |
| 3.1.5 | [x] | single patterns | engine/candidates.py::single_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.6 | [x] | pair patterns | engine/candidates.py::context_pair_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.7 | [!] | random pair sampling | engine/planted.py::generate<br>engine/candidates.py::random_pair_candidates | 7 | 16 |
| 3.1.8 | [x] | A AND B UNLESS C | engine/candidates.py::unless_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.9 | [x] | bank re-testing | engine/candidates.py::bank_candidates<br>engine/patterns.py::export_bank | 2 | 4 |
| 3.1.10 | [x] | deterministic candidate IDs | engine/candidates.py::Candidate<br>engine/planted.py::generate | 4 | 10 |
| 3.2.1 | [x] | canonical expression | engine/pattern_identity.py::canonical_expression<br>engine/patterns.py::PatternMiner | 10 | 8 |
| 3.2.2 | [x] | hash | engine/candidates.py<br>engine/pattern_identity.py::pattern_hash | 3 | 4 |
| 3.2.3 | [~] | feature set | engine/candidates.py::features_used<br>engine/pattern_identity.py::IdentityError | 12 | 12 |
| 3.2.4 | [~] | transformation | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 10 | 8 |
| 3.2.5 | [~] | target | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 3 | 3 |
| 3.2.6 | [~] | discovery dates | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 12 | 11 |
| 3.2.7 | [~] | validation dates | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 12 | 11 |
| 3.2.8 | [x] | context scope | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 10 | 8 |
| 3.2.9 | [~] | statistics | engine/pattern_identity.py::IdentityError<br>engine/patterns.py::PatternMiner | 10 | 8 |
| 3.2.10 | [x] | lifecycle state | engine/pattern_identity.py::IdentityError<br>engine/pattern_lifecycle.py::Lifecycle | 5 | 3 |
| 3.3.1 | [?] | 3.3 Relevance weighting: relevance = recency_weight × context_similarity_weight × era_weight | engine/pattern_stats.py::context_similarity<br>engine/patterns.py::_weights | 14 | 26 |
| 3.4.1 | [~] | week-clustered observations | engine/pattern_stats.py::validation_gain_gate<br>engine/patterns.py | 10 | 8 |
| 3.4.2 | [x] | weighted mean | engine/pattern_stats.py::weighted_mean<br>engine/patterns.py::_weights | 12 | 26 |
| 3.4.3 | [~] | effective sample size | engine/pattern_stats.py::effective_n<br>engine/memory.py::validate | 1 | 3 |
| 3.4.4 | [x] | t-statistic | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 12 | 26 |
| 3.4.5 | [~] | p-value | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 10 | 8 |
| 3.4.6 | [x] | FDR | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 12 | 26 |
| 3.4.7 | [~] | permutation null | engine/patterns.py<br>engine/antioverfit.py::_null_summary | 12 | 10 |
| 3.4.8 | [x] | local false-discovery estimate | engine/pattern_stats.py::local_fdr<br>engine/patterns.py | 1 | 3 |
| 3.4.9 | [~] | later confirmation | engine/pattern_stats.py::confirmation_factor<br>engine/patterns.py | 10 | 8 |
| 3.4.10 | [~] | P(real) | engine/pattern_stats.py::p_real<br>engine/patterns.py | 14 | 29 |
| 3.4.11 | [~] | effect shrinkage | engine/pattern_stats.py::effective_n<br>engine/patterns.py | 10 | 8 |
| 3.5.1 | [x] | P(real) >= 0.80 | engine/pattern_stats.py::admission_flags<br>engine/patterns.py::PatternMiner | 1 | 3 |
| 3.5.2 | [~] | survives redundancy rules | engine/pattern_stats.py::admission_flags<br>engine/patterns.py::PatternMiner | 1 | 3 |
| 3.5.3 | [?] | passes validation-gain threshold | engine/pattern_stats.py::validation_gain_gate<br>engine/heavy_tests.py::_oos_pattern_check | 1 | 3 |
| 3.5.4 | [?] | confirmation preserves sign | engine/pattern_stats.py::admission_flags<br>engine/pattern_identity.py::PatternBook | 1 | 3 |
| 3.5.5 | [ ] | does not violate anti-leak rules | - | 0 | 0 |
| 3.6.1 | [~] | 3.6 Redundancy: Implement overlap comparison | engine/patterns.py::_overlap | 10 | 7 |
| 3.7.1 | [?] | 3.7 Validation-gain gate: The pattern must improve genuinely unseen prediction quality | engine/patterns.py::PatternMiner<br>engine/pattern_stats.py::validation_gain_gate | 10 | 8 |

### Phase 4: PATTERN LIFECYCLE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 4.0.1 | [x] | failure detector | engine/pattern_lifecycle.py::detect_failure | 2 | 1 |
| 4.0.2 | [~] | context terciles | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 3 | 14 |
| 4.0.3 | [x] | rescoping search | engine/pattern_lifecycle.py::_resolve_search<br>engine/patterns.py::search | 10 | 7 |
| 4.0.4 | [x] | long-run test | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 6 | 17 |
| 4.0.5 | [~] | discovery test | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 5 | 14 |
| 4.0.6 | [?] | confirmation test | engine/pattern_lifecycle.py<br>engine/pattern_stats.py::confirmation_factor | 3 | 3 |
| 4.0.7 | [~] | recent-stretch test | engine/pattern_lifecycle.py | 2 | 1 |
| 4.0.8 | [x] | sign-consistency test | engine/pattern_lifecycle.py::_sign | 2 | 1 |
| 4.0.9 | [x] | discard reason | engine/pattern_lifecycle.py<br>scripts/run_pattern_bank.py | 2 | 1 |

### Phase 5: LONG-TERM PATTERN BANK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 5.0.1 | [x] | pattern identity | engine/pattern_lifecycle.py::pattern_id<br>engine/pattern_identity.py::IdentityError | 11 | 8 |
| 5.0.2 | [?] | history | engine/pattern_bank.py::history<br>engine/pattern_lifecycle.py | 7 | 20 |
| 5.0.3 | [?] | discovery evidence | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 4 | 16 |
| 5.0.4 | [?] | failure evidence | engine/pattern_bank.py<br>engine/pattern_lifecycle.py::detect_failure | 5 | 16 |
| 5.0.5 | [x] | rescoping | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 3 | 14 |
| 5.0.6 | [~] | current relevance | engine/pattern_bank.py | 2 | 14 |
| 5.0.7 | [x] | modernity | engine/pattern_bank.py | 2 | 14 |
| 5.0.8 | [!] | context | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 6 | 17 |
| 5.0.9 | [x] | effect | engine/pattern_bank.py<br>engine/pattern_stats.py::effective_n | 5 | 34 |
| 5.0.10 | [x] | confidence | engine/pattern_bank.py<br>engine/analog_weighting.py::confidence_calibration | 5 | 14 |
| 5.0.11 | [!] | last validation | engine/pattern_bank.py<br>engine/adaptive.py::validate_cfg | 13 | 36 |

### Phase 6: PATTERN → FIND VOLATILITY INTEGRATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 6.0.1 | [x] | pattern movement score | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::MovementCalibrator | 2 | 1 |
| 6.0.2 | [x] | movement probability | engine/pattern_movers.py::MovementCalibrator | 2 | 1 |
| 6.0.3 | [~] | pattern contribution attribution | engine/pattern_movers.py::PatternMoverModel | 2 | 1 |
| 6.0.4 | [x] | interaction with existing mover model | engine/pattern_movers.py::MoverModel<br>engine/livesim.py::_fit_mover | 2 | 1 |
| 6.0.5 | [~] | out-of-sample comparison | engine/pattern_movers.py<br>engine/parity.py::sample_dates | 4 | 3 |
| 6.0.6 | [x] | mover model alone | engine/heavy_tests.py<br>engine/pattern_movers.py::MoverModel | 2 | 1 |
| 6.0.7 | [~] | pattern score alone | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 4 | 7 |
| 6.0.8 | [x] | mover + pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 2 | 3 |
| 6.0.9 | [x] | mover + random pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 2 | 1 |
| 6.0.10 | [x] | mover + shuffled pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 2 | 2 |

### Phase 7: HEAVY ALGORITHM TESTING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 7.0.1 | [~] | multiple eras | engine/heavy_tests.py::ablation_across_eras | 2 | 0 |
| 7.0.2 | [~] | bull markets | engine/heavy_tests.py | 2 | 0 |
| 7.0.3 | [~] | bear markets | engine/heavy_tests.py | 2 | 0 |
| 7.0.4 | [~] | high-volatility regimes | engine/heavy_tests.py::by_regime<br>engine/gaprisk.py::_week_regime | 3 | 8 |
| 7.0.5 | [?] | low-volatility regimes | engine/heavy_tests.py::by_regime<br>engine/pattern_movers.py | 3 | 9 |
| 7.0.6 | [~] | pre-decimal era | engine/heavy_tests.py | 2 | 0 |
| 7.0.7 | [~] | post-decimal era | engine/heavy_tests.py | 2 | 0 |
| 7.0.8 | [~] | post-electronic era | engine/heavy_tests.py | 2 | 0 |
| 7.0.9 | [~] | modern market | engine/heavy_tests.py | 2 | 0 |
| 7.0.10 | [~] | random hidden windows | engine/heavy_tests.py::hidden_windows<br>scripts/quality_gate.py::check_hidden_global_state | 3 | 0 |
| 7.0.11 | [x] | movement prediction | engine/heavy_tests.py<br>engine/pattern_movers.py::MovementCalibrator | 7 | 39 |
| 7.0.12 | [~] | direction prediction | engine/heavy_tests.py<br>engine/pattern_movers.py::_direction_accuracy | 5 | 35 |
| 7.0.13 | [~] | IC | engine/heavy_tests.py::run_heavy<br>scripts/heavy_algo_real.py | 2 | 0 |
| 7.0.14 | [x] | rank IC | engine/heavy_tests.py<br>engine/pattern_movers.py | 3 | 1 |
| 7.0.15 | [x] | t-stat | engine/heavy_tests.py::series_stats<br>scripts/heavy_algo_real.py | 11 | 3 |
| 7.0.16 | [~] | false discoveries | engine/heavy_tests.py::false_discovery_summary | 2 | 0 |
| 7.0.17 | [x] | pattern count | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 4 | 5 |
| 7.0.18 | [~] | pattern survival | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 5 | 3 |
| 7.0.19 | [x] | turnover | engine/heavy_tests.py<br>engine/retester.py::turnover | 3 | 1 |
| 7.0.20 | [x] | costs | engine/heavy_tests.py::cost_sensitivity<br>scripts/heavy_algo_real.py | 4 | 20 |
| 7.0.21 | [x] | stability | engine/heavy_tests.py::stability<br>engine/exits.py::selection_stability | 7 | 23 |
| 7.0.22 | [~] | regime sensitivity | engine/heavy_tests.py::by_regime | 2 | 0 |

### Phase 8: ANALOG ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 8.1.1 | [~] | market drawdown | engine/analog_weighting.py::market_context<br>engine/analogs_sector.py::sector_fingerprints | 2 | 1 |
| 8.1.2 | [x] | 1m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 2 |
| 8.1.3 | [x] | 3m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 2 |
| 8.1.4 | [x] | 6m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 2 |
| 8.1.5 | [x] | 12m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 2 |
| 8.1.6 | [~] | acceleration | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 2 | 1 |
| 8.1.7 | [~] | distance from MA200 | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 1 | 0 |
| 8.1.8 | [~] | volatility | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 3 | 0 |
| 8.1.9 | [~] | volatility ratio | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 3 | 0 |
| 8.1.10 | [x] | VIX | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 2 | 1 |
| 8.1.11 | [~] | VIX term structure | engine/analog_weighting.py<br>engine/analogs.py::fingerprints | 2 | 1 |
| 8.1.12 | [x] | breadth | engine/analogs_sector.py::breadth_dispersion_crowding<br>engine/analogs.py::fingerprints | 2 | 1 |
| 8.1.13 | [x] | dispersion | engine/analogs_sector.py::breadth_dispersion_crowding<br>engine/analogs.py::fingerprints | 4 | 1 |
| 8.1.14 | [x] | sector crowding | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs.py::fingerprints | 2 | 1 |
| 8.1.15 | [x] | top-sector concentration change | engine/analogs_sector.py::sector_fingerprints | 1 | 0 |
| 8.1.16 | [~] | FRED data | engine/parity_suite.py::fingerprints_builder<br>engine/analogs_stock.py::data_sufficient | 1 | 2 |
| 8.1.17 | [~] | 10Y yield change | engine/analogs_sector.py::sector_fingerprints | 1 | 0 |
| 8.1.18 | [~] | credit-spread change | engine/pit.py::fingerprint<br>engine/analogs_sector.py::sector_fingerprints | 1 | 1 |
| 8.2.1 | [~] | 8.2 Point-in-time standardization: Standardize using only historical information available at the m… | engine/analog_weighting.py::pit_moments | 1 | 0 |
| 8.3.1 | [~] | analog end >= 63 sessions before prediction | engine/analogs.py::Analogs<br>engine/analog_weighting.py::predict_series | 2 | 1 |
| 8.3.2 | [~] | analogs at least 21 sessions apart | engine/analogs.py::Analogs<br>engine/analogs_stock.py::PooledStockAnalogs | 2 | 1 |
| 8.3.3 | [~] | one analog per episode | engine/analogs.py::Analogs<br>engine/analog_weighting.py::select_episodes | 1 | 1 |
| 8.4.1 | [x] | 8.4 Feature weighting: Implement learnable feature weights | engine/analog_weighting.py::WeightSchedule<br>engine/analogs_sector.py::panel_features | 1 | 0 |
| 8.5.1 | [x] | 8.5 Sector analogs: Implement sector fingerprints | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 2 | 1 |
| 8.6.1 | [x] | 8.6 Stock-level analogs: Implement stock-level fingerprints where data suffices | engine/analogs_stock.py::PooledStockAnalogs<br>engine/analog_weighting.py::level_agreement | 1 | 0 |
| 8.7.1 | [~] | analog dates | engine/analog_weighting.py<br>engine/timeline.py::DialOutput | 3 | 6 |
| 8.7.2 | [~] | distances | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.7.3 | [~] | ages | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.7.4 | [x] | forecast return | engine/analog_weighting.py::blend_forecasts<br>engine/timeline.py::forecast_from_analogs | 3 | 6 |
| 8.7.5 | [~] | volatility | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.7.6 | [~] | drawdown | engine/analog_weighting.py<br>engine/timeline.py::DialOutput | 3 | 6 |
| 8.7.7 | [~] | uniqueness | engine/analogs_sector.py::SectorAnalogs<br>engine/analogs_stock.py::PooledStockAnalogs | 1 | 0 |
| 8.7.8 | [~] | analog count | engine/analog_weighting.py<br>engine/timeline.py::DialOutput | 3 | 6 |
| 8.7.9 | [~] | confidence | engine/analog_weighting.py::confidence_calibration<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.1 | [x] | no analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.2 | [x] | market analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.3 | [x] | sector analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.4 | [x] | stock analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.5 | [x] | shuffled analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |
| 8.8.6 | [x] | nearest-neighbor random control | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 0 |

### Phase 9: MEMORY SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 9.0.1 | [x] | recency | engine/memory.py<br>engine/memory_diagnostics.py | 8 | 21 |
| 9.0.2 | [x] | market similarity | engine/memory.py<br>engine/memory_diagnostics.py | 5 | 21 |
| 9.0.3 | [~] | reliability shrinkage | engine/memory.py::reliability | 4 | 18 |
| 9.0.4 | [x] | shock detection | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::detect_lag | 4 | 18 |
| 9.0.5 | [x] | long-term prior | engine/memory.py<br>engine/pattern_bank.py::prior_frame | 6 | 32 |
| 9.0.6 | [x] | modernity | engine/memory.py<br>engine/memory_diagnostics.py | 4 | 18 |
| 9.1.1 | [~] | situation fingerprint | engine/lessons.py::_situation<br>engine/memory.py::Memory | 11 | 32 |
| 9.1.2 | [~] | relevant features | engine/memory.py::Memory<br>engine/data_sources.py::keep_relevant_forms | 4 | 18 |
| 9.1.3 | [~] | context | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 7 | 29 |
| 9.1.4 | [~] | outcome | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 9 | 29 |
| 9.1.5 | [x] | error type | engine/memory.py::Memory<br>engine/memory_diagnostics.py::error_profile_by_arm | 4 | 18 |
| 9.1.6 | [~] | source experiment | engine/memory.py::Memory<br>engine/isolation.py::RecordingClient | 4 | 18 |
| 9.1.7 | [!] | date | engine/memory.py::Memory<br>engine/blind_gates.py::check_decisions_use_recorded_info | 20 | 39 |
| 9.1.8 | [?] | era | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 7 | 21 |
| 9.1.9 | [~] | relevance | engine/memory.py::Memory<br>engine/patterns.py::export_records | 14 | 25 |
| 9.1.10 | [~] | reliability | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 9.1.11 | [x] | shock state | engine/memory.py::shock_state<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 9.2.1 | [x] | ticker | engine/memory.py::coarse_date | 4 | 18 |
| 9.2.2 | [x] | exact future outcome | engine/memory.py::coarse_date<br>engine/antioverfit.py::_relabel_tickers | 7 | 24 |
| 9.2.3 | [?] | hidden test identifier | engine/memory.py::coarse_date | 4 | 18 |
| 9.2.4 | [?] | information that uniquely identifies a test window | engine/antioverfit.py::_relabel_tickers<br>engine/memory.py::coarse_date | 6 | 21 |
| 9.3.1 | [~] | Adapter: 8 weeks | engine/memory.py<br>scripts/run_b15_memory_adapter.py::build_weeks | 4 | 18 |
| 9.3.2 | [~] | miner: 4 years | engine/memory.py<br>engine/pattern_stats.py::recency_weight | 14 | 26 |
| 9.4.1 | [ ] | 9.4 Market similarity: Implement Gaussian similarity | - | 0 | 0 |
| 9.5.1 | [~] | 9.5 Reliability shrinkage: Implement pseudo-count shrinkage | engine/memory.py::counts | 4 | 18 |
| 9.6.1 | [~] | 9.6 Shock detection: Implement two-sided CUSUM | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::detect_lag | 4 | 18 |

### Phase 10: LESSON MEMORY / LEARNING FROM MISTAKES

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 10.0.1 | [~] | losing decisions | engine/lessons.py | 3 | 11 |
| 10.0.2 | [!] | missed winners | engine/missed_winners.py::MissedWinnerDetector | 2 | 4 |
| 10.0.3 | [ ] | false positives | - | 0 | 0 |
| 10.0.4 | [ ] | false negatives | - | 0 | 0 |
| 10.0.5 | [x] | pattern failures | engine/exits.py::pattern_fail_flags<br>engine/parity.py::ParityFailure | 7 | 35 |
| 10.0.6 | [~] | regime failures | engine/antioverfit.py::regime_labels | 2 | 3 |
| 10.0.7 | [~] | situation | engine/lessons.py::_situation | 3 | 11 |
| 10.0.8 | [x] | features | engine/antimemo.py::snap_features<br>engine/lessons.py::usable_features | 5 | 14 |
| 10.0.9 | [!] | context | engine/lessons.py<br>engine/analog_weighting.py::market_context | 10 | 33 |
| 10.0.10 | [x] | decision | engine/antimemo.py<br>engine/lessons.py | 6 | 15 |
| 10.0.11 | [x] | outcome | engine/antimemo.py<br>engine/lessons.py | 6 | 14 |
| 10.0.12 | [x] | error category | engine/lessons.py::category_counts | 3 | 11 |
| 10.0.13 | [x] | confidence | engine/lessons.py<br>engine/analog_weighting.py::confidence_calibration | 6 | 11 |
| 10.0.14 | [x] | counterfactual | engine/lessons.py::_counterfactual | 3 | 11 |

### Phase 11: RERUN ANTI-MEMORIZATION EXPERIMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 11.0.1 | [?] | play window A | engine/antimemo.py::play_window<br>engine/blind_gates.py::seal_window | 7 | 7 |
| 11.0.2 | [~] | create lessons | engine/lessons.py::Lesson<br>scripts/lessons_archive.py | 3 | 11 |
| 11.0.3 | [x] | rerun A under a fresh disguise | engine/antimemo.py::disguise | 3 | 3 |
| 11.0.4 | [~] | measure improvement | engine/antimemo.py::improvement_by_era | 3 | 3 |
| 11.0.5 | [?] | play unseen window B | engine/antimemo.py::play_window | 3 | 3 |
| 11.0.6 | [x] | keep lessons only when B is not harmed | engine/antimemo.py::harmed<br>engine/lessons.py::Lesson | 3 | 3 |
| 11.0.7 | [!] | improvement on A | engine/antimemo.py::improvement_by_era<br>engine/improve.py | 4 | 5 |
| 11.0.8 | [x] | improvement on disguised A | engine/antimemo.py::disguise | 3 | 3 |
| 11.0.9 | [!] | improvement on B | engine/antimemo.py::improvement_by_era<br>engine/improve.py | 4 | 5 |
| 11.0.10 | [ ] | degradation on B | - | 0 | 0 |
| 11.0.11 | [x] | lesson count | engine/lessons.py::Lesson<br>engine/memory.py::Lesson | 11 | 31 |
| 11.0.12 | [~] | accepted lessons | engine/lessons.py::Lesson<br>scripts/lessons_archive.py | 3 | 11 |
| 11.0.13 | [x] | rejected lessons | engine/antimemo.py<br>engine/lessons.py::Lesson | 3 | 3 |

### Phase 12: PER-STOCK-TYPE TRUST TABLES

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 12.0.1 | [x] | SIC division | engine/trust.py::sic_division<br>engine/trust_store.py | 4 | 29 |
| 12.0.2 | [x] | size tercile | engine/trust.py::_tercile_labels | 2 | 27 |
| 12.0.3 | [x] | volatility tercile | engine/trust.py::_tercile_labels<br>scripts/antioverfit_real.py::vol_types | 2 | 30 |
| 12.0.4 | [~] | trend state | engine/trust.py | 2 | 27 |
| 12.0.5 | [ ] | theme/industry momentum | - | 0 | 0 |
| 12.0.6 | [~] | attention state | engine/trust.py | 2 | 27 |
| 12.0.7 | [~] | week-clustered statistics | engine/trust.py::_week_label<br>engine/direction_ablate.py::week_bootstrap | 6 | 48 |
| 12.0.8 | [x] | shrinkage | engine/trust.py | 2 | 27 |
| 12.0.9 | [~] | minimum sample size | engine/health.py::memory_growth_mb_per_min | 2 | 1 |
| 12.0.10 | [x] | confidence | engine/direction.py<br>scripts/direction_study.py | 7 | 34 |
| 12.0.11 | [x] | recent relevance | engine/trust.py<br>engine/trust_store.py | 3 | 30 |

### Phase 13: DIRECTION ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 13.0.1 | [x] | pattern direction | engine/direction.py::DirectionEngine<br>engine/direction_ablate.py | 6 | 34 |
| 13.0.2 | [x] | analogs | engine/direction.py<br>scripts/direction_study.py | 6 | 35 |
| 13.0.3 | [x] | per-type trust | engine/direction.py<br>engine/direction_calib.py::PerTypeCalibrator | 6 | 34 |
| 13.0.4 | [!] | missed-winner detector | engine/missed_winners.py::MissedWinnerDetector | 2 | 4 |
| 13.0.5 | [x] | model prediction | engine/direction.py::predict<br>scripts/direction_study.py | 7 | 42 |
| 13.0.6 | [~] | evidence where justified | engine/direction_ablate.py<br>scripts/bible_trace.py::EvidenceFile | 2 | 1 |
| 13.0.7 | [?] | calibration curves | engine/direction.py<br>engine/missed_winners.py::calibration | 9 | 63 |
| 13.0.8 | [x] | Brier score | engine/direction.py::brier<br>engine/missed_winners.py::_base_score | 7 | 39 |
| 13.0.9 | [x] | log loss | engine/direction.py::log_loss<br>engine/direction_ablate.py | 5 | 42 |
| 13.0.10 | [~] | reliability diagram | engine/direction.py::reliability_bins | 3 | 34 |
| 13.0.11 | [~] | out-of-sample direction accuracy | engine/direction.py::direction_accuracy<br>scripts/direction_study.py::direction_acc_at | 6 | 35 |

### Phase 14: MISSED-WINNER DETECTOR

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 14.0.1 | [?] | evidence ranks | engine/memory.py::rank_arms<br>engine/missed_winners.py::rank_ic | 7 | 22 |
| 14.0.2 | [!] | mu_raw | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.3 | [~] | vol20 | engine/missed_winners.py | 2 | 4 |
| 14.0.4 | [~] | max20 | engine/missed_winners.py | 2 | 4 |
| 14.0.5 | [!] | log_dv | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.6 | [!] | r5 | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.7 | [x] | other approved inputs only if they pass point-in-time rules | engine/pattern_stats.py::_check_point_in_time<br>engine/pit.py::passed | 10 | 8 |
| 14.0.8 | [~] | detector alone | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.9 | [x] | detector + base | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.10 | [x] | shuffled detector | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 14.0.11 | [x] | future-scrambled detector | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 4 |

### Phase 15: EXIT LEARNER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 15.0.1 | [x] | hit rate | engine/exits.py<br>engine/analog_weighting.py::rate | 7 | 24 |
| 15.0.2 | [x] | average gain | engine/exits.py::oos_gain_interval | 2 | 19 |
| 15.0.3 | [~] | average loss | engine/exits.py<br>engine/stops.py::loss_risk | 4 | 19 |
| 15.0.4 | [x] | tail loss | engine/exits.py<br>engine/stops.py::loss_risk | 3 | 26 |
| 15.0.5 | [~] | time held | engine/exits.py | 2 | 19 |
| 15.0.6 | [x] | turnover | engine/retester.py::turnover | 1 | 1 |
| 15.0.7 | [x] | costs | engine/exits.py::CostModel<br>engine/stops.py | 5 | 19 |
| 15.0.8 | [~] | weekly band behavior | engine/exits.py::WeekTable | 2 | 19 |

### Phase 16: STOP / LOSS ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 16.0.1 | [~] | ATR-based | engine/stops.py | 2 | 19 |
| 16.0.2 | [~] | volatility percentile | engine/stops.py::VolPercentileStop | 2 | 19 |
| 16.0.3 | [~] | stock-type specific | engine/stops.py::type_gap_table | 2 | 19 |
| 16.0.4 | [x] | gap-aware | engine/gaprisk.py<br>engine/stops.py::GapAwareStop | 2 | 26 |
| 16.0.5 | [~] | position-size-aware | engine/gaprisk.py::p_position<br>engine/stops.py::SizeAwareStop | 2 | 26 |
| 16.0.6 | [~] | pattern invalidation | engine/stops.py::InvalidationStop | 2 | 19 |
| 16.0.7 | [~] | hybrid | engine/exits.py<br>engine/stops.py::HybridStop | 3 | 19 |

### Phase 17: TIMELINE DIAL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 17.0.1 | [!] | regime | engine/timeline.py::regime_from_market<br>engine/antioverfit.py::regime_labels | 8 | 12 |
| 17.0.2 | [x] | analog forecast | engine/timeline.py::forecast_from_analogs<br>engine/analog_weighting.py::blend_forecasts | 3 | 6 |
| 17.0.3 | [~] | year-to-date progress | engine/timeline.py::_reject_dates | 2 | 6 |
| 17.0.4 | [~] | weekly performance trajectory | engine/timeline.py | 2 | 6 |
| 17.0.5 | [x] | exposure | engine/timeline.py<br>engine/pit.py::restatement_exposure | 3 | 7 |
| 17.0.6 | [!] | k | engine/timeline.py::DialAdmission<br>scripts/dial_offline.py::DialSession | 2 | 6 |
| 17.0.7 | [x] | pool_q | engine/timeline.py<br>engine/analogs_stock.py::PooledStockAnalogs | 3 | 6 |
| 17.0.8 | [x] | brake | engine/timeline.py | 2 | 6 |
| 17.0.9 | [!] | improvement | engine/timeline.py<br>engine/antimemo.py::improvement_by_era | 6 | 9 |
| 17.0.10 | [ ] | no unacceptable risk deterioration | - | 0 | 0 |
| 17.0.11 | [~] | no overfitting | engine/timeline.py | 2 | 6 |
| 17.0.12 | [!] | stability across eras | engine/timeline.py<br>engine/heavy_tests.py::ablation_across_eras | 4 | 6 |

### Phase 18: WEEKLY ADAPTER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 18.0.1 | [x] | w_model | engine/adaptive.py<br>engine/missed_winners.py | 11 | 30 |
| 18.0.2 | [x] | liq_q | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 19 |
| 18.0.3 | [x] | k | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 19 |
| 18.0.4 | [x] | pool_q | engine/analogs_stock.py::PooledStockAnalogs | 1 | 0 |
| 18.0.5 | [x] | w_move | engine/adaptive.py<br>engine/memory_diagnostics.py | 11 | 4 |
| 18.0.6 | [x] | w_mom | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 19 |
| 18.0.7 | [x] | base | engine/missed_winners.py::_base_score<br>scripts/run_b15_memory_adapter.py | 2 | 4 |
| 18.0.8 | [ ] | neighbor | - | 0 | 0 |
| 18.0.9 | [?] | evidence | engine/adaptive.py<br>engine/memory.py | 10 | 19 |
| 18.0.10 | [?] | effective sample | engine/adaptive.py<br>engine/memory.py | 11 | 22 |
| 18.0.11 | [x] | confidence | engine/analog_weighting.py::confidence_calibration<br>engine/heavy_tests.py::era_confidence | 3 | 0 |
| 18.0.12 | [!] | improvement | engine/adaptive.py<br>engine/antimemo.py::improvement_by_era | 11 | 5 |
| 18.0.13 | [x] | standard error | engine/adaptive.py::GuardRailError<br>engine/memory.py::classify_error | 12 | 23 |
| 18.0.14 | [x] | cooldown | engine/adaptive.py | 8 | 2 |
| 18.0.15 | [x] | last switch | engine/adaptive.py<br>scripts/run_b15_memory_adapter.py::weekly_last_sessions | 8 | 2 |
| 18.0.16 | [x] | revert state | engine/adaptive.py::KnobState<br>engine/analog_weighting.py::price_state | 9 | 2 |
| 18.0.17 | [?] | minimum evidence | engine/adaptive.py<br>scripts/run_b15_memory_adapter.py | 9 | 3 |
| 18.0.18 | [x] | cooldown | engine/adaptive.py | 8 | 2 |
| 18.0.19 | [ ] | statistically meaningful improvement | - | 0 | 0 |
| 18.0.20 | [x] | one-step movement | engine/adaptive.py::one_step<br>engine/memory_diagnostics.py | 11 | 5 |
| 18.0.21 | [ ] | rapid revert when deterioration occurs | - | 0 | 0 |

### Phase 19: TRAIN THE TRAINING BASIS

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 19.0.1 | [x] | 24 random starting configurations | engine/basis_search.py<br>scripts/livesim_loop2.py | 4 | 3 |
| 19.0.2 | [?] | 10 random archived screening windows | engine/basis_search.py<br>scripts/livesim_loop2.py::as_search_window | 4 | 3 |
| 19.0.3 | [x] | top 3 candidates | engine/basis_search.py::Candidate<br>engine/objective.py | 13 | 7 |
| 19.0.4 | [~] | confirmation across all archived windows | engine/antimemo.py::disguise_window<br>scripts/livesim_loop2.py::as_search_window | 3 | 3 |
| 19.0.5 | [x] | tiered objective | engine/basis_search.py<br>engine/objective.py::TierScore | 6 | 6 |
| 19.0.6 | [x] | next-basis adoption | engine/basis_search.py::BasisHistory | 2 | 3 |
| 19.0.7 | [x] | half-life | engine/memory.py::half_life | 4 | 18 |
| 19.0.8 | [x] | prior weeks | engine/exits.py::WeekTable<br>engine/missed_winners.py::prior_correct | 14 | 30 |
| 19.0.9 | [ ] | switch threshold | - | 0 | 0 |
| 19.0.10 | [~] | minimum weeks | engine/blind_gates.py::is_week_end<br>engine/improve.py::weekly | 7 | 4 |
| 19.0.11 | [ ] | cooldown | - | 0 | 0 |
| 19.0.12 | [ ] | revert drop | - | 0 | 0 |
| 19.0.13 | [ ] | IC beta | - | 0 | 0 |
| 19.0.14 | [!] | detector max | engine/adaptive.py::detector_scores | 8 | 2 |
| 19.0.15 | [x] | detector minimum weeks | engine/adaptive.py::detector_scores<br>scripts/run_b15_memory_adapter.py::build_weeks | 8 | 3 |
| 19.0.16 | [x] | memory half-life | engine/memory.py::half_life<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 19.0.17 | [ ] | bandwidth | - | 0 | 0 |
| 19.0.18 | [x] | prior scale | engine/memory.py::_fit_scale<br>engine/missed_winners.py::prior_correct | 5 | 19 |
| 19.0.19 | [ ] | shrinkage | - | 0 | 0 |
| 19.0.20 | [~] | shock parameters | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::shock_report | 4 | 18 |

### Phase 20: TIERED OBJECTIVE FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 20.TIER-1.1 | [x] | most weeks approximately 5–10% | engine/exits.py::WeekTable<br>engine/objective.py::TierScore | 6 | 22 |
| 20.TIER-1.2 | [x] | yearly average move near 7% | engine/objective.py::TierScore<br>engine/timeline.py::expected_abs_move | 4 | 3 |
| 20.TIER-2.1 | [x] | worst 5% week | engine/objective.py::TierScore<br>engine/stops.py | 7 | 22 |
| 20.TIER-2.2 | [~] | max drawdown | engine/objective.py::TierScore<br>engine/run_report.py::max_drawdown | 6 | 3 |
| 20.TIER-2.3 | [!] | weeks outside band | scripts/timeline_basis_report.py::load_weekly<br>engine/exits.py::WeekTable | 7 | 25 |
| 20.TIER-2.4 | [x] | catastrophic losses | engine/objective.py::TierScore<br>engine/stops.py::loss_risk | 6 | 22 |
| 20.TIER-3.1 | [!] | positive in-band weeks | engine/exits.py::WeekTable<br>engine/objective.py::TierScore | 7 | 25 |
| 20.TIER-3.2 | [~] | successful +10% outcomes | engine/objective.py::TierScore | 4 | 3 |
| 20.TIER-3.3 | [~] | directional precision | engine/objective.py::TierScore | 4 | 3 |

### Phase 21: BLIND SIMULATOR HARDENING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 21.SEALED-WINDOW.1 | [?] | Sealed window | engine/blind_gates.py::seal_window<br>engine/livesim.py::SealedYear | 4 | 4 |
| 21.DISGUISE.1 | [x] | Disguise | engine/blind_gates.py::check_disguise_signature<br>engine/livesim.py::_disguise | 6 | 7 |
| 21.WARM-UP.1 | [~] | Warm-up | engine/blind_gates.py<br>engine/livesim.py | 3 | 4 |
| 21.REVEAL.1 | [x] | adjustments locked | engine/blind_gates.py::lock_adjustments<br>engine/livesim.py::reveal | 3 | 4 |
| 21.REVEAL.2 | [~] | all predictions recorded | engine/blind_gates.py::RevealGate | 3 | 4 |
| 21.REVEAL.3 | [~] | all trades completed | engine/blind_gates.py::RevealGate<br>engine/livesim.py::BlindTrader | 3 | 4 |
| 21.REVEAL.4 | [~] | all learning decisions finalized | engine/blind_gates.py::RevealGate<br>engine/livesim.py::reveal | 3 | 4 |

### Phase 23: RE-TESTER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 23.0.1 | [!] | holdings | engine/blind_gates.py<br>engine/retester.py::cmp_holdings | 12 | 4 |
| 23.0.2 | [!] | trades | engine/blind_gates.py<br>engine/retester.py::cmp_trades | 12 | 4 |
| 23.0.3 | [x] | scores | engine/retester.py::cmp_scores<br>engine/ablation.py::score | 10 | 7 |
| 23.0.4 | [x] | weekly returns | engine/blind_gates.py::is_week_end<br>engine/retester.py::cmp_weekly_returns | 12 | 4 |
| 23.0.5 | [~] | adaptation events | engine/retester.py::cmp_events | 1 | 1 |
| 23.0.6 | [~] | pattern activation | engine/retester.py::cmp_pattern_activation | 1 | 1 |
| 23.0.7 | [x] | memory state | engine/blind_gates.py::check_memory_bank_causality<br>engine/retester.py::cmp_memory | 14 | 21 |

### Phase 24: WORKER HEALTH

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 24.0.1 | [x] | start | engine/blind_gates.py<br>engine/health.py | 5 | 4 |
| 24.0.2 | [x] | configuration | engine/adaptive.py::audit_matches_cfg<br>engine/timeline.py::cfg_overrides | 13 | 6 |
| 24.0.3 | [?] | window | engine/blind_gates.py::seal_window<br>engine/health.py::window_breakdown | 5 | 4 |
| 24.0.4 | [x] | seed | engine/blind_gates.py<br>engine/health.py | 5 | 4 |
| 24.0.5 | [x] | memory | engine/blind_gates.py::check_memory_bank_causality<br>engine/health.py::memory_growth_mb_per_min | 7 | 4 |
| 24.0.6 | [x] | completion | engine/blind_gates.py<br>engine/health.py | 5 | 4 |
| 24.0.7 | [x] | crash | engine/health.py<br>engine/retester.py | 3 | 1 |
| 24.0.8 | [x] | timeout | engine/health.py<br>engine/pattern_bank.py::BankLockTimeout | 4 | 15 |
| 24.0.9 | [x] | OOM | engine/health.py::WorkerLog<br>scripts/blind_gates_real.py::section_health | 5 | 2 |
| 24.0.10 | [x] | invalid result | engine/health.py::result_fingerprint<br>engine/basis_search.py::SearchResult | 5 | 23 |

### Phase 25: PLANTED-PATTERN CALIBRATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 25.0.1 | [x] | detection rate | engine/health.py::failure_rates<br>engine/pattern_lifecycle.py::detect_failure | 4 | 2 |
| 25.0.2 | [ ] | false discovery rate | - | 0 | 0 |
| 25.0.3 | [!] | P(real) calibration | engine/planted.py::calibration_table<br>engine/missed_winners.py::calibration | 7 | 13 |
| 25.0.4 | [x] | rejection rate | engine/pattern_lifecycle.py::scope_null_rate<br>engine/pattern_stats.py::bh_reject | 5 | 9 |
| 25.0.5 | [ ] | recovery rate | - | 0 | 0 |
| 25.0.6 | [ ] | rescoping behavior | - | 0 | 0 |

### Phase 26: ANTI-OVERFITTING BATTERY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 26.A.1 | [~] | A. Future scramble: Future data must not affect earlier decisions | engine/ablation.py | 1 | 3 |
| 26.B.1 | [!] | B. Label permutation: Random labels should destroy apparent predictive performance | engine/antioverfit.py::label_permutation | 2 | 3 |
| 26.C.1 | [!] | C. Ticker permutation: Ticker identities should not create fake signal | engine/antioverfit.py::ticker_permutation | 2 | 3 |
| 26.D.1 | [x] | D. Date disguise: Changing absolute dates while preserving structure should preserve behavior | engine/antioverfit.py::date_disguise | 2 | 3 |
| 26.E.1 | [~] | E. Randomized outcomes: Pattern discovery should collapse toward chance | engine/antioverfit.py::randomized_outcomes | 2 | 3 |
| 26.F.1 | [!] | F. Feature shuffle: Important features should lose signal when shuffled | engine/antioverfit.py::feature_shuffle | 2 | 3 |
| 26.G.1 | [x] | G. Dead-feature injection: Adding random features must not create stable predictive power | engine/antioverfit.py::dead_feature_injection | 2 | 3 |
| 26.H.1 | [?] | H. Duplicate-feature injection: Duplicates must not double-count evidence | engine/antioverfit.py::duplicate_feature_injection | 2 | 3 |
| 26.I.1 | [?] | I. Regime split: Signal must be evaluated independently across regimes | engine/missed_winners.py::evaluate<br>engine/pattern_stats.py::evaluate_masks | 3 | 7 |
| 26.J.1 | [?] | J. Walk-forward split: No future observations in training | engine/antioverfit.py::walk_forward_splits<br>engine/missed_winners.py::walk_forward | 6 | 11 |

### Phase 27: DATA EXPANSION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 27.0.1 | [?] | schema validation | engine/data_sources.py::ValidationReport | 1 | 1 |
| 27.0.2 | [x] | date validation | engine/data_sources.py::ValidationReport<br>engine/antioverfit.py::date_disguise | 9 | 22 |
| 27.0.3 | [x] | source provenance | engine/data_sources.py::ProvenanceLedger<br>scripts/data_live_audit.py | 4 | 19 |
| 27.0.4 | [~] | missing-data handling | engine/data_sources.py | 1 | 1 |
| 27.0.5 | [x] | duplicate detection | engine/data_sources.py::detect_splits<br>engine/pattern_lifecycle.py::detect_failure | 3 | 2 |
| 27.0.6 | [?] | point-in-time validation | engine/data_sources.py::point_in_time_filter<br>engine/pattern_stats.py::_check_point_in_time | 3 | 5 |
| 27.0.7 | [~] | parity tests where applicable | engine/data_sources.py::parity | 1 | 1 |

### Phase 28: LIVE/RESEARCH SEPARATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 28.0.1 | [x] | can simulate | engine/portfolio.py::simulate<br>engine/replay.py::simulate | 2 | 6 |
| 28.0.2 | [x] | cannot submit broker orders | engine/champion.py::order<br>engine/isolation.py::submit_order | 4 | 5 |
| 28.0.3 | [?] | uses validated configuration | engine/timeline.py::cfg_overrides<br>engine/data_sources.py::ValidationReport | 2 | 6 |
| 28.0.4 | [~] | requires explicit activation | engine/isolation.py::verify_activation | 2 | 1 |
| 28.0.5 | [x] | uses current data only | engine/data_sources.py<br>engine/isolation.py::data_is_current | 3 | 1 |
| 28.0.6 | [~] | respects trading hours | engine/isolation.py::audit_hours_firewall | 2 | 1 |
| 28.0.7 | [ ] | respects broker constraints | - | 0 | 0 |

### Phase 29: PUBLIC EXPLANATION SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 29.DASHBOARD.1 | [ ] | Dashboard | - | 0 | 0 |
| 29.PATTERN-EXPLORER.1 | [~] | expression | scripts/site_pages.py::_pattern_row | 0 | 0 |
| 29.PATTERN-EXPLORER.2 | [~] | target | scripts/site_pages.py::_pattern_row | 0 | 0 |
| 29.PATTERN-EXPLORER.3 | [x] | effect | scripts/site_sources.py::_pattern_from_row<br>scripts/site_pages.py::_pattern_row | 1 | 0 |
| 29.PATTERN-EXPLORER.4 | [~] | confidence | scripts/site_pages.py::_pattern_row | 0 | 0 |
| 29.PATTERN-EXPLORER.5 | [~] | sample size | scripts/site_pages.py::_pattern_row<br>engine/pattern_movers.py::PatternMoverModel | 0 | 0 |
| 29.PATTERN-EXPLORER.6 | [x] | P(real) | scripts/site_sources.py::_pattern_from_row<br>scripts/site_pages.py::_pattern_row | 1 | 0 |
| 29.PATTERN-EXPLORER.7 | [~] | recent performance | scripts/site_pages.py::_pattern_row | 0 | 0 |
| 29.PATTERN-EXPLORER.8 | [~] | historical performance | scripts/site_pages.py::_pattern_row | 0 | 0 |
| 29.PATTERN-EXPLORER.9 | [x] | context | scripts/site_sources.py::_pattern_from_row<br>scripts/site_pages.py::_pattern_row | 1 | 0 |
| 29.PATTERN-EXPLORER.10 | [x] | lifecycle | scripts/site_sources.py::_pattern_from_row<br>scripts/site_pages.py::_pattern_row | 1 | 0 |
| 29.PATTERN-EXPLORER.11 | [?] | validation evidence | scripts/site_pages.py::_evidence_html<br>scripts/site_sources.py::_pattern_from_row | 5 | 16 |
| 29.SENSITIVITY-PAGE.1 | [x] | parameter | scripts/site_audit.py::audit_page<br>scripts/site_pages.py | 1 | 1 |
| 29.SENSITIVITY-PAGE.2 | [x] | tested values | scripts/site_sources.py::build_sensitivity<br>scripts/site_render.py::page | 1 | 0 |
| 29.SENSITIVITY-PAGE.3 | [x] | effect | scripts/site_sources.py::build_sensitivity<br>scripts/site_audit.py::audit_page | 1 | 0 |
| 29.SENSITIVITY-PAGE.4 | [~] | uncertainty | scripts/site_pages.py<br>scripts/site_audit.py::audit_page | 0 | 0 |
| 29.SENSITIVITY-PAGE.5 | [~] | stability | engine/exits.py::cost_sensitivity<br>scripts/site_audit.py::audit_page | 2 | 19 |
| 29.SENSITIVITY-PAGE.6 | [x] | selected value | scripts/site_charts.py<br>scripts/site_render.py::page | 3 | 19 |
| 29.SENSITIVITY-PAGE.7 | [~] | reason selected | engine/exits.py::Selection<br>scripts/site_render.py::page | 2 | 19 |

### Phase 30: EXPERIMENT MEMORY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 30.WHAT-CHANGED.1 | [~] | What changed? | engine/checkpoint.py<br>engine/registry.py | 7 | 5 |
| 30.WHY-DID.1 | [~] | Why did it change? | engine/checkpoint.py<br>engine/registry.py | 7 | 5 |
| 30.WHAT-DATA.1 | [x] | What data was used? | engine/registry.py<br>engine/run_report.py | 5 | 4 |
| 30.WHAT-WAS.1 | [ ] | What was unseen? | - | 0 | 0 |
| 30.WHAT-WAS2.1 | [x] | What was the baseline? | engine/champion.py::freeze_baseline<br>engine/run_report.py::baseline_metrics | 3 | 3 |
| 30.WHAT-IMPROVED.1 | [!] | What improved? | engine/champion.py<br>engine/registry.py | 9 | 9 |
| 30.WHAT-WORSENED.1 | [ ] | What worsened? | - | 0 | 0 |
| 30.WAS-IMPROVEMENT.1 | [ ] | Was improvement statistically meaningful? | - | 0 | 0 |
| 30.DID-RISK.1 | [~] | Did risk change? | engine/champion.py::_check_risk | 1 | 2 |
| 30.DID-THE.1 | [ ] | Did the improvement survive another window? | - | 0 | 0 |
| 30.WAS-IT.1 | [x] | Was it adopted? | engine/experiment_memory.py<br>engine/run_report.py | 5 | 3 |
| 30.IF-REJECTED.1 | [~] | If rejected, why?: Never let an experiment become an orphan | engine/registry.py::ExperimentMemory<br>engine/experiment_memory.py | 5 | 2 |

### Phase 31: CODE QUALITY FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 31.0.1 | [ ] | type-safe interfaces where appropriate | - | 0 | 0 |
| 31.0.2 | [ ] | explicit inputs | - | 0 | 0 |
| 31.0.3 | [~] | explicit outputs | scripts/quality_gate.py<br>scripts/bible_trace.py::write_outputs | 2 | 0 |
| 31.0.4 | [ ] | deterministic behavior | - | 0 | 0 |
| 31.0.5 | [ ] | logging | - | 0 | 0 |
| 31.0.6 | [~] | error handling | engine/livesim.py::BlindGateError<br>scripts/site_audit.py::handle_data | 3 | 3 |
| 31.0.7 | [~] | unit tests | scripts/quality_gate.py<br>scripts/test_inventory.py | 1 | 0 |
| 31.0.8 | [~] | integration tests | scripts/test_inventory.py<br>scripts/bible_trace.py::integration_pending | 2 | 0 |
| 31.0.9 | [ ] | documentation | - | 0 | 0 |
| 31.0.10 | [~] | no hidden global state | scripts/quality_gate.py::check_hidden_global_state | 1 | 0 |
| 31.0.11 | [~] | no accidental randomness | scripts/quality_gate.py::check_randomness | 1 | 0 |

### Phase 32: TEST PYRAMID

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 32.L1.1 | [~] | Level 1 — Unit: Test mathematical correctness | scripts/test_inventory.py::_level_words | 1 | 0 |
| 32.L2.1 | [~] | Level 2 — Integration: Test interaction with neighboring systems | scripts/test_inventory.py::_level_words | 1 | 0 |
| 32.L3.1 | [~] | Level 3 — Historical: Test on known historical data | scripts/quality_gate.py::check_tests_touch_data<br>engine/analog_weighting.py::level_agreement | 6 | 5 |
| 32.L4.1 | [?] | Level 4 — Walk-forward: Test chronologically | scripts/test_inventory.py::_level_words<br>engine/analog_weighting.py::learn_weights_walk_forward | 6 | 22 |
| 32.L5.1 | [~] | Level 5 — Blind: Test disguised hidden windows | engine/blind_gates.py::BlindClock<br>engine/livesim.py::BlindGateError | 3 | 4 |
| 32.L6.1 | [~] | Level 6 — Adversarial: Try to make it leak | engine/blind_gates.py::make_ticker_map<br>scripts/test_inventory.py::_level_words | 3 | 4 |
| 32.L7.1 | [~] | Level 7 — Reproducibility: Run twice and compare | engine/repro.py::run_twice<br>engine/direction_calib.py::compare_gates | 6 | 4 |

### Phase 33: REQUIRED REPRODUCIBILITY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 33.0.1 | [x] | same configuration hash | engine/repro.py::artifact_hash<br>engine/adaptive.py::audit_matches_cfg | 12 | 4 |
| 33.0.2 | [x] | same seed | engine/ablation.py<br>engine/antioverfit.py | 3 | 5 |
| 33.0.3 | [x] | same candidate ordering | engine/repro.py<br>engine/candidates.py::Candidate | 5 | 7 |
| 33.0.4 | [~] | same model configuration | engine/repro.py<br>engine/adaptive.py::audit_matches_cfg | 12 | 3 |
| 33.0.5 | [x] | same predictions | engine/antioverfit.py<br>engine/repro.py | 7 | 37 |
| 33.0.6 | [x] | same trades | engine/repro.py<br>engine/adaptive.py::_trade | 11 | 23 |
| 33.0.7 | [x] | same metrics | engine/ablation.py<br>engine/antioverfit.py | 5 | 5 |

### Phase 34: REQUIRED ABLATION FRAMEWORK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 34.0.1 | [x] | baseline | engine/ablation.py<br>engine/analog_weighting.py::baseline | 4 | 4 |
| 34.0.2 | [x] | baseline + feature | engine/ablation.py::feature_ablation<br>engine/analog_weighting.py::baseline | 4 | 3 |
| 34.0.3 | [~] | feature alone | engine/ablation.py::feature_ablation<br>engine/pattern_movers.py::attribute_by_feature | 6 | 9 |
| 34.0.4 | [~] | randomized feature | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::dead_feature_injection | 2 | 5 |
| 34.0.5 | [!] | shuffled feature | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::feature_shuffle | 5 | 6 |
| 34.0.6 | [x] | future-scrambled feature where applicable | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::future_scramble | 1 | 3 |

### Phase 35: CHAMPION / CHALLENGER SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 35.0.1 | [x] | Champion | engine/champion.py::ChampionError<br>engine/timeline.py::champion | 4 | 8 |
| 35.0.2 | [x] | Challenger | engine/champion.py::ChallengerQueue<br>engine/improve.py::spawn_challengers | 3 | 4 |
| 35.0.3 | [x] | Candidate | engine/champion.py::add_candidate<br>engine/adaptive.py::_evaluate_candidates | 12 | 9 |
| 35.0.4 | [x] | Rejected | engine/champion.py::reject<br>engine/run_report.py | 5 | 11 |
| 35.0.5 | [~] | required gates pass | engine/champion.py::_check_gates<br>engine/livesim.py::BlindGateError | 6 | 7 |
| 35.0.6 | [~] | enough independent evidence exists | scripts/bible_trace.py::EvidenceFile<br>engine/champion.py::_check_evidence | 1 | 0 |
| 35.0.7 | [ ] | higher-priority objective is not harmed | - | 0 | 0 |
| 35.0.8 | [ ] | risk does not violate constraints | - | 0 | 0 |
| 35.0.9 | [~] | reproducibility passes | scripts/final_report.py::s_reproducibility<br>engine/registry.py::reproducibility_conflicts | 1 | 0 |
| 35.0.10 | [~] | blind validation passes | engine/champion.py::_check_blind | 1 | 2 |

### Phase 36: REQUIRED REPORT AFTER EACH MAJOR RUN

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 36.TIER-1.1 | [x] | TIER 1: weekly average move | engine/objective.py::TierScore<br>engine/heavy_tests.py | 4 | 3 |
| 36.TIER-1.2 | [x] | TIER 1: weekly median move | engine/heavy_tests.py::vs_past_median<br>engine/gaprisk.py::_week_regime | 8 | 14 |
| 36.TIER-1.3 | [x] | TIER 1: share 5–10% | engine/run_report.py<br>engine/objective.py::TierScore | 5 | 3 |
| 36.TIER-1.4 | [x] | TIER 1: share >10% | engine/run_report.py<br>engine/objective.py::TierScore | 5 | 3 |
| 36.TIER-1.5 | [x] | TIER 1: share <5% | engine/run_report.py<br>engine/objective.py::TierScore | 5 | 3 |
| 36.TIER-2.1 | [~] | TIER 2: max drawdown | engine/run_report.py::max_drawdown<br>engine/objective.py::TierScore | 6 | 3 |
| 36.TIER-2.2 | [x] | TIER 2: worst week | engine/objective.py::TierScore<br>scripts/livesim_loop2.py::tiered | 5 | 3 |
| 36.TIER-2.3 | [~] | TIER 2: 5th percentile week | engine/stops.py::VolPercentileStop | 2 | 19 |
| 36.TIER-2.4 | [x] | TIER 2: catastrophic losses | engine/objective.py::TierScore<br>engine/stops.py::loss_risk | 6 | 22 |
| 36.TIER-2.5 | [~] | TIER 2: turnover | engine/run_report.py | 1 | 0 |
| 36.TIER-2.6 | [x] | TIER 2: costs | engine/exits.py::CostModel<br>engine/heavy_tests.py::cost_sensitivity | 2 | 19 |
| 36.TIER-3.1 | [x] | TIER 3: positive in-band percentage | engine/run_report.py<br>engine/exits.py::band_metrics | 7 | 22 |
| 36.TIER-3.2 | [ ] | TIER 3: direction accuracy | - | 0 | 0 |
| 36.TIER-3.3 | [?] | TIER 3: calibration | engine/run_report.py::calibration<br>engine/stops.py::gap_calibration | 3 | 19 |
| 36.TIER-3.4 | [~] | TIER 3: 8/10 rate | engine/run_report.py | 1 | 0 |
| 36.ALGORITHM.1 | [~] | ALGORITHM: pattern count | engine/heavy_tests.py::_oos_pattern_check<br>scripts/site_sources.py::_pattern_from_row | 3 | 0 |
| 36.ALGORITHM.2 | [x] | ALGORITHM: active pattern count | engine/pattern_bank.py::PatternBank<br>engine/pattern_lifecycle.py::counts | 5 | 15 |
| 36.ALGORITHM.3 | [~] | ALGORITHM: failed patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 3 | 0 |
| 36.ALGORITHM.4 | [~] | ALGORITHM: rescoped patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 3 | 0 |
| 36.ALGORITHM.5 | [~] | ALGORITHM: discarded patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 3 | 0 |
| 36.ALGORITHM.6 | [~] | ALGORITHM: P(real) distribution | engine/heavy_tests.py | 2 | 0 |
| 36.ALGORITHM.7 | [~] | ALGORITHM: false discovery diagnostics | engine/heavy_tests.py::false_discovery_summary<br>engine/run_report.py::algorithm_section | 3 | 0 |
| 36.ANALOGS.1 | [x] | ANALOGS: analog count | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 3 | 6 |
| 36.ANALOGS.2 | [ ] | ANALOGS: distance distribution | - | 0 | 0 |
| 36.ANALOGS.3 | [~] | ANALOGS: forecast quality | engine/run_report.py::analogs_section | 1 | 0 |
| 36.MEMORY.1 | [x] | MEMORY: lesson count | engine/registry.py::ExperimentMemory<br>engine/lessons.py::Lesson | 11 | 31 |
| 36.MEMORY.2 | [~] | MEMORY: accepted lessons | engine/blind_gates.py::check_memory_bank_causality<br>engine/registry.py::ExperimentMemory | 3 | 4 |
| 36.MEMORY.3 | [~] | MEMORY: rejected lessons | engine/champion.py::reject<br>engine/registry.py::ExperimentMemory | 5 | 4 |
| 36.MEMORY.4 | [ ] | MEMORY: shock events | - | 0 | 0 |
| 36.MEMORY.5 | [~] | MEMORY: memory switches | engine/memory_diagnostics.py::memory_health<br>engine/registry.py::ExperimentMemory | 1 | 1 |
| 36.ADAPTATION.1 | [~] | ADAPTATION: parameter changes | engine/adaptive.py::adaptation_report<br>engine/basis_search.py::changed_keys | 9 | 3 |
| 36.ADAPTATION.2 | [x] | ADAPTATION: reverts | engine/adaptive.py::adaptation_report | 8 | 2 |
| 36.ADAPTATION.3 | [?] | ADAPTATION: evidence | engine/run_report.py<br>engine/adaptive.py::adaptation_report | 9 | 2 |
| 36.ADAPTATION.4 | [~] | ADAPTATION: confidence | engine/heavy_tests.py::era_confidence | 2 | 0 |
| 36.GATES.1 | [?] | GATES: parity | engine/data_sources.py::parity<br>engine/livesim.py::BlindGateError | 5 | 5 |
| 36.GATES.2 | [!] | GATES: retester | scripts/blind_gates_real.py::section_retest<br>scripts/check_retester.py | 0 | 1 |
| 36.GATES.3 | [x] | GATES: future scramble | engine/antioverfit.py::future_scramble<br>engine/blind_gates.py::RevealGate | 6 | 8 |
| 36.GATES.4 | [x] | GATES: time fence | engine/adaptive.py::TimeFence<br>engine/pit.py::fence | 9 | 3 |
| 36.GATES.5 | [~] | GATES: worker health | scripts/blind_gates_real.py::section_health<br>scripts/livesim_loop2.py::reset_worker_files | 3 | 1 |
| 36.GATES.6 | [~] | GATES: reproducibility | scripts/final_report.py::s_reproducibility<br>engine/registry.py::reproducibility_conflicts | 1 | 0 |

### Phase 38: MASTER CHECKLIST

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 38.ALGORITHM.1 | [?] | ALGORITHM: Candles and micro-signals | engine/candidates.py<br>engine/planted.py | 2 | 2 |
| 38.ALGORITHM.2 | [~] | ALGORITHM: Pattern miner | engine/pattern_stats.py<br>engine/candidates.py | 4 | 4 |
| 38.ALGORITHM.3 | [~] | ALGORITHM: Week-clustered statistics | engine/pattern_stats.py<br>engine/pattern_identity.py | 2 | 2 |
| 38.ALGORITHM.4 | [~] | ALGORITHM: Heavy tests across eras | engine/heavy_tests.py | 5 | 5 |
| 38.ALGORITHM.5 | [!] | ALGORITHM: Long-term pattern bank | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 5 | 5 |
| 38.ALGORITHM.6 | [?] | ALGORITHM: Pattern scores → Find Volatility | engine/heavy_tests.py<br>engine/pattern_movers.py | 1 | 4 |
| 38.ALGORITHM.7 | [?] | ALGORITHM: Minute collector | engine/data_sources.py | 2 | 3 |
| 38.ALGORITHM.8 | [~] | ALGORITHM: Self-tuning Algorithm | engine/basis_search.py<br>engine/memory.py | 5 | 5 |
| 38.ALGORITHM.9 | [~] | ALGORITHM: Market analog engine | engine/analog_weighting.py<br>engine/analogs_sector.py | 5 | 4 |
| 38.ALGORITHM.10 | [x] | ALGORITHM: Sector analog engine | engine/analog_weighting.py | 1 | 1 |
| 38.ALGORITHM.11 | [x] | ALGORITHM: Stock analog engine | engine/analogs_stock.py | 1 | 0 |
| 38.ALGORITHM.12 | [!] | ALGORITHM: Timeline dial | engine/timeline.py | 2 | 4 |
| 38.ALGORITHM.13 | [~] | ALGORITHM: Lesson memory | engine/memory.py<br>engine/lessons.py | 5 | 5 |
| 38.ALGORITHM.14 | [!] | ALGORITHM: Rerun anti-memorization test | engine/antimemo.py<br>engine/lessons.py | 2 | 4 |
| 38.ALGORITHM.15 | [?] | ALGORITHM: Improve-or-discard lifecycle | engine/pattern_lifecycle.py<br>engine/pattern_bank.py | 4 | 5 |
| 38.ALGORITHM.16 | [?] | ALGORITHM: Additional data sources | engine/data_sources.py | 2 | 3 |
| 38.ALGORITHM.17 | [!] | ALGORITHM: Planted-pattern calibration | engine/health.py<br>engine/pattern_lifecycle.py | 2 | 3 |
| 38.FIND-VOLATILITY.1 | [?] | FIND VOLATILITY: 95% mover target where sufficient candidates exist | engine/direction.py<br>engine/heavy_tests.py | 5 | 5 |
| 38.FIND-VOLATILITY.2 | [?] | FIND VOLATILITY: Direction ≥80% calibrated confidence | engine/direction.py<br>engine/direction_ablate.py | 4 | 5 |
| 38.FIND-VOLATILITY.3 | [~] | FIND VOLATILITY: Per-stock-type trust tables | engine/trust.py<br>engine/direction.py | 5 | 4 |
| 38.FIND-VOLATILITY.4 | [?] | FIND VOLATILITY: Exit learner | engine/exits.py<br>engine/retester.py | 4 | 2 |
| 38.FIND-VOLATILITY.5 | [~] | FIND VOLATILITY: Stop/risk learner | engine/stops.py<br>engine/gaprisk.py | 2 | 1 |
| 38.FIND-VOLATILITY.6 | [~] | FIND VOLATILITY: 8-of-10 +10% objective across blind eras | engine/heavy_tests.py<br>engine/objective.py | 5 | 5 |
| 38.TEST.1 | [~] | TEST: Archive | engine/blind_gates.py | 2 | 1 |
| 38.TEST.2 | [~] | TEST: Blind loop | engine/blind_gates.py | 1 | 1 |
| 38.TEST.3 | [!] | TEST: Tiered 7% objective | engine/objective.py<br>engine/exits.py | 3 | 2 |
| 38.TEST.4 | [?] | TEST: Weekly self-adjustment | engine/adaptive.py<br>engine/analog_weighting.py | 3 | 5 |
| 38.TEST.5 | [?] | TEST: Insider leak protection | engine/pit.py<br>engine/blind_gates.py | 5 | 4 |
| 38.TEST.6 | [?] | TEST: 13D correctness | engine/data_sources.py | 2 | 3 |
| 38.TEST.7 | [~] | TEST: Explorer | scripts/site_pages.py<br>scripts/site_sources.py | 2 | 1 |
| 38.TEST.8 | [~] | TEST: Sensitivity | engine/exits.py<br>scripts/site_sources.py | 2 | 2 |
| 38.TEST.9 | [!] | TEST: Re-tester parity | engine/blind_gates.py<br>engine/retester.py | 2 | 3 |
| 38.TEST.10 | [x] | TEST: Future scramble | engine/pit.py<br>engine/ablation.py | 4 | 4 |
| 38.TEST.11 | [~] | TEST: Time fence | engine/pit.py | 1 | 1 |
| 38.TEST.12 | [x] | TEST: Worker health | engine/blind_gates.py<br>engine/health.py | 5 | 3 |
| 38.TEST.13 | [x] | TEST: Reproducibility | engine/repro.py<br>engine/ablation.py | 3 | 2 |
| 38.TEST.14 | [!] | TEST: Label permutation | engine/antioverfit.py | 1 | 1 |
| 38.TEST.15 | [!] | TEST: Feature shuffle | engine/antioverfit.py | 1 | 1 |
| 38.TEST.16 | [!] | TEST: Ticker permutation | engine/antioverfit.py | 1 | 1 |
| 38.TEST.17 | [!] | TEST: Planted-pattern calibration | engine/health.py<br>engine/pattern_lifecycle.py | 2 | 3 |
| 38.LIVE.1 | [~] | LIVE: Only consider upgrade after validated research edge | engine/isolation.py<br>engine/champion.py | 4 | 2 |
| 38.LIVE.2 | [~] | LIVE: Verify paper-only safeguards | engine/isolation.py<br>engine/champion.py | 4 | 2 |
| 38.LIVE.3 | [~] | LIVE: Verify trading-hours firewall | engine/isolation.py<br>engine/champion.py | 4 | 2 |
| 38.LIVE.4 | [~] | LIVE: Verify broker safety | engine/isolation.py<br>engine/champion.py | 4 | 2 |
| 38.LIVE.5 | [~] | LIVE: Verify research/live isolation | engine/isolation.py<br>engine/champion.py | 4 | 2 |

### Phase 39: DEFINITION OF DONE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 39.0.1 | [~] | every required module exists | engine/heavy_tests.py<br>engine/memory.py | 5 | 5 |
| 39.0.2 | [~] | every required module is tested | engine/heavy_tests.py<br>engine/memory.py | 5 | 5 |
| 39.0.3 | [?] | every data source is validated | engine/data_sources.py | 2 | 3 |
| 39.0.4 | [?] | every point-in-time rule is enforced | engine/pit.py<br>engine/livesim.py | 5 | 5 |
| 39.0.5 | [~] | blind simulation works | engine/blind_gates.py | 2 | 1 |
| 39.0.6 | [!] | re-tester works | engine/blind_gates.py<br>engine/retester.py | 2 | 3 |
| 39.0.7 | [~] | parity works | engine/parity.py<br>engine/adaptive.py | 4 | 4 |
| 39.0.8 | [x] | future scramble works | engine/pit.py<br>engine/ablation.py | 4 | 4 |
| 39.0.9 | [~] | time fence works | engine/pit.py | 1 | 1 |
| 39.0.10 | [x] | worker health works | engine/blind_gates.py<br>engine/health.py | 5 | 3 |
| 39.0.11 | [x] | deterministic replay works | engine/repro.py<br>engine/ablation.py | 3 | 2 |
| 39.0.12 | [~] | pattern discovery works | engine/pattern_stats.py<br>engine/candidates.py | 4 | 4 |
| 39.0.13 | [~] | pattern validation works | engine/pattern_stats.py<br>engine/pattern_identity.py | 2 | 2 |
| 39.0.14 | [?] | pattern lifecycle works | engine/pattern_lifecycle.py<br>engine/pattern_bank.py | 4 | 5 |
| 39.0.15 | [~] | analog engine works | engine/analog_weighting.py<br>engine/analogs_sector.py | 5 | 4 |
| 39.0.16 | [~] | memory works | engine/memory.py<br>engine/antioverfit.py | 3 | 5 |
| 39.0.17 | [~] | lesson learning works | engine/lessons.py<br>engine/antimemo.py | 5 | 5 |
| 39.0.18 | [!] | anti-memorization test works | engine/antimemo.py<br>engine/lessons.py | 2 | 4 |
| 39.0.19 | [?] | direction engine works | engine/direction.py<br>engine/direction_ablate.py | 4 | 5 |
| 39.0.20 | [?] | exit engine works | engine/exits.py<br>engine/retester.py | 4 | 2 |
| 39.0.21 | [~] | stop/risk engine works | engine/stops.py<br>engine/gaprisk.py | 2 | 1 |
| 39.0.22 | [~] | per-type trust works | engine/trust.py<br>engine/direction.py | 5 | 4 |
| 39.0.23 | [!] | timeline dial exists and is tested | engine/timeline.py | 2 | 4 |
| 39.0.24 | [~] | outer training-basis loop works | engine/basis_search.py<br>engine/memory.py | 5 | 5 |
| 39.0.25 | [!] | planted-pattern calibration works | engine/health.py<br>engine/pattern_lifecycle.py | 2 | 3 |
| 39.0.26 | [~] | experiment registry works | engine/registry.py<br>engine/experiment_memory.py | 5 | 5 |
| 39.0.27 | [ ] | dashboard works | - | 0 | 0 |
| 39.0.28 | [~] | Pattern Explorer works | scripts/site_pages.py<br>scripts/site_sources.py | 2 | 1 |
| 39.0.29 | [~] | Sensitivity page works | engine/exits.py<br>scripts/site_sources.py | 2 | 2 |
| 39.0.30 | [ ] | documentation reflects actual state | - | 0 | 0 |
| 39.0.31 | [~] | all negative findings are preserved | engine/checkpoint.py<br>engine/champion.py | 5 | 5 |
| 39.0.32 | [~] | champion/challenger system works | engine/champion.py<br>scripts/bible_trace.py | 5 | 5 |

### Phase 42: THE STANDARD FOR EVERY CLAIM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 42.0.1 | [ ] | exact pattern definition | - | 0 | 0 |
| 42.0.2 | [~] | discovery sample | engine/heavy_tests.py::false_discovery_summary | 2 | 0 |
| 42.0.3 | [?] | confirmation sample | engine/pattern_stats.py::confirmation_factor | 1 | 3 |
| 42.0.4 | [~] | later sample | engine/basis_search.py::sample_candidate<br>scripts/antioverfit_real.py::load_sample | 2 | 6 |
| 42.0.5 | [~] | week-clustered statistics | engine/direction_ablate.py::week_bootstrap<br>engine/exits.py::WeekTable | 6 | 48 |
| 42.0.6 | [!] | permutation evidence | engine/antioverfit.py::label_permutation | 2 | 3 |
| 42.0.7 | [x] | FDR result | engine/adaptive.py::result<br>engine/analog_weighting.py::audit_result | 12 | 6 |
| 42.0.8 | [x] | P(real) | engine/antioverfit.py::_real_has_signal<br>engine/livesim.py::real_end | 5 | 8 |
| 42.0.9 | [~] | effect size | engine/pattern_stats.py::effective_n<br>engine/stops.py::SizeAwareStop | 3 | 21 |
| 42.0.10 | [~] | incremental validation gain | engine/data_sources.py::ValidationReport | 1 | 1 |
| 42.0.11 | [~] | regime distribution | engine/heavy_tests.py::by_regime | 2 | 0 |
| 42.0.12 | [ ] | cost-adjusted result | - | 0 | 0 |
| 42.0.13 | [x] | baseline | engine/analog_weighting.py::baseline<br>engine/backtest.py::baseline_random | 4 | 3 |
| 42.0.14 | [x] | candidate | engine/adaptive.py::_evaluate_candidates<br>engine/basis_search.py::Candidate | 12 | 9 |
| 42.0.15 | [?] | independent windows | engine/blind_gates.py::seal_window<br>engine/pattern_identity.py::Window | 6 | 7 |
| 42.0.16 | [x] | risk | engine/champion.py::_check_risk<br>engine/live.py::risk | 5 | 21 |
| 42.0.17 | [ ] | statistical comparison | - | 0 | 0 |
| 42.0.18 | [x] | stability | engine/exits.py::selection_stability<br>engine/heavy_tests.py::stability | 7 | 23 |
| 42.0.19 | [x] | adoption reason | engine/adaptive.py::hold_reasons | 8 | 2 |

### Phase 43: PRIORITY WHEN TIME/COMPUTE IS LIMITED

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 43.0.1 | [ ] | anti-leak correctness | - | 0 | 0 |
| 43.0.2 | [~] | Algorithm | engine/run_report.py::algorithm_section | 1 | 0 |
| 43.0.3 | [!] | Pattern validation | engine/exits.py::pattern_fail_flags<br>engine/heavy_tests.py::_oos_pattern_check | 9 | 34 |
| 43.0.4 | [x] | Pattern → mover integration | engine/pattern_movers.py::PatternMoverModel | 2 | 1 |
| 43.0.5 | [?] | Analog validation | engine/run_report.py::analogs_section<br>engine/timeline.py::forecast_from_analogs | 3 | 6 |
| 43.0.6 | [x] | Memory | engine/blind_gates.py::check_memory_bank_causality<br>engine/experiment_memory.py | 6 | 4 |
| 43.0.7 | [~] | Direction | engine/direction.py::DirectionEngine<br>engine/direction_ablate.py | 5 | 35 |
| 43.0.8 | [!] | Exit/stop | engine/exits.py::ExitResult<br>engine/stops.py::AtrStop | 3 | 19 |
| 43.0.9 | [ ] | outer-loop tuning | - | 0 | 0 |
| 43.0.10 | [ ] | dashboard | - | 0 | 0 |
| 43.0.11 | [x] | Live | engine/isolation.py::live_mode<br>engine/live.py::predict_live | 4 | 4 |

### Phase 44: RESOURCE MANAGEMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 44.0.1 | [?] | Windows PC | engine/antimemo.py::disguise_window<br>engine/blind_gates.py::seal_window | 9 | 7 |
| 44.0.2 | [~] | 8 CPU cores | engine/gaprisk.py::_fit_core | 1 | 8 |
| 44.0.3 | [ ] | 16 GB RAM | - | 0 | 0 |
| 44.0.4 | [ ] | 3–7 workers depending on memory | - | 0 | 0 |
| 44.0.5 | [ ] | memory-aware scheduling | - | 0 | 0 |
| 44.0.6 | [x] | deterministic worker seeds | engine/blind_gates.py::check_sealed_before_workers<br>engine/heavy_tests.py::_combine_seeds | 5 | 4 |
| 44.0.7 | [ ] | automatic cleanup | - | 0 | 0 |
| 44.0.8 | [ ] | stale-process detection | - | 0 | 0 |
| 44.0.9 | [~] | worker health reports | engine/health.py::WorkerLog<br>engine/memory_diagnostics.py::capacity_report | 5 | 2 |

### Phase 46: FINAL REPORT REQUIREMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 46.1.1 | [~] | 1. Implementation report: Every module changed | engine/adaptive.py::adaptation_report<br>engine/parity.py::ParityReport | 10 | 4 |
| 46.2.1 | [~] | 2. Validation report: Every gate | scripts/final_report.py::s_validation<br>engine/adaptive.py::_global_gate | 12 | 6 |
| 46.3.1 | [~] | 3. Research report: What genuinely improved | scripts/final_report.py::s_research_and_failures<br>engine/improve.py::report | 2 | 2 |
| 46.4.1 | [x] | 4. Failure report: What did not work | engine/blind_gates.py::check_sealed_before_workers<br>engine/exits.py::format_report | 7 | 23 |
| 46.5.1 | [~] | 5. Remaining uncertainty report: What is still unproven | scripts/final_report.py::s_uncertainty | 1 | 0 |
| 46.6.1 | [~] | 6. Champion configuration: Exact configuration and hash | scripts/final_report.py::cfg_hash<br>engine/adaptive.py::audit_matches_cfg | 1 | 0 |
| 46.7.1 | [~] | 7. Challenger configurations: Exact configurations and evidence | scripts/final_report.py::cfg_hash<br>engine/adaptive.py::audit_matches_cfg | 1 | 0 |
| 46.8.1 | [~] | median | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.2 | [~] | mean | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.3 | [~] | standard deviation | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.4 | [~] | percentiles | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.5 | [~] | worst | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.6 | [~] | best | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.7 | [~] | drawdown | scripts/final_report.py::distribution | 1 | 0 |
| 46.8.8 | [~] | era breakdown | scripts/final_report.py::distribution | 1 | 0 |
| 46.9.1 | [x] | 9. Leakage audit: Explicit pass/fail | scripts/final_report.py::audit<br>engine/livesim.py::_raise_on_fail | 6 | 4 |
| 46.10.1 | [~] | 10. Reproducibility audit: Explicit pass/fail | scripts/final_report.py::audit<br>engine/pattern_lifecycle.py::_fail | 4 | 2 |
| 46.11.1 | [~] | 11. Checklist: Every item marked honestly | scripts/site_sources.py::parse_checklist | 1 | 0 |

### Phase 22: BLIND CLOCK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 22.0.1 | [~] | BLIND CLOCK: Every simulated day: | engine/blind_gates.py::BlindClock<br>engine/livesim.py::BlindGateError | 3 | 4 |

### Phase 37: CHECKLIST STATE MACHINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 37.0.1 | [~] | CHECKLIST STATE MACHINE: Create/maintain: | scripts/bible_trace.py<br>scripts/test_inventory.py | 2 | 0 |

### Phase 40: WHAT "KEEP WORKING" MEANS

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 40.0.1 | [~] | WHAT "KEEP WORKING" MEANS: After each completed phase: | engine/ablation.py::mean_arm<br>engine/analog_weighting.py::_expanding_target_mean | 5 | 8 |

### Phase 41: NEVER SETTLE FOR A COSMETIC IMPLEMENTATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 41.0.1 | [ ] | NEVER SETTLE FOR A COSMETIC IMPLEMENTATION: Examples of unacceptable completion: | - | 0 | 0 |

### Phase 45: CONTINUOUS RESEARCH LOOP

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 45.0.1 | [ ] | CONTINUOUS RESEARCH LOOP: Once the required implementation is complete, do not consider the researc… | - | 0 | 0 |
