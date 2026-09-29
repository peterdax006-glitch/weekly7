# BIBLE TRACE

Generated 2026-09-29T05:20Z by scripts/bible_trace.py from BIBLE.md (sha deb456974707d57e). Indexed 148 code files, 72 test files, 443 evidence files.

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
| [ ] | NOT STARTED | 6 | 0.9% |
| [~] | IMPLEMENTED / TESTING | 268 | 40.2% |
| [?] | UNPROVEN | 74 | 11.1% |
| [x] | VALIDATED | 274 | 41.1% |
| [!] | FAILED | 45 | 6.7% |

By requirement kind:

| kind | [ ] | [~] | [?] | [x] | [!] |
|---|---:|---:|---:|---:|---:|
| checkbox | 1 | 31 | 11 | 30 | 5 |
| checklist | 0 | 12 | 18 | 5 | 10 |
| heading | 0 | 25 | 7 | 19 | 3 |
| list | 3 | 179 | 33 | 204 | 26 |
| phase | 2 | 3 | 0 | 0 | 0 |
| report | 0 | 18 | 5 | 16 | 1 |

## Per phase

| phase | title | reqs | [ ] | [~] | [?] | [x] | [!] | linked code lines | Bible range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | BASELINE AND CONTROL SYSTEM | 43 | 0 | 37 | 0 | 6 | 0 | 1,810 | 1,000-2,000 |
| 1 | POINT-IN-TIME DATA FIREWALL | 16 | 0 | 4 | 0 | 12 | 0 | 1,790 | 1,500-3,000 |
| 2 | FEATURE/PARITY FIREWALL | 9 | 0 | 4 | 0 | 5 | 0 | 1,085 | 1,000-2,000 |
| 3 | PATTERN MINER HARDENING | 39 | 0 | 15 | 2 | 21 | 1 | 2,603 | 3,000-6,000 |
| 4 | PATTERN LIFECYCLE | 9 | 0 | 3 | 1 | 5 | 0 | 1,253 | 1,000-2,000 |
| 5 | LONG-TERM PATTERN BANK | 11 | 0 | 1 | 0 | 8 | 2 | 1,253 | 700-1,500 |
| 6 | PATTERN → FIND VOLATILITY INTEGRATION | 10 | 0 | 2 | 1 | 7 | 0 | 1,708 | 700-1,500 |
| 7 | HEAVY ALGORITHM TESTING | 22 | 0 | 14 | 1 | 7 | 0 | 1,708 | 1,500-3,000 |
| 8 | ANALOG ENGINE | 40 | 0 | 20 | 1 | 19 | 0 | 1,561 | 2,000-4,000 |
| 9 | MEMORY SYSTEM | 26 | 0 | 12 | 2 | 12 | 0 | 1,811 | 2,000-4,000 |
| 10 | LESSON MEMORY / LEARNING FROM MISTAKES | 14 | 0 | 3 | 0 | 9 | 2 | 1,505 | 1,500-3,000 |
| 11 | RERUN ANTI-MEMORIZATION EXPERIMENT | 13 | 0 | 3 | 0 | 8 | 2 | 1,592 | 800-1,500 |
| 12 | PER-STOCK-TYPE TRUST TABLES | 11 | 0 | 4 | 1 | 6 | 0 | 1,197 | 1,000-2,000 |
| 13 | DIRECTION ENGINE | 11 | 0 | 3 | 0 | 7 | 1 | 1,256 | 1,500-3,000 |
| 14 | MISSED-WINNER DETECTOR | 11 | 0 | 4 | 1 | 3 | 3 | 2,587 | 700-1,500 |
| 15 | EXIT LEARNER | 8 | 0 | 3 | 0 | 5 | 0 | 1,376 | 1,000-2,000 |
| 16 | STOP / LOSS ENGINE | 7 | 0 | 6 | 0 | 1 | 0 | 1,638 | 1,000-2,000 |
| 17 | TIMELINE DIAL | 12 | 0 | 3 | 0 | 4 | 5 | 962 | 1,000-2,000 |
| 18 | WEEKLY ADAPTER | 21 | 0 | 0 | 6 | 14 | 1 | 2,587 | 1,500-3,000 |
| 19 | TRAIN THE TRAINING BASIS | 20 | 0 | 3 | 5 | 11 | 1 | 1,323 | 1,500-3,000 |
| 20 | TIERED OBJECTIVE FIREWALL | 9 | 0 | 3 | 0 | 4 | 2 | 2,031 | 700-1,500 |
| 21 | BLIND SIMULATOR HARDENING | 7 | 0 | 4 | 0 | 3 | 0 | 2,678 | 1,500-3,000 |
| 22 | BLIND CLOCK | 1 | 0 | 1 | 0 | 0 | 0 | 2,678 | 1,000-2,000 |
| 23 | RE-TESTER | 7 | 0 | 1 | 0 | 4 | 2 | 2,207 | 700-1,500 |
| 24 | WORKER HEALTH | 10 | 0 | 0 | 1 | 9 | 0 | 1,777 | 500-1,000 |
| 25 | PLANTED-PATTERN CALIBRATION | 6 | 0 | 0 | 1 | 4 | 1 | 743 | 500-1,000 |
| 26 | ANTI-OVERFITTING BATTERY | 10 | 0 | 2 | 0 | 5 | 3 | 1,657 | 1,000-2,000 |
| 27 | DATA EXPANSION | 7 | 0 | 2 | 2 | 3 | 0 | 1,759 | 1,000-2,500 |
| 28 | LIVE/RESEARCH SEPARATION | 7 | 0 | 2 | 2 | 3 | 0 | 1,513 | 500-1,000 |
| 29 | PUBLIC EXPLANATION SYSTEM | 19 | 0 | 10 | 1 | 8 | 0 | 2,076 | 1,500-3,000 |
| 30 | EXPERIMENT MEMORY | 12 | 0 | 4 | 4 | 4 | 0 | 3,284 | 500-1,000 |
| 31 | CODE QUALITY FIREWALL | 11 | 3 | 6 | 2 | 0 | 0 | 1,150 | - |
| 32 | TEST PYRAMID | 7 | 0 | 6 | 0 | 1 | 0 | 1,150 | - |
| 33 | REQUIRED REPRODUCIBILITY | 7 | 0 | 1 | 0 | 6 | 0 | 1,495 | - |
| 34 | REQUIRED ABLATION FRAMEWORK | 6 | 0 | 2 | 0 | 3 | 1 | 3,365 | - |
| 35 | CHAMPION / CHALLENGER SYSTEM | 10 | 0 | 4 | 2 | 4 | 0 | 1,489 | - |
| 36 | REQUIRED REPORT AFTER EACH MAJOR RUN | 40 | 0 | 18 | 5 | 16 | 1 | 3,572 | - |
| 37 | CHECKLIST STATE MACHINE | 1 | 0 | 1 | 0 | 0 | 0 | 1,633 | - |
| 38 | MASTER CHECKLIST | 45 | 0 | 12 | 18 | 5 | 10 | 2,908 | - |
| 39 | DEFINITION OF DONE | 32 | 1 | 15 | 9 | 3 | 4 | 1,766 | - |
| 40 | WHAT "KEEP WORKING" MEANS | 1 | 0 | 1 | 0 | 0 | 0 | 0 | - |
| 41 | NEVER SETTLE FOR A COSMETIC IMPLEMENTAT… | 1 | 1 | 0 | 0 | 0 | 0 | 0 | - |
| 42 | THE STANDARD FOR EVERY CLAIM | 19 | 0 | 6 | 4 | 8 | 1 | 1,407 | - |
| 43 | PRIORITY WHEN TIME/COMPUTE IS LIMITED | 11 | 0 | 2 | 0 | 7 | 2 | 1,407 | - |
| 44 | RESOURCE MANAGEMENT | 9 | 0 | 6 | 1 | 2 | 0 | 1,407 | - |
| 45 | CONTINUOUS RESEARCH LOOP | 1 | 1 | 0 | 0 | 0 | 0 | 0 | - |
| 46 | FINAL REPORT REQUIREMENT | 18 | 0 | 15 | 1 | 2 | 0 | 270 | - |

## The 20 most important MISSING items

Ranked by Phase 43's priority order (anti-leak first), then leak/gate/fail-closed wording; at most 5 per phase.

1. `41.0.1` (phase 41, Bible line 2015): NEVER SETTLE FOR A COSMETIC IMPLEMENTATION: Examples of unacceptable completion:
2. `45.0.1` (phase 45, Bible line 2153): CONTINUOUS RESEARCH LOOP: Once the required implementation is complete, do not consider the research finished
3. `31.0.1` (phase 31, Bible line 1635): type-safe interfaces where appropriate
4. `31.0.2` (phase 31, Bible line 1636): explicit inputs
5. `31.0.5` (phase 31, Bible line 1639): logging
6. `39.0.30` (phase 39, Bible line 1952): documentation reflects actual state

## Failed (45)

- `3.1.7` random pair sampling -> failed verdict recorded in tests/test_candidates.py::test_generator_reserves_budget_for_context_pairs_and_random_pairs
- `5.0.8` context -> failed verdict recorded in tests/test_candidates.py::test_generator_reserves_budget_for_context_pairs_and_random_pairs
- `5.0.11` last validation -> failed verdict recorded in state/research/pattern_bank/bank_v3/bank_v000004.json
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
- `17.0.10` no unacceptable risk deterioration -> linked; failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `17.0.12` stability across eras -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `18.0.12` improvement -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `19.0.14` detector max -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `20.TIER-2.3` weeks outside band -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `20.TIER-3.1` positive in-band weeks -> failed verdict recorded in state/research/timeline_basis/dial_offline_seed0.json
- `23.0.1` holdings -> failed verdict recorded in state/research/blind_gates/report.md
- `23.0.2` trades -> failed verdict recorded in state/research/blind_gates/report.md
- `25.0.2` false discovery rate -> linked; failed verdict recorded in state/research/algorithm/planted/report.md
- `26.B.1` B. Label permutation: Random labels should destroy apparent predictive performance -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `26.C.1` C. Ticker permutation: Ticker identities should not create fake signal -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `26.F.1` F. Feature shuffle: Important features should lose signal when shuffled -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `34.0.5` shuffled feature -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `36.GATES.2` GATES: retester -> failed verdict recorded in state/research/blind_gates/results.json
- `38.ALGORITHM.5` ALGORITHM: Long-term pattern bank -> derived from 11 requirements: 1 [~], 8 [x], 2 [!]
- `38.ALGORITHM.12` ALGORITHM: Timeline dial -> derived from 12 requirements: 3 [~], 4 [x], 5 [!]
- `38.ALGORITHM.14` ALGORITHM: Rerun anti-memorization test -> derived from 13 requirements: 3 [~], 8 [x], 2 [!]
- `38.ALGORITHM.17` ALGORITHM: Planted-pattern calibration -> derived from 6 requirements: 1 [?], 4 [x], 1 [!]
- `38.TEST.3` TEST: Tiered 7% objective -> derived from 9 requirements: 3 [~], 4 [x], 2 [!]
- `38.TEST.9` TEST: Re-tester parity -> derived from 7 requirements: 1 [~], 4 [x], 2 [!]
- `38.TEST.14` TEST: Label permutation -> derived from 1 requirements: 1 [!]
- `38.TEST.15` TEST: Feature shuffle -> derived from 1 requirements: 1 [!]
- `38.TEST.16` TEST: Ticker permutation -> derived from 1 requirements: 1 [!]
- `38.TEST.17` TEST: Planted-pattern calibration -> derived from 6 requirements: 1 [?], 4 [x], 1 [!]
- `39.0.6` re-tester works -> derived from 7 requirements: 1 [~], 4 [x], 2 [!]
- `39.0.18` anti-memorization test works -> derived from 13 requirements: 3 [~], 8 [x], 2 [!]
- `39.0.23` timeline dial exists and is tested -> derived from 12 requirements: 3 [~], 4 [x], 5 [!]
- `39.0.25` planted-pattern calibration works -> derived from 6 requirements: 1 [?], 4 [x], 1 [!]
- `42.0.6` permutation evidence -> failed verdict recorded in state/research/antioverfit/real_v1.log
- `43.0.3` Pattern validation -> failed verdict recorded in state/research/pattern_bank/bank_v3/bank_v000004.json
- `43.0.8` Exit/stop -> failed verdict recorded in tests/test_gaprisk.py::test_empty_and_unfitted_and_stop_rule_hook

## Bible claims the trace cannot support (35)

- `0.1.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.2`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.4`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `1.1.6`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `2.0.3`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
- `2.0.5`  claim [ ] vs computed [x]: Bible checkbox is stale (unchecked but traceable)
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
... and 10 more (all in bible_trace.json).

## Task-file output not on disk

None.

## Code no requirement links to

- engine/candles.py
- engine/config.py
- engine/data.py
- engine/edgar.py
- engine/explain.py
- engine/fill_audit.py
- engine/model.py
- engine/options.py
- engine/scoring.py
- engine/shadows.py
- engine/tick.py
- engine/train.py
- engine/universe.py
- scripts/algo_test.py
- scripts/analog_test.py
- scripts/archive_opens.py
- scripts/audit_registry.py
- scripts/backfill_market_snaps.py
- scripts/backtest_summary.py
- scripts/basis_offline.py
- scripts/bootstrap_history.py
- scripts/diag.py
- scripts/download_1962.py
- scripts/experiment.py
- scripts/extend_history.py
- scripts/fetch_macro.py
- scripts/frontier.py
- scripts/fv_eval.py
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
- scripts/run_13d_fix.py
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

- improve.py: 1 pending, e.g. major runs call checkpoint.write_checkpoint, run_report.build_report/write_report; Board.promote re…
- site_build.py: 1 pending, e.g. CI: site_build.py --verify; optional site_publish.py --interval 300 (no --push)
- site_publish.py: 1 pending, e.g. CI: site_build.py --verify; optional site_publish.py --interval 300 (no --push)
- train.py: 2 pending, e.g. train.py: pit.purged_training_set(X, y, as_of, horizon, cal) instead of ad-hoc label cutting

## Trace invariants (0 violations)

All hold.

## Explicit links (92 honoured of 92 declared, 0 broken)

No broken links.

State changes caused by honoured links: [ ] -> [!] x2, [ ] -> [?] x15, [ ] -> [x] x9, [?] -> [x] x20, [x] -> [?] x4, [x] -> [~] x3, [~] -> [?] x7


## Genuine gaps (6)

Requirements with no implementing code after explicit links were applied and the code was read. A reason means a reviewer confirmed it is missing; 'UNREVIEWED' means the word match found nothing and nobody has checked.

- `31.0.1` (phase 31, Bible line 1635, list): type-safe interfaces where appropriate -- reviewed quality_gate.py (check_* list): no check requires or counts type annotations; nothing in engine/ enforces type-safe interfaces
- `31.0.2` (phase 31, Bible line 1636, list): explicit inputs -- reviewed quality_gate.py: check_mutable_defaults and check_hidden_global_state cover parts of 'explicit inputs', but no check that a module reads its inputs only from arguments
- `31.0.5` (phase 31, Bible line 1639, list): logging -- reviewed quality_gate.py: check_print_debug forbids print() in engine/ but nothing requires modules to log; the repo has no logging usage to check
- `39.0.30` (phase 39, Bible line 1952, checkbox): documentation reflects actual state -- no requirement maps to it: nothing compares README/BLUEPRINT/state docs with the code except this trace itself
- `41.0.1` (phase 41, Bible line 2015, phase): NEVER SETTLE FOR A COSMETIC IMPLEMENTATION: Examples of unacceptable completion: -- NOT CODE: a directive about how to work (no cosmetic implementations); it is enforced by review, and the trace/quality gate are the closest mechanisms
- `45.0.1` (phase 45, Bible line 2153, phase): CONTINUOUS RESEARCH LOOP: Once the required implementation is complete, do not consider the research finished -- NOT CODE: a directive to keep researching after implementation; scripts/loop_years.py runs the C10 fresh-year loop only, there is no general research-loop scheduler

## Detail by phase


### Phase 0: BASELINE AND CONTROL SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 0.1.1 | [~] | Read CANON.md | engine/champion.py::_canon | 1 | 2 |
| 0.1.2 | [~] | Hash the canon | engine/champion.py::_canon<br>engine/provenance.py::code_hash | 3 | 26 |
| 0.1.3 | [x] | Verify current hash | engine/basis_search.py::_hash<br>engine/data_sources.py::frame_hash | 7 | 21 |
| 0.1.4 | [~] | Add a canon integrity check | engine/provenance.py::IntegrityError | 2 | 24 |
| 0.1.5 | [~] | Ensure experiments record the canon hash | engine/champion.py::_canon<br>engine/memory.py::record | 5 | 20 |
| 0.1.6 | [~] | Ensure experiments record blueprint version | engine/champion.py::_canon<br>engine/provenance.py | 3 | 26 |
| 0.1.7 | [~] | Ensure code commits are recorded | engine/provenance.py::code_hash | 2 | 24 |
| 0.2.1 | [x] | experiment ID | engine/registry.py::ExperimentMemory<br>engine/antimemo.py::archive_experiment | 10 | 10 |
| 0.2.2 | [~] | timestamp | engine/registry.py::ExperimentMemory | 4 | 2 |
| 0.2.3 | [~] | git commit | engine/provenance.py::git_commit<br>engine/registry.py::ExperimentMemory | 2 | 24 |
| 0.2.4 | [x] | canon hash | engine/data_sources.py::DelistedRegistry<br>engine/provenance.py::code_hash | 7 | 37 |
| 0.2.5 | [~] | blueprint version | engine/provenance.py<br>engine/improve.py::log_experiment | 3 | 24 |
| 0.2.6 | [~] | configuration hash | scripts/final_report.py::cfg_hash<br>engine/baseline.py | 8 | 7 |
| 0.2.7 | [~] | data snapshot | engine/antimemo.py::archive_experiment<br>engine/baseline.py | 3 | 6 |
| 0.2.8 | [~] | random seed | engine/antimemo.py::archive_experiment<br>engine/registry.py::ExperimentMemory | 3 | 6 |
| 0.2.9 | [x] | window IDs | engine/registry.py::REQUIRED | 1 | 1 |
| 0.2.10 | [~] | model parameters | engine/experiment_memory.py::import_registry<br>engine/antimemo.py::archive_experiment | 9 | 27 |
| 0.2.11 | [~] | training range | engine/experiment_memory.py::import_registry<br>engine/registry.py::ExperimentMemory | 3 | 0 |
| 0.2.12 | [~] | validation range | engine/experiment_memory.py::import_registry<br>engine/exits.py::validate | 6 | 19 |
| 0.2.13 | [~] | test range | engine/experiment_memory.py::import_registry<br>engine/registry.py::ExperimentMemory | 3 | 0 |
| 0.2.14 | [~] | metrics | engine/registry.py::ExperimentMemory<br>engine/exits.py::band_metrics | 9 | 21 |
| 0.2.15 | [~] | gates | engine/registry.py::ExperimentMemory | 4 | 2 |
| 0.2.16 | [~] | outcome | engine/registry.py::ExperimentMemory<br>engine/antimemo.py::archive_experiment | 10 | 10 |
| 0.2.17 | [x] | reason for adoption/rejection | engine/experiment_memory.py::import_registry<br>engine/run_report.py | 3 | 0 |
| 0.3.1 | [~] | config | engine/checkpoint.py::CheckpointError | 2 | 1 |
| 0.3.2 | [~] | logs | engine/checkpoint.py::CheckpointError | 2 | 1 |
| 0.3.3 | [~] | metrics | engine/baseline.py::extract_metrics<br>engine/checkpoint.py::CheckpointError | 3 | 1 |
| 0.3.4 | [~] | artifact manifest | engine/checkpoint.py::CheckpointError | 2 | 1 |
| 0.3.5 | [~] | random seeds | engine/checkpoint.py::CheckpointError | 2 | 1 |
| 0.3.6 | [~] | hashes | engine/checkpoint.py::CheckpointError | 2 | 1 |
| 0.3.7 | [~] | output summary | engine/pit.py::summary<br>engine/checkpoint.py::CheckpointError | 1 | 1 |
| 0.4.1 | [~] | current blind-window results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::BlindGateError | 10 | 7 |
| 0.4.2 | [~] | current mover accuracy | engine/baseline.py::diff_vs_baseline<br>engine/run_report.py::baseline_metrics | 2 | 0 |
| 0.4.3 | [~] | current model results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::snapshot_from | 10 | 7 |
| 0.4.4 | [~] | current pattern results | engine/baseline.py::diff_vs_baseline<br>engine/adaptive.py::from_snapshot | 13 | 9 |
| 0.4.5 | [~] | current analog results | engine/adaptive.py::from_snapshot<br>engine/livesim.py::snapshot_from | 13 | 9 |
| 0.4.6 | [x] | current weekly distribution | scripts/final_report.py::distribution<br>engine/run_report.py::analog_distance_distribution | 3 | 0 |
| 0.4.7 | [~] | max drawdown | engine/run_report.py::max_drawdown<br>engine/analog_weighting.py::baseline | 3 | 1 |
| 0.4.8 | [~] | worst weeks | engine/live.py::week_state<br>engine/baseline.py::diff_vs_baseline | 2 | 0 |
| 0.4.9 | [~] | turnover | engine/run_report.py::baseline_metrics<br>engine/baseline.py::diff_vs_baseline | 4 | 2 |
| 0.4.10 | [~] | costs | engine/baseline.py::diff_vs_baseline<br>engine/champion.py::freeze_baseline | 3 | 2 |
| 0.4.11 | [~] | missed winners | engine/baseline.py::diff_vs_baseline<br>engine/champion.py::freeze_baseline | 3 | 2 |
| 0.4.12 | [~] | direction accuracy where available | engine/run_report.py::direction_accuracy_report<br>engine/baseline.py::diff_vs_baseline | 3 | 0 |

### Phase 1: POINT-IN-TIME DATA FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 1.1.1 | [~] | Every feature carries an effective date | engine/lessons.py::usable_features<br>engine/antioverfit.py::date_disguise | 3 | 21 |
| 1.1.2 | [x] | Every filing carries publication/availability date | engine/pit.py::IntegrityError<br>engine/data_sources.py::_rank_ic_by_date | 2 | 3 |
| 1.1.3 | [x] | Every macro series has publication lag | engine/pit.py::IntegrityError<br>scripts/pit_audit_real.py::macro_lags | 1 | 1 |
| 1.1.4 | [x] | Every label has a close date | engine/pit.py::label_close_dates<br>scripts/pit_audit_real.py::panel_integrity | 4 | 7 |
| 1.1.5 | [~] | Every training row proves its label closed before prediction | engine/blind_gates.py::check_label_alignment<br>engine/pit.py::label_close_dates | 3 | 6 |
| 1.1.6 | [x] | Future rows cannot be queried | engine/pit.py::IntegrityError<br>engine/analog_weighting.py::query | 2 | 2 |
| 1.2.1 | [~] | 1.2 Time-fence enforcement: Implement a hard object-level time fence | engine/pit.py::fence | 1 | 1 |
| 1.3.1 | [x] | predictions unchanged | engine/pit.py::future_scramble<br>engine/antioverfit.py::future_scramble | 7 | 11 |
| 1.3.2 | [x] | scores unchanged | engine/pit.py::future_scramble<br>engine/ablation.py::score | 4 | 9 |
| 1.3.3 | [x] | pattern selection unchanged | engine/pit.py::future_scramble<br>engine/fv_pipeline.py::Selection | 2 | 1 |
| 1.3.4 | [x] | memory unchanged | engine/livesim.py::long_term_memory<br>engine/pit.py::future_scramble | 4 | 8 |
| 1.3.5 | [x] | adaptive decisions unchanged | engine/pit.py::future_scramble<br>engine/blind_gates.py::check_decisions_use_recorded_info | 6 | 9 |
| 1.4.1 | [x] | decision after close | engine/pit.py::audit_fills<br>scripts/pit_audit_real.py::live_decisions | 5 | 7 |
| 1.4.2 | [x] | fill next open | engine/pit.py::fill_next_open<br>scripts/pit_audit_real.py | 5 | 7 |
| 1.4.3 | [x] | no same-close execution | engine/pit.py::audit_fills<br>scripts/pit_audit_real.py | 5 | 7 |
| 1.4.4 | [~] | no next-day information leakage | engine/livesim.py::_information_sources<br>engine/blind_gates.py::InformationLedger | 2 | 4 |

### Phase 2: FEATURE/PARITY FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 2.0.1 | [~] | feature parity harness | engine/parity.py::FeatureContract<br>engine/parity_suite.py::analog_find_parity | 2 | 2 |
| 2.0.2 | [x] | random-date sampling | engine/parity.py::sample_dates | 1 | 1 |
| 2.0.3 | [x] | strict recomputation | engine/parity.py::strict_row<br>engine/parity_suite.py::strict | 2 | 2 |
| 2.0.4 | [~] | fast recomputation | engine/parity.py<br>scripts/run_parity.py::fast_git_commit | 2 | 2 |
| 2.0.5 | [x] | exact comparison | engine/parity.py<br>engine/claims.py::Comparison | 3 | 2 |
| 2.0.6 | [~] | maximum absolute error | engine/adaptive.py::GuardRailError<br>engine/parity.py::max_abs | 8 | 7 |
| 2.0.7 | [x] | maximum relative error | engine/parity.py::ParityReport.max_rel | 1 | 1 |
| 2.0.8 | [x] | NaN mismatch detection | engine/parity.py::compare_panels | 1 | 1 |
| 2.0.9 | [~] | missing-data mismatch detection | scripts/bible_trace.py::top_missing<br>engine/data_sources.py::detect_splits | 3 | 2 |

### Phase 3: PATTERN MINER HARDENING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 3.1.1 | [x] | all 54 engine features | engine/candidates.py::Candidate<br>engine/pattern_identity.py::features | 2 | 4 |
| 3.1.2 | [x] | all 25 candle/micro signals | engine/candidates.py::Candidate<br>engine/patterns.py::with_candles | 2 | 4 |
| 3.1.3 | [x] | market context | engine/candidates.py::context_pair_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.4 | [x] | quintile transformation | engine/candidates.py::Candidate<br>engine/planted.py::generate | 4 | 10 |
| 3.1.5 | [x] | single patterns | engine/candidates.py::single_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.6 | [x] | pair patterns | engine/candidates.py::context_pair_candidates<br>engine/planted.py::generate | 4 | 10 |
| 3.1.7 | [!] | random pair sampling | engine/planted.py::generate<br>engine/candidates.py::random_pair_candidates | 7 | 21 |
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
| 3.3.1 | [?] | 3.3 Relevance weighting: relevance = recency_weight × context_similarity_weight × era_weight | engine/pattern_stats.py::relevance<br>engine/patterns.py::PatternMiner | 1 | 0 |
| 3.4.1 | [~] | week-clustered observations | engine/pattern_stats.py::validation_gain_gate<br>engine/patterns.py | 10 | 8 |
| 3.4.2 | [x] | weighted mean | engine/pattern_stats.py::weighted_mean<br>engine/patterns.py::_weights | 13 | 26 |
| 3.4.3 | [~] | effective sample size | engine/pattern_stats.py::effective_n<br>engine/memory.py::validate | 1 | 3 |
| 3.4.4 | [x] | t-statistic | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 13 | 26 |
| 3.4.5 | [~] | p-value | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 10 | 8 |
| 3.4.6 | [x] | FDR | engine/pattern_identity.py<br>engine/pattern_stats.py::validation_gain_gate | 13 | 26 |
| 3.4.7 | [~] | permutation null | engine/patterns.py<br>engine/antioverfit.py::_null_summary | 12 | 10 |
| 3.4.8 | [x] | local false-discovery estimate | engine/pattern_stats.py::local_fdr<br>engine/patterns.py | 1 | 3 |
| 3.4.9 | [~] | later confirmation | engine/pattern_stats.py::confirmation_factor<br>engine/patterns.py | 10 | 8 |
| 3.4.10 | [~] | P(real) | engine/pattern_stats.py::p_real<br>engine/patterns.py | 15 | 29 |
| 3.4.11 | [~] | effect shrinkage | engine/pattern_stats.py::effective_n<br>engine/patterns.py | 10 | 8 |
| 3.5.1 | [x] | P(real) >= 0.80 | engine/pattern_stats.py::admission_flags<br>engine/patterns.py::PatternMiner | 1 | 3 |
| 3.5.2 | [~] | survives redundancy rules | engine/pattern_stats.py::admission_flags<br>engine/patterns.py::PatternMiner | 1 | 3 |
| 3.5.3 | [x] | passes validation-gain threshold | engine/pattern_stats.py::validation_gain_gate | 1 | 1 |
| 3.5.4 | [x] | confirmation preserves sign | engine/pattern_stats.py::admission_flags | 1 | 1 |
| 3.5.5 | [?] | does not violate anti-leak rules | engine/pattern_stats.py::LookAheadError<br>engine/patterns.py::PatternMiner.fit | 1 | 0 |
| 3.6.1 | [~] | 3.6 Redundancy: Implement overlap comparison | engine/patterns.py::_overlap | 10 | 7 |
| 3.7.1 | [x] | 3.7 Validation-gain gate: The pattern must improve genuinely unseen prediction quality | engine/pattern_stats.py::validation_gain_gate | 1 | 1 |

### Phase 4: PATTERN LIFECYCLE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 4.0.1 | [x] | failure detector | engine/pattern_lifecycle.py::detect_failure | 2 | 1 |
| 4.0.2 | [~] | context terciles | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 3 | 14 |
| 4.0.3 | [x] | rescoping search | engine/pattern_lifecycle.py::_resolve_search<br>engine/patterns.py::search | 10 | 7 |
| 4.0.4 | [x] | long-run test | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 6 | 20 |
| 4.0.5 | [~] | discovery test | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 5 | 14 |
| 4.0.6 | [?] | confirmation test | engine/pattern_lifecycle.py::run_tests | 1 | 0 |
| 4.0.7 | [~] | recent-stretch test | engine/pattern_lifecycle.py | 2 | 1 |
| 4.0.8 | [x] | sign-consistency test | engine/pattern_lifecycle.py::_sign | 2 | 1 |
| 4.0.9 | [x] | discard reason | engine/pattern_lifecycle.py<br>scripts/run_pattern_bank.py | 2 | 1 |

### Phase 5: LONG-TERM PATTERN BANK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 5.0.1 | [x] | pattern identity | engine/pattern_lifecycle.py::pattern_id<br>engine/pattern_identity.py::IdentityError | 11 | 8 |
| 5.0.2 | [x] | history | engine/pattern_bank.py::PatternBank.history | 1 | 1 |
| 5.0.3 | [x] | discovery evidence | engine/pattern_identity.py::PatternRecord | 1 | 1 |
| 5.0.4 | [x] | failure evidence | engine/pattern_lifecycle.py::detect_failure | 1 | 1 |
| 5.0.5 | [x] | rescoping | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 3 | 14 |
| 5.0.6 | [~] | current relevance | engine/pattern_bank.py | 2 | 14 |
| 5.0.7 | [x] | modernity | engine/pattern_bank.py | 2 | 14 |
| 5.0.8 | [!] | context | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 6 | 18 |
| 5.0.9 | [x] | effect | engine/pattern_bank.py<br>engine/pattern_stats.py::effective_n | 5 | 34 |
| 5.0.10 | [x] | confidence | engine/pattern_bank.py<br>engine/analog_weighting.py::confidence_calibration | 5 | 15 |
| 5.0.11 | [!] | last validation | engine/pattern_bank.py<br>engine/adaptive.py::validate_cfg | 14 | 42 |

### Phase 6: PATTERN → FIND VOLATILITY INTEGRATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 6.0.1 | [x] | pattern movement score | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::MovementCalibrator | 3 | 1 |
| 6.0.2 | [x] | movement probability | engine/pattern_movers.py::MovementCalibrator | 2 | 1 |
| 6.0.3 | [~] | pattern contribution attribution | engine/pattern_movers.py::PatternMoverModel | 2 | 1 |
| 6.0.4 | [x] | interaction with existing mover model | engine/pattern_movers.py::MoverModel<br>engine/fv_pipeline.py::MoverStage | 2 | 1 |
| 6.0.5 | [?] | out-of-sample comparison | engine/pattern_movers.py<br>engine/claims.py::Comparison | 5 | 3 |
| 6.0.6 | [x] | mover model alone | engine/heavy_tests.py<br>engine/pattern_movers.py::MoverModel | 2 | 1 |
| 6.0.7 | [~] | pattern score alone | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 4 | 7 |
| 6.0.8 | [x] | mover + pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 3 | 2 |
| 6.0.9 | [x] | mover + random pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 3 | 1 |
| 6.0.10 | [x] | mover + shuffled pattern | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 2 | 2 |

### Phase 7: HEAVY ALGORITHM TESTING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 7.0.1 | [~] | multiple eras | engine/heavy_tests.py::ablation_across_eras | 2 | 0 |
| 7.0.2 | [~] | bull markets | engine/heavy_tests.py | 2 | 0 |
| 7.0.3 | [~] | bear markets | engine/heavy_tests.py | 2 | 0 |
| 7.0.4 | [~] | high-volatility regimes | engine/heavy_tests.py::by_regime<br>engine/gaprisk.py::_week_regime | 3 | 8 |
| 7.0.5 | [?] | low-volatility regimes | engine/heavy_tests.py::by_regime | 1 | 0 |
| 7.0.6 | [~] | pre-decimal era | engine/heavy_tests.py | 2 | 0 |
| 7.0.7 | [~] | post-decimal era | engine/heavy_tests.py | 2 | 0 |
| 7.0.8 | [~] | post-electronic era | engine/heavy_tests.py | 2 | 0 |
| 7.0.9 | [~] | modern market | engine/heavy_tests.py | 2 | 0 |
| 7.0.10 | [~] | random hidden windows | engine/heavy_tests.py::hidden_windows<br>scripts/quality_gate.py::check_hidden_global_state | 4 | 0 |
| 7.0.11 | [x] | movement prediction | engine/heavy_tests.py<br>engine/pattern_movers.py::MovementCalibrator | 7 | 41 |
| 7.0.12 | [~] | direction prediction | engine/heavy_tests.py<br>engine/pattern_movers.py::_direction_accuracy | 5 | 35 |
| 7.0.13 | [~] | IC | engine/heavy_tests.py::run_heavy<br>scripts/heavy_algo_real.py | 2 | 0 |
| 7.0.14 | [x] | rank IC | engine/heavy_tests.py<br>engine/pattern_movers.py | 3 | 6 |
| 7.0.15 | [x] | t-stat | engine/heavy_tests.py::series_stats<br>scripts/heavy_algo_real.py | 11 | 9 |
| 7.0.16 | [~] | false discoveries | engine/heavy_tests.py::false_discovery_summary | 2 | 0 |
| 7.0.17 | [x] | pattern count | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 4 | 5 |
| 7.0.18 | [~] | pattern survival | engine/heavy_tests.py::_oos_pattern_check<br>engine/pattern_movers.py::PatternMoverModel | 5 | 3 |
| 7.0.19 | [x] | turnover | engine/heavy_tests.py<br>engine/retester.py::turnover | 3 | 1 |
| 7.0.20 | [x] | costs | engine/heavy_tests.py::cost_sensitivity<br>scripts/heavy_algo_real.py | 6 | 20 |
| 7.0.21 | [x] | stability | engine/heavy_tests.py::stability<br>engine/exits.py::selection_stability | 8 | 25 |
| 7.0.22 | [~] | regime sensitivity | engine/heavy_tests.py::by_regime | 2 | 0 |

### Phase 8: ANALOG ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 8.1.1 | [~] | market drawdown | engine/analog_weighting.py::market_context<br>engine/analogs_sector.py::sector_fingerprints | 2 | 15 |
| 8.1.2 | [x] | 1m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 12 |
| 8.1.3 | [x] | 3m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 12 |
| 8.1.4 | [x] | 6m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 12 |
| 8.1.5 | [x] | 12m return | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 9 | 12 |
| 8.1.6 | [~] | acceleration | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 2 | 15 |
| 8.1.7 | [~] | distance from MA200 | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 1 | 5 |
| 8.1.8 | [~] | volatility | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 3 | 5 |
| 8.1.9 | [~] | volatility ratio | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs_stock.py::stock_fingerprint | 3 | 5 |
| 8.1.10 | [x] | VIX | engine/analog_weighting.py<br>engine/analogs_sector.py::sector_fingerprints | 2 | 15 |
| 8.1.11 | [~] | VIX term structure | engine/analog_weighting.py<br>engine/analogs.py::fingerprints | 2 | 15 |
| 8.1.12 | [x] | breadth | engine/analogs_sector.py::breadth_dispersion_crowding<br>engine/analogs.py::fingerprints | 2 | 15 |
| 8.1.13 | [x] | dispersion | engine/analogs_sector.py::breadth_dispersion_crowding<br>engine/analogs.py::fingerprints | 4 | 15 |
| 8.1.14 | [x] | sector crowding | engine/analogs_sector.py::sector_fingerprints<br>engine/analogs.py::fingerprints | 2 | 15 |
| 8.1.15 | [x] | top-sector concentration change | engine/analogs_sector.py::sector_fingerprints | 1 | 3 |
| 8.1.16 | [~] | FRED data | engine/parity_suite.py::fingerprints_builder<br>engine/analogs_stock.py::data_sufficient | 1 | 2 |
| 8.1.17 | [~] | 10Y yield change | engine/analogs_sector.py::sector_fingerprints | 1 | 3 |
| 8.1.18 | [~] | credit-spread change | engine/pit.py::fingerprint<br>engine/analogs_sector.py::sector_fingerprints | 1 | 1 |
| 8.2.1 | [?] | 8.2 Point-in-time standardization: Standardize using only historical information available at the m… | engine/analog_weighting.py::pit_moments | 1 | 0 |
| 8.3.1 | [~] | analog end >= 63 sessions before prediction | engine/analogs.py::Analogs<br>engine/analog_weighting.py::predict_series | 2 | 15 |
| 8.3.2 | [~] | analogs at least 21 sessions apart | engine/analogs.py::Analogs<br>engine/analogs_stock.py::PooledStockAnalogs | 2 | 15 |
| 8.3.3 | [~] | one analog per episode | engine/analogs.py::Analogs<br>engine/analog_weighting.py::select_episodes | 1 | 15 |
| 8.4.1 | [x] | 8.4 Feature weighting: Implement learnable feature weights | engine/analog_weighting.py::WeightSchedule<br>engine/analogs_sector.py::panel_features | 1 | 5 |
| 8.5.1 | [x] | 8.5 Sector analogs: Implement sector fingerprints | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 2 | 15 |
| 8.6.1 | [x] | 8.6 Stock-level analogs: Implement stock-level fingerprints where data suffices | engine/analogs_stock.py::PooledStockAnalogs<br>engine/analog_weighting.py::level_agreement | 1 | 3 |
| 8.7.1 | [~] | analog dates | engine/analog_weighting.py<br>engine/run_report.py::analog_distance_distribution | 5 | 7 |
| 8.7.2 | [~] | distances | engine/analog_weighting.py<br>engine/run_report.py::analog_distance_distribution | 3 | 1 |
| 8.7.3 | [~] | ages | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 1 | 1 |
| 8.7.4 | [x] | forecast return | engine/analog_weighting.py::blend_forecasts<br>engine/run_report.py::analog_distance_distribution | 5 | 7 |
| 8.7.5 | [~] | volatility | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 1 | 1 |
| 8.7.6 | [~] | drawdown | engine/analog_weighting.py<br>engine/run_report.py::analog_distance_distribution | 5 | 7 |
| 8.7.7 | [~] | uniqueness | engine/analogs_sector.py::SectorAnalogs<br>engine/analogs_stock.py::PooledStockAnalogs | 1 | 5 |
| 8.7.8 | [~] | analog count | engine/analog_weighting.py<br>engine/timeline.py::DialOutput | 3 | 7 |
| 8.7.9 | [~] | confidence | engine/analog_weighting.py::confidence_calibration<br>engine/run_report.py::analog_distance_distribution | 3 | 1 |
| 8.8.1 | [x] | no analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 5 |
| 8.8.2 | [x] | market analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 5 |
| 8.8.3 | [x] | sector analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 3 |
| 8.8.4 | [x] | stock analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 5 |
| 8.8.5 | [x] | shuffled analog | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 3 |
| 8.8.6 | [x] | nearest-neighbor random control | engine/analog_weighting.py::ablation_table<br>engine/analogs_sector.py::SectorAnalogs | 1 | 1 |

### Phase 9: MEMORY SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 9.0.1 | [x] | recency | engine/memory.py<br>engine/memory_diagnostics.py | 8 | 21 |
| 9.0.2 | [x] | market similarity | engine/memory.py<br>engine/memory_diagnostics.py | 5 | 21 |
| 9.0.3 | [~] | reliability shrinkage | engine/memory.py::reliability | 4 | 18 |
| 9.0.4 | [x] | shock detection | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::detect_lag | 4 | 18 |
| 9.0.5 | [x] | long-term prior | engine/memory.py<br>engine/pattern_bank.py::prior_frame | 6 | 32 |
| 9.0.6 | [x] | modernity | engine/memory.py<br>engine/memory_diagnostics.py | 4 | 18 |
| 9.1.1 | [~] | situation fingerprint | engine/lessons.py::_situation<br>engine/memory.py::Memory | 11 | 42 |
| 9.1.2 | [~] | relevant features | engine/memory.py::Memory<br>engine/data_sources.py::keep_relevant_forms | 4 | 18 |
| 9.1.3 | [~] | context | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 7 | 39 |
| 9.1.4 | [~] | outcome | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 10 | 39 |
| 9.1.5 | [x] | error type | engine/memory.py::Memory<br>engine/memory_diagnostics.py::error_profile_by_arm | 4 | 18 |
| 9.1.6 | [~] | source experiment | engine/memory.py::Memory<br>engine/isolation.py::RecordingClient | 4 | 18 |
| 9.1.7 | [x] | date | engine/memory.py::Memory<br>engine/blind_gates.py::check_decisions_use_recorded_info | 13 | 42 |
| 9.1.8 | [x] | era | engine/memory.py::era_of | 1 | 1 |
| 9.1.9 | [~] | relevance | engine/memory.py::Memory<br>engine/patterns.py::export_records | 14 | 25 |
| 9.1.10 | [~] | reliability | engine/memory.py::Memory<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 9.1.11 | [x] | shock state | engine/memory.py::shock_state<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 9.2.1 | [x] | ticker | engine/memory.py::coarse_date | 4 | 18 |
| 9.2.2 | [x] | exact future outcome | engine/memory.py::coarse_date<br>engine/antioverfit.py::_relabel_tickers | 7 | 24 |
| 9.2.3 | [?] | hidden test identifier | engine/memory.py::coarse_date<br>engine/memory_diagnostics.py::leak_check | 1 | 0 |
| 9.2.4 | [?] | information that uniquely identifies a test window | engine/memory.py::coarse_date<br>engine/memory_diagnostics.py::leak_check | 1 | 0 |
| 9.3.1 | [~] | Adapter: 8 weeks | engine/memory.py<br>scripts/run_b15_memory_adapter.py::build_weeks | 4 | 18 |
| 9.3.2 | [~] | miner: 4 years | engine/memory.py<br>engine/pattern_stats.py::recency_weight | 14 | 26 |
| 9.4.1 | [x] | 9.4 Market similarity: Implement Gaussian similarity | engine/memory.py::Memory.factors | 1 | 1 |
| 9.5.1 | [~] | 9.5 Reliability shrinkage: Implement pseudo-count shrinkage | engine/memory.py::counts | 4 | 18 |
| 9.6.1 | [~] | 9.6 Shock detection: Implement two-sided CUSUM | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::detect_lag | 4 | 18 |

### Phase 10: LESSON MEMORY / LEARNING FROM MISTAKES

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 10.0.1 | [~] | losing decisions | engine/lessons.py | 3 | 21 |
| 10.0.2 | [!] | missed winners | engine/missed_winners.py::MissedWinnerDetector | 2 | 6 |
| 10.0.3 | [x] | false positives | engine/lessons.py::CATEGORIES | 1 | 1 |
| 10.0.4 | [x] | false negatives | engine/lessons.py::CATEGORIES | 1 | 1 |
| 10.0.5 | [x] | pattern failures | engine/exits.py::pattern_fail_flags<br>engine/parity.py::ParityFailure | 8 | 35 |
| 10.0.6 | [~] | regime failures | engine/antioverfit.py::regime_labels | 2 | 3 |
| 10.0.7 | [~] | situation | engine/lessons.py::_situation | 3 | 21 |
| 10.0.8 | [x] | features | engine/antimemo.py::snap_features<br>engine/lessons.py::usable_features | 5 | 28 |
| 10.0.9 | [!] | context | engine/lessons.py<br>engine/analog_weighting.py::market_context | 10 | 44 |
| 10.0.10 | [x] | decision | engine/antimemo.py<br>engine/lessons.py | 6 | 25 |
| 10.0.11 | [x] | outcome | engine/antimemo.py<br>engine/lessons.py | 6 | 25 |
| 10.0.12 | [x] | error category | engine/lessons.py::category_counts | 3 | 21 |
| 10.0.13 | [x] | confidence | engine/lessons.py<br>engine/analog_weighting.py::confidence_calibration | 6 | 22 |
| 10.0.14 | [x] | counterfactual | engine/lessons.py::_counterfactual | 3 | 21 |

### Phase 11: RERUN ANTI-MEMORIZATION EXPERIMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 11.0.1 | [x] | play window A | engine/antimemo.py::play_window | 1 | 1 |
| 11.0.2 | [~] | create lessons | engine/lessons.py::Lesson<br>scripts/lessons_archive.py | 3 | 21 |
| 11.0.3 | [x] | rerun A under a fresh disguise | engine/antimemo.py::disguise | 3 | 6 |
| 11.0.4 | [~] | measure improvement | engine/antimemo.py::improvement_by_era | 3 | 6 |
| 11.0.5 | [x] | play unseen window B | engine/antimemo.py::run_experiment | 1 | 1 |
| 11.0.6 | [x] | keep lessons only when B is not harmed | engine/antimemo.py::harmed<br>engine/lessons.py::Lesson | 3 | 6 |
| 11.0.7 | [!] | improvement on A | engine/antimemo.py::improvement_by_era<br>engine/improve.py | 4 | 8 |
| 11.0.8 | [x] | improvement on disguised A | engine/antimemo.py::disguise | 3 | 6 |
| 11.0.9 | [!] | improvement on B | engine/antimemo.py::improvement_by_era<br>engine/improve.py | 4 | 8 |
| 11.0.10 | [x] | degradation on B | engine/antimemo.py::run_experiment | 1 | 1 |
| 11.0.11 | [x] | lesson count | engine/lessons.py::Lesson<br>engine/memory.py::Lesson | 11 | 41 |
| 11.0.12 | [~] | accepted lessons | engine/lessons.py::Lesson<br>scripts/lessons_archive.py | 3 | 21 |
| 11.0.13 | [x] | rejected lessons | engine/antimemo.py<br>engine/lessons.py::Lesson | 3 | 6 |

### Phase 12: PER-STOCK-TYPE TRUST TABLES

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 12.0.1 | [x] | SIC division | engine/trust.py::sic_division<br>engine/trust_store.py | 4 | 31 |
| 12.0.2 | [x] | size tercile | engine/trust.py::_tercile_labels | 2 | 27 |
| 12.0.3 | [x] | volatility tercile | engine/trust.py::_tercile_labels<br>scripts/antioverfit_real.py::vol_types | 2 | 30 |
| 12.0.4 | [~] | trend state | engine/trust.py | 2 | 27 |
| 12.0.5 | [?] | theme/industry momentum | engine/trust.py::NUMERIC | 1 | 0 |
| 12.0.6 | [~] | attention state | engine/trust.py | 2 | 27 |
| 12.0.7 | [~] | week-clustered statistics | engine/trust.py::_week_label<br>engine/direction_ablate.py::week_bootstrap | 7 | 48 |
| 12.0.8 | [x] | shrinkage | engine/trust.py | 2 | 27 |
| 12.0.9 | [~] | minimum sample size | engine/health.py::memory_growth_mb_per_min | 2 | 3 |
| 12.0.10 | [x] | confidence | engine/direction.py<br>scripts/direction_study.py | 7 | 35 |
| 12.0.11 | [x] | recent relevance | engine/trust.py<br>engine/trust_store.py | 3 | 30 |

### Phase 13: DIRECTION ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 13.0.1 | [x] | pattern direction | engine/direction.py::DirectionEngine<br>engine/direction_ablate.py | 5 | 34 |
| 13.0.2 | [x] | analogs | engine/direction.py<br>scripts/direction_study.py | 6 | 49 |
| 13.0.3 | [x] | per-type trust | engine/direction.py<br>engine/direction_calib.py::PerTypeCalibrator | 6 | 34 |
| 13.0.4 | [!] | missed-winner detector | engine/missed_winners.py::MissedWinnerDetector | 2 | 6 |
| 13.0.5 | [x] | model prediction | engine/direction.py::predict<br>scripts/direction_study.py | 7 | 42 |
| 13.0.6 | [~] | evidence where justified | engine/direction_ablate.py<br>scripts/bible_trace.py::EvidenceFile | 3 | 1 |
| 13.0.7 | [x] | calibration curves | engine/direction.py::reliability_bins | 1 | 1 |
| 13.0.8 | [x] | Brier score | engine/direction.py::brier<br>engine/missed_winners.py::_base_score | 7 | 41 |
| 13.0.9 | [x] | log loss | engine/direction.py::log_loss<br>engine/direction_ablate.py | 5 | 35 |
| 13.0.10 | [~] | reliability diagram | engine/direction.py::reliability_bins | 3 | 34 |
| 13.0.11 | [~] | out-of-sample direction accuracy | engine/direction.py::direction_accuracy<br>scripts/direction_study.py::direction_acc_at | 6 | 35 |

### Phase 14: MISSED-WINNER DETECTOR

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 14.0.1 | [?] | evidence ranks | engine/missed_winners.py::MissedWinnerDetector | 1 | 0 |
| 14.0.2 | [!] | mu_raw | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.3 | [~] | vol20 | engine/missed_winners.py | 2 | 6 |
| 14.0.4 | [~] | max20 | engine/missed_winners.py | 2 | 6 |
| 14.0.5 | [!] | log_dv | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.6 | [!] | r5 | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.7 | [~] | other approved inputs only if they pass point-in-time rules | engine/fv_pipeline.py::ComboRule<br>engine/pattern_stats.py::_check_point_in_time | 3 | 4 |
| 14.0.8 | [~] | detector alone | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.9 | [x] | detector + base | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.10 | [x] | shuffled detector | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 14.0.11 | [x] | future-scrambled detector | engine/missed_winners.py::MissedWinnerDetector<br>scripts/run_b15_memory_adapter.py | 2 | 6 |

### Phase 15: EXIT LEARNER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 15.0.1 | [x] | hit rate | engine/exits.py<br>engine/analog_weighting.py::rate | 8 | 27 |
| 15.0.2 | [x] | average gain | engine/exits.py::oos_gain_interval | 3 | 19 |
| 15.0.3 | [~] | average loss | engine/exits.py<br>engine/stops.py::loss_risk | 5 | 20 |
| 15.0.4 | [x] | tail loss | engine/exits.py<br>engine/stops.py::loss_risk | 4 | 26 |
| 15.0.5 | [~] | time held | engine/exits.py | 3 | 19 |
| 15.0.6 | [x] | turnover | engine/retester.py::turnover | 1 | 1 |
| 15.0.7 | [x] | costs | engine/exits.py::CostModel<br>engine/stops.py | 5 | 19 |
| 15.0.8 | [~] | weekly band behavior | engine/exits.py::WeekTable | 3 | 19 |

### Phase 16: STOP / LOSS ENGINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 16.0.1 | [~] | ATR-based | engine/stops.py | 2 | 19 |
| 16.0.2 | [~] | volatility percentile | engine/stops.py::VolPercentileStop | 2 | 19 |
| 16.0.3 | [~] | stock-type specific | engine/stops.py::type_gap_table | 2 | 19 |
| 16.0.4 | [x] | gap-aware | engine/gaprisk.py<br>engine/stops.py::GapAwareStop | 2 | 26 |
| 16.0.5 | [~] | position-size-aware | engine/gaprisk.py::p_position<br>engine/stops.py::SizeAwareStop | 2 | 26 |
| 16.0.6 | [~] | pattern invalidation | engine/stops.py::InvalidationStop | 2 | 19 |
| 16.0.7 | [~] | hybrid | engine/exits.py<br>engine/stops.py::HybridStop | 4 | 19 |

### Phase 17: TIMELINE DIAL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 17.0.1 | [!] | regime | engine/timeline.py::regime_from_market<br>engine/antioverfit.py::regime_labels | 8 | 14 |
| 17.0.2 | [x] | analog forecast | engine/timeline.py::forecast_from_analogs<br>engine/analog_weighting.py::blend_forecasts | 3 | 11 |
| 17.0.3 | [~] | year-to-date progress | engine/timeline.py::_reject_dates | 2 | 6 |
| 17.0.4 | [~] | weekly performance trajectory | engine/timeline.py | 2 | 6 |
| 17.0.5 | [x] | exposure | engine/timeline.py<br>engine/pit.py::restatement_exposure | 3 | 7 |
| 17.0.6 | [!] | k | engine/timeline.py::DialAdmission<br>scripts/dial_offline.py::DialSession | 2 | 6 |
| 17.0.7 | [x] | pool_q | engine/timeline.py<br>engine/analogs_stock.py::PooledStockAnalogs | 3 | 9 |
| 17.0.8 | [x] | brake | engine/timeline.py | 2 | 6 |
| 17.0.9 | [!] | improvement | engine/timeline.py<br>engine/antimemo.py::improvement_by_era | 6 | 12 |
| 17.0.10 | [!] | no unacceptable risk deterioration | engine/timeline.py::promotion_gate | 1 | 1 |
| 17.0.11 | [~] | no overfitting | engine/timeline.py | 2 | 6 |
| 17.0.12 | [!] | stability across eras | engine/timeline.py<br>engine/heavy_tests.py::ablation_across_eras | 4 | 6 |

### Phase 18: WEEKLY ADAPTER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 18.0.1 | [x] | w_model | engine/adaptive.py<br>engine/missed_winners.py | 11 | 28 |
| 18.0.2 | [x] | liq_q | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 24 |
| 18.0.3 | [x] | k | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 24 |
| 18.0.4 | [x] | pool_q | engine/analogs_stock.py::PooledStockAnalogs | 1 | 3 |
| 18.0.5 | [x] | w_move | engine/adaptive.py<br>engine/memory_diagnostics.py | 11 | 9 |
| 18.0.6 | [x] | w_mom | engine/adaptive.py::Adapter<br>engine/memory.py | 10 | 24 |
| 18.0.7 | [x] | base | engine/missed_winners.py::_base_score<br>scripts/run_b15_memory_adapter.py | 2 | 6 |
| 18.0.8 | [?] | neighbor | engine/adaptive.py::KnobState | 1 | 0 |
| 18.0.9 | [?] | evidence | engine/adaptive.py::KnobState | 1 | 0 |
| 18.0.10 | [?] | effective sample | engine/adaptive.py::KnobState | 1 | 0 |
| 18.0.11 | [x] | confidence | engine/analog_weighting.py::confidence_calibration<br>engine/heavy_tests.py::era_confidence | 3 | 1 |
| 18.0.12 | [!] | improvement | engine/adaptive.py<br>engine/antimemo.py::improvement_by_era | 11 | 11 |
| 18.0.13 | [x] | standard error | engine/adaptive.py::GuardRailError<br>engine/memory.py::classify_error | 12 | 28 |
| 18.0.14 | [x] | cooldown | engine/adaptive.py | 8 | 7 |
| 18.0.15 | [x] | last switch | engine/adaptive.py<br>scripts/run_b15_memory_adapter.py::weekly_last_sessions | 8 | 7 |
| 18.0.16 | [x] | revert state | engine/adaptive.py::KnobState<br>engine/analog_weighting.py::price_state | 9 | 8 |
| 18.0.17 | [?] | minimum evidence | engine/adaptive.py::Adapter._global_gate | 1 | 0 |
| 18.0.18 | [x] | cooldown | engine/adaptive.py | 8 | 7 |
| 18.0.19 | [?] | statistically meaningful improvement | engine/adaptive.py::Adapter._evaluate_candidates | 1 | 0 |
| 18.0.20 | [x] | one-step movement | engine/adaptive.py::one_step<br>engine/memory_diagnostics.py | 11 | 10 |
| 18.0.21 | [?] | rapid revert when deterioration occurs | engine/adaptive.py::Adapter._learn | 1 | 0 |

### Phase 19: TRAIN THE TRAINING BASIS

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 19.0.1 | [x] | 24 random starting configurations | engine/basis_search.py<br>scripts/livesim_loop2.py | 4 | 5 |
| 19.0.2 | [x] | 10 random archived screening windows | engine/basis_search.py::BasisSearch | 1 | 1 |
| 19.0.3 | [x] | top 3 candidates | engine/basis_search.py::Candidate<br>engine/objective.py | 13 | 12 |
| 19.0.4 | [~] | confirmation across all archived windows | engine/antimemo.py::disguise_window<br>scripts/livesim_loop2.py::as_search_window | 3 | 6 |
| 19.0.5 | [x] | tiered objective | engine/basis_search.py<br>engine/objective.py::TierScore | 6 | 8 |
| 19.0.6 | [x] | next-basis adoption | engine/basis_search.py::BasisHistory | 2 | 5 |
| 19.0.7 | [x] | half-life | engine/memory.py::half_life | 4 | 18 |
| 19.0.8 | [x] | prior weeks | engine/exits.py::WeekTable<br>engine/missed_winners.py::prior_correct | 15 | 32 |
| 19.0.9 | [x] | switch threshold | engine/fv_pipeline.py::certify_threshold | 1 | 0 |
| 19.0.10 | [~] | minimum weeks | engine/blind_gates.py::is_week_end<br>engine/fv_pipeline.py::mover_weekly | 7 | 6 |
| 19.0.11 | [?] | cooldown | engine/adaptive.py::META_DEFAULT<br>engine/basis_search.py::META_SPACE | 1 | 0 |
| 19.0.12 | [?] | revert drop | engine/adaptive.py::META_DEFAULT<br>engine/basis_search.py::META_SPACE | 1 | 0 |
| 19.0.13 | [?] | IC beta | engine/adaptive.py::META_DEFAULT<br>engine/basis_search.py::META_SPACE | 1 | 0 |
| 19.0.14 | [!] | detector max | engine/adaptive.py::detector_scores | 8 | 7 |
| 19.0.15 | [x] | detector minimum weeks | engine/adaptive.py::detector_scores<br>scripts/run_b15_memory_adapter.py::build_weeks | 8 | 8 |
| 19.0.16 | [x] | memory half-life | engine/memory.py::half_life<br>engine/memory_diagnostics.py::memory_health | 4 | 18 |
| 19.0.17 | [?] | bandwidth | engine/memory.py::MEM_DEFAULT<br>engine/basis_search.py::META_SPACE | 1 | 0 |
| 19.0.18 | [x] | prior scale | engine/memory.py::_fit_scale<br>engine/missed_winners.py::prior_correct | 5 | 21 |
| 19.0.19 | [?] | shrinkage | engine/adaptive.py::META_DEFAULT<br>engine/memory.py::MEM_DEFAULT | 1 | 0 |
| 19.0.20 | [~] | shock parameters | engine/memory.py::_shock_update<br>engine/memory_diagnostics.py::shock_report | 4 | 18 |

### Phase 20: TIERED OBJECTIVE FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 20.TIER-1.1 | [x] | most weeks approximately 5–10% | engine/exits.py::WeekTable<br>engine/objective.py::TierScore | 7 | 24 |
| 20.TIER-1.2 | [x] | yearly average move near 7% | engine/objective.py::TierScore<br>engine/timeline.py::expected_abs_move | 4 | 5 |
| 20.TIER-2.1 | [x] | worst 5% week | engine/objective.py::TierScore<br>engine/stops.py | 7 | 24 |
| 20.TIER-2.2 | [~] | max drawdown | engine/objective.py::TierScore<br>engine/run_report.py::max_drawdown | 7 | 5 |
| 20.TIER-2.3 | [!] | weeks outside band | scripts/timeline_basis_report.py::load_weekly<br>engine/exits.py::WeekTable | 8 | 27 |
| 20.TIER-2.4 | [x] | catastrophic losses | engine/objective.py::TierScore<br>engine/stops.py::loss_risk | 6 | 24 |
| 20.TIER-3.1 | [!] | positive in-band weeks | engine/exits.py::WeekTable<br>engine/objective.py::TierScore | 8 | 27 |
| 20.TIER-3.2 | [~] | successful +10% outcomes | engine/objective.py::TierScore | 4 | 5 |
| 20.TIER-3.3 | [~] | directional precision | engine/objective.py::TierScore | 4 | 5 |

### Phase 21: BLIND SIMULATOR HARDENING

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 21.SEALED-WINDOW.1 | [x] | Sealed window | engine/blind_gates.py::seal_window | 1 | 1 |
| 21.DISGUISE.1 | [x] | Disguise | engine/blind_gates.py::check_disguise_signature<br>engine/livesim.py::_disguise | 6 | 10 |
| 21.WARM-UP.1 | [~] | Warm-up | engine/blind_gates.py<br>engine/livesim.py | 3 | 6 |
| 21.REVEAL.1 | [x] | adjustments locked | engine/blind_gates.py::lock_adjustments<br>engine/livesim.py::reveal | 3 | 6 |
| 21.REVEAL.2 | [~] | all predictions recorded | engine/blind_gates.py::RevealGate | 3 | 6 |
| 21.REVEAL.3 | [~] | all trades completed | engine/blind_gates.py::RevealGate<br>engine/livesim.py::BlindTrader | 3 | 6 |
| 21.REVEAL.4 | [~] | all learning decisions finalized | engine/blind_gates.py::RevealGate<br>engine/livesim.py::reveal | 3 | 6 |

### Phase 23: RE-TESTER

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 23.0.1 | [!] | holdings | engine/blind_gates.py<br>engine/retester.py::cmp_holdings | 12 | 9 |
| 23.0.2 | [!] | trades | engine/blind_gates.py<br>engine/retester.py::cmp_trades | 12 | 9 |
| 23.0.3 | [x] | scores | engine/retester.py::cmp_scores<br>engine/ablation.py::score | 10 | 15 |
| 23.0.4 | [x] | weekly returns | engine/blind_gates.py::is_week_end<br>engine/retester.py::cmp_weekly_returns | 12 | 9 |
| 23.0.5 | [x] | adaptation events | engine/retester.py::cmp_events<br>engine/run_report.py::memory_shock_events | 3 | 1 |
| 23.0.6 | [~] | pattern activation | engine/retester.py::cmp_pattern_activation | 1 | 1 |
| 23.0.7 | [x] | memory state | engine/blind_gates.py::check_memory_bank_causality<br>engine/retester.py::cmp_memory | 14 | 26 |

### Phase 24: WORKER HEALTH

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 24.0.1 | [x] | start | engine/blind_gates.py<br>engine/health.py | 5 | 6 |
| 24.0.2 | [x] | configuration | engine/adaptive.py::audit_matches_cfg<br>engine/timeline.py::cfg_overrides | 14 | 11 |
| 24.0.3 | [?] | window | engine/health.py::window_breakdown | 1 | 0 |
| 24.0.4 | [x] | seed | engine/blind_gates.py<br>engine/health.py | 5 | 6 |
| 24.0.5 | [x] | memory | engine/blind_gates.py::check_memory_bank_causality<br>engine/health.py::memory_growth_mb_per_min | 8 | 6 |
| 24.0.6 | [x] | completion | engine/blind_gates.py<br>engine/health.py | 5 | 6 |
| 24.0.7 | [x] | crash | engine/health.py<br>engine/retester.py | 3 | 3 |
| 24.0.8 | [x] | timeout | engine/health.py<br>engine/pattern_bank.py::BankLockTimeout | 4 | 17 |
| 24.0.9 | [x] | OOM | engine/health.py::WorkerLog<br>scripts/blind_gates_real.py::section_health | 5 | 4 |
| 24.0.10 | [x] | invalid result | engine/health.py::result_fingerprint<br>engine/basis_search.py::SearchResult | 6 | 25 |

### Phase 25: PLANTED-PATTERN CALIBRATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 25.0.1 | [x] | detection rate | engine/planted.py::summarise | 1 | 1 |
| 25.0.2 | [!] | false discovery rate | engine/planted.py::summarise | 1 | 1 |
| 25.0.3 | [x] | P(real) calibration | engine/planted.py::calibration_table<br>engine/missed_winners.py::calibration | 7 | 15 |
| 25.0.4 | [x] | rejection rate | engine/planted.py::summarise | 1 | 1 |
| 25.0.5 | [?] | recovery rate | engine/planted.py::summarise | 1 | 0 |
| 25.0.6 | [x] | rescoping behavior | engine/pattern_lifecycle.py::Lifecycle.scope_null_rate | 1 | 1 |

### Phase 26: ANTI-OVERFITTING BATTERY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 26.A.1 | [~] | A. Future scramble: Future data must not affect earlier decisions | engine/ablation.py | 1 | 6 |
| 26.B.1 | [!] | B. Label permutation: Random labels should destroy apparent predictive performance | engine/antioverfit.py::label_permutation | 2 | 3 |
| 26.C.1 | [!] | C. Ticker permutation: Ticker identities should not create fake signal | engine/antioverfit.py::ticker_permutation | 2 | 3 |
| 26.D.1 | [x] | D. Date disguise: Changing absolute dates while preserving structure should preserve behavior | engine/antioverfit.py::date_disguise | 1 | 1 |
| 26.E.1 | [~] | E. Randomized outcomes: Pattern discovery should collapse toward chance | engine/antioverfit.py::randomized_outcomes | 2 | 3 |
| 26.F.1 | [!] | F. Feature shuffle: Important features should lose signal when shuffled | engine/antioverfit.py::feature_shuffle | 2 | 3 |
| 26.G.1 | [x] | G. Dead-feature injection: Adding random features must not create stable predictive power | engine/antioverfit.py::dead_feature_injection | 1 | 1 |
| 26.H.1 | [x] | H. Duplicate-feature injection: Duplicates must not double-count evidence | engine/antioverfit.py::duplicate_feature_injection | 1 | 1 |
| 26.I.1 | [x] | I. Regime split: Signal must be evaluated independently across regimes | engine/antioverfit.py::regime_split | 1 | 1 |
| 26.J.1 | [x] | J. Walk-forward split: No future observations in training | engine/antioverfit.py::walk_forward_splits | 1 | 1 |

### Phase 27: DATA EXPANSION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 27.0.1 | [?] | schema validation | engine/data_sources.py::validate_prices | 1 | 0 |
| 27.0.2 | [x] | date validation | engine/data_sources.py::ValidationReport<br>engine/antioverfit.py::date_disguise | 6 | 5 |
| 27.0.3 | [x] | source provenance | engine/data_sources.py::ProvenanceLedger<br>scripts/data_live_audit.py | 4 | 24 |
| 27.0.4 | [~] | missing-data handling | engine/data_sources.py | 1 | 2 |
| 27.0.5 | [x] | duplicate detection | engine/data_sources.py::detect_splits<br>engine/pattern_lifecycle.py::detect_failure | 3 | 3 |
| 27.0.6 | [?] | point-in-time validation | engine/data_sources.py::point_in_time_filter | 1 | 0 |
| 27.0.7 | [~] | parity tests where applicable | engine/data_sources.py::parity | 1 | 2 |

### Phase 28: LIVE/RESEARCH SEPARATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 28.0.1 | [x] | can simulate | engine/portfolio.py::simulate<br>engine/replay.py::simulate | 2 | 8 |
| 28.0.2 | [x] | cannot submit broker orders | engine/champion.py::order<br>engine/isolation.py::submit_order | 6 | 5 |
| 28.0.3 | [?] | uses validated configuration | engine/adaptive.py::validate_cfg<br>engine/isolation.py::verify_activation | 2 | 0 |
| 28.0.4 | [~] | requires explicit activation | engine/isolation.py::verify_activation | 2 | 1 |
| 28.0.5 | [x] | uses current data only | engine/data_sources.py<br>engine/isolation.py::data_is_current | 3 | 2 |
| 28.0.6 | [~] | respects trading hours | engine/isolation.py::audit_hours_firewall | 2 | 1 |
| 28.0.7 | [?] | respects broker constraints | engine/isolation.py::check_order | 1 | 0 |

### Phase 29: PUBLIC EXPLANATION SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 29.DASHBOARD.1 | [~] | Dashboard | engine/site_data.py::write_site | 0 | 0 |
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
| 29.PATTERN-EXPLORER.11 | [?] | validation evidence | scripts/site_pages.py::_evidence_html | 1 | 0 |
| 29.SENSITIVITY-PAGE.1 | [x] | parameter | scripts/site_audit.py::audit_page<br>scripts/site_pages.py | 1 | 1 |
| 29.SENSITIVITY-PAGE.2 | [x] | tested values | scripts/site_sources.py::build_sensitivity<br>scripts/site_render.py::page | 1 | 0 |
| 29.SENSITIVITY-PAGE.3 | [x] | effect | scripts/site_sources.py::build_sensitivity<br>scripts/site_audit.py::audit_page | 1 | 0 |
| 29.SENSITIVITY-PAGE.4 | [~] | uncertainty | scripts/site_pages.py<br>scripts/site_audit.py::audit_page | 0 | 0 |
| 29.SENSITIVITY-PAGE.5 | [~] | stability | engine/exits.py::cost_sensitivity<br>scripts/site_audit.py::audit_page | 3 | 19 |
| 29.SENSITIVITY-PAGE.6 | [x] | selected value | scripts/site_charts.py<br>scripts/site_render.py::page | 4 | 19 |
| 29.SENSITIVITY-PAGE.7 | [~] | reason selected | engine/exits.py::Selection<br>scripts/site_render.py::page | 3 | 19 |

### Phase 30: EXPERIMENT MEMORY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 30.WHAT-CHANGED.1 | [~] | What changed? | engine/checkpoint.py<br>engine/registry.py | 8 | 8 |
| 30.WHY-DID.1 | [~] | Why did it change? | engine/checkpoint.py<br>engine/registry.py | 8 | 8 |
| 30.WHAT-DATA.1 | [x] | What data was used? | engine/registry.py<br>engine/run_report.py | 6 | 5 |
| 30.WHAT-WAS.1 | [?] | What was unseen? | engine/registry.py::QUESTIONS | 2 | 0 |
| 30.WHAT-WAS2.1 | [x] | What was the baseline? | engine/champion.py::freeze_baseline<br>engine/claims.py | 5 | 2 |
| 30.WHAT-IMPROVED.1 | [x] | What improved? | engine/champion.py<br>engine/experiment_memory.py | 9 | 10 |
| 30.WHAT-WORSENED.1 | [?] | What worsened? | engine/registry.py::QUESTIONS | 1 | 0 |
| 30.WAS-IMPROVEMENT.1 | [?] | Was improvement statistically meaningful? | engine/registry.py::QUESTIONS | 1 | 0 |
| 30.DID-RISK.1 | [~] | Did risk change? | engine/champion.py::_check_risk | 1 | 2 |
| 30.DID-THE.1 | [?] | Did the improvement survive another window? | engine/registry.py::QUESTIONS | 1 | 0 |
| 30.WAS-IT.1 | [x] | Was it adopted? | engine/claims.py<br>engine/experiment_memory.py | 7 | 5 |
| 30.IF-REJECTED.1 | [~] | If rejected, why?: Never let an experiment become an orphan | engine/registry.py::ExperimentMemory<br>engine/resources.py | 8 | 2 |

### Phase 31: CODE QUALITY FIREWALL

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 31.0.1 | [ ] | type-safe interfaces where appropriate | - | 0 | 0 |
| 31.0.2 | [ ] | explicit inputs | - | 0 | 0 |
| 31.0.3 | [~] | explicit outputs | scripts/quality_gate.py<br>scripts/bible_trace.py::write_outputs | 3 | 0 |
| 31.0.4 | [?] | deterministic behavior | scripts/quality_gate.py::check_randomness | 1 | 0 |
| 31.0.5 | [ ] | logging | - | 0 | 0 |
| 31.0.6 | [~] | error handling | engine/livesim.py::BlindGateError<br>scripts/site_audit.py::handle_data | 3 | 5 |
| 31.0.7 | [~] | unit tests | scripts/quality_gate.py<br>scripts/test_inventory.py | 2 | 0 |
| 31.0.8 | [~] | integration tests | scripts/test_inventory.py<br>scripts/bible_trace.py::integration_pending | 3 | 0 |
| 31.0.9 | [?] | documentation | scripts/quality_gate.py::check_module_docstrings | 1 | 0 |
| 31.0.10 | [~] | no hidden global state | scripts/quality_gate.py::check_hidden_global_state | 2 | 0 |
| 31.0.11 | [~] | no accidental randomness | scripts/quality_gate.py::check_randomness | 2 | 0 |

### Phase 32: TEST PYRAMID

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 32.L1.1 | [~] | Level 1 — Unit: Test mathematical correctness | scripts/test_inventory.py::_level_words | 1 | 0 |
| 32.L2.1 | [~] | Level 2 — Integration: Test interaction with neighboring systems | scripts/test_inventory.py::_level_words | 1 | 0 |
| 32.L3.1 | [~] | Level 3 — Historical: Test on known historical data | scripts/quality_gate.py::check_tests_touch_data<br>engine/analog_weighting.py::level_agreement | 7 | 6 |
| 32.L4.1 | [x] | Level 4 — Walk-forward: Test chronologically | engine/antioverfit.py::walk_forward_splits | 1 | 1 |
| 32.L5.1 | [~] | Level 5 — Blind: Test disguised hidden windows | engine/blind_gates.py::BlindClock<br>engine/livesim.py::BlindGateError | 3 | 6 |
| 32.L6.1 | [~] | Level 6 — Adversarial: Try to make it leak | engine/blind_gates.py::make_ticker_map<br>scripts/test_inventory.py::_level_words | 3 | 6 |
| 32.L7.1 | [~] | Level 7 — Reproducibility: Run twice and compare | engine/repro.py::run_twice<br>engine/direction_calib.py::compare_gates | 6 | 4 |

### Phase 33: REQUIRED REPRODUCIBILITY

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 33.0.1 | [x] | same configuration hash | engine/repro.py::config_hash | 1 | 1 |
| 33.0.2 | [x] | same seed | engine/repro.py::config_hash | 1 | 1 |
| 33.0.3 | [x] | same candidate ordering | engine/candidates.py::CandidateGenerator | 1 | 1 |
| 33.0.4 | [~] | same model configuration | engine/repro.py<br>engine/adaptive.py::audit_matches_cfg | 13 | 8 |
| 33.0.5 | [x] | same predictions | engine/repro.py::run_twice | 1 | 1 |
| 33.0.6 | [x] | same trades | engine/adaptive.py::replay_check | 2 | 1 |
| 33.0.7 | [x] | same metrics | engine/repro.py::diagnose | 1 | 1 |

### Phase 34: REQUIRED ABLATION FRAMEWORK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 34.0.1 | [x] | baseline | engine/ablation.py<br>engine/analog_weighting.py::baseline | 4 | 8 |
| 34.0.2 | [x] | baseline + feature | engine/ablation.py::feature_ablation<br>engine/analog_weighting.py::baseline | 5 | 7 |
| 34.0.3 | [~] | feature alone | engine/ablation.py::feature_ablation<br>engine/pattern_movers.py::attribute_by_feature | 6 | 14 |
| 34.0.4 | [~] | randomized feature | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::dead_feature_injection | 2 | 8 |
| 34.0.5 | [!] | shuffled feature | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::feature_shuffle | 5 | 10 |
| 34.0.6 | [x] | future-scrambled feature where applicable | engine/ablation.py::feature_ablation<br>engine/antioverfit.py::future_scramble | 1 | 6 |

### Phase 35: CHAMPION / CHALLENGER SYSTEM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 35.0.1 | [x] | Champion | engine/champion.py::ChampionError<br>engine/timeline.py::champion | 5 | 8 |
| 35.0.2 | [x] | Challenger | engine/champion.py::ChallengerQueue<br>engine/improve.py::spawn_challengers | 4 | 6 |
| 35.0.3 | [x] | Candidate | engine/champion.py::add_candidate<br>engine/adaptive.py::_evaluate_candidates | 12 | 14 |
| 35.0.4 | [x] | Rejected | engine/champion.py::reject<br>engine/run_report.py | 6 | 11 |
| 35.0.5 | [~] | required gates pass | engine/champion.py::_check_gates<br>engine/livesim.py::BlindGateError | 6 | 9 |
| 35.0.6 | [~] | enough independent evidence exists | scripts/bible_trace.py::EvidenceFile<br>engine/champion.py::_check_evidence | 2 | 0 |
| 35.0.7 | [?] | higher-priority objective is not harmed | engine/champion.py::_check_priority | 1 | 0 |
| 35.0.8 | [?] | risk does not violate constraints | engine/champion.py::_check_risk | 1 | 0 |
| 35.0.9 | [~] | reproducibility passes | scripts/final_report.py::s_reproducibility<br>engine/registry.py::reproducibility_conflicts | 2 | 0 |
| 35.0.10 | [~] | blind validation passes | engine/champion.py::_check_blind | 1 | 2 |

### Phase 36: REQUIRED REPORT AFTER EACH MAJOR RUN

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 36.TIER-1.1 | [x] | TIER 1: weekly average move | engine/objective.py::TierScore<br>engine/heavy_tests.py | 4 | 5 |
| 36.TIER-1.2 | [x] | TIER 1: weekly median move | engine/heavy_tests.py::vs_past_median<br>engine/gaprisk.py::_week_regime | 8 | 16 |
| 36.TIER-1.3 | [x] | TIER 1: share 5–10% | engine/run_report.py<br>engine/objective.py::TierScore | 6 | 5 |
| 36.TIER-1.4 | [x] | TIER 1: share >10% | engine/run_report.py<br>engine/objective.py::TierScore | 6 | 5 |
| 36.TIER-1.5 | [x] | TIER 1: share <5% | engine/run_report.py<br>engine/objective.py::TierScore | 6 | 5 |
| 36.TIER-2.1 | [~] | TIER 2: max drawdown | engine/run_report.py::max_drawdown<br>engine/objective.py::TierScore | 7 | 5 |
| 36.TIER-2.2 | [x] | TIER 2: worst week | engine/objective.py::TierScore<br>scripts/livesim_loop2.py::tiered | 5 | 5 |
| 36.TIER-2.3 | [~] | TIER 2: 5th percentile week | engine/stops.py::VolPercentileStop | 2 | 19 |
| 36.TIER-2.4 | [x] | TIER 2: catastrophic losses | engine/objective.py::TierScore<br>engine/stops.py::loss_risk | 6 | 24 |
| 36.TIER-2.5 | [~] | TIER 2: turnover | engine/run_report.py | 2 | 0 |
| 36.TIER-2.6 | [x] | TIER 2: costs | engine/exits.py::CostModel<br>engine/claims.py::CostAdjustment | 3 | 19 |
| 36.TIER-3.1 | [x] | TIER 3: positive in-band percentage | engine/run_report.py<br>engine/exits.py::band_metrics | 9 | 24 |
| 36.TIER-3.2 | [?] | TIER 3: direction accuracy | engine/run_report.py::tier3 | 1 | 0 |
| 36.TIER-3.3 | [?] | TIER 3: calibration | engine/run_report.py::calibration | 1 | 0 |
| 36.TIER-3.4 | [~] | TIER 3: 8/10 rate | engine/run_report.py | 2 | 0 |
| 36.ALGORITHM.1 | [~] | ALGORITHM: pattern count | engine/heavy_tests.py::_oos_pattern_check<br>scripts/site_sources.py::_pattern_from_row | 3 | 0 |
| 36.ALGORITHM.2 | [x] | ALGORITHM: active pattern count | engine/pattern_bank.py::PatternBank<br>engine/pattern_lifecycle.py::counts | 6 | 15 |
| 36.ALGORITHM.3 | [~] | ALGORITHM: failed patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 4 | 0 |
| 36.ALGORITHM.4 | [~] | ALGORITHM: rescoped patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 4 | 0 |
| 36.ALGORITHM.5 | [~] | ALGORITHM: discarded patterns | engine/heavy_tests.py::_oos_pattern_check<br>engine/run_report.py::algorithm_section | 4 | 0 |
| 36.ALGORITHM.6 | [~] | ALGORITHM: P(real) distribution | engine/heavy_tests.py<br>engine/run_report.py::algorithm_section | 2 | 0 |
| 36.ALGORITHM.7 | [~] | ALGORITHM: false discovery diagnostics | engine/heavy_tests.py::false_discovery_summary<br>engine/run_report.py::algorithm_section | 4 | 0 |
| 36.ANALOGS.1 | [x] | ANALOGS: analog count | engine/analog_weighting.py<br>engine/analogs_sector.py::SectorAnalogs | 3 | 9 |
| 36.ANALOGS.2 | [?] | ANALOGS: distance distribution | engine/run_report.py::analogs_section | 1 | 0 |
| 36.ANALOGS.3 | [~] | ANALOGS: forecast quality | engine/run_report.py::analog_distance_distribution | 2 | 0 |
| 36.MEMORY.1 | [x] | MEMORY: lesson count | engine/registry.py::ExperimentMemory<br>engine/lessons.py::Lesson | 11 | 41 |
| 36.MEMORY.2 | [~] | MEMORY: accepted lessons | engine/blind_gates.py::check_memory_bank_causality<br>engine/registry.py::ExperimentMemory | 3 | 6 |
| 36.MEMORY.3 | [~] | MEMORY: rejected lessons | engine/champion.py::reject<br>engine/run_report.py::memory_shock_events | 6 | 4 |
| 36.MEMORY.4 | [?] | MEMORY: shock events | engine/run_report.py::build_report | 1 | 0 |
| 36.MEMORY.5 | [~] | MEMORY: memory switches | engine/memory_diagnostics.py::memory_health<br>engine/registry.py::ExperimentMemory | 1 | 1 |
| 36.ADAPTATION.1 | [~] | ADAPTATION: parameter changes | engine/adaptive.py::adaptation_report<br>engine/basis_search.py::changed_keys | 9 | 8 |
| 36.ADAPTATION.2 | [x] | ADAPTATION: reverts | engine/adaptive.py::adaptation_report | 8 | 7 |
| 36.ADAPTATION.3 | [?] | ADAPTATION: evidence | engine/adaptive.py::adaptation_report | 1 | 0 |
| 36.ADAPTATION.4 | [~] | ADAPTATION: confidence | engine/run_report.py<br>engine/heavy_tests.py::era_confidence | 2 | 0 |
| 36.GATES.1 | [x] | GATES: parity | engine/data_sources.py::parity<br>engine/livesim.py::BlindGateError | 5 | 8 |
| 36.GATES.2 | [!] | GATES: retester | scripts/blind_gates_real.py::section_retest<br>scripts/check_retester.py | 0 | 1 |
| 36.GATES.3 | [x] | GATES: future scramble | engine/antioverfit.py::future_scramble<br>engine/blind_gates.py::RevealGate | 6 | 10 |
| 36.GATES.4 | [x] | GATES: time fence | engine/adaptive.py::TimeFence<br>engine/pit.py::fence | 9 | 8 |
| 36.GATES.5 | [~] | GATES: worker health | scripts/blind_gates_real.py::section_health<br>scripts/livesim_loop2.py::reset_worker_files | 3 | 3 |
| 36.GATES.6 | [~] | GATES: reproducibility | scripts/final_report.py::s_reproducibility<br>engine/registry.py::reproducibility_conflicts | 2 | 0 |

### Phase 38: MASTER CHECKLIST

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 38.ALGORITHM.1 | [?] | ALGORITHM: Candles and micro-signals | engine/candidates.py<br>engine/planted.py | 2 | 2 |
| 38.ALGORITHM.2 | [~] | ALGORITHM: Pattern miner | engine/pattern_stats.py<br>engine/candidates.py | 4 | 4 |
| 38.ALGORITHM.3 | [~] | ALGORITHM: Week-clustered statistics | engine/pattern_stats.py<br>engine/pattern_identity.py | 2 | 2 |
| 38.ALGORITHM.4 | [~] | ALGORITHM: Heavy tests across eras | engine/heavy_tests.py | 5 | 5 |
| 38.ALGORITHM.5 | [!] | ALGORITHM: Long-term pattern bank | engine/pattern_bank.py<br>engine/pattern_lifecycle.py | 5 | 5 |
| 38.ALGORITHM.6 | [?] | ALGORITHM: Pattern scores → Find Volatility | engine/heavy_tests.py<br>engine/pattern_movers.py | 3 | 4 |
| 38.ALGORITHM.7 | [?] | ALGORITHM: Minute collector | engine/data_sources.py | 2 | 3 |
| 38.ALGORITHM.8 | [?] | ALGORITHM: Self-tuning Algorithm | engine/adaptive.py<br>engine/basis_search.py | 5 | 5 |
| 38.ALGORITHM.9 | [~] | ALGORITHM: Market analog engine | engine/analog_weighting.py<br>engine/analogs_sector.py | 5 | 5 |
| 38.ALGORITHM.10 | [x] | ALGORITHM: Sector analog engine | engine/analog_weighting.py | 1 | 1 |
| 38.ALGORITHM.11 | [x] | ALGORITHM: Stock analog engine | engine/analogs_stock.py | 1 | 1 |
| 38.ALGORITHM.12 | [!] | ALGORITHM: Timeline dial | engine/timeline.py | 3 | 5 |
| 38.ALGORITHM.13 | [~] | ALGORITHM: Lesson memory | engine/memory.py<br>engine/lessons.py | 5 | 5 |
| 38.ALGORITHM.14 | [!] | ALGORITHM: Rerun anti-memorization test | engine/antimemo.py<br>engine/lessons.py | 2 | 4 |
| 38.ALGORITHM.15 | [?] | ALGORITHM: Improve-or-discard lifecycle | engine/pattern_lifecycle.py<br>engine/pattern_bank.py | 5 | 4 |
| 38.ALGORITHM.16 | [?] | ALGORITHM: Additional data sources | engine/data_sources.py | 2 | 3 |
| 38.ALGORITHM.17 | [!] | ALGORITHM: Planted-pattern calibration | engine/planted.py<br>engine/pattern_lifecycle.py | 3 | 2 |
| 38.FIND-VOLATILITY.1 | [?] | FIND VOLATILITY: 95% mover target where sufficient candidates exist | engine/direction.py<br>engine/heavy_tests.py | 5 | 5 |
| 38.FIND-VOLATILITY.2 | [?] | FIND VOLATILITY: Direction ≥80% calibrated confidence | engine/direction.py<br>engine/direction_ablate.py | 5 | 5 |
| 38.FIND-VOLATILITY.3 | [?] | FIND VOLATILITY: Per-stock-type trust tables | engine/trust.py<br>engine/direction.py | 5 | 4 |
| 38.FIND-VOLATILITY.4 | [?] | FIND VOLATILITY: Exit learner | engine/exits.py<br>engine/retester.py | 5 | 3 |
| 38.FIND-VOLATILITY.5 | [~] | FIND VOLATILITY: Stop/risk learner | engine/stops.py<br>engine/gaprisk.py | 2 | 1 |
| 38.FIND-VOLATILITY.6 | [~] | FIND VOLATILITY: 8-of-10 +10% objective across blind eras | engine/heavy_tests.py<br>engine/objective.py | 5 | 5 |
| 38.TEST.1 | [~] | TEST: Archive | engine/blind_gates.py | 2 | 1 |
| 38.TEST.2 | [~] | TEST: Blind loop | engine/blind_gates.py | 1 | 1 |
| 38.TEST.3 | [!] | TEST: Tiered 7% objective | engine/objective.py<br>engine/exits.py | 3 | 2 |
| 38.TEST.4 | [?] | TEST: Weekly self-adjustment | engine/adaptive.py<br>engine/analog_weighting.py | 3 | 5 |
| 38.TEST.5 | [?] | TEST: Insider leak protection | engine/pit.py<br>engine/blind_gates.py | 5 | 5 |
| 38.TEST.6 | [?] | TEST: 13D correctness | engine/data_sources.py | 2 | 3 |
| 38.TEST.7 | [~] | TEST: Explorer | scripts/site_pages.py<br>scripts/site_sources.py | 1 | 0 |
| 38.TEST.8 | [~] | TEST: Sensitivity | engine/exits.py<br>scripts/site_sources.py | 2 | 2 |
| 38.TEST.9 | [!] | TEST: Re-tester parity | engine/blind_gates.py<br>engine/retester.py | 3 | 3 |
| 38.TEST.10 | [x] | TEST: Future scramble | engine/pit.py<br>engine/ablation.py | 4 | 4 |
| 38.TEST.11 | [~] | TEST: Time fence | engine/pit.py | 1 | 1 |
| 38.TEST.12 | [x] | TEST: Worker health | engine/health.py<br>engine/blind_gates.py | 5 | 3 |
| 38.TEST.13 | [x] | TEST: Reproducibility | engine/repro.py<br>engine/adaptive.py | 4 | 1 |
| 38.TEST.14 | [!] | TEST: Label permutation | engine/antioverfit.py | 1 | 1 |
| 38.TEST.15 | [!] | TEST: Feature shuffle | engine/antioverfit.py | 1 | 1 |
| 38.TEST.16 | [!] | TEST: Ticker permutation | engine/antioverfit.py | 1 | 1 |
| 38.TEST.17 | [!] | TEST: Planted-pattern calibration | engine/planted.py<br>engine/pattern_lifecycle.py | 3 | 2 |
| 38.LIVE.1 | [?] | LIVE: Only consider upgrade after validated research edge | engine/isolation.py<br>engine/adaptive.py | 5 | 2 |
| 38.LIVE.2 | [?] | LIVE: Verify paper-only safeguards | engine/isolation.py<br>engine/adaptive.py | 5 | 2 |
| 38.LIVE.3 | [?] | LIVE: Verify trading-hours firewall | engine/isolation.py<br>engine/adaptive.py | 5 | 2 |
| 38.LIVE.4 | [?] | LIVE: Verify broker safety | engine/isolation.py<br>engine/adaptive.py | 5 | 2 |
| 38.LIVE.5 | [?] | LIVE: Verify research/live isolation | engine/isolation.py<br>engine/adaptive.py | 5 | 2 |

### Phase 39: DEFINITION OF DONE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 39.0.1 | [~] | every required module exists | engine/adaptive.py<br>engine/heavy_tests.py | 5 | 5 |
| 39.0.2 | [~] | every required module is tested | engine/adaptive.py<br>engine/heavy_tests.py | 5 | 5 |
| 39.0.3 | [?] | every data source is validated | engine/data_sources.py | 2 | 3 |
| 39.0.4 | [?] | every point-in-time rule is enforced | engine/pit.py<br>engine/livesim.py | 5 | 5 |
| 39.0.5 | [~] | blind simulation works | engine/blind_gates.py | 2 | 1 |
| 39.0.6 | [!] | re-tester works | engine/blind_gates.py<br>engine/retester.py | 3 | 3 |
| 39.0.7 | [~] | parity works | engine/parity.py<br>engine/adaptive.py | 4 | 3 |
| 39.0.8 | [x] | future scramble works | engine/pit.py<br>engine/ablation.py | 4 | 4 |
| 39.0.9 | [~] | time fence works | engine/pit.py | 1 | 1 |
| 39.0.10 | [x] | worker health works | engine/health.py<br>engine/blind_gates.py | 5 | 3 |
| 39.0.11 | [x] | deterministic replay works | engine/repro.py<br>engine/adaptive.py | 4 | 1 |
| 39.0.12 | [~] | pattern discovery works | engine/pattern_stats.py<br>engine/candidates.py | 4 | 4 |
| 39.0.13 | [~] | pattern validation works | engine/pattern_stats.py<br>engine/pattern_identity.py | 2 | 3 |
| 39.0.14 | [?] | pattern lifecycle works | engine/pattern_lifecycle.py<br>engine/pattern_bank.py | 5 | 4 |
| 39.0.15 | [~] | analog engine works | engine/analog_weighting.py<br>engine/analogs_sector.py | 5 | 5 |
| 39.0.16 | [~] | memory works | engine/memory.py<br>engine/lessons.py | 5 | 5 |
| 39.0.17 | [?] | lesson learning works | engine/lessons.py<br>engine/antimemo.py | 5 | 5 |
| 39.0.18 | [!] | anti-memorization test works | engine/antimemo.py<br>engine/lessons.py | 2 | 4 |
| 39.0.19 | [?] | direction engine works | engine/direction.py<br>engine/direction_ablate.py | 5 | 5 |
| 39.0.20 | [?] | exit engine works | engine/exits.py<br>engine/retester.py | 5 | 3 |
| 39.0.21 | [~] | stop/risk engine works | engine/stops.py<br>engine/gaprisk.py | 2 | 1 |
| 39.0.22 | [?] | per-type trust works | engine/trust.py<br>engine/direction.py | 5 | 4 |
| 39.0.23 | [!] | timeline dial exists and is tested | engine/timeline.py | 3 | 5 |
| 39.0.24 | [?] | outer training-basis loop works | engine/adaptive.py<br>engine/basis_search.py | 5 | 5 |
| 39.0.25 | [!] | planted-pattern calibration works | engine/planted.py<br>engine/pattern_lifecycle.py | 3 | 2 |
| 39.0.26 | [~] | experiment registry works | engine/registry.py<br>engine/experiment_memory.py | 5 | 5 |
| 39.0.27 | [~] | dashboard works | engine/site_data.py | 0 | 0 |
| 39.0.28 | [~] | Pattern Explorer works | scripts/site_pages.py<br>scripts/site_sources.py | 1 | 0 |
| 39.0.29 | [~] | Sensitivity page works | engine/exits.py<br>scripts/site_sources.py | 2 | 2 |
| 39.0.30 | [ ] | documentation reflects actual state | - | 0 | 0 |
| 39.0.31 | [~] | all negative findings are preserved | engine/checkpoint.py<br>engine/registry.py | 5 | 5 |
| 39.0.32 | [?] | champion/challenger system works | engine/champion.py<br>scripts/bible_trace.py | 5 | 5 |

### Phase 42: THE STANDARD FOR EVERY CLAIM

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 42.0.1 | [?] | exact pattern definition | engine/pattern_identity.py::pattern_id | 1 | 0 |
| 42.0.2 | [~] | discovery sample | engine/claims.py<br>engine/heavy_tests.py::false_discovery_summary | 3 | 0 |
| 42.0.3 | [?] | confirmation sample | engine/pattern_stats.py::confirmation_factor | 1 | 0 |
| 42.0.4 | [~] | later sample | engine/claims.py<br>engine/basis_search.py::sample_candidate | 3 | 8 |
| 42.0.5 | [~] | week-clustered statistics | engine/claims.py<br>engine/direction_ablate.py::week_bootstrap | 6 | 22 |
| 42.0.6 | [!] | permutation evidence | engine/claims.py<br>engine/antioverfit.py::label_permutation | 3 | 3 |
| 42.0.7 | [x] | FDR result | engine/adaptive.py::result<br>engine/analog_weighting.py::audit_result | 12 | 13 |
| 42.0.8 | [x] | P(real) | engine/claims.py<br>engine/resources.py | 5 | 3 |
| 42.0.9 | [~] | effect size | engine/claims.py<br>engine/pattern_stats.py::effective_n | 4 | 21 |
| 42.0.10 | [~] | incremental validation gain | engine/data_sources.py::ValidationReport | 1 | 2 |
| 42.0.11 | [~] | regime distribution | engine/claims.py<br>engine/heavy_tests.py::by_regime | 5 | 0 |
| 42.0.12 | [?] | cost-adjusted result | engine/objective.py::week_row | 1 | 0 |
| 42.0.13 | [x] | baseline | engine/claims.py<br>engine/analog_weighting.py::baseline | 4 | 2 |
| 42.0.14 | [x] | candidate | engine/claims.py<br>engine/adaptive.py::_evaluate_candidates | 12 | 12 |
| 42.0.15 | [x] | independent windows | engine/basis_search.py::BasisSearch | 1 | 1 |
| 42.0.16 | [x] | risk | engine/claims.py<br>engine/champion.py::_check_risk | 6 | 21 |
| 42.0.17 | [?] | statistical comparison | engine/ablation.py::paired_ci<br>engine/objective.py::paired_bootstrap | 1 | 0 |
| 42.0.18 | [x] | stability | engine/claims.py<br>engine/exits.py::selection_stability | 7 | 20 |
| 42.0.19 | [x] | adoption reason | engine/claims.py<br>engine/adaptive.py::hold_reasons | 9 | 7 |

### Phase 43: PRIORITY WHEN TIME/COMPUTE IS LIMITED

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 43.0.1 | [x] | anti-leak correctness | engine/pit.py::full_check | 1 | 1 |
| 43.0.2 | [~] | Algorithm | engine/run_report.py::algorithm_section | 2 | 0 |
| 43.0.3 | [!] | Pattern validation | engine/exits.py::pattern_fail_flags<br>engine/fv_pipeline.py::validate_bars | 7 | 33 |
| 43.0.4 | [x] | Pattern → mover integration | engine/pattern_movers.py::PatternMoverModel | 2 | 1 |
| 43.0.5 | [x] | Analog validation | engine/run_report.py::analogs_section<br>engine/timeline.py::forecast_from_analogs | 2 | 1 |
| 43.0.6 | [x] | Memory | engine/resources.py::memory_gb<br>scripts/dashboard_build.py::s_memory | 8 | 6 |
| 43.0.7 | [x] | Direction | engine/direction.py::DirectionEngine<br>engine/direction_ablate.py | 4 | 34 |
| 43.0.8 | [!] | Exit/stop | engine/exits.py::ExitResult<br>engine/fv_pipeline.py::exit_specs | 4 | 19 |
| 43.0.9 | [x] | outer-loop tuning | engine/basis_search.py::BasisSearch | 1 | 1 |
| 43.0.10 | [~] | dashboard | engine/site_data.py::write_site | 0 | 0 |
| 43.0.11 | [x] | Live | engine/resources.py<br>engine/isolation.py::live_mode | 5 | 3 |

### Phase 44: RESOURCE MANAGEMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 44.0.1 | [?] | Windows PC | engine/claims.py<br>engine/resources.py | 6 | 6 |
| 44.0.2 | [~] | 8 CPU cores | engine/resources.py::cpu_cores<br>engine/gaprisk.py::_fit_core | 3 | 8 |
| 44.0.3 | [~] | 16 GB RAM | engine/resources.py::memory_gb | 0 | 0 |
| 44.0.4 | [~] | 3–7 workers depending on memory | engine/resources.py::worker_count | 0 | 0 |
| 44.0.5 | [~] | memory-aware scheduling | engine/resources.py::JobQueue | 0 | 0 |
| 44.0.6 | [x] | deterministic worker seeds | engine/resources.py::worker_seed<br>engine/blind_gates.py::check_sealed_before_workers | 7 | 6 |
| 44.0.7 | [~] | automatic cleanup | engine/resources.py::cleanup | 0 | 0 |
| 44.0.8 | [~] | stale-process detection | engine/resources.py::stale_report | 0 | 0 |
| 44.0.9 | [x] | worker health reports | engine/resources.py::worker_health<br>engine/health.py::WorkerLog | 5 | 4 |

### Phase 46: FINAL REPORT REQUIREMENT

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 46.1.1 | [~] | 1. Implementation report: Every module changed | engine/adaptive.py::adaptation_report<br>engine/parity.py::ParityReport | 10 | 9 |
| 46.2.1 | [~] | 2. Validation report: Every gate | scripts/final_report.py::s_validation<br>engine/adaptive.py::_global_gate | 13 | 12 |
| 46.3.1 | [~] | 3. Research report: What genuinely improved | scripts/bible_trace.py::genuine_gaps<br>scripts/final_report.py::s_research_and_failures | 4 | 4 |
| 46.4.1 | [x] | 4. Failure report: What did not work | engine/blind_gates.py::check_sealed_before_workers<br>engine/exits.py::format_report | 8 | 26 |
| 46.5.1 | [~] | 5. Remaining uncertainty report: What is still unproven | scripts/final_report.py::s_uncertainty | 2 | 0 |
| 46.6.1 | [~] | 6. Champion configuration: Exact configuration and hash | scripts/final_report.py::cfg_hash<br>engine/adaptive.py::audit_matches_cfg | 2 | 0 |
| 46.7.1 | [~] | 7. Challenger configurations: Exact configurations and evidence | scripts/final_report.py::cfg_hash<br>engine/adaptive.py::audit_matches_cfg | 2 | 0 |
| 46.8.1 | [~] | median | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.2 | [~] | mean | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.3 | [~] | standard deviation | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.4 | [~] | percentiles | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.5 | [~] | worst | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.6 | [~] | best | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.7 | [~] | drawdown | scripts/final_report.py::distribution | 2 | 0 |
| 46.8.8 | [~] | era breakdown | scripts/final_report.py::distribution | 2 | 0 |
| 46.9.1 | [x] | 9. Leakage audit: Explicit pass/fail | scripts/final_report.py::audit<br>engine/livesim.py::_raise_on_fail | 7 | 6 |
| 46.10.1 | [?] | 10. Reproducibility audit: Explicit pass/fail | scripts/final_report.py::audit<br>engine/pattern_lifecycle.py::_fail | 5 | 2 |
| 46.11.1 | [~] | 11. Checklist: Every item marked honestly | scripts/site_sources.py::parse_checklist | 1 | 0 |

### Phase 22: BLIND CLOCK

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 22.0.1 | [~] | BLIND CLOCK: Every simulated day: | engine/blind_gates.py::BlindClock<br>engine/livesim.py::BlindGateError | 3 | 6 |

### Phase 37: CHECKLIST STATE MACHINE

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 37.0.1 | [~] | CHECKLIST STATE MACHINE: Create/maintain: | scripts/bible_trace.py<br>scripts/test_inventory.py | 3 | 0 |

### Phase 40: WHAT "KEEP WORKING" MEANS

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 40.0.1 | [~] | WHAT "KEEP WORKING" MEANS: After each completed phase: | engine/ablation.py::mean_arm<br>engine/analog_weighting.py::_expanding_target_mean | 4 | 9 |

### Phase 41: NEVER SETTLE FOR A COSMETIC IMPLEMENTATION

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 41.0.1 | [ ] | NEVER SETTLE FOR A COSMETIC IMPLEMENTATION: Examples of unacceptable completion: | - | 0 | 0 |

### Phase 45: CONTINUOUS RESEARCH LOOP

| id | state | requirement | code | tests | evidence |
|---|---|---|---|---:|---:|
| 45.0.1 | [ ] | CONTINUOUS RESEARCH LOOP: Once the required implementation is complete, do not consider the researc… | - | 0 | 0 |
