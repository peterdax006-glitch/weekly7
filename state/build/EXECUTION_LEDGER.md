# EXECUTION LEDGER - canon C69 master state audit (sections 2, 4, 5, 6, 24, 36)

Written 2026-09-29 (~17:00-18:00 MDT) by a read-only Opus audit. This is the ONLY file the audit wrote in the repo. Scratch output
(reachability dumps, a one-cycle planted research-loop run, AST scans) went to the session scratchpad only. `state/livesim/` was not read.
No git stash/reset/pull. The full test suite was NOT run. Two test files were run: `tests/test_research_loop.py` and
`tests/test_research_selection_exits.py` gave **54 passed in 56 s**.

**Builders still writing (treat as IN PROGRESS, not judged final):**
- P03: `engine/research/market_expectations.py`, `change_points.py`, `regime_memory.py`, `tests/test_research_market_change.py`.
- W02: `engine/research/feeds.py`, `evidence.py` (untracked), and `loop.py` (modified in the working tree).

Git at audit time: HEAD `eeb393c` (C69 saved). Working tree: `engine/research/loop.py` modified; `engine/research/evidence.py`
untracked; 5 loop2 round checkpoints (rounds 004-008 of 29 Sep) untracked; `prof.out` untracked.

**Nothing may disappear from this ledger (C69 section 2).** Future sessions update rows in place and append. They never delete a row.

## 0. Headline facts (each is re-derivable from the command named)

1. **341 of 369 checklist items are not VALIDATED.** C62 has 173, C66 has 104 and C68 has 64 (source: the three JSONs; table in section 1).
2. **73 items carry a stale JSON status or note.** Most are C66 items still marked IN_PROGRESS or NOT_STARTED although their
   wave-1 module exists. Examples: RF03 observer, RF08 knowability, RF21 autopsy, RF23 multiscale and RF32 brain health are all
   still NOT_STARTED, and all five modules exist with tests.
3. **Static reachability, default mode, passes.** `scripts/reachability.py` gives hooks 69 REACHED, 7 RESEARCH-ONLY, 0 UNREACHED,
   and modules 64/64 REACHED. So most of the 39 MISSING hooks in INTEGRATION_AUDIT.md (12:21) are now reached statically, via the
   new `engine/learning/loop_hooks.py` and `test_path.py`. INTEGRATION_AUDIT.md section 1a is therefore **stale**.
4. **`scripts/reachability.py --package engine.research` FAILS (exit 1).** 47 modules are REACHED and **15 are UNREACHED**. The
   unreached modules are all 13 C68 modules and W02's two:
   - the ten accepted C68 modules: expectations, outcomes, prediction_error, error_research, pattern_change, what_changed,
     selection_constraint, exit_research, calibration_target, self_correct;
   - P03's three: market_expectations, change_points, regime_memory;
   - W02's feeds and evidence.
   **Every C68 module is dead in production today.**
5. **Data does not flow.** One planted cycle was run into the scratchpad:
   `scripts/research_loop.py --once --root <scratch> --checkpoint off`. Result: **51 stages; 27 OK, 24 SKIPPED_NO_INPUT**;
   knowledge 0; section-47 chains 0; knowledge->decision chains 0.
   Several stages report OK with 0 in and 0 out (experiment_memory, research_graph, replication, brain_health).
   `two_stage` took 80 rows in and put 0 out. `regimes` took 4 in and put 0 out. The skipped stages are listed in section 2.
6. **Nothing has ever run in production with persistence.**
   - `state/learning/` does not exist, so there has been no curator store and no hub sinks.
   - `state/research/research_loop/` does not exist, so the research loop has never run at its default root.
   - No `"legit"` key appears in any loop2 round checkpoint, so the C62 learner has never shadowed a real window.
   - The only production decision-maker is still `engine/adaptive.Session` (via `scripts/livesim_loop2.py`).
7. **Scientific state.**
   - No real-data learning result has ever passed: learning curve 0/6 windows rising; cross-year NO_TRANSFER; lessons harmful
     (-0.188%/wk); missed-winner uplift 0; direction 51.8% (coin flip); pattern reliability AUC 0.4975.
   - The only positive real result is the movement edge (57.9% vs 14.6% base), and it comes from a **survivor-only** panel.
   - The planted C62 acceptance_mini improved on **1 of 3 seeds** (`state/research/acceptance_mini/summary.json`).
8. **All 10 C66 VALIDATED items (RS01, RS05-RS12, RS14) and several C62 VALIDATED items rest on the survivor-only panel.**
   RG04 is "CURRENTLY VIOLATED". C69 section 23 forbids survivor-only results silently becoming final evidence, and section 39
   requires "no unjustified VALIDATED states". **These VALIDATED labels must be re-labelled or caveated** (work item W-03).

## 1. Every item that is NOT VALIDATED (C69 section 2), by checklist and status

Status legend: *IMPLEMENTED - not validated* means code plus unit tests exist and there is no evidence it works as intended.
"Stale?" marks rows whose JSON status or notes contradict the code on disk today. Stale rows need a checklist update through
`scripts/contract_checklist.py`, not a code change. C62's VALIDATED rows (18) and C66's (10) are not listed here, but see
headline 8. The C62 IMPLEMENTED list is long (146) because the C63 foundation wave marked everything with code as IMPLEMENTED.
None of those 146 has passed a real-data or intended-behaviour check.

| checklist | items | VALIDATED | IMPLEMENTED (not validated) | IN_PROGRESS | NOT_STARTED | FAILED | not VALIDATED total | flagged stale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C62 | 191 | 18 | 146 | 7 | 7 | 13 | 173 | 6 |
| C66 | 114 | 10 | 4 | 85 | 15 | 0 | 104 | 42 |
| C68 | 64 | 0 | 18 | 0 | 46 | 0 | 64 | 25 |
| **total** | 369 | 28 | 168 | 92 | 68 | 13 | 341 | 73 |

### 1.1 C62 - `state/build/SELF_LEARNING_MASTER_CHECKLIST.json` (191 items; 173 not VALIDATED)
Counts: FAILED 13, IN_PROGRESS 7, NOT_STARTED 7, IMPLEMENTED 146, VALIDATED 18

#### C62 FAILED (13)
| id | description | code path(s) | stale? |
|---|---|---|---|
| C03 | Implement retrieval. | analogs*.py; learning/context.py; learning/learner.py; learning/retrieval.py; learning/similarity.py; learning/situation.py; memory.py; pattern_memory.py | **STALE:** status FAILED but newest note says IMPLEMENTED - NOT VALIDATED (S12); contract_checklist.py refuses the downgrade, so the row is internally inconsistent |
| D13 | Missed-winner analysis. | learning/missed_winners.py; missed_winners.py | **STALE:** status FAILED; newest note is an S05 IMPLEMENTED note; the FAILED evidence (real uplift 0, p 0.64) is older and still the only real-data result |
| D15 | OOS validation of failure lessons. | antimemo.py; scripts/lessons_archive.py |  |
| E04 | Legitimate learner. | learners.py (6 learners); learning/controls.py; learning/same_year.py; learning_delta.py | **STALE:** status FAILED; newest note is S16 pass 2; the FAILED evidence (B22 lc1: 0/6 windows rising) still stands |
| G01 | Experiment registry. | improve.py; learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py; registry.py; state/experiments.jsonl | **STALE:** status FAILED; newest note is S09 pass 1; the registry audit failure (provenance on 3.8% of records) still stands |
| H02 | Future-data firewall. | antioverfit.py; leak_audit.py; parity.py; pit.py |  |
| H03 | Memory firewall. | blind_gates.py; leak_audit.py; learning_delta.py; pattern_memory.py |  |
| I10 | Planted regime pattern. | learning/planted_world.py; planted.py (regime plant) |  |
| I13 | Planted unless pattern. | learning/planted_world.py; planted.py (unless plant) |  |
| I16 | Real-data validation. | scripts/learning_delta.py; scripts/learner_search.py; scripts/lessons_archive.py; scripts/pattern_memory_real.py |  |
| I18 | Cross-year validation. | learning_delta.py; learners.py |  |
| L13 | Future-information audit passed. | leak_audit.py |  |
| L14 | Provenance audit passed. | scripts/audit_registry.py |  |

#### C62 IN_PROGRESS (7)
| id | description | code path(s) | stale? |
|---|---|---|---|
| I01 | Unit tests. | tests/*.py (78 files) | **STALE:** note says pattern_reliability.py has no tests; tests/test_pattern_reliability.py now exists |
| I12 | Planted conditional pattern. | learning/planted_world.py; pattern_reliability.py; planted.py |  |
| I17 | Fresh holdout seeds. | scripts/planted_calibration.py; scripts/learner_search.py (seeds 11,12) |  |
| L08 | Cross-regime transfer measured. | - |  |
| L15 | Reproducibility passed. | retester.py; repro.py; scripts/check_retester.py |  |
| L16 | Stability passed. | scripts/learner_search.py |  |
| L24 | All critical failures resolved or explicitly classified as unresolved scientific limitations. | leak_audit.py (LEAK/QUARANTINED classes) |  |

#### C62 NOT_STARTED (7)
| id | description | code path(s) | stale? |
|---|---|---|---|
| L09 | Cross-stock transfer measured. | - |  |
| L10 | Cross-sector transfer measured. | - |  |
| L21 | Learning delta independently reproduced. | - |  |
| L22 | Fresh holdout validation passed. | - |  |
| L23 | Knowledge promotion gates passed. | - | **STALE:** note says 'No knowledge promotion gate exists'; engine/learning/promotion.py (10 gates) and engine/research/quality_gate.py exist - NOT_STARTED should read 'gate exists, never passed anything real' |
| L25 | Final Masterstock updated. | - |  |
| L26 | Final checklist independently audited. | - |  |

#### C62 IMPLEMENTED - not validated (146)
| id | description | code path(s) | stale? |
|---|---|---|---|
| A01 | Define KnowledgeObject. | learning/core.py; learning/knowledge.py; lessons.py; memory.py; pattern_identity.py |  |
| A02 | Define epistemic states. | learning/core.py; learning/epistemic.py |  |
| A03 | Define immutable experience records. | learning/archive.py; lessons.py; pattern_memory.py; retester.py |  |
| A04 | Define situation representation. | learning/context.py; learning/retrieval.py; learning/similarity.py; learning/situation.py; learning_delta.py; lessons.py; memory.py |  |
| A05 | Define provenance schema. | learning/core.py; learning/knowledge.py; provenance.py; registry.py |  |
| A06 | Define knowledge versioning. | leak_audit.py; learning/core.py; learning/knowledge.py; pattern_bank.py; trust_store.py |  |
| A07 | Define lifecycle states. | learning/core.py; learning/epistemic.py; pattern_lifecycle.py |  |
| A08 | Define confidence dimensions. | learning/core.py; learning/epistemic.py; pattern_reliability.py; pattern_stats.py |  |
| A09 | Define reliability dimensions. | learning/core.py; learning/epistemic.py; pattern_reliability.py; trust.py |  |
| A10 | Implement serialization/deserialization. | learning/core.py; learning/knowledge.py; pattern_bank.py; pattern_identity.py; trust.py |  |
| A11 | Implement schema validation. | claims.py; learning/core.py; learning/knowledge.py; pattern_identity.py; registry.py |  |
| A13 | Add code/data/config provenance. | learning/core.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/knowledge.py; learning/memory_firewall.py; learning/reproducibility.py; provenance.py |  |
| B01 | Build raw experience archive. | learning/archive.py; pattern_memory.py; state/experiments.jsonl |  |
| B02 | Build episode archive. | learning/archive.py; lessons.py; memory.py |  |
| B03 | Build situation archive. | learning/archive.py |  |
| B04 | Build knowledge archive. | learning/archive.py; pattern_bank.py; pattern_memory.py; trust_store.py |  |
| B05 | Build retrieval index. | learning/archive.py; memory.py; pattern_memory.py |  |
| B06 | Build context index. | learning/archive.py |  |
| B07 | Build temporal index. | learning/archive.py; pattern_memory.py |  |
| B08 | Build reliability index. | learning/archive.py |  |
| B09 | Build failure index. | learning/archive.py; lessons.py; memory_diagnostics.py; pattern_reliability.py |  |
| B10 | Build contradiction index. | learning/archive.py |  |
| B11 | Build recovery index. | learning/retirement.py |  |
| B12 | Implement memory snapshots. | checkpoint.py; learning/archive.py; learning_delta.py; pattern_memory.py |  |
| B13 | Implement immutable historical records. | checkpoint.py; learning/archive.py; pattern_bank.py; pattern_memory.py (hash chain, ChainCorrupt); trust_store.py |  |
| B14 | Implement dynamic influence. | learning/archive.py; memory.py; pattern_bank.py; pattern_memory.py |  |
| B15 | Implement memory audit. | blind_gates.py; learning/archive.py; lessons.py; memory_diagnostics.py; pattern_memory.py |  |
| C01 | Implement observe stage. | adaptive.py; learning/learner.py; livesim.py |  |
| C02 | Implement situation description. | learning/context.py; learning/learner.py; learning/retrieval.py; learning/similarity.py; learning/situation.py; lessons.py; memory.py |  |
| C04 | Implement expectation formation. | direction.py; learning/learner.py; memory.py |  |
| C05 | Implement decision capture. | adaptive.py; fill_audit.py; learning/learner.py; learning_delta.py |  |
| C06 | Implement outcome capture. | learning/learner.py; learning_delta.py; lessons.py; pattern_memory.py (mature_date) |  |
| C07 | Implement surprise detection. | learning/learner.py; learning/surprise.py; memory.py; run_report.py |  |
| C08 | Implement credit assignment. | ablation.py; learning/credit.py; learning/learner.py; memory_diagnostics.py |  |
| C09 | Implement blame assignment. | learning/learner.py; learning/separation.py; lessons.py |  |
| C10 | Implement belief updating. | learning/belief.py; learning/learner.py; pattern_stats.py; trust.py |  |
| C11 | Implement reliability updating. | learning/learner.py; pattern_reliability.py; trust_store.py |  |
| C12 | Implement condition discovery. | candidates.py; learning/boundary.py; learning/competition.py; learning/learner.py; pattern_lifecycle.py (rescoped); pattern_reliability.py |  |
| C13 | Implement anti-condition discovery. | candidates.py; learning/boundary.py; learning/competition.py; learning/learner.py; pattern_reliability.py |  |
| C15 | Implement knowledge storage. | learning/learner.py; lessons.py; pattern_bank.py; pattern_memory.py; trust_store.py |  |
| C16 | Implement meta-learning update. | learning/experiment_memory.py; learning/failed_learners.py; learning/learner.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| C17 | Implement research-priority update. | improve.py; learning/experiment_memory.py; learning/failed_learners.py; learning/learner.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| D01 | Loss classifier. | learning/failure.py; lessons.py |  |
| D02 | Selection failure detector. | learning/failure.py; lessons.py (bad_entry); memory.py; missed_winners.py |  |
| D03 | Timing failure detector. | exits.py; learning/failure.py; lessons.py (missed_exit) |  |
| D04 | Direction failure detector. | direction.py; learning/failure.py; learning_delta.py |  |
| D05 | Risk failure detector. | gaprisk.py; learning/failure.py; lessons.py (oversized_loser); stops.py |  |
| D06 | Pattern failure detector. | learning/failure.py; pattern_lifecycle.py; pattern_movers.py; pattern_reliability.py; patterns.py (death) |  |
| D07 | Context failure detector. | learning/failure.py; pattern_reliability.py |  |
| D08 | Regime failure detector. | antioverfit.py; learning/failure.py; lessons.py (regime_misread); pattern_reliability.py |  |
| D09 | Measurement failure detector. | data_sources.py; learning/failure.py; parity.py; pit.py |  |
| D10 | Unknown failure state. | learning/failure.py; pattern_reliability.py |  |
| D11 | Pattern-break analysis. | learning/break_detection.py; learning/health.py; learning/lifecycle.py; learning/reliability.py; pattern_reliability.py; pattern_reliability.py |  |
| D12 | Recovery analysis. | learning/break_detection.py; learning/health.py; learning/lifecycle.py; learning/reliability.py; pattern_reliability.py |  |
| D14 | Counterfactual distinction discovery. | learning/missed_winners.py; lessons.py; missed_winners.py |  |
| E01 | Same-year rerun harness. | learning/controls.py; learning/same_year.py; learning_delta.py; scripts/learning_curve.py; scripts/learning_delta.py |  |
| E02 | Disguised rerun mechanism. | antimemo.py; blind_gates.py; learning/controls.py; learning/same_year.py; learning_delta.py |  |
| E06 | Random learner control. | learning/controls.py; learning/same_year.py; learning_delta.py |  |
| E09 | Cross-regime test. | antioverfit.py; heavy_tests.py; learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; learning_delta.py |  |
| E10 | Cross-stock test. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py |  |
| E11 | Cross-sector test. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py |  |
| E12 | Cross-volatility test. | antioverfit.py (type quiet/middle/wild); learners.py; learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py |  |
| E13 | Transfer ratio. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; learning_delta.py |  |
| E15 | Identity gap. | antimemo.py; learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; learning_delta.py |  |
| E16 | Transfer stability. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; scripts/learner_search.py (seeds) |  |
| F01 | Supports edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F02 | Contradicts edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F03 | Contains edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; pattern_identity.py; pattern_stats.py |  |
| F04 | Specializes edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; pattern_lifecycle.py (rescoped); trust.py |  |
| F05 | Generalizes edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; trust.py |  |
| F06 | Causes-failure edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F07 | Recovers-with edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F08 | Redundant-with edges. | learning/redundancy.py; pattern_stats.py; patterns.py (duplicate status) |  |
| F09 | Complements edges. | learning/redundancy.py |  |
| F10 | Depends-on edges. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F11 | Graph traversal. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py |  |
| F12 | Contradiction investigation. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; pattern_reliability.py |  |
| F13 | Parent/child shrinkage. | learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; pattern_stats.py; trust.py |  |
| F14 | Knowledge lineage. | leak_audit.py; learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; pattern_bank.py (window history); pattern_memory.py (source run id) |  |
| F15 | Decision lineage. | adaptive.py; explain.py; learning/contradiction.py; learning/hierarchy.py; learning/knowledge_graph.py; memory_diagnostics.py |  |
| G02 | Question generation. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G03 | Competing hypothesis generation. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G04 | Expected outcome generation. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G05 | Information-gain scoring. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G06 | Research priority queue. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G07 | Duplicate experiment detection. | experiment_memory.py; learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py; registry.py |  |
| G08 | Failed experiment memory. | experiment_memory.py; learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py; state/experiments.jsonl |  |
| G09 | Failed learner memory. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py; learning_delta.py; state/research/learners/*/report.md |  |
| G10 | Exploration/exploitation policy. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| G11 | Compute-aware research allocation. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py; resources.py |  |
| G12 | Meta-learning research policy. | learning/experiment_memory.py; learning/failed_learners.py; learning/meta_learning.py; learning/research_policy.py; learning/research_priority.py |  |
| H01 | Point-in-time firewall. | learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; pit.py (PITStore, Guard, purged_training_set) |  |
| H04 | Identity firewall. | antioverfit.py; blind_gates.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; learning_delta.py |  |
| H05 | Provenance firewall. | claims.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; provenance.py; retester.py |  |
| H07 | Revision-data firewall. | edgar.py; leak_audit.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py |  |
| H08 | Survivorship firewall. | leak_audit.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; pit.py; scripts/fetch_delisted.py |  |
| H10 | Evaluation contamination firewall. | blind_gates.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; learning_delta.py |  |
| H11 | Cache/network firewall. | isolation.py; leak_audit.py; learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py |  |
| H12 | Fail-closed behavior. | learning/firewalls.py; learning/future_firewall.py; learning/identity_firewall.py; learning/memory_firewall.py; learning/reproducibility.py; learning_delta.py; pattern_memory.py; pit.py |  |
| I02 | Integration tests. | learning/test_path.py; learning/wiring.py; tests/integration/ |  |
| I03 | Property tests. | tests/test_learning_properties.py |  |
| I04 | Determinism tests. | repro.py |  |
| I05 | Negative tests. | (planted-defect test per file, CONTEXT rule 5) |  |
| J01 | Champion knowledge. | champion.py; learning/champion.py |  |
| J02 | Challenger knowledge. | champion.py; improve.py; learning/champion.py |  |
| J03 | Shadow knowledge. | learning/champion.py; shadows.py |  |
| J04 | Promotion gate. | champion.py; learning/promotion.py; objective.py |  |
| J05 | Retirement gate. | learning/retirement.py; pattern_bank.py; pattern_lifecycle.py; pattern_memory.py (disregarded) |  |
| J06 | Recovery gate. | learning/retirement.py; pattern_bank.py; pattern_memory.py (requalify) |  |
| J07 | Reliability monitoring. | learning/break_detection.py; learning/champion.py; learning/health.py; learning/lifecycle.py; learning/promotion.py; learning/reliability.py; pattern_reliability.py; trust_store.py |  |
| J08 | Calibration monitoring. | direction_calib.py; learning/calibration.py; pattern_reliability.py |  |
| J09 | Contradiction monitoring. | learning/champion.py; learning/contradiction_monitor.py; learning/promotion.py |  |
| J10 | Surprise monitoring. | learning/surprise.py |  |
| J11 | Knowledge health. | learning/break_detection.py; learning/champion.py; learning/health.py; learning/lifecycle.py; learning/promotion.py; learning/reliability.py; memory_diagnostics.py; pattern_reliability.py |  |
| J12 | Decision impact monitoring. | adaptive.py; learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; pattern_movers.py (ablation) |  |
| J13 | Learning curve. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; learning_delta.py; scripts/learning_curve.py |  |
| J14 | Learning scorecard. | learning/learning_curve.py; learning/portfolio_value.py; learning/scorecard.py; learning/transfer.py; learning/transfer_score.py; learning_delta.py |  |
| J15 | Research priority dashboard. | learning/reports.py; scripts/learning_report.py |  |
| K01 | Learning curve report. | learning/reports.py; learning_delta.py; scripts/learning_report.py |  |
| K02 | Transfer report. | learning/reports.py; learning_delta.py; scripts/learner_search.py; scripts/learning_report.py |  |
| K03 | Memorization report. | learning/reports.py; learning_delta.py; scripts/learning_report.py |  |
| K04 | Identity-gap report. | antimemo.py; learning/reports.py; scripts/learning_report.py |  |
| K05 | Knowledge health report. | learning/reports.py; memory_diagnostics.py; pattern_reliability.py; scripts/learning_report.py |  |
| K06 | Failure report. | learning/reports.py; lessons.py; scripts/learning_report.py; scripts/run_pattern_fail.py |  |
| K07 | Recovery report. | learning/reports.py; scripts/learning_report.py |  |
| K08 | Contradiction report. | learning/reports.py; scripts/learning_report.py |  |
| K09 | Missed-winner report. | learning/reports.py; missed_winners.py; scripts/learning_report.py; scripts/run_b15_memory_adapter.py |  |
| K10 | Experiment-memory report. | experiment_memory.py; learning/reports.py; scripts/audit_registry.py; scripts/learning_report.py |  |
| K11 | Research-priority report. | learning/reports.py; scripts/learning_report.py |  |
| K12 | Meta-learning report. | learning/reports.py; scripts/learning_report.py |  |
| K13 | Promotion/rejection report. | champion.py; improve.py; learning/reports.py; scripts/learning_report.py |  |
| K14 | Provenance audit report. | learning/reports.py; scripts/audit_registry.py; scripts/leak_audit.py; scripts/learning_report.py; scripts/pit_audit_real.py |  |
| K15 | Full learning-system report. | learning/reports.py; scripts/learning_report.py |  |
| L01 | No-learning baseline frozen. | baseline.py; learning/controls.py; learning/same_year.py; learning_delta.py |  |
| L02 | Legitimate learner frozen. | learning/controls.py; learning/learner.py |  |
| L03 | Memorizer control frozen. | learning/controls.py; learning/same_year.py; learning_delta.py |  |
| L04 | Random learner control frozen. | learning/controls.py; learning/learner.py; learning/same_year.py |  |
| L05 | Leaky learner control frozen. | learning/controls.py; learning/same_year.py; learning_delta.py |  |
| L06 | Same-year learning measured. | scripts/learning_curve.py; scripts/learning_delta.py |  |
| L07 | Cross-year transfer measured. | scripts/learning_delta.py; scripts/learner_search.py |  |
| L11 | Memorization gap measured. | learning_delta.py |  |
| L12 | Identity gap measured. | antimemo.py |  |
| L17 | Risk impact measured. | learning_delta.py |  |
| L18 | Direction impact measured. | learning_delta.py |  |
| L19 | Movement impact measured. | learning_delta.py |  |
| L20 | Portfolio impact measured. | learning_delta.py |  |

### 1.2 C66 - `state/build/RESEARCH_BRAIN_CHECKLIST.json` (114 items; 104 not VALIDATED)
Counts: IN_PROGRESS 85, NOT_STARTED 15, IMPLEMENTED 4, VALIDATED 10

#### C66 IN_PROGRESS (85)
| id | description | code path(s) | stale? |
|---|---|---|---|
| RF01 | Research priority engine implemented | learning/research_priority.py; learning/research_policy.py | **STALE:** research/priority.py (R15 accepted) + test_research_priority.py |
| RF02 | Autonomous research scheduler implemented | learning/research_priority.py; learning/checkpoints.py; learning/learner.py | **STALE:** research/loop.py, controller.py, two_stage.py (W01); runs, but 24/51 stages SKIPPED_NO_INPUT (this audit's planted run) |
| RF04 | Winner research implemented | learning/failure.py; learning/postmortem.py; learning/separation.py; learning/credit.py | **STALE:** research/winners_losers.py (R05) + test_research_winners_losers.py |
| RF05 | Loser research implemented | learning/failure.py; learning/postmortem.py; learning/separation.py; learning/credit.py | **STALE:** research/winners_losers.py, loss_pipeline.py (R05) |
| RF10 | Volatility laboratory implemented | fv_pipeline.py; pattern_movers.py; direction_features.py; scripts/movers.py | **STALE:** research/volatility_lab.py + vol_hypotheses.py (R07) + test_research_volatility.py |
| RF11 | Direction laboratory implemented | direction.py; direction_calib.py; direction_ablate.py; direction_features.py; fv_pipeline.py; scripts/direction_research.py; scripts/direction_study.py | **STALE:** research/direction_lab.py (R08) + test_research_direction.py |
| RF12 | Conditional accuracy frontier implemented | direction_features.py; direction.py | **STALE:** research/frontier.py (R09) + test_research_frontier.py |
| RF13 | Pattern discovery expansion implemented | patterns.py; candidates.py; pattern_stats.py; pattern_identity.py; pattern_bank.py; pattern_lifecycle.py; pattern_memory.py; pattern_memory_eval.py; learning/knowledge.py; planted.py | **STALE:** research/discovery.py + discovery_sources.py (R10) + test_research_discovery*.py |
| RF14 | Pattern-break research integrated | learning/break_detection.py; pattern_reliability.py; learning/boundary.py | **STALE:** research/break_research.py (R11) + test_research_breaks.py |
| RF15 | Experiment memory integrated | learning/experiment_memory.py; experiment_memory.py; registry.py | **STALE:** research/experiments.py (R12) + test_research_experiments.py |
| RF17 | Failed-learner registry integrated | learning/failed_learners.py | **STALE:** research/failed_lab.py (R12) |
| RF18 | Compute manager implemented | learning/compute.py; resources.py; health.py; learning/research_policy.py | **STALE:** research/compute_manager.py (R14) + test_research_compute.py |
| RF20 | Compute-waste controller implemented | learning/research_priority.py; learning/research_policy.py | **STALE:** research/waste.py (R14) |
| RF22 | Daily research generator implemented | learning/research_priority.py; learning/surprise.py; learning/contradiction_monitor.py | **STALE:** research/targets.py (R15) |
| RF24 | Cross-sectional research implemented | analogs_sector.py; trust.py; learning/situation.py | **STALE:** research/cross_section.py (R16) |
| RF25 | Regime research implemented | learning/context.py; learning/hierarchy.py; learning/situation.py | **STALE:** research/regimes.py (R16) + test_research_regimes.py |
| RF26 | Interaction discovery implemented | learning/complexity.py; learning/missed_winners.py; learning/break_detection.py; pattern_movers.py; candidates.py (pairs, A and B UNLESS C) | **STALE:** research/interactions.py (R17) + test_research_interactions.py |
| RF27 | Knowledge graph integrated | learning/knowledge_graph.py | **STALE:** research/research_graph.py (R18) + test_research_graph.py |
| RF28 | Knowledge-to-decision bridge integrated | learning/decision_contract.py; learning/champion.py | **STALE:** research/decision_bridge.py (R18) |
| RF29 | Research/trader firewall integrated | learning/trader_view.py; learning/curator.py; learning/test_path.py; learning/firewalls.py; learning/future_firewall.py; learning/memory_firewall.py; learning/identity_firewall.py | **STALE:** research/firewall.py + namespaces.py (R01) + test_research_firewall.py |
| RF30 | Replication system implemented | learning/transfer.py; learning/transfer_score.py; learning/promotion.py | **STALE:** research/replication.py (R19) + test_research_gate.py |
| RF31 | Unknown-cause system integrated | learning/unknowns.py | **STALE:** research/unknown_cause.py (R02) |
| RF33 | Research-diversity system implemented | learning/research_policy.py | **STALE:** research/diversity.py (R20) |
| RF34 | Scientific long-term memory integrated | learning/archive.py; learning/interpretation.py; learning/belief.py; learning/epistemic.py | **STALE:** research/science_memory.py (R18) |
| RF35 | Question generator implemented | learning/research_priority.py | **STALE:** research/questions.py (R15) |
| RF36 | Hypothesis tree implemented | learning/competition.py | **STALE:** research/hypothesis_tree.py (R15) |
| RF37 | Promotion quality gate implemented | learning/promotion.py; champion.py | **STALE:** research/quality_gate.py (R19) + test_research_gate.py |
| RT01 | Unit tests complete | tests/test_learning_*.py (37 files) | **STALE:** 30 tests/test_research_*.py files now exist; notes still say 'C62 modules only' |
| RT02 | Integration tests complete | tests/test_learning_integration.py; tests/test_learning_reachability_loop.py; tests/test_learning_reachability_system.py | **STALE:** tests/test_research_loop.py (23 tests) exists; notes say 'none' |
| RT03 | Planted volatility world passes | fv_pipeline.py; tests/test_fv_pipeline.py; tests/test_pattern_movers.py |  |
| RT04 | Planted direction world passes | direction_features.py; tests/test_direction_features.py |  |
| RT05 | Planted failure-regime world passes | learning/planted_world.py; tests/test_learning_planted_world.py |  |
| RT06 | Null world does not generate fake signal | learning/meta_learning.py; learning/missed_winners.py; planted.py |  |
| RT07 | Future-leak tests pass | tests/test_learning_future_firewall.py; tests/test_leak_audit.py; tests/test_learning_test_path.py |  |
| RT08 | Identity-leak tests pass | tests/test_learning_identity_firewall.py; tests/test_tiebreak.py |  |
| RT09 | Year-recognition tests pass | tests/test_learning_curator.py; tests/test_leak_audit_verdicts.py |  |
| RT10 | Memorization controls pass | tests/test_learning_same_year.py; tests/test_antimemo.py |  |
| RT11 | Same-year controls pass | tests/test_learning_same_year.py |  |
| RT12 | Cross-year transfer tests pass | tests/test_learning_transfer.py |  |
| RT13 | Cross-stock transfer tests pass | tests/test_learning_transfer.py |  |
| RT14 | Cross-sector transfer tests pass | tests/test_learning_transfer.py |  |
| RT15 | Cross-regime transfer tests pass | tests/test_learning_transfer.py |  |
| RT16 | Research-priority tests pass | tests/test_learning_research.py |  |
| RT17 | Compute-allocation tests pass | tests/test_learning_compute.py; tests/test_learning_research.py |  |
| RT18 | Compute-waste tests pass | tests/test_learning_research.py |  |
| RT20 | Unknown-cause tests pass | tests/test_learning_epistemic.py | **STALE:** unknown_cause.py forced-label detector exists (R02); notes say none |
| RT21 | Replication tests pass | tests/test_learning_promotion.py; tests/test_repro.py | **STALE:** research/replication.py + test_research_gate.py exist; notes say replication missing |
| RT22 | Promotion gates pass | tests/test_learning_promotion.py |  |
| RS02 | Volatility calibration measured | fv_pipeline.py |  |
| RS03 | Volatility coverage measured | fv_pipeline.py |  |
| RS13 | Pattern transfer measured | heavy_tests.py |  |
| RS15 | Research-policy transfer measured | learning/research_policy.py |  |
| RS16 | Meta-learning transfer measured | learning/meta_learning.py |  |
| RS17 | Failed-learner memory demonstrated useful | learning/failed_learners.py |  |
| RS18 | Research compute efficiency measured | learning/compute.py; learning/scorecard.py |  |
| RA01 | System generates its own research questions | learning/research_priority.py |  |
| RA02 | System ranks research questions | learning/research_priority.py; learning/research_policy.py |  |
| RA03 | System allocates compute | learning/research_policy.py; learning/compute.py |  |
| RA04 | System runs experiments | learning/compute.py |  |
| RA05 | System records results | learning/experiment_memory.py |  |
| RA06 | System remembers failures | learning/failed_learners.py; learning/experiment_memory.py |  |
| RA07 | System remembers successes | learning/experiment_memory.py; learning/archive.py |  |
| RA08 | System studies winners | learning/credit.py |  |
| RA09 | System studies losers | learning/failure.py; lessons.py |  |
| RA10 | System studies missed winners | missed_winners.py; learning/missed_winners.py |  |
| RA12 | System investigates pattern breaks | learning/break_detection.py; pattern_reliability.py |  |
| RA13 | System identifies unknown causes | learning/unknowns.py |  |
| RA14 | System promotes validated knowledge | learning/promotion.py; improve.py |  |
| RA15 | System downgrades broken knowledge | learning/retirement.py; learning/health.py |  |
| RA16 | System detects wasted research | learning/research_priority.py; learning/research_policy.py |  |
| RA17 | System redirects compute | learning/research_policy.py |  |
| RA18 | System learns which research methods work | learning/meta_learning.py |  |
| RA19 | System continuously generates the next research agenda | learning/research_priority.py |  |
| RA20 | System can run indefinitely with checkpoints and recovery | learning/checkpoints.py |  |
| RG01 | No critical future-information pathway exists | leak_audit.py; learning/future_firewall.py |  |
| RG02 | No hidden year information reaches the trader | learning/trader_view.py; learning/curator.py |  |
| RG03 | No stock-identity memorization pathway remains | learning/identity_firewall.py |  |
| RG04 | No survivor-only evaluation remains where it affects conclusions | scripts/fetch_delisted.py; data_sources.py |  |
| RG05 | No major research result depends on a single lucky experiment | learning/promotion.py; learning/transfer.py |  |
| RG06 | No major promoted knowledge lacks OOS evidence | learning/promotion.py |  |
| RG07 | No major promoted knowledge lacks provenance | provenance.py; registry.py |  |
| RG08 | No critical failure is silently ignored | learning/checkpoints.py |  |
| RG13 | Full contract checklist is reconciled | state/build/SELF_LEARNING_MASTER_CHECKLIST.json |  |
| RG14 | Full autonomous-research checklist is reconciled | state/build/RESEARCH_MAPPING.md |  |
| RG16 | Masterstock contains the complete state | - |  |

#### C66 NOT_STARTED (15)
| id | description | code path(s) | stale? |
|---|---|---|---|
| RF03 | Market-wide observation layer implemented | learning/missed_winners.py; fv_pipeline.py; missed_winners.py | **STALE:** research/observer.py (R04 accepted) + test_research_observer.py - JSON says NOT_STARTED |
| RF08 | Knowability engine implemented | pit.py; edgar.py; learning/future_firewall.py; learning/unknowns.py | **STALE:** research/knowability.py (R02, 2,508 lines) + test_research_knowability.py - JSON says NOT_STARTED |
| RF19 | Experiment-value accounting implemented | learning/research_policy.py; learning/scorecard.py | **STALE:** research/value_accounting.py (R14) - JSON says NOT_STARTED |
| RF21 | Daily autopsy implemented | learning/reports.py; learning/surprise.py | **STALE:** research/autopsy.py (R04) - JSON says NOT_STARTED |
| RF23 | Multi-scale research implemented | candles.py; learning/situation.py | **STALE:** research/multiscale.py (R16) + test_research_scales.py - JSON says NOT_STARTED |
| RF32 | Research-health system implemented | learning/research_priority.py; learning/research_policy.py | **STALE:** research/brain_health.py (R20) + test_research_health.py - JSON says NOT_STARTED |
| RT19 | Winner/loser symmetry tests pass | - | **STALE:** tests/test_research_symmetry.py exists (R09); JSON says 'No symmetry engine' |
| RS04 | Predictable-vs-unknowable classification evaluated | - | **STALE:** knowability classifier now exists (R02); only the EVALUATION is missing - note 'No knowability classifier' is stale |
| RA11 | System studies missed losers | - | **STALE:** research/missed.py missed-loser path (R06; RF07 IMPLEMENTED) - JSON says 'No missed-loser code' |
| RG09 | No research branch consumes large compute without value accounting | - | **STALE:** research/value_accounting.py exists; notes say 'value accounting is missing' |
| RG10 | No code is padded merely to satisfy line counts | - | **STALE:** C66 code now exists (61 modules) so the padding audit can run; notes say 'no C66 code written yet' |
| RG11 | Existing C62–C65 requirements remain intact | - |  |
| RG12 | Existing tests remain intact unless legitimately updated for changed behavior | - |  |
| RG15 | Independent audit completed | - |  |
| RG17 | Only after ALL of the above: evaluate whether the test system is ready for the next deployment phase | - |  |

#### C66 IMPLEMENTED - not validated (4)
| id | description | code path(s) | stale? |
|---|---|---|---|
| RF06 | Missed-winner research implemented | learning/missed_winners.py; missed_winners.py; research/missed.py |  |
| RF07 | Missed-loser research implemented | learning/missed_winners.py; missed_winners.py; research/missed.py |  |
| RF09 | Counterfactual information reconstruction implemented | parity.py; parity_suite.py; research/counterfactual.py |  |
| RF16 | Meta-learning integrated | learning/meta_learning.py; research/meta_research.py |  |

### 1.3 C68 - `state/build/PREDICTION_ERROR_CHECKLIST.json` (64 items; 64 not VALIDATED)
Counts: NOT_STARTED 46, IMPLEMENTED 18

#### C68 NOT_STARTED (46)
| id | description | code path(s) | stale? |
|---|---|---|---|
| PEF | Checklist F: MARKET EXPECTATION VS MARKET REALITY | - | **STALE:** research/market_expectations.py + tests/test_research_market_change.py exist (P03, still writing) |
| PEG | Checklist G: MARKET CONTRACTION RESEARCH | - | **STALE:** market_expectations.py 'contraction' tests exist (P03, still writing) |
| PEI | Checklist I: CHANGE-POINT DETECTION | - | **STALE:** research/change_points.py exists (P03, still writing) |
| PER | Checklist R: EARLY REGIME WARNING SYSTEM | - | **STALE:** early-warning tests in test_research_market_change.py (P03, still writing) |
| PEW | Checklist W: REGIME-CHANGE MEMORY | - | **STALE:** research/regime_memory.py exists (P03, still writing) |
| PEX | Checklist X: NO RETROSPECTIVE REGIME CHEATING | - | **STALE:** test_no_retrospective_cheating_scrambling_the_future_leaves_the_past_bit_identical exists (P03, still writing) |
| PEY | Checklist Y: FULL ERROR-TO-IMPROVEMENT PIPELINE | - |  |
| PEZ | Checklist Z: FINAL INTEGRATION TESTS | - |  |
| PZ01 | predictions cannot be rewritten after outcomes | - | **STALE:** unit test exists: test_research_expectations.py::test_rewrite_refused_and_identical_rerecord_idempotent |
| PZ02 | exits cannot be manipulated to improve prediction statistics | - | **STALE:** unit test exists: test_research_selection_exits.py::test_each_outcome_abuse_is_detected |
| PZ03 | ±1% evaluation does not control selling | - | **STALE:** unit test exists: test_research_selection_exits.py::test_changing_the_calibration_target_leaves_every_exit_identical |
| PZ04 | future information cannot enter error analysis | - | **STALE:** unit test exists: test_research_expectations.py::test_future_information_cannot_enter_error_analysis |
| PZ05 | future information cannot enter regime detection | - | **STALE:** unit test exists (P03, in progress): test_research_market_change.py::test_no_retrospective_cheating_... |
| PZ06 | unknowable events are allowed to remain unknowable | - | **STALE:** unit test exists: test_research_what_changed.py::test_unknowable_outcome_is_preserved_and_proposes_nothing |
| PZ07 | tiny errors do not consume excessive compute | - | **STALE:** unit test exists: test_research_error_research.py::test_tiny_errors_stay_cheap_even_when_statistically_large |
| PZ08 | major errors trigger deeper investigation | - | **STALE:** unit test exists: test_research_error_research.py::test_planted_confident_failure_triggers_deep_investigation_with_all_15_questions |
| PZ09 | repeated errors escalate research priority | - | **STALE:** unit test exists: test_research_error_research.py::test_repeated_expected_8_10_realised_3_6_escalates_monotonically |
| PZ10 | pattern changes can be detected | - | **STALE:** unit test exists: test_research_what_changed.py::test_structural_step_is_caught_and_never_deleted |
| PZ11 | false regime changes do not unnecessarily disable successful patterns | - | **STALE:** unit test exists (P03): test_research_market_change.py::test_a_false_regime_change_does_not_disable_a_successful_pattern |
| PZ12 | genuinely degraded patterns can reduce influence | - | **STALE:** unit test exists: test_research_what_changed.py::test_gradual_decay_is_weakening |
| PZ13 | recovered patterns can regain influence | - | **STALE:** unit test exists: test_research_what_changed.py::test_returning_pattern_after_a_failure_spell |
| PZ14 | market-wide changes can be distinguished from individual-stock anomalies | - | **STALE:** unit test exists: test_research_what_changed.py::test_single_stock_extreme_day_is_level_four_not_market |
| PZ15 | exit timing is learned independently | - | **STALE:** unit test exists: test_research_selection_exits.py::test_learned_exit_beats_holding_on_planted_fade_and_never_on_trend |
| PZ16 | the 5–10% selection constraint is enforced | - | **STALE:** unit test exists: test_research_selection_exits.py::test_selection_band_is_realisable_gain_under_intended_policy |
| PZ17 | stocks outside the predicted 5–10% outcome range cannot be selected merely to improve another metric | - | **STALE:** unit test exists: test_research_selection_exits.py::test_cherry_picking_and_target_change_are_refused |
| PZ18 | the system cannot game the ±1% target | - | **STALE:** unit test exists: test_research_selection_exits.py::test_target_measured_honestly_achieved_and_not_achieved |
| PZ19 | discoveries must pass out-of-sample validation | - | **STALE:** unit test exists: test_research_selection_exits.py::test_fixes_are_tested_independently_and_promoted_only_through_the_gate |
| PZ20 | the learner improves only when evidence justifies improvement | - |  |
| PC01 | Every checklist item is implemented. | - |  |
| PC02 | Every item has tests. | - |  |
| PC03 | Every integration point is reachable. | - |  |
| PC04 | No future leakage exists. | - |  |
| PC05 | Deterministic replay works. | - |  |
| PC06 | Prediction expectations are immutable. | - |  |
| PC07 | Error analysis works at stock, sector, market, pattern, and regime levels. | - |  |
| PC08 | Exit learning works independently of prediction calibration. | - |  |
| PC09 | The 5–10% selection constraint is enforced. | - |  |
| PC10 | The ±1% target is measured honestly. | - |  |
| PC11 | Confident failures trigger deeper investigation. | - |  |
| PC12 | Repeated errors produce escalating research priority. | - |  |
| PC13 | Regime-change detection has forward-time validation. | - |  |
| PC14 | Knowable and unknowable causes are distinguished. | - |  |
| PC15 | Research discoveries are validated out of sample. | - |  |
| PC16 | The system demonstrably learns from prediction errors. | - |  |
| PC17 | The existing Weekly7 checklist remains intact. | - |  |
| PC18 | No existing test or scientific gate is weakened to make this pass. | - |  |

#### C68 IMPLEMENTED - not validated (18)
| id | description | code path(s) | stale? |
|---|---|---|---|
| PEA | Checklist A: IMMUTABLE PRE-PREDICTION EXPECTATION | research/expectations.py; research/outcomes.py; research/prediction_error.py |  |
| PEB | Checklist B: COMPLETE OUTCOME RECONSTRUCTION | research/expectations.py; research/outcomes.py; research/prediction_error.py |  |
| PEC | Checklist C: PREDICTION ERROR ENGINE | research/expectations.py; research/outcomes.py; research/prediction_error.py |  |
| PED | Checklist D: ERROR-SIZE-BASED RESEARCH | research/error_research.py |  |
| PEE | Checklist E: CONFIDENT-WRONG DETECTOR | research/error_research.py |  |
| PEH | Checklist H: PATTERN CHANGE DETECTION | research/pattern_change.py; research/what_changed.py |  |
| PEJ | Checklist J: "WHAT CHANGED?" INVESTIGATION TREE | research/pattern_change.py; research/what_changed.py |  |
| PEK | Checklist K: KNOWABILITY CLASSIFICATION | research/pattern_change.py; research/what_changed.py |  |
| PEL | Checklist L: 5–10% SELECTION CONSTRAINT | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PEM | Checklist M: DO NOT GAME THE 5–10% STATISTIC | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PEN | Checklist N: OPTIMAL EXIT RESEARCH | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PEO | Checklist O: EXIT INDEPENDENCE RULE | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PEP | Checklist P: ±1 PERCENTAGE-POINT CALIBRATION TARGET | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PEQ | Checklist Q: SELF-CORRECTING PREDICTION NETWORK | research/calibration_target.py; research/exit_research.py; research/selection_constraint.py; research/self_correct.py |  |
| PES | Checklist S: REPEATED ERROR ESCALATION | research/error_research.py |  |
| PET | Checklist T: SELF-RESEARCH LOOP | research/error_research.py |  |
| PEU | Checklist U: COMPUTE ALLOCATION | research/error_research.py |  |
| PEV | Checklist V: EVERY DISCOVERY MUST BE TESTABLE | research/pattern_change.py; research/what_changed.py |  |

## 2. Section-4 anti-bare-minimum gate, per subsystem (with section-24 TECHNICAL and SCIENTIFIC verdicts)

### Evidence keys

- **[R]** `scripts/reachability.py` (default mode): 64/64 learning modules REACHED; hooks 69 REACHED, 7 RESEARCH-ONLY, 0 UNREACHED.
- **[RR]** `scripts/reachability.py --package engine.research --show all`: 47 REACHED, 15 UNREACHED, exit 1. The per-module
  "definitions reached" fractions are quoted where they are low. The script says REACHED is an upper bound and UNREACHED is
  trustworthy.
- **[PL]** One planted cycle, `scripts/research_loop.py --once --checkpoint off --root <scratchpad>`: stage status with n_in/n_out
  from `reports/cycle_00000.json`. It gave 27 OK and 24 SKIPPED_NO_INPUT; knowledge 0; section-47 chains 0; knowledge->decision
  chains 0.
- **[PROD]** Production-path evidence on disk:
  - `state/learning/` is absent;
  - `state/research/research_loop/` is absent;
  - no `legit` key appears in any loop2 round checkpoint;
  - `livesim_loop2.py:27-39` has `--learner` default `off`;
  - `scripts/research_loop.py` docstring: "only the planted source has been run".
- **[T]** Tests that import the module (by regex over `tests/`), with grep counts in those files for determinism words
  (determinis/identical/reproduc) and adversarial words (leak/future/FirewallBreach/tamper/refuse/forged/raises).
  These are counts, not proofs.
- **[J]** Masterstock `JOURNAL.md` acceptance entries (29 Sep 15:23-16:44) and INTEGRATION.md known defects.
- **[S]** Scientific result files named in STATUS.md and RESEARCH_MAPPING.md ("Evidence already on disk").

Cells read YES / NO / PARTIAL / UNKNOWN. "Data in" means real or planted data enters the module on the production path; in this
ledger that is the research loop [PL]. "Output->next" means its output measurably changed a downstream stage in [PL].
"Prod path" means the module has actually been used by a process that runs, not just statically reached.

| # | Subsystem (modules) | Exist | Reach | Data in | Output->next | Persist | Prov | Determ | Fail-safe | Adversarial | Prod path | Sci valid | TECHNICAL | SCIENTIFIC/BEHAVIOURAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Research loop + controller + two-stage (`research/loop.py`, `controller.py`, `two_stage.py`) | YES | YES [RR] loop 165/169, two_stage 35/38 | PARTIAL: planted only; 24/51 stages SKIPPED_NO_INPUT [PL] | NO: two_stage 80 in -> 0 out; knowledge 0; section-47 chains 0 [PL] | YES: stage/cycle checkpoints (`loop.py` state_*.pkl, :845) | YES (provenance.json written) | PARTIAL: kill/resume invariant tested (test_research_loop.py 23 passed); no replay-equality test on a full cycle | YES: stage errors -> FAILED/REFUSED_LEAK/SKIPPED, cycle continues (`loop.py:2523-2549`) | PARTIAL: FirewallBreach -> REFUSED_LEAK; no planted-leak run through the loop yet (W02 item 4d) | NO [PROD] | NO | RUNS (planted cycle 4.8 s, exit 0) | **NOT WORKING AS INTENDED**: it cannot yet carry a single planted pattern to a decision change |
| 2 | Feeds + evidence bundle (`research/feeds.py`, `evidence.py`) - **W02 IN PROGRESS** | YES (evidence untracked) | **NO**: UNREACHED [RR] | NO (not wired into the loop yet) | NO | UNKNOWN | YES by text (32 prov refs) | UNKNOWN | UNKNOWN | UNKNOWN (0 tests import them yet) | NO | NO | IN PROGRESS | IN PROGRESS - the fix for rows 1, 3-18 depends on it |
| 3 | Research/trader firewall + namespaces (`research/firewall.py`, `namespaces.py`; learning `trader_view`, `curator`, `firewalls`, `future_firewall`, `memory_firewall`, `identity_firewall`) | YES | YES; firewall 60/119 defs [RR]; learning modules REACHED [R] | NO: `update.firewall_release` SKIPPED_NO_INPUT [PL]; curator store never created [PROD] | NO: nothing released in [PL] | PARTIAL: curator store root configured (`livesim_loop2.py:40`) but never written | YES | PARTIAL | PARTIAL: **3 firewall checks fail OPEN on ImportError** (`firewalls.py:527`, `future_firewall.py:318`, `memory_firewall.py:584`) | YES in unit tests: R01 refused 28 planted leaks [J]; trader_view 550 adversarial hits [T]; `assert_trader_path_clean` in CI | NO | PARTIAL: leak audit has 1 LEAK (channel 6, m_* identifies the year; answered by C64 design, not closed) and 4 QUARANTINED [S: S18] | WORKS in unit tests | **NOT PROVEN**: never exercised on a real release; H02/H03/L13 FAILED |
| 4 | Market observer + daily autopsy (`observer.py`, `autopsy.py`) | YES | YES; observer 76/126, autopsy 66/135 | **NO**: both SKIPPED_NO_INPUT [PL] | NO | YES (ObserverLedger, AutopsyLedger) | PARTIAL (autopsy 0 prov refs) | PARTIAL (8/2 det hits [T]) | YES (FirewallBreach paths) | PARTIAL | NO | NO | WORKS in unit tests (~55 ms per 3,000-name day [J]) | NOT VERIFIED: the market has never been observed; C69 section 8.2 all open |
| 5 | C67 mover-episode lab (`episodes.py`, `episode_paths.py`, `precursors.py`) | YES | YES; episode_paths 15/31 | **NO**: precursors and sweeps SKIPPED_NO_INPUT [PL] | NO | YES (coverage book, `_atomic_text`) | YES | PARTIAL | YES | YES (planted volume spike found; noise world 0 candidates over 7 seeds [J]) | NO | NO | WORKS in unit tests | NOT VERIFIED: 0 real episodes studied; C69 section 9 all open |
| 6 | Volatility lab + H1-H9 hypotheses (`volatility_lab.py`, `vol_hypotheses.py`) | YES | YES; volatility_lab **70/148** | PARTIAL: planted 3,120 rows in, 1 out [PL] | PARTIAL: feeds controller "18% compute to volatility" [PL] | YES | YES | PARTIAL (2 det hits) | YES | YES (planted/null worlds, 18 studies [J]) | NO | NO: the only real volatility evidence is the older fv run1 (57.9% vs 14.6%), **survivor-only** [S] | WORKS | NOT VERIFIED: section 10 gate (transfer across periods, stocks, sectors, regimes) not run; **a 5th mover model** (see section 4) |
| 7 | Direction lab + accuracy frontier (`direction_lab.py`, `frontier.py`) | YES | YES | PARTIAL: direction_lab 1,600 in, 1 out; **frontier SKIPPED_NO_INPUT** [PL] | NO (direction gated: "waits for volatility evidence") | YES | YES | PARTIAL | YES | YES (truncation test caught 2 own future-reading features [J]) | NO | Real: 51.8% coin flip; 0 cells at 80% [S] | WORKS | NOT WORKING AS INTENDED (no direction edge exists yet; C69 section 11 open) |
| 8 | Winner/loser research, loss pipeline, symmetry (`winners_losers.py`, `loss_pipeline.py`, `symmetry.py`) | YES | YES | **NO**: decision_losses, winners_losers, loss_pipeline and symmetry all SKIPPED_NO_INPUT [PL] | NO | PARTIAL (LossRiskBank) | YES | PARTIAL (symmetry 0 det hits) | YES | YES (10 planted causes named exactly [J]) | NO | NO | WORKS in unit tests | NOT VERIFIED |
| 9 | Missed winners/losers (`research/missed.py` over `engine/missed_winners.py` and `learning/missed_winners.py`) | YES | YES; S05.missed_week hook REACHED [R] | **NO**: missed SKIPPED_NO_INPUT [PL] | NO | YES (NearMiss/Filter/Path ledgers) | PARTIAL | PARTIAL | YES | PARTIAL | NO | **FAILED**: D13 real uplift 0, p 0.64 [S] | WORKS | **FAILED** (C62 D13) |
| 10 | Knowability, counterfactual, unknown cause (`knowability.py`, `counterfactual.py`, `unknown_cause.py`) | YES | YES; knowability **101/213**, unknown_cause 51/103 | **NO**: all three SKIPPED_NO_INPUT [PL] | NO | PARTIAL | YES (counterfactual 76, knowability 124 prov refs) | PARTIAL | YES (ClassificationGate is the only release door) | PARTIAL: **known defect - null world labelled POTENTIALLY_PREDICTABLE in 6/10 seeds** (INTEGRATION.md R03) | NO | NO (RS04 NOT_STARTED) | WORKS | **DEFECTIVE**: false positives on null worlds |
| 11 | Pattern discovery + interactions (`discovery.py`, `discovery_sources.py`, `interactions.py`) | YES | YES | **NO**: discovery and interactions SKIPPED_NO_INPUT [PL]; feature_screen OK (35 in, 6 out) | NO | YES | YES | PARTIAL | YES | YES (beta-neutral control; null worlds 0 trusted findings [J]) | NO | Real miner: 0.14 live patterns per refit, rank IC ~0 [INTEGRATION.md B04] | WORKS | NOT VERIFIED; the known limit is that an XOR pair with no marginal association can be dropped [J R15] |
| 12 | Pattern-break research (`break_research.py` over `learning/break_detection.py` and `pattern_reliability.py`) | YES | YES; S09.signals_from_health REACHED via research_loop [R] | **NO**: breaks.knowledge and break_research SKIPPED_NO_INPUT [PL] | NO | **NO** (0 persistence refs) | YES | **UNKNOWN** (0 det hits [T]) | YES | PARTIAL | NO | Real: reliability AUC 0.4975; all 25 real breaks "unknown cause" [S] | WORKS | NOT WORKING AS INTENDED (breaks are not predictable on real data) |
| 13 | Regimes, cross-section, multiscale (`regimes.py`, `cross_section.py`, `multiscale.py`) | YES | YES | PARTIAL: regimes 4 in, 0 out; **cross_section and multiscale SKIPPED_NO_INPUT** [PL] | NO | **NO** (0 persistence refs in any of the three) | PARTIAL | PARTIAL | YES | YES (two-regime world gives k=2 at every refit [J]) | NO | NO | WORKS | NOT VERIFIED; C69 section 14 open |
| 14 | Research agenda: priority, targets, questions, hypothesis tree (`priority.py`, `targets.py`, `questions.py`, `hypothesis_tree.py`) | YES | YES; questions **66/138** | PARTIAL: generate 7 in / 7 out, trees 7/7, priority 7 -> 1; **targets SKIPPED_NO_INPUT** [PL] | YES (1 experiment selected and launched) | YES (targets, questions and tree have save paths) | **NO** (priority, targets, questions and tree have 0 prov refs) | PARTIAL | YES | YES (null-world defects caught twice [J R15]) | NO | NO (RS15 research-policy transfer unmeasured) | WORKS | PARTIAL: generates and ranks questions, but on no real failures |
| 15 | Compute governance: compute_manager, value_accounting, waste, diversity, brain_health | YES | YES; diversity **37/90**, compute_manager 74/139 | PARTIAL: compute 1/1, diversity 0 -> 10, waste 1/0, brain_health **0/0** [PL] | PARTIAL | YES | PARTIAL (waste 0 prov refs) | **NO** evidence (0 det hits [T] for compute_manager, value_accounting, waste) | YES | PARTIAL | NO | NO (RS18 unmeasured) | WORKS | NOT VERIFIED |
| 16 | Experiments, meta-research, failed-learner lab (`experiments.py`, `meta_research.py`, `failed_lab.py`) | YES | YES | PARTIAL: experiment_memory 0/0, meta_research 0 -> 1, failed_lab 1 -> 0 [PL] | NO | YES | YES | PARTIAL | YES | PARTIAL | NO | NO (RS16, RS17 unmeasured) | WORKS | NOT VERIFIED |
| 17 | Knowledge -> decision: research_graph, science_memory, decision_bridge | YES | YES; science_memory **62/143** | PARTIAL: research_graph 0/0, science_memory 1/1; **decision_bridge SKIPPED_NO_INPUT** [PL] | **NO**: knowledge->decision chains 0 [PL] | PARTIAL (the science_memory graph fold is not idempotent; W02 is fixing it) | YES | PARTIAL | YES (falsifiers cannot be loosened) | YES | NO | NO | WORKS | **NOT WORKING AS INTENDED**: no knowledge reaches a decision |
| 18 | Replication, quality gate, research scorecard (`replication.py`, `quality_gate.py`, `research/scorecard.py`) | YES | YES | **NO**: replication 0/0; quality_gate and scorecard SKIPPED_NO_INPUT [PL]; W01 saw 10/14 gates lacking evidence | NO | PARTIAL (ReplicationLedger only) | YES | PARTIAL | YES: missing evidence blocks (fail closed) | YES | NO | NO: nothing promoted, ever (RG06) | WORKS | NOT VERIFIED (the gate has never seen a candidate) |
| 19 | C68 expectations, outcomes, prediction error (`expectations.py`, `outcomes.py`, `prediction_error.py`) | YES | **NO**: UNREACHED [RR] | NO | NO | YES (hash-chained ChainFile, `test_engine_persists_and_reloads`) | PARTIAL (outcomes and prediction_error 0 prov refs) | YES (`test_engine_step_is_incremental_deterministic...`) | YES | YES (rewrite, late write and future info refused; forged chain caught) | NO | NO; GAP: nothing supplied trajectory fields until P05's PathModel, and that is not wired | WORKS in unit tests (59) | **DEAD** until P06 wires it |
| 20 | C68 error research + confident-wrong (`error_research.py`) | YES | **NO** [RR] | NO | NO (its step feeds `research/priority` only in tests) | **NO** (0 persistence refs; DepthLedger is in-memory) | PARTIAL | PARTIAL | YES | YES | NO | NO | WORKS in unit tests (26) | DEAD |
| 21 | C68 pattern change + what-changed tree (`pattern_change.py`, `what_changed.py`) | YES | **NO** [RR] | NO | NO | **NO** (0 persistence refs) | PARTIAL | YES (`test_investigation_is_deterministic`, scrambled-future test) | YES | YES | NO | NO | WORKS in unit tests (51) | DEAD; and it duplicates break research (section 4) |
| 22 | C68 5-10% selection, learned exits, +-1pp target, self-correction (`selection_constraint.py`, `exit_research.py`, `calibration_target.py`, `self_correct.py`) | YES | **NO** [RR] | NO | **NO**: selection does not gate two_stage or adaptive; exits do not control any real exit | **NO** (0 persistence refs in all four) | PARTIAL (exit_research 0 prov refs) | PARTIAL | YES | YES (exits identical when the +-1pp target changes; target-aware exit caught; every abuse detected: tests passed in this audit) | NO | NO | WORKS in unit tests (31) | **DEAD**: C69 sections 16 and 17 are the classic "filter bypassed by another selector" (section 31) until wired |
| 23 | C68 market expectation, change points, regime memory (`market_expectations.py`, `change_points.py`, `regime_memory.py`) - **P03 IN PROGRESS** | YES | **NO** [RR] | NO | NO | PARTIAL (checkpoint round-trip test) | PARTIAL | YES by test name (forward-only, bit-identical under a scrambled future) | YES | YES | NO | NO | IN PROGRESS | IN PROGRESS |
| 24 | C62 blind learner loop (`learner.py`, `loop_hooks.py`, `test_path.py`, `curator.py`, `trader_view.py`) | YES | YES [R]: all 19 section-4 stages; S19.run_day REACHED | PARTIAL: planted only (acceptance_mini, worker_smoke) [PROD] | **NO**: SHADOW ONLY; picks never trade (INTEGRATION_AUDIT section 3) | **NO**: `state/learning/curator/` never created | YES | PARTIAL | YES | YES | **NO**: `--learner` defaults to off | **FAILED**: C03/E04/I16/I18 FAILED; acceptance_mini improved 1/3 seeds | RUNS on planted data | **NOT WORKING AS INTENDED** |
| 25 | C62 knowledge stack (`knowledge`, `epistemic`, `belief`, `archive`, `knowledge_graph`, `interpretation`, `unknowns`) | YES | YES [R] | PARTIAL: through the learner shadow only | PARTIAL | PARTIAL (archive ChainFile; the hub root is None in production per INTEGRATION_AUDIT section 1b) | YES | PARTIAL | YES | YES | NO | NO | WORKS | NOT VERIFIED |
| 26 | C62 promotion, champion, decision contract (`learning/promotion.py`, `learning/champion.py`, `decision_contract.py`) | YES | YES; wiring.promotion_gate is **RESEARCH-ONLY (parked live path)** [R] | NO real candidate | NO | PARTIAL | YES | YES (det hits 29/33) | YES | YES (planted defect per gate) | NO | NO: L23 no knowledge ever promoted | WORKS | NOT VERIFIED |
| 27 | C62 failure learning (`failure`, `postmortem`, `credit`, `separation`, `lessons.py`) | YES | YES; wiring.post_mortem RESEARCH-ONLY [R] | PARTIAL: research scripts only | NO | PARTIAL | YES | PARTIAL | YES | YES | NO | **FAILED**: D15 lessons harmful -0.188%/wk | WORKS | **FAILED** |
| 28 | C62 health, reliability, lifecycle, retirement (`health`, `reliability`, `lifecycle`, `retirement`, `break_detection`) | YES | YES [R] | PARTIAL: learner shadow only | NO | PARTIAL | PARTIAL | PARTIAL | YES | PARTIAL: **KNOWN**: the B27 health monitor false-alarms on stationary series (winner's curse); the S15 placebo let 1/6 through | NO | NO | WORKS | **DEFECTIVE** (false alarms) |
| 29 | C62 trader-side research brain (`research_priority`, `research_policy`, `meta_learning`, `surprise`) | YES | YES; S09.propose_selected and update_from_result now REACHED via test_path [R] | PARTIAL: learner shadow, planted only | PARTIAL | PARTIAL | PARTIAL (research_priority 0 prov refs) | PARTIAL | YES | YES | NO | NO | WORKS | NOT VERIFIED; RESEARCH_MAPPING risk 1: this runs INSIDE the blind trader and should shrink to consuming released agenda items |
| 30 | Legacy hook bus (`learning/wiring.py` into improve/lessons/memory/registry/patterns) | YES | PARTIAL: 7 hooks RESEARCH-ONLY (post_mortem, decision_sources, production_config, pre_launch, registry_audit, weight, promotion_gate) [R] | PARTIAL | PARTIAL: `wiring.weight` is identity while no board is configured | **NO**: HUB.root is None unless configure() is called (INTEGRATION_AUDIT section 1b; `state/learning/hub` absent) | YES | - | YES | PARTIAL | PARTIAL | NO | RUNS | **SHALLOW**: sinks do not persist |
| 31 | Production Test system (`adaptive.Session`, `livesim.py`, `scripts/livesim_loop2.py`, `basis_search`, `objective`) | YES | YES | YES (real survivor-only panel; loop2 rounds 004-008 ran 29 Sep) | YES (it trades the sim) | YES (`state/checkpoints/loop2_round*`) | YES (provenance and fill_audit gates) | YES (re-tester, Phase 23) | YES | YES (blind gates: 40 seals, 0 overlap) | **YES - the only live decision path** | **FAILED**: OOS in-band 22%, risk -0.31; dial failed its gate; timeline memory +0.00008/wk (t 0.54) [INTEGRATION.md B09] | WORKS | **NOT WORKING AS INTENDED** (misses the +7%/week band; no learning) |
| 32 | Legacy Algorithm and Find-volatility (`patterns.PatternMiner`, `fv_pipeline`, `pattern_movers`, `direction*`, `exits.py`, `stops.py`) | YES | YES | YES (real, survivor-only) | PARTIAL: fv M+D placed **0 bets over 716 weeks** (direction gate never opened) | YES | PARTIAL | YES | YES | YES | PARTIAL | PARTIAL: movement edge real (AUC 0.733); stops cut P(loss>20%) 3.26% -> 1.22% but the worst loss is -73% | WORKS | PARTIAL (movement yes, direction no, gap-loss cap no) |

**Summary.** 0 of 32 subsystems pass all eleven section-4 questions. Rows 1-29 fail "Prod path". No row has YES on "Sci valid": row 32 is
PARTIAL and every other row is NO or FAILED. C69 section 4: *IT IS NOT DONE.*

The two blocking causes common to most rows:
- (a) **no feed**: 24 stages have no input (W02);
- (b) **no production run**: nothing has been run with persistence on real data, which C63 deferred to after the foundation.

## 3. Section-5 placeholder / stub / bypass review (engine/ and scripts/, tests excluded)

Method:
- a grep for TODO|FIXME|XXX|HACK|NotImplemented|placeholder|stub|bypass|dummy|fake;
- an AST scan for pass-only, ellipsis-only and docstring-only bodies, constant returns, and `except: pass` / `except: continue`;
- the reachability dead-code lists.

Result: **no TODO or FIXME markers in engine/ or scripts/.** Every `todo` hit is a variable name.

| Site | Kind | Class | Reason |
|---|---|---|---|
| `engine/stops.py:194` StopRule.dist | NotImplementedError | LEGITIMATE | abstract base; 7 subclasses (AtrStop..HybridStop) |
| `engine/stops.py:206` StopRule._learn | pass body | LEGITIMATE | default no-learning hook; VolPercentileStop overrides (:246) |
| `engine/exits.py:337` Rule.run | NotImplementedError | LEGITIMATE | abstract rule base |
| `engine/learners.py:723` propose | NotImplementedError | LEGITIMATE | abstract learner base |
| `engine/learning/firewalls.py:333` FirewallLayer.inspect | NotImplementedError | LEGITIMATE | abstract layer base |
| `engine/learning/credit.py:266,274` Combiner | NotImplementedError | LEGITIMATE | abstract combiner |
| `engine/learning/controls.py:339` Control.decide; `:322,:334,:341` end_run/observe/attach_oracle return None | abstract + no-op hooks | LEGITIMATE | base control; the five frozen controls override |
| `engine/research/break_research.py:290,293` GateRule | NotImplementedError | LEGITIMATE | abstract gate rule |
| `engine/research/namespaces.py:439` NamespaceStore._admit | docstring-only | LEGITIMATE | both concrete stores (ResearchStore :568, LiveStore :616) override |
| `engine/research/loop.py:251,253` Feed; `feeds.py:272,274` Source | Protocol `...` | LEGITIMATE | typing Protocols |
| `engine/learning/promotion.py:271,273` | @overload `...` | LEGITIMATE | typing overloads |
| `engine/broker.py:35` LocalBroker.is_session -> None | constant return | LEGITIMATE | documented fallback; parked Live path (C65) |
| `engine/research/meta_research.py:1482` `fake = {q: OosResult(... NOT_VALIDATED, "inner fit")}` | placeholder OOS results | LEGITIMATE (review) | used only inside the outer walk-forward replay, which is the OOS test; labelled NOT_VALIDATED. Verify it cannot leak into a released advice record |
| `scripts/legacy/frontier.py:22` `ret = (... * 0)  # placeholder` | placeholder | **DEFECT (low)** | dead legacy script; retire it or move it to an archive dir so nobody reads its output as a result |
| `engine/learning/firewalls.py:527`, `future_firewall.py:318`, `memory_firewall.py:584` `except ImportError: pass` | fail-open | **DEFECT (firewall)** | a missing `blind_gates`/`leak_audit` module silently disables a firewall check. Should fail closed (raise or add a finding) |
| `engine/learning/future_firewall.py:533,1095` `except NetworkBlocked: pass` | swallow | LEGITIMATE | the blocked attempts are still returned as NetworkEvents from `g.blocked` |
| `engine/data.py:62` `except Exception: continue` in yfinance download | silent data loss | **DEFECT (data quality)** | failed ticker batches vanish without a count. This can silently thin the universe, on top of the survivor bias |
| `engine/learning/research_policy.py:1344` `except Exception: continue` on timestamp parse | silent skip | **DEFECT (minor)** | unparseable stamps should be counted or refused, not dropped |
| `engine/learners.py:470` `except Exception: pass` | broad swallow | **DEFECT (minor)** | benign (it rebuilds the panel), but it hides real errors; narrow the except |
| `engine/learning/memory_firewall.py:412` `except FirewallBreach: continue` | silent drop | **REVIEW** | breaching items are dropped from the lineage view uncounted. Confirm the breach is reported elsewhere, or count it |
| `engine/research/break_research.py:1635,2188,2205,2495` `except (…FirewallBreach…): continue` | skip | REVIEW | a skip is correct but refusals are not counted; a silent-failure risk (C69 section 26) |
| `engine/research/discovery.py:1716` `rec.gate(now)` FirewallBreach -> continue | filter | LEGITIMATE | immature records are withheld by design |
| `engine/learning/scorecard.py:148`, `engine/resources.py:47,61`, `engine/repro.py:207,213`, `engine/research/loop.py:845` | best-effort swallow | LEGITIMATE | metering/probe/cleanup; each documents "never a fake number" or has a reasoned exemption (INTEGRATION.md B17) |
| `learning/wiring.py` `weight` hook | identity adapter | **DEFECT (shallow)** | returns the legacy weight unchanged while HUB.board is None; RESEARCH-ONLY [R] |
| `learning/wiring.py` sinks (`HUB.root None`) | non-persisting adapter | **DEFECT (shallow)** | every sink except the experiment ledger is lost at exit (INTEGRATION_AUDIT section 1b) |
| `improve.py` promotion_gate (`claims_learning` never True) | bypassed gate | **DEFECT (bypass)** | the scorecard, firewall and identity checks never execute (INTEGRATION_AUDIT S10); now also only via the parked live path [R] |
| 15 UNREACHED engine.research modules [RR] | dead adapters | **DEFECT (integration)** | all C68 modules plus feeds/evidence: code that claims prediction-error, selection and exit behaviour but runs nowhere |
| low definition coverage: volatility_lab 70/148, knowability 101/213, questions 66/138, science_memory 62/143, diversity 37/90, autopsy 66/135 [RR] | partly unused implementation | REVIEW | about half of each module's defs are unreached even by the generous static graph; check for dead helpers vs. entry points only tests call |
| `scripts/_save_*.py`, `arch_doc.py`, `bible_trace.py`, `reachability.py`, `quality_gate.py`, `site_*.py` except/continue | tooling | LEGITIMATE | tooling that skips unparsable files and reports them |

No hard-coded verdicts or constant scores were found in research or learning decision code. The AST scan found no function that
returns a constant score from real inputs. Two built-in detectors look for exactly this pattern:
`replication.py:947 constant_effects` and `unknowns.py:407` ("0.5 placeholder").

## 4. Section-6 duplication check (parallel implementations of one job)

| Conceptual job | Implementations | Canonical (intended) | Actually used by a running process | Verdict |
|---|---|---|---|---|
| Predict big movers (volatility) | `fv_pipeline.MoverStage`; `pattern_movers.MoverModel` / `PatternMoverModel`; `direction_features.mover_walk_forward`; `scripts/movers.py`; `learners.MoverUseLearner`; **`research/vol_hypotheses.fit_hypothesis` (own LightGBM, :458)** | RESEARCH_MAPPING WP3: fv_pipeline.MoverStage as the single champion | adaptive.Session (production) uses none of the lab models; volatility_lab uses vol_hypotheses and imports only `pattern_movers.auc` | **UNRESOLVED; grew from 4 to 5.** The lab does not include MoverStage as a competitor, so lab and fv results are not comparable |
| Pattern stores | `pattern_bank`, `pattern_memory`, `pattern_lifecycle` / `pattern_identity`, `learning/knowledge.KnowledgeStore`, `research/research_graph` + `science_memory` | knowledge.KnowledgeStore (R10 files discoveries there) | loop2/adaptive read pattern_memory / pattern_bank; the research side writes KnowledgeStore | UNRESOLVED ("choose ONE store" still open) |
| Pattern degradation / break | `learning/break_detection`, `engine/pattern_reliability`, `research/break_research` (wraps both), **`research/pattern_change` (P04, imports learning health/lifecycle/reliability, NOT break_research)**, `learning/health` | break_research (the WP10 "one break engine") | none in production | **NEW DUPLICATE**: pattern_change is a 2nd break/degradation classifier beside break_research |
| Knowability classification | `research/knowability` (PREDICTABLE..DATA_FAILURE), **`research/what_changed` five-way K (P04)**, `unknown_cause`, `learning/unknowns`, `pattern_reliability` UNKNOWN_CAUSE | research/knowability | none | **NEW DUPLICATE**: what_changed has its own five-way map instead of calling knowability |
| +-1pp calibration statistic | **`prediction_error.honest_tolerance` (P01, :459)** and **`calibration_target.tolerance_profile` / `naive_share` (P05, :361, :464)** | not named | neither (both unreached) | **NEW DUPLICATE**: two answers to the same C68 section-P question |
| Experiment ledgers | `registry.py`, `engine/experiment_memory.py`, `learning/experiment_memory.ExperimentLedger`, `learning/compute.ExperimentLedger` (same class name, execution ledger), `research/experiments.py`, `improve` experiments.jsonl, `research_policy.RoundLedger` | learning/experiment_memory (front) | improve.log_experiment -> wiring.import_legacy; research loop -> research/experiments + compute ledger | UNRESOLVED (3 known + 2 name-clashing) |
| Firewalls | learning firewalls (8 layers), future_firewall, memory_firewall, identity_firewall, trader_view, curator, blind_gates, leak_audit, pit, isolation, research/firewall + namespaces | trader_view + curator for the trader; research/firewall as a new LAYER | loop2 uses blind_gates + fill_audit; the learner shadow uses the learning firewalls | ISOLATED by design (research/firewall imports FW, MF, TV). Still 11 surfaces; OK only if the layering is documented |
| Regime detection | `learning/situation.classify_regime`, `learning/context`, `learning/hierarchy`, `research/regimes` (own k-means via cross_section), P03 `change_points` / `regime_memory` (import regimes) | research/regimes | none in production | PARTIAL: P03 extends regimes (good); situation/context remain parallel on the trader side |
| Missed winners | `engine/missed_winners`, `learning/missed_winners`, `research/missed` (layered) | layered (research extends both) | adaptive calls MissedLedger.add; observe() hook REACHED [R] | OK (extension, not a copy) |
| Priority / questions / scorecard | `learning/research_priority` + `research/priority`; `learning/questions.QuestionEngine` + `research/questions` (extends research_priority, **not** learning/questions); `learning/scorecard` + `research/scorecard` | research extends learning | learner shadow uses learning; loop uses research | PARTIAL: learning/questions.QuestionEngine is a parallel question generator |
| Loss-risk bank / trial ledger | `loss_pipeline.LossRiskBank` <- `symmetry.LossRiskBank`; `interactions.TrialLedger` <- `discovery.TrialLedger` <- `frontier.TestLedger` | subclassed (fixed in R09/R10) | - | RESOLVED |
| Hash-chained logs | `archive.ChainFile` (used by expectations, science_memory) | ChainFile | - | RESOLVED (R19's ChainLog removed) |
| Health ledgers | `pattern_reliability.HealthLedger` vs `research/brain_health.HealthLedger` | different jobs (pattern vs research-brain) | - | NAME CLASH only; rename to avoid confusion |

## 5. Categorised known issues (section 2)

**Scientific limitations**
1. No direction edge on movers: 51.8%, Brier equals the base rate; 0 cells reach 80% (memory; direction2 run1). RS05-RS09.
2. The 7% band is structurally hard: in-band vs risk -0.69; OOS in-band 22% (memory; INTEGRATION.md B09).
3. Pattern reliability is unpredictable (AUC 0.4975) and all 25 real breaks have unknown cause (STATUS; pattern_reliability report).
4. Learning curve 0/6 windows rising; the old +0.23%/wk was void (leak) (STATUS; B22 lc1). C62 E04 FAILED.
5. Cross-year NO_TRANSFER; lessons harmful -0.188%/wk; memory skill negative (C62 I16, I18, D15).
6. Timeline memory has no OOS value (+0.00008/wk, t 0.54) (STATUS; B26).
7. Stops cannot cap gap losses (worst -73%) (RS10-RS12; memory).
8. Planted C62 acceptance improved 1/3 seeds (acceptance_mini summary.json). Planted regime/unless patterns admitted only
   0.12-0.50 (C62 I10, I13).
9. Movement edge real (57.9% vs 14.6%; AUC 0.733) but unproven off the survivor panel (RESEARCH_MAPPING evidence).

**Integration gaps**
1. 15 engine.research modules UNREACHED: all C68 plus feeds/evidence [RR].
2. 24/51 loop stages SKIPPED_NO_INPUT; knowledge 0; no knowledge->decision chain [PL]; W02 brief.
3. The quality gate cannot promote: 10/14 gates lack evidence (journal 16:23).
4. The science_memory graph fold is not idempotent, so the loop skips it (journal 16:23; W02).
5. The C62 learner is shadow-only and off by default; the curator store has never been created (INTEGRATION_AUDIT section 3).
6. Hub sinks do not persist (configure() is never called); wiring.weight is identity; promotion_gate is degenerate
   (INTEGRATION_AUDIT section 1b, S10).
7. C68 selection and exit modules do not control two_stage/adaptive selection or exits (row 22). P06 is not started.
8. INTEGRATION.md open B-series hooks:
   - fills through PITStore.executor + audit_fills in backtest/livesim;
   - pit.purged_training_set in train.py;
   - future_scramble_store over the real pipeline;
   - parity.require_parity in live;
   - TrustTable/DirectionEngine hooks;
   - checkpoint/run_report/Board.promote;
   - Board.record_shadow_session;
   - experiment_memory.check before grids.
9. The research engine runs inside the blind learner (RESEARCH_MAPPING design risk 1): stage_research should consume released
   agenda items only.
10. Reachability REACHED counts are an upper bound. Low definition coverage in several research modules [RR].

**Firewall concerns**
1. Leak channel 6 (m_* market context identifies the year) is still LEAK; C64 answers it by design but it is not closed
   (C62 H02, L13).
2. 4 channels QUARANTINED: survivorship; metadata; META_DEFAULT learned-state residual; 8d. train_basis is ungated
   (C62 H03).
3. Three firewall checks fail open on ImportError (section 3).
4. Same-year rerun leak: research filed under year Y must not release during a disguised replay of Y. Enforced in
   counterfactual/decision_bridge/episodes by unit test only; never exercised in a real run (RESEARCH_MAPPING risk 1b).
5. Hindsight labels (knowability classes, "unpredictable") must never become trader training labels without a PIT gate
   (RESEARCH_MAPPING risk 1c). No integration test proves this.
6. The curator release design gap (weights sum to 1) was fixed with a strength band (INTEGRATION_AUDIT S19). Verify it on
   real data.
7. Live/sim parity: live decides at 15:42 ET while sims decide at close and fill at the next open (INTEGRATION.md B01).
   Parked by C65.
8. trader_view rejects counts near 1900-2100 as years; public summaries send log10 (journal 15:43). This is a
   false-positive risk, not a leak.

**Data-quality problems**
1. panel.parquet has no candle columns: the 25 candle signals reach the miner only if merged (INTEGRATION.md B14).
2. m_* context columns were never miner candidates; 12 sparse flag features fill only 2-4 quintiles (B14).
3. Insider data: 146 impossible rows were dropped at load (fixed, B01). Saturday filings need a backward busday roll (B01 note).
4. `engine/data.py:62` silently drops failed download batches (section 3).
5. The pit scrambled store shifts fabricated rows only ~400 days, so stale test histories fail invariance spuriously
   (journal 15:24).
6. Panel-wide normalisation reads the future: R08 caught two of its own features (journal 15:40). Every feature that uses
   panel-wide stats needs auditing.
7. Registry provenance is present on only 3.8% of historical records (C62 G01, L14 FAILED).

**Shallow implementations**
1. wiring.weight identity; non-persisting hub sinks; degenerate promotion gate (section 3).
2. Many stages report OK with 0 in and 0 out (experiment_memory, research_graph, replication, brain_health) [PL]. Section 31
   calls this "technically satisfies the checklist while failing the objective".
3. Loop stages are gated on inputs that no feed yet provides (W02).
4. C66 line budgets were met on the ruler (36/37) but function was not proven. C69 section 32 says to investigate depth, not
   count.

**Duplicates**: see section 4. The critical ones are the 5th mover model, pattern_change vs break_research, what_changed K vs
knowability, and the two +-1pp statistics.

**Survivor-only limitations**
1. The panel has 5,243 tickers, of which only 2 end before the last date; it is 100% the current universe (INTEGRATION.md B01;
   PIT_AUDIT).
2. The delisted fetch covered 9,250 terminal events, resolved 307 tickers and recovered prices for only **77** names
   (`state/research/delisted/report.md`). The survivor bias is essentially unmitigated.
3. Every real result is affected, including all C66 VALIDATED RS items and the movement edge. RG04 says "CURRENTLY VIOLATED".
4. No label or gate yet stops survivor-only results from becoming final evidence (C69 section 23).

**Future-leak concerns**
1. Channel 6 (year from m_* context). Mitigated by the C64 curator but not closed.
2. Features normalised by panel-wide statistics (above).
3. Hindsight research outputs reaching the trader: only the ClassificationGate/MaturedRecord.gate doors exist, and they are
   unit-tested only.
4. The macro revision risk check (`leak_audit.macro_revision_risk`) is skipped silently if its import fails (section 3).
5. Regime/change detection must be forward-only. P03 has a scrambled-future test, but P03 is still in progress.
6. The model labels enter at close t while the real fill is at the next open. This was fixed for Test via
   features.labels(entry='open'); Live is unchanged (B01).

## 6. Prioritised work list (C69 section 36 order)

Each row is a work item. Ids W-nn are stable; append, never renumber.

**P1: blockers to the foundation**
- W-01 Finish W02 (feeds + evidence): zero SKIPPED_NO_INPUT on the planted world, and the full section-47 chain
  (question -> experiment -> replication -> PROMOTE -> firewall -> curator -> two_stage decision change -> outcome -> new
  question). Noise must never be promoted. The science_memory fold must be idempotent. (Rows 1-18.)
- W-02 P06: wire C68 into the loop and into the decision path. The 13 UNREACHED modules become REACHED; expectations are
  recorded before every two_stage decision; errors feed error_research -> priority; selection_constraint gates two_stage
  picks; the learned exit controls exits. Prove it with `reachability.py --package engine.research --fail-on all` exit 0.
- W-03 Survivor bias.
  - Label every survivor-only VALIDATED item: C66 RS01, RS05-RS12, RS14, and the relevant C62 items.
  - Add a machine gate so survivor-only evidence cannot be final.
  - Scale up delisted price recovery beyond 77 names (RG04, C69 section 23).

**P2: broken core**
- W-04 Knowability null-world defect: 6/10 null seeds read POTENTIALLY_PREDICTABLE (row 10).
- W-05 B27 health monitor false alarms (winner's curse), and the S15 placebo that let 1/6 through (row 28).
- W-06 The quality gate must be able to reach PROMOTE on a planted genuine effect, which it never has. Close the 10 evidence
  gaps through evidence.py.
- W-07 The C62 learner: acceptance_mini improved 1/3 seeds. Diagnose on the planted world before any real run (C03/E04 FAILED).

**P3: security, anti-cheating and firewalls**
- W-08 Make the 3 `except ImportError: pass` firewall checks fail closed.
- W-09 Integration adversarial suite (C68 PZ01-PZ20; C66 RT07-RT09) run through the loop. Today these tests exist at unit
  level only.
- W-10 Leak channel 6 and the 4 QUARANTINED channels: re-audit with the curator path running; gate train_basis (C62 H02,
  H03, L13).
- W-11 An integration test for the same-year rerun leak and the hindsight-label leak (research -> trader) in a real disguised replay.

**P4: integration failures**
- W-12 Run the research loop at its default root on the planted feed with persistence. Then run it on a real-cache frame once
  W-01 passes and C63 allows (the first persisted production run).
- W-13 `livesim_loop2 --learner legit` shadow run on real windows (creates state/learning/curator). Call wiring.configure() in
  production so the hub sinks persist and the board weights are real. Make promotion_gate claims_learning reachable
  (row 30).
- W-14 Close the open INTEGRATION.md B-series hooks listed in section 5 (PIT fills, purged training, parity, trust/direction,
  shadow sessions, experiment_memory.check).
- W-15 Update the stale checklist rows through `scripts/contract_checklist.py`: 42 C66, 25 C68 and 6 C62 (section 1). Mark
  INTEGRATION_AUDIT.md section 1a as superseded by today's reachability result.

**P5: scientific validity**
- W-16 The C69 section-10 volatility-first gate on real, survivor-corrected data: transfer across periods, stocks, sectors and
  regimes, and proof that the system does not merely pick historical big movers.
- W-17 Re-measure RS01-RS14 through the labs (not the legacy scripts) once W-03 lands.
- W-18 Evaluate knowability (RS04), research-policy transfer (RS15), meta-learning transfer (RS16), failed-learner usefulness
  (RS17) and compute efficiency (RS18).

**P6: major performance bottlenecks**
- W-19 Per-event could-I-have-known reconstruction cost at market scale. Enforce the budget: one PIT snapshot per day and the
  top-K events (RESEARCH_MAPPING risk 3). provenance.stamp costs ~0.3 s per call, so hash once per step (journal 15:35).

**P7: shallow implementations and duplicates**
- W-20 Resolve the duplicates in section 4:
  - name ONE mover champion and add MoverStage to the lab's competitors;
  - fold pattern_change into break_research;
  - have what_changed call knowability;
  - keep one +-1pp statistic;
  - choose one pattern store;
  - put one front on the experiment ledgers.
- W-21 Stages that are OK with 0 in and 0 out must report NO_INPUT, not OK (row 1; C69 section 31).
- W-22 Count silent refusals and skips (memory_firewall :412, break_research, data.py :62, research_policy :1344).
- W-23 Review the low definition coverage in volatility_lab, knowability, questions, science_memory, diversity and autopsy.
  Wire the unused capability or delete it (section 32).
- W-24 Retire `scripts/legacy/frontier.py`, and move the trader-side research brain toward consuming released agenda items
  (row 29).

**P8-P10: prediction/research weakness, improvements, polish**
- W-25 Direction: accept NO_RELIABLE_SIGNAL as a valid answer; the controller must shift compute honestly (RESEARCH_MAPPING
  risk 2).
- W-26 Exits and gap risk: sizing or event avoidance for gap losses (RS10-RS12 follow-up).
- W-27 Rename the colliding classes (two ExperimentLedger, two HealthLedger).
- W-28 An independent audit (C66 RG15, C62 L26), after W-01 to W-18.
