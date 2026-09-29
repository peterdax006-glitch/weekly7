# Contract section 88 mapping - existing WEEKLY7 code vs SELF_LEARNING_CONTRACT.md (canon C62)

Generated 2026-09-29 by a read-only audit of engine/, scripts/, tests/, state/research and state/experiments.jsonl. The machine-readable result is `state/build/SELF_LEARNING_MASTER_CHECKLIST.json` (fields status, code_paths, tests, evidence, actual_lines, notes). This is a self-audit, not the independent audit L26 requires.

Status rule used: VALIDATED only where an artefact on disk meets the item's own wording; code that exists but is unproven is `IMPLEMENTED — NOT VALIDATED`; code whose validation ran and failed is FAILED (`IMPLEMENTED — FAILED VALIDATION`); partial work is IN_PROGRESS. Note tags give the section-88 class: already implemented / partial / implemented-but-unvalidated / missing / contradictory / unsafe / duplicated / under-tested / under-depth.

Audit test run (synthetic, 9 files): 170 passed, 1 failed - `tests/test_provenance.py::test_integrity_fails_closed_on_tampered_bible` fails because `provenance.verify_integrity` now needs SELF_LEARNING_CONTRACT.md and the test's fake repo does not copy it (introduced with C62).

## Summary

| status | items |
|---|---|
| NOT_STARTED | 42 |
| IN_PROGRESS | 81 |
| IMPLEMENTED | 37 |
| TESTING | 0 |
| VALIDATED | 18 |
| FAILED | 13 |
| BLOCKED | 0 |
| **total** | **191** |

Section-88 classes (from note tags): partial 75, missing 42, already implemented 18, unsafe 11, under-tested 9, duplicated 5, contradictory 4, under-depth 3, implemented-but-unvalidated 2, missing plant 1. Tags are in addition to status: the 37 IMPLEMENTED items are the implemented-but-unvalidated class and the 13 FAILED items were validated and failed.

Line budget: **27,153 meaningful lines exist that genuinely do a contract job, against 56,700 required** (48%). 4 of 57 sections meet their minimum: Situation similarity, Future-information firewall, Reproducibility, Test infrastructure for the above. Existing depth is concentrated in firewalls, harnesses and tests; the knowledge layer itself (objects, states, graph, belief, research policy, meta-learning) is 0.

What the evidence says about learning today: every real-data learning run is negative - same-year gain is MEMORISATION (ld1, ld2), cross-year transfer is NO_TRANSFER (ld2), none of 6 cross-year learners passes (ls3), lessons from other windows cost -0.188%/week (lessons archive), the episodic memory's walk-forward skill is -0.16 and the missed-winner detector adds nothing. The controls that say so (memoriser, leak probe, no-learning) are the best-validated part of the system.

## Per-phase tables

### A FOUNDATION - IN_PROGRESS 8, NOT_STARTED 1, IMPLEMENTED 3, VALIDATED 1

| id | item | status | main code | note |
|---|---|---|---|---|
| A01 | Define KnowledgeObject. | IN_PROGRESS | engine/pattern_identity.py::PatternRecord; engine/lessons.py::Lesson | [partial][duplicated] Typed PatternRecord (validated, hashed) covers patterns only; ~15 of ~60 contract fields. Three other record types hold knowledge (lessons.Lesson, memory.Lesson, pattern_bank dicts). No KnowledgeObject. |
| A02 | Define epistemic states. | NOT_STARTED | - | [missing][contradictory] No epistemic states. Three unrelated vocabularies exist: pattern_lifecycle.STATES, pattern_memory modes (universal/local/disregarded), pattern_reliability.VERDICTS. |
| A03 | Define immutable experience records. | IN_PROGRESS | engine/pattern_memory.py::PatternMemory; engine/lessons.py::Episode | [partial] Pattern observations are append-only and hash-chained; decision episodes (lessons.Episode, memory.Memory episodes) are mutable and memory.py evicts at capacity. |
| A04 | Define situation representation. | IN_PROGRESS | engine/memory.py::context_of; engine/lessons.py::_situation | [under-depth] 7 market-context columns and a 4-field lesson situation; no identity-free situation type. |
| A05 | Define provenance schema. | IMPLEMENTED | engine/provenance.py::stamp; engine/registry.py::REQUIRED | IMPLEMENTED — NOT VALIDATED. [unsafe] Schema defined, but registry_audit ok=false (51 of 53 records missing git/canon/config/seed fields). test_integrity_fails_closed_on_tampered_bible FAILS now: its fake repo lacks SELF_LEARNING_CONTRACT.md that verify_integrity started requiring with C62. |
| A06 | Define knowledge versioning. | IN_PROGRESS | engine/trust_store.py::TrustStore; engine/pattern_bank.py::PatternBank | [partial] Versioned trust tables, content-addressed bank commits, basis lineage. No knowledge-object versioning. |
| A07 | Define lifecycle states. | IMPLEMENTED | engine/pattern_lifecycle.py::STATES,ALLOWED,Lifecycle | IMPLEMENTED — NOT VALIDATED. [partial] Bible Phase-4 states with enforced transitions; not the contract's ACTIVE/DEGRADED/DORMANT/RETIRED or birth..retirement phases. |
| A08 | Define confidence dimensions. | IN_PROGRESS | engine/pattern_stats.py::p_real; engine/pattern_reliability.py::MetaModel | [partial] truth confidence = P(real) (calibrated on planted truth); current reliability = P(hold). transfer_confidence, context_confidence, failure_risk missing. |
| A09 | Define reliability dimensions. | IN_PROGRESS | engine/pattern_reliability.py::walk_forward,HealthLedger; engine/trust.py::TrustTable | [partial][under-tested] One reliability dimension; pattern_reliability has no test file and is being live-edited. |
| A10 | Implement serialization/deserialization. | IN_PROGRESS | engine/pattern_identity.py::PatternRecord; engine/pattern_bank.py | [partial] JSON round-trips exist per store; none for a knowledge object. |
| A11 | Implement schema validation. | IN_PROGRESS | engine/pattern_identity.py::IdentityError; engine/registry.py::Registry | [partial] Validation per store; no knowledge schema. |
| A12 | Add deterministic hashing. | VALIDATED | engine/pattern_identity.py::pattern_hash; engine/pattern_bank.py::canon_hash | [already implemented] Order-independent canonical hashing, tested for determinism and tamper detection (those files passed in the audit run). Must be reused, not re-written, by the knowledge layer. |
| A13 | Add code/data/config provenance. | IMPLEMENTED | engine/provenance.py::stamp,code_hash,code_mixed,data_snapshot,config_hash | IMPLEMENTED — NOT VALIDATED. [unsafe] Works where called; not applied to most experiment records; one provenance test failing (see A05). |

### B MEMORY - IN_PROGRESS 6, NOT_STARTED 5, IMPLEMENTED 4

| id | item | status | main code | note |
|---|---|---|---|---|
| B01 | Build raw experience archive. | IN_PROGRESS | engine/pattern_memory.py::PatternMemory; state/experiments.jsonl | [partial] No L0 raw-observation layer; sealed livesim archives exist but are the test answers (must not be read). |
| B02 | Build episode archive. | IN_PROGRESS | engine/lessons.py::LessonStore,Episode; engine/memory.py::Memory | [partial][contradictory] memory.Memory evicts episodes at capacity (eviction_regret measured) - contract section 49 says never destroy evidence. |
| B03 | Build situation archive. | NOT_STARTED | - | [missing] No situation archive. |
| B04 | Build knowledge archive. | IN_PROGRESS | engine/pattern_memory.py::PatternMemory; engine/pattern_bank.py::PatternBank | [partial][duplicated] Three knowledge stores (timeline memory, pattern bank, trust store). Pick pattern_memory's hash chain as the archive core. |
| B05 | Build retrieval index. | IN_PROGRESS | engine/memory.py::Memory; engine/pattern_memory.py::PatternMemory.view | [partial] Linear kernel scans, no index; retrieval skill on real data negative (C03). |
| B06 | Build context index. | NOT_STARTED | - | [missing] Context used only as a kernel weight; no context index. |
| B07 | Build temporal index. | IN_PROGRESS | engine/pattern_memory.py::PatternMemory.view,Analytics | [partial] mature_date gating and a dated owner-side timeline; no temporal index over knowledge. |
| B08 | Build reliability index. | NOT_STARTED | - | [missing] pattern_reliability.PredLog holds per-pattern predictions but is not an index. |
| B09 | Build failure index. | IN_PROGRESS | engine/lessons.py::LessonBook,kind_report; engine/memory_diagnostics.py::error_profile_by_arm | [partial] Failures grouped by lesson kind; no index across knowledge. |
| B10 | Build contradiction index. | NOT_STARTED | - | [missing] |
| B11 | Build recovery index. | NOT_STARTED | - | [missing] pattern_memory requalifies silently; recoveries are not recorded. |
| B12 | Implement memory snapshots. | IMPLEMENTED | engine/pattern_memory.py::PatternMemory.view,audit_prefix_invariance; engine/learning_delta.py::LearnedState | IMPLEMENTED — NOT VALIDATED. Past-as-of views are prefix-invariant and learner state S0 is snapshotted per pair; the snapshot is not yet part of the reproducibility key (section 56). |
| B13 | Implement immutable historical records. | IMPLEMENTED | engine/pattern_memory.py (hash chain, ChainCorrupt); engine/pattern_bank.py | IMPLEMENTED — NOT VALIDATED. Tamper detection is tested and passing for these stores; memory.py and lessons stores are mutable, so coverage is partial. |
| B14 | Implement dynamic influence. | IMPLEMENTED | engine/pattern_memory.py::_pool,PatternMemory.view(gate=); engine/memory.py::Memory | IMPLEMENTED — NOT VALIDATED. Influence changes without deletion (universal/local/disregarded, gate hook); not shown to improve decisions. |
| B15 | Implement memory audit. | IMPLEMENTED | engine/memory_diagnostics.py::leak_check; engine/lessons.py::audit_identity | IMPLEMENTED — NOT VALIDATED. [unsafe] Per-store audits exist; leak_audit channel 4 (learned state from the future) is LEAK in the default loop. |

### C LEARNING - IN_PROGRESS 12, FAILED 1, IMPLEMENTED 2, VALIDATED 1, NOT_STARTED 1

| id | item | status | main code | note |
|---|---|---|---|---|
| C01 | Implement observe stage. | IN_PROGRESS | engine/adaptive.py::Session; engine/livesim.py | [partial] Decisions are recorded by the blind session; no OBSERVE stage of a learning loop. |
| C02 | Implement situation description. | IN_PROGRESS | engine/memory.py::context_of; engine/lessons.py::_situation | [under-depth] See A04. |
| C03 | Implement retrieval. | FAILED | engine/memory.py::Memory; engine/pattern_memory.py::PatternMemory.view | IMPLEMENTED — FAILED VALIDATION. Retrieval exists; real walk-forward skill -0.16 (CI -0.20..-0.12), IC -0.029 (memory_adapter/results.json). |
| C04 | Implement expectation formation. | IN_PROGRESS | engine/memory.py::classify_error; engine/direction.py | [partial] Expected outcome used only after the fact to classify errors; not recorded before the decision. |
| C05 | Implement decision capture. | IN_PROGRESS | engine/adaptive.py::Session; engine/learning_delta.py::Run | [partial] Picks and fills captured; confidence, prediction, retrieved memories, abstentions, uncertainty and explanation (section 3 BEFORE record) are not. |
| C06 | Implement outcome capture. | IMPLEMENTED | engine/pattern_memory.py (mature_date); engine/lessons.py::post_mortem | IMPLEMENTED — NOT VALIDATED. Outcomes captured only after labels mature. |
| C07 | Implement surprise detection. | IN_PROGRESS | engine/memory.py::classify_error; engine/run_report.py::memory_shock_events | [under-depth] FP/FN classification and z-score shocks; no surprise magnitude/direction/persistence. |
| C08 | Implement credit assignment. | IN_PROGRESS | engine/ablation.py; engine/memory_diagnostics.py::factor_attribution,factor_ablation | [partial] Component ablation, not per-decision credit. |
| C09 | Implement blame assignment. | IN_PROGRESS | engine/lessons.py::classify_kind,_counterfactual | [partial] 5 lesson kinds with a counterfactual; learned lessons hurt on the real archive (D15). |
| C10 | Implement belief updating. | IN_PROGRESS | engine/pattern_stats.py::eb_shrink; engine/trust.py::TrustTable | [partial] Shrinkage only; no prior -> evidence-strength -> posterior belief record. |
| C11 | Implement reliability updating. | IN_PROGRESS | engine/pattern_reliability.py::walk_forward,health_monitor; engine/trust_store.py::drift | [partial][under-tested] pattern_reliability live-edited by builders; no tests yet. |
| C12 | Implement condition discovery. | IN_PROGRESS | engine/pattern_reliability.py::explain_breaks,confirm_driver; engine/candidates.py::context_pair_candidates | [partial] Regime-conditional planted pattern admitted only 38-50% (I10). |
| C13 | Implement anti-condition discovery. | IN_PROGRESS | engine/candidates.py::unless_candidates; engine/pattern_reliability.py::explain_breaks | [partial] 'unless' patterns admitted directly only 12-25% (I13). |
| C14 | Implement transfer evaluation. | VALIDATED | engine/learning_delta.py::headline_transfer,run_pair,harness_selfcheck; engine/learners.py | [already implemented] The EVALUATION is validated (planted generaliser/memoriser/non-learner/leak self-check VALID, past-only pairs). Its result on real data is NO_TRANSFER - that is recorded under E04/I16, not here. |
| C15 | Implement knowledge storage. | IMPLEMENTED | engine/pattern_memory.py; engine/pattern_bank.py | IMPLEMENTED — NOT VALIDATED. [duplicated] Four stores; see B04. |
| C16 | Implement meta-learning update. | NOT_STARTED | - | [missing] |
| C17 | Implement research-priority update. | IN_PROGRESS | engine/improve.py::diagnose,spawn_challengers | [partial][contradictory] Spawns parameter challengers only - the tuning loop section 35 warns against; untested. |

### D FAILURE LEARNING - IN_PROGRESS 11, IMPLEMENTED 1, NOT_STARTED 1, FAILED 2

| id | item | status | main code | note |
|---|---|---|---|---|
| D01 | Loss classifier. | IN_PROGRESS | engine/lessons.py::classify_kind,CATEGORIES | [partial] 5 kinds vs the 14 contract causes (no FALSE_PATTERN/REGIME_CHANGE/MEASUREMENT_ERROR/UNKNOWN...). |
| D02 | Selection failure detector. | IN_PROGRESS | engine/lessons.py (bad_entry); engine/memory.py::classify_error | [partial] |
| D03 | Timing failure detector. | IN_PROGRESS | engine/lessons.py (missed_exit); engine/exits.py | [partial] No entry-timing detector. |
| D04 | Direction failure detector. | IN_PROGRESS | engine/direction.py; engine/learning_delta.py::pick_hits (dir_hit) | [partial] Direction hit rate measured; not attributed per loss. |
| D05 | Risk failure detector. | IN_PROGRESS | engine/lessons.py (oversized_loser); engine/gaprisk.py | [partial] |
| D06 | Pattern failure detector. | IMPLEMENTED | engine/pattern_lifecycle.py::detect_failure; engine/patterns.py (death) | IMPLEMENTED — NOT VALIDATED. Planted decaying patterns are not held (miner level); live break detection (pattern_reliability) is untested. |
| D07 | Context failure detector. | IN_PROGRESS | engine/pattern_reliability.py::driver_table,explain_breaks | [partial][under-tested] Live-edited; no tests. |
| D08 | Regime failure detector. | IN_PROGRESS | engine/lessons.py (regime_misread); engine/antioverfit.py::regime_split | [partial] |
| D09 | Measurement failure detector. | IN_PROGRESS | engine/pit.py::IntegrityError; engine/parity.py | [partial] Data-quality checks exist; not a loss-cause detector. |
| D10 | Unknown failure state. | IN_PROGRESS | engine/pattern_reliability.py::VERDICTS(OPEN),unknown_cause_share | [partial][under-tested] UNKNOWN exists only for pattern breaks. |
| D11 | Pattern-break analysis. | IN_PROGRESS | engine/pattern_reliability.py::explain_breaks,driver_table,onset_shifts,confirm_driver,investigate | [partial][under-tested] Success-vs-failure contrasts on market drivers with a family-wise null; builders live-editing; no test file. |
| D12 | Recovery analysis. | NOT_STARTED | - | [missing] Recovery happens (pattern_memory requalification) but is never analysed. |
| D13 | Missed-winner analysis. | FAILED | engine/missed_winners.py::MissedWinnerDetector,why_missed,walk_forward,evaluate | IMPLEMENTED — FAILED VALIDATION. Real 2017-2026: detector uplift 0 (verdict false, p_vs_control 0.64). |
| D14 | Counterfactual distinction discovery. | IN_PROGRESS | engine/missed_winners.py::missed_profile; engine/lessons.py::_counterfactual | [partial] Profiles, no OOS-validated separating distinction. |
| D15 | OOS validation of failure lessons. | FAILED | engine/antimemo.py::archive_experiment; scripts/lessons_archive.py | IMPLEMENTED — FAILED VALIDATION. Lessons from other windows: -0.188%/wk, 90% CI -0.337%..-0.019% (harmful). |

### E TRANSFER - IN_PROGRESS 6, IMPLEMENTED 2, VALIDATED 5, FAILED 1, NOT_STARTED 2

| id | item | status | main code | note |
|---|---|---|---|---|
| E01 | Same-year rerun harness. | IN_PROGRESS | engine/learning_delta.py::run_pair,play_curve; scripts/learning_delta.py | [implemented-but-unvalidated] run1->run2 works (self-check VALID); the run 1..N curve (C57) is running (lc1) and being edited by builders. |
| E02 | Disguised rerun mechanism. | IMPLEMENTED | engine/learning_delta.py::make_presentation,audit_presentation; engine/antimemo.py::disguise | IMPLEMENTED — NOT VALIDATED. [contradictory] Codes/dates are disguised and audited, but the same year's numbers are identical, so any numeric memory recognises the rerun (ld2 caveat); year-fingerprint channel 6 QUARANTINED. Section 1.2 forbids 'identifying a disguised rerun as a rerun'. |
| E03 | No-learning control. | VALIDATED | engine/learning_delta.py::NullLearner | [already implemented] Noise control = exactly 0 on 12 real pairs; planted non-learner = NO_EFFECT. |
| E04 | Legitimate learner. | FAILED | engine/learners.py (6 learners); engine/learning_delta.py::MemoryBankLearner,BasisLearner | IMPLEMENTED — FAILED VALIDATION. No learner passes past-only transfer; band_cfg lifts in_band but worsens worst5/max_dd. |
| E05 | Identity memorizer control. | VALIDATED | engine/learning_delta.py::MemoriserLearner,IdentityRecallLearner | [already implemented] Planted memoriser flagged MEMORISATION (same-year +0.088, transfer 0). |
| E06 | Random learner control. | IN_PROGRESS | engine/learning_delta.py::aggregate (shuffle ctrl) | [partial] A shuffled-outcome control exists; no random learner (control D) with random updates of equal size. |
| E07 | Leaky learner control. | VALIDATED | engine/learning_delta.py::LeakyGate,leak_gate_caught,leak_probe | [already implemented] Leak probe caught = True in every self-check. |
| E08 | Cross-year test. | VALIDATED | engine/learning_delta.py::headline_transfer,choose_pairs,causal_bank; engine/learners.py | [already implemented] The TEST is valid; result NO_TRANSFER (see E04/I18). |
| E09 | Cross-regime test. | IN_PROGRESS | engine/learning_delta.py::aggregate (by era); engine/antioverfit.py::regime_split | [partial] Era breakdowns only; no held-out-regime transfer test. |
| E10 | Cross-stock test. | NOT_STARTED | - | [missing] No held-out-stock transfer (ticker permutation tests invariance, not transfer). |
| E11 | Cross-sector test. | NOT_STARTED | - | [missing] |
| E12 | Cross-volatility test. | IN_PROGRESS | engine/antioverfit.py (type quiet/middle/wild); engine/learners.py::RegimeMapLearner | [partial] Per-volatility-type IC for the miner; no learned-knowledge transfer across volatility states. |
| E13 | Transfer ratio. | IN_PROGRESS | engine/learning_delta.py::aggregate (gap) | [partial] Gap computed, ratio not. |
| E14 | Memorization gap. | VALIDATED | engine/learning_delta.py::aggregate,verdict (gap) | [already implemented] Measured with bootstrap CI: ld1 gap +0.00259 [+0.00012,+0.00585]; ld2 +0.00095. |
| E15 | Identity gap. | IMPLEMENTED | engine/antimemo.py::archive_experiment (b - c); engine/learning_delta.py::IdentityRecallLearner | IMPLEMENTED — NOT VALIDATED. Identity gap b-c = +0.000% on the lessons archive; not yet in the learning scorecard. |
| E16 | Transfer stability. | IN_PROGRESS | scripts/learner_search.py (seeds) | [partial] Replicates by seed exist; ls2 learner self-check INVALID; no stability statistic. |

### F KNOWLEDGE GRAPH - NOT_STARTED 7, IN_PROGRESS 7, IMPLEMENTED 1

| id | item | status | main code | note |
|---|---|---|---|---|
| F01 | Supports edges. | NOT_STARTED | - | [missing] No SUPPORTS edges; no knowledge graph exists. |
| F02 | Contradicts edges. | NOT_STARTED | - | [missing] No CONTRADICTS edges; no knowledge graph exists. |
| F03 | Contains edges. | IN_PROGRESS | engine/pattern_stats.py::containment; engine/pattern_identity.py::Scope | [partial] Row containment computed for pruning; not stored as an edge. |
| F04 | Specializes edges. | IN_PROGRESS | engine/trust.py::parent_key; engine/pattern_lifecycle.py (rescoped) | [partial] Parent/child exists for trust types and rescoped patterns; not edges. |
| F05 | Generalizes edges. | IN_PROGRESS | engine/trust.py::parent_key | [partial] See F04. |
| F06 | Causes-failure edges. | NOT_STARTED | - | [missing] No CAUSES_FAILURE_OF edges; no knowledge graph exists. |
| F07 | Recovers-with edges. | NOT_STARTED | - | [missing] No RECOVERS_WITH edges; no knowledge graph exists. |
| F08 | Redundant-with edges. | IN_PROGRESS | engine/pattern_stats.py::prune_redundant; engine/patterns.py (duplicate status) | [partial] Redundant patterns are pruned, not linked (the distinction is thrown away). |
| F09 | Complements edges. | NOT_STARTED | - | [missing] No COMPLEMENTS edges; no knowledge graph exists. |
| F10 | Depends-on edges. | NOT_STARTED | - | [missing] No DEPENDS_ON edges; no knowledge graph exists. |
| F11 | Graph traversal. | NOT_STARTED | - | [missing] |
| F12 | Contradiction investigation. | IN_PROGRESS | engine/pattern_reliability.py::explain_breaks | [partial][under-tested] Explains a pattern's own worked-vs-broke periods; no investigation of disagreement between two knowledge items. |
| F13 | Parent/child shrinkage. | IMPLEMENTED | engine/trust.py::TrustTable (shrink to parent); engine/pattern_stats.py::eb_shrink | IMPLEMENTED — NOT VALIDATED. Shrinkage to parent works for trust tables; not applied to a knowledge hierarchy. |
| F14 | Knowledge lineage. | IN_PROGRESS | engine/pattern_bank.py (window history); engine/pattern_memory.py (source run id) | [partial] |
| F15 | Decision lineage. | IN_PROGRESS | engine/adaptive.py::adaptation_report,hold_reasons; engine/explain.py | [partial] Explains knob changes and top reasons; no link from a decision to knowledge ids. |

### G RESEARCH INTELLIGENCE - FAILED 1, NOT_STARTED 7, IMPLEMENTED 1, IN_PROGRESS 3

| id | item | status | main code | note |
|---|---|---|---|---|
| G01 | Experiment registry. | FAILED | engine/improve.py::log_experiment; engine/registry.py::Registry | IMPLEMENTED — FAILED VALIDATION. [unsafe] registry_audit ok=false: experiment_id/git/config/seed present on 3.8% of audited records. |
| G02 | Question generation. | NOT_STARTED | - | [missing] |
| G03 | Competing hypothesis generation. | NOT_STARTED | - | [missing] P(real)/P(hallucinated)/P(coincidence) is the only hypothesis split, and it is fixed. |
| G04 | Expected outcome generation. | NOT_STARTED | - | [missing] |
| G05 | Information-gain scoring. | NOT_STARTED | - | [missing] |
| G06 | Research priority queue. | NOT_STARTED | - | [missing] resources.JobQueue can execute a queue; nothing ranks research. |
| G07 | Duplicate experiment detection. | IMPLEMENTED | engine/experiment_memory.py::TriedIndex; engine/registry.py::fingerprint,ExperimentMemory | IMPLEMENTED — NOT VALIDATED. [duplicated] Two dedup mechanisms (TriedIndex, registry.ExperimentMemory) - merge. |
| G08 | Failed experiment memory. | IN_PROGRESS | engine/experiment_memory.py::worsened,answer_phase30; state/experiments.jsonl | [partial] Failed experiments are in the log but mostly without the fields to recognise them. |
| G09 | Failed learner memory. | IN_PROGRESS | state/research/learners/*/report.md; engine/learning_delta.py::CAVEATS | [partial] Failed learners exist only as markdown reports; no structured failed-learner store. |
| G10 | Exploration/exploitation policy. | NOT_STARTED | - | [missing] |
| G11 | Compute-aware research allocation. | IN_PROGRESS | engine/resources.py::worker_count,wait_for_memory,JobQueue | [partial] Compute-aware execution, not research allocation. |
| G12 | Meta-learning research policy. | NOT_STARTED | - | [missing] |

### H FIREWALLS - IMPLEMENTED 7, FAILED 2, VALIDATED 2, IN_PROGRESS 1

| id | item | status | main code | note |
|---|---|---|---|---|
| H01 | Point-in-time firewall. | IMPLEMENTED | engine/pit.py (PITStore, Guard, purged_training_set) | IMPLEMENTED — NOT VALIDATED. [unsafe] Firewall and real-cache audit exist; channels 2 (back-adjusted prices) and 8c (feed shape) are LEAK. |
| H02 | Future-data firewall. | FAILED | engine/leak_audit.py; engine/pit.py::future_scramble | IMPLEMENTED — FAILED VALIDATION. [unsafe] Audit works (planted leaks caught) but 3 of 12 channels are LEAK in the default blind path - not fail-closed. |
| H03 | Memory firewall. | FAILED | engine/pattern_memory.py::check_no_future; engine/blind_gates.py::check_memory_bank_causality | IMPLEMENTED — FAILED VALIDATION. [unsafe] Channel 4 LEAK: basis cfg/meta trained on future or same windows in the current loop; the lineage filter exists but is not the default. |
| H04 | Identity firewall. | IMPLEMENTED | engine/blind_gates.py::check_ticker_map,check_disguised_frame; engine/antioverfit.py::ticker_permutation,date_disguise | IMPLEMENTED — NOT VALIDATED. No-lesson results invariant under rename (0 of 45 windows change); year fingerprint QUARANTINED; name-order tie-breaks leaked 3x historically. |
| H05 | Provenance firewall. | IMPLEMENTED | engine/provenance.py::verify_integrity,stale; engine/claims.py::ClaimBook | IMPLEMENTED — NOT VALIDATED. [unsafe] See A05 (failing test, incomplete registry). |
| H06 | Code-version firewall. | VALIDATED | engine/provenance.py::code_hash,code_mixed,stale; engine/health.py::classify | [already implemented] Section-62 Test 12 equivalent: a code edit makes a worker result stale and it is excluded (tests passed in the audit run). |
| H07 | Revision-data firewall. | IMPLEMENTED | engine/leak_audit.py::pit_macro,macro_revision_risk; engine/edgar.py | IMPLEMENTED — NOT VALIDATED. Macro revisions CLEAN (channel 5); split back-adjustment (a revision of history) is channel 2 LEAK. |
| H08 | Survivorship firewall. | IMPLEMENTED | engine/leak_audit.py::inject_dead_names,survivorship_haircut,pit_universe; engine/pit.py::survivorship_report | IMPLEMENTED — NOT VALIDATED. [unsafe] Panel is survivor-only; channel 1 QUARANTINED (measured, not removed). |
| H09 | Worker-staleness firewall. | VALIDATED | engine/health.py::exclusion_report,classify; engine/resources.py::stale_report | [already implemented] Planted stale/crashed/OOM workers are excluded; tests passed. |
| H10 | Evaluation contamination firewall. | IMPLEMENTED | engine/blind_gates.py::seal_window,RevealGate,check_seal; engine/learning_delta.py::revealed_pool,causal_bank | IMPLEMENTED — NOT VALIDATED. [unsafe] Sealing works; the loop's basis learning still trains on evaluation-era windows (channel 4). |
| H11 | Cache/network firewall. | IMPLEMENTED | engine/leak_audit.py::NetworkGuard,FileAccessRecorder,import_closure; engine/isolation.py::research_mode | IMPLEMENTED — NOT VALIDATED. Network CLEAN (channel 7); run-time file reads QUARANTINED (channel 8d). |
| H12 | Fail-closed behavior. | IN_PROGRESS | engine/pit.py::FailClosed; engine/pattern_memory.py::LeakError | [partial][unsafe] Fail-closed exceptions exist per module; open LEAK channels are reported, not blocked. |

### I VALIDATION - IN_PROGRESS 4, NOT_STARTED 1, IMPLEMENTED 2, VALIDATED 9, FAILED 4

| id | item | status | main code | note |
|---|---|---|---|---|
| I01 | Unit tests. | IN_PROGRESS | tests/*.py (78 files) | [partial] Unit tests exist for current modules; pattern_reliability.py (1,543 lines) has none; no knowledge-layer modules to test yet. |
| I02 | Integration tests. | IN_PROGRESS | tests/integration/ | [under-tested] 4 files, 314 lines; no learning-loop integration test. |
| I03 | Property tests. | NOT_STARTED | - | [missing] No property-based tests (no hypothesis library). |
| I04 | Determinism tests. | IMPLEMENTED | engine/repro.py::run_twice,assert_reproducible | IMPLEMENTED — NOT VALIDATED. Determinism tests exist and pass for current modules; none for learning state. |
| I05 | Negative tests. | IMPLEMENTED | (planted-defect test per file, CONTEXT rule 5) | IMPLEMENTED — NOT VALIDATED. Negative tests exist per module; not per contract subsystem. |
| I06 | Planted strong pattern. | VALIDATED | engine/planted.py; scripts/planted_calibration.py | [already implemented] strong_detection 1.000 (rule >= 0.9). Miner-level; the knowledge layer must be re-run. |
| I07 | Planted weak pattern. | VALIDATED | engine/planted.py | [already implemented] weak admitted 1.00 standard / 0.88 heavy tails, sign 1.0. Not a verdict criterion; miner-level. |
| I08 | Planted negative pattern. | VALIDATED | engine/planted.py | [already implemented] negative_detection 1.000, sign 1.000. Miner-level. |
| I09 | Planted pair. | VALIDATED | engine/planted.py | [already implemented] pair admitted 1.00 in standard and heavy tails. Miner-level. |
| I10 | Planted regime pattern. | FAILED | engine/planted.py (regime plant) | IMPLEMENTED — FAILED VALIDATION. regime plant admitted 0.38 (standard) / 0.50 (heavy tails), effect ratio ~0.2. Not a verdict criterion, so the overall VALIDATED verdict hides it. |
| I11 | Planted decaying pattern. | VALIDATED | engine/planted.py (decaying plant) | [already implemented] decaying_not_held 0.000 (rule <= 0.2). Detection of decay DURING use (pattern_reliability 'phantom' world) is untested. |
| I12 | Planted conditional pattern. | IN_PROGRESS | engine/planted.py; engine/pattern_reliability.py::planted_world('regime') | [missing plant] No distinct 'conditional' plant; the closest (regime) fails - see I10. |
| I13 | Planted unless pattern. | FAILED | engine/planted.py (unless plant) | IMPLEMENTED — FAILED VALIDATION. unless admitted directly 0.12-0.25; 0.75-0.88 only via a child/duplicate. |
| I14 | Planted noise pattern. | VALIDATED | engine/planted.py (noise_only) | [already implemented] zero_rejected 0.000, noise_only_quiet 0.25 active/run (rule <= 1.0). |
| I15 | Hallucinated pattern. | VALIDATED | engine/planted.py (hallucinated plant) | [already implemented] hallucinated_rejected 0.000 false admission (rule <= 0.2). |
| I16 | Real-data validation. | FAILED | scripts/learning_delta.py; scripts/learner_search.py | IMPLEMENTED — FAILED VALIDATION. Every real-data learning run so far: MEMORISATION or NO_TRANSFER; lessons harmful; memory skill negative. |
| I17 | Fresh holdout seeds. | IN_PROGRESS | scripts/planted_calibration.py; scripts/learner_search.py (seeds 11,12) | [partial] Multi-seed runs; no reserved never-used holdout-seed protocol. |
| I18 | Cross-year validation. | FAILED | engine/learning_delta.py::headline_transfer; engine/learners.py | IMPLEMENTED — FAILED VALIDATION. Cross-year: NO_TRANSFER (ld2 CI -0.00034..+0.00030); no learner passes (ls3). |
| I19 | Memorization control. | VALIDATED | engine/learning_delta.py::MemoriserLearner,IdentityRecallLearner,verdict | [already implemented] The control works: planted memoriser flagged; real memory_bank learner correctly labelled MEMORISATION. |
| I20 | Future-leak control. | VALIDATED | engine/learning_delta.py::LeakyGate,leak_probe; engine/leak_audit.py | [already implemented] The control works (planted leaks caught). The leaks it found are open - see H02. |

### J PRODUCTION INTELLIGENCE - IN_PROGRESS 11, IMPLEMENTED 1, NOT_STARTED 3

| id | item | status | main code | note |
|---|---|---|---|---|
| J01 | Champion knowledge. | IN_PROGRESS | engine/champion.py::Board | [partial] Champion of configs/models, not knowledge. |
| J02 | Challenger knowledge. | IN_PROGRESS | engine/champion.py::ChallengerQueue; engine/improve.py::spawn_challengers | [partial][duplicated] Two challenger mechanisms. |
| J03 | Shadow knowledge. | IN_PROGRESS | engine/shadows.py | [partial][under-tested] Shadow baseline portfolios only (44 lines, no test). |
| J04 | Promotion gate. | IN_PROGRESS | engine/champion.py::_check_*; engine/objective.py::firewall | [partial] 6 of 10 gates; missing anti-memorisation, future-info audit, cross-context transfer, provenance completeness. |
| J05 | Retirement gate. | IMPLEMENTED | engine/pattern_lifecycle.py; engine/pattern_bank.py | IMPLEMENTED — NOT VALIDATED. Retire-not-delete holds for patterns; no knowledge-level retirement gate. |
| J06 | Recovery gate. | IN_PROGRESS | engine/pattern_memory.py (requalify); engine/pattern_bank.py | [partial] Automatic requalification; no explicit evidence threshold for recovery. |
| J07 | Reliability monitoring. | IN_PROGRESS | engine/pattern_reliability.py::health_monitor,HealthLedger; engine/trust_store.py::drift | [partial][under-tested] |
| J08 | Calibration monitoring. | IN_PROGRESS | engine/pattern_reliability.py::brier,ece,reliability_table; engine/direction_calib.py | [partial] Calibration measured; no drift monitor or influence reduction. |
| J09 | Contradiction monitoring. | NOT_STARTED | - | [missing] |
| J10 | Surprise monitoring. | NOT_STARTED | - | [missing] run_report.memory_shock_events is a shock detector, not a surprise monitor. |
| J11 | Knowledge health. | IN_PROGRESS | engine/pattern_reliability.py::HealthLedger; engine/memory_diagnostics.py::memory_health | [partial] healthy/suspect/broken only. |
| J12 | Decision impact monitoring. | IN_PROGRESS | engine/pattern_movers.py (ablation); engine/adaptive.py::adaptation_value | [partial] |
| J13 | Learning curve. | IN_PROGRESS | engine/learning_delta.py::play_curve,curve_verdict; scripts/learning_curve.py | [implemented-but-unvalidated] Builders editing; real run lc1 in flight. |
| J14 | Learning scorecard. | IN_PROGRESS | engine/learning_delta.py::aggregate,render_report | [partial] 8 metrics + controls; ~10 of 19 scorecard fields. |
| J15 | Research priority dashboard. | NOT_STARTED | - | [missing] |

### K REPORTING - IN_PROGRESS 5, IMPLEMENTED 5, NOT_STARTED 5

| id | item | status | main code | note |
|---|---|---|---|---|
| K01 | Learning curve report. | IN_PROGRESS | engine/learning_delta.py::render_curve_report | [partial] Report code exists; no report on disk yet. |
| K02 | Transfer report. | IMPLEMENTED | engine/learning_delta.py::render_report; scripts/learner_search.py | IMPLEMENTED — NOT VALIDATED. Cross-year transfer only; no regime/stock/sector sections. |
| K03 | Memorization report. | IMPLEMENTED | engine/learning_delta.py::render_report | IMPLEMENTED — NOT VALIDATED.  |
| K04 | Identity-gap report. | IMPLEMENTED | engine/antimemo.py::archive_markdown | IMPLEMENTED — NOT VALIDATED. Lessons archive only. |
| K05 | Knowledge health report. | IN_PROGRESS | engine/pattern_reliability.py::render_report; engine/memory_diagnostics.py::report | [partial] |
| K06 | Failure report. | IN_PROGRESS | scripts/run_pattern_fail.py; engine/lessons.py::kind_report | [partial] |
| K07 | Recovery report. | NOT_STARTED | - | [missing] |
| K08 | Contradiction report. | NOT_STARTED | - | [missing] |
| K09 | Missed-winner report. | IMPLEMENTED | engine/missed_winners.py::summary_text; scripts/run_b15_memory_adapter.py | IMPLEMENTED — NOT VALIDATED.  |
| K10 | Experiment-memory report. | IN_PROGRESS | scripts/audit_registry.py; engine/experiment_memory.py::answer_coverage | [partial] |
| K11 | Research-priority report. | NOT_STARTED | - | [missing] |
| K12 | Meta-learning report. | NOT_STARTED | - | [missing] |
| K13 | Promotion/rejection report. | IN_PROGRESS | engine/champion.py::Ledger; engine/improve.py::report | [partial] |
| K14 | Provenance audit report. | IMPLEMENTED | scripts/audit_registry.py; scripts/leak_audit.py | IMPLEMENTED — NOT VALIDATED. Reports exist; their verdicts fail (registry ok=false, 3 LEAK channels). |
| K15 | Full learning-system report. | NOT_STARTED | - | [missing] scripts/final_report.py is the Bible Phase-46 trading report, not a learning-system report. |

### L FINAL SCIENTIFIC VALIDATION - IN_PROGRESS 7, NOT_STARTED 9, IMPLEMENTED 8, FAILED 2

| id | item | status | main code | note |
|---|---|---|---|---|
| L01 | No-learning baseline frozen. | IN_PROGRESS | engine/learning_delta.py::NullLearner; engine/baseline.py::freeze_baseline | [partial] Control exists; not frozen as a hashed learning baseline. |
| L02 | Legitimate learner frozen. | NOT_STARTED | - | [missing] No learner has passed; nothing to freeze. |
| L03 | Memorizer control frozen. | IN_PROGRESS | engine/learning_delta.py::MemoriserLearner | [partial] Not frozen. |
| L04 | Random learner control frozen. | NOT_STARTED | - | [missing] Random learner control does not exist. |
| L05 | Leaky learner control frozen. | IN_PROGRESS | engine/learning_delta.py::LeakyGate | [partial] Not frozen. |
| L06 | Same-year learning measured. | IMPLEMENTED | scripts/learning_delta.py; scripts/learning_curve.py | IMPLEMENTED — NOT VALIDATED. run1->run2 measured (MEMORISATION); the run 1..N curve is in flight; must be re-measured on the frozen learner. |
| L07 | Cross-year transfer measured. | IMPLEMENTED | scripts/learning_delta.py; scripts/learner_search.py | IMPLEMENTED — NOT VALIDATED. Measured = NO_TRANSFER for current learners. |
| L08 | Cross-regime transfer measured. | IN_PROGRESS | - | [partial] By-era breakdown only. |
| L09 | Cross-stock transfer measured. | NOT_STARTED | - | [missing] |
| L10 | Cross-sector transfer measured. | NOT_STARTED | - | [missing] |
| L11 | Memorization gap measured. | IMPLEMENTED | engine/learning_delta.py::aggregate | IMPLEMENTED — NOT VALIDATED. Measured for memory_bank learners. |
| L12 | Identity gap measured. | IMPLEMENTED | engine/antimemo.py::archive_experiment | IMPLEMENTED — NOT VALIDATED. Measured on lessons archive only. |
| L13 | Future-information audit passed. | FAILED | engine/leak_audit.py | IMPLEMENTED — FAILED VALIDATION. 3 LEAK, 4 QUARANTINED channels. |
| L14 | Provenance audit passed. | FAILED | scripts/audit_registry.py | IMPLEMENTED — FAILED VALIDATION. registry_audit ok=false; provenance test failing. |
| L15 | Reproducibility passed. | IN_PROGRESS | engine/retester.py; engine/repro.py | [partial] Re-tester for trading windows; learning experiments not reproduced from the full section-56 key. |
| L16 | Stability passed. | IN_PROGRESS | scripts/learner_search.py | [partial] Seed replicates disagree in validity (ls2 self-check INVALID). |
| L17 | Risk impact measured. | IMPLEMENTED | engine/learning_delta.py::run_metrics (worst5, max_dd) | IMPLEMENTED — NOT VALIDATED. Measured for current learners only. |
| L18 | Direction impact measured. | IMPLEMENTED | engine/learning_delta.py::pick_hits (dir_hit) | IMPLEMENTED — NOT VALIDATED. Measured for current learners only. |
| L19 | Movement impact measured. | IMPLEMENTED | engine/learning_delta.py::pick_hits (mover_hit) | IMPLEMENTED — NOT VALIDATED. Measured for current learners only. |
| L20 | Portfolio impact measured. | IMPLEMENTED | engine/learning_delta.py::run_metrics (mean_week, in_band, year_return) | IMPLEMENTED — NOT VALIDATED. Measured for current learners only. |
| L21 | Learning delta independently reproduced. | NOT_STARTED | - | [missing] |
| L22 | Fresh holdout validation passed. | NOT_STARTED | - | [missing] |
| L23 | Knowledge promotion gates passed. | NOT_STARTED | - | [missing] No knowledge promotion gate exists. |
| L24 | All critical failures resolved or explicitly classified as unresolved scientific limitations. | IN_PROGRESS | engine/leak_audit.py (LEAK/QUARANTINED classes) | [partial] Leak channels classified; learning failures (E04, D13, D15, C03) not yet classified as fixed vs scientific limitation. |
| L25 | Final Masterstock updated. | NOT_STARTED | - | [missing] Final state only. |
| L26 | Final checklist independently audited. | NOT_STARTED | - | [missing] Final state only. This mapping is a self-audit, not an independent one. |

## Line budget (57 sections)

| section | min | actual | status | where |
|---|---|---|---|---|
| Knowledge objects | 900 | 444 | IN_PROGRESS | engine/pattern_identity.py (444) |
| Epistemic states | 450 | 0 | NOT_STARTED | - |
| Observation/interpretation | 600 | 0 | NOT_STARTED | - |
| Context modeling | 1200 | 169 | IN_PROGRESS | engine/memory.py (19); engine/pattern_memory.py (8); engine/pattern_stats.py (62); engine/pattern_reliability.py (80) |
| Failure learning | 1100 | 507 | IN_PROGRESS | engine/lessons.py (368); engine/pattern_reliability.py (139) |
| Pattern-break engine | 1300 | 296 | IN_PROGRESS | engine/pattern_reliability.py (255); engine/pattern_lifecycle.py (41) |
| Dynamic reliability | 900 | 420 | IN_PROGRESS | engine/pattern_reliability.py (420) |
| Pattern lifecycle | 800 | 401 | IN_PROGRESS | engine/pattern_lifecycle.py (401) |
| Retirement/recovery | 450 | 384 | IN_PROGRESS | engine/pattern_bank.py (384) |
| Time-aware memory | 700 | 132 | IN_PROGRESS | engine/pattern_memory.py (95); engine/heavy_tests.py (25); engine/memory_diagnostics.py (12) |
| Situation representation | 1000 | 67 | IN_PROGRESS | engine/lessons.py (42); engine/memory.py (21); engine/learning_delta.py (4) |
| Situation similarity | 900 | 1101 | IN_PROGRESS | engine/analog_weighting.py (687); engine/analogs.py (99); engine/analogs_sector.py (151); engine/analogs_stock.py (164) |
| Knowledge retrieval | 1100 | 416 | IN_PROGRESS | engine/memory.py (345); engine/memory_diagnostics.py (71) |
| Contradiction graph | 850 | 0 | NOT_STARTED | - |
| Knowledge hierarchy | 900 | 291 | IN_PROGRESS | engine/trust.py (253); engine/pattern_stats.py (38) |
| Credit assignment | 1200 | 266 | IN_PROGRESS | engine/ablation.py (171); engine/direction_ablate.py (69); engine/memory_diagnostics.py (8); engine/heavy_tests.py (18) |
| Redundancy intelligence | 600 | 69 | IN_PROGRESS | engine/pattern_stats.py (45); engine/antioverfit.py (24) |
| Missed-winner learning | 1000 | 431 | FAILED | engine/missed_winners.py (431) |
| Selection/timing/risk separation | 700 | 34 | IN_PROGRESS | engine/lessons.py (22); engine/memory.py (12) |
| Loss postmortems | 800 | 88 | IN_PROGRESS | engine/lessons.py (88) |
| Same-year learning harness | 1300 | 1277 | IN_PROGRESS | engine/learning_delta.py (720); engine/antimemo.py (433); scripts/learning_delta.py (124) |
| Cross-year transfer | 1100 | 1041 | FAILED | engine/learners.py (739); scripts/learner_search.py (212); engine/learning_delta.py (90) |
| Transfer scoring | 400 | 126 | IN_PROGRESS | engine/learning_delta.py (126) |
| Memory firewall | 900 | 196 | IN_PROGRESS | engine/pattern_memory.py (72); engine/memory.py (28); engine/memory_diagnostics.py (2); engine/lessons.py (9); +2 more |
| Identity firewall | 750 | 236 | IN_PROGRESS | engine/antioverfit.py (46); engine/blind_gates.py (78); engine/leak_audit.py (112) |
| Future-information firewall | 1000 | 2116 | FAILED | engine/pit.py (1030); engine/parity.py (379); engine/parity_suite.py (304); engine/fill_audit.py (32); +1 more |
| Experiment memory | 800 | 541 | IN_PROGRESS | engine/experiment_memory.py (293); engine/registry.py (225); engine/improve.py (23) |
| Belief updating | 800 | 0 | NOT_STARTED | - |
| Real/useful/current separation | 600 | 144 | IN_PROGRESS | engine/pattern_stats.py (144) |
| Portfolio-value learning | 850 | 829 | IN_PROGRESS | engine/objective.py (193); engine/pattern_movers.py (636) |
| Exploration/exploitation | 750 | 0 | NOT_STARTED | - |
| Information-gain research | 900 | 0 | NOT_STARTED | - |
| Meta-learning | 1200 | 0 | NOT_STARTED | - |
| Failed-learner memory | 500 | 0 | NOT_STARTED | - |
| Knowledge competition | 700 | 0 | NOT_STARTED | - |
| Complexity control | 500 | 0 | IN_PROGRESS | - |
| Boundary learning | 700 | 0 | IN_PROGRESS | - |
| Unknown handling | 400 | 129 | IN_PROGRESS | engine/direction_calib.py (129) |
| Knowledge→decision contract | 450 | 0 | NOT_STARTED | - |
| Champion/challenger | 700 | 429 | IN_PROGRESS | engine/champion.py (241); engine/improve.py (144); engine/shadows.py (44) |
| Promotion gates | 700 | 66 | IN_PROGRESS | engine/champion.py (66) |
| Knowledge health | 650 | 209 | IN_PROGRESS | engine/pattern_reliability.py (129); engine/memory_diagnostics.py (38); engine/trust_store.py (42) |
| Learning scorecard | 700 | 106 | IN_PROGRESS | engine/learning_delta.py (106) |
| Learning curve | 500 | 346 | IN_PROGRESS | engine/learning_delta.py (194); scripts/learning_curve.py (152) |
| Knowledge archive | 1000 | 483 | IN_PROGRESS | engine/pattern_memory.py (351); engine/lessons.py (73); engine/trust_store.py (59) |
| Knowledge graph | 1000 | 0 | NOT_STARTED | - |
| Research priority | 900 | 0 | NOT_STARTED | - |
| Surprise engine | 500 | 28 | IN_PROGRESS | engine/run_report.py (28) |
| Disagreement engine | 650 | 0 | NOT_STARTED | - |
| Calibration | 650 | 69 | IN_PROGRESS | engine/pattern_reliability.py (21); engine/memory_diagnostics.py (26); engine/run_report.py (11); engine/planted.py (11) |
| Learning firewalls | 1100 | 666 | IN_PROGRESS | engine/blind_gates.py (290); engine/isolation.py (376) |
| Reproducibility | 500 | 739 | IMPLEMENTED | engine/repro.py (296); engine/provenance.py (92); engine/retester.py (351) |
| Compute manager | 900 | 702 | IN_PROGRESS | engine/resources.py (396); engine/health.py (306) |
| Continuous execution/checkpointing | 700 | 517 | IN_PROGRESS | engine/checkpoint.py (135); scripts/livesim_loop2.py (382) |
| Reporting/dashboard/inspection | 1500 | 1166 | IN_PROGRESS | scripts/dashboard_build.py (395); scripts/final_report.py (212); engine/run_report.py (202); engine/claims.py (302); +1 more |
| Test infrastructure for the above | 8000 | 8871 | IN_PROGRESS | tests/test_learning_delta.py (278); tests/test_learning_curve.py (196); tests/test_learners.py (253); tests/test_lessons.py (393); +37 more |
| Integration/refactoring/type safety | 4000 | 605 | IN_PROGRESS | engine/adaptive.py (605) |
| **total** | **56,700** | **27,153** | | |

Counting rule: a module is counted whole only when all of it does that job; otherwise only the named functions (listed with per-part counts in the JSON code_paths). No line is counted in two sections. Code that exists but serves trading rather than learning (site_* pages, broker, features, fv_pipeline, exits, stops, gaprisk, direction) is not counted.

## Gap list

### Unsafe (fix before anything learns from new state - contract section 85 priority 1)

1. **Three open LEAK channels in the default blind path** (`state/research/leak_audit/report.md`): 2 back-adjusted price levels, 4 learned state (basis cfg/meta) trained on future or same windows, 8c feed shape (alphabetical column order, future-IPO columns, absolute SPY level). Fixes exist as tested hooks (`leak_audit.BasisLineage`/`strict_training_windows`, `split_invariant_tradable`, `hardened_feed_class`) but are not the default. Four more channels are QUARANTINED (survivorship, today's metadata, year fingerprint, run-time file reads).
2. **Provenance not applied**: `state/research/registry_audit.json` ok=false - experiment_id/git/canon/config/seed present on 2 of 53 audited records; `tests/test_provenance.py::test_integrity_fails_closed_on_tampered_bible` fails today (fixture lacks SELF_LEARNING_CONTRACT.md).
3. **Fail-closed is per-module only**: pit/pattern_memory/learning_delta raise, but nothing blocks a learning result from promotion when a firewall layer has an open finding.
4. `engine/pattern_reliability.py` (1,543 lines, being live-edited) has **no test file** - its guards (`audit_context_builder`, `causality_audit`) are untested code.

### Contradictory

- `engine/memory.py` evicts episodes at capacity; contract section 49 says never destroy evidence.
- `engine/lessons.py` lessons reweight decisions directly; section 24 says a postmortem must first become a hypothesis. (Lessons also measured harmful.)
- `engine/improve.py::spawn_challengers` is a parameter-tuning loop; section 35 says not to spend compute re-tuning known parameters.
- Same-year disguised reruns (C54/C57) replay identical numbers, so any numeric memory recognises them; section 1.2 forbids "identifying a disguised rerun as a rerun" (ld2 report asks the owner to rule C54 vs C56).
- The planted-calibration verdict is VALIDATED while the regime plant is admitted only 38-50% and the unless plant 12-25% directly - neither is a verdict criterion.
- Redundancy pruning (`pattern_stats.prune_redundant`) deletes the predictive-vs-regime-protection distinction section 21 wants learned.
- Four state vocabularies (lifecycle STATES, pattern_memory modes, reliability VERDICTS/HealthLedger, bank PRIOR_STATES) and none is the contract's epistemic set.

### Missing (no code at all)

Epistemic states; observation/interpretation separation and competing explanations; belief updating (prior -> posterior); contradiction graph and knowledge graph (all 10 edge types, traversal, lineage queries); situation archive, context/reliability/contradiction/recovery indexes; recovery analysis; cross-stock and cross-sector transfer tests; random-learner control (D); transfer ratio; question/hypothesis/expected-outcome generation; information-gain priority and research queue; exploration/exploitation; meta-learning; failed-learner store; knowledge competition; knowledge->decision contract; contradiction and surprise monitoring; research-priority dashboard; recovery, contradiction, research-priority, meta-learning and full learning-system reports; property tests; frozen controls and the whole L-phase.

### Partial / under-depth (code exists, well below the contract)

KnowledgeObject (pattern-only record), situation representation (67 lines), similarity (one distance, no components), retrieval (no reliability/transfer/contradiction terms; real skill negative), dynamic reliability (1 of 5 dimensions), time-aware memory (fixed kernels, no learned temporal class), pattern lifecycle (no decay-type classification), failure learning (5 of 14 causes), loss postmortems (few of 13 fields), selection/timing/risk separation (34 lines), credit assignment (component, not per-decision), hierarchy (trust tables only), redundancy (predictive only), promotion gates (6 of 10, configs not knowledge), calibration (measured, not monitored or acted on), scorecard (~10 of 19 fields).

### Implemented but scientifically unvalidated or failed

Validated as instruments: learning-delta harness self-check, memoriser/no-learning/leak controls, memorisation gap, transfer evaluation, stale-code and stale-worker firewalls, deterministic hashing, planted strong/weak/negative/pair/decaying/noise/hallucinated detection (miner level).
Failed on real data: retrieval/episodic memory (skill -0.16), missed-winner detector (uplift 0), lessons OOS (-0.188%/wk), every cross-year learner (NO_TRANSFER), regime and unless plants, future-information audit (3 LEAK), provenance audit.

## Duplicate / parallel systems to avoid (reuse, do not add another)

| concern | existing parallel implementations | rule for new work |
|---|---|---|
| knowledge store | `pattern_memory.PatternMemory` (hash chain), `pattern_bank.PatternBank`, `trust_store.TrustStore`, `lessons.LessonStore`, `memory.Memory` | `engine/learning/archive.py` sits on the pattern_memory chain; others get adapters, not a sixth store |
| state vocabulary | `pattern_lifecycle.STATES`, pattern_memory modes, `pattern_reliability.VERDICTS` + HealthLedger statuses, `pattern_bank.PRIOR_STATES` | `engine/learning/epistemic.py` defines the contract states and a mapping from each |
| champion/challenger | `champion.Board/ChallengerQueue`, `improve.test_and_promote/spawn_challengers` | `engine/learning/promotion.py` reuses champion.Ledger; retire improve.py's path for knowledge |
| experiment dedup | `experiment_memory.TriedIndex`, `registry.fingerprint/ExperimentMemory` | one façade in `engine/learning/experiment_memory.py` |
| lesson records | `lessons.Lesson`/`KINDS`, `memory.Lesson`/`ERROR_TYPES` | both become producers of failure hypotheses in `engine/learning/failure.py` |
| hashing | `pattern_bank.canon_hash`, `pattern_memory._hash`, `repro.artifact_hash`, `pit.fingerprint`, `trust_store._content_hash`, `checkpoint._body_hash`, `provenance.config_hash` vs `repro.config_hash` | use `repro.artifact_hash` + `pattern_identity.pattern_hash`; add none |
| multiple testing / stats | `pattern_reliability.holm`, `pattern_memory.bh_adjust`, `pattern_stats.bh_qvalues`; `_p_two_sided` x4; `boot_ci` x2; `signflip_p`/`sign_flip_p`; `max_drawdown` x2; `era_of` x4 | import from `pattern_stats` (and `learning_delta` for bootstrap); no new copies |
| calibration tables | pattern_stats, planted, memory_diagnostics, missed_winners, run_report, pattern_reliability (brier/ece) | `engine/learning/calibration.py` wraps pattern_reliability.brier/ece + pattern_stats.calibration_table |
| disguise | `antimemo.disguise`, `learning_delta.make_presentation`, `blind_gates.make_ticker_map`, livesim disguise | learning_delta.make_presentation is the audited one |
| ablation | `ablation.ablate`, `direction_ablate`, `pattern_movers` ablation, `heavy_tests.ablation_across_eras`, `memory_diagnostics.factor_ablation` | `engine/learning/credit.py` calls `ablation.ablate` |
| health | `health.py` (workers) vs `pattern_reliability.HealthLedger` (patterns) | `engine/learning/health.py` wraps HealthLedger; name clash only |

## Proposed build order (contract section 85)

1. P1 future-information and provenance correctness -> WP1.
2. P2 learning validity -> WP2 (the object the firewalls and controls must reason about), WP3 (identity-free situations).
3. P3 same-year learning delta -> WP4 (complete controls A-E, curve, scorecard).
4. P4 cross-year transfer -> WP5 (the legitimate learner closing the loop, cross-context transfer).
5. P5 failure learning -> WP6.
6. P6 pattern reliability and lifecycle -> WP7 (after the builders release pattern_reliability.py).
7. P7 knowledge retrieval/context -> WP8, WP9.
8. P8 meta-learning -> WP10.
9. P9 production integration -> WP11.
10. P10 presentation -> WP12.

Hold-points: WP4 and WP5 must not edit `engine/learning_delta.py`, `engine/pattern_memory.py`, `engine/pattern_reliability.py` or `scripts/learning_curve*` while builders hold them - import and extend from `engine/learning/`.

## Work packages

Every package: new code under `engine/learning/`, synthetic tests under `tests/learning/`, a real-cache runner under `scripts/` writing to `state/research/<module>/` with provenance, and the section-62 adversarial test(s) it owns (each of the 12 is owned exactly once).

| WP | name | target modules (engine/learning/) | reuses | items | budget rows | owns section-62 tests |
|---|---|---|---|---|---|---|
| WP1 | Firewall chain + provenance closure | `firewall.py` (8 layers DATA/TIME/MEMORY/IDENTITY/PROVENANCE/EXPERIMENT/EVALUATION/CODE-VERSION; `admit(item, as_of)` fail-closed), `checkpoints.py` (run key = code+data+config+seed+worker cfg+memory snapshot+experiment id) | pit, leak_audit, blind_gates, provenance, repro, health, registry | A05 A13 B15 G01 H01-H12 K14 L13 L14 | Future-info firewall, Memory firewall, Learning firewalls, Reproducibility | T7 hidden-from-evaluation not used; T11 future info rejected; T12 code change mid-run -> stale rejected (re-point existing health test) |
| WP2 | Knowledge core | `knowledge.py` (typed KnowledgeObject, versioning, (de)serialisation, schema validation), `epistemic.py` (9 states + UNKNOWN/INSUFFICIENT_DATA/CONFLICTED/UNTESTED, transitions, mapping from the 4 existing vocabularies), `archive.py` (L0-L9 layers on the pattern_memory hash chain; snapshots) | pattern_identity, pattern_memory, repro.artifact_hash | A01-A04 A06-A12 B01-B04 B12 B13 C15 | Knowledge objects, Epistemic states, Observation/interpretation, Knowledge archive, Unknown handling | (property/determinism tests: round-trip, state-transition invariants, hash stability) |
| WP3 | Situation + identity | `situation.py` (identity-free situation from decision-time data; ticker/date/year/sector scramble), `similarity.py` (10 stored component scores) | lessons.identity_proxy_columns, memory.context_of, analog_weighting, antioverfit permutations | A04 B03 C02 E02 H04 | Situation representation, Situation similarity, Identity firewall | T8 ticker identities changed -> no collapse; T9 years changed -> transfer when equivalent |
| WP4 | Same-year harness completion + controls | `transfer.py` (transfer_ratio with zero/negative guards, over-specialisation flag, cross-regime/stock/sector/volatility splits, stability), control learners A, C, D, E in `learner.py` (random learner D is new; all five frozen with hashes), versioned learning scorecard and curve summary in `transfer.py` | learning_delta (import only), learners, antimemo | E01 E03 E05-E07 E09-E16 J13 J14 K01-K04 L01 L03-L06 L11 L12 | Same-year harness, Cross-year transfer, Transfer scoring, Learning scorecard, Learning curve | T10 memoriser stays distinguishable |
| WP5 | Legitimate learner (the loop) | `learner.py` (OBSERVE -> ... -> SELECT NEXT QUESTION, every stage a function; BEFORE/EXPERIENCE/AFTER record), `retrieval.py` hook | WP2-WP4, memory.Memory, pattern_memory.view | C01 C04 C05 C06 E04 E08 I16 I18 L02 L07 | Integration/refactoring, Portfolio-value learning | T1 planted real pattern discovered and used |
| WP6 | Failure learning | `failure.py` (14-cause classifier with UNKNOWN, 13-field postmortem -> hypothesis, selection/timing/direction/risk/exit separation), `credit.py` (per-decision credit/blame by ablation + interaction), `missed_winners.py` (distinction discovery validated OOS) | lessons, missed_winners, ablation, exits, gaprisk | C08 C09 D01-D05 D07-D10 D13-D15 | Failure learning, Loss postmortems, Selection/timing/risk, Credit assignment, Missed-winner learning | (planted mis-attribution: a well-selected badly-timed trade must not teach selection) |
| WP7 | Reliability, belief, lifecycle | `reliability.py` (5 dimensions; wraps pattern_reliability P(hold) as current_reliability), `belief.py` (prior/likelihood/posterior, evidence-quality weighting), `lifecycle.py` (birth..retirement, decay-type classifier, ACTIVE/DEGRADED/DORMANT/RETIRED, evidence-gated recovery, temporal classes) | pattern_reliability, pattern_lifecycle, pattern_bank, pattern_memory | A08 A09 C10 C11 D06 D12 J05 J06 B11 | Dynamic reliability, Belief updating, Pattern lifecycle, Retirement/recovery, Time-aware memory | T2 fake pattern rejected; T3 decaying pattern degradation detected |
| WP8 | Context, conditions, boundaries, hierarchy | `hierarchy.py` (general -> market -> sector -> stock-type -> vol -> interaction with shrinkage; context pooling), `break_detection.py` (success vs failure populations over all 18 section-10 dimensions, OOS-tested candidate conditions, boundaries stored as knowledge) | trust, pattern_stats.eb_shrink, pattern_reliability.explain_breaks | C12 C13 D11 F13 I10 I12 I13 | Context modeling, Pattern-break engine, Knowledge hierarchy, Boundary learning, Complexity control | T4 regime-specific pattern -> condition learned |
| WP9 | Graph, contradiction, redundancy, retrieval | `knowledge_graph.py` (10 edge types, traversal, knowledge and decision lineage), `contradiction.py` (context that explains the disagreement), `redundancy.py` (5 redundancy kinds; link, don't prune), `retrieval.py` (10 ranking terms, explanation text) | pattern_stats.jaccard/containment, memory.Memory, memory_diagnostics.explain | B05-B10 C03 F01-F12 F14 F15 J09 K08 | Contradiction graph, Knowledge graph, Redundancy intelligence, Knowledge retrieval, Knowledge competition | T5 reversed pattern -> contradiction; T6 duplicated pattern -> redundancy |
| WP10 | Research intelligence + meta-learning | `experiment_memory.py` (11-field experiment record, dedup façade, failed experiments/learners), `research_policy.py` (questions, competing hypotheses, expected outcomes, information-gain priority, exploration/exploitation, compute-aware allocation via resources), `surprise.py`, `disagreement.py`, `meta_learning.py` (evaluated OOS) | experiment_memory, registry, resources, learners reports | C07 C16 C17 G02-G12 J10 J15 K10-K12 | Experiment memory, Information-gain, Exploration/exploitation, Meta-learning, Failed-learner memory, Research priority, Surprise engine, Disagreement engine, Compute manager | (planted: a repeated experiment is refused; a known dead-end learner is not re-proposed) |
| WP11 | Production intelligence | `promotion.py` (champion/challenger/shadow/retired knowledge; all 10 gates, one critical failure blocks; knowledge->decision contract), `health.py` (9 health states, why/evidence/research), `calibration.py` (drift, per-context, influence cut when overconfident) | champion.Ledger, pattern_reliability.HealthLedger, direction_calib | J01-J04 J07 J08 J11 J12 K05 K13 L23 | Champion/challenger, Promotion gates, Knowledge health, Calibration, Knowledge->decision contract, Real/useful/current | (planted: a memorising or leaky challenger is blocked even with better OOS P&L) |
| WP12 | Reports + final validation | `scripts/learning_report.py` (K01-K15 from artefacts only), dashboard section; L-phase runs: frozen controls, fresh holdout seeds, independent reproduction | claims, run_report, dashboard_build | I17 K06 K07 K09 K15 L08-L10 L15-L22 L24-L26 | Reporting/dashboard, Continuous execution/checkpointing, Test infrastructure | (claim scan: no number in a report without an artefact) |

Test-infrastructure row (8,000) and integration row (4,000) are shared by all packages; the existing 8,868 counted test lines cover the pre-contract modules, so new test lines accrue per package.
