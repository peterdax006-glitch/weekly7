# Integration audit: is the self-learning code really wired in? (29 Sep 2026, read-only)

Method: an AST index of every import and call in 390 non-venv .py files (tests excluded from "production"), an
import-reachability closure from engine/*.py and scripts/*.py, and manual reads of each host function. wiring.HOOKS was
not trusted. No file was edited except this one. No tests were run. state/livesim/ was not touched.

## 0. Headline

- wiring.unwired_hooks() passes, but it only checks that the text `wiring.<fn>(` appears in the file. 5 of its 11 hooks sit
  in functions that no production path calls, or that do nothing in production (section 1b).
- 39 of the hook calls queued in INTEGRATION.md's C62 section have no non-test call site at all (section 1a).
  The checklist note for the Integration row ("every queued hook applied") is false.
- 17 of 62 engine/learning modules are imported by no production code. Only tests import them (section 2).
- The section-4 loop is real: all 19 stages call their owning modules. It runs only in two places: as a SHADOW behind
  `livesim_loop2 --learner legit` (default off), and in scripts/acceptance_mini.py on a planted world.
  state/learning/curator/ does not exist, so it has never run on real data (section 3).
- Section 83 verdict: INCOMPLETE. Functionality is missing. Uncredited integration code elsewhere does exist
  (4,521 lines on the ruler if you count generously, 3,294 if you count strictly), but a line count cannot close missing
  hooks (section 4).

## 1a. C62 contract wave 1 hooks: real call sites

In the table, "learner" means engine/learning/learner.py. That file is production-reachable only through
test_path (loop2 --learner legit) and scripts/acceptance_mini.py.

| Builder | Hook in INTEGRATION.md | Actual non-test call site | Verdict |
|---|---|---|---|
| S01 | promotion/health/archive call decision_contract.check | only learner.py:622 (`DC.check` in `_contract_allows`); promotion.py, health.py and archive.py do not import decision_contract | MISSING in promotion/health/archive |
| S01 | decision_contract.policy_check | none (defined decision_contract.py:591) | MISSING |
| S01 | readiness | none (decision_contract.py:500, champion.py:941) | MISSING |
| S01 | firewalls use KnowledgeStore.as_of + audit_future | as_of at learner.py:618,657; audit_future at learner.py:1263; firewalls.py never imports knowledge | MISSING in firewalls |
| S01 | failure/credit/experiment_memory call with_failure, with_relation | none (knowledge.py:1144,1157) | MISSING |
| S01 | DecisionLog | learner.py:347 (construct), :658 (`record_all`) | wired (learner only) |
| S01 | research policy uses interpretation.next_test + unknowns.rank_unknowns | none (both modules are dead) | MISSING |
| S05 | post_mortem -> records_from_lessons_frame -> LossClassifier.classify | lessons.py:260 -> wiring.py:275,284 | wired, but post_mortem is reached only via engine/antimemo.py:157,284,459 <- scripts/lessons_archive.py, lessons_real.py |
| S05 | LessonBook.learn + memory lesson store -> hypotheses_from_lessons | lessons.py:351,366 and memory.py:530 -> wiring.py:319 | wired; LessonBook is built only in antimemo.py; Memory.export_lessons is called only by memory_diagnostics.py:166,356 (loop2 calls mem.export()) |
| S05 | MissedLedger.add -> week_from_base -> add_week | hook placed in MissedLedger.observe (missed_winners.py:298) -> wiring.py:331,335 | DEAD: observe() has 0 non-test callers; adaptive.py:358 calls ledger.add(), which has no hook |
| S08 | RetirementLedger.evaluate/attempt_recovery from health verdicts via temporal.to_evidence | evaluate/attempt_recovery at learner.py:959,961 fed by RT.series_evidence; to_evidence has 0 callers | PARTIAL (health->retirement path MISSING) |
| S08 | calibration.combined_influence at the decision contract | none (calibration.py:764) | MISSING (learner uses TP.expected_influence + retirement.influence instead, L520-533) |
| S08 | SurpriseTracker.research_priority/research_questions -> research priority | not called; learner.py:1356 uses RPR.signals_from_surprise_rows | equivalent path in learner only |
| S11 | production readers call KnowledgeBoard.weight | wiring.py:546 <- effective_weight <- lessons.py:544,555 | NO-OP: HUB.board is None unless configure() is called, and configure() has 0 non-test callers, so the legacy weight is returned unchanged |
| S11 | effective_champion | none (champion.py:538) | MISSING |
| S11 | audit_decision_sources after each decision run | none (champion.py:820) | MISSING |
| S11 | compute.job_for wraps experiments as resources.Job | none (compute.py:552) | MISSING |
| S14 | none needed | planted_world.as_claim used internally (:804) | n/a |
| S07 | PatternMiner rows -> evidence_from_pattern_row -> BeliefLedger.update | update at learner.py:775 from the learner's own outcomes; evidence_from_pattern_row has 0 callers; patterns.py never reaches belief | MISSING (miner->belief) |
| S07 | QuestionEngine.ask_many + boundary.to_knowledge -> knowledge store | none | MISSING |
| S07 | competition.boundary_field | none | MISSING |
| S07 | questions.py imports boundary.py | yes, but questions.py itself is dead | dead |
| S06 | credit.update_proposals -> belief updates | none (credit.py:1472) | MISSING |
| S06 | RedundancyReport.edges + credit.credit_edges -> knowledge graph | none (credit.py:1243; redundancy is dead) | MISSING |
| S06 | redundancy takes credit.masked_pairs | none (credit.py:1540) | MISSING |
| S09 | log_experiment -> ExperimentLedger.record_result / import_legacy | improve.py:79 -> wiring.on_experiment -> import_legacy wiring.py:382; record_result has 0 callers | wired (import_legacy path); log_experiment is reached from improve.weekly <- live.py:251 / tick.py:56 |
| S09 | pre-launch ExperimentLedger.already_tested | improve.py:198 (spawn_challengers <- weekly L337) -> wiring.py:408 | wired |
| S09 | MetaLearner.update(now).advice -> PolicyContext(meta=) | learner.py:1344-1345 -> research.step(meta=) :1367 -> research_priority.py:749 | wired (learner only) |
| S09 | ResearchPriorityEngine.step -> propose_selected -> update_from_result | step at learner.py:1367; propose_selected and update_from_result have 0 callers | PARTIAL: the loop never closes |
| S09 | signals_from_* for failure/surprise/health/missed-winner/leak-audit | failure :1360, surprise :1356, missed :1364 (learner); signals_from_health and signals_from_data_audit have 0 callers | PARTIAL |
| S09 | failed_learners.seed_registry + check_proposal before any new learner | none (module dead) | MISSING |
| S10 | promotion calls scorecard.gate_improvement_claim | wiring.py:576 <- promotion_gate <- promotion_allowed <- improve.py:245 (test_and_promote L259 <- weekly L345). promotion.py's PromotionGate does not call it | DEGENERATE: it runs only when claims_learning=True; no challenger ever sets that, and register_evidence has 0 callers, so scorecard, firewall and identity checks never execute in production |
| S10 | scorecard_for_learner -> ScorecardStore.append | none | MISSING |
| S10 | learning_curve.delta_from_records / curve_from_play_records | none (scripts/learning_curve*.py do not call them) | MISSING |
| S11 p2 | compute.run_experiment_process | none | MISSING |
| S11 p2 | checkpoints.resume_verified at every resume | none | MISSING |
| S11 p2 | write_interruption on shutdown | none | MISSING |
| S15 | lifecycle.apply_to_ledger -> retirement gate | none | MISSING |
| S15 | health.inputs_from_knowledge + epistemic_proposals | none | MISSING |
| S15 | reliability.contexts_from_condition | none | MISSING |
| S19 | Curator.run_day once per simulated day | test_path.py:462 (PathRunner.on_tick <- livesim.py:606 hook) | wired (loop2 --learner legit only) |
| S19 | trader gets ONLY TraderDay | test_path.py:471-474 | wired |
| S19 | trader_view.assert_trader_path_clean in CI | .github/workflows/ci.yml:46; also test_path.py:395,507 | wired |
| S19 | store root state/learning/curator/ | livesim_loop2.py:40 | wired (the directory does not exist: never run) |
| S19 | archive _YEAR regex fix | archive.py:46 (digit-only lookarounds) | done |
| S19 | calibrated strength band | curator.py:72-73, strength_band :329 | done |
| S20 | period loop calls ContradictionMonitor.run_period | wiring.py:497-498 inside on_period; on_period has 0 callers | MISSING (no period runner) |
| S20 | mon.signals -> ResearchPriorityEngine.step | wiring.research_step (wiring.py:509) has 0 callers | MISSING |
| S20 | dashboard_rows -> reports.health_dashboard | wiring.write_dashboard_inputs (wiring.py:524) has 0 callers; health_dashboard is called only in reports.py:1634 | MISSING |
| S20 | add KnowledgeGraph.contradiction_keys(now) | not defined anywhere | MISSING |

Counted as MISSING or dead (39): policy_check, readiness, decision_contract.check in promotion/health/archive,
as_of+audit_future in firewalls, with_failure, with_relation, next_test, rank_unknowns, missed-week (dead host),
temporal.to_evidence, combined_influence, effective_champion, audit_decision_sources, job_for, evidence_from_pattern_row,
ask_many, to_knowledge, boundary_field, update_proposals, credit_edges/RedundancyReport.edges, masked_pairs, record_result,
propose_selected, update_from_result, signals_from_health, signals_from_data_audit, seed_registry/check_proposal,
scorecard_for_learner/ScorecardStore.append, delta_from_records/curve_from_play_records, run_experiment_process,
resume_verified, write_interruption, apply_to_ledger, inputs_from_knowledge/epistemic_proposals, contexts_from_condition,
run_period/on_period, research_step, write_dashboard_inputs, contradiction_keys.

## 1b. wiring.HOOKS (the 11 "connected" hooks): what actually fires in production

| Hook | Site | Does its host run in production? |
|---|---|---|
| experiment | improve.py:79 | yes, via log_experiment (31 non-test callers; improve.weekly from live/tick). This is the ONLY sink that persists (experiment_ledger.jsonl beside the registry). None exists under state/, so it has not fired since it landed |
| pre_launch | improve.py:198 | yes, via improve.weekly -> spawn_challengers |
| repro | improve.py:74 | yes, via log_experiment |
| promotion_gate | improve.py:245 | yes, but degenerate (see S10 above) |
| redundancy | patterns.py:382 | yes, via PatternMiner.fit (in-memory graph only) |
| post_mortem | lessons.py:260 | research scripts only (antimemo <- lessons_archive/lessons_real) |
| lessons | lessons.py:351,366; memory.py:530 | research scripts / memory_diagnostics only |
| weight | lessons.py:544,555 | runs, but it is an identity function (no board configured) |
| registry_audit | registry.py:194 | only scripts/audit_registry.py |
| memory_entry | registry.py:274 | NO: registry.ExperimentMemory is constructed only in tests |
| missed_week | missed_winners.py:298 | NO: MissedLedger.observe has 0 non-test callers |

Extra problem: HUB.root is None in production because configure() is never called outside tests. So Hub.persist() is a
no-op, and every sink except the experiment ledger (hypotheses, failure ledger, missed ledger, redundancy graph,
promotion decisions) is lost when the process exits. hook_report(), audit_hub(), state_digest() and
assert_hooks_healthy() have 0 non-test callers, so nothing in production checks that the sinks fired.

## 2. engine/learning modules by non-test importer

Production-reachable through engine code (via wiring.py, imported by improve, lessons, memory, missed_winners, patterns
and registry): archive, champion, context, contradiction, contradiction_monitor, core, experiment_memory, failure,
firewalls, future_firewall, identity_firewall, knowledge_graph, memory_firewall, missed_winners, postmortem, promotion,
reproducibility, research_policy, research_priority, scorecard, separation, situation, transfer, transfer_score, wiring.
(Imported is not the same as called. For example, contradiction_monitor is imported by wiring, but its only call is in
the uncalled on_period.)

Reachable only through the learner/shadow path (scripts/livesim_loop2.py -> test_path; scripts/acceptance_mini.py ->
learner): learner, test_path, curator, trader_view, belief, boundary, calibration, credit, decision_contract, epistemic,
knowledge, learning_curve (via reports/same_year), meta_learning, planted_world, reliability, retirement, retrieval,
similarity, surprise, temporal. reports is reached only through scripts/learning_report.py.

**DEAD IN PRODUCTION (only tests import them, directly or through another dead module): 17 modules**

| Module | Only non-test importer |
|---|---|
| break_detection | none |
| checkpoints | none |
| competition | none |
| complexity | competition (dead) |
| compute | none (its `python -m` worker entry is launched only by run_experiment_process, which has 0 callers) |
| controls | same_year (dead) |
| disagreement | none |
| failed_learners | none |
| health | none |
| hierarchy | none |
| interpretation | none |
| lifecycle | health (dead) |
| portfolio_value | none |
| questions | none |
| redundancy | disagreement (dead) |
| same_year | none |
| unknowns | none |

Note: the section-60 rows "Compute manager" and "Continuous execution/checkpointing" are marked IMPLEMENTED on the strength
of compute.py and checkpoints.py, and both modules are dead in production.

## 3. Section-4 loop trace

learner.py `LegitimateLearner` has one method per stage, and each stage calls its owning module:
OBSERVE (stage_observe L463: FW gate) -> DESCRIBE (L491: ST.SituationBuilder.build_panel) -> RETRIEVE (L504:
KN.KnowledgeStore.visible, RV retriever) -> ASSESS (L535: RL tracker, RT.influence, TP.expected_influence) -> EXPECTATIONS
(L557: SU.scale_at, CX.ContextModel.estimate) -> DECIDE (L589: DC.check, DC.DecisionLog, RV.register_prediction) ->
OUTCOME (L669: RV.resolve_outcome) -> SURPRISE (L703: SU.observe) -> CREDIT (L728: CR.CreditEngine, SP.SubsystemLedger)
-> BELIEFS (L756: BL.BeliefLedger.update) -> FAILURE (L792: FL.LossClassifier, PM.build_postmortem, SP.attribute,
MW.add_week) -> CONDITIONS (L844: CX.discover, BD.learn_pattern_boundaries) -> ANTI-CONDITIONS (L882: CX.rules, CT.triage)
-> RELIABILITY (L917: RL.update, CB.add/assess, TP.estimate, RT.evaluate/attempt_recovery) -> TRANSFER (L977:
TR.RuleTransferLedger, RV.transfer_evidence) -> STORE (L1142: KN store, AR.Archive, CH.KnowledgeBoard, PR.PromotionGate)
-> GRAPH (L1272: KG) -> META (L1318: ML.MetaLearner.update) -> RESEARCH (L1349: RPR.step with meta advice).
decide_batch (L1373) runs stages 1-6 and resolve_and_learn (L1389) runs stages 7-19. **Every stage is in code.**

Gaps inside the loop: credit results never reach beliefs (update_proposals is unused); the research queue never closes
(propose_selected and update_from_result are unused); calibration is observed but not applied (combined_influence is unused);
break_detection, lifecycle, health, hierarchy, questions, competition, complexity, unknowns and interpretation never run.

Driver: `livesim_loop2.py --learner legit` (learner_flag L27-38; default off) -> _worker L246-249
`TP.hook_factory(PathConfig(store_root=state/learning/curator))` -> `livesim.run(hook_factory=...)` L250 ->
livesim.py:601-606 calls `trader.hook.on_tick()` each session -> test_path.PathRunner.on_tick L454 ->
`curator.run_day` L462 -> PathTrader.act -> `learner.decide_batch` L231 / `resolve_and_learn` L217 -> MemoryFiler.file
L496 -> report into result2.json["legit"] (loop2 L312). run_workers forwards LEARNER_ARGS to the child processes.
**Verdict: the trace is genuine, but the learner is SHADOW ONLY.** Its picks never trade; adaptive.Session does.
The mode is off by default, and the only legit run on record is the synthetic state/build/smoke/worker_smoke.py
(state/learning/curator/ is absent).

## 4. Section 83: row "Integration/refactoring/type safety" (minimum 4,000; mapped 1,610)

Ruler (scripts/contract_lines.py): adaptive.py 605 + wiring.py 503 + test_path.py 431 + acceptance_mini.py 71 = 1,610 (reproduced).

Q1 "Is functionality missing?" **Yes.** It belongs in these files:
- promotion.py / health.py / archive.py: decision_contract.check, policy_check, readiness.
- firewalls.py: KnowledgeStore.as_of + audit_future.
- failure.py / credit.py / experiment_memory.py: knowledge.with_failure, with_relation.
- research_policy.py: interpretation.next_test, unknowns.rank_unknowns.
- engine/adaptive.py (the MissedLedger.add call at :358) or missed_winners.py: move the hook to add(), or call observe().
- learner.py (or the period runner): temporal.to_evidence -> retirement, calibration.combined_influence, credit.update_proposals
  -> belief, credit_edges / RedundancyReport.edges -> graph, masked_pairs -> redundancy, propose_selected / update_from_result,
  signals_from_health / data_audit, evidence_from_pattern_row from PatternMiner rows, QuestionEngine.ask_many /
  boundary.to_knowledge, competition.boundary_field, failed_learners.seed_registry / check_proposal,
  lifecycle.apply_to_ledger, health.inputs_from_knowledge / epistemic_proposals, reliability.contexts_from_condition.
- A period runner (scripts/ or loop2) that calls wiring.on_period, research_step and write_dashboard_inputs, plus
  KnowledgeGraph.contradiction_keys in knowledge_graph.py.
- Production configure(root=..., board=...) in whatever starts improve.weekly / loop2, so the sinks persist and the board weights.
- compute.run_experiment_process and job_for; checkpoints.resume_verified and write_interruption in loop2's resume and shutdown.
- scorecard_for_learner -> ScorecardStore.append in a real runner; learning_curve.delta_from_records in scripts/learning_curve*.py.
- register_evidence plus claims_learning in the Test loop, so the promotion gate's learning checks ever run.
- effective_champion / audit_decision_sources after each decision run.

Q2 "Does equivalent integration code genuinely live elsewhere?" Some does, and no section-60 row credits it:
| File | Ruler lines | Integration? |
|---|---:|---|
| engine/learning/learner.py | 1,433 | yes: the section-4 conductor that wires 29 modules; credited to no line_budget row |
| engine/learning/core.py | 251 | yes: the shared typed vocabulary (the "type safety" part); credited to no line_budget row |
| engine/learning/curator.py | 673 | arguable: C64 memory curation (closer to time-aware memory / firewalls); credited to no row |
| engine/learning/trader_view.py | 554 | arguable: an import-closure firewall (closer to learning firewalls); credited to no row |

Strict count: 1,610 + learner 1,433 + core 251 = **3,294** (still 706 under).
Generous count: + curator 673 + trader_view 554 = **4,521** (the ruler run over all 8 files gives 4,521).

**Verdict: INCOMPLETE.** The row has missing functionality (section 1a: 39 unconnected hooks, a no-op board adapter,
non-persisting sinks, 17 dead modules). Section 83 says missing functionality means continue implementing. Crediting
learner.py and core.py is legitimate and should be added to the mapping, but even the generous 4,521 cannot count as
"complete" while the hooks above are unconnected. Do not pad. Wire the listed calls.
