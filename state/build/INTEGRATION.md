# Integration queue (hooks reported by builders; applied by the main session after the foundation wave)

## B01 pit (engine/pit.py, 49 tests)
- [~] (F08 2026-09-29: engine/train.py future_scramble_gate = pit.future_scramble (frames variant; these components take frames, not a Guard) over the REAL LightGBM fit, PatternMiner, Memory and adaptive Session; labels closing after as_of count as future; retrain_guarded runs it before a swap, fail closed; RESEARCH-ONLY: train.py is reached only via the parked live path) future_scramble_store(pipeline, store, as_of) over real PatternMiner / Memory / adaptive outputs
- [x] (loop2 fills proven per fill by engine/fill_audit.gate -> pit.audit_fills; F08 2026-09-29: backtest.run + run_topk fill via PITStore.executor(d).fill_next_open and prove the ledger with fill_audit.audit_session, FailClosed on failure; fill='close' = legacy control that fails the audit; livesim.py is B10's; backtest is RESEARCH-ONLY per reachability) backtest.py + livesim.py: fills via PITStore.executor(as_of).fill_next_open; run pit.audit_fills on results
- [x] (F08 2026-09-29: train.training_rows; purge removes LABEL_HORIZON x names rows at the cut; parked live path) train.py: pit.purged_training_set(X, y, as_of, horizon, cal) instead of ad-hoc label cutting
- note: Saturday filing -> strict next session needs busday roll='backward'

## B02 parity (engine/parity.py)
- [ ] live.todays_features: parity.require_parity(parity.run_parity(parity.Inputs(stocks, market, ev, ins, sic), fast=X, mode="live"))
- [ ] research runners: parity.cached_pass(path) before dependent experiments

## B07 trust/direction (engine/trust.py, engine/direction.py)
- [~] (F08 2026-09-29: backtest.ScoreHooks / fit_score_hooks, trust_on OFF by default, OFF byte-identical; RESEARCH-ONLY; no value claimed) TrustTable().fit(X, y, info, now) after PatternMiner.fit; tt.neutralize(scores, info_day) where miner scores are consumed
- [~] (F08 2026-09-29: backtest.ScoreHooks direction_on OFF by default; ON trades only bet & side=+1 rows; a closed engine (the real case) trades nothing; RESEARCH-ONLY) DirectionEngine().fit(build_inputs(...), up, now, movers); eng.decide(F_day) -> only bet=True rows trade

## B12 registry/checkpoint/champion/run_report
- [x] improve.log_experiment ids -> sha256 (done 2026-09-28)
- [x] log_experiment writes outcome, reason, metrics (+ all Phase 0.2 fields; 2026-09-28)
- [~] (F08 2026-09-29: backtest.record_major_run; backtest.run records by default; scripts/research_loop.py record_run REACHED; loop2 is F06's; Board.promote NOT done) major runs call checkpoint.write_checkpoint, run_report.build_report/write_report; Board.promote replaces improve.py promotion

## B12 second pass (engine/baseline.py, engine/experiment_memory.py, scripts/audit_registry.py; 82 tests)
- [x] log_experiment writes outcome/reason/metrics + Phase 0.2 fields (2026-09-28)
- [x] freeze the baseline once with the champion cfg: state/baseline/B_2026-09-28_5a62d137 (2026-09-28)
- [ ] live trader: Board.record_shadow_session daily, shadow_review after each session
- [~] (F08 2026-09-29: experiment_memory.launch_index/prelaunch/filter_batch/record_launch; scripts/grid_runner.py + scripts/learner_search.py skip exact repeats; engine/research/loop.py candidates NOT gated (not F08's file)) experiment_memory.check() before launching grid/loop candidates
- audit (real 53 records): 0 complete, 2 with provenance -> fixed going forward only; history stays as-is (append-only)

## Ownership transfers (2026-09-28)
- engine/memory.py, engine/adaptive.py -> B15 (asked to add Session.fills for the fill audit)
- engine/livesim.py, scripts/check_retester.py, scripts/livesim_cycle.py -> B10 (apply blind-gate/retester/health hooks)
- scripts/livesim_loop2.py -> B09 (objective.evaluate + BasisSearch; keep provenance + fill_audit gates)

## B09 objective/timeline/basis (54 tests)
- note: adaptive.py reads meta det_max / det_min_weeks not in META_DEFAULT (B15 to add)
- [x] (B15: dial hook in Session, meta dial_on, off by default; dial failed its real gate so it stays off) adaptive weekly step: merge timeline.cfg_overrides(timeline.step(...)) (needs gross-weight scale)

## B10 blind gates/retester/health (84 tests)
- real-cache: 40 seals 0 overlap; disguise 0 failures/12 windows; look-ahead probe 6/6 honest pass, 6/6 peeking caught

## B16 public pages (29 tests; real-data build CLEAN)
- [x] nav links in docs/index.html (2026-09-28)
- [ ] visual/mobile check of explorer/sensitivity2/runs/checklist (not yet viewed in a browser)
- [ ] CI: site_build.py --verify; optional site_publish.py --interval 300 (no --push)
- note: pattern bank path state/pattern_bank is B16's guess; align with B03's pattern_bank.py location

## B01 real-cache PIT audit findings (state/research/pit/PIT_AUDIT.md)
- [!] SURVIVOR-ONLY PANEL: 5,243 tickers, only 2 end before the last date, 100% in current universe.csv - every
      backtest/sim inherits survivorship bias. Needs delisted history (B13 data_sources delisted registry) - FOUNDATION.
- [ ] Live decides 15:42-15:44 ET (before close) while sims decide at close and fill next open: live/sim parity gap.
      Decide: move live to decide-after-close + fill at next open (matches C33), or model the 15:45 decision in sims.
- [x] analogs.LAG: UNRATE 30, UMCSENT 25, USREC 460 (2026-09-28)
- [x] (2026-09-28: features.labels(entry='open'); Test uses it via livesim label_entry; Live unchanged) model labels (features.labels) enter at close t; real fill is next open (mean gap +0.05%, abs 0.7%; 2020 1.3%)
- [x] (2026-09-28: features.clean_insider drops 146 impossible rows; tests/test_insider_clean.py) insider data: 142 rows filed before trade date, 6 year typos (13, 24) - clean at load
- [~] (F08 2026-09-29: backtest + train done, see B01) route fills through PITStore.executor(...).fill_next_open + pit.audit_fills; train.py purged_training_set;
      real pipeline through pit.future_scramble_store
- features.build is future-invariant (5 cuts, 107k rows, planted shift(-5) caught)

## B14 miner hardening (3,600 lines, 109 tests) - accepted; B14 now OWNS engine/patterns.py to integrate
- decision: P(real) via BH-adjusted p (blueprint sec. 30 outranks code, which used Bonferroni)
- decision: overlap-robust (HAC) cluster test when rows overlap the label horizon (per-date clustering overstated
  significance: >20% false rejections on overlapping 5-day returns in synthetic test)
- finding: panel.parquet has NO candle columns - the 25 candle signals only reach the miner if merged (algo_test.py)
- finding: m_* context columns were never candidates (weights/scope only); 12 sparse flag features fill 2-4 quintiles
- regression gate: planted tests + planted_calibration verdict before/after

## B08 exits/stops (43 tests) - sent back for conditional gap model + loss-cap analysis
- hooks: exits.learn(paths, default_rules(), as_of).apply(paths); stops.walk_forward_stops; stops.fit_gap_model + size_positions

## B09 second pass (loop2 integrated; 65+ tests)
- [x] loop2: objective.evaluate, BasisSearch via train_basis(), gates intact, seeded search, archive_dirs skips _ dirs
      (my _w01c_original backup had been trained on twice)
- real-system basis search (39 windows, 155 replays): OOS in-band 22%, risk -0.31; nothing adopted
- [x] dial on real replays: DONE - FAILS its gate (in-band 22% -> 14%); not adopted (state/research/timeline_basis/dial_offline_seed0.json)
- [x] adaptive.py: META_DEFAULT det_max/det_min_weeks; dial hook off by default (B15, 171 tests)

## B17 quality gate (70 tests) - gate findings to fix (baseline state/quality/baseline.json)
- [x] (2026-09-28: engine/explain.py; tick live-side; reviewed couplings; gate PASS) import boundaries: parity_suite -> engine.live; tick.py -> alpaca.trading; research scripts data_live_audit,
      patterns, replay_year, three_way, site_build import engine.live/broker (move plain/explain helpers to a neutral module)
- [x] print() in engine: improve -> warnings; live allowed (job log); universe in __main__ (gate exempts CLI blocks)
- [x] (reasoned exemption: the probe must touch global RNG to detect it) engine/repro.py seeds global RNGs; tests/test_repro.py draws from them
- [x] (both fixed) mutable default scripts/antioverfit_real.py:43; tests/test_livesim_gates.py:172 has no assert
- [x] A6 collector test (tests/test_collect_intraday.py, 2026-09-28)
- [x] CI: quality_gate.py --baseline state/quality/baseline.json (.github/workflows/ci.yml)

## B04 pattern_movers + heavy_tests (52 tests) - accepted; real run in progress (state/research/heavy_algo_real_run.log)
- hook: Find volatility uses PatternMoverModel(cfg).fit(X, y, now).features(Xday, as_of) -> pat_dir, pat_mov, p_move, deployed
- smoke: 0.14 live patterns per refit, rank IC ~0 (miner rarely admits patterns on real price-only data)
- bug sent to B14: score() uses full CTX list vs fitted subset for regime scopes

## A7 self-tuning (main session): engine/miner_tuning.py, scripts/miner_tune_real.py (7 tests); real run: data/tune_move0.log

## C62 contract wave 1 (S-series, 29 Sep 2026) - hooks to apply in wave-2 integration (S17), not before all code exists (C63)
- S01 knowledge: promotion/health/archive call decision_contract.check, policy_check, readiness; firewalls use
  KnowledgeStore.as_of + audit_future; failure/credit/experiment_memory call with_failure, with_relation, DecisionLog;
  research policy uses interpretation.next_test + unknowns.rank_unknowns. Shared terms Level/HypKind/Mode/Applicability live in S01 modules.
- S05 failure: engine/lessons.py post_mortem() -> failure.records_from_lessons_frame(frame, X, now) -> LossClassifier().classify;
  LessonBook.learn() + memory.py lesson store -> failure.hypotheses_from_lessons; engine/missed_winners.py MissedLedger.add
  call site -> learning.missed_winners.week_from_base(...) -> MissedLearningLedger.add_week. Shared types: failure.Hypothesis,
  learning.missed_winners.RejectionReason.
- S08 time/calibration: RetirementLedger.evaluate/attempt_recovery consume B27 health verdicts via retirement.Evidence
  (temporal.to_evidence builds it); calibration.combined_influence(kid, now, ledger, monitor, profile) at the decision contract;
  SurpriseTracker.research_priority/research_questions -> research priority.
- S11 production: production readers call KnowledgeBoard.weight + effective_champion; audit_decision_sources after each decision
  run; compute.job_for wraps an experiment as engine.resources.Job.
- S14 planted world: scoring PatternMiner output via planted_world.as_claim (key_named/effect duck-typing); no hook needed.
- S07 belief: PatternMiner rows -> belief.evidence_from_pattern_row(row, now, n_candidates=len(R)) -> BeliefLedger.update;
  questions.QuestionEngine.ask_many + boundary.to_knowledge(bset) -> knowledge store; competition.boundary_field(...) feeds
  boundary proposals into competitions; questions.py imports boundary.py.
- S06 credit: CreditReport -> credit.update_proposals() -> belief updates; RedundancyReport.edges() + credit.credit_edges() ->
  knowledge graph; redundancy takes credit.masked_pairs() as candidate pairs. Local enums: UpdateAction, CreditVerdict, Utility,
  PairClass, RedundancyKind, Reason, Predictiveness.
- S09 research: improve.log_experiment -> ExperimentLedger.record_result / em.import_legacy; pre-launch ExperimentLedger.already_tested(question, design, now);
  MetaLearner.update(now).advice -> PolicyContext(meta=...); nightly ResearchPriorityEngine.step -> propose_selected -> update_from_result;
  signals_from_* adapters for failure/surprise/health/missed-winner/leak-audit; failed_learners.seed_registry + check_proposal before any new learner.
- S10 transfer/scoring: promotion calls scorecard.gate_improvement_claim(card).allowed; real runner -> scorecard_for_learner -> ScorecardStore.append;
  scripts/learning_curve* records -> learning_curve.delta_from_records / curve_from_play_records.
- S11 pass 2: compute.run_experiment_process for real workers; checkpoints.resume_verified at every resume; write_interruption on shutdown.
  CAVEAT: current_code_hash() moves while builders edit engine/, so real launches will be refused as stale until edits stop - by design.
- S15 break/health: lifecycle.apply_to_ledger -> retirement gate; health.inputs_from_knowledge + epistemic_proposals (caller writes new versions);
  reliability.contexts_from_condition turns validated break conditions into contexts/anti_contexts. KNOWN: B27 health monitor
  false-alarms on stationary series (expected effect from a shrunk early mean = winner's curse) - fix in Stage 3.
- S19 curator (C64): the trusted runner calls Curator.run_day(date, m_state, k) once per simulated day; the trader gets ONLY TraderDay;
  EVERY trader-side memory read must go through the Curator; run trader_view.assert_trader_path_clean() in CI; choose the store root
  (proposal: state/learning/curator/). FIX: archive.py _YEAR regex misses AAPL_2008 - use trader_view's digit-only lookarounds.
  DESIGN GAP: release weights sum to 1, so a lone weak memory reads as weight 1.0 - pass a coarse calibrated strength band too
  (era-free) so the trader cannot over-rely on a weak memory.
- S20: period loop calls ContradictionMonitor(graph, ledger).run_period(now); mon.signals(report) -> ResearchPriorityEngine.step;
  mon.dashboard_rows(report)["rows"] -> reports.health_dashboard. Add public KnowledgeGraph.contradiction_keys(now) (monitor reads _edges).
  KNOWN GAPS for Stage 3: transfer_ratio(inf, x) not guarded; FW.reference_context seeds 11/17/41 trip the implausible-IC check by chance;
  calibration.ece_equal_mass NaN with <3 rows per bin.

## C66 wave 1 (R-series) hooks and known defects
- R03 counterfactual: loop entry counterfactual.step(store, events, now, providers); market-wide run_year(...); R02 hook
  Providers(classifier=adapt_classifier(r02_engine)); ONLY release path ClassificationGate().add(report).release(now, replaying_years);
  add the cf_ prefix + HINDSIGHT_NAMES to the trader-side leak scan. DEFECT: null world 6/10 seeds -> POTENTIALLY_PREDICTABLE
  from magnitude-only pointers (must be UNKNOWN). pit scrambled stores bump fabricated rows only ~400 days.
- R06 missed: submit_checked(state, day) -> step(state, now) or sweep(...); research_targets(report) -> priority engine;
  trader_handoff(record, now) is its only door. trader_view forbids a 'reasons' key even inside MaturedRecord payloads.
- R19 (partial, sent back): replication.authorize_system_change before any research-derived change reaches the trader,
  passing replayed_years; QualityGate.evaluate before PromotionGate; scorecard priority weights -> research priority.

## C68 (P-series) hooks and fixes queued for P06
- P01 expectations/outcomes/errors: expectations_from_day(day, PathModel..., ctx, now) -> ExpectationLedger.record(exp, now) before the fill;
  outcomes.reconstruct -> OutcomeLedger.add after exit; prediction_error.ErrorEngine(tracker=shared SurpriseTracker).step(outcomes, now);
  store ExpectationLedger.anchor() outside the ledger and verify(anchors) periodically.
- P02 error_research.step(state, now, records, contexts, priority_state, results, budget, seed) after P01 records mature; Q15 follow-ups
  -> precursors/discovery; knowability via InvestigationContext.
- P03 market_expectations.step / change_points.step / regime_memory.step each day; feed_tracker into the shared SurpriseTracker;
  DUPLICATE: market_expectations has its own small hash chain -> move onto archive ChainFile lanes (like P01's exp68/out68/err68).
- P04 pattern_change.step daily + mark_investigated; what_changed.step after errors mature; plan_test -> questions; new_trees ->
  priority; Conclusions reach the trader only via MaturedRecord.gate.
- P05 two_stage.run_day -> selection_constraint.select/apply_to_positions; exit_research.LearnedExitRule into exits.walk_forward;
  CommitmentBook().commit at expectation time; calibration_target.evaluate for grading; self_correct.step.
- CROSS-CUTTING: current_code_hash() re-reads and hashes all engine sources on every call - at least 4 builders hit 0.3 s-40 s
  slowdowns and cached it locally. Needs ONE canonical per-process cache that still detects on-disk edits (stale-code guard).
