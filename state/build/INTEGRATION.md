# Integration queue (hooks reported by builders; applied by the main session after the foundation wave)

## B01 pit (engine/pit.py, 49 tests)
- [ ] future_scramble_store(pipeline, store, as_of) over real PatternMiner / Memory / adaptive outputs
- [~] (loop2 fills proven per fill by engine/fill_audit.gate -> pit.audit_fills; backtest.py not yet) backtest.py + livesim.py: fills via PITStore.executor(as_of).fill_next_open; run pit.audit_fills on results
- [ ] train.py: pit.purged_training_set(X, y, as_of, horizon, cal) instead of ad-hoc label cutting
- note: Saturday filing -> strict next session needs busday roll='backward'

## B02 parity (engine/parity.py)
- [ ] live.todays_features: parity.require_parity(parity.run_parity(parity.Inputs(stocks, market, ev, ins, sic), fast=X, mode="live"))
- [ ] research runners: parity.cached_pass(path) before dependent experiments

## B07 trust/direction (engine/trust.py, engine/direction.py)
- [ ] TrustTable().fit(X, y, info, now) after PatternMiner.fit; tt.neutralize(scores, info_day) where miner scores are consumed
- [ ] DirectionEngine().fit(build_inputs(...), up, now, movers); eng.decide(F_day) -> only bet=True rows trade

## B12 registry/checkpoint/champion/run_report
- [x] improve.log_experiment ids -> sha256 (done 2026-09-28)
- [x] log_experiment writes outcome, reason, metrics (+ all Phase 0.2 fields; 2026-09-28)
- [ ] major runs call checkpoint.write_checkpoint, run_report.build_report/write_report; Board.promote replaces improve.py promotion

## B12 second pass (engine/baseline.py, engine/experiment_memory.py, scripts/audit_registry.py; 82 tests)
- [x] log_experiment writes outcome/reason/metrics + Phase 0.2 fields (2026-09-28)
- [x] freeze the baseline once with the champion cfg: state/baseline/B_2026-09-28_5a62d137 (2026-09-28)
- [ ] live trader: Board.record_shadow_session daily, shadow_review after each session
- [ ] experiment_memory.check() before launching grid/loop candidates
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
- [ ] route fills through PITStore.executor(...).fill_next_open + pit.audit_fills; train.py purged_training_set;
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
