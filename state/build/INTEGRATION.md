# Integration queue (hooks reported by builders; applied by the main session after the foundation wave)

## B01 pit (engine/pit.py, 49 tests)
- [ ] future_scramble_store(pipeline, store, as_of) over real PatternMiner / Memory / adaptive outputs
- [ ] backtest.py + livesim.py: fills via PITStore.executor(as_of).fill_next_open; run pit.audit_fills on results
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
- [ ] log_experiment writes outcome, reason, metrics
- [ ] major runs call checkpoint.write_checkpoint, run_report.build_report/write_report; Board.promote replaces improve.py promotion

## B12 second pass (engine/baseline.py, engine/experiment_memory.py, scripts/audit_registry.py; 82 tests)
- [x] log_experiment writes outcome/reason/metrics + Phase 0.2 fields (2026-09-28)
- [ ] freeze the baseline once with the champion cfg: baseline.freeze(...)
- [ ] live trader: Board.record_shadow_session daily, shadow_review after each session
- [ ] experiment_memory.check() before launching grid/loop candidates
- audit (real 53 records): 0 complete, 2 with provenance -> fixed going forward only; history stays as-is (append-only)

## Ownership transfers (2026-09-28)
- engine/memory.py, engine/adaptive.py -> B15 (asked to add Session.fills for the fill audit)
- engine/livesim.py, scripts/check_retester.py, scripts/livesim_cycle.py -> B10 (apply blind-gate/retester/health hooks)
- scripts/livesim_loop2.py -> B09 (objective.evaluate + BasisSearch; keep provenance + fill_audit gates)

## B09 objective/timeline/basis (54 tests)
- note: adaptive.py reads meta det_max / det_min_weeks not in META_DEFAULT (B15 to add)
- [ ] adaptive weekly step: merge timeline.cfg_overrides(timeline.step(...)) (needs gross-weight scale)

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
- [ ] model labels (features.labels) enter at close t; real fill is next open (mean gap +0.05%, abs 0.7%; 2020 1.3%)
- [ ] insider data: 142 rows filed before trade date, 6 year typos (13, 24) - clean at load
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
