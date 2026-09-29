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
