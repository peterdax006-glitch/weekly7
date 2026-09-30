# F08_open_hooks  [model: opus]
C69 ledger W-14 (integration); C69 sections 5, 20, 27; INTEGRATION.md B01, B07, B12 open boxes.
Read state/build/CONTEXT.md (rules 1-28), state/build/INTEGRATION.md (B01, B02, B07, B12, B12 second pass), EXECUTION_LEDGER.md
section 5 "Integration gaps" item 8.
You OWN: engine/backtest.py, engine/train.py, engine/pit.py (call sites only), engine/experiment_memory.py (call sites), the grid /
search entry scripts that launch candidate batches (scripts/*grid*.py, scripts/*search*.py - list which you touched), and NEW
tests/test_open_hooks.py. NEVER edit scripts/livesim_loop2.py or engine/leak_audit.py (F06), engine/learning/learner.py,
loop_hooks.py, test_path.py, planted_world.py, scripts/acceptance_mini.py (F07), engine/research/error_loop.py + C68 modules +
two_stage.py (P06), or anything under live/ (C65: Live is parked - B02 live parity and Board.record_shadow_session in the live
trader stay open; say so).
Do (each hook: applied on the production path, proven REACHABLE with scripts/reachability.py, and proven to CHANGE behaviour with a
test that fails if the hook is removed):
1. B01: backtest.py fills via PITStore.executor(as_of).fill_next_open and pit.audit_fills on its results (loop2 already does this
   via fill_audit.gate - reuse, do not duplicate). train.py uses pit.purged_training_set(X, y, as_of, horizon, cal) instead of
   ad-hoc label cutting; show the purge removes the overlapping rows. future_scramble_store over the real PatternMiner / Memory /
   adaptive outputs (a test: scrambling data after as_of changes nothing at as_of).
2. B07: TrustTable and DirectionEngine are real modules with no production caller. Direction has no measured edge (51.8%), so wire
   them behind explicit config flags, OFF by default, with a test proving that ON changes decisions and OFF is byte-identical to
   today. Do not claim they add value.
3. B12: major runs (backtest, research_loop, livesim_loop2 is F06's - leave it) call checkpoint.write_checkpoint and
   run_report.build_report/write_report; experiment_memory.check() runs before any grid/loop candidate launch and skips exact
   repeats (test with a planted repeat).
Report: each hook -> file:line, reachability line, test name; ruler counts; test results. Tick the INTEGRATION.md boxes you closed.
