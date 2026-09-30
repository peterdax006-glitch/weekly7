# F13_stale_promotion_rollback  [model: opus]
C69 sections 12-15, 27, 28, 31; C68 (PREDICTION_ERROR_ADDITION.md: forward-only regimes, self-correction, rollback); ledger W-06.
Read state/build/CONTEXT.md (rules 1-29), the Masterstock JOURNAL entry "F11 DONE" (30 Sep), state/research/c68_fix_promote/
(f11_flip2_n60 = the valid flip runs; f11_flip_n60 = rejected exploding runs, ignore), scripts/c68_fix_promote.py,
engine/research/self_correct.py, error_loop.py, change_points.py, regime_memory.py, tests/test_c68_fix_promote.py.
You OWN: engine/research/self_correct.py, error_loop.py (promotion/rollback/monitor code), scripts/c68_fix_promote.py,
tests/test_c68_fix_promote.py, NEW tests for this brief. Do NOT edit engine/research/loop.py, evidence.py, quality_gate.py or
engine/learning/*. Nine research-loop runs (tests/test_regate_sequential.py loop, PIDs 26264 11332 8260 26552 20240 5740 11744
18484 8744) are running on the current code: never kill them; keep your own runs to <= 2 processes while they run.
Finding (F11): flip seed 0 - the planted error STOPPED at 80% of the sample, the fix was PROMOTED AFTER the flip on stale
evidence (its OOS evidence predated the flip) and was NOT rolled back before the data ended (monitor t only -1.57).
Do:
1. Reproduce flip seed 0. Explain exactly why stale evidence could promote after a detected change point (C68 forward-only
   regimes): does the promotion gate ignore change_points / regime_memory, or does the evidence window straddle the change?
2. Fix at the source: evidence for a promotion must be drawn from the regime in force at the decision (or be shown to hold across the
   change); a detected change point must trigger re-validation of fixes whose evidence predates it. Never lower a gate.
3. Rollback: a promoted fix that stops working must be rolled back within a stated bound; measure the delay distribution.
4. Prove over >= 6 flip seeds + the 6 planted + 6 null twins (report the same table as F11): promotions after the flip on stale
   evidence = 0; planted promote rate not worse than F11's 3/6 without cause; null promotions 0. Tests under 90 s per file.
Report ruler counts, tests, tables. Nothing VALIDATED by you.
