# F10_learner_enforce_evidence  [model: opus]
C69 ledger W-07 continued; C62 C03, E04, I16, I18; C69 sections 12-14, 27, 28, 31.
Read state/build/CONTEXT.md (rules 1-28), state/research/acceptance_mini/diagnosis.md (F07), engine/learning/scorecard.py
(scorecard_for_learner, LearningScorecard claim gate), engine/learning/learner.py, retirement.py, wiring.py, scripts/acceptance_mini.py.
You OWN: engine/learning/scorecard.py, learner.py, retirement.py, wiring.py (identity-blocker wording only), planted_world.py,
scripts/acceptance_mini.py, their tests. NEVER edit engine/research/error_loop.py, the C68 modules or two_stage.py (P06 is running).

OWNER-SIDE RULING (main session, 29 Sep): `learning_claim="enforce"` stays the default and its bar is NOT lowered. F07 found it
deadlocks because scorecard_for_learner never measures forward-year folds or portfolio fields in a one-year world. The fix is to
SUPPLY the evidence, not to relax the gate.
Do:
1. A multi-year planted acceptance world (>= 4 simulated years, truths that persist across years plus one that decays) so
   forward-year folds exist; the null world gets the same length.
2. A portfolio-level card: from the learner's simulated weekly picks compute the fields the claim gate needs (risk, drawdown, band
   share, calibration) with the same Measured/bootstrap conventions; merge it into the learner's scorecard. Untestable fields stay
   UNTESTED, never filled with defaults.
3. Measure F07's next hypothesis: does the 8-week retirement window with degrade_t 1.0 falsely degrade true items? Report the false
   degrade rate on planted truths and on null items; change it only if the evidence says so, with a test.
4. Wiring: word the identity blocker honestly (F07 note); adopt recover_degraded in retirement.py natively if it belongs there.
5. Run acceptance_mini under enforce AND record on >= 6 seeds + the null world (detached, RAM >= 2.5 GB). Report the table: improved
   seeds, mean vs none/control, null false improvements (must stay 0), changed-without-knowledge (must stay 0), and which gate
   blocks each seed that still fails. Keep each test file under 90 s (mark long runs W7_LONG_TESTS).
Report ruler counts, test results, tables. Nothing is VALIDATED by you.
