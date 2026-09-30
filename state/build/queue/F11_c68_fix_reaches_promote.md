# F11_c68_fix_reaches_promote  [model: opus]
C69 ledger W-06 (C68 part); C68 PZ20, PC11, PC12, PC16; C69 sections 12-14, 27, 31.
Read state/build/CONTEXT.md (rules 1-29), PREDICTION_ERROR_ADDITION.md (the spec; never edit), engine/research/error_loop.py,
self_correct.py, error_research.py, prediction_error.py, calibration_target.py, quality_gate.py, evidence.py, and P06's notes in
the Masterstock journal (29 Sep "P06 DONE").
You OWN: engine/research/error_loop.py, self_correct.py, error_research.py, prediction_error.py, calibration_target.py, and their
tests (tests/test_research_error_loop.py, tests/test_c68_adversarial.py, the P01-P05 test files). NEVER edit engine/learning/*
(F10 is running) or loosen quality_gate.py / evidence.py thresholds (read-only for you; report any real defect).
Facts: in P06's planted run no self-correction fix ever reached PROMOTE on its own evidence (promote/rollback tested only by forcing
the verdict); escalation happens only on stock-type groups, never on pattern groups or 'all'.
Do:
1. Plant a GENUINE, fixable, persistent prediction error in the W02 planted world (e.g. a miscalibration of p_move in one stock
   type that persists across years) and a NULL twin with the same noise but no error. Prove on the real loop that the fix is
   proposed, validated out of sample, and PROMOTED by the real gate on its own evidence, and that the null twin's fix is never
   promoted (>= 6 seeds each; state the rates).
2. Escalation: repeated confident errors on a pattern group and on 'all' escalate research priority (C68 PC11, PC12); test both,
   plus a null control that does not escalate.
3. Rollback: a promoted fix that degrades out of sample is rolled back by the real path (not a forced verdict).
4. Keep each test file under 90 s (long runs behind W7_LONG_TESTS=1, run them once and report).
Report ruler counts, test results, rate tables. Nothing is VALIDATED by you.
