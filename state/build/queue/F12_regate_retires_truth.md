# F12_regate_retires_truth  [model: opus]
C69 ledger W-06 / W-12 finding (29-30 Sep); C69 sections 12-14, 27, 28, 31; C66 quality gate.
Read state/build/CONTEXT.md (rules 1-29), state/research/research_loop_w12.log + state/research/research_loop_w12_summary.json
(the W-12 run: 60 cycles, default settings, knowledge 0), engine/research/loop.py st_quality_and_knowledge (REGATE_MAX_LOOKS=3,
REGATE_NEW_DATES=13), quality_gate.py, evidence.py, feeds.py planted truth, the run dir state/research/research_loop/w12_planted_persisted/.
You OWN: engine/research/loop.py (st_quality_and_knowledge and the re-gate constants only), engine/research/evidence.py,
tests/test_research_dataflow.py, NEW tests/test_regate_sequential.py. quality_gate.py thresholds are NOT to be lowered (you may fix
real defects, with a test). NEVER edit engine/learning/* (F10) or error_loop.py/self_correct.py/error_research.py/
prediction_error.py/calibration_target.py (F11).
Finding: at default settings finding DBR44ef231394d2 was gated at cycle 31 (FAILED), 45 and 59 (NEEDS_MORE_EVIDENCE) and then
RETIRED after 3 looks; W02 measured that the default min_effect needs ~30 fresh weeks of holdout, but 3 looks x 13 dates < that.
Do:
1. Identify DBR44ef231394d2 (and DBR2d20b91b1f58) against the planted truth: genuine (lv20 family), coincidence, or other?
2. If a genuine effect can be retired before the evidence it needs can exist, that is a defect: replace the hard look cap with an
   honest sequential design (e.g. alpha-spending across looks, or looks scheduled only when the holdout reaches the power the gate
   needs) so repeated looks cannot manufacture a pass AND a true effect is not killed for lack of time. Do not lower min_effect.
3. Prove over >= 6 seeds on the planted world at DEFAULT settings (long runs behind W7_LONG_TESTS, detached, RAM >= 2.5 GB): the
   genuine pattern reaches PROMOTE in a stated number of cycles; the coincidence is never promoted; a null world promotes nothing
   (state the false-promotion rate bound).
Report ruler counts, test results, per-seed table.
