# F14_gate_calendar_and_episode_floor  [model: opus]
C69 sections 12-14, 27, 28, 31; C66 quality gate; ledger W-06. Follow-up to F12 (Masterstock JOURNAL "F12 DONE", 30 Sep).
Read state/build/CONTEXT.md (rules 1-29), engine/research/evidence.py (SequentialPlan, failure-episode evidence, complexity),
engine/research/quality_gate.py, engine/research/loop.py st_quality_and_knowledge, tests/test_regate_sequential.py,
state/research/regate_sequential/SUMMARY_f12_loops.md and gate_*_s*.json.
You OWN: engine/research/evidence.py, tests/test_regate_sequential.py (NOT its run_loop/main runner), NEW tests for this brief.
quality_gate.py and loop.py are READ-ONLY for you (report a real defect there with a failing test; the main session applies it).
Nine long runs of `tests/test_regate_sequential.py loop` are running (PIDs 26264 11332 8260 26552 20240 5740 11744 18484 8744):
never kill them, never edit run_loop/main, keep your own runs to <= 2 processes. Never run git.
F12 recorded two structural issues:
(a) every December look FAILS because the sliding three-year frame leaves a single unseen calendar year at that point;
(b) a very strong genuine effect cannot collect 5 failure episodes in time, so it waits at NEEDS_MORE_EVIDENCE (seeds 0 and 4).
Do:
1. Confirm each with a planted case and a test that fails on today's code.
2. (a) The unseen-years requirement must not depend on where the calendar boundary falls: count unseen evidence by matured
   dates/weeks, or align the frame, so a look in December is judged on the same amount of unseen evidence as one in June.
3. (b) A failure-episode floor must not punish a strong effect for failing rarely: make the requirement ask for enough
   evidence about failures (e.g. an upper confidence bound on the failure rate, or episodes OR sufficient exposure), with the
   same false-promotion control. Never lower min_effect or the alpha plan.
4. Re-run the gate proof (`tests/test_regate_sequential.py gate --seeds 0 1 2 3 4 5` for default and null; <= 2 processes):
   genuine promoted on more seeds with no false promotions (coincidence/null 0/21 before, report the new count).
Report ruler counts, tests, tables. Nothing VALIDATED by you.
