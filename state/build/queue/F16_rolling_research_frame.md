# F16_rolling_research_frame  [model: opus]
C69 sections 12-14, 21, 27, 28; canon C56; follow-up to F14 (Masterstock JOURNAL "F14", 30 Sep).
Read state/build/CONTEXT.md (rules 1-29), engine/research/feeds.py (FrameStore, store.upto, research_frame), engine/research/evidence.py
(train_cut, unseen_dates), engine/research/loop.py (how the frame reaches the gate), tests/test_gate_calendar_and_episode_floor.py
(the test that documents the January shortfall), tests/test_research_dataflow.py.
You OWN: engine/research/feeds.py (FrameStore / upto / research_frame only), tests/test_research_dataflow.py, the documenting test in
tests/test_gate_calendar_and_episode_floor.py, NEW tests. evidence.py / loop.py / quality_gate.py are read-only. Never run git.
F15 is working on engine/livesim.py, features.py, parity.py - never edit those.
Finding (F14): `store.upto` keeps whole calendar years, so early in a year the research frame spans only 2 years and the gate can
see only 1 unseen year - every January look is starved exactly as December looks were before F14.
Do:
1. Replace the whole-calendar-year window with a rolling window of matured sessions ending at `now` (e.g. 156 weeks), strictly
   past-only (the fail-closed stage-input audit must still refuse anything at/after now), with a test that fails on today's code
   (a January look gets the same unseen evidence as a July look).
2. Memory/CPU: the frame is built lazily year by year today; keep the rolling window no more expensive than today's (state the
   measured time and peak RAM before/after on the planted world).
3. Re-run the gate proof (`python tests/test_regate_sequential.py gate --seeds 0 1 2 3 4 5 --world default` and `--world null`,
   <= 2 processes): genuine promoted >= F14's 5/6, coincidence/null still 0/30.
Report ruler counts, tests, table.
