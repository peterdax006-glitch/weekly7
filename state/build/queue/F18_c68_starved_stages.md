# F18_c68_starved_stages  [model: opus]
C68 (PREDICTION_ERROR_ADDITION.md sections on error research depth and what-changed); C69 sections 5, 27, 31 (a stage that never
receives input is shallow, not done). Read state/build/CONTEXT.md (rules 1-29), state/research/regate_sequential/SUMMARY_clean_loops.md,
engine/research/error_loop.py (stage registry, st_* for research_depth, what_changed, error_research), what_changed.py,
error_research.py, the run folders state/research/regate_sequential/f12_default_s{0..5}/reports/cycle_*.json.
Finding: in the clean full loops (129 cycles, default settings, planted world) c68.research_depth and c68.what_changed were
SKIPPED_NO_INPUT in 129/129 cycles and c68.error_research ran OK only 4-8 times; every other C68 stage ran.
You OWN: engine/research/error_loop.py, what_changed.py, error_research.py and their tests. Never edit loop.py, evidence.py,
feeds.py, quality_gate.py, engine/learning/*. Never run git. A real-data Test-loop run (livesim_loop2, PID 26356) is running:
use at most ONE process of your own, never kill others.
Do:
1. From the recorded reasons, say exactly which input each starved stage waits for and why it never arrives (never produced,
   produced under another key, gated behind a threshold the planted world never reaches, or dependent on a stage that is off).
2. If the planted world simply never produces the trigger (e.g. no confident failure repeated enough), plant one honestly and show
   the stage runs; if the wiring is wrong, fix the wiring. Never lower a trigger threshold to make a stage run.
3. Test: each stage runs at least once through the real loop on a world that has its trigger, and stays SKIPPED on a null world
   without it. Tests under 90 s per file.
Report cause per stage, fixes, tests, ruler counts.
