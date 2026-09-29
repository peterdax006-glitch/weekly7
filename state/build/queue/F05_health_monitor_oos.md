# F05_health_monitor_oos  [model: sonnet]
C69 ledger W-05 (remainder) and sections 15, 27, 28. Read state/build/CONTEXT.md (rules 1-28) and state/build/EXECUTION_LEDGER.md W-05.
You OWN: engine/pattern_reliability.py (health monitor only), engine/research/break_research.py (the explain_break call site only),
tests/test_pattern_reliability.py, tests/test_research_breaks.py. Never lower a test or threshold; every fix gets a test that fails on
the old behaviour.
F02 (29 Sep) trimmed the health monitor's false alarms on weak stationary patterns only from 0.86 to 0.75 of the nominal budget
(205 -> 178 alarms over 240 patterns; 42% flagged) by capping the shrunk mean at a lower confidence bound (effect_lcb_z = 0.5); a
stronger cap broke test_phantom_pattern_is_caught_within_a_bounded_delay. The named real fix: estimate each pattern's EXPECTED effect
OUT OF SAMPLE (e.g. a rolling cross-fitted estimate: the effect the monitor compares against is estimated on data disjoint from the
data the monitor is currently judging, past-only), so the winner's curse no longer sets the bar. Build it; show on >= 240 weak
stationary patterns that false alarms fall materially below 0.75 of budget (state the new bound in a test) while planted decays are
still caught (>= 90%, median delay not worse than today's <= 60 weeks) and the phantom-pattern test still passes. Then switch
engine/research/break_research.py (~line 2122) from explain_break to break_detection.explain_break_controlled so the pattern-break
investigations also get F02's placebo bar; test that a placebo-only explanation is refused there. Keep each test file under 90 s.
