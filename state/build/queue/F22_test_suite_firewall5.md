# F22_test_suite_firewall5  [model: opus]
C75 Firewall 5 (never lower tests: no skipped seeds, no asserting a bug, no excluded hard cases) and section 4 known failures
("unguarded infinite transfer ratio", "reference-context IC false alarms"). Read state/build/CONTEXT.md (rules 1-29) and
state/build/C75_PHASE0_MAPPING.md (gap 16 and section 4).
Findings (F20, 30 Sep): `REFERENCE_SEEDS` in the test suite skips the failing seeds 11/17/41; `test_known_gap_infinite_gain_inputs_are_not_guarded`
ASSERTS the bug. Search the whole tests/ tree for every other instance of the same pattern (skipped/filtered seeds, xfail/skip
that hides a real defect, tests asserting known-bad behaviour, reduced populations, `known_gap` / `known bug` wording) and list them.
You OWN: the tests you must change, and the engine modules the underlying defects live in (name them; do NOT touch
engine/research/pattern_benchmark.py (F19), engine/learning type annotations being edited by F21 - coordinate by editing only the
lines your fix needs, and never run git). One process only; a real-data Test loop is running.
Do: for each instance, restore the honest test (all seeds, the correct expected behaviour) - it will fail - then FIX THE SYSTEM at the
source (e.g. guard transfer_ratio against inf/zero denominators with a defined UNDEFINED/UNTESTED result; find why reference seeds
11/17/41 fail and fix the cause). If a requirement is genuinely impossible, keep the test failing-but-documented as a strict xfail with
the scientific reason and say so - never make it pass by weakening it. Report each instance, cause, fix, test result.
