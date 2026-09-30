# F25_miner_noise_admissions  [model: opus]
C75 section 4 (open problems), Firewall 5, Phase 3 (noise rejection), canon C70-C74 (zero false positives). Read CONTEXT.md
(rules 1-29) and state/build/C75_PHASE0_MAPPING.md.
Finding (F22, 30 Sep): tests/test_patterns_integration.py::test_noise_false_admissions_within_budget is a strict xfail - on PURE NOISE
the PatternMiner (engine/patterns.py) admits false patterns beyond its stated budget (the Phase 25 false-discovery excess). This is a
real open defect, not an impossible requirement.
You OWN: engine/patterns.py and its tests (tests/test_patterns*.py). CONTEXT rule 29: patterns is research-only and must never
become reachable from the trader path - do not import it from anything the trader reaches. Never run git. One process only; a real
Test loop and other builders are running. Do not edit engine/research/pattern_benchmark.py (F19), feeds.py/two_stage.py/
decision_bridge.py/adaptive.py (F23), health.py/failed_lab.py/waste.py (F24).
Do: 1. Measure the false-admission rate on >= 200 pure-noise panels (state the stated budget and the measured rate with a CI).
2. Find WHY it exceeds the budget (multiplicity not controlled across the search, null distribution too narrow, look-ahead in a
filter, data-dependent choices before the null, etc.) with evidence. 3. Fix at the source so the rate is within budget, WITHOUT
lowering power on planted real patterns more than necessary - report recall on the existing planted patterns before/after.
4. Remove the xfail; the test must pass honestly (never widen the budget). Report rates, cause, fix, tests, ruler counts.
