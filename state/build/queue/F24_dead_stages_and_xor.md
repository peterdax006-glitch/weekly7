# F24_dead_stages_and_xor  [model: sonnet]
C75 Phase 1 and section 4 ("health-book round-trip failure", "conjunction pre-screening can lose XOR relationships"), Firewall 3
(a stage reporting OK with 0 output is not integrated). Read CONTEXT.md (rules 1-29) and state/build/C75_PHASE0_MAPPING.md (gaps 5, 8, 18).
Findings: persisted health books do not reload (5 of 6 fail with "record altered" - reproduced by F20); the failed_lab and waste stages
of the research loop report OK with 0 output in all 129 cycles of every clean run; the conjunction pre-screen drops XOR pairs, so
XOR/interaction patterns can never be found.
You OWN: engine/learning/health.py (book persistence), engine/research/failed_lab.py, engine/research/waste.py, the conjunction
pre-screen module (find it: discovery.py / interactions.py / patterns.py - name it), and their tests. Never edit loop.py, evidence.py,
feeds.py, two_stage.py, decision_bridge.py, adaptive.py (F23), pattern_benchmark.py (F19). Never run git. One process only.
Do: (1) health-book round trip: find why reloaded records read as altered, fix at the source, test save->load->verify for all 6 cases.
(2) failed_lab / waste: find what input they wait for in the loop and why they produce nothing; make them do real work on the planted
world (a failed learner registered and reused; a wasteful research line detected and its compute redirected) with a test through the
real loop - or, if their input genuinely never occurs in the planted world, say exactly what is missing (do not fake it).
(3) XOR: a planted A XOR B (neither alone predictive) and A AND B pattern must survive the pre-screen and be found; keep the
pre-screen's cost bounded (state it); null features must not be admitted more than before (state the rate). Report causes, fixes, tests.
