# F28_benchmark_round2  [model: opus]
C75 Phase 3 / Phase 10 repair loop; canon C70-C74; Firewalls 5, 6, 10. Read CONTEXT.md (rules 1-29), F26's report (Masterstock JOURNAL
"F26"), state/research/pattern_benchmark/f26/SUMMARY.md, engine/research/evidence.py, quality_gate.py, pattern_benchmark.py.
State after F26 (20 worlds, final-look): real found 71/246 dev, 25/100 held-out; false positives ~0.9 dev / 1.3 held-out per world;
null worlds 0. Remaining false positives (F26): PROXIES 16 of 21 - a candidate correlated with a real pattern that adds nothing beyond
it; identity_null 2 - a name-level artefact counted as name-week units; weakest leaks (strength 0.3) in low-power worlds, z 2.6-3.4
under a 4.06 bar. Defects F26 reported in modules it did not own:
 a. engine/learning/calibration.platt_slope diverges (slopes ~1e7) - needs damping;
 b. engine/learning/complexity.compare - worst-fold tolerance units error (must be in each fold's own SE);
 c. engine/learning/identity_firewall - adopt F26's vectorised scoring; `_multiset_key` unstable when data hold both -0.0 and 0.0;
 d. the research firewall's IDENTITY layer quarantines rules that merely have no skill (no skill is not an identity leak).
You OWN: engine/research/evidence.py, quality_gate.py, pattern_benchmark.py, engine/learning/calibration.py, complexity.py,
identity_firewall.py, the research firewall's IDENTITY layer (engine/research/firewall.py - that layer only) and their tests.
Do NOT edit candidate generation (discovery*.py, targets.py, interactions.py, candidate_forms.py - F27 is running), feeds.py,
two_stage.py, decision_bridge.py, adaptive.py, patterns.py, loop.py (report needed hooks). Never run git. <= 2 processes.
Do (tune on DEVELOPMENT seeds only; held-out scored only after each fix is frozen; never change scoring/answer key/difficulty/floor):
1. Proxy test: a candidate is promoted only if it carries incremental out-of-sample value beyond the strongest correlated candidate
   (and beyond already-filed knowledge); report the proxy FP count before/after and real-pattern loss.
2. Identity units: tests whose effective sample is names (not name-weeks) must count names; fix and report.
3. Weak leaks: explain why the leak screen's bar misses strength-0.3 leaks in low-power worlds and close it without refusing genuine
   new-information signals (F26 flagged: a real scheduled-event magnitude signal would trip the leak screen - plant one on dev seeds
   and show it is not refused).
4. Fix a-d at the source with tests; re-run the affected learning test files (they are used by the C62 learner).
5. Scale: run the benchmark on >= 100 worlds (dev + held-out, final-look protocol; F26 measured ~69 s/world) and report the full
   table by kind, strength band, era, noise family, tier and null worlds, with CIs. State what blocks "all detectable real found, 0 FP".
Report tables, causes, fixes, tests, ruler counts.
