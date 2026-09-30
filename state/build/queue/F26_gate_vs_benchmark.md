# F26_gate_vs_benchmark  [model: opus]
C75 Phase 3 (3B, 3C, 3E), Phase 10 loop (FIND -> CLASSIFY -> UNDERSTAND -> FIX -> TEST -> ADVERSARIAL -> FRESH HOLDOUT), Firewalls 5, 6,
10; canon C70-C74. Read CONTEXT.md (rules 1-29), F19's report (Masterstock JOURNAL "F19"), state/research/pattern_benchmark/
final_look/SUMMARY.md, engine/research/pattern_benchmark.py, engine/research/evidence.py, engine/research/quality_gate.py.
Benchmark baseline (20 worlds, final-look protocol, current screen + evidence.gate): detectable real found 3/346; false positives
5.6 per world (stated alpha allows ~0.003); precision 2.6%; null worlds promote 6-8.
F19's ranked gate failures (you own these):
 1. Leak-shaped features are promoted: 111/112 false positives; the gate's leak screen only checks |corr| > 0.9 with the SIGNED
    return, so a feature built from |outcome| (or any non-linear function of the future) passes it, and passes calibration and
    complexity too. The honest defence is point-in-time: a feature whose value at t depends on data after t must be refused -
    e.g. a truncation / future-scramble test on the candidate's own construction (the F09 method), plus a dependence test against
    functions of the future outcome (|y|, y^2, rank), not only linear correlation with y.
 2. Calibration blocks 96% and replication 91% of gated REAL patterns (complexity 53%, risk 25%): obvious/moderate linear,
    conditional, changing and lifecycle real patterns all end FAILED. Find why per gate with the new per-gate detail (does
    calibration test a probability model the pattern never claimed? does replication demand the same effect in halves too small
    to have power?). Fix a gate only when it is WRONG for the claim being tested; never lower a threshold to pass a pattern.
 5. Nothing retires: a FAILED candidate stays live and is re-gated every quarter (growing load).
 6. Multiplicity understated: the gate is told how many candidates were RAISED (~60-100), not how many were SCANNED (~615).
You OWN: engine/research/evidence.py, engine/research/quality_gate.py, engine/research/pattern_benchmark.py (also fix its 11 mypy
errors and delete its `exclude` line in pyproject.toml), their tests. loop.py read-only (report defects). Do not edit
engine/patterns.py (F25), feeds.py/two_stage.py/decision_bridge.py/adaptive.py (F23), failed_lab/waste/health/interaction screen (F24).
Never run git. <= 2 processes (a real Test loop and builders are running).
Rules: tune ONLY on development seeds; the held-out seeds (state/research/pattern_benchmark/heldout_seeds.json) are scored only after
each fix is frozen, and reported separately. Never change the scoring, the answer key, the world generator's difficulty or the
detectability floor to improve the score.
Report after each fix: dev and held-out table (real found by kind/strength, false positives per world by noise kind, null worlds,
precision, calibration/Brier), what changed, and the gate-by-gate block rates before/after. Also measure and cut the 3.3 s per gate call
(mostly the identity harness) without weakening it - 500 worlds must become feasible; state the new per-world cost.
