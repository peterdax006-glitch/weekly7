# B25_direction_research
Checklist V2/V5 (BIBLE.md master checklist; canon C23/C24, C56). Estimated code: 1,000-2,000 lines (a requirement).
You own: engine/direction_features.py, scripts/direction_research.py, tests/test_direction_features.py, state/research/direction2/. Read (do not edit) engine/direction.py, engine/fv_pipeline.py, engine/features.py, engine/candles.py, engine/patterns.py.

Fact base: movement is predictable (movers touch +-10% in-week 57.9% vs 14.6% base; miner IC ~0.35 on |move|). Direction on movers
is a coin flip so far (B07: 51.8%, Brier = base rate; V5 run: the 80% gate never opened in 716 weeks) - BUT B07 used stand-in inputs,
not the system's real signals, and scored realised movers, not predicted ones.

Task: test, walk-forward and point-in-time (C56), whether ANY direction signal exists on PREDICTED movers using real,
literature-backed inputs, and map the accuracy/coverage frontier instead of only the 80% gate:
- post-earnings-announcement drift (earnings 8-K timing + the earnings-reaction features in engine/features.py);
- insider buying clusters (ins_buyers30 / opportunistic);
- short-term reversal vs 12-1 momentum and 52-week-high distance, by mover type;
- the pattern miner's direction score (engine.patterns, trained past-only);
- event type of the move (earnings / filing / none) and market regime.
Report per input family and combined: accuracy, Brier vs base rate, calibration, and the frontier "accuracy achievable at coverage
c" for c in {1%, 5%, 10%, 25%, 100%} with bootstrap CIs over weeks, per era; plus random/shuffled-label controls that must read ~50%.
Honesty rules: every threshold is chosen on earlier blocks only; report the 80% question plainly (at what coverage, if any, is a
calibrated 80% reached with a CI lower bound >= 0.80?). Use a seeded ticker sample, RAM >= 2.5 GB, long runs DETACHED (PowerShell
Start-Process) with checkpoints. Tests: a planted world with a real direction signal in one event type (must be found at that
coverage) and a null world (must stay ~50% at all coverages). Never run git; never kill by name.
