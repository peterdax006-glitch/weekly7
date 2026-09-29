# B07_trust_direction
Bible: PHASES 12 and 13 (lines 965-1040). Estimated code: 2,500-5,000 lines.
You own: engine/trust.py, engine/direction.py, tests/test_trust.py, tests/test_direction.py

Per-stock-type trust tables (type = sector x size x vol bucket etc; shrinkage to parent; min support); direction engine: P(up) for predicted movers, calibration (isotonic/Platt on time-ordered holdout), Brier, log loss, reliability bins, >=80% confidence gate that abstains when calibration is insufficient.
