# B11_antioverfit
Bible: PHASES 26, 33, 34 (lines 1447-1494, 1693-1729). Estimated code: 2,000-4,000 lines.
You own: engine/antioverfit.py, engine/repro.py, engine/ablation.py, tests/test_antioverfit.py, tests/test_repro.py, tests/test_ablation.py

Anti-overfitting battery: label permutation, feature shuffle, ticker permutation, dead-feature injection, duplicate features, date-shift; each returns a verdict vs the real run. Reproducibility (run twice, compare artefacts by hash). Generic ablation framework (remove component X, compare with CI).
