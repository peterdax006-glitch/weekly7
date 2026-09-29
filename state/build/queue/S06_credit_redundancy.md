# S06_credit_redundancy
Contract sections 20, 21, 53. Checklist C08, F08-F09 support. Minimum lines: 1,200+600+650 = 2,450.
Read state/build/CONTEXT.md fully (rules 1-18) and SELF_LEARNING_CONTRACT.md sections 0-4, 58-62, 83, 85 plus the sections below, in full. C63: foundation first - all code + unit tests now, no real-data runs or tuning.
You own: engine/learning/credit.py, redundancy.py, disagreement.py; tests/test_learning_credit.py, test_learning_disagreement.py.

Build: Credit assignment over decision components (patterns, analogs, memory, direction, timing, risk) by ablation / leave-one-out / Shapley-style counterfactuals with interaction terms and stability across resamples; no blanket credit. Redundancy of five kinds (predictive, context, mechanistic, operational, risk) - two patterns may predict the same thing but protect different regimes. Disagreement engine: who disagreed, why, who was right, in what context, and whether disagreement itself predicts outcomes. Build on engine/ablation.py and engine/direction_ablate.py.
