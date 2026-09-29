# R09_accuracy_frontier_symmetry  [model: sonnet]
C66 §11 (1,500-2,500), §12 winner/loser symmetry + loss-risk bank (2,000-3,000).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/frontier.py, engine/research/symmetry.py; tests/test_research_frontier.py.
Build on (import, extend - never copy): engine/learning/calibration.py, scorecard.py, engine/direction_calib.py.

Build: Coverage->accuracy frontier (100/50/25/10/5/1% and finer) with n, CI, calibration, base rate, weeks, stocks, independent obs, era/sector/regime stability; detects 'high accuracy only from a tiny sample'; the 80% question answered as accuracy + coverage + calibration + OOS + transfer + risk and able to output 'No reliable 80% directional region has been found.' Symmetry engine: TP/FP/TN/FN, missed winners/losers per pattern incl. wrong-direction, large-loss, regime-transition, external-event, near-miss, high-vol, low-liquidity cases; a LOSS-RISK knowledge bank separate from the opportunity bank.
