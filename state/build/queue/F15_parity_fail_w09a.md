# F15_parity_fail_w09a  [model: opus]
C69 sections 21-22 (anti-cheating), 27, 28; canon C56 (only live information), C64; ledger W-13.
Read state/build/CONTEXT.md (rules 1-29), engine/livesim.py (parity_test ~line 540-570, Feed), engine/features.py,
engine/parity.py, scripts/livesim_loop2.py, state/research/feature_leak/report.md (F09 truncation audit: main builder CLEAN).
Finding (30 Sep 03:19, first run of `scripts/livesim_loop2.py 30 --learner legit`): window w09a's worker died with
`RuntimeError: PARITY FAIL: fast features differ from live-computed ones (max diff 1) - possible leakage` (traceback in
state/livesim/w09a/health2.err). w09b/w09c passed parity (they crashed later on a learner NaN bug, now fixed). Earlier rounds
(windows 1-24) passed parity. Recent edits that touch the feature/feed path: F09 (volatility_lab.event_inputs filing_n5 NaN->0,
feeds.market_proxy bfill), F06 (livesim Feed.long_term_memory bank exclusions, train_basis gate), F08 (train.py purge).
You OWN: engine/livesim.py (parity_test and Feed only), engine/features.py, engine/parity.py, NEW tests/test_parity_w09a.py.
Never run git. state/livesim is sealed: you may read w09a's worker files (health2.err/jsonl, parquet artefacts) ONLY to
reproduce the parity check; never read or print the real year or anything that identifies it, and never read other windows' results.
Do:
1. Reproduce the parity failure for w09a deterministically (same sealed window, same seed) and name the exact feature(s), dates
   and tickers (tickers/dates may be shown as disguised ids only) where fast != live, and why (diff of exactly 1 suggests a
   count/flag/rank feature).
2. Decide with evidence: is it a REAL future leak in the fast path, a live-path bug, or a benign edge (e.g. a tie, a boundary
   session)? A benign edge is only benign if you can prove neither path reads the future; the parity tolerance is NOT to be widened.
3. Fix at the source, with a test that fails on today's code; re-run parity on w09a and on 3 other sealed windows.
4. Report: cause, fix, test, and whether any earlier result used the leaking feature.
