# Weekly7 builder context pack (read fully before coding)

Repo: C:\Users\Peter\weekly7. Python: `.venv/Scripts/python` (3.11 x64; pandas, numpy, lightgbm, scipy, sklearn, pytest).
Governing doc: `BIBLE.md` (read the phase section your task file names, in full). Design: `MASTER_BLUEPRINT.md`.
Goal of the system: pick stocks so a $1,000 simulated portfolio moves ~7%/week (band 5-10%), then minimise risk, then
raise the share of positive weeks. Four projects: Algorithm (pattern discovery), Find volatility (movers -> direction ->
exits -> stops), Test (blind simulated years), Live (Alpaca PAPER only).

## Hard rules
1. You own ONLY the new files named in your task file. Never edit existing files (others are editing in parallel).
   If integration into an existing file is needed, write it as a function in YOUR file and list the exact hook
   (file, function, one-line call) under "INTEGRATION" in your final report.
2. Real, meaningful code. No padding, no placeholder `pass` bodies, no TODO stubs, no duplicated boilerplate to raise
   line count. Detail comes from real mechanisms: validation, edge cases, statistics, diagnostics, reports, tests.
   Bible line-count estimate for your phase is in the task file; land inside it with substance.
3. No look-ahead: every function that uses time takes an explicit `now`/`as_of` and must never read data after it.
   Decisions at a close fill at the NEXT session's open. No after-hours/weekend trading. No crypto. Paper only.
4. Deterministic: every random draw takes an explicit seed / np.random.Generator.
5. Tests: pytest files in `tests/` named in your task. Synthetic data only, each test file < 90 s and < 400 MB RAM
   (the machine is shared with long experiments - never load the full caches in tests). Run your tests; they must pass.
   Include at least one test that plants a known effect/defect and proves the code catches it (a check that cannot
   fail is worthless), and one on the empty/degenerate case.
6. Interfaces already in the repo you may IMPORT (read, don't modify):
   - `engine.config as K` (paths K.ROOT, K.DATA, K.CACHE, K.STATE)
   - `engine.provenance.stamp(cfg, seed)`, `engine.improve.log_experiment(rec, cfg=None, seed=None)`
   - `engine.patterns.PatternMiner` (fit(X, y, now) -> .patterns DataFrame with key_named, effect, t_disc, t_conf,
     p_real, p_hallucinated, p_coincidence, status in active/rescoped/no_gain/duplicate/discarded/rejected; .score(Xday))
   - Panel convention: X = DataFrame indexed by MultiIndex (date, ticker), feature columns; market-context columns
     start with `m_`. y = forward return Series on the same index.
   - `engine.memory.Memory`, `engine.adaptive` (Session, replay, Adapter), `engine.analogs.Analogs`
7. Don't use the network in tests. Don't touch `state/livesim/` (sealed test windows) or `data/cache/`.
8. Style: match the repo - compact, docstring per module stating which Bible phase and canon it serves, comments only
   where the why is non-obvious.

## Final report (keep under 200 words)
Files + line counts; test result line (N passed); what is honestly UNPROVEN; INTEGRATION hooks; anything you learned
that the owner should record.

## Added after wave 1 (2026-09-28)
9. Line range is a requirement: every wave-1 builder undershot and was sent back. Depth = a real-cache runner script
   (scripts/, background, results to state/research/<module>/ with provenance), report outputs, per-era/per-type
   breakdowns, planted-defect tests per mechanism.
10. RAM is shared by ~13 builders plus long experiments. Before launching any real-data run, check free memory
    (`.venv/Scripts/python -c "import psutil;print(psutil.virtual_memory().available/1e9)"`); if under 2.5 GB,
    wait and retry (poll every 60 s, give up after 20 min and report). Use float32, column subsets, seeded ticker samples.
