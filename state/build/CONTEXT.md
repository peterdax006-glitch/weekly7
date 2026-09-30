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
11. NEVER kill processes by image name (`taskkill /IM python.exe`, `pkill python`): on 2026-09-28 a builder did this
    and killed every experiment on the machine (a 91%-done SEC job, a movers grid). Kill only a PID you started
    yourself (record it when you launch). Long jobs must checkpoint so a kill costs minutes, not hours.
12. Git: never run `git pull --rebase`, `git stash` or `git reset` - other builders' uncommitted work lives in the
    same working tree. You do not commit; the main session (and an auto-snapshot every 15 min) does.

## Self-Learning contract builders (S-series, from 2026-09-29; canon C62 + C63)
13. THE SPEC is `SELF_LEARNING_CONTRACT.md` (owner contract, verbatim; never edit). Read sections 0-4, 58-62, 83, 85 and every
    section your task file names, in full. Your task file names the checklist ids (A01..L26) you own.
14. C63 FOUNDATION FIRST: write ALL the code for your sections now, to at least each section's minimum meaningful line count
    (section 60) with real mechanisms. Unit tests are part of the code: every module gets tests that run in seconds and include
    a planted case it must catch plus the empty case. NO real-data runs, no tuning, no perfecting in this wave - that is the
    next wave. Mark your work "IMPLEMENTED — NOT VALIDATED" (never VALIDATED).
15. Package: `engine/learning/`. Import the shared vocabulary from `engine.learning.core` (Epistemic, Lifecycle, Promotion,
    Health, Unknown, FailureCause, Subsystem, TemporalClass, DecisionEffect, Edge, Layer, Provenance, Confidence,
    KnowledgeLike, FirewallBreach, stable_hash, require_past, current_code_hash). Do NOT edit core.py; if you need a new
    shared term, define it in your module and list it under INTEGRATION. Do not import other S-builders' modules (they are
    being written in parallel); work on KnowledgeLike duck types and plain records.
16. REUSE, don't duplicate: before writing, grep engine/ for the job (e.g. experiment_memory.py, champion.py, health.py,
    pattern_lifecycle.py, antimemo.py, leak_audit.py, pit.py, isolation.py, repro.py, trust.py, missed_winners.py,
    checkpoint.py, patterns.py, pattern_memory.py, pattern_reliability.py). Wrap/extend them from your new file; never a
    parallel copy. Say in your report what existing code you built on.
17. Types: frozen/slotted dataclasses or explicit classes with validate() - never one giant untyped dict (section 5).
    Everything time-aware takes `now` and must fail closed (FirewallBreach) on anything dated at/after it.
18. Report additionally: per contract section, meaningful lines written (non-blank, non-comment) vs the minimum.
19. ONE RULER for line minimums: `.venv/Scripts/python scripts/contract_lines.py <your files>` (excludes docstrings, comments,
    blank lines). Report those numbers. A module under its minimum is incomplete (contract section 83) unless you name where
    the equivalent functionality genuinely lives.

## Research-brain builders (R-series, from 2026-09-29; canon C66 + C63)
20. THE SPEC is `RESEARCH_BRAIN_CONTRACT.md` (owner, verbatim; never edit). Read sections 0, 1, 29-31, 43, 44, 48, 53 plus
    every section your brief names, in full. It EXTENDS C62 (SELF_LEARNING_CONTRACT.md); both bind.
21. Package `engine/research/`. Shared vocabulary: `engine.research.core` (Namespace, Knowability, Availability, MoveCategory,
    ResearchState, Stage, GateVerdict, Problem, Horizon, ExperimentValue, ResearchQuestion, MaturedRecord) and
    `engine.learning.core`. Do NOT edit either core; define new shared terms in your module and list them under INTEGRATION.
22. EXTEND, never duplicate: your brief names the existing engine/learning or engine modules that already do part of your job.
    Import and build on them. A parallel copy of an existing mechanism is a defect. Say in your report what you built on.
23. C63 (re-affirmed by the owner for C66): code + unit tests only. No real-data runs, no tuning, no section-46 test programme.
    Every module gets fast tests with a planted case it must catch, a null case where it must find nothing, and the empty case.
24. Blind trader vs research world (C64, C66 sections 29-31): anything built from matured outcomes lives in
    MATURED_RESEARCH_STATE and may reach a decision only through MaturedRecord.gate(now) / engine.learning.curator. No real
    date, year or ticker may appear in anything handed to the trader (engine.learning.trader_view refuses them).
25. Expose ONE clear public entry per module (e.g. `step(state, now)`, `run_day(...)`), because the wave-2 research loop will
    call it. List it under INTEGRATION. Reachability is proven in wave 2 by scripts/reachability.py, which must reach your module.
26. Ruler: `.venv/Scripts/python scripts/contract_lines.py <your files>`. Report counts against your brief's minimum. Scratch
    files must use unique names in your own scratchpad (never shared /tmp names - builders clobbered each other before).
27. From state/build/RESEARCH_MAPPING.md (read your row): research that uses hindsight runs on the TRUSTED (curator) side,
    never inside the blind learner. Research filed under a real year must NEVER be released while that same year is being
    replayed in disguise (the same-year rerun leak). Market-wide work must STREAM year by year: ~38M name-days (~8 GB float32)
    will not fit, so keep exception rows only, with one point-in-time snapshot per day. Do not grow the known duplicates
    (4 mover models, 3 pattern stores, 2 break engines, 3 experiment ledgers, 7 firewalls): extend the one the mapping names.

## Prediction-error addition builders (P-series, canon C68)
28. THE SPEC is `PREDICTION_ERROR_ADDITION.md` (owner, verbatim; never edit). It is an ADDITION to C66: read its header and the
    checklists your brief names in full, plus its COMPLETION REQUIREMENT and FINAL PRINCIPLE. Package `engine/research/`.
    EXTEND the existing surprise (engine/learning/surprise.py), calibration, experiment memory, research priority, break research,
    regimes, knowability, winners/losers, observer, exits (engine/exits.py, engine/stops.py) - never a second system. Research
    depth feeds the EXISTING priority/compute_manager (no second scheduler). Non-negotiables: an expectation recorded before a
    prediction is IMMUTABLE (hash-chained; any later rewrite is detected and refused); exits are decided by the learned exit
    policy ONLY - the ±1pp target is evaluation, never an input to any exit or hold decision (prove it with a test that perturbs
    the target and shows identical exits); unknowable outcomes stay unknowable; regime/change detection runs forward in time
    with only data available at each step (a test must show the detector's decision at t is identical when all data after t is
    scrambled). C63: code + unit tests only.
29. Trader import closure (29 Sep, leak 8d reopened by a lazy import): engine/train.py, engine/backtest.py, engine/livesim.py,
    engine/adaptive.py, engine/memory.py, engine/leak_audit.py and everything they import are reachable by the blind trader. Never
    add an import (even inside a function) of patterns, pattern_*, trust*, direction*, lessons, analogs*, curator or research
    modules there; put such code in a separate module the trader never imports (see engine/scramble_audit.py, engine/score_hooks.py).
    Check with `python scripts/leak_audit.py --parts static,assemble` (8d must not be LEAK) and tests/test_leak_audit.py.
