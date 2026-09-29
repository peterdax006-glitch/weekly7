# WEEKLY7 FINAL REPORT (Bible Phase 46)

_Generated 2026-09-28 22:38 from artefacts by scripts/final_report.py. Nothing here is typed by hand._

## 1. Implementation

```json
{
 "engine_modules": [
  {
   "module": "__init__.py",
   "lines": 0,
   "summary": "",
   "commits": 1
  },
  {
   "module": "ablation.py",
   "lines": 229,
   "summary": "Bible Phase 34 (required ablation framework); serves the incremental-out-of-sample-value canon.",
   "commits": 1
  },
  {
   "module": "adaptive.py",
   "lines": 776,
   "summary": "Self-training inside a year (canon C15-C18). One pure, deterministic module used by BOTH the blind live",
   "commits": 7
  },
  {
   "module": "analog_weighting.py",
   "lines": 898,
   "summary": "Analog engine core, feature weighting and ablation (Bible PHASE 8: 8.2, 8.3, 8.4, 8.7, 8.8; serves canon C41).",
   "commits": 1
  },
  {
   "module": "analogs.py",
   "lines": 125,
   "summary": "Rare-analog engine (canon C41): notice when today resembles the only other time something happened.",
   "commits": 2
  },
  {
   "module": "analogs_sector.py",
   "lines": 211,
   "summary": "Sector analog engine (Bible PHASE 8.5; canon C41): when a sector last looked like this, what did it do next?",
   "commits": 1
  },
  {
   "module": "analogs_stock.py",
   "lines": 209,
   "summary": "Stock-level analog engine (Bible PHASE 8.6; canon C41): \"where data suffices\".",
   "commits": 1
  },
  {
   "module": "antimemo.py",
   "lines": 557,
   "summary": "Bible Phase 11 - rerun anti-memorisation experiment (firewall against \"learning the answer\").",
   "commits": 2
  },
  {
   "module": "antioverfit.py",
   "lines": 644,
   "summary": "Bible Phase 26 (anti-overfitting battery); serves the no-look-ahead and no-fake-signal canons.",
   "commits": 1
  },
  {
   "module": "backtest.py",
   "lines": 253,
   "summary": "Daily event-driven backtest of the full pipeline (Blueprint Part H1) plus the baselines.",
   "commits": 3
  },
  {
   "module": "baseline.py",
   "lines": 147,
   "summary": "Bible Phase 0.4: the immutable baseline snapshot, and the comparison of any later run against it.",
   "commits": 1
  },
  {
   "module": "basis_search.py",
   "lines": 210,
   "summary": "Outer training-basis search (Bible PHASE 19; canon C11, C15-C21, C34, C39).",
   "commits": 1
  },
  {
   "module": "blind_gates.py",
   "lines": 643,
   "summary": "Bible Phases 21 and 22 (canons C11, C19, C33): blind-simulator hardening gates and the blind clock.",
   "commits": 1
  },
  {
   "module": "broker.py",
   "lines": 125,
   "summary": "Execution (Blueprint Part F). Two interchangeable brokers:",
   "commits": 2
  },
  {
   "module": "candidates.py",
   "lines": 542,
   "summary": "Bible Phase 3.1 - candidate generation for the pattern miner (canons C35, C37, C43).",
   "commits": 3
  },
  {
   "module": "candles.py",
   "lines": 50,
   "summary": "Multi-timeframe candle features (canon C35): daily, weekly (5 sessions) and monthly (21 sessions) candles and",
   "commits": 1
  },
  {
   "module": "champion.py",
   "lines": 437,
   "summary": "Bible Phase 0.4 (immutable baseline snapshot) and Phase 35 (champion / challenger).",
   "commits": 1
  },
  {
   "module": "checkpoint.py",
   "lines": 159,
   "summary": "Bible Phase 0.3: checkpoint bundle for every major run.",
   "commits": 1
  },
  {
   "module": "config.py",
   "lines": 62,
   "summary": "Central settings. Numbers here are the blueprint's numbers; the learning loop (Part M)",
   "commits": 2
  },
  {
   "module": "data.py",
   "lines": 101,
   "summary": "Daily bar store. Wide parquet frames (date x ticker) per field, split-and-dividend adjusted.",
   "commits": 1
  },
  {
   "module": "data_sources.py",
   "lines": 733,
   "summary": "Data expansion adapters (Bible PHASE 27): sector ETF history, backup price source, delisted-company registry.",
   "commits": 2
  },
  {
   "module": "direction.py",
   "lines": 330,
   "summary": "Direction engine (Bible PHASE 13; canons: direction comes AFTER movement identification, calibrate, abstain).",
   "commits": 1
  },
  {
   "module": "direction_ablate.py",
   "lines": 92,
   "summary": "Direction-input ablation (Bible PHASE 13: \"evidence where justified\"; generic ablation pattern: drop one input",
   "commits": 1
  },
  {
   "module": "direction_calib.py",
   "lines": 186,
   "summary": "Direction calibration alternatives and abstention gates (Bible PHASE 13; canon: \"if the system cannot produce",
   "commits": 1
  },
  {
   "module": "edgar.py",
   "lines": 358,
   "summary": "SEC EDGAR: point-in-time corporate events and insider purchases (all free, 10 req/s limit).",
   "commits": 3
  },
  {
   "module": "exits.py",
   "lines": 669,
   "summary": "Bible Phase 15 (Exit learner), serving the tiered objective of Phase 20 and the \"no look-ahead, fills at next open\" rules.",
   "commits": 1
  },
  {
   "module": "experiment_memory.py",
   "lines": 177,
   "summary": "Bible Phase 30 (experiment memory), the search-space half: a normalised index of WHAT WAS TRIED.",
   "commits": 1
  },
  {
   "module": "explain.py",
   "lines": 45,
   "summary": "Plain-English names for features and market readings, and the top reasons behind a pick. Pure text helpers",
   "commits": 1
  },
  {
   "module": "features.py",
   "lines": 278,
   "summary": "Point-in-time feature panel (Blueprint Part B).",
   "commits": 7
  },
  {
   "module": "fill_audit.py",
   "lines": 56,
   "summary": "Bible Phase 1.4 / canon C33, as a Test-loop gate: prove from a finished Session that every decision was taken at a",
   "commits": 1
  },
  {
   "module": "gaprisk.py",
   "lines": 390,
   "summary": "Bible Phase 16 (Stop / loss engine), gap-risk part: the honest answer to \"losers never worse than -20%\".",
   "commits": 1
  },
  {
   "module": "health.py",
   "lines": 415,
   "summary": "Bible Phase 24 (canons C11, C19): worker health for parallel research.",
   "commits": 1
  },
  {
   "module": "heavy_tests.py",
   "lines": 676,
   "summary": "Bible Phase 7 (heavy algorithm testing), with Phase 34 controls and the Phase 36 provenance block.",
   "commits": 2
  },
  {
   "module": "improve.py",
   "lines": 334,
   "summary": "The self-improvement engine (Blueprint Part M, canon C5).",
   "commits": 4
  },
  {
   "module": "isolation.py",
   "lines": 521,
   "summary": "Live/research separation and live safety audits (Bible PHASE 28; checklist L2-L5).",
   "commits": 2
  },
  {
   "module": "lessons.py",
   "lines": 801,
   "summary": "Bible Phase 10 - lesson memory / learning from mistakes.",
   "commits": 2
  },
  {
   "module": "live.py",
   "lines": 251,
   "summary": "Live jobs (Blueprint Part F). Entry: python -m engine.live <job>",
   "commits": 9
  },
  {
   "module": "livesim.py",
   "lines": 540,
   "summary": "Blind live-clock simulation of a random hidden year (canon C11).",
   "commits": 12
  },
  {
   "module": "memory.py",
   "lines": 636,
   "summary": "Factor-weighted episodic memory for the self-learning system (Bible Phase 9, canon C34).",
   "commits": 3
  },
  {
   "module": "memory_diagnostics.py",
   "lines": 384,
   "summary": "Why did this memory count? Diagnostics for the factor-weighted memory (Bible Phase 9, canon C34).",
   "commits": 2
  },
  {
   "module": "miner_tuning.py",
   "lines": 177,
   "summary": "A7 (canon C34-C37, blueprint: the Algorithm tunes itself): walk-forward self-tuning of the pattern miner's own",
   "commits": 1
  },
  {
   "module": "missed_winners.py",
   "lines": 598,
   "summary": "Missed-winner detector and its evaluation (Bible Phase 14, canon C20).",
   "commits": 2
  },
  {
   "module": "model.py",
   "lines": 123,
   "summary": "Learning layer (Blueprint Part C): evidence composite + LightGBM ranker + triple-barrier",
   "commits": 3
  },
  {
   "module": "objective.py",
   "lines": 260,
   "summary": "Tiered objective firewall (Bible PHASE 20; canon C38 band, C39 lexicographic tiers).",
   "commits": 1
  },
  {
   "module": "options.py",
   "lines": 55,
   "summary": "Live options features for the top candidates (Blueprint B2). Free, 15-minute delayed.",
   "commits": 1
  },
  {
   "module": "parity.py",
   "lines": 496,
   "summary": "Bible Phase 2: feature/parity firewall (canon: no look-ahead; the fast path must reproduce the strict live path).",
   "commits": 1
  },
  {
   "module": "parity_suite.py",
   "lines": 422,
   "summary": "Bible Phase 2 (parity firewall), all feature families. `engine.parity` is the harness; this module points it at",
   "commits": 1
  },
  {
   "module": "pattern_bank.py",
   "lines": 495,
   "summary": "Bible Phase 5 - long-term pattern bank (canon C43; \"no pattern is trusted forever, none deleted for one bad regime\").",
   "commits": 2
  },
  {
   "module": "pattern_identity.py",
   "lines": 588,
   "summary": "Bible Phase 3.2 - pattern identity (canons C35, C37, C43).",
   "commits": 2
  },
  {
   "module": "pattern_lifecycle.py",
   "lines": 582,
   "summary": "Bible Phase 4 - pattern lifecycle state machine (canon C43: never hold a failed pattern).",
   "commits": 2
  },
  {
   "module": "pattern_movers.py",
   "lines": 842,
   "summary": "Bible Phase 6 (pattern -> Find volatility integration) and Phase 34 (required ablation framework).",
   "commits": 2
  },
  {
   "module": "pattern_stats.py",
   "lines": 748,
   "summary": "Bible Phase 3.3 + 3.4 - statistical validation library and relevance weighting (canons C35, C37).",
   "commits": 2
  },
  {
   "module": "patterns.py",
   "lines": 493,
   "summary": "Self-learning pattern miner with relevance-weighted memory (canon C35), on the Phase 3 hardening libraries.",
   "commits": 5
  },
  {
   "module": "pit.py",
   "lines": 1348,
   "summary": "Bible PHASE 1 - point-in-time data firewall (canon: no look-ahead; decide at a close, fill at the NEXT open; a fence",
   "commits": 1
  },
  {
   "module": "planted.py",
   "lines": 299,
   "summary": "Bible Phase 25 (mandatory): planted-pattern calibration. Build synthetic markets whose true patterns are KNOWN,",
   "commits": 1
  },
  {
   "module": "policy.py",
   "lines": 181,
   "summary": "The trading policy, shared by the backtest and live trading so that what is tested is what trades.",
   "commits": 7
  },
  {
   "module": "portfolio.py",
   "lines": 135,
   "summary": "Portfolio layer (Blueprint Part D + E).",
   "commits": 1
  },
  {
   "module": "provenance.py",
   "lines": 126,
   "summary": "Bible Phase 0: provenance for every experiment - git commit, canon hash, Bible hash, blueprint version, config hash,",
   "commits": 2
  },
  {
   "module": "registry.py",
   "lines": 286,
   "summary": "Bible Phase 0.2 + Phase 30: read-side of the experiment registry and the experiment memory.",
   "commits": 1
  },
  {
   "module": "replay.py",
   "lines": 174,
   "summary": "Live-as-if replays of any year 1976-2025 (canon C9, C10).",
   "commits": 2
  },
  {
   "module": "repro.py",
   "lines": 360,
   "summary": "Bible Phase 33 (required reproducibility); serves the deterministic-experiment canon.",
   "commits": 1
  },
  {
   "module": "retester.py",
   "lines": 492,
   "summary": "Bible Phase 23 (canons C11, C19): the re-tester. Replays an archived blind window and demands it reproduces the",
   "commits": 1
  },
  {
   "module": "run_report.py",
   "lines": 448,
   "summary": "Bible Phase 36: the required machine-readable and human-readable report after each major run.",
   "commits": 1
  },
  {
   "module": "scoring.py",
   "lines": 65,
   "summary": "Universe-wide prediction scoring (Blueprint Part H2 level 3, Part M1 'Selection'/'Forecast').",
   "commits": 1
  },
  {
   "module": "shadows.py",
   "lines": 56,
   "summary": "Shadow baseline portfolios (Blueprint Part H2), each with its own $1,000:",
   "commits": 1
  },
  {
   "module": "site_data.py",
   "lines": 129,
   "summary": "Writes site/data.json \u2014 everything the dashboard shows (Blueprint Part J).",
   "commits": 4
  },
  {
   "module": "stops.py",
   "lines": 400,
   "summary": "Bible Phase 16 (Stop / loss engine), under the tiered objective of Phase 20.",
   "commits": 2
  },
  {
   "module": "tick.py",
   "li
```

## 2. Validation (every gate)

```json
{
 "quality_gate": {
  "exit_code": 0,
  "failing": 0
 },
 "planted_calibration": {
  "validated": false,
  "failed": [
   "decaying_not_held",
   "fdr",
   "p_real_top_bin"
  ]
 },
 "parity": {
  "passed": true,
  "failed": []
 },
 "blind_gates": {
  "seals": {
   "n": 40,
   "seal_fails": 0,
   "pairwise_overlap_fails": 0,
   "coverage": {
    "n": 40,
    "per_bin": {
     "1965-1974": 8,
     "1975-1984": 7,
     "1985-1994": 5,
     "1995-2004": 8,
     "2005-2014": 7,
     "2015-2025": 5
    },
    "per_era": {
     "pre1997": 22,
     "2001+": 16,
     "1997-2000": 2
    },
    "empty_bins": [],
    "max_share": 0.2
   },
   "coverage_findings": [],
   "rows": [
    {
     "i": 0,
     "start": "1977-03-01",
     "era": "pre1997",
     "shift_weeks": 9591,
     "fails": 0
    },
    {
     "i": 1,
     "start": "1978-10-01",
     "era": "pre1997",
     "shift_weeks": 9050,
     "fails": 0
    },
    {
     "i": 2,
     "start": "2001-04-01",
     "era": "2001+",
     "shift_weeks": 9252,
     "fails": 0
    },
    {
     "i": 3,
     "start": "1985-09-01",
     "era": "pre1997",
     "shift_weeks": 9272,
     "fails": 0
    },
    {
     "i": 4,
     "start": "1996-12-01",
     "era": "pre1997",
     "shift_weeks": 10890,
     "fails": 0
    },
    {
     "i": 5,
     "start": "1969-11-01",
     "era": "pre1997",
     "shift_weeks": 8995,
     "fails": 0
    },
    {
     "i": 6,
     "start": "2008-06-01",
     "era": "2001+",
     "shift_weeks": 9546,
     "fails": 0
    },
    {
     "i": 7,
     "start": "1968-08-01",
     "era": "pre1997",
     "shift_weeks": 9261,
     "fails": 0
    },
    {
     "i": 8,
     "start": "2017-02-01",
     "era": "2001+",
     "shift_weeks": 8644,
     "fails": 0
    },
    {
     "i": 9,
     "start": "1982-10-01",
     "era": "pre1997",
     "shift_weeks": 8925,
     "fails": 0
    },
    {
     "i": 10,
     "start": "1990-01-01",
     "era": "pre1997",
     "shift_weeks": 8175,
     "fails": 0
    },
    {
     "i": 11,
     "start": "1971-01-01",
     "era": "pre1997",
     "shift_weeks": 9086,
     "fails": 0
    },
    {
     "i": 12,
     "start": "1973-07-01",
     "era": "pre1997",
     "shift_weeks": 9359,
     "fails": 0
    },
    {
     "i": 13,
     "start": "1977-10-01",
     "era": "pre1997",
     "shift_weeks": 9894,
     "fails": 0
    },
    {
     "i": 14,
     "start": "1990-09-01",
     "era": "pre1997",
     "shift_weeks": 9353,
     "fails": 0
    },
    {
     "i": 15,
     "start": "1997-12-01",
     "era": "1997-2000",
     "shift_weeks": 10008,
     "fails": 0
    },
    {
     "i": 16,
     "start": "2021-10-01",
     "era": "2001+",
     "shift_weeks": 8700,
     "fails": 0
    },
    {
     "i": 17,
     "start": "1980-07-01",
     "era": "pre1997",
     "shift_weeks": 9649,
     "fails": 0
    },
    {
     "i": 18,
     "start": "1974-01-01",
     "era": "pre1997",
     "shift_weeks": 10022,
     "fails": 0
    },
    {
     "i": 19,
     "start": "2009-03-01",
     "era": "2001+",
     "shift_weeks": 10858,
     "fails": 0
    },
    {
     "i": 20,
     "start": "2009-09-01",
     "era": "2001+",
     "shift_weeks": 9301,
     "fails": 0
    },
    {
     "i": 21,
     "start": "2014-12-01",
     "era": "2001+",
     "shift_weeks": 9325,
     "fails": 0
    },
    {
     "i": 22,
     "start": "1981-08-01",
     "era": "pre1997",
     "shift_weeks": 9452,
     "fails": 0
    },
    {
     "i": 23,
     "start": "1998-09-01",
     "era": "1997-2000",
     "shift_weeks": 10030,
     "fails": 0
    },
    {
     "i": 24,
     "start": "2003-05-01",
     "era": "2001+",
     "shift_weeks": 9619,
     "fails": 0
    },
    {
     "i": 25,
     "start": "2003-12-01",
     "era": "2001+",
     "shift_weeks": 10325,
     "fails": 0
    },
    {
     "i": 26,
     "start": "2025-06-01",
     "era": "2001+",
     "shift_weeks": 8000,
     "fails": 0
    },
    {
     "i": 27,
     "start": "1984-11-01",
     "era": "pre1997",
     "shift_weeks": 9323,
     "fails": 0
    },
    {
     "i": 28,
     "start": "2005-05-01",
     "era": "2001+",
     "shift_weeks": 10544,
     "fails": 0
    },
    {
     "i": 29,
     "start": "1971-12-01",
     "era": "pre1997",
     "shift_weeks": 8875,
     "fails": 0
    },
    {
     "i": 30,
     "start": "1995-02-01",
     "era": "pre1997",
     "shift_weeks": 9820,
     "fails": 0
    },
    {
     "i": 31,
     "start": "2004-09-01",
     "era": "2001+",
     "shift_weeks": 8925,
     "fails": 0
    },
    {
     "i": 32,
     "start": "2012-07-01",
     "era": "2001+",
     "shift_weeks": 10684,
     "fails": 0
    },
    {
     "i": 33,
     "start": "1986-10-01",
     "era": "pre1997",
     "shift_weeks": 9650,
     "fails": 0
    },
    {
     "i": 34,
     "start": "1969-04-01",
     "era": "pre1997",
     "shift_weeks": 10347,
     "fails": 0
    },
    {
     "i": 35,
     "start": "2018-11-01",
     "era": "2001+",
     "shift_weeks": 10659,
     "fails": 0
    },
    {
     "i": 36,
     "start": "1967-01-01",
     "era": "pre1997",
     "shift_weeks": 9336,
     "fails": 0
    },
    {
     "i": 37,
     "start": "1988-11-01",
     "era": "pre1997",
     "shift_weeks": 9280,
     "fails": 0
    },
    {
     "i": 38,
     "start": "2011-03-01",
     "era": "2001+",
     "shift_weeks": 8982,
     "fails": 0
    },
    {
     "i": 39,
     "start": "2015-08-01",
     "era": "2001+",
     "shift_weeks": 9392,
     "fails": 0
    }
   ]
  },
  "disguise": [
   {
    "start": "1969-04-01",
    "era": "pre1997",
    "n_live": 2,
    "sessions": 1734,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1969-11-01",
    "era": "pre1997",
    "n_live": 2,
    "sessions": 1735,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1971-12-01",
    "era": "pre1997",
    "n_live": 3,
    "sessions": 1740,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1974-01-01",
    "era": "pre1997",
    "n_live": 14,
    "sessions": 1739,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1980-07-01",
    "era": "pre1997",
    "n_live": 33,
    "sessions": 1769,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1981-08-01",
    "era": "pre1997",
    "n_live": 37,
    "sessions": 1768,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1986-10-01",
    "era": "pre1997",
    "n_live": 60,
    "sessions": 1769,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1990-09-01",
    "era": "pre1997",
    "n_live": 73,
    "sessions": 1768,
    "warmup_years": 5.99,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "1997-12-01",
    "era": "1997-2000",
    "n_live": 117,
    "sessions": 1769,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "2009-09-01",
    "era": "2001+",
    "n_live": 53,
    "sessions": 1763,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "2015-08-01",
    "era": "2001+",
    "n_live": 69,
    "sessions": 1761,
    "warmup_years": 5.99,
    "hard_fails": [],
    "duplicate_paths": []
   },
   {
    "start": "2018-11-01",
    "era": "2001+",
    "n_live": 74,
    "sessions": 1762,
    "warmup_years": 6.0,
    "hard_fails": [],
    "duplicate_paths": []
   }
  ],
  "probe": [
   {
    "now": "1981-06-25",
    "honest_passes": true,
    "peeker_caught": true
   },
   {
    "now": "1989-03-20",
    "honest_passes": true,
    "peeker_caught": true
   },
   {
    "now": "1996-12-10",
    "honest_passes": true,
    "peeker_caught": true
   },
   {
    "now": "2004-09-20",
    "honest_passes": true,
    "peeker_caught": true
   },
   {
    "now": "2012-06-22",
    "honest_passes": true,
    "peeker_caught": true
   },
   {
    "now": "2020-04-01",
    "honest_passes": true,
    "peeker_caught": true
   }
  ],
  "retest": {
   "faithful": {
    "status": "PASS",
    "summary": "PASS: 819 items across 7 components",
    "compared": {
     "holdings": 400,
     "trades": 338,
     "scores": 40,
     "weekly_returns": 40,
     "adaptation_events": 0,
     "pattern_activation": 0,
     "memory_state": 1
    },
    "worst": {
     "holdings": [
      0,
      0.0
     ],
     "trades": [
      0,
      0.0
     ],
     "scores": [
      0,
      0.0
     ],
     "weekly_returns": [
      0,
      0.0
     ],
     "adaptation_events": [
      0,
      0.0
     ],
     "pattern_activation": [
      0,
      0.0
     ],
     "memory_state": [
      0,
      0.0
     ]
    },
    "noise_floor": 0.0,
    "first_divergent_week": null,
    "turnover_archive_vs_replay": [
     0.4077,
     0.4077
    ]
   },
   "one_day_leak": {
    "status": "FAIL",
    "summary": "FAIL: 570 items across 7 components; 367 unexplained divergence(s), first: holdings/('2012-03-01', 'CLS') present in only archive",
    "compared": {
     "holdings": 317,
     "trades": 172,
     "scores": 40,
     "weekly_returns": 40,
     "adaptation_events": 0,
     "pattern_activation": 0,
     "memory_state": 1
    },
    "worst": {
     "holdings": [
      166,
      null
     ],
     "trades": [
      163,
      null
     ],
     "scores": [
      0,
      0.0
     ],
     "weekly_returns": [
      38,
      0.81808
     ],
     "adaptation_events": [
      0,
      0.0
     ],
     "pattern_activation": [
      0,
      0.0
     ],
     "memory_state": [
      0,
      0.0
     ]
    },
    "noise_floor": null,
    "first_divergent_week": "w00",
    "turnover_archive_vs_replay": [
     0.4077,
     0.3974
    ]
   },
   "fill_drift_2pct": {
    "status": "FAIL",
    "summary": "FAIL: 819 items across 7 components; 169 unexplained divergence(s), first: trades/2012-03-02/BBGI/buy.price 58.2729 vs 59.4384",
    "compared": {
     "holdings": 400,
     "trades": 338,
     "scores": 40,
     "weekly_returns": 40,
     "adaptation_events": 0,
     "pattern_activation": 0,
     "memory_state": 1
    },
    "worst": {
     "holdings": [
      0,
      0.0
     ],
     "trades": [
      169,
      0.01961
     ],
     "scores": [
      0,
      0.0
     ],
     "weekly_returns": [
      0,
      0.0
     ],
     "adaptation_events": [
      0,
      0.0
     ],
     "pattern_activation": [
      0,
      0.0
     ],
     "memory_state": [
      0,
      0.0
     ]
    },
    "noise_floor": 0.05,
    "first_divergent_week": null,
    "turnover_archive_vs_replay": [
     0.4077,
     0.4077
    ]
   },
   "other_code": {
    "status": "STALE_CODE",
    "summary": "STALE_CODE: 0 items across 0 components (archive code_hash codeA != replay codeB: stale code, not a parity failure)",
    "compared": {},
    "worst": {},
    "noise_floor": null,
    "first_divergent_week": null,
    "turnover_archive_vs_replay": [
     0.4077,
     0.4077
    ]
   }
  },
  "health": {
   "counts": {
    "OK": 8,
    "CRASHED": 1,
    "OOM": 1,
    "TIMEOUT": 1,
    "INVALID": 1,
    "STALE_CODE": 1
   },
   "excluded": [
    {
     "worker": "w08",
     "status": "CRASHED",
     "why": "return code -11",
     "peak_mb": 715.2072018243628
    },
    {
     "worker": "w09",
     "status": "OOM",
     "why": "rss 7900 MB over limit",
     "peak_mb": 570.8722087559754
    },
    {
     "worker": "w10",
     "status": "TIMEOUT",
     "why": "exceeded wall-clock or heartbeat limit",
     "peak_mb": 433.87410424198464
    },
    {
     "worker": "w11",
     "status": "INVALID",
     "why": "1 non-finite numbers in result",
     "peak_mb": 899.7858049750351
    },
    {
     "worker": "w12",
     "status": "STALE_CODE",
     "why": "worker ran code old, disk has cur",
     "peak_mb": 381.8558682596205
    }
   ],
   "window_breakdown": {
    "1969-04-01": {
     "assigned": 1,
     "ok": 1
    },
    "1969-11-01":
```

## 3. Research: what genuinely improved

```json
"NO EVIDENCE"
```

## 4. Failures

```json
[
 {
  "event": "planted_calibration",
  "reason": "Phase 25 NOT VALIDATED: decaying_not_held, fdr, p_real_top_bin"
 },
 {
  "checklist": "- [!] A13 Planted-pattern calibration (engine/planted.py; 64 runs, 8 scenarios): NOT VALIDATED."
 },
 {
  "checklist": "- [!] T17 Planted-pattern calibration: same as A13 (NOT VALIDATED)."
 }
]
```

## 5. Remaining uncertainty

```json
{
 "checklist_open": [
  "- [~] P0.2 Experiment registry with provenance. `log_experiment` stamps code_hash (the modules the run loaded), code_mixed, git commit, canon/Bible hashes, blueprint version, config hash, data snapshot, seed, and all Pha",
  "- [~] P0.3 Checkpoint bundle per major run: engine/checkpoint.py (write/verify; tampered, deleted and added files detected). Not yet called by the major-run scripts.",
  "- [~] P0.4 Immutable baseline snapshot: engine/baseline.py (freeze/verify/diff_vs_baseline). Not yet frozen for the champion.",
  "- [~] A1 Candles and micro-signals. tests/test_candles.py: 12 pass, including point-in-time truncation with a NaN-aware control that must fail on a peeking feature.",
  "- [~] A2 Pattern miner. Redundancy and gain gate now ordered by evidence (ordering by raw effect hid P(real)=1.0 patterns). Bit-reproducible across runs and PYTHONHASHSEED (B11). Hardening library (B14: identity, statist",
  "- [~] A2b Week-clustered statistics, permutation null, P(real), redundancy pruning, validation gate.",
  "- [~] A3 Heavy tests across eras.",
  "- [~] A4 Long-term pattern bank: engine/pattern_bank.py (B03).",
  "- [~] A5 Pattern scores \u2192 Find volatility: engine/pattern_movers.py with random and shuffled controls (B04); grid6 (_ipat) running.",
  "- [~] A6 Minute collector (1.09M 1-minute, 1.87M 5-minute bars; scheduling pending).",
  "- [ ] A7 Self-tuning Algorithm (walk-forward over its own settings).",
  "- [~] A8 Market analog engine. Macro lags corrected by the PIT audit (UNRATE 30, UMCSENT 25, USREC 460 sessions). Heavy test pending.",
  "- [~] A8b/A8c Sector and stock analog engines: engine/analogs_sector.py, analogs_stock.py, analog_weighting.py (B05).",
  "- [~] A9 Timeline dial and yearly pacing: engine/timeline.py (B09). On the real-cache proxy the dial fails its gate; integration via livesim_loop2 in progress.",
  "- [~] A10 Lesson memory: engine/lessons.py (B06).",
  "- [~] A10b Rerun anti-memorisation: engine/antimemo.py (B06).",
  "- [~] A11 Improve-or-discard lifecycle: engine/pattern_lifecycle.py (B03). The planted decaying pattern is still held in 25% of runs (see A13).",
  "- [~] A12 Additional data: FRED done. Sector ETFs, backup prices and DELISTED HISTORY (now priority, because of survivor bias) are with B13.",
  "- [~] V1 95% mover target. Best is `_irf` at 85.6% top-10 (97.4% at the 92% bar on 5.6 picks/week); `_i` is 84.7%. The ab/ad/abad/st variants are within \u00b10.3 pt, so no gain. Survivor-biased.",
  "- [~] V2 Direction at \u226580% calibrated confidence: engine/direction.py (B07). Stacked model; Platt/isotonic calibration on a later block; Brier, log loss, ECE and reliability; the gate abstains unless calibration is prove",
  "- [~] V2b Per-stock-type trust tables: engine/trust.py (B07). Six nested type levels, empirical-Bayes shrinkage to the parent, and an own-data bar.",
  "- [~] V3 Exit learner: engine/exits.py (B08). Real walk-forward (19,304 positions, 2005-2022): no proven edge (CIs straddle 0; the chosen rule flips between folds).",
  "- [~] V4 Stop/risk learner: engine/stops.py (B08).",
  "- [ ] V5 8 of 10 at +10% across blind eras.",
  "- [~] T2 Blind loop. Round 1 on the new rules: +0.64 to +0.77%/week; stopped by a gate. Restart after the B09, B10 and B15 integrations.",
  "- [ ] T3 Tiered 7% objective reached. Currently 19\u201325% of weeks in the band; needs \u226550%. On the real-cache proxy, no 1-25 stock volatility basket reaches 30% in band, and band share vs risk has rank correlation -0.69.",
  "- [~] T4 Weekly self-adjustment (fires 0\u20132 times per window).",
  "- [~] T6 13D correctness (fix running).",
  "- [~] T7 Pattern Explorer: rebuilt as docs/explorer.html from state artefacts (B16, audit clean). Visual check pending.",
  "- [~] T8 Sensitivity page: docs/sensitivity2.html (28 settings, 106 values; 2 pass the |t|\u22653.5 bar). Visual check pending.",
  "- [~] T9 Re-tester parity. The w01c cause is FOUND: memory.py was edited mid-run.",
  "- [~] T10 Future scramble. Passes every window so far. tests/test_session.py includes a peeking-rule control that must fail it; the blind_gates look-ahead probe caught 6/6.",
  "- [~] T11 Time fence. The Session raises on future prices (tested); engine/pit.py adds a Guard and a hash-chained audit log; features.build is future-invariant on real data (5 cuts, 107k rows).",
  "- [~] T12 Worker health: engine/health.py classifies timeouts, OOM, crashes and stale code. Wiring in progress (B10).",
  "- [~] T13 Reproducibility: engine/repro.py. PatternMiner is identical across 2 runs and 2 hash seeds.",
  "- [~] T14/T15/T16 Label permutation, feature shuffle, ticker permutation: engine/antioverfit.py (tests A-J). On real data the reference evaluator's IC is about 0-0.02 per era, so it cannot be separated from noise.",
  "- [~] T18 Fill audit (C33): the engine/fill_audit.py gate in loop2 proves next-open fills per fill; 5 planted defects caught.",
  "- [ ] L1 Upgrade only after a validated research edge (owner decision).",
  "- [~] L2 Paper-only safeguards (paper=True hard-coded; formal tests in B13).",
  "- [~] L3 Trading-hours firewall (broker guard and scheduler; B13). The PIT audit found live decides at 15:42-15:44 ET, before the close, unlike the simulations. Parity gap; decision pending.",
  "- [~] L4 Broker safety audit (B13).",
  "- [~] L5 Research/live isolation test: engine/isolation.py (B13)."
 ],
 "bible_trace": "NO EVIDENCE",
 "standing_caveats": [
  "price panel is survivor-only (Phase 1 audit): historical returns are upper bounds"
 ]
}
```

## 6. Champion configuration

```json
{
 "live_champion": {
  "version": "1.2.1",
  "w_model": 0.5,
  "w_options": 0.15,
  "trained_through": "2026-09-25",
  "champion": "top-4 equal weight, keep while top 20%, weekly, -8% brake, max 2 per sector (v1.2; v1.3 rolled back)",
  "vol_filter": true,
  "min_dv": 20000000.0
 },
 "live_champion_hash": "f71db54964b98721",
 "test_basis": {
  "version": 1,
  "cfg": {
   "k": 2,
   "exit_q": 0.8,
   "rebalance_weeks": 1,
   "brake": null,
   "max_per_sector": null,
   "w_model": 1.0,
   "pick": "hivol",
   "pool_q": 0.7,
   "liq_q": 0.0,
   "vol_filter": false,
   "stress_thr": null,
   "stress_k": 2,
   "trend_filter": null,
   "trend_gross": 0.0
  },
  "meta": {
   "half_life": 6,
   "prior_weeks": 8,
   "switch_z": 2.0,
   "min_weeks": 4,
   "cooldown": 3,
   "revert_drop": 0.03,
   "ic_beta": 0.5,
   "ic_clip": 1.0,
   "adaptive_knobs": [
    "w_model",
    "liq_q",
    "k",
    "pool_q",
    "w_move",
    "w_mom"
   ]
  },
  "hash": "4ca4f7a9c7f8fc37"
 }
}
```

## 7. Challenger configurations

```json
[
 {
  "event": "challenger_historical",
  "id": "sector_cap",
  "t": "2026-09-28T19:31:16"
 },
 {
  "event": "rolled_back",
  "id": "v1.3",
  "t": "2026-09-28T20:29:59"
 }
]
```

## 8. Performance distribution

```json
{
 "windows_revealed": 39,
 "note": "replay skipped"
}
```

## 9. Leakage audit

```json
{
 "verdict": "PASS",
 "checks": {
  "feature_parity": "pass",
  "future_scramble_probe": "pass",
  "features_future_invariant": "pass"
 }
}
```

## 10. Reproducibility audit

```json
{
 "verdict": "PASS",
 "checks": {
  "reference_evaluator_reproducible": "pass",
  "fresh_process_hash_seeds": "pass",
  "pattern_miner_reproducible": "pass",
  "retester_faithful_replay": "pass",
  "retester_catches_planted_divergence": "pass"
 }
}
```

## 11. Checklist

```json
{
 "counts": {
  "not started": 4,
  "failed": 2,
  "validated": 3,
  "implemented/testing": 38
 },
 "file": "state/CHECKLIST.md"
}
```
