# Weekly7 master checklist (the Bible, Phase 37–38; canon C32, C47)

**States:** `[ ]` not started · `[~]` implemented / testing · `[?]` unproven · `[x]` validated (evidence below) · `[!]` failed.
Nothing is marked `[x]` merely because code exists. Work continues until every box is `[x]`, or is `[?]` with the exact technical reason.
Foundation status per Bible phase (line ranges, tests): `state/build/FOUNDATION.md`. Integration hooks: `state/build/INTEGRATION.md`.

> **Standing caveat (2026-09-28):** the price panel is SURVIVOR-ONLY (5,243 tickers, 2 end early; Phase 1 audit). Every
> historical return below is an upper bound until delisted history is integrated (Phase 27).

## Phase 0: control
- [x] P0.1 Canon integrity. Evidence: `canon/build_canon.py --verify` (52 directives); `engine/provenance.verify_integrity()` checks the canon and the Bible and fails closed (tests/test_provenance.py: a tampered Bible raises).
- [~] P0.2 Experiment registry with provenance. `log_experiment` stamps code_hash (the modules the run loaded), code_mixed, git commit, canon/Bible hashes, blueprint version, config hash, data snapshot, seed, and all Phase 0.2 fields (missing ones warned); sha256 ids. Registry audit of the historic 53 records: 0 complete, 2 with provenance (append-only; fixed going forward). Open: older writers pass no cfg/seed.
- [~] P0.3 Checkpoint bundle per major run: engine/checkpoint.py (write/verify; tampered, deleted and added files detected). Not yet called by the major-run scripts.
- [~] P0.4 Immutable baseline snapshot: engine/baseline.py (freeze/verify/diff_vs_baseline). Not yet frozen for the champion.

## Algorithm
- [~] A1 Candles and micro-signals. tests/test_candles.py: 12 pass, including point-in-time truncation with a NaN-aware control that must fail on a peeking feature.
- [~] A2 Pattern miner. Redundancy and gain gate now ordered by evidence (ordering by raw effect hid P(real)=1.0 patterns). Bit-reproducible across runs and PYTHONHASHSEED (B11). Hardening library (B14: identity, statistics, candidates).
- [~] A2b Week-clustered statistics, permutation null, P(real), redundancy pruning, validation gate.
- [~] A3 Heavy tests across eras.
  - Movement patterns (test 2020+): IC 0.383, t 38.5; 56 active of 4,382; null t95 1.93 vs real 42.8.
  - Direction: 2017 v1 IC 0.010; 2020 v1 0.042; 2022 v2 0.006; 2020 v2 nothing passed.
  - Harness: engine/heavy_tests.py (B04).
- [~] A4 Long-term pattern bank: engine/pattern_bank.py (B03).
- [~] A5 Pattern scores → Find volatility: engine/pattern_movers.py with random and shuffled controls (B04); grid6 (_ipat) running.
- [~] A6 Minute collector (1.09M 1-minute, 1.87M 5-minute bars; scheduling pending).
- [ ] A7 Self-tuning Algorithm (walk-forward over its own settings).
- [~] A8 Market analog engine. Macro lags corrected by the PIT audit (UNRATE 30, UMCSENT 25, USREC 460 sessions). Heavy test pending.
- [~] A8b/A8c Sector and stock analog engines: engine/analogs_sector.py, analogs_stock.py, analog_weighting.py (B05).
- [~] A9 Timeline dial and yearly pacing: engine/timeline.py (B09). On the real-cache proxy the dial fails its gate; integration via livesim_loop2 in progress.
- [~] A10 Lesson memory: engine/lessons.py (B06).
- [~] A10b Rerun anti-memorisation: engine/antimemo.py (B06).
- [~] A11 Improve-or-discard lifecycle: engine/pattern_lifecycle.py (B03). The planted decaying pattern is still held in 25% of runs (see A13).
- [~] A12 Additional data: FRED done. Sector ETFs, backup prices and DELISTED HISTORY (now priority, because of survivor bias) are with B13.
- [!] A13 Planted-pattern calibration (engine/planted.py; 64 runs, 8 scenarios): NOT VALIDATED.
  - Pass: strong/weak/negative/pair detection 100%; hallucinated 0% admitted; zero 0%; noise-only 0.4 active/run; P(real)>0.9 bin 86% truly real.
  - FAIL: FDR 15.9% (limit 10%); decaying pattern held 25% (limit 20%).
  - Also: effect sizes about 25% of truth; regime pattern found 50% and never rescoped; unless pattern untested in 6/8 runs.
  - Evidence: state/research/algorithm/planted/report.md

## Find volatility
- [~] V1 95% mover target. Best is `_irf` at 85.6% top-10 (97.4% at the 92% bar on 5.6 picks/week); `_i` is 84.7%. The ab/ad/abad/st variants are within ±0.3 pt, so no gain. Survivor-biased.
- [~] V2 Direction at ≥80% calibrated confidence: engine/direction.py (B07). Stacked model; Platt/isotonic calibration on a later block; Brier, log loss, ECE and reliability; the gate abstains unless calibration is proven out of sample. Real-data study in progress.
- [~] V2b Per-stock-type trust tables: engine/trust.py (B07). Six nested type levels, empirical-Bayes shrinkage to the parent, and an own-data bar.
- [~] V3 Exit learner: engine/exits.py (B08). Real walk-forward (19,304 positions, 2005-2022): no proven edge (CIs straddle 0; the chosen rule flips between folds).
- [~] V4 Stop/risk learner: engine/stops.py (B08).
  - Stops cut P(loss>20%) from 3.3% to 1.2%, but the worst loss is still -73% because of overnight gaps.
  - The gap model is miscalibrated: 11-16% of nights beyond its 95% tail, against a 5% target.
  - A conditional gap model is being built.
- [ ] V5 8 of 10 at +10% across blind eras.

## Test
- [x] T1 Archive (39 windows rebuilt with the move signal, insider data and opening prices; regen logs).
- [~] T2 Blind loop. Round 1 on the new rules: +0.64 to +0.77%/week; stopped by a gate. Restart after the B09, B10 and B15 integrations.
- [ ] T3 Tiered 7% objective reached. Currently 19–25% of weeks in the band; needs ≥50%. On the real-cache proxy, no 1-25 stock volatility basket reaches 30% in band, and band share vs risk has rank correlation -0.69.
- [~] T4 Weekly self-adjustment (fires 0–2 times per window).
- [x] T5 Insider leak protection (parity 0.0 on 4 windows × 8 days with insider data on).
- [~] T6 13D correctness (fix running).
- [~] T7 Pattern Explorer: rebuilt as docs/explorer.html from state artefacts (B16, audit clean). Visual check pending.
- [~] T8 Sensitivity page: docs/sensitivity2.html (28 settings, 106 values; 2 pass the |t|≥3.5 bar). Visual check pending.
- [~] T9 Re-tester parity. The w01c cause is FOUND: memory.py was edited mid-run.
  - Provenance now records the modules a run loaded and any mid-run edits; the loop reruns stale windows.
  - engine/retester.py (B10): 0.5% relative tolerance, code hash compared first.
  - Needs one clean round.
- [~] T10 Future scramble. Passes every window so far. tests/test_session.py includes a peeking-rule control that must fail it; the blind_gates look-ahead probe caught 6/6.
- [~] T11 Time fence. The Session raises on future prices (tested); engine/pit.py adds a Guard and a hash-chained audit log; features.build is future-invariant on real data (5 cuts, 107k rows).
- [~] T12 Worker health: engine/health.py classifies timeouts, OOM, crashes and stale code. Wiring in progress (B10).
- [~] T13 Reproducibility: engine/repro.py. PatternMiner is identical across 2 runs and 2 hash seeds.
- [~] T14/T15/T16 Label permutation, feature shuffle, ticker permutation: engine/antioverfit.py (tests A-J). On real data the reference evaluator's IC is about 0-0.02 per era, so it cannot be separated from noise.
- [!] T17 Planted-pattern calibration: same as A13 (NOT VALIDATED).
- [~] T18 Fill audit (C33): the engine/fill_audit.py gate in loop2 proves next-open fills per fill; 5 planted defects caught.

## Live
- [ ] L1 Upgrade only after a validated research edge (owner decision).
- [~] L2 Paper-only safeguards (paper=True hard-coded; formal tests in B13).
- [~] L3 Trading-hours firewall (broker guard and scheduler; B13). The PIT audit found live decides at 15:42-15:44 ET, before the close, unlike the simulations. Parity gap; decision pending.
- [~] L4 Broker safety audit (B13).
- [~] L5 Research/live isolation test: engine/isolation.py (B13).
