# Weekly7 master checklist (the Bible, Phase 37–38; canon C32, C47)

**States:** `[ ]` not started · `[~]` implemented / testing · `[?]` unproven · `[x]` validated (evidence below) · `[!]` failed.
Nothing is marked `[x]` merely because code exists. Work continues until every box is `[x]`, or is `[?]` with the exact technical reason.
Foundation status per Bible phase (line ranges, tests): `state/build/FOUNDATION.md`. Integration hooks: `state/build/INTEGRATION.md`.

> **Standing caveat (2026-09-28):** the price panel is SURVIVOR-ONLY (5,243 tickers, 2 end early; Phase 1 audit). Every
> historical return below is an upper bound until delisted history is integrated (Phase 27).

## Phase 0: control
- [x] P0.1 Canon integrity. Evidence: `canon/build_canon.py --verify` (56 directives, C1-C56); `engine/provenance.verify_integrity()` checks the canon and the Bible and fails closed (tests/test_provenance.py).
- [~] P0.2 Experiment registry with provenance. `log_experiment` stamps code_hash (the modules the run loaded), code_mixed, git commit, canon/Bible hashes, blueprint version, config hash, data snapshot, seed, and all Phase 0.2 fields (missing ones warned); sha256 ids. Registry audit of the historic 53 records: 0 complete, 2 with provenance (append-only; fixed going forward). Open: older writers pass no cfg/seed.
- [~] P0.3 Checkpoint bundle per major run: engine/checkpoint.py (write/verify; tampered, deleted and added files detected). Not yet called by the major-run scripts.
- [x] P0.4 Immutable baseline snapshot. Frozen 2026-09-28 23:25: state/baseline/B_2026-09-28_5a62d137 (Live champion + Test basis v1, 19 registry metrics, whole-engine code hash, data snapshot); verify() ok, no drift. Unmeasured at freeze (no registry source): mover/pattern/analog/missed-winner/direction metrics.

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
- [~] A6 Minute collector: scheduled after every weekday close (.github/workflows/intraday.yml -> monthly intraday-YYYY-MM releases; scripts/sync_intraday.py; 6 tests). First scheduled runs not yet verified.
- [~] A7 Self-tuning Algorithm: engine/miner_tuning.py (walk-forward coordinate search, fold-share + week-bootstrap rule, untouched holdout; 7 tests). Real run (movement, 600 tickers, 2008+): no change adopted, holdout IC 0.352; fdr_q found inert.
- [~] A8 Market analog engine. Macro lags corrected by the PIT audit (UNRATE 30, UMCSENT 25, USREC 460 sessions). Heavy test pending.
- [~] A8b/A8c Sector and stock analog engines: engine/analogs_sector.py, analogs_stock.py, analog_weighting.py (B05).
- [~] A9 Timeline dial and yearly pacing: engine/timeline.py (B09). On the real-cache proxy the dial fails its gate; integration via livesim_loop2 in progress.
- [~] A10 Lesson memory: engine/lessons.py (B06).
- [~] A10b Rerun anti-memorisation: engine/antimemo.py (B06).
- [~] A11 Improve-or-discard lifecycle: engine/pattern_lifecycle.py (B03). The planted decaying pattern is still held in 25% of runs (see A13).
- [~] A12 Additional data: FRED done. Sector ETFs, backup prices and DELISTED HISTORY (now priority, because of survivor bias) are with B13.
- [x] A13 Planted-pattern calibration: VALIDATED on BOTH seed sets with the final code (2026-09-29) - original seeds 100+ (FDR 6.7%, decaying 0% held, P(real)>0.9 bin 97% real) and fresh hold-out seeds 300+ (FDR 9.9%, decaying 0%, 96%); all 10 criteria pass. Evidence: state/research/algorithm/planted/report_final_seeds100.md, report.md (hold-out), data/planted_cal9_holdout.log, data/planted_cal10_origseeds.log. Fixes: exact-truth discovery scoring, decay death rule, Bonferroni-corrected rescue scored at in-scope size. Still open (separate, stricter): tests/test_patterns_integration.py strict xfail - small fast-setting noise panels admit ~0.67 false patterns/run. History:
  - Pass: strong/weak/negative/pair detection 100%; hallucinated 0% admitted; zero 0%; noise-only 0.4 active/run; P(real)>0.9 bin 86% truly real.
  - FAIL: FDR 15.9% (limit 10%); decaying pattern held 25% (limit 20%).
  - Rerun 2026-09-29 on the hardened miner (BH P(real), full-search null, empirical-Bayes effects, content-ordered rows): effect sizes now 0.80-1.17 of truth (was 0.25); regime 62% (was 50%); unless untested 4/8 (was 6/8); noise-only 0.25 active/run. Still FAIL: FDR 18.2%, decaying held 25%.
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
- [~] V5 8 of 10 at +10% across blind eras: engine/fv_pipeline.py (B19; 23 tests: reaches 8/10 when direction is knowable, abstains on noise). Real run (scripts/fv_eval.py run1) in progress.

## Test
- [x] T1 Archive (39 windows rebuilt with the move signal, insider data and opening prices; regen logs).
- [~] T2 Blind loop: running with all gates (round 2 done 2026-09-28: w02a +1.85%/wk, w02b +0.33%, w02c -0.80%; all gates OK). Continues round by round.
- [ ] T3 Tiered 7% objective reached. Currently 19–25% of weeks in the band; needs ≥50%. On the real-cache proxy, no 1-25 stock volatility basket reaches 30% in band, and band share vs risk has rank correlation -0.69.
- [~] T4 Weekly self-adjustment (fires 0–2 times per window).
- [x] T5 Insider leak protection (parity 0.0 on 4 windows × 8 days with insider data on).
- [~] T6 13D correctness (fix running).
- [~] T7 Pattern Explorer: rebuilt as docs/explorer.html from state artefacts (B16, audit clean). Visual check pending.
- [~] T8 Sensitivity page: docs/sensitivity2.html (28 settings, 106 values; 2 pass the |t|≥3.5 bar). Visual check pending.
- [x] T9 Re-tester parity. Round 2 (2026-09-28): re-tester OK on 3/3 windows under the provenance/stale-code gate; engine/retester.py (0.5% relative, code hash first). Evidence: data/loop2_run3.log.
  - Provenance now records the modules a run loaded and any mid-run edits; the loop reruns stale windows.
  - engine/retester.py (B10): 0.5% relative tolerance, code hash compared first.
  - Needs one clean round.
- [~] T10 Future scramble. Passes every window so far. tests/test_session.py includes a peeking-rule control that must fail it; the blind_gates look-ahead probe caught 6/6.
- [~] T11 Time fence. The Session raises on future prices (tested); engine/pit.py adds a Guard and a hash-chained audit log; features.build is future-invariant on real data (5 cuts, 107k rows).
- [x] T12 Worker health: engine/health.py wired into loop2 (supervise, heartbeat, exclusion report); on real data it correctly EXCLUDED 3 windows whose results held NaN (cause fixed, 8ef3835). Evidence: data/loop2.log, state/livesim/w02*/health2.jsonl.
- [~] T13 Reproducibility: engine/repro.py. PatternMiner is identical across 2 runs and 2 hash seeds.
- [~] T14/T15/T16 Label permutation, feature shuffle, ticker permutation: engine/antioverfit.py (tests A-J). On real data the reference evaluator's IC is about 0-0.02 per era, so it cannot be separated from noise.
- [x] T17 Planted-pattern calibration: same as A13 (VALIDATED on original and hold-out seeds).
- [x] T18 Fill audit (C33): round 2 proved all 490 fills at the next open (164+142+184) via engine/fill_audit.gate -> pit.audit_fills; 5 planted defects caught in tests.

- [ ] T19 Learning delta (canon C54/C55): the change in results when the same year is rerun after learning, DISGUISED so the system cannot know it is the same year; with no-learning, transfer and memoriser controls. B22 building engine/learning_delta.py. Current evidence (B06): memorisation 0, lessons from other windows -0.19%/wk.
- [ ] T20 Future-leak audit (canon C56): the system only ever sees what was available live at that moment - survivorship, split-adjusted prices, today's metadata, learned state from later years, macro revisions, year fingerprints, network. B23 auditing every channel.

## Live
- [ ] L1 Upgrade only after a validated research edge (owner decision).
- [~] L2 Paper-only safeguards (paper=True hard-coded; formal tests in B13).
- [~] L3 Trading-hours firewall: FIXED 2026-09-29 - broker.regular_hours now uses the strict NYSE calendar (holidays + half days; it had allowed 1,090 closed slots); tests/test_broker_hours.py. Open: live decides at 15:42 ET vs sims at the close (owner decision).
- [~] L4 Broker safety audit (B13).
- [~] L5 Research/live isolation test: engine/isolation.py (B13).
