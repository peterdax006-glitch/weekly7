# Weekly7 master checklist (the Bible, Phase 37–38; canon C32, C47)

**States:** `[ ]` not started · `[~]` implemented / testing · `[?]` unproven · `[x]` validated (evidence below) · `[!]` failed.
Nothing is marked `[x]` merely because code exists. Work continues until every box is `[x]`, or is `[?]` with the exact technical reason.

## Phase 0: control
- [x] P0.1 Canon integrity. Evidence: `canon/build_canon.py --verify` passes; `engine/provenance.verify_integrity()` checks the canon and the Bible and fails closed.
- [~] P0.2 Experiment registry with provenance (commit, canon/Bible hash, blueprint version, config hash, data snapshot, seed). Built in `log_experiment`; still to do: every writer passes cfg and seed.
- [ ] P0.3 Checkpoint bundle per major run (config, logs, metrics, manifest, seeds, hashes)
- [ ] P0.4 Immutable baseline snapshot

## Algorithm
- [~] A1 Candles and micro-signals (built; unit tests pending)
- [~] A2 Pattern miner (built; planted-pattern calibration pending)
- [~] A2b Week-clustered statistics, permutation null, P(real), redundancy pruning, validation gate
- [~] A3 Heavy tests across eras (2017 v1: IC 0.010 t 0.6; 2020 v1: 0.042 t 2.8; 2022 v2: 0.006 t 1.1; 2020 v2 direction: nothing passed P(real) ≥ 0.8; movement: running)
- [~] A4 Long-term pattern bank (name-keyed, earlier windows only)
- [~] A5 Pattern scores → Find volatility (grid 6 queued; ablations with random and shuffled pattern controls required)
- [~] A6 Minute collector (first pass: 1.09M 1-minute, 1.87M 5-minute bars; scheduling pending)
- [ ] A7 Self-tuning Algorithm (walk-forward over its own settings)
- [~] A8 Market analog engine (built; heavy test pending)
- [ ] A8b Sector analog engine
- [ ] A8c Stock analog engine
- [ ] A9 Timeline dial and yearly pacing
- [ ] A10 Lesson memory
- [ ] A10b Rerun anti-memorisation test
- [~] A11 Improve-or-discard lifecycle (built; unit tests pending)
- [~] A12 Additional data (FRED done; sector ETFs, backup price source and delisted history pending)
- [ ] A13 Planted-pattern calibration

## Find volatility
- [~] V1 95% mover target where enough candidates exist (84.7% average; 95%+ in 5–6 of 13; ~97% at the 92% bar on 5–6 picks a week)
- [ ] V2 Direction at ≥80% calibrated confidence
- [ ] V2b Per-stock-type trust tables
- [ ] V3 Exit learner
- [ ] V4 Stop/risk learner
- [ ] V5 8 of 10 at +10% across blind eras

## Test
- [x] T1 Archive (39 windows rebuilt with the move signal, insider data and opening prices; regen logs)
- [~] T2 Blind loop (round 1 on new rules: +0.64 to +0.77%/week; stopped by a gate)
- [ ] T3 Tiered 7% objective reached (19–25% of weeks in band; needs ≥50%)
- [~] T4 Weekly self-adjustment (fires: 0–2 per window)
- [x] T5 Insider leak protection (parity 0.0 on 4 windows × 8 days with insider data on)
- [~] T6 13D correctness (fix running)
- [~] T7 Pattern Explorer (published; stale)
- [~] T8 Sensitivity page (published; stale)
- [!] T9 Re-tester parity (w01c mismatch: live 43.0% vs replay 62.7%; reproduction test running)
- [~] T10 Future scramble (passes every window so far; automated per round)
- [~] T11 Time fence (Session and Adapter raise on future prices; feed-level fence pending)
- [~] T12 Worker health (crashed workers excluded and reported; timeouts and out-of-memory reports pending)
- [ ] T13 Reproducibility (run twice, compare)
- [ ] T14 Label permutation
- [ ] T15 Feature shuffle
- [ ] T16 Ticker permutation
- [ ] T17 Planted-pattern calibration (same as A13)

## Live
- [ ] L1 Upgrade only after a validated research edge (owner decision)
- [~] L2 Paper-only safeguards (paper=True hard-coded; formal test pending)
- [~] L3 Trading-hours firewall (broker guard + scheduler; formal test pending)
- [ ] L4 Broker safety audit
- [ ] L5 Research/live isolation test
