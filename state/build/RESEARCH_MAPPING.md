# C66 mapping: existing WEEKLY7 code vs RESEARCH_BRAIN_CONTRACT.md (29 Sep 2026, read-only)

Scope: this audit maps existing code against the owner's C66 build prompt (`RESEARCH_BRAIN_CONTRACT.md`, contract sha 787fd48c...). It is written as input to the build waves. Machine-readable results are in `state/build/RESEARCH_BRAIN_CHECKLIST.json`. Only the fields status, code_paths and notes on items were edited, plus actual_lines, code_paths, notes and status on the 37 line_budget rows. The JSON was parsed before it was saved.

Method. Every module in engine/learning (64 modules), engine/*.py and the relevant scripts/ was read at docstring and definition level. The key hosts were read further. Lines were counted with the single ruler `scripts/contract_lines.py`. Where only part of a module does a C66 job, the same ruler was restricted to the named top-level definitions. The partial modules are research_priority, research_policy, fv_pipeline, direction_features, direction, pattern_movers, break_detection, missed_winners, situation, scorecard, promotion and learner. The helper script is in the session scratchpad.

A module is credited to one row only. Code that is only a building block is named in the notes but not credited. INTEGRATION_AUDIT.md findings were reused, not re-derived. No tests were run, no files were edited apart from the two named above, and state/livesim/ was not read.

## Headline

- **44,453 meaningful lines of existing code genuinely do C66 jobs, against a section-45 minimum of 61,600** (the table's upper bound is 99,400). The prompt's own "~58,000–90,000" text is lower than its table.
- 26 of the 37 rows are below their minimum. The combined shortfall is **28,040 lines**, and that assumes every credited line counts in full.
- 11 rows reach their minimum on the ruler, but none is functionally covered. Six of the 37 rows are **duplicated-risk**: parallel implementations already exist, and a C66 builder is likely to add another.
- No row is `covered`. The row classes are: partial 24, missing 7, duplicated-risk 6.
- Item status (114 items): NOT_STARTED 16, IN_PROGRESS 88, IMPLEMENTED 0, VALIDATED 10.
  - All 10 VALIDATED items are SCIENTIFIC VALIDATION *measurements* with artefacts on disk: RS01, RS05-RS12 and RS14.
  - Every one of those artefacts was made from the **survivor-only** price panel (RG04 is currently violated).
  - All 10 must be re-measured by the C66 labs.
- The research engine that exists is real but runs only inside the blind learner. It runs as one stage (`learner.stage_research`), as a shadow, on planted worlds. `state/learning/curator/` has never been created.
  - The loop never closes: `propose_selected` and `update_from_result` have 0 production callers.
  - C66's central requirement is a persistent, market-wide, self-allocating research loop. That loop does not exist.

## Per-component table (section 45 order)

Credited = meaningful lines, by the ruler, that genuinely do the C66 job. Gap = minimum minus credited, with a floor of zero.

| # | component | min | credited | gap | class | existing code it must build on |
|---|---|---:|---:|---:|---|---|
| 1 | Research priority engine | 1500 | 884 | 616 | partial | research_priority (ranking), research_policy (EIG/EVSI/PriorityFunction/EigCalibrator) |
| 2 | Autonomous research loop | 2000 | 980 | 1020 | partial | research_priority.ResearchQueue, checkpoints, learner.stage_research/stage_meta |
| 3 | Market-wide observer | 1500 | 0 | 1500 | missing | blocks: fv_pipeline.Panel, missed_winners.week_from_base |
| 4 | Winner/loser research | 2500 | 3888 | 0 | partial | failure, postmortem, separation, credit (own trades only) |
| 5 | Missed opportunity engine | 1800 | 1443 | 357 | partial | engine/missed_winners, learning/missed_winners (winners only) |
| 6 | Knowability/external-factor | 2500 | 0 | 2500 | missing | blocks: pit (availability/lags), edgar (accepted ts), future_firewall, unknowns |
| 7 | Counterfactual reconstruction | 2000 | 683 | 1317 | partial | parity, parity_suite (strict PIT recomputation) |
| 8 | Volatility laboratory | 3000 | 1188 | 1812 | duplicated-risk | fv_pipeline.MoverStage, pattern_movers, direction_features.mover_walk_forward, scripts/movers.py |
| 9 | Direction laboratory | 3000 | 1106 | 1894 | partial | direction, direction_calib, direction_ablate, direction_features, fv DirectionStage, scripts/direction_research, direction_study |
| 10 | Conditional accuracy frontier | 1500 | 155 | 1345 | partial | direction_features.frontier/eighty_question, direction.wilson/ece |
| 11 | Winner/loser symmetry | 2000 | 0 | 2000 | missing | blocks: pattern_stats, failure, boundary |
| 12 | Pattern discovery | 4000 | 4632 | 0 | duplicated-risk | patterns, candidates, pattern_stats, pattern_identity, pattern_bank, pattern_lifecycle, pattern_memory(+eval), knowledge, planted |
| 13 | Pattern-break research | 2000 | 3558 | 0 | duplicated-risk | learning/break_detection, engine/pattern_reliability, boundary |
| 14 | Experiment memory | 1500 | 1373 | 127 | duplicated-risk | learning/experiment_memory, engine/experiment_memory, registry |
| 15 | Meta-learning | 2500 | 1213 | 1287 | partial | meta_learning |
| 16 | Failed-learner lab | 1200 | 509 | 691 | partial | failed_learners (no production caller) |
| 17 | Compute manager | 1500 | 2031 | 0 | partial | compute, resources, engine/health, research_policy compute defs |
| 18 | Experiment value accounting | 1200 | 129 | 1071 | missing | research_policy.RealisedGain/RoundLedger, scorecard.ComputeMeter |
| 19 | Compute-waste controller | 800 | 64 | 736 | partial | research_priority.barren_streak, research_policy.marginal_return_verdict; reuse retirement.py for DORMANT |
| 20 | Daily market autopsy | 1500 | 0 | 1500 | missing | blocks: reports.Builder, surprise |
| 21 | Daily research target generator | 1500 | 898 | 602 | partial | research_priority.signals_from_*, surprise, contradiction_monitor |
| 22 | Multi-scale learning | 1200 | 45 | 1155 | missing | candles, situation.horizon_state |
| 23 | Cross-sectional learning | 1500 | 453 | 1047 | partial | analogs_sector, trust, situation.CrossSection |
| 24 | Regime-aware research | 1500 | 2198 | 0 | partial | context, hierarchy, situation.classify_regime/PopulationDriftMonitor |
| 25 | Interaction discovery | 2000 | 592 | 1408 | duplicated-risk | complexity + 4 scattered interaction searches |
| 26 | Knowledge graph | 1500 | 1008 | 492 | partial | knowledge_graph |
| 27 | Knowledge-to-decision bridge | 1000 | 1188 | 0 | partial | decision_contract, learning/champion (hooks 0 callers) |
| 28 | Research/trader firewall | 2000 | 5544 | 0 | duplicated-risk | trader_view, curator, test_path, firewalls, future/memory/identity_firewall |
| 29 | Replication system | 1000 | 1568 | 0 | partial | transfer, transfer_score, promotion rerun gate |
| 30 | Unknown-cause system | 1000 | 409 | 591 | partial | unknowns |
| 31 | Self-improvement scorecard | 1000 | 1205 | 0 | partial | scorecard, learning_curve |
| 32 | Research-brain health | 1200 | 92 | 1108 | missing | research_policy.policy_health/audit_spend/allocation_entropy |
| 33 | Research diversity controller | 1000 | 340 | 660 | partial | research_policy.TargetAllocator family |
| 34 | Scientific long-term memory | 1500 | 3009 | 0 | partial | archive, interpretation, belief, epistemic |
| 35 | Question generator | 1200 | 294 | 906 | partial | research_priority.QuestionGenerator/CandidateBuilder |
| 36 | Hypothesis tree | 1000 | 702 | 298 | partial | competition |
| 37 | Quality/promotion gate | 1000 | 1072 | 0 | partial | learning/promotion, engine/champion |
| | **total** | **61,600** | **44,453** | **28,040** | | |

The exact definitions counted for each row are listed in each line_budget row's `notes`.

Rows at or above their minimum are still `partial` for two reasons. First, the credited code does the C62 version of the job: own trades rather than the market, patterns rather than research branches, trader-side knowledge rather than research discoveries. Second, the hooks that would make it do the C66 job have no callers. Closing those gaps is integration work, not new line count, and it must not be padded.

## What C62 code already provides and must be EXTENDED, not duplicated

| C66 need | existing code to extend | the extension |
|---|---|---|
| Research priority formula (s.2) | `research_policy.PriorityFunction`, `EigCalibrator`, `evsi_normal`, `joint_information`, `registry_overfit_risk` | Add the terms `expected_loss_reduction`, `volatility_value` and `direction_value`. Validate the priority model itself out of sample. Close `propose_selected -> update_from_result`. |
| Research queue and checkpoints (s.3) | `research_priority.ResearchQueue` (snapshot/restore/replay), `checkpoints` (resume_verified, write_interruption) | Lift the queue out of `learner.stage_research` into a standalone loop. Wire resume_verified. |
| Loss research (s.5, s.34) | `failure.LossClassifier` (8 detectors), `postmortem`, `separation`, `credit` | Feed them market-wide losers and winners, not only held trades. Use their causes for loss-risk bank entries. |
| Missed winners (s.6) | `engine/missed_winners` (detector, ledger), `learning/missed_winners.RejectionAnalyzer` | Mirror the code for missed LOSERS. Rank by the s.6 list. Fix the dead `MissedLedger.observe` hook (adaptive calls `add`). |
| PIT reconstruction (s.8) | `parity`, `parity_suite` (strict truncated recomputation per family), `pit` availability dates | Reuse the strict path to snapshot the full information state at T once per day, shared by all events that day. |
| Volatility models (s.9) | `fv_pipeline.MoverStage` (LightGBM ranks, Platt, certified threshold) | Make it the single champion. The other three mover implementations become challengers or are retired. |
| Direction on predicted movers (s.10-11) | `direction_features` (walk-forward mover picks, frontier, `eighty_question`, planted/leak/shuffled controls), `direction`, `direction_calib` | Wrap them in lab objects. Add per-coverage independence, sector and regime stability. |
| Pattern discovery (s.13) | `patterns.PatternMiner`, `candidates`, `pattern_stats` (BH, cluster permutations), `pattern_identity`, `knowledge.KnowledgeObject` | Build the missing miner -> KnowledgeObject bridge (`evidence_from_pattern_row` has 0 callers). Add filings, insider, breadth, dispersion and liquidity candidates. Choose ONE store. |
| Pattern breaks (s.14) | `break_detection.explain_break`, `engine/pattern_reliability` (C60 gating) | Merge them behind one interface. Schedule investigations through `signals_from_health`, which currently has 0 callers. |
| Experiment memory (s.15) | `learning/experiment_memory.ExperimentLedger` (already_tested wired via improve.py) | Add data/config hash, controls, compute cost, research value and decision impact. Make it the only front for `registry` and `engine/experiment_memory`. |
| Meta-learning (s.16) | `meta_learning.MetaLearner` (walk-forward, null control) | Add validation-method yield, dataset false-discovery rates and question -> decision-change learning. |
| Failed learners (s.17) | `failed_learners` (six real failures seeded) | Wire `check_proposal` before any learner search. Surface per-class failure counts. |
| Compute (s.18) | `compute` (deterministic workers, RAM admission, reconcile), `resources`, `research_policy.successive_halving`/`sprt_bernoulli` | Add the research-state machine QUEUED..RETIRED and the five-stage escalation ladder. |
| Diversity (s.38) | `research_policy.TargetAllocator`/`DiscountedAllocator` | Add targets VOLATILITY, DIRECTION, RISK_LOSS and PATTERN_BREAK to `ResearchTarget`. |
| Knowledge graph and bridge (s.27-28) | `knowledge_graph`, `decision_contract`, `learning/champion` | Add node types for regime, event, learner and question. Wire `policy_check`/`readiness`. Record "informational only" value. |
| Firewall (s.29-31) | `trader_view`, `curator`, `test_path`, `firewalls.LearningFirewallGate` (8 layers), `future/memory/identity_firewall`, `leak_audit` | Add one new gate LAYER and the research namespace, not a ninth parallel firewall. |
| Replication and gate (s.32, s.42) | `transfer` (6 axes), `promotion` (10 gates), `complexity` | Add gates for replication (as distinct from reproducibility), calibration, complexity and failure behaviour. Add the outcomes QUARANTINED, NEEDS_MORE_EVIDENCE and UNKNOWN. |
| Unknown cause (s.33) | `unknowns` (non-numeric UNKNOWN states), `pattern_reliability` UNKNOWN_CAUSE verdict | Use one vocabulary. Track the unknown rate by regime, volatility, sector and confidence. Add a forced-label detector. |
| Scientific memory (s.39) | `archive` (10 hash-chained layers), `belief` (evidence chains), `interpretation`, `epistemic` | Add the query API "why do you believe X / what would stop you / what replaced it". |
| Scorecard (s.36) | `scorecard.LearningScorecard` + claim gate, `learning_curve` | Add the C66 fields for volatility, direction-at-coverage, avoidable loss and research efficiency. |

## Genuinely new systems (nothing to extend)

1. The market-wide daily observer: the universe record in categories A-I, including near misses, false positives, false negatives and abstentions.
2. The daily market autopsy (s.21).
3. The knowability engine: the PREDICTABLE...DATA_FAILURE classes, and the known-before, known-after, simultaneous, uncertain and unavailable labels.
4. Counterfactual "could I have known" reports beyond features: patterns, memory, macro and events at T versus what was learned later.
5. The volatility hypothesis programme: H1-H9 kept as live competitors, able to discover H10 and later hypotheses.
6. Winner/loser symmetry, and a loss-risk knowledge bank separate from the opportunity bank.
7. Missed-LOSER research.
8. Experiment value accounting: decision, risk, prediction, transfer and redundancy change per job.
9. Research-brain health: duplicate rate, the researcher's false discovery rate, easy-question bias, abandonment of hard questions.
10. A hypothesis tree with branch kill and redirect.
11. Multi-scale horizon research and cross-horizon transfer.
12. Stock-versus-cohort cross-sectional decomposition.
13. The research state machine and escalation ladder.
14. The `MATURED_RESEARCH_STATE` namespace and the research-side planted-leak suite.

## Key design risks

**1. Research/trader firewall versus the C64 curator and trader_view.**
Today the only research engine (`research_priority`, `meta_learning`, `research_policy`) is imported by `learner.py`, and runs *inside the blind trader*. C66 wants a research side that studies matured outcomes. For knowability it also wants hindsight: information known only after the event, and real-date causes. That work can only live on the trusted side next to the curator, which knows the real year.

- **(a) Research artefacts carry forbidden values.** They will hold real dates, years and ticker-like strings. Anything they release to the trader must go through `curator` -> `TraderRelease` -> `trader_view.assert_trader_safe`.
- **(b) The same-year rerun leak (C55/C64).** Research filed under real year Y must not be released while the trader replays a disguised Y. Maturity must be checked in REAL time against the window's real `now`, never in disguised time.
- **(c) Hindsight labels leaking into training.** Hindsight labels include "unpredictable", "external cause" and knowability classes. They must never become trader training labels, except through an explicit point-in-time training gate that dates each label at its maturity.
- **(d) Research results and pattern health are future state.** Each needs a result timestamp and must fail closed in `memory_firewall`.
- **(e) Import-path enforcement.** `trader_view.assert_trader_path_clean` checks the import closure. A separate `engine/research/` package therefore lets the existing CI check forbid the trader from importing research code by path. This is the main justification for a new package.
  - Blind-safe primitives stay in `engine/learning`: pure functions with `now` and no year, such as the priority maths and allocators.
  - The trader-side `stage_research` should shrink to consuming released agenda items.

**2. The volatility-first two-stage architecture versus the existing fv_pipeline and movers.**
- **Two-stage already exists.** `fv_pipeline` already implements M->D->E->S with a certified threshold. On real data the direction gate never opened, so M+D placed 0 bets over 716 weeks (`state/research/fv/run1/report.md`).
- **Four mover models exist.** They are `fv_pipeline.MoverStage`, `pattern_movers.MoverModel`/`PatternMoverModel`, `direction_features.mover_walk_forward` and `scripts/movers.py`. They use different labels: a ±10% weekly touch versus up_first/up_sign, and different pools. The volatility lab must name one champion and one label/horizon contract. Otherwise a fifth model appears and the lab and fv results stop being comparable.
- **Direction must read predicted movers.** The direction lab must consume the champion's walk-forward picks, as `direction2` does, and never realised movers. The older `state/research/direction/summary.json` itself carries the caveat "realised movers".
- **The volatility-evidence gate must be an object.** "Direction only after volatility has evidence" needs an explicit gate that the priority engine can read. The fv 0.80/0.95 certification and the s.11 frontier must share one definition of the 80% target.
- **Existing findings to carry forward.** Direction on movers has been at base rate (51-52%). The 7% band is structurally hard. The planned volatility-to-direction compute shift in s.50 will likely show "direction yields nothing". The research-health system must treat that as a valid answer, not as a stall.

**3. The market-wide observer's cost on about 3,000 names per day.**
The figures below are arithmetic estimates, not measurements.
- **Scale.** 3,000 names × 252 days is about 0.76M name-days per year, or about 38M over 50 years. At 54 float32 features that is about 8 GB, which the 16 GB shared machine cannot hold. CONTEXT rule 10 allows about 2.5 GB free.
- **Classification cost.** Labelling categories A-I is cheap when vectorised per day (ranks and quantiles over 3,000 rows).
- **Snapshot cost.** The real cost is the per-event "could I have known" reconstruction, because `parity`'s strict recomputation rebuilds features for a date.
- **Design.**
  - Stream year-partitioned parquet with only needed columns in float32.
  - Classify every name every day, but persist only the exception rows: the top and bottom 1% tails, near misses, false positives, false negatives and abstentions. That is roughly 100-200 rows per day, or 25-50k per year.
  - Build one PIT snapshot per day and share it across that day's events.
  - Cap counterfactual reports at the top K events per day, with a seeded sample of the rest.
  - Checkpoint per year.
- **Survivorship.** Adding delisted names (RG04) grows the universe. The per-day budget must be set before that happens.

## Work packages (C66 priority order: anti-cheating first, then volatility, then losses, then direction, then the research brain)

"Rows" gives the section-45 rows each package closes. "Min new lines" is the sum of row gaps. Where a row's gap is 0 because existing code was credited, a design floor is given for the missing function and marked (floor). A floor is an estimate, not a ruler number. Every package also owns planted-leak tests for its own outputs (46D).

| WP | package | target modules | builds on | rows | min new lines | owns section-46 tests |
|---|---|---|---|---|---:|---|
| 1 | **Research/trader firewall and research namespace** | NEW `engine/research/{__init__,namespace,release}.py`; a RESEARCH layer in `engine/learning/firewalls.py`; extend `trader_view` forbidden closure to `engine.research` | trader_view, curator, test_path, firewalls, future/memory/identity_firewall, leak_audit | 28 | 1,000 (floor) | D (future price, label, event, year identity, future pattern status, future experiment result); RT07-RT09; E (research outputs) |
| 2 | **Market-wide observer and daily autopsy** (plus a survivor-free universe) | NEW `engine/research/observer.py`, `engine/research/autopsy.py` | fv_pipeline.build_panel/Panel, learning/missed_winners.week_from_base, engine/missed_winners, data_sources delisted registry, reports.Builder, surprise | 3, 20 | 3,000 | A; planted A-I categories; empty-day; RAM/time budget test |
| 3 | **Volatility laboratory** (Objective 1) | NEW `engine/research/volatility_lab.py` (+ hypothesis specs H1-H9, open H10+) | fv_pipeline.MoverStage (champion), pattern_movers, direction_features.mover_walk_forward (fold in), candidates, pattern_stats, transfer | 8 | 1,812 | C planted volatility world; C null world; F transfer (years/stocks/sectors/regimes); RS01-RS03 re-measured survivor-free |
| 4 | **Knowability, "could I have known" and unknown cause** | NEW `engine/research/knowability.py`, `engine/research/counterfactual.py`; extend `engine/learning/unknowns.py` | pit, edgar, parity/parity_suite, future_firewall, curator, unknowns, pattern_reliability verdicts | 6, 7, 30 | 4,408 | RT20 unknown-cause (incl. forced-labelling); RS04; D (classification never reaches the trader) |
| 5 | **Loss-first outcome research**: market-wide winners and losers, missed winners and losers, symmetry, loss-risk bank | NEW `engine/research/outcomes.py`, `engine/research/symmetry.py`; extend `engine/learning/missed_winners.py` (missed losers) | failure, postmortem, separation, credit, both missed_winners, pattern_stats, boundary | 4, 5, 11 | 3,157 (incl. 800 floor for row 4) | RT19 symmetry; H loss-priority (world data); planted missed-loser |
| 6 | **Direction laboratory and conditional frontier** (gated on WP3 evidence) | NEW `engine/research/direction_lab.py`, `engine/research/frontier.py` | direction, direction_calib, direction_ablate, direction_features (frontier, eighty_question), fv DirectionStage | 9, 10 | 3,239 | C planted direction world; C null world; RS05-RS09 re-measured survivor-free |
| 7 | **Research agenda**: priority engine, question generator, hypothesis tree, daily target generator | extend `engine/learning/research_priority.py`, `research_policy.py` (blind-safe maths); NEW `engine/research/agenda.py`, `engine/research/hypothesis_tree.py` | research_priority, research_policy, competition, experiment_memory hypotheses, surprise, contradiction_monitor | 1, 21, 35, 36 | 2,422 | G research-policy (A huge information value vs B tiny); H loss-priority; RT16 |
| 8 | **Autonomous loop and compute governance**: loop, escalation ladder, value accounting, waste, brain health, diversity | NEW `engine/research/{loop,ladder,accounting,waste,brain_health}.py`; extend `research_policy.ResearchTarget`/allocators | checkpoints, compute (execution ledger, run_experiment_process), resources, research_policy (SPRT, successive_halving), retirement (DORMANT) | 2, 17, 18, 19, 32, 33 | 5,095 (incl. 500 floor for ladder) | I compute-waste; RT17; RT18; crash/resume (RA20) |
| 9 | **Discovery expansion**: pattern discovery, interactions, multi-scale, cross-sectional, regime | NEW `engine/research/{discovery,interactions,horizons,cross_section,regimes}.py` | patterns, candidates, pattern_stats, pattern_identity, knowledge, complexity, context, hierarchy, situation, analogs_sector, trust, candles | 12, 22, 23, 24, 25 | 4,410 (incl. 400 floor miner->KnowledgeObject bridge, 400 floor regime discovery) | C planted interaction / regime; null world with multiple testing; F cross-stock/sector/regime |
| 10 | **Pattern-break research integration** (one break engine, scheduled) | extend `engine/learning/break_detection.py` (wrap `engine/pattern_reliability`) | break_detection, pattern_reliability, boundary, health, lifecycle | 13 | 600 (floor) | C planted failure-regime world; RS14 re-run |
| 11 | **Experiment memory, meta-learning and failed-learner lab** (one ledger) | extend `engine/learning/experiment_memory.py`, `meta_learning.py`, `failed_learners.py` | learning/experiment_memory, engine/experiment_memory, registry, compute ledger, meta_learning, failed_learners | 14, 15, 16 | 2,105 | duplicate-experiment tests; RS16; RS17 |
| 12 | **Knowledge to decision**: graph, bridge, scientific memory query, replication, promotion gate | extend `knowledge_graph.py`, `decision_contract.py`, `promotion.py`; NEW `engine/research/{replication,why}.py` | knowledge_graph, decision_contract, champion, archive, belief, interpretation, transfer, promotion, complexity | 26, 27, 29, 34, 37 | 1,692 (492 + floors: 400 replication, 400 why-query, 400 gate) | RT21 replication; RT22 gates; RG05-RG07 |
| 13 | **Scorecard and end-to-end validation harness** | extend `engine/learning/scorecard.py`; NEW `engine/research/planted_worlds.py`; `tests/integration/` | scorecard, learning_curve, planted_world, same_year, controls, transfer | 31 | 400 (floor) | B integration (brain -> learner -> decision); C all worlds end to end; E memorisation; F transfer; section-47 cycle |

The package sum (33,340 = 28,040 row gap + 5,300 of floors) exceeds the 28,040 row gap because of the floors. Neither figure is a target: the ruler minimum per row is.

## Evidence already on disk that C66 must respect

- **Movement has an edge.** The fv mover hit rate is 57.9% against a 14.6% base (`state/research/fv/run1/report.md`). OOS movement AUC is 0.733 (`state/research/heavy_algo/H9b8e5c64e6b5a404.md`).
- **Direction on predicted movers is at chance.** Accuracy is about 0.51. The 80% question found 0 cells in 6 configurations, while a planted signal was recovered at 0.92 (`state/research/direction2/run1/`).
- **Pattern reliability is not predictable.** Skill is -0.20% and AUC 0.498 (`state/research/pattern_reliability/report.md`).
- **Stops lower P(loss>20%) but cannot cap gap losses.** P(loss>20%) falls from 3.26% to 1.22% out of sample, but the worst loss is still -73% (`state/research/exits_stops/report.txt`).
- **All four results come from the survivor-only panel.**

C66 section 48 applies: none of these is market knowledge until it survives unseen time, stocks, regimes and fresh seeds, on a panel that includes delisted names.
