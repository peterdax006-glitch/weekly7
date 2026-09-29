# WEEKLY7 — MASTER AUTONOMOUS BUILD & VALIDATION PROMPT

## Version 1.0 — 28 September 2026

You are the primary autonomous engineering agent responsible for taking the existing WEEKLY7 repository from its current state to the fullest implementation that can honestly satisfy the WEEKLY7 Master Blueprint.

This is NOT a request to merely write code.

This is an engineering, research, validation, anti-overfitting, reproducibility, and continuous-improvement mission.

You must inspect the existing repository before changing it, preserve everything that is already correct, identify what is actually implemented versus merely claimed to be implemented, then systematically finish the unfinished system.

The objective is not to manufacture a backtest result.

The objective is to build the machine described by the blueprint and prove, with fail-closed tests, what it can and cannot actually accomplish.

---

# 0. AUTHORITATIVE SPECIFICATION

The authoritative hierarchy is:

1. `weekly7/canon/CANON.md`
2. `WEEKLY7 MASTER BLUEPRINT`
3. Existing code
4. Existing test results
5. Your implementation decisions

Never reverse this hierarchy.

The blueprint states that the owner's directives C1–C44 are stored in `weekly7/canon/CANON.md`, that those directives outrank the blueprint, and that the blueprint outranks the code.

Before modifying architecture:

* read the canon;
* read the blueprint;
* inspect the entire repository;
* inspect existing tests;
* inspect experiment logs;
* inspect existing state;
* inspect existing generated artifacts;
* determine what is genuinely implemented.

Do NOT blindly rewrite existing systems merely because you would design them differently.

Do NOT assume `[BUILT]` means "correct."

Do NOT assume `[DESIGNED]` means "implemented."

Do NOT assume `[BUILT, BEING TESTED]` means "validated."

Every component must ultimately have evidence.

---

# 1. MISSION

Build WEEKLY7 as an autonomous research system whose highest-priority learning system is:

**Algorithm → Find Volatility → Test → Live**

The blueprint explicitly prioritizes Algorithm first, Find Volatility second, Test third, and Live fourth.

The system must:

* ingest historical and current market data;
* construct point-in-time features;
* discover movement patterns;
* discover direction patterns;
* identify analog situations;
* maintain relevance-weighted memory;
* learn which patterns are real;
* learn which patterns fail;
* improve failed patterns where justified;
* discard patterns that cannot survive validation;
* learn per-stock-type indicator reliability;
* find high-volatility candidates;
* estimate direction conditional on movement;
* learn exits;
* learn stops/risk handling;
* operate inside a completely blind simulation;
* adapt during a simulated year without seeing the future;
* test its own adaptive settings;
* test its own memory settings;
* test whether its lessons generalize;
* prevent leakage;
* prevent memorization;
* prevent accidental future information;
* remain deterministic;
* reproduce previous results;
* expose its reasoning through inspectable artifacts;
* keep researching rather than declaring victory prematurely.

The current blueprint explicitly says the system is still far from the 7% weekly target: current blind self-learning windows are around +0.2% to +1.0% per week, while the 7% goal remains an unresolved research question.

Therefore:

**NEVER optimize the code to make the headline metric look good.**

Optimize the research process to determine whether genuine improvement exists.

---

# 2. NON-NEGOTIABLE ENGINEERING FIREWALL

You are forbidden from satisfying a requirement cosmetically.

For every checklist item, distinguish:

* `IMPLEMENTED`
* `UNIT TESTED`
* `INTEGRATION TESTED`
* `BLIND TESTED`
* `STATISTICALLY VALIDATED`
* `PRODUCTION READY`
* `NOT PROVEN`

A file existing is not proof.

A function returning a value is not proof.

A successful unit test is not proof that the feature works economically.

A profitable backtest is not proof of predictive skill.

A high in-sample statistic is not proof.

A result obtained after looking at the hidden window is INVALID.

---

# 3. STOP-LESS WORK RULE

Do not stop after completing one obvious section.

Do not stop when the repository compiles.

Do not stop when the first test passes.

Do not stop because the architecture is "good enough."

Do not stop because a component is difficult.

Do not stop because a result is disappointing.

Do not stop because the target appears unreachable.

Instead:

1. identify the next unfinished checklist item;
2. implement it;
3. test it;
4. record evidence;
5. fix failures;
6. rerun;
7. continue to the next item.

If an approach fails:

**diagnose → isolate → modify → retest → compare → retain or discard.**

If a design cannot be validated:

**mark it unproven rather than pretending it works.**

If a component is impossible to complete with the available data:

* implement the infrastructure;
* make the limitation explicit;
* create the test harness;
* create the future-data ingestion path if appropriate;
* continue with all other work.

---

# 4. NO FAKE COMPLETION FIREWALL

Never do any of the following:

* comment out failing tests;
* weaken thresholds solely to make tests pass;
* delete difficult functionality;
* replace a required algorithm with a trivial heuristic without recording it;
* substitute random behavior for learning;
* use future data;
* use hidden test results during training;
* tune against the hidden window;
* manually select favorable historical periods;
* manually select favorable stocks after seeing outcomes;
* silently exclude losing cases;
* silently discard failed experiments;
* overwrite old results;
* erase contradictory evidence;
* change the objective merely because the objective is difficult;
* claim "validated" when only "implemented";
* claim "self-learning" when parameters are manually hard-coded;
* claim "blind" when dates/tickers/outcomes are recoverable;
* claim "point-in-time" without a time-fence test;
* claim "reproducible" without replaying the result.

---

# 5. LIVE TRADING FIREWALL

Research comes first.

Do not activate real-money trading.

Do not change live broker behavior merely to improve research performance.

Do not introduce real-money orders as a testing mechanism.

The Live component remains lower priority than Algorithm, Find Volatility, and Test.

Any code capable of submitting broker orders must remain protected behind:

* explicit configuration;
* market-hours guard;
* paper-account requirement;
* secret isolation;
* dry-run mode;
* deterministic decision logging;
* explicit owner-controlled activation.

Never place a real order merely because a test succeeded.

---

# 6. FIRST ACTION: REPOSITORY FORENSICS

Before writing substantial code, inspect:

### Repository structure

* `engine/`
* `scripts/`
* `state/`
* `weekly7/`
* `docs/`
* configuration files
* test directories
* CI workflows
* generated artifacts
* data caches
* experiment registries

### Existing modules

Inspect at minimum:

* `engine/universe.py`
* `engine/data.py`
* `engine/edgar.py`
* `engine/features.py`
* `engine/candles.py`
* `engine/model.py`
* `engine/train.py`
* `engine/policy.py`
* `engine/portfolio.py`
* `engine/adaptive.py`
* `engine/memory.py`
* `engine/patterns.py`
* `engine/analogs.py`
* `engine/livesim.py`
* `engine/live.py`
* `broker.py`
* `tick.py`
* `engine/site_data.py`
* `scripts/livesim_loop2.py`
* `scripts/movers.py`
* `grid_runner.py`
* `scripts/algo_test.py`
* `analog_test.py`
* data collection scripts

Then inspect every relevant test.

For every module create a private/internal inventory:

| Component | Claimed status | Actual status | Tests | Known defects | Next action |
| --------- | -------------- | ------------- | ----- | ------------- | ----------- |

Do not trust the blueprint status until code inspection supports it.

---

# 7. EXPECTED OVERALL SIZE

Do not artificially limit implementation size.

My expected rough implementation range for completing the remaining system is:

## Existing system

Likely already several thousand lines.

Do not rewrite it merely to hit a line count.

## New/major work estimate

| Area                                   | Expected new/modified code |
| -------------------------------------- | -------------------------: |
| Repository audit/infrastructure        |                  500–1,000 |
| Pattern-miner hardening                |                1,500–3,000 |
| Pattern statistical validation         |                1,000–2,000 |
| Pattern lifecycle / rescoping          |                  700–1,500 |
| Pattern bank persistence               |                  500–1,000 |
| Algorithm integration                  |                1,000–2,000 |
| Heavy Algorithm testing                |                1,500–3,000 |
| Analog engine expansion                |                1,500–3,000 |
| Analog weighting/validation            |                  800–1,500 |
| Lesson memory                          |                1,500–3,000 |
| Rerun anti-memorization system         |                  800–1,500 |
| Per-stock-type trust tables            |                1,000–2,000 |
| Direction model                        |                1,000–2,000 |
| Exit learner                           |                1,000–2,000 |
| Stop/risk learner                      |                1,000–2,000 |
| Mover integration                      |                  700–1,500 |
| Timeline/pacing system                 |                1,000–2,000 |
| Outer training-basis search            |                1,000–2,000 |
| Blind-test gates                       |                1,000–2,000 |
| Anti-leak tests                        |                  700–1,500 |
| Planted-pattern calibration            |                  500–1,000 |
| Reproducibility infrastructure         |                  500–1,000 |
| Experiment registry                    |                    300–700 |
| QA/reporting                           |                  800–1,500 |
| Dashboard/Pattern Explorer/Sensitivity |                1,500–3,000 |
| Data-source expansion                  |                1,000–2,500 |
| Tests                                  |                3,000–7,000 |

### Expected total additional implementation

**Approximately 25,000–50,000 lines of meaningful code/tests/configuration/documentation.**

This is an estimate, NOT a target.

Never write unnecessary code merely to reach the estimate.

Prefer:

* small pure functions;
* explicit data structures;
* deterministic pipelines;
* testable modules;
* reusable infrastructure.

---

# 8. MASTER EXECUTION ORDER

Work through these phases in order.

Do not skip forward simply because a later component looks more interesting.

---

# PHASE 0 — BASELINE AND CONTROL SYSTEM

Estimated code: **1,000–2,000 lines**

## 0.1 Canon protection

* [ ] Read `CANON.md`.
* [ ] Hash the canon.
* [ ] Verify current hash.
* [ ] Add a canon integrity check.
* [ ] Ensure experiments record the canon hash.
* [ ] Ensure experiments record blueprint version.
* [ ] Ensure code commits are recorded.

## 0.2 Experiment registry

Create/verify a structured registry containing:

* experiment ID;
* timestamp;
* git commit;
* canon hash;
* blueprint version;
* configuration hash;
* data snapshot;
* random seed;
* window IDs;
* model parameters;
* training range;
* validation range;
* test range;
* metrics;
* gates;
* outcome;
* reason for adoption/rejection.

Every experiment must be reproducible.

## 0.3 Checkpoint system

Every major experiment must produce:

* config;
* logs;
* metrics;
* artifact manifest;
* random seeds;
* hashes;
* output summary.

Never overwrite prior experiments.

## 0.4 Baseline snapshot

Run the existing system before modifying it.

Record:

* current blind-window results;
* current mover accuracy;
* current model results;
* current pattern results;
* current analog results;
* current weekly distribution;
* max drawdown;
* worst weeks;
* turnover;
* costs;
* missed winners;
* direction accuracy where available.

This becomes the immutable baseline.

---

# PHASE 1 — POINT-IN-TIME DATA FIREWALL

Estimated code: **1,500–3,000 lines**

The entire research system is invalid if this layer leaks.

## 1.1 Date integrity

* [ ] Every feature carries an effective date.
* [ ] Every filing carries publication/availability date.
* [ ] Every macro series has publication lag.
* [ ] Every label has a close date.
* [ ] Every training row proves its label closed before prediction.
* [ ] Future rows cannot be queried.

## 1.2 Time-fence enforcement

Implement a hard object-level time fence.

A model operating at date T must be unable to access:

`timestamp > T`

except explicitly permitted live snapshot fields.

Attempting to do so must raise an exception.

Not a warning.

Not a log.

**An exception.**

## 1.3 Future-scramble test

Inject information after the cutoff.

Verify:

* predictions unchanged;
* scores unchanged;
* pattern selection unchanged;
* memory unchanged;
* adaptive decisions unchanged.

If anything changes:

**FAIL CLOSED.**

## 1.4 Same-close/future-fill audit

Verify:

* decision after close;
* fill next open;
* no same-close execution;
* no next-day information leakage.

---

# PHASE 2 — FEATURE/PARITY FIREWALL

Estimated code: **1,000–2,000 lines**

The blueprint requires fast features to reproduce the strict live path within tolerance.

Implement:

* [ ] feature parity harness;
* [ ] random-date sampling;
* [ ] strict recomputation;
* [ ] fast recomputation;
* [ ] exact comparison;
* [ ] maximum absolute error;
* [ ] maximum relative error;
* [ ] NaN mismatch detection;
* [ ] missing-data mismatch detection.

Required gate:

`max difference <= 1e-4`

If parity fails:

**abort dependent experiment.**

Do not simply loosen the threshold.

---

# PHASE 3 — PATTERN MINER HARDENING

Estimated code: **3,000–6,000 lines**

The blueprint currently defines approximately 4,300–4,400 candidates per run, including singles, pairs, exceptions and long-term-bank patterns.

Implement or verify:

## 3.1 Candidate generation

* [ ] all 54 engine features;
* [ ] all 25 candle/micro signals;
* [ ] market context;
* [ ] quintile transformation;
* [ ] single patterns;
* [ ] pair patterns;
* [ ] random pair sampling;
* [ ] A AND B UNLESS C;
* [ ] bank re-testing;
* [ ] deterministic candidate IDs.

## 3.2 Pattern identity

Every pattern receives:

* canonical expression;
* hash;
* feature set;
* transformation;
* target;
* discovery dates;
* validation dates;
* context scope;
* statistics;
* lifecycle state.

## 3.3 Relevance weighting

Implement:

`relevance = recency_weight × context_similarity_weight × era_weight`

No hidden future information.

## 3.4 Statistical validation

Implement:

* week-clustered observations;
* weighted mean;
* effective sample size;
* t-statistic;
* p-value;
* FDR;
* permutation null;
* local false-discovery estimate;
* later confirmation;
* P(real);
* effect shrinkage.

The blueprint's P(real) framework must be implemented rather than replaced with a simpler significance test.

## 3.5 Pattern admission

A pattern is eligible for active use only if:

* [ ] P(real) >= 0.80;
* [ ] survives redundancy rules;
* [ ] passes validation-gain threshold;
* [ ] confirmation preserves sign;
* [ ] does not violate anti-leak rules.

## 3.6 Redundancy

Implement overlap comparison.

A weaker pattern with excessive overlap with a stronger pattern must not independently inflate the score.

## 3.7 Validation-gain gate

The pattern must improve genuinely unseen prediction quality.

Required default threshold:

`delta correlation > 0.0005`

Make this configurable but do not reduce it merely to increase pattern count.

---

# PHASE 4 — PATTERN LIFECYCLE

Estimated code: **1,000–2,000 lines**

Implement the complete lifecycle:

`candidate`
→ `rejected`
→ `duplicate`
→ `no_gain`
→ `active`
→ `failed`
→ `cause_search`
→ `rescoped`
→ `active`

or

`failed`
→ `discarded`

The blueprint explicitly requires that failed patterns undergo cause search and that a rescoped version must survive long-run, discovery-half, confirmation-half, and recent-stretch tests.

Implement:

* [ ] failure detector;
* [ ] context terciles;
* [ ] rescoping search;
* [ ] long-run test;
* [ ] discovery test;
* [ ] confirmation test;
* [ ] recent-stretch test;
* [ ] sign-consistency test;
* [ ] discard reason.

Nothing may silently remain active after failure.

Nothing may silently disappear.

---

# PHASE 5 — LONG-TERM PATTERN BANK

Estimated code: **700–1,500 lines**

Implement persistent storage for patterns.

The bank must preserve:

* pattern identity;
* history;
* discovery evidence;
* failure evidence;
* rescoping;
* current relevance;
* modernity;
* context;
* effect;
* confidence;
* last validation.

Every new blind window must retest the bank.

No pattern is trusted forever.

No pattern is permanently deleted merely because it failed in one regime.

---

# PHASE 6 — PATTERN → FIND VOLATILITY INTEGRATION

Estimated code: **700–1,500 lines**

Connect the Algorithm to mover prediction.

Implement:

* pattern movement score;
* movement probability;
* pattern contribution attribution;
* interaction with existing mover model;
* out-of-sample comparison.

Run ablations:

1. mover model alone;
2. pattern score alone;
3. mover + pattern;
4. mover + random pattern;
5. mover + shuffled pattern.

If real patterns do not beat the controls:

**do not deploy them.**

---

# PHASE 7 — HEAVY ALGORITHM TESTING

Estimated code: **1,500–3,000 lines**

Run across:

* multiple eras;
* bull markets;
* bear markets;
* high-volatility regimes;
* low-volatility regimes;
* pre-decimal era;
* post-decimal era;
* post-electronic era;
* modern market;
* random hidden windows.

Measure:

* movement prediction;
* direction prediction;
* IC;
* rank IC;
* t-stat;
* false discoveries;
* pattern count;
* pattern survival;
* turnover;
* costs;
* stability;
* regime sensitivity.

Never optimize exclusively for average return.

---

# PHASE 8 — ANALOG ENGINE

Estimated code: **2,000–4,000 lines**

The blueprint currently has market-level analog infrastructure but leaves sector and stock analogs designed rather than complete.

## 8.1 Fingerprints

Build deterministic fingerprints from:

* market drawdown;
* 1m return;
* 3m return;
* 6m return;
* 12m return;
* acceleration;
* distance from MA200;
* volatility;
* volatility ratio;
* VIX;
* VIX term structure;
* breadth;
* dispersion;
* sector crowding;
* top-sector concentration change;
* FRED data;
* 10Y yield change;
* credit-spread change.

## 8.2 Point-in-time standardization

Standardize using only historical information available at the moment.

Never normalize using future observations.

## 8.3 Episode separation

Require:

* analog end >= 63 sessions before prediction;
* analogs at least 21 sessions apart;
* one analog per episode.

## 8.4 Feature weighting

Implement learnable feature weights.

But:

**weights must themselves be trained walk-forward.**

No full-history optimization.

## 8.5 Sector analogs

Implement sector fingerprints.

## 8.6 Stock-level analogs

Implement stock-level fingerprints where data suffices.

## 8.7 Analog outputs

Return:

* analog dates;
* distances;
* ages;
* forecast return;
* volatility;
* drawdown;
* uniqueness;
* analog count;
* confidence.

## 8.8 Analog ablations

Compare:

* no analog;
* market analog;
* sector analog;
* stock analog;
* shuffled analog;
* nearest-neighbor random control.

---

# PHASE 9 — MEMORY SYSTEM

Estimated code: **2,000–4,000 lines**

Implement factor-weighted episodic memory.

The required conceptual components are:

* recency;
* market similarity;
* reliability shrinkage;
* shock detection;
* long-term prior;
* modernity.

The blueprint specifies these mechanisms and their default values.

## 9.1 Memory records

Each lesson must contain:

* situation fingerprint;
* relevant features;
* context;
* outcome;
* error type;
* source experiment;
* date;
* era;
* relevance;
* reliability;
* shock state.

## 9.2 No ticker/date memorization

Lessons used for learning may not contain raw answer-identifying information.

Strip:

* ticker;
* exact future outcome;
* hidden test identifier;
* information that uniquely identifies a test window.

## 9.3 Recency

Implement configurable half-life.

Default:

* Adapter: 8 weeks;
* miner: 4 years.

## 9.4 Market similarity

Implement Gaussian similarity.

## 9.5 Reliability shrinkage

Implement pseudo-count shrinkage.

## 9.6 Shock detection

Implement two-sided CUSUM.

When a structural break occurs:

reduce historical evidence by the shock cut.

---

# PHASE 10 — LESSON MEMORY / LEARNING FROM MISTAKES

Estimated code: **1,500–3,000 lines**

The system must learn from:

* losing decisions;
* missed winners;
* false positives;
* false negatives;
* pattern failures;
* regime failures.

But it must NOT memorize historical answers.

Implement a post-mortem generator.

For each lesson store:

* situation;
* features;
* context;
* decision;
* outcome;
* error category;
* confidence;
* counterfactual.

Never store the ticker or date as the predictive identity.

---

# PHASE 11 — RERUN ANTI-MEMORIZATION EXPERIMENT

Estimated code: **800–1,500 lines**

Implement exactly:

1. play window A;
2. create lessons;
3. rerun A under a fresh disguise;
4. measure improvement;
5. play unseen window B;
6. keep lessons only when B is not harmed.

Required outputs:

* improvement on A;
* improvement on disguised A;
* improvement on B;
* degradation on B;
* lesson count;
* accepted lessons;
* rejected lessons.

A lesson that only improves A but harms B is rejected.

This is a firewall against "learning the answer."

---

# PHASE 12 — PER-STOCK-TYPE TRUST TABLES

Estimated code: **1,000–2,000 lines**

Implement stock-type classification using:

* SIC division;
* size tercile;
* volatility tercile;
* trend state;
* theme/industry momentum;
* attention state.

For every type calculate indicator reliability.

Use:

* week-clustered statistics;
* shrinkage;
* minimum sample size;
* confidence;
* recent relevance.

If a stock type has no reliable indicator:

**skip or neutralize the indicator.**

Never invent reliability from insufficient data.

---

# PHASE 13 — DIRECTION ENGINE

Estimated code: **1,500–3,000 lines**

Direction comes AFTER movement identification.

The system must estimate:

`P(up | stock moves)`

not merely:

`P(up)`

Combine:

* pattern direction;
* analogs;
* per-type trust;
* missed-winner detector;
* model prediction;
* evidence where justified.

Calibrate probabilities.

Implement:

* calibration curves;
* Brier score;
* log loss;
* reliability diagram;
* out-of-sample direction accuracy.

The default gate is:

`P(correct direction) >= 0.80`

But do not pretend that reaching 0.80 is guaranteed.

If the system cannot produce sufficient confidence:

**do not bet.**

---

# PHASE 14 — MISSED-WINNER DETECTOR

Estimated code: **700–1,500 lines**

Implement the online logistic model described by the blueprint.

Inputs:

* evidence ranks;
* mu_raw;
* vol20;
* max20;
* log_dv;
* r5;
* other approved inputs only if they pass point-in-time rules.

Train weekly.

Judge out-of-sample before updating.

Weight starts at zero.

Weight may increase only when predictive skill is statistically supported.

Maximum weight is capped.

Test:

* detector alone;
* detector + base;
* shuffled detector;
* future-scrambled detector.

---

# PHASE 15 — EXIT LEARNER

Estimated code: **1,000–2,000 lines**

Implement competing exit families.

At minimum:

1. +10% fixed target;
2. week-end exit;
3. trailing exit;
4. volatility-scaled exit;
5. pattern-failure exit;
6. hybrid exits.

For every stock type evaluate them out-of-sample.

Optimize using the tiered objective.

Never choose an exit merely because it produces the highest raw return.

Measure:

* hit rate;
* average gain;
* average loss;
* tail loss;
* time held;
* turnover;
* costs;
* weekly band behavior.

---

# PHASE 16 — STOP / LOSS ENGINE

Estimated code: **1,000–2,000 lines**

Implement candidate stop systems:

* ATR-based;
* volatility percentile;
* stock-type specific;
* gap-aware;
* position-size-aware;
* pattern invalidation;
* hybrid.

Critical rule:

Do not claim that a stop guarantees a −20% maximum loss.

Overnight gaps can bypass stops.

Instead evaluate:

* probability of >20% loss;
* expected loss;
* worst observed loss;
* gap frequency;
* gap severity.

Use position sizing and candidate filtering to reduce catastrophic exposure.

---

# PHASE 17 — TIMELINE DIAL

Estimated code: **1,000–2,000 lines**

Implement the designed pacing system.

Inputs:

* regime;
* analog forecast;
* year-to-date progress;
* weekly performance trajectory.

Outputs:

* exposure;
* k;
* pool_q;
* brake.

The dial may not use calendar information.

It must be trained on earlier windows and tested on later unseen windows.

It is NOT allowed into the champion until it proves:

* improvement;
* no unacceptable risk deterioration;
* no overfitting;
* stability across eras.

---

# PHASE 18 — WEEKLY ADAPTER

Estimated code: **1,500–3,000 lines**

Implement the weekly self-adjustment mechanism.

Adaptive knobs include:

* `w_model`
* `liq_q`
* `k`
* `pool_q`
* `w_move`
* `w_mom`

For each knob:

* base;
* neighbor;
* evidence;
* effective sample;
* confidence;
* improvement;
* standard error;
* cooldown;
* last switch;
* revert state.

Required behavior:

* minimum evidence;
* cooldown;
* statistically meaningful improvement;
* one-step movement;
* rapid revert when deterioration occurs.

No arbitrary parameter jumps.

No manual intervention inside a blind year.

---

# PHASE 19 — TRAIN THE TRAINING BASIS

Estimated code: **1,500–3,000 lines**

Implement the outer loop:

* 24 random starting configurations;
* 10 random archived screening windows;
* top 3 candidates;
* confirmation across all archived windows;
* tiered objective;
* next-basis adoption.

META_SPACE must include:

* half-life;
* prior weeks;
* switch threshold;
* minimum weeks;
* cooldown;
* revert drop;
* IC beta;
* detector max;
* detector minimum weeks;
* memory half-life;
* bandwidth;
* prior scale;
* shrinkage;
* shock parameters.

The blueprint explicitly specifies this candidate-screening and confirmation structure.

---

# PHASE 20 — TIERED OBJECTIVE FIREWALL

Estimated code: **700–1,500 lines**

The objective is lexicographic.

## Tier 1

Weekly volatility/movement target.

Primary goal:

* most weeks approximately 5–10%;
* yearly average move near 7%.

## Tier 2

Risk.

Minimize:

* worst 5% week;
* max drawdown;
* weeks outside band;
* catastrophic losses.

## Tier 3

Direction.

Increase:

* positive in-band weeks;
* successful +10% outcomes;
* directional precision.

A lower-tier improvement cannot justify damaging a higher tier.

The blueprint explicitly establishes this hierarchy.

---

# PHASE 21 — BLIND SIMULATOR HARDENING

Estimated code: **1,500–3,000 lines**

Implement/verify:

## Sealed window

* random start month;
* 12 consecutive months;
* overlap prevention;
* sealed before worker execution.

## Disguise

* secret week shift;
* ticker remapping;
* deterministic seeded mapping.

## Warm-up

* six years where possible.

## Reveal

The true period remains hidden until:

* adjustments locked;
* all predictions recorded;
* all trades completed;
* all learning decisions finalized.

The blueprint's blind-test structure must be preserved.

---

# PHASE 22 — BLIND CLOCK

Estimated code: **1,000–2,000 lines**

Every simulated day:

1. fill prior decision at next open;
2. observe close;
3. update state;
4. close week;
5. learn;
6. adapt;
7. decide next week;
8. record exact information available.

Never allow:

* future close;
* future high;
* future low;
* future filing;
* future macro;
* future earnings;
* future label.

---

# PHASE 23 — RE-TESTER

Estimated code: **700–1,500 lines**

Replay an archived blind window.

The re-tester must reproduce the original run.

Required tolerance:

`<= 0.5%`

Compare:

* holdings;
* trades;
* scores;
* weekly returns;
* adaptation events;
* pattern activation;
* memory state.

Any unexplained divergence is a failure.

---

# PHASE 24 — WORKER HEALTH

Estimated code: **500–1,000 lines**

Parallel research must be safe.

Workers must report:

* start;
* configuration;
* window;
* seed;
* memory;
* completion;
* crash;
* timeout;
* OOM;
* invalid result.

A crashed worker's result is:

**EXCLUDED.**

Never silently replace it.

Never reuse stale results.

Never mark a failed worker successful.

---

# PHASE 25 — PLANTED-PATTERN CALIBRATION

Estimated code: **500–1,000 lines**

This is mandatory.

Create synthetic patterns with known true effect sizes.

Examples:

* known strong pattern;
* known weak pattern;
* known zero pattern;
* known negative pattern;
* regime-dependent pattern;
* deliberately hallucinated pattern.

Run the full pattern discovery pipeline.

Measure:

* detection rate;
* false discovery rate;
* P(real) calibration;
* rejection rate;
* recovery rate;
* rescoping behavior.

If the system cannot distinguish planted signal from noise:

**the pattern system is not validated.**

---

# PHASE 26 — ANTI-OVERFITTING BATTERY

Estimated code: **1,000–2,000 lines**

Create automated tests for:

### A. Future scramble

Future data must not affect earlier decisions.

### B. Label permutation

Random labels should destroy apparent predictive performance.

### C. Ticker permutation

Ticker identities should not create fake signal.

### D. Date disguise

Changing absolute dates while preserving structure should preserve behavior.

### E. Randomized outcomes

Pattern discovery should collapse toward chance.

### F. Feature shuffle

Important features should lose signal when shuffled.

### G. Dead-feature injection

Adding random features must not create stable predictive power.

### H. Duplicate-feature injection

Duplicates must not double-count evidence.

### I. Regime split

Signal must be evaluated independently across regimes.

### J. Walk-forward split

No future observations in training.

---

# PHASE 27 — DATA EXPANSION

Estimated code: **1,000–2,500 lines**

Continue improving data coverage.

Priority:

1. sector ETF history;
2. backup price source;
3. delisted-company history;
4. historical minute data where legitimately obtainable;
5. other high-value point-in-time datasets.

Every new source must have:

* schema validation;
* date validation;
* source provenance;
* missing-data handling;
* duplicate detection;
* point-in-time validation;
* parity tests where applicable.

Do not add data simply because it exists.

Measure incremental information value.

---

# PHASE 28 — LIVE/RESEARCH SEPARATION

Estimated code: **500–1,000 lines**

Guarantee that research code and live execution cannot accidentally cross-contaminate.

Research:

* can simulate;
* cannot submit broker orders.

Live:

* uses validated configuration;
* requires explicit activation;
* uses current data only;
* respects trading hours;
* respects broker constraints.

No research experiment may automatically change Live.

---

# PHASE 29 — PUBLIC EXPLANATION SYSTEM

Estimated code: **1,500–3,000 lines**

Maintain/update:

## Dashboard

* equity;
* weekly performance;
* target comparison;
* holdings;
* reasons;
* ledger;
* learning loop;
* backtest.

## Pattern Explorer

For every active pattern show:

* expression;
* target;
* effect;
* confidence;
* sample size;
* P(real);
* recent performance;
* historical performance;
* context;
* lifecycle;
* validation evidence.

## Sensitivity page

Show:

* parameter;
* tested values;
* effect;
* uncertainty;
* stability;
* selected value;
* reason selected.

Never hide negative experiments.

---

# PHASE 30 — EXPERIMENT MEMORY

Estimated code: **500–1,000 lines**

Every experiment must answer:

### What changed?

### Why did it change?

### What data was used?

### What was unseen?

### What was the baseline?

### What improved?

### What worsened?

### Was improvement statistically meaningful?

### Did risk change?

### Did the improvement survive another window?

### Was it adopted?

### If rejected, why?

Never let an experiment become an orphan.

---

# PHASE 31 — CODE QUALITY FIREWALL

Every major module must have:

* type-safe interfaces where appropriate;
* explicit inputs;
* explicit outputs;
* deterministic behavior;
* logging;
* error handling;
* unit tests;
* integration tests;
* documentation;
* no hidden global state;
* no accidental randomness.

Avoid giant monolithic functions.

Avoid duplicated implementations.

Avoid magical constants.

Every configurable value must have a documented reason.

---

# PHASE 32 — TEST PYRAMID

For every major feature:

## Level 1 — Unit

Test mathematical correctness.

## Level 2 — Integration

Test interaction with neighboring systems.

## Level 3 — Historical

Test on known historical data.

## Level 4 — Walk-forward

Test chronologically.

## Level 5 — Blind

Test disguised hidden windows.

## Level 6 — Adversarial

Try to make it leak.

## Level 7 — Reproducibility

Run twice and compare.

No component should be considered complete before passing the levels appropriate to it.

---

# PHASE 33 — REQUIRED REPRODUCIBILITY

Run the same experiment twice.

Require:

* same configuration hash;
* same seed;
* same candidate ordering;
* same model configuration;
* same predictions;
* same trades;
* same metrics.

If results differ:

find why.

Do not merely add tolerance.

---

# PHASE 34 — REQUIRED ABLATION FRAMEWORK

Every major new Algorithm feature must be tested against:

1. baseline;
2. baseline + feature;
3. feature alone;
4. randomized feature;
5. shuffled feature;
6. future-scrambled feature where applicable.

The feature only earns deployment consideration if it demonstrates incremental out-of-sample value.

---

# PHASE 35 — CHAMPION / CHALLENGER SYSTEM

Never overwrite the current champion merely because a challenger is newer.

Maintain:

* Champion;
* Challenger;
* Candidate;
* Rejected.

A challenger replaces the champion only when:

* required gates pass;
* enough independent evidence exists;
* higher-priority objective is not harmed;
* risk does not violate constraints;
* reproducibility passes;
* blind validation passes.

---

# PHASE 36 — REQUIRED REPORT AFTER EACH MAJOR RUN

Produce a machine-readable and human-readable report containing:

```text
RUN ID
GIT COMMIT
CANON HASH
BLUEPRINT VERSION
CONFIG HASH
DATA SNAPSHOT
RANDOM SEED

WINDOWS
TRAIN PERIOD
VALIDATION PERIOD
TEST PERIOD

TIER 1
weekly average move
weekly median move
share 5–10%
share >10%
share <5%

TIER 2
max drawdown
worst week
5th percentile week
catastrophic losses
turnover
costs

TIER 3
positive in-band percentage
direction accuracy
calibration
8/10 rate

ALGORITHM
pattern count
active pattern count
failed patterns
rescoped patterns
discarded patterns
P(real) distribution
false discovery diagnostics

ANALOGS
analog count
distance distribution
forecast quality

MEMORY
lesson count
accepted lessons
rejected lessons
shock events
memory switches

ADAPTATION
parameter changes
reverts
evidence
confidence

GATES
parity
retester
future scramble
time fence
worker health
reproducibility

DECISION
ADOPT
REJECT
CONTINUE TESTING

REASON
```

---

# PHASE 37 — CHECKLIST STATE MACHINE

Create/maintain:

`state/CHECKLIST.md`

Each item must contain:

```text
[ ] NOT STARTED
[~] IMPLEMENTED / TESTING
[?] UNPROVEN
[x] VALIDATED
[!] FAILED
```

Never mark `[x] VALIDATED` merely because implementation exists.

---

# PHASE 38 — MASTER CHECKLIST

## ALGORITHM

* [x/~] A1 Candles and micro-signals
* [x/~] A2 Pattern miner
* [x/~] A2b Week-clustered statistics
* [ ] A3 Heavy tests across eras
* [x/~] A4 Long-term pattern bank
* [ ] A5 Pattern scores → Find Volatility
* [x/~] A6 Minute collector
* [ ] A7 Self-tuning Algorithm
* [x/~] A8 Market analog engine
* [ ] A8b Sector analog engine
* [ ] A8c Stock analog engine
* [ ] A9 Timeline dial
* [ ] A10 Lesson memory
* [ ] A10b Rerun anti-memorization test
* [x/~] A11 Improve-or-discard lifecycle
* [ ] A12 Additional data sources
* [ ] A13 Planted-pattern calibration

## FIND VOLATILITY

* [ ] V1 95% mover target where sufficient candidates exist
* [ ] V2 Direction ≥80% calibrated confidence
* [ ] V2b Per-stock-type trust tables
* [ ] V3 Exit learner
* [ ] V4 Stop/risk learner
* [ ] V5 8-of-10 +10% objective across blind eras

## TEST

* [x/~] T1 Archive
* [x/~] T2 Blind loop
* [ ] T3 Tiered 7% objective
* [x/~] T4 Weekly self-adjustment
* [x/~] T5 Insider leak protection
* [x/~] T6 13D correctness
* [ ] T7 Explorer
* [ ] T8 Sensitivity
* [ ] T9 Re-tester parity
* [ ] T10 Future scramble
* [ ] T11 Time fence
* [ ] T12 Worker health
* [ ] T13 Reproducibility
* [ ] T14 Label permutation
* [ ] T15 Feature shuffle
* [ ] T16 Ticker permutation
* [ ] T17 Planted-pattern calibration

## LIVE

* [ ] L1 Only consider upgrade after validated research edge
* [ ] L2 Verify paper-only safeguards
* [ ] L3 Verify trading-hours firewall
* [ ] L4 Verify broker safety
* [ ] L5 Verify research/live isolation

---

# PHASE 39 — DEFINITION OF DONE

The project is NOT finished merely because all code is written.

The project reaches "engineering complete" only when:

* [ ] every required module exists;
* [ ] every required module is tested;
* [ ] every data source is validated;
* [ ] every point-in-time rule is enforced;
* [ ] blind simulation works;
* [ ] re-tester works;
* [ ] parity works;
* [ ] future scramble works;
* [ ] time fence works;
* [ ] worker health works;
* [ ] deterministic replay works;
* [ ] pattern discovery works;
* [ ] pattern validation works;
* [ ] pattern lifecycle works;
* [ ] analog engine works;
* [ ] memory works;
* [ ] lesson learning works;
* [ ] anti-memorization test works;
* [ ] direction engine works;
* [ ] exit engine works;
* [ ] stop/risk engine works;
* [ ] per-type trust works;
* [ ] timeline dial exists and is tested;
* [ ] outer training-basis loop works;
* [ ] planted-pattern calibration works;
* [ ] experiment registry works;
* [ ] dashboard works;
* [ ] Pattern Explorer works;
* [ ] Sensitivity page works;
* [ ] documentation reflects actual state;
* [ ] all negative findings are preserved;
* [ ] champion/challenger system works.

And critically:

**If the 7% target is not achieved honestly, the system must report that fact.**

Failure to achieve 7% is not permission to cheat.

Failure to achieve 7% is not permission to redefine 7%.

Failure to achieve 7% is not permission to selectively choose favorable windows.

Failure to achieve 7% is not permission to remove difficult historical periods.

The system's job is to discover the best defensible result.

---

# PHASE 40 — WHAT "KEEP WORKING" MEANS

After each completed phase:

1. run its tests;
2. inspect failures;
3. repair failures;
4. rerun;
5. update checklist;
6. record experiment;
7. identify the next unchecked item;
8. continue.

When a test fails:

**do not stop and ask what to do.**

First investigate it yourself.

Use the repository, tests, logs, experiment history, and specification.

Only request owner input when the decision genuinely changes an owner-level product/canon decision.

Do not ask permission for ordinary implementation decisions.

Do not ask whether you should fix a bug.

Fix it.

Do not ask whether you should add tests.

Add them.

Do not ask whether you should investigate a failed gate.

Investigate it.

Do not ask whether you should continue to the next checklist item.

Continue.

---

# PHASE 41 — NEVER SETTLE FOR A COSMETIC IMPLEMENTATION

Examples of unacceptable completion:

### BAD

"We now have a pattern miner."

If it merely generates patterns but does not statistically validate them.

### BAD

"We now have memory."

If memory is just a dictionary of prior winners.

### BAD

"We now have self-learning."

If parameters are manually selected.

### BAD

"We now have anti-overfitting."

If only a train/test split exists.

### BAD

"We now have analogs."

If nearest neighbors use future normalization.

### BAD

"We now have direction."

If it simply assumes the mover goes up.

### BAD

"We now have risk control."

If the stop can be bypassed by an overnight gap and the system claims guaranteed losses.

### BAD

"We achieved 7%."

If the result came from tuning on the test window.

---

# PHASE 42 — THE STANDARD FOR EVERY CLAIM

Every significant claim must have an evidence trail.

For example:

> "Pattern X is useful."

Must correspond to:

* exact pattern definition;
* discovery sample;
* confirmation sample;
* later sample;
* week-clustered statistics;
* permutation evidence;
* FDR result;
* P(real);
* effect size;
* incremental validation gain;
* regime distribution;
* cost-adjusted result.

Likewise:

> "Setting X is better."

Must include:

* baseline;
* candidate;
* independent windows;
* risk;
* statistical comparison;
* stability;
* adoption reason.

---

# PHASE 43 — PRIORITY WHEN TIME/COMPUTE IS LIMITED

If compute is limited, prioritize:

1. anti-leak correctness;
2. Algorithm;
3. Pattern validation;
4. Pattern → mover integration;
5. Analog validation;
6. Memory;
7. Direction;
8. Exit/stop;
9. outer-loop tuning;
10. dashboard;
11. Live.

Never sacrifice anti-cheating infrastructure to run more experiments.

---

# PHASE 44 — RESOURCE MANAGEMENT

The target environment is approximately:

* Windows PC;
* 8 CPU cores;
* 16 GB RAM.

Use approximately:

* 3–7 workers depending on memory;
* memory-aware scheduling;
* deterministic worker seeds;
* automatic cleanup;
* stale-process detection;
* worker health reports.

Do not launch unlimited parallel jobs.

Do not allow OOM thrashing.

Do not let a worker silently die.

---

# PHASE 45 — CONTINUOUS RESEARCH LOOP

Once the required implementation is complete, do not consider the research finished.

The continuing loop is:

```text
NEW DATA
   ↓
FEATURES
   ↓
PATTERN DISCOVERY
   ↓
STATISTICAL FILTER
   ↓
ANALOG SEARCH
   ↓
MEMORY
   ↓
MOVER MODEL
   ↓
DIRECTION
   ↓
EXIT / STOP
   ↓
BLIND TEST
   ↓
DIAGNOSIS
   ↓
LESSONS
   ↓
ADAPTER
   ↓
OUTER BASIS SEARCH
   ↓
NEW CHALLENGER
   ↓
BLIND VALIDATION
   ↓
CHAMPION OR REJECT
   ↓
MORE DATA
   ↺
```

This loop must be automated wherever possible.

---

# PHASE 46 — FINAL REPORT REQUIREMENT

When the entire checklist has been worked through, produce:

## 1. Implementation report

Every module changed.

## 2. Validation report

Every gate.

## 3. Research report

What genuinely improved.

## 4. Failure report

What did not work.

## 5. Remaining uncertainty report

What is still unproven.

## 6. Champion configuration

Exact configuration and hash.

## 7. Challenger configurations

Exact configurations and evidence.

## 8. Performance distribution

Not just averages.

Include:

* median;
* mean;
* standard deviation;
* percentiles;
* worst;
* best;
* drawdown;
* era breakdown.

## 9. Leakage audit

Explicit pass/fail.

## 10. Reproducibility audit

Explicit pass/fail.

## 11. Checklist

Every item marked honestly.

---

# FINAL OPERATING DIRECTIVE

You are not being asked to make WEEKLY7 look complete.

You are being asked to make WEEKLY7 **actually complete wherever completion is technically possible, and rigorously prove what remains impossible or unproven.**

The central rule is:

> **Implement → test → attack → validate → record → improve → repeat.**

Never:

> implement → assume → declare done.

The system is explicitly intended to learn continuously, but only when evidence justifies learning. The blueprint's own rule is that patterns, settings, and lessons should only be used when they improve out-of-sample results beyond noise; everything else must be recorded and left unused.

Therefore:

**Do not optimize for looking successful.**

**Optimize for becoming correct.**

**Do not stop because the checklist is long.**

**Work the checklist.**

**Do not hide failures.**

**Turn failures into experiments.**

**Do not weaken gates to pass them.**

**Fix the underlying system.**

**Do not use future information.**

**Fail closed.**

**Do not erase history.**

**Preserve every experiment.**

**Do not declare a capability merely because code exists.**

**Prove it.**

And when one phase is complete, immediately move to the next unfinished phase until the entire master checklist has either been genuinely completed and validated or has been explicitly marked `UNPROVEN` with the exact technical reason and the infrastructure required to resolve it later.

# BEGIN NOW

First:

1. Read `CANON.md`.
2. Read the complete WEEKLY7 blueprint.
3. Inspect the entire repository.
4. Establish the baseline.
5. Build the implementation inventory.
6. Identify the first unchecked/highest-priority item.
7. Begin implementation.
8. Run tests.
9. Fix failures.
10. Continue.

Do not give me a high-level plan and stop.

Execute the plan.
