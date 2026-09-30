# WEEKLY7 — ULTIMATE MASTER BUILD, TEST, PERFECT & COMPLETION PROMPT

## MASTER DIRECTIVE

You are now responsible for completing the entire WEEKLY7 system to the owner's intended end state.

This is **not** a normal coding task.

This is not:

> "implement the requested features."

It is:

> **build the complete foundation → prove the entire foundation works → prove it works exactly as intended → scientifically test it under hostile conditions → identify every weakness → fix every weakness → repeatedly retest → perfect the system toward the owner's actual end goal → independently verify completion → do not stop until the master checklist is genuinely complete.**

The owner has explicitly required that the system **not stop short of the intended design**.

Do not interpret "implemented" as "done."

Do not interpret "tests exist" as "works."

Do not interpret "tests pass" as "the intended behavior is correct."

Do not interpret "the code is large" as "the system is intelligent."

Do not interpret "one benchmark passed" as "the system generalizes."

Do not stop because the current codebase is large.

Do not stop because the original contract's line minimums are satisfied.

Do not stop because the unit suite is green.

Do not stop because the system looks architecturally complete.

Do not stop because a subagent reports completion.

Do not stop because the checklist contains checkmarks that are not backed by evidence.

Do not stop because a problem is difficult.

Do not stop because a long experiment is inconvenient.

Do not stop because the current result is disappointing.

Do not lower a requirement because it is difficult to satisfy.

Do not remove a test because the system fails it.

Do not weaken a test to make it pass.

Do not redefine success after seeing the result.

Do not tune against the final holdout.

Do not manufacture evidence.

Do not pad the code.

Do not create duplicate systems where an existing system should be extended.

Do not replace a difficult requirement with an easier approximation without explicit authorization.

The project stops **only when the completion gate at the end of this document passes.**

---

# 1. AUTHORITATIVE PROJECT MATERIAL

Before modifying anything, read and use the existing authoritative project material.

The following remain binding:

* the existing WEEKLY7 blueprint;
* CANON C1–C68;
* C62 Self-Learning Engine contract;
* C64 blind-trader / knowing-curator architecture;
* C65 Live-is-parked rule;
* C66 Autonomous Deep-Research Self-Learning Engine;
* C67 mover-episode research requirement;
* C68 prediction-error → market-change → self-calibration → adaptive-exit addition;
* the existing machine checklists;
* existing tests;
* existing research artifacts;
* existing failure records;
* existing Masterstock;
* existing provenance rules.

The new Real Pattern vs Noise validation system in this prompt is an **addition and integration**, not a replacement.

Never delete or weaken an existing requirement merely because this prompt introduces a newer layer.

---

# 2. MASTER CHECKLIST AUTHORITY

The ultimate checklist is the union of:

### Existing learning system

**C62 — 191-item Self-Learning Engine contract**

### Autonomous research brain

**C66 — 114-item Autonomous Deep-Research contract**

### Mover research

**C67 — hundreds of 5–10% movers/day and what happened next**

### Prediction-error intelligence

**C68 — 64-item prediction-error / market-change / calibration / exit addition**

### New owner-required benchmark

**REAL PATTERN vs NOISE MASTER VALIDATION**

No item from any of these may silently disappear.

The machine-readable checklists remain authoritative:

* `state/build/SELF_LEARNING_MASTER_CHECKLIST.json`
* `state/build/RESEARCH_BRAIN_CHECKLIST.json`
* C68's checklist/provenance artifacts
* corresponding canon/contract files
* the new benchmark checklist created from this prompt

If there is any disagreement between a summary and the actual contract/checklist, inspect the authoritative contract and record the discrepancy rather than guessing.

---

# 3. CURRENT KNOWN STATE — DO NOT ASSUME THESE ARE FINISHED

At the documented snapshot:

### C62

* 191 total items
* 18 validated
* 13 failed
* 146 implemented but not validated
* 7 in progress
* 7 not started

### C66

* 114 total items
* 37 foundation components
* 22 testing items
* 18 scientific-validation items
* 20 autonomous-operation items
* 17 final-gate items
* documented status counts included 85 in progress, 15 not started, 10 validated, 4 implemented
* statuses require independent remapping after the current code wave

### C68

* 64 items
* only 8 were implemented at the documented snapshot
* P01–P05 were being built
* P06 integration was still required
* full C68 testing and independent remapping remained

### C67

The mover-episode research requirement must be fully integrated and validated.

The system must study hundreds of 5–10% movers and what happened afterward, including:

* consolidation;
* increasing volatility;
* stopping;
* spiking;
* reversing;
* continuation;
* other meaningful trajectories.

This must become useful research rather than merely a data-collection script.

---

# 4. CURRENT KNOWN FAILURES / OPEN PROBLEMS

These are not allowed to disappear from the project simply because newer work exists.

Every one must be resolved, scientifically explained, or explicitly measured as an irreducible limitation with the correct UNKNOWN behavior.

Known issues include:

* counterfactual engine incorrectly labels some null seeds POTENTIALLY_PREDICTABLE;
* placebo false positives;
* winner's-curse health-monitor false alarms;
* insufficient contradiction shuffles;
* conjunction pre-screening can lose XOR relationships;
* ladder audit can fire on true signals;
* unguarded infinite transfer ratio;
* reference-context IC false alarms;
* small-bin ECE NaN;
* health-book round-trip failure;
* contradiction-monitor raw IDs triggering identity firewall;
* enforce-mode promotion failing to promote within one window;
* survivor-only price panel;
* year-identifiability through market-context features;
* META_DEFAULT residual;
* numerous untuned thresholds/prior assumptions;
* open future-leak channels;
* registry audit issues;
* planted regime recall weaknesses;
* UNLESS recall weaknesses;
* research stages previously skipping due to missing data;
* quality gate previously lacking sufficient evidence;
* real research loop not yet proven end-to-end;
* research results previously dependent on survivor-only data;
* cross-stock/sector transfer incompletely measured;
* research-policy transfer incompletely measured;
* meta-learning transfer incompletely measured;
* failed-learner usefulness incompletely demonstrated;
* research compute efficiency incompletely measured.

Do not merely mark these "known."

For every failure:

1. reproduce it;
2. isolate the cause;
3. fix the cause;
4. create a regression test;
5. run fresh tests;
6. run adversarial tests;
7. verify that the fix did not weaken another requirement.

---

# 5. GLOBAL FIREWALLS

These rules govern the entire project.

## FIREWALL 1 — FOUNDATION BEFORE VALIDATION

Do not tune the research brain against real results before the complete foundation is built.

The order is:

**BUILD → GENERAL TEST → INTENDED-BEHAVIOR TEST → SCIENTIFIC VALIDATION → PERFECTING**

Never reverse this order merely because an early experiment looks interesting.

---

## FIREWALL 2 — NO FALSE COMPLETION

A checklist item may only become:

**VALIDATED**

when it has:

* implementation;
* relevant tests;
* successful execution;
* evidence artifact;
* provenance;
* correct data flow;
* correct behavior;
* no unresolved blocking defect.

"Code exists" = implemented.

"Unit test exists" = tested.

"Evidence proves intended behavior" = validated.

Do not collapse these states.

---

## FIREWALL 3 — REACHABILITY IS NOT ENOUGH

A module counts as integrated only if:

1. it is reachable;
2. it receives the correct data;
3. it produces the expected output;
4. the output reaches the next stage;
5. downstream behavior actually changes when the module should affect it.

The project's existing AST reachability check must remain.

Add data-flow verification on top of it.

---

## FIREWALL 4 — COMPUTED VERDICTS ONLY

Never manually type:

* PASS;
* VALIDATED;
* IMPROVED;
* CLEAN;
* NO LEAK;
* PROMOTED;
* SUCCESS.

Verdicts must be calculated from artifacts.

Missing evidence = **UNMEASURED**, not PASS.

---

## FIREWALL 5 — NEVER LOWER TESTS

If a test fails:

**fix the system.**

Do not:

* lower the threshold;
* remove the test;
* reduce the test population;
* remove difficult cases;
* change the scoring definition;
* exclude inconvenient seeds;
* hide failed runs.

If the requirement itself is scientifically impossible, preserve the test and report the limitation honestly.

---

## FIREWALL 6 — NO FUTURE INFORMATION

The learner may only use information available at the simulated decision timestamp.

This includes:

* prices;
* fundamentals;
* filings;
* insider information;
* macro information;
* market context;
* pattern memory;
* research results;
* experiment outcomes;
* regime information;
* knowledge;
* failure information.

No future information may enter indirectly.

---

## FIREWALL 7 — BLIND TRADER

The trader side must never know:

* the real year;
* future dates;
* hidden simulation identity;
* truth labels;
* generator configuration;
* planted-pattern IDs;
* noise IDs;
* evaluator answers.

The trusted curator may know historical metadata, but the trader cannot access it.

---

## FIREWALL 8 — ANSWER-KEY SEPARATION

The Real Pattern vs Noise benchmark must have two separate worlds:

### Truth world

Contains:

* real patterns;
* noise patterns;
* generator configuration;
* ground-truth effect;
* exact pattern identity.

### Learner world

Contains only:

* market observations;
* permitted historical information;
* learner memory.

The learner must never access the truth world.

---

## FIREWALL 9 — ANSWER KEY EXISTS FIRST

Every benchmark simulation must follow:

```text
SEED
↓
WORLD GENERATION
↓
TRUTH GENERATION
↓
TRUTH SEAL
↓
MARKET RELEASE
↓
LEARNER
↓
RESULT FREEZE
↓
EVALUATOR
↓
SCORE
```

Never:

```text
learner
↓
inspect result
↓
create answer key
```

The truth artifact must cryptographically prove it existed before learner output.

---

## FIREWALL 10 — FINAL HOLDOUT

Never tune against the final benchmark.

Once the final benchmark is sealed:

* learner frozen;
* configuration frozen;
* thresholds frozen;
* research policy frozen;
* memory state frozen;
* scoring frozen.

A final benchmark failure requires fixing the system using non-final data and creating a **new** final benchmark.

---

# 6. MASTER EXECUTION ORDER

Everything must now be completed in this exact broad sequence.

---

# PHASE 0 — INVENTORY AND BASELINE

### Checklist

* [ ] Read every authoritative contract.
* [ ] Read every machine checklist.
* [ ] Read current Masterstock.
* [ ] Read current research mapping.
* [ ] Read current integration audit.
* [ ] Read current failure/open-problem list.
* [ ] Inspect current Git state.
* [ ] Verify contract hashes.
* [ ] Verify provenance.
* [ ] Verify current test baseline.
* [ ] Verify current line-budget baseline.
* [ ] Verify current reachability baseline.
* [ ] Verify current type-check baseline.
* [ ] Create a frozen starting snapshot.
* [ ] Map every C62 item.
* [ ] Map every C66 item.
* [ ] Map every C67 requirement.
* [ ] Map every C68 item.
* [ ] Map every new benchmark requirement.
* [ ] Identify duplicates.
* [ ] Identify missing functionality.
* [ ] Identify existing systems that must be extended rather than duplicated.

Do not begin optimization here.

---

# PHASE 1 — BUILD THE COMPLETE FOUNDATION

The goal is to make the complete architecture exist before large-scale validation.

## 1A — C62 LEARNING FOUNDATION

Complete and validate the implementation of the entire C62 architecture, including every outstanding:

* knowledge object;
* epistemic state;
* interpretation;
* situation representation;
* similarity;
* retrieval;
* failure learning;
* missed-winner/missed-loser learning;
* credit assignment;
* redundancy/disagreement;
* belief;
* questions;
* competition;
* complexity;
* boundary detection;
* retirement;
* temporal memory;
* surprise;
* calibration;
* experiment memory;
* research policy;
* meta-learning;
* failed-learner memory;
* transfer;
* transfer score;
* portfolio value;
* scorecard;
* learning curve;
* promotion;
* champion/challenger;
* compute allocation;
* checkpoints;
* firewalls;
* future firewall;
* identity firewall;
* blind trader;
* curator;
* research/live separation;
* health;
* contradiction handling;
* provenance;
* deterministic replay.

Every remaining C62 item must be mapped to implementation and tests.

---

# 1B — C66 RESEARCH FOUNDATION

Complete all 37 C66 foundation components:

* [ ] RF01 Research priority engine
* [ ] RF02 Autonomous research scheduler
* [ ] RF03 Market-wide observation
* [ ] RF04 Winner research
* [ ] RF05 Loser research
* [ ] RF06 Missed-winner research
* [ ] RF07 Missed-loser research
* [ ] RF08 Knowability engine
* [ ] RF09 Counterfactual information reconstruction
* [ ] RF10 Volatility laboratory
* [ ] RF11 Direction laboratory
* [ ] RF12 Conditional accuracy frontier
* [ ] RF13 Pattern discovery expansion
* [ ] RF14 Pattern-break research
* [ ] RF15 Experiment memory
* [ ] RF16 Meta-learning
* [ ] RF17 Failed-learner registry
* [ ] RF18 Compute manager
* [ ] RF19 Experiment-value accounting
* [ ] RF20 Compute-waste controller
* [ ] RF21 Daily autopsy
* [ ] RF22 Daily research generator
* [ ] RF23 Multi-scale research
* [ ] RF24 Cross-sectional research
* [ ] RF25 Regime research
* [ ] RF26 Interaction discovery
* [ ] RF27 Knowledge graph
* [ ] RF28 Knowledge-to-decision bridge
* [ ] RF29 Research/trader firewall
* [ ] RF30 Replication system
* [ ] RF31 Unknown-cause system
* [ ] RF32 Research-health system
* [ ] RF33 Research-diversity system
* [ ] RF34 Scientific long-term memory
* [ ] RF35 Question generator
* [ ] RF36 Hypothesis tree
* [ ] RF37 Promotion quality gate

These correspond to the existing C66 foundation list.

---

# 1C — C67 MOVER-EPISODE FOUNDATION

Build the complete always-on mover research system.

For hundreds of stocks per day, study stocks entering the 5–10% volatility range and record what happens next.

Study:

* consolidation;
* continuation;
* acceleration;
* spike;
* reversal;
* collapse;
* stagnation;
* gap behavior;
* intraday path;
* close-to-close;
* open-to-close;
* range;
* close location;
* next-day behavior;
* subsequent-week behavior.

The system must search for even very small predictable patterns.

It must operate continuously through the research brain rather than becoming a disconnected report generator.

---

# 1D — C68 PREDICTION-ERROR FOUNDATION

Integrate the entire C68 addition.

## Immutable expectations

Before every prediction, permanently capture:

* expected return;
* expected distribution;
* direction;
* confidence;
* volatility;
* time-to-peak;
* best exit window;
* MFE;
* MAE;
* regime;
* sector regime;
* relevant patterns;
* pattern strength;
* interactions;
* holding period;
* probability distribution;
* selection reasoning;
* rejected alternatives;
* uncertainty;
* model version;
* feature state;
* knowledge state;
* timestamp;
* legal information set.

## Outcome reconstruction

After maturity, reconstruct:

* actual path;
* return;
* MFE;
* MAE;
* volatility;
* direction;
* timing;
* market movement;
* sector movement;
* peer movement;
* pattern behavior;
* regime;
* event indicators;
* best historical exit opportunities.

## Error engine

Measure separately:

* return error;
* direction error;
* volatility error;
* timing error;
* exit error;
* confidence error;
* market-regime error;
* sector-regime error;
* pattern-strength error;
* interaction error.

Never collapse these into one generic error number.

## Error research

Implement:

* error magnitude prioritization;
* confident-wrong investigation;
* repeated-error escalation;
* self-generated research questions;
* value-aware compute allocation.

## Market-change intelligence

Implement:

* expectation vs reality;
* contraction detection;
* change points;
* early regime warning;
* regime memory;
* no-retrospective-cheating protection.

## Pattern-change intelligence

Implement:

* what-changed tree;
* per-pattern change detection;
* knowability classification;
* precursor discovery;
* forward validation.

## Selection / exit intelligence

Implement:

* 5–10% selection constraint;
* anti-gaming protection;
* exit learning;
* exit independence from the ±1pp calibration target;
* honest calibration measurement;
* self-correction.

---

# 1E — COMPLETE INTEGRATION

Every research module must connect to the actual production learning/research pathway.

The complete chain must operate:

```text
market data
↓
point-in-time state
↓
market-wide observation
↓
autopsy
↓
winner/loser study
↓
missed-winner/missed-loser study
↓
knowability
↓
counterfactual reconstruction
↓
pattern-break research
↓
volatility research
↓
direction research
↓
frontier
↓
symmetry
↓
pattern discovery
↓
interaction discovery
↓
multi-scale research
↓
cross-sectional research
↓
regime research
↓
episode research
↓
question generation
↓
hypothesis tree
↓
research priority
↓
experiments
↓
failed-learner research
↓
meta-learning
↓
compute allocation
↓
replication
↓
quality gate
↓
research graph
↓
decision bridge
↓
scientific memory
↓
brain health
↓
research diversity
↓
research/trader firewall
↓
curator
↓
decision
↓
outcome
↓
prediction error
↓
new research question
```

No stage may silently skip because data is missing.

If input is unavailable:

> record `SKIPPED_MISSING_MODULE` or `SKIPPED_NO_INPUT`.

Never treat skipped work as successful work.

Then build the missing input.

The final rich planted-world run must demonstrate **zero skipped stages**.

---

# PHASE 2 — GENERAL FUNCTIONAL TESTING

Only begin this phase after the entire foundation is built.

This phase asks:

> **Does the machinery actually work at all?**

Not yet:

> Is it scientifically excellent?

Test every module independently.

---

## 2A — UNIT TESTS

Complete every C62/C66/C68 unit test.

Test:

* normal inputs;
* empty inputs;
* malformed inputs;
* boundary inputs;
* duplicate inputs;
* missing data;
* contradictory data;
* extreme values;
* NaN;
* infinity;
* tiny sample;
* huge sample;
* deterministic replay;
* serialization;
* restoration;
* checkpoints;
* crash recovery.

---

## 2B — INTEGRATION TESTS

Prove:

* modules call one another;
* data flows correctly;
* outputs are consumed;
* state persists correctly;
* research results reach the correct destination;
* firewalls actually block prohibited flows;
* learner state changes only when it should;
* promotion requires evidence.

---

## 2C — FULL SYSTEM SMOKE TEST

Run a small complete world.

Prove:

* all stages execute;
* no stage silently disappears;
* no data is lost;
* no stage consumes future information;
* research questions are generated;
* experiments execute;
* evidence is stored;
* knowledge is promoted only when permitted;
* outcomes feed back into learning;
* checkpoints work;
* recovery works.

---

# PHASE 3 — EXACT-INTENDED-BEHAVIOR TESTING

Now ask:

> **Does the system do exactly what the owner intended, rather than merely functioning technically?**

This is where the new benchmark becomes central.

---

# 3A — REAL PATTERN vs NOISE WORLD GENERATOR

Create the benchmark generator.

Every simulation must contain by default:

> **~50 genuine patterns**

and:

> **~500 noise processes**

with configurable ranges.

The genuine patterns must span:

* obvious;
* moderate;
* subtle;
* rare;
* conditional;
* interactive;
* delayed;
* regime-dependent;
* changing;
* extremely subtle.

Noise must span:

* pure randomness;
* short-lived coincidence;
* multiple-testing noise;
* autocorrelation traps;
* regime-correlated noise;
* volatility-correlated noise;
* sample-size traps;
* threshold illusions;
* near-pattern noise;
* historical coincidence;
* reversal noise;
* delayed coincidence;
* interaction decoys;
* XOR-like traps;
* selection bias;
* survivor bias;
* persistent weak noise;
* strong nontransferable noise;
* adversarial near-signal noise.

---

# 3B — PATTERN DISCOVERY TEST

The learner must:

1. search broadly;
2. identify potential patterns;
3. normalize them;
4. deduplicate equivalent hypotheses;
5. collect evidence;
6. classify them;
7. estimate confidence;
8. decide whether to promote, quarantine, reject, or remain UNKNOWN.

The evaluator must separately measure:

* discovery recall;
* discovery precision;
* noise rejection;
* false-discovery rate;
* false-negative rate;
* calibration.

---

# 3C — ANSWER-KEY FIREWALL

For every simulation:

1. generate seed;
2. generate truth;
3. seal truth;
4. release observations;
5. run learner;
6. freeze learner result;
7. score against truth.

The truth must be created before learner output.

The learner cannot know:

* number of real patterns;
* number of noise patterns;
* pattern IDs;
* truth labels;
* generator parameters.

---

# 3D — PARTIAL MATCH EVALUATION

A discovery is not merely RIGHT/WRONG.

Measure:

* feature overlap;
* condition similarity;
* threshold accuracy;
* lag accuracy;
* effect direction;
* effect magnitude;
* regime accuracy;
* transfer accuracy.

A close discovery receives appropriate partial credit.

A superficially similar but fundamentally wrong pattern does not.

---

# 3E — NULL WORLDS

Create worlds containing:

> **zero genuine patterns**

but hundreds of convincing noise relationships.

The learner must not manufacture signal.

Correct behavior may be:

> UNKNOWN / NO RELIABLE SIGNAL.

---

# 3F — DIFFICULTY CURVE

Create:

* Tier 1 obvious;
* Tier 2 moderate;
* Tier 3 subtle;
* Tier 4 adversarial;
* Tier 5 highly conditional;
* Tier 6 maximum difficulty.

---

# 3G — PATTERN LIFECYCLE TESTS

Test:

* pattern appears;
* pattern strengthens;
* pattern weakens;
* pattern dies;
* pattern reverses;
* pattern returns.

The learner must preserve history while changing present influence.

---

# 3H — UNKNOWN-CAUSE TEST

Some patterns must fail for genuinely unknowable reasons.

Correct answer:

> UNKNOWN.

The system must not invent retrospective explanations.

---

# 3I — INTERACTION TEST

Test:

```text
A alone = no signal
B alone = no signal
A+B = real signal
```

and XOR-like relationships.

---

# 3J — ADVERSARIAL NOISE

If the system is especially vulnerable to a noise family, create harder versions.

But these adversarial examples must not become the final benchmark.

---

# 3K — CROSS-YEAR / CROSS-ERA TEST

The system must work regardless of the year.

It must not recognize the year.

It must transfer useful knowledge across eras while recognizing conditional patterns.

---

# 3L — CROSS-STOCK TEST

A pattern learned on one stock must not be assumed valid elsewhere.

Test:

* same stock;
* new stock;
* same sector;
* new sector;
* different market conditions.

---

# 3M — MEMORIZATION TEST

Rename:

* tickers;
* dates;
* simulation IDs;
* file names;
* pattern IDs.

Shift dates where possible.

The behavior should remain invariant.

---

# 3N — GENERATOR MUTATION

After the learner becomes good at one generator:

change:

* seeds;
* feature correlations;
* noise distributions;
* pattern strength;
* lag;
* regime duration;
* pattern frequency;
* market distributions;
* generator structure.

The learner must continue working.

---

# PHASE 4 — SCIENTIFIC VALIDATION

Now test whether the complete system is actually useful.

The existing C66 scientific-validation checklist must all be completed:

* [ ] volatility prediction measured OOS
* [ ] volatility calibration measured
* [ ] volatility coverage measured
* [ ] predictable vs unknowable classification evaluated
* [ ] direction evaluated only on predicted movers
* [ ] direction accuracy measured
* [ ] direction calibration measured
* [ ] direction coverage frontier measured
* [ ] 80% question evaluated honestly
* [ ] loss reduction measured
* [ ] worst-case behavior measured
* [ ] tail risk measured
* [ ] pattern transfer measured
* [ ] pattern-break prediction measured
* [ ] research-policy transfer measured
* [ ] meta-learning transfer measured
* [ ] failed-learner memory usefulness demonstrated
* [ ] research compute efficiency measured

These are the actual C66 scientific gates, not optional extras.

---

# PHASE 5 — LEARNING VALIDATION

Now determine whether the system actually learns.

Run:

* no-learning control;
* transfer control;
* memorizer control;
* random control;
* leaky control where appropriate;
* identity control;
* same-year control;
* cross-year control;
* cross-stock control;
* cross-sector control;
* cross-regime control.

The key measurement is:

> **Does previous experience improve performance on genuinely new situations?**

Not:

> Does the system get better at the situations it already saw?

---

# PHASE 6 — LARGE-SCALE PATTERN BENCHMARK

Now run the complete benchmark.

Minimum main benchmark:

> **500 simulations**

Each simulation:

> **hundreds of noise processes**

and:

> **dozens of genuine patterns**

Target default:

> **~50 genuine + ~500 noise per simulation**

This produces approximately:

* 25,000 genuine patterns;
* 250,000 noise processes.

Then run a larger independent benchmark if resources allow.

---

# PHASE 7 — LEARNING-CURVE VALIDATION

Measure performance after:

* 1 simulation;
* 5;
* 10;
* 25;
* 50;
* 100;
* 250;
* 500.

Track:

* genuine recall;
* noise rejection;
* precision;
* false-discovery rate;
* calibration;
* partial-match score;
* cross-world transfer.

The important curve is:

> **performance on unseen worlds as historical experience increases.**

If only previously seen worlds improve, that is not sufficient evidence of learning.

---

# PHASE 8 — ADVERSARIAL GENERALIZATION

Now deliberately attack the system.

Test:

* extreme noise;
* null worlds;
* subtle signals;
* misleading regimes;
* changing regimes;
* pattern reversals;
* strong false correlations;
* multiple-testing traps;
* interaction traps;
* near-identical real/noise relationships;
* unseen generator configurations.

The system must remain calibrated.

It must not become more confident merely because the environment became harder.

---

# PHASE 9 — C68 PREDICTION-ERROR VALIDATION

Prove that the system can answer:

> **"I expected X. Reality produced Y. Why was I wrong, how much was predictable, what changed, and what should I do differently?"**

Test:

* immutable expectation;
* complete outcome reconstruction;
* separated error dimensions;
* error magnitude prioritization;
* confident-wrong detection;
* repeated-error escalation;
* market change detection;
* regime change;
* pattern change;
* knowability;
* self-research;
* research priority;
* adaptive calibration;
* exit learning;
* self-correction.

No retrospective information may leak into the supposed prediction-time state.

---

# PHASE 10 — PERFECTING

This phase does not begin until the complete system has been shown to work generally and according to the intended design.

Now improve it.

For every weakness:

```text
FIND
↓
CLASSIFY
↓
UNDERSTAND
↓
FIX
↓
TEST
↓
ADVERSARIAL TEST
↓
FRESH HOLDOUT
↓
RECHECK OTHER SYSTEMS
```

Repeat until no meaningful unresolved weakness remains.

---

# PHASE 11 — PRODUCT END-GOAL

The final product should be the system originally envisioned:

## It should continuously research.

Not occasionally.

Not only when manually instructed.

It should continually identify valuable research questions.

## It should continuously discover patterns.

Including small, subtle, conditional and interacting patterns.

## It should distinguish signal from noise.

It must not confuse statistical coincidence with transferable knowledge.

## It should learn from failures.

Especially:

* confident failures;
* repeated failures;
* missed winners;
* missed losers;
* regime changes;
* pattern breaks.

## It should know when it does not know.

UNKNOWN must be a real state.

## It should know when old knowledge has degraded.

Broken patterns must be gated rather than blindly trusted.

## It should know when old knowledge has returned.

Recovered patterns should be capable of regaining influence when evidence supports it.

## It should learn how to research.

It should improve:

* what questions it asks;
* what experiments it runs;
* where it spends compute;
* which research methods are productive;
* which research methods waste resources.

## It should learn how to learn.

The research brain itself should improve.

---

# PHASE 12 — AUTONOMOUS OPERATION GATE

The C66 autonomous-operation requirements must all be demonstrated:

* [ ] system generates research questions;
* [ ] ranks questions;
* [ ] allocates compute;
* [ ] runs experiments;
* [ ] records results;
* [ ] remembers failures;
* [ ] remembers successes;
* [ ] studies winners;
* [ ] studies losers;
* [ ] studies missed winners;
* [ ] studies missed losers;
* [ ] investigates pattern breaks;
* [ ] identifies unknown causes;
* [ ] promotes validated knowledge;
* [ ] downgrades broken knowledge;
* [ ] detects wasted research;
* [ ] redirects compute;
* [ ] learns which research methods work;
* [ ] continually generates the next research agenda;
* [ ] runs indefinitely with checkpoint/recovery.

These are directly part of the existing C66 master checklist.

---

# PHASE 13 — REAL-DATA SCIENTIFIC REVALIDATION

After synthetic testing is complete, run the research architecture against real historical data.

But:

**real data does not replace the synthetic benchmark.**

The real-data system must remain:

* point-in-time;
* walk-forward;
* leak-free;
* provenance tracked;
* independently reproducible.

The known survivor-only problem must be fixed or conclusions must remain explicitly qualified.

All previously validated scientific results that depended on survivor-only data must be re-measured with the corrected universe.

---

# PHASE 14 — FINAL INDEPENDENT AUDIT

Before completion, a separate audit must inspect:

* code;
* tests;
* architecture;
* data flow;
* provenance;
* future-leak paths;
* identity paths;
* year paths;
* memorization;
* benchmark isolation;
* truth-key isolation;
* deterministic replay;
* scientific reports;
* promotion evidence;
* holdout integrity;
* checklist status.

The person/process performing the audit must not simply trust the builder's completion report.

---

# PHASE 15 — FINAL HOLDOUT

Generate a completely fresh final benchmark.

Requirements:

* fresh seeds;
* fresh worlds;
* fresh pattern parameters;
* fresh noise parameters;
* unseen generator configuration;
* unseen combinations;
* no tuning afterward.

Run the frozen system.

Then score it.

Do not change the system after seeing the final result.

---

# PHASE 16 — FINAL C62/C66/C67/C68 RE-MAPPING

Run an independent mapping.

Every item must resolve to exactly one state:

* VALIDATED;
* FAILED;
* INSUFFICIENT EVIDENCE;
* UNMEASURED.

There must be no vague:

> "basically done."

Every VALIDATED item needs evidence.

Every FAILED item needs either a fix or an explicit documented scientific limitation.

Every INSUFFICIENT EVIDENCE item blocks completion.

Every UNMEASURED item blocks completion.

---

# PHASE 17 — FINAL COMPLETION GATE

The project is **NOT COMPLETE** unless every applicable condition below passes.

## FOUNDATION

* [ ] C62 complete
* [ ] C66 foundation complete
* [ ] C67 integrated
* [ ] C68 complete
* [ ] all integrations reachable
* [ ] all required data flows proven
* [ ] no duplicate architecture where extension is required
* [ ] line-budget requirements satisfied without padding
* [ ] type safety clean
* [ ] full unit suite passes

## GENERAL FUNCTION

* [ ] complete system executes
* [ ] checkpoints work
* [ ] recovery works
* [ ] deterministic replay works
* [ ] every stage executes
* [ ] no silent skipped stages
* [ ] missing data is explicitly represented
* [ ] provenance works
* [ ] reports work

## INTENDED BEHAVIOR

* [ ] blind trader works
* [ ] knowing curator works
* [ ] no year leakage
* [ ] no identity memorization
* [ ] no future information
* [ ] research/trader firewall works
* [ ] pattern lifecycle works
* [ ] UNKNOWN works
* [ ] broken-pattern gating works
* [ ] recovered-pattern behavior works
* [ ] prediction-error pipeline works
* [ ] autonomous research loop works

## PATTERN/NOISE BENCHMARK

* [ ] answer key generated before learner output
* [ ] answer key cryptographically sealed
* [ ] learner cannot access truth
* [ ] hundreds of noise per simulation
* [ ] dozens of genuine patterns per simulation
* [ ] subtle genuine patterns
* [ ] convincing noise
* [ ] null worlds
* [ ] conditional patterns
* [ ] interaction patterns
* [ ] changing patterns
* [ ] reversed patterns
* [ ] adversarial noise
* [ ] candidate discovery measured
* [ ] genuine recall measured
* [ ] noise rejection measured
* [ ] false-discovery rate measured
* [ ] calibration measured
* [ ] partial-match quality measured
* [ ] cross-year transfer measured
* [ ] cross-era transfer measured
* [ ] cross-stock transfer measured
* [ ] cross-sector transfer measured
* [ ] cross-regime transfer measured
* [ ] unseen-generator transfer measured
* [ ] 500+ simulation benchmark completed
* [ ] learning curve demonstrated
* [ ] unseen-world improvement demonstrated

## SCIENTIFIC VALIDATION

* [ ] volatility OOS validated
* [ ] direction OOS validated
* [ ] predicted-mover universe respected
* [ ] calibration validated
* [ ] coverage measured
* [ ] 80% question answered honestly
* [ ] loss reduction measured
* [ ] worst-case behavior measured
* [ ] tail risk measured
* [ ] pattern transfer measured
* [ ] pattern-break prediction measured
* [ ] research-policy transfer measured
* [ ] meta-learning transfer measured
* [ ] failed-learner memory usefulness measured
* [ ] compute efficiency measured

## C68

* [ ] immutable expectation
* [ ] complete outcome reconstruction
* [ ] separated error dimensions
* [ ] confident-wrong investigation
* [ ] repeated-error escalation
* [ ] market-change detection
* [ ] regime-change detection
* [ ] pattern-change detection
* [ ] knowability classification
* [ ] self-research
* [ ] adaptive calibration
* [ ] exit learning
* [ ] self-correction
* [ ] all C68 adversarial tests

## ANTI-CHEATING

* [ ] no future path
* [ ] no hidden year
* [ ] no identity memorization
* [ ] no simulation-ID memorization
* [ ] no answer-key access
* [ ] no generator leakage
* [ ] no survivor-only conclusions
* [ ] no lucky single experiment
* [ ] no final-holdout tuning
* [ ] no benchmark manipulation
* [ ] no manual verdicts
* [ ] no padded implementation
* [ ] no weakened tests

## FINAL EVIDENCE

* [ ] every promoted item has provenance
* [ ] every scientific claim has OOS evidence
* [ ] every improvement claim has a control
* [ ] every learning claim has an unseen-world test
* [ ] every major result has replication
* [ ] independent audit passes
* [ ] Masterstock complete
* [ ] all contracts remain intact
* [ ] all hashes verify
* [ ] final holdout remains sealed until frozen evaluation
* [ ] final checklist is machine-verifiable

---

# 18. WHAT "PERFECT" MEANS

Do not interpret "perfect" as:

> every random simulation must produce 100% classification accuracy.

That would reward false certainty.

Instead:

> **The system must become as accurate as the evidence permits while remaining calibrated and scientifically honest.**

If the truth is genuinely distinguishable:

> find it.

If the truth is subtle but discoverable:

> learn to find it.

If the noise is convincing but distinguishable with sufficient evidence:

> reject it.

If the evidence is genuinely insufficient:

> UNKNOWN.

If a pattern worked historically but no longer works:

> detect the degradation and gate it.

If a pattern returns:

> recognize the evidence and restore appropriate influence.

The objective is not forced certainty.

The objective is **maximum transferable intelligence with minimum false discovery and honest uncertainty.**

---

# 19. NON-STOP EXECUTION RULE

You are not finished because one phase is finished.

When Phase N finishes:

1. verify Phase N;
2. check whether any prerequisite remains incomplete;
3. proceed immediately to the next incomplete phase;
4. continue until the entire master checklist passes.

If a test fails:

> enter repair loop.

If a builder fails:

> inspect its artifacts and resume/reassign the work.

If a process stops:

> restart from the saved state.

If a module is incomplete:

> complete it.

If a requirement is ambiguous:

> consult the authoritative contract, canon, Masterstock and owner directives before making assumptions.

Do not stop and ask the owner whether you should continue when the checklist already tells you what to do.

Do not substitute a progress report for actual completion.

Do not stop merely because the current session has been running a long time.

Do not declare completion until the machine-verifiable final gate passes.

---

# 20. BUILDER RULES

Use:

* Sonnet for normal implementation;
* Opus for integration;
* Opus for firewalls;
* Opus for anti-cheating;
* Opus for scientific judgement;
* Opus for final audit.

Normal builders must:

* read their assigned contract;
* inspect existing implementation;
* extend rather than duplicate;
* write tests;
* prove reachability;
* prove data flow;
* report exact evidence;
* commit their work.

Integration/judgement builders must independently inspect the work rather than trusting builder claims.

---

# 21. GIT / PROCESS SAFETY

While builders are working:

* commit;
* fetch;
* merge.

Do not:

* rebase;
* stash;
* reset;
* overwrite another builder's work;
* kill processes by name.

Preserve evidence.

Never allow an update to erase evidence earned by another process.

---

# 22. REPORTING REQUIREMENT

At the end of every major phase, produce:

### What was supposed to happen

### What actually happened

### Tests run

### Tests passed

### Tests failed

### Known limitations

### Evidence artifacts

### Checklist changes

### Remaining blockers

### Next unfinished checklist items

Do not write:

> "Everything looks good."

Use measurable evidence.

---

# 23. FINAL PRODUCT DEFINITION

The finished WEEKLY7 system should be a continuously improving research organism that:

```text
OBSERVES
↓
PREDICTS
↓
RECORDS EXPECTATIONS
↓
OBSERVES REALITY
↓
MEASURES ERROR
↓
IDENTIFIES SURPRISE
↓
STUDIES WINNERS
↓
STUDIES LOSERS
↓
STUDIES MISSED OPPORTUNITIES
↓
DISCOVERS PATTERNS
↓
DISTINGUISHES PATTERNS FROM NOISE
↓
TESTS PATTERNS
↓
TESTS BREAKS
↓
IDENTIFIES REGIMES
↓
IDENTIFIES KNOWABLE VS UNKNOWN
↓
GENERATES QUESTIONS
↓
ALLOCATES RESEARCH
↓
RUNS EXPERIMENTS
↓
REPLICATES
↓
PROMOTES VALIDATED KNOWLEDGE
↓
GATES BROKEN KNOWLEDGE
↓
MEASURES WHETHER LEARNING TRANSFERS
↓
LEARNS WHICH RESEARCH METHODS WORK
↓
IMPROVES ITS OWN RESEARCH PROCESS
↓
REPEATS
```

The ultimate objective is:

> **The system should become better at discovering what is genuinely predictable, better at rejecting what only looks predictable, better at recognizing when old knowledge stops working, better at discovering why it failed, and better at deciding what to investigate next — without ever cheating by seeing information it would not have had at the time.**

---

# 24. ABSOLUTE STOP CONDITION

There is exactly one legitimate reason to stop the master build:

> **The complete master checklist has been independently verified as complete and the final gate has passed.**

Not:

* "most of it is done."
* "the foundation is basically done."
* "all important parts are done."
* "the tests mostly pass."
* "the benchmark is pretty good."
* "we reached the line count."
* "the model looks intelligent."
* "the remaining items are minor."
* "we can finish them later."

Only:

> **EVERY REQUIRED ITEM → IMPLEMENTED → TESTED → INTENDED BEHAVIOR VERIFIED → SCIENTIFICALLY VALIDATED WHERE APPLICABLE → INTEGRATED → INDEPENDENTLY AUDITED.**

Until then:

# DO NOT STOP.

# DO NOT DECLARE COMPLETE.

# DO NOT MOVE TO DEPLOYMENT.

# KEEP WORKING THE NEXT UNFINISHED CHECKLIST ITEM.

And when the entire checklist finally passes, produce the final completion report containing:

1. complete checklist;
2. evidence for every validated item;
3. all remaining scientific limitations;
4. final benchmark results;
5. unseen-world learning curve;
6. final anti-cheating audit;
7. final provenance audit;
8. final independent audit;
9. final holdout result;
10. exact statement of what has and has not been demonstrated.

Only then is WEEKLY7 considered complete.
