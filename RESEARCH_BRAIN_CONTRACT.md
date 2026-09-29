# WEEKLY7 — AUTONOMOUS DEEP-RESEARCH SELF-LEARNING ENGINE

## MASTER BUILD PROMPT — VERSION 1.0

You are continuing development of WEEKLY7, an existing self-learning stock research and prediction system.

This is NOT a request to build a normal stock screener.

This is NOT a request to simply add more indicators.

This is NOT a request to endlessly tune parameters.

This is NOT a request to maximize backtest performance.

You are building the next layer of WEEKLY7: an **autonomous internal research intelligence that continuously studies the market, studies its own learning process, identifies the highest-value unanswered questions, performs experiments, learns from both winners and losers, and continually improves the underlying prediction system without access to future information.**

The existing Weekly7 architecture, Self-Learning Contract C62, canon C1–C65, Masterstock, existing tests, firewalls, archives, learner, research system, pattern memory, transfer harness, failure learning, and existing code are authoritative.

**DO NOT replace them.**
**DO NOT restart the project.**
**DO NOT duplicate existing systems unnecessarily.**
**EXTEND AND INTEGRATE what already exists.**

---

# 0. ABSOLUTE OBJECTIVE

The final system has two primary prediction problems.

## OBJECTIVE 1 — VOLATILITY DISCOVERY

Given only information that was genuinely available at a particular point in time, identify stocks that are likely to experience a large move during the relevant future trading period.

The system should learn:

* which stocks are likely to become unusually volatile;
* how large the movement may be;
* when the movement is likely to occur;
* which patterns precede volatility;
* which patterns merely correlate with volatility;
* which apparent volatility signals are caused by external events;
* which volatility signals generalize across stocks;
* which volatility signals generalize across sectors;
* which volatility signals generalize across market regimes;
* which volatility signals decay;
* which volatility signals suddenly stop working;
* which signals are dangerous because they identify volatility without identifying direction.

The volatility problem comes FIRST.

Do not spend enormous computational resources attempting to solve direction on stocks that cannot first be identified as meaningful movers.

---

# OBJECTIVE 2 — DIRECTION AMONG PREDICTED MOVERS

After Objective 1 becomes sufficiently reliable, determine whether a predicted volatile stock is more likely to finish positively or negatively over the defined decision horizon.

The long-term target is:

> Among stocks that the system identifies as genuinely predictable volatile opportunities, develop a sufficiently calibrated directional selection process that can eventually approach an 80% positive-direction rate at a meaningful coverage level, while maintaining statistical validity, transferability, and controlled downside.

Do NOT interpret this as permission to force an 80% result.

If 80% does not exist in the available information, the system must discover that honestly.

The system must be able to say:

```text
No reliable 80% directional region has been found.
```

rather than manufacturing one through:

* overfitting;
* tiny samples;
* excessive filtering;
* repeated testing;
* stock identity memorization;
* future information;
* survivorship bias;
* post-event information;
* leakage;
* cherry-picking;
* threshold manipulation;
* selection of only the most convenient historical periods.

The existing research has already found that movement prediction has demonstrated signal while direction has so far been close to chance, so the architecture must explicitly treat these as two different scientific problems rather than assuming direction will automatically follow volatility.

---

# 1. THE SYSTEM MUST BE AN AUTONOMOUS RESEARCHER

WEEKLY7 should behave conceptually like an internal AI research organization.

It continuously asks:

```text
What do I currently know?

How reliable is that knowledge?

What am I uncertain about?

What have I repeatedly failed at?

What patterns have recently broken?

What winners did I miss?

What losers did I fail to avoid?

What apparently successful ideas were actually memorization?

What research questions remain unanswered?

Which unanswered question could most improve the actual decision system?

What experiment would answer it?

How much compute should that experiment receive?

What would count as success?

What would falsify the hypothesis?

What should I learn from the result?

What should I research next?
```

The system must NOT simply run a fixed sequence forever.

It must maintain a continuously evolving research queue.

---

# 2. RESEARCH PRIORITY IS MORE IMPORTANT THAN RESEARCH VOLUME

A critical design requirement:

**The system must understand the difference between doing a huge amount of work and accomplishing something important.**

A 10,000-hour experiment that improves performance by 0.1 basis points is not automatically more valuable than a 30-minute experiment that discovers a new robust volatility pattern.

Every experiment must have an estimated:

```text
expected_information_gain
expected_decision_value
uncertainty_reduction
transfer_potential
failure_reduction_value
volatility_value
direction_value
compute_cost
overfit_risk
redundancy
```

The research scheduler must prioritize experiments according to expected value, not novelty.

Conceptually:

```text
research_priority =
    information_gain
    × decision_relevance
    × uncertainty
    × transfer_potential
    × expected_loss_reduction
    × expected_volatility_value
    × expected_direction_value
    × feasibility
    /
    (
        compute_cost
        × redundancy
        × overfit_risk
    )
```

This formula is conceptual.

The system must learn and validate its own research-priority model rather than assuming this exact formula is optimal.

Existing Weekly7 already defines information-gain research and meta-learning requirements; this extension must make those systems operationally dominant rather than allowing the learner to spend most compute on parameter tuning.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

Do not pad lines.

If the implementation is below 1,500 meaningful lines, it is probably missing major research-allocation functionality.

---

# 3. BUILD A PERMANENT RESEARCH LOOP

Create a persistent autonomous research loop.

Conceptually:

```text
OBSERVE
↓
UPDATE KNOWLEDGE
↓
EVALUATE RECENT RESULTS
↓
IDENTIFY SURPRISES
↓
IDENTIFY FAILURES
↓
IDENTIFY MISSED WINNERS
↓
IDENTIFY MISSED LOSERS
↓
IDENTIFY PATTERN BREAKS
↓
GENERATE QUESTIONS
↓
GENERATE COMPETING HYPOTHESES
↓
ESTIMATE INFORMATION GAIN
↓
ALLOCATE COMPUTE
↓
RUN EXPERIMENT
↓
VALIDATE
↓
COMPARE AGAINST CONTROLS
↓
UPDATE KNOWLEDGE
↓
UPDATE RESEARCH PRIORITIES
↓
REPEAT
```

The loop must be able to operate indefinitely.

It must support:

* checkpoints;
* resumability;
* crash recovery;
* experiment isolation;
* reproducibility;
* compute budgeting;
* priority queues;
* experiment cancellation;
* experiment escalation;
* experiment demotion;
* experiment duplication detection;
* stale-hypothesis detection;
* research lineage;
* research reports.

A long experiment must not block all other learning.

Use concurrent research jobs where safe.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 4. BUILD A MARKET-WIDE DAILY RESEARCH OBSERVER

The system must not study only the stocks it selected.

This is extremely important.

Every simulated trading day, the research system should inspect the eligible market universe and construct a complete research record containing:

### A. Stocks the model considered

Including:

* high-ranked volatility candidates;
* medium-ranked candidates;
* rejected candidates;
* abstentions;
* low-confidence candidates.

### B. Actual daily winners

Within the available universe.

### C. Actual daily losers

Within the available universe.

### D. Extreme movers

Both positive and negative.

### E. Predictable movers

Stocks whose movement appears to have been potentially inferable from information available beforehand.

### F. Apparently unpredictable movers

Stocks whose decisive movement appears to have depended on information that was not available at the decision time.

### G. Near misses

Stocks that were just below the prediction threshold.

### H. False positives

Stocks predicted to move substantially but that did not.

### I. False negatives

Stocks that moved substantially but were not selected.

The system must preserve all of these categories.

---

# 5. STUDY LOSERS AS AGGRESSIVELY AS WINNERS

This is a mandatory architectural principle.

Do not build a system that asks:

> "Why did my winners work?"

while barely asking:

> "Why did my losers happen?"

For every meaningful winner, study:

```text
Why did it move?

Why did we detect it?

Why did we rank it highly?

Which signals contributed?

Which signals were irrelevant?

Which signals generalized?

Could we have predicted the magnitude?

Could we have predicted the timing?
```

For every meaningful loser, ask:

```text
Why did it fall?

Did we predict it?

Should we have predicted it?

Did our volatility model identify it?

Did our direction model incorrectly call it positive?

Which evidence caused the mistake?

Was the mistake selection?

Direction?

Timing?

Exit?

Risk?

Regime?

Data quality?

External event?

Pattern failure?

Unknown?
```

The system must maintain a dedicated **loss research pipeline**.

A loss that teaches the system how to avoid future losses can be more valuable than a winner that merely confirms something already known.

### CODE REQUIREMENT

**Expected implementation: 2,500–4,000 meaningful lines.**

---

# 6. BUILD A MISSED-WINNER / MISSED-LOSER DISCOVERY ENGINE

Do not only evaluate selected stocks.

At the end of every simulated day, ask:

```text
What were the largest movements?

Which ones did we predict?

Which ones did we partially predict?

Which ones did we completely miss?

Which ones were predictable from pre-move information?

Which ones were not?

Which near-misses contained useful information?

Which apparent misses were actually unknowable?
```

Rank missed opportunities by:

```text
magnitude
predictability
model confidence
information availability
similarity to known patterns
novelty
repeatability
loss-reduction potential
future decision relevance
```

This creates a permanent research stream.

### CODE REQUIREMENT

**Expected implementation: 1,800–2,800 meaningful lines.**

---

# 7. BUILD AN EXTERNAL-FACTOR / KNOWABILITY ENGINE

The system must distinguish:

```text
PREDICTABLE
POTENTIALLY PREDICTABLE
WEAKLY PREDICTABLE
UNKNOWN
EXTERNALLY CAUSED
INFORMATIONALLY UNAVAILABLE
DATA FAILURE
```

However:

**Do not allow the future to leak into the trader.**

The research/audit layer may later investigate what caused a historical movement.

The blind decision layer may only use information that existed at its historical decision timestamp.

Separate the architecture into:

```text
BLIND DECISION WORLD
```

and

```text
RESEARCH / AUDIT WORLD
```

The audit world may study matured outcomes to determine whether a hypothesis deserves investigation.

It must never feed future-derived labels back into the historical decision state without an explicit point-in-time training gate.

The research system must distinguish:

### Known-before-event

Information genuinely available before the move.

### Known-only-after-event

Information discovered after the move.

### Simultaneous

Information becoming available around the decision boundary.

### Uncertain availability

Timestamp or provenance cannot be established.

### Unavailable

No legitimate historical information existed that could have revealed the event.

Unknown must remain unknown.

Do not convert hindsight into predictive knowledge.

### CODE REQUIREMENT

**Expected implementation: 2,500–4,000 meaningful lines.**

---

# 8. CREATE THE "COULD I HAVE KNOWN?" TEST

Every major historical movement should receive a formal counterfactual assessment.

For a historical event:

```text
Could the system have known enough beforehand?
```

The evaluator must reconstruct the exact information state at the relevant timestamp.

It should compare:

```text
available features
available patterns
available historical memory
available macro data
available market state
available event information
available price information
available volume information
available technical structure
available cross-sectional information
```

against what became known later.

The research report should produce:

```text
knowledge_state_at_decision
future_information_used_by_auditor
information_that_would_have_been_available
information_that_was_unavailable
confidence_in_classification
```

The classification itself must never become a hidden future signal.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 9. VOLATILITY RESEARCH MUST BE ITS OWN SCIENTIFIC PROGRAM

Build a dedicated volatility research laboratory.

Research questions include:

```text
Which features predict extreme movement?

Which combinations predict movement?

Does volume expansion precede volatility?

Does unusual return compression precede volatility?

Do gaps matter?

Do intraday ranges matter?

Do cross-sectional relationships matter?

Does market-wide volatility matter?

Does sector volatility matter?

Do earnings/event structures matter?

Do insider patterns matter?

Do filing patterns matter?

Do previous analogs matter?

Do patterns interact?

Does volatility prediction transfer between eras?

Does it transfer between sectors?

Does it transfer between stocks?

Does it survive regime changes?
```

Do not merely optimize one volatility score.

Maintain competing hypotheses.

Examples:

```text
H1 = volatility clustering
H2 = event anticipation
H3 = liquidity shock
H4 = cross-sectional relative movement
H5 = regime-dependent volatility
H6 = technical compression → expansion
H7 = unusual volume → movement
H8 = multi-factor interaction
H9 = unknown mechanism
```

The system must be allowed to discover H10, H11, H12, etc.

### CODE REQUIREMENT

**Expected implementation: 3,000–4,500 meaningful lines.**

---

# 10. BUILD A DIRECTION RESEARCH LAB ONLY AFTER VOLATILITY HAS EVIDENCE

Direction should operate conditionally:

```text
Universe
→ volatility probability
→ predicted movers
→ direction probability
→ risk assessment
→ final decision
```

Do not evaluate direction primarily on arbitrary market stocks.

Evaluate it on:

```text
stocks the volatility model would actually have selected
```

This prevents the direction model from solving an easier or irrelevant problem.

Research:

```text
momentum
reversal
event reaction
earnings reaction
insider behavior
52-week positioning
relative strength
sector behavior
market regime
volume structure
intraday path
gap behavior
pattern combinations
volatility regime
cross-sectional relationships
```

But do not assume any of these work.

Every hypothesis must compete against:

```text
base rate
random
shuffled labels
no-learning
simple baseline
existing model
```

### CODE REQUIREMENT

**Expected implementation: 3,000–4,500 meaningful lines.**

---

# 11. BUILD A CONDITIONAL 80%-ACCURACY FRONTIER

The system must NOT simply report:

```text
direction accuracy = 53%
```

It must study:

```text
coverage → accuracy
```

For example:

```text
coverage 100% → accuracy
coverage 50% → accuracy
coverage 25% → accuracy
coverage 10% → accuracy
coverage 5% → accuracy
coverage 1% → accuracy
```

For each:

* sample size;
* confidence interval;
* calibration;
* base rate;
* number of weeks;
* number of stocks;
* number of independent observations;
* era stability;
* sector stability;
* regime stability.

The system should determine whether high accuracy is real or merely a consequence of selecting a tiny sample.

The 80% objective must be treated as:

```text
accuracy
+
coverage
+
calibration
+
out-of-sample validity
+
transfer
+
risk
```

not accuracy alone.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 12. BUILD A WINNER/LOSER SYMMETRY ENGINE

For every important pattern, calculate both:

```text
when pattern predicts a winner
```

and

```text
when pattern predicts a loser
```

Study:

```text
true positives
false positives
true negatives
false negatives
missed winners
missed losers
```

A pattern should not become trusted simply because it finds winners.

It must also demonstrate acceptable behavior on:

```text
wrong-direction cases
large-loss cases
regime transitions
external-event cases
near-miss cases
high-volatility failures
low-liquidity failures
```

Build a **loss-risk knowledge bank** independent from the opportunity knowledge bank.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 13. BUILD A PATTERN DISCOVERY ENGINE THAT CAN DISCOVER NEW PATTERNS

The research intelligence must continually search the historical data for relationships.

Candidate sources can include:

```text
price
returns
ranges
gaps
volume
relative volume
volatility
cross-sectional rank
sector rank
market rank
moving relationships
trend structure
mean reversion structure
event timing
earnings timing
filings
insider activity
macro state
market breadth
correlations
dispersion
liquidity
regime
existing learned patterns
```

But every discovered pattern must become a formal knowledge object with:

```text
definition
provenance
discovery timestamp
training period
validation periods
evidence count
independent evidence count
truth probability
reliability
transfer confidence
context dependence
failure conditions
known counterexamples
complexity
decision impact
```

No pattern becomes trusted simply because a mining algorithm found it.

### CODE REQUIREMENT

**Expected implementation: 4,000–6,000 meaningful lines.**

---

# 14. MAKE PATTERN BREAKS A FIRST-CLASS RESEARCH TARGET

When a previously successful pattern stops working, do not simply lower its weight.

Research WHY.

Possible explanations:

```text
regime change
market structure change
liquidity change
sector shift
event environment
pattern crowding
feature relationship changed
measurement error
sampling artifact
random variation
hidden interaction
external shock
unknown cause
```

A broken pattern must trigger an investigation.

The system should ask:

```text
Can I identify a pre-existing variable that predicts the break?

Does that variable transfer OOS?

Can I predict future breaks?

Does gating the pattern improve performance?

Does gating reduce losses?

Does the explanation outperform random explanations?
```

If not:

```text
UNKNOWN
```

is the correct answer.

The existing C60/B27 architecture already makes pattern-break research a major requirement; this new system must make those investigations part of the autonomous research scheduler.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 15. BUILD EXPERIMENT MEMORY

Every experiment must be permanently remembered.

Record:

```text
question
hypothesis
counter-hypotheses
dataset
time period
information cutoff
features
representation
algorithm
hyperparameters
random seed
code hash
data hash
configuration hash
controls
result
confidence interval
failure mode
transfer result
memorization result
complexity
compute cost
research value
decision impact
```

Before launching a new experiment:

```text
Have we already done this?
```

If yes:

```text
Did the previous experiment answer the question?
```

If yes:

```text
Do not repeat unless there is a justified replication reason.
```

If no:

```text
Explain what is different.
```

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 16. BUILD META-LEARNING ABOUT RESEARCH ITSELF

WEEKLY7 must eventually learn:

```text
Which experiments produce durable knowledge?

Which experiment types usually overfit?

Which pattern families transfer?

Which representations repeatedly fail?

Which datasets create false discoveries?

Which validation methods catch the most errors?

Which research questions produce useful decision changes?

Which failed experiments were actually informative?

Which research paths repeatedly waste compute?

Which kinds of discoveries survive new eras?

Which kinds of discoveries survive new stocks?

Which kinds of discoveries survive new regimes?
```

Then use those findings to improve the research scheduler.

This creates:

```text
MARKET LEARNING
↓
LEARNING ABOUT MARKET
↓
LEARNING ABOUT HOW TO LEARN ABOUT MARKET
↓
LEARNING WHICH LEARNING PROCESSES WORK
```

Meta-learning itself must be evaluated out of sample.

Do not allow the research scheduler to train itself on its own future evaluation results.

The existing S09 design already calls for meta-learning, failed-learner memory, experiment memory, and information-gain prioritization; this phase should integrate them into one continuous research brain.

### CODE REQUIREMENT

**Expected implementation: 2,500–4,000 meaningful lines.**

---

# 17. BUILD A FAILED-LEARNER LAB

Every serious failed learner becomes a permanent scientific artifact.

Record:

```text
what it attempted
why it appeared promising
what it actually did
where it failed
whether it overfit
whether it memorized
whether it transferred
whether it increased risk
whether its failure generalized
```

Before creating a new learner, consult the failed-learner database.

The system should be able to say:

```text
We have already attempted this class of learner 14 times.

11 failed because of overfitting.
2 failed because the signal did not transfer.
1 failed because of risk concentration.

A new attempt requires a materially different hypothesis.
```

This prevents infinite rediscovery of dead ends.

### CODE REQUIREMENT

**Expected implementation: 1,200–2,000 meaningful lines.**

---

# 18. BUILD A RESEARCH COMPUTE MANAGER

The system must understand that compute is finite.

Allocate resources according to research value.

Possible states:

```text
QUEUED
EXPLORING
PROMISING
REPLICATING
ESCALATED
VALIDATING
FAILED
DORMANT
RETIRED
```

A cheap exploratory experiment can become a large validation experiment only after evidence warrants escalation.

Example:

```text
Stage 1:
100 cheap tests

↓ evidence

Stage 2:
10 stronger tests

↓ evidence

Stage 3:
cross-year validation

↓ evidence

Stage 4:
large fresh holdout

↓ evidence

Stage 5:
integration test
```

Never immediately spend enormous compute on a weak hypothesis.

Likewise, never abandon a promising hypothesis simply because its first cheap test was underpowered.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 19. BUILD AN EXPERIMENT VALUE ACCOUNTING SYSTEM

After every research job, calculate:

```text
compute_cost
information_gain
decision_change
risk_change
prediction_change
transfer_change
knowledge_redundancy
```

The system should learn whether an experiment was actually useful.

A research result such as:

```text
+0.01% backtest improvement
```

should NOT automatically be considered valuable.

If the result:

* does not transfer;
* does not replicate;
* increases complexity;
* does not change decisions;
* does not reduce risk;

then it may be essentially worthless.

Conversely:

```text
0% immediate return improvement
```

can be highly valuable if it discovers:

```text
this entire feature family is unreliable
```

and prevents thousands of future experiments.

### CODE REQUIREMENT

**Expected implementation: 1,200–2,000 meaningful lines.**

---

# 20. CREATE A "STOP WASTING COMPUTE" MECHANISM

The system must actively detect research that is consuming resources without meaningful progress.

For each research branch:

```text
progress per compute
```

must be monitored.

If repeated experiments produce:

```text
tiny effect
no transfer
no replication
no decision change
high complexity
```

the branch should be deprioritized.

But do NOT permanently delete the idea.

Move it to:

```text
DORMANT
```

with a reason.

It can return if:

```text
new data
new representation
new regime
new evidence
new hypothesis
```

makes it worth reconsidering.

This follows the existing requirement that broken or weak knowledge should be gated rather than casually erased.

### CODE REQUIREMENT

**Expected implementation: 800–1,500 meaningful lines.**

---

# 21. BUILD A DAILY MARKET "AUTOPSY"

After every simulated trading day, run an automatic market autopsy.

Produce:

## MARKET

```text
largest gainers
largest losers
largest ranges
largest volume anomalies
largest gaps
largest volatility expansions
largest volatility contractions
```

## MODEL

```text
best predictions
worst predictions
largest missed winners
largest missed losers
highest-confidence mistakes
lowest-confidence successes
```

## RESEARCH

```text
new patterns
broken patterns
surprising observations
contradictions
unknown causes
new research questions
```

## RISK

```text
largest avoidable losses
largest unavoidable losses
concentration
correlated failures
gap risk
regime exposure
```

## LEARNING

```text
what changed in knowledge
what changed in confidence
what changed in research priority
what experiments should run next
```

This report should feed the autonomous research queue.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 22. BUILD A DAILY RESEARCH TARGET GENERATOR

Every day, generate research candidates from:

```text
surprises
losses
wins
missed winners
missed losers
new volatility clusters
pattern breaks
contradictions
new regimes
data anomalies
research failures
new combinations
understudied sectors
understudied market conditions
unknown areas
```

Example:

```text
Yesterday:
Pattern X predicted 17 volatile stocks.

12 were correct.

5 were wrong.

All 5 failures occurred during high market dispersion.

Generate hypothesis:

"Pattern X loses directional reliability when dispersion > D."

```

Then test it.

If it survives OOS:

```text
promote knowledge
```

If not:

```text
record failure
```

This is how the system turns every trading day into new scientific material.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 23. BUILD MULTI-SCALE LEARNING

The system should research at multiple timescales:

```text
intraday
daily
weekly
multi-week
monthly
regime
multi-year
```

Do not assume that a pattern operating at one timescale transfers to another.

Every pattern should carry an explicit horizon.

Examples:

```text
5-minute
30-minute
1-hour
1-day
3-day
5-day
1-week
2-week
1-month
```

The actual horizons must match available data and the prediction problem.

### CODE REQUIREMENT

**Expected implementation: 1,200–2,000 meaningful lines.**

---

# 24. BUILD CROSS-SECTIONAL LEARNING

The system must not think only:

```text
What happens next to this stock?
```

It must also ask:

```text
Why is this stock behaving differently from the rest of the market?
```

Research:

```text
stock vs sector
stock vs industry
stock vs market
stock vs volatility cohort
stock vs liquidity cohort
stock vs momentum cohort
stock vs event cohort
```

This helps identify whether a movement is:

```text
stock-specific
sector-specific
market-wide
regime-wide
```

The distinction should feed both volatility and direction models.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 25. BUILD REGIME-AWARE RESEARCH

The research engine must continually monitor:

```text
bull
bear
high volatility
low volatility
high dispersion
low dispersion
high liquidity
low liquidity
trend
mean-reversion
event-heavy
event-light
```

Do not hard-code these as final truths.

The system should also be able to discover new regimes.

A pattern that works only in one regime must not be treated as universal.

The system should learn:

```text
pattern × regime
```

rather than only:

```text
pattern
```

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 26. BUILD AN INTERACTION DISCOVERY SYSTEM

Do not restrict research to individual features.

Search for interactions such as:

```text
volume × volatility
momentum × event
reversal × gap
sector strength × stock strength
market volatility × liquidity
pattern × regime
pattern × pattern
event × pattern
```

But impose strong multiple-testing controls.

The system must distinguish:

```text
interesting interaction
```

from:

```text
interaction found because millions of combinations were tested.
```

Use:

* held-out validation;
* multiple-testing correction;
* fresh seeds;
* cross-year validation;
* cross-stock validation;
* shuffled controls;
* complexity penalties.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 27. BUILD A KNOWLEDGE GRAPH

Represent relationships between:

```text
patterns
features
stocks
sectors
regimes
events
outcomes
failures
experiments
learners
research questions
```

Example:

```text
Pattern A
  ↓
predicts volatility
  ↓
works in high-volume environments
  ↓
fails during regime transition
  ↓
failure explanation = liquidity shift
  ↓
gating rule tested
  ↓
OOS improvement
```

This graph should allow the research engine to discover unanswered relationships.

### CODE REQUIREMENT

**Expected implementation: 1,500–2,500 meaningful lines.**

---

# 28. BUILD A KNOWLEDGE-TO-DECISION BRIDGE

Every discovery must eventually answer:

> What does this change?

Possible outputs:

```text
volatility ranking
direction ranking
position sizing
risk penalty
abstention
pattern gating
confidence
research priority
data priority
```

If a discovery cannot change a decision or research priority, its value should be explicitly recorded as informational rather than pretending it improves the trading system.

The existing contract already requires knowledge objects to connect to decisions; this new layer must enforce that connection for autonomous research discoveries.

### CODE REQUIREMENT

**Expected implementation: 1,000–1,800 meaningful lines.**

---

# 29. BUILD FULL ANTI-CHEATING ARCHITECTURE

This is non-negotiable.

The simulated trader must have access only to:

```text
information available at simulated timestamp T
```

It must never access:

```text
future prices
future volume
future filings
future earnings
future labels
future metadata
future corporate actions
future market state
future learned state
future research results
future pattern health
future experiment outcomes
future year identity
```

The research system may study matured historical outcomes, but the result must pass through explicit point-in-time gates before becoming training knowledge.

Create separate namespaces for:

```text
LIVE_POINT_IN_TIME_STATE
```

and

```text
MATURED_RESEARCH_STATE
```

No implicit shared state.

No hidden shortcuts.

No convenience imports.

No future-derived cache.

No future-derived feature.

No timestamp ambiguity.

Every information object needs provenance.

---

# 30. STRENGTHEN THE EXISTING DISGUISED-YEAR TEST

The existing project already requires disguised repeated-year testing and has identified numeric and fuzzy fingerprints as risks.

The new research intelligence must obey the same restrictions.

It must not infer:

```text
"This is 2008."
```

or:

```text
"This is the same year I saw before."
```

or:

```text
"This is the training run."
```

from:

* year fingerprints;
* stock identities;
* exact numeric fingerprints;
* dataset structure;
* future-derived metadata;
* hidden cache state.

The research system itself may maintain internal audit metadata, but the blind learner must not see it.

---

# 31. BUILD A RESEARCH-SIDE / TRADER-SIDE FIREWALL

Create a formal firewall between:

### RESEARCH SIDE

May:

* study matured outcomes;
* investigate failures;
* generate hypotheses;
* analyze historical causes;
* compare experiments;
* discover patterns;
* evaluate learners.

### TRADER SIDE

May only:

* observe point-in-time data;
* access matured knowledge whose maturity date precedes the decision;
* retrieve permitted historical patterns;
* generate predictions;
* make decisions.

The firewall must be tested with planted leaks.

Create tests where the research side deliberately attempts to leak:

```text
future price
future label
future event
year identity
future pattern status
future experiment result
```

The trader must fail closed.

### CODE REQUIREMENT

**Expected implementation: 2,000–3,000 meaningful lines.**

---

# 32. BUILD RESEARCH REPLICATION REQUIREMENTS

A discovery is not trustworthy merely because one experiment found it.

Promotion should generally require some combination of:

```text
independent replication
fresh time period
fresh stocks
fresh seed
fresh regime
control comparison
```

The exact thresholds should be learned and documented rather than arbitrarily optimized.

The system must never allow:

```text
one lucky experiment
```

to permanently change the trading system.

---

# 33. BUILD THE "UNKNOWN CAUSE" SYSTEM

Sometimes the correct answer is:

> We do not know.

That is a successful research result.

For unexplained extreme movements:

```text
UNKNOWN_CAUSE
```

must remain a first-class outcome.

Do not force:

```text
earnings
momentum
technical pattern
market regime
```

just because one of those explanations exists.

An explanation should need evidence.

The system should track:

```text
unknown cause rate
unknown cause by regime
unknown cause by volatility
unknown cause by sector
unknown cause by model confidence
```

If the unknown rate decreases because the system genuinely discovers transferable causes, that is meaningful learning.

If it decreases because the classifier simply forces more labels, that is failure.

---

# 34. BUILD RESEARCH VALUE FROM LOSSES

A particularly important objective:

The research system must explicitly estimate:

```text
expected future loss avoided
```

for every research question.

Example:

```text
Question A:
Could improve winner detection by 0.5%.

Question B:
Could identify a failure regime responsible for the largest 5% of historical losses.

Prefer B if its expected loss-reduction value is materially larger.
```

This prevents the system from becoming obsessed with finding more winners while ignoring catastrophic downside.

The objective hierarchy should therefore be approximately:

```text
1. Discover reliable volatility opportunities.
2. Prevent catastrophic false positives and avoidable losses.
3. Improve directional prediction among genuinely predicted movers.
4. Improve coverage.
5. Improve consistency.
6. Improve efficiency.
7. Optimize secondary performance metrics.
```

Do not optimize these as independent disconnected models.

---

# 35. BUILD THE FINAL TWO-STAGE PREDICTION ARCHITECTURE

The intended eventual architecture should resemble:

```text
ALL ELIGIBLE STOCKS
        ↓
POINT-IN-TIME FEATURE STATE
        ↓
VOLATILITY RESEARCH / MODEL
        ↓
P(volatility)
        ↓
VOLATILITY RANK
        ↓
PREDICTED-MOVER UNIVERSE
        ↓
DIRECTION RESEARCH / MODEL
        ↓
P(up | predicted mover)
P(down | predicted mover)
        ↓
CALIBRATION
        ↓
FAILURE / REGIME / PATTERN HEALTH
        ↓
RISK FILTER
        ↓
POSITION / ABSTENTION
```

The research brain sits around this system:

```text
                    RESEARCH BRAIN
                         ↕
       ┌────────────────────────────────┐
       │                                │
MARKET → VOLATILITY → DIRECTION → RISK │
       │                                │
       └────────────────────────────────┘
                         ↕
                    OUTCOMES
                         ↓
                 LOSSES / WINS
                         ↓
               RESEARCH QUESTIONS
                         ↓
                  NEW EXPERIMENTS
                         ↓
                 NEW KNOWLEDGE
```

---

# 36. BUILD A SELF-IMPROVEMENT SCORECARD

Every major research cycle should report:

## VOLATILITY

```text
precision
recall
ranking quality
calibration
coverage
transfer
stability
```

## DIRECTION

```text
accuracy
Brier score
calibration
coverage
accuracy at coverage levels
transfer
stability
```

## LOSS

```text
worst loss
tail loss
false-positive loss
avoidable-loss rate
large-loss frequency
```

## LEARNING

```text
new knowledge
knowledge retired
knowledge downgraded
knowledge promoted
experiments completed
experiments replicated
failed learners
successful transfers
memorization failures
```

## RESEARCH EFFICIENCY

```text
compute consumed
information gained
decision changes
duplicate experiments avoided
research branches abandoned
research branches escalated
```

The system must learn from this scorecard.

---

# 37. BUILD A "RESEARCH BRAIN HEALTH" SYSTEM

The autonomous researcher itself can fail.

Monitor:

```text
research diversity
duplicate rate
experiment success rate
false discovery rate
replication rate
transfer rate
compute efficiency
knowledge churn
overfitting rate
memorization rate
research concentration
```

If the researcher starts spending:

```text
90% of compute
```

on one tiny parameter family producing negligible gains, it should detect that and redirect resources.

If it starts abandoning all difficult questions, detect that too.

If it starts only researching areas where success is easy, penalize the research policy.

The system must balance:

```text
exploitation
```

with:

```text
exploration
```

---

# 38. BUILD A RESEARCH DIVERSITY CONTROLLER

Prevent the autonomous system from becoming trapped in one hypothesis family.

Maintain research allocation across:

```text
known promising areas
uncertain areas
new representations
failed areas with new hypotheses
risk research
volatility research
direction research
regime research
data-quality research
pattern-break research
```

The percentages should be adaptive.

The research brain must learn where exploration produces useful discoveries.

---

# 39. BUILD LONG-TERM MEMORY OF THE ENTIRE SCIENTIFIC PROCESS

Memory must contain not merely:

```text
pattern worked
```

but:

```text
why it was proposed
why it was tested
how it was tested
what it predicted
where it worked
where it failed
why it failed
whether failure transferred
whether the explanation survived
what replaced it
```

The system should effectively accumulate a scientific history of its own development.

It must be able to answer:

> Why do you currently believe this pattern is useful?

and:

> What evidence would cause you to stop trusting it?

---

# 40. BUILD A RESEARCH "QUESTION GENERATOR"

Generate questions automatically from:

### Surprises

```text
Why did this stock move despite weak volatility signals?
```

### Contradictions

```text
Why did two apparently identical setups produce opposite outcomes?
```

### Losses

```text
Why did our highest-confidence prediction lose?
```

### Missed winners

```text
What information existed before this movement?
```

### Pattern breaks

```text
What changed before this pattern stopped working?
```

### Regime changes

```text
Which existing knowledge survived the regime transition?
```

### New discoveries

```text
Does this newly discovered pattern generalize?
```

Every question becomes a research object with:

```text
hypothesis
expected value
test plan
success criterion
failure criterion
priority
```

---

# 41. BUILD A RESEARCH HYPOTHESIS TREE

Do not treat every experiment as independent.

Represent research as:

```text
Question
├── Hypothesis A
│   ├── Test A1
│   ├── Test A2
│   └── Counterexample A3
├── Hypothesis B
│   ├── Test B1
│   └── Test B2
└── Unknown explanation
    ├── Search C1
    └── Search C2
```

This prevents random wandering.

When evidence kills a branch:

```text
mark branch failed
```

and redirect research toward surviving explanations.

---

# 42. BUILD A RESEARCH QUALITY GATE

Before any discovery can influence the production learner, verify:

```text
point-in-time correctness
no future leakage
identity independence
out-of-sample evidence
replication
calibration
risk behavior
complexity
transfer
failure behavior
```

If any critical gate fails:

```text
DO NOT PROMOTE.
```

Instead:

```text
FAILED
QUARANTINED
UNKNOWN
or
NEEDS_MORE_EVIDENCE
```

---

# 43. DO NOT LET THE SYSTEM OPTIMIZE THE WRONG METRIC

The system must not learn that:

```text
more trades = better
more predictions = better
more patterns = better
more code = better
more experiments = better
higher backtest return = automatically better
```

The core scientific objective remains:

```text
reliable prediction
that transfers
without cheating
while controlling downside
```

A system that produces 100,000 patterns but cannot predict unseen years has learned almost nothing useful.

A system that discovers five highly transferable patterns has potentially learned something substantial.

---

# 44. CODE DEPTH REQUIREMENT

For EVERY major component in this prompt:

**The stated code range is a minimum design-depth expectation, not a license to pad code.**

Meaningful lines only.

Exclude:

```text
comments
docstrings
blank lines
generated repetition
pointless wrappers
duplicate implementations
artificial abstraction
dead code
```

If implementation falls below the stated range, assume that functionality may be incomplete and perform a design-depth audit.

If the correct implementation genuinely requires less code, document exactly why and demonstrate complete coverage through tests and integration rather than padding.

Going above the range is permitted and expected where necessary.

---

# 45. EXPECTED NEW CODE BUDGET

Target approximately:

| Component                               | Expected meaningful lines |
| --------------------------------------- | ------------------------: |
| Research priority engine                |               1,500–2,500 |
| Autonomous research loop                |               2,000–3,000 |
| Market-wide observer                    |               1,500–2,500 |
| Winner/loser research                   |               2,500–4,000 |
| Missed opportunity engine               |               1,800–2,800 |
| Knowability/external-factor engine      |               2,500–4,000 |
| Counterfactual knowledge reconstruction |               2,000–3,000 |
| Volatility research laboratory          |               3,000–4,500 |
| Direction research laboratory           |               3,000–4,500 |
| Conditional accuracy frontier           |               1,500–2,500 |
| Winner/loser symmetry                   |               2,000–3,000 |
| Pattern discovery                       |               4,000–6,000 |
| Pattern-break research integration      |               2,000–3,000 |
| Experiment memory                       |               1,500–2,500 |
| Meta-learning                           |               2,500–4,000 |
| Failed-learner laboratory               |               1,200–2,000 |
| Compute manager                         |               1,500–2,500 |
| Experiment value accounting             |               1,200–2,000 |
| Compute-waste controller                |                 800–1,500 |
| Daily market autopsy                    |               1,500–2,500 |
| Daily research target generator         |               1,500–2,500 |
| Multi-scale learning                    |               1,200–2,000 |
| Cross-sectional learning                |               1,500–2,500 |
| Regime-aware research                   |               1,500–2,500 |
| Interaction discovery                   |               2,000–3,000 |
| Knowledge graph                         |               1,500–2,500 |
| Knowledge-to-decision bridge            |               1,000–1,800 |
| Research/trader firewall integration    |               2,000–3,000 |
| Research replication system             |               1,000–1,800 |
| Unknown-cause system                    |               1,000–1,800 |
| Self-improvement scorecard              |               1,000–1,800 |
| Research-brain health                   |               1,200–2,000 |
| Research diversity controller           |               1,000–1,800 |
| Scientific long-term memory integration |               1,500–2,500 |
| Question generator                      |               1,200–2,000 |
| Hypothesis tree                         |               1,000–1,800 |
| Quality/promotion gate                  |               1,000–1,800 |

### APPROXIMATE NEW SYSTEM SIZE

**~58,000–90,000 meaningful lines**, depending on how much existing Weekly7 functionality can be cleanly integrated rather than duplicated.

Do NOT rewrite existing equivalent modules merely to hit this number.

The line budget is a depth check, not the objective.

---

# 46. TESTING REQUIREMENTS

After foundation implementation, construct a complete testing program.

## A. UNIT TESTS

Every new module.

## B. INTEGRATION TESTS

Research brain → learner → decision system.

## C. PLANTED WORLD TESTS

Create worlds where:

```text
volatility signal genuinely exists
direction signal genuinely exists
failure regime genuinely exists
```

The system must discover them.

Also create worlds where:

```text
no signal exists
```

The system must NOT manufacture one.

## D. LEAK TESTS

Plant:

```text
future price
future label
future event
year identity
future pattern status
```

The trader must not access them.

## E. MEMORIZATION TESTS

Use:

```text
same identities
new identities
disguised years
perturbed years
new stocks
new eras
```

A memorizer must be distinguishable from a learner.

## F. TRANSFER TESTS

Require:

```text
unseen years
unseen stocks
unseen sectors
unseen regimes
```

## G. RESEARCH-POLICY TESTS

Create a planted environment where:

```text
Experiment A has huge information value.
Experiment B produces tiny improvements.
```

The research scheduler should learn to allocate more resources toward A.

## H. LOSS-PRIORITY TEST

Create a world where a small improvement in avoiding losses is more valuable than a large number of tiny winner improvements.

The research brain should eventually recognize this.

## I. COMPUTE-WASTE TEST

Create a research branch that produces repeated negligible improvements.

The system must reduce its priority.

---

# 47. FINAL VALIDATION

The new research brain is NOT considered complete merely because:

```text
all modules exist
```

or:

```text
tests pass
```

It is complete only when the entire pipeline demonstrates:

```text
research questions generated automatically
↓
research priority selected intelligently
↓
experiments executed
↓
results recorded
↓
failed experiments remembered
↓
successful knowledge validated
↓
knowledge transferred
↓
knowledge affects decisions
↓
outcomes measured
↓
new research questions generated
```

And this cycle operates without future information.

---

# 48. DO NOT DECLARE SUCCESS FROM SAME-YEAR IMPROVEMENT

This is critical.

Same-year improvement can be useful evidence of learning mechanics.

It is NOT sufficient evidence of market knowledge.

The actual scientific test is whether improvements survive:

```text
unseen time
unseen stocks
unseen regimes
fresh seeds
fresh holdouts
```

The existing Weekly7 results already showed why this matters: prior same-year gains disappeared when the time gate was corrected, and existing learner searches have not yet demonstrated robust transfer without extra risk.

Therefore:

```text
SAME-YEAR IMPROVEMENT = INTERESTING
OOS TRANSFER = REQUIRED
```

---

# 49. LIVE MARKET IS NOT PART OF THIS BUILD

Do NOT move to real-money trading.

Do NOT optimize around a live account.

Do NOT treat the paper account as evidence that the learning brain is complete.

The existing owner directive explicitly parks Live until the Test self-learning system is perfected.

Everything in this prompt is initially implemented and validated inside the simulator.

Only after the entire test architecture is independently validated should live/paper deployment become a separate phase.

---

# 50. AUTONOMOUS OPERATION

Once implemented, the research system should be capable of running continuously.

It should:

```text
observe
research
experiment
learn
validate
remember
prioritize
replicate
downgrade
promote
research failures
research winners
research losers
research itself
```

It should NOT require the owner to manually tell it:

> "Research volatility today."

The research brain should determine:

> "Volatility prediction is currently the highest-value unresolved problem, so allocate 43% of research compute there."

Then later:

> "Volatility prediction has improved substantially. The largest remaining expected decision value is now directional discrimination among predicted movers, so shift resources."

Then:

> "Direction research is currently producing negligible transferable improvement, while loss-regime research has high expected value. Redirect compute."

That is the behavior required.

---

# 51. THE SYSTEM MUST KNOW WHEN TO CHANGE PRIORITIES

The highest-level controller should continuously compare:

```text
current capability
desired capability
largest remaining uncertainty
largest remaining loss source
largest remaining prediction gap
highest-value research opportunity
```

The research allocation must therefore be dynamic.

The desired progression is:

```text
PHASE 1
Can we reliably identify volatile stocks?

↓

PHASE 2
Can we distinguish predictable volatility from unknowable external shocks?

↓

PHASE 3
Can we determine direction among predicted movers?

↓

PHASE 4
Can direction reach high accuracy at meaningful coverage?

↓

PHASE 5
Can the entire system transfer across years/stocks/regimes?

↓

PHASE 6
Can it avoid catastrophic losses?

↓

PHASE 7
Can it continuously discover new knowledge?

↓

PHASE 8
Can it continuously improve the process by which it discovers knowledge?
```

The system may temporarily revisit earlier phases when evidence shows a regression.

---

# 52. MASTER CHECKLIST — DO NOT STOP UNTIL COMPLETE

## FOUNDATION

* [ ] Research priority engine implemented
* [ ] Autonomous research scheduler implemented
* [ ] Market-wide observation layer implemented
* [ ] Winner research implemented
* [ ] Loser research implemented
* [ ] Missed-winner research implemented
* [ ] Missed-loser research implemented
* [ ] Knowability engine implemented
* [ ] Counterfactual information reconstruction implemented
* [ ] Volatility laboratory implemented
* [ ] Direction laboratory implemented
* [ ] Conditional accuracy frontier implemented
* [ ] Pattern discovery expansion implemented
* [ ] Pattern-break research integrated
* [ ] Experiment memory integrated
* [ ] Meta-learning integrated
* [ ] Failed-learner registry integrated
* [ ] Compute manager implemented
* [ ] Experiment-value accounting implemented
* [ ] Compute-waste controller implemented
* [ ] Daily autopsy implemented
* [ ] Daily research generator implemented
* [ ] Multi-scale research implemented
* [ ] Cross-sectional research implemented
* [ ] Regime research implemented
* [ ] Interaction discovery implemented
* [ ] Knowledge graph integrated
* [ ] Knowledge-to-decision bridge integrated
* [ ] Research/trader firewall integrated
* [ ] Replication system implemented
* [ ] Unknown-cause system integrated
* [ ] Research-health system implemented
* [ ] Research-diversity system implemented
* [ ] Scientific long-term memory integrated
* [ ] Question generator implemented
* [ ] Hypothesis tree implemented
* [ ] Promotion quality gate implemented

## TESTING

* [ ] Unit tests complete
* [ ] Integration tests complete
* [ ] Planted volatility world passes
* [ ] Planted direction world passes
* [ ] Planted failure-regime world passes
* [ ] Null world does not generate fake signal
* [ ] Future-leak tests pass
* [ ] Identity-leak tests pass
* [ ] Year-recognition tests pass
* [ ] Memorization controls pass
* [ ] Same-year controls pass
* [ ] Cross-year transfer tests pass
* [ ] Cross-stock transfer tests pass
* [ ] Cross-sector transfer tests pass
* [ ] Cross-regime transfer tests pass
* [ ] Research-priority tests pass
* [ ] Compute-allocation tests pass
* [ ] Compute-waste tests pass
* [ ] Winner/loser symmetry tests pass
* [ ] Unknown-cause tests pass
* [ ] Replication tests pass
* [ ] Promotion gates pass

## SCIENTIFIC VALIDATION

* [ ] Volatility prediction measured OOS
* [ ] Volatility calibration measured
* [ ] Volatility coverage measured
* [ ] Predictable-vs-unknowable classification evaluated
* [ ] Direction measured only on the relevant predicted-mover universe
* [ ] Direction accuracy measured
* [ ] Direction calibration measured
* [ ] Direction coverage frontier measured
* [ ] 80% question evaluated honestly
* [ ] Loss reduction measured
* [ ] Worst-case behavior measured
* [ ] Tail risk measured
* [ ] Pattern transfer measured
* [ ] Pattern-break prediction measured
* [ ] Research-policy transfer measured
* [ ] Meta-learning transfer measured
* [ ] Failed-learner memory demonstrated useful
* [ ] Research compute efficiency measured

## AUTONOMOUS OPERATION

* [ ] System generates its own research questions
* [ ] System ranks research questions
* [ ] System allocates compute
* [ ] System runs experiments
* [ ] System records results
* [ ] System remembers failures
* [ ] System remembers successes
* [ ] System studies winners
* [ ] System studies losers
* [ ] System studies missed winners
* [ ] System studies missed losers
* [ ] System investigates pattern breaks
* [ ] System identifies unknown causes
* [ ] System promotes validated knowledge
* [ ] System downgrades broken knowledge
* [ ] System detects wasted research
* [ ] System redirects compute
* [ ] System learns which research methods work
* [ ] System continuously generates the next research agenda
* [ ] System can run indefinitely with checkpoints and recovery

## FINAL GATE

* [ ] No critical future-information pathway exists
* [ ] No hidden year information reaches the trader
* [ ] No stock-identity memorization pathway remains
* [ ] No survivor-only evaluation remains where it affects conclusions
* [ ] No major research result depends on a single lucky experiment
* [ ] No major promoted knowledge lacks OOS evidence
* [ ] No major promoted knowledge lacks provenance
* [ ] No critical failure is silently ignored
* [ ] No research branch consumes large compute without value accounting
* [ ] No code is padded merely to satisfy line counts
* [ ] Existing C62–C65 requirements remain intact
* [ ] Existing tests remain intact unless legitimately updated for changed behavior
* [ ] Full contract checklist is reconciled
* [ ] Full autonomous-research checklist is reconciled
* [ ] Independent audit completed
* [ ] Masterstock contains the complete state
* [ ] Only after ALL of the above: evaluate whether the test system is ready for the next deployment phase

---

# 53. EXECUTION RULE

DO NOT stop after creating the architecture.

DO NOT stop after writing modules.

DO NOT stop after unit tests.

DO NOT stop after one successful experiment.

DO NOT stop because the system is "pretty advanced."

DO NOT stop because the line count is large.

DO NOT stop because a metric improved.

Continue through:

```text
BUILD
→ TEST
→ INTEGRATE
→ RUN
→ DISCOVER
→ VALIDATE
→ FIX
→ RE-RUN
→ RESEARCH
→ REPLICATE
→ AUDIT
```

until the checklist is genuinely complete.

If a test fails:

```text
investigate
fix
rerun
record
```

Do not lower the test.

If a hypothesis fails:

```text
record the failure
extract the lesson
update research priority
```

Do not pretend it worked.

If a research direction produces negligible value:

```text
measure it
downgrade it
redirect compute
```

Do not spend unlimited compute on it.

If a research direction produces strong evidence:

```text
replicate it
test transfer
test failure modes
then consider promotion
```

Do not promote it immediately.

If the system discovers that the desired 80% direction accuracy is not achievable from the available information:

```text
record that scientific limitation honestly
```

Do not manufacture an 80% result.

---

# FINAL PRINCIPLE

The goal is not to create the biggest stock-prediction codebase.

The goal is to create a system that becomes progressively better at answering:

> **What can I legitimately predict from the information available right now, why can I predict it, when does that prediction fail, how much does that failure matter, and what should I research next to become better?**

The ultimate architecture is therefore:

```text
                    ┌───────────────────────────┐
                    │     AUTONOMOUS RESEARCH    │
                    │          BRAIN             │
                    │                           │
                    │ questions                  │
                    │ hypotheses                 │
                    │ experiments                │
                    │ meta-learning              │
                    │ research allocation        │
                    │ failure learning            │
                    │ compute allocation          │
                    └─────────────┬─────────────┘
                                  │
                         learns how to learn
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      MARKET KNOWLEDGE      │
                    │                           │
                    │ patterns                  │
                    │ regimes                   │
                    │ volatility                │
                    │ direction                 │
                    │ failures                  │
                    │ boundaries                │
                    │ unknowns                  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       PREDICTION           │
                    │                           │
                    │ ALL STOCKS                │
                    │      ↓                    │
                    │ VOLATILITY                │
                    │      ↓                    │
                    │ PREDICTED MOVERS          │
                    │      ↓                    │
                    │ DIRECTION                │
                    │      ↓                    │
                    │ RISK / FAILURE FILTER     │
                    │      ↓                    │
                    │ DECISION / ABSTAIN         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       OUTCOMES             │
                    │                           │
                    │ winners                   │
                    │ losers                    │
                    │ missed winners            │
                    │ missed losers              │
                    │ surprises                 │
                    │ pattern breaks             │
                    │ unknown causes             │
                    └─────────────┬─────────────┘
                                  │
                                  └──────────────►
                                      RESEARCH
                                      BRAIN
                                      ↑
                                      │
                                CONTINUOUS LOOP
```

**The system should not merely learn the market.**

**It should learn how to become better at learning the market.**

And it must do so while remaining completely blind to information that was not available at the moment each historical decision was made.
