# WEEKLY7 — MASTER SELF-LEARNING ENGINE

## AUTONOMOUS BUILD, LEARNING, VALIDATION & NON-STOP COMPLETION CONTRACT

---

# 0. YOUR ROLE

You are the autonomous research engineer responsible for building the **learning intelligence of WEEKLY7**.

The goal is not to make WEEKLY7 contain more code.

The goal is to make WEEKLY7 **learn from experience in a way that produces measurable improvement on future decisions without memorizing the answers**.

The system must progressively become better at:

* recognizing situations it has encountered before;
* distinguishing situations that merely look similar from situations that actually transfer;
* determining when a previously useful pattern is still useful;
* determining when a pattern has weakened, reversed, or become conditional;
* learning from failures;
* learning from missed winners;
* identifying why a decision succeeded or failed;
* separating useful information from redundant information;
* discovering interactions;
* identifying regime dependence;
* transferring knowledge across years;
* transferring knowledge across stocks;
* transferring knowledge across sectors;
* transferring knowledge across market conditions;
* knowing when it does not have enough evidence;
* deciding what experiment should be run next;
* deciding what knowledge deserves influence;
* reducing the influence of knowledge that has degraded;
* recovering knowledge when the original conditions return;
* avoiding ticker/date/year memorization;
* avoiding future-information contamination;
* avoiding learning from its own test answers;
* and demonstrating that learning itself improves the system.

The existing WEEKLY7 blueprint, CANON, Masterstock, and validation architecture remain authoritative.

You are not replacing them.

You are implementing the **learning brain that operates inside them**.

---

# 1. NON-NEGOTIABLE OWNER DIRECTIVES

These are standing requirements.

## 1.1 The learning system is the product

Do not treat learning as a side feature.

A system that has 50,000 lines of trading infrastructure but does not demonstrably improve from experience has failed the central objective.

---

## 1.2 No fake learning

The following do NOT count as learning:

* changing a parameter after seeing the answer;
* hardcoding a winning year;
* remembering a ticker/date combination;
* increasing the weight of a pattern merely because it recently won;
* tuning against the evaluation window;
* adding features until the test improves;
* storing outcome labels in a retrievable form;
* selecting the best historical run;
* manually changing thresholds after seeing results;
* using future information indirectly;
* using test results to alter training state;
* identifying a disguised rerun as a rerun;
* memorizing exact market sequences;
* storing a hidden fingerprint of an evaluation year;
* creating a "memory" table that is actually an answer lookup table.

Learning must produce **transferable knowledge**.

---

# 2. CURRENT PROJECT STATE

The current Masterstock establishes that:

* the foundation has already grown substantially;
* the project has already exceeded the original 25,000-line foundation floor;
* the learning delta remains the critical unresolved issue;
* the current real-archive memorization control is effectively zero;
* lessons from other windows have not yet produced positive learning delta;
* the system already has pattern mining, memory, blind simulation, provenance, parity, planted-pattern validation, and other infrastructure;
* the project therefore does NOT need another generic trading framework;
* it needs the learning mechanism to actually convert experience into future improvement.

The project has already demonstrated that infrastructure can become large without solving the central learning problem.

Therefore:

**Do not confuse implementation volume with intelligence.**

---

# 3. THE CENTRAL LEARNING DEFINITION

Define learning as:

> A change in the system's future behavior caused by prior experience that improves performance on information the system has not previously been allowed to know, while remaining robust against identity memorization, future leakage, overfitting, and unacceptable risk.

The minimum scientific test is:

### BEFORE

Run the system without the newly acquired lesson.

Record:

* decision;
* confidence;
* prediction;
* expected outcome;
* risk;
* selected patterns;
* retrieved memories;
* abstentions;
* uncertainty;
* explanation.

### EXPERIENCE

Reveal the outcome.

### LEARNING

Allow the system to process the experience.

### AFTER

Run the same disguised situation again under a legitimate learner.

The learner should behave differently **only if the prior experience provides transferable information**.

### VALIDATION

Determine whether the change improves future behavior.

---

# 4. THE LEARNING LOOP

Implement the complete loop:

```text
OBSERVE
    ↓
DESCRIBE SITUATION
    ↓
RETRIEVE RELEVANT KNOWLEDGE
    ↓
ASSESS KNOWLEDGE RELIABILITY
    ↓
FORM EXPECTATIONS
    ↓
DECIDE
    ↓
OBSERVE OUTCOME
    ↓
MEASURE SURPRISE
    ↓
ASSIGN CREDIT / BLAME
    ↓
UPDATE BELIEFS
    ↓
INVESTIGATE FAILURE
    ↓
LEARN CONDITIONS
    ↓
LEARN ANTI-CONDITIONS
    ↓
UPDATE RELIABILITY
    ↓
TEST TRANSFER
    ↓
STORE KNOWLEDGE
    ↓
UPDATE KNOWLEDGE GRAPH
    ↓
UPDATE META-KNOWLEDGE
    ↓
SELECT NEXT RESEARCH QUESTION
```

Every stage must be represented in code.

No stage may exist only as documentation.

---

# 5. REQUIRED KNOWLEDGE OBJECT

Implement a durable versioned `KnowledgeObject`.

Minimum conceptual fields:

```text
knowledge_id
version
created_at
updated_at

source_experiences
source_experiments
source_runs

observation
interpretation
hypothesis

contexts
anti_contexts

market_context
sector_context
stock_context
stock_type_context
volatility_context
liquidity_context
regime_context
time_context

effect
effect_direction
effect_size
effect_uncertainty

truth_confidence
current_reliability
context_confidence
transfer_confidence
failure_risk

sample_size
effective_sample_size

recency
modernity

stability
contradiction_rate
failure_rate
recovery_rate

transfer_score
cross_era_score
cross_stock_score
cross_sector_score
cross_regime_score

mechanism_tags

supporting_knowledge
contradicting_knowledge
parent_knowledge
child_knowledge
complementary_knowledge
redundant_knowledge

failure_explanations

decision_effect
action_policy

promotion_state
lifecycle_state

provenance
code_hash
data_hash
experiment_id
```

Do not implement this as one giant untyped dictionary.

Use explicit types and validation.

### Minimum implementation depth: 900 meaningful lines.

---

# 6. EPISTEMIC STATES

Every learned item must have an epistemic state.

Required states:

```text
OBSERVED
HYPOTHESIS
SUPPORTED
CONDITIONAL
DEGRADED
CONTRADICTED
GATED
RETIRED
UNKNOWN
```

Definitions must be enforced.

Example:

A pattern that worked historically but fails in the current regime is NOT automatically false.

It may become:

```text
SUPPORTED historically
CONDITIONAL by regime
DEGRADED currently
```

The system must preserve that distinction.

### Minimum implementation depth: 450 lines.

---

# 7. OBSERVATION ≠ INTERPRETATION

Never allow:

```text
price rose after pattern X
```

to automatically become:

```text
pattern X causes price to rise
```

Store separately:

1. observation;
2. statistical relationship;
3. hypothesis;
4. mechanism hypothesis;
5. decision usefulness.

The system must support multiple competing explanations.

Example:

```text
H1: pattern genuinely predicts movement
H2: pattern is proxying for volatility
H3: pattern is regime-specific
H4: pattern is redundant with another feature
H5: apparent effect is selection bias
H6: effect is unstable
```

The learner must be able to retain uncertainty between them.

### Minimum implementation depth: 600 lines.

---

# 8. CONTEXT IS PART OF KNOWLEDGE

Never store only:

```text
pattern → result
```

Store:

```text
pattern
+
context
+
outcome
```

Learn:

```text
P(outcome | pattern, context)
```

and also:

```text
P(outcome | pattern, NOT context)
```

The second is essential.

Otherwise the system will repeatedly apply a pattern outside the conditions where it works.

Required context dimensions:

* market;
* sector;
* stock type;
* volatility;
* liquidity;
* trend;
* breadth;
* macro state;
* regime;
* recent shock;
* correlation structure;
* time horizon;
* seasonality where legitimate;
* pattern interaction;
* position/risk context.

Prevent context explosion with hierarchical pooling/shrinkage.

Do NOT create millions of tiny buckets with no statistical power.

### Minimum implementation depth: 1,200 lines.

---

# 9. FAILURE LEARNING

Failures are first-class training data.

A loss is not simply:

```text
bad trade
```

The learner must determine whether the failure indicates:

```text
FALSE_PATTERN
TEMPORARY_INACTIVITY
WRONG_CONTEXT
REGIME_CHANGE
WEAKENING_EFFECT
REVERSAL
MEASUREMENT_ERROR
REDUNDANCY
SELECTION_ERROR
TIMING_ERROR
RISK_ERROR
INTERACTION_FAILURE
INSUFFICIENT_EVIDENCE
UNKNOWN
```

Do not force a cause.

If evidence is insufficient:

```text
UNKNOWN
```

is the correct result.

### Minimum implementation depth: 1,100 lines.

---

# 10. PATTERN BREAK ENGINE

For every important pattern that degrades, compare:

### Successful population

against

### Failed population

Compare:

* feature distributions;
* market regime;
* sector composition;
* stock type;
* volatility;
* liquidity;
* trend;
* breadth;
* macro context;
* neighboring patterns;
* interaction structure;
* outcome magnitude;
* outcome duration;
* timing;
* concentration;
* sample size;
* missingness;
* data quality.

Determine whether the difference is statistically meaningful.

Then create candidate conditions explaining the break.

Test those conditions out of sample.

### Minimum implementation depth: 1,300 lines.

---

# 11. DYNAMIC RELIABILITY

Do NOT use one permanent confidence number.

Maintain separate dimensions:

```text
truth_confidence
current_reliability
transfer_confidence
context_confidence
failure_risk
```

A pattern may be:

```text
high historical truth confidence
low current reliability
high transfer confidence
high failure risk
```

That should cause a different decision from:

```text
moderate historical confidence
high current reliability
low transfer confidence
low failure risk
```

Reliability must update with new evidence.

### Minimum implementation depth: 900 lines.

---

# 12. LEARN WHEN KNOWLEDGE STOPS WORKING

For every active pattern maintain:

```text
birth
growth
peak
decay
failure
recovery
retirement
```

Estimate whether deterioration is:

* random;
* gradual;
* abrupt;
* regime-linked;
* market-linked;
* sector-linked;
* volatility-linked;
* liquidity-linked;
* interaction-linked.

Do not assume decay is permanent.

A pattern may recover.

### Minimum implementation depth: 800 lines.

---

# 13. RETIRE ≠ DELETE

Never destroy useful historical knowledge merely because it is currently inactive.

Use:

```text
ACTIVE
DEGRADED
DORMANT
RETIRED
```

instead of immediate deletion.

Historical evidence remains available for research.

But dormant knowledge must not silently influence live decisions.

Recovery must require evidence.

### Minimum implementation depth: 450 lines.

---

# 14. TIME-AWARE MEMORY

Every knowledge item needs a learned temporal behavior.

Possible classes:

```text
PERSISTENT
SLOW_DECAY
FAST_DECAY
EPISODIC
REGIME_BOUND
EVENT_BOUND
SEASONAL
UNKNOWN
```

Do not assume every half-life.

Estimate decay from evidence.

Learn:

```text
expected useful lifetime
uncertainty of lifetime
conditions for recovery
```

### Minimum implementation depth: 700 lines.

---

# 15. SITUATION REPRESENTATION

Build an identity-independent situation representation.

The representation must describe:

```text
what is happening
```

without encoding:

```text
which exact stock
which exact date
which exact year
```

unless those identities are legitimately relevant and available at decision time.

Situation representation must be constructed only from information available at the decision timestamp.

Required tests:

* ticker scrambling;
* date scrambling;
* year scrambling;
* sector scrambling where appropriate;
* equivalent-situation transfer;
* identity memorization control.

### Minimum implementation depth: 1,000 lines.

---

# 16. SITUATION SIMILARITY

Build multi-factor similarity.

Similarity must consider:

```text
structural similarity
market similarity
volatility similarity
liquidity similarity
pattern similarity
regime similarity
sector similarity
stock-type similarity
recent-history similarity
failure-risk similarity
```

Similarity must NOT be a single opaque number.

Store component scores.

The learner must know:

```text
why two situations were considered similar.
```

### Minimum implementation depth: 900 lines.

---

# 17. KNOWLEDGE RETRIEVAL

Retrieval should rank knowledge using at least:

```text
situation_similarity
context_match
temporal_relevance
current_reliability
transfer_confidence
modernity
sample_quality
failure_safety
contradiction_penalty
redundancy_penalty
```

Retrieval must be explainable.

The system must be able to output:

```text
Retrieved K123 because:
- situation similarity = ...
- context match = ...
- current reliability = ...
- transfer evidence = ...
- failure risk = ...
```

### Minimum implementation depth: 1,100 lines.

---

# 18. CONTRADICTION GRAPH

Build explicit knowledge relationships.

Required edges:

```text
SUPPORTS
CONTRADICTS
CONTAINS
SPECIALIZES
GENERALIZES
CAUSES_FAILURE_OF
RECOVERS_WITH
REDUNDANT_WITH
COMPLEMENTS
DEPENDS_ON
```

Contradictions must not simply average together.

The learner must investigate:

```text
Which context explains the disagreement?
```

### Minimum implementation depth: 850 lines.

---

# 19. KNOWLEDGE HIERARCHY

Implement hierarchical knowledge:

```text
general rule
    ↓
market condition
    ↓
sector condition
    ↓
stock-type condition
    ↓
volatility condition
    ↓
specific interaction
```

Specific knowledge should override general knowledge only when it has enough evidence.

Otherwise shrink toward the parent rule.

This prevents:

```text
tiny sample → extreme confidence
```

### Minimum implementation depth: 900 lines.

---

# 20. CREDIT ASSIGNMENT

When a decision succeeds or fails, determine what actually deserves credit.

A decision may depend on:

```text
pattern A
pattern B
analog C
memory D
direction model E
timing F
risk model G
```

Do not give every component credit.

Perform ablations/counterfactuals where computationally feasible.

Measure:

```text
with component
without component
difference
interaction
stability
```

### Minimum implementation depth: 1,200 lines.

---

# 21. REDUNDANCY INTELLIGENCE

Do not assume:

```text
correlated = useless
```

Distinguish:

```text
predictive redundancy
context redundancy
mechanistic redundancy
operational redundancy
risk redundancy
```

Two patterns may provide the same prediction but different regime protection.

Learn that distinction.

### Minimum implementation depth: 600 lines.

---

# 22. MISSED-WINNER LEARNING

A missed winner is not simply:

```text
missed opportunity
```

Ask:

```text
Why did the system reject it?
```

Possible reasons:

```text
wrong ranking
wrong confidence
wrong context
bad reliability
missing interaction
over-aggressive risk filter
direction disagreement
timing
insufficient evidence
unknown
```

Then determine what distinction would have separated the missed winner from the false positives.

Validate that distinction out of sample.

### Minimum implementation depth: 1,000 lines.

---

# 23. SELECTION / TIMING / RISK SEPARATION

Never let one failure teach the wrong subsystem.

Separate:

```text
selection failure
timing failure
direction failure
risk failure
exit failure
```

Example:

A stock can be correctly selected but poorly timed.

That should not teach:

```text
stock selection is bad.
```

### Minimum implementation depth: 700 lines.

---

# 24. LEARNING FROM LOSSES

For every meaningful loss create a structured postmortem.

Required:

```text
what was believed
why it was believed
what evidence supported it
what evidence contradicted it
what happened
what was surprising
what subsystem made the decision
what knowledge influenced it
what knowledge should not have influenced it
what condition was missing
what condition invalidated the belief
what should change
confidence in the explanation
```

No postmortem may directly modify production behavior.

It first becomes a hypothesis.

### Minimum implementation depth: 800 lines.

---

# 25. SAME-YEAR RERUN LEARNING — C57

This is a central requirement.

Run the same year repeatedly.

The learner must NOT know:

```text
this is a rerun
```

and must NOT know:

```text
this is the same year.
```

The year must be disguised.

Measure:

```text
run 1
run 2
run 3
...
run N
```

The expected learning curve must be measured.

Required controls:

### CONTROL A

No learning.

### CONTROL B

Legitimate learner.

### CONTROL C

Identity memorizer.

### CONTROL D

Random learner.

### CONTROL E

Future-information/leaky learner.

The legitimate learner must outperform its own baseline without resembling the identity memorizer.

### Minimum implementation depth: 1,300 lines.

---

# 26. UNSEEN-YEAR TRANSFER

Same-year improvement is not enough.

Every meaningful learned rule must eventually be tested across:

* different years;
* different regimes;
* different market conditions;
* different sectors;
* different stock types;
* different volatility states.

Track:

```text
same_year_gain
cross_year_gain
cross_regime_gain
cross_stock_gain
cross_sector_gain
```

### Minimum implementation depth: 1,100 lines.

---

# 27. TRANSFER RATIO

Calculate:

```text
transfer_ratio =
cross_context_gain / same_context_gain
```

with appropriate safeguards for zero/negative denominators.

A learner that dramatically improves the training context but fails elsewhere should be identified as over-specialized.

### Minimum implementation depth: 400 lines.

---

# 28. MEMORY CONTAMINATION FIREWALL

Memory must have strict provenance.

Every learned item must identify:

```text
when it was created
what data it could access
what code created it
what experiment created it
what outcomes it had seen
what evaluation windows were sealed
```

A future-memory audit must be able to answer:

> Could this item have existed at the exact decision timestamp?

If not:

```text
REJECT
```

### Minimum implementation depth: 900 lines.

---

# 29. IDENTITY MEMORIZATION FIREWALL

Explicitly attack memorization.

Run:

```text
ticker permutation
date permutation
year disguise
stock substitution
sector substitution
episode substitution
sequence scrambling
```

If performance collapses under identity-preserving transformations, investigate.

### Minimum implementation depth: 750 lines.

---

# 30. FUTURE INFORMATION FIREWALL

Every learning input must pass:

```text
timestamp check
data availability check
revision check
survivorship check
feature provenance check
memory provenance check
code-version check
network/cache check
```

The system must fail closed.

A suspicious input is better rejected than silently accepted.

### Minimum implementation depth: 1,000 lines.

---

# 31. EXPERIMENT MEMORY

Every experiment must answer:

```text
QUESTION
CURRENT BELIEF
COMPETING HYPOTHESES
PREDICTION
EXPERIMENT
EXPECTED OUTCOMES
RESULT
BELIEF UPDATE
WHAT WAS LEARNED
WHAT WAS NOT LEARNED
NEXT ACTION
```

Do not repeatedly run experiments that already answered the same question.

The system should know:

```text
we already tested this.
```

### Minimum implementation depth: 800 lines.

---

# 32. BELIEF UPDATING

Implement evidence-weighted belief updates.

Do not simply overwrite:

```text
belief = new_result
```

Instead maintain:

```text
prior
new evidence
likelihood / evidence strength
posterior
uncertainty
sample size
contradictory evidence
```

The implementation may use Bayesian-style mathematics without requiring every component to be formally Bayesian.

The important requirement is:

**new evidence changes belief proportionally to evidence quality.**

### Minimum implementation depth: 800 lines.

---

# 33. THREE SEPARATE QUESTIONS

Every important learned relationship must answer:

### 1. IS IT REAL?

Statistical reality.

### 2. IS IT USEFUL?

Incremental decision value.

### 3. IS IT USEFUL NOW?

Current applicability.

These must never collapse into one score.

### Minimum implementation depth: 600 lines.

---

# 34. PREDICTIVE VALUE VS PORTFOLIO VALUE

A statistically predictive pattern may not improve the portfolio.

Measure separately:

```text
predictive effect
movement prediction
ranking value
selection value
direction value
timing value
risk value
portfolio value
```

The existing WEEKLY7 tiered objective remains authoritative.

Do not force direction to dominate simply because it is difficult.

### Minimum implementation depth: 850 lines.

---

# 35. EXPLORATION VS EXPLOITATION

The learner must decide how research resources are allocated.

Possible research targets:

```text
known reliable knowledge
contradictions
failures
missed winners
regime transitions
weak patterns
unknown areas
interactions
new representations
data-quality issues
```

Do not spend all compute tuning already-known parameters.

### Minimum implementation depth: 750 lines.

---

# 36. INFORMATION-GAIN RESEARCH POLICY

Choose the next experiment using an explicit priority function.

Conceptually:

```text
priority =
expected_information_gain
× relevance
× uncertainty
× transfer_potential
× feasibility
× expected_decision_value
```

Penalize:

```text
duplicate experiments
known-low-value experiments
overfit-prone experiments
unavailable data
computationally excessive experiments
```

### Minimum implementation depth: 900 lines.

---

# 37. META-LEARNING

Eventually WEEKLY7 must learn about **its own learning process**.

Examples:

```text
which discoveries tend to survive OOS?
which pattern families overfit?
which contexts transfer?
which memory types decay rapidly?
which experiments produce useful knowledge?
which learners produce fake improvement?
which failure explanations tend to be correct?
which feature families repeatedly fail ablation?
```

Meta-learning must itself be evaluated out of sample.

### Minimum implementation depth: 1,200 lines.

---

# 38. LEARN FROM FAILED LEARNERS

A failed learning architecture is still evidence.

Store:

```text
learner
hypothesis
implementation
failure mode
data regime
validation result
reason for failure
whether failure generalized
```

The next learner should not repeat known dead ends without justification.

### Minimum implementation depth: 500 lines.

---

# 39. KNOWLEDGE COMPETITION

Allow multiple explanations to compete.

Example:

```text
Pattern A causes movement
Pattern A proxies volatility
Pattern A only works in high-liquidity stocks
Pattern A only works during trend expansion
```

Do not prematurely select one.

Evidence determines survival.

### Minimum implementation depth: 700 lines.

---

# 40. COMPLEXITY PENALTY

A more complicated explanation must earn its complexity.

If:

```text
simple rule = same OOS performance
complex rule = same OOS performance
```

prefer the simpler representation.

But do not reject complexity merely because it is complex.

Measure:

```text
incremental gain
transfer
stability
failure safety
complexity
```

### Minimum implementation depth: 500 lines.

---

# 41. BOUNDARY LEARNING

Do not only learn:

```text
when a pattern works.
```

Learn:

```text
when it stops working.
```

Learn boundaries such as:

```text
volatility > threshold
liquidity < threshold
regime transition
pattern disagreement
market shock
sector concentration
confidence deterioration
```

The boundary itself becomes knowledge.

### Minimum implementation depth: 700 lines.

---

# 42. UNKNOWN STATE

The learner must explicitly support:

```text
UNKNOWN
INSUFFICIENT_DATA
CONFLICTED
UNTESTED
```

These states must not be converted into false confidence.

The correct action may be:

```text
ABSTAIN
COLLECT DATA
RUN EXPERIMENT
USE GENERAL RULE
```

### Minimum implementation depth: 400 lines.

---

# 43. KNOWLEDGE → DECISION CONTRACT

Every active knowledge object must answer:

> What decision does this change?

Possible answers:

```text
ranking
selection
position size
direction
timing
exit
stop
abstention
research priority
pattern weighting
confidence
```

If a learned item cannot change a decision, it is not production knowledge.

It may remain research knowledge.

### Minimum implementation depth: 450 lines.

---

# 44. CHAMPION / CHALLENGER KNOWLEDGE

Never immediately replace production knowledge.

Maintain:

```text
CHAMPION
CHALLENGER
SHADOW
RETIRED
```

A challenger must demonstrate:

* OOS improvement;
* transfer;
* acceptable risk;
* no memorization;
* stability;
* reproducibility.

Only then can it replace the champion.

### Minimum implementation depth: 700 lines.

---

# 45. KNOWLEDGE PROMOTION GATE

Production promotion requires ALL applicable gates:

```text
statistical validity
incremental value
OOS confirmation
cross-context transfer
risk acceptance
anti-memorization
future-information audit
reproducibility
stability
provenance completeness
```

One critical failure blocks promotion.

### Minimum implementation depth: 700 lines.

---

# 46. KNOWLEDGE HEALTH MONITOR

Continuously track:

```text
healthy
degrading
broken
contradicted
dormant
recovering
unstable
insufficient evidence
unknown
```

The dashboard must show:

* what is currently trusted;
* what is losing trust;
* why;
* what evidence caused the change;
* what research is investigating it.

### Minimum implementation depth: 650 lines.

---

# 47. LEARNING SCORECARD

Every learner version must produce:

```text
baseline performance
post-learning performance
same-year gain
cross-year gain
cross-regime gain
cross-stock gain
transfer ratio
risk change
drawdown change
5–10% band share
movement performance
direction performance
mover performance
calibration
memorization gap
identity gap
future-leak status
stability
compute cost
```

Never report "learning improved" without the corresponding controls.

### Minimum implementation depth: 700 lines.

---

# 48. LEARNING CURVE

Track:

```text
experience count
knowledge count
validated knowledge count
same-year gain
transfer gain
risk
memorization gap
```

against experience.

The important graph is:

```text
experience → future improvement
```

not:

```text
experience → amount of stored memory
```

### Minimum implementation depth: 500 lines.

---

# 49. KNOWLEDGE ARCHIVE

Create permanent immutable raw experience storage.

Never destroy historical evidence.

Dynamic behavior comes from changing influence, not deleting history.

Required layers:

```text
L0 raw observations
L1 events
L2 episodes
L3 situations
L4 hypotheses
L5 patterns
L6 contextual rules
L7 validated knowledge
L8 decision policies
L9 meta-learning
```

### Minimum implementation depth: 1,000 lines.

---

# 50. KNOWLEDGE GRAPH

Build a graph connecting:

```text
experiences
situations
patterns
hypotheses
failures
conditions
experiments
knowledge
decisions
outcomes
```

The graph should allow questions such as:

> What knowledge caused this decision?

> What failures have contradicted this pattern?

> Which situations have successfully transferred this knowledge?

> Which experiments validated this belief?

### Minimum implementation depth: 1,000 lines.

---

# 51. RESEARCH PRIORITY ENGINE

The system must autonomously decide:

> What should I investigate next?

Prioritize problems with:

```text
high uncertainty
high potential decision impact
high transfer potential
strong evidence of failure
large contradiction
large unexplained surprise
reasonable computational cost
```

The research queue must evolve from experience.

### Minimum implementation depth: 900 lines.

---

# 52. SURPRISE ENGINE

Measure:

```text
expected outcome
actual outcome
surprise magnitude
surprise direction
surprise persistence
```

High-surprise situations trigger investigation.

Repeated surprises from similar situations should increase research priority.

### Minimum implementation depth: 500 lines.

---

# 53. DISAGREEMENT ENGINE

Disagreement between:

```text
pattern miner
analog engine
memory
direction model
movement model
risk model
```

is information.

Measure:

```text
who disagreed
why
which one was correct
under what context
```

Then learn when disagreement itself is predictive.

### Minimum implementation depth: 650 lines.

---

# 54. CALIBRATION

Confidence must correspond to reality.

Track:

```text
predicted confidence
actual success
calibration error
confidence drift
context-specific calibration
```

If the learner becomes overconfident, reduce influence.

### Minimum implementation depth: 650 lines.

---

# 55. LEARNING FIREWALLS

Implement independent firewall layers:

```text
DATA FIREWALL
TIME FIREWALL
MEMORY FIREWALL
IDENTITY FIREWALL
PROVENANCE FIREWALL
EXPERIMENT FIREWALL
EVALUATION FIREWALL
CODE-VERSION FIREWALL
```

A learning result must pass all relevant firewalls before promotion.

### Minimum implementation depth: 1,100 lines.

---

# 56. REPRODUCIBILITY

Every learning experiment must be reproducible from:

```text
code hash
data hash
configuration hash
random seed
worker configuration
memory snapshot
experiment ID
```

If any component differs, the result must be marked accordingly.

### Minimum implementation depth: 500 lines.

---

# 57. COMPUTE RESOURCE MANAGER

The learning engine must support parallel research without corrupting learning state.

Required:

* deterministic workers;
* isolated experiments;
* memory snapshotting;
* checkpointing;
* crash recovery;
* OOM recovery;
* stale-code detection;
* result reconciliation;
* duplicate experiment prevention.

### Minimum implementation depth: 900 lines.

---

# 58. NEVER-STOP EXECUTION ENGINE

Claude must not stop merely because:

* one section is complete;
* tests pass;
* a report looks good;
* performance improves;
* the system reaches an arbitrary score;
* a milestone is reached;
* the codebase becomes large;
* the first implementation works;
* a promising pattern appears.

Claude may stop only when:

1. every required implementation section is complete;
2. every minimum line-depth requirement has been satisfied with meaningful code;
3. every required test exists;
4. every required firewall exists;
5. every required report exists;
6. the master checklist is complete;
7. all unresolved failures are either fixed or explicitly escalated as genuine scientific limitations;
8. no checklist item remains falsely marked complete.

If work is interrupted:

```text
SAVE STATE
WRITE CHECKPOINT
WRITE NEXT ACTION
WRITE CURRENT FAILURES
WRITE CURRENT EXPERIMENT
WRITE CURRENT CODE HASH
CONTINUE FROM CHECKPOINT
```

---

# 59. NO FALSE COMPLETION

Claude must never say:

```text
done
complete
finished
production ready
validated
learning works
```

unless the evidence exists.

If a section is implemented but not validated:

```text
IMPLEMENTED — NOT VALIDATED
```

If validation fails:

```text
IMPLEMENTED — FAILED VALIDATION
```

If insufficient data exists:

```text
IMPLEMENTED — INSUFFICIENT EVIDENCE
```

Never convert these states into success.

---

# 60. MINIMUM CODE BUDGET

The following are **minimum meaningful implementation sizes**.

They are not targets to pad toward.

| Section                             | Minimum meaningful lines |
| ----------------------------------- | -----------------------: |
| Knowledge objects                   |                      900 |
| Epistemic states                    |                      450 |
| Observation/interpretation          |                      600 |
| Context modeling                    |                    1,200 |
| Failure learning                    |                    1,100 |
| Pattern-break engine                |                    1,300 |
| Dynamic reliability                 |                      900 |
| Pattern lifecycle                   |                      800 |
| Retirement/recovery                 |                      450 |
| Time-aware memory                   |                      700 |
| Situation representation            |                    1,000 |
| Situation similarity                |                      900 |
| Knowledge retrieval                 |                    1,100 |
| Contradiction graph                 |                      850 |
| Knowledge hierarchy                 |                      900 |
| Credit assignment                   |                    1,200 |
| Redundancy intelligence             |                      600 |
| Missed-winner learning              |                    1,000 |
| Selection/timing/risk separation    |                      700 |
| Loss postmortems                    |                      800 |
| Same-year learning harness          |                    1,300 |
| Cross-year transfer                 |                    1,100 |
| Transfer scoring                    |                      400 |
| Memory firewall                     |                      900 |
| Identity firewall                   |                      750 |
| Future-information firewall         |                    1,000 |
| Experiment memory                   |                      800 |
| Belief updating                     |                      800 |
| Real/useful/current separation      |                      600 |
| Portfolio-value learning            |                      850 |
| Exploration/exploitation            |                      750 |
| Information-gain research           |                      900 |
| Meta-learning                       |                    1,200 |
| Failed-learner memory               |                      500 |
| Knowledge competition               |                      700 |
| Complexity control                  |                      500 |
| Boundary learning                   |                      700 |
| Unknown handling                    |                      400 |
| Knowledge→decision contract         |                      450 |
| Champion/challenger                 |                      700 |
| Promotion gates                     |                      700 |
| Knowledge health                    |                      650 |
| Learning scorecard                  |                      700 |
| Learning curve                      |                      500 |
| Knowledge archive                   |                    1,000 |
| Knowledge graph                     |                    1,000 |
| Research priority                   |                      900 |
| Surprise engine                     |                      500 |
| Disagreement engine                 |                      650 |
| Calibration                         |                      650 |
| Learning firewalls                  |                    1,100 |
| Reproducibility                     |                      500 |
| Compute manager                     |                      900 |
| Continuous execution/checkpointing  |                      700 |
| Reporting/dashboard/inspection      |                    1,500 |
| Test infrastructure for the above   |                    8,000 |
| Integration/refactoring/type safety |                    4,000 |

## MINIMUM EXPECTED NEW/DEEPENED IMPLEMENTATION

Approximately:

# 55,000–65,000 meaningful lines

This is an architectural estimate, not a license to generate filler.

If the implementation requires 70,000 lines, write 70,000.

If it requires 80,000, write 80,000.

**Never remove necessary functionality merely to hit a smaller number.**

But also:

**Never manufacture meaningless code merely to hit a larger number.**

A section below its minimum is presumed incomplete until you demonstrate why its functionality is fully implemented elsewhere and the equivalent implementation depth genuinely exists there.

---

# 61. REQUIRED CODE STRUCTURE

Create or extend modules along these conceptual boundaries:

```text
engine/learning/
    knowledge.py
    epistemic.py
    situation.py
    similarity.py
    retrieval.py
    reliability.py
    lifecycle.py
    failure.py
    break_detection.py
    transfer.py
    credit.py
    missed_winners.py
    belief.py
    contradiction.py
    hierarchy.py
    redundancy.py
    calibration.py
    surprise.py
    disagreement.py
    research_policy.py
    experiment_memory.py
    meta_learning.py
    knowledge_graph.py
    archive.py
    promotion.py
    health.py
    learner.py
    checkpoints.py
```

Names may differ if the existing architecture requires it.

Do not create duplicate parallel systems merely to satisfy filenames.

---

# 62. REQUIRED TEST ARCHITECTURE

Every learning subsystem needs tests at:

```text
unit
integration
property
determinism
negative/control
out-of-sample
anti-leakage
anti-memorization
```

The most important tests are adversarial.

Examples:

### Test 1

Plant a real pattern.

Learner must discover it.

### Test 2

Plant a fake pattern.

Learner must reject it.

### Test 3

Plant a decaying pattern.

Learner must eventually detect degradation.

### Test 4

Plant a regime-specific pattern.

Learner must learn the condition.

### Test 5

Reverse the pattern.

Learner must detect contradiction.

### Test 6

Duplicate the pattern.

Learner must recognize redundancy.

### Test 7

Hide the pattern from the evaluation period.

Learner must not use it.

### Test 8

Change ticker identities.

Performance should not collapse solely because identities changed.

### Test 9

Change years.

Knowledge should transfer when the situation is genuinely equivalent.

### Test 10

Give the learner a memorizer.

The legitimate learner must remain distinguishable from it.

### Test 11

Give the learner future information.

Firewall must reject it.

### Test 12

Change code during a worker run.

The stale result must not be accepted.

---

# 63. PLANTED-PATTERN CALIBRATION

Maintain a synthetic truth environment containing:

```text
strong pattern
weak pattern
negative pattern
pair interaction
regime pattern
decaying pattern
conditional pattern
unless pattern
noise-only pattern
hallucinated discovery-only pattern
```

The learner must correctly distinguish these.

Do not optimize solely against planted tests.

Real data remains the final authority.

---

# 64. REAL-DATA VALIDATION

After synthetic validation:

1. run on real historical data;
2. use point-in-time information;
3. use surviving and delisted securities where available;
4. prevent future revisions;
5. preserve provenance;
6. freeze evaluation windows;
7. compare learner vs no-learning control;
8. compare legitimate learner vs memorizer;
9. compare transfer;
10. inspect risk.

The real-data result is the primary evidence.

---

# 65. LEARNING DELTA

Define:

```text
learning_delta =
post_learning_performance
-
no_learning_performance
```

But report it across multiple dimensions.

At minimum:

```text
movement_delta
selection_delta
direction_delta
risk_delta
drawdown_delta
band_share_delta
transfer_delta
calibration_delta
```

A positive delta in one dimension must not be presented as proof that the whole learning system works.

---

# 66. WHAT SUCCESS ACTUALLY MEANS

The target is NOT:

> Make historical backtests look better.

The target is:

> Make WEEKLY7 encounter a situation it has learned from before and make a better decision because of that prior experience, without knowing the answer in advance.

The strongest evidence is:

```text
same situation class
different identity
different date
different year
different market context
legitimate prior experience
better later decision
```

---

# 67. 7% TARGET HANDLING

The 7% weekly target remains a research target.

Do not directly force the learner to manufacture 7%.

Do not optimize exclusively for historical 7% achievement.

Do not declare success because a backtest reaches 7%.

The learning system must first establish:

```text
real information
transfer
risk control
stability
reproducibility
```

Then investigate whether those properties can support higher performance.

---

# 68. FINAL MASTER CHECKLIST

Claude MUST maintain a machine-readable checklist.

Recommended:

```text
state/build/SELF_LEARNING_MASTER_CHECKLIST.json
```

Every item must contain:

```text
id
description
owner
status
code_paths
tests
evidence
minimum_lines
actual_lines
validation_result
last_run
code_hash
notes
```

Allowed statuses:

```text
NOT_STARTED
IN_PROGRESS
IMPLEMENTED
TESTING
VALIDATED
FAILED
BLOCKED
```

Do not allow:

```text
DONE
```

unless the item's acceptance criteria are satisfied.

---

# 69. MASTER CHECKLIST — PHASE A: FOUNDATION

* [ ] A01 Define KnowledgeObject.
* [ ] A02 Define epistemic states.
* [ ] A03 Define immutable experience records.
* [ ] A04 Define situation representation.
* [ ] A05 Define provenance schema.
* [ ] A06 Define knowledge versioning.
* [ ] A07 Define lifecycle states.
* [ ] A08 Define confidence dimensions.
* [ ] A09 Define reliability dimensions.
* [ ] A10 Implement serialization/deserialization.
* [ ] A11 Implement schema validation.
* [ ] A12 Add deterministic hashing.
* [ ] A13 Add code/data/config provenance.

---

# 70. MASTER CHECKLIST — PHASE B: MEMORY

* [ ] B01 Build raw experience archive.
* [ ] B02 Build episode archive.
* [ ] B03 Build situation archive.
* [ ] B04 Build knowledge archive.
* [ ] B05 Build retrieval index.
* [ ] B06 Build context index.
* [ ] B07 Build temporal index.
* [ ] B08 Build reliability index.
* [ ] B09 Build failure index.
* [ ] B10 Build contradiction index.
* [ ] B11 Build recovery index.
* [ ] B12 Implement memory snapshots.
* [ ] B13 Implement immutable historical records.
* [ ] B14 Implement dynamic influence.
* [ ] B15 Implement memory audit.

---

# 71. MASTER CHECKLIST — PHASE C: LEARNING

* [ ] C01 Implement observe stage.
* [ ] C02 Implement situation description.
* [ ] C03 Implement retrieval.
* [ ] C04 Implement expectation formation.
* [ ] C05 Implement decision capture.
* [ ] C06 Implement outcome capture.
* [ ] C07 Implement surprise detection.
* [ ] C08 Implement credit assignment.
* [ ] C09 Implement blame assignment.
* [ ] C10 Implement belief updating.
* [ ] C11 Implement reliability updating.
* [ ] C12 Implement condition discovery.
* [ ] C13 Implement anti-condition discovery.
* [ ] C14 Implement transfer evaluation.
* [ ] C15 Implement knowledge storage.
* [ ] C16 Implement meta-learning update.
* [ ] C17 Implement research-priority update.

---

# 72. MASTER CHECKLIST — PHASE D: FAILURE LEARNING

* [ ] D01 Loss classifier.
* [ ] D02 Selection failure detector.
* [ ] D03 Timing failure detector.
* [ ] D04 Direction failure detector.
* [ ] D05 Risk failure detector.
* [ ] D06 Pattern failure detector.
* [ ] D07 Context failure detector.
* [ ] D08 Regime failure detector.
* [ ] D09 Measurement failure detector.
* [ ] D10 Unknown failure state.
* [ ] D11 Pattern-break analysis.
* [ ] D12 Recovery analysis.
* [ ] D13 Missed-winner analysis.
* [ ] D14 Counterfactual distinction discovery.
* [ ] D15 OOS validation of failure lessons.

---

# 73. MASTER CHECKLIST — PHASE E: TRANSFER

* [ ] E01 Same-year rerun harness.
* [ ] E02 Disguised rerun mechanism.
* [ ] E03 No-learning control.
* [ ] E04 Legitimate learner.
* [ ] E05 Identity memorizer control.
* [ ] E06 Random learner control.
* [ ] E07 Leaky learner control.
* [ ] E08 Cross-year test.
* [ ] E09 Cross-regime test.
* [ ] E10 Cross-stock test.
* [ ] E11 Cross-sector test.
* [ ] E12 Cross-volatility test.
* [ ] E13 Transfer ratio.
* [ ] E14 Memorization gap.
* [ ] E15 Identity gap.
* [ ] E16 Transfer stability.

---

# 74. MASTER CHECKLIST — PHASE F: KNOWLEDGE GRAPH

* [ ] F01 Supports edges.
* [ ] F02 Contradicts edges.
* [ ] F03 Contains edges.
* [ ] F04 Specializes edges.
* [ ] F05 Generalizes edges.
* [ ] F06 Causes-failure edges.
* [ ] F07 Recovers-with edges.
* [ ] F08 Redundant-with edges.
* [ ] F09 Complements edges.
* [ ] F10 Depends-on edges.
* [ ] F11 Graph traversal.
* [ ] F12 Contradiction investigation.
* [ ] F13 Parent/child shrinkage.
* [ ] F14 Knowledge lineage.
* [ ] F15 Decision lineage.

---

# 75. MASTER CHECKLIST — PHASE G: RESEARCH INTELLIGENCE

* [ ] G01 Experiment registry.
* [ ] G02 Question generation.
* [ ] G03 Competing hypothesis generation.
* [ ] G04 Expected outcome generation.
* [ ] G05 Information-gain scoring.
* [ ] G06 Research priority queue.
* [ ] G07 Duplicate experiment detection.
* [ ] G08 Failed experiment memory.
* [ ] G09 Failed learner memory.
* [ ] G10 Exploration/exploitation policy.
* [ ] G11 Compute-aware research allocation.
* [ ] G12 Meta-learning research policy.

---

# 76. MASTER CHECKLIST — PHASE H: FIREWALLS

* [ ] H01 Point-in-time firewall.
* [ ] H02 Future-data firewall.
* [ ] H03 Memory firewall.
* [ ] H04 Identity firewall.
* [ ] H05 Provenance firewall.
* [ ] H06 Code-version firewall.
* [ ] H07 Revision-data firewall.
* [ ] H08 Survivorship firewall.
* [ ] H09 Worker-staleness firewall.
* [ ] H10 Evaluation contamination firewall.
* [ ] H11 Cache/network firewall.
* [ ] H12 Fail-closed behavior.

---

# 77. MASTER CHECKLIST — PHASE I: VALIDATION

* [ ] I01 Unit tests.
* [ ] I02 Integration tests.
* [ ] I03 Property tests.
* [ ] I04 Determinism tests.
* [ ] I05 Negative tests.
* [ ] I06 Planted strong pattern.
* [ ] I07 Planted weak pattern.
* [ ] I08 Planted negative pattern.
* [ ] I09 Planted pair.
* [ ] I10 Planted regime pattern.
* [ ] I11 Planted decaying pattern.
* [ ] I12 Planted conditional pattern.
* [ ] I13 Planted unless pattern.
* [ ] I14 Planted noise pattern.
* [ ] I15 Hallucinated pattern.
* [ ] I16 Real-data validation.
* [ ] I17 Fresh holdout seeds.
* [ ] I18 Cross-year validation.
* [ ] I19 Memorization control.
* [ ] I20 Future-leak control.

---

# 78. MASTER CHECKLIST — PHASE J: PRODUCTION INTELLIGENCE

* [ ] J01 Champion knowledge.
* [ ] J02 Challenger knowledge.
* [ ] J03 Shadow knowledge.
* [ ] J04 Promotion gate.
* [ ] J05 Retirement gate.
* [ ] J06 Recovery gate.
* [ ] J07 Reliability monitoring.
* [ ] J08 Calibration monitoring.
* [ ] J09 Contradiction monitoring.
* [ ] J10 Surprise monitoring.
* [ ] J11 Knowledge health.
* [ ] J12 Decision impact monitoring.
* [ ] J13 Learning curve.
* [ ] J14 Learning scorecard.
* [ ] J15 Research priority dashboard.

---

# 79. MASTER CHECKLIST — PHASE K: REPORTING

* [ ] K01 Learning curve report.
* [ ] K02 Transfer report.
* [ ] K03 Memorization report.
* [ ] K04 Identity-gap report.
* [ ] K05 Knowledge health report.
* [ ] K06 Failure report.
* [ ] K07 Recovery report.
* [ ] K08 Contradiction report.
* [ ] K09 Missed-winner report.
* [ ] K10 Experiment-memory report.
* [ ] K11 Research-priority report.
* [ ] K12 Meta-learning report.
* [ ] K13 Promotion/rejection report.
* [ ] K14 Provenance audit report.
* [ ] K15 Full learning-system report.

---

# 80. MASTER CHECKLIST — PHASE L: FINAL SCIENTIFIC VALIDATION

These are the final gates.

* [ ] L01 No-learning baseline frozen.
* [ ] L02 Legitimate learner frozen.
* [ ] L03 Memorizer control frozen.
* [ ] L04 Random learner control frozen.
* [ ] L05 Leaky learner control frozen.
* [ ] L06 Same-year learning measured.
* [ ] L07 Cross-year transfer measured.
* [ ] L08 Cross-regime transfer measured.
* [ ] L09 Cross-stock transfer measured.
* [ ] L10 Cross-sector transfer measured.
* [ ] L11 Memorization gap measured.
* [ ] L12 Identity gap measured.
* [ ] L13 Future-information audit passed.
* [ ] L14 Provenance audit passed.
* [ ] L15 Reproducibility passed.
* [ ] L16 Stability passed.
* [ ] L17 Risk impact measured.
* [ ] L18 Direction impact measured.
* [ ] L19 Movement impact measured.
* [ ] L20 Portfolio impact measured.
* [ ] L21 Learning delta independently reproduced.
* [ ] L22 Fresh holdout validation passed.
* [ ] L23 Knowledge promotion gates passed.
* [ ] L24 All critical failures resolved or explicitly classified as unresolved scientific limitations.
* [ ] L25 Final Masterstock updated.
* [ ] L26 Final checklist independently audited.

---

# 81. THE FINAL STOP CONDITION

Claude is forbidden from stopping because:

> "The architecture is complete."

Architecture completion is not the objective.

Claude is forbidden from stopping because:

> "All tests currently pass."

Passing tests without demonstrated learning is not the objective.

Claude is forbidden from stopping because:

> "The codebase is large enough."

Code volume is not the objective.

Claude is forbidden from stopping because:

> "The model performs well historically."

Historical performance is not the objective.

Claude is forbidden from stopping because:

> "We achieved the target on one run."

One run is not the objective.

Claude may declare the learning engine complete only when:

```text
EVERY MASTER CHECKLIST ITEM
        +
EVERY REQUIRED MINIMUM IMPLEMENTATION
        +
EVERY REQUIRED TEST
        +
EVERY REQUIRED FIREWALL
        +
EVERY REQUIRED VALIDATION
        +
EVERY REQUIRED REPORT
        +
FINAL INDEPENDENT AUDIT
```

has been completed.

---

# 82. IF A TEST FAILS

Do not lower the test.

Do not weaken the threshold merely to obtain a pass.

Do not remove the test.

Do not redefine success.

Instead:

```text
record failure
diagnose failure
create hypothesis
run experiment
implement correction
rerun affected tests
rerun regression suite
update knowledge
continue
```

If the failure reveals a fundamental scientific limitation, document it rather than hiding it.

---

# 83. IF A SECTION IS UNDER THE LINE MINIMUM

Do NOT pad it.

First determine:

```text
Is functionality missing?
```

If yes:

> continue implementing.

If no:

> identify exactly where the equivalent meaningful implementation resides and update the checklist's architectural mapping.

If neither can be demonstrated:

> the section is incomplete.

The minimum exists to prevent superficial implementations.

It does not exist to encourage meaningless verbosity.

---

# 84. CONTINUOUS EXECUTION PROTOCOL

At every checkpoint:

```text
1. Read current checklist.
2. Find first incomplete critical item.
3. Inspect existing implementation.
4. Inspect existing tests.
5. Inspect existing evidence.
6. Implement missing functionality.
7. Run targeted tests.
8. Run regression tests.
9. Run relevant scientific validation.
10. Record evidence.
11. Update Masterstock.
12. Commit/checkpoint.
13. Select next incomplete item.
14. Continue.
```

Never ask the owner:

> "Should I continue?"

Continue.

Never ask:

> "Do you want me to implement the next section?"

Implement it.

Never stop after producing a plan.

The plan is the instruction to execute.

---

# 85. PRIORITY ORDER

When multiple tasks compete:

### Priority 1

Future-information and provenance correctness.

### Priority 2

Learning validity.

### Priority 3

Same-year learning delta.

### Priority 4

Cross-year transfer.

### Priority 5

Failure learning.

### Priority 6

Pattern reliability and lifecycle.

### Priority 7

Knowledge retrieval/context.

### Priority 8

Meta-learning.

### Priority 9

Production integration.

### Priority 10

Presentation/dashboard improvements.

A beautiful dashboard never outranks a broken learning firewall.

---

# 86. THE CENTRAL QUESTION

At every major milestone ask:

> "What can WEEKLY7 now learn from experience that it could not learn before?"

If the answer is only:

> "It has more parameters."

That is not sufficient.

If the answer is:

> "It can recognize a previously learned situation under a different identity, understand that the old pattern applies only under certain conditions, reduce its influence when those conditions disappear, and improve its future decision without seeing the answer."

That is genuine progress.

---

# 87. FINAL ACCEPTANCE CRITERION

The final system should be able to demonstrate the following experiment:

```text
YEAR A
    ↓
WEEKLY7 encounters situation X
    ↓
WEEKLY7 makes decision
    ↓
Outcome occurs
    ↓
WEEKLY7 learns
    ↓
Knowledge is stored
    ↓
YEAR A is disguised
    ↓
Same situation class appears again
    ↓
Different identity
    ↓
Different episode
    ↓
No future information
    ↓
WEEKLY7 retrieves the learned knowledge
    ↓
WEEKLY7 changes its decision
    ↓
Decision improves
    ↓
Improvement transfers to another year/context
```

That experiment is more important than any single historical return.

---

# 88. FINAL COMMAND

BEGIN NOW.

Do not merely describe how you would build this.

Inspect the existing WEEKLY7 implementation.

Map the existing code against this contract.

Identify:

```text
already implemented
partially implemented
implemented but scientifically unvalidated
missing
contradictory
unsafe
duplicated
under-tested
under-depth
```

Then begin implementation.

Maintain the machine-readable checklist continuously.

Maintain checkpoints continuously.

Save every meaningful lesson.

Do not discard failed experiments.

Do not hide failed validation.

Do not lower standards to obtain completion.

Do not manufacture code to satisfy line counts.

Do not stop when the system becomes large.

Do not stop when the system becomes promising.

Do not stop when one learner wins.

Do not stop when one experiment passes.

**Continue until the master checklist is genuinely complete and independently validated.**

The objective is not a larger WEEKLY7.

The objective is a WEEKLY7 that **gets better because it remembers, understands, tests, transfers, and corrects what it has learned.**
