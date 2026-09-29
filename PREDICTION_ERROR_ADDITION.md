# WEEKLY7 — MASTER CHECKLIST ADDITION

## PREDICTION ERROR → MARKET CHANGE → SELF-CALIBRATION → ADAPTIVE EXIT INTELLIGENCE

This is an **ADDITION to the previous Weekly7 Autonomous Deep-Research Self-Learning Master Checklist.**

Do not replace, weaken, duplicate, or restart the previous checklist.

Integrate this layer into the existing Weekly7 architecture, especially:

* research intelligence
* surprise detection
* calibration
* experiment memory
* research priority
* meta-learning
* pattern lifecycle/health
* pattern-break research
* market-wide observation
* winner/loser research
* compute allocation
* anti-leak architecture
* research/trader firewall
* disguised-year testing
* autonomous research loop
* volatility research
* direction research
* prediction provenance
* deterministic replay
* Masterstock

The existing system already has generic surprise and calibration infrastructure. Extend and specialize it for prediction-error-driven market discovery rather than creating redundant parallel systems.

---

# ADDITIONAL OBJECTIVE

The system must become exceptionally good at answering:

> **"I expected X. Reality produced Y. Why was I wrong, how much of the difference was predictable, what changed, and what should I do differently next time?"**

Every prediction becomes an experiment.

Every meaningful prediction error becomes a research opportunity.

Every repeated prediction error becomes evidence of a potentially deeper model failure.

Every repeated unexpected market behavior becomes a potential regime-change signal.

The system must continuously learn both:

1. **how to predict the market better**, and
2. **how to recognize when the rules of the market have changed.**

It must never use future information to discover a change that supposedly could have been recognized beforehand.

---

# CHECKLIST A — IMMUTABLE PRE-PREDICTION EXPECTATION

Before every prediction, permanently record:

* predicted return
* predicted return distribution
* predicted direction
* confidence
* predicted volatility
* predicted time-to-peak
* predicted best exit window
* predicted maximum favorable excursion
* predicted maximum adverse excursion
* expected market regime
* expected sector regime
* expected relevant patterns
* expected pattern strengths
* expected interactions
* expected holding period
* expected probability distribution
* reasons for selection
* reasons competing candidates were rejected
* uncertainty
* alternative hypotheses
* model/version
* feature state
* knowledge state
* timestamp
* all information legally available at prediction time

The original expectation must become immutable.

No later information may rewrite history.

---

# CHECKLIST B — COMPLETE OUTCOME RECONSTRUCTION

After the position exits, reconstruct the entire realized trajectory.

Record:

* actual return at exit
* maximum favorable excursion
* maximum adverse excursion
* maximum return before exit
* minimum return before exit
* time of maximum return
* time of minimum return
* actual volatility
* actual direction
* actual holding period
* all meaningful intraperiod movements
* market movement
* sector movement
* relevant peer movement
* pattern behavior
* pattern changes
* external-event indicators
* actual best historical exit opportunity

Do not merely compare entry and exit.

The system must understand the path that produced the outcome.

---

# CHECKLIST C — PREDICTION ERROR ENGINE

Calculate separately:

* return error
* directional error
* volatility error
* timing error
* exit error
* confidence error
* market-regime error
* sector-regime error
* pattern-strength error
* interaction error

Example:

Prediction = +10%

Actual trajectory = +15%, then +11%, then +7%

The system must understand that:

* the prediction was directionally correct
* the magnitude was underestimated
* the market may have provided additional upside
* the optimal exit may have occurred before the eventual decline
* the exit model must be evaluated independently from the return predictor

Do not collapse these into one score.

---

# CHECKLIST D — ERROR-SIZE-BASED RESEARCH

Research intensity must scale with:

**prediction error × confidence × repeatability × potential future value × market significance.**

Tiny errors should not consume enormous computation.

Large unexpected errors should trigger substantially deeper research.

A highly confident prediction that is dramatically wrong must receive substantially more investigation than a low-confidence prediction that is slightly wrong.

---

# CHECKLIST E — CONFIDENT-WRONG DETECTOR

Whenever the system is substantially wrong despite high confidence:

Automatically create a research event.

Ask:

1. Which assumption failed?
2. Which feature failed?
3. Which pattern failed?
4. Did the pattern itself change?
5. Did another pattern override it?
6. Did an interaction change?
7. Did volatility change?
8. Did market correlation change?
9. Did sector behavior change?
10. Did timing change?
11. Did the optimal exit change?
12. Was there evidence available before the prediction that could have revealed the failure?
13. If yes, why did the system fail to recognize it?
14. If no, classify the cause as currently unknowable.
15. Can a precursor now be discovered for future detection?

---

# CHECKLIST F — MARKET EXPECTATION VS MARKET REALITY

The system must create market-level expectations, not only stock-level expectations.

Before each research/trading period estimate:

* expected market opportunity
* expected volatility
* expected breadth
* expected dispersion
* expected correlations
* expected sector leadership
* expected momentum persistence
* expected reversal probability
* expected gap behavior
* expected number of meaningful movers
* expected number of qualifying 5–10% opportunities

Then compare against reality.

Example:

Expected opportunity = approximately 10%

Actual opportunity = approximately 15%

The system must investigate why.

Possible explanations must be tested rather than assumed:

* volatility expansion
* stronger momentum persistence
* stronger sector concentration
* increased cross-sectional dispersion
* changing correlations
* pattern interaction
* changed holding-period behavior
* changed market microstructure
* identifiable external information
* previously weak pattern becoming strong
* multiple small changes combining into a large change

The objective is not merely to explain the past.

The objective is:

> **Can the system identify the precursor before the next occurrence?**

---

# CHECKLIST G — MARKET CONTRACTION RESEARCH

If expected opportunity is 10% but actual opportunity falls to 6%, investigate the reduction with equal seriousness.

Determine:

* which patterns deteriorated
* which patterns remained stable
* which patterns reversed
* which stocks stopped responding
* which sectors changed
* whether volatility compressed
* whether momentum decayed
* whether reversal behavior increased
* whether holding periods changed
* whether optimal exits changed
* whether formerly successful patterns became unreliable
* whether the market regime changed
* whether the apparent decline was actually caused by candidate-selection failure

Do not conclude:

> "The strategy stopped working."

until the system has investigated **what specifically stopped working and why.**

---

# CHECKLIST H — PATTERN CHANGE DETECTION

Every important pattern must have continuously updated estimates of:

* historical reliability
* recent reliability
* reliability by regime
* reliability by volatility
* reliability by sector
* reliability by holding period
* reliability by confidence
* recent prediction error
* error direction
* error magnitude
* degradation rate
* recovery rate

The system must distinguish:

* temporary noise
* normal variance
* weakening
* strengthening
* regime-specific failure
* genuine structural change
* permanent obsolescence
* returning patterns

Never immediately delete a failing pattern.

First investigate why it failed.

---

# CHECKLIST I — CHANGE-POINT DETECTION

Build dedicated change detection for:

* feature → outcome relationships
* pattern → outcome relationships
* volatility behavior
* direction behavior
* sector relationships
* market correlations
* breadth
* momentum persistence
* reversal behavior
* holding periods
* optimal exits
* confidence calibration
* prediction-error distributions

The system should attempt to recognize meaningful changes **as early as statistically justified**.

But:

> A change may only be declared using information available at the time the change was supposedly detectable.

No retrospective cheating.

---

# CHECKLIST J — "WHAT CHANGED?" INVESTIGATION TREE

Every significant error must automatically produce a structured investigation:

### Level 1 — Market

Did the whole market behave differently?

### Level 2 — Sector

Did this sector behave differently?

### Level 3 — Cross-section

Did comparable stocks behave differently?

### Level 4 — Individual stock

Did this stock have unusual behavior?

### Level 5 — Pattern

Did the relevant pattern change?

### Level 6 — Interaction

Did combinations of patterns behave differently?

### Level 7 — Timing

Was the prediction correct but the timing wrong?

### Level 8 — Exit

Was the prediction correct but the optimal exit different?

### Level 9 — External information

Was the outcome caused by information that genuinely could not have been known?

### Level 10 — Model failure

If the information was available, why did the model fail to recognize it?

Every investigation must end with:

**CAUSE**
→ **EVIDENCE**
→ **PRE-OUTCOME DETECTABILITY**
→ **REPEATABILITY**
→ **NEW KNOWLEDGE**
→ **MODEL CHANGE**
→ **TEST**

---

# CHECKLIST K — KNOWABILITY CLASSIFICATION

Every major unexpected outcome must be classified into one of:

1. **Predictable but missed**
2. **Partially predictable**
3. **Predictable only under a newly discovered condition**
4. **Currently unexplained**
5. **Genuinely unknowable from available information**

The system must not force every event into a predictable explanation.

If an event was genuinely unknowable, preserve that conclusion as knowledge.

This prevents the system from inventing explanations simply because an outcome occurred.

---

# CHECKLIST L — 5–10% SELECTION CONSTRAINT

The prediction network must be optimized around the actual trading constraint.

A stock is eligible for selection only when the system predicts that its eventual realizable gain under its intended holding/exit strategy falls within:

**+5% to +10%.**

Do not select a stock merely because it has a high probability of going up.

Do not select a stock merely because it has high volatility.

Do not select a stock because it improves historical statistics.

The system must predict:

> **Will this stock produce the required 5–10% gain under the strategy's actual optimal exit behavior?**

The prediction system must continuously improve its ability to identify this region.

---

# CHECKLIST M — DO NOT GAME THE 5–10% STATISTIC

The system must NEVER:

* hold a position longer simply to make its prediction look correct
* delay an exit to improve calibration statistics
* choose a worse exit because it makes the prediction closer
* redefine the prediction after seeing the outcome
* suppress losing predictions
* select only historically convenient examples
* change the target after seeing the result

The prediction statistic is for evaluation.

The actual trading decision remains governed by the learned optimal-exit model.

---

# CHECKLIST N — OPTIMAL EXIT RESEARCH

The system must independently learn:

> **When should this position be sold?**

Analyze:

* expected return trajectory
* probability of continued upside
* probability of reversal
* momentum decay
* volatility decay
* time since entry
* pattern decay
* market regime
* sector behavior
* stock-specific behavior
* maximum favorable excursion
* opportunity cost
* expected future return conditional on holding

The system must learn the expected best exit **before the future exit point occurs**.

---

# CHECKLIST O — EXIT INDEPENDENCE RULE

The system must always sell when its learned decision policy says the expected value of remaining invested has fallen below the learned exit threshold.

It must **never** wait merely because:

> "I need the final return to be within 1 percentage point of my prediction."

The ±1 percentage-point criterion is an **evaluation target**, not an exit rule.

After selling, the system evaluates:

> How close was the realized outcome to the original prediction?

This preserves scientific integrity.

---

# CHECKLIST P — ±1 PERCENTAGE-POINT CALIBRATION TARGET

The long-term research objective is to develop predictions whose realized outcomes fall within approximately **±1 percentage point of the original prediction at least 80% of the time**, subject to sufficient sample size and rigorous out-of-sample validation.

This is a target to optimize toward.

It must never be manufactured by:

* changing predictions afterward
* delaying exits
* cherry-picking
* suppressing failed predictions
* using future information
* narrowing the sample after seeing results
* manipulating the target distribution

If the system cannot achieve this honestly, it must report the actual result.

---

# CHECKLIST Q — SELF-CORRECTING PREDICTION NETWORK

When prediction accuracy deteriorates, the system must investigate whether the problem is:

* feature weighting
* missing feature
* obsolete feature
* pattern interaction
* regime recognition
* confidence calibration
* candidate selection
* volatility estimation
* direction estimation
* timing
* exit estimation
* market-level conditioning
* sector conditioning
* overfitting
* underfitting

It must test candidate fixes independently.

A proposed improvement may only become part of the production learner after passing the existing validation and anti-leak architecture.

---

# CHECKLIST R — EARLY REGIME WARNING SYSTEM

Build an early-warning system using **only information available up to the current point in time**.

It should search for precursors such as:

* widespread prediction-error shifts
* simultaneous pattern degradation
* simultaneous pattern strengthening
* changing volatility
* changing breadth
* changing correlations
* changing sector leadership
* changing holding-period behavior
* changing optimal exits
* abnormal cross-sectional dispersion
* repeated confident failures

The system should distinguish:

**single-stock anomaly**

from

**sector change**

from

**market-wide regime change.**

---

# CHECKLIST S — REPEATED ERROR ESCALATION

One surprising result may be noise.

Repeated similar surprises must increase research priority.

For example:

Prediction repeatedly underestimates strong moves.

The system should investigate:

> "Is there a systematic missing factor causing underprediction?"

Likewise:

Prediction repeatedly expects 8–10% but realizes 3–6%.

Investigate:

> "What changed that is consistently suppressing the previously successful behavior?"

The system must learn from **error patterns**, not isolated errors.

---

# CHECKLIST T — SELF-RESEARCH LOOP

The system must research itself continuously.

It should ask:

* Which prediction errors are increasing?
* Which patterns are degrading?
* Which predictions are systematically biased?
* Which confidence levels are miscalibrated?
* Which regimes produce the worst errors?
* Which stock types produce the worst errors?
* Which sectors produce the worst errors?
* Which exit decisions produce the largest missed gains?
* Which research discoveries actually improve OOS performance?
* Which research approaches repeatedly produce nothing?
* What should be researched next?

This research must feed the existing research-priority/compute-allocation system rather than creating a second independent scheduler.

---

# CHECKLIST U — COMPUTE ALLOCATION

Research depth must be allocated according to expected value.

Prioritize investigations with high:

* error magnitude
* confidence
* frequency
* repeatability
* market impact
* potential predictive value
* transferability
* likelihood of revealing a regime change

Deprioritize investigations where:

* the effect is tiny
* the event is extremely isolated
* the cause is demonstrably unknowable
* repeated research produces no improvement
* expected information gain is negligible

The system must become better at deciding **what deserves research.**

---

# CHECKLIST V — EVERY DISCOVERY MUST BE TESTABLE

Whenever the system claims:

> "I figured out why I was wrong."

it must create a hypothesis.

Then:

1. formulate the hypothesis
2. identify supporting evidence
3. identify contradictory evidence
4. create an appropriate historical test
5. run out-of-sample evaluation
6. test multiple periods
7. test multiple stocks
8. test multiple regimes
9. test against controls/placebos
10. determine whether the discovery transfers
11. record the result permanently

An explanation that cannot survive testing remains an explanation hypothesis, not knowledge.

---

# CHECKLIST W — REGIME-CHANGE MEMORY

When a genuine regime change is discovered, record:

* what the old regime looked like
* what the new regime looked like
* earliest detectable evidence
* patterns affected
* patterns strengthened
* patterns weakened
* prediction errors before detection
* detection latency
* which signals could have detected it earlier
* which signals were misleading
* conditions for returning to the old regime
* confidence in the regime classification

The system should eventually learn:

> **"When I start seeing X, the market is beginning to behave differently, and these formerly successful patterns should be treated differently."**

---

# CHECKLIST X — NO RETROSPECTIVE REGIME CHEATING

A regime detector must be evaluated on the information it possessed at the time.

It cannot identify a regime using:

* future returns
* future volatility
* future labels
* future pattern performance
* future market state
* future knowledge memory

Then claim that the regime was detectable earlier.

The detector must operate forward through time exactly as it would have historically.

---

# CHECKLIST Y — FULL ERROR-TO-IMPROVEMENT PIPELINE

Every significant prediction must be capable of flowing through:

**Prediction**

↓

**Immutable expectation**

↓

**Outcome**

↓

**Error measurement**

↓

**Error classification**

↓

**Cause investigation**

↓

**Knowability determination**

↓

**Pattern/regime investigation**

↓

**Hypothesis**

↓

**Research**

↓

**Out-of-sample test**

↓

**Model update**

↓

**Validation**

↓

**Promotion or rejection**

↓

**Future monitoring**

This pipeline must be persistent and auditable.

---

# CHECKLIST Z — FINAL INTEGRATION TESTS

Add adversarial tests proving that:

* predictions cannot be rewritten after outcomes
* exits cannot be manipulated to improve prediction statistics
* ±1% evaluation does not control selling
* future information cannot enter error analysis
* future information cannot enter regime detection
* unknowable events are allowed to remain unknowable
* tiny errors do not consume excessive compute
* major errors trigger deeper investigation
* repeated errors escalate research priority
* pattern changes can be detected
* false regime changes do not unnecessarily disable successful patterns
* genuinely degraded patterns can reduce influence
* recovered patterns can regain influence
* market-wide changes can be distinguished from individual-stock anomalies
* exit timing is learned independently
* the 5–10% selection constraint is enforced
* stocks outside the predicted 5–10% outcome range cannot be selected merely to improve another metric
* the system cannot game the ±1% target
* discoveries must pass out-of-sample validation
* the learner improves only when evidence justifies improvement

---

# COMPLETION REQUIREMENT

Do not declare this addition complete because the code exists.

It is complete only when:

1. Every checklist item is implemented.
2. Every item has tests.
3. Every integration point is reachable.
4. No future leakage exists.
5. Deterministic replay works.
6. Prediction expectations are immutable.
7. Error analysis works at stock, sector, market, pattern, and regime levels.
8. Exit learning works independently of prediction calibration.
9. The 5–10% selection constraint is enforced.
10. The ±1% target is measured honestly.
11. Confident failures trigger deeper investigation.
12. Repeated errors produce escalating research priority.
13. Regime-change detection has forward-time validation.
14. Knowable and unknowable causes are distinguished.
15. Research discoveries are validated out of sample.
16. The system demonstrably learns from prediction errors.
17. The existing Weekly7 checklist remains intact.
18. No existing test or scientific gate is weakened to make this pass.

**Never lower a test threshold merely because the system currently cannot pass it.**

The objective is not to make the statistics look good.

The objective is to make the prediction system genuinely better.

# FINAL PRINCIPLE

The system should eventually behave like this:

> **"I predicted 10%. I got 15%. I know exactly how wrong I was. I investigated what produced the extra 5%. I determined whether that information was knowable beforehand. I discovered whether the market changed. I tested whether that change was repeatable. I learned whether my model should respond differently next time. And if the same conditions appear again, I can make a better prediction without ever having seen the future."**

And when it predicts 10% but gets 6%:

> **"I know exactly why I missed. I know whether my previously successful patterns weakened. I know whether the market changed. I know when the change became detectable. I know whether my model should reduce exposure to that pattern, replace it, or recognize that this was an isolated failure."**

The system must continuously improve both:

**its predictions**

and

**its ability to recognize when the world has stopped behaving the way its predictions assumed.**
