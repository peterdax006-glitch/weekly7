# WEEKLY7 — 10-HOUR NON-STOP MASTER EXECUTION CHECKLIST

## FOUNDATION → WORKING → INTENDED BEHAVIOR → OPERATION → FAILURE DISCOVERY → IMPROVEMENT → PERFECTION → INDEPENDENT PROOF

---

# 0. ABSOLUTE EXECUTION ORDER

You are taking autonomous ownership of the Weekly7 engineering/research system for this work session.

Your job is NOT to finish a convenient number of checklist items.

Your job is to continue working through the complete system until the current checklist is genuinely complete or every remaining item has a documented, technically justified scientific limitation.

The required order is:

### PHASE 1

**BUILD THE FOUNDATION**

↓

### PHASE 2

**MAKE SURE THE FOUNDATION ACTUALLY WORKS**

↓

### PHASE 3

**MAKE SURE IT WORKS THE WAY THE OWNER INTENDED**

↓

### PHASE 4

**RUN THE INTEGRATED SYSTEM**

↓

### PHASE 5

**FIND WHAT IS WEAK, WRONG, SHALLOW, MISWIRED, OR MISLEADING**

↓

### PHASE 6

**FIX IT**

↓

### PHASE 7

**RUN IT AGAIN**

↓

### PHASE 8

**PERFECT THE SYSTEM**

↓

### PHASE 9

**INDEPENDENTLY VERIFY THAT THE IMPROVEMENTS ARE REAL**

↓

### PHASE 10

**ONLY THEN DECLARE COMPLETION**

Do not skip ahead because a later test is easier.

Do not perform superficial testing of an unfinished foundation.

Do not declare a feature complete because its file exists.

Do not declare a feature complete because unit tests pass.

Do not declare the system complete because the full test suite passes.

---

# 1. NON-NEGOTIABLE OWNER RULES

The existing Weekly7 contracts remain authoritative.

Preserve all existing requirements, including:

* C62 self-learning contract
* C63 foundation-first rule
* C64 blind trader / knowing curator
* C65 live-system parking
* C66 autonomous deep-research engine
* C67 mover-episode research
* C68 prediction-error/self-calibration
* all existing canon directives
* all anti-cheating requirements
* all provenance requirements
* all existing validation requirements

C63 specifically requires the code foundation to be built before moving into real-data testing and perfecting.

Do not weaken any existing requirement to make this checklist easier.

Do not delete a failing test because it is inconvenient.

Do not lower a threshold because the system currently cannot pass it.

Do not redefine success after seeing results.

---

# 2. MASTER STATE AUDIT — FIRST

Before changing code:

* [ ] Read the current Masterstock.
* [ ] Read the current C62/C66/C67/C68 contracts.
* [ ] Read the current machine checklist state.
* [ ] Inspect current git state.
* [ ] Inspect current builder/integration state.
* [ ] Identify every IMPLEMENTED item that is not VALIDATED.
* [ ] Identify every IN_PROGRESS item.
* [ ] Identify every NOT_STARTED item.
* [ ] Identify every FAILED item.
* [ ] Identify every known scientific limitation.
* [ ] Identify every known integration gap.
* [ ] Identify every known firewall concern.
* [ ] Identify every known data-quality problem.
* [ ] Identify every known shallow implementation.
* [ ] Identify every known duplicate implementation.
* [ ] Identify every known survivor-only limitation.
* [ ] Identify every known future-leak concern.

Create a current execution ledger.

Nothing is allowed to disappear from the ledger.

---

# 3. FOUNDATION PHASE

## 3.1 COMPLETE ALL FOUNDATION CODE

For every unfinished foundation requirement:

* [ ] Implement it.
* [ ] Integrate it with the existing architecture.
* [ ] Add its tests.
* [ ] Add provenance.
* [ ] Add deterministic behavior where required.
* [ ] Add failure handling.
* [ ] Add health monitoring where appropriate.
* [ ] Add observability.
* [ ] Add audit information.
* [ ] Ensure it is reachable.
* [ ] Ensure data actually flows through it.

Do not build isolated demonstration modules.

A component is not complete merely because it can be imported.

---

# 4. ANTI-BARE-MINIMUM FOUNDATION GATE

For every component, ask:

### Existence

* [ ] Does the implementation exist?

### Reachability

* [ ] Can the real system actually call it?

### Data flow

* [ ] Does real data enter it?

### Output flow

* [ ] Does its output actually influence the next appropriate component?

### Persistence

* [ ] Is its learned state stored correctly?

### Provenance

* [ ] Can we determine where its result came from?

### Determinism

* [ ] Can the same inputs reproduce the same result where required?

### Failure behavior

* [ ] Does it fail safely?

### Adversarial behavior

* [ ] Does it reject deliberately invalid input?

### Integration

* [ ] Is the production path using it?

### Scientific validity

* [ ] Is there evidence that it does what it claims?

If any answer is "no":

**IT IS NOT DONE.**

---

# 5. NO PLACEHOLDER IMPLEMENTATIONS

Search for:

* TODO
* FIXME
* placeholder
* stub
* pass
* NotImplemented
* fake result
* hardcoded result
* constant score
* manually typed verdict
* bypass
* compatibility shortcut
* dead adapter
* unused implementation

For each one:

* [ ] Determine whether it is legitimate.
* [ ] If not legitimate, replace it with real functionality.
* [ ] Test it.
* [ ] Integrate it.
* [ ] Verify it.

No component may claim to perform research, learning, prediction, calibration, regime detection, or causal analysis if it merely returns a placeholder result.

---

# 6. DUPLICATION FIREWALL

For every new implementation:

* [ ] Search for existing functionality.
* [ ] Determine whether the capability already exists.
* [ ] Extend the existing implementation if appropriate.
* [ ] Use adapters only when architecturally justified.
* [ ] Do not create parallel competing sources of truth.

If two systems perform the same conceptual job:

* [ ] Identify the canonical implementation.
* [ ] Retire or isolate the duplicate.
* [ ] Prove which implementation is actually used.

"Both exist" is not an acceptable architecture.

---

# 7. FOUNDATION TEST GATE

After foundation implementation:

* [ ] Run unit tests.
* [ ] Run property tests.
* [ ] Run integration tests.
* [ ] Run adversarial tests.
* [ ] Run deterministic replay.
* [ ] Run reachability.
* [ ] Run type checks.
* [ ] Run provenance checks.
* [ ] Run architecture/firewall checks.
* [ ] Run the complete existing suite.

Every failure becomes a work item.

Do not simply record the failure and continue.

---

# 8. INTENDED-BEHAVIOR PHASE

This is a separate phase.

Passing tests is NOT enough.

For each major system, verify that its behavior matches the owner's intended purpose.

---

## 8.1 AUTONOMOUS RESEARCHER

Verify that it actually:

* [ ] generates research questions
* [ ] prioritizes them
* [ ] allocates compute intelligently
* [ ] investigates failures
* [ ] investigates winners
* [ ] investigates losers
* [ ] researches missed opportunities
* [ ] learns from research outcomes
* [ ] remembers useful discoveries
* [ ] rejects useless discoveries
* [ ] avoids endlessly repeating failed experiments

---

## 8.2 MARKET-WIDE OBSERVER

Verify that it genuinely observes the broad market rather than a convenient subset.

* [ ] Daily market observation works.
* [ ] Movers are discovered independently of the current portfolio.
* [ ] Winners are studied.
* [ ] Losers are studied.
* [ ] Stocks the system never selected are still available for research.
* [ ] Intraday behavior can be analyzed.
* [ ] Subsequent behavior can be analyzed.
* [ ] Consolidation is studied.
* [ ] continuation is studied.
* [ ] reversal is studied.
* [ ] volatility expansion is studied.
* [ ] volatility contraction is studied.

---

# 9. C67 MOVER-EPISODE DEEP-RESEARCH GATE

For the hundreds of relevant mover episodes available in historical research:

* [ ] Identify the episode.
* [ ] Reconstruct what was knowable before the movement.
* [ ] Analyze what happened during the movement.
* [ ] Analyze what happened afterward.
* [ ] Determine whether it consolidated.
* [ ] Determine whether it accelerated.
* [ ] Determine whether it stopped.
* [ ] Determine whether it reversed.
* [ ] Determine whether it continued the following day.
* [ ] Search for small repeatable precursors.
* [ ] Search for interactions.
* [ ] Search for regime dependence.
* [ ] Search for patterns that transfer.
* [ ] Record unknown causes honestly.

The goal is not to produce a large number of observations.

The goal is to extract **validated predictive knowledge**.

---

# 10. VOLATILITY-FIRST GATE

Before expecting strong direction performance:

* [ ] Verify volatility discovery.
* [ ] Verify candidate selection.
* [ ] Verify that volatility predictions transfer.
* [ ] Verify that the system does not simply select historically large movers.
* [ ] Verify out-of-sample performance.
* [ ] Verify across multiple periods.
* [ ] Verify across stocks.
* [ ] Verify across sectors.
* [ ] Verify across regimes.

---

# 11. DIRECTION GATE

After volatility foundation is proven:

* [ ] Test direction prediction.
* [ ] Test confidence calibration.
* [ ] Test conditional direction.
* [ ] Test pattern interactions.
* [ ] Test regime conditioning.
* [ ] Test winner/loser symmetry.
* [ ] Test whether direction improves over baseline.
* [ ] Test whether improvements transfer.

Do not assume that volatility knowledge automatically produces direction knowledge.

---

# 12. PREDICTION-ERROR LOOP

For every meaningful prediction:

* [ ] Save immutable pre-outcome expectation.
* [ ] Save confidence.
* [ ] Save predicted return.
* [ ] Save predicted range.
* [ ] Save expected regime.
* [ ] Save relevant patterns.
* [ ] Save expected timing.
* [ ] Save expected exit.
* [ ] Save available information.

After the outcome:

* [ ] Reconstruct the actual trajectory.
* [ ] Calculate error.
* [ ] Calculate error magnitude.
* [ ] Calculate directional error.
* [ ] Calculate timing error.
* [ ] Calculate exit error.
* [ ] Investigate why it happened.
* [ ] Determine whether the cause was knowable.
* [ ] Generate a research hypothesis.
* [ ] Test the hypothesis.
* [ ] Promote only if evidence supports it.

---

# 13. CONFIDENT-WRONG FIREWALL

Whenever the system is highly confident and substantially wrong:

**AUTOMATICALLY ESCALATE.**

It must investigate:

* [ ] pattern failure
* [ ] regime change
* [ ] missing feature
* [ ] feature degradation
* [ ] interaction change
* [ ] volatility change
* [ ] sector change
* [ ] market-wide change
* [ ] timing failure
* [ ] exit failure
* [ ] unknowable external event

No confident failure may simply disappear into the aggregate statistics.

---

# 14. MARKET REGIME CHANGE GATE

Continuously compare:

**EXPECTED MARKET**

against

**ACTUAL MARKET.**

Detect:

* [ ] volatility shifts
* [ ] breadth shifts
* [ ] correlation shifts
* [ ] dispersion shifts
* [ ] sector leadership shifts
* [ ] momentum shifts
* [ ] reversal shifts
* [ ] holding-period shifts
* [ ] prediction-error shifts
* [ ] pattern-performance shifts

Then ask:

> Did the market actually change, or did our model merely have a bad sample?

The system must distinguish those possibilities.

---

# 15. PATTERN HEALTH GATE

Every important pattern must have:

* [ ] historical performance
* [ ] recent performance
* [ ] regime performance
* [ ] confidence calibration
* [ ] degradation tracking
* [ ] recovery tracking
* [ ] failure explanation
* [ ] transfer evidence
* [ ] health state
* [ ] promotion state

Do not delete failed patterns automatically.

Gate them.

Investigate them.

Allow recovery when evidence supports recovery.

---

# 16. 5–10% TARGET GATE

The selection system must enforce the intended objective:

> Select only stocks for which the system predicts a realizable strategy outcome within approximately +5% to +10%.

Verify:

* [ ] selection is based on prediction, not hindsight
* [ ] selection is not based solely on volatility
* [ ] selection is not based solely on direction
* [ ] selection cannot be altered after outcome
* [ ] candidates outside the range cannot be selected merely to improve statistics
* [ ] the actual exit decision remains independent

---

# 17. EXIT INTELLIGENCE GATE

The system must learn:

> **When is the best time to sell?**

Test:

* [ ] continuation probability
* [ ] reversal probability
* [ ] momentum decay
* [ ] volatility decay
* [ ] time since entry
* [ ] pattern decay
* [ ] market regime
* [ ] sector regime
* [ ] expected remaining return
* [ ] opportunity cost

The system must sell according to its learned exit policy.

It must NEVER delay an exit merely to make the prediction closer to the eventual outcome.

The ±1% criterion is an evaluation metric, not an exit trigger.

---

# 18. ±1% CALIBRATION RESEARCH

Measure whether realized outcomes fall within approximately ±1 percentage point of prediction.

Track:

* [ ] overall rate
* [ ] by confidence
* [ ] by regime
* [ ] by sector
* [ ] by pattern
* [ ] by holding period
* [ ] by prediction magnitude

Target:

**80% within ±1 percentage point**, if the data genuinely supports it.

Do not manufacture this statistic.

---

# 19. SELF-RESEARCH PHASE

Now stop thinking only about stock prediction.

Make the system research itself.

Ask:

* [ ] Where is prediction error increasing?
* [ ] Which patterns are weakening?
* [ ] Which patterns are strengthening?
* [ ] Which regimes produce poor predictions?
* [ ] Which features contribute little?
* [ ] Which features suddenly matter?
* [ ] Which experiments repeatedly fail?
* [ ] Which research methods produce useful discoveries?
* [ ] Which research methods waste compute?
* [ ] What should be investigated next?

The system must become better at deciding **what it should learn.**

---

# 20. COMPUTE-VALUE FIREWALL

Before expensive research:

Calculate expected research value.

Consider:

* error magnitude
* frequency
* confidence
* repeatability
* potential market impact
* transferability
* probability of discovering something actionable
* expected compute cost

Reject or deprioritize research whose expected information gain is too low.

Do not confuse:

**more computation**

with

**better research.**

---

# 21. ANTI-BACKFIT FIREWALL

Every discovered improvement must survive:

* [ ] unseen dates
* [ ] unseen years
* [ ] unseen stocks
* [ ] unseen sectors
* [ ] unseen regimes
* [ ] shuffled controls
* [ ] placebo tests
* [ ] identity tests
* [ ] memorization tests
* [ ] leakage audits
* [ ] independent reproduction

A discovery that only works on the data that inspired it is not a discovery.

---

# 22. BLIND-TRADER FIREWALL

The running Test trader must never know information it would not have possessed at the time.

Verify:

* [ ] no future prices
* [ ] no future labels
* [ ] no future pattern results
* [ ] no future regime labels
* [ ] no future research conclusions
* [ ] no hidden year
* [ ] no ticker identity leakage
* [ ] no future-derived memory
* [ ] no hindsight-generated ranking

The knowing curator may operate on the trusted side only under the existing architecture.

The trader-facing side must remain blind.

---

# 23. SURVIVOR-BIAS FIREWALL

Explicitly track the known survivor-only panel limitation.

Do not describe survivor-only results as equivalent to a full historical market.

Until delisted history is integrated:

* [ ] label affected results appropriately
* [ ] prevent survivor-only results from silently becoming final evidence
* [ ] test whether conclusions survive broader data

The current Masterstock explicitly identifies survivor-only data as a scientific limitation.

---

# 24. "WORKS" VS "WORKS AS INTENDED" GATE

For every major subsystem produce two separate verdicts:

### TECHNICAL VERDICT

Does the code execute correctly?

### SCIENTIFIC/BEHAVIORAL VERDICT

Does it actually accomplish the intended purpose?

Both must pass.

Example:

A research engine that successfully generates 10,000 questions but never discovers useful predictive information:

**TECHNICALLY WORKING**

but

**NOT WORKING AS INTENDED.**

Do not mark it complete.

---

# 25. REAL OPERATION PHASE

Once foundation and integration gates pass:

Run the actual Test self-learning system.

Observe it.

Do not merely run static tests.

Measure:

* [ ] research throughput
* [ ] learning throughput
* [ ] memory growth
* [ ] useful discovery rate
* [ ] failed experiment rate
* [ ] compute allocation
* [ ] prediction calibration
* [ ] pattern health
* [ ] regime detection
* [ ] volatility selection
* [ ] direction selection
* [ ] exit behavior
* [ ] portfolio consequences

Let it operate long enough to expose interaction failures.

---

# 26. AUTOPSY PHASE

After operation:

Perform a complete autopsy.

Find:

* [ ] silent failures
* [ ] false positives
* [ ] false negatives
* [ ] redundant research
* [ ] missed research opportunities
* [ ] shallow research
* [ ] incorrect conclusions
* [ ] bad memory
* [ ] bad prioritization
* [ ] bad calibration
* [ ] pattern failures
* [ ] regime-detection failures
* [ ] exit failures
* [ ] integration failures
* [ ] performance bottlenecks

Every meaningful weakness becomes a new work item.

---

# 27. PERFECTING PHASE

Now improve the system.

For every weakness:

1. [ ] reproduce it
2. [ ] identify cause
3. [ ] formulate fix
4. [ ] implement fix
5. [ ] test fix
6. [ ] run regression tests
7. [ ] run adversarial tests
8. [ ] run OOS tests
9. [ ] compare against previous version
10. [ ] retain the change only if it genuinely improves the system

Never accept:

> "The new version is different."

Require:

> **"The new version is demonstrably better."**

---

# 28. NO REGRESSION FIREWALL

Every improvement must prove that it does not silently damage:

* [ ] existing learning
* [ ] anti-leak guarantees
* [ ] blind operation
* [ ] memory integrity
* [ ] pattern health
* [ ] deterministic replay
* [ ] research diversity
* [ ] compute controls
* [ ] existing validated behavior

An improvement that fixes one component while breaking another is not an improvement.

---

# 29. INDEPENDENT REPRODUCTION

At the end of the improvement cycle:

* [ ] rerun from a clean state
* [ ] rerun with fresh seeds where appropriate
* [ ] rerun with independent test conditions
* [ ] verify results independently
* [ ] compare results to the original evidence
* [ ] verify that the conclusion does not depend on one lucky run

---

# 30. FINAL SCIENTIFIC AUDIT

Before completion, independently audit:

### Leakage

* [ ] no future leakage

### Memorization

* [ ] no ticker/date/year memorization

### Survivor bias

* [ ] limitations disclosed and gated

### Calibration

* [ ] predictions measured honestly

### Causality

* [ ] explanations are not accepted merely because they fit after the fact

### Transfer

* [ ] knowledge transfers beyond the data that produced it

### Robustness

* [ ] results survive controls and perturbations

### Reproducibility

* [ ] results can be reproduced

### Integration

* [ ] all intended systems actually interact

### Learning

* [ ] learning demonstrably changes future behavior appropriately

### Self-improvement

* [ ] research itself improves where justified

---

# 31. BARE-MINIMUM DETECTION FIREWALL

At every stage ask:

> **"Could this implementation technically satisfy the checklist while failing to accomplish the actual objective?"**

If yes:

**DO NOT ACCEPT IT.**

Examples of unacceptable bare-minimum implementations:

* generating research questions without learning from them
* recording prediction errors without investigating them
* detecting regime changes only after the regime is over
* detecting patterns only because they were profitable historically
* calculating calibration without changing anything from the result
* creating a memory database that the trader never uses
* creating a learner that never reaches the decision path
* creating a research engine that never influences future research
* creating a pattern health score that never gates patterns
* creating an exit model that never controls exits
* creating a 5–10% filter that is bypassed by another selector
* creating tests that only test happy paths
* passing unit tests while the real pipeline bypasses the component
* producing large amounts of code without corresponding capability
* increasing line count without increasing validated functionality

Any such implementation must be rejected and rebuilt.

---

# 32. CODE-DEPTH FIREWALL

Line count is NOT the definition of quality.

However, the existing C66 line budgets remain binding.

Therefore:

* [ ] Never pad code to reach a line target.
* [ ] Never stop below a required budget merely because a minimal implementation technically works.
* [ ] If significantly below budget, investigate whether required depth is actually missing.
* [ ] If above budget, verify that the additional code represents real functionality.
* [ ] Every major code block must correspond to a real capability.
* [ ] Every major capability must have tests.
* [ ] Every important test must test meaningful behavior.

The goal is:

**depth of capability, not artificial length.**

---

# 33. CHECKLIST STATE FIREWALL

The machine checklist must never say:

`VALIDATED`

merely because:

* code exists
* tests exist
* tests pass
* builder says complete
* a file reached its line budget

A component becomes VALIDATED only when evidence demonstrates that it works as intended.

Use:

* IMPLEMENTED
* IN_PROGRESS
* FAILED
* VALIDATED
* SCIENTIFIC LIMITATION

honestly.

---

# 34. DO NOT STOP FOR COSMETIC COMPLETION

Do not stop because:

* the suite is green
* the line budget is met
* the code compiles
* the checklist has been edited
* a builder reported completion
* the system ran once
* one historical test improved
* one benchmark improved
* one seed passed

Continue until the relevant evidence standard is met.

---

# 35. CONTINUOUS WORK LOOP

After every completed work package:

```text
INSPECT
→ IMPLEMENT
→ TEST
→ INTEGRATE
→ RUN
→ OBSERVE
→ FIND WEAKNESS
→ FIX
→ RETEST
→ VALIDATE
→ UPDATE CHECKLIST
→ SELECT NEXT HIGHEST-VALUE WORK
→ CONTINUE
```

Never end the loop merely because one phase completed.

---

# 36. AUTONOMOUS PRIORITY RULE

When choosing what to work on next, prioritize:

1. blocker to foundation
2. broken core functionality
3. security/anti-cheating/firewall issue
4. integration failure
5. scientific-validity failure
6. major performance bottleneck
7. shallow implementation
8. major prediction/research weakness
9. high-value improvement
10. lower-value polish

Do not spend an hour polishing a minor component while a major scientific failure remains unresolved.

---

# 37. 10-HOUR EXECUTION REQUIREMENT

This is intended to be a **long autonomous work session**.

Do not finish early because the obvious tasks are complete.

If the primary checklist becomes temporarily blocked:

* [ ] investigate the blocker
* [ ] work on another unblocked foundation item
* [ ] strengthen tests
* [ ] perform integration analysis
* [ ] inspect for shallow implementations
* [ ] run audits
* [ ] improve documentation/provenance
* [ ] investigate scientific weaknesses
* [ ] prepare the next executable work package

Never sit idle waiting for the owner when useful autonomous work remains.

Do not ask the owner what to do next if this checklist already determines the next action.

---

# 38. 10-HOUR CHECKPOINTS

At approximately:

### HOUR 1

* [ ] complete state audit
* [ ] identify highest-priority unfinished foundation
* [ ] begin foundation completion

### HOUR 2

* [ ] continue foundation
* [ ] integrate completed components
* [ ] eliminate obvious stubs/duplicates

### HOUR 3

* [ ] foundation verification
* [ ] unit/property/integration testing
* [ ] fix failures

### HOUR 4

* [ ] intended-behavior verification
* [ ] anti-cheating verification
* [ ] blind-trader verification

### HOUR 5

* [ ] real integrated Test operation
* [ ] observe actual behavior
* [ ] collect failures

### HOUR 6

* [ ] deep autopsy
* [ ] identify research/model weaknesses
* [ ] investigate prediction errors

### HOUR 7

* [ ] implement highest-value fixes
* [ ] improve research intelligence
* [ ] improve pattern/regime detection

### HOUR 8

* [ ] rerun system
* [ ] measure improvements
* [ ] attack improvements adversarially

### HOUR 9

* [ ] independent reproduction
* [ ] OOS validation
* [ ] regression audit
* [ ] remaining defects

### HOUR 10

* [ ] final deep audit
* [ ] finish all possible checklist items
* [ ] update evidence
* [ ] identify only genuinely unresolved scientific limitations

These are checkpoints, NOT deadlines.

If a phase takes longer, continue it.

If it finishes early, immediately proceed to the next phase.

---

# 39. FINAL COMPLETION GATE

You may only declare the system complete when ALL applicable conditions are satisfied:

* [ ] foundation complete
* [ ] foundation tested
* [ ] integration complete
* [ ] integration tested
* [ ] intended behavior verified
* [ ] autonomous operation exercised
* [ ] weaknesses discovered
* [ ] weaknesses investigated
* [ ] fixes implemented
* [ ] fixes tested
* [ ] regression testing complete
* [ ] anti-leak testing complete
* [ ] anti-memorization testing complete
* [ ] blind-trader testing complete
* [ ] survivor-bias limitation handled honestly
* [ ] prediction-error loop operational
* [ ] regime-change research operational
* [ ] pattern health operational
* [ ] volatility research operational
* [ ] direction research operational
* [ ] 5–10% selection constraint operational
* [ ] exit intelligence operational
* [ ] self-research operational
* [ ] compute allocation operational
* [ ] C67 mover research operational
* [ ] C68 prediction-error research operational
* [ ] independent reproduction complete
* [ ] Masterstock updated
* [ ] checklist evidence updated
* [ ] no unjustified "VALIDATED" states
* [ ] no known critical blocker ignored

---

# 40. IF NOT COMPLETE

If the complete checklist is not finished after the initial 10-hour work period:

**DO NOT CLAIM COMPLETION.**

Instead:

1. Continue from the exact next unchecked item in the next autonomous session.
2. Preserve all work.
3. Preserve all evidence.
4. Preserve all failures.
5. Preserve all discoveries.
6. Preserve all unfinished items.
7. Do not reset progress.
8. Do not lower requirements.
9. Do not redefine the checklist.
10. Do not declare success because "substantial progress" was made.

---

# 41. FINAL RULE

The standard is NOT:

> "Did we write enough code?"

The standard is NOT:

> "Did the tests pass?"

The standard is NOT:

> "Did the system run?"

The standard is:

> **"Did we build the foundation, prove that it works, prove that it works the way it was intended to work, run it against reality, discover where it fails, teach it from those failures, improve it, and independently prove that the improvement is real?"**

If the answer is not yet yes:

# KEEP WORKING.

# DO NOT SETTLE.

# DO NOT LOWER THE BAR.

# DO NOT STOP.

The system should become better because it learned—not merely because more code was added.
