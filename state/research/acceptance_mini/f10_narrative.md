## What F10 changed (the gate did not change)

Owner ruling (29 Sep): `learning_claim="enforce"` stays the default and its bar is not lowered; the evidence must be supplied.
`gate_improvement_claim` and the PromotionGate are untouched. What was added:

1. **A world long enough to judge a claim.** `planted_world.multi_year_spec` (5 years here): a strong long, a strong short and a
   regime-gated long that persist through every year, one item that decays (full to year 2, dead from year 3), two noise cells.
   The null world has the same length. `scorecard_for_learner` needs two training years per held-out year and its stability check
   three held-out years, so a claim can be allowed from year 5 at the earliest; 4 years leave stability UNTESTED.
2. **The evidence card breaks the F07 circle.** The learner keeps a *shadow book*: for every row of every learning week, what all
   held knowledge (production and not yet promoted) expected, written before the outcome existed. `LegitimateLearner.evidence_card`
   runs `scorecard_for_learner` on that book (controls A-E over forward-year folds, stock and regime folds, memorisation and identity
   gaps, leak probe) and merges `scorecard.portfolio_card` into it: risk (worst-5% week), drawdown and 5-10% band share of the book the
   learner would hold against a same-size no-knowledge book (per 13-week window, `learning_curve.compute_learning_delta`), and
   calibration (ECE of Phi(expected/scale)). A field that cannot be measured stays UNTESTED. The card is registered for the learning
   claim; an invalid card registers nothing, and a learner is never judged on another learner's card left in the process-wide hub.
3. **False degrades were measured and the window changed on that evidence** (`--degrade-study`, table in f10_degrade_study.md):
   with an 8-week window at degrade_t 1.0 every planted true item was degraded at least once in five years and spent a large share
   of weeks inactive, and the decaying item was falsely degraded before its change in about half the histories. 16 weeks, same
   degrade_t, cut both sharply, still demotes noise almost always and still detects the real decay (with a delay). `retire_window`
   is now 16; the ledger-wide floor `min_n` stays 8 because the lifecycle machine writes to the same ledger. The regime item stays
   badly served by any unconditional window (known C62 I10/I13).
4. **Bugs the 5-year runs exposed and fixed** (each with a test): a relabelled challenger could be promoted by the board and then
   refused by the knowledge schema; a DORMANT item could never recover (it was offered an 8-week window against a 16-outcome bar) and
   a probation item was never asked again (`retirement.window_check` now serves both doors, shared by the learner and the study); two
   writers in one tick (window check and lifecycle machine) crashed on an illegal move (`RetirementLedger.settled`); retiring an item
   years after its birth failed the archive's learned-at check.
5. `recover_degraded` now lives in `RetirementLedger`; the identity blocker says what is true of the report (collapsed / not judged /
   nondeterministic / harness finding) instead of always "collapsed".
