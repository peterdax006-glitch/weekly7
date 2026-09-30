# F10: the learning claim under `enforce`, with the evidence supplied (C69 W-07; C62 C03, E04, I16, I18)

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world only (C63). Every table is rendered by `scripts/acceptance_mini.py --f10-report` from the summary.json files named here; no number is typed.

Runs: `enforce` = state/research/acceptance_mini/f10_enforce_a,state/research/acceptance_mini/f10_enforce_b,state/research/acceptance_mini/f10_enforce_c,state/research/acceptance_mini/f10_enforce_null, `record` = state/research/acceptance_mini/f10_record_a,state/research/acceptance_mini/f10_record_b,state/research/acceptance_mini/f10_record_c,state/research/acceptance_mini/f10_record_null

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

## Before / after

| run | seeds | improved | mean vs none (picking seeds) | mean vs none (all seeds) | mean vs control | null seeds | null false improvements | null changed decisions | changed without knowledge |
|---|---|---|---|---|---|---|---|---|---|
| enforce | 4 | 1 | +0.01024 | +0.00256 | +0.01024 | 6 | 0 | 0 | 0 |
| record | 6 | 5 | +0.01724 | +0.01724 | +0.01724 | 6 | 0 | 0 | 0 |

## Degrade calls against the planted truth (totals)

| run | false DEGRADEs | correct DEGRADEs | null-world DEGRADEs | false by kind | weeks not ACTIVE by kind (per seed) |
|---|---|---|---|---|---|
| enforce | 11 | 4 | 0 | {'decaying': 1, 'negative': 3, 'regime': 6, 'strong': 1} | {'decaying': [112, 194, 105, 100], 'negative': [13, 75, 75, 0], 'regime': [171, 234, 106, 172], 'strong': [0, 0, 75, 0]} |
| record | 17 | 6 | 0 | {'decaying': 2, 'negative': 6, 'regime': 8, 'strong': 1} | {'decaying': [112, 140, 194, 105, 100, 112], 'negative': [13, 66, 75, 75, 0, 33], 'regime': [171, 22, 234, 106, 172, 190], 'strong': [0, 0, 0, 75, 0, 0]} |

## Per seed: what the claim gate saw and what still blocks

| run | seed | world | verdict | production | vs none | vs control | cards valid / built | first valid card | first card the claim gate allowed | claim-gate blockers (last card) | last promotion attempt per pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enforce | 3 | planted | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 26 / 41 | 2011-02-04 | never | not_memoriser | f0:q4: learning_claim; f1:q0: learning_claim; f1:q4: learning_claim; f2:q0: anti_memorization,learning_claim,oos_confirmation; f2:q4: anti_memorization,learning_claim; f3:q0: learning_claim; f5:q0: an |
| enforce | 5 | planted | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 26 / 41 | 2011-02-04 | never | not_memoriser | f0:q0: anti_memorization,learning_claim,oos_confirmation,statistical_validity; f0:q4: learning_claim; f1:q0: learning_claim,oos_confirmation; f1:q4: learning_claim; f2:q4: anti_memorization,learning_c |
| enforce | 6 | planted | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 15 / 29 | 2011-02-04 | never | not_memoriser | f0:q0: learning_claim,statistical_validity; f0:q4: learning_claim; f1:q0: learning_claim,statistical_validity; f1:q4: learning_claim; f2:q4: anti_memorization,learning_claim; f5:q0: learning_claim,oos |
| enforce | 7 | planted | IMPROVED | 1 | +0.01024 | +0.01024 | 26 / 41 | 2011-02-04 | 2013-01-18 | - | f0:q0: learning_claim,oos_confirmation; f0:q4: learning_claim; f1:q0: learning_claim; f1:q4: PROMOTED; f2:q4: anti_memorization,learning_claim; f5:q4: learning_claim |
| enforce | 3 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 2 | - | never | refused: E_leaky_learner: 'detected' must be recorded (was the planted defect seen?) | f5:q0: learning_claim,oos_confirmation,statistical_validity |
| enforce | 4 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |
| enforce | 5 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |
| enforce | 6 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 2 / 2 | 2011-02-04 | never | learning_gain, beats_luck_floor, not_memoriser, cross_year_transfer, stability | f0:q0: anti_memorization,learning_claim,oos_confirmation |
| enforce | 7 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 2 | - | never | refused: E_leaky_learner: 'detected' must be recorded (was the planted defect seen?) | f1:q4: anti_memorization,learning_claim,oos_confirmation,statistical_validity |
| enforce | 8 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |
| record | 3 | planted | IMPROVED | 4 | +0.03861 | +0.03861 | 11 / 20 | 2011-04-29 | never | not_memoriser | f0:q4: PROMOTED; f1:q0: PROMOTED; f1:q4: PROMOTED; f2:q0: anti_memorization,oos_confirmation; f2:q4: PROMOTED; f3:q0: PROMOTED; f5:q0: anti_memorization,oos_confirmation; f5:q4: PROMOTED |
| record | 4 | planted | IMPROVED | 4 | +0.01904 | +0.01904 | 10 / 22 | 2012-09-14 | 2013-01-18 | - | f0:q0: PROMOTED; f0:q4: PROMOTED; f1:q4: PROMOTED; f2:q4: PROMOTED; f5:q0: anti_memorization,oos_confirmation; f5:q4: PROMOTED |
| record | 5 | planted | IMPROVED | 2 | +0.01162 | +0.01162 | 8 / 19 | 2011-02-04 | never | stability | f0:q0: anti_memorization,oos_confirmation,statistical_validity; f0:q4: PROMOTED; f1:q0: oos_confirmation; f1:q4: PROMOTED; f2:q4: anti_memorization,oos_confirmation,statistical_validity; f5:q0: anti_m |
| record | 6 | planted | CHANGED, NOT IMPROVED | 3 | -0.00801 | -0.00801 | 5 / 16 | 2012-05-11 | never | not_memoriser | f0:q0: statistical_validity; f0:q4: PROMOTED; f1:q0: statistical_validity; f1:q4: PROMOTED; f2:q4: anti_memorization; f5:q0: oos_confirmation,statistical_validity; f5:q4: PROMOTED |
| record | 7 | planted | IMPROVED | 4 | +0.02121 | +0.02121 | 0 / 14 | - | never | refused: E_leaky_learner: 'detected' must be recorded (was the planted defect seen?) | f0:q0: oos_confirmation; f0:q4: PROMOTED; f1:q0: PROMOTED; f1:q4: PROMOTED; f2:q4: PROMOTED; f5:q4: PROMOTED |
| record | 8 | planted | IMPROVED | 4 | +0.02099 | +0.02099 | 2 / 8 | 2013-11-08 | 2013-11-08 | - | f0:q0: anti_memorization,oos_confirmation; f0:q4: PROMOTED; f1:q4: PROMOTED; f2:q4: PROMOTED; f5:q4: PROMOTED |
| record | 3 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 2 | - | never | refused: E_leaky_learner: 'detected' must be recorded (was the planted defect seen?) | f5:q0: oos_confirmation,statistical_validity |
| record | 4 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |
| record | 5 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |
| record | 6 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 2 / 2 | 2011-02-04 | never | learning_gain, beats_luck_floor, not_memoriser, cross_year_transfer, stability | f0:q0: anti_memorization,oos_confirmation |
| record | 7 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 2 | - | never | refused: E_leaky_learner: 'detected' must be recorded (was the planted defect seen?) | f1:q4: anti_memorization,oos_confirmation,statistical_validity |
| record | 8 | null | NO CHANGE: the experience did not alter any decision | 0 | n/a | n/a | 0 / 0 | - | never | - | no attempt |

## Every planted item's lifecycle

| run | seed | world | item | kind | registered (week) | DEGRADEs | of them FALSE | full recoveries | DORMANT | weeks not ACTIVE |
|---|---|---|---|---|---|---|---|---|---|---|
| enforce | 3 | planted | decaying | decaying | 7 | 1 | 0 | 0 | 1 | 112 |
| enforce | 3 | planted | negative | negative | 15 | 1 | 1 | 0 | 0 | 13 |
| enforce | 3 | planted | regime | regime | 37 | 1 | 1 | 0 | 1 | 171 |
| enforce | 3 | planted | strong | strong | 18 | 0 | 0 | 0 | 0 | 0 |
| enforce | 5 | planted | decaying | decaying | 12 | 2 | 1 | 1 | 2 | 194 |
| enforce | 5 | planted | negative | negative | 52 | 1 | 1 | 1 | 1 | 75 |
| enforce | 5 | planted | regime | regime | 7 | 1 | 1 | 0 | 1 | 234 |
| enforce | 5 | planted | strong | strong | 9 | 0 | 0 | 0 | 0 | 0 |
| enforce | 6 | planted | decaying | decaying | 12 | 1 | 0 | 0 | 1 | 105 |
| enforce | 6 | planted | negative | negative | 48 | 1 | 1 | 1 | 1 | 75 |
| enforce | 6 | planted | regime | regime | 54 | 2 | 2 | 1 | 2 | 106 |
| enforce | 6 | planted | strong | strong | 16 | 1 | 1 | 1 | 1 | 75 |
| enforce | 7 | planted | decaying | decaying | 18 | 1 | 0 | 0 | 1 | 100 |
| enforce | 7 | planted | negative | negative | 7 | 0 | 0 | 0 | 0 | 0 |
| enforce | 7 | planted | regime | regime | 39 | 2 | 2 | 1 | 2 | 172 |
| enforce | 7 | planted | strong | strong | 17 | 0 | 0 | 0 | 0 | 0 |
| record | 3 | planted | decaying | decaying | 7 | 1 | 0 | 0 | 1 | 112 |
| record | 3 | planted | negative | negative | 15 | 1 | 1 | 0 | 0 | 13 |
| record | 3 | planted | regime | regime | 37 | 1 | 1 | 0 | 1 | 171 |
| record | 3 | planted | strong | strong | 18 | 0 | 0 | 0 | 0 | 0 |
| record | 4 | planted | decaying | decaying | 9 | 2 | 1 | 1 | 1 | 140 |
| record | 4 | planted | negative | negative | 32 | 2 | 2 | 2 | 0 | 66 |
| record | 4 | planted | regime | regime | 181 | 1 | 1 | 0 | 1 | 22 |
| record | 4 | planted | strong | strong | 16 | 0 | 0 | 0 | 0 | 0 |
| record | 5 | planted | decaying | decaying | 12 | 2 | 1 | 1 | 2 | 194 |
| record | 5 | planted | negative | negative | 52 | 1 | 1 | 1 | 1 | 75 |
| record | 5 | planted | regime | regime | 7 | 1 | 1 | 0 | 1 | 234 |
| record | 5 | planted | strong | strong | 9 | 0 | 0 | 0 | 0 | 0 |
| record | 6 | planted | decaying | decaying | 12 | 1 | 0 | 0 | 1 | 105 |
| record | 6 | planted | negative | negative | 48 | 1 | 1 | 1 | 1 | 75 |
| record | 6 | planted | regime | regime | 54 | 2 | 2 | 1 | 2 | 106 |
| record | 6 | planted | strong | strong | 16 | 1 | 1 | 1 | 1 | 75 |
| record | 7 | planted | decaying | decaying | 18 | 1 | 0 | 0 | 1 | 100 |
| record | 7 | planted | negative | negative | 7 | 0 | 0 | 0 | 0 | 0 |
| record | 7 | planted | regime | regime | 39 | 2 | 2 | 1 | 2 | 172 |
| record | 7 | planted | strong | strong | 17 | 0 | 0 | 0 | 0 | 0 |
| record | 8 | planted | decaying | decaying | 9 | 1 | 0 | 0 | 1 | 112 |
| record | 8 | planted | negative | negative | 28 | 1 | 1 | 1 | 0 | 33 |
| record | 8 | planted | regime | regime | 39 | 1 | 1 | 0 | 1 | 190 |
| record | 8 | planted | strong | strong | 7 | 0 | 0 | 0 | 0 | 0 |

## Truth trace

Stages, in order: never found -> found, not admitted -> admitted, not usable at the probe -> usable, never carried a decision -> acted on, wrong side -> acted on, right side.

| run | seed | item | kind | pattern | sign | belief | role | lifecycle | skill | weighted rows | LONG rows | outcome | lost at | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enforce | 3 | strong | strong | f0:q4 | +1 | 0.01321 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning |
| enforce | 3 | negative | negative | f1:q4 | -1 | -0.01159 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; anti_memor |
| enforce | 3 | regime | regime | f2:q4 | +1 | 0.00829 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 3 | decaying | decaying | f5:q4 | +1 | 0.00599 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 3 | noise_a | noise | f3:q4 | +0 | 0.00172 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 3 | noise_b | noise | f4:q0 | +0 | 0.00014 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 5 | strong | strong | f0:q4 | +1 | 0.01457 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning |
| enforce | 5 | negative | negative | f1:q4 | -1 | -0.01033 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 5 | regime | regime | f2:q4 | +1 | 0.00669 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 5 | decaying | decaying | f5:q4 | +1 | 0.00627 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 5 | noise_a | noise | f3:q4 | +0 | -0.00138 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 5 | noise_b | noise | f4:q0 | +0 | -0.00087 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 6 | strong | strong | f0:q4 | +1 | 0.01376 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 6 | negative | negative | f1:q4 | -1 | -0.01135 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 6 | regime | regime | f2:q4 | +1 | 0.00981 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 6 | decaying | decaying | f5:q4 | +1 | 0.00446 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 6 | noise_a | noise | f3:q4 | +0 | 1e-05 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 6 | noise_b | noise | f4:q0 | +0 | 0.00074 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 7 | strong | strong | f0:q4 | +1 | 0.01437 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning |
| enforce | 7 | negative | negative | f1:q4 | -1 | -0.01206 | CHAMPION | ACTIVE | PROVEN | 191 | 73 | 0.00258 | acted on, right side |  |
| enforce | 7 | regime | regime | f2:q4 | +1 | 0.007 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 7 | decaying | decaying | f5:q4 | +1 | 0.00544 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| enforce | 7 | noise_a | noise | f3:q4 | +0 | 0.00142 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| enforce | 7 | noise_b | noise | f4:q0 | +0 | 0.00061 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 3 | strong | strong | f0:q4 | +1 | 0.01321 | CHAMPION | ACTIVE | PROVEN | 58 | 58 | 0.03861 | acted on, right side |  |
| record | 3 | negative | negative | f1:q4 | -1 | -0.01159 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 3 | regime | regime | f2:q4 | +1 | 0.00829 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 3 | decaying | decaying | f5:q4 | +1 | 0.00599 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 3 | noise_a | noise | f3:q4 | +0 | 0.00172 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 3 | noise_b | noise | f4:q0 | +0 | 0.00014 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 4 | strong | strong | f0:q4 | +1 | 0.01411 | CHAMPION | ACTIVE | PROVEN | 270 | 240 | 0.01975 | acted on, right side |  |
| record | 4 | negative | negative | f1:q4 | -1 | -0.00992 | CHAMPION | ACTIVE | PROVEN | 5 | 1 | -0.0394 | acted on, wrong side |  |
| record | 4 | regime | regime | f2:q4 | +1 | 0.00607 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8; ; ; ; ; ; ; ; anti_memorization; ; ; epistemic G |
| record | 4 | decaying | decaying | f5:q4 | +1 | 0.0058 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 4 | noise_a | noise | f3:q4 | +0 | 0.00019 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 4 | noise_b | noise | f4:q0 | +0 | 0.00121 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 5 | strong | strong | f0:q4 | +1 | 0.01457 | CHAMPION | ACTIVE | PROVEN | 258 | 235 | 0.01194 | acted on, right side |  |
| record | 5 | negative | negative | f1:q4 | -1 | -0.01033 | CHAMPION | ACTIVE | PROVEN | 19 | 0 | 0.00553 | acted on, right side |  |
| record | 5 | regime | regime | f2:q4 | +1 | 0.00669 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| record | 5 | decaying | decaying | f5:q4 | +1 | 0.00627 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| record | 5 | noise_a | noise | f3:q4 | +0 | -0.00138 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 5 | noise_b | noise | f4:q0 | +0 | -0.00087 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 6 | strong | strong | f0:q4 | +1 | 0.01376 | CHAMPION | ACTIVE | PROVEN | 18 | 18 | 0.01652 | acted on, right side |  |
| record | 6 | negative | negative | f1:q4 | -1 | -0.01135 | CHAMPION | ACTIVE | PROVEN | 83 | 83 | 0.01333 | acted on, wrong side |  |
| record | 6 | regime | regime | f2:q4 | +1 | 0.00981 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion;  |
| record | 6 | decaying | decaying | f5:q4 | +1 | 0.00446 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 6 | noise_a | noise | f3:q4 | +0 | 1e-05 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 6 | noise_b | noise | f4:q0 | +0 | 0.00074 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 7 | strong | strong | f0:q4 | +1 | 0.01437 | CHAMPION | ACTIVE | PROVEN | 339 | 258 | 0.02051 | acted on, right side |  |
| record | 7 | negative | negative | f1:q4 | -1 | -0.01206 | CHAMPION | ACTIVE | PROVEN | 191 | 2 | 0.00258 | acted on, right side |  |
| record | 7 | regime | regime | f2:q4 | +1 | 0.007 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 7 | decaying | decaying | f5:q4 | +1 | 0.00544 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 7 | noise_a | noise | f3:q4 | +0 | 0.00142 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 7 | noise_b | noise | f4:q0 | +0 | 0.00061 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 8 | strong | strong | f0:q4 | +1 | 0.01345 | CHAMPION | ACTIVE | PROVEN | 346 | 258 | 0.01721 | acted on, right side |  |
| record | 8 | negative | negative | f1:q4 | -1 | -0.01148 | CHAMPION | ACTIVE | PROVEN | 60 | 2 | -0.00297 | acted on, wrong side |  |
| record | 8 | regime | regime | f2:q4 | +1 | 0.00766 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 8 | decaying | decaying | f5:q4 | +1 | 0.00407 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| record | 8 | noise_a | noise | f3:q4 | +0 | 0.0012 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| record | 8 | noise_b | noise | f4:q0 | +0 | -0.00117 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
