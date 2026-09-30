# F07 learner diagnosis (C69 W-07; C62 C03, E04, I16, I18)

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world only (C63). Every table below is rendered by `scripts/acceptance_mini.py --diagnose` from the summary.json files named in its header; no number is typed.

Runs: `before-original-3seeds` = state/research/acceptance_mini/before_f07, `pre-F07-code,record` = state/research/acceptance_mini/pre_f07_record_a,state/research/acceptance_mini/pre_f07_record_b, `F07,record` = state/research/acceptance_mini/f07_record_a,state/research/acceptance_mini/f07_record_b, `F07,enforce(default)` = state/research/acceptance_mini/f07_enforce_a,state/research/acceptance_mini/f07_enforce_b

## What was found (method: trace each planted item stage by stage; `learner.truth_trace` now does this on every run)

The original 1/3 result came from code that has since changed. On the code as it stood when F07 started, the learner's default
`learning_claim="enforce"` blocked every promotion on every seed, so nothing could be acted on at all. Two separate problems, in the
order a planted item meets them:

**A. Under `enforce`: found, not admitted (all seeds).** Every promotion attempt of a true item failed the learning-claim gate with
two blockers:
1. *identity: "collapsed under an attack"* - false. The learner audits a rule on its last few weeks (4 held-out dates). The
   `episode_substitution` attack swaps 20-date blocks, so on 4 dates it moved nothing; `verify_transform` then raised a
   `no-op-transform` FAIL and `IdentityReport.passed` went False with every attack verdict OK. FIXED at the source of the call
   (`loop_hooks.episode_attack_kwargs`: the block is sized so the window always holds two episodes). A planted memoriser still
   collapses under the resized attack (test). The wiring blocker text is misleading ("collapsed" when no attack collapsed): it is
   worth fixing in wiring.py (not F07's file).
2. *scorecard: "no learning scorecard supplied"* - structural, NOT fixed. `scorecard_for_learner` needs forward-YEAR folds; a
   50-week planted world lives inside one calendar year, so control E cannot run and the card is refused. Even with a valid card,
   `gate_improvement_claim` requires band share, risk change and drawdown change to be measured, and `scorecard_for_learner`
   leaves those UNTESTED by design ("the claim gate will (correctly) refuse until a portfolio-level card supplies them"). So under
   `enforce` no learned item can ever reach production in any run shorter than several years, and even then only once a
   portfolio-level card exists. It is also circular: the card measures the learner's decisions, and the learner makes no
   decisions until an item is promoted. This is an OWNER DECISION (should per-item promotion require the whole-learner
   section-47 claim?), not something to fix by lowering a gate. The `enforce` run below shows the consequence; the `record` runs
   isolate everything downstream of that gate.

**B. Under `record`: the loss on the failing seeds happened AFTER admission.** Two defects:
1. *Admitted, retrieved, but not acted on - retrieval skill UNPROVEN/FAILED.* The walk-forward `SkillMonitor` only ever saw
   predictions made after an item became CHAMPION. A late promotion left it a few dozen rows (UNPROVEN on seed 4; FAILED on seed
   5 because the rank correlation was slightly negative on near-constant expectations) and `influence=False` withheld every
   decision. FIXED (`learner.shadow_monitoring`, default on): held, not-yet-production items are retrieved by a shadow retriever
   whose predictions are registered before their outcomes exist and scored by the same monitor. It has no monitor of its own,
   never reaches a decision, and registers nothing while frozen or probing (tests). Not a threshold change: the gate's bar is
   untouched, it now sees the shadow period's evidence instead of discarding it.
2. *Admitted, then silently removed - a one-way degradation.* A true champion hit one noisy 8-week window with t below 1.0 and the
   retirement gate moved it to DEGRADED; `RetirementLedger` has no DEGRADED -> ACTIVE door unless the item first went DORMANT,
   the learner only asked for recovery from DORMANT, and the knowledge object never followed a recovery anyway. The decision
   contract then refused it ("lifecycle DEGRADED") for the rest of the run. FIXED (`learner.RecoveringLedger.recover_degraded` +
   `stage_store`): the ledger's OWN recovery bar (recover_min_n outcomes all dated after the degrade, t >= recover_t, which is
   stricter than degrade_t) restores the item through `transition`, and the object's lifecycle/epistemic follow. A test shows
   the old ledger refuses the same evidence; weak, thin, stale, future and empty evidence are all refused.

## What still fails (plainly)

- Under the default `enforce`, the learner improves on 0 seeds: see A2. Nothing is VALIDATED.
- Under `record`, remaining losses (see the trace table): a true item degraded too late in the year to accumulate the 16 post-degrade
  weeks recovery needs; a true item that fails the promotion gate's statistical_validity; the regime item is usually found but not
  admitted (known C62 I10/I13); a long-only learner cannot profit from a negative-only lesson ("CHANGED, NOT IMPROVED").
- NEXT HYPOTHESIS: the 8-week retirement window with degrade_t = 1.0 has low power for planted effects of this size, so true items
  are degraded by noise (false-degrade rate per champion is the next thing to measure on the planted world: count DEGRADE
  transitions on items the truth ledger says are live). Second: the per-item learning-claim deadlock (A2) needs an owner ruling.

## Before / after

| run | seeds | improved | mean vs none (picking seeds) | mean vs none (all seeds) | mean vs control | null seeds | null false improvements | null changed decisions | changed without knowledge |
|---|---|---|---|---|---|---|---|---|---|
| before-original-3seeds | 3 | 1 | +0.01272 | +0.00424 | n/a | not run | not run | not run | 0 |
| pre-F07-code,record | 6 | 1 | +0.01272 | +0.00212 | +0.01272 | 6 | 0 | 0 | 0 |
| F07,record | 6 | 4 | +0.01369 | +0.00913 | +0.01369 | 6 | 0 | 0 | 0 |
| F07,enforce(default) | 6 | 0 | n/a | +0.00000 | n/a | 6 | 0 | 0 | 0 |

## Per seed

| run | seed | world | verdict | production | picks | vs none | skill at probe | gate blockers |
|---|---|---|---|---|---|---|---|---|
| before-original-3seeds | 3 | planted | IMPROVED | 2 | 250 | +0.01272 | n/a (n=n/a) | - |
| before-original-3seeds | 4 | planted | NO CHANGE: the experience did not alter any decision | 2 | 0 | n/a | n/a (n=n/a) | - |
| before-original-3seeds | 5 | planted | NO CHANGE: the experience did not alter any decision | 1 | 0 | n/a | n/a (n=n/a) | - |
| pre-F07-code,record | 3 | planted | IMPROVED | 2 | 250 | +0.01272 | PROVEN (n=75) | anti_memorization, oos_confirmation, statistical_validity |
| pre-F07-code,record | 3 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | anti_memorization, oos_confirmation, statistical_validity |
| pre-F07-code,record | 4 | planted | NO CHANGE: the experience did not alter any decision | 2 | 0 | n/a | UNPROVEN (n=32) | statistical_validity |
| pre-F07-code,record | 4 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| pre-F07-code,record | 5 | planted | NO CHANGE: the experience did not alter any decision | 1 | 0 | n/a | FAILED (n=40) | anti_memorization, oos_confirmation, statistical_validity |
| pre-F07-code,record | 5 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| pre-F07-code,record | 6 | planted | NO CHANGE: the experience did not alter any decision | 1 | 0 | n/a | FAILED (n=40) | oos_confirmation, statistical_validity |
| pre-F07-code,record | 6 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| pre-F07-code,record | 7 | planted | CHANGED, NOT IMPROVED | 1 | 0 | n/a | UNPROVEN (n=40) | statistical_validity |
| pre-F07-code,record | 7 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | anti_memorization, oos_confirmation, statistical_validity |
| pre-F07-code,record | 8 | planted | CHANGED, NOT IMPROVED | 3 | 0 | n/a | UNPROVEN (n=73) | anti_memorization, oos_confirmation, statistical_validity |
| pre-F07-code,record | 8 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,record | 3 | planted | IMPROVED | 2 | 250 | +0.01272 | PROVEN (n=714) | anti_memorization, oos_confirmation, statistical_validity |
| F07,record | 3 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | UNPROVEN (n=112) | anti_memorization, oos_confirmation, statistical_validity |
| F07,record | 4 | planted | IMPROVED | 2 | 249 | +0.02009 | PROVEN (n=364) | statistical_validity |
| F07,record | 4 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,record | 5 | planted | IMPROVED | 1 | 250 | +0.00883 | PROVEN (n=476) | anti_memorization, oos_confirmation, statistical_validity |
| F07,record | 5 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,record | 6 | planted | NO CHANGE: the experience did not alter any decision | 1 | 0 | n/a | PROVEN (n=353) | oos_confirmation, statistical_validity |
| F07,record | 6 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,record | 7 | planted | CHANGED, NOT IMPROVED | 1 | 0 | n/a | PROVEN (n=583) | statistical_validity |
| F07,record | 7 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | FAILED (n=104) | anti_memorization, oos_confirmation, statistical_validity |
| F07,record | 8 | planted | IMPROVED | 3 | 250 | +0.01311 | PROVEN (n=641) | anti_memorization, oos_confirmation, statistical_validity |
| F07,record | 8 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,enforce(default) | 3 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=714) | anti_memorization, learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 3 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | UNPROVEN (n=112) | anti_memorization, learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 4 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=364) | learning_claim, statistical_validity |
| F07,enforce(default) | 4 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,enforce(default) | 5 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=476) | anti_memorization, learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 5 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,enforce(default) | 6 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=353) | learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 6 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |
| F07,enforce(default) | 7 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=583) | learning_claim, statistical_validity |
| F07,enforce(default) | 7 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | FAILED (n=104) | anti_memorization, learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 8 | planted | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | PROVEN (n=641) | anti_memorization, learning_claim, oos_confirmation, statistical_validity |
| F07,enforce(default) | 8 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | n/a | INSUFFICIENT_EVIDENCE (n=0) | - |

## Truth trace: each planted item through store -> retrieval -> knowledge -> decision -> outcome

Stages, in order: never found -> found, not admitted -> admitted, not usable at the probe -> usable, never carried a decision -> acted on, wrong side -> acted on, right side. `outcome` = planted sign x mean realised excess return of the probe rows the item carried (> 0: the probe world paid the call).

| run | seed | item | kind | pattern | sign | belief | role | lifecycle | skill | weighted rows | LONG rows | outcome | lost at | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F07,record | 3 | strong | strong | f0:q4 | +1 | 0.01425 | CHAMPION | ACTIVE | PROVEN | 400 | 250 | 0.01543 | acted on, right side |  |
| F07,record | 3 | negative | negative | f1:q4 | -1 | -0.01223 | CHAMPION | ACTIVE | PROVEN | 400 | 1 | 0.00967 | acted on, right side |  |
| F07,record | 3 | regime | regime | f2:q4 | +1 | 0.00792 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8;  |
| F07,record | 3 | noise_a | noise | f3:q4 | +0 | 0.00201 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 3 | noise_b | noise | f4:q0 | +0 | -0.00106 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 4 | strong | strong | f0:q4 | +1 | 0.01428 | CHAMPION | ACTIVE | PROVEN | 336 | 249 | 0.01896 | acted on, right side |  |
| F07,record | 4 | negative | negative | f1:q4 | -1 | -0.00933 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| F07,record | 4 | regime | regime | f2:q4 | +1 | -0.00036 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 4 | noise_a | noise | f3:q4 | +0 | -0.00145 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 4 | noise_b | noise | f4:q0 | +0 | -0.00056 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 5 | strong | strong | f0:q4 | +1 | 0.01652 | CHAMPION | ACTIVE | PROVEN | 400 | 250 | 0.01136 | acted on, right side |  |
| F07,record | 5 | negative | negative | f1:q4 | -1 | -0.00581 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 5 | regime | regime | f2:q4 | +1 | 0.01118 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | nfirmation,anti_memorization; statistical_validity,oos_confirmation,anti_memorization; sta |
| F07,record | 5 | noise_a | noise | f3:q4 | +0 | -0.00376 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 5 | noise_b | noise | f4:q0 | +0 | -0.00036 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 6 | strong | strong | f0:q4 | +1 | 0.01007 | CHAMPION | DEGRADED | PROVEN | 0 | 0 | None | admitted, not usable at the probe | contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED |
| F07,record | 6 | negative | negative | f1:q4 | -1 | -0.00645 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | tion; statistical_validity,oos_confirmation; statistical_validity,oos_confirmation; statis |
| F07,record | 6 | regime | regime | f2:q4 | +1 | 0.00442 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 6 | noise_a | noise | f3:q4 | +0 | 0.00178 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 6 | noise_b | noise | f4:q0 | +0 | 0.00119 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 7 | strong | strong | f0:q4 | +1 | 0.0123 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity; statistical_validity; ;  |
| F07,record | 7 | negative | negative | f1:q4 | -1 | -0.01712 | CHAMPION | ACTIVE | PROVEN | 338 | 0 | 0.01153 | acted on, right side |  |
| F07,record | 7 | regime | regime | f2:q4 | +1 | 0.00842 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8;  |
| F07,record | 7 | noise_a | noise | f3:q4 | +0 | -0.00054 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 7 | noise_b | noise | f4:q0 | +0 | 0.00097 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 8 | strong | strong | f0:q4 | +1 | 0.01735 | CHAMPION | ACTIVE | PROVEN | 400 | 249 | 0.01418 | acted on, right side |  |
| F07,record | 8 | negative | negative | f1:q4 | -1 | -0.01022 | CHAMPION | ACTIVE | PROVEN | 400 | 1 | 0.00728 | acted on, right side |  |
| F07,record | 8 | regime | regime | f2:q4 | +1 | 0.00765 | CHAMPION | ACTIVE | PROVEN | 400 | 12 | 0.01282 | acted on, right side |  |
| F07,record | 8 | noise_a | noise | f3:q4 | +0 | 0.00359 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,record | 8 | noise_b | noise | f4:q0 | +0 | -0.00185 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 3 | strong | strong | f0:q4 | +1 | 0.01425 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | adow sessions < 8; statistical_validity,learning_claim; statistical_validity,learning_clai |
| F07,enforce(default) | 3 | negative | negative | f1:q4 | -1 | -0.01223 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | adow sessions < 8; statistical_validity,learning_claim; statistical_validity,learning_clai |
| F07,enforce(default) | 3 | regime | regime | f2:q4 | +1 | 0.00792 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8; learning_claim |
| F07,enforce(default) | 3 | noise_a | noise | f3:q4 | +0 | 0.00201 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 3 | noise_b | noise | f4:q0 | +0 | -0.00106 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 4 | strong | strong | f0:q4 | +1 | 0.01428 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity,learning_claim; statistical |
| F07,enforce(default) | 4 | negative | negative | f1:q4 | -1 | -0.00933 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-88386f3da512@v1: 7 shadow sessions < 8; learning_claim; learning_claim |
| F07,enforce(default) | 4 | regime | regime | f2:q4 | +1 | -0.00036 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 4 | noise_a | noise | f3:q4 | +0 | -0.00145 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 4 | noise_b | noise | f4:q0 | +0 | -0.00056 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 5 | strong | strong | f0:q4 | +1 | 0.01652 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ty,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_clai |
| F07,enforce(default) | 5 | negative | negative | f1:q4 | -1 | -0.00581 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 5 | regime | regime | f2:q4 | +1 | 0.01118 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | rmation,anti_memorization,learning_claim; statistical_validity,oos_confirmation,learning_c |
| F07,enforce(default) | 5 | noise_a | noise | f3:q4 | +0 | -0.00376 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 5 | noise_b | noise | f4:q0 | +0 | -0.00036 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 6 | strong | strong | f0:q4 | +1 | 0.01007 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity,learning_claim; statistical |
| F07,enforce(default) | 6 | negative | negative | f1:q4 | -1 | -0.00645 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | statistical_validity,oos_confirmation,learning_claim; statistical_validity,oos_confirmatio |
| F07,enforce(default) | 6 | regime | regime | f2:q4 | +1 | 0.00442 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 6 | noise_a | noise | f3:q4 | +0 | 0.00178 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 6 | noise_b | noise | f4:q0 | +0 | 0.00119 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 7 | strong | strong | f0:q4 | +1 | 0.0123 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity,learning_claim; statistical |
| F07,enforce(default) | 7 | negative | negative | f1:q4 | -1 | -0.01712 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ty,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_clai |
| F07,enforce(default) | 7 | regime | regime | f2:q4 | +1 | 0.00842 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8; learning_claim |
| F07,enforce(default) | 7 | noise_a | noise | f3:q4 | +0 | -0.00054 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 7 | noise_b | noise | f4:q0 | +0 | 0.00097 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 8 | strong | strong | f0:q4 | +1 | 0.01735 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | ty,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_clai |
| F07,enforce(default) | 8 | negative | negative | f1:q4 | -1 | -0.01022 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-88386f3da512@v1: 7 shadow sessions < 8; learning_claim; learning_claim |
| F07,enforce(default) | 8 | regime | regime | f2:q4 | +1 | 0.00765 | RESEARCH | BIRTH | PROVEN | 0 | 0 | None | found, not admitted | K-40350d0a993a@v1: 7 shadow sessions < 8; learning_claim |
| F07,enforce(default) | 8 | noise_a | noise | f3:q4 | +0 | 0.00359 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
| F07,enforce(default) | 8 | noise_b | noise | f4:q0 | +0 | -0.00185 | None | None | PROVEN | 0 | 0 | None | never found | no supported belief became knowledge |
