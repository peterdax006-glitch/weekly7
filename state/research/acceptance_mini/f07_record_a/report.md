# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [3, 4, 5], 50 weeks, 40 stocks, learning_claim `record`; code hash `090248da16477772`; git `0ea8e2b956e9323f02fd980f1b5cc555e6b626dc+dirty`
- improved in 3 of 3 seeds; INVALID verdicts 0; identity-invariant 3; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.01388385535774553; over all seeds (0 where no pick): 0.01388385535774553; vs shuffled-outcome control: 0.01388385535774553
- null world (no planted truth): 3 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'acted on, right side': 4, 'found, not admitted': 2, 'admitted, not usable at the probe': 1, 'never found': 2}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | planted | IMPROVED | 5 | 2 | 250 | +0.01272 | +3.59 | +0.01272 | +0.01272 | True |
| 3 | null | NO CHANGE: the experience did not alter any decision | 1 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 4 | planted | IMPROVED | 2 | 2 | 249 | +0.02009 | +5.09 | +0.02009 | +0.02009 | True |
| 4 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 5 | planted | IMPROVED | 2 | 1 | 250 | +0.00883 | +2.11 | +0.00883 | +0.00883 | True |
| 5 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 3, planted world (256.2 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 5 knowledge items, 2 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 250 picks, mean edge +0.01272, weekly t +3.59, 722 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 722/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.01272, vs control +0.01272
  planted truth: {'claims': 5, 'true_positive': 4, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.2}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: found, not admitted (K-40350d0a993a@v1: 7 shadow sessions < 8; )
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 3, null world (90.6 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 1 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 1, 'true_positive': 0, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 1.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 4, planted world (160.3 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 2 knowledge items, 2 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 249 picks, mean edge +0.02009, weekly t +5.09, 336 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 336/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.02009, vs control +0.02009
  planted truth: {'claims': 2, 'true_positive': 2, 'false_positive': 0, 'recall': 0.667, 'false_discovery_rate': 0.0}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace regime     f2:q4    sign +1: never found (no supported belief became knowledge)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 4, null world (48.7 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 0 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 0, 'true_positive': 0, 'false_positive': 0, 'recall': 1.0, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 5, planted world (130.6 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 2 knowledge items, 1 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 250 picks, mean edge +0.00883, weekly t +2.11, 400 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 400/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.00883, vs control +0.00883
  planted truth: {'claims': 2, 'true_positive': 2, 'false_positive': 0, 'recall': 0.667, 'false_discovery_rate': 0.0}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: never found (no supported belief became knowledge)
  trace regime     f2:q4    sign +1: found, not admitted (nfirmation,anti_memorization; statistical_validity,oos_confirmation,anti_memorization; statistical_validity,oos_confirmation; oos_confirmation; oos_confirmation)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 5, null world (48.0 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 0 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 0, 'true_positive': 0, 'false_positive': 0, 'recall': 1.0, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
