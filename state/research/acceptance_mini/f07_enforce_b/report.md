# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [6, 7, 8], 50 weeks, 40 stocks, learning_claim `enforce`; code hash `b34981b71d97b82f`; git `0ea8e2b956e9323f02fd980f1b5cc555e6b626dc+dirty`
- improved in 0 of 3 seeds; INVALID verdicts 0; identity-invariant 3; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): None; over all seeds (0 where no pick): 0.0; vs shuffled-outcome control: None
- null world (no planted truth): 3 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'found, not admitted': 8, 'never found': 1}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | planted | NO CHANGE: the experience did not alter any decision | 2 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 6 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 7 | planted | NO CHANGE: the experience did not alter any decision | 3 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 7 | null | NO CHANGE: the experience did not alter any decision | 1 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 8 | planted | NO CHANGE: the experience did not alter any decision | 4 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 8 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 6, planted world (90.6 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 2 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 2, 'true_positive': 2, 'false_positive': 0, 'recall': 0.667, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity,learning_claim; statistical_validity,learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (statistical_validity,oos_confirmation,learning_claim; statistical_validity,oos_confirmation,learning_claim; statistical_validity,oos_confirmation,learning_claim)
  trace regime     f2:q4    sign +1: never found (no supported belief became knowledge)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 6, null world (53.9 s)

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

## seed 7, planted world (135.9 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 3 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 3, 'true_positive': 3, 'false_positive': 0, 'recall': 1.0, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity,learning_claim; statistical_validity,learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (ty,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_claim; learning_claim; learning_claim)
  trace regime     f2:q4    sign +1: found, not admitted (K-40350d0a993a@v1: 7 shadow sessions < 8; learning_claim)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 7, null world (76.4 s)

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

## seed 8, planted world (155.1 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 4 knowledge items, 0 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 4, 'true_positive': 3, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.25}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (ty,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_claim; statistical_validity,learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (K-88386f3da512@v1: 7 shadow sessions < 8; learning_claim; learning_claim)
  trace regime     f2:q4    sign +1: found, not admitted (K-40350d0a993a@v1: 7 shadow sessions < 8; learning_claim)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 8, null world (49.3 s)

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
