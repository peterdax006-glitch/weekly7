# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [6, 7, 8], 50 weeks, 40 stocks, learning_claim `record`; code hash `090248da16477772`; git `0ea8e2b956e9323f02fd980f1b5cc555e6b626dc+dirty`
- improved in 1 of 3 seeds; INVALID verdicts 0; identity-invariant 3; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.013113574721540318; over all seeds (0 where no pick): 0.004371191573846773; vs shuffled-outcome control: 0.013113574721540318
- null world (no planted truth): 3 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'admitted, not usable at the probe': 1, 'found, not admitted': 3, 'never found': 1, 'acted on, right side': 4}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | planted | NO CHANGE: the experience did not alter any decision | 2 | 1 | 0 | n/a | n/a | n/a | n/a | True |
| 6 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 7 | planted | CHANGED, NOT IMPROVED | 3 | 1 | 0 | n/a | n/a | n/a | n/a | True |
| 7 | null | NO CHANGE: the experience did not alter any decision | 1 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 8 | planted | IMPROVED | 4 | 3 | 250 | +0.01311 | +3.22 | +0.01311 | +0.01311 | True |
| 8 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 6, planted world (121.4 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 2 knowledge items, 1 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 2, 'true_positive': 2, 'false_positive': 0, 'recall': 0.667, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  trace strong     f0:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace negative   f1:q4    sign -1: found, not admitted (tion; statistical_validity,oos_confirmation; statistical_validity,oos_confirmation; statistical_validity,oos_confirmation; statistical_validity,oos_confirmation)
  trace regime     f2:q4    sign +1: never found (no supported belief became knowledge)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 6, null world (52.8 s)

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

## seed 7, planted world (181.1 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 3 knowledge items, 1 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 338 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 338/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 3, 'true_positive': 3, 'false_positive': 0, 'recall': 1.0, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (CHANGED, NOT IMPROVED)
  trace strong     f0:q4    sign +1: found, not admitted (K-1f1a7d9267ed@v1: 7 shadow sessions < 8; statistical_validity; statistical_validity; ; )
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: found, not admitted (K-40350d0a993a@v1: 7 shadow sessions < 8; )
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 7, null world (66.0 s)

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

## seed 8, planted world (258.6 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 4 knowledge items, 3 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 250 picks, mean edge +0.01311, weekly t +3.22, 981 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 981/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.01311, vs control +0.01311
  planted truth: {'claims': 4, 'true_positive': 3, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.25}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: acted on, right side
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 8, null world (48.5 s)

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
