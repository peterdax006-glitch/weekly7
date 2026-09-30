# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [7, 8], 50 weeks, 40 stocks, learning_claim `record`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 2 of 2 seeds; INVALID verdicts 0; identity-invariant 2; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.021098235973221927; over all seeds (0 where no pick): 0.021098235973221927; vs shuffled-outcome control: 0.021098235973221927
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'acted on, right side': 3, 'admitted, not usable at the probe': 4, 'acted on, wrong side': 1}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | planted | IMPROVED | 6 | 4 | 260 | +0.02121 | +5.98 | +0.02121 | +0.02121 | True |
| 8 | planted | IMPROVED | 5 | 4 | 260 | +0.02099 | +5.36 | +0.02099 | +0.02099 | True |

## seed 7, planted world (4521.7 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 6 knowledge items, 4 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 260 picks, mean edge +0.02121, weekly t +5.98, 530 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 530/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.02121, vs control +0.02121
  planted truth: {'claims': 6, 'true_positive': 5, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.167}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace decaying   f5:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 8, planted world (4271.0 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 5 knowledge items, 4 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 260 picks, mean edge +0.02099, weekly t +5.36, 406 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 406/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.02099, vs control +0.02099
  planted truth: {'claims': 5, 'true_positive': 4, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.2}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, wrong side
  trace regime     f2:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace decaying   f5:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
