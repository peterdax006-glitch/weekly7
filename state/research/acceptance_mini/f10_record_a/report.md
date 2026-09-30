# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [3, 4], 50 weeks, 40 stocks, learning_claim `record`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 2 of 2 seeds; INVALID verdicts 0; identity-invariant 2; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.028823974257430354; over all seeds (0 where no pick): 0.028823974257430354; vs shuffled-outcome control: 0.028823974257430354
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'acted on, right side': 2, 'admitted, not usable at the probe': 4, 'acted on, wrong side': 1, 'found, not admitted': 1}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | planted | IMPROVED | 8 | 4 | 58 | +0.03861 | +3.88 | +0.03861 | +0.03861 | True |
| 4 | planted | IMPROVED | 6 | 4 | 241 | +0.01904 | +4.34 | +0.01904 | +0.01904 | True |

## seed 3, planted world (4587.4 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 8 knowledge items, 4 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 58 picks, mean edge +0.03861, weekly t +3.88, 58 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 58/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.03861, vs control +0.03861
  planted truth: {'claims': 8, 'true_positive': 5, 'false_positive': 3, 'recall': 1.0, 'false_discovery_rate': 0.375}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace regime     f2:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace decaying   f5:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 4, planted world (5027.8 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 6 knowledge items, 4 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 241 picks, mean edge +0.01904, weekly t +4.34, 275 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 275/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.01904, vs control +0.01904
  planted truth: {'claims': 6, 'true_positive': 4, 'false_positive': 2, 'recall': 1.0, 'false_discovery_rate': 0.333}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, wrong side
  trace regime     f2:q4    sign +1: found, not admitted (K-40350d0a993a@v1: 7 shadow sessions < 8; ; ; ; ; ; ; ; anti_memorization; ; ; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: admitted, not usable at the probe (contract allows=False, retrieval skill PROVEN, lifecycle DEGRADED)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
