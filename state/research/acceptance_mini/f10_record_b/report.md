# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [5, 6], 50 weeks, 40 stocks, learning_claim `record`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 1 of 1 seeds; INVALID verdicts 0; identity-invariant 1; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.011618434617749477; over all seeds (0 where no pick): 0.011618434617749477; vs shuffled-outcome control: 0.011618434617749477
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'acted on, right side': 2, 'found, not admitted': 2}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | planted | IMPROVED | 7 | 2 | 235 | +0.01162 | +2.10 | +0.01162 | +0.01162 | True |

## seed 5, planted world (3183.9 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 7 knowledge items, 2 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 235 picks, mean edge +0.01162, weekly t +2.10, 277 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 277/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.01162, vs control +0.01162
  planted truth: {'claims': 7, 'true_positive': 5, 'false_positive': 2, 'recall': 1.0, 'false_discovery_rate': 0.286}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: acted on, right side
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
