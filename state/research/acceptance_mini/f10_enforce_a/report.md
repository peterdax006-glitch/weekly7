# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [3, 4], 50 weeks, 40 stocks, learning_claim `enforce`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 0 of 1 seeds; INVALID verdicts 0; identity-invariant 1; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): None; over all seeds (0 where no pick): 0.0; vs shuffled-outcome control: None
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'found, not admitted': 4}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | planted | NO CHANGE: the experience did not alter any decision | 8 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 3, planted world (6063.8 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 8 knowledge items, 0 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 8, 'true_positive': 5, 'false_positive': 3, 'recall': 1.0, 'false_discovery_rate': 0.375}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; anti_memorization,learning_claim; learning_claim; learning_claim; learning_claim)
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
