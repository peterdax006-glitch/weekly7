# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [3, 4], 50 weeks, 40 stocks, learning_claim `enforce`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 1 of 2 seeds; INVALID verdicts 0; identity-invariant 2; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.019042535803607784; over all seeds (0 where no pick): 0.009521267901803892; vs shuffled-outcome control: 0.019042535803607784
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'found, not admitted': 6, 'acted on, right side': 1, 'acted on, wrong side': 1}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | planted | NO CHANGE: the experience did not alter any decision | 8 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 4 | planted | IMPROVED | 6 | 2 | 241 | +0.01904 | +4.34 | +0.01904 | +0.01904 | True |

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

## seed 4, planted world (8638.5 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 6 knowledge items, 2 in production
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
  trace regime     f2:q4    sign +1: found, not admitted (sions < 8; learning_claim; learning_claim; learning_claim; ; ; ; ; anti_memorization; ; ; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
