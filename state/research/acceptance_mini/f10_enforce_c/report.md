# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [7, 8], 50 weeks, 40 stocks, learning_claim `enforce`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 1 of 2 seeds; INVALID verdicts 0; identity-invariant 2; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.010243590705928176; over all seeds (0 where no pick): 0.005121795352964088; vs shuffled-outcome control: 0.010243590705928176
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'found, not admitted': 7, 'acted on, right side': 1}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | planted | IMPROVED | 6 | 1 | 73 | +0.01024 | +1.87 | +0.01024 | +0.01024 | True |
| 8 | planted | NO CHANGE: the experience did not alter any decision | 5 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 7, planted world (5439.7 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 6 knowledge items, 1 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 73 picks, mean edge +0.01024, weekly t +1.87, 191 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 191/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson +0.01024, vs control +0.01024
  planted truth: {'claims': 6, 'true_positive': 5, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.167}
  verdict: IMPROVED  (IMPROVED)
  trace strong     f0:q4    sign +1: found, not admitted (; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: acted on, right side
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 8, planted world (7983.3 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 5 knowledge items, 0 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 5, 'true_positive': 4, 'false_positive': 1, 'recall': 1.0, 'false_discovery_rate': 0.2}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim)
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
