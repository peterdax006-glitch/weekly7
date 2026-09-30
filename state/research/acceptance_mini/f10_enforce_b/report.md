# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [5, 6], 50 weeks, 40 stocks, learning_claim `enforce`; code hash `44cc7fb9f6086889`; git `4338632d74e3fb87c98eaef7bb841829167b1139+dirty`
- improved in 0 of 2 seeds; INVALID verdicts 0; identity-invariant 2; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): None; over all seeds (0 where no pick): 0.0; vs shuffled-outcome control: None
- null world (no planted truth): 0 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {'found, not admitted': 8}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | planted | NO CHANGE: the experience did not alter any decision | 7 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 6 | planted | NO CHANGE: the experience did not alter any decision | 7 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 5, planted world (4119.7 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 7 knowledge items, 0 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 7, 'true_positive': 5, 'false_positive': 2, 'recall': 1.0, 'false_discovery_rate': 0.286}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim; learning_claim)
  trace negative   f1:q4    sign -1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```

## seed 6, planted world (2699.2 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 260 weeks -> 7 knowledge items, 0 in production
  probe: 52 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2080 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 7, 'true_positive': 5, 'false_positive': 2, 'recall': 1.0, 'false_discovery_rate': 0.286}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
  note: no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on
  trace strong     f0:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace negative   f1:q4    sign -1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace regime     f2:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace decaying   f5:q4    sign +1: found, not admitted (nnot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion; epistemic GATED cannot be champion)
  trace noise_a    f3:q4    sign +0: never found (no supported belief became knowledge)
  trace noise_b    f4:q0    sign +0: never found (no supported belief became knowledge)
```
