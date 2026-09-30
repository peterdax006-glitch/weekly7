# Section-87 acceptance, miniature

Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by `scripts/acceptance_mini.py`, none are typed.

- seeds [3, 4, 5], 50 weeks, 40 stocks, learning_claim `record`; code hash `9e97cc203ca2d234`; git ``
- improved in 1 of 3 seeds; INVALID verdicts 0; identity-invariant 3; decisions changed without knowledge behind them: 0
- mean improvement vs no lesson (seeds that picked): 0.012724441266454983; over all seeds (0 where no pick): 0.004241480422151661; vs shuffled-outcome control: 0.012724441266454983
- null world (no planted truth): 3 seeds, false improvements 0, changed decisions 0, items in production 0
- planted true items by the stage their signal was lost at: {}

| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | planted | IMPROVED | 5 | 2 | 250 | +0.01272 | +3.59 | +0.01272 | +0.01272 | True |
| 3 | null | NO CHANGE: the experience did not alter any decision | 1 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 4 | planted | NO CHANGE: the experience did not alter any decision | 2 | 2 | 0 | n/a | n/a | n/a | n/a | True |
| 4 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |
| 5 | planted | NO CHANGE: the experience did not alter any decision | 2 | 1 | 0 | n/a | n/a | n/a | n/a | True |
| 5 | null | NO CHANGE: the experience did not alter any decision | 0 | 0 | 0 | n/a | n/a | n/a | n/a | True |

## seed 3, planted world (118.4 s)

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
```

## seed 3, null world (55.7 s)

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
```

## seed 4, planted world (95.5 s)

```
Section-87 acceptance (miniature)   [IMPLEMENTED - NOT VALIDATED]
  learned 50 weeks -> 2 knowledge items, 2 in production
  probe: 50 weeks of a different episode under new identities
  lesson learner : 0 picks, mean edge n/a, weekly t n/a, 0 rows backed by knowledge
  control learner: 0 picks, mean edge n/a, weekly t n/a
  decisions changed vs a learner without the lesson: 0/2000 (0 without knowledge)
  identity-invariant on year A under new identities: True
  improvement vs no lesson n/a, vs control n/a
  planted truth: {'claims': 2, 'true_positive': 2, 'false_positive': 0, 'recall': 0.667, 'false_discovery_rate': 0.0}
  verdict: NOT DEMONSTRATED  (NO CHANGE: the experience did not alter any decision)
```

## seed 4, null world (47.7 s)

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
```

## seed 5, planted world (79.2 s)

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
```

## seed 5, null world (45.5 s)

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
```
