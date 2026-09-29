# Pattern reliability on the real weekly panel

49 patterns, 716 weeks. Every number below is out of sample: predictions, gates and verdicts use only evidence matured before the week they act on.

## Can reliability be predicted? (walk-forward, calibrated)
| model | n | hit rate | Brier model | Brier pooled base | Brier own base | skill vs pool | skill t | ECE | AUC |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 9939 | 0.504 | 0.2506 | 0.2501 | 0.2538 | -0.20% | -1.78 | 0.023 | 0.498 |
| pooled_ctx | 9939 | 0.504 | 0.2510 | 0.2501 | 0.2538 | -0.39% | -2.15 | 0.024 | 0.497 |
| own_history | 9939 | 0.504 | 0.2506 | 0.2501 | 0.2538 | -0.22% | -2.13 | 0.025 | 0.498 |
| per_pattern | 9939 | 0.504 | 0.2858 | 0.2501 | 0.2538 | -14.28% | -11.39 | 0.135 | 0.496 |
| base_own | 9939 | 0.504 | 0.2538 | 0.2501 | 0.2538 | -1.51% | -3.92 | 0.050 | 0.504 |

## What gating buys (mean weekly portfolio return of equal-weighted patterns; off = cash)

| policy | weeks | mean | Sharpe | +weeks | worst 5% | avg weight | gain vs always-on [95% CI] | gain vs best discard [95% CI] |
|---|---|---|---|---|---|---|---|---|
| always_on | 466 | +0.002% | 0.04 | 50.4% | -0.65% | 1.00 |  | -0.004% [-0.032%, +0.020%] |
| discard_forever | 466 | -0.006% | -0.19 | 45.3% | -0.26% | 0.12 | -0.008% [-0.043%, +0.027%] | -0.012% [-0.034%, +0.008%] |
| discard_year | 466 | +0.006% | 0.13 | 52.8% | -0.38% | 0.37 | +0.003% [-0.024%, +0.032%] | -0.001% [-0.014%, +0.010%] |
| discard_revive | 466 | +0.007% | 0.14 | 50.4% | -0.46% | 0.45 | +0.004% [-0.020%, +0.032%] |  |
| gated_raw | 466 | -0.011% | -0.22 | 39.1% | -0.56% | 0.67 | -0.013% [-0.040%, +0.013%] | -0.018% [-0.046%, +0.008%] |
| gated_c60 | 466 | +0.017% | 0.34 | 51.7% | -0.48% | 0.55 | +0.015% [-0.008%, +0.038%] | +0.010% [-0.015%, +0.032%] |
| gated_c61 | 466 | +0.007% | 0.14 | 50.4% | -0.46% | 0.45 | +0.004% [-0.020%, +0.032%] | +0.000% [+0.000%, +0.000%] |
| oracle | 466 | +0.586% | 12.04 | 100.0% | +0.13% | 0.50 | +0.584% [+0.536%, +0.641%] | +0.580% [+0.530%, +0.631%] |

Best discard baseline: discard_revive. Switched-off vs used pattern-weeks (mean signed return):
- gated_raw: used +0.002% (n=7446), switched off +0.075% (n=2493)
- gated_c60: used +0.043% (n=5372), switched off -0.006% (n=4567)
- gated_c61: used +0.044% (n=4463), switched off +0.001% (n=5476)

## Per era: gain over always-on
| era | policy | weeks | policy mean | always-on mean | gain [95% CI] |
|---|---|---|---|---|---|
| 2016-2018 | discard_revive | 63 | +0.005% | -0.015% | +0.020% [-0.041%, +0.086%] |
| 2016-2018 | gated_c60 | 63 | +0.006% | -0.015% | +0.021% [-0.031%, +0.072%] |
| 2016-2018 | gated_c61 | 63 | +0.005% | -0.015% | +0.020% [-0.041%, +0.086%] |
| 2019-2021 | discard_revive | 157 | -0.027% | -0.043% | +0.016% [-0.037%, +0.075%] |
| 2019-2021 | gated_c60 | 157 | +0.005% | -0.043% | +0.048% [+0.006%, +0.099%] |
| 2019-2021 | gated_c61 | 157 | -0.027% | -0.043% | +0.016% [-0.037%, +0.075%] |
| 2022-2024 | discard_revive | 156 | +0.010% | +0.040% | -0.030% [-0.061%, +0.001%] |
| 2022-2024 | gated_c60 | 156 | +0.032% | +0.040% | -0.007% [-0.038%, +0.023%] |
| 2022-2024 | gated_c61 | 156 | +0.010% | +0.040% | -0.030% [-0.061%, +0.001%] |
| 2025-2026 | discard_revive | 90 | +0.060% | +0.029% | +0.031% [-0.022%, +0.090%] |
| 2025-2026 | gated_c60 | 90 | +0.019% | +0.029% | -0.010% [-0.054%, +0.036%] |
| 2025-2026 | gated_c61 | 90 | +0.060% | +0.029% | +0.031% [-0.022%, +0.090%] |

## Pattern health and investigations (C61)

Broken episodes detected: 66 (phantom at birth: 41); broken now: 26; false-alarm design: one per 500 pattern-weeks.
Verdicts: PHANTOM_DISCARDED=41, DISCARDED_UNPREDICTABLE=25.
Unknown-cause share of resolved breaks: 25/25 = 100%.
Invariant violations: 0

## Why patterns stop working (family-wise controlled over every candidate driver)

No driver among 47 searched beat the family-wise noise baseline (108 breaks, 35 independent clusters): the honest answer is 'unknown cause'.

## Status at the last week

Counts: disregarded=26, steady=23
