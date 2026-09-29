# Find volatility pipeline (V1-V5)

weeks traded: 716   origins: 56   picks/week: 10   gate: 0.8

## V1 movers (touch +-10% inside the week)
- hit rate 57.9% (90% CI 55.8%-60.2%) vs base rate of all tradable stocks 14.6%; predicted 52.5%
- 95% target NOT met; weeks with >=10 candidates above tau95: 0.1% (mean 0.0 qualified)

## Ablation (same picks, paired CIs)

variant  weeks  mean_week  mean_week_lo  mean_week_hi  share10_gross  share10_gross_lo  coverage  dir_acc  cat_rate  worst_position  goal8_weeks
      M    716     0.0034       -0.0005        0.0072         0.1503            0.1395       1.0   0.4751    0.0317         -0.7066       0.0028
    M+D    716     0.0000        0.0000        0.0000         0.0000            0.0000       0.0   0.0000    0.0000          0.0000       0.0000
  M+D+E    716     0.0000        0.0000        0.0000         0.0000            0.0000       0.0   0.0000    0.0000          0.0000       0.0000
  M+D+S    716     0.0000        0.0000        0.0000         0.0000            0.0000       0.0   0.0000    0.0000          0.0000       0.0000
M+D+E+S    716     0.0000        0.0000        0.0000         0.0000            0.0000       0.0   0.0000    0.0000          0.0000       0.0000

 from      to  d_mean_week  d_mean_week_lo  d_mean_week_hi  d_share10  d_share10_lo  d_share10_hi             verdict
    M     M+D      -0.0034         -0.0073          0.0002    -0.1503       -0.1619       -0.1397 not distinguishable
  M+D   M+D+E       0.0000          0.0000          0.0000     0.0000        0.0000        0.0000 not distinguishable
M+D+E   M+D+S       0.0000          0.0000          0.0000     0.0000        0.0000        0.0000 not distinguishable
M+D+S M+D+E+S       0.0000          0.0000          0.0000     0.0000        0.0000        0.0000 not distinguishable

## Per era (full pipeline)

 era  weeks  mean_week  pos_weeks  in_band  mover_hit  coverage  dir_acc  share10_gross  goal8_weeks  worst_position  cat  bets
2013     52        0.0        0.0      0.0      0.254       0.0      0.0            0.0          0.0             0.0    0     0
2014     52        0.0        0.0      0.0      0.419       0.0      0.0            0.0          0.0             0.0    0     0
2015     53        0.0        0.0      0.0      0.472       0.0      0.0            0.0          0.0             0.0    0     0
2016     52        0.0        0.0      0.0      0.519       0.0      0.0            0.0          0.0             0.0    0     0
2017     52        0.0        0.0      0.0      0.421       0.0      0.0            0.0          0.0             0.0    0     0
2018     52        0.0        0.0      0.0      0.538       0.0      0.0            0.0          0.0             0.0    0     0
2019     52        0.0        0.0      0.0      0.550       0.0      0.0            0.0          0.0             0.0    0     0
2020     53        0.0        0.0      0.0      0.777       0.0      0.0            0.0          0.0             0.0    0     0
2021     52        0.0        0.0      0.0      0.675       0.0      0.0            0.0          0.0             0.0    0     0
2022     52        0.0        0.0      0.0      0.760       0.0      0.0            0.0          0.0             0.0    0     0
2023     52        0.0        0.0      0.0      0.619       0.0      0.0            0.0          0.0             0.0    0     0
2024     52        0.0        0.0      0.0      0.706       0.0      0.0            0.0          0.0             0.0    0     0
2025     52        0.0        0.0      0.0      0.704       0.0      0.0            0.0          0.0             0.0    0     0
2026     38        0.0        0.0      0.0      0.732       0.0      0.0            0.0          0.0             0.0    0     0

## V5 verdict

- share10 lower bound 0.00 < 0.80

No stop guarantees a floor: gaps fill through stops. Read cat_rate and cat_wilson_hi, not a promise.