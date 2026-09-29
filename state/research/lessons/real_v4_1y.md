# Lessons on real data (real_v4_1y)

Summary: {"pairs": 7, "lessons": 33, "kept": 6, "mean_final_B": 0.0007563820907047819, "mean_holdout_C": 0.00020549470906729234, "pairs_beating_null": 2, "mean_memorisation_gap": 0.0, "seconds": 397}

# A 2017-01-01..2018-01-01 -> B 2018-01-01..2019-01-01

- improvement on A: +0.044%  |  on disguised A: +0.044%  |  memorisation gap: +0.000%
- improvement on unseen B: +0.103%  |  degradation on B: -0.000% (full book, before the firewall)
- lessons: 2  accepted: 1  rejected: 1  |  identity-invariant: True
- after the firewall: A +0.044%, disguised A +0.044%, B +0.103%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| c8e2d6484ba9 | kept | +0.044% | +0.044% | +0.103% |
| 09aa2fb9ae61 | no_gain_A | +0.000% | +0.000% | +0.000% |

## Per period on B (kept book)
```
         n  improvement      base      with
2018  52.0     0.001027  0.005051  0.006078
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high  9630          75          75            0.027444            0.025681             34
 low  2895          91          91            0.002832            0.003119             14
 mid  5294          94          94           -0.010669           -0.006698             38
```
- keyed on names? rename-only rerun +0.044%; on dates? shift-only rerun +0.044%
- never-used window C: -0.004% (90% CI -0.200% .. +0.165%), harmed: False
- optimism control (returns shuffled within day): null mean -0.035%, max +0.000%, real A +0.044%, p 0.167, exceeds null: True

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2018-01-01..2019-01-01 -> B 2019-01-01..2020-01-01

- improvement on A: +0.065%  |  on disguised A: +0.065%  |  memorisation gap: +0.000%
- improvement on unseen B: +0.090%  |  degradation on B: -0.000% (full book, before the firewall)
- lessons: 5  accepted: 0  rejected: 5  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 0be8331b6ab5 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 76d8f828e600 | no_gain_A | -0.159% | -0.159% | +0.061% |
| 3ce80c42bda4 | no_gain_A | -0.353% | -0.353% | +0.301% |
| b2dfbb9b5117 | no_gain_A | +0.000% | +0.000% | +0.000% |
| f7443f86f718 | no_gain_A | -0.020% | -0.020% | -0.006% |
- keyed on names? rename-only rerun +0.065%; on dates? shift-only rerun +0.065%
- never-used window C: +0.000% (90% CI +0.000% .. +0.000%), harmed: False
- optimism control (returns shuffled within day): null mean +0.090%, max +0.385%, real A +0.065%, p 0.667, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2019-01-01..2020-01-01 -> B 2020-01-01..2021-01-01

- improvement on A: +0.038%  |  on disguised A: +0.038%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.465%  |  degradation on B: -0.465% (full book, before the firewall)
- lessons: 2  accepted: 0  rejected: 2  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 5ec0434e2a1c | no_gain_A | +0.000% | +0.000% | +0.000% |
| db1fed1624ec | harms_B | +0.038% | +0.038% | -0.465% |
- keyed on names? rename-only rerun +0.038%; on dates? shift-only rerun +0.038%
- never-used window C: +0.000% (90% CI +0.000% .. +0.000%), harmed: False
- optimism control (returns shuffled within day): null mean -0.131%, max +0.076%, real A +0.038%, p 0.333, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2020-01-01..2021-01-01 -> B 2021-01-01..2022-01-01

- improvement on A: +0.484%  |  on disguised A: +0.484%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.672%  |  degradation on B: -0.672% (full book, before the firewall)
- lessons: 7  accepted: 1  rejected: 6  |  identity-invariant: True
- after the firewall: A +0.016%, disguised A +0.016%, B +0.115%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 5dc02981d789 | kept | +0.016% | +0.016% | +0.115% |
| 1545dddc1193 | no_gain_A | +0.000% | +0.000% | +0.000% |
| f284b93826ae | harms_B | -0.160% | -0.160% | -0.419% |
| 01883c3690fa | no_gain_A | -0.255% | -0.255% | +0.106% |
| f8b2c307f8c5 | harms_B | -0.315% | -0.315% | -0.403% |
| e8388dd4b357 | no_gain_A | -0.145% | -0.145% | -0.209% |
| 7c38658fcdb9 | no_gain_A | -0.494% | -0.494% | -0.085% |

## Per period on B (kept book)
```
         n  improvement      base      with
2021  52.0     0.001148  0.001306  0.002454
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high  2575          38          63            0.004833            0.007751             25
 low 11153         116           0           -0.002803                 NaN            116
 mid  6503         106         197            0.004537            0.000760            141
```
- keyed on names? rename-only rerun +0.484%; on dates? shift-only rerun +0.484%
- never-used window C: +0.054% (90% CI -0.407% .. +0.442%), harmed: False
- optimism control (returns shuffled within day): null mean -0.042%, max +0.272%, real A +0.484%, p 0.167, exceeds null: True

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2021-01-01..2022-01-01 -> B 2022-01-01..2023-01-01

- improvement on A: -0.226%  |  on disguised A: -0.226%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.215%  |  degradation on B: -0.215% (full book, before the firewall)
- lessons: 4  accepted: 0  rejected: 4  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| d17f88d6c460 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 637fcc8bfddb | no_gain_A | +0.000% | +0.000% | +0.000% |
| 398a8dd1d6ea | harms_B | -0.226% | -0.226% | -0.215% |
| 405922fe6a1c | no_gain_A | +0.000% | +0.000% | +0.000% |
- keyed on names? rename-only rerun -0.226%; on dates? shift-only rerun -0.226%
- never-used window C: +0.000% (90% CI +0.000% .. +0.000%), harmed: False
- optimism control (returns shuffled within day): null mean -0.065%, max +0.201%, real A -0.226%, p 0.833, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2022-01-01..2023-01-01 -> B 2023-01-01..2024-01-01

- improvement on A: -0.003%  |  on disguised A: -0.003%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.207%  |  degradation on B: -0.207% (full book, before the firewall)
- lessons: 7  accepted: 2  rejected: 5  |  identity-invariant: True
- after the firewall: A +0.399%, disguised A +0.399%, B +0.247%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 817af3144f95 | kept | +0.280% | +0.280% | +0.212% |
| 165af7ef9995 | kept | +0.520% | +0.520% | +0.006% |
| e5b2aee51c02 | harms_B | -0.008% | -0.008% | -0.227% |
| 7c38658fcdb9 | harms_B | -0.142% | -0.142% | -0.174% |
| 051af9d46747 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 189a6668711f | no_gain_A | +0.000% | +0.000% | +0.000% |
| 75c0f73fdcf2 | harms_B | -0.143% | -0.143% | -0.312% |

## Per period on B (kept book)
```
         n  improvement      base      with
2023  52.0     0.002472  0.002232  0.004704
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high  2920           9          11            0.021545            0.017774             12
 low 11842         206         192           -0.002155            0.001925            334
 mid  5377          45          57            0.018453            0.011543             88
```
- keyed on names? rename-only rerun -0.003%; on dates? shift-only rerun -0.003%
- never-used window C: +0.094% (90% CI -0.299% .. +0.640%), harmed: False
- optimism control (returns shuffled within day): null mean +0.060%, max +0.415%, real A -0.003%, p 0.667, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2023-01-01..2024-01-01 -> B 2024-01-01..2025-01-01

- improvement on A: +0.194%  |  on disguised A: +0.194%  |  memorisation gap: +0.000%
- improvement on unseen B: +0.133%  |  degradation on B: -0.000% (full book, before the firewall)
- lessons: 6  accepted: 2  rejected: 4  |  identity-invariant: True
- after the firewall: A +0.533%, disguised A +0.533%, B +0.065%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 7d5e76f33754 | kept | +0.048% | +0.048% | +0.136% |
| 262f2250d1f1 | kept | +0.533% | +0.533% | +0.065% |
| 5ba105752f19 | no_gain_A | -0.117% | -0.117% | +0.249% |
| 739545949d21 | no_gain_A | -0.058% | -0.058% | -0.092% |
| f7dec18c6cb5 | no_gain_A | -0.063% | -0.063% | -0.154% |
| 9c298abc7e4c | no_gain_A | -0.024% | -0.024% | +0.268% |

## Per period on B (kept book)
```
         n  improvement      base      with
2024  52.0     0.000648  0.001867  0.002515
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high  6134          17          49            0.001991           -0.008659             36
 low  7807         172          45            0.002900            0.002152            201
 mid  6166          71         166           -0.000665            0.005912            173
```
- keyed on names? rename-only rerun +0.194%; on dates? shift-only rerun +0.194%
- never-used window C: +0.000% (90% CI -0.467% .. +0.576%), harmed: False
- optimism control (returns shuffled within day): null mean +0.046%, max +0.289%, real A +0.194%, p 0.500, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_
