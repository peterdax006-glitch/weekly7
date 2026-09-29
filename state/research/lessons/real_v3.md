# Lessons on real data (real_v3)

Summary: {"pairs": 2, "lessons": 9, "kept": 1, "mean_final_B": 0.0001155416170756022, "mean_holdout_C": 0.0, "pairs_beating_null": 0, "mean_memorisation_gap": 0.0, "seconds": 121}

# A 2017-01-01..2019-01-01 -> B 2019-01-01..2021-01-01

- improvement on A: +0.059%  |  on disguised A: +0.059%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.146%  |  degradation on B: -0.146% (full book, before the firewall)
- lessons: 3  accepted: 1  rejected: 2  |  identity-invariant: True
- after the firewall: A +0.157%, disguised A +0.157%, B +0.023%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| a9ec07882724 | kept | +0.157% | +0.157% | +0.023% |
| ab51eab7dde2 | no_gain_A | +0.000% | +0.000% | +0.000% |
| d0547c9bef9d | no_support_B | +0.073% | +0.073% | -0.175% |

## Per period on B (kept book)
```
         n  improvement      base      with
2019  52.0     0.000467  0.003775  0.004241
2020  53.0     0.000000  0.011160  0.011160
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high 20823         147         165            0.020990            0.018705             18
 low  6447         215         202            0.001347            0.001774             25
 mid 10630         163         158            0.003458            0.003896             21
```
- keyed on names? rename-only rerun +0.059%; on dates? shift-only rerun +0.059%
- never-used window C: +0.000% (90% CI +0.000% .. +0.000%), harmed: False
- optimism control (returns shuffled within day): null mean +0.279%, max +0.411%, real A +0.059%, p 1.000, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_

# A 2019-01-01..2021-01-01 -> B 2021-01-01..2023-01-01

- improvement on A: -0.259%  |  on disguised A: -0.259%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.208%  |  degradation on B: -0.208% (full book, before the firewall)
- lessons: 6  accepted: 0  rejected: 6  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 3ba2743411ba | no_gain_A | +0.000% | +0.000% | +0.000% |
| 0793f162d596 | no_gain_A | -0.043% | -0.043% | -0.044% |
| d3caadda6078 | harms_B | +0.054% | +0.054% | -0.111% |
| 485537d61841 | harms_B | +0.098% | +0.098% | -0.294% |
| 741422b4e9e8 | no_gain_A | -0.040% | -0.040% | -0.111% |
| 64fa8c75241b | no_gain_A | -0.357% | -0.357% | -0.113% |
- keyed on names? rename-only rerun -0.259%; on dates? shift-only rerun -0.259%
- never-used window C: +0.000% (90% CI +0.000% .. +0.000%), harmed: False
- optimism control (returns shuffled within day): null mean -0.139%, max -0.033%, real A -0.259%, p 1.000, exceeds null: False

_B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure_
