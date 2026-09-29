# Lessons on real data (real_v2)

Summary: {"pairs": 3, "lessons": 14, "kept": 1, "mean_final_B": 7.051057540453396e-05, "mean_memorisation_gap": 0.0, "seconds": 86}

# A 2017-01-01..2019-01-01 -> B 2019-01-01..2021-01-01

- improvement on A: +0.026%  |  on disguised A: +0.026%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.296%  |  degradation on B: -0.296% (full book, before the firewall)
- lessons: 3  accepted: 0  rejected: 3  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| ab51eab7dde2 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 62fd7b3ad94d | no_gain_A | -0.039% | -0.039% | -0.032% |
| d0547c9bef9d | no_support_B | +0.073% | +0.073% | -0.175% |

_B is used to reject lessons; use a further unseen window for an unbiased final figure_

# A 2019-01-01..2021-01-01 -> B 2021-01-01..2023-01-01

- improvement on A: -0.248%  |  on disguised A: -0.248%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.207%  |  degradation on B: -0.207% (full book, before the firewall)
- lessons: 6  accepted: 1  rejected: 5  |  identity-invariant: True
- after the firewall: A +0.012%, disguised A +0.012%, B +0.021%  ->  kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| dee1c3f0f02c | kept | +0.012% | +0.012% | +0.021% |
| a50cbd192b0d | no_gain_A | +0.000% | +0.000% | +0.000% |
| fce97c900c1b | harms_B | +0.060% | +0.060% | -0.088% |
| 514cc5ca1375 | no_gain_A | -0.150% | -0.150% | +0.136% |
| 932c4978f07f | harms_B | -0.379% | -0.379% | -0.168% |
| 13d44002bbd7 | no_gain_A | -0.150% | -0.150% | -0.014% |

## Per period on B (kept book)
```
         n  improvement      base      with
2021  52.0     0.000114  0.001306  0.001420
2022  52.0     0.000309 -0.002337 -0.002028
```

## Per stock type on B (kept book)
```
type  rows  picks_base  picks_with  mean_pick_pnl_base  mean_pick_pnl_with  picks_changed
high 12941         181         189            0.007499            0.009869              8
 low 10705         122         115            0.002510            0.001185             19
 mid 16850         217         216           -0.008902           -0.009998              7
```

_B is used to reject lessons; use a further unseen window for an unbiased final figure_

# A 2021-01-01..2023-01-01 -> B 2023-01-01..2025-01-01

- improvement on A: +0.053%  |  on disguised A: +0.053%  |  memorisation gap: +0.000%
- improvement on unseen B: -0.251%  |  degradation on B: -0.251% (full book, before the firewall)
- lessons: 5  accepted: 0  rejected: 5  |  identity-invariant: True
- after the firewall: A +0.000%, disguised A +0.000%, B +0.000%  ->  none_kept

| lesson | verdict | A | A disguised | B |
|---|---|---|---|---|
| 6cd41a720763 | harms_B | +0.123% | +0.123% | -0.179% |
| 22255885fad7 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 54e016a5dfa3 | no_gain_A | -0.072% | -0.072% | -0.008% |
| 089f3bc7edc1 | no_gain_A | +0.000% | +0.000% | +0.000% |
| 7cab1c0a1625 | no_gain_A | +0.000% | +0.000% | +0.000% |

_B is used to reject lessons; use a further unseen window for an unbiased final figure_
