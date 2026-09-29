# Planted-pattern calibration (Bible Phase 25)

**Verdict: NOT VALIDATED** (64 runs, 98s)

| criterion | value | rule | pass |
|---|---|---|---|
| strong_detection | 1.000 | detection_rate >= 0.9 | yes |
| negative_detection | 1.000 | detection_rate >= 0.8 | yes |
| negative_sign | 1.000 | sign_accuracy >= 0.95 | yes |
| hallucinated_rejected | 0.000 | false_admission_rate <= 0.2 | yes |
| decaying_not_held | 0.250 | false_admission_rate <= 0.2 | NO |
| zero_rejected | 0.000 | false_admission_rate <= 0.1 | yes |
| noise_only_quiet | 0.250 | active_per_run <= 1.0 | yes |
| fdr | 0.186 | false_discovery_rate <= 0.10 | NO |
| p_real_top_bin | 0.840 | P(real) in (0.9,1] truly real >= 0.85 | NO |

## heavy_tails (8 runs)

active/run 14.4, false active/run 2.62, FDR 18.3%, P(real) Brier 0.19966875286178107, ECE 0.1765372448979591

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.35 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.12 | 0.06 | 1.0 | {'rejected': 4, 'active': 4} |
| pair | pair | True | 0.38 | 0.00 | 0.31 | 1.0 | {'not_tested': 5, 'active': 3} |
| unless | unless | True | 0.00 | 0.25 | 0.33 | 1.0 | {'not_tested': 6, 'duplicate': 1, 'no_gain': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.62 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.00 | 0.50 | 0.21 | 1.0 | {'no_gain': 7, 'rejected': 1} |
| zero | zero | False | 0.00 | 0.12 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 19% (n=1033)
- (0.2, 0.5]: mean P 0.36 -> truly real 25% (n=362)
- (0.5, 0.8]: mean P 0.65 -> truly real 41% (n=731)
- (0.8, 0.9]: mean P 0.85 -> truly real 51% (n=299)
- (0.9, 1.0]: mean P 0.99 -> truly real 85% (n=1887)

## noise_only (8 runs)

active/run 0.2, false active/run 0.25, FDR 100.0%, P(real) Brier 0.005610268736891168, ECE 0.01921326031228152

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 0% (n=4179)
- (0.2, 0.5]: mean P 0.31 -> truly real 0% (n=82)
- (0.5, 0.8]: mean P 0.59 -> truly real 0% (n=27)
- (0.8, 0.9]: mean P 0.84 -> truly real 0% (n=2)
- (0.9, 1.0]: mean P 0.98 -> truly real 0% (n=1)

## power_0.002 (8 runs)

active/run 0.5, false active/run 0.12, FDR 25.0%, P(real) Brier 0.1337146849557522, ECE 0.11344923148579414

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.25 | 0.00 | 0.28 | 1.0 | {'rejected': 6, 'active': 2} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 13% (n=4011)
- (0.2, 0.5]: mean P 0.31 -> truly real 44% (n=217)
- (0.5, 0.8]: mean P 0.59 -> truly real 51% (n=57)
- (0.8, 0.9]: mean P 0.83 -> truly real 50% (n=4)
- (0.9, 1.0]: mean P 0.96 -> truly real 80% (n=5)

## power_0.004 (8 runs)

active/run 2.6, false active/run 0.25, FDR 9.5%, P(real) Brier 0.09053077616511628, ECE 0.06640039534883721

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.88 | 0.00 | 0.26 | 1.0 | {'active': 7, 'rejected': 1} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 7% (n=3275)
- (0.2, 0.5]: mean P 0.34 -> truly real 47% (n=423)
- (0.5, 0.8]: mean P 0.65 -> truly real 84% (n=384)
- (0.8, 0.9]: mean P 0.85 -> truly real 98% (n=110)
- (0.9, 1.0]: mean P 0.95 -> truly real 100% (n=108)

## power_0.006 (8 runs)

active/run 7.1, false active/run 0.62, FDR 8.8%, P(real) Brier 0.0414980770204366, ECE 0.04549170924291685

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 1% (n=2774)
- (0.2, 0.5]: mean P 0.31 -> truly real 17% (n=468)
- (0.5, 0.8]: mean P 0.67 -> truly real 77% (n=308)
- (0.8, 0.9]: mean P 0.86 -> truly real 94% (n=217)
- (0.9, 1.0]: mean P 0.97 -> truly real 99% (n=539)

## power_0.009 (8 runs)

active/run 7.4, false active/run 1.50, FDR 20.3%, P(real) Brier 0.04584797563341067, ECE 0.09874997679814387

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.04 -> truly real 0% (n=2330)
- (0.2, 0.5]: mean P 0.32 -> truly real 0% (n=677)
- (0.5, 0.8]: mean P 0.63 -> truly real 14% (n=187)
- (0.8, 0.9]: mean P 0.86 -> truly real 63% (n=90)
- (0.9, 1.0]: mean P 0.99 -> truly real 98% (n=1026)

## power_0.012 (8 runs)

active/run 2.6, false active/run 1.12, FDR 42.9%, P(real) Brier 0.08370669387845048, ECE 0.16213205752725585

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.04 -> truly real 0% (n=1887)
- (0.2, 0.5]: mean P 0.34 -> truly real 0% (n=862)
- (0.5, 0.8]: mean P 0.62 -> truly real 1% (n=344)
- (0.8, 0.9]: mean P 0.84 -> truly real 4% (n=56)
- (0.9, 1.0]: mean P 0.99 -> truly real 93% (n=1162)

## standard (8 runs)

active/run 18.1, false active/run 3.38, FDR 18.6%, P(real) Brier 0.19720244790120592, ECE 0.16172302875695738

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.37 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| regime | regime | True | 0.62 | 0.12 | 0.07 | 1.0 | {'rejected': 3, 'active': 5} |
| pair | pair | True | 0.38 | 0.00 | 0.24 | 1.0 | {'active': 3, 'not_tested': 5} |
| unless | unless | True | 0.00 | 0.12 | 0.33 | 1.0 | {'not_tested': 7, 'duplicate': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.50 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.25 | 0.62 | 0.22 | 1.0 | {'active': 2, 'no_gain': 6} |
| zero | zero | False | 0.00 | 0.25 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 19% (n=1071)
- (0.2, 0.5]: mean P 0.36 -> truly real 29% (n=356)
- (0.5, 0.8]: mean P 0.65 -> truly real 48% (n=697)
- (0.8, 0.9]: mean P 0.85 -> truly real 54% (n=267)
- (0.9, 1.0]: mean P 0.99 -> truly real 84% (n=1921)
