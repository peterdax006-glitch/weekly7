# Planted-pattern calibration (Bible Phase 25)

**Verdict: NOT VALIDATED** (64 runs, 112s)

| criterion | value | rule | pass |
|---|---|---|---|
| strong_detection | 1.000 | detection_rate >= 0.9 | yes |
| negative_detection | 1.000 | detection_rate >= 0.8 | yes |
| negative_sign | 1.000 | sign_accuracy >= 0.95 | yes |
| hallucinated_rejected | 0.000 | false_admission_rate <= 0.2 | yes |
| decaying_not_held | 0.250 | false_admission_rate <= 0.2 | NO |
| zero_rejected | 0.000 | false_admission_rate <= 0.1 | yes |
| noise_only_quiet | 0.375 | active_per_run <= 1.0 | yes |
| fdr | 0.159 | false_discovery_rate <= 0.10 | NO |
| p_real_top_bin | 0.856 | P(real) in (0.9,1] truly real >= 0.85 | yes |

## heavy_tails (8 runs)

active/run 14.9, false active/run 1.75, FDR 11.8%, P(real) Brier 0.32360029006712965, ECE 0.31929969907407413

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.35 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| regime | regime | True | 0.25 | 0.25 | 0.06 | 1.0 | {'rejected': 6, 'active': 2} |
| pair | pair | True | 0.88 | 0.00 | 0.3 | 1.0 | {'active': 7, 'not_tested': 1} |
| unless | unless | True | 0.12 | 0.50 | 0.33 | 1.0 | {'not_tested': 4, 'duplicate': 2, 'active': 1, 'no_gain': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.38 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.00 | 0.75 | 0.21 | 1.0 | {'no_gain': 7, 'rejected': 1} |
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 50% (n=1901)
- (0.2, 0.5]: mean P 0.38 -> truly real 68% (n=97)
- (0.5, 0.8]: mean P 0.66 -> truly real 38% (n=377)
- (0.8, 0.9]: mean P 0.85 -> truly real 44% (n=198)
- (0.9, 1.0]: mean P 0.99 -> truly real 87% (n=1747)

## noise_only (8 runs)

active/run 0.4, false active/run 0.38, FDR 100.0%, P(real) Brier 0.004234469664257402, ECE 0.007598064816973654

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| zero | zero | False | 0.00 | 0.12 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 0% (n=4225)
- (0.2, 0.5]: mean P 0.35 -> truly real 0% (n=36)
- (0.5, 0.8]: mean P 0.64 -> truly real 0% (n=23)
- (0.8, 0.9]: mean P 0.85 -> truly real 0% (n=4)
- (0.9, 1.0]: mean P 0.98 -> truly real 0% (n=1)

## power_0.002 (8 runs)

active/run 1.2, false active/run 0.50, FDR 40.0%, P(real) Brier 0.14361783068933395, ECE 0.13863758733115975

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.38 | 0.12 | 0.28 | 1.0 | {'rejected': 5, 'active': 3} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 14% (n=4171)
- (0.2, 0.5]: mean P 0.35 -> truly real 43% (n=60)
- (0.5, 0.8]: mean P 0.63 -> truly real 53% (n=47)
- (0.8, 0.9]: mean P 0.86 -> truly real 50% (n=14)
- (0.9, 1.0]: mean P 0.97 -> truly real 50% (n=2)

## power_0.004 (8 runs)

active/run 2.1, false active/run 0.12, FDR 5.9%, P(real) Brier 0.1466976589046512, ECE 0.15661313953488373

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 15% (n=3850)
- (0.2, 0.5]: mean P 0.38 -> truly real 86% (n=87)
- (0.5, 0.8]: mean P 0.66 -> truly real 93% (n=207)
- (0.8, 0.9]: mean P 0.85 -> truly real 94% (n=80)
- (0.9, 1.0]: mean P 0.95 -> truly real 99% (n=76)

## power_0.006 (8 runs)

active/run 3.6, false active/run 0.12, FDR 3.4%, P(real) Brier 0.06557612026499303, ECE 0.08262319851231982

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 7% (n=3428)
- (0.2, 0.5]: mean P 0.37 -> truly real 93% (n=54)
- (0.5, 0.8]: mean P 0.68 -> truly real 96% (n=217)
- (0.8, 0.9]: mean P 0.85 -> truly real 98% (n=163)
- (0.9, 1.0]: mean P 0.96 -> truly real 100% (n=440)

## power_0.009 (8 runs)

active/run 6.0, false active/run 0.25, FDR 4.2%, P(real) Brier 0.014314984593967517, ECE 0.009000417633410678

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 1% (n=3191)
- (0.2, 0.5]: mean P 0.35 -> truly real 17% (n=29)
- (0.5, 0.8]: mean P 0.68 -> truly real 66% (n=53)
- (0.8, 0.9]: mean P 0.86 -> truly real 91% (n=67)
- (0.9, 1.0]: mean P 0.99 -> truly real 99% (n=970)

## power_0.012 (8 runs)

active/run 1.9, false active/run 0.38, FDR 20.0%, P(real) Brier 0.016857727828306263, ECE 0.01820821345707658

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 0% (n=3104)
- (0.2, 0.5]: mean P 0.38 -> truly real 11% (n=28)
- (0.5, 0.8]: mean P 0.65 -> truly real 10% (n=42)
- (0.8, 0.9]: mean P 0.86 -> truly real 14% (n=28)
- (0.9, 1.0]: mean P 1.00 -> truly real 97% (n=1108)

## standard (8 runs)

active/run 17.2, false active/run 2.75, FDR 15.9%, P(real) Brier 0.3259414202314815, ECE 0.3156850925925926

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.37 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.12 | 0.07 | 1.0 | {'rejected': 4, 'active': 4} |
| pair | pair | True | 1.00 | 0.00 | 0.28 | 1.0 | {'active': 8} |
| unless | unless | True | 0.12 | 0.25 | 0.32 | 1.0 | {'not_tested': 6, 'active': 1, 'duplicate': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.50 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.25 | 0.75 | 0.22 | 1.0 | {'active': 2, 'no_gain': 6} |
| zero | zero | False | 0.00 | 0.25 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.00 -> truly real 51% (n=1919)
- (0.2, 0.5]: mean P 0.36 -> truly real 62% (n=119)
- (0.5, 0.8]: mean P 0.67 -> truly real 44% (n=322)
- (0.8, 0.9]: mean P 0.85 -> truly real 56% (n=196)
- (0.9, 1.0]: mean P 0.99 -> truly real 86% (n=1764)
