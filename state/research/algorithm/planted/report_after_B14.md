# Planted-pattern calibration (Bible Phase 25)

**Verdict: NOT VALIDATED** (64 runs, 120s)

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
| p_real_top_bin | 0.849 | P(real) in (0.9,1] truly real >= 0.85 | NO |

## heavy_tails (8 runs)

active/run 16.2, false active/run 2.12, FDR 13.1%, P(real) Brier 0.22299522820269016, ECE 0.17202553339517626

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.35 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.12 | 0.06 | 1.0 | {'rejected': 4, 'active': 4} |
| pair | pair | True | 1.00 | 0.00 | 0.29 | 1.0 | {'active': 8} |
| unless | unless | True | 0.12 | 0.88 | 0.31 | 1.0 | {'not_tested': 2, 'duplicate': 3, 'no_gain': 2, 'active': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.25 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.00 | 0.62 | 0.21 | 1.0 | {'no_gain': 7, 'rejected': 1} |
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 27% (n=1286)
- (0.2, 0.5]: mean P 0.35 -> truly real 34% (n=546)
- (0.5, 0.8]: mean P 0.65 -> truly real 47% (n=717)
- (0.8, 0.9]: mean P 0.85 -> truly real 49% (n=201)
- (0.9, 1.0]: mean P 0.99 -> truly real 85% (n=1562)

## noise_only (8 runs)

active/run 0.2, false active/run 0.25, FDR 100.0%, P(real) Brier 0.006288347219375874, ECE 0.021486679087098275

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 0% (n=4131)
- (0.2, 0.5]: mean P 0.30 -> truly real 0% (n=139)
- (0.5, 0.8]: mean P 0.59 -> truly real 0% (n=22)
- (0.9, 1.0]: mean P 0.94 -> truly real 0% (n=2)

## power_0.002 (8 runs)

active/run 0.6, false active/run 0.25, FDR 40.0%, P(real) Brier 0.09211292723993483, ECE 0.07975990225738888

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.25 | 0.00 | 0.28 | 1.0 | {'rejected': 6, 'active': 2} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 9% (n=3952)
- (0.2, 0.5]: mean P 0.34 -> truly real 21% (n=267)
- (0.5, 0.8]: mean P 0.60 -> truly real 36% (n=70)
- (0.8, 0.9]: mean P 0.85 -> truly real 100% (n=1)
- (0.9, 1.0]: mean P 0.94 -> truly real 71% (n=7)

## power_0.004 (8 runs)

active/run 2.2, false active/run 0.25, FDR 11.1%, P(real) Brier 0.07855845369302325, ECE 0.05260460465116278

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.88 | 0.00 | 0.26 | 1.0 | {'active': 7, 'rejected': 1} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 6% (n=3524)
- (0.2, 0.5]: mean P 0.35 -> truly real 48% (n=326)
- (0.5, 0.8]: mean P 0.66 -> truly real 76% (n=305)
- (0.8, 0.9]: mean P 0.85 -> truly real 93% (n=61)
- (0.9, 1.0]: mean P 0.96 -> truly real 99% (n=84)

## power_0.006 (8 runs)

active/run 5.4, false active/run 0.75, FDR 14.0%, P(real) Brier 0.032665229495348835, ECE 0.035525000000000015

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 1% (n=3147)
- (0.2, 0.5]: mean P 0.32 -> truly real 22% (n=286)
- (0.5, 0.8]: mean P 0.67 -> truly real 87% (n=256)
- (0.8, 0.9]: mean P 0.85 -> truly real 95% (n=187)
- (0.9, 1.0]: mean P 0.96 -> truly real 99% (n=424)

## power_0.009 (8 runs)

active/run 4.8, false active/run 0.62, FDR 13.2%, P(real) Brier 0.02658790876075331, ECE 0.05910606835619625

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 0% (n=2923)
- (0.2, 0.5]: mean P 0.31 -> truly real 1% (n=347)
- (0.5, 0.8]: mean P 0.65 -> truly real 23% (n=117)
- (0.8, 0.9]: mean P 0.85 -> truly real 85% (n=74)
- (0.9, 1.0]: mean P 0.99 -> truly real 98% (n=840)

## power_0.012 (8 runs)

active/run 1.6, false active/run 0.38, FDR 23.1%, P(real) Brier 0.044556360751162795, ECE 0.09686797674418604

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.04 -> truly real 0% (n=2628)
- (0.2, 0.5]: mean P 0.31 -> truly real 0% (n=534)
- (0.5, 0.8]: mean P 0.63 -> truly real 1% (n=155)
- (0.8, 0.9]: mean P 0.85 -> truly real 21% (n=28)
- (0.9, 1.0]: mean P 0.99 -> truly real 95% (n=955)

## standard (8 runs)

active/run 18.1, false active/run 3.38, FDR 18.6%, P(real) Brier 0.2215807202666976, ECE 0.17233096011131727

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.25 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 0.37 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.26 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.38 | 0.07 | 1.0 | {'rejected': 4, 'active': 4} |
| pair | pair | True | 1.00 | 0.00 | 0.28 | 1.0 | {'active': 8} |
| unless | unless | True | 0.25 | 0.75 | 0.3 | 1.0 | {'not_tested': 4, 'duplicate': 2, 'active': 2} |
| hallucinated | hallucinated | False | 0.00 | 0.62 | 0.11 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.25 | 0.50 | 0.22 | 1.0 | {'active': 2, 'no_gain': 6} |
| zero | zero | False | 0.00 | 0.25 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 27% (n=1308)
- (0.2, 0.5]: mean P 0.35 -> truly real 38% (n=554)
- (0.5, 0.8]: mean P 0.65 -> truly real 46% (n=702)
- (0.8, 0.9]: mean P 0.86 -> truly real 54% (n=214)
- (0.9, 1.0]: mean P 0.99 -> truly real 85% (n=1534)
