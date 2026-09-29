# Planted-pattern calibration (Bible Phase 25)

**Verdict: VALIDATED** (64 runs, 95s)

| criterion | value | rule | pass |
|---|---|---|---|
| strong_detection | 1.000 | detection_rate >= 0.9 | yes |
| negative_detection | 1.000 | detection_rate >= 0.8 | yes |
| negative_sign | 1.000 | sign_accuracy >= 0.95 | yes |
| hallucinated_rejected | 0.000 | false_admission_rate <= 0.2 | yes |
| decaying_not_held | 0.000 | false_admission_rate <= 0.2 | yes |
| zero_rejected | 0.000 | false_admission_rate <= 0.1 | yes |
| noise_only_quiet | 0.250 | active_per_run <= 1.0 | yes |
| fdr | 0.099 | false_discovery_rate <= 0.10 (exact truth) | yes |
| fdr_by_condition | 0.134 | reported only (legacy definition) | yes |
| p_real_top_bin | 0.963 | P(real) in (0.9,1] truly real >= 0.85 (exact truth) | yes |

## heavy_tails (8 runs)

active/run 17.4, false active/run 2.62, FDR 15.1%, P(real) Brier 0.22012508737244896, ECE 0.1758821196660482

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.77 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 1.07 | 1.0 | {'active': 7, 'rescoped': 1} |
| negative | negative | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 7, 'rescoped': 1} |
| regime | regime | True | 0.25 | 0.50 | 0.2 | 1.0 | {'rejected': 5, 'active': 2, 'no_gain': 1} |
| pair | pair | True | 1.00 | 0.00 | 0.86 | 1.0 | {'active': 8} |
| unless | unless | True | 0.12 | 0.88 | 0.92 | 1.0 | {'no_gain': 3, 'active': 1, 'not_tested': 2, 'duplicate': 2} |
| hallucinated | hallucinated | False | 0.00 | 0.50 | 0.38 | 1.0 | {'no_gain': 1, 'rejected': 6, 'discarded': 1} |
| decaying | decaying | False | 0.00 | 0.75 | 0.6 | 1.0 | {'rejected': 2, 'no_gain': 4, 'discarded': 2} |
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 29% (n=1264)
- (0.2, 0.5]: mean P 0.36 -> truly real 36% (n=555)
- (0.5, 0.8]: mean P 0.65 -> truly real 40% (n=724)
- (0.8, 0.9]: mean P 0.85 -> truly real 51% (n=232)
- (0.9, 1.0]: mean P 0.99 -> truly real 88% (n=1537)

## noise_only (8 runs)

active/run 0.2, false active/run 0.25, FDR 100.0%, P(real) Brier 0.010263593639551191, ECE 0.022588055165965407

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 0% (n=4139)
- (0.2, 0.5]: mean P 0.34 -> truly real 0% (n=67)
- (0.5, 0.8]: mean P 0.64 -> truly real 0% (n=63)
- (0.8, 0.9]: mean P 0.83 -> truly real 0% (n=8)
- (0.9, 1.0]: mean P 0.93 -> truly real 0% (n=1)

## power_0.002 (8 runs)

active/run 0.9, false active/run 0.25, FDR 28.6%, P(real) Brier 0.09819477878299462, ECE 0.08269850502219107

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.25 | 0.12 | 0.75 | 1.0 | {'rejected': 6, 'active': 2} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 9% (n=3962)
- (0.2, 0.5]: mean P 0.31 -> truly real 41% (n=160)
- (0.5, 0.8]: mean P 0.63 -> truly real 42% (n=102)
- (0.8, 0.9]: mean P 0.86 -> truly real 67% (n=33)
- (0.9, 1.0]: mean P 0.94 -> truly real 88% (n=24)

## power_0.004 (8 runs)

active/run 3.0, false active/run 0.25, FDR 8.3%, P(real) Brier 0.0799262000513299, ECE 0.06055991600559962

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.8 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 6% (n=3490)
- (0.2, 0.5]: mean P 0.31 -> truly real 49% (n=355)
- (0.5, 0.8]: mean P 0.64 -> truly real 78% (n=162)
- (0.8, 0.9]: mean P 0.85 -> truly real 98% (n=100)
- (0.9, 1.0]: mean P 0.97 -> truly real 98% (n=179)

## power_0.006 (8 runs)

active/run 4.2, false active/run 0.50, FDR 11.8%, P(real) Brier 0.029837972029356945, ECE 0.02847015377446411

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 1% (n=3113)
- (0.2, 0.5]: mean P 0.31 -> truly real 20% (n=315)
- (0.5, 0.8]: mean P 0.67 -> truly real 81% (n=192)
- (0.8, 0.9]: mean P 0.86 -> truly real 94% (n=134)
- (0.9, 1.0]: mean P 0.98 -> truly real 100% (n=538)

## power_0.009 (8 runs)

active/run 2.9, false active/run 0.62, FDR 21.7%, P(real) Brier 0.027366526206896554, ECE 0.0646903541472507

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 0% (n=2860)
- (0.2, 0.5]: mean P 0.31 -> truly real 0% (n=399)
- (0.5, 0.8]: mean P 0.64 -> truly real 20% (n=110)
- (0.8, 0.9]: mean P 0.86 -> truly real 85% (n=54)
- (0.9, 1.0]: mean P 0.99 -> truly real 97% (n=869)

## power_0.012 (8 runs)

active/run 2.2, false active/run 0.62, FDR 27.8%, P(real) Brier 0.048304829517819704, ECE 0.10537894246447706

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.04 -> truly real 0% (n=2534)
- (0.2, 0.5]: mean P 0.32 -> truly real 0% (n=608)
- (0.5, 0.8]: mean P 0.63 -> truly real 1% (n=154)
- (0.8, 0.9]: mean P 0.85 -> truly real 5% (n=37)
- (0.9, 1.0]: mean P 1.00 -> truly real 95% (n=960)

## standard (8 runs)

active/run 17.8, false active/run 2.38, FDR 13.4%, P(real) Brier 0.21693901375231914, ECE 0.164543599257885

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 1.18 | 1.0 | {'active': 7, 'rescoped': 1} |
| negative | negative | True | 1.00 | 0.00 | 0.8 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.12 | 0.22 | 0.875 | {'active': 4, 'rejected': 3, 'no_gain': 1} |
| pair | pair | True | 1.00 | 0.00 | 0.95 | 1.0 | {'active': 8} |
| unless | unless | True | 0.50 | 0.50 | 0.94 | 1.0 | {'active': 4, 'duplicate': 1, 'not_tested': 3} |
| hallucinated | hallucinated | False | 0.00 | 0.50 | 0.38 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.00 | 0.88 | 0.67 | 1.0 | {'discarded': 7, 'no_gain': 1} |
| zero | zero | False | 0.00 | 0.12 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 29% (n=1328)
- (0.2, 0.5]: mean P 0.36 -> truly real 36% (n=566)
- (0.5, 0.8]: mean P 0.65 -> truly real 47% (n=655)
- (0.8, 0.9]: mean P 0.85 -> truly real 57% (n=235)
- (0.9, 1.0]: mean P 0.99 -> truly real 87% (n=1528)
