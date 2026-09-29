# Planted-pattern calibration (Bible Phase 25)

**Verdict: NOT VALIDATED** (64 runs, 111s)

| criterion | value | rule | pass |
|---|---|---|---|
| strong_detection | 1.000 | detection_rate >= 0.9 | yes |
| negative_detection | 1.000 | detection_rate >= 0.8 | yes |
| negative_sign | 1.000 | sign_accuracy >= 0.95 | yes |
| hallucinated_rejected | 0.000 | false_admission_rate <= 0.2 | yes |
| decaying_not_held | 0.250 | false_admission_rate <= 0.2 | NO |
| zero_rejected | 0.000 | false_admission_rate <= 0.1 | yes |
| noise_only_quiet | 0.250 | active_per_run <= 1.0 | yes |
| fdr | 0.182 | false_discovery_rate <= 0.10 | NO |
| p_real_top_bin | 0.851 | P(real) in (0.9,1] truly real >= 0.85 | yes |

## heavy_tails (8 runs)

active/run 16.2, false active/run 2.12, FDR 13.1%, P(real) Brier 0.22245849747912802, ECE 0.17211876159554731

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.81 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 1.12 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.77 | 1.0 | {'active': 8} |
| regime | regime | True | 0.50 | 0.12 | 0.18 | 1.0 | {'rejected': 4, 'active': 4} |
| pair | pair | True | 1.00 | 0.00 | 0.94 | 1.0 | {'active': 8} |
| unless | unless | True | 0.12 | 0.88 | 0.93 | 1.0 | {'not_tested': 2, 'duplicate': 3, 'no_gain': 2, 'active': 1} |
| hallucinated | hallucinated | False | 0.00 | 0.25 | 0.36 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.00 | 0.62 | 0.65 | 1.0 | {'no_gain': 7, 'rejected': 1} |
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 27% (n=1268)
- (0.2, 0.5]: mean P 0.35 -> truly real 34% (n=549)
- (0.5, 0.8]: mean P 0.65 -> truly real 46% (n=731)
- (0.8, 0.9]: mean P 0.85 -> truly real 53% (n=227)
- (0.9, 1.0]: mean P 0.99 -> truly real 85% (n=1537)

## noise_only (8 runs)

active/run 0.2, false active/run 0.25, FDR 100.0%, P(real) Brier 0.009281667615277132, ECE 0.022359944108057753

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| zero | zero | False | 0.00 | 0.00 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 0% (n=4136)
- (0.2, 0.5]: mean P 0.33 -> truly real 0% (n=98)
- (0.5, 0.8]: mean P 0.61 -> truly real 0% (n=50)
- (0.8, 0.9]: mean P 0.84 -> truly real 0% (n=6)
- (0.9, 1.0]: mean P 0.93 -> truly real 0% (n=4)

## power_0.002 (8 runs)

active/run 0.6, false active/run 0.38, FDR 60.0%, P(real) Brier 0.09924186529671863, ECE 0.0902909937165464

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.12 | 0.00 | 0.75 | 1.0 | {'rejected': 7, 'active': 1} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.01 -> truly real 9% (n=4041)
- (0.2, 0.5]: mean P 0.34 -> truly real 20% (n=146)
- (0.5, 0.8]: mean P 0.63 -> truly real 23% (n=95)
- (0.8, 0.9]: mean P 0.84 -> truly real 0% (n=7)
- (0.9, 1.0]: mean P 0.93 -> truly real 38% (n=8)

## power_0.004 (8 runs)

active/run 2.0, false active/run 0.25, FDR 12.5%, P(real) Brier 0.08458855603023255, ECE 0.05602048837209303

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 0.75 | 0.00 | 0.77 | 1.0 | {'rejected': 2, 'active': 6} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.02 -> truly real 6% (n=3557)
- (0.2, 0.5]: mean P 0.34 -> truly real 47% (n=302)
- (0.5, 0.8]: mean P 0.65 -> truly real 82% (n=297)
- (0.8, 0.9]: mean P 0.84 -> truly real 86% (n=85)
- (0.9, 1.0]: mean P 0.96 -> truly real 90% (n=59)

## power_0.006 (8 runs)

active/run 4.1, false active/run 0.25, FDR 6.1%, P(real) Brier 0.03557410252325581, ECE 0.03560430232558137

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.78 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 1% (n=3164)
- (0.2, 0.5]: mean P 0.32 -> truly real 25% (n=279)
- (0.5, 0.8]: mean P 0.67 -> truly real 85% (n=282)
- (0.8, 0.9]: mean P 0.85 -> truly real 97% (n=176)
- (0.9, 1.0]: mean P 0.96 -> truly real 99% (n=399)

## power_0.009 (8 runs)

active/run 5.2, false active/run 0.88, FDR 16.7%, P(real) Brier 0.02704368250174378, ECE 0.0604825854452453

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.79 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 0% (n=2895)
- (0.2, 0.5]: mean P 0.30 -> truly real 2% (n=387)
- (0.5, 0.8]: mean P 0.63 -> truly real 24% (n=102)
- (0.8, 0.9]: mean P 0.86 -> truly real 82% (n=83)
- (0.9, 1.0]: mean P 0.98 -> truly real 98% (n=834)

## power_0.012 (8 runs)

active/run 2.0, false active/run 0.75, FDR 37.5%, P(real) Brier 0.04391399953488373, ECE 0.09777032558139535

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.79 | 1.0 | {'active': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.04 -> truly real 0% (n=2606)
- (0.2, 0.5]: mean P 0.32 -> truly real 0% (n=582)
- (0.5, 0.8]: mean P 0.62 -> truly real 2% (n=128)
- (0.8, 0.9]: mean P 0.86 -> truly real 19% (n=36)
- (0.9, 1.0]: mean P 0.99 -> truly real 95% (n=948)

## standard (8 runs)

active/run 17.9, false active/run 3.25, FDR 18.2%, P(real) Brier 0.22133507745825604, ECE 0.16978320964749538

| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |
|---|---|---|---|---|---|---|---|
| strong | strong | True | 1.00 | 0.00 | 0.8 | 1.0 | {'active': 8} |
| weak | weak | True | 1.00 | 0.00 | 1.17 | 1.0 | {'active': 8} |
| negative | negative | True | 1.00 | 0.00 | 0.83 | 1.0 | {'active': 8} |
| regime | regime | True | 0.62 | 0.25 | 0.21 | 1.0 | {'rejected': 3, 'active': 5} |
| pair | pair | True | 1.00 | 0.00 | 0.95 | 1.0 | {'active': 8} |
| unless | unless | True | 0.25 | 0.75 | 0.92 | 1.0 | {'not_tested': 4, 'duplicate': 2, 'active': 2} |
| hallucinated | hallucinated | False | 0.00 | 0.50 | 0.35 | 1.0 | {'rejected': 7, 'no_gain': 1} |
| decaying | decaying | False | 0.25 | 0.50 | 0.65 | 1.0 | {'active': 2, 'no_gain': 6} |
| zero | zero | False | 0.00 | 0.25 | None | None | {'rejected': 8} |

P(real) reliability:

- (-0.01, 0.2]: mean P 0.03 -> truly real 27% (n=1302)
- (0.2, 0.5]: mean P 0.35 -> truly real 38% (n=554)
- (0.5, 0.8]: mean P 0.64 -> truly real 47% (n=718)
- (0.8, 0.9]: mean P 0.85 -> truly real 53% (n=223)
- (0.9, 1.0]: mean P 0.99 -> truly real 85% (n=1515)
