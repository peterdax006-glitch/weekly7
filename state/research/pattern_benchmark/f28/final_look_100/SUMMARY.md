# F28 benchmark round 2 - IMPLEMENTED, NOT VALIDATED

Protocol: F19 final-look (one screen + gate at each world's last date), every F26 + F28 fix on, seeds W00000..W00099 (72 development + 28 held-out; held-out split fixed by the salted hash, never used for tuning; F28 tuning used development seeds 1, 9, 17, 19 and 2 only). Refused by the seal check: 0. Code hash at launch: ['939fadd26f6447b9']. Answer key, scoring, world generator and detectability floor unchanged.

- **development**: worlds 72; detectable real found 323/1265 = 25.5% [23.1%, 27.7%]; all real found 323/3350; promotions 334; precision 96.7%; FP 11 = 0.15/world [0.07, 0.25] (worlds with 0 FP: 61/72); FP kinds {'proxy': np.int64(10), 'leak': np.int64(1)}; null worlds 5, promoting 1
- **heldout**: worlds 28; detectable real found 142/543 = 26.2% [23.1%, 29.7%]; all real found 142/1350; promotions 146; precision 97.3%; FP 4 = 0.14/world [0.00, 0.32] (worlds with 0 FP: 25/28); FP kinds {'proxy': np.int64(4)}; null worlds 1, promoting 0
- **all**: worlds 100; detectable real found 465/1808 = 25.7% [23.8%, 27.6%]; all real found 465/4700; promotions 480; precision 96.9%; FP 15 = 0.15/world [0.08, 0.23] (worlds with 0 FP: 86/100); FP kinds {'proxy': np.int64(14), 'leak': np.int64(1)}; null worlds 6, promoting 1

## Same seeds as F26's final stage (S4), before -> after F28

- development: S4 worlds 14; detectable real found 71/246 = 28.9% [23.5%, 33.5%]; all real found 71/650; promotions 84; precision 84.5%; FP 13 = 0.93/world [0.29, 1.71] (worlds with 0 FP: 9/14); FP kinds {'proxy': np.int64(9), 'leak': np.int64(3), 'identity_null': np.int64(1)}; null worlds 1, promoting 0
- development: F28 worlds 14; detectable real found 71/246 = 28.9% [23.5%, 33.5%]; all real found 71/650; promotions 72; precision 98.6%; FP 1 = 0.07/world [0.00, 0.21] (worlds with 0 FP: 13/14); FP kinds {'proxy': np.int64(1)}; null worlds 1, promoting 0
- heldout: S4 worlds 6; detectable real found 25/100 = 25.0% [17.1%, 31.6%]; all real found 25/250; promotions 33; precision 75.8%; FP 8 = 1.33/world [0.33, 2.33] (worlds with 0 FP: 2/6); FP kinds {'proxy': np.int64(7), 'identity_null': np.int64(1)}; null worlds 1, promoting 0
- heldout: F28 worlds 6; detectable real found 25/100 = 25.0% [17.1%, 31.6%]; all real found 25/250; promotions 27; precision 92.6%; FP 2 = 0.33/world [0.00, 1.00] (worlds with 0 FP: 5/6); FP kinds {'proxy': np.int64(2)}; null worlds 1, promoting 0

# F19 real-vs-noise benchmark (C70-C74) - IMPLEMENTED, NOT VALIDATED

Worlds scored: 100 (72 development, 28 held-out). Cost: 198 s per world system time (median 157).

## C72 score: right / wrong against the sealed answer key

- Real patterns found (detectable): 465 of 1808; all real planted: 4700 (1673 undetectable in principle, 1219 detectable only in their true form).
- Noise correctly rejected: 54625 of 54640; false positives: 15 in total, 0.15 per world (target 0; the gate's stated alpha allows 0.000 per world on the pure nulls it gated, 0.0002 if the search size were every feature scored).

## Totals (world-cluster bootstrap 95% intervals)

| set | worlds | real right (detectable promoted, right sign) | precision (right real / all promotions) | FDR | FNR (detectable) | real right / all real | undetectable in principle | not representable | noise correctly rejected | false positives / world | expected FP / world at stated alpha | expected FP / world at honest alpha | candidate recall real | candidate recall noise | partial credit (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 100 | 25.7% [23.8%, 27.6%] (n=1808) | 96.9% (465/480) | 3.1% | 74.3% | 465/4700 | 1673 | 1219 | 100.0% [100.0%, 100.0%] (n=54640) | 0.15 (max 2; worlds with 0: 86/100) | 0.000 | 0.0002 | 32.9% [31.6%, 34.3%] (n=4700) | 6.8% [6.4%, 7.2%] (n=54640) | 0.117 |
| development | 72 | 25.5% [23.1%, 27.7%] (n=1265) | 96.7% (323/334) | 3.3% | 74.5% | 323/3350 | 1207 | 878 | 100.0% [100.0%, 100.0%] (n=39350) | 0.15 (max 1; worlds with 0: 61/72) | 0.000 | 0.0002 | 32.5% [31.1%, 34.0%] (n=3350) | 6.9% [6.4%, 7.4%] (n=39350) | 0.115 |
| heldout | 28 | 26.2% [23.1%, 29.7%] (n=543) | 97.3% (142/146) | 2.7% | 73.8% | 142/1350 | 466 | 341 | 100.0% [99.9%, 100.0%] (n=15290) | 0.14 (max 2; worlds with 0: 25/28) | 0.000 | 0.0002 | 33.9% [31.9%, 36.1%] (n=1350) | 6.6% [5.9%, 7.3%] (n=15290) | 0.122 |

## Per era (the era covering most of the final evaluation window)

| split | era | worlds | TP recall (detectable) | FP / world | noise rejected | detectable share |
|---|---|---|---|---|---|---|
| development | calm | 13 | 24.6% [17.9%, 31.9%] (n=171) | 0.08 | 100.0% [100.0%, 100.0%] (n=7125) | 31.1% |
| development | choppy | 13 | 24.5% [17.6%, 31.6%] (n=237) | 0.08 | 100.0% [100.0%, 100.0%] (n=7090) | 36.5% |
| development | crisis | 15 | 26.8% [22.6%, 31.8%] (n=276) | 0.27 | 100.0% [99.9%, 100.0%] (n=8240) | 42.5% |
| development | trending | 17 | 25.8% [23.1%, 28.6%] (n=310) | 0.12 | 100.0% [99.9%, 100.0%] (n=9230) | 36.5% |
| development | volatile | 14 | 25.5% [20.4%, 30.2%] (n=271) | 0.21 | 100.0% [99.9%, 100.0%] (n=7665) | 41.7% |
| heldout | calm | 4 | 30.9% [26.5%, 36.7%] (n=68) | 0.00 | 100.0% [100.0%, 100.0%] (n=2230) | 34.0% |
| heldout | choppy | 2 | 18.4% [5.9%, 28.6%] (n=38) | 0.00 | 100.0% [100.0%, 100.0%] (n=1070) | 38.0% |
| heldout | crisis | 5 | 28.4% [20.0%, 37.5%] (n=109) | 0.40 | 99.9% [99.8%, 100.0%] (n=2765) | 43.6% |
| heldout | trending | 10 | 22.0% [17.5%, 26.3%] (n=168) | 0.00 | 100.0% [100.0%, 100.0%] (n=5390) | 37.3% |
| heldout | volatile | 7 | 28.7% [23.2%, 34.2%] (n=160) | 0.29 | 99.9% [99.9%, 100.0%] (n=3835) | 45.7% |

## Real patterns per kind and strength band

| split | kind | band | n | detectable | not representable | undetectable | TP recall (detectable) | surfaced | mean oracle power | median delay looks |
|---|---|---|---|---|---|---|---|---|---|---|
| development | changing | extremely_subtle | 67 | 8 | 0 | 59 | 0.0% [0.0%, 0.0%] (n=8) | 10.4% | 0.18 |  |
| development | changing | faint | 67 | 24 | 0 | 43 | 0.0% [0.0%, 0.0%] (n=24) | 19.4% | 0.39 |  |
| development | changing | moderate | 67 | 61 | 0 | 6 | 27.9% [17.2%, 39.1%] (n=61) | 86.6% | 0.92 | 0 |
| development | changing | obvious | 67 | 67 | 0 | 0 | 38.8% [26.9%, 50.7%] (n=67) | 98.5% | 1.00 | 0 |
| development | changing | subtle | 67 | 47 | 0 | 20 | 10.6% [2.3%, 20.0%] (n=47) | 65.7% | 0.70 | 0 |
| development | conditional | extremely_subtle | 67 | 1 | 9 | 57 | 0.0% [0.0%, 0.0%] (n=1) | 7.5% | 0.22 |  |
| development | conditional | faint | 67 | 26 | 23 | 18 | 0.0% [0.0%, 0.0%] (n=26) | 28.4% | 0.67 |  |
| development | conditional | moderate | 67 | 66 | 1 | 0 | 30.3% [19.4%, 41.5%] (n=66) | 94.0% | 1.00 | 0 |
| development | conditional | obvious | 67 | 67 | 0 | 0 | 61.2% [50.7%, 73.1%] (n=67) | 98.5% | 1.00 | 0 |
| development | conditional | subtle | 67 | 56 | 10 | 1 | 3.6% [0.0%, 9.1%] (n=56) | 64.2% | 0.96 | 0 |
| development | delayed | extremely_subtle | 67 | 0 | 4 | 63 | n/a | 0.0% | 0.11 |  |
| development | delayed | faint | 67 | 4 | 25 | 38 | 0.0% [0.0%, 0.0%] (n=4) | 3.0% | 0.46 |  |
| development | delayed | moderate | 67 | 24 | 43 | 0 | 0.0% [0.0%, 0.0%] (n=24) | 25.4% | 0.99 |  |
| development | delayed | obvious | 67 | 37 | 30 | 0 | 2.7% [0.0%, 8.6%] (n=37) | 43.3% | 1.00 | 0 |
| development | delayed | subtle | 67 | 7 | 44 | 16 | 0.0% [0.0%, 0.0%] (n=7) | 10.4% | 0.71 |  |
| development | interactive | extremely_subtle | 67 | 1 | 12 | 54 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.25 |  |
| development | interactive | faint | 67 | 2 | 43 | 22 | 0.0% [0.0%, 0.0%] (n=2) | 0.0% | 0.65 |  |
| development | interactive | moderate | 67 | 0 | 67 | 0 | n/a | 0.0% | 1.00 |  |
| development | interactive | obvious | 67 | 1 | 66 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 1.00 |  |
| development | interactive | subtle | 67 | 1 | 65 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.95 |  |
| development | lifecycle | extremely_subtle | 67 | 8 | 1 | 58 | 0.0% [0.0%, 0.0%] (n=8) | 3.0% | 0.17 |  |
| development | lifecycle | faint | 67 | 0 | 6 | 61 | n/a | 3.0% | 0.10 |  |
| development | lifecycle | moderate | 67 | 67 | 0 | 0 | 50.7% [40.3%, 62.7%] (n=67) | 100.0% | 1.00 | 0 |
| development | lifecycle | obvious | 67 | 67 | 0 | 0 | 52.2% [40.3%, 64.2%] (n=67) | 95.5% | 1.00 | 0 |
| development | lifecycle | subtle | 67 | 32 | 0 | 35 | 0.0% [0.0%, 0.0%] (n=32) | 29.9% | 0.48 |  |
| development | linear | extremely_subtle | 67 | 14 | 0 | 53 | 0.0% [0.0%, 0.0%] (n=14) | 9.0% | 0.20 |  |
| development | linear | faint | 67 | 27 | 0 | 40 | 0.0% [0.0%, 0.0%] (n=27) | 23.9% | 0.39 |  |
| development | linear | moderate | 67 | 67 | 0 | 0 | 35.8% [25.4%, 46.3%] (n=67) | 98.5% | 0.99 | 0 |
| development | linear | obvious | 67 | 67 | 0 | 0 | 79.1% [70.1%, 88.1%] (n=67) | 100.0% | 1.00 | 0 |
| development | linear | subtle | 67 | 53 | 0 | 14 | 0.0% [0.0%, 0.0%] (n=53) | 55.2% | 0.77 |  |
| development | rare | extremely_subtle | 67 | 0 | 6 | 61 | n/a | 4.5% | 0.12 |  |
| development | rare | faint | 67 | 1 | 15 | 51 | 0.0% [0.0%, 0.0%] (n=1) | 4.5% | 0.26 |  |
| development | rare | moderate | 67 | 33 | 34 | 0 | 0.0% [0.0%, 0.0%] (n=33) | 31.3% | 0.95 |  |
| development | rare | obvious | 67 | 56 | 11 | 0 | 5.4% [0.0%, 12.3%] (n=56) | 64.2% | 1.00 | 0 |
| development | rare | subtle | 67 | 5 | 35 | 27 | 0.0% [0.0%, 0.0%] (n=5) | 6.0% | 0.57 |  |
| development | regime | extremely_subtle | 67 | 5 | 2 | 60 | 0.0% [0.0%, 0.0%] (n=5) | 9.0% | 0.15 |  |
| development | regime | faint | 67 | 21 | 7 | 39 | 0.0% [0.0%, 0.0%] (n=21) | 29.9% | 0.38 |  |
| development | regime | moderate | 67 | 32 | 2 | 33 | 50.0% [33.3%, 68.6%] (n=32) | 41.8% | 0.51 | 0 |
| development | regime | obvious | 67 | 32 | 3 | 32 | 46.9% [29.6%, 63.6%] (n=32) | 43.3% | 0.52 | 0 |
| development | regime | subtle | 67 | 26 | 6 | 35 | 26.9% [9.5%, 45.8%] (n=26) | 40.3% | 0.48 | 0 |
| development | threshold | extremely_subtle | 67 | 0 | 2 | 65 | n/a | 4.5% | 0.10 |  |
| development | threshold | faint | 67 | 3 | 10 | 54 | 0.0% [0.0%, 0.0%] (n=3) | 4.5% | 0.25 |  |
| development | threshold | moderate | 67 | 51 | 13 | 3 | 0.0% [0.0%, 0.0%] (n=51) | 53.7% | 0.92 |  |
| development | threshold | obvious | 67 | 66 | 1 | 0 | 36.4% [24.2%, 47.8%] (n=66) | 92.5% | 1.00 | 0 |
| development | threshold | subtle | 67 | 29 | 13 | 25 | 0.0% [0.0%, 0.0%] (n=29) | 23.9% | 0.59 |  |
| development | xor | extremely_subtle | 67 | 0 | 15 | 52 | n/a | 0.0% | 0.28 |  |
| development | xor | faint | 67 | 0 | 57 | 10 | n/a | 0.0% | 0.79 |  |
| development | xor | moderate | 67 | 1 | 66 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 1.00 |  |
| development | xor | obvious | 67 | 2 | 65 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 0.0% | 1.00 |  |
| development | xor | subtle | 67 | 0 | 66 | 1 | n/a | 0.0% | 0.97 |  |
| heldout | changing | extremely_subtle | 27 | 9 | 0 | 18 | 0.0% [0.0%, 0.0%] (n=9) | 14.8% | 0.26 |  |
| heldout | changing | faint | 27 | 8 | 0 | 19 | 0.0% [0.0%, 0.0%] (n=8) | 11.1% | 0.29 |  |
| heldout | changing | moderate | 27 | 25 | 0 | 2 | 24.0% [8.0%, 41.7%] (n=25) | 85.2% | 0.92 | 0 |
| heldout | changing | obvious | 27 | 27 | 0 | 0 | 40.7% [22.2%, 59.3%] (n=27) | 100.0% | 0.99 | 0 |
| heldout | changing | subtle | 27 | 20 | 0 | 7 | 5.0% [0.0%, 16.7%] (n=20) | 44.4% | 0.63 | 0 |
| heldout | conditional | extremely_subtle | 27 | 1 | 1 | 25 | 0.0% [0.0%, 0.0%] (n=1) | 3.7% | 0.14 |  |
| heldout | conditional | faint | 27 | 12 | 9 | 6 | 0.0% [0.0%, 0.0%] (n=12) | 25.9% | 0.72 |  |
| heldout | conditional | moderate | 27 | 27 | 0 | 0 | 29.6% [14.8%, 48.1%] (n=27) | 100.0% | 1.00 | 0 |
| heldout | conditional | obvious | 27 | 27 | 0 | 0 | 70.4% [51.9%, 88.9%] (n=27) | 100.0% | 1.00 | 0 |
| heldout | conditional | subtle | 27 | 25 | 2 | 0 | 4.0% [0.0%, 12.5%] (n=25) | 55.6% | 0.97 | 0 |
| heldout | delayed | extremely_subtle | 27 | 0 | 3 | 24 | n/a | 3.7% | 0.13 |  |
| heldout | delayed | faint | 27 | 1 | 9 | 17 | 0.0% [0.0%, 0.0%] (n=1) | 11.1% | 0.39 |  |
| heldout | delayed | moderate | 27 | 10 | 17 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 29.6% | 0.99 |  |
| heldout | delayed | obvious | 27 | 19 | 8 | 0 | 10.5% [0.0%, 26.7%] (n=19) | 59.3% | 1.00 | 0 |
| heldout | delayed | subtle | 27 | 6 | 17 | 4 | 0.0% [0.0%, 0.0%] (n=6) | 14.8% | 0.78 |  |
| heldout | interactive | extremely_subtle | 27 | 0 | 5 | 22 | n/a | 0.0% | 0.17 |  |
| heldout | interactive | faint | 27 | 0 | 18 | 9 | n/a | 0.0% | 0.61 |  |
| heldout | interactive | moderate | 27 | 1 | 26 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 1.00 |  |
| heldout | interactive | obvious | 27 | 1 | 26 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 1.00 |  |
| heldout | interactive | subtle | 27 | 0 | 27 | 0 | n/a | 0.0% | 0.98 |  |
| heldout | lifecycle | extremely_subtle | 27 | 2 | 1 | 24 | 0.0% [0.0%, 0.0%] (n=2) | 7.4% | 0.13 |  |
| heldout | lifecycle | faint | 27 | 1 | 2 | 24 | 0.0% [0.0%, 0.0%] (n=1) | 3.7% | 0.10 |  |
| heldout | lifecycle | moderate | 27 | 27 | 0 | 0 | 51.9% [33.3%, 70.4%] (n=27) | 100.0% | 1.00 | 0 |
| heldout | lifecycle | obvious | 27 | 27 | 0 | 0 | 44.4% [25.9%, 63.0%] (n=27) | 85.2% | 1.00 | 0 |
| heldout | lifecycle | subtle | 27 | 14 | 0 | 13 | 0.0% [0.0%, 0.0%] (n=14) | 40.7% | 0.52 |  |
| heldout | linear | extremely_subtle | 27 | 6 | 0 | 21 | 0.0% [0.0%, 0.0%] (n=6) | 3.7% | 0.24 |  |
| heldout | linear | faint | 27 | 12 | 0 | 15 | 0.0% [0.0%, 0.0%] (n=12) | 14.8% | 0.42 |  |
| heldout | linear | moderate | 27 | 27 | 0 | 0 | 44.4% [25.9%, 63.0%] (n=27) | 100.0% | 1.00 | 0 |
| heldout | linear | obvious | 27 | 27 | 0 | 0 | 85.2% [70.4%, 96.3%] (n=27) | 100.0% | 1.00 | 0 |
| heldout | linear | subtle | 27 | 22 | 0 | 5 | 9.1% [0.0%, 22.8%] (n=22) | 66.7% | 0.79 | 0 |
| heldout | rare | extremely_subtle | 27 | 0 | 1 | 26 | n/a | 14.8% | 0.12 |  |
| heldout | rare | faint | 27 | 0 | 3 | 24 | n/a | 11.1% | 0.22 |  |
| heldout | rare | moderate | 27 | 8 | 19 | 0 | 0.0% [0.0%, 0.0%] (n=8) | 25.9% | 0.96 |  |
| heldout | rare | obvious | 27 | 25 | 2 | 0 | 4.0% [0.0%, 12.5%] (n=25) | 66.7% | 1.00 | 0 |
| heldout | rare | subtle | 27 | 2 | 13 | 12 | 0.0% [0.0%, 0.0%] (n=2) | 11.1% | 0.58 |  |
| heldout | regime | extremely_subtle | 27 | 7 | 2 | 18 | 0.0% [0.0%, 0.0%] (n=7) | 11.1% | 0.26 |  |
| heldout | regime | faint | 27 | 12 | 0 | 15 | 8.3% [0.0%, 25.0%] (n=12) | 37.0% | 0.43 | 0 |
| heldout | regime | moderate | 27 | 13 | 0 | 14 | 53.8% [25.0%, 80.0%] (n=13) | 40.7% | 0.48 | 0 |
| heldout | regime | obvious | 27 | 13 | 0 | 14 | 69.2% [42.9%, 92.3%] (n=13) | 51.9% | 0.48 | 0 |
| heldout | regime | subtle | 27 | 12 | 1 | 14 | 50.0% [23.1%, 78.6%] (n=12) | 44.4% | 0.48 | 0 |
| heldout | threshold | extremely_subtle | 27 | 1 | 1 | 25 | 0.0% [0.0%, 0.0%] (n=1) | 7.4% | 0.14 |  |
| heldout | threshold | faint | 27 | 4 | 3 | 20 | 0.0% [0.0%, 0.0%] (n=4) | 3.7% | 0.27 |  |
| heldout | threshold | moderate | 27 | 23 | 4 | 0 | 0.0% [0.0%, 0.0%] (n=23) | 63.0% | 0.96 |  |
| heldout | threshold | obvious | 27 | 27 | 0 | 0 | 25.9% [11.1%, 40.7%] (n=27) | 85.2% | 0.99 | 0 |
| heldout | threshold | subtle | 27 | 12 | 7 | 8 | 0.0% [0.0%, 0.0%] (n=12) | 40.7% | 0.60 |  |
| heldout | xor | extremely_subtle | 27 | 0 | 10 | 17 | n/a | 0.0% | 0.39 |  |
| heldout | xor | faint | 27 | 0 | 23 | 4 | n/a | 0.0% | 0.76 |  |
| heldout | xor | moderate | 27 | 0 | 27 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | obvious | 27 | 0 | 27 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | subtle | 27 | 0 | 27 | 0 | n/a | 0.0% | 0.99 |  |

## Noise per kind

| split | kind | n | promoted (FP) | FP rate | surfaced | mean best gate share | last verdicts |
|---|---|---|---|---|---|---|---|
| development | adversarial_near | 1344 | 0 | 0.0% [0.0%, 0.0%] (n=1344) | 1.6% | 0.69 | {'FAILED': np.int64(21)} |
| development | autocorr_trap | 2256 | 0 | 0.0% [0.0%, 0.0%] (n=2256) | 8.7% | 0.70 | {'FAILED': np.int64(195), 'UNKNOWN': np.int64(1)} |
| development | base_null | 2520 | 0 | 0.0% [0.0%, 0.0%] (n=2520) | 4.4% | 0.71 | {'FAILED': np.int64(111), 'NEEDS_MORE_EVIDENCE': np.int64(1)} |
| development | coincidence | 1080 | 0 | 0.0% [0.0%, 0.0%] (n=1080) | 1.9% | 0.68 | {'FAILED': np.int64(21)} |
| development | context | 335 | 0 | 0.0% [0.0%, 0.0%] (n=335) | 10.7% | 0.70 | {'FAILED': np.int64(35), 'UNKNOWN': np.int64(1)} |
| development | delayed_coincidence | 1080 | 0 | 0.0% [0.0%, 0.0%] (n=1080) | 4.4% | 0.70 | {'FAILED': np.int64(47)} |
| development | early_decay | 720 | 0 | 0.0% [0.0%, 0.0%] (n=720) | 1.5% | 0.66 | {'FAILED': np.int64(11)} |
| development | fluke | 720 | 0 | 0.0% [0.0%, 0.0%] (n=720) | 3.2% | 0.70 | {'FAILED': np.int64(23)} |
| development | identity_null | 2160 | 0 | 0.0% [0.0%, 0.0%] (n=2160) | 15.6% | 0.70 | {'FAILED': np.int64(334), 'QUARANTINED': np.int64(2)} |
| development | interaction_component | 1340 | 0 | 0.0% [0.0%, 0.0%] (n=1340) | 2.2% | 0.68 | {'FAILED': np.int64(29)} |
| development | interaction_decoy | 720 | 0 | 0.0% [0.0%, 0.0%] (n=720) | 4.0% | 0.69 | {'FAILED': np.int64(29)} |
| development | leak | 907 | 1 | 0.1% [0.0%, 0.3%] (n=907) | 100.0% | 0.91 | {'QUARANTINED': np.int64(898), 'UNKNOWN': np.int64(8), 'PROMOTE': np.int64(1)} |
| development | mt_winner | 5400 | 0 | 0.0% [0.0%, 0.0%] (n=5400) | 2.0% | 0.67 | {'FAILED': np.int64(107)} |
| development | near_pattern | 1344 | 0 | 0.0% [0.0%, 0.0%] (n=1344) | 2.6% | 0.67 | {'FAILED': np.int64(35)} |
| development | null | 7230 | 0 | 0.0% [0.0%, 0.0%] (n=7230) | 2.3% | 0.67 | {'FAILED': np.int64(167)} |
| development | proxy | 1080 | 10 | 0.9% [0.5%, 1.5%] (n=1080) | 29.7% | 0.80 | {'FAILED': np.int64(303), 'PROMOTE': np.int64(10), 'NEEDS_MORE_EVIDENCE': np.int64(7)} |
| development | regime_corr | 1440 | 0 | 0.0% [0.0%, 0.0%] (n=1440) | 2.3% | 0.67 | {'FAILED': np.int64(32), 'NEEDS_MORE_EVIDENCE': np.int64(1)} |
| development | reversal | 1080 | 0 | 0.0% [0.0%, 0.0%] (n=1080) | 0.9% | 0.73 | {'FAILED': np.int64(10)} |
| development | sample_size | 1440 | 0 | 0.0% [0.0%, 0.0%] (n=1440) | 0.0% | n/a | {} |
| development | selection_bias | 720 | 0 | 0.0% [0.0%, 0.0%] (n=720) | 1.2% | 0.60 | {'FAILED': np.int64(8), 'QUARANTINED': np.int64(1)} |
| development | strong_nontransferable | 907 | 0 | 0.0% [0.0%, 0.0%] (n=907) | 8.7% | 0.71 | {'FAILED': np.int64(76), 'NEEDS_MORE_EVIDENCE': np.int64(3)} |
| development | survivor_bias | 907 | 0 | 0.0% [0.0%, 0.0%] (n=907) | 3.9% | 0.67 | {'FAILED': np.int64(35)} |
| development | threshold_illusion | 1800 | 0 | 0.0% [0.0%, 0.0%] (n=1800) | 2.0% | 0.66 | {'FAILED': np.int64(36)} |
| development | vol_corr | 1440 | 0 | 0.0% [0.0%, 0.0%] (n=1440) | 7.4% | 0.70 | {'FAILED': np.int64(107)} |
| development | xor_trap | 720 | 0 | 0.0% [0.0%, 0.0%] (n=720) | 4.4% | 0.72 | {'FAILED': np.int64(32)} |
| heldout | adversarial_near | 487 | 0 | 0.0% [0.0%, 0.0%] (n=487) | 0.8% | 0.69 | {'FAILED': np.int64(4)} |
| heldout | autocorr_trap | 813 | 0 | 0.0% [0.0%, 0.0%] (n=813) | 8.7% | 0.70 | {'FAILED': np.int64(70), 'QUARANTINED': np.int64(1)} |
| heldout | base_null | 980 | 0 | 0.0% [0.0%, 0.0%] (n=980) | 3.0% | 0.71 | {'FAILED': np.int64(29)} |
| heldout | coincidence | 420 | 0 | 0.0% [0.0%, 0.0%] (n=420) | 2.1% | 0.67 | {'FAILED': np.int64(9)} |
| heldout | context | 135 | 0 | 0.0% [0.0%, 0.0%] (n=135) | 24.4% | 0.72 | {'FAILED': np.int64(32), 'NEEDS_MORE_EVIDENCE': np.int64(1)} |
| heldout | delayed_coincidence | 420 | 0 | 0.0% [0.0%, 0.0%] (n=420) | 4.8% | 0.66 | {'FAILED': np.int64(20)} |
| heldout | early_decay | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 2.5% | 0.67 | {'FAILED': np.int64(7)} |
| heldout | fluke | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 1.4% | 0.77 | {'FAILED': np.int64(4)} |
| heldout | identity_null | 840 | 0 | 0.0% [0.0%, 0.0%] (n=840) | 13.7% | 0.71 | {'FAILED': np.int64(115)} |
| heldout | interaction_component | 540 | 0 | 0.0% [0.0%, 0.0%] (n=540) | 2.2% | 0.71 | {'FAILED': np.int64(12)} |
| heldout | interaction_decoy | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 3.9% | 0.63 | {'FAILED': np.int64(11)} |
| heldout | leak | 328 | 0 | 0.0% [0.0%, 0.0%] (n=328) | 100.0% | 0.91 | {'QUARANTINED': np.int64(328)} |
| heldout | mt_winner | 1950 | 0 | 0.0% [0.0%, 0.0%] (n=1950) | 1.9% | 0.65 | {'FAILED': np.int64(37)} |
| heldout | near_pattern | 487 | 0 | 0.0% [0.0%, 0.0%] (n=487) | 3.1% | 0.67 | {'FAILED': np.int64(15)} |
| heldout | null | 3204 | 0 | 0.0% [0.0%, 0.0%] (n=3204) | 2.0% | 0.67 | {'FAILED': np.int64(63)} |
| heldout | proxy | 420 | 4 | 1.0% [0.0%, 2.1%] (n=420) | 35.0% | 0.80 | {'FAILED': np.int64(141), 'PROMOTE': np.int64(4), 'NEEDS_MORE_EVIDENCE': np.int64(2)} |
| heldout | regime_corr | 560 | 0 | 0.0% [0.0%, 0.0%] (n=560) | 2.0% | 0.62 | {'FAILED': np.int64(11)} |
| heldout | reversal | 420 | 0 | 0.0% [0.0%, 0.0%] (n=420) | 1.9% | 0.74 | {'FAILED': np.int64(8)} |
| heldout | sample_size | 560 | 0 | 0.0% [0.0%, 0.0%] (n=560) | 0.0% | n/a | {} |
| heldout | selection_bias | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 3.2% | 0.62 | {'FAILED': np.int64(9)} |
| heldout | strong_nontransferable | 328 | 0 | 0.0% [0.0%, 0.0%] (n=328) | 6.7% | 0.70 | {'FAILED': np.int64(22)} |
| heldout | survivor_bias | 328 | 0 | 0.0% [0.0%, 0.0%] (n=328) | 4.9% | 0.66 | {'FAILED': np.int64(16)} |
| heldout | threshold_illusion | 650 | 0 | 0.0% [0.0%, 0.0%] (n=650) | 1.4% | 0.75 | {'FAILED': np.int64(9)} |
| heldout | vol_corr | 560 | 0 | 0.0% [0.0%, 0.0%] (n=560) | 5.4% | 0.69 | {'FAILED': np.int64(30)} |
| heldout | xor_trap | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 4.3% | 0.65 | {'FAILED': np.int64(12)} |

## C73: how close (confidence = 1 - the screen's BH q at the last look)

| split | label | n | brier | mean confidence | mean |effect error| | mean effect error | sign agreement | median final rank | just below the line (t in [1.5, 2)) | partial credit | near misses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| development | NOISE | 39350 | 0.1161 | 0.2203 | 0.01222 | 0.004128 | 0.7952 | 321 | 1822 | 0 | {'false positive (a proxy: informative but adds nothing)': np.int64(10), 'false positive': np.int64(1)} |
| development | REAL | 3350 | 0.515 | 0.4077 | 0.00671 | -0.001932 | 0.9954 | 85 | 227 | 385.5 | {'surfaced, not promoted': np.int64(520), 'found': np.int64(323), 'one gate short at the last look': np.int64(244), 'a proxy of it was promoted instead': np.int64(3)} |
| heldout | NOISE | 15290 | 0.1113 | 0.2122 | 0.01197 | 0.003671 | 0.7718 | 322 | 659 | 0 | {'false positive (a proxy: informative but adds nothing)': np.int64(4)} |
| heldout | REAL | 1350 | 0.509 | 0.4175 | 0.006909 | -0.001791 | 0.9913 | 74 | 97 | 165.2 | {'surfaced, not promoted': np.int64(224), 'found': np.int64(142), 'one gate short at the last look': np.int64(91), 'a proxy of it was promoted instead': np.int64(1)} |

## C73: calibration of that confidence

| bin | n | share_real | mean_conf | promoted |
|---|---|---|---|---|
| [0.0, 0.5) | 50008 | 0.05781 | 0.1282 | 0 |
| [0.5, 0.8) | 4564 | 0.09882 | 0.6315 | 0 |
| [0.8, 0.9) | 1063 | 0.1458 | 0.8486 | 0.001881 |
| [0.9, 0.95) | 582 | 0.2268 | 0.9241 | 0.01203 |
| [0.95, 0.99) | 515 | 0.3223 | 0.97 | 0.02524 |
| [0.99, 1.0) | 2608 | 0.347 | 1 | 0.1756 |

## C75 3E/3F: per difficulty tier (0 = NULL world)

| split | tier | worlds | recall (detectable) | precision | FDR | FP / world | noise rejected | worlds with nothing promoted |
|---|---|---|---|---|---|---|---|---|
| development | 0 | 5 | n/a | 0.0% | 100.0% | 0.20 | 100.0% [99.9%, 100.0%] (n=2650) | 4/5 |
| development | 1 | 10 | 29.1% [25.0%, 33.6%] (n=213) | 96.9% | 3.1% | 0.20 | 100.0% [99.9%, 100.0%] (n=5350) | 0/10 |
| development | 2 | 14 | 27.3% [22.4%, 32.2%] (n=297) | 97.6% | 2.4% | 0.14 | 100.0% [99.9%, 100.0%] (n=7490) | 0/14 |
| development | 3 | 9 | 29.4% [22.5%, 35.4%] (n=163) | 96.0% | 4.0% | 0.22 | 100.0% [99.9%, 100.0%] (n=4815) | 0/9 |
| development | 4 | 7 | 25.9% [22.4%, 29.0%] (n=147) | 97.4% | 2.6% | 0.14 | 100.0% [99.9%, 100.0%] (n=4060) | 0/7 |
| development | 5 | 15 | 22.8% [17.5%, 28.6%] (n=254) | 95.1% | 4.9% | 0.20 | 100.0% [99.9%, 100.0%] (n=8025) | 0/15 |
| development | 6 | 12 | 18.8% [13.7%, 24.0%] (n=191) | 100.0% | 0.0% | 0.00 | 100.0% [100.0%, 100.0%] (n=6960) | 1/12 |
| heldout | 0 | 1 | n/a | n/a | n/a | 0.00 | 100.0% [100.0%, 100.0%] (n=530) | 1/1 |
| heldout | 1 | 5 | 31.0% [26.0%, 35.8%] (n=113) | 92.1% | 7.9% | 0.60 | 99.9% [99.7%, 100.0%] (n=2675) | 0/5 |
| heldout | 2 | 6 | 26.8% [20.0%, 33.3%] (n=127) | 97.1% | 2.9% | 0.17 | 100.0% [99.9%, 100.0%] (n=3210) | 0/6 |
| heldout | 3 | 6 | 22.9% [19.6%, 25.6%] (n=118) | 100.0% | 0.0% | 0.00 | 100.0% [100.0%, 100.0%] (n=3210) | 0/6 |
| heldout | 4 | 3 | 26.3% [17.4%, 40.0%] (n=57) | 100.0% | 0.0% | 0.00 | 100.0% [100.0%, 100.0%] (n=1740) | 0/3 |
| heldout | 5 | 3 | 18.8% [5.9%, 26.7%] (n=48) | 100.0% | 0.0% | 0.00 | 100.0% [100.0%, 100.0%] (n=1605) | 0/3 |
| heldout | 6 | 4 | 27.5% [15.8%, 39.5%] (n=80) | 100.0% | 0.0% | 0.00 | 100.0% [100.0%, 100.0%] (n=2320) | 0/4 |

## F28: real patterns per kind (all bands)

| split | kind | n | detectable | TP recall (detectable) | found any status | surfaced (detectable) |
|---|---|---|---|---|---|---|
| development | changing | 335 | 207 | 23.2% [18.0%, 28.7%] (n=207) | 48 | 86.0% [81.6%, 90.3%] (n=207) |
| development | conditional | 335 | 216 | 29.2% [24.2%, 33.9%] (n=216) | 63 | 87.0% [81.9%, 91.2%] (n=216) |
| development | delayed | 335 | 72 | 1.4% [0.0%, 4.3%] (n=72) | 1 | 68.1% [56.3%, 79.1%] (n=72) |
| development | interactive | 335 | 5 | 0.0% [0.0%, 0.0%] (n=5) | 0 | 0.0% [0.0%, 0.0%] (n=5) |
| development | lifecycle | 335 | 174 | 39.7% [33.9%, 45.7%] (n=174) | 69 | 84.5% [80.1%, 88.7%] (n=174) |
| development | linear | 335 | 228 | 33.8% [28.8%, 38.5%] (n=228) | 77 | 79.8% [74.9%, 84.8%] (n=228) |
| development | rare | 335 | 95 | 3.2% [0.0%, 7.1%] (n=95) | 3 | 63.2% [52.7%, 72.6%] (n=95) |
| development | regime | 335 | 116 | 32.8% [23.3%, 41.8%] (n=116) | 38 | 89.7% [82.4%, 95.4%] (n=116) |
| development | threshold | 335 | 149 | 16.1% [10.7%, 21.2%] (n=149) | 24 | 73.8% [67.7%, 79.7%] (n=149) |
| development | xor | 335 | 3 | 0.0% [0.0%, 0.0%] (n=3) | 0 | 0.0% [0.0%, 0.0%] (n=3) |
| heldout | changing | 135 | 89 | 20.2% [12.8%, 28.9%] (n=89) | 18 | 75.3% [65.9%, 84.0%] (n=89) |
| heldout | conditional | 135 | 92 | 30.4% [23.3%, 38.5%] (n=92) | 28 | 82.6% [77.4%, 88.2%] (n=92) |
| heldout | delayed | 135 | 36 | 5.6% [0.0%, 13.9%] (n=36) | 2 | 69.4% [54.3%, 84.2%] (n=36) |
| heldout | interactive | 135 | 2 | 0.0% [0.0%, 0.0%] (n=2) | 0 | 0.0% [0.0%, 0.0%] (n=2) |
| heldout | lifecycle | 135 | 71 | 36.6% [27.4%, 46.4%] (n=71) | 26 | 83.1% [76.4%, 90.1%] (n=71) |
| heldout | linear | 135 | 94 | 39.4% [32.3%, 46.4%] (n=94) | 37 | 79.8% [74.2%, 85.7%] (n=94) |
| heldout | rare | 135 | 35 | 2.9% [0.0%, 8.6%] (n=35) | 1 | 65.7% [51.4%, 79.4%] (n=35) |
| heldout | regime | 135 | 57 | 40.4% [24.4%, 55.7%] (n=57) | 23 | 82.5% [72.3%, 92.5%] (n=57) |
| heldout | threshold | 135 | 67 | 10.4% [4.3%, 17.5%] (n=67) | 7 | 70.1% [58.2%, 81.0%] (n=67) |
| heldout | xor | 135 | 0 | n/a | 0 | n/a |

## F28: real patterns per strength band (all kinds)

| split | band | n | detectable | TP recall (detectable) | found any status | surfaced (detectable) |
|---|---|---|---|---|---|---|
| development | extremely_subtle | 670 | 37 | 0.0% [0.0%, 0.0%] (n=37) | 0 | 32.4% [17.1%, 47.7%] (n=37) |
| development | faint | 670 | 108 | 0.0% [0.0%, 0.0%] (n=108) | 0 | 55.6% [46.6%, 64.3%] (n=108) |
| development | moderate | 670 | 402 | 27.6% [23.6%, 31.3%] (n=402) | 111 | 87.6% [84.8%, 90.1%] (n=402) |
| development | obvious | 670 | 462 | 42.9% [37.9%, 48.1%] (n=462) | 198 | 91.3% [88.9%, 93.7%] (n=462) |
| development | subtle | 670 | 256 | 5.5% [3.1%, 8.0%] (n=256) | 14 | 67.2% [61.0%, 73.0%] (n=256) |
| heldout | extremely_subtle | 270 | 26 | 0.0% [0.0%, 0.0%] (n=26) | 0 | 34.6% [15.8%, 51.7%] (n=26) |
| heldout | faint | 270 | 50 | 2.0% [0.0%, 6.1%] (n=50) | 1 | 44.0% [31.2%, 55.6%] (n=50) |
| heldout | moderate | 270 | 161 | 29.2% [22.2%, 36.0%] (n=161) | 47 | 87.6% [83.2%, 91.6%] (n=161) |
| heldout | obvious | 270 | 193 | 43.5% [38.5%, 49.5%] (n=193) | 84 | 90.2% [86.6%, 93.7%] (n=193) |
| heldout | subtle | 270 | 113 | 8.8% [4.2%, 14.2%] (n=113) | 10 | 64.6% [52.6%, 75.4%] (n=113) |

## F28: false positives per noise family

| split | family | n | promoted (FP) | FP / world | FP rate | kinds promoted |
|---|---|---|---|---|---|---|
| development | correlated with a real pattern (proxy-shaped) | 3768 | 10 | 0.14 | 0.3% [0.1%, 0.4%] (n=3768) | {'proxy': np.int64(10)} |
| development | future-derived (leak-shaped) | 1814 | 1 | 0.01 | 0.1% [0.0%, 0.2%] (n=1814) | {'leak': np.int64(1)} |
| development | interaction component (part of a real pattern) | 1340 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=1340) | {} |
| development | per-name persistent (identity-shaped) | 8711 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=8711) | {} |
| development | pure null (per-date effect zero everywhere) | 18030 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=18030) | {} |
| development | transient or local effect | 7027 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=7027) | {} |
| heldout | correlated with a real pattern (proxy-shaped) | 1394 | 4 | 0.14 | 0.3% [0.0%, 0.7%] (n=1394) | {'proxy': np.int64(4)} |
| heldout | future-derived (leak-shaped) | 656 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=656) | {} |
| heldout | interaction component (part of a real pattern) | 540 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=540) | {} |
| heldout | per-name persistent (identity-shaped) | 3328 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=3328) | {} |
| heldout | pure null (per-date effect zero everywhere) | 7204 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=7204) | {} |
| heldout | transient or local effect | 2708 | 0 | 0.00 | 0.0% [0.0%, 0.0%] (n=2708) | {} |

## F28: null worlds (the right answer is nothing)

| split | null worlds | worlds promoting anything | promotions |
|---|---|---|---|
| development | 5 | 1 | 1 |
| heldout | 1 | 0 | 0 |

## F28: each fix switched off (counterfactual verdicts of the same bundles; exact for the final-look protocol)

| split | configuration | real found (detectable) | real found (any) | FP | FP / world | proxy FP | identity_null FP | leak FP | other FP |
|---|---|---|---|---|---|---|---|---|---|
| development | with all F28 fixes | 323/1265 | 323 | 11 | 0.15 | 10 | 0 | 1 | {} |
| development | no_rival | 323/1265 | 323 | 74 | 1.03 | 71 | 0 | 1 | {'vol_corr': np.int64(2)} |
| development | no_name_units | 323/1265 | 323 | 18 | 0.25 | 10 | 3 | 1 | {'vol_corr': np.int64(3), 'base_null': np.int64(1)} |
| development | no_leak_suspect | 329/1265 | 329 | 19 | 0.26 | 10 | 0 | 9 | {} |
| development | no_f28 | 329/1265 | 329 | 92 | 1.28 | 73 | 3 | 9 | {'vol_corr': np.int64(6), 'base_null': np.int64(1)} |
| heldout | with all F28 fixes | 142/543 | 142 | 4 | 0.14 | 4 | 0 | 0 | {} |
| heldout | no_rival | 142/543 | 142 | 35 | 1.25 | 35 | 0 | 0 | {} |
| heldout | no_name_units | 142/543 | 142 | 5 | 0.18 | 4 | 1 | 0 | {} |
| heldout | no_leak_suspect | 145/543 | 145 | 4 | 0.14 | 4 | 0 | 0 | {} |
| heldout | no_f28 | 145/543 | 145 | 36 | 1.29 | 35 | 1 | 0 | {} |

## Which gate blocked what (last look)

| gate | real (detectable, not promoted) | noise (gated) |
|---|---|---|
| calibration | 15.6% of 972 | 14.5% of 3720 |
| complexity | 33.8% of 972 | 39.4% of 3720 |
| failure_behavior | 0.1% of 972 | 1.3% of 3720 |
| identity | 1.4% of 972 | 9.0% of 3720 |
| leakage | 2.0% of 972 | 35.4% of 3720 |
| out_of_sample | 50.6% of 972 | 65.8% of 3720 |
| replication | 92.7% of 972 | 61.8% of 3720 |
| transfer | 25.2% of 972 | 40.5% of 3720 |

## C75 Phase 7: next unseen worlds after k worlds

| history_worlds | next_worlds | recall_next | fp_per_world_next | noise_rejection_next |
|---|---|---|---|---|
| 1 | 5 | 0.2421 | 0 | 1 |
| 5 | 5 | 0.2029 | 0 | 1 |
| 10 | 10 | 0.3444 | 0.3 | 0.9994 |
| 25 | 25 | 0.2389 | 0.2 | 0.9996 |
| 50 | 50 | 0.2602 | 0.12 | 0.9998 |

## Ranked concrete failures (drive the next briefs)

| failure | where | eras | count | per_100_worlds |
|---|---|---|---|---|
| real only detectable in its true form (single-feature screen cannot represent it): xor | candidate generation | {'trending': np.int64(104), 'volatile': np.int64(88), 'crisis': np.int64(68), 'calm': np.int64(63), 'choppy': np.int64(60)} | 383 | 383 |
| real only detectable in its true form (single-feature screen cannot represent it): interactive | candidate generation | {'trending': np.int64(101), 'volatile': np.int64(79), 'crisis': np.int64(68), 'calm': np.int64(54), 'choppy': np.int64(53)} | 355 | 355 |
| real only detectable in its true form (single-feature screen cannot represent it): delayed | candidate generation | {'trending': np.int64(54), 'volatile': np.int64(45), 'crisis': np.int64(44), 'calm': np.int64(34), 'choppy': np.int64(23)} | 200 | 200 |
| real only detectable in its true form (single-feature screen cannot represent it): rare | candidate generation | {'trending': np.int64(37), 'volatile': np.int64(31), 'crisis': np.int64(27), 'choppy': np.int64(23), 'calm': np.int64(21)} | 139 | 139 |
| detectable real missed: linear (extremely_subtle, faint, moderate, subtle) | screen: never raised | {'trending': np.int64(15), 'volatile': np.int64(14), 'crisis': np.int64(13), 'choppy': np.int64(13), 'calm': np.int64(10)} | 65 | 65 |
| detectable real missed: conditional (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'volatile': np.int64(17), 'trending': np.int64(15), 'crisis': np.int64(15), 'calm': np.int64(8), 'choppy': np.int64(6)} | 61 | 61 |
| detectable real missed: threshold (faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(19), 'crisis': np.int64(15), 'volatile': np.int64(11), 'calm': np.int64(7), 'choppy': np.int64(7)} | 59 | 59 |
| real only detectable in its true form (single-feature screen cannot represent it): conditional | candidate generation | {'trending': np.int64(14), 'calm': np.int64(13), 'volatile': np.int64(12), 'crisis': np.int64(8), 'choppy': np.int64(8)} | 55 | 55 |
| detectable real missed: changing (faint, moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(11), 'calm': np.int64(11), 'crisis': np.int64(11), 'volatile': np.int64(11), 'choppy': np.int64(10)} | 54 | 54 |
| real only detectable in its true form (single-feature screen cannot represent it): threshold | candidate generation | {'trending': np.int64(15), 'volatile': np.int64(12), 'choppy': np.int64(10), 'calm': np.int64(10), 'crisis': np.int64(7)} | 54 | 54 |
| detectable real missed: linear (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(15), 'crisis': np.int64(12), 'calm': np.int64(10), 'choppy': np.int64(8), 'volatile': np.int64(6)} | 51 | 51 |
| detectable real missed: changing (extremely_subtle, faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(17), 'volatile': np.int64(12), 'calm': np.int64(10), 'choppy': np.int64(6), 'crisis': np.int64(6)} | 51 | 51 |
| detectable real missed: lifecycle (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(14), 'choppy': np.int64(10), 'calm': np.int64(9), 'volatile': np.int64(9), 'crisis': np.int64(8)} | 50 | 50 |
| detectable real missed: rare (faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(11), 'volatile': np.int64(11), 'crisis': np.int64(9), 'choppy': np.int64(8), 'calm': np.int64(8)} | 47 | 47 |
| detectable real missed: conditional (faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(11), 'volatile': np.int64(10), 'choppy': np.int64(9), 'calm': np.int64(7), 'crisis': np.int64(7)} | 44 | 44 |
| detectable real missed: threshold (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(15), 'calm': np.int64(9), 'crisis': np.int64(8), 'volatile': np.int64(8), 'choppy': np.int64(2)} | 42 | 42 |
| detectable real missed: lifecycle (extremely_subtle, obvious, subtle) | screen: never raised | {'trending': np.int64(12), 'calm': np.int64(8), 'volatile': np.int64(8), 'crisis': np.int64(6), 'choppy': np.int64(5)} | 39 | 39 |
| detectable real missed: delayed (faint, moderate, obvious, subtle) | screen: never raised | {'choppy': np.int64(10), 'crisis': np.int64(10), 'trending': np.int64(9), 'volatile': np.int64(4), 'calm': np.int64(1)} | 34 | 34 |
| detectable real missed: regime (faint, moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'crisis': np.int64(17), 'volatile': np.int64(11), 'trending': np.int64(2), 'choppy': np.int64(1)} | 31 | 31 |
| detectable real missed: changing (extremely_subtle, faint, moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication | {'trending': np.int64(8), 'volatile': np.int64(5), 'crisis': np.int64(5), 'calm': np.int64(3), 'choppy': np.int64(2)} | 23 | 23 |
| real only detectable in its true form (single-feature screen cannot represent it): regime | candidate generation | {'choppy': np.int64(9), 'trending': np.int64(6), 'crisis': np.int64(4), 'volatile': np.int64(3), 'calm': np.int64(1)} | 23 | 23 |
| detectable real missed: regime (extremely_subtle, faint, moderate, obvious, subtle) | screen: never raised | {'crisis': np.int64(8), 'volatile': np.int64(6), 'choppy': np.int64(4), 'trending': np.int64(2), 'calm': np.int64(2)} | 22 | 22 |
| detectable real missed: linear (faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication | {'trending': np.int64(8), 'volatile': np.int64(6), 'crisis': np.int64(4), 'calm': np.int64(2), 'choppy': np.int64(2)} | 22 | 22 |
| detectable real missed: delayed (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(6), 'calm': np.int64(4), 'volatile': np.int64(4), 'crisis': np.int64(3), 'choppy': np.int64(3)} | 20 | 20 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by complexity,out_of_sample,replication | {'volatile': np.int64(6), 'trending': np.int64(6), 'crisis': np.int64(3), 'choppy': np.int64(2), 'calm': np.int64(1)} | 18 | 18 |
| detectable real missed: conditional (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication | {'volatile': np.int64(8), 'calm': np.int64(3), 'trending': np.int64(2), 'choppy': np.int64(2), 'crisis': np.int64(2)} | 17 | 17 |
| detectable real missed: conditional (faint, moderate, subtle) | live: FAILED blocked by complexity,out_of_sample,replication | {'crisis': np.int64(5), 'trending': np.int64(4), 'volatile': np.int64(4), 'calm': np.int64(3), 'choppy': np.int64(1)} | 17 | 17 |
| detectable real missed: threshold (extremely_subtle, faint, moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication | {'volatile': np.int64(7), 'trending': np.int64(3), 'calm': np.int64(3), 'choppy': np.int64(3)} | 16 | 16 |
| detectable real missed: changing (extremely_subtle, moderate, obvious, subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'volatile': np.int64(6), 'choppy': np.int64(5), 'trending': np.int64(4), 'crisis': np.int64(1)} | 16 | 16 |
| detectable real missed: linear (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by complexity,out_of_sample,replication | {'trending': np.int64(5), 'calm': np.int64(4), 'crisis': np.int64(3), 'volatile': np.int64(2), 'choppy': np.int64(2)} | 16 | 16 |
| detectable real missed: conditional (faint, moderate, obvious, subtle) | live: FAILED blocked by replication,transfer | {'trending': np.int64(4), 'choppy': np.int64(4), 'crisis': np.int64(4), 'volatile': np.int64(3)} | 15 | 15 |
| detectable real missed: lifecycle (moderate, obvious, subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'volatile': np.int64(7), 'trending': np.int64(4), 'calm': np.int64(2), 'crisis': np.int64(1), 'choppy': np.int64(1)} | 15 | 15 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by out_of_sample,replication | {'trending': np.int64(7), 'calm': np.int64(4), 'choppy': np.int64(3), 'volatile': np.int64(1)} | 15 | 15 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by out_of_sample,replication | {'choppy': np.int64(6), 'trending': np.int64(4), 'crisis': np.int64(3), 'volatile': np.int64(2)} | 15 | 15 |
| noise promoted: proxy | promoted | {'crisis': np.int64(6), 'volatile': np.int64(5), 'trending': np.int64(2), 'choppy': np.int64(1)} | 14 | 14 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'trending': np.int64(3), 'choppy': np.int64(3), 'calm': np.int64(3), 'crisis': np.int64(3), 'volatile': np.int64(1)} | 13 | 13 |
| detectable real missed: changing (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(6), 'calm': np.int64(3), 'choppy': np.int64(2), 'volatile': np.int64(1), 'crisis': np.int64(1)} | 13 | 13 |
| detectable real missed: regime (faint, moderate, obvious, subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'crisis': np.int64(7), 'volatile': np.int64(6)} | 13 | 13 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'trending': np.int64(4), 'crisis': np.int64(3), 'volatile': np.int64(3), 'calm': np.int64(1), 'choppy': np.int64(1)} | 12 | 12 |
| detectable real missed: changing (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'trending': np.int64(4), 'crisis': np.int64(3), 'volatile': np.int64(3), 'calm': np.int64(1), 'choppy': np.int64(1)} | 12 | 12 |
| detectable real missed: changing (faint, moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'trending': np.int64(4), 'crisis': np.int64(2), 'calm': np.int64(2), 'choppy': np.int64(2), 'volatile': np.int64(1)} | 11 | 11 |
| detectable real missed: changing (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication | {'volatile': np.int64(5), 'choppy': np.int64(3), 'calm': np.int64(2), 'trending': np.int64(1)} | 11 | 11 |
| detectable real missed: linear (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'trending': np.int64(5), 'volatile': np.int64(2), 'choppy': np.int64(2), 'crisis': np.int64(1), 'calm': np.int64(1)} | 11 | 11 |
| detectable real missed: delayed (faint, moderate, obvious) | live: FAILED blocked by complexity,out_of_sample,replication | {'trending': np.int64(5), 'choppy': np.int64(3), 'volatile': np.int64(2), 'calm': np.int64(1)} | 11 | 11 |
| detectable real missed: linear (faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'choppy': np.int64(3), 'volatile': np.int64(3), 'calm': np.int64(2), 'trending': np.int64(2)} | 10 | 10 |
| detectable real missed: conditional (faint, moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'trending': np.int64(5), 'choppy': np.int64(2), 'crisis': np.int64(2), 'volatile': np.int64(1)} | 10 | 10 |
| real only detectable in its true form (single-feature screen cannot represent it): lifecycle | candidate generation | {'volatile': np.int64(4), 'crisis': np.int64(2), 'trending': np.int64(2), 'calm': np.int64(1), 'choppy': np.int64(1)} | 10 | 10 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'crisis': np.int64(4), 'choppy': np.int64(4), 'volatile': np.int64(1), 'trending': np.int64(1)} | 10 | 10 |
| detectable real missed: rare (moderate, obvious, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'trending': np.int64(4), 'crisis': np.int64(2), 'choppy': np.int64(2), 'calm': np.int64(1)} | 9 | 9 |
| detectable real missed: linear (moderate, obvious, subtle) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'trending': np.int64(6), 'volatile': np.int64(2), 'calm': np.int64(1)} | 9 | 9 |
| detectable real missed: delayed (moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'trending': np.int64(3), 'volatile': np.int64(3), 'calm': np.int64(1), 'choppy': np.int64(1), 'crisis': np.int64(1)} | 9 | 9 |
| detectable real missed: regime (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'volatile': np.int64(6), 'trending': np.int64(2), 'crisis': np.int64(1)} | 9 | 9 |
| detectable real missed: conditional (moderate, obvious) | live: FAILED blocked by transfer | {'trending': np.int64(3), 'volatile': np.int64(2), 'choppy': np.int64(2), 'crisis': np.int64(1), 'calm': np.int64(1)} | 9 | 9 |
| detectable real missed: delayed (faint, moderate, obvious, subtle) | live: FAILED blocked by out_of_sample,replication | {'trending': np.int64(2), 'crisis': np.int64(2), 'volatile': np.int64(2), 'calm': np.int64(2), 'choppy': np.int64(1)} | 9 | 9 |
| detectable real missed: lifecycle (moderate, obvious, subtle) | live: FAILED blocked by complexity,out_of_sample,replication | {'trending': np.int64(3), 'crisis': np.int64(2), 'volatile': np.int64(2), 'calm': np.int64(1)} | 8 | 8 |
| detectable real missed: lifecycle (obvious, subtle) | live: FAILED blocked by out_of_sample,replication | {'trending': np.int64(2), 'volatile': np.int64(2), 'crisis': np.int64(1), 'calm': np.int64(1), 'choppy': np.int64(1)} | 7 | 7 |
| detectable real missed: rare (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'crisis': np.int64(3), 'volatile': np.int64(1), 'trending': np.int64(1), 'calm': np.int64(1), 'choppy': np.int64(1)} | 7 | 7 |
| detectable real missed: conditional (faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'trending': np.int64(3), 'calm': np.int64(1), 'volatile': np.int64(1), 'choppy': np.int64(1), 'crisis': np.int64(1)} | 7 | 7 |
| detectable real missed: interactive (extremely_subtle, faint, moderate, obvious, subtle) | screen: never raised | {'choppy': np.int64(4), 'crisis': np.int64(2), 'volatile': np.int64(1)} | 7 | 7 |
| detectable real missed: regime (extremely_subtle, faint, moderate, obvious) | live: FAILED blocked by complexity,out_of_sample,replication | {'volatile': np.int64(4), 'trending': np.int64(1), 'calm': np.int64(1)} | 6 | 6 |
| detectable real missed: delayed (moderate, obvious, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'trending': np.int64(2), 'choppy': np.int64(2), 'volatile': np.int64(1), 'crisis': np.int64(1)} | 6 | 6 |
| detectable real missed: regime (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'volatile': np.int64(3), 'crisis': np.int64(2), 'choppy': np.int64(1)} | 6 | 6 |
| detectable real missed: changing (moderate, obvious, subtle) | live: FAILED blocked by replication,transfer | {'calm': np.int64(2), 'volatile': np.int64(1), 'crisis': np.int64(1), 'choppy': np.int64(1)} | 5 | 5 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(2), 'choppy': np.int64(2), 'volatile': np.int64(1)} | 5 | 5 |
| detectable real missed: regime (extremely_subtle, faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication | {'volatile': np.int64(4), 'crisis': np.int64(1)} | 5 | 5 |
| detectable real missed: lifecycle (moderate) | live: FAILED blocked by replication,transfer | {'volatile': np.int64(2), 'calm': np.int64(2), 'crisis': np.int64(1)} | 5 | 5 |
| detectable real missed: linear (moderate, obvious) | live: UNKNOWN blocked by leakage | {'choppy': np.int64(2), 'volatile': np.int64(1), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: regime (moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'volatile': np.int64(3), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: lifecycle (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'choppy': np.int64(2), 'crisis': np.int64(1), 'volatile': np.int64(1)} | 4 | 4 |
| detectable real missed: delayed (moderate, subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'choppy': np.int64(2), 'calm': np.int64(1), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: conditional (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'volatile': np.int64(2), 'trending': np.int64(1), 'choppy': np.int64(1)} | 4 | 4 |
| detectable real missed: lifecycle (faint, subtle) | live: FAILED blocked by complexity,out_of_sample,replication,transfer | {'volatile': np.int64(3), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: changing (moderate, subtle) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(3), 'volatile': np.int64(1)} | 4 | 4 |
| detectable real missed: changing (faint, moderate, subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'choppy': np.int64(1), 'trending': np.int64(1), 'volatile': np.int64(1), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: conditional (faint, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(2), 'crisis': np.int64(1), 'calm': np.int64(1)} | 4 | 4 |
| detectable real missed: changing (extremely_subtle, moderate) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'volatile': np.int64(3), 'crisis': np.int64(1)} | 4 | 4 |
| detectable real missed: regime (faint, moderate) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'crisis': np.int64(3), 'volatile': np.int64(1)} | 4 | 4 |
| detectable real missed: regime (faint, moderate, subtle) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'volatile': np.int64(2), 'crisis': np.int64(2)} | 4 | 4 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'choppy': np.int64(2), 'calm': np.int64(1), 'volatile': np.int64(1)} | 4 | 4 |
| detectable real missed: threshold (moderate, subtle) | live: FAILED blocked by out_of_sample | {'crisis': np.int64(2), 'trending': np.int64(1), 'choppy': np.int64(1)} | 4 | 4 |
| detectable real missed: conditional (faint, moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'crisis': np.int64(2), 'trending': np.int64(2)} | 4 | 4 |
| detectable real missed: threshold (moderate, obvious) | live: FAILED blocked by replication,transfer | {'volatile': np.int64(3), 'trending': np.int64(1)} | 4 | 4 |
| detectable real missed: lifecycle (obvious) | live: FAILED blocked by calibration,out_of_sample | {'choppy': np.int64(2), 'crisis': np.int64(1)} | 3 | 3 |
| detectable real missed: lifecycle (obvious) | live: UNKNOWN blocked by leakage | {'crisis': np.int64(1), 'volatile': np.int64(1), 'choppy': np.int64(1)} | 3 | 3 |
| detectable real missed: rare (obvious) | live: FAILED blocked by complexity,identity,out_of_sample,replication,transfer | {'trending': np.int64(2), 'calm': np.int64(1)} | 3 | 3 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'volatile': np.int64(1), 'crisis': np.int64(1), 'trending': np.int64(1)} | 3 | 3 |
| detectable real missed: linear (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(1), 'crisis': np.int64(1), 'calm': np.int64(1)} | 3 | 3 |
| detectable real missed: linear (extremely_subtle, moderate, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'crisis': np.int64(2), 'trending': np.int64(1)} | 3 | 3 |
| detectable real missed: delayed (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'trending': np.int64(1), 'choppy': np.int64(1), 'volatile': np.int64(1)} | 3 | 3 |
| detectable real missed: regime (faint, obvious) | live: FAILED blocked by calibration,replication | {'volatile': np.int64(3)} | 3 | 3 |
| detectable real missed: xor (moderate, obvious) | screen: never raised | {'crisis': np.int64(1), 'trending': np.int64(1), 'choppy': np.int64(1)} | 3 | 3 |
| detectable real missed: threshold (moderate) | live: NEEDS_MORE_EVIDENCE blocked by complexity,replication | {'trending': np.int64(2), 'calm': np.int64(1)} | 3 | 3 |
| detectable real missed: conditional (moderate, subtle) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'choppy': np.int64(1), 'crisis': np.int64(1), 'trending': np.int64(1)} | 3 | 3 |
| detectable real missed: conditional (faint, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'choppy': np.int64(1), 'volatile': np.int64(1), 'crisis': np.int64(1)} | 3 | 3 |
| detectable real missed: changing (faint, obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(3)} | 3 | 3 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,complexity,replication | {'volatile': np.int64(1), 'choppy': np.int64(1), 'trending': np.int64(1)} | 3 | 3 |
| detectable real missed: changing (moderate) | live: FAILED blocked by transfer | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 2 |
| detectable real missed: changing (obvious) | live: UNKNOWN blocked by leakage,replication | {'volatile': np.int64(2)} | 2 | 2 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by complexity,replication,transfer | {'choppy': np.int64(1), 'trending': np.int64(1)} | 2 | 2 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by complexity,transfer | {'trending': np.int64(2)} | 2 | 2 |
| detectable real missed: changing (moderate, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'crisis': np.int64(2)} | 2 | 2 |
| detectable real missed: conditional (extremely_subtle, faint) | live: FAILED blocked by complexity,identity,out_of_sample,replication | {'choppy': np.int64(1), 'crisis': np.int64(1)} | 2 | 2 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration,replication | {'calm': np.int64(1), 'volatile': np.int64(1)} | 2 | 2 |
| detectable real missed: changing (moderate, obvious) | live: FAILED blocked by complexity,replication,transfer | {'trending': np.int64(1), 'choppy': np.int64(1)} | 2 | 2 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'volatile': np.int64(1), 'crisis': np.int64(1)} | 2 | 2 |
| detectable real missed: threshold (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(1), 'crisis': np.int64(1)} | 2 | 2 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by calibration,out_of_sample,replication | {'volatile': np.int64(1), 'trending': np.int64(1)} | 2 | 2 |
| detectable real missed: rare (obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(2)} | 2 | 2 |
| detectable real missed: lifecycle (subtle) | live: FAILED blocked by out_of_sample,replication,transfer | {'volatile': np.int64(1), 'crisis': np.int64(1)} | 2 | 2 |
| detectable real missed: linear (obvious, subtle) | live: FAILED blocked by complexity,transfer | {'volatile': np.int64(1), 'trending': np.int64(1)} | 2 | 2 |
| detectable real missed: lifecycle (moderate, obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(1), 'calm': np.int64(1)} | 2 | 2 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(2)} | 2 | 2 |
| detectable real missed: linear (moderate) | live: FAILED blocked by replication,transfer | {'choppy': np.int64(2)} | 2 | 2 |
| detectable real missed: linear (faint) | live: FAILED blocked by complexity,identity,out_of_sample,replication,transfer | {'volatile': np.int64(1), 'choppy': np.int64(1)} | 2 | 2 |
| detectable real missed: delayed (moderate) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'trending': np.int64(1), 'volatile': np.int64(1)} | 2 | 2 |
| detectable real missed: lifecycle (extremely_subtle, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1), 'volatile': np.int64(1)} | 2 | 2 |
| detectable real missed: threshold (moderate, obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'volatile': np.int64(1), 'crisis': np.int64(1)} | 2 | 2 |
| detectable real missed: linear (moderate) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(1), 'choppy': np.int64(1)} | 2 | 2 |
| noise promoted: leak | promoted | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (extremely_subtle) | live: FAILED blocked by calibration,complexity,identity,out_of_sample,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,out_of_sample | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by complexity,out_of_sample,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by complexity,out_of_sample | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (faint) | live: FAILED blocked by complexity,identity,out_of_sample,replication,transfer | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by calibration,out_of_sample,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (faint) | live: FAILED blocked by calibration,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (faint) | live: FAILED blocked by complexity,leakage,out_of_sample,replication,transfer | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (faint) | live: FAILED blocked by complexity,identity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,complexity | {'choppy': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (moderate) | live: FAILED blocked by calibration,complexity,identity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (moderate) | live: UNKNOWN blocked by leakage,replication | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (obvious) | live: FAILED blocked by leakage,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by out_of_sample | {'choppy': np.int64(1)} | 1 | 1 |
| detectable real missed: conditional (moderate) | live: UNKNOWN blocked by leakage | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: lifecycle (subtle) | live: FAILED blocked by calibration,complexity,leakage,out_of_sample,replication,transfer | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (moderate) | live: FAILED blocked by transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (subtle) | live: FAILED blocked by leakage,out_of_sample,replication | {'choppy': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (moderate) | live: FAILED blocked by calibration,complexity,replication | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (subtle) | live: FAILED blocked by complexity,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (moderate) | live: FAILED blocked by calibration,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: linear (subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: lifecycle (subtle) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: lifecycle (moderate) | live: FAILED blocked by complexity,replication,transfer | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: lifecycle (extremely_subtle) | live: FAILED blocked by out_of_sample | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,leakage,out_of_sample,replication | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by complexity,out_of_sample | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,out_of_sample,transfer | {'choppy': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (moderate) | live: FAILED blocked by complexity,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (moderate) | live: FAILED blocked by leakage,out_of_sample,replication,transfer | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: delayed (obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,identity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: regime (subtle) | live: FAILED blocked by complexity,out_of_sample | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: regime (subtle) | live: FAILED blocked by complexity,replication,transfer | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: regime (subtle) | live: FAILED blocked by calibration,out_of_sample | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: regime (faint) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: rare (subtle) | live: FAILED blocked by complexity,out_of_sample | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: rare (obvious) | live: FAILED blocked by replication,transfer | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: rare (moderate) | live: FAILED blocked by transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by complexity,transfer | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (subtle) | live: FAILED blocked by calibration,replication | {'calm': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (subtle) | live: FAILED blocked by calibration,complexity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (subtle) | live: FAILED blocked by calibration,complexity,identity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 1 |
| detectable real missed: regime (moderate) | live: UNKNOWN blocked by leakage | {'volatile': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (subtle) | retired: FAILED blocked by calibration,complexity,failure_behavior,identity,out_of_sample,replication,transfer | {'crisis': np.int64(1)} | 1 | 1 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by transfer | {'crisis': np.int64(1)} | 1 | 1 |
