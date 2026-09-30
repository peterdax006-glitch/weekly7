# F19 real-vs-noise benchmark (C70-C74) - IMPLEMENTED, NOT VALIDATED

Worlds scored: 20 (14 development, 6 held-out). Cost: 215 s per world (median 214), 1.2 CPU-hours in total. Refused (seal check failed): 0.

## C72 score: right / wrong against the sealed answer key

- Real patterns found (detectable): 3 of 346; all real planted: 900 (314 undetectable in principle, 240 detectable only in their true form).
- Noise correctly rejected: 10713 of 10825; false positives: 112 in total, 5.60 per world (target 0; the gate's stated alpha allows 0.003 per world on the pure nulls it gated, 0.0003 if the search size were every feature scored).

## Totals (world-cluster bootstrap 95% intervals)

| set | worlds | real right (detectable promoted, right sign) | precision (right real / all promotions) | FDR | FNR (detectable) | real right / all real | undetectable in principle | not representable | noise correctly rejected | false positives / world | expected FP / world at stated alpha | expected FP / world at honest alpha | candidate recall real | candidate recall noise | partial credit (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 20 | 0.9% [0.0%, 1.8%] (n=346) | 2.6% (3/115) | 97.4% | 99.1% | 3/900 | 314 | 240 | 99.0% [98.7%, 99.3%] (n=10825) | 5.60 (max 13; worlds with 0: 2/20) | 0.003 | 0.0003 | 33.1% [31.2%, 35.3%] (n=900) | 6.3% [5.5%, 7.1%] (n=10825) | 0.005 |
| development | 14 | 0.8% [0.0%, 1.9%] (n=246) | 2.5% (2/81) | 97.5% | 99.2% | 2/650 | 223 | 181 | 99.0% [98.6%, 99.3%] (n=7575) | 5.64 (max 11; worlds with 0: 2/14) | 0.004 | 0.0003 | 32.8% [30.2%, 35.7%] (n=650) | 6.3% [5.4%, 7.5%] (n=7575) | 0.005 |
| heldout | 6 | 1.0% [0.0%, 2.8%] (n=100) | 2.9% (1/34) | 97.1% | 99.0% | 1/250 | 91 | 59 | 99.0% [98.4%, 99.4%] (n=3250) | 5.50 (max 13; worlds with 0: 0/6) | 0.002 | 0.0002 | 34.0% [31.2%, 37.2%] (n=250) | 6.1% [4.9%, 7.7%] (n=3250) | 0.004 |

## Per era (the era covering most of the final evaluation window)

| split | era | worlds | TP recall (detectable) | FP / world | noise rejected | detectable share |
|---|---|---|---|---|---|---|
| development | calm | 2 | 0.0% [0.0%, 0.0%] (n=35) | 8.50 | 98.4% [98.3%, 98.5%] (n=1070) | 35.0% |
| development | choppy | 1 | 0.0% [0.0%, 0.0%] (n=19) | 4.00 | 99.3% [99.3%, 99.3%] (n=535) | 38.0% |
| development | crisis | 3 | 2.4% [0.0%, 4.8%] (n=42) | 6.00 | 98.9% [98.1%, 100.0%] (n=1645) | 42.0% |
| development | trending | 6 | 0.0% [0.0%, 0.0%] (n=106) | 5.17 | 99.0% [98.5%, 99.6%] (n=3255) | 35.3% |
| development | volatile | 2 | 2.3% [0.0%, 4.2%] (n=44) | 4.50 | 99.2% [98.7%, 99.6%] (n=1070) | 44.0% |
| heldout | choppy | 1 | 0.0% [0.0%, 0.0%] (n=21) | 4.00 | 99.3% [99.3%, 99.3%] (n=535) | 42.0% |
| heldout | crisis | 1 | 4.2% [4.2%, 4.2%] (n=24) | 1.00 | 99.8% [99.8%, 99.8%] (n=535) | 48.0% |
| heldout | trending | 4 | 0.0% [0.0%, 0.0%] (n=55) | 7.00 | 98.7% [98.1%, 99.2%] (n=2180) | 36.7% |

## Real patterns per kind and strength band

| split | kind | band | n | detectable | not representable | undetectable | TP recall (detectable) | surfaced | mean oracle power | median delay looks |
|---|---|---|---|---|---|---|---|---|---|---|
| development | changing | extremely_subtle | 13 | 2 | 0 | 11 | 0.0% [0.0%, 0.0%] (n=2) | 15.4% | 0.21 |  |
| development | changing | faint | 13 | 3 | 0 | 10 | 0.0% [0.0%, 0.0%] (n=3) | 15.4% | 0.26 |  |
| development | changing | moderate | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 92.3% | 1.00 |  |
| development | changing | obvious | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | changing | subtle | 13 | 11 | 0 | 2 | 0.0% [0.0%, 0.0%] (n=11) | 69.2% | 0.84 |  |
| development | conditional | extremely_subtle | 13 | 0 | 1 | 12 | n/a | 7.7% | 0.18 |  |
| development | conditional | faint | 13 | 5 | 4 | 4 | 0.0% [0.0%, 0.0%] (n=5) | 30.8% | 0.60 |  |
| development | conditional | moderate | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | conditional | obvious | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | conditional | subtle | 13 | 11 | 2 | 0 | 0.0% [0.0%, 0.0%] (n=11) | 53.8% | 0.95 |  |
| development | delayed | extremely_subtle | 13 | 0 | 1 | 12 | n/a | 0.0% | 0.14 |  |
| development | delayed | faint | 13 | 0 | 4 | 9 | n/a | 0.0% | 0.34 |  |
| development | delayed | moderate | 13 | 4 | 9 | 0 | 0.0% [0.0%, 0.0%] (n=4) | 15.4% | 1.00 |  |
| development | delayed | obvious | 13 | 7 | 6 | 0 | 0.0% [0.0%, 0.0%] (n=7) | 38.5% | 1.00 |  |
| development | delayed | subtle | 13 | 3 | 9 | 1 | 0.0% [0.0%, 0.0%] (n=3) | 15.4% | 0.79 |  |
| development | interactive | extremely_subtle | 13 | 0 | 3 | 10 | n/a | 0.0% | 0.30 |  |
| development | interactive | faint | 13 | 0 | 10 | 3 | n/a | 0.0% | 0.79 |  |
| development | interactive | moderate | 13 | 0 | 13 | 0 | n/a | 0.0% | 1.00 |  |
| development | interactive | obvious | 13 | 0 | 13 | 0 | n/a | 0.0% | 1.00 |  |
| development | interactive | subtle | 13 | 0 | 13 | 0 | n/a | 0.0% | 0.96 |  |
| development | lifecycle | extremely_subtle | 13 | 3 | 0 | 10 | 0.0% [0.0%, 0.0%] (n=3) | 0.0% | 0.26 |  |
| development | lifecycle | faint | 13 | 0 | 2 | 11 | n/a | 0.0% | 0.17 |  |
| development | lifecycle | moderate | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | lifecycle | obvious | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | lifecycle | subtle | 13 | 4 | 0 | 9 | 0.0% [0.0%, 0.0%] (n=4) | 23.1% | 0.44 |  |
| development | linear | extremely_subtle | 13 | 4 | 0 | 9 | 0.0% [0.0%, 0.0%] (n=4) | 7.7% | 0.27 |  |
| development | linear | faint | 13 | 8 | 0 | 5 | 0.0% [0.0%, 0.0%] (n=8) | 30.8% | 0.48 |  |
| development | linear | moderate | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | linear | obvious | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | linear | subtle | 13 | 12 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=12) | 76.9% | 0.90 |  |
| development | rare | extremely_subtle | 13 | 0 | 3 | 10 | n/a | 0.0% | 0.21 |  |
| development | rare | faint | 13 | 0 | 5 | 8 | n/a | 7.7% | 0.33 |  |
| development | rare | moderate | 13 | 5 | 8 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 38.5% | 0.98 |  |
| development | rare | obvious | 13 | 11 | 2 | 0 | 0.0% [0.0%, 0.0%] (n=11) | 69.2% | 1.00 |  |
| development | rare | subtle | 13 | 1 | 8 | 4 | 0.0% [0.0%, 0.0%] (n=1) | 15.4% | 0.66 |  |
| development | regime | extremely_subtle | 13 | 1 | 0 | 12 | 0.0% [0.0%, 0.0%] (n=1) | 7.7% | 0.12 |  |
| development | regime | faint | 13 | 4 | 1 | 8 | 0.0% [0.0%, 0.0%] (n=4) | 30.8% | 0.37 |  |
| development | regime | moderate | 13 | 5 | 1 | 7 | 20.0% [0.0%, 60.1%] (n=5) | 30.8% | 0.46 | 0 |
| development | regime | obvious | 13 | 5 | 1 | 7 | 20.0% [0.0%, 60.0%] (n=5) | 38.5% | 0.46 | 0 |
| development | regime | subtle | 13 | 4 | 1 | 8 | 0.0% [0.0%, 0.0%] (n=4) | 30.8% | 0.40 |  |
| development | threshold | extremely_subtle | 13 | 0 | 1 | 12 | n/a | 0.0% | 0.11 |  |
| development | threshold | faint | 13 | 1 | 3 | 9 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.36 |  |
| development | threshold | moderate | 13 | 12 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=12) | 61.5% | 0.99 |  |
| development | threshold | obvious | 13 | 13 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=13) | 100.0% | 1.00 |  |
| development | threshold | subtle | 13 | 3 | 2 | 8 | 0.0% [0.0%, 0.0%] (n=3) | 15.4% | 0.43 |  |
| development | xor | extremely_subtle | 13 | 0 | 4 | 9 | n/a | 0.0% | 0.31 |  |
| development | xor | faint | 13 | 0 | 12 | 1 | n/a | 0.0% | 0.87 |  |
| development | xor | moderate | 13 | 0 | 13 | 0 | n/a | 0.0% | 1.00 |  |
| development | xor | obvious | 13 | 0 | 13 | 0 | n/a | 0.0% | 1.00 |  |
| development | xor | subtle | 13 | 0 | 12 | 1 | n/a | 0.0% | 0.95 |  |
| heldout | changing | extremely_subtle | 5 | 1 | 0 | 4 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.18 |  |
| heldout | changing | faint | 5 | 0 | 0 | 5 | n/a | 0.0% | 0.07 |  |
| heldout | changing | moderate | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 1.00 |  |
| heldout | changing | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 0.99 |  |
| heldout | changing | subtle | 5 | 4 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=4) | 60.0% | 0.71 |  |
| heldout | conditional | extremely_subtle | 5 | 0 | 0 | 5 | n/a | 0.0% | 0.08 |  |
| heldout | conditional | faint | 5 | 4 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=4) | 60.0% | 0.74 |  |
| heldout | conditional | moderate | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 1.00 |  |
| heldout | conditional | obvious | 5 | 5 | 0 | 0 | 20.0% [0.0%, 60.0%] (n=5) | 100.0% | 1.00 | 0 |
| heldout | conditional | subtle | 5 | 4 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=4) | 20.0% | 0.94 |  |
| heldout | delayed | extremely_subtle | 5 | 0 | 1 | 4 | n/a | 0.0% | 0.24 |  |
| heldout | delayed | faint | 5 | 1 | 2 | 2 | 0.0% [0.0%, 0.0%] (n=1) | 20.0% | 0.56 |  |
| heldout | delayed | moderate | 5 | 4 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=4) | 60.0% | 0.99 |  |
| heldout | delayed | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 60.0% | 1.00 |  |
| heldout | delayed | subtle | 5 | 1 | 3 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 20.0% | 0.78 |  |
| heldout | interactive | extremely_subtle | 5 | 0 | 1 | 4 | n/a | 0.0% | 0.14 |  |
| heldout | interactive | faint | 5 | 0 | 3 | 2 | n/a | 0.0% | 0.58 |  |
| heldout | interactive | moderate | 5 | 0 | 5 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | interactive | obvious | 5 | 1 | 4 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 1.00 |  |
| heldout | interactive | subtle | 5 | 0 | 5 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | lifecycle | extremely_subtle | 5 | 1 | 0 | 4 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.20 |  |
| heldout | lifecycle | faint | 5 | 0 | 0 | 5 | n/a | 0.0% | 0.02 |  |
| heldout | lifecycle | moderate | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 1.00 |  |
| heldout | lifecycle | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 80.0% | 1.00 |  |
| heldout | lifecycle | subtle | 5 | 2 | 0 | 3 | 0.0% [0.0%, 0.0%] (n=2) | 20.0% | 0.37 |  |
| heldout | linear | extremely_subtle | 5 | 2 | 0 | 3 | 0.0% [0.0%, 0.0%] (n=2) | 0.0% | 0.30 |  |
| heldout | linear | faint | 5 | 2 | 0 | 3 | 0.0% [0.0%, 0.0%] (n=2) | 20.0% | 0.34 |  |
| heldout | linear | moderate | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 1.00 |  |
| heldout | linear | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 100.0% | 1.00 |  |
| heldout | linear | subtle | 5 | 4 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=4) | 80.0% | 0.81 |  |
| heldout | rare | extremely_subtle | 5 | 0 | 0 | 5 | n/a | 40.0% | 0.12 |  |
| heldout | rare | faint | 5 | 0 | 1 | 4 | n/a | 0.0% | 0.34 |  |
| heldout | rare | moderate | 5 | 1 | 4 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 20.0% | 1.00 |  |
| heldout | rare | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 80.0% | 0.99 |  |
| heldout | rare | subtle | 5 | 0 | 4 | 1 | n/a | 0.0% | 0.65 |  |
| heldout | regime | extremely_subtle | 5 | 0 | 0 | 5 | n/a | 20.0% | 0.01 |  |
| heldout | regime | faint | 5 | 1 | 0 | 4 | 0.0% [0.0%, 0.0%] (n=1) | 20.0% | 0.23 |  |
| heldout | regime | moderate | 5 | 2 | 0 | 3 | 0.0% [0.0%, 0.0%] (n=2) | 40.0% | 0.40 |  |
| heldout | regime | obvious | 5 | 2 | 0 | 3 | 0.0% [0.0%, 0.0%] (n=2) | 40.0% | 0.40 |  |
| heldout | regime | subtle | 5 | 1 | 1 | 3 | 0.0% [0.0%, 0.0%] (n=1) | 20.0% | 0.38 |  |
| heldout | threshold | extremely_subtle | 5 | 0 | 0 | 5 | n/a | 0.0% | 0.10 |  |
| heldout | threshold | faint | 5 | 0 | 1 | 4 | n/a | 20.0% | 0.22 |  |
| heldout | threshold | moderate | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 80.0% | 0.97 |  |
| heldout | threshold | obvious | 5 | 5 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 80.0% | 0.99 |  |
| heldout | threshold | subtle | 5 | 2 | 2 | 1 | 0.0% [0.0%, 0.0%] (n=2) | 40.0% | 0.63 |  |
| heldout | xor | extremely_subtle | 5 | 0 | 1 | 4 | n/a | 0.0% | 0.25 |  |
| heldout | xor | faint | 5 | 0 | 4 | 1 | n/a | 0.0% | 0.77 |  |
| heldout | xor | moderate | 5 | 0 | 5 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | obvious | 5 | 0 | 5 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | subtle | 5 | 0 | 5 | 0 | n/a | 0.0% | 1.00 |  |

## Noise per kind

| split | kind | n | promoted (FP) | FP rate | surfaced | mean best gate share | last verdicts |
|---|---|---|---|---|---|---|---|
| development | adversarial_near | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 0.0% | n/a | {} |
| development | autocorr_trap | 350 | 0 | 0.0% [0.0%, 0.0%] (n=350) | 8.6% | 0.60 | {'FAILED': np.int64(28), 'QUARANTINED': np.int64(2)} |
| development | base_null | 490 | 0 | 0.0% [0.0%, 0.0%] (n=490) | 2.9% | 0.51 | {'QUARANTINED': np.int64(8), 'FAILED': np.int64(6)} |
| development | coincidence | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 3.3% | 0.58 | {'FAILED': np.int64(7)} |
| development | context | 65 | 0 | 0.0% [0.0%, 0.0%] (n=65) | 7.7% | 0.55 | {'FAILED': np.int64(5)} |
| development | delayed_coincidence | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 6.2% | 0.59 | {'FAILED': np.int64(10), 'QUARANTINED': np.int64(3)} |
| development | early_decay | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 2.1% | 0.53 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| development | fluke | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 4.3% | 0.57 | {'FAILED': np.int64(4), 'QUARANTINED': np.int64(2)} |
| development | identity_null | 420 | 0 | 0.0% [0.0%, 0.0%] (n=420) | 15.5% | 0.62 | {'FAILED': np.int64(63), 'QUARANTINED': np.int64(2)} |
| development | interaction_component | 260 | 0 | 0.0% [0.0%, 0.0%] (n=260) | 1.9% | 0.60 | {'FAILED': np.int64(4), 'QUARANTINED': np.int64(1)} |
| development | interaction_decoy | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 4.3% | 0.53 | {'FAILED': np.int64(5), 'QUARANTINED': np.int64(1)} |
| development | leak | 142 | 78 | 54.9% [34.0%, 75.6%] (n=142) | 100.0% | 0.95 | {'PROMOTE': np.int64(78), 'FAILED': np.int64(57), 'NEEDS_MORE_EVIDENCE': np.int64(7)} |
| development | mt_winner | 840 | 0 | 0.0% [0.0%, 0.0%] (n=840) | 3.2% | 0.52 | {'FAILED': np.int64(16), 'QUARANTINED': np.int64(11)} |
| development | near_pattern | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 3.3% | 0.51 | {'FAILED': np.int64(4), 'QUARANTINED': np.int64(3)} |
| development | null | 1904 | 0 | 0.0% [0.0%, 0.0%] (n=1904) | 2.4% | 0.58 | {'FAILED': np.int64(37), 'QUARANTINED': np.int64(8)} |
| development | proxy | 210 | 1 | 0.5% [0.0%, 1.4%] (n=210) | 29.0% | 0.70 | {'FAILED': np.int64(58), 'QUARANTINED': np.int64(2), 'PROMOTE': np.int64(1)} |
| development | regime_corr | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 1.4% | 0.62 | {'FAILED': np.int64(4)} |
| development | reversal | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 1.0% | 0.62 | {'FAILED': np.int64(2)} |
| development | sample_size | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 0.0% | n/a | {} |
| development | selection_bias | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 0.7% | 0.50 | {'FAILED': np.int64(1)} |
| development | strong_nontransferable | 142 | 0 | 0.0% [0.0%, 0.0%] (n=142) | 8.5% | 0.58 | {'FAILED': np.int64(12)} |
| development | survivor_bias | 142 | 0 | 0.0% [0.0%, 0.0%] (n=142) | 3.5% | 0.50 | {'FAILED': np.int64(3), 'QUARANTINED': np.int64(2)} |
| development | threshold_illusion | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 2.1% | 0.49 | {'FAILED': np.int64(5), 'QUARANTINED': np.int64(1)} |
| development | vol_corr | 280 | 0 | 0.0% [0.0%, 0.0%] (n=280) | 5.0% | 0.49 | {'QUARANTINED': np.int64(8), 'FAILED': np.int64(6)} |
| development | xor_trap | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 3.6% | 0.62 | {'FAILED': np.int64(5)} |
| heldout | adversarial_near | 91 | 0 | 0.0% [0.0%, 0.0%] (n=91) | 1.1% | 0.50 | {'FAILED': np.int64(1)} |
| heldout | autocorr_trap | 149 | 0 | 0.0% [0.0%, 0.0%] (n=149) | 12.1% | 0.58 | {'FAILED': np.int64(14), 'QUARANTINED': np.int64(4)} |
| heldout | base_null | 210 | 0 | 0.0% [0.0%, 0.0%] (n=210) | 5.2% | 0.51 | {'FAILED': np.int64(6), 'QUARANTINED': np.int64(5)} |
| heldout | coincidence | 90 | 0 | 0.0% [0.0%, 0.0%] (n=90) | 3.3% | 0.61 | {'FAILED': np.int64(3)} |
| heldout | context | 25 | 0 | 0.0% [0.0%, 0.0%] (n=25) | 28.0% | 0.56 | {'FAILED': np.int64(7)} |
| heldout | delayed_coincidence | 90 | 0 | 0.0% [0.0%, 0.0%] (n=90) | 3.3% | 0.47 | {'QUARANTINED': np.int64(2), 'FAILED': np.int64(1)} |
| heldout | early_decay | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 0.0% | n/a | {} |
| heldout | fluke | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 0.0% | n/a | {} |
| heldout | identity_null | 180 | 0 | 0.0% [0.0%, 0.0%] (n=180) | 15.0% | 0.60 | {'FAILED': np.int64(25), 'QUARANTINED': np.int64(2)} |
| heldout | interaction_component | 100 | 0 | 0.0% [0.0%, 0.0%] (n=100) | 3.0% | 0.61 | {'FAILED': np.int64(3)} |
| heldout | interaction_decoy | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 5.0% | 0.53 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| heldout | leak | 60 | 33 | 55.0% [38.2%, 68.9%] (n=60) | 100.0% | 0.96 | {'PROMOTE': np.int64(33), 'FAILED': np.int64(25), 'NEEDS_MORE_EVIDENCE': np.int64(2)} |
| heldout | mt_winner | 360 | 0 | 0.0% [0.0%, 0.0%] (n=360) | 1.4% | 0.48 | {'QUARANTINED': np.int64(3), 'FAILED': np.int64(2)} |
| heldout | near_pattern | 91 | 0 | 0.0% [0.0%, 0.0%] (n=91) | 2.2% | 0.58 | {'FAILED': np.int64(2)} |
| heldout | null | 824 | 0 | 0.0% [0.0%, 0.0%] (n=824) | 0.7% | 0.61 | {'FAILED': np.int64(6)} |
| heldout | proxy | 90 | 0 | 0.0% [0.0%, 0.0%] (n=90) | 25.6% | 0.70 | {'FAILED': np.int64(20), 'QUARANTINED': np.int64(3)} |
| heldout | regime_corr | 120 | 0 | 0.0% [0.0%, 0.0%] (n=120) | 2.5% | 0.64 | {'FAILED': np.int64(3)} |
| heldout | reversal | 90 | 0 | 0.0% [0.0%, 0.0%] (n=90) | 2.2% | 0.67 | {'FAILED': np.int64(2)} |
| heldout | sample_size | 120 | 0 | 0.0% [0.0%, 0.0%] (n=120) | 0.0% | n/a | {} |
| heldout | selection_bias | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 5.0% | 0.53 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| heldout | strong_nontransferable | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 6.7% | 0.56 | {'FAILED': np.int64(4)} |
| heldout | survivor_bias | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 8.3% | 0.58 | {'FAILED': np.int64(4), 'QUARANTINED': np.int64(1)} |
| heldout | threshold_illusion | 120 | 0 | 0.0% [0.0%, 0.0%] (n=120) | 1.7% | 0.50 | {'FAILED': np.int64(2)} |
| heldout | vol_corr | 120 | 0 | 0.0% [0.0%, 0.0%] (n=120) | 5.0% | 0.60 | {'FAILED': np.int64(6)} |
| heldout | xor_trap | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 5.0% | 0.58 | {'FAILED': np.int64(3)} |

## C73: how close (confidence = 1 - the screen's BH q at the last look)

| split | label | n | brier | mean confidence | mean |effect error| | mean effect error | sign agreement | median final rank | just below the line (t in [1.5, 2)) | partial credit | near misses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| development | NOISE | 7575 | 0.1097 | 0.2107 | 0.01152 | 0.003145 | 0.7704 | 319 | 348 | 0 | {'false positive': np.int64(78), 'false positive (a proxy: informative but adds nothing)': np.int64(1)} |
| development | REAL | 650 | 0.5136 | 0.4116 | 0.006639 | -0.00172 | 1 | 72.5 | 55 | 3.5 | {'surfaced, not promoted': np.int64(206), 'one gate short at the last look': np.int64(4), 'found': np.int64(2), 'a proxy of it was promoted instead': np.int64(1)} |
| heldout | NOISE | 3250 | 0.1015 | 0.196 | 0.0126 | 0.003251 | 0.75 | 319 | 147 | 0 | {'false positive': np.int64(33)} |
| heldout | REAL | 250 | 0.5324 | 0.4018 | 0.006403 | -0.001914 | 1 | 71 | 13 | 1 | {'surfaced, not promoted': np.int64(84), 'found': np.int64(1)} |

## C73: calibration of that confidence

| bin | n | share_real | mean_conf | promoted |
|---|---|---|---|---|
| [0.0, 0.5) | 9922 | 0.05553 | 0.12 | 0 |
| [0.5, 0.8) | 939 | 0.08626 | 0.626 | 0 |
| [0.8, 0.9) | 195 | 0.1692 | 0.8507 | 0 |
| [0.9, 0.95) | 117 | 0.2564 | 0.9223 | 0 |
| [0.95, 0.99) | 94 | 0.3511 | 0.9674 | 0 |
| [0.99, 1.0) | 458 | 0.3755 | 1 | 0.2511 |

## C75 3E/3F: per difficulty tier (0 = NULL world)

| split | tier | worlds | recall (detectable) | precision | FDR | FP / world | noise rejected | worlds with nothing promoted |
|---|---|---|---|---|---|---|---|---|
| development | 0 | 1 | n/a | 0.0% | 100.0% | 8.00 | 98.5% [98.5%, 98.5%] (n=530) | 0/1 |
| development | 1 | 2 | 0.0% [0.0%, 0.0%] (n=39) | 0.0% | 100.0% | 3.00 | 99.4% [99.3%, 99.6%] (n=1070) | 0/2 |
| development | 2 | 4 | 1.3% [0.0%, 3.4%] (n=76) | 4.2% | 95.8% | 5.75 | 98.9% [98.6%, 99.3%] (n=2140) | 0/4 |
| development | 3 | 5 | 1.1% [0.0%, 3.1%] (n=91) | 3.1% | 96.9% | 6.20 | 98.8% [98.2%, 99.6%] (n=2675) | 1/5 |
| development | 4 | 1 | 0.0% [0.0%, 0.0%] (n=21) | n/a | n/a | 0.00 | 100.0% [100.0%, 100.0%] (n=580) | 1/1 |
| development | 6 | 1 | 0.0% [0.0%, 0.0%] (n=19) | 0.0% | 100.0% | 11.00 | 98.1% [98.1%, 98.1%] (n=580) | 0/1 |
| heldout | 0 | 1 | n/a | 0.0% | 100.0% | 6.00 | 98.9% [98.9%, 98.9%] (n=530) | 0/1 |
| heldout | 1 | 2 | 2.3% [0.0%, 4.2%] (n=43) | 14.3% | 85.7% | 3.00 | 99.4% [99.1%, 99.8%] (n=1070) | 0/2 |
| heldout | 3 | 2 | 0.0% [0.0%, 0.0%] (n=39) | 0.0% | 100.0% | 4.00 | 99.3% [99.3%, 99.3%] (n=1070) | 0/2 |
| heldout | 6 | 1 | 0.0% [0.0%, 0.0%] (n=18) | 0.0% | 100.0% | 13.00 | 97.8% [97.8%, 97.8%] (n=580) | 0/1 |

## Which gate blocked what (last look)

| gate | real (detectable, not promoted) | noise (gated) |
|---|---|---|
| calibration | 96.0% of 277 | 73.9% of 677 |
| complexity | 53.1% of 277 | 62.2% of 677 |
| failure_behavior | 0.0% of 277 | 1.5% of 677 |
| identity | 1.8% of 277 | 11.2% of 677 |
| leakage | 1.8% of 277 | 11.2% of 677 |
| out_of_sample | 19.1% of 277 | 55.4% of 677 |
| replication | 91.3% of 277 | 71.0% of 677 |
| reproducibility | 0.0% of 277 | 2.8% of 677 |
| risk | 25.3% of 277 | 27.6% of 677 |
| transfer | 15.2% of 277 | 42.4% of 677 |

## C75 Phase 7: next unseen worlds after k worlds

| history_worlds | next_worlds | recall_next | fp_per_world_next | noise_rejection_next |
|---|---|---|---|---|
| 1 | 5 | 0 | 3.8 | 0.9931 |
| 5 | 5 | 0 | 6.2 | 0.9884 |
| 10 | 10 | 0.01667 | 5.3 | 0.9901 |

## Ranked concrete failures (drive the next briefs)

| failure | where | eras | count | per_100_worlds |
|---|---|---|---|---|
| noise promoted: leak | promoted | {'trending': np.int64(58), 'crisis': np.int64(19), 'calm': np.int64(17), 'volatile': np.int64(9), 'choppy': np.int64(8)} | 111 | 555 |
| real only detectable in its true form (single-feature screen cannot represent it): xor | candidate generation | {'trending': np.int64(37), 'crisis': np.int64(11), 'calm': np.int64(9), 'volatile': np.int64(9), 'choppy': np.int64(8)} | 74 | 370 |
| real only detectable in its true form (single-feature screen cannot represent it): interactive | candidate generation | {'trending': np.int64(34), 'crisis': np.int64(12), 'volatile': np.int64(10), 'choppy': np.int64(7), 'calm': np.int64(7)} | 70 | 350 |
| real only detectable in its true form (single-feature screen cannot represent it): delayed | candidate generation | {'trending': np.int64(18), 'crisis': np.int64(7), 'volatile': np.int64(6), 'calm': np.int64(4), 'choppy': np.int64(1)} | 36 | 180 |
| real only detectable in its true form (single-feature screen cannot represent it): rare | candidate generation | {'trending': np.int64(19), 'crisis': np.int64(7), 'volatile': np.int64(4), 'choppy': np.int64(3), 'calm': np.int64(2)} | 35 | 175 |
| detectable real missed: linear (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication | {'crisis': np.int64(5), 'trending': np.int64(5), 'calm': np.int64(4), 'choppy': np.int64(1), 'volatile': np.int64(1)} | 16 | 80 |
| detectable real missed: linear (faint, moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(10), 'volatile': np.int64(2), 'crisis': np.int64(1), 'choppy': np.int64(1), 'calm': np.int64(1)} | 15 | 75 |
| detectable real missed: linear (extremely_subtle, faint, subtle) | screen: never raised | {'trending': np.int64(8), 'crisis': np.int64(3), 'choppy': np.int64(2), 'calm': np.int64(1), 'volatile': np.int64(1)} | 15 | 75 |
| detectable real missed: conditional (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication | {'trending': np.int64(4), 'volatile': np.int64(4), 'calm': np.int64(3), 'choppy': np.int64(2), 'crisis': np.int64(1)} | 14 | 70 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(6), 'crisis': np.int64(3), 'calm': np.int64(2), 'choppy': np.int64(1), 'volatile': np.int64(1)} | 13 | 65 |
| detectable real missed: changing (moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(6), 'volatile': np.int64(3), 'choppy': np.int64(2), 'calm': np.int64(1), 'crisis': np.int64(1)} | 13 | 65 |
| detectable real missed: conditional (faint, subtle) | screen: never raised | {'trending': np.int64(7), 'choppy': np.int64(1), 'calm': np.int64(1), 'volatile': np.int64(1)} | 10 | 50 |
| detectable real missed: delayed (faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(4), 'choppy': np.int64(3), 'crisis': np.int64(3)} | 10 | 50 |
| real only detectable in its true form (single-feature screen cannot represent it): threshold | candidate generation | {'trending': np.int64(7), 'choppy': np.int64(2), 'calm': np.int64(1)} | 10 | 50 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication | {'trending': np.int64(5), 'volatile': np.int64(2), 'calm': np.int64(1), 'crisis': np.int64(1)} | 9 | 45 |
| detectable real missed: threshold (faint, moderate, obvious, subtle) | screen: never raised | {'trending': np.int64(3), 'crisis': np.int64(2), 'volatile': np.int64(2), 'calm': np.int64(1), 'choppy': np.int64(1)} | 9 | 45 |
| detectable real missed: changing (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication | {'trending': np.int64(5), 'calm': np.int64(2), 'crisis': np.int64(1), 'volatile': np.int64(1)} | 9 | 45 |
| real only detectable in its true form (single-feature screen cannot represent it): conditional | candidate generation | {'trending': np.int64(5), 'crisis': np.int64(1), 'calm': np.int64(1), 'volatile': np.int64(1)} | 8 | 40 |
| detectable real missed: lifecycle (extremely_subtle, obvious, subtle) | screen: never raised | {'trending': np.int64(3), 'crisis': np.int64(2), 'choppy': np.int64(1), 'calm': np.int64(1), 'volatile': np.int64(1)} | 8 | 40 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,replication,risk | {'trending': np.int64(3), 'choppy': np.int64(2), 'volatile': np.int64(2), 'calm': np.int64(1)} | 8 | 40 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(6), 'crisis': np.int64(2)} | 8 | 40 |
| detectable real missed: threshold (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(5), 'calm': np.int64(2)} | 7 | 35 |
| detectable real missed: changing (extremely_subtle, faint, moderate, subtle) | screen: never raised | {'trending': np.int64(3), 'calm': np.int64(2), 'choppy': np.int64(1), 'volatile': np.int64(1)} | 7 | 35 |
| detectable real missed: regime (faint, obvious, subtle) | live: FAILED blocked by calibration,complexity,replication | {'crisis': np.int64(4), 'trending': np.int64(2)} | 6 | 30 |
| detectable real missed: conditional (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(2), 'choppy': np.int64(1), 'calm': np.int64(1), 'trending': np.int64(1), 'volatile': np.int64(1)} | 6 | 30 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(5), 'crisis': np.int64(1)} | 6 | 30 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,risk | {'crisis': np.int64(2), 'choppy': np.int64(2), 'trending': np.int64(1)} | 5 | 25 |
| detectable real missed: delayed (moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(2), 'choppy': np.int64(1), 'calm': np.int64(1), 'volatile': np.int64(1)} | 5 | 25 |
| detectable real missed: changing (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(3), 'crisis': np.int64(1), 'choppy': np.int64(1)} | 5 | 25 |
| detectable real missed: changing (faint, moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,replication,risk | {'trending': np.int64(2), 'choppy': np.int64(1), 'crisis': np.int64(1), 'volatile': np.int64(1)} | 5 | 25 |
| detectable real missed: changing (moderate, obvious, subtle) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(2), 'calm': np.int64(1), 'volatile': np.int64(1), 'trending': np.int64(1)} | 5 | 25 |
| detectable real missed: conditional (faint, moderate) | live: FAILED blocked by calibration,complexity,replication,risk | {'trending': np.int64(3), 'crisis': np.int64(2)} | 5 | 25 |
| detectable real missed: linear (faint, moderate, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(3), 'crisis': np.int64(1), 'calm': np.int64(1)} | 5 | 25 |
| real only detectable in its true form (single-feature screen cannot represent it): regime | candidate generation | {'trending': np.int64(5)} | 5 | 25 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(2), 'choppy': np.int64(2), 'trending': np.int64(1)} | 5 | 25 |
| detectable real missed: rare (moderate, obvious) | screen: never raised | {'trending': np.int64(3), 'choppy': np.int64(1)} | 4 | 20 |
| detectable real missed: delayed (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(3), 'calm': np.int64(1)} | 4 | 20 |
| detectable real missed: conditional (obvious) | live: FAILED blocked by calibration | {'trending': np.int64(3), 'calm': np.int64(1)} | 4 | 20 |
| detectable real missed: lifecycle (obvious, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(1), 'calm': np.int64(1), 'volatile': np.int64(1), 'trending': np.int64(1)} | 4 | 20 |
| detectable real missed: conditional (faint, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(2), 'choppy': np.int64(1), 'crisis': np.int64(1)} | 4 | 20 |
| detectable real missed: linear (obvious) | live: FAILED blocked by calibration | {'choppy': np.int64(2), 'trending': np.int64(1), 'volatile': np.int64(1)} | 4 | 20 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(3), 'volatile': np.int64(1)} | 4 | 20 |
| detectable real missed: rare (moderate, obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'volatile': np.int64(2), 'crisis': np.int64(1), 'choppy': np.int64(1)} | 4 | 20 |
| detectable real missed: threshold (moderate, obvious, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(3), 'crisis': np.int64(1)} | 4 | 20 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration | {'trending': np.int64(2), 'calm': np.int64(1)} | 3 | 15 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,replication,risk | {'choppy': np.int64(1), 'calm': np.int64(1), 'crisis': np.int64(1)} | 3 | 15 |
| detectable real missed: regime (moderate, subtle) | live: FAILED blocked by calibration,replication | {'crisis': np.int64(3)} | 3 | 15 |
| detectable real missed: conditional (faint, moderate, obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(3)} | 3 | 15 |
| real only detectable in its true form (single-feature screen cannot represent it): lifecycle | candidate generation | {'crisis': np.int64(1), 'volatile': np.int64(1)} | 2 | 10 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,replication,transfer | {'trending': np.int64(1), 'choppy': np.int64(1)} | 2 | 10 |
| detectable real missed: regime (extremely_subtle, moderate) | screen: never raised | {'trending': np.int64(1), 'crisis': np.int64(1)} | 2 | 10 |
| detectable real missed: regime (obvious) | live: FAILED blocked by risk | {'crisis': np.int64(2)} | 2 | 10 |
| detectable real missed: linear (obvious) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(1), 'volatile': np.int64(1)} | 2 | 10 |
| detectable real missed: linear (moderate, obvious) | live: FAILED blocked by calibration,replication,risk | {'choppy': np.int64(1), 'trending': np.int64(1)} | 2 | 10 |
| detectable real missed: linear (moderate, subtle) | live: FAILED blocked by calibration,complexity,replication,risk | {'trending': np.int64(2)} | 2 | 10 |
| detectable real missed: conditional (obvious) | live: FAILED blocked by replication,risk | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 10 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,complexity,replication,risk | {'trending': np.int64(1), 'choppy': np.int64(1)} | 2 | 10 |
| detectable real missed: changing (faint, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(2)} | 2 | 10 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(2)} | 2 | 10 |
| detectable real missed: regime (faint, moderate) | live: FAILED blocked by calibration,complexity,replication,risk | {'volatile': np.int64(2)} | 2 | 10 |
| detectable real missed: linear (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'crisis': np.int64(1), 'volatile': np.int64(1)} | 2 | 10 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: conditional (faint) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication | {'choppy': np.int64(1)} | 1 | 5 |
| detectable real missed: delayed (moderate) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'choppy': np.int64(1)} | 1 | 5 |
| detectable real missed: changing (obvious) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,risk,transfer | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: changing (obvious) | live: NEEDS_MORE_EVIDENCE blocked by complexity | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration,complexity,replication,risk,transfer | {'choppy': np.int64(1)} | 1 | 5 |
| noise promoted: proxy | promoted | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: changing (extremely_subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: conditional (subtle) | live: FAILED blocked by calibration,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: lifecycle (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: lifecycle (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,risk | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: lifecycle (obvious) | live: FAILED blocked by calibration,complexity | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: interactive (obvious) | screen: never raised | {'choppy': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'calm': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (moderate) | live: FAILED blocked by calibration,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,risk | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (subtle) | live: FAILED blocked by calibration,out_of_sample,replication,risk,transfer | {'calm': np.int64(1)} | 1 | 5 |
| detectable real missed: linear (obvious) | live: FAILED blocked by complexity,risk | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: linear (faint) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: linear (moderate) | live: FAILED blocked by calibration,replication,transfer | {'choppy': np.int64(1)} | 1 | 5 |
| detectable real missed: lifecycle (obvious) | live: FAILED blocked by calibration,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: lifecycle (obvious) | live: FAILED blocked by calibration,risk | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: linear (subtle) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: linear (obvious) | live: FAILED blocked by calibration,risk | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,risk,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,risk | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,risk,transfer | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (faint) | live: FAILED blocked by calibration,complexity,replication,risk,transfer | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (subtle) | live: FAILED blocked by calibration,complexity,replication,transfer | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (obvious) | live: FAILED blocked by calibration | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (moderate) | live: FAILED blocked by replication,risk | {'crisis': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (moderate) | live: NEEDS_MORE_EVIDENCE blocked by replication | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: regime (subtle) | live: FAILED blocked by calibration,replication,risk | {'volatile': np.int64(1)} | 1 | 5 |
| detectable real missed: threshold (subtle) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 5 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,complexity,replication,risk | {'calm': np.int64(1)} | 1 | 5 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 5 |
