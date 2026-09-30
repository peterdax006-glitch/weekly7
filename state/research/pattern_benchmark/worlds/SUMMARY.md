# F19 real-vs-noise benchmark (C70-C74) - IMPLEMENTED, NOT VALIDATED

Worlds scored: 3 (2 development, 1 held-out). Cost: 1251 s per world (median 1263), 1.0 CPU-hours in total. Refused (seal check failed): 0.

## C72 score: right / wrong against the sealed answer key

- Real patterns found (detectable): 1 of 57; all real planted: 150 (53 undetectable in principle, 40 detectable only in their true form).
- Noise correctly rejected: 1670 of 1695; false positives: 25 in total, 8.33 per world (target 0; the gate's stated alpha allows 0.009 per world on the pure nulls it gated, 0.0010 if the search size were every feature scored).

## Totals (world-cluster bootstrap 95% intervals)

| set | worlds | real right (detectable promoted, right sign) | precision (right real / all promotions) | FDR | FNR (detectable) | real right / all real | undetectable in principle | not representable | noise correctly rejected | false positives / world | expected FP / world at stated alpha | expected FP / world at honest alpha | candidate recall real | candidate recall noise | partial credit (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 3 | 1.8% [0.0%, 4.8%] (n=57) | 3.8% (1/26) | 96.2% | 98.2% | 1/150 | 53 | 40 | 98.5% [97.4%, 99.3%] (n=1695) | 8.33 (max 15; worlds with 0: 0/3) | 0.009 | 0.0010 | 42.0% [40.0%, 46.0%] (n=150) | 13.9% [11.8%, 15.9%] (n=1695) | 0.007 |
| development | 2 | 2.6% [0.0%, 4.8%] (n=39) | 9.1% (1/11) | 90.9% | 97.4% | 1/100 | 30 | 31 | 99.1% [99.0%, 99.3%] (n=1115) | 5.00 (max 6; worlds with 0: 0/2) | 0.010 | 0.0011 | 43.0% [40.0%, 46.0%] (n=100) | 12.9% [11.8%, 14.0%] (n=1115) | 0.010 |
| heldout | 1 | 0.0% [0.0%, 0.0%] (n=18) | 0.0% (0/15) | 100.0% | 100.0% | 0/50 | 23 | 9 | 97.4% [97.4%, 97.4%] (n=580) | 15.00 (max 15; worlds with 0: 0/1) | 0.006 | 0.0007 | 40.0% [40.0%, 40.0%] (n=50) | 15.9% [15.9%, 15.9%] (n=580) | 0.000 |

## Per era (the era covering most of the final evaluation window)

| split | era | worlds | TP recall (detectable) | FP / world | noise rejected | detectable share |
|---|---|---|---|---|---|---|
| development | crisis | 1 | 4.8% [4.8%, 4.8%] (n=21) | 6.00 | 99.0% [99.0%, 99.0%] (n=580) | 42.0% |
| development | trending | 1 | 0.0% [0.0%, 0.0%] (n=18) | 4.00 | 99.3% [99.3%, 99.3%] (n=535) | 36.0% |
| heldout | trending | 1 | 0.0% [0.0%, 0.0%] (n=18) | 15.00 | 97.4% [97.4%, 97.4%] (n=580) | 36.0% |

## Real patterns per kind and strength band

| split | kind | band | n | detectable | not representable | undetectable | TP recall (detectable) | surfaced | mean oracle power | median delay looks |
|---|---|---|---|---|---|---|---|---|---|---|
| development | changing | extremely_subtle | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.19 |  |
| development | changing | faint | 2 | 0 | 0 | 2 | n/a | 50.0% | 0.19 |  |
| development | changing | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 50.0% | 1.00 |  |
| development | changing | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | changing | subtle | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.52 |  |
| development | conditional | extremely_subtle | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.17 |  |
| development | conditional | faint | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.40 |  |
| development | conditional | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | conditional | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | conditional | subtle | 2 | 1 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.81 |  |
| development | delayed | extremely_subtle | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.08 |  |
| development | delayed | faint | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.35 |  |
| development | delayed | moderate | 2 | 0 | 2 | 0 | n/a | 0.0% | 1.00 |  |
| development | delayed | obvious | 2 | 1 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 50.0% | 1.00 |  |
| development | delayed | subtle | 2 | 1 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 50.0% | 0.79 |  |
| development | interactive | extremely_subtle | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.27 |  |
| development | interactive | faint | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.67 |  |
| development | interactive | moderate | 2 | 0 | 2 | 0 | n/a | 0.0% | 1.00 |  |
| development | interactive | obvious | 2 | 0 | 2 | 0 | n/a | 0.0% | 1.00 |  |
| development | interactive | subtle | 2 | 0 | 2 | 0 | n/a | 0.0% | 0.96 |  |
| development | lifecycle | extremely_subtle | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.25 |  |
| development | lifecycle | faint | 2 | 0 | 0 | 2 | n/a | 50.0% | 0.12 |  |
| development | lifecycle | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | lifecycle | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | lifecycle | subtle | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.67 |  |
| development | linear | extremely_subtle | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.25 |  |
| development | linear | faint | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 50.0% | 0.40 |  |
| development | linear | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | linear | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | linear | subtle | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 0.90 |  |
| development | rare | extremely_subtle | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.33 |  |
| development | rare | faint | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.40 |  |
| development | rare | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | rare | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | rare | subtle | 2 | 0 | 2 | 0 | n/a | 0.0% | 0.69 |  |
| development | regime | extremely_subtle | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.08 |  |
| development | regime | faint | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 50.0% | 0.65 |  |
| development | regime | moderate | 2 | 1 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| development | regime | obvious | 2 | 1 | 1 | 0 | 100.0% [100.0%, 100.0%] (n=1) | 100.0% | 1.00 | 0 |
| development | regime | subtle | 2 | 1 | 0 | 1 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.62 |  |
| development | threshold | extremely_subtle | 2 | 0 | 1 | 1 | n/a | 0.0% | 0.40 |  |
| development | threshold | faint | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.27 |  |
| development | threshold | moderate | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 0.96 |  |
| development | threshold | obvious | 2 | 2 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=2) | 100.0% | 1.00 |  |
| development | threshold | subtle | 2 | 1 | 1 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.81 |  |
| development | xor | extremely_subtle | 2 | 0 | 0 | 2 | n/a | 0.0% | 0.00 |  |
| development | xor | faint | 2 | 0 | 2 | 0 | n/a | 0.0% | 0.90 |  |
| development | xor | moderate | 2 | 0 | 2 | 0 | n/a | 0.0% | 1.00 |  |
| development | xor | obvious | 2 | 0 | 2 | 0 | n/a | 0.0% | 1.00 |  |
| development | xor | subtle | 2 | 0 | 2 | 0 | n/a | 0.0% | 0.96 |  |
| heldout | changing | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | changing | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | changing | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | changing | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.96 |  |
| heldout | changing | subtle | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 0.0% | 0.54 |  |
| heldout | conditional | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.08 |  |
| heldout | conditional | faint | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.62 |  |
| heldout | conditional | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | conditional | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | conditional | subtle | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | delayed | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.00 |  |
| heldout | delayed | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.42 |  |
| heldout | delayed | moderate | 1 | 0 | 1 | 0 | n/a | 0.0% | 0.96 |  |
| heldout | delayed | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | delayed | subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.00 |  |
| heldout | interactive | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.08 |  |
| heldout | interactive | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.25 |  |
| heldout | interactive | moderate | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | interactive | obvious | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | interactive | subtle | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | lifecycle | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.29 |  |
| heldout | lifecycle | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.08 |  |
| heldout | lifecycle | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | lifecycle | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | lifecycle | subtle | 1 | 0 | 0 | 1 | n/a | 100.0% | 0.12 |  |
| heldout | linear | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | linear | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | linear | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | linear | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | linear | subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.21 |  |
| heldout | rare | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 100.0% | 0.04 |  |
| heldout | rare | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.08 |  |
| heldout | rare | moderate | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | rare | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.96 |  |
| heldout | rare | subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.21 |  |
| heldout | regime | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | regime | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.17 |  |
| heldout | regime | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | regime | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 1.00 |  |
| heldout | regime | subtle | 1 | 0 | 1 | 0 | n/a | 0.0% | 0.88 |  |
| heldout | threshold | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.00 |  |
| heldout | threshold | faint | 1 | 0 | 0 | 1 | n/a | 100.0% | 0.08 |  |
| heldout | threshold | moderate | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.83 |  |
| heldout | threshold | obvious | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.96 |  |
| heldout | threshold | subtle | 1 | 1 | 0 | 0 | 0.0% [0.0%, 0.0%] (n=1) | 100.0% | 0.71 |  |
| heldout | xor | extremely_subtle | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.04 |  |
| heldout | xor | faint | 1 | 0 | 0 | 1 | n/a | 0.0% | 0.29 |  |
| heldout | xor | moderate | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | obvious | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |
| heldout | xor | subtle | 1 | 0 | 1 | 0 | n/a | 0.0% | 1.00 |  |

## Noise per kind

| split | kind | n | promoted (FP) | FP rate | surfaced | mean best gate share | last verdicts |
|---|---|---|---|---|---|---|---|
| development | adversarial_near | 45 | 0 | 0.0% [0.0%, 0.0%] (n=45) | 6.7% | 0.53 | {'QUARANTINED': np.int64(3)} |
| development | autocorr_trap | 75 | 0 | 0.0% [0.0%, 0.0%] (n=75) | 13.3% | 0.55 | {'QUARANTINED': np.int64(6), 'FAILED': np.int64(4)} |
| development | base_null | 70 | 0 | 0.0% [0.0%, 0.0%] (n=70) | 4.3% | 0.58 | {'QUARANTINED': np.int64(2), 'FAILED': np.int64(1)} |
| development | coincidence | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 3.3% | 0.67 | {'QUARANTINED': np.int64(1)} |
| development | context | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 30.0% | 0.50 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| development | delayed_coincidence | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 3.3% | 0.42 | {'QUARANTINED': np.int64(1)} |
| development | early_decay | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 5.0% | 0.75 | {'FAILED': np.int64(1)} |
| development | fluke | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 5.0% | 0.67 | {'FAILED': np.int64(1)} |
| development | identity_null | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 20.0% | 0.62 | {'FAILED': np.int64(7), 'QUARANTINED': np.int64(5)} |
| development | interaction_component | 40 | 0 | 0.0% [0.0%, 0.0%] (n=40) | 7.5% | 0.58 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| development | interaction_decoy | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 0.0% | n/a | {} |
| development | leak | 30 | 10 | 33.3% [30.0%, 40.0%] (n=30) | 100.0% | 0.93 | {'FAILED': np.int64(20), 'PROMOTE': np.int64(10)} |
| development | mt_winner | 180 | 0 | 0.0% [0.0%, 0.0%] (n=180) | 13.9% | 0.56 | {'QUARANTINED': np.int64(21), 'FAILED': np.int64(4)} |
| development | near_pattern | 45 | 0 | 0.0% [0.0%, 0.0%] (n=45) | 13.3% | 0.61 | {'QUARANTINED': np.int64(4), 'FAILED': np.int64(2)} |
| development | null | 140 | 0 | 0.0% [0.0%, 0.0%] (n=140) | 7.1% | 0.58 | {'FAILED': np.int64(5), 'QUARANTINED': np.int64(5)} |
| development | proxy | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 46.7% | 0.75 | {'FAILED': np.int64(11), 'QUARANTINED': np.int64(3)} |
| development | regime_corr | 40 | 0 | 0.0% [0.0%, 0.0%] (n=40) | 0.0% | n/a | {} |
| development | reversal | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 0.0% | n/a | {} |
| development | sample_size | 40 | 0 | 0.0% [0.0%, 0.0%] (n=40) | 0.0% | n/a | {} |
| development | selection_bias | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 0.0% | n/a | {} |
| development | strong_nontransferable | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 16.7% | 0.55 | {'QUARANTINED': np.int64(3), 'FAILED': np.int64(2)} |
| development | survivor_bias | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 13.3% | 0.52 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(2)} |
| development | threshold_illusion | 60 | 0 | 0.0% [0.0%, 0.0%] (n=60) | 6.7% | 0.48 | {'FAILED': np.int64(3), 'QUARANTINED': np.int64(1)} |
| development | vol_corr | 40 | 0 | 0.0% [0.0%, 0.0%] (n=40) | 12.5% | 0.50 | {'QUARANTINED': np.int64(4), 'FAILED': np.int64(1)} |
| development | xor_trap | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 30.0% | 0.51 | {'QUARANTINED': np.int64(5), 'FAILED': np.int64(1)} |
| heldout | adversarial_near | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 13.3% | 0.58 | {'QUARANTINED': np.int64(4)} |
| heldout | autocorr_trap | 50 | 0 | 0.0% [0.0%, 0.0%] (n=50) | 24.0% | 0.61 | {'FAILED': np.int64(6), 'QUARANTINED': np.int64(6)} |
| heldout | base_null | 35 | 0 | 0.0% [0.0%, 0.0%] (n=35) | 22.9% | 0.51 | {'QUARANTINED': np.int64(5), 'FAILED': np.int64(3)} |
| heldout | coincidence | 15 | 0 | 0.0% [0.0%, 0.0%] (n=15) | 0.0% | n/a | {} |
| heldout | context | 5 | 0 | 0.0% [0.0%, 0.0%] (n=5) | 60.0% | 0.58 | {'FAILED': np.int64(3)} |
| heldout | delayed_coincidence | 15 | 0 | 0.0% [0.0%, 0.0%] (n=15) | 13.3% | 0.58 | {'FAILED': np.int64(1), 'QUARANTINED': np.int64(1)} |
| heldout | early_decay | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 10.0% | 0.67 | {'QUARANTINED': np.int64(1)} |
| heldout | fluke | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 0.0% | n/a | {} |
| heldout | identity_null | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 30.0% | 0.64 | {'FAILED': np.int64(5), 'QUARANTINED': np.int64(4)} |
| heldout | interaction_component | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 0.0% | n/a | {} |
| heldout | interaction_decoy | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 20.0% | 0.50 | {'QUARANTINED': np.int64(2)} |
| heldout | leak | 20 | 15 | 75.0% [75.0%, 75.0%] (n=20) | 100.0% | 0.98 | {'PROMOTE': np.int64(15), 'FAILED': np.int64(3), 'NEEDS_MORE_EVIDENCE': np.int64(2)} |
| heldout | mt_winner | 120 | 0 | 0.0% [0.0%, 0.0%] (n=120) | 5.0% | 0.53 | {'QUARANTINED': np.int64(4), 'FAILED': np.int64(2)} |
| heldout | near_pattern | 30 | 0 | 0.0% [0.0%, 0.0%] (n=30) | 0.0% | n/a | {} |
| heldout | null | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 20.0% | 0.46 | {'QUARANTINED': np.int64(1), 'FAILED': np.int64(1)} |
| heldout | proxy | 15 | 0 | 0.0% [0.0%, 0.0%] (n=15) | 13.3% | 0.75 | {'FAILED': np.int64(2)} |
| heldout | regime_corr | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 5.0% | 0.75 | {'FAILED': np.int64(1)} |
| heldout | reversal | 15 | 0 | 0.0% [0.0%, 0.0%] (n=15) | 0.0% | n/a | {} |
| heldout | sample_size | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 0.0% | n/a | {} |
| heldout | selection_bias | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 0.0% | n/a | {} |
| heldout | strong_nontransferable | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 10.0% | 0.54 | {'FAILED': np.int64(2)} |
| heldout | survivor_bias | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 25.0% | 0.60 | {'FAILED': np.int64(3), 'QUARANTINED': np.int64(2)} |
| heldout | threshold_illusion | 40 | 0 | 0.0% [0.0%, 0.0%] (n=40) | 7.5% | 0.53 | {'FAILED': np.int64(2), 'QUARANTINED': np.int64(1)} |
| heldout | vol_corr | 20 | 0 | 0.0% [0.0%, 0.0%] (n=20) | 40.0% | 0.56 | {'FAILED': np.int64(5), 'QUARANTINED': np.int64(3)} |
| heldout | xor_trap | 10 | 0 | 0.0% [0.0%, 0.0%] (n=10) | 20.0% | 0.54 | {'QUARANTINED': np.int64(2)} |

## C73: how close (confidence = 1 - the screen's BH q at the last look)

| split | label | n | brier | mean confidence | mean |effect error| | mean effect error | sign agreement | median final rank | just below the line (t in [1.5, 2)) | partial credit | near misses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| development | NOISE | 1115 | 0.1207 | 0.2211 | 0.01184 | 0.003789 | 0.7815 | 326 | 46 | 0 | {'false positive': np.int64(10)} |
| development | REAL | 100 | 0.523 | 0.4005 | 0.007549 | -0.002966 | 1 | 80 | 6 | 1 | {'surfaced, not promoted': np.int64(42), 'found': np.int64(1)} |
| heldout | NOISE | 580 | 0.1778 | 0.2987 | 0.01735 | 0.008684 | 0.6512 | 336 | 35 | 0 | {'false positive': np.int64(15)} |
| heldout | REAL | 50 | 0.5137 | 0.4063 | 0.006771 | -0.001227 | 1 | 139.5 | 2 | 0 | {'surfaced, not promoted': np.int64(20)} |

## C73: calibration of that confidence

| bin | n | share_real | mean_conf | promoted |
|---|---|---|---|---|
| [0.0, 0.5) | 1480 | 0.06284 | 0.131 | 0 |
| [0.5, 0.8) | 185 | 0.07027 | 0.6247 | 0 |
| [0.8, 0.9) | 39 | 0.2051 | 0.8458 | 0 |
| [0.9, 0.95) | 35 | 0.1714 | 0.9174 | 0 |
| [0.95, 0.99) | 17 | 0.3529 | 0.9752 | 0 |
| [0.99, 1.0) | 89 | 0.2697 | 1 | 0.2921 |

## C75 3E/3F: per difficulty tier (0 = NULL world)

| split | tier | worlds | recall (detectable) | precision | FDR | FP / world | noise rejected | worlds with nothing promoted |
|---|---|---|---|---|---|---|---|---|
| development | 3 | 1 | 0.0% [0.0%, 0.0%] (n=18) | 0.0% | 100.0% | 4.00 | 99.3% [99.3%, 99.3%] (n=535) | 0/1 |
| development | 4 | 1 | 4.8% [4.8%, 4.8%] (n=21) | 14.3% | 85.7% | 6.00 | 99.0% [99.0%, 99.0%] (n=580) | 0/1 |
| heldout | 6 | 1 | 0.0% [0.0%, 0.0%] (n=18) | 0.0% | 100.0% | 15.00 | 97.4% [97.4%, 97.4%] (n=580) | 0/1 |

## Which gate blocked what (last look)

| gate | real (detectable, not promoted) | noise (gated) |
|---|---|---|
| calibration | 100.0% of 50 | 86.9% of 236 |
| complexity | 64.0% of 50 | 73.3% of 236 |
| failure_behavior | 0.0% of 50 | 1.7% of 236 |
| identity | 12.0% of 50 | 43.6% of 236 |
| leakage | 12.0% of 50 | 43.6% of 236 |
| out_of_sample | 50.0% of 50 | 75.0% of 236 |
| replication | 100.0% of 50 | 82.6% of 236 |
| reproducibility | 0.0% of 50 | 6.4% of 236 |
| risk | 8.0% of 50 | 21.2% of 236 |
| transfer | 32.0% of 50 | 57.6% of 236 |

## C75 Phase 7: next unseen worlds after k worlds

| history_worlds | next_worlds | recall_next | fp_per_world_next | noise_rejection_next |
|---|---|---|---|---|
| 1 | 2 | 0.02564 | 5 | 0.991 |

## Ranked concrete failures (drive the next briefs)

| failure | where | eras | count | per_100_worlds |
|---|---|---|---|---|
| noise promoted: leak | promoted | {'trending': np.int64(19), 'crisis': np.int64(6)} | 25 | 833.3 |
| real only detectable in its true form (single-feature screen cannot represent it): interactive | candidate generation | {'trending': np.int64(7), 'crisis': np.int64(4)} | 11 | 366.7 |
| real only detectable in its true form (single-feature screen cannot represent it): xor | candidate generation | {'trending': np.int64(7), 'crisis': np.int64(4)} | 11 | 366.7 |
| real only detectable in its true form (single-feature screen cannot represent it): delayed | candidate generation | {'trending': np.int64(3), 'crisis': np.int64(3)} | 6 | 200 |
| real only detectable in its true form (single-feature screen cannot represent it): rare | candidate generation | {'trending': np.int64(3), 'crisis': np.int64(2)} | 5 | 166.7 |
| detectable real missed: conditional (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(2), 'crisis': np.int64(1)} | 3 | 100 |
| detectable real missed: linear (moderate, subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(2), 'trending': np.int64(1)} | 3 | 100 |
| detectable real missed: linear (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(2), 'crisis': np.int64(1)} | 3 | 100 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,replication | {'trending': np.int64(2), 'crisis': np.int64(1)} | 3 | 100 |
| detectable real missed: lifecycle (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(2), 'crisis': np.int64(1)} | 3 | 100 |
| real only detectable in its true form (single-feature screen cannot represent it): regime | candidate generation | {'trending': np.int64(3)} | 3 | 100 |
| detectable real missed: changing (moderate, subtle) | screen: never raised | {'trending': np.int64(2)} | 2 | 66.67 |
| detectable real missed: conditional (moderate) | live: FAILED blocked by calibration,complexity,replication | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 66.67 |
| detectable real missed: changing (obvious) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 66.67 |
| detectable real missed: changing (moderate, obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(1), 'crisis': np.int64(1)} | 2 | 66.67 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1), 'crisis': np.int64(1)} | 2 | 66.67 |
| detectable real missed: regime (faint, subtle) | live: FAILED blocked by calibration,complexity,replication | {'crisis': np.int64(2)} | 2 | 66.67 |
| real only detectable in its true form (single-feature screen cannot represent it): conditional | candidate generation | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 66.67 |
| detectable real missed: linear (extremely_subtle, faint) | screen: never raised | {'crisis': np.int64(1), 'trending': np.int64(1)} | 2 | 66.67 |
| real only detectable in its true form (single-feature screen cannot represent it): threshold | candidate generation | {'trending': np.int64(2)} | 2 | 66.67 |
| detectable real missed: conditional (subtle) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(2)} | 2 | 66.67 |
| detectable real missed: changing (moderate) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: delayed (obvious) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: delayed (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: delayed (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: conditional (obvious) | live: FAILED blocked by calibration,complexity,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: conditional (faint) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: changing (subtle) | live: FAILED blocked by calibration,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: linear (obvious) | live: FAILED blocked by calibration,complexity,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: lifecycle (extremely_subtle) | screen: never raised | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: regime (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: regime (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: rare (obvious) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,risk,transfer | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: rare (moderate) | live: FAILED blocked by calibration,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: rare (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: rare (moderate) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: linear (subtle) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: lifecycle (subtle) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,replication,risk | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (subtle) | quarantined: QUARANTINED blocked by calibration,complexity,identity,leakage,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (obvious) | live: FAILED blocked by calibration,complexity,out_of_sample,replication | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: regime (moderate) | live: FAILED blocked by calibration,replication | {'crisis': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (moderate) | live: FAILED blocked by calibration,out_of_sample,replication,transfer | {'trending': np.int64(1)} | 1 | 33.33 |
| detectable real missed: threshold (subtle) | screen: never raised | {'crisis': np.int64(1)} | 1 | 33.33 |
