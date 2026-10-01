# F28 x F27: gated result with and without the candidate forms (same seeds, final-look protocol) - IMPLEMENTED, NOT VALIDATED

Worlds in both runs: 14. A promoted composite that is no real pattern in any form counts as a false positive ('false_composite'); a composite that is a real pattern in its TRUE form is credited to that pattern.

| split | system | worlds | real found (detectable) | real found (not representable as a single) | real found (any status) | FP | FP / world | FP kinds | forms promoted | history-dependent forms promoted | median search size | s / world | refused |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| development | singles (F28 100-world run) | 14 | 71/246 = 28.9% [23.5%, 33.5%] | 0/181 | 71 | 0 | 0.00 | {} | 0 | 0 | 615 | 73 | 0 |
| development | singles + F27 forms | 14 | 73/246 = 29.7% [24.2%, 34.5%] | 34/181 | 107 | 6 | 0.43 | {'false_composite': 5, 'proxy': 1} | 64 | 17 | 7.302e+05 | 93 | 0 |

## With forms: found per kind (all statuses)

| split | kind | n | found |
|---|---|---|---|
| development | changing | 65 | 10 |
| development | conditional | 65 | 16 |
| development | delayed | 65 | 11 |
| development | interactive | 65 | 16 |
| development | lifecycle | 65 | 17 |
| development | linear | 65 | 11 |
| development | rare | 65 | 3 |
| development | regime | 65 | 8 |
| development | threshold | 65 | 4 |
| development | xor | 65 | 11 |
