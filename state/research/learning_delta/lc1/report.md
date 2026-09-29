# Repeated-run learning curve - lc1

Harness self-check (planted pattern learner rises, reset control flat, identity control flat under disguise and rising without it): **VALID**

Windows showing a rising same-year curve: 0 of 6. Every run used a fresh disguise (new order-preserving codes and a new date shift); memory was carried forward and never wiped, and released to each run only as evidence whose outcome had matured by that simulated moment (C58, checked independently after every run).
Multiple-testing bar: 4029 candidate patterns were tried across all runs and windows, so the slope must beat p < 1.2e-05 (0.05 / 4029).

| window | era | runs | first-5 mean_week | last-5 | diff [CI] | slope/run [CI] | plateau | reset ctrl slope | identity ctrl slope | perm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r02b | pre1997 | 20 | +0.01615 | +0.01494 | -0.00121 [-0.00200, -0.00006] | -0.00006 [-0.00014, +0.00003] | none | -0.00000 | +0.00000 | 0.12 | NO_CURVE |
| w07c | pre1997 | 20 | +0.00529 | +0.00466 | -0.00063 [-0.00169, +0.00022] | -0.00009 [-0.00021, -0.00000] | none | -0.00000 | +0.00000 | 0.098 | NO_CURVE |
| w03b | 1997-2000 | 20 | +0.01018 | +0.01159 | +0.00141 [-0.00071, +0.00361] | -0.00000 [-0.00020, +0.00015] | none | -0.00000 | n/a | 0.98 | NO_CURVE |
| r04b | 2001+ | 20 | +0.00428 | +0.00538 | +0.00110 [-0.00028, +0.00182] | +0.00009 [-0.00004, +0.00021] | none | -0.00000 | n/a | 0.097 | NO_CURVE |
| r01a | 2001+ | 20 | +0.00051 | -0.00028 | -0.00079 [-0.00177, +0.00015] | -0.00001 [-0.00009, +0.00011] | none | -0.00000 | n/a | 0.83 | NO_CURVE |
| r09c | 2001+ | 20 | +0.01513 | +0.01480 | -0.00034 [-0.00053, -0.00021] | -0.00010 [-0.00036, +0.00006] | run 16 | -0.00000 | n/a | 0.42 | NO_CURVE |

## Other metrics (slope per run, last-5 minus first-5)

| window | mean_week | in_band | worst5 | max_dd | pos_share | dir_hit | mover_hit | year_return |
|---|---|---|---|---|---|---|---|---|
| r02b | -0.00006 / -0.00121 | +0.00084 / +0.01538 | -0.00015 / -0.00348 | -0.00128 / -0.02269 | -0.00116 / -0.02308 | +0.00039 / +0.00517 | -0.00033 / -0.00720 | -0.00704 / -0.14282 |
| w07c | -0.00009 / -0.00063 | +0.00021 / +0.01132 | -0.00083 / -0.01228 | +0.00008 / +0.00183 | +0.00663 / +0.12830 | +0.00024 / +0.01291 | -0.00703 / -0.11415 | -0.00606 / -0.03517 |
| w03b | -0.00000 / +0.00141 | -0.00045 / +0.00000 | +0.00207 / +0.03309 | +0.00354 / +0.05566 | -0.00188 / -0.01923 | -0.00118 / -0.01998 | +0.00020 / +0.01314 | +0.00103 / +0.14274 |
| r04b | +0.00009 / +0.00110 | +0.00499 / +0.06923 | -0.00063 / -0.00930 | -0.00263 / -0.04112 | +0.00106 / +0.01154 | -0.00156 / -0.02188 | +0.00116 / +0.01642 | +0.00490 / +0.05402 |
| r01a | -0.00001 / -0.00079 | +0.00234 / +0.02692 | -0.00026 / -0.00314 | -0.00064 / -0.00996 | -0.00147 / -0.04231 | -0.00069 / -0.02054 | +0.00206 / +0.02493 | -0.00066 / -0.04281 |
| r09c | -0.00010 / -0.00034 | +0.00172 / +0.02642 | +0.00000 / +0.00000 | +0.00000 / +0.00000 | -0.00009 / -0.00377 | -0.00152 / -0.01235 | +0.00071 / +0.02222 | -0.01192 / -0.03228 |

## Do gains persist after other years?

| window | first-5 | last-5 | after 3 other years | retained | other years' mean_week |
|---|---|---|---|---|---|
| r02b | +0.01615 | +0.01494 | +0.01494 | 100% | -0.00084 |
| w07c | +0.00529 | +0.00466 | +0.00439 | 142% | +0.00445 |
| w03b | +0.01018 | +0.01159 | +0.01217 | 141% | +0.00469 |
| r04b | +0.00428 | +0.00538 | +0.00528 | 91% | +0.00683 |
| r01a | +0.00051 | -0.00028 | -0.00174 | 285% | +0.00908 |
| r09c | +0.01513 | +0.01480 | +0.01408 | 314% | +0.00118 |

## Verdict reasons

- r02b: NO_CURVE - slope -5.9e-05/run (CI -0.000143..+3.29e-05) does not clear zero
- w07c: NO_CURVE - slope -9.34e-05/run (CI -0.000213..-4.46e-06) does not clear zero
- w03b: NO_CURVE - slope -2.36e-06/run (CI -0.000196..+0.000154) does not clear zero
- r04b: NO_CURVE - slope +9.2e-05/run (CI -3.53e-05..+0.00021) does not clear zero
- r01a: NO_CURVE - slope -1.2e-05/run (CI -9.46e-05..+0.000114) does not clear zero
- r09c: NO_CURVE - slope -0.000105/run (CI -0.000364..+5.89e-05) does not clear zero

## What this does not prove

- The learners that exist in the adaptive replay are the long-term memory bank (and an optional basis search). The pattern bank and lessons are not wired into engine.adaptive.replay, so their contribution to a curve is not measured here.
- Runs in one chain share one year's data, so a rising curve is a same-year effect; C57 wants exactly that, but it says nothing on its own about other years (the interleaved runs and the transfer test in the pair harness speak to that).
- The memory bank matches on market-context fingerprints that no disguise changes; a curve that rises through them is learning the year's patterns, not its stock identities, which is why the identity-recall control (keyed on code and date) is the memorisation test.
- Run-to-run CIs treat runs as exchangeable although they are a chain: read them as a screen, not as exact coverage.
- The replay does not retrain the return model, so improvements that would need model retraining are outside these curves.
