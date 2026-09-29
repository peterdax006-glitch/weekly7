# Learning delta - ld2_basis

## HEADLINE: transfer delta (learning applied to unseen years, past-only)

**NO_TRANSFER** - transfer CI -0.000342..+0.000297 contains zero

Weekly-mean transfer delta -0.00002 [-0.00034, +0.00030] over 12 past-only pairs (0 of 12 transfer pairs dropped because the learning window came AFTER the transfer year); sign-flip p +1.00000.
Other metrics: in_band +0.00157; worst5 +0.00004; max_dd +0.00000; pos_share -0.00314; dir_hit -0.00146; mover_hit -0.00001; year_return +0.00075

## Second: same-year disguised delta (C54/C55) - read with the caveat below

Caveat: the same real year replayed under a new disguise is the same numbers with new labels. Any stored numeric memory recognises it (the paths are identical), so this delta can be memorisation, not learning; owner to rule (C54 vs C56). It is not the headline.

Same-year verdict on weekly mean: **MEMORISATION** - same-year effect +0.0009321 (CI +0.000205..+0.00177) far above transfer -1.516e-05; gap CI +0.000225..+0.00176
Verdict on share of weeks in the 5-10% band: **NO_EFFECT** - effect CI -0.011..+0.0299 and transfer CI contain zero


Harness self-check (planted generaliser / memoriser / non-learner / leak): **VALID**
- generaliser: GENERALISING; memoriser: MEMORISATION (same-year +0.0885, transfer +0.0000); non-learner: NO_EFFECT; leak probe caught: True

Pairs: 12. Learner: memory_bank+basis. S0 basis version 1. Transfer learned-from-the-future share: 0%.

## Learning delta per metric (Run 2 - Run 1; 95% bootstrap CI over windows)

| metric | run1 | run2 | same-year delta | noise ctrl | shuffle ctrl | learning effect | transfer | gap |
|---|---|---|---|---|---|---|---|---|
| mean_week | +0.00350 | +0.00444 | +0.00093 [+0.00023, +0.00177] | +0.00000 [+0.00000, +0.00000] | +0.00004 [-0.00033, +0.00044] | +0.00093 [+0.00021, +0.00177] | -0.00002 [-0.00034, +0.00030] | +0.00095 [+0.00023, +0.00176] |
| in_band | +0.15306 | +0.16092 | +0.00786 [-0.01101, +0.03145] | +0.00000 [+0.00000, +0.00000] | +0.01264 [+0.00314, +0.02522] | +0.00786 [-0.01101, +0.02987] | +0.00157 [+0.00000, +0.00472] | +0.00629 [-0.01572, +0.02987] |
| worst5 | -0.07006 | -0.06935 | +0.00071 [-0.00287, +0.00497] | +0.00000 [+0.00000, +0.00000] | -0.00090 [-0.00335, +0.00078] | +0.00071 [-0.00307, +0.00465] | +0.00004 [+0.00000, +0.00013] | +0.00067 [-0.00294, +0.00470] |
| max_dd | -0.30105 | -0.28811 | +0.01294 [-0.00338, +0.03617] | +0.00000 [+0.00000, +0.00000] | -0.00055 [-0.01132, +0.00953] | +0.01294 [-0.00364, +0.03664] | +0.00000 [+0.00000, +0.00000] | +0.01294 [-0.00320, +0.03743] |
| pos_share | +0.53961 | +0.54590 | +0.00629 [-0.00783, +0.01887] | +0.00000 [+0.00000, +0.00000] | +0.00314 [+0.00000, +0.00786] | +0.00629 [-0.00632, +0.01736] | -0.00314 [-0.00943, +0.00000] | +0.00943 [-0.00626, +0.02365] |
| dir_hit | +0.55637 | +0.56726 | +0.01089 [+0.00353, +0.01994] | +0.00000 [+0.00000, +0.00000] | +0.00364 [+0.00020, +0.00838] | +0.01089 [+0.00336, +0.02002] | -0.00146 [-0.00394, +0.00000] | +0.01235 [+0.00361, +0.02371] |
| mover_hit | +0.16712 | +0.16675 | -0.00037 [-0.00691, +0.00617] | +0.00000 [+0.00000, +0.00000] | +0.00067 [-0.00595, +0.00610] | -0.00037 [-0.00711, +0.00575] | -0.00001 [-0.00309, +0.00304] | -0.00035 [-0.00720, +0.00847] |
| year_return | +0.18171 | +0.23289 | +0.05118 [-0.00635, +0.10737] | +0.00000 [+0.00000, +0.00000] | +0.00143 [-0.02548, +0.02726] | +0.05118 [-0.00526, +0.10963] | +0.00075 [-0.02221, +0.02446] | +0.05043 [-0.00927, +0.10842] |

Learning effect = same-year delta minus the no-learning rerun delta (disguise noise removed). Gap = learning effect minus transfer gain; a gap above zero with transfer near zero is MEMORISATION.

## By era of the learning window

| era | n | mean_week delta | in_band delta | worst5 delta | transfer mean_week |
|---|---|---|---|---|---|
| 2001+ | 3 | +0.00114 | -0.01887 | +0.00641 | +0.00040 |
| pre1997 | 9 | +0.00086 | +0.01677 | -0.00118 | -0.00015 |

## By size of the starting memory bank

| S0 bank | n | same-year mean_week delta | transfer mean_week delta |
|---|---|---|---|
| with_bank | 12 | +0.00093 | -0.00002 |

Sign-flip p-values on mean_week: same-year +0.04297, learning effect +0.04297, transfer +1.00000. Tie-break luck floor (mean |random relabel - run 1|): +0.00035.

## Per pair (mean_week)

| window | era | bank | run1 | run2 | noise ctrl | transfer S0 | transfer S1 |
|---|---|---|---|---|---|---|---|
| r09a | pre1997 | 2949 | +0.01849 | +0.01711 | +0.01849 | -0.00130 | -0.00130 |
| r08c | pre1997 | 933 | +0.00562 | +0.00575 | +0.00562 | +0.01766 | +0.01766 |
| r12a | pre1997 | 2949 | +0.00270 | +0.00327 | +0.00270 | +0.00156 | +0.00156 |
| r03b | pre1997 | 2949 | +0.00490 | +0.00550 | +0.00490 | +0.00209 | +0.00209 |
| r02c | 2001+ | 9711 | +0.00542 | +0.00542 | +0.00542 | -0.00072 | -0.00072 |
| r12b | pre1997 | 2949 | +0.00498 | +0.00821 | +0.00498 | -0.00446 | -0.00446 |
| r10a | 2001+ | 11438 | +0.00097 | +0.00441 | +0.00097 | +0.01307 | +0.01426 |
| r03a | pre1997 | 2949 | +0.00215 | +0.00331 | +0.00215 | +0.00280 | +0.00280 |
| r02a | pre1997 | 2949 | +0.00280 | +0.00419 | +0.00280 | +0.00419 | +0.00419 |
| r04a | pre1997 | 2949 | +0.00192 | +0.00397 | +0.00192 | +0.00490 | +0.00353 |
| r10b | 2001+ | 9711 | -0.00546 | -0.00546 | -0.00546 | +0.00585 | +0.00585 |
| r05a | pre1997 | 6299 | -0.00243 | -0.00243 | -0.00243 | +0.00348 | +0.00348 |

## Memoriser control on real windows

Verdict MEMORISATION: same-year delta +0.02209, transfer +0.00000 over 4 pairs. This must read as large-and-near-zero for the harness to be able to see memorisation.

## Blindness (C55)

Hand-overs audited: 96; failures: 0. Every player hand-over passed the structural, date, name and same-year-fidelity checks; a failing hand-over would abort the pair.

## What this does not prove

- Runs replay archived weekly snapshots (engine.adaptive.replay): the return model is NOT retrained, so learning that lives in model fitting is not measured here. Only the adaptive layer's own learning (long-term memory bank, optional basis update) is.
- Snapshot features carry the market context (m_* columns), unchanged by any disguise. The memory bank matches on that context, so recognising a year through it is possible and is exactly what the transfer control and the planted memoriser exist to expose.
- Direction and mover hit rates use close-to-close proxies over the holding period, identical in both runs; they are not the fill-accurate figures.
- The learning windows are the revealed cycle years; their snapshots came from the model that was trained before them, so Run 1 is an honest blind play, but transfer windows learned from a LATER year are marked anachronistic and are a generalisation test, not a live-time claim.
- With few pairs the bootstrap intervals are wide; INCONCLUSIVE is the correct label until they narrow.
