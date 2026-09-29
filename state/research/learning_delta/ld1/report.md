# Learning delta - ld1

Verdict on weekly mean: **MEMORISATION** - same-year effect +0.002269 (CI +6.76e-05..+0.00544) far above transfer -0.0003211; gap CI +0.000124..+0.00585
Verdict on share of weeks in the 5-10% band: **HARMFUL** - learning lowered the metric: effect -0.0143, transfer +0.003175


Harness self-check (planted generaliser / memoriser / non-learner / leak): **VALID**
- generaliser: GENERALISING; memoriser: MEMORISATION (same-year +0.0878, transfer +0.0000); non-learner: NO_EFFECT; leak probe caught: True

Pairs: 12. Learner: memory_bank. S0 basis version 1. Transfer learned-from-the-future share: 8%.

## Learning delta per metric (Run 2 - Run 1; 95% bootstrap CI over windows)

| metric | run1 | run2 | same-year delta | noise ctrl | shuffle ctrl | learning effect | transfer | gap |
|---|---|---|---|---|---|---|---|---|
| mean_week | +0.00829 | +0.01056 | +0.00227 [+0.00007, +0.00547] | +0.00000 [+0.00000, +0.00000] | -0.00135 [-0.00378, +0.00007] | +0.00227 [+0.00007, +0.00544] | -0.00032 [-0.00090, +0.00022] | +0.00259 [+0.00012, +0.00585] |
| in_band | +0.20371 | +0.18940 | -0.01430 [-0.02854, -0.00314] | +0.00000 [+0.00000, +0.00000] | +0.00157 [-0.00943, +0.01415] | -0.01430 [-0.02866, -0.00314] | +0.00317 [-0.00324, +0.01116] | -0.01748 [-0.03653, -0.00469] |
| worst5 | -0.08016 | -0.07993 | +0.00023 [-0.00541, +0.00411] | +0.00000 [+0.00000, +0.00000] | -0.00273 [-0.00820, +0.00000] | +0.00023 [-0.00530, +0.00454] | +0.00018 [-0.00052, +0.00119] | +0.00005 [-0.00539, +0.00428] |
| max_dd | -0.36684 | -0.35367 | +0.01317 [-0.01226, +0.04181] | +0.00000 [+0.00000, +0.00000] | -0.00726 [-0.02031, -0.00000] | +0.01317 [-0.01298, +0.04252] | -0.00084 [-0.01195, +0.00859] | +0.01401 [-0.01086, +0.03981] |
| pos_share | +0.54046 | +0.54995 | +0.00949 [-0.00623, +0.02682] | +0.00000 [+0.00000, +0.00000] | -0.00626 [-0.01415, +0.00163] | +0.00949 [-0.00623, +0.02673] | +0.00327 [-0.00472, +0.01282] | +0.00623 [-0.00953, +0.02661] |
| dir_hit | +0.51376 | +0.52168 | +0.00792 [-0.00118, +0.02189] | +0.00000 [+0.00000, +0.00000] | -0.00519 [-0.01255, +0.00030] | +0.00792 [-0.00106, +0.02188] | -0.00182 [-0.00867, +0.00324] | +0.00974 [-0.00223, +0.02593] |
| mover_hit | +0.23343 | +0.21843 | -0.01501 [-0.03115, -0.00247] | +0.00000 [+0.00000, +0.00000] | -0.00335 [-0.00804, +0.00044] | -0.01501 [-0.03110, -0.00259] | -0.00348 [-0.01033, +0.00315] | -0.01153 [-0.02374, -0.00101] |
| year_return | +0.52507 | +0.63473 | +0.10966 [-0.00747, +0.22902] | +0.00000 [+0.00000, +0.00000] | -0.04264 [-0.10733, +0.00602] | +0.10966 [-0.00631, +0.24249] | -0.06714 [-0.21804, +0.02479] | +0.17680 [+0.00721, +0.39949] |

Learning effect = same-year delta minus the no-learning rerun delta (disguise noise removed). Gap = learning effect minus transfer gain; a gap above zero with transfer near zero is MEMORISATION.

## By era of the learning window

| era | n | mean_week delta | in_band delta | worst5 delta | transfer mean_week |
|---|---|---|---|---|---|
| 2001+ | 4 | +0.00524 | -0.01923 | +0.00242 | -0.00043 |
| pre1997 | 8 | +0.00078 | -0.01184 | -0.00086 | -0.00027 |

## By size of the starting memory bank

| S0 bank | n | same-year mean_week delta | transfer mean_week delta |
|---|---|---|---|
| empty_bank | 1 | -0.00065 | -0.00145 |
| with_bank | 11 | +0.00253 | -0.00022 |

Sign-flip p-values on mean_week: same-year +0.10156, learning effect +0.10156, transfer +0.34375. Tie-break luck floor (mean |random relabel - run 1|): +0.00154.

## Per pair (mean_week)

| window | era | bank | run1 | run2 | noise ctrl | transfer S0 | transfer S1 |
|---|---|---|---|---|---|---|---|
| r09c | 2001+ | 6696 | +0.00316 | +0.00316 | +0.00316 | -0.00583 | -0.00583 |
| r04a | pre1997 | 2016 | -0.00360 | +0.00113 | -0.00360 | +0.04332 | +0.04067 |
| r08b | pre1997 | 2016 | +0.02586 | +0.02586 | +0.02586 | +0.00822 | +0.00822 |
| r06b | pre1997 | 2016 | +0.02191 | +0.01931 | +0.02191 | -0.00648 | -0.00648 |
| r01a | 2001+ | 6696 | +0.00515 | +0.00890 | +0.00515 | +0.00316 | +0.00306 |
| r03a | pre1997 | 2016 | +0.01281 | +0.01308 | +0.01281 | -0.01134 | -0.01134 |
| r03b | pre1997 | 2016 | +0.01735 | +0.01924 | +0.01735 | +0.02191 | +0.02270 |
| r06c | 2001+ | 6696 | +0.00149 | +0.00149 | +0.00149 | -0.00144 | -0.00144 |
| r06a | pre1997 | 0 | +0.00210 | +0.00145 | +0.00210 | +0.00228 | +0.00084 |
| r02a | pre1997 | 2016 | +0.01569 | +0.01830 | +0.01569 | +0.01774 | +0.01774 |
| r10b | 2001+ | 4969 | +0.00447 | +0.02170 | +0.00447 | -0.01177 | -0.01340 |
| r07b | pre1997 | 3478 | -0.00691 | -0.00691 | -0.00691 | +0.00254 | +0.00372 |

## Memoriser control on real windows

Verdict MEMORISATION: same-year delta +0.07502, transfer +0.00000 over 4 pairs. This must read as large-and-near-zero for the harness to be able to see memorisation.

## Blindness (C55)

Hand-overs audited: 96; failures: 0. Every player hand-over passed the structural, date, name and same-year-fidelity checks; a failing hand-over would abort the pair.

## What this does not prove

- Runs replay archived weekly snapshots (engine.adaptive.replay): the return model is NOT retrained, so learning that lives in model fitting is not measured here. Only the adaptive layer's own learning (long-term memory bank, optional basis update) is.
- Snapshot features carry the market context (m_* columns), unchanged by any disguise. The memory bank matches on that context, so recognising a year through it is possible and is exactly what the transfer control and the planted memoriser exist to expose.
- Direction and mover hit rates use close-to-close proxies over the holding period, identical in both runs; they are not the fill-accurate figures.
- The learning windows are the revealed cycle years; their snapshots came from the model that was trained before them, so Run 1 is an honest blind play, but transfer windows learned from a LATER year are marked anachronistic and are a generalisation test, not a live-time claim.
- With few pairs the bootstrap intervals are wide; INCONCLUSIVE is the correct label until they narrow.
