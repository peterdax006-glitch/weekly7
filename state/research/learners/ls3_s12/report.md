# Generalising learners - ls3_s12

Validity: harness self-check VALID; learner self-check (planted worlds) VALID; real-window memoriser control seen. 

Pairs (learning window -> later transfer window): r01a->r06c, r09a->r02b, r08b->r11a, r06c->r03c, w02a->r04b, r11a->r12b, r03c->r09c, r08a->r09a, r12a->w03c, r05c->r11b, w03c->r13a, r10c->r13c, r12b->r06b, w02c->r05a, r02c->r10c, r05b->r03b

A learner PASSES only if the past-only transfer delta of its design metric has a 95% CI above zero AND no tier (in_band, worst5, max_dd) is significantly worse (C39). Eight metrics x six learners are looked at, so one lone marginal pass is suggestive, not proof.

## band_cfg

Design metric **in_band**: past-only transfer +0.01771 [+0.00000, +0.04495] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: worst5, max_dd); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | -0.00006 [-0.00128, +0.00124] | +0.00092 [-0.00015, +0.00218] | +0.00098 [-0.00098, +0.00286] | +0.00000 |
| in_band | +0.01771 [+0.00000, +0.04495] | +0.00116 [-0.00601, +0.00710] | -0.01655 [-0.04012, +0.00000] | +0.00000 |
| worst5 | -0.00499 [-0.00894, -0.00134] | -0.00415 [-0.00771, -0.00093] | +0.00084 [-0.00428, +0.00531] | +0.00000 |
| max_dd | -0.02689 [-0.05903, -0.00198] | -0.01173 [-0.03072, +0.00568] | +0.01516 [-0.01152, +0.05125] | +0.00000 |
| pos_share | -0.01422 [-0.03721, +0.00726] | +0.00968 [-0.01297, +0.03474] | +0.02390 [-0.00236, +0.05631] | +0.00000 |
| dir_hit | -0.02364 [-0.05177, +0.00670] | -0.00765 [-0.03530, +0.02107] | +0.01599 [-0.02233, +0.05814] | +0.00000 |
| mover_hit | +0.02174 [+0.00632, +0.04416] | +0.01429 [+0.00651, +0.02312] | -0.00745 [-0.02565, +0.00536] | +0.00000 |
| year_return | -0.02485 [-0.10972, +0.05802] | +0.03587 [-0.03503, +0.10936] | +0.06072 [-0.05537, +0.19398] | +0.00000 |

State changed in 8 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI -0.000156..+0.00213 and transfer CI contain zero. In-band verdict: GENERALISING.

What it adopted:
- r01a -> r06c: nothing
- r09a -> r02b: round 1: adopt [vol_filter=False] delta +0.0457 over 13 years (11 positive, t 4.4, held-out +0.0338)
- r08b -> r11a: round 1: adopt [vol_filter=False] delta +0.0453 over 11 years (9 positive, t 3.7, held-out +0.0098)
- r06c -> r03c: nothing
- w02a -> r04b: nothing
- r11a -> r12b: round 1: adopt [vol_filter=False] delta +0.0397 over 14 years (11 positive, t 3.5, held-out +0.0192)
- r03c -> r09c: nothing
- r08a -> r09a: round 1: adopt [vol_filter=False] delta +0.0479 over 10 years (8 positive, t 3.6, held-out +0.0065)
- r12a -> w03c: round 1: adopt [vol_filter=False] delta +0.0326 over 20 years (14 positive, t 3.4, held-out +0.0082)
- r05c -> r11b: nothing
- w03c -> r13a: nothing
- r10c -> r13c: nothing
- r12b -> r06b: round 1: adopt [vol_filter=False] delta +0.0407 over 17 years (14 positive, t 4.2, held-out +0.0322)
- w02c -> r05a: round 1: adopt [vol_filter=False] delta +0.0293 over 21 years (14 positive, t 3.0, held-out +0.0085)
- r02c -> r10c: nothing
- r05b -> r03b: round 1: adopt [vol_filter=False] delta +0.0447 over 12 years (10 positive, t 3.9, held-out +0.0194)

## band_pool

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: none); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| in_band | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| worst5 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| max_dd | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| pos_share | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| dir_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| mover_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| year_return | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |

State changed in 0 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r09a -> r02b: nothing
- r08b -> r11a: nothing
- r06c -> r03c: nothing
- w02a -> r04b: nothing
- r11a -> r12b: nothing
- r03c -> r09c: nothing
- r08a -> r09a: nothing
- r12a -> w03c: nothing
- r05c -> r11b: nothing
- w03c -> r13a: nothing
- r10c -> r13c: nothing
- r12b -> r06b: nothing
- w02c -> r05a: nothing
- r02c -> r10c: nothing
- r05b -> r03b: nothing

## regime_map

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: none); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| in_band | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| worst5 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| max_dd | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| pos_share | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| dir_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| mover_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| year_return | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |

State changed in 0 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: regime_map: 40 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 7/11 years; t 1.31 < 2.8671155158417543); bucket 1: kept global (positive in 10/18 years; t 1.41 < 2.6425031778246066); bucket 2: kept glo
- r09a -> r02b: regime_map: 12 past windows + current, buckets on m_vix; no market reading in the history: no map
- r08b -> r11a: regime_map: 10 past windows + current, buckets on m_vix; no market reading in the history: no map
- r06c -> r03c: regime_map: 43 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 7/13 years; t 0.48 < 2.6219672888827246; not positive in the latest held-out years); bucket 1: kept global (positive in 9/20 years; t 2.15
- w02a -> r04b: regime_map: 29 past windows + current, buckets on m_vix; bucket 0: kept global (only 5 years (< 8); positive in 2/5 years; t 0.24 < 4.209791349758732; not positive in the latest held-out years); bucket 1: kept global (positive in 
- r11a -> r12b: regime_map: 13 past windows + current, buckets on m_vix; no market reading in the history: no map
- r03c -> r09c: regime_map: 45 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 8/15 years; t 0.61 < 2.581778022270516; not positive in the latest held-out years); bucket 1: kept global (positive in 12/22 years; t 1.27
- r08a -> r09a: regime_map: 9 past windows + current, buckets on m_vix; no market reading in the history: no map
- r12a -> w03c: regime_map: 19 past windows + current, buckets on m_vix; no market reading in the history: no map
- r05c -> r11b: regime_map: 28 past windows + current, buckets on m_vix; bucket 0: kept global (only 4 years (< 8); positive in 1/4 years; t 0.29 < 4.043160560499268; not positive in the earlier years); bucket 1: kept global (only 7 years (< 8); 
- w03c -> r13a: regime_map: 22 past windows + current, buckets on m_vix; bucket 0: kept global (only 2 years (< 8); positive in 2/2 years; t 1.76 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 2 years (< 8); positi
- r10c -> r13c: regime_map: 34 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 5/8 years; t 0.77 < 3.0203965604655325); bucket 1: kept global (positive in 7/13 years; t 1.09 < 2.6078567480996684); bucket 2: kept globa
- r12b -> r06b: regime_map: 16 past windows + current, buckets on m_vix; no market reading in the history: no map
- w02c -> r05a: regime_map: 20 past windows + current, buckets on m_vix; bucket 0: kept global (only 1 years (< 8); positive in 1/1 years; t 0.00 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 1 years (< 8); positi
- r02c -> r10c: regime_map: 32 past windows + current, buckets on m_vix; bucket 0: kept global (only 7 years (< 8); positive in 4/7 years; t 0.74 < 2.9312314970848807); bucket 1: kept global (positive in 5/11 years; t 1.75 < 2.577567186164762); b
- r05b -> r03b: regime_map: 11 past windows + current, buckets on m_vix; no market reading in the history: no map

## lessons

Design metric **mean_week**: past-only transfer +0.00000 [+0.00000, +0.00000] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: none); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| in_band | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| worst5 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| max_dd | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| pos_share | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| dir_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| mover_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| year_return | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |

State changed in 0 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r09a -> r02b: nothing
- r08b -> r11a: nothing
- r06c -> r03c: nothing
- w02a -> r04b: nothing
- r11a -> r12b: nothing
- r03c -> r09c: nothing
- r08a -> r09a: nothing
- r12a -> w03c: nothing
- r05c -> r11b: nothing
- w03c -> r13a: nothing
- r10c -> r13c: nothing
- r12b -> r06b: nothing
- w02c -> r05a: nothing
- r02c -> r10c: nothing
- r05b -> r03b: nothing

## mover_use

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: none); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| in_band | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| worst5 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| max_dd | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| pos_share | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| dir_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| mover_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| year_return | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |

State changed in 0 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r09a -> r02b: nothing
- r08b -> r11a: nothing
- r06c -> r03c: nothing
- w02a -> r04b: nothing
- r11a -> r12b: nothing
- r03c -> r09c: nothing
- r08a -> r09a: nothing
- r12a -> w03c: nothing
- r05c -> r11b: nothing
- w03c -> r13a: nothing
- r10c -> r13c: nothing
- r12b -> r06b: nothing
- w02c -> r05a: nothing
- r02c -> r10c: nothing
- r05b -> r03b: nothing

## chain

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 16 pairs -> **no pass** (design metric not shown up; tiers significantly worse: none); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| in_band | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| worst5 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| max_dd | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| pos_share | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| dir_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| mover_hit | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |
| year_return | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 |

State changed in 0 of 16 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r09a -> r02b: nothing
- r08b -> r11a: nothing
- r06c -> r03c: nothing
- w02a -> r04b: nothing
- r11a -> r12b: nothing
- r03c -> r09c: nothing
- r08a -> r09a: nothing
- r12a -> w03c: nothing
- r05c -> r11b: nothing
- w03c -> r13a: nothing
- r10c -> r13c: nothing
- r12b -> r06b: nothing
- w02c -> r05a: nothing
- r02c -> r10c: nothing
- r05b -> r03b: nothing

## Verdict

Passing learners: none

## What this does not prove

- Ledgers are close-to-close, held-nothing weekly picks; the harness plays the real adaptive replay, so a ledger gain can vanish there.
- Pairs share history (W is the latest window before B), so pair deltas are not independent and the CIs are optimistic.
- Snapshots were produced by a model trained before each window; learners cannot fix that model, only how its output is used.
- The price panel is survivor-only (memory: weekly7-price-panel-is-survivor-only): high-volatility pools look better than they were.
