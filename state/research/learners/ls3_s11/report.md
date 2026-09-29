# Generalising learners - ls3_s11

Validity: harness self-check VALID; learner self-check (planted worlds) VALID; real-window memoriser control seen. 

Pairs (learning window -> later transfer window): r08a->r09a, r03b->r01c, r11a->r12b, r10c->r13c, r09a->r03b, r04c->r10a, r06c->r03c, r02c->r10c, r12a->w03c, r07b->r05c, r04a->r05b, r02b->r02a, r03c->r09c, r12b->r06b, r13c->w04c, r12c->r07b

A learner PASSES only if the past-only transfer delta of its design metric has a 95% CI above zero AND no tier (in_band, worst5, max_dd) is significantly worse (C39). Eight metrics x six learners are looked at, so one lone marginal pass is suggestive, not proof.

## band_cfg

Design metric **in_band**: past-only transfer +0.02120 [+0.00118, +0.04833] over 16 pairs -> **no pass** (design metric up; tiers significantly worse: worst5, max_dd); headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | -0.00102 [-0.00221, +0.00022] | +0.00105 [+0.00010, +0.00211] | +0.00208 [+0.00062, +0.00364] | +0.00000 |
| in_band | +0.02120 [+0.00118, +0.04833] | +0.00825 [-0.00243, +0.02007] | -0.01295 [-0.03781, +0.00599] | +0.00000 |
| worst5 | -0.00394 [-0.00839, -0.00001] | -0.00310 [-0.00589, -0.00058] | +0.00084 [-0.00337, +0.00486] | +0.00000 |
| max_dd | -0.03154 [-0.06260, -0.00511] | -0.00876 [-0.02962, +0.01055] | +0.02278 [-0.00730, +0.06049] | +0.00000 |
| pos_share | -0.03554 [-0.05928, -0.01533] | +0.00844 [-0.01057, +0.02889] | +0.04397 [+0.01317, +0.07722] | +0.00000 |
| dir_hit | -0.04442 [-0.06922, -0.02125] | -0.01185 [-0.04313, +0.01889] | +0.03257 [+0.00475, +0.06692] | +0.00000 |
| mover_hit | +0.03097 [+0.01067, +0.05726] | +0.01090 [+0.00461, +0.01820] | -0.02007 [-0.04118, -0.00452] | +0.00000 |
| year_return | -0.07832 [-0.16106, +0.00609] | +0.04868 [-0.01411, +0.10785] | +0.12700 [+0.01833, +0.24319] | +0.00000 |

State changed in 8 of 16 pairs. Same-year verdict (weekly mean): MEMORISATION - same-year effect +0.001053 (CI +0.000137..+0.00208) far above transfer -0.001023; gap CI +0.000621..+0.00364. In-band verdict: GENERALISING.

What it adopted:
- r08a -> r09a: round 1: adopt [vol_filter=False] delta +0.0479 over 10 years (8 positive, t 3.6, held-out +0.0065)
- r03b -> r01c: round 1: adopt [vol_filter=False] delta +0.0420 over 16 years (13 positive, t 4.1, held-out +0.0348)
- r11a -> r12b: round 1: adopt [vol_filter=False] delta +0.0397 over 14 years (11 positive, t 3.5, held-out +0.0192)
- r10c -> r13c: nothing
- r09a -> r03b: round 1: adopt [vol_filter=False] delta +0.0457 over 13 years (11 positive, t 4.4, held-out +0.0338)
- r04c -> r10a: nothing
- r06c -> r03c: nothing
- r02c -> r10c: nothing
- r12a -> w03c: round 1: adopt [vol_filter=False] delta +0.0326 over 20 years (14 positive, t 3.4, held-out +0.0082)
- r07b -> r05c: nothing
- r04a -> r05b: round 1: adopt [vol_filter=False] delta +0.0510 over 9 years (7 positive, t 3.5, held-out +0.0128)
- r02b -> r02a: round 1: adopt [vol_filter=False] delta +0.0422 over 15 years (12 positive, t 3.9, held-out +0.0310)
- r03c -> r09c: nothing
- r12b -> r06b: round 1: adopt [vol_filter=False] delta +0.0407 over 17 years (14 positive, t 4.2, held-out +0.0322)
- r13c -> w04c: nothing
- r12c -> r07b: nothing

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
- r08a -> r09a: nothing
- r03b -> r01c: nothing
- r11a -> r12b: nothing
- r10c -> r13c: nothing
- r09a -> r03b: nothing
- r04c -> r10a: nothing
- r06c -> r03c: nothing
- r02c -> r10c: nothing
- r12a -> w03c: nothing
- r07b -> r05c: nothing
- r04a -> r05b: nothing
- r02b -> r02a: nothing
- r03c -> r09c: nothing
- r12b -> r06b: nothing
- r13c -> w04c: nothing
- r12c -> r07b: nothing

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
- r08a -> r09a: regime_map: 9 past windows + current, buckets on m_vix; no market reading in the history: no map
- r03b -> r01c: regime_map: 15 past windows + current, buckets on m_vix; no market reading in the history: no map
- r11a -> r12b: regime_map: 13 past windows + current, buckets on m_vix; no market reading in the history: no map
- r10c -> r13c: regime_map: 34 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 5/8 years; t 0.77 < 3.0203965604655325); bucket 1: kept global (positive in 7/13 years; t 1.09 < 2.6078567480996684); bucket 2: kept globa
- r09a -> r03b: regime_map: 12 past windows + current, buckets on m_vix; no market reading in the history: no map
- r04c -> r10a: regime_map: 41 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 7/11 years; t 0.68 < 2.8484178944866168; not positive in the earlier years); bucket 1: kept global (positive in 11/18 years; t 1.34 < 2.64
- r06c -> r03c: regime_map: 43 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 7/13 years; t 0.48 < 2.6219672888827246; not positive in the latest held-out years); bucket 1: kept global (positive in 9/20 years; t 2.15
- r02c -> r10c: regime_map: 32 past windows + current, buckets on m_vix; bucket 0: kept global (only 7 years (< 8); positive in 4/7 years; t 0.74 < 2.9312314970848807); bucket 1: kept global (positive in 5/11 years; t 1.75 < 2.577567186164762); b
- r12a -> w03c: regime_map: 19 past windows + current, buckets on m_vix; no market reading in the history: no map
- r07b -> r05c: regime_map: 26 past windows + current, buckets on m_vix; bucket 0: kept global (only 3 years (< 8); positive in 2/3 years; t 2.00 < 4.087258921340241; not positive in the latest held-out years); bucket 1: kept global (only 5 years
- r04a -> r05b: regime_map: 8 past windows + current, buckets on m_vix; no market reading in the history: no map
- r02b -> r02a: regime_map: 14 past windows + current, buckets on m_vix; no market reading in the history: no map
- r03c -> r09c: regime_map: 45 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 8/15 years; t 0.61 < 2.581778022270516; not positive in the latest held-out years); bucket 1: kept global (positive in 12/22 years; t 1.27
- r12b -> r06b: regime_map: 16 past windows + current, buckets on m_vix; no market reading in the history: no map
- r13c -> w04c: regime_map: 37 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 4/8 years; t 1.14 < 3.14266429379535); bucket 1: kept global (t 1.36 < 2.763544203092895); bucket 2: kept global (positive in 16/30 years;
- r12c -> r07b: regime_map: 23 past windows + current, buckets on m_vix; bucket 0: kept global (only 2 years (< 8); positive in 2/2 years; t 1.09 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 3 years (< 8); positi

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
- r08a -> r09a: nothing
- r03b -> r01c: nothing
- r11a -> r12b: nothing
- r10c -> r13c: nothing
- r09a -> r03b: nothing
- r04c -> r10a: nothing
- r06c -> r03c: nothing
- r02c -> r10c: nothing
- r12a -> w03c: nothing
- r07b -> r05c: nothing
- r04a -> r05b: nothing
- r02b -> r02a: nothing
- r03c -> r09c: nothing
- r12b -> r06b: nothing
- r13c -> w04c: nothing
- r12c -> r07b: nothing

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
- r08a -> r09a: nothing
- r03b -> r01c: nothing
- r11a -> r12b: nothing
- r10c -> r13c: nothing
- r09a -> r03b: nothing
- r04c -> r10a: nothing
- r06c -> r03c: nothing
- r02c -> r10c: nothing
- r12a -> w03c: nothing
- r07b -> r05c: nothing
- r04a -> r05b: nothing
- r02b -> r02a: nothing
- r03c -> r09c: nothing
- r12b -> r06b: nothing
- r13c -> w04c: nothing
- r12c -> r07b: nothing

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
- r08a -> r09a: nothing
- r03b -> r01c: nothing
- r11a -> r12b: nothing
- r10c -> r13c: nothing
- r09a -> r03b: nothing
- r04c -> r10a: nothing
- r06c -> r03c: nothing
- r02c -> r10c: nothing
- r12a -> w03c: nothing
- r07b -> r05c: nothing
- r04a -> r05b: nothing
- r02b -> r02a: nothing
- r03c -> r09c: nothing
- r12b -> r06b: nothing
- r13c -> w04c: nothing
- r12c -> r07b: nothing

## Verdict

Passing learners: none

## What this does not prove

- Ledgers are close-to-close, held-nothing weekly picks; the harness plays the real adaptive replay, so a ledger gain can vanish there.
- Pairs share history (W is the latest window before B), so pair deltas are not independent and the CIs are optimistic.
- Snapshots were produced by a model trained before each window; learners cannot fix that model, only how its output is used.
- The price panel is survivor-only (memory: weekly7-price-panel-is-survivor-only): high-volatility pools look better than they were.
