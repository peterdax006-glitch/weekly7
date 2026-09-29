# Generalising learners - ls1

Validity: harness self-check VALID; learner self-check (planted worlds) VALID; real-window memoriser control seen. 

Pairs (learning window -> later transfer window): r01a->r06c, r06c->r03c, r12b->r12a, r08b->r11a, r11a->r03b, r10a->r07a, w02c->r05a, w02a->r02c, r13c->r01a, r10c->r13c, r05c->r11b, r03c->r09c

A learner PASSES only if the past-only transfer delta of its design metric has a 95% CI above zero. Eight metrics x six learners are looked at, so one lone marginal pass is suggestive, not proof.

## band_cfg

Design metric **in_band**: past-only transfer +0.01107 [+0.00160, +0.02204] over 12 pairs -> **PASS**; headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | +0.00083 [-0.00002, +0.00197] | +0.00096 [-0.00027, +0.00244] | +0.00013 [-0.00168, +0.00177] | +0.00000 |
| in_band | +0.01107 [+0.00160, +0.02204] | -0.00324 [-0.01442, +0.00472] | -0.01430 [-0.03332, -0.00006] | +0.00000 |
| worst5 | -0.00277 [-0.00676, +0.00069] | -0.00198 [-0.00638, +0.00105] | +0.00079 [-0.00567, +0.00677] | +0.00000 |
| max_dd | -0.01670 [-0.04045, +0.00150] | +0.00117 [-0.01144, +0.01494] | +0.01787 [-0.00632, +0.05129] | +0.00000 |
| pos_share | +0.00962 [-0.00629, +0.02861] | +0.01427 [+0.00160, +0.02857] | +0.00466 [-0.00962, +0.02174] | +0.00000 |
| dir_hit | -0.00177 [-0.03314, +0.02681] | -0.00673 [-0.02925, +0.02256] | -0.00495 [-0.05355, +0.05182] | +0.00000 |
| mover_hit | +0.00910 [+0.00285, +0.01654] | +0.01052 [+0.00046, +0.02263] | +0.00141 [-0.00660, +0.01029] | +0.00000 |
| year_return | +0.04210 [+0.00083, +0.09996] | +0.05434 [-0.01635, +0.13117] | +0.01224 [-0.08635, +0.10320] | +0.00000 |

State changed in 5 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI -0.000252..+0.0024 and transfer CI contain zero. In-band verdict: GENERALISING.

What it adopted:
- r01a -> r06c: nothing
- r06c -> r03c: nothing
- r12b -> r12a: round 1: adopt [vol_filter=False] delta +0.0373 over 16 years (13 positive, t 3.8, held-out +0.0309)
- r08b -> r11a: round 1: adopt [vol_filter=False] delta +0.0404 over 10 years (8 positive, t 3.2, held-out +0.0131)
- r11a -> r03b: round 1: adopt [vol_filter=False] delta +0.0354 over 13 years (10 positive, t 3.1, held-out +0.0191)
- r10a -> r07a: nothing
- w02c -> r05a: round 1: adopt [vol_filter=False] delta +0.0260 over 20 years (13 positive, t 2.7, held-out +0.0085)
- w02a -> r02c: nothing
- r13c -> r01a: nothing
- r10c -> r13c: nothing
- r05c -> r11b: nothing
- r03c -> r09c: round 1: adopt [liq_q=0.0] delta +0.0184 over 40 years (26 positive, t 2.6, held-out +0.0404)

## band_pool

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 12 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

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

State changed in 0 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r06c -> r03c: nothing
- r12b -> r12a: nothing
- r08b -> r11a: nothing
- r11a -> r03b: nothing
- r10a -> r07a: nothing
- w02c -> r05a: nothing
- w02a -> r02c: nothing
- r13c -> r01a: nothing
- r10c -> r13c: nothing
- r05c -> r11b: nothing
- r03c -> r09c: nothing

## regime_map

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 12 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

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

State changed in 0 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: regime_map: 34 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 5/9 years; t 1.20 < 2.5; not positive in the earlier years); bucket 1: kept global (positive in 6/14 years; t 1.84 < 2.5); bucket 2: kept 
- r06c -> r03c: regime_map: 37 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 7/12 years; t 1.67 < 2.5; not positive in the latest held-out years); bucket 1: kept global (positive in 9/17 years; t 0.83 < 2.5); bucket
- r12b -> r12a: regime_map: 15 past windows + current, buckets on m_vix; no market reading in the history: no map
- r08b -> r11a: regime_map: 9 past windows + current, buckets on m_vix; no market reading in the history: no map
- r11a -> r03b: regime_map: 12 past windows + current, buckets on m_vix; no market reading in the history: no map
- r10a -> r07a: regime_map: 38 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 6/12 years; t 1.24 < 2.5; not positive in the latest held-out years); bucket 1: kept global (positive in 6/19 years; t 0.55 < 2.5); bucket
- w02c -> r05a: regime_map: 19 past windows + current, buckets on m_vix; bucket 0: kept global (only 1 years (< 8); positive in 1/1 years; t 0.00 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 1 years (< 8); positi
- w02a -> r02c: regime_map: 25 past windows + current, buckets on m_vix; bucket 0: kept global (only 3 years (< 8); positive in 2/3 years; t 2.00 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 5 years (< 8); positi
- r13c -> r01a: regime_map: 32 past windows + current, buckets on m_vix; bucket 0: kept global (only 7 years (< 8); positive in 4/7 years; t 0.87 < 2.5; not positive in the earlier years); bucket 1: kept global (positive in 6/12 years; t 2.05 < 2
- r10c -> r13c: regime_map: 30 past windows + current, buckets on m_vix; bucket 0: kept global (only 7 years (< 8); positive in 4/7 years; t 1.90 < 2.5); bucket 1: kept global (positive in 5/9 years; t 0.70 < 2.5); bucket 2: kept global (positive
- r05c -> r11b: regime_map: 25 past windows + current, buckets on m_vix; bucket 0: kept global (only 3 years (< 8); positive in 2/3 years; t 2.00 < 2.5; not positive in the latest held-out years); bucket 1: kept global (only 5 years (< 8); positi
- r03c -> r09c: regime_map: 39 past windows + current, buckets on m_vix; bucket 0: kept global (positive in 6/13 years; t 1.18 < 2.5; not positive in the latest held-out years); bucket 1: kept global (positive in 8/19 years; t 2.04 < 2.5); bucket

## lessons

Design metric **mean_week**: past-only transfer +0.00000 [+0.00000, +0.00000] over 12 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

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

State changed in 0 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r06c -> r03c: nothing
- r12b -> r12a: nothing
- r08b -> r11a: nothing
- r11a -> r03b: nothing
- r10a -> r07a: nothing
- w02c -> r05a: nothing
- w02a -> r02c: nothing
- r13c -> r01a: nothing
- r10c -> r13c: nothing
- r05c -> r11b: nothing
- r03c -> r09c: nothing

## mover_use

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 12 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

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

State changed in 0 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r06c -> r03c: nothing
- r12b -> r12a: nothing
- r08b -> r11a: nothing
- r11a -> r03b: nothing
- r10a -> r07a: nothing
- w02c -> r05a: nothing
- w02a -> r02c: nothing
- r13c -> r01a: nothing
- r10c -> r13c: nothing
- r05c -> r11b: nothing
- r03c -> r09c: nothing

## chain

Design metric **in_band**: past-only transfer +0.00000 [+0.00000, +0.00000] over 12 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

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

State changed in 0 of 12 pairs. Same-year verdict (weekly mean): NO_EFFECT - effect CI +0..+0 and transfer CI contain zero. In-band verdict: NO_EFFECT.

What it adopted:
- r01a -> r06c: nothing
- r06c -> r03c: nothing
- r12b -> r12a: nothing
- r08b -> r11a: nothing
- r11a -> r03b: nothing
- r10a -> r07a: nothing
- w02c -> r05a: nothing
- w02a -> r02c: nothing
- r13c -> r01a: nothing
- r10c -> r13c: nothing
- r05c -> r11b: nothing
- r03c -> r09c: nothing

## Verdict

Passing learners: band_cfg

## What this does not prove

- Ledgers are close-to-close, held-nothing weekly picks; the harness plays the real adaptive replay, so a ledger gain can vanish there.
- Pairs share history (W is the latest window before B), so pair deltas are not independent and the CIs are optimistic.
- Snapshots were produced by a model trained before each window; learners cannot fix that model, only how its output is used.
- The price panel is survivor-only (memory: weekly7-price-panel-is-survivor-only): high-volatility pools look better than they were.
