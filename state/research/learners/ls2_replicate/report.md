# Generalising learners - ls2_replicate

Validity: harness self-check VALID; learner self-check (planted worlds) INVALID; real-window memoriser control seen. NO VERDICT BELOW MAY BE USED.

Pairs (learning window -> later transfer window): r04a->r05b, r13c->r13b, r12b->r06b, r11a->r03b, r07b->r05c, w04a->r01a, r03b->w02c, r02c->r10c, w02c->r12c, r13a->w02a, w02a->r04b, w03c->r13a, r09a->r02b, r10a->r07a, r04c->r10a, r10c->w04c

A learner PASSES only if the past-only transfer delta of its design metric has a 95% CI above zero. Eight metrics x six learners are looked at, so one lone marginal pass is suggestive, not proof.

## band_cfg

Design metric **in_band**: past-only transfer +0.01073 [-0.00007, +0.02504] over 16 pairs -> **no pass**; headline verdict on weekly mean: NO_TRANSFER

| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |
|---|---|---|---|---|
| mean_week | -0.00032 [-0.00133, +0.00069] | +0.00115 [+0.00023, +0.00220] | +0.00147 [+0.00014, +0.00317] | +0.00000 |
| in_band | +0.01073 [-0.00007, +0.02504] | +0.00825 [+0.00000, +0.01887] | -0.00247 [-0.02041, +0.01297] | +0.00000 |
| worst5 | -0.00367 [-0.00705, -0.00084] | -0.00200 [-0.00455, +0.00020] | +0.00167 [-0.00215, +0.00588] | +0.00000 |
| max_dd | -0.03180 [-0.06091, -0.00886] | -0.00610 [-0.02384, +0.00965] | +0.02571 [-0.00394, +0.06237] | +0.00000 |
| pos_share | -0.01794 [-0.04286, +0.00565] | +0.00354 [-0.01179, +0.02005] | +0.02148 [-0.01626, +0.05849] | +0.00000 |
| dir_hit | -0.02958 [-0.05147, -0.00960] | -0.01124 [-0.03801, +0.01427] | +0.01834 [-0.01365, +0.05624] | +0.00000 |
| mover_hit | +0.01390 [+0.00445, +0.02477] | +0.01019 [+0.00321, +0.01886] | -0.00371 [-0.00785, -0.00027] | +0.00000 |
| year_return | -0.03605 [-0.10466, +0.03500] | +0.05842 [-0.00001, +0.12295] | +0.09446 [+0.00328, +0.20162] | +0.00000 |

State changed in 6 of 16 pairs. Same-year verdict (weekly mean): MEMORISATION - same-year effect +0.001154 (CI +0.000244..+0.00222) far above transfer -0.0003168; gap CI +0.000141..+0.00317. In-band verdict: NO_EFFECT.

What it adopted:
- r04a -> r05b: round 1: adopt [vol_filter=False] delta +0.0510 over 9 years (7 positive, t 3.5, held-out +0.0128)
- r13c -> r13b: nothing
- r12b -> r06b: round 1: adopt [vol_filter=False] delta +0.0407 over 17 years (14 positive, t 4.2, held-out +0.0322)
- r11a -> r03b: round 1: adopt [vol_filter=False] delta +0.0397 over 14 years (11 positive, t 3.5, held-out +0.0192)
- r07b -> r05c: nothing
- w04a -> r01a: nothing
- r03b -> w02c: round 1: adopt [vol_filter=False] delta +0.0420 over 16 years (13 positive, t 4.1, held-out +0.0348)
- r02c -> r10c: nothing
- w02c -> r12c: round 1: adopt [vol_filter=False] delta +0.0293 over 21 years (14 positive, t 3.0, held-out +0.0085)
- r13a -> w02a: nothing
- w02a -> r04b: nothing
- w03c -> r13a: nothing
- r09a -> r02b: round 1: adopt [vol_filter=False] delta +0.0457 over 13 years (11 positive, t 4.4, held-out +0.0338)
- r10a -> r07a: nothing
- r04c -> r10a: nothing
- r10c -> w04c: nothing

## Verdict

Passing learners: none

## What this does not prove

- Ledgers are close-to-close, held-nothing weekly picks; the harness plays the real adaptive replay, so a ledger gain can vanish there.
- Pairs share history (W is the latest window before B), so pair deltas are not independent and the CIs are optimistic.
- Snapshots were produced by a model trained before each window; learners cannot fix that model, only how its output is used.
- The price panel is survivor-only (memory: weekly7-price-panel-is-survivor-only): high-volatility pools look better than they were.
