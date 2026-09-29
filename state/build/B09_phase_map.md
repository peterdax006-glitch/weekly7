# B09 phase map: BIBLE Phases 17, 19, 20 (lines 1142-1174, 1217-1292)

Every requirement -> implementation -> proving test. Files: `engine/timeline.py` (T), `engine/basis_search.py` (B),
`engine/objective.py` (O); tests `tests/test_timeline.py`, `test_basis_search.py`, `test_objective.py`,
`test_livesim_loop2.py`. Status: DONE = implemented and tested; PROXY = implemented, proven only on a proxy/synthetic
series (real-system evidence is negative or absent, see notes).

## Phase 17 - Timeline dial

| Requirement | Implementation | Test | Status |
|---|---|---|---|
| Implement the designed pacing system | T.step (controller), T.simulate, T.train_dial | test_calm_raises_and_overshoot_lowers_aggressiveness, test_rate_limit_and_deadband | DONE |
| Input: regime | T.regime_from_market (m_vix_term -> stress), T.step(regime=) | test_regime_from_market_dict_series_and_missing, test_forecast_and_regime_move_the_dial | DONE |
| Input: analog forecast | T.forecast_from_analogs (Analogs.find -> weekly mean/sd, uniqueness widens sd), T.expected_abs_move | test_forecast_from_analogs_scaling_uniqueness_and_missing, test_forecast_and_regime_move_the_dial | DONE (beta = portfolio/market swing ratio is an assumption) |
| Input: year-to-date progress | T.pacing_state (cum vs 1.07^n path, gap, weeks_behind, required, projected), used in T.pace_terms | test_pacing_state_on_the_path_behind_and_ahead, test_pacing_state_edges, test_pacing_feeds_the_dial_and_losing_is_defended_not_chased | DONE |
| Input: weekly performance trajectory | T.pace_terms (EWMA of |week|, over-band share, trailing drawdown) | test_calm_raises_and_overshoot_lowers_aggressiveness, test_over_limit_forces_derisking_even_with_bullish_forecast | DONE |
| Output: exposure | T.DialOutput.exposure (bounds + brake) | test_outputs_inside_bounds_for_extreme_inputs, test_brake_exposure_never_exceeds_normal_exposure | DONE |
| Output: k | T.DialOutput.k, T.cfg_overrides | test_outputs_inside_bounds_for_extreme_inputs, test_cfg_overrides_keys, test_simulate_proxy_leverage_scales_with_k | DONE |
| Output: pool_q | T.DialOutput.pool_q (direction assumed: higher = bolder) | test_outputs_inside_bounds_for_extreme_inputs | DONE (direction UNPROVEN) |
| Output: brake | T.step brake trip/release/hysteresis | test_brake_trips_caps_and_releases | DONE |
| Dial may not use calendar information | T._reject_dates (datetime, Timestamp, date-indexed data, dicts holding dates) | test_calendar_information_is_rejected, test_decisions_do_not_depend_on_absolute_time_or_future | DONE |
| Trained on earlier windows | T.train_dial (default params always a contender, seeded) | test_train_dial_is_deterministic_and_requires_data | DONE |
| Tested on later unseen windows | T.promotion_gate (raises if train follows test), T.gate_rows | test_gate_fails_closed_and_checks_order, test_gate_rows_matches_promotion_gate_and_rejects_unpaired | DONE |
| Not allowed into champion until proves: improvement | gate_rows check "improvement" (firewall + paired bootstrap on the deciding tier) | test_planted_effect_dial_passes_gate_on_unseen_windows, test_gate_rejects_dial_when_no_effect_to_exploit | DONE |
| ... no unacceptable risk deterioration | gate_rows check "risk" (tier-2 risk, max DD, catastrophic rate) | test_gate_catches_a_risk_deteriorating_dial | DONE |
| ... no overfitting | gate_rows check "no_overfit" (train gain must carry to test) | test_planted_effect_dial_passes_gate_on_unseen_windows (train gain > 0 carries) | DONE |
| ... stability across eras | gate_rows check "stable_eras" (era thirds) | test_gate_flags_a_dial_that_only_works_in_one_era | DONE |
| Champion admission actually enforced | T.DialAdmission (needs all five checks; hash-chained log; revoke) - was MISSING | test_admission_refuses_unproven_and_admits_proven, test_admission_persists_verifies_and_detects_tampering | DONE (new) |
| Dial cannot win by gaming the objective | gate_rows check "not_gaming" (O.gaming_flags) - was MISSING | test_gate_flags_a_dial_that_wins_by_hiding_in_cash | DONE (new) |
| Real-system evidence | scripts/dial_offline.py (adaptive.replay via DialSession, selfcheck) | run result state/research/timeline_basis/dial_offline_seed0.json | PROXY+REAL: gate FAILS on real replay (in-band 22% -> 14%); dial is NOT admitted |

## Phase 19 - Train the training basis

| Requirement | Implementation | Test | Status |
|---|---|---|---|
| 24 random starting configurations | B.SearchConfig.n_start=24, B.sample_candidate (validated draws, dedup) | test_default_funnel_is_24_starts_10_screen_3_finalists, test_duplicate_candidates_are_evaluated_once, test_invalid_candidates_never_reach_evaluation | DONE (new explicit test) |
| 10 random archived screening windows | B.SearchConfig.n_screen=10; BasisSearch.run screening | test_default_funnel_is_24_starts_10_screen_3_finalists, test_funnel_structure_and_eval_budget | DONE |
| Top 3 candidates | n_top=3, finalists sorted by (key, soft) | test_funnel_structure_and_eval_budget | DONE |
| Confirmation across all archived windows | BasisSearch.run confirm loop on all eligible windows, held-out evidence, Bonferroni | test_funnel_structure_and_eval_budget, test_screening_windows_cannot_testify_for_their_own_winner, test_fallback_without_holdout_is_stricter_than_with_it | DONE |
| Tiered objective | O.evaluate / O.firewall inside the search | test_planted_effect_is_found, test_incumbent_never_replaced_by_a_worse_confirmed_candidate | DONE |
| Next-basis adoption | SearchResult.adopted/cfg/meta; loop2 train_basis | test_planted_effect_is_found, test_train_basis_adopts_a_planted_improvement_and_tags_windows (loop2 tests) | DONE |
| Version history of bases | B.BasisHistory (append-only, hash-chained, atomic write, lineage) - was MISSING | test_history_records_adoptions_and_lineage, test_history_persists_and_tampering_is_detected | DONE (new) |
| Rollback of a bad basis | B.BasisHistory.rollback, B.review_current_basis (firewall + bootstrap) - was MISSING | test_rollback_reinstalls_an_older_basis_without_deleting_anything, test_review_rolls_back_only_on_reliable_evidence | DONE (new) |
| Anti-overfit | penalty_se / penalty_gap, held-out bootstrap, min_win_share | test_penalty_is_monotone_in_noise, test_pure_noise_is_rarely_adopted, test_screen_luck_is_not_enough | DONE |
| No look-ahead | run(as_of) hides later windows | test_no_lookahead_windows_after_as_of_never_evaluated | DONE |
| META_SPACE: half-life, prior weeks, switch threshold, minimum weeks, cooldown, revert drop, IC beta, detector max, detector minimum weeks, memory half-life, bandwidth, prior scale, shrinkage, shock parameters | B.META_SPACE (15 keys: half_life, prior_weeks, switch_z, min_weeks, cooldown, revert_drop, ic_beta, det_max, det_min_weeks, mem_half_life, mem_bandwidth, mem_prior_scale, mem_shrink, mem_shock_k, mem_shock_cut) | test_meta_space_covers_bible_list, test_validate_meta_catches_nonsense | DONE (det_max/det_min_weeks are read by adaptive via meta.get, absent from META_DEFAULT) |
| Search refuses gamed gains | O.firewall(inc_rows, cand_rows) inside BasisSearch | test_search_refuses_a_leverage_only_candidate_and_the_firewall_alone_would_pass_it, test_search_refuses_a_cash_hiding_candidate | DONE (new) |
| Wired into the real loop | scripts/livesim_loop2.py train_basis, reveal after decision | tests/test_livesim_loop2.py (25) | DONE |
| Real-system evidence | scripts/basis_offline.py | state/research/timeline_basis/basis_offline_seed0.json | REAL: nothing adopted on 39 revealed windows (no reliable improvement) |

## Phase 20 - Tiered objective firewall

| Requirement | Implementation | Test | Status |
|---|---|---|---|
| Objective is lexicographic | O.evaluate packs an integer key; O.firewall decides at the first differing element | test_order_is_transitive_and_matches_scalar, test_tier1_dominates_property, test_tier2_dominates_tier3_once_reached, test_monotone_in_each_tier | DONE |
| Tier 1: most weeks ~5-10% | O.week_row in_band, O.evaluate key[0] (with overshoot penalty) | test_week_row_known_series, test_overshoot_costs_tier1_even_when_band_share_is_equal | DONE |
| Tier 1: yearly average move near 7% | O.evaluate closeness (key[1], from abs_mean) - was MISSING | test_yearly_average_move_near_7pct_orders_equal_band_shares, test_closeness_cannot_outrank_the_band_share_and_risk_cannot_outrank_closeness, test_decisive_diffs_can_be_decided_by_closeness | DONE (new) |
| Tier 2: worst 5% week | week_row worst5 -> risk | test_tier2_dominates_tier3_once_reached | DONE |
| Tier 2: max drawdown | week_row max_dd -> risk (loop2 uses the daily-path drawdown) | test_tier2_dominates_tier3_once_reached, test_run_window_builds_tier_row_from_the_session | DONE |
| Tier 2: weeks outside band | over_band term in risk (OVER_RISK_W) | test_overshoot_costs_tier1_even_when_band_share_is_equal | DONE |
| Tier 2: catastrophic losses | cat_rate (week <= -20%) in risk; WIPEOUT floor | test_catastrophic_weeks_hurt_tier2, test_wipeout_floor | DONE |
| Tier 2 only judged once tier 1 met (C39) | reached gate on key[2], key[3] | test_risk_ignored_before_tier1_reached | DONE |
| Tier 3: positive in-band weeks | pos_in_band (weight 0.6) | test_tier3_composite_rewards_plus10_and_precision_but_only_after_tier1 | DONE |
| Tier 3: successful +10% outcomes | plus10 (weight 0.2, scaled) in _t3_composite - was computed but UNUSED | same test | DONE (new) |
| Tier 3: directional precision | dir_precision (weight 0.2) in _t3_composite - was computed but UNUSED | same test | DONE (new) |
| Lower tier cannot justify damaging a higher tier | firewall order + property tests; blended-score counter-example | test_planted_defect_naive_sum_would_be_fooled, test_tier1_dominates_property | DONE |
| Not gameable by leverage alone | O.gaming_flags "no_edge" (pooled t-stat of the mean week), firewall refuses NEW flags - was MISSING | test_leverage_only_strategy_is_caught_as_no_edge, test_levering_a_real_edge_is_not_flagged | DONE (new) |
| Not gameable by hiding in cash | "cash_hiding" (flat share) | test_cash_hiding_is_caught | DONE (new) |
| Not carried by one lucky window | "single_window_carry" | test_one_lucky_window_carry_is_caught_and_both_sides_flagged_is_not_new | DONE (new) |
| Legacy parity with scripts/livesim_loop2.tiered | O.tiered_legacy | test_legacy_parity | DONE |
| Empty / NaN / missing-field inputs | _check, evaluate([]) floor | test_empty_and_degenerate, test_gaming_flags_degenerate_inputs | DONE |
| Reporting per era / type | O.by_group, O.summary, O.rank_correlation_of_tiers | test_by_group_reports_each_era_and_never_drops_one, test_rank_correlation_of_tiers_detects_planted_conflict | DONE |

## Still MISSING or weak (honest list)
- No real-system evidence that any dial, basis or objective choice raises the share of in-band weeks: real replay showed ~22% in band
  before and after; the dial made it worse. The machinery is proven to reject non-improvements, not to find improvements.
- Tier-1 rewards swing size by design (Bible); the gaming guard catches leverage on noise, but a strategy with a real small edge
  levered into the band is (correctly) not flagged and will win tier 1 while raising tier-2 risk; only tier 2 then judges it.
- pool_q direction and the analog beta (portfolio/market swing ratio, default 3.0) are assumptions.
- The dial's exposure hook needs a change inside engine/adaptive.py (owned elsewhere); it is exercised only through DialSession.
- `simulate` (proxy leverage) is not the trading system; real evidence comes only from scripts/dial_offline.py.
