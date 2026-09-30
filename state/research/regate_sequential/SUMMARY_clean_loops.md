# Clean full-loop runs (after F14, F16, F17 fixes) - 30 Sep 08:35-10:55 MDT

First runs where the harness measures the gate cleanly: fresh run folders (open_loop moves old runs aside), reconcile judges only
the current code run's attempts. Default settings, W02 planted world, 129 cycles each.

| run | knowledge filed (feature: first cycle) | all genuine | failed stages |
|---|---|---|---|
| default s0 | lv20: 55, xs_atr_rank: 81 | yes | 0 |
| default s1 | lv20: 56, vol_over_mkt: 114 | yes | 0 |
| default s2 | lv20: 52 | yes | 0 |
| default s3 | vol_over_mkt: 36, lv20: 84, xs_atr_rank: 120 | yes | 0 |
| default s4 | xs_atr_rank: 125 | yes | 0 |
| default s5 | - | - | 0 |
| null s0-s2 | - | - | 0 |

- Genuine knowledge on 5/6 seeds, 9 items, 0 false; nulls 0/3; 0 failed stages.
- History: W-12 0/1 -> pre-fix 2/6 -> final 4/6 (harness-contaminated) -> F14+F16 3/6 (harness-contaminated) -> clean 5/6.
- C68 stages now RUN (seeds 0 and 3 checked: 0 refused; expectations, market_regime, monitor_audit, pattern_change,
  selection_policy OK every cycle; outcomes_errors and validate_promote OK from cycle 3-7). OPEN: c68.research_depth and
  c68.what_changed were SKIPPED_NO_INPUT in 129/129 cycles and error_research ran only 4-8 times - their inputs never arrive.
