# Full research loop at DEFAULT settings on the final code (F12 sequential re-gate + complexity fix + attempt-folder fix)

Run 30 Sep ~01:40-03:06 MDT, 9 detached processes, `python tests/test_regate_sequential.py loop --seed N --world W --fresh`
(W02 planted world, LoopConfig defaults, gate_min_weeks 26, the feed's sweeps, C68 stages registered; free_gb pinned to 12 =
machine RAM admission only). Every run reached the end of the planted data (129 cycles, 2017-05-12 .. 2019-10-25).
Note: F13 edited self_correct.py/error_loop.py while these ran; each process kept the code it loaded at start.

| run | knowledge filed (feature: first cycle) | all genuine? | still open at the end (feature: verdict, looks) | cycles with a failed stage |
|---|---|---|---|---|
| default s0 | - | - | vol_over_mkt FAILED x8, xs_vol_rank NME x6, lv20 NME x2 | 1 (graph id read as a year; fixed f2123dc3) |
| default s1 | xs_atr_rank: 58, xs_vol_rank: 102 | yes | - | 0 |
| default s2 | - | - | xs_atr_rank NME x6, lv20 NME x1 | 0 |
| default s3 | xs_vol_rank: 77, lv20: 112 | yes | xs_atr_rank NME x7 | 0 |
| default s4 | xs_range_rank: 109 | yes | xs_atr_rank NME x7, lv20 NME x5 | 0 |
| default s5 | vol_over_mkt: 47 | yes | - | 0 |
| null s0-s2 | - | - | nothing gated | 0 |

- Genuine knowledge filed on 4/6 default seeds (pre-fix code: 2/6; W-12 before F12: 0/1). Every filed item is a planted genuine
  feature; false knowledge 0; null worlds 0/3. No genuine finding was retired by the look cap.
- Time to first knowledge: 47-109 cycles (weeks) - slow; F14 is addressing the December-look failures and the 5-failure-episode floor.
- Stage failures fell from 1-10 per run to 1 in all nine runs (the remaining one is fixed).
