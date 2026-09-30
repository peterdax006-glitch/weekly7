# C68 fix promotion (F11)

IMPLEMENTED - NOT VALIDATED. Planted worlds only.

| kind | seed | promoted_sector_slope | first_promote | promoted_other | rolled_back | final_production | last_verdict | last_blocking | last_effect | last_t | picked_weak_sector_share | post_flip_promotions | promotion_regime | held_by_regime | rollback_at | rollback_why | rollback_delay_weeks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flip | 0 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample,regime_in_force | -0.00013652349232662805 | -0.5264596145083007 | 1.0 | 0 | nan | 1 | nan | nan | nan |
| flip | 1 | 1 | 2019-05-10 |  | 1 | incumbent | FAILED | out_of_sample | -0.00016345032648366423 | -0.697030277461229 | 0.9615384615384616 | 0 | NO_CHANGE | 0 | 2020-07-31 | revalidation | 14.6 |
| flip | 2 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample,regime_in_force | -0.0007614025108610738 | -2.635430215002205 | 0.9818181818181818 | 0 | nan | 0 | nan | nan | nan |
| flip | 3 | 1 | 2020-02-14 |  | 1 | incumbent | FAILED | out_of_sample | -2.961228453695269e-05 | -0.20422599924796647 | 0.8113207547169812 | 0 | NO_CHANGE | 0 | 2020-06-05 | revalidation | 6.6 |
| flip | 4 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample,regime_in_force | -0.000670058270442034 | -2.5733575272049447 | 1.0 | 0 | nan | 0 | nan | nan | nan |
| flip | 5 | 1 | 2019-05-10 |  | 1 | incumbent | FAILED | out_of_sample | -5.397984339918935e-05 | -0.1823179095690992 | 0.9038461538461539 | 0 | NO_CHANGE | 0 | 2020-09-25 | degradation | 22.6 |
| null | 0 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -2.2824709465993156e-05 | -1.6018492205224482 | 0.4883720930232558 | 0 | nan | 0 | nan | nan | nan |
| null | 1 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | 1.012204063532117e-05 | 0.4159782237320051 | 0.047619047619047616 | 0 | nan | 0 | nan | nan | nan |
| null | 2 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -5.843318573505129e-05 | -1.7014921211135678 | 0.38095238095238093 | 0 | nan | 0 | nan | nan | nan |
| null | 3 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -7.000895987285305e-05 | -1.67515977823973 | 0.07142857142857142 | 0 | nan | 0 | nan | nan | nan |
| null | 4 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -1.3375571381682902e-05 | -0.9330849094122459 | 0.3488372093023256 | 0 | nan | 0 | nan | nan | nan |
| null | 5 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -3.221923348873984e-05 | -1.1567057297290755 | 0.7619047619047619 | 0 | nan | 0 | nan | nan | nan |
| planted | 0 | 1 | 2020-01-17 |  | 0 | sector_slope | NEEDS_MORE_EVIDENCE | complexity,replication | 0.0009738944055652207 | 5.52136182936915 | 1.0 | 0 | NO_CHANGE | 0 | nan | nan | nan |
| planted | 1 | 0 | nan |  | 0 | incumbent | NEEDS_MORE_EVIDENCE | replication | 0.001059996740402764 | 6.710912447081989 | 1.0 | 0 | nan | 0 | nan | nan | nan |
| planted | 2 | 0 | nan |  | 0 | incumbent | FAILED | risk | 0.0011642914789787794 | 6.580421199573915 | 1.0 | 0 | nan | 0 | nan | nan | nan |
| planted | 3 | 1 | 2019-08-30 |  | 0 | sector_slope | FAILED | out_of_sample,transfer,regime_in_force | 0.00035129905512819033 | 1.8843758165023892 | 1.0 | 0 | NO_CHANGE | 0 | nan | nan | nan |
| planted | 4 | 1 | 2019-05-10 |  | 0 | sector_slope | FAILED | out_of_sample,regime_in_force | -0.00020847950468490064 | -0.9983813226925294 | 1.0 | 0 | NO_CHANGE | 2 | nan | nan | nan |
| planted | 5 | 0 | nan |  | 0 | incumbent | QUARANTINED | leakage | 0.0006072566564445174 | 2.7851043277870833 | 1.0 | 0 | nan | 0 | nan | nan | nan |

```
{
 "flip": {
  "n": 6,
  "promote_rate": 0.5,
  "promoted_seeds": [
   1,
   3,
   5
  ],
  "any_other_fix_promoted": 0,
  "rollback_rate": 0.5,
  "first_promote": [
   "2019-05-10",
   "2019-05-10",
   "2020-02-14"
  ],
  "last_verdicts": {
   "FAILED": 6
  },
  "stage_failures": 0,
  "post_flip_promotions": 0,
  "held_by_regime": 1,
  "rollback_delay_weeks": [
   6.6,
   14.6,
   22.6
  ],
  "rollback_why": {
   "revalidation": 2,
   "degradation": 1
  }
 },
 "null": {
  "n": 6,
  "promote_rate": 0.0,
  "promoted_seeds": [],
  "any_other_fix_promoted": 0,
  "rollback_rate": 0.0,
  "first_promote": [],
  "last_verdicts": {
   "FAILED": 6
  },
  "stage_failures": 0,
  "post_flip_promotions": 0,
  "held_by_regime": 0,
  "rollback_delay_weeks": [],
  "rollback_why": {}
 },
 "planted": {
  "n": 6,
  "promote_rate": 0.5,
  "promoted_seeds": [
   0,
   3,
   4
  ],
  "any_other_fix_promoted": 0,
  "rollback_rate": 0.0,
  "first_promote": [
   "2019-05-10",
   "2019-08-30",
   "2020-01-17"
  ],
  "last_verdicts": {
   "FAILED": 3,
   "NEEDS_MORE_EVIDENCE": 2,
   "QUARANTINED": 1
  },
  "stage_failures": 0,
  "post_flip_promotions": 0,
  "held_by_regime": 2,
  "rollback_delay_weeks": [],
  "rollback_why": {}
 },
 "rollback_delay_curve_snr0.8": [
  {
   "shift": -1.0,
   "rollback_prob": 0.56,
   "median_weeks": 17.5,
   "p90_weeks": 46.0,
   "max_weeks_seen": 51.0,
   "false_rollback": 0.0
  },
  {
   "shift": -1.5,
   "rollback_prob": 0.9,
   "median_weeks": 10.0,
   "p90_weeks": 27.10000000000001,
   "max_weeks_seen": 49.0,
   "false_rollback": 0.0
  },
  {
   "shift": -2.0,
   "rollback_prob": 0.99,
   "median_weeks": 7.0,
   "p90_weeks": 16.200000000000003,
   "max_weeks_seen": 35.0,
   "false_rollback": 0.0
  },
  {
   "shift": -3.0,
   "rollback_prob": 1.0,
   "median_weeks": 6.0,
   "p90_weeks": 6.0,
   "max_weeks_seen": 13.0,
   "false_rollback": 0.0
  }
 ]
}
```