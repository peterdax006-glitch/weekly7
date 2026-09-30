# C68 fix promotion (F11)

IMPLEMENTED - NOT VALIDATED. Planted worlds only.

| kind | seed | promoted_sector_slope | first_promote | promoted_other | rolled_back | final_production | last_verdict | last_blocking | last_effect | last_t | picked_weak_sector_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| null | 0 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -2.2824709465993156e-05 | -1.6018492205224482 | 0.4883720930232558 |
| null | 1 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | 1.012204063532117e-05 | 0.4159782237320051 | 0.047619047619047616 |
| null | 2 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -5.843318573505129e-05 | -1.7014921211135678 | 0.38095238095238093 |
| null | 3 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -7.000895987285305e-05 | -1.67515977823973 | 0.07142857142857142 |
| null | 4 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -1.3375571381682902e-05 | -0.9330849094122459 | 0.3488372093023256 |
| null | 5 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -3.221923348873984e-05 | -1.1567057297290755 | 0.7619047619047619 |
| planted | 0 | 1 | 2020-01-17 |  | 0 | sector_slope | NEEDS_MORE_EVIDENCE | complexity,replication | 0.0009738944055652207 | 5.52136182936915 | 1.0 |
| planted | 1 | 0 | nan |  | 0 | incumbent | NEEDS_MORE_EVIDENCE | replication | 0.001059996740402764 | 6.710912447081989 | 1.0 |
| planted | 2 | 0 | nan |  | 0 | incumbent | FAILED | risk | 0.0011642914789787794 | 6.580421199573915 | 1.0 |
| planted | 3 | 1 | 2019-08-30 |  | 0 | sector_slope | FAILED | out_of_sample,transfer | 0.00035129905512819033 | 1.8843758165023892 | 1.0 |
| planted | 4 | 1 | 2019-05-10 |  | 0 | sector_slope | FAILED | out_of_sample | -0.00020847950468490064 | -0.9983813226925294 | 1.0 |
| planted | 5 | 0 | nan |  | 0 | incumbent | QUARANTINED | leakage | 0.0006072566564445174 | 2.7851043277870833 | 1.0 |

```
{
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
  "stage_failures": 0
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
  "stage_failures": 0
 }
}
```