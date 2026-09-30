# C68 fix promotion (F11)

IMPLEMENTED - NOT VALIDATED. Planted worlds only.

| kind | seed | promoted_sector_slope | first_promote | promoted_other | rolled_back | final_production | last_verdict | last_blocking | last_effect | last_t | picked_weak_sector_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flip | 0 | 1 | 2020-07-03 |  | 0 | sector_slope | FAILED | out_of_sample | -0.0007403176937646226 | -2.3792805659639074 | 1.0 |
| flip | 1 | 1 | 2019-05-10 |  | 1 | incumbent | FAILED | out_of_sample | -0.0005413963673611845 | -2.21590844481821 | 0.9423076923076923 |
| flip | 2 | 0 | nan |  | 0 | incumbent | FAILED | out_of_sample | -0.0007614025108610738 | -2.635430215002205 | 0.9818181818181818 |
| flip | 3 | 1 | 2020-02-14 |  | 1 | incumbent | FAILED | out_of_sample | -0.0002742789075023823 | -1.2288141427483097 | 0.9245283018867925 |

```
{
 "flip": {
  "n": 4,
  "promote_rate": 0.75,
  "promoted_seeds": [
   0,
   1,
   3
  ],
  "any_other_fix_promoted": 0,
  "rollback_rate": 0.5,
  "first_promote": [
   "2019-05-10",
   "2020-02-14",
   "2020-07-03"
  ],
  "last_verdicts": {
   "FAILED": 4
  },
  "stage_failures": 0
 }
}
```