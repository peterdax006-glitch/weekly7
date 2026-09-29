# Timeline pattern memory on the real weekly panel

500 seeded tickers, 1286 patterns registered from 23 half-year miner blocks, 27176 post-registration quarterly observations. Cumulative candidate tries: 38663 (best-of-many null |t| ~ 4.60). Chain ok: True. Prefix-invariance violations: 0.

## Patterns by mode over real time

         T  registered  with_evidence  universal  local  disregarded  cum_tries
2015-07-01          80              0          0      0            0      38663
2016-01-01         149             67          0      0           67      38663
2016-07-01         214            133          0      0          133      38663
2017-01-01         263            133          0      0          133      38663
2017-07-01         327            238          0      0          238      38663
2018-01-01         352            295          0      0          295      38663
2018-07-01         390            317          0      0          317      38663
2019-01-01         465            353          0      0          353      38663
2019-07-01         521            412          0      0          412      38663
2020-01-01         590            455          0      0          455      38663
2020-07-01         656            522          0      0          522      38663
2021-01-01         727            592          0      0          592      38663
2021-07-01         795            651          0      0          651      38663
2022-01-01         851            720          0      0          720      38663
2022-07-01         918            775          0      0          775      38663
2023-01-01         980            775          0      0          775      38663
2023-07-01        1029            899          0      0          899      38663
2024-01-01        1070            945          0      0          945      38663
2024-07-01        1116            982          0      0          982      38663
2025-01-01        1166           1023          0      0         1023      38663
2025-07-01        1209           1066          0      0         1066      38663
2026-01-01        1260           1108          0      0         1108      38663
2026-07-01        1286           1157          0      1         1156      38663

## Out-of-sample value of the view (defaults, all blocks)

Signed weekly excess return in the NEXT half-year for patterns the view uses (weight > 0), the ones it disregards, a random subset of the same size drawn from the view, and random patterns. Blocks are the independent unit.

- blocks: 23 (with >= 5 used patterns: 0); mean used 0.0, mean disregarded 591.8
- used nan  weighted nan  disregarded nan  random subset nan  random patterns nan  registry baseline nan
- used minus disregarded nan (t nan); used minus random subset nan (t nan); used minus random patterns nan (t nan)

## Walk-forward threshold fit

Fit blocks (early): 2015-07-01 .. 2020-01-01; judged on later blocks 2020-07-01 .. 2026-07-01.

- chosen: {'n_universal': 4, 't_fail': 2.5, 'tau_time_years': 6.0, 'intra_k': 4, 'use_fdr': False, 't_min_local': 1.0}
- defaults: {'n_universal': 6, 't_fail': 1.5, 'tau_time_years': 3.0, 'intra_k': 3, 'use_fdr': True, 't_min_local': 1.5}
- fit objective chosen +0.00006 vs defaults +0.00000
- JUDGE objective chosen +0.00016 vs defaults +0.00000
- chosen_active (best config that used >= 5 patterns on average in the fit blocks): {'n_universal': 4, 't_fail': 2.5, 'tau_time_years': 6.0, 'intra_k': 4, 'use_fdr': False, 't_min_local': 1.0}
- judge_chosen: used +0.00041 weighted +0.00048 disregarded +0.00034 random subset +0.00033 random +0.00032; used-minus-random-subset +0.00008 (t +0.54); mean used 237.5
- judge_chosen_active: used +0.00041 weighted +0.00048 disregarded +0.00034 random subset +0.00033 random +0.00032; used-minus-random-subset +0.00008 (t +0.54); mean used 237.5
- judge_defaults: used nan weighted nan disregarded nan random subset nan random nan; used-minus-random-subset nan (t nan); mean used 0.1

## Same, with the cumulative-tries FDR switched off (defaults otherwise)

- blocks with >= 5 used: 22; used +0.00030 disregarded +0.00022 random subset +0.00026 random patterns +0.00021; used minus random subset +0.00003 (t +0.23); used minus disregarded +0.00008 (t +0.44)

## First-noticed lag (registration to first usable evidence, days)

{'count': 1157.0, 'mean': 102.39498703543647, 'std': 56.36800187826513, 'min': 83.0, '25%': 85.0, '50%': 87.0, '75%': 89.0, 'max': 914.0}

## Final-state report

# Pattern memory report as of 2026-09-25

- observations stored: 27176; patterns: 1157; chain ok: True
- cumulative candidate tries: 38663 over 24 runs (1157 distinct keys); best-of-many null |t| ~ 4.60
- by mode: disregarded=1154, local=3

| key | mode | weight | pooled t | periods | first noticed | held | failed |
|---|---|---|---|---|---|---|---|
| days_to_earn q2 & ear q2 | local | 0.299 | 4.96 | 6 | 2021-09-24 | 5 | 1 |
| ear q2 & min20 q0 | local | 0.197 | 5.09 | 9 | 2018-09-28 | 6 | 0 |
| ear_volsurge q2 & skew60 q4 | local | 0.186 | 4.56 | 2 | 2025-03-28 | 2 | 0 |
| atr_pct q0 & ind_mom60 q4 | disregarded | 0.000 | 0.86 | 8 | 2019-09-27 | 1 | 1 |
| atr_pct q0 & intraday20 q3 | disregarded | 0.000 | 0.25 | 11 | 2016-03-24 | 2 | 5 |
| atr_pct q0 & skew60 q4 | disregarded | 0.000 | 0.04 | 11 | 2016-12-30 | 0 | 6 |
| atr_pct q1 & dist_ma50 q2 | disregarded | 0.000 | 1.49 | 5 | 2022-12-30 | 2 | 1 |
| atr_pct q1 & ind_mom60 q4 | disregarded | 0.000 | 1.86 | 5 | 2022-12-30 | 3 | 1 |
| atr_pct q1 & overnight20 q3 | disregarded | 0.000 | 1.29 | 6 | 2021-09-24 | 1 | 1 |
| atr_pct q1 & r120 q3 | disregarded | 0.000 | 0.23 | 5 | 2022-12-30 | 1 | 3 |
| atr_pct q1 & r5_nonews q4 | disregarded | 0.000 | 0.07 | 10 | 2017-09-29 | 0 | 5 |
| atr_pct q2 | disregarded | 0.000 | 1.87 | 8 | 2019-03-29 | 4 | 2 |
| atr_pct q2 & ev_red_flag q2 | disregarded | 0.000 | 2.06 | 8 | 2019-03-29 | 4 | 2 |
| atr_pct q2 & intraday20 q1 | disregarded | 0.000 | 0.26 | 4 | 2023-03-31 | 1 | 2 |
| atr_pct q2 & max20 q4 | disregarded | 0.000 | 0.06 | 1 | 2026-03-27 | 0 | 0 |
| atr_pct q2 & r1 q1 | disregarded | 0.000 | 1.17 | 4 | 2023-09-29 | 1 | 1 |
| atr_pct q2 & r1 q4 | disregarded | 0.000 | 0.44 | 6 | 2021-09-24 | 1 | 3 |
| atr_pct q2 & rel_ind20 q0 | disregarded | 0.000 | 0.53 | 11 | 2016-03-24 | 3 | 5 |
| atr_pct q2 & rel_ind20 q2 | disregarded | 0.000 | 0.86 | 11 | 2016-12-30 | 2 | 3 |
| atr_pct q3 | disregarded | 0.000 | 2.47 | 6 | 2021-09-24 | 3 | 1 |
| atr_pct q3 & close_loc q2 | disregarded | 0.000 | 0.62 | 3 | 2024-03-28 | 0 | 1 |
| atr_pct q3 & days_since_earn q1 | disregarded | 0.000 | 1.51 | 6 | 2021-09-24 | 1 | 1 |
| atr_pct q3 & days_since_earn q2 | disregarded | 0.000 | 0.27 | 8 | 2019-03-29 | 2 | 3 |
| atr_pct q3 & days_to_earn q3 | disregarded | 0.000 | 1.76 | 6 | 2021-09-24 | 1 | 0 |
| atr_pct q3 & dist_ma50 q1 | disregarded | 0.000 | 1.29 | 3 | 2024-03-28 | 2 | 1 |

## Era breakdown

                                                     key       era  periods  held  failed    mean_t
                               atr_pct q0 & ind_mom60 q4 2019-2022        4     0       1 -0.117660
                               atr_pct q0 & ind_mom60 q4 2023-2026        4     1       0 -0.490896
                              atr_pct q0 & intraday20 q3 2015-2018        3     0       1 -0.183444
                              atr_pct q0 & intraday20 q3 2019-2022        4     1       2  0.033863
                              atr_pct q0 & intraday20 q3 2023-2026        4     1       2 -0.105323
                                  atr_pct q0 & skew60 q4 2015-2018        3     0       3  0.584370
                                  atr_pct q0 & skew60 q4 2019-2022        4     0       2 -0.029208
                                  atr_pct q0 & skew60 q4 2023-2026        4     0       1 -0.441269
                               atr_pct q1 & dist_ma50 q2 2019-2022        1     1       0  1.311881
                               atr_pct q1 & dist_ma50 q2 2023-2026        4     1       1  0.506261
                               atr_pct q1 & ind_mom60 q4 2019-2022        1     0       0  0.507002
                               atr_pct q1 & ind_mom60 q4 2023-2026        4     3       1  0.912008
                             atr_pct q1 & overnight20 q3 2019-2022        2     1       0  0.882534
                             atr_pct q1 & overnight20 q3 2023-2026        4     0       1  0.346443
                                    atr_pct q1 & r120 q3 2019-2022        1     0       0  0.498074
                                    atr_pct q1 & r120 q3 2023-2026        4     1       3  0.005988
                               atr_pct q1 & r5_nonews q4 2015-2018        2     0       1 -0.261370
                               atr_pct q1 & r5_nonews q4 2019-2022        4     0       1 -0.105015
                               atr_pct q1 & r5_nonews q4 2023-2026        4     0       3  0.183511
                                              atr_pct q2 2019-2022        4     3       0  1.205668
                                              atr_pct q2 2023-2026        4     1       2  0.118704
                             atr_pct q2 & ev_red_flag q2 2019-2022        4     3       0  1.283044
                             atr_pct q2 & ev_red_flag q2 2023-2026        4     1       2  0.173737
                              atr_pct q2 & intraday20 q1 2023-2026        4     1       2  0.130884
                                   atr_pct q2 & max20 q4 2023-2026        1     0       0  0.055715
                                      atr_pct q2 & r1 q1 2023-2026        4     1       1  0.587467
                                      atr_pct q2 & r1 q4 2019-2022        2     0       2  0.535689
                                      atr_pct q2 & r1 q4 2023-2026        4     1       1 -0.537665
                               atr_pct q2 & rel_ind20 q0 2015-2018        3     1       0 -0.985852
                               atr_pct q2 & rel_ind20 q0 2019-2022        4     0       3  0.718281
                               atr_pct q2 & rel_ind20 q0 2023-2026        4     2       2 -0.421025
                               atr_pct q2 & rel_ind20 q2 2015-2018        3     0       1  0.204276
                               atr_pct q2 & rel_ind20 q2 2019-2022        4     2       0  0.899067
                               atr_pct q2 & rel_ind20 q2 2023-2026        4     0       2 -0.339368
                                              atr_pct q3 2019-2022        2     0       1  0.328487
                                              atr_pct q3 2023-2026        4     3       0  1.345314
                               atr_pct q3 & close_loc q2 2023-2026        3     0       1  0.356079
                         atr_pct q3 & days_since_earn q1 2019-2022        2     0       0  0.631303
                         atr_pct q3 & days_since_earn q1 2023-2026        4     1       1  0.608752
                         atr_pct q3 & days_since_earn q2 2019-2022        4     2       1  0.272086
                         atr_pct q3 & days_since_earn q2 2023-2026        4     0       2 -0.083017
                            atr_pct q3 & days_to_earn q3 2019-2022        2     0       0  0.675815
                            atr_pct q3 & days_to_earn q3 2023-2026        4     1       0  0.740150
                               atr_pct q3 & dist_ma50 q1 2023-2026        3     2       1  0.743157
                               atr_pct q3 & dist_ma50 q3 2019-2022        4     2       1 -0.888294
                               atr_pct q3 & dist_ma50 q3 2023-2026        4     1       2  0.419240
                            atr_pct q3 & ear_volsurge q3 2019-2022        2     0       0 -0.957636
                            atr_pct q3 & ear_volsurge q3 2023-2026        4     0       2  0.055987
                            atr_pct q3 & earn_in_week q2 2019-2022        2     0       0  0.388450
                            atr_pct q3 & earn_in_week q2 2023-2026        4     3       0  1.713924
                            atr_pct q3 & ev_agreement q2 2023-2026        3     2       0  1.437816
                             atr_pct q3 & ev_offering q2 2023-2026        3     2       1  0.694467
                                atr_pct q3 & ev_shelf q2 2023-2026        3     2       1  0.746028
                                    atr_pct q3 & frog q1 2023-2026        3     1       1 -0.436378
                                    atr_pct q3 & frog q2 2023-2026        3     0       2 -0.005896
                               atr_pct q3 & gap_today q2 2019-2022        1     0       0 -0.140251
                               atr_pct q3 & gap_today q2 2023-2026        4     0       2 -0.040756
                     atr_pct q3 & ins_opportunistic30 q2 2023-2026        3     2       1  0.741293
                                  atr_pct q3 & log_dv q1 2019-2022        2     0       2 -0.691881
                                  atr_pct q3 & log_dv q1 2023-2026        4     1       1  0.502701
                                  atr_pct q3 & log_dv q4 2019-2022        4     1       3  0.029798
                                  atr_pct q3 & log_dv q4 2023-2026        4     1       0  0.830242
                                atr_pct q3 & mom_12_1 q2 2015-2018        1     0       0  0.152835
                                atr_pct q3 & mom_12_1 q2 2019-2022        4     2       1  0.602741
                                atr_pct q3 & mom_12_1 q2 2023-2026        4     1       2  0.301619
                                   atr_pct q3 & news5 q2 2023-2026        4     3       0  1.296206
                             atr_pct q3 & overnight20 q0 2019-2022        1     0       1 -1.633433
                             atr_pct q3 & overnight20 q0 2023-2026        4     1       2  0.641779
                                      atr_pct q3 & r5 q1 2019-2022        2     1       1 -0.723524
                                      atr_pct q3 & r5 q1 2023-2026        4     0       2  0.301261
                  atr_pct q3 & r5 q1 unless close_loc q0 2019-2022        2     1       1 -0.862679
                  atr_pct q3 & r5 q1 unless close_loc q0 2023-2026        4     0       2  0.159928
                  atr_pct q3 & r5 q1 unless close_loc q4 2019-2022        2     0       1 -0.863070
                  atr_pct q3 & r5 q1 unless close_loc q4 2023-2026        4     2       2  0.493492
               atr_pct q3 & r5 q1 unless earn_in_week q4 2019-2022        2     1       1 -0.248745
               atr_pct q3 & r5 q1 unless earn_in_week q4 2023-2026        4     2       2  0.492352
                atr_pct q3 & r5 q1 unless ev_offering q4 2019-2022        2     1       1 -0.759384
                atr_pct q3 & r5 q1 unless ev_offering q4 2023-2026        4     0       2  0.298474
                atr_pct q3 & r5 q1 unless ev_red_flag q4 2019-2022        2     0       1 -0.703320
                atr_pct q3 & r5 q1 unless ev_red_flag q4 2023-2026        4     2       2  0.371712
                       atr_pct q3 & r5 q1 unless frog q0 2019-2022        2     1       1 -0.367942
                       atr_pct q3 & r5 q1 unless frog q0 2023-2026        4     0       2  0.103705
                       atr_pct q3 & r5 q1 unless frog q4 2019-2022        2     1       1 -0.205246
                       atr_pct q3 & r5 q1 unless frog q4 2023-2026        4     1       2  0.158543
                  atr_pct q3 & r5 q1 unless gap_today q4 2019-2022        2     0       1 -0.228873
                  atr_pct q3 & r5 q1 unless gap_today q4 2023-2026        4     0       2  0.071440
                      atr_pct q3 & r5 q1 unless news5 q4 2019-2022        2     1       1 -1.111636
                      atr_pct q3 & r5 q1 unless news5 q4 2023-2026        4     0       2  0.096291
                atr_pct q3 & r5 q1 unless overnight20 q0 2019-2022        2     1       1 -0.782749
                atr_pct q3 & r5 q1 unless overnight20 q0 2023-2026        4     0       1 -0.207136
                        atr_pct q3 & r5 q1 unless r20 q0 2019-2022        2     1       1 -0.632519
                        atr_pct q3 & r5 q1 unless r20 q0 2023-2026        4     0       3 -0.025319
                        atr_pct q3 & r5 q1 unless r20 q4 2019-2022        2     1       1 -0.582480
                        atr_pct q3 & r5 q1 unless r20 q4 2023-2026        4     1       2  0.090845
                    atr_pct q3 & r5 q1 unless r5_news q4 2019-2022        2     1       1 -0.768945
                    atr_pct q3 & r5 q1 unless r5_news q4 2023-2026        4     0       2  0.161027
                        atr_pct q3 & r5 q1 unless r60 q0 2019-2022        2     1       1 -0.409938
                        atr_pct q3 & r5 q1 unless r60 q0 2023-2026        4     0       3  0.186375
                        atr_pct q3 & r5 q1 unless r60 q4 2019-2022        2     1       1 -0.244291
                        atr_pct q3 & r5 q1 unless r60 q4 2023-2026        4     1       1 -0.704907
                                 atr_pct q3 & r5_news q2 2023-2026        3     2       0  1.242565
                          atr_pct q3 & range_compress q0 2019-2022        1     0       1 -0.998776
                          atr_pct q3 & range_compress q0 2023-2026        4     2       1  0.685951
                          atr_pct q3 & range_compress q2 2023-2026        3     0       1  0.226819
                               atr_pct q3 & rel_ind60 q2 2023-2026        4     0       2 -0.114381
                                  atr_pct q3 & skew60 q2 2019-2022        4     0       3 -0.350991
                                  atr_pct q3 & skew60 q2 2023-2026        4     3       0  1.402446
                               atr_pct q3 & vol_ratio q1 2023-2026        3     1       1  0.556986
                              atr_pct q3 & vol_surge1 q1 2019-2022        1     0       1 -0.534839
                              atr_pct q3 & vol_surge1 q1 2023-2026        4     2       1  1.022272
                                              atr_pct q4 2023-2026        4     1       0 -0.815902
                         atr_pct q4 & days_since_earn q4 2023-2026        1     1       0 -2.206798
                                     atr_pct q4 & ear q2 2023-2026        1     1       0 -2.065030
                                   atr_pct q4 & min20 q0 2023-2026        1     0       0 -0.390216
                                   atr_pct q4 & min20 q2 2019-2022        1     0       0  0.646808
                                   atr_pct q4 & min20 q2 2023-2026        4     0       2  0.118735
                                atr_pct q4 & mom_12_1 q2 2023-2026        1     0       0 -0.119542
                                   atr_pct q4 & vol20 q3 2019-2022        3     1       2 -0.407564
                                   atr_pct q4 & vol20 q3 2023-2026        4     1       2  0.529940
                              atr_pct q4 & vol_surge1 q0 2023-2026        1     0       0 -0.882663
                              atr_pct q4 & vol_surge5 q0 2023-2026        1     0       0 -0.506495
                             close_loc q0 & dist_ma50 q1 2023-2026        4     0       1 -0.269896
                             close_loc q0 & gap_today q4 2019-2022        4     2       1 -0.615415
                             close_loc q0 & gap_today q4 2023-2026        4     0       1 -0.237082
                          close_loc q0 & ins_buyers30 q4 2019-2022        2     1       1 -0.902801
                          close_loc q0 & ins_buyers30 q4 2023-2026        4     1       2 -0.146714
                            close_loc q0 & intraday20 q4 2019-2022        1     0       0 -0.467619
                            close_loc q0 & intraday20 q4 2023-2026        4     2       2 -0.654851
                                close_loc q0 & log_dv q1 2019-2022        4     1       1 -0.694419
                                close_loc q0 & log_dv q1 2023-2026        4     0       2 -0.057400
                                close_loc q0 & skew60 q2 2023-2026        2     1       0  0.687660
                                 close_loc q0 & vol20 q2 2023-2026        2     0       0  0.839171
                       close_loc q1 & days_since_earn q0 2015-2018        2     1       1  0.809385
                       close_loc q1 & days_since_earn q0 2019-2022        4     1       2  0.037480
                       close_loc q1 & days_since_earn q0 2023-2026        4     0       2 -0.328397
                       close_loc q1 & days_since_earn q3 2015-2018        4     1       1  0.389434
                       close_loc q1 & days_since_earn q3 2019-2022        4     2       0  0.729692
                       close_loc q1 & days_since_earn q3 2023-2026        4     2       1  0.538611
                          close_loc q1 & days_to_earn q2 2015-2018        2     1       0 -1.452513
                          close_loc q1 & days_to_earn q2 2019-2022        4     0       3  0.282755
                          close_loc q1 & days_to_earn q2 2023-2026        4     2       2 -0.453437
                          close_loc q1 & ear_volsurge q3 2023-2026        2     1       0 -0.973798
                            close_loc q1 & intraday20 q3 2015-2018        3     2       1  0.727178
                            close_loc q1 & intraday20 q3 2019-2022        4     1       2  0.266914
                            close_loc q1 & intraday20 q3 2023-2026        4     0       2 -0.056226
                                 close_loc q1 & min20 q2 2023-2026        2     0       0 -0.560160
                                 close_loc q1 & min20 q4 2015-2018        2     0       0  0.153201
                                 close_loc q1 & min20 q4 2019-2022        4     2       2  0.519859
                                 close_loc q1 & min20 q4 2023-2026        4     0       2 -0.006536
                                 close_loc q1 & news5 q4 2015-2018        2     0       1 -0.326303
                                 close_loc q1 & news5 q4 2019-2022        4     2       0 -1.495183
                                 close_loc q1 & news5 q4 2023-2026        4     2       2 -0.975844
                                  close_loc q1 & r120 q2 2015-2018        2     0       1 -0.042680
                                  close_loc q1 & r120 q2 2019-2022        4     0       1  0.408797
                                  close_loc q1 & r120 q2 2023-2026        4     0       2 -0.344546
                                   close_loc q1 & r20 q2 2015-2018        2     1       0  0.701567
                                   close_loc q1 & r20 q2 2019-2022        4     1       1  0.413791
                                   close_loc q1 & r20 q2 2023-2026        4     0       2 -0.451075
                                close_loc q1 & skew60 q3 2023-2026        2     0       0 -0.876703
                             close_loc q2 & gap_today q0 2019-2022        3     1       0 -0.685285
                             close_loc q2 & gap_today q0 2023-2026        4     0       1 -0.042486
                                   close_loc q3 & ear q3 2015-2018        3     1       1 -0.637699
                                   close_loc q3 & ear q3 2019-2022        4     0       3  0.351188
                                   close_loc q3 & ear q3 2023-2026        4     0       1 -0.147792
                                  close_loc q3 & frog q3 2015-2018        1     0       0 -0.012613
                                  close_loc q3 & frog q3 2019-2022        4     1       2  0.079384
                                  close_loc q3 & frog q3 2023-2026        4     2       1 -0.428651
                                   close_loc q3 & r20 q3 2015-2018        3     0       2  0.089178
                                   close_loc q3 & r20 q3 2019-2022        4     0       3  0.035632
                                   close_loc q3 & r20 q3 2023-2026        4     2       1 -0.666521
                                    close_loc q3 & r5 q2 2019-2022        2     0       1  0.206090
                                    close_loc q3 & r5 q2 2023-2026        4     1       2 -0.211270
                        close_loc q3 & range_compress q3 2015-2018        3     1       2 -0.241717
                        close_loc q3 & range_compress q3 2019-2022        4     0       1 -0.097825
                        close_loc q3 & range_compress q3 2023-2026        4     2       0  0.961499
                            close_loc q3 & vol_surge1 q3 2015-2018        1     1       0  1.270794
                            close_loc q3 & vol_surge1 q3 2019-2022        4     0       3 -0.256478
                            close_loc q3 & vol_surge1 q3 2023-2026        4     0       2  0.122195
                       close_loc q4 & days_since_earn q3 2015-2018        3     1       1 -0.382897
                       close_loc q4 & days_since_earn q3 2019-2022        4     0       0 -0.648476
                       close_loc q4 & days_since_earn q3 2023-2026        4     0       1  0.145014
                          close_loc q4 & days_to_earn q3 2015-2018        3     0       1 -0.168675
                          close_loc q4 & days_to_earn q3 2019-2022        4     1       0 -0.667258
                          close_loc q4 & days_to_earn q3 2023-2026        4     1       2  0.181420
                             close_loc q4 & dist_ma50 q2 2019-2022        1     0       0 -0.373276
                             close_loc q4 & dist_ma50 q2 2023-2026        4     2       2 -0.305798
                          close_loc q4 & ev_agreement q4 2023-2026        4     1       0 -0.978799
                            close_loc q4 & intraday20 q4 2023-2026        4     0       1  0.098951
                              close_loc q4 & mom_12_1 q3 2015-2018        3     0       0 -0.728899
                              close_loc q4 & mom_12_1 q3 2019-2022        4     1       1 -0.451321
                              close_loc q4 & mom_12_1 q3 2023-2026        4     0       3  0.421136
                                   close_loc q4 & r20 q4 2023-2026        4     1       2 -0.250038
                                   close_loc q4 & r60 q3 2019-2022        1     0       0  0.490831
                                   close_loc q4 & r60 q3 2023-2026        4     1       1  0.442947
                             close_loc q4 & rel_ind60 q4 2019-2022        1     0       0 -0.851974
                             close_loc q4 & rel_ind60 q4 2023-2026        4     1       1 -0.441599
                                      days_since_earn q0 2015-2018        3     1       1 -0.291314
                                      days_since_earn q0 2019-2022        4     1       1 -0.854979
                                      days_since_earn q0 2023-2026        4     0       3  0.273676
                    days_since_earn q0 & earn_in_week q2 2019-2022        2     1       0 -1.235815
                    days_since_earn q0 & earn_in_week q2 2023-2026        4     0       2  0.281172
                       days_since_earn q0 & ind_mom20 q4 2019-2022        3     0       2  0.186142
                       days_since_earn q0 & ind_mom20 q4 2023-2026        4     1       1 -0.414486
                       days_since_earn q0 & ind_mom60 q0 2015-2018        3     0       3 -0.608317
                       days_since_earn q0 & ind_mom60 q0 2019-2022        4     2       1  0.846417
                       days_since_earn q0 & ind_mom60 q0 2023-2026        4     0       2 -0.130156
                       days_since_earn q0 & ind_mom60 q1 2019-2022        4     3       1 -1.247529
                       days_since_earn q0 & ind_mom60 q1 2023-2026        4     0       4  0.692743
                   days_since_earn q0 & ins_officer30 q2 2015-2018        3     1       1 -0.303093
                   days_since_earn q0 & ins_officer30 q2 2019-2022        4     1       1 -0.986159
                   days_since_earn q0 & ins_officer30 q2 2023-2026        4     0       3  0.323152
                      days_since_earn q0 & intraday20 q3 2015-2018        4     0       0 -0.649503
                      days_since_earn q0 & intraday20 q3 2019-2022        4     2       2 -0.432439
                      days_since_earn q0 & intraday20 q3 2023-2026        4     0       2  0.155227
                           days_since_earn q0 & min20 q3 2019-2022        2     1       1  0.718713
                           days_since_earn q0 & min20 q3 2023-2026        4     0       2 -0.038787
                        days_since_earn q0 & mom_12_1 q3 2015-2018        3     0       3 -0.934781
                        days_since_earn q0 & mom_12_1 q3 2019-2022        4     1       2  0.080702
                        days_since_earn q0 & mom_12_1 q3 2023-2026        4     1       0  0.748282
                            days_since_earn q0 & r120 q1 2015-2018        2     2       0 -1.086572
                            days_since_earn q0 & r120 q1 2019-2022        4     1       1 -0.581691
                            days_since_earn q0 & r120 q1 2023-2026        4     1       2 -0.063733
                             days_since_earn q0 & r20 q3 2015-2018        3     1       2 -0.126038
                             days_since_earn q0 & r20 q3 2019-2022        4     2       0 -1.317491
                             days_since_earn q0 & r20 q3 2023-2026        4     0       1 -0.454050
                             days_since_earn q0 & r20 q4 2019-2022        3     2       1 -0.597502
                             days_since_earn q0 & r20 q4 2023-2026        4     0       2  0.168839
                              days_since_earn q0 & r5 q1 2023-2026        3     1       0  0.922448
                  days_since_earn q0 & range_compress q1 2019-2022        2     1       0 -0.879812
                  days_since_earn q0 & range_compress q1 2023-2026        4     0       2  0.296093
                  days_since_earn q0 & range_compress q2 2019-2022        2     0       0 -0.316850
                  days_since_earn q0 & range_compress q2 2023-2026        4     1       1 -0.266083
                       days_since_earn q0 & rel_ind20 q3 2015-2018        3     0       2  0.282646
                       days_since_earn q0 & rel_ind20 q3 2019-2022        4     1       3 -0.187363
                       days_since_earn q0 & rel_ind20 q3 2023-2026        4     0       2 -0.169423
                       days_since_earn q0 & rel_ind60 q3 2015-2018        3     1       1 -0.876961
                       days_since_earn q0 & rel_ind60 q3 2019-2022        4     0       2  0.231743
                       days_since_earn q0 & rel_ind60 q3 2023-2026        4     0       1 -0.280233
                                      days_since_earn q1 2019-2022        2     0       1  0.261588
                                      days_since_earn q1 2023-2026        4     1       1 -0.386807
                       days_since_earn q1 & dist_ma50 q0 2015-2018        3     1       1 -0.497415
                       days_since_earn q1 & dist_ma50 q0 2019-2022        4     0       1 -0.137652
                       days_since_earn q1 & dist_ma50 q0 2023-2026        4     0       2 -0.152062
                       days_since_earn q1 & dist_ma50 q2 2015-2018        2     0       0  0.839338
                       days_since_earn q1 & dist_ma50 q2 2019-2022        4     1       2  0.056217
                       days_since_earn q1 & dist_ma50 q2 2023-2026        4     0       2 -0.265141
                       days_since_earn q1 & dist_ma50 q4 2015-2018        1     0       0  0.895679
                       days_since_earn q1 & dist_ma50 q4 2019-2022        4     0       2 -0.087946
                       days_since_earn q1 & dist_ma50 q4 2023-2026        4     2       0  1.043223
                            days_since_earn q1 & frog q3 2019-2022        4     1       2 -0.205398
                            days_since_earn q1 & frog q3 2023-2026        4     2       1 -0.734549
                       days_since_earn q1 & gap_today q0 2019-2022        3     1       1 -0.568386
                       days_since_earn q1 & gap_today q0 2023-2026        4     2       0 -0.976411
                       days_since_earn q1 & ind_mom20 q1 2023-2026        2     1       0 -0.614444
                       days_since_earn q1 & ind_mom20 q2 2015-2018        3     1       1  0.447235
                       days_since_earn q1 & ind_mom20 q2 2019-2022        4     0       2 -0.308520
                       days_since_earn q1 & ind_mom20 q2 2023-2026        4     1       2  0.247747
days_since_earn q1 & ind_mom20 q2 unless ear_volsurge q4 2019-2022        2     1       0 -0.841502
days_since_earn q1 & ind_mom20 q2 unless ear_volsurge q4 2023-2026        4     0       2  0.275226
   days_since_earn q1 & ind_mom20 q2 unless vol_ratio q4 2019-2022        2     1       0 -1.003515
   days_since_earn q1 & ind_mom20 q2 unless vol_ratio q4 2023-2026        4     0       2  0.239233
  days_since_earn q1 & ind_mom20 q2 unless vol_surge1 q4 2019-2022        2     1       0 -1.121991
  days_since_earn q1 & ind_mom20 q2 unless vol_surge1 q4 2023-2026        4     0       3  0.393692
                       days_since_earn q1 & ind_mom20 q4 2015-2018        1     0       1 -0.476717
                       days_since_earn q1 & ind_mom20 q4 2019-2022        4     0       0  0.333332
                       days_since_earn q1 & ind_mom20 q4 2023-2026        4     0       2 -0.070990
                      days_since_earn q1 & intraday20 q3 2015-2018        3     0       2  0.039887
                      days_since_earn q1 & intraday20 q3 2019-2022        4     0       1 -0.245774
                      days_since_earn q1 & intraday20 q3 2023-2026        4     2       2 -0.631018
                          days_since_earn q1 & log_dv q1 2019-2022        1     0       1 -1.410347
                          days_since_earn q1 & log_dv q1 2023-2026        4     1       0  0.556056
                           days_since_earn q1 & max20 q0 2015-2018        3     0       1 -0.208514
                           days_since_earn q1 & max20 q0 2019-2022        4     0       3  0.676784
                           days_since_earn q1 & max20 q0 2023-2026        4     0       0 -0.627057
                            days_since_earn q1 & r120 q1 2015-2018        2     0       1 -0.442395
                            days_since_earn q1 & r120 q1 2019-2022        4     1       2  0.109018
                            days_since_earn q1 & r120 q1 2023-2026        4     2       1  0.537480
                             days_since_earn q1 & r20 q2 2015-2018        2     1       1 -0.158559
                             days_since_earn q1 & r20 q2 2019-2022        4     2       2 -0.131933
                             days_since_earn q1 & r20 q2 2023-2026        4     1       0 -0.540737
                             days_since_earn q1 & r20 q4 2015-2018        1     0       0  0.551752
                             days_since_earn q1 & r20 q4 2019-2022        4     0       4 -0.679549
                             days_since_earn q1 & r20 q4 2023-2026        4     1       1  0.611950
                  days_since_earn q1 & range_compress q3 2015-2018        3     0       1  0.443148
                  days_since_earn q1 & range_compress q3 2019-2022        4     1       2  0.205018
                  days_since_earn q1 & range_compress q3 2023-2026        4     0       2 -0.137268
                       days_since_earn q1 & rel_ind60 q1 2015-2018        2     1       1 -0.519978
                       days_since_earn q1 & rel_ind60 q1 2019-2022        4     1       2 -0.015966
                       days_since_earn q1 & rel_ind60 q1 2023-2026        4     1       2 -0.340697
                       days_since_earn q1 & vol_ratio q0 2015-2018        4     1       2  0.282767
                       days_since_earn q1 & vol_ratio q0 2019-2022        4     2       2 -0.047353
                       days_since_earn q1 & vol_ratio q0 2023-2026        4     0       1  0.102567
                                      days_since_earn q2 2023-2026        2     1       1 -0.524183
                       days_since_earn q2 & ind_mom60 q1 2023-2026        4     1       2  0.385347
                          days_since_earn q2 & log_dv q1 2019-2022        4     1       0  0.856235
                          days_since_earn q2 & log_dv q1 2023-2026        4     2       1  0.921681
                           days_since_earn q2 & max20 q1 2019-2022        4     1       3  0.050806
                           days_since_earn q2 & max20 q1 2023-2026        4     1       1 -0.331007
                              days_since_earn q2 & r5 q1 2019-2022        3     2       0  1.294430
                              days_since_earn q2 & r5 q1 2023-2026        4     1       3 -0.493524
                             days_since_earn q2 & r60 q3 2015-2018        1     0       1 -1.033372
                             days_since_earn q2 & r60 q3 2019-2022        4     3       0  1.261051
                             days_since_earn q2 & r60 q3 2023-2026        4     1       2  0.377208
                  days_since_earn q2 & range_compress q2 2023-2026        4     2       1  0.765407
                       days_since_earn q2 & vol_ratio q1 2019-2022        4     1       1  0.416295
                       days_since_earn q2 & vol_ratio q1 2023-2026        4     1       2  0.328102
                      days_since_earn q2 & vol_surge1 q2 2023-2026        3     1       2  0.198939
                                      days_since_earn q3 2019-2022        3     1       0  0.702241
                                      days_since_earn q3 2023-2026        4     3       0  1.640394
                    days_since_earn q3 & days_to_earn q1 2015-2018        3     2       1  0.629122
                    days_since_earn q3 & days_to_earn q1 2019-2022        4     3       1  0.975393
                    days_since_earn q3 & days_to_earn q1 2023-2026        4     1       2  0.732937
                       days_since_earn q3 & dist_ma50 q1 2023-2026        4     2       1  1.020656
                            days_since_earn q3 & frog q0 2023-2026        3     1       1  0.067123
                            days_since_earn q3 & frog q4 2023-2026        3     1       1  0.552352
                       days_since_earn q3 & ind_mom60 q4 2019-2022        3     0       1  0.087805
                       days_since_earn q3 & ind_mom60 q4 2023-2026        4     1       0  0.666199
                          days_since_earn q3 & log_dv q3 2015-2018        3     0       1  0.015909
                          days_since_earn q3 & log_dv q3 2019-2022        4     1       1  0.380515
                          days_since_earn q3 & log_dv q3 2023-2026        4     1       0  0.914077
                           days_since_earn q3 & max20 q1 2023-2026        4     3       1  1.006248
                           days_since_earn q3 & min20 q4 2015-2018        1     1       0  1.447106
                           days_since_earn q3 & min20 q4 2019-2022        4     1       2 -0.166837
                           days_since_earn q3 & min20 q4 2023-2026        4     1       2  0.242816
                     days_since_earn q3 & overnight20 q3 2015-2018        2     1       1  0.481911
                     days_since_earn q3 & overnight20 q3 2019-2022        4     3       0  1.769343
                     days_since_earn q3 & overnight20 q3 2023-2026        4     2       1  0.609067
                          days_since_earn q3 & skew60 q1 2023-2026        3     1       0  1.319622
                       days_since_earn q3 & vol_ratio q2 2015-2018        3     0       2 -0.577173
                       days_since_earn q3 & vol_ratio q2 2019-2022        4     1       2  0.236572
                       days_since_earn q3 & vol_ratio q2 2023-2026        4     2       0  1.238345
                       days_since_earn q3 & vol_ratio q3 2019-2022        4     1       1  0.464030
                       days_since_earn q3 & vol_ratio q3 2023-2026        4     1       2 -0.018015
                      days_since_earn q3 & vol_surge1 q3 2023-2026        4     2       1  0.858220
                      days_since_earn q3 & vol_surge5 q2 2023-2026        3     2       0  1.442757
                                      days_since_earn q4 2015-2018        2     1       0 -1.099547
                                      days_since_earn q4 2019-2022        4     1       2 -0.244234
                                      days_since_earn q4 2023-2026        4     4       0 -1.867164
                             days_since_earn q4 & ear q2 2023-2026        1     1       0 -2.578788
                     days_since_earn q4 & ev_offering q2 2015-2018        2     1       0 -1.025128
                     days_since_earn q4 & ev_offering q2 2019-2022        4     1       2 -0.176487
                     days_since_earn q4 & ev_offering q2 2023-2026        4     3       0 -1.809559
                            days_since_earn q4 & frog q2 2015-2018        4     0       3  0.067048
                            days_since_earn q4 & frog q2 2019-2022        4     2       1 -0.868541
                            days_since_earn q4 & frog q2 2023-2026        4     2       0 -1.347463
                       days_since_earn q4 & gap_today q0 2019-2022        3     1       0 -0.649681
                       days_since_earn q4 & gap_today q0 2023-2026        4     3       0 -1.490849
                           days_since_earn q4 & max20 q4 2023-2026        1     1       0 -2.316482
                           days_since_earn q4 & min20 q0 2023-2026        1     1       0 -2.988779
                        days_since_earn q4 & mom_12_1 q2 2015-2018        4     0       1 -0.163022
                        days_since_earn q4 & mom_12_1 q2 2019-2022        4     1       0 -1.084080
                        days_since_earn q4 & mom_12_1 q2 2023-2026        4     2       1 -0.821626
                              days_since_earn q4 & r5 q1 2019-2022        3     1       1  0.501635
                              days_since_earn q4 & r5 q1 2023-2026        4     1       2  0.472187
                  days_since_earn q4 & range_compress q0 2023-2026        1     1       0 -1.795458
                       days_since_earn q4 & rel_ind20 q2 2015-2018        3     0       1 -0.392454
                       days_since_earn q4 & rel_ind20 q2 2019-2022        4     0       3  0.312235
                       days_since_earn q4 & rel_ind20 q2 2023-2026        4     0       1 -0.151736
                          days_since_earn q4 & skew60 q4 2023-2026        1     0       0 -0.765985
                           days_since_earn q4 & vol20 q3 2023-2026        3     0       1  0.048775
                           days_since_earn q4 & vol20 q4 2023-2026        1     1       0 -1.749498
                      days_since_earn q4 & vol_surge5 q0 2023-2026        1     1       0 -3.816335
                              days_to_earn q0 & min20 q1 2023-2026        3     3       0  2.148515
                              days_to_earn q0 & news5 q4 2023-2026        1     0       0 -0.243209
                                 days_to_earn q0 & r5 q3 2023-2026        3     2       1  1.160123
                         days_to_earn q0 & vol_surge1 q1 2023-2026        3     2       1  0.724930
                         days_to_earn q0 & vol_surge5 q1 2015-2018        4     0       3 -0.649639
                         days_to_earn q0 & vol_surge5 q1 2019-2022        4     1       2  0.308035
                         days_to_earn q0 & vol_surge5 q1 2023-2026        4     3       1  1.536013
                                         days_to_earn q1 2015-2018        3     2       1  0.956548
                                         days_to_earn q1 2019-2022        4     1       1  0.645701
                                         days_to_earn q1 2023-2026        4     2       1  0.760172
                          days_to_earn q1 & gap_today q3 2015-2018        4     0       2 -0.376154
                          days_to_earn q1 & gap_today q3 2019-2022        4     3       1  0.854228
                          days_to_earn q1 & gap_today q3 2023-2026        4     0       3 -0.331027
                          days_to_earn q1 & ind_mom20 q0 2015-2018        4     1       1  0.229632
                          days_to_earn q1 & ind_mom20 q0 2019-2022        4     2       1  0.694755
                          days_to_earn q1 & ind_mom20 q0 2023-2026        4     0       4 -0.557477
                          days_to_earn q1 & ind_mom20 q1 2019-2022        4     2       2  0.890385
                          days_to_earn q1 & ind_mom20 q1 2023-2026        4     0       2  0.104910
                      days_to_earn q1 & ins_officer30 q2 2015-2018        3     2       1  0.936836
                      days_to_earn q1 & ins_officer30 q2 2019-2022        4     1       1  0.650999
                      days_to_earn q1 & ins_officer30 q2 2023-2026        4     1       1  0.764274
                         days_to_earn q1 & intraday20 q0 2015-2018        3     0       2 -0.321132
                         days_to_earn q1 & intraday20 q0 2019-2022        4     2       1  0.846111
                         days_to_earn q1 & intraday20 q0 2023-2026        4     0       0  0.302660
                             days_to_earn q1 & log_dv q2 2015-2018        1     0       1 -0.613166
                             days_to_earn q1 & log_dv q2 2019-2022        4     2       1  1.084893
                             days_to_earn q1 & log_dv q2 2023-2026        4     1       2  0.157156
                        days_to_earn q1 & overnight20 q3 2015-2018        2     1       1  0.564134
                        days_to_earn q1 & overnight20 q3 2019-2022        4     1       0  0.924169
                        days_to_earn q1 & overnight20 q3 2023-2026        4     0       2 -0.039987
                                days_to_earn q1 & r20 q2 2023-2026        4     1       0  0.900501
                          days_to_earn q1 & r5_nonews q0 2019-2022        3     2       0  1.243309
                          days_to_earn q1 & r5_nonews q0 2023-2026        4     2       0  0.992631
                          days_to_earn q1 & rel_ind60 q3 2023-2026        4     2       1  0.606236
                          days_to_earn q1 & vol_ratio q4 2015-2018        4     1       2  0.198308
                          days_to_earn q1 & vol_ratio q4 2019-2022        4     0       2 -0.080676
                          days_to_earn q1 & vol_ratio q4 2023-2026        4     1       1  0.640034
                                         days_to_earn q2 2019-2022        1     0       1  0.146134
                                         days_to_earn q2 2023-2026        4     3       0 -1.979090
                          days_to_earn q2 & dist_52wh q3 2019-2022        1     0       1 -0.404865
                          days_to_earn q2 & dist_52wh q3 2023-2026        4     1       2  0.321119
                          days_to_earn q2 & dist_ma50 q2 2019-2022        1     0       1  1.209216
                          days_to_earn q2 & dist_ma50 q2 2023-2026        4     3       0 -1.482120
                                days_to_earn q2 & ear q2 2019-2022        2     1       1 -0.487409
                                days_to_earn q2 & ear q2 2023-2026        4     4       0 -2.796619
                       days_to_earn q2 & earn_in_week q2 2019-2022        2     0       1  0.082283
                       days_to_earn q2 & earn_in_week q2 2023-2026        4     2       1 -1.447945
                       days_to_earn q2 & ev_agreement q4 2023-2026        3     2       0 -1.991970
                        days_to_earn q2 & ev_offering q2 2019-2022        1     0       1  0.073684
                        days_to_earn q2 & ev_offering q2 2023-2026        4     3       0 -1.927243
                               days_to_earn q2 & frog q0 2019-2022        3     0       1 -0.175982
                               days_to_earn q2 & frog q0 2023-2026        4     4       0 -2.036441
                               days_to_earn q2 & frog q2 2019-2022        2     1       1 -0.958157
                               days_to_earn q2 & frog q2 2023-2026        4     2       0 -1.202461
                               days_to_earn q2 & frog q3 2015-2018        4     2       1 -0.850558
                               days_to_earn q2 & frog q3 2019-2022        4     1       2  0.018948
                               days_to_earn q2 & frog q3 2023-2026        4     1       2  0.143083
                          days_to_earn q2 & gap_today q0 2019-2022        3     0       0 -0.230735
                          days_to_earn q2 & gap_today q0 2023-2026        4     4       0 -1.493448
                          days_to_earn q2 & gap_today q1 2019-2022        1     0       0 -0.707459
                          days_to_earn q2 & gap_today q1 2023-2026        4     0       2 -0.017128
                          days_to_earn q2 & ind_mom60 q4 2019-2022        1     0       0 -0.088720
                          days_to_earn q2 & ind_mom60 q4 2023-2026        4     2       0 -0.956195
                      days_to_earn q2 & ins_officer30 q2 2019-2022        1     0       1  0.145824
                      days_to_earn q2 & ins_officer30 q2 2023-2026        4     3       0 -1.809091
                         days_to_earn q2 & intraday20 q4 2019-2022        2     0       0 -0.707937
                         days_to_earn q2 & intraday20 q4 2023-2026        4     1       0 -1.041340
                              days_to_earn q2 & max20 q2 2015-2018        2     0       2 -1.579815
                              days_to_earn q2 & max20 q2 2019-2022        4     1       2  0.309822
                              days_to_earn q2 & max20 q2 2023-2026        4     2       1  0.674355
                              days_to_earn q2 & max20 q4 2019-2022        1     0       0 -0.325642
                              days_to_earn q2 & max20 q4 2023-2026        4     4       0 -1.891275
                           days_to_earn q2 & mom_12_1 q0 2015-2018        4     0       2 -0.140755
                           days_to_earn q2 & mom_12_1 q0 2019-2022        4     1       3  0.127480
                           days_to_earn q2 & mom_12_1 q0 2023-2026        4     2       0 -1.076813
                           days_to_earn q2 & mom_12_1 q2 2019-2022        1     0       1  0.099501
                           days_to_earn q2 & mom_12_1 q2 2023-2026        4     2       0 -0.996301
                        days_to_earn q2 & overnight20 q4 2015-2018        2     0       1 -0.147100
                        days_to_earn q2 & overnight20 q4 2019-2022        4     2       2 -0.158361
                        days_to_earn q2 & overnight20 q4 2023-2026        4     2       1 -1.070098
                                days_to_earn q2 & r20 q4 2019-2022        2     0       0 -0.915422
                                days_to_earn q2 & r20 q4 2023-2026        4     3       0 -1.385339
                                 days_to_earn q2 & r5 q1 2019-2022        3     2       0  1.328092
                                 days_to_earn q2 & r5 q1 2023-2026        4     2       2  0.082155
                          days_to_earn q2 & r5_nonews q2 2015-2018        2     0       2  0.610518
                          days_to_earn q2 & r5_nonews q2 2019-2022        4     1       1 -1.023757
                          days_to_earn q2 & r5_nonews q2 2023-2026        4     2       1 -0.866231
                     days_to_earn q2 & range_compress q3 2015-2018        4     1       1  0.622649
                     days_to_earn q2 & range_compress q3 2019-2022        4     0       1  0.144024
                     days_to_earn q2 & range_compress q3 2023-2026        4     1       3 -0.146716
                     days_to_earn q2 & range_compress q4 2015-2018        2     1       0 -1.213100
                     days_to_earn q2 & range_compress q4 2019-2022        4     1       2 -0.038704
                     days_to_earn q2 & range_compress q4 2023-2026        4     1       2 -0.254655
                          days_to_earn q2 & rel_ind60 q4 2019-2022        4     2       1 -0.808286
                          days_to_earn q2 & rel_ind60 q4 2023-2026        4     2       1 -0.575660
                         days_to_earn q2 & vol_surge5 q1 2019-2022        3     0       1 -0.146530
                         days_to_earn q2 & vol_surge5 q1 2023-2026        4     0       2  0.016036
                         days_to_earn q3 & dist_ma200 q0 2015-2018        2     1       0 -1.305586
                         days_to_earn q3 & dist_ma200 q0 2019-2022        4     1       1 -0.620217
                         days_to_earn q3 & dist_ma200 q0 2023-2026        4     2       0 -0.794415
                          days_to_earn q3 & dist_ma50 q0 2015-2018        3     1       1 -0.219222
                          days_to_earn q3 & dist_ma50 q0 2019-2022        4     0       1 -0.288419
                          days_to_earn q3 & dist_ma50 q0 2023-2026        4     0       1 -0.117094
                       days_to_earn q3 & ear_volsurge q0 2019-2022        2     1       0 -1.931538
                       days_to_earn q3 & ear_volsurge q0 2023-2026        4     1       1 -0.670109
                          days_to_earn q3 & ind_mom20 q3 2019-2022        1     0       0 -0.736767
                          days_to_earn q3 & ind_mom20 q3 2023-2026        4     1       1 -0.511572
                         days_to_earn q3 & intraday20 q0 2015-2018        2     0       2  0.870565
                         days_to_earn q3 & intraday20 q0 2019-2022        4     1       2 -0.502030
                         days_to_earn q3 & intraday20 q0 2023-2026        4     1       2 -0.112278
                         days_to_earn q3 & intraday20 q1 2015-2018        1     0       0 -0.348522
                         days_to_earn q3 & intraday20 q1 2019-2022        4     0       2  0.227941
                         days_to_earn q3 & intraday20 q1 2023-2026        4     1       1 -0.618932
                              days_to_earn q3 & max20 q0 2015-2018        3     1       1 -0.528447
                              days_to_earn q3 & max20 q0 2019-2022        4     0       4  0.813778
                              days_to_earn q3 & max20 q0 2023-2026        4     2       1 -0.878023
                           days_to_earn q3 & mom_12_1 q3 2015-2018        3     2       0 -0.906556
                           days_to_earn q3 & mom_12_1 q3 2019-2022        4     0       3  0.489409
                           days_to_earn q3 & mom_12_1 q3 2023-2026        4     0       1 -0.263004
                                days_to_earn q3 & r20 q0 2015-2018        2     0       1  0.299621
                                days_to_earn q3 & r20 q0 2019-2022        4     1       1 -0.442124
                                days_to_earn q3 & r20 q0 2023-2026        4     1       2 -0.199156
                                days_to_earn q3 & r20 q4 2015-2018        2     1       1 -0.046556
                                days_to_earn q3 & r20 q4 2019-2022        4     0       3 -0.535983
                                days_to_earn q3 & r20 q4 2023-2026        4     1       1  0.575431
                                days_to_earn q3 & r60 q1 2019-2022        2     0       1  0.279196
                                days_to_earn q3 & r60 q1 2023-2026        4     0       1 -0.341752
                     days_to_earn q3 & range_compress q1 2019-2022        2     1       0 -0.836447
                     days_to_earn q3 & range_compress q1 2023-2026        4     2       2 -0.534175
                          days_to_earn q3 & rel_ind60 q1 2015-2018        2     1       1 -0.011838
                          days_to_earn q3 & rel_ind60 q1 2019-2022        4     1       2  0.008949
                          days_to_earn q3 & rel_ind60 q1 2023-2026        4     1       2 -0.340270
                          days_to_earn q3 & rel_ind60 q2 2019-2022        2     1       1 -0.425663
                          days_to_earn q3 & rel_ind60 q2 2023-2026        4     1       1 -0.539080
                          days_to_earn q4 & dist_ma50 q0 2019-2022        1     0       0 -0.666173
                          days_to_earn q4 & dist_ma50 q0 2023-2026        4     0       2 -0.049833
                                days_to_earn q4 & ear q2 2015-2018        4     0       2  0.187791
                                days_to_earn q4 & ear q2 2019-2022        4     2       1 -0.933606
                                days_to_earn q4 & ear q2 2023-2026        4     0       4  0.609556
                          days_to_earn q4 & gap_today q2 2019-2022        1     0       1 -0.417804
                          days_to_earn q4 & gap_today q2 2023-2026        4     1       2  0.126911
                          days_to_earn q4 & ind_mom20 q1 2019-2022        1     0       1  0.407097
                          days_to_earn q4 & ind_mom20 q1 2023-2026        4     0       1 -0.163162
                          days_to_earn q4 & ind_mom20 q2 2015-2018        3     0       3 -0.452377
                          days_to_earn q4 & ind_mom20 q2 2019-2022        4     1       1  0.224525
                          days_to_earn q4 & ind_mom20 q2 2023-2026        4     1       2  0.299020
                          days_to_earn q4 & ind_mom20 q4 2019-2022        3     0       2  0.069434
                          days_to_earn q4 & ind_mom20 q4 2023-2026        4     1       1 -0.476345
                          days_to_earn q4 & ind_mom60 q0 2019-2022        1     0       0  0.246581
                          days_to_earn q4 & ind_mom60 q0 2023-2026        4     0       2 -0.000116
                       days_to_earn q4 & ins_buyers30 q2 2015-2018        4     0       2  0.028728
                       days_to_earn q4 & ins_buyers30 q2 2019-2022        4     2       1 -0.554613
                       days_to_earn q4 & ins_buyers30 q2 2023-2026        4     0       4  0.198142
                      days_to_earn q4 & ins_officer30 q2 2015-2018        4     1       1 -0.018602
                      days_to_earn q4 & ins_officer30 q2 2019-2022        4     2       1 -0.804567
                      days_to_earn q4 & ins_officer30 q2 2023-2026        4     0       4  0.253746
                         days_to_earn q4 & intraday20 q3 2015-2018        4     2       0 -0.828631
                         days_to_earn q4 & intraday20 q3 2019-2022        4     1       2 -0.253574
                         days_to_earn q4 & intraday20 q3 2023-2026        4     0       2  0.093530
                              days_to_earn q4 & min20 q3 2019-2022        2     1       1  0.499091
                              days_to_earn q4 & min20 q3 2023-2026        4     1       2  0.002871
                                 days_to_earn q4 & r1 q2 2015-2018        4     0       3  0.096071
                                 days_to_earn q4 & r1 q2 2019-2022        4     2       1 -0.820916
                                 days_to_earn q4 & r1 q2 2023-2026        4     0       2 -0.045122
                               days_to_earn q4 & r120 q1 2015-2018        2     1       0 -0.755606
                               days_to_earn q4 & r120 q1 2019-2022        4     2       1 -0.668786
                               days_to_earn q4 & r120 q1 2023-2026        4     0       2 -0.023109
                                days_to_earn q4 & r20 q2 2019-2022        1     0       0  0.188979
                                days_to_earn q4 & r20 q2 2023-2026        4     1       2  0.182397
                                days_to_earn q4 & r20 q3 2015-2018        4     1       3  0.214496
                                days_to_earn q4 & r20 q3 2019-2022        4     2       0 -0.960603
                                days_to_earn q4 & r20 q3 2023-2026        4     0       2 -0.181179
                     days_to_earn q4 & range_compress q2 2019-2022        2     0       0 -0.389322
                     days_to_earn q4 & range_compress q2 2023-2026        4     1       2 -0.196507
                          days_to_earn q4 & rel_ind20 q3 2015-2018        4     0       0  0.538571
                          days_to_earn q4 & rel_ind20 q3 2019-2022        4     1       1  0.432026
                          days_to_earn q4 & rel_ind20 q3 2023-2026        4     0       2 -0.100083
                          days_to_earn q4 & vol_ratio q1 2015-2018        3     0       2  0.340788
                          days_to_earn q4 & vol_ratio q1 2019-2022        4     0       1 -0.309768
                          days_to_earn q4 & vol_ratio q1 2023-2026        4     0       1 -0.038989
                                            dist_52wh q0 2015-2018        3     0       1 -0.085961
                                            dist_52wh q0 2019-2022        4     1       1 -0.579337
                                            dist_52wh q0 2023-2026        4     2       1 -0.747848
                                   dist_52wh q0 & ear q2 2023-2026        1     1       0 -2.599069
                                 dist_52wh q0 & min20 q0 2023-2026        1     1       0 -1.362745
                                dist_52wh q0 & skew60 q4 2023-2026        1     1       0 -1.881221
                                 dist_52wh q0 & vol20 q4 2023-2026        1     1       0 -1.826241
                                   dist_52wh q1 & ear q1 2015-2018        2     0       1 -0.379407
                                   dist_52wh q1 & ear q1 2019-2022        4     1       0  0.611870
                                   dist_52wh q1 & ear q1 2023-2026        4     0       2 -0.328070
                          dist_52wh q1 & ear_volsurge q0 2023-2026        4     0       2 -0.191308
                              dist_52wh q1 & ev_shelf q2 2015-2018        2     0       0  0.531313
                              dist_52wh q1 & ev_shelf q2 2019-2022        4     2       1  0.792144
                              dist_52wh q1 & ev_shelf q2 2023-2026        4     1       1  0.575367
                                  dist_52wh q1 & frog q0 2015-2018        4     1       3 -0.329417
                                  dist_52wh q1 & frog q0 2019-2022        4     2       0  0.993623
                                  dist_52wh q1 & frog q0 2023-2026        4     0       0  0.663940
                                  dist_52wh q1 & frog q3 2015-2018        2     1       1 -0.260195
                                  dist_52wh q1 & frog q3 2019-2022        4     1       2  0.239067
                                  dist_52wh q1 & frog q3 2023-2026        4     0       2  0.059951
                             dist_52wh q1 & ind_mom60 q0 2015-2018        2     1       1 -0.327041
                             dist_52wh q1 & ind_mom60 q0 2019-2022        4     2       1  0.622889
                             dist_52wh q1 & ind_mom60 q0 2023-2026        4     0       2  0.032935
                           dist_52wh q1 & ins_value30 q2 2019-2022        1     0       0  0.270651
                           dist_52wh q1 & ins_value30 q2 2023-2026        4     2       0  0.775031
                            dist_52wh q1 & intraday20 q3 2019-2022        3     1       2  0.164946
                            dist_52wh q1 & intraday20 q3 2023-2026        4     2       1  0.822546
                                dist_52wh q1 & log_dv q4 2023-2026        3     1       1  0.349620
                                 dist_52wh q1 & min20 q1 2015-2018        3     0       0  0.719682
                                 dist_52wh q1 & min20 q1 2019-2022        4     1       2  0.110342
                                 dist_52wh q1 & min20 q1 2023-2026        4     3       0  1.582199
                                 dist_52wh q1 & news5 q2 2023-2026        4     1       1  0.436932
                           dist_52wh q1 & overnight20 q3 2015-2018        3     2       0  0.812725
                           dist_52wh q1 & overnight20 q3 2019-2022        4     0       2  0.040843
                           dist_52wh q1 & overnight20 q3 2023-2026        4     2       1  0.972092
                                   dist_52wh q1 & r20 q2 2015-2018        2     1       0  0.784798
                                   dist_52wh q1 & r20 q2 2019-2022        4     1       2  0.054657
                                   dist_52wh q1 & r20 q2 2023-2026        4     1       2  0.039171
                                   dist_52wh q1 & r20 q3 2019-2022        4     1       1 -0.517503
                                   dist_52wh q1 & r20 q3 2023-2026        4     0       3  0.446813
                               dist_52wh q1 & r5_news q4 2019-2022        2     2       0  1.270470
                               dist_52wh q1 & r5_news q4 2023-2026        3     0       3 -0.546619
                             dist_52wh q1 & r5_nonews q4 2015-2018        2     0       2  0.235693
                             dist_52wh q1 & r5_nonews q4 2019-2022        4     2       2 -0.585613
                             dist_52wh q1 & r5_nonews q4 2023-2026        4     0       1  0.030940
                                   dist_52wh q1 & r60 q3 2015-2018        2     1       1  0.903967
                                   dist_52wh q1 & r60 q3 2019-2022        4     2       2  0.312187
                                   dist_52wh q1 & r60 q3 2023-2026        4     0       2 -0.164708
                             dist_52wh q1 & rel_ind20 q3 2015-2018        2     0       1  0.565062
                             dist_52wh q1 & rel_ind20 q3 2019-2022        4     1       1 -0.322413
                             dist_52wh q1 & rel_ind20 q3 2023-2026        4     0       1 -0.231147
                             dist_52wh q1 & rel_ind60 q1 2019-2022        1     1       0  1.087383
                             dist_52wh q1 & rel_ind60 q1 2023-2026        4     1       1  0.376122
                            dist_52wh q1 & vol_surge1 q1 2023-2026        3     0       1  0.320126
                            dist_52wh q1 & vol_surge1 q3 2015-2018        2     2       0  1.303250
                            dist_52wh q1 & vol_surge1 q3 2019-2022        4     1       1  0.365458
                            dist_52wh q1 & vol_surge1 q3 2023-2026        4     2       2  0.645752
                            dist_52wh q1 & vol_surge5 q1 2015-2018        4     0       2 -0.225254
                            dist_52wh q1 & vol_surge5 q1 2019-2022        4     1       3  0.162875
                            dist_52wh q1 & vol_surge5 q1 2023-2026        4     1       1  0.971627
                                            dist_52wh q2 2015-2018        4     0       3 -0.376257
                                            dist_52wh q2 2019-2022        4     1       1  0.627282
                                            dist_52wh q2 2023-2026        4     1       1  0.384203
                                  dist_52wh q2 & frog q0 2023-2026        4     3       0  1.343458
                             dist_52wh q2 & gap_today q4 2015-2018        4     1       0 -0.790739
                             dist_52wh q2 & gap_today q4 2019-2022        4     1       2  0.343051
                             dist_52wh q2 & gap_today q4 2023-2026        4     1       3  0.045395
                             dist_52wh q2 & ind_mom20 q4 2015-2018        1     0       1 -1.287663
                             dist_52wh q2 & ind_mom20 q4 2019-2022        4     2       1  0.378396
                             dist_52wh q2 & ind_mom20 q4 2023-2026        4     2       0  1.111205
                            dist_52wh q2 & intraday20 q0 2015-2018        4     1       3 -0.304033
                            dist_52wh q2 & intraday20 q0 2019-2022        4     1       2  0.069692
                            dist_52wh q2 & intraday20 q0 2023-2026        4     1       1  0.536870
                                 dist_52wh q2 & min20 q4 2015-2018        2     1       1  0.826821
                                 dist_52wh q2 & min20 q4 2019-2022        4     1       1  0.496363
                                 dist_52wh q2 & min20 q4 2023-2026        4     0       3 -0.537627
                           dist_52wh q2 & overnight20 q4 2019-2022        1     1       0  1.559786
                           dist_52wh q2 & overnight20 q4 2023-2026        4     1       1  0.282470
                                    dist_52wh q2 & r1 q0 2015-2018        4     0       2 -0.252696
                                    dist_52wh q2 & r1 q0 2019-2022        4     1       1  0.420098
                                    dist_52wh q2 & r1 q0 2023-2026        4     2       1  0.678492
                                    dist_52wh q2 & r1 q1 2019-2022        1     0       1 -0.895797
                                    dist_52wh q2 & r1 q1 2023-2026        4     1       1  0.519754
                             dist_52wh q2 & rel_ind20 q0 2015-2018        4     2       0 -0.970736
                             dist_52wh q2 & rel_ind20 q0 2019-2022        4     1       3  0.250172
                             dist_52wh q2 & rel_ind20 q0 2023-2026        4     0       2  0.088881
                             dist_52wh q3 & dist_ma50 q4 2019-2022        1     0       1 -0.474528
                             dist_52wh q3 & dist_ma50 q4 2023-2026        4     2       1  0.802424
                                   dist_52wh q3 & ear q4 2019-2022        1     0       0  0.466852
                                   dist_52wh q3 & ear q4 2023-2026        4     1       1  0.701565
                                dist_52wh q3 & log_dv q3 2019-2022        1     0       0  0.621711
                                dist_52wh q3 & log_dv q3 2023-2026        4     1       2  0.266446
                                 dist_52wh q3 & min20 q4 2019-2022        4     1       3 -0.496354
                                 dist_52wh q3 & min20 q4 2023-2026        4     1       2 -0.433087
                                  dist_52wh q3 & r120 q3 2019-2022        1     0       1  0.093018
                                  dist_52wh q3 & r120 q3 2023-2026        4     0       2 -0.266317
                                   dist_52wh q3 & r20 q3 2015-2018        3     1       0  0.998876
                                   dist_52wh q3 & r20 q3 2019-2022        4     1       2  0.279976
                                   dist_52wh q3 & r20 q3 2023-2026        4     0       2  0.099857
                             dist_52wh q3 & r5_nonews q3 2019-2022        2     0       0  0.503891
                             dist_52wh q3 & r5_nonews q3 2023-2026        4     0       0  0.535596
                             dist_52wh q3 & r5_nonews q4 2023-2026        4     2       1  0.795763
                                   dist_52wh q3 & r60 q2 2015-2018        4     0       1  0.154383
                                   dist_52wh q3 & r60 q2 2019-2022        4     0       1 -0.094460
                                   dist_52wh q3 & r60 q2 2023-2026        4     2       2  0.382639
                             dist_52wh q3 & rel_ind20 q3 2015-2018        4     1       2  0.399814
                             dist_52wh q3 & rel_ind20 q3 2019-2022        4     1       2  0.139499
                             dist_52wh q3 & rel_ind20 q3 2023-2026        4     2       0  0.904551
                             dist_52wh q3 & rel_ind60 q4 2019-2022        1     0       0  0.060973
                             dist_52wh q3 & rel_ind60 q4 2023-2026        4     1       1  0.791136
                                 dist_52wh q3 & vol20 q0 2015-2018        3     1       1 -0.330908
                                 dist_52wh q3 & vol20 q0 2019-2022        4     1       3  0.159918
                                 dist_52wh q3 & vol20 q0 2023-2026        4     1       1 -0.602238
                            dist_52wh q3 & vol_surge1 q3 2019-2022        1     0       0  0.056703
                            dist_52wh q3 & vol_surge1 q3 2023-2026        4     2       0  0.942115
                            dist_ma200 q0 & gap_today q1 2019-2022        3     0       2  0.030290
                            dist_ma200 q0 & gap_today q1 2023-2026        4     1       2  0.256911
                                           dist_ma200 q1 2023-2026        4     2       0  1.287022
                                 dist_ma200 q1 & frog q4 2019-2022        4     2       2  0.782382
                                 dist_ma200 q1 & frog q4 2023-2026        4     0       1  0.521113
                            dist_ma200 q1 & ind_mom60 q1 2015-2018        1     0       1 -1.415033
                            dist_ma200 q1 & ind_mom60 q1 2019-2022        4     1       1  0.373139
                            dist_ma200 q1 & ind_mom60 q1 2023-2026        4     2       0  0.869267
                               dist_ma200 q1 & skew60 q4 2023-2026        1     0       0  0.647980
                           dist_ma200 q1 & vol_surge5 q3 2023-2026        4     3       1  1.852248
                            dist_ma200 q2 & gap_today q0 2023-2026        4     0       1 -0.187374
                           dist_ma200 q2 & intraday20 q4 2023-2026        2     0       0  0.343385
                            dist_ma200 q2 & vol_ratio q3 2015-2018        1     0       1 -0.380865
                            dist_ma200 q2 & vol_ratio q3 2019-2022        4     1       2  0.084464
                            dist_ma200 q2 & vol_ratio q3 2023-2026        4     2       1  0.430083
                                  dist_ma200 q3 & ear q3 2015-2018        3     1       2  0.309496
                                  dist_ma200 q3 & ear q3 2019-2022        4     2       2  0.282389
                                  dist_ma200 q3 & ear q3 2023-2026        4     0       2 -0.075309
                                  dist_ma200 q4 & ear q0 2023-2026        2     1       0  0.784609
                             dist_ma200 q4 & mom_12_1 q0 2023-2026        4     1       2 -0.252735
                               dist_ma200 q4 & skew60 q2 2019-2022        4     2       2 -0.170730
                               dist_ma200 q4 & skew60 q2 2023-2026        4     0       2  0.120313
                                  dist_ma50 q0 & frog q0 2019-2022        3     0       3  0.581426
                                  dist_ma50 q0 & frog q0 2023-2026        4     2       0 -1.229046
                                  dist_ma50 q0 & frog q4 2019-2022        4     1       1 -0.454482
                                  dist_ma50 q0 & frog q4 2023-2026        4     1       2  0.341832
                             dist_ma50 q0 & gap_today q0 2019-2022        3     1       1 -0.396172
                             dist_ma50 q0 & gap_today q0 2023-2026        4     3       0 -1.254117
                                dist_ma50 q0 & log_dv q1 2015-2018        3     2       0  1.468747
                                dist_ma50 q0 & log_dv q1 2019-2022        4     0       3 -0.124009
                                dist_ma50 q0 & log_dv q1 2023-2026        4     1       3 -0.552647
                                            dist_ma50 q1 2015-2018        3     0       1 -0.051225
                                            dist_ma50 q1 2019-2022        4     3       0  1.321259
                                            dist_ma50 q1 2023-2026        4     1       0  1.027669
                          dist_ma50 q1 & ear_volsurge q3 2015-2018        1     1       0 -1.404465
                          dist_ma50 q1 & ear_volsurge q3 2019-2022        4     0       2  0.054739
                          dist_ma50 q1 & ear_volsurge q3 2023-2026        4     0       1  0.171980
                          dist_ma50 q1 & ev_agreement q2 2023-2026        4     2       0  1.174374
                           dist_ma50 q1 & ev_offering q2 2015-2018        3     0       2 -0.136886
                           dist_ma50 q1 & ev_offering q2 2019-2022        4     3       0  1.390074
                           dist_ma50 q1 & ev_offering q2 2023-2026        4     1       0  1.049854
                                  dist_ma50 q1 & frog q0 2023-2026        4     1       2  0.514015
                             dist_ma50 q1 & gap_today q1 2023-2026        4     1       0  0.926498
                             dist_ma50 q1 & ind_mom20 q2 2023-2026        4     2       0  1.267635
                                dist_ma50 q1 & log_dv q3 2023-2026        4     1       1  0.501845
                                 dist_ma50 q1 & min20 q1 2023-2026        3     2       0  1.347828
                                 dist_ma50 q1 & news5 q2 2023-2026        4     2       0  1.098125
                           dist_ma50 q1 & overnight20 q0 2023-2026        1     0       0  0.444833
                           dist_ma50 q1 & overnight20 q2 2015-2018        1     1       0  1.998513
                           dist_ma50 q1 & overnight20 q2 2019-2022        4     1       1  0.332884
                           dist_ma50 q1 & overnight20 q2 2023-2026        4     3       1  1.041854
                                  dist_ma50 q1 & r120 q3 2019-2022        4     1       2 -0.041073
                                  dist_ma50 q1 & r120 q3 2023-2026        4     1       1 -0.560280
                               dist_ma50 q1 & r5_news q2 2023-2026        4     2       0  1.273095
                             dist_ma50 q1 & r5_nonews q0 2023-2026        4     3       0  1.685274
                        dist_ma50 q1 & range_compress q2 2023-2026        4     2       0  1.401935
                             dist_ma50 q1 & rel_ind20 q1 2023-2026        2     0       0  0.619178
                                 dist_ma50 q1 & vol20 q2 2023-2026        1     0       0 -0.670995
                            dist_ma50 q1 & vol_surge1 q0 2019-2022        2     2       0 -1.573268
                            dist_ma50 q1 & vol_surge1 q0 2023-2026        4     1       3  0.093351
                            dist_ma50 q1 & vol_surge1 q1 2023-2026        4     3       1  0.880647
                            dist_ma50 q1 & vol_surge1 q3 2023-2026        4     1       2 -0.080854
                            dist_ma50 q1 & vol_surge5 q2 2023-2026        3     1       1  0.303364
                         dist_ma50 q2 & ins_officer30 q2 2019-2022        1     0       0  0.449632
                         dist_ma50 q2 & ins_officer30 q2 2023-2026        4     1       2  0.057095
                            dist_ma50 q2 & intraday20 q2 2019-2022        1     1       0  1.485401
                            dist_ma50 q2 & intraday20 q2 2023-2026        4     0       2 -0.050985
                              dist_ma50 q2 & mom_12_1 q3 2019-2022        1     0       0  0.909304
                              dist_ma50 q2 & mom_12_1 q3 2023-2026        4     2       2  0.392534
                                  dist_ma50 q2 & r120 q3 2019-2022        1     1       0  1.144959
                                  dist_ma50 q2 & r120 q3 2023-2026        4     0       1  0.200842
                                   dist_ma50 q2 & r20 q3 2015-2018        3     1       2 -0.370291
                                   dist_ma50 q2 & r20 q3 2019-2022        4     1       2 -0.028487
                                   dist_ma50 q2 & r20 q3 2023-2026        4     0       0  0.558374
                                   dist_ma50 q2 & r60 q3 2019-2022        1     0       1  0.261064
                                   dist_ma50 q2 & r60 q3 2023-2026        4     1       3 -0.173414
                             dist_ma50 q2 & rel_ind20 q4 2015-2018        3     0       1 -0.354775
                             dist_ma50 q2 & rel_ind20 q4 2019-2022        4     1       0 -0.792658
                             dist_ma50 q2 & rel_ind20 q4 2023-2026        4     0       1 -0.415580
                             dist_ma50 q2 & rel_ind60 q3 2019-2022        1     0       0  0.585132
                             dist_ma50 q2 & rel_ind60 q3 2023-2026        4     1       1  0.178531
                             dist_ma50 q2 & rel_ind60 q4 2015-2018        1     1       0 -1.710178
                             dist_ma50 q2 & rel_ind60 q4 2019-2022        4     2       1 -0.636905
                             dist_ma50 q2 & rel_ind60 q4 2023-2026        4     2       1 -0.504891
                            dist_ma50 q2 & vol_surge1 q1 2015-2018        2     1       1  0.320655
                            dist_ma50 q2 & vol_surge1 q1 2019-2022        4     2       2  0.237764
                            dist_ma50 q2 & vol_surge1 q1 2023-2026        4     0       0  0.309231
                            dist_ma50 q2 & vol_surge1 q3 2015-2018        2     0       1 -0.008964
                            dist_ma50 q2 & vol_surge1 q3 2019-2022        4     1       1 -0.344684
                            dist_ma50 q2 & vol_surge1 q3 2023-2026        4     0       3  0.301956
                          dist_ma50 q3 & ear_volsurge q0 2019-2022        4     2       0 -1.016187
                          dist_ma50 q3 & ear_volsurge q0 2023-2026        4     1       2 -0.141375
                          dist_ma50 q3 & ev_agreement q4 2019-2022        2     0       2  0.979727
                          dist_ma50 q3 & ev_agreement q4 2023-2026        3     2       0 -1.977062
                                  dist_ma50 q3 & frog q0 2019-2022        2     2       0 -1.370822
                                  dist_ma50 q3 & frog q0 2023-2026        4     3       1 -0.940890
                             dist_ma50 q3 & gap_today q3 2019-2022        4     0       3  0.318862
                             dist_ma50 q3 & gap_today q3 2023-2026        4     1       1 -0.484908
                           dist_ma50 q3 & ins_value30 q4 2023-2026        1     0       0 -0.736075
                                dist_ma50 q3 & log_dv q1 2019-2022        4     1       2 -0.299219
                                dist_ma50 q3 & log_dv q1 2023-2026        4     0       2 -0.079053
                                dist_ma50 q3 & log_dv q3 2019-2022        1     0       1  3.072167
                                dist_ma50 q3 & log_dv q3 2023-2026        4     1       0 -0.808494
                              dist_ma50 q3 & mom_12_1 q0 2019-2022        4     1       1  0.382709
                              dist_ma50 q3 & mom_12_1 q0 2023-2026        4     0       3 -0.252801
                               dist_ma50 q3 & r5_news q4 2019-2022        4     1       2 -0.332552
                               dist_ma50 q3 & r5_news q4 2023-2026        4     0       2  0.140054
                             dist_ma50 q3 & r5_nonews q3 2015-2018        3     1       0  0.710134
                             dist_ma50 q3 & r5_nonews q3 2019-2022        4     0       2 -0.430373
                             dist_ma50 q3 & r5_nonews q3 2023-2026        4     1       2  0.248311
                        dist_ma50 q3 & range_compress q2 2015-2018        1     1       0  1.114458
                        dist_ma50 q3 & range_compress q2 2019-2022        4     1       2  0.043933
                        dist_ma50 q3 & range_compress q2 2023-2026        4     2       2 -0.279816
                             dist_ma50 q3 & rel_ind60 q4 2015-2018        1     0       1  0.755800
                             dist_ma50 q3 & rel_ind60 q4 2019-2022        4     0       0 -0.634738
                             dist_ma50 q3 & rel_ind60 q4 2023-2026        4     1       0 -0.586498
                                 dist_ma50 q3 & vol20 q3 2019-2022        4     0       3 -0.076399
                                 dist_ma50 q3 & vol20 q3 2023-2026        4     1       1  0.250793
                             dist_ma50 q3 & vol_ratio q0 2019-2022        4     0       2  0.215913
                             dist_ma50 q3 & vol_ratio q0 2023-2026        4     1       1 -0.331507
                            dist_ma50 q3 & vol_surge1 q3 2015-2018        3     1       1 -0.529956
                            dist_ma50 q3 & vol_surge1 q3 2019-2022        4     2       2 -0.555587
                            dist_ma50 q3 & vol_surge1 q3 2023-2026        4     0       2 -0.246118
                            dist_ma50 q3 & vol_surge5 q1 2019-2022        3     1       0 -1.023204
                            dist_ma50 q3 & vol_surge5 q1 2023-2026        4     1       0 -1.107401
                             dist_ma50 q4 & gap_today q0 2019-2022        3     1       0 -0.932652
                             dist_ma50 q4 & gap_today q0 2023-2026        4     2       1 -1.086235
                          dist_ma50 q4 & ins_buyers30 q2 2019-2022        4     2       1 -0.714900
                          dist_ma50 q4 & ins_buyers30 q2 2023-2026        4     0       3  0.491622
                            dist_ma50 q4 & intraday20 q4 2023-2026        4     1       2 -0.195405
                                dist_ma50 q4 & log_dv q1 2019-2022        1     0       1 -0.862150
                                dist_ma50 q4 & log_dv q1 2023-2026        4     0       1  0.450680
                                dist_ma50 q4 & log_dv q3 2019-2022        1     0       1  0.255941
                                dist_ma50 q4 & log_dv q3 2023-2026        4     3       0 -0.978423
                 dist_ma50 q4 & log_dv q3 unless frog q4 2023-2026        1     1       0 -1.094295
                             dist_ma50 q4 & r5_nonews q0 2015-2018        2     1       1 -0.607525
                             dist_ma50 q4 & r5_nonews q0 2019-2022        4     0       2 -0.057269
                             dist_ma50 q4 & r5_nonews q0 2023-2026        4     2       2 -0.555144
                             dist_ma50 q4 & r5_nonews q4 2023-2026        4     1       3  0.180862
                                 dist_ma50 q4 & vol20 q2 2015-2018        4     0       2 -0.244240
                                 dist_ma50 q4 & vol20 q2 2019-2022        4     1       1  0.384656
                                 dist_ma50 q4 & vol20 q2 2023-2026        4     2       2  1.103344
                             dist_ma50 q4 & vol_ratio q3 2019-2022        1     1       0 -1.601174
                             dist_ma50 q4 & vol_ratio q3 2023-2026        4     0       1 -0.155577
                            dist_ma50 q4 & vol_surge5 q0 2023-2026        4     0       1  0.075089
                                                  ear q0 2019-2022        2     0       1  0.033117
                                                  ear q0 2023-2026        4     2       1  0.480191
                                ear q0 & earn_in_week q2 2019-2022        2     0       2 -0.592303
                                ear q0 & earn_in_week q2 2023-2026        4     2       2  0.432239
                                 ear q0 & ev_offering q2 2019-2022        2     0       1  0.017115
                                 ear q0 & ev_offering q2 2023-2026        4     1       1  0.467231
                                        ear q0 & frog q1 2019-2022        1     0       0  0.880231
                                        ear q0 & frog q1 2023-2026        4     1       2  0.129103
                                   ear q0 & ind_mom20 q2 2019-2022        2     0       2 -0.494355
                                   ear q0 & ind_mom20 q2 2023-2026        4     1       2  0.714702
                                   ear q0 & ind_mom60 q0 2019-2022        1     1       0  1.357744
                                   ear q0 & ind_mom60 q0 2023-2026        4     0       3 -0.286713
                              ear q0 & range_compress q2 2019-2022        2     0       2 -0.114989
                              ear q0 & range_compress q2 2023-2026        4     0       1  0.155393
                                                  ear q1 2015-2018        3     0       2 -0.446384
                                                  ear q1 2019-2022        4     1       1  0.260276
                                                  ear q1 2023-2026        4     2       0  1.029489
                                ear q1 & ear_volsurge q3 2015-2018        1     0       0  0.261287
                                ear q1 & ear_volsurge q3 2019-2022        4     1       1  0.159217
                                ear q1 & ear_volsurge q3 2023-2026        4     0       2 -0.042039
                                    ear q1 & ev_shelf q2 2015-2018        2     0       0  0.628346
                                    ear q1 & ev_shelf q2 2019-2022        4     1       1  0.302776
                                    ear q1 & ev_shelf q2 2023-2026        4     3       1  0.995256
                                        ear q1 & frog q0 2019-2022        4     1       1  0.633341
                                        ear q1 & frog q0 2023-2026        4     0       2  0.061285
                                   ear q1 & gap_today q3 2015-2018        1     0       0  0.241663
                                   ear q1 & gap_today q3 2019-2022        4     2       1  0.406470
                                   ear q1 & gap_today q3 2023-2026        4     1       2  0.453997
                                   ear q1 & ind_mom60 q1 2019-2022        4     0       4 -0.742800
                                   ear q1 & ind_mom60 q1 2023-2026        4     2       0  0.980068
                                ear q1 & ins_buyers30 q2 2015-2018        2     0       1  0.300222
                                ear q1 & ins_buyers30 q2 2019-2022        4     1       1  0.497266
                                ear q1 & ins_buyers30 q2 2023-2026        4     2       0  0.863025
                               ear q1 & ins_officer30 q2 2015-2018        2     0       0  0.410718
                               ear q1 & ins_officer30 q2 2019-2022        4     1       1  0.303238
                               ear q1 & ins_officer30 q2 2023-2026        4     2       0  1.116512
                                 ear q1 & ins_value30 q2 2015-2018        2     0       1  0.300222
                                 ear q1 & ins_value30 q2 2019-2022        4     1       1  0.497266
                                 ear q1 & ins_value30 q2 2023-2026        4     2       0  0.863025
                                  ear q1 & intraday20 q2 2019-2022        4     2       2  0.710233
                                  ear q1 & intraday20 q2 2023-2026        4     0       1  0.115248
                                  ear q1 & intraday20 q3 2015-2018        1     0       1  0.223160
                                  ear q1 & intraday20 q3 2019-2022        4     1       0 -0.844892
                                  ear q1 & intraday20 q3 2023-2026        4     1       3  0.638664
                                      ear q1 & log_dv q3 2015-2018        1     1       0  2.390930
                                      ear q1 & log_dv q3 2019-2022        4     0       2 -0.088571
                                      ear q1 & log_dv q3 2023-2026        4     0       2  0.018039
                                       ear q1 & max20 q0 2019-2022        3     1       2 -0.715080
                                       ear q1 & max20 q0 2023-2026        4     1       0 -0.769937
                                       ear q1 & max20 q3 2015-2018        4     1       2 -0.114443
                                       ear q1 & max20 q3 2019-2022        4     1       2  0.529331
                                       ear q1 & max20 q3 2023-2026        4     0       2  0.351153
                                       ear q1 & max20 q4 2019-2022        1     0       1 -0.495125
                                       ear q1 & max20 q4 2023-2026        4     2       0  0.886073
                                       ear q1 & min20 q1 2023-2026        3     1       0  0.886897
                                       ear q1 & min20 q3 2019-2022        2     0       2  0.852210
                                       ear q1 & min20 q3 2023-2026        4     1       1 -0.455484
                                         ear q1 & r20 q4 2019-2022        2     0       2 -1.251962
                                         ear q1 & r20 q4 2023-2026        4     1       0  0.952237
                                     ear q1 & r5_news q2 2015-2018        1     1       0  1.007213
                                     ear q1 & r5_news q2 2019-2022        4     1       1  0.612854
                                     ear q1 & r5_news q2 2023-2026        4     2       0  0.870386
                                   ear q1 & r5_nonews q2 2019-2022        2     0       2  0.886946
                                   ear q1 & r5_nonews q2 2023-2026        4     1       1 -0.561211
                              ear q1 & range_compress q0 2019-2022        1     0       1 -0.128664
                              ear q1 & range_compress q0 2023-2026        4     1       0  0.795546
                                   ear q1 & rel_ind20 q4 2019-2022        3     0       3 -1.055632
                                   ear q1 & rel_ind20 q4 2023-2026        4     2       1  1.048180
                                   ear q1 & rel_ind60 q2 2015-2018        1     0       0  0.498658
                                   ear q1 & rel_ind60 q2 2019-2022        4     1       2  0.161064
                                   ear q1 & rel_ind60 q2 2023-2026        4     0       2 -0.039012
                                   ear q1 & rel_ind60 q4 2015-2018        1     0       0  0.817400
                                   ear q1 & rel_ind60 q4 2019-2022        4     1       3 -0.376323
                                   ear q1 & rel_ind60 q4 2023-2026        4     1       1  0.504311
                                      ear q1 & skew60 q1 2019-2022        4     2       1  0.699242
                                      ear q1 & skew60 q1 2023-2026        4     2       0  1.195957
                                   ear q1 & vol_ratio q3 2019-2022        2     0       1 -0.074115
                                   ear q1 & vol_ratio q3 2023-2026        4     2       1  0.794837
                                                  ear q2 2015-2018        2     1       0 -1.333805
                                                  ear q2 2019-2022        4     2       2 -0.955226
                                                  ear q2 2023-2026        4     4       0 -1.836608
                                ear q2 & ear_volsurge q2 2023-2026        2     2       0 -2.640638
                                ear q2 & earn_in_week q2 2019-2022        2     1       1 -0.469556
                                ear q2 & earn_in_week q2 2023-2026        4     4       0 -1.495038
                                ear q2 & ev_agreement q2 2023-2026        2     1       0 -1.638961
                                ear q2 & ev_agreement q4 2023-2026        2     1       0 -1.930962
                                 ear q2 & ev_offering q2 2023-2026        2     1       0 -1.890336
                                 ear q2 & ev_red_flag q2 2023-2026        2     1       0 -1.796653
                                 ear q2 & ev_red_flag q4 2023-2026        2     2       0 -1.784045
                                    ear q2 & ev_shelf q2 2023-2026        2     1       0 -1.276783
                    ear q2 & frog q2 unless dist_52wh q4 2023-2026        2     1       0 -1.647759
                    ear q2 & frog q2 unless ind_mom20 q4 2023-2026        1     1       0 -1.579916
                   ear q2 & frog q2 unless vol_surge1 q4 2023-2026        1     1       0 -1.492717
                   ear q2 & frog q2 unless vol_surge5 q4 2023-2026        2     1       0 -1.261373
                                   ear q2 & gap_today q0 2023-2026        2     1       0 -0.858353
                         ear q2 & ins_opportunistic30 q2 2023-2026        2     2       0 -2.035094
                                  ear q2 & intraday20 q4 2023-2026        1     0       0 -0.953293
                                      ear q2 & log_dv q3 2023-2026        2     0       0 -0.769495
                                       ear q2 & max20 q4 2023-2026        2     2       0 -2.359383
                                       ear q2 & min20 q0 2015-2018        1     0       0 -0.683258
                                       ear q2 & min20 q0 2019-2022        4     2       0 -1.302822
                                       ear q2 & min20 q0 2023-2026        4     4       0 -2.346890
                                       ear q2 & news5 q2 2023-2026        1     0       0 -0.584060
                                          ear q2 & r1 q0 2019-2022        1     0       1  0.349654
                                          ear q2 & r1 q0 2023-2026        4     3       0 -1.851008
                                          ear q2 & r1 q4 2023-2026        1     0       0  0.643536
                                     ear q2 & r5_news q0 2023-2026        2     1       0 -2.065451
                                     ear q2 & r5_news q2 2023-2026        2     1       0 -1.530985
                                         ear q2 & r60 q0 2023-2026        2     1       0 -2.034743
                                         ear q2 & r60 q4 2023-2026        2     1       0 -1.127509
                              ear q2 & range_compress q0 2023-2026        2     2       0 -1.818419
                                   ear q2 & rel_ind60 q4 2023-2026        2     1       0 -1.363941
                                      ear q2 & skew60 q0 2023-2026        2     0       1 -0.146447
                                      ear q2 & skew60 q1 2019-2022        4     2       2  0.546936
                                      ear q2 & skew60 q1 2023-2026        4     1       2 -0.200423
                                      ear q2 & skew60 q4 2023-2026        2     1       0 -2.118372
                                       ear q2 & vol20 q4 2023-2026        2     2       0 -2.214957
                                   ear q2 & vol_ratio q2 2023-2026        2     0       1 -0.149925
                                  ear q2 & vol_surge1 q0 2023-2026        2     1       0 -1.306600
                                  ear q2 & vol_surge5 q0 2023-2026        1     1       0 -1.947378
                                  ear q2 & vol_surge5 q3 2023-2026        2     1       1  0.133770
                                                  ear q3 2015-2018        3     0       0  0.310515
                                                  ear q3 2019-2022        4     1       2 -0.144902
                                                  ear q3 2023-2026        4     1       1  0.226362
                                ear q3 & ins_buyers30 q2 2015-2018        3     0       2 -0.009706
                                ear q3 & ins_buyers30 q2 2019-2022        4     1       2 -0.104173
                                ear q3 & ins_buyers30 q2 2023-2026        4     1       1  0.385349
                               ear q3 & ins_officer30 q2 2015-2018        3     0       1 -0.025447
                               ear q3 & ins_officer30 q2 2019-2022        4     1       2 -0.202035
                               ear q3 & ins_officer30 q2 2023-2026        4     1       1  0.272007
                                 ear q3 & ins_value30 q2 2015-2018        3     0       2 -0.009706
                                 ear q3 & ins_value30 q2 2019-2022        4     1       2 -0.104173
                                 ear q3 & ins_value30 q2 2023-2026        4     1       1  0.385349
                                      ear q3 & log_dv q1 2023-2026        2     0       1 -0.053321
                                    ear q3 & mom_12_1 q3 2015-2018        3     0       0  0.328659
                                    ear q3 & mom_12_1 q3 2019-2022        4     1       1  0.397247
                                    ear q3 & mom_12_1 q3 2023-2026        4     1       1  0.521886
                                 ear q3 & overnight20 q1 2015-2018        3     1       2 -0.589280
                                 ear q3 & overnight20 q1 2019-2022        4     1       1 -0.472398
                                 ear q3 & overnight20 q1 2023-2026        4     0       2  0.490914
                                         ear q3 & r20 q3 2015-2018        3     0       1 -0.220973
                                         ear q3 & r20 q3 2019-2022        4     1       1 -0.556197
                                         ear q3 & r20 q3 2023-2026        4     0       2  0.081577
                              ear q3 & range_compress q1 2015-2018        2     1       0  0.976299
                              ear q3 & range_compress q1 2019-2022        4     0       3 -0.240755
                              ear q3 & range_compress q1 2023-2026        4     1       1  0.151905
                                  ear q3 & vol_surge1 q2 2023-2026        3     1       1 -0.449696
                                  ear q3 & vol_surge5 q2 2023-2026        2     1       1 -0.475503
                                                  ear q4 2015-2018        2     0       0  0.554780
                                                  ear q4 2019-2022        4     1       1  0.561375
                                                  ear q4 2023-2026        4     2       0  0.694412
                                 ear q4 & ev_offering q2 2023-2026        2     0       0 -0.231812
                                      ear q4 & log_dv q3 2015-2018        1     0       1  0.793850
                                      ear q4 & log_dv q3 2019-2022        4     1       1 -0.562265
                                      ear q4 & log_dv q3 2023-2026        4     0       3  0.077615
                                       ear q4 & vol20 q2 2023-2026        4     0       1 -0.038744
                                         ear_volsurge q0 2015-2018        4     0       3 -0.037680
                                         ear_volsurge q0 2019-2022        4     2       0 -0.745000
                                         ear_volsurge q0 2023-2026        4     2       1 -0.759349
                          ear_volsurge q0 & ind_mom20 q4 2019-2022        4     2       1 -0.465521
                          ear_volsurge q0 & ind_mom20 q4 2023-2026        4     1       1 -0.564878
                             ear_volsurge q0 & log_dv q4 2019-2022        4     0       3  0.463565
                             ear_volsurge q0 & log_dv q4 2023-2026        4     3       1 -1.354315
                           ear_volsurge q0 & mom_12_1 q3 2015-2018        3     1       1 -0.392372
                           ear_volsurge q0 & mom_12_1 q3 2019-2022        4     1       2 -0.237348
                           ear_volsurge q0 & mom_12_1 q3 2023-2026        4     1       2 -0.390610
                          ear_volsurge q0 & rel_ind20 q4 2019-2022        3     2       1 -1.563230
                          ear_volsurge q0 & rel_ind20 q4 2023-2026        4     2       2 -0.418418
                                         ear_volsurge q1 2023-2026        3     0       0  0.819768
                               ear_volsurge q1 & frog q4 2019-2022        4     3       1  0.968768
                               ear_volsurge q1 & frog q4 2023-2026        4     1       1  0.355758
                          ear_volsurge q1 & gap_today q0 2019-2022        3     2       1  0.631244
                          ear_volsurge q1 & gap_today q0 2023-2026        4     0       2 -0.125858
                             ear_volsurge q1 & skew60 q4 2019-2022        4     0       3 -0.204742
                             ear_volsurge q1 & skew60 q4 2023-2026        4     1       0  0.468762
                         ear_volsurge q1 & vol_surge5 q3 2023-2026        4     2       1  0.690733
                                         ear_volsurge q2 2015-2018        2     0       0 -0.341246
                                         ear_volsurge q2 2019-2022        4     1       1 -0.180898
                                         ear_volsurge q2 2023-2026        4     2       0 -1.342683
                          ear_volsurge q2 & gap_today q0 2019-2022        3     0       1 -0.050858
                          ear_volsurge q2 & gap_today q0 2023-2026        4     4       0 -1.733166
                             ear_volsurge q2 & log_dv q3 2015-2018        3     0       1 -0.406455
                             ear_volsurge q2 & log_dv q3 2019-2022        4     0       1 -0.151143
                             ear_volsurge q2 & log_dv q3 2023-2026        4     1       1 -0.437364
                           ear_volsurge q2 & mom_12_1 q3 2015-2018        3     2       1 -0.492439
                           ear_volsurge q2 & mom_12_1 q3 2019-2022        4     0       1 -0.144734
                           ear_volsurge q2 & mom_12_1 q3 2023-2026        4     0       2  0.313445
                             ear_volsurge q2 & skew60 q4 2023-2026        2     2       0 -3.222663
                         ear_volsurge q2 & vol_surge5 q1 2019-2022        3     1       0 -1.020485
                         ear_volsurge q2 & vol_surge5 q1 2023-2026        4     2       2 -0.510157
                         ear_volsurge q2 & vol_surge5 q3 2023-2026        2     0       0  0.784795
                                         ear_volsurge q3 2015-2018        1     0       1 -0.617041
                                         ear_volsurge q3 2019-2022        4     1       2  0.455872
                                         ear_volsurge q3 2023-2026        4     1       0  0.823313
                        ear_volsurge q3 & ev_offering q2 2023-2026        2     0       0  0.425631
                          ear_volsurge q3 & ind_mom20 q1 2023-2026        2     1       1 -0.448531
                         ear_volsurge q3 & intraday20 q3 2019-2022        2     1       1  0.669476
                         ear_volsurge q3 & intraday20 q3 2023-2026        4     1       2  0.344443
                         ear_volsurge q3 & intraday20 q4 2019-2022        3     1       2  0.044831
                         ear_volsurge q3 & intraday20 q4 2023-2026        4     2       0  1.135502
                              ear_volsurge q3 & min20 q2 2023-2026        2     0       0 -0.661971
                        ear_volsurge q3 & overnight20 q2 2019-2022        2     0       0  0.147483
                        ear_volsurge q3 & overnight20 q2 2023-2026        4     2       1  0.594873
                               ear_volsurge q3 & r120 q3 2019-2022        1     1       0  1.005947
                               ear_volsurge q3 & r120 q3 2023-2026        4     1       2  0.043940
                            ear_volsurge q3 & r5_news q2 2023-2026        3     0       2  0.210131
                          ear_volsurge q3 & rel_ind60 q2 2023-2026        4     0       1  0.373530
                          ear_volsurge q3 & rel_ind60 q3 2019-2022        1     0       0 -0.556182
                          ear_volsurge q3 & rel_ind60 q3 2023-2026        4     1       2 -0.042552
                              ear_volsurge q3 & vol20 q2 2019-2022        2     1       0  1.375078
                              ear_volsurge q3 & vol20 q2 2023-2026        4     2       0  0.871142
                         ear_volsurge q3 & vol_surge1 q2 2023-2026        4     1       2  0.258326
                                         ear_volsurge q4 2019-2022        1     0       1 -1.016837
                                         ear_volsurge q4 2023-2026        4     1       1  0.371568
                               ear_volsurge q4 & frog q4 2019-2022        4     1       2 -0.463337
                               ear_volsurge q4 & frog q4 2023-2026        4     1       3  0.333366
                          ear_volsurge q4 & ind_mom60 q1 2015-2018        4     1       1  0.115937
                          ear_volsurge q4 & ind_mom60 q1 2019-2022        4     1       0  0.657897
                          ear_volsurge q4 & ind_mom60 q1 2023-2026        4     0       1  0.222908
                             ear_volsurge q4 & log_dv q1 2019-2022        4     1       1 -0.304276
                             ear_volsurge q4 & log_dv q1 2023-2026        4     1       2  0.133247
                        ear_volsurge q4 & overnight20 q4 2019-2022        1     0       0 -0.353527
                        ear_volsurge q4 & overnight20 q4 2023-2026        4     1       2 -0.177487
                                ear_volsurge q4 & r20 q3 2015-2018        4     0       2  0.153367
                                ear_volsurge q4 & r20 q3 2019-2022        4     2       2 -0.505373
                                ear_volsurge q4 & r20 q3 2023-2026        4     1       0 -0.729060
                     ear_volsurge q4 & range_compress q2 2019-2022        2     1       0  0.683518
                     ear_volsurge q4 & range_compress q2 2023-2026        4     1       1  0.555293
                              ear_volsurge q4 & vol20 q3 2019-2022        4     0       2  0.009925
                              ear_volsurge q4 & vol20 q3 2023-2026        4     3       0  1.456940
                         ear_volsurge q4 & vol_surge1 q1 2019-2022        4     1       1 -0.552377
                         ear_volsurge q4 & vol_surge1 q1 2023-2026        4     0       2  0.483403
                         ear_volsurge q4 & vol_surge5 q3 2015-2018        3     1       1 -0.379633
                         ear_volsurge q4 & vol_surge5 q3 2019-2022        4     0       2  0.225698
                         ear_volsurge q4 & vol_surge5 q3 2023-2026        4     0       2 -0.054882
                       earn_in_week q2 & ins_buyers30 q2 2015-2018        4     1       3 -0.193839
                       earn_in_week q2 & ins_buyers30 q2 2019-2022        4     2       1  1.093215
                       earn_in_week q2 & ins_buyers30 q2 2023-2026        4     0       0  0.366385
                           earn_in_week q2 & mom_12_1 q3 2015-2018        3     0       3 -1.315362
                           earn_in_week q2 & mom_12_1 q3 2019-2022        4     2       2  0.479874
                           earn_in_week q2 & mom_12_1 q3 2023-2026        4     0       0  0.558882
                                 earn_in_week q4 & r5 q1 2019-2022        1     1       0 -1.613363
                                         ev_agreement q2 2015-2018        1     0       0  0.720506
                                         ev_agreement q2 2019-2022        4     3       1  1.637436
                                         ev_agreement q2 2023-2026        4     1       1  0.684013
                        ev_agreement q2 & ev_offering q2 2019-2022        4     3       1  1.612319
                        ev_agreement q2 & ev_offering q2 2023-2026        4     1       1  1.008281
                        ev_agreement q2 & ev_offering q4 2019-2022        4     0       2 -0.187285
                        ev_agreement q2 & ev_offering q4 2023-2026        3     0       1 -0.183711
                               ev_agreement q2 & frog q1 2019-2022        2     0       2 -0.495138
                               ev_agreement q2 & frog q1 2023-2026        4     0       0  0.432407
                         ev_agreement q2 & intraday20 q4 2023-2026        4     0       1 -0.123936
                                 ev_agreement q2 & r5 q1 2019-2022        3     1       1  0.471458
                                 ev_agreement q2 & r5 q1 2023-2026        4     2       2  0.593784
                          ev_agreement q2 & r5_nonews q4 2023-2026        4     1       1 -0.287427
                     ev_agreement q2 & range_compress q0 2023-2026        4     1       2 -0.344790
                          ev_agreement q2 & rel_ind60 q2 2019-2022        3     1       1  1.021167
                          ev_agreement q2 & rel_ind60 q2 2023-2026        4     1       3 -0.165613
      ev_agreement q2 & rel_ind60 q2 unless close_loc q0 2023-2026        4     1       1 -0.181963
      ev_agreement q2 & rel_ind60 q2 unless close_loc q4 2023-2026        4     0       2 -0.150159
      ev_agreement q2 & rel_ind60 q2 unless dist_52wh q0 2023-2026        4     0       1 -0.051352
      ev_agreement q2 & rel_ind60 q2 unless dist_52wh q4 2023-2026        4     1       2  0.011569
      ev_agreement q2 & rel_ind60 q2 unless dist_ma50 q0 2023-2026        4     0       2  0.035232
      ev_agreement q2 & rel_ind60 q2 unless ind_mom20 q4 2023-2026        4     1       2 -0.249583
           ev_agreement q2 & rel_ind60 q2 unless r120 q0 2023-2026        4     0       1 -0.283292
           ev_agreement q2 & rel_ind60 q2 unless r120 q4 2023-2026        4     1       1 -0.219494
        ev_agreement q2 & rel_ind60 q2 unless r5_news q0 2023-2026        4     1       1 -0.252418
     ev_agreement q2 & rel_ind60 q2 unless vol_surge1 q4 2023-2026        4     1       1 -0.516157
                          ev_agreement q2 & rel_ind60 q4 2019-2022        1     1       0 -1.007889
                          ev_agreement q2 & rel_ind60 q4 2023-2026        4     1       2 -0.328284
                              ev_agreement q2 & vol20 q2 2023-2026        2     2       0  1.378692
                              ev_agreement q2 & vol20 q3 2019-2022        4     2       2  0.287506
                              ev_agreement q2 & vol20 q3 2023-2026        4     3       0  1.879431
                          ev_agreement q2 & vol_ratio q3 2019-2022        2     0       1  0.090376
                          ev_agreement q2 & vol_ratio q3 2023-2026        4     1       1  0.683422
                          ev_agreement q2 & vol_ratio q4 2015-2018        2     1       1  0.181020
                          ev_agreement q2 & vol_ratio q4 2019-2022        4     1       1 -0.423363
                          ev_agreement q2 & vol_ratio q4 2023-2026        4     2       2 -0.168763
                         ev_agreement q2 & vol_surge5 q0 2023-2026        4     2       1 -0.396920
                         ev_agreement q2 & vol_surge5 q2 2023-2026        4     1       0  0.774752
                         ev_agreement q2 & vol_surge5 q3 2019-2022        1     1       0  1.553127
                         ev_agreement q2 & vol_surge5 q3 2023-2026        4     2       1  1.088825
                         ev_agreement q2 & vol_surge5 q4 2023-2026        3     1       1  0.111954
                                         ev_agreement q4 2019-2022        4     2       1 -1.319098
                                         ev_agreement q4 2023-2026        4     1       1 -0.861479
                           ev_agreement q4 & ev_shelf q2 2019-2022        4     2       1 -1.362488
                           ev_agreement q4 & ev_shelf q2 2023-2026        4     1       2 -0.533733
                          ev_agreement q4 & ind_mom60 q1 2023-2026        2     1       1 -0.886775
                           ev_agreement q4 & mom_12_1 q3 2023-2026        3     1       1  0.530480
                                 ev_agreement q4 & r5 q3 2023-2026        3     0       1 -0.209383
                     ev_agreement q4 & range_compress q0 2019-2022        3     1       2 -0.258847
                     ev_agreement q4 & range_compress q0 2023-2026        4     1       2 -0.481329
                          ev_agreement q4 & rel_ind60 q4 2019-2022        4     2       1 -0.806360
                          ev_agreement q4 & rel_ind60 q4 2023-2026        4     2       1 -1.731293
                         ev_agreement q4 & vol_surge1 q1 2019-2022        2     0       2  0.704122
                         ev_agreement q4 & vol_surge1 q1 2023-2026        3     1       0 -0.875433
                         ev_agreement q4 & vol_surge1 q2 2023-2026        2     0       0  0.479845
                         ev_agreement q4 & vol_surge1 q4 2019-2022        3     1       2 -0.290266
                         ev_agreement q4 & vol_surge1 q4 2023-2026        4     0       0 -0.376448
                                          ev_offering q2 2019-2022        2     0       0  0.669465
                                          ev_offering q2 2023-2026        4     2       2  0.718694
                            ev_offering q2 & ev_shelf q2 2023-2026        3     2       1  1.864449
                          ev_offering q2 & intraday20 q4 2023-2026        4     0       1 -0.158956
                               ev_offering q2 & min20 q1 2023-2026        3     1       1  0.505663
                           ev_offering q2 & rel_ind60 q2 2019-2022        3     1       1  0.926910
                           ev_offering q2 & rel_ind60 q2 2023-2026        4     0       2 -0.109253
                           ev_offering q2 & rel_ind60 q4 2019-2022        1     1       0 -1.032619
                           ev_offering q2 & rel_ind60 q4 2023-2026        4     1       2 -0.354404
                              ev_offering q2 & skew60 q1 2019-2022        1     0       0  0.685773
                              ev_offering q2 & skew60 q1 2023-2026        4     3       0  1.168440
                               ev_offering q2 & vol20 q2 2023-2026        2     2       0  1.161383
                               ev_offering q2 & vol20 q3 2019-2022        4     2       2  0.322473
                               ev_offering q2 & vol20 q3 2023-2026        4     2       0  1.680471
                          ev_offering q2 & vol_surge5 q3 2023-2026        4     2       0  1.015435
                                          ev_red_flag q2 2019-2022        1     1       0  1.051464
                                          ev_red_flag q2 2023-2026        4     1       0  0.892534
                                ev_red_flag q2 & frog q0 2019-2022        4     1       2  0.767714
                                ev_red_flag q2 & frog q0 2023-2026        4     0       2 -0.512193
                                ev_red_flag q2 & frog q1 2019-2022        2     0       0 -0.474834
                                ev_red_flag q2 & frog q1 2023-2026        4     0       1  0.084639
                               ev_red_flag q2 & news5 q2 2019-2022        2     1       0  0.851482
                               ev_red_flag q2 & news5 q2 2023-2026        4     1       0  0.975264
                                  ev_red_flag q2 & r5 q1 2019-2022        4     1       1  0.458239
                                  ev_red_flag q2 & r5 q1 2023-2026        4     2       2  0.657527
                                 ev_red_flag q2 & r60 q3 2023-2026        1     0       0  0.407669
                      ev_red_flag q2 & range_compress q2 2019-2022        4     0       1 -0.268286
                      ev_red_flag q2 & range_compress q2 2023-2026        4     1       2 -0.065891
                           ev_red_flag q2 & rel_ind20 q4 2019-2022        3     1       1 -0.554666
                           ev_red_flag q2 & rel_ind20 q4 2023-2026        4     1       3  0.036234
                           ev_red_flag q2 & rel_ind60 q2 2019-2022        3     1       1  0.798243
                           ev_red_flag q2 & rel_ind60 q2 2023-2026        4     0       2 -0.250519
                               ev_red_flag q2 & vol20 q3 2019-2022        4     2       2  0.234023
                               ev_red_flag q2 & vol20 q3 2023-2026        4     2       0  1.636523
                           ev_red_flag q2 & vol_ratio q3 2019-2022        2     1       1  0.255775
                           ev_red_flag q2 & vol_ratio q3 2023-2026        4     2       1  0.771235
                                          ev_red_flag q4 2019-2022        1     0       0 -0.921040
                                          ev_red_flag q4 2023-2026        4     1       1 -0.797295
                                             ev_shelf q2 2015-2018        3     0       2 -0.871940
                                             ev_shelf q2 2019-2022        4     0       0  0.662056
                                             ev_shelf q2 2023-2026        4     3       0  2.657486
                                  ev_shelf q2 & max20 q3 2019-2022        4     0       3 -0.131894
                                  ev_shelf q2 & max20 q3 2023-2026        4     1       1  0.944265
                                  ev_shelf q2 & min20 q1 2023-2026        3     1       1  0.561802
                                  ev_shelf q2 & news5 q2 2019-2022        2     1       0  1.029773
                                  ev_shelf q2 & news5 q2 2023-2026        4     3       0  1.572115
                                    ev_shelf q2 & r20 q1 2023-2026        3     1       2  0.141442
                                    ev_shelf q2 & r60 q3 2023-2026        1     0       0  0.479762
                                  ev_shelf q2 & vol20 q3 2023-2026        3     1       1  0.329300
                              ev_shelf q2 & vol_ratio q3 2019-2022        2     0       1  0.213911
                              ev_shelf q2 & vol_ratio q3 2023-2026        4     1       1  0.674809
                             ev_shelf q2 & vol_surge1 q3 2015-2018        2     1       0  1.045631
                             ev_shelf q2 & vol_surge1 q3 2019-2022        4     2       2  0.305948
                             ev_shelf q2 & vol_surge1 q3 2023-2026        4     0       2  0.154154
                                                 frog q0 2019-2022        4     0       2  0.630835
                                                 frog q0 2023-2026        4     2       0 -0.940206
                                  frog q0 & gap_today q4 2023-2026        3     0       1 -0.091572
                                  frog q0 & ind_mom20 q4 2019-2022        3     0       2  0.216956
                                  frog q0 & ind_mom20 q4 2023-2026        4     2       1 -1.123241
                                 frog q0 & intraday20 q4 2019-2022        3     1       0 -1.083143
                                 frog q0 & intraday20 q4 2023-2026        4     3       0 -1.137489
                                     frog q0 & log_dv q3 2019-2022        4     2       1  0.789476
                                     frog q0 & log_dv q3 2023-2026        4     0       3 -0.604059
                                      frog q0 & max20 q1 2023-2026        4     1       1  0.255147
                                         frog q0 & r1 q0 2015-2018        4     1       2  0.173270
                                         frog q0 & r1 q0 2019-2022        4     1       2 -0.228462
                                         frog q0 & r1 q0 2023-2026        4     2       1 -0.847847
                                  frog q0 & r5_nonews q4 2019-2022        1     1       0 -1.187915
                                  frog q0 & r5_nonews q4 2023-2026        4     2       2 -0.629271
                                        frog q0 & r60 q2 2015-2018        2     0       0 -0.171008
                                        frog q0 & r60 q2 2019-2022        4     0       1 -0.384039
                                        frog q0 & r60 q2 2023-2026        4     1       2 -0.367812
                             frog q0 & range_compress q0 2019-2022        2     1       0 -1.234196
                             frog q0 & range_compress q0 2023-2026        4     2       0 -1.091547
                                  frog q0 & rel_ind60 q0 2015-2018        4     2       1 -0.708166
                                  frog q0 & rel_ind60 q0 2019-2022        4     0       3  0.453775
                                  frog q0 & rel_ind60 q0 2023-2026        4     0       1 -0.127644
                                  frog q0 & vol_ratio q2 2023-2026        3     1       1  0.658873
                                 frog q0 & vol_surge5 q1 2019-2022        3     0       1  0.237049
                                 frog q0 & vol_surge5 q1 2023-2026        4     1       1  0.445507
                                                 frog q1 2019-2022        2     0       0 -0.574412
                                                 frog q1 2023-2026        4     0       2  0.077339
                              frog q1 & ins_officer30 q2 2019-2022        2     0       0 -0.509395
                              frog q1 & ins_officer30 q2 2023-2026        4     0       2  0.244705
                                 frog q1 & intraday20 q1 2019-2022        2     1       0  0.914516
                                 frog q1 & intraday20 q1 2023-2026        4     1       2  0.073218
                                 frog q1 & intraday20 q4 2023-2026        4     1       0  0.463572
                                     frog q1 & log_dv q3 2015-2018        1     0       1  1.065175
                                     frog q1 & log_dv q3 2019-2022        4     0       2  0.340196
                                     frog q1 & log_dv q3 2023-2026        4     1       0 -0.734659
                                      frog q1 & max20 q0 2019-2022        1     0       0  0.719815
                                      frog q1 & max20 q0 2023-2026        4     0       3 -0.064268
                                      frog q1 & min20 q1 2023-2026        3     0       1  0.355736
                                      frog q1 & min20 q3 2023-2026        3     0       1  0.352936
                                      frog q1 & news5 q2 2019-2022        2     0       2 -0.716497
                                      frog q1 & news5 q2 2023-2026        4     1       1  0.460366
                                         frog q1 & r5 q1 2015-2018        1     0       1 -0.475660
                                         frog q1 & r5 q1 2019-2022        4     1       0  1.032326
                                         frog q1 & r5 q1 2023-2026        4     0       2 -0.042235
                             frog q1 & range_compress q2 2019-2022        2     1       1 -0.046813
                             frog q1 & range_compress q2 2023-2026        4     1       1  0.622199
                             frog q1 & range_compress q3 2023-2026        3     1       1 -0.366262
                                  frog q1 & rel_ind60 q1 2015-2018        3     1       1  0.619681
                                  frog q1 & rel_ind60 q1 2019-2022        4     2       1  0.790855
                                  frog q1 & rel_ind60 q1 2023-2026        4     0       2  0.104265
                                  frog q1 & rel_ind60 q2 2019-2022        3     1       0  0.653826
                                  frog q1 & rel_ind60 q2 2023-2026        4     2       1  0.980111
                                  frog q1 & rel_ind60 q4 2023-2026        2     1       1 -0.600240
                                      frog q1 & vol20 q2 2023-2026        2     0       0  0.468407
                                 frog q1 & vol_surge1 q1 2023-2026        4     1       2  0.247804
                                 frog q1 & vol_surge5 q3 2019-2022        1     1       0 -1.598418
                                 frog q1 & vol_surge5 q3 2023-2026        4     0       2 -0.121631
                                                 frog q2 2023-2026        4     1       0 -0.722088
                                  frog q2 & gap_today q0 2019-2022        3     0       1 -0.220676
                                  frog q2 & gap_today q0 2023-2026        4     0       0 -0.587255
                                     frog q2 & log_dv q0 2023-2026        3     2       0 -0.997289
                                     frog q2 & log_dv q3 2019-2022        2     0       1  0.133932
                                     frog q2 & log_dv q3 2023-2026        4     1       0 -1.098849
                                      frog q2 & min20 q1 2023-2026        3     0       1 -0.013768
                                      frog q2 & min20 q3 2019-2022        2     1       0  1.879575
                                      frog q2 & min20 q3 2023-2026        4     0       3 -0.582658
                                frog q2 & overnight20 q3 2019-2022        2     0       1 -0.042736
                                frog q2 & overnight20 q3 2023-2026        4     1       2  0.132373
                                        frog q2 & r20 q4 2019-2022        3     1       0 -0.585384
                                        frog q2 & r20 q4 2023-2026        4     0       2 -0.098573
                                  frog q2 & r5_nonews q3 2019-2022        2     1       1 -0.323408
                                  frog q2 & r5_nonews q3 2023-2026        4     2       1 -0.691744
                             frog q2 & range_compress q0 2019-2022        2     1       0 -1.120522
                             frog q2 & range_compress q0 2023-2026        4     0       1 -0.014287
                                  frog q2 & rel_ind20 q0 2023-2026        3     0       1  0.034086
                                  frog q2 & rel_ind20 q4 2019-2022        3     1       0 -0.728172
                                  frog q2 & rel_ind20 q4 2023-2026        4     0       0 -0.562804
                                  frog q2 & rel_ind60 q2 2019-2022        3     1       1 -0.471161
                                  frog q2 & rel_ind60 q2 2023-2026        4     2       1 -0.997357
                                     frog q2 & skew60 q0 2023-2026        2     0       0  0.243610
                                      frog q2 & vol20 q3 2023-2026        3     1       2 -0.134896
                                 frog q2 & vol_surge1 q0 2023-2026        1     1       0 -1.028209
                                 frog q2 & vol_surge1 q2 2023-2026        2     0       0  0.299929
                                  frog q3 & gap_today q4 2019-2022        2     1       1  0.302287
                                  frog q3 & gap_today q4 2023-2026        4     0       1  0.131238
                                  frog q3 & ind_mom20 q2 2023-2026        4     1       2 -0.336758
                                  frog q3 & ind_mom20 q4 2019-2022        2     1       1  0.204402
                                  frog q3 & ind_mom20 q4 2023-2026        4     2       1  0.798907
                                 frog q3 & intraday20 q4 2023-2026        2     1       0  1.466153
                                     frog q3 & log_dv q3 2019-2022        2     0       1 -0.093852
                                     frog q3 & log_dv q3 2023-2026        4     2       1  0.472178
                                      frog q3 & max20 q1 2015-2018        2     0       2 -0.941827
                                      frog q3 & max20 q1 2019-2022        4     2       1  0.472097
                                      frog q3 & max20 q1 2023-2026        4     1       0  0.653724
                                      frog q3 & min20 q3 2019-2022        2     2       0  1.482621
                                      frog q3 & min20 q3 2023-2026        4     1       2  0.235368
                                   frog q3 & mom_12_1 q2 2015-2018        4     0       1  0.236820
                                   frog q3 & mom_12_1 q2 2019-2022        4     1       2  0.101452
                                   frog q3 & mom_12_1 q2 2023-2026        4     1       3  0.226005
                                         frog q3 & r1 q4 2015-2018        2     1       0 -1.075303
                                         frog q3 & r1 q4 2019-2022        4     1       0 -1.186289
                                         frog q3 & r1 q4 2023-2026        4     0       2  0.167476
                                        frog q3 & r20 q4 2019-2022        3     0       3 -0.938071
                                        frog q3 & r20 q4 2023-2026        4     2       1  0.800056
                                  frog q3 & vol_ratio q4 2019-2022        2     1       1 -0.561607
                                  frog q3 & vol_ratio q4 2023-2026        4     2       2  0.056664
                                 frog q3 & vol_surge5 q1 2019-2022        3     1       1 -0.642850
                                 frog q3 & vol_surge5 q1 2023-2026        4     0       3  0.408012
                                                 frog q4 2019-2022        3     0       2 -0.089198
                                                 frog q4 2023-2026        4     1       1  0.721580
                                  frog q4 & ind_mom60 q2 2019-2022        4     2       1  0.952690
                                  frog q4 & ind_mom60 q2 2023-2026        4     1       1  0.275677
                                  frog q4 & r5_nonews q0 2015-2018        3     2       1  0.382580
                                  frog q4 & r5_nonews q0 2019-2022        4     0       1  0.112571
                                  frog q4 & r5_nonews q0 2023-2026        4     1       1  0.480195
                                      frog q4 & vol20 q1 2015-2018        4     0       2 -0.281078
                                      frog q4 & vol20 q1 2019-2022        4     2       1  0.916972
                                      frog q4 & vol20 q1 2023-2026        4     1       1  0.588782
                                  frog q4 & vol_ratio q1 2019-2022        4     0       2 -0.473256
                                  frog q4 & vol_ratio q1 2023-2026        4     0       0  0.556958
                                 frog q4 & vol_surge5 q3 2023-2026        2     1       0  1.438085
                                            gap_today q0 2015-2018        2     0       2  0.124056
                                            gap_today q0 2019-2022        4     1       1 -0.361025
                                            gap_today q0 2023-2026        4     2       0 -0.805661
                             gap_today q0 & ind_mom20 q4 2019-2022        4     1       2  0.215185
                             gap_today q0 & ind_mom20 q4 2023-2026        4     1       2  0.274736
                   gap_today q0 & ins_opportunistic30 q2 2019-2022        3     1       1 -0.495346
                   gap_today q0 & ins_opportunistic30 q2 2023-2026        4     1       0 -0.730127
                            gap_today q0 & intraday20 q4 2019-2022        3     1       1 -0.683029
                            gap_today q0 & intraday20 q4 2023-2026        4     1       2 -0.399016
                                gap_today q0 & log_dv q2 2015-2018        1     1       0  1.137464
                                gap_today q0 & log_dv q2 2019-2022        4     1       2  0.067720
                                gap_today q0 & log_dv q2 2023-2026        4     0       0  0.481369
                                gap_today q0 & log_dv q3 2019-2022        3     1       1 -0.891348
                                gap_today q0 & log_dv q3 2023-2026        4     0       1 -0.461133
                                 gap_today q0 & max20 q3 2019-2022        3     1       2  0.689204
                                 gap_today q0 & max20 q3 2023-2026        4     1       0  0.679328
                                 gap_today q0 & min20 q1 2015-2018        1     0       0  0.519394
                                 gap_today q0 & min20 q1 2019-2022        4     1       2  0.128021
                                 gap_today q0 & min20 q1 2023-2026        4     4       0  1.549989
                              gap_today q0 & mom_12_1 q2 2019-2022        1     0       1  0.299731
                              gap_today q0 & mom_12_1 q2 2023-2026        4     2       1 -0.984530
                           gap_today q0 & overnight20 q1 2019-2022        3     0       1  0.034504
                           gap_today q0 & overnight20 q1 2023-2026        4     1       1  0.585445
                           gap_today q0 & overnight20 q3 2015-2018        1     0       0  0.385609
                           gap_today q0 & overnight20 q3 2019-2022        4     2       2  0.849572
                           gap_today q0 & overnight20 q3 2023-2026        4     0       3 -0.398888
                                    gap_today q0 & r1 q0 2019-2022        3     0       1 -0.001705
                                    gap_today q0 & r1 q0 2023-2026        4     1       2 -0.444676
                                  gap_today q0 & r120 q3 2015-2018        1     0       1 -0.622972
                                  gap_today q0 & r120 q3 2019-2022        4     0       1  0.561832
                                  gap_today q0 & r120 q3 2023-2026        4     0       2 -0.146737
                                   gap_today q0 & r20 q0 2019-2022        3     0       1  0.182693
                                   gap_today q0 & r20 q0 2023-2026        4     2       1 -0.920675
                                   gap_today q0 & r20 q4 2019-2022        3     1       2 -0.302012
                                   gap_today q0 & r20 q4 2023-2026        4     1       1 -0.404788
                                    gap_today q0 & r5 q1 2019-2022        3     1       1  0.342784
                                    gap_today q0 & r5 q1 2023-2026        4     1       2 -0.199097
                             gap_today q0 & r5_nonews q1 2019-2022        3     1       2  0.443231
                             gap_today q0 & r5_nonews q1 2023-2026        4     1       2 -0.137797
                             gap_today q0 & r5_nonews q3 2019-2022        1     0       0  0.157078
                             gap_today q0 & r5_nonews q3 2023-2026        4     1       1  0.666785
                                   gap_today q0 & r60 q4 2019-2022        3     1       0 -1.125897
                                   gap_today q0 & r60 q4 2023-2026        4     2       0 -1.087329
                        gap_today q0 & range_compress q1 2015-2018        1     0       1 -0.738760
                        gap_today q0 & range_compress q1 2019-2022        4     2       2  0.839010
                        gap_today q0 & range_compress q1 2023-2026        4     0       3 -0.534267
                             gap_today q0 & rel_ind20 q0 2019-2022        3     1       1 -0.406613
                             gap_today q0 & rel_ind20 q0 2023-2026        4     2       1 -0.737712
                             gap_today q0 & rel_ind20 q4 2019-2022        3     1       1 -0.331681
                             gap_today q0 & rel_ind20 q4 2023-2026        4     1       0 -0.620048
                             gap_today q0 & rel_ind60 q4 2019-2022        3     1       0 -0.964389
                             gap_today q0 & rel_ind60 q4 2023-2026        4     1       0 -0.963065
                            gap_today q0 & vol_surge1 q1 2019-2022        4     1       1 -0.165161
                            gap_today q0 & vol_surge1 q1 2023-2026        4     1       2 -0.397157
                            gap_today q1 & intraday20 q2 2019-2022        1     0       1 -0.791399
                            gap_today q1 & intraday20 q2 2023-2026        4     2       1  0.541399
                            gap_today q1 & intraday20 q4 2023-2026        4     0       2 -0.061772
                              gap_today q1 & mom_12_1 q0 2019-2022        3     1       0 -0.739559
                              gap_today q1 & mom_12_1 q0 2023-2026        4     0       1 -0.195488
                                   gap_today q1 & r20 q2 2019-2022        1     0       1 -0.487412
                                   gap_today q1 & r20 q2 2023-2026        4     1       2  0.438426
                             gap_today q1 & r5_nonews q4 2023-2026        4     1       1 -0.347744
                            gap_today q1 & vol_surge5 q0 2023-2026        4     2       2  0.602270
                                            gap_today q2 2015-2018        3     0       1  0.136773
                                            gap_today q2 2019-2022        4     2       1  0.518373
                                            gap_today q2 2023-2026        4     0       3 -0.200014
                                gap_today q2 & log_dv q1 2019-2022        2     0       1 -0.091141
                                gap_today q2 & log_dv q1 2023-2026        4     0       2 -0.053390
                                 gap_today q2 & min20 q3 2019-2022        1     0       1  0.026193
                                 gap_today q2 & min20 q3 2023-2026        4     0       0 -0.346494
                    gap_today q2 & r60 q0 unless frog q4 2019-2022        1     0       1  0.553154
                    gap_today q2 & r60 q0 unless frog q4 2023-2026        4     1       2 -0.171681
                 gap_today q2 & r60 q0 unless r5_news q0 2019-2022        1     0       0  0.567550
                 gap_today q2 & r60 q0 unless r5_news q0 2023-2026        4     0       2  0.042721
               gap_today q2 & r60 q0 unless r5_nonews q4 2019-2022        1     0       1  0.201041
               gap_today q2 & r60 q0 unless r5_nonews q4 2023-2026        4     1       2 -0.221306
                             gap_today q2 & rel_ind20 q1 2023-2026        4     2       1  0.595444
                                            gap_today q3 2015-2018        4     0       3 -0.199179
                                            gap_today q3 2019-2022        4     2       1  0.647560
                                            gap_today q3 2023-2026        4     0       1  0.221652
                             gap_today q3 & ind_mom60 q0 2015-2018        4     0       2  0.069306
                             gap_today q3 & ind_mom60 q0 2019-2022        4     0       2  0.441915
                             gap_today q3 & ind_mom60 q0 2023-2026        4     2       1 -0.792860
                            gap_today q3 & intraday20 q2 2019-2022        4     1       1  0.512895
                            gap_today q3 & intraday20 q2 2023-2026        4     0       2 -0.230314
                            gap_today q3 & intraday20 q3 2015-2018        2     0       1  0.096704
                            gap_today q3 & intraday20 q3 2019-2022        4     2       2  0.279204
                            gap_today q3 & intraday20 q3 2023-2026        4     1       3 -0.320626
                                gap_today q3 & log_dv q3 2019-2022        3     2       0  1.194006
                                gap_today q3 & log_dv q3 2023-2026        4     0       2 -0.142900
                                 gap_today q3 & min20 q2 2019-2022        4     1       1  0.789521
                                 gap_today q3 & min20 q2 2023-2026        4     2       2  0.182499
                                 gap_today q3 & min20 q4 2019-2022        2     2       0  1.181531
                                 gap_today q3 & min20 q4 2023-2026        4     0       2 -0.330259
                              gap_today q3 & mom_12_1 q2 2015-2018        4     0       2  0.104465
                              gap_today q3 & mom_12_1 q2 2019-2022        4     0       1 -0.218856
                              gap_today q3 & mom_12_1 q2 2023-2026        4     1       1 -0.286663
                                  gap_today q3 & r120 q3 2019-2022        1     0       0  0.928258
                                  gap_today q3 & r120 q3 2023-2026        4     1       2  0.345864
                                    gap_today q3 & r5 q3 2019-2022        3     1       1  0.391388
                                    gap_today q3 & r5 q3 2023-2026        4     2       1  0.972473
                               gap_today q3 & r5_news q4 2019-2022        3     0       2  0.235752
                               gap_today q3 & r5_news q4 2023-2026        3     1       0 -1.286926
                        gap_today q3 & range_compress q3 2019-2022        4     3       1  0.383039
                        gap_today q3 & range_compress q3 2023-2026        4     1       1  0.723781
                                 gap_today q3 & vol20 q3 2023-2026        3     1       2  0.410802
                            gap_today q3 & vol_surge5 q1 2019-2022        3     2       0  1.038894
                            gap_today q3 & vol_surge5 q1 2023-2026        4     0       2  0.200011
                            gap_today q3 & vol_surge5 q3 2015-2018        4     0       2 -0.075093
                            gap_today q3 & vol_surge5 q3 2019-2022        4     2       2  0.721788
                            gap_today q3 & vol_surge5 q3 2023-2026        4     2       2  0.019042
                                            gap_today q4 2019-2022        3     0       1 -0.139016
                                            gap_today q4 2023-2026        4     0       2  0.011406
                                gap_today q4 & log_dv q1 2019-2022        2     0       1  0.262770
                                gap_today q4 & log_dv q1 2023-2026        4     0       2  0.010119
                                gap_today q4 & log_dv q3 2015-2018        3     1       0 -0.884536
                                gap_today q4 & log_dv q3 2019-2022        4     1       2 -0.221818
                                gap_today q4 & log_dv q3 2023-2026        4     1       2 -0.505102
                                 gap_today q4 & min20 q4 2019-2022        2     0       0  0.511878
                                 gap_today q4 & min20 q4 2023-2026        4     0       3 -0.216141
                             gap_today q4 & r5_nonews q2 2019-2022        2     1       1  0.983411
                             gap_today q4 & r5_nonews q2 2023-2026        4     1       2 -0.126920
                        gap_today q4 & range_compress q0 2015-2018        2     1       0 -1.430302
                        gap_today q4 & range_compress q0 2019-2022        4     0       1 -0.043878
                        gap_today q4 & range_compress q0 2023-2026        4     2       2 -0.563288
                             gap_today q4 & rel_ind60 q0 2015-2018        4     2       1 -0.924044
                             gap_today q4 & rel_ind60 q0 2019-2022        4     0       1 -0.364310
                             gap_today q4 & rel_ind60 q0 2023-2026        4     2       0 -0.912100
                             gap_today q4 & rel_ind60 q2 2019-2022        3     2       1 -0.900325
                             gap_today q4 & rel_ind60 q2 2023-2026        4     0       3  0.474548
                                gap_today q4 & skew60 q4 2023-2026        2     0       0 -0.226374
                             gap_today q4 & vol_ratio q3 2019-2022        2     1       0  0.618389
                             gap_today q4 & vol_ratio q3 2023-2026        4     0       2 -0.095400
                            gap_today q4 & vol_surge1 q3 2019-2022        2     0       1  0.633153
                            gap_today q4 & vol_surge1 q3 2023-2026        4     1       1 -0.426294
                                            ind_mom20 q0 2015-2018        1     0       0 -0.521721
                                            ind_mom20 q0 2019-2022        4     0       3  0.746529
                                            ind_mom20 q0 2023-2026        4     2       2 -0.675698
                            ind_mom20 q0 & intraday20 q2 2015-2018        3     1       1 -0.755426
                            ind_mom20 q0 & intraday20 q2 2019-2022        4     0       3  0.782369
                            ind_mom20 q0 & intraday20 q2 2023-2026        4     2       2 -0.345708
                              ind_mom20 q0 & mom_12_1 q3 2015-2018        3     0       2 -0.953626
                              ind_mom20 q0 & mom_12_1 q3 2019-2022        4     1       1  0.568163
                              ind_mom20 q0 & mom_12_1 q3 2023-2026        4     1       1  0.191776
                            ind_mom20 q0 & vol_surge5 q3 2019-2022        3     2       1  1.280177
                            ind_mom20 q0 & vol_surge5 q3 2023-2026        4     0       3 -0.696498
                                            ind_mom20 q1 2019-2022        2     0       1  0.213037
                                            ind_mom20 q1 2023-2026        4     0       1 -0.407874
                            ind_mom20 q1 & intraday20 q4 2023-2026        2     0       1 -0.275261
                             ind_mom20 q1 & rel_ind60 q2 2019-2022        1     1       0  3.097933
                             ind_mom20 q1 & rel_ind60 q2 2023-2026        4     1       1 -0.049486
                                            ind_mom20 q2 2015-2018        3     1       2 -0.389413
                                            ind_mom20 q2 2019-2022        4     1       1  0.537002
                                            ind_mom20 q2 2023-2026        4     1       1  0.757838
                          ind_mom20 q2 & ins_buyers30 q2 2015-2018        3     1       2 -0.196024
                          ind_mom20 q2 & ins_buyers30 q2 2019-2022        4     1       1  0.531888
                          ind_mom20 q2 & ins_buyers30 q2 2023-2026        4     1       1  0.730138
                         ind_mom20 q2 & ins_officer30 q2 2015-2018        3     0       2 -0.385514
                         ind_mom20 q2 & ins_officer30 q2 2019-2022        4     1       1  0.460351
                         ind_mom20 q2 & ins_officer30 q2 2023-2026        4     2       1  0.864195
                           ind_mom20 q2 & ins_value30 q2 2015-2018        3     1       2 -0.196024
                           ind_mom20 q2 & ins_value30 q2 2019-2022        4     1       1  0.531888
                           ind_mom20 q2 & ins_value30 q2 2023-2026        4     1       1  0.730138
                                ind_mom20 q2 & log_dv q0 2015-2018        3     1       0 -1.039683
                                ind_mom20 q2 & log_dv q0 2019-2022        4     2       2 -0.528670
                                ind_mom20 q2 & log_dv q0 2023-2026        4     0       3  1.005297
                                ind_mom20 q2 & log_dv q1 2019-2022        2     1       1  0.346820
                                ind_mom20 q2 & log_dv q1 2023-2026        4     1       3 -0.042783
                                 ind_mom20 q2 & news5 q2 2019-2022        1     0       1 -1.123078
                                 ind_mom20 q2 & news5 q2 2023-2026        4     2       1  0.310215
                           ind_mom20 q2 & overnight20 q2 2015-2018        2     1       1  0.553393
                           ind_mom20 q2 & overnight20 q2 2019-2022        4     2       2  0.563514
                           ind_mom20 q2 & overnight20 q2 2023-2026        4     0       1  0.353458
                                    ind_mom20 q2 & r5 q1 2015-2018        3     1       0  0.827067
                                    ind_mom20 q2 & r5 q1 2019-2022        4     1       2 -0.097785
                                    ind_mom20 q2 & r5 q1 2023-2026        4     1       1  0.817145
                             ind_mom20 q2 & rel_ind20 q3 2015-2018        4     2       2  0.229671
                             ind_mom20 q2 & rel_ind20 q3 2019-2022        4     1       1  0.426108
                             ind_mom20 q2 & rel_ind20 q3 2023-2026        4     1       1  0.911125
                            ind_mom20 q2 & vol_surge5 q1 2015-2018        3     1       2  0.460784
                            ind_mom20 q2 & vol_surge5 q1 2019-2022        4     1       0  0.882265
                            ind_mom20 q2 & vol_surge5 q1 2023-2026        4     1       2  0.341688
                                ind_mom20 q3 & log_dv q3 2015-2018        2     1       1 -0.664867
                                ind_mom20 q3 & log_dv q3 2019-2022        4     3       0 -1.069567
                                ind_mom20 q3 & log_dv q3 2023-2026        4     0       2 -0.327709
                                    ind_mom20 q3 & r5 q3 2019-2022        2     1       0 -0.962524
                                    ind_mom20 q3 & r5 q3 2023-2026        4     0       2  0.441619
                             ind_mom20 q3 & rel_ind20 q3 2015-2018        4     1       3 -0.458763
                             ind_mom20 q3 & rel_ind20 q3 2019-2022        4     2       0 -1.259929
                             ind_mom20 q3 & rel_ind20 q3 2023-2026        4     0       2 -0.029156
                            ind_mom20 q3 & vol_surge5 q3 2015-2018        3     0       1 -0.066349
                            ind_mom20 q3 & vol_surge5 q3 2019-2022        4     0       2 -0.063397
                            ind_mom20 q3 & vol_surge5 q3 2023-2026        4     1       1  0.758237
                            ind_mom20 q4 & intraday20 q3 2019-2022        2     0       1  0.095197
                            ind_mom20 q4 & intraday20 q3 2023-2026        4     0       2  0.123876
                            ind_mom20 q4 & intraday20 q4 2019-2022        3     0       0 -0.852737
                            ind_mom20 q4 & intraday20 q4 2023-2026        4     0       3  0.384478
                                ind_mom20 q4 & log_dv q0 2019-2022        4     1       2 -0.095569
                                ind_mom20 q4 & log_dv q0 2023-2026        4     2       2  0.414159
                                ind_mom20 q4 & log_dv q3 2019-2022        2     1       1  0.384649
                                ind_mom20 q4 & log_dv q3 2023-2026        4     1       1 -0.325459
                                   ind_mom20 q4 & r20 q3 2015-2018        1     0       1 -1.567034
                                   ind_mom20 q4 & r20 q3 2019-2022        4     0       2 -0.211194
                                   ind_mom20 q4 & r20 q3 2023-2026        4     1       0  0.882517
                                    ind_mom20 q4 & r5 q4 2015-2018        1     1       0 -1.218305
                                    ind_mom20 q4 & r5 q4 2019-2022        4     1       0 -0.937812
                                    ind_mom20 q4 & r5 q4 2023-2026        4     0       3  0.394827
                               ind_mom20 q4 & r5_news q4 2019-2022        4     3       1 -0.654835
                               ind_mom20 q4 & r5_news q4 2023-2026        4     1       1 -0.461785
                             ind_mom20 q4 & r5_nonews q0 2019-2022        4     1       1  0.492457
                             ind_mom20 q4 & r5_nonews q0 2023-2026        4     1       2  0.339150
                             ind_mom20 q4 & rel_ind20 q4 2019-2022        3     0       1 -0.178203
                             ind_mom20 q4 & rel_ind20 q4 2023-2026        4     0       2  0.080310
                             ind_mom20 q4 & vol_ratio q1 2023-2026        3     1       1 -0.494373
                             ind_mom20 q4 & vol_ratio q4 2015-2018        4     1       2 -0.292216
                             ind_mom20 q4 & vol_ratio q4 2019-2022        4     2       1 -0.827982
                             ind_mom20 q4 & vol_ratio q4 2023-2026        4     1       1  0.113482
                            ind_mom20 q4 & vol_surge1 q1 2023-2026        3     0       1 -0.074597
                              ind_mom60 q0 & mom_12_1 q3 2019-2022        4     1       2  0.201098
                              ind_mom60 q0 & mom_12_1 q3 2023-2026        4     1       1 -0.300174
                                   ind_mom60 q0 & r20 q2 2015-2018        2     0       1  0.933619
                                   ind_mom60 q0 & r20 q2 2019-2022        4     2       2 -0.213192
                                   ind_mom60 q0 & r20 q2 2023-2026        4     1       2 -0.404593
                                   ind_mom60 q0 & r20 q4 2019-2022        4     1       1  0.447421
                                   ind_mom60 q0 & r20 q4 2023-2026        4     2       2  0.316402
                             ind_mom60 q0 & rel_ind60 q2 2019-2022        2     1       0  1.135049
                             ind_mom60 q0 & rel_ind60 q2 2023-2026        4     0       4 -0.551401
                                            ind_mom60 q1 2015-2018        1     0       0  0.186666
                                            ind_mom60 q1 2019-2022        4     0       3 -0.439327
                                            ind_mom60 q1 2023-2026        4     1       1  0.544457
                            ind_mom60 q1 & intraday20 q2 2019-2022        1     1       0  1.336806
                            ind_mom60 q1 & intraday20 q2 2023-2026        4     2       1  1.125389
                            ind_mom60 q1 & intraday20 q4 2019-2022        3     2       1 -1.119743
                            ind_mom60 q1 & intraday20 q4 2023-2026        4     0       0 -0.270860
                                ind_mom60 q1 & log_dv q1 2015-2018        1     0       0  0.572495
                                ind_mom60 q1 & log_dv q1 2019-2022        4     0       4 -0.802092
                                ind_mom60 q1 & log_dv q1 2023-2026        4     2       1  0.719661
                                 ind_mom60 q1 & max20 q4 2019-2022        1     0       0 -0.846301
                                 ind_mom60 q1 & max20 q4 2023-2026        4     0       2  0.103419
                                 ind_mom60 q1 & min20 q3 2019-2022        2     2       0  2.377543
                                 ind_mom60 q1 & min20 q3 2023-2026        4     0       3 -0.182152
                                 ind_mom60 q1 & min20 q4 2015-2018        1     0       0 -0.845544
                                 ind_mom60 q1 & min20 q4 2019-2022        4     0       1  0.017373
                                 ind_mom60 q1 & min20 q4 2023-2026        4     1       3  0.190189
                                   ind_mom60 q1 & r20 q1 2019-2022        3     1       0  1.023012
                                   ind_mom60 q1 & r20 q1 2023-2026        4     1       0  0.635692
                                   ind_mom60 q1 & r20 q4 2019-2022        3     1       0 -1.442216
                                   ind_mom60 q1 & r20 q4 2023-2026        4     0       1 -0.275380
                                   ind_mom60 q1 & r60 q0 2019-2022        4     1       2 -0.429449
                                   ind_mom60 q1 & r60 q0 2023-2026        4     0       2  0.281603
                        ind_mom60 q1 & range_compress q0 2019-2022        1     1       0 -1.094546
                        ind_mom60 q1 & range_compress q0 2023-2026        4     0       2  0.191069
                        ind_mom60 q1 & range_compress q4 2019-2022        1     0       0  0.672751
                        ind_mom60 q1 & range_compress q4 2023-2026        4     1       2  0.441325
                             ind_mom60 q1 & rel_ind20 q4 2019-2022        3     1       0 -1.202818
                             ind_mom60 q1 & rel_ind20 q4 2023-2026        4     0       1 -0.387647
                                 ind_mom60 q1 & vol20 q3 2023-2026        3     1       2 -0.454390
                            ind_mom60 q1 & vol_surge1 q1 2023-2026        3     0       1 -0.122960
                                            ind_mom60 q2 2019-2022        4     1       2 -0.016788
                                            ind_mom60 q2 2023-2026        4     0       1 -0.188624
                                ind_mom60 q2 & log_dv q4 2019-2022        3     0       2 -0.191793
                                ind_mom60 q2 & log_dv q4 2023-2026        4     1       1  0.535240
                                 ind_mom60 q2 & vol20 q3 2019-2022        4     3       0 -1.120638
                                 ind_mom60 q2 & vol20 q3 2023-2026        4     0       2  0.663232
                            ind_mom60 q3 & intraday20 q3 2015-2018        4     1       0 -0.712654
                            ind_mom60 q3 & intraday20 q3 2019-2022        4     0       1 -0.027080
                            ind_mom60 q3 & intraday20 q3 2023-2026        4     0       3  0.272072
                                ind_mom60 q3 & log_dv q0 2015-2018        3     1       0 -1.272222
                                ind_mom60 q3 & log_dv q0 2019-2022        4     1       2  0.003685
                                ind_mom60 q3 & log_dv q0 2023-2026        4     0       2  0.169133
                              ind_mom60 q3 & mom_12_1 q3 2015-2018        3     0       3 -1.227972
                              ind_mom60 q3 & mom_12_1 q3 2019-2022        4     1       2  0.570089
                              ind_mom60 q3 & mom_12_1 q3 2023-2026        4     1       2  0.402127
                                   ind_mom60 q3 & r20 q2 2015-2018        4     0       2 -0.305756
                                   ind_mom60 q3 & r20 q2 2019-2022        4     0       0  0.551303
                                   ind_mom60 q3 & r20 q2 2023-2026        4     0       2 -0.153189
                                    ind_mom60 q3 & r5 q3 2019-2022        1     0       1  0.244543
                                    ind_mom60 q3 & r5 q3 2023-2026        4     1       2 -0.169570
                                   ind_mom60 q3 & r60 q1 2015-2018        1     0       1 -0.732121
                                   ind_mom60 q3 & r60 q1 2019-2022        4     2       1  0.557223
                                   ind_mom60 q3 & r60 q1 2023-2026        4     0       1  0.384164
                            ind_mom60 q3 & vol_surge1 q3 2015-2018        2     1       0  1.225162
                            ind_mom60 q3 & vol_surge1 q3 2019-2022        4     2       1  0.820397
                            ind_mom60 q3 & vol_surge1 q3 2023-2026        4     0       2 -0.123549
                                   ind_mom60 q4 & r20 q3 2015-2018        4     1       1 -0.391720
                                   ind_mom60 q4 & r20 q3 2019-2022        4     1       1 -0.553823
                                   ind_mom60 q4 & r20 q3 2023-2026        4     0       3  0.340719
                                         ins_buyers30 q2 2015-2018        3     0       3 -0.626706
                                         ins_buyers30 q2 2019-2022        4     2       1  1.124565
                                         ins_buyers30 q2 2023-2026        4     1       1  0.497842
                      ins_buyers30 q2 & ins_officer30 q2 2015-2018        3     0       3 -0.626706
                      ins_buyers30 q2 & ins_officer30 q2 2019-2022        4     2       1  1.124565
                      ins_buyers30 q2 & ins_officer30 q2 2023-2026        4     1       1  0.497842
                        ins_buyers30 q2 & ins_value30 q2 2015-2018        3     0       3 -0.626706
                        ins_buyers30 q2 & ins_value30 q2 2019-2022        4     2       1  1.124565
                        ins_buyers30 q2 & ins_value30 q2 2023-2026        4     1       1  0.497842
                           ins_buyers30 q2 & mom_12_1 q3 2015-2018        3     2       0 -1.638648
                           ins_buyers30 q2 & mom_12_1 q3 2019-2022        4     0       3  0.372041
                           ins_buyers30 q2 & mom_12_1 q3 2023-2026        4     0       3  0.737915
                        ins_buyers30 q2 & overnight20 q3 2015-2018        2     1       1  0.189153
                        ins_buyers30 q2 & overnight20 q3 2019-2022        4     1       1  0.490031
                        ins_buyers30 q2 & overnight20 q3 2023-2026        4     2       2  0.324580
                                ins_buyers30 q2 & r20 q3 2015-2018        3     0       1 -0.170202
                                ins_buyers30 q2 & r20 q3 2019-2022        4     1       1 -0.590750
                                ins_buyers30 q2 & r20 q3 2023-2026        4     0       2  0.106255
                                ins_buyers30 q2 & r20 q4 2015-2018        2     1       1 -0.337767
                                ins_buyers30 q2 & r20 q4 2019-2022        4     2       1 -0.680790
                                ins_buyers30 q2 & r20 q4 2023-2026        4     0       3  0.099646
                                 ins_buyers30 q2 & r5 q1 2015-2018        1     0       0  0.979727
                                 ins_buyers30 q2 & r5 q1 2019-2022        4     1       1  0.230714
                                 ins_buyers30 q2 & r5 q1 2023-2026        4     2       2  0.698045
                                 ins_buyers30 q2 & r5 q3 2019-2022        3     0       1  0.001961
                                 ins_buyers30 q2 & r5 q3 2023-2026        4     1       1  0.474869
                                ins_buyers30 q2 & r60 q3 2015-2018        3     0       1 -0.361461
                                ins_buyers30 q2 & r60 q3 2019-2022        4     3       1  0.994067
                                ins_buyers30 q2 & r60 q3 2023-2026        4     0       1  0.360284
                          ins_buyers30 q2 & rel_ind20 q1 2023-2026        2     0       1  0.126665
                         ins_buyers30 q2 & vol_surge1 q1 2015-2018        2     0       0  0.151687
                         ins_buyers30 q2 & vol_surge1 q1 2019-2022        4     1       0  0.775447
                         ins_buyers30 q2 & vol_surge1 q1 2023-2026        4     2       1  0.627388
                         ins_buyers30 q2 & vol_surge1 q3 2015-2018        2     1       0  0.884119
                         ins_buyers30 q2 & vol_surge1 q3 2019-2022        4     2       2  0.123950
                         ins_buyers30 q2 & vol_surge1 q3 2023-2026        4     0       1  0.113210
                         ins_buyers30 q4 & vol_surge1 q3 2023-2026        2     1       1 -0.471485
                                        ins_officer30 q2 2015-2018        4     0       3 -0.872804
                                        ins_officer30 q2 2019-2022        4     1       2  0.336130
                                        ins_officer30 q2 2023-2026        4     1       0  1.078105
                       ins_officer30 q2 & ins_value30 q2 2015-2018        3     0       3 -0.626706
                       ins_officer30 q2 & ins_value30 q2 2019-2022        4     2       1  1.124565
                       ins_officer30 q2 & ins_value30 q2 2023-2026        4     1       1  0.497842
                        ins_officer30 q2 & intraday20 q3 2015-2018        3     0       2 -0.141881
                        ins_officer30 q2 & intraday20 q3 2019-2022        4     0       3 -0.324420
                        ins_officer30 q2 & intraday20 q3 2023-2026        4     1       1  0.615353
                             ins_officer30 q2 & min20 q1 2015-2018        3     1       1  0.735245
                             ins_officer30 q2 & min20 q1 2019-2022        4     2       2  0.569027
                             ins_officer30 q2 & min20 q1 2023-2026        4     3       1  1.536139
                          ins_officer30 q2 & mom_12_1 q3 2015-2018        3     2       0 -1.561189
                          ins_officer30 q2 & mom_12_1 q3 2019-2022        4     0       2  0.305667
                          ins_officer30 q2 & mom_12_1 q3 2023-2026        4     0       3  0.766667
                       ins_officer30 q2 & overnight20 q3 2015-2018        2     1       1  0.332353
                       ins_officer30 q2 & overnight20 q3 2019-2022        4     1       1  0.392495
                       ins_officer30 q2 & overnight20 q3 2023-2026        4     2       2  0.306301
                               ins_officer30 q2 & r20 q3 2015-2018        3     0       1 -0.173144
                               ins_officer30 q2 & r20 q3 2019-2022        4     1       1 -0.684580
                               ins_officer30 q2 & r20 q3 2023-2026        4     0       3  0.105243
                               ins_officer30 q2 & r20 q4 2015-2018        2     1       1 -0.354444
                               ins_officer30 q2 & r20 q4 2019-2022        4     2       0 -0.890342
                               ins_officer30 q2 & r20 q4 2023-2026        4     0       3  0.106471
                                ins_officer30 q2 & r5 q0 2015-2018        3     1       0  0.921597
                                ins_officer30 q2 & r5 q0 2019-2022        4     2       1  0.412919
                                ins_officer30 q2 & r5 q0 2023-2026        4     0       2 -0.196143
                         ins_officer30 q2 & rel_ind60 q2 2019-2022        3     1       1  0.949423
                         ins_officer30 q2 & rel_ind60 q2 2023-2026        4     0       2 -0.090009
                         ins_officer30 q2 & rel_ind60 q4 2019-2022        1     0       0 -0.980561
                         ins_officer30 q2 & rel_ind60 q4 2023-2026        4     1       2 -0.326184
                        ins_officer30 q2 & vol_surge1 q1 2015-2018        2     0       1  0.228347
                        ins_officer30 q2 & vol_surge1 q1 2019-2022        4     1       0  0.621507
                        ins_officer30 q2 & vol_surge1 q1 2023-2026        4     2       1  0.668798
                        ins_officer30 q2 & vol_surge1 q3 2015-2018        2     2       0  1.147414
                        ins_officer30 q2 & vol_surge1 q3 2019-2022        4     2       2  0.156814
                        ins_officer30 q2 & vol_surge1 q3 2023-2026        4     1       2  0.157120
                        ins_officer30 q2 & vol_surge5 q1 2019-2022        3     0       1  0.395821
                        ins_officer30 q2 & vol_surge5 q1 2023-2026        4     2       0  1.084491
                                  ins_opportunistic30 q2 2015-2018        4     0       3 -0.571820
                                  ins_opportunistic30 q2 2019-2022        4     1       1  0.402856
                                  ins_opportunistic30 q2 2023-2026        4     1       1  0.724788
                       ins_opportunistic30 q2 & min20 q1 2023-2026        3     1       1  0.544752
                    ins_opportunistic30 q2 & mom_12_1 q3 2015-2018        3     2       0 -1.638153
                    ins_opportunistic30 q2 & mom_12_1 q3 2019-2022        4     0       2  0.304200
                    ins_opportunistic30 q2 & mom_12_1 q3 2023-2026        4     0       3  0.708954
                         ins_opportunistic30 q2 & r20 q1 2023-2026        3     1       2  0.071620
              ins_opportunistic30 q2 & range_compress q2 2015-2018        2     1       0  0.887797
              ins_opportunistic30 q2 & range_compress q2 2019-2022        4     0       2 -0.106748
              ins_opportunistic30 q2 & range_compress q2 2023-2026        4     0       2 -0.126127
                       ins_opportunistic30 q2 & vol20 q2 2023-2026        2     2       0  1.367249
                       ins_opportunistic30 q2 & vol20 q3 2023-2026        3     1       1  0.366290
                  ins_opportunistic30 q2 & vol_surge5 q1 2019-2022        3     0       1  0.371493
                  ins_opportunistic30 q2 & vol_surge5 q1 2023-2026        4     1       0  0.937741
                  ins_opportunistic30 q2 & vol_surge5 q3 2023-2026        2     1       0  1.033754
                                          ins_value30 q2 2015-2018        3     0       3 -0.626706
                                          ins_value30 q2 2019-2022        4     2       1  1.124565
                                          ins_value30 q2 2023-2026        4     1       1  0.497842
                            ins_value30 q2 & mom_12_1 q3 2015-2018        3     2       0 -1.638648
                            ins_value30 q2 & mom_12_1 q3 2019-2022        4     0       3  0.372041
                            ins_value30 q2 & mom_12_1 q3 2023-2026        4     0       3  0.737915
                         ins_value30 q2 & overnight20 q3 2015-2018        2     1       1  0.189153
                         ins_value30 q2 & overnight20 q3 2019-2022        4     1       1  0.490031
                         ins_value30 q2 & overnight20 q3 2023-2026        4     2       2  0.324580
                                 ins_value30 q2 & r20 q3 2015-2018        3     0       1 -0.170202
                                 ins_value30 q2 & r20 q3 2019-2022        4     1       1 -0.590750
                                 ins_value30 q2 & r20 q3 2023-2026        4     0       2  0.106255
                                 ins_value30 q2 & r20 q4 2015-2018        2     1       1 -0.337767
                                 ins_value30 q2 & r20 q4 2019-2022        4     2       1 -0.680790
                                 ins_value30 q2 & r20 q4 2023-2026        4     0       3  0.099646
                                  ins_value30 q2 & r5 q1 2015-2018        1     0       0  0.979727
                                  ins_value30 q2 & r5 q1 2019-2022        4     1       1  0.230714
                                  ins_value30 q2 & r5 q1 2023-2026        4     2       2  0.698045
                               ins_value30 q2 & vol20 q3 2019-2022        4     2       2  0.312678
                               ins_value30 q2 & vol20 q3 2023-2026        4     3       0  1.866465
                          ins_value30 q2 & vol_surge1 q1 2015-2018        2     0       0  0.151687
                          ins_value30 q2 & vol_surge1 q1 2019-2022        4     1       0  0.775447
                          ins_value30 q2 & vol_surge1 q1 2023-2026        4     2       1  0.627388
                          ins_value30 q2 & vol_surge1 q3 2015-2018        2     1       0  0.884119
                          ins_value30 q2 & vol_surge1 q3 2019-2022        4     2       2  0.123950
                          ins_value30 q2 & vol_surge1 q3 2023-2026        4     0       1  0.113210
                             intraday20 q0 & mom_12_1 q2 2015-2018        4     1       3  0.348289
                             intraday20 q0 & mom_12_1 q2 2019-2022        4     2       1 -0.337475
                             intraday20 q0 & mom_12_1 q2 2023-2026        4     2       2 -0.479771
                             intraday20 q0 & mom_12_1 q3 2015-2018        2     0       1  0.184722
                             intraday20 q0 & mom_12_1 q3 2019-2022        4     1       2  0.724028
                             intraday20 q0 & mom_12_1 q3 2023-2026        4     0       0  0.451739
                                   intraday20 q0 & r5 q0 2019-2022        3     1       1 -0.361562
                                   intraday20 q0 & r5 q0 2023-2026        4     1       2 -0.484257
                                           intraday20 q1 2023-2026        3     1       0  0.830419
                                intraday20 q1 & max20 q1 2019-2022        1     1       0  1.923184
                                intraday20 q1 & max20 q1 2023-2026        4     4       0  1.346546
                                intraday20 q1 & min20 q4 2015-2018        1     0       1  1.021431
                                intraday20 q1 & min20 q4 2019-2022        4     1       2 -0.541003
                                intraday20 q1 & min20 q4 2023-2026        4     0       1 -0.083838
                                   intraday20 q1 & r1 q0 2019-2022        2     0       1 -0.083106
                                   intraday20 q1 & r1 q0 2023-2026        4     1       1  0.451587
                       intraday20 q1 & range_compress q1 2015-2018        2     0       1  0.107599
                       intraday20 q1 & range_compress q1 2019-2022        4     2       1  0.807283
                       intraday20 q1 & range_compress q1 2023-2026        4     2       1  0.320297
                            intraday20 q1 & rel_ind60 q2 2023-2026        4     0       1 -0.145282
        intraday20 q1 & rel_ind60 q2 unless close_loc q0 2023-2026        4     0       1 -0.040587
  intraday20 q1 & rel_ind60 q2 unless days_since_earn q4 2023-2026        4     1       2  0.032292
         intraday20 q1 & rel_ind60 q2 unless mom_12_1 q0 2023-2026        4     1       1  0.450654
         intraday20 q1 & rel_ind60 q2 unless mom_12_1 q4 2023-2026        4     1       1 -0.523553
        intraday20 q1 & rel_ind60 q2 unless r5_nonews q4 2023-2026        4     0       2  0.037267
   intraday20 q1 & rel_ind60 q2 unless range_compress q0 2023-2026        4     0       0 -0.328670
            intraday20 q1 & rel_ind60 q2 unless vol20 q4 2023-2026        4     0       1 -0.490179
                           intraday20 q1 & vol_surge1 q2 2023-2026        4     0       1 -0.394694
                                           intraday20 q2 2015-2018        2     0       1 -0.050142
                                           intraday20 q2 2019-2022        4     2       1  0.840458
                                           intraday20 q2 2023-2026        4     0       3 -0.184610
                          intraday20 q2 & overnight20 q3 2019-2022        4     1       0  0.802120
                          intraday20 q2 & overnight20 q3 2023-2026        4     1       2  0.030903
                                  intraday20 q2 & r20 q4 2019-2022        3     0       0 -0.453576
                                  intraday20 q2 & r20 q4 2023-2026        4     0       2  0.324487
                       intraday20 q2 & range_compress q2 2019-2022        1     1       0  1.641663
                       intraday20 q2 & range_compress q2 2023-2026        4     0       2  0.089255
                                intraday20 q2 & vol20 q2 2019-2022        3     1       1  0.923997
                                intraday20 q2 & vol20 q2 2023-2026        4     0       1 -0.052166
                                intraday20 q3 & max20 q0 2015-2018        3     1       2  0.038647
                                intraday20 q3 & max20 q0 2019-2022        4     1       1  0.226882
                                intraday20 q3 & max20 q0 2023-2026        4     0       1  0.253255
                                intraday20 q3 & min20 q1 2015-2018        1     1       0  1.108044
                                intraday20 q3 & min20 q1 2019-2022        4     0       3 -1.334029
                                intraday20 q3 & min20 q1 2023-2026        4     1       0  1.082830
                             intraday20 q3 & mom_12_1 q3 2015-2018        3     3       0 -1.521749
                             intraday20 q3 & mom_12_1 q3 2019-2022        4     1       1 -0.334043
                             intraday20 q3 & mom_12_1 q3 2023-2026        4     0       3  1.063418
                          intraday20 q3 & overnight20 q1 2015-2018        4     1       0  0.456450
                          intraday20 q3 & overnight20 q1 2019-2022        4     0       2 -0.366528
                          intraday20 q3 & overnight20 q1 2023-2026        4     1       1  0.349624
                          intraday20 q3 & overnight20 q3 2015-2018        3     2       1 -1.197289
                          intraday20 q3 & overnight20 q3 2019-2022        4     0       2  0.145893
                          intraday20 q3 & overnight20 q3 2023-2026        4     0       1 -0.402993
                                   intraday20 q3 & r1 q2 2015-2018        4     1       3 -0.100121
                                   intraday20 q3 & r1 q2 2019-2022        4     1       3  0.125895
                                   intraday20 q3 & r1 q2 2023-2026        4     0       2 -0.044822
                                  intraday20 q3 & r20 q3 2015-2018        3     0       2  0.111186
                                  intraday20 q3 & r20 q3 2019-2022        4     1       2 -0.246859
                                  intraday20 q3 & r20 q3 2023-2026        4     0       3  0.133737
                                   intraday20 q3 & r5 q1 2015-2018        4     2       0  1.400063
                                   intraday20 q3 & r5 q1 2019-2022        4     1       1  0.045469
                                   intraday20 q3 & r5 q1 2023-2026        4     1       2  0.340815
                       intraday20 q3 & range_compress q3 2015-2018        3     0       1  0.066668
                       intraday20 q3 & range_compress q3 2019-2022        4     0       2 -0.276945
                       intraday20 q3 & range_compress q3 2023-2026        4     0       0  0.576007
                            intraday20 q3 & rel_ind60 q1 2015-2018        1     0       0  0.159347
                            intraday20 q3 & rel_ind60 q1 2019-2022        4     1       3 -0.651262
                            intraday20 q3 & rel_ind60 q1 2023-2026        4     0       0  0.624892
                                           intraday20 q4 2015-2018        2     0       1  0.008793
                                           intraday20 q4 2019-2022        4     3       0 -1.503137
                                           intraday20 q4 2023-2026        4     0       1 -0.194611
                               intraday20 q4 & log_dv q1 2023-2026        4     1       2  0.261821
                               intraday20 q4 & log_dv q3 2019-2022        2     2       0 -1.513562
                               intraday20 q4 & log_dv q3 2023-2026        4     1       1 -0.619539
                                intraday20 q4 & max20 q4 2019-2022        1     1       0 -2.070356
                                intraday20 q4 & max20 q4 2023-2026        4     0       1 -0.121932
                          intraday20 q4 & overnight20 q2 2019-2022        3     1       2 -0.053252
                          intraday20 q4 & overnight20 q2 2023-2026        4     0       1  0.284218
                                  intraday20 q4 & r20 q4 2023-2026        4     0       1 -0.137694
                            intraday20 q4 & r5_nonews q4 2023-2026        4     0       1 -0.068503
                            intraday20 q4 & rel_ind20 q1 2023-2026        4     2       2 -0.249518
                            intraday20 q4 & rel_ind20 q4 2023-2026        4     1       2 -0.258369
                            intraday20 q4 & rel_ind60 q4 2023-2026        4     2       2 -0.395262
                               intraday20 q4 & skew60 q0 2023-2026        2     1       0  0.776750
                           intraday20 q4 & vol_surge1 q0 2023-2026        2     1       0 -0.599455
                           intraday20 q4 & vol_surge5 q0 2023-2026        4     1       2 -0.197773
                                    log_dv q0 & news5 q2 2015-2018        4     1       2 -0.116915
                                    log_dv q0 & news5 q2 2019-2022        4     1       2 -0.563207
                                    log_dv q0 & news5 q2 2023-2026        4     1       1 -0.585331
                                      log_dv q0 & r20 q1 2015-2018        1     1       0 -2.021581
                                      log_dv q0 & r20 q1 2019-2022        4     0       1 -0.055833
                                      log_dv q0 & r20 q1 2023-2026        4     0       2  0.325290
                                log_dv q0 & rel_ind60 q2 2023-2026        3     1       1 -0.592295
                                               log_dv q1 2015-2018        2     1       0  0.824188
                                               log_dv q1 2019-2022        4     0       3 -0.439124
                                               log_dv q1 2023-2026        4     0       1  0.327925
                                    log_dv q1 & max20 q4 2023-2026        2     1       0 -1.021460
                                 log_dv q1 & mom_12_1 q3 2015-2018        3     0       1  0.278097
                                 log_dv q1 & mom_12_1 q3 2019-2022        4     0       2 -0.030104
                                 log_dv q1 & mom_12_1 q3 2023-2026        4     0       1  0.216922
                                     log_dv q1 & r120 q3 2015-2018        3     1       1  0.598644
                                     log_dv q1 & r120 q3 2019-2022        4     2       2  0.276449
                                     log_dv q1 & r120 q3 2023-2026        4     2       2  0.181585
                                     log_dv q1 & r120 q4 2023-2026        4     1       2  0.187925
                                      log_dv q1 & r20 q4 2019-2022        1     0       1 -0.990027
                                      log_dv q1 & r20 q4 2023-2026        4     0       1  0.248704
                                       log_dv q1 & r5 q1 2015-2018        1     1       0  1.798485
                                       log_dv q1 & r5 q1 2019-2022        4     1       2  0.028114
                                       log_dv q1 & r5 q1 2023-2026        4     1       1  0.439728
                     log_dv q1 & r5 q1 unless atr_pct q4 2019-2022        2     0       1 -0.059868
                     log_dv q1 & r5 q1 unless atr_pct q4 2023-2026        4     2       1  0.644054
                   log_dv q1 & r5 q1 unless close_loc q0 2019-2022        3     2       0 -1.263870
                   log_dv q1 & r5 q1 unless close_loc q0 2023-2026        4     0       1 -0.243719
                   log_dv q1 & r5 q1 unless close_loc q4 2019-2022        3     0       2 -0.342811
                   log_dv q1 & r5 q1 unless close_loc q4 2023-2026        4     1       1  0.551636
                   log_dv q1 & r5 q1 unless dist_52wh q0 2019-2022        3     1       1  0.337912
                   log_dv q1 & r5 q1 unless dist_52wh q0 2023-2026        4     1       2  0.069932
                   log_dv q1 & r5 q1 unless dist_52wh q4 2019-2022        3     1       1 -0.594707
                   log_dv q1 & r5 q1 unless dist_52wh q4 2023-2026        4     1       3  0.345563
                   log_dv q1 & r5 q1 unless dist_ma50 q0 2019-2022        3     0       2 -0.542321
                   log_dv q1 & r5 q1 unless dist_ma50 q0 2023-2026        4     2       2  0.444462
                   log_dv q1 & r5 q1 unless dist_ma50 q4 2019-2022        3     0       2 -0.469983
                   log_dv q1 & r5 q1 unless dist_ma50 q4 2023-2026        4     2       1  0.602170
                log_dv q1 & r5 q1 unless ear_volsurge q0 2019-2022        3     0       2 -0.300953
                log_dv q1 & r5 q1 unless ear_volsurge q0 2023-2026        4     3       1  0.590239
                log_dv q1 & r5 q1 unless ear_volsurge q4 2019-2022        3     2       1 -0.993457
                log_dv q1 & r5 q1 unless ear_volsurge q4 2023-2026        4     0       3  0.392162
                        log_dv q1 & r5 q1 unless frog q0 2019-2022        3     1       2 -0.104666
                        log_dv q1 & r5 q1 unless frog q0 2023-2026        4     2       1  0.535100
                   log_dv q1 & r5 q1 unless gap_today q0 2019-2022        3     2       1 -0.818677
                   log_dv q1 & r5 q1 unless gap_today q0 2023-2026        4     0       3  0.239601
                   log_dv q1 & r5 q1 unless gap_today q4 2019-2022        3     1       2  0.056063
                   log_dv q1 & r5 q1 unless gap_today q4 2023-2026        4     2       1  0.565066
                   log_dv q1 & r5 q1 unless ind_mom20 q0 2019-2022        3     1       1 -0.862895
                   log_dv q1 & r5 q1 unless ind_mom20 q0 2023-2026        4     0       3  0.348548
                   log_dv q1 & r5 q1 unless ind_mom20 q4 2019-2022        3     2       1 -0.619133
                   log_dv q1 & r5 q1 unless ind_mom20 q4 2023-2026        4     0       2  0.087238
                        log_dv q1 & r5 q1 unless r120 q0 2019-2022        3     1       1 -0.239287
                        log_dv q1 & r5 q1 unless r120 q0 2023-2026        4     0       2 -0.036650
                        log_dv q1 & r5 q1 unless r120 q4 2019-2022        3     1       2  0.208402
                        log_dv q1 & r5 q1 unless r120 q4 2023-2026        4     2       1  0.642670
                     log_dv q1 & r5 q1 unless r5_news q0 2019-2022        3     0       2 -0.407186
                     log_dv q1 & r5 q1 unless r5_news q0 2023-2026        4     3       1  0.621689
                   log_dv q1 & r5 q1 unless rel_ind20 q4 2019-2022        2     1       0 -1.077061
                   log_dv q1 & r5 q1 unless rel_ind20 q4 2023-2026        4     0       3  0.376970
                   log_dv q1 & r5 q1 unless vol_ratio q0 2019-2022        3     0       2 -0.301924
                   log_dv q1 & r5 q1 unless vol_ratio q0 2023-2026        4     2       1  0.666005
                  log_dv q1 & r5 q1 unless vol_surge1 q0 2019-2022        3     1       2 -0.482508
                  log_dv q1 & r5 q1 unless vol_surge1 q0 2023-2026        4     3       1  0.537690
                  log_dv q1 & r5 q1 unless vol_surge5 q0 2019-2022        3     0       2 -0.007216
                  log_dv q1 & r5 q1 unless vol_surge5 q0 2023-2026        4     1       1  0.340184
                  log_dv q1 & r5 q1 unless vol_surge5 q4 2019-2022        2     1       0 -1.827378
                  log_dv q1 & r5 q1 unless vol_surge5 q4 2023-2026        4     0       3  0.347017
                           log_dv q1 & range_compress q0 2023-2026        2     0       1  0.026785
                           log_dv q1 & range_compress q1 2019-2022        4     1       2  0.629827
                           log_dv q1 & range_compress q1 2023-2026        4     3       1  1.042378
                           log_dv q1 & range_compress q3 2019-2022        4     0       0  0.443211
                           log_dv q1 & range_compress q3 2023-2026        4     1       1  0.386251
                                log_dv q1 & rel_ind20 q1 2019-2022        1     1       0 -1.220316
                                log_dv q1 & rel_ind20 q1 2023-2026        4     0       2  0.004730
                                log_dv q1 & rel_ind20 q3 2015-2018        3     0       2 -0.311004
                                log_dv q1 & rel_ind20 q3 2019-2022        4     0       2 -0.305310
                                log_dv q1 & rel_ind20 q3 2023-2026        4     2       0  1.372841
                                log_dv q1 & rel_ind20 q4 2019-2022        3     3       0 -1.358040
                                log_dv q1 & rel_ind20 q4 2023-2026        4     0       3  0.427097
                                log_dv q1 & rel_ind60 q4 2023-2026        4     1       2 -0.064913
                                   log_dv q1 & skew60 q4 2023-2026        2     1       1 -0.154228
                                log_dv q1 & vol_ratio q0 2019-2022        4     0       1  0.079502
                                log_dv q1 & vol_ratio q0 2023-2026        4     0       1  0.276470
                                               log_dv q2 2023-2026        3     1       1  0.275750
                                log_dv q2 & r5_nonews q0 2023-2026        3     0       2  0.077935
                                    log_dv q3 & max20 q4 2019-2022        1     1       0 -1.129005
                                    log_dv q3 & max20 q4 2023-2026        4     1       1 -0.493247
                              log_dv q3 & overnight20 q3 2019-2022        1     0       1  0.876314
                              log_dv q3 & overnight20 q3 2023-2026        4     1       3 -0.284447
                                       log_dv q3 & r5 q4 2019-2022        1     0       0 -0.561676
                                       log_dv q3 & r5 q4 2023-2026        4     1       2 -0.350280
                                log_dv q3 & r5_nonews q4 2019-2022        1     1       0 -1.109886
                                log_dv q3 & r5_nonews q4 2023-2026        4     1       3 -0.181250
                           log_dv q3 & range_compress q0 2019-2022        1     0       0 -0.154880
                           log_dv q3 & range_compress q0 2023-2026        4     1       1 -0.402632
                                log_dv q3 & rel_ind20 q4 2019-2022        1     0       0 -0.142802
                                log_dv q3 & rel_ind20 q4 2023-2026        4     2       1 -1.292102
                                log_dv q3 & rel_ind60 q1 2019-2022        4     1       1  0.164125
                                log_dv q3 & rel_ind60 q1 2023-2026        4     3       1  1.034254
                                log_dv q3 & vol_ratio q3 2019-2022        2     1       1  0.324668
                                log_dv q3 & vol_ratio q3 2023-2026        4     1       2 -0.022574
                                       log_dv q4 & r5 q1 2023-2026        3     0       1  0.111186
                                log_dv q4 & rel_ind60 q2 2019-2022        3     2       0  1.812473
                                log_dv q4 & rel_ind60 q2 2023-2026        4     0       1  0.107653
                               max20 q0 & overnight20 q3 2019-2022        2     0       2  0.837229
                               max20 q0 & overnight20 q3 2023-2026        4     1       2 -0.436738
                                       max20 q1 & r20 q1 2023-2026        3     0       0  0.503336
                                 max20 q1 & r5_nonews q4 2023-2026        4     1       1 -0.202554
                                 max20 q1 & rel_ind60 q2 2023-2026        4     1       1 -0.457030
                                     max20 q1 & vol20 q2 2023-2026        1     1       0 -1.214152
                                max20 q1 & vol_surge1 q2 2023-2026        4     1       2  0.199090
                                max20 q1 & vol_surge5 q2 2023-2026        4     1       1  0.566054
                                     max20 q2 & news5 q2 2023-2026        4     3       0  1.210595
                               max20 q2 & overnight20 q4 2015-2018        2     0       2 -0.579692
                               max20 q2 & overnight20 q4 2019-2022        4     2       1  0.562589
                               max20 q2 & overnight20 q4 2023-2026        4     2       2  0.053807
                                 max20 q2 & r5_nonews q3 2019-2022        3     1       0 -0.724680
                                 max20 q2 & r5_nonews q3 2023-2026        4     0       2 -0.181882
                                max20 q2 & vol_surge1 q4 2015-2018        2     0       1  0.096724
                                max20 q2 & vol_surge1 q4 2019-2022        4     0       2 -0.493713
                                max20 q2 & vol_surge1 q4 2023-2026        4     2       1  0.908914
                                                max20 q3 2019-2022        4     0       3 -0.192324
                                                max20 q3 2023-2026        4     1       1  0.805695
                                 max20 q3 & r5_nonews q0 2019-2022        1     0       1 -0.000367
                                 max20 q3 & r5_nonews q0 2023-2026        4     1       2  0.281541
                                                max20 q4 2019-2022        1     1       0 -1.390010
                                                max20 q4 2023-2026        4     1       1 -0.387142
                                  max20 q4 & mom_12_1 q2 2023-2026        1     0       0  0.176067
                            max20 q4 & range_compress q3 2019-2022        3     0       2  0.260478
                            max20 q4 & range_compress q3 2023-2026        4     2       2 -0.235857
                                    max20 q4 & skew60 q0 2023-2026        2     1       1  0.747726
                                max20 q4 & vol_surge1 q0 2023-2026        1     1       0 -1.020459
                                  min20 q0 & mom_12_1 q2 2023-2026        1     0       0  0.247689
                                min20 q0 & vol_surge5 q0 2023-2026        1     1       0 -1.021155
                                                min20 q1 2015-2018        3     1       1  0.687094
                                                min20 q1 2019-2022        4     2       2  0.622021
                                                min20 q1 2023-2026        4     3       1  1.529319
                                  min20 q1 & mom_12_1 q3 2015-2018        3     0       2 -0.332813
                                  min20 q1 & mom_12_1 q3 2019-2022        4     0       2 -0.448232
                                  min20 q1 & mom_12_1 q3 2023-2026        4     3       1  1.107898
                                     min20 q1 & news5 q2 2023-2026        3     2       0  1.286561
                                       min20 q1 & r20 q1 2019-2022        4     3       1  1.051587
                                       min20 q1 & r20 q1 2023-2026        4     1       0  0.707676
                                        min20 q1 & r5 q1 2023-2026        3     1       1  0.098095
                                   min20 q1 & r5_news q2 2015-2018        3     1       0  0.828111
                                   min20 q1 & r5_news q2 2019-2022        4     2       2  0.819862
                                   min20 q1 & r5_news q2 2023-2026        4     3       1  1.346647
                                       min20 q1 & r60 q2 2023-2026        3     0       1  0.066465
                                 min20 q1 & vol_ratio q1 2019-2022        4     1       2  0.301788
                                 min20 q1 & vol_ratio q1 2023-2026        4     1       1  0.522032
                                min20 q1 & vol_surge1 q1 2023-2026        3     2       1  1.076025
                                                min20 q2 2023-2026        4     1       0  0.372240
                                        min20 q2 & r1 q0 2023-2026        2     1       1 -0.306654
                                 min20 q2 & r5_nonews q1 2019-2022        4     2       1  1.045176
                                 min20 q2 & r5_nonews q1 2023-2026        4     1       2 -0.118095
                                    min20 q2 & skew60 q2 2023-2026        4     2       0  0.603021
                                     min20 q2 & vol20 q3 2019-2022        4     1       3  0.087758
                                     min20 q2 & vol20 q3 2023-2026        4     1       2  0.131371
                                min20 q2 & vol_surge1 q1 2019-2022        4     0       2 -0.050632
                                min20 q2 & vol_surge1 q1 2023-2026        4     1       2  0.080533
                               min20 q3 & overnight20 q2 2019-2022        2     1       0  0.995332
                               min20 q3 & overnight20 q2 2023-2026        4     1       1  0.655168
                                        min20 q3 & r5 q2 2023-2026        4     4       0 -1.386351
                                 min20 q3 & r5_nonews q1 2019-2022        2     1       0  1.846715
                                 min20 q3 & r5_nonews q1 2023-2026        4     0       1  0.063747
                                       min20 q3 & r60 q3 2019-2022        1     1       0  1.412440
                                       min20 q3 & r60 q3 2023-2026        4     0       2  0.229491
                                 min20 q3 & rel_ind60 q1 2019-2022        2     2       0  1.701649
                                 min20 q3 & rel_ind60 q1 2023-2026        4     0       2  0.125962
                                    min20 q3 & skew60 q3 2019-2022        2     2       0  1.815850
                                    min20 q3 & skew60 q3 2023-2026        4     0       3 -0.515481
                                                min20 q4 2015-2018        3     0       3  0.307395
                                                min20 q4 2019-2022        4     1       2 -0.219330
                                                min20 q4 2023-2026        4     1       3 -0.293001
                                        min20 q4 & r1 q4 2019-2022        2     0       1  0.227717
                                        min20 q4 & r1 q4 2023-2026        4     0       1 -0.392982
                                      min20 q4 & r120 q1 2015-2018        2     0       1 -0.632709
                                      min20 q4 & r120 q1 2019-2022        4     1       2  0.302543
                                      min20 q4 & r120 q1 2023-2026        4     0       1  0.182408
                                       min20 q4 & r20 q2 2015-2018        2     0       1 -0.325123
                                       min20 q4 & r20 q2 2019-2022        4     1       2  0.315649
                                       min20 q4 & r20 q2 2023-2026        4     1       2 -0.168510
                                   min20 q4 & r5_news q4 2019-2022        4     2       2  0.507388
                                   min20 q4 & r5_news q4 2023-2026        4     1       2 -0.199038
                            min20 q4 & range_compress q2 2015-2018        1     0       1  0.334092
                            min20 q4 & range_compress q2 2019-2022        4     1       2 -0.364296
                            min20 q4 & range_compress q2 2023-2026        4     0       1 -0.236126
                                 min20 q4 & rel_ind60 q1 2015-2018        2     1       1  0.582560
                                 min20 q4 & rel_ind60 q1 2019-2022        4     1       2  0.010265
                                 min20 q4 & rel_ind60 q1 2023-2026        4     2       2  0.370950
                                    mom_12_1 q0 & r20 q3 2019-2022        4     1       1 -0.269433
                                    mom_12_1 q0 & r20 q3 2023-2026        4     2       2 -0.592692
                              mom_12_1 q0 & rel_ind20 q3 2015-2018        2     1       1  0.793029
                              mom_12_1 q0 & rel_ind20 q3 2019-2022        4     1       2  0.017279
                              mom_12_1 q0 & rel_ind20 q3 2023-2026        4     0       1 -0.065381
     mom_12_1 q0 & rel_ind20 q4 unless range_compress q0 2019-2022        4     1       1 -0.711628
     mom_12_1 q0 & rel_ind20 q4 unless range_compress q0 2023-2026        4     1       2 -0.414802
                                  mom_12_1 q0 & vol20 q3 2019-2022        2     1       0 -1.072885
                                  mom_12_1 q0 & vol20 q3 2023-2026        4     0       2  0.035576
                             mom_12_1 q0 & vol_surge1 q3 2015-2018        2     0       1  0.499494
                             mom_12_1 q0 & vol_surge1 q3 2019-2022        4     0       2  0.093546
                             mom_12_1 q0 & vol_surge1 q3 2023-2026        4     1       1 -0.368997
                         mom_12_1 q1 & range_compress q1 2015-2018        3     0       1 -0.172494
                         mom_12_1 q1 & range_compress q1 2019-2022        4     0       3 -0.068960
                         mom_12_1 q1 & range_compress q1 2023-2026        4     0       2  0.093147
                                             mom_12_1 q2 2015-2018        2     0       1 -0.116827
                                             mom_12_1 q2 2019-2022        4     2       2 -0.151647
                                             mom_12_1 q2 2023-2026        4     1       1 -0.414143
                                    mom_12_1 q2 & r20 q0 2015-2018        4     1       3  0.052597
                                    mom_12_1 q2 & r20 q0 2019-2022        4     1       1 -0.587106
                                    mom_12_1 q2 & r20 q0 2023-2026        4     0       1 -0.133649
                                    mom_12_1 q2 & r60 q3 2019-2022        4     2       0  0.818840
                                    mom_12_1 q2 & r60 q3 2023-2026        4     0       0  0.402404
                         mom_12_1 q2 & range_compress q2 2019-2022        4     0       2  0.060872
                         mom_12_1 q2 & range_compress q2 2023-2026        4     1       2 -0.367969
                                  mom_12_1 q2 & vol20 q4 2023-2026        1     0       0  0.630424
                             mom_12_1 q2 & vol_surge1 q3 2015-2018        4     1       1  0.532692
                             mom_12_1 q2 & vol_surge1 q3 2019-2022        4     0       3 -0.615536
                             mom_12_1 q2 & vol_surge1 q3 2023-2026        4     0       1  0.375400
                             mom_12_1 q2 & vol_surge5 q3 2023-2026        4     1       1  0.485950
                                             mom_12_1 q3 2015-2018        4     1       3 -0.656125
                                             mom_12_1 q3 2019-2022        4     1       2  0.254487
                                             mom_12_1 q3 2023-2026        4     2       1  0.783346
                                    mom_12_1 q3 & r20 q3 2015-2018        3     2       0 -1.230225
                                    mom_12_1 q3 & r20 q3 2019-2022        4     0       4  0.480931
                                    mom_12_1 q3 & r20 q3 2023-2026        4     0       2  0.116064
                                     mom_12_1 q3 & r5 q0 2015-2018        3     1       2 -0.814283
                                     mom_12_1 q3 & r5 q0 2019-2022        4     1       2  0.344806
                                     mom_12_1 q3 & r5 q0 2023-2026        4     2       1  0.905337
                                mom_12_1 q3 & r5_news q2 2015-2018        3     2       0 -1.231004
                                mom_12_1 q3 & r5_news q2 2019-2022        4     0       3  0.272640
                                mom_12_1 q3 & r5_news q2 2023-2026        4     0       2  0.569462
                         mom_12_1 q3 & range_compress q0 2015-2018        3     0       2 -0.457723
                         mom_12_1 q3 & range_compress q0 2019-2022        4     1       1  0.340231
                         mom_12_1 q3 & range_compress q0 2023-2026        4     2       1  0.551078
                         mom_12_1 q3 & range_compress q4 2015-2018        3     2       0 -1.419676
                         mom_12_1 q3 & range_compress q4 2019-2022        4     0       3  0.512686
                         mom_12_1 q3 & range_compress q4 2023-2026        4     0       3  0.206067
                              mom_12_1 q3 & rel_ind20 q3 2015-2018        3     0       0 -0.444864
                              mom_12_1 q3 & rel_ind20 q3 2019-2022        4     2       1 -0.594120
                              mom_12_1 q3 & rel_ind20 q3 2023-2026        4     0       4  0.850337
                              mom_12_1 q3 & rel_ind60 q2 2019-2022        1     0       1 -0.270974
                              mom_12_1 q3 & rel_ind60 q2 2023-2026        4     2       2  0.246355
                                  mom_12_1 q3 & vol20 q3 2019-2022        4     0       3 -0.240651
                                  mom_12_1 q3 & vol20 q3 2023-2026        4     2       1  0.890712
                              mom_12_1 q3 & vol_ratio q0 2019-2022        2     0       1  0.050708
                              mom_12_1 q3 & vol_ratio q0 2023-2026        4     2       1 -0.490898
                              mom_12_1 q3 & vol_ratio q2 2015-2018        3     2       0 -0.936340
                              mom_12_1 q3 & vol_ratio q2 2019-2022        4     0       1 -0.094507
                              mom_12_1 q3 & vol_ratio q2 2023-2026        4     1       3  0.726567
                             mom_12_1 q3 & vol_surge1 q2 2023-2026        4     1       3 -0.283745
                                             mom_12_1 q4 2015-2018        2     1       0  0.765561
                                             mom_12_1 q4 2019-2022        4     1       2  0.423587
                                             mom_12_1 q4 2023-2026        4     0       0  0.422708
                              mom_12_1 q4 & vol_ratio q4 2019-2022        4     0       2 -0.011370
                              mom_12_1 q4 & vol_ratio q4 2023-2026        4     0       0  0.328099
                             mom_12_1 q4 & vol_surge1 q2 2015-2018        1     1       0  1.465568
                             mom_12_1 q4 & vol_surge1 q2 2019-2022        4     1       1  0.230666
                             mom_12_1 q4 & vol_surge1 q2 2023-2026        4     1       1  0.341403
                                                news5 q2 2019-2022        4     2       1  0.196574
                                                news5 q2 2023-2026        4     2       0  1.266892
                                       news5 q2 & r60 q3 2023-2026        1     0       0 -0.228384
                                 news5 q2 & vol_ratio q3 2019-2022        2     1       1  0.407454
                                 news5 q2 & vol_ratio q3 2023-2026        4     1       1  0.520633
                                news5 q2 & vol_surge5 q4 2019-2022        3     0       2  0.187304
                                news5 q2 & vol_surge5 q4 2023-2026        4     2       2 -1.364331
                                       news5 q4 & r20 q4 2019-2022        4     2       0 -1.399291
                                       news5 q4 & r20 q4 2023-2026        4     1       2 -0.226887
                                          overnight20 q0 2019-2022        4     2       2 -0.315123
                                          overnight20 q0 2023-2026        4     2       1 -0.546795
                                overnight20 q0 & r120 q3 2019-2022        1     0       1 -1.052182
                                overnight20 q0 & r120 q3 2023-2026        4     2       1  0.557533
                           overnight20 q0 & rel_ind60 q4 2019-2022        1     0       0 -0.864931
                           overnight20 q0 & rel_ind60 q4 2023-2026        4     0       2 -0.052705
                              overnight20 q0 & skew60 q1 2019-2022        1     1       0 -1.559404
                              overnight20 q0 & skew60 q1 2023-2026        4     1       2  0.332489
                                          overnight20 q1 2019-2022        1     0       0  0.194151
                                          overnight20 q1 2023-2026        4     2       2 -0.022638
                                  overnight20 q1 & r5 q3 2019-2022        3     1       2  0.285196
                                  overnight20 q1 & r5 q3 2023-2026        4     2       0  1.049387
                              overnight20 q1 & skew60 q0 2015-2018        4     0       3  0.696594
                              overnight20 q1 & skew60 q0 2019-2022        4     2       2 -0.439049
                              overnight20 q1 & skew60 q0 2023-2026        4     2       2 -0.703579
                                          overnight20 q2 2019-2022        4     0       2 -0.119561
                                          overnight20 q2 2023-2026        4     2       1  0.665889
                      overnight20 q2 & range_compress q0 2019-2022        1     0       0  0.298190
                      overnight20 q2 & range_compress q0 2023-2026        4     2       2  0.136358
                      overnight20 q2 & range_compress q2 2015-2018        1     0       0  0.317713
                      overnight20 q2 & range_compress q2 2019-2022        4     2       2  0.332498
                      overnight20 q2 & range_compress q2 2023-2026        4     1       2  0.262888
                           overnight20 q2 & rel_ind60 q1 2019-2022        2     1       0  1.440091
                           overnight20 q2 & rel_ind60 q1 2023-2026        4     2       0  0.888588
                               overnight20 q2 & vol20 q2 2023-2026        1     0       0  0.704426
                          overnight20 q2 & vol_surge5 q3 2019-2022        1     0       0  0.244782
                          overnight20 q2 & vol_surge5 q3 2023-2026        4     1       1  0.348508
                                          overnight20 q3 2015-2018        2     1       1  0.278733
                                          overnight20 q3 2019-2022        4     1       1  0.427501
                                          overnight20 q3 2023-2026        4     2       2  0.209822
                                  overnight20 q3 & r1 q1 2019-2022        2     0       1  0.047452
                                  overnight20 q3 & r1 q1 2023-2026        4     1       2  0.171908
                                 overnight20 q3 & r20 q3 2015-2018        3     1       1 -0.687214
                                 overnight20 q3 & r20 q3 2019-2022        4     1       3  0.282881
                                 overnight20 q3 & r20 q3 2023-2026        4     1       2 -0.054154
                      overnight20 q3 & range_compress q1 2015-2018        2     0       2 -0.482113
                      overnight20 q3 & range_compress q1 2019-2022        4     2       1  0.635127
                      overnight20 q3 & range_compress q1 2023-2026        4     1       2  0.257274
                                       r1 q0 & skew60 q4 2023-2026        2     2       0 -2.001818
                                        r1 q0 & vol20 q2 2023-2026        2     0       0  0.239555
                                        r1 q0 & vol20 q3 2023-2026        2     0       0  0.121726
                                   r1 q0 & vol_surge5 q3 2023-2026        2     0       1 -0.375186
                                                   r1 q1 2019-2022        3     2       1  0.791815
                                                   r1 q1 2023-2026        4     1       1  0.463727
                                        r1 q1 & vol20 q2 2023-2026        2     0       0  0.694273
                                           r1 q2 & r5 q1 2015-2018        3     1       1  0.500415
                                           r1 q2 & r5 q1 2019-2022        4     1       1  0.758552
                                           r1 q2 & r5 q1 2023-2026        4     1       3 -0.226940
                                    r1 q2 & rel_ind60 q1 2015-2018        2     1       0 -1.313771
                                    r1 q2 & rel_ind60 q1 2019-2022        4     1       2 -0.016416
                                    r1 q2 & rel_ind60 q1 2023-2026        4     0       2  0.540255
                                       r1 q2 & skew60 q2 2015-2018        3     2       1  0.944950
                                       r1 q2 & skew60 q2 2019-2022        4     0       4 -0.527560
                                       r1 q2 & skew60 q2 2023-2026        4     2       2  0.430623
                                       r1 q2 & skew60 q3 2019-2022        4     1       0 -0.456768
                                       r1 q2 & skew60 q3 2023-2026        4     2       1 -0.547133
                                   r1 q4 & vol_surge1 q2 2023-2026        2     1       0  0.760292
                                                 r120 q1 2023-2026        4     3       1  0.944645
                                        r120 q3 & r60 q3 2019-2022        1     0       0  0.788357
                                        r120 q3 & r60 q3 2023-2026        4     0       2 -0.012527
                                     r120 q3 & skew60 q0 2023-2026        2     1       0  0.848284
                                         r120 q4 & r5 q2 2015-2018        1     0       0  0.949209
                                         r120 q4 & r5 q2 2019-2022        4     2       1  0.800596
                                         r120 q4 & r5 q2 2023-2026        4     0       4 -0.341146
                                                  r20 q1 2015-2018        3     0       1 -0.057281
                                                  r20 q1 2019-2022        4     3       0  1.549299
                                                  r20 q1 2023-2026        4     1       2  0.286970
                              r20 q1 & range_compress q2 2019-2022        3     1       1  0.552273
                              r20 q1 & range_compress q2 2023-2026        4     1       3 -0.041199
                              r20 q1 & range_compress q4 2023-2026        4     1       2 -0.458090
                                   r20 q1 & rel_ind20 q0 2023-2026        3     1       2  0.145367
                                   r20 q1 & rel_ind60 q2 2019-2022        3     3       0  2.132393
                                   r20 q1 & rel_ind60 q2 2023-2026        4     0       2  0.075880
                                      r20 q1 & skew60 q2 2023-2026        2     1       0  0.558361
                                   r20 q1 & vol_ratio q0 2019-2022        4     1       1  0.575513
                                   r20 q1 & vol_ratio q0 2023-2026        4     0       2 -0.188502
                                   r20 q1 & vol_ratio q1 2019-2022        4     2       2  0.600795
                                   r20 q1 & vol_ratio q1 2023-2026        4     0       3 -0.055019
                                  r20 q1 & vol_surge1 q1 2019-2022        3     1       1  0.541452
                                  r20 q1 & vol_surge1 q1 2023-2026        4     2       1  0.597791
                              r20 q2 & range_compress q2 2019-2022        1     0       1  1.263160
                              r20 q2 & range_compress q2 2023-2026        4     1       1 -0.369023
                                   r20 q2 & rel_ind20 q4 2023-2026        4     1       2  0.224980
                                                  r20 q3 2015-2018        4     0       2  0.284096
                                                  r20 q3 2019-2022        4     1       1 -0.704472
                                                  r20 q3 2023-2026        4     0       2  0.007127
                              r20 q3 & range_compress q2 2015-2018        4     2       1  0.636610
                              r20 q3 & range_compress q2 2019-2022        4     1       1  0.169941
                              r20 q3 & range_compress q2 2023-2026        4     1       3 -0.395312
                                   r20 q3 & rel_ind20 q3 2015-2018        4     1       3  0.156793
                                   r20 q3 & rel_ind20 q3 2019-2022        4     2       1 -1.240908
                                   r20 q3 & rel_ind20 q3 2023-2026        4     0       2  0.329348
                                   r20 q3 & rel_ind60 q1 2019-2022        4     1       2  0.107877
                                   r20 q3 & rel_ind60 q1 2023-2026        4     2       0  1.099899
                                       r20 q3 & vol20 q0 2015-2018        3     0       1 -0.432146
                                       r20 q3 & vol20 q0 2019-2022        4     1       2 -0.043000
                                       r20 q3 & vol20 q0 2023-2026        4     1       2 -0.242390
                                   r20 q3 & vol_ratio q4 2019-2022        4     1       2 -0.232205
                                   r20 q3 & vol_ratio q4 2023-2026        4     0       1 -0.143853
                                                  r20 q4 2015-2018        2     1       1 -0.497823
                                                  r20 q4 2019-2022        4     2       0 -0.928383
                                                  r20 q4 2023-2026        4     0       3  0.091623
                                   r20 q4 & r5_nonews q4 2023-2026        4     1       3  0.055293
                              r20 q4 & range_compress q2 2019-2022        1     0       1 -0.926378
                              r20 q4 & range_compress q2 2023-2026        4     0       0  0.330769
                                   r20 q4 & rel_ind60 q2 2019-2022        3     1       1 -0.392138
                                   r20 q4 & rel_ind60 q2 2023-2026        4     1       2  0.061897
                                      r20 q4 & skew60 q1 2019-2022        4     0       0 -0.317546
                                      r20 q4 & skew60 q1 2023-2026        4     0       1 -0.378979
                                  r20 q4 & vol_surge5 q0 2023-2026        4     2       2 -0.431183
                                                   r5 q0 2015-2018        3     1       0  0.884434
                                                   r5 q0 2019-2022        4     2       1  0.442977
                                                   r5 q0 2023-2026        4     0       2 -0.258006
                                    r5 q0 & rel_ind60 q1 2019-2022        2     1       0  1.959970
                                    r5 q0 & rel_ind60 q1 2023-2026        4     0       1  0.294061
                                   r5 q0 & vol_surge5 q4 2023-2026        1     1       0  1.103013
                                                   r5 q1 2015-2018        4     3       1  1.538664
                                                   r5 q1 2019-2022        4     1       1  0.483222
                                                   r5 q1 2023-2026        4     2       2  0.655842
                  r5 q1 & r5_news q2 unless gap_today q4 2023-2026        3     0       1  0.143317
                 r5 q1 & r5_news q2 unless vol_surge1 q0 2023-2026        3     1       1  0.115579
                                          r5 q1 & r60 q3 2015-2018        2     1       0  0.654817
                                          r5 q1 & r60 q3 2019-2022        4     1       0  0.780995
                                          r5 q1 & r60 q3 2023-2026        4     1       2  0.057150
                               r5 q1 & range_compress q2 2019-2022        3     0       1 -0.031416
                               r5 q1 & range_compress q2 2023-2026        4     0       2 -0.081956
                                    r5 q1 & rel_ind60 q0 2019-2022        2     0       0 -0.626045
                                    r5 q1 & rel_ind60 q0 2023-2026        4     1       3  0.278715
                                    r5 q1 & rel_ind60 q2 2019-2022        3     1       1  0.608014
                                    r5 q1 & rel_ind60 q2 2023-2026        4     1       2  0.159690
                                        r5 q1 & vol20 q1 2023-2026        3     0       0  0.321769
                                    r5 q1 & vol_ratio q1 2019-2022        2     1       0 -1.102200
                                    r5 q1 & vol_ratio q1 2023-2026        4     1       2 -0.043700
                                    r5 q1 & vol_ratio q3 2023-2026        2     1       0  1.502646
                                   r5 q1 & vol_surge5 q2 2023-2026        2     1       1 -0.579756
                                                   r5 q3 2019-2022        3     0       2 -0.411945
                                                   r5 q3 2023-2026        4     1       1  0.380310
                                    r5 q3 & r5_nonews q3 2019-2022        3     0       1  0.005524
                                    r5 q3 & r5_nonews q3 2023-2026        4     0       1  0.281975
                                        r5 q3 & vol20 q3 2023-2026        3     1       1  0.313153
                                                   r5 q4 2019-2022        3     2       1 -1.048750
                                                   r5 q4 2023-2026        4     1       2 -0.070853
                                   r5 q4 & vol_surge1 q2 2023-2026        3     1       1 -0.560980
                                              r5_news q0 2023-2026        4     2       1 -0.566557
                                              r5_news q2 2019-2022        2     0       0  0.583557
                                              r5_news q2 2023-2026        4     3       1  0.952561
                               r5_news q2 & rel_ind60 q2 2019-2022        3     1       1  0.931505
                               r5_news q2 & rel_ind60 q2 2023-2026        4     0       3 -0.441923
                                   r5_news q2 & vol20 q3 2023-2026        3     1       1  0.400604
                                 r5_nonews q0 & vol20 q3 2019-2022        3     1       0  0.647862
                                 r5_nonews q0 & vol20 q3 2023-2026        4     2       1  0.797973
                            r5_nonews q0 & vol_surge5 q3 2023-2026        2     1       1 -0.153315
                                            r5_nonews q1 2015-2018        3     1       1 -0.281128
                                            r5_nonews q1 2019-2022        4     1       3  0.209476
                                            r5_nonews q1 2023-2026        4     2       2 -0.160690
                                   r5_nonews q1 & r60 q4 2023-2026        3     1       0 -0.623767
                        r5_nonews q1 & range_compress q2 2019-2022        2     0       1 -0.058739
                        r5_nonews q1 & range_compress q2 2023-2026        4     0       2 -0.031168
                                 r5_nonews q1 & vol20 q1 2023-2026        3     0       0 -0.234783
                             r5_nonews q1 & vol_ratio q3 2023-2026        2     1       1  1.104900
                             r5_nonews q2 & rel_ind60 q2 2019-2022        3     1       1  0.826055
                             r5_nonews q2 & rel_ind60 q2 2023-2026        4     0       2 -0.221674
                             r5_nonews q2 & rel_ind60 q4 2023-2026        2     2       0 -1.642166
                                r5_nonews q3 & skew60 q3 2019-2022        2     0       0  0.683284
                                r5_nonews q3 & skew60 q3 2023-2026        4     1       2 -0.148359
                            r5_nonews q3 & vol_surge5 q2 2023-2026        2     0       1  0.061266
                            r5_nonews q3 & vol_surge5 q4 2015-2018        2     1       1 -1.238250
                            r5_nonews q3 & vol_surge5 q4 2019-2022        4     1       2 -0.223025
                            r5_nonews q3 & vol_surge5 q4 2023-2026        4     2       1 -0.828040
                                            r5_nonews q4 2019-2022        3     2       1 -1.366275
                                            r5_nonews q4 2023-2026        4     1       1 -0.565850
                            r5_nonews q4 & vol_surge1 q2 2019-2022        3     2       0 -1.159385
                            r5_nonews q4 & vol_surge1 q2 2023-2026        4     2       1 -0.389551
                            r5_nonews q4 & vol_surge5 q0 2019-2022        1     1       0 -3.668581
                            r5_nonews q4 & vol_surge5 q0 2023-2026        4     2       1 -0.690558
                                                  r60 q1 2019-2022        4     2       1  0.955886
                                                  r60 q1 2023-2026        4     3       0  1.018625
                                   r60 q1 & rel_ind60 q2 2019-2022        2     1       0  1.222968
                                   r60 q1 & rel_ind60 q2 2023-2026        4     1       3  0.206283
                                                  r60 q3 2015-2018        3     0       1 -0.400491
                                                  r60 q3 2019-2022        4     2       1  0.864805
                                                  r60 q3 2023-2026        4     0       1  0.393942
                              r60 q3 & range_compress q0 2023-2026        2     0       0  0.718214
                                      r60 q4 & skew60 q4 2023-2026        2     1       0 -1.026476
                                   r60 q4 & vol_ratio q3 2019-2022        2     0       1  0.102640
                                   r60 q4 & vol_ratio q3 2023-2026        4     2       1 -0.731817
                        range_compress q0 & rel_ind20 q1 2019-2022        1     1       0 -2.541211
                        range_compress q0 & rel_ind20 q1 2023-2026        4     0       2  0.241210
                        range_compress q0 & vol_ratio q3 2019-2022        2     0       0 -0.604659
                        range_compress q0 & vol_ratio q3 2023-2026        4     2       2 -0.496333
                        range_compress q1 & rel_ind20 q3 2019-2022        3     1       0 -0.600647
                        range_compress q1 & rel_ind20 q3 2023-2026        4     3       0 -1.293501
                        range_compress q2 & rel_ind20 q3 2023-2026        3     1       1  0.229995
                        range_compress q2 & rel_ind20 q4 2019-2022        3     0       0 -0.808938
                        range_compress q2 & rel_ind20 q4 2023-2026        4     0       0 -0.491687
                        range_compress q2 & rel_ind60 q2 2015-2018        1     0       1  1.066679
                        range_compress q2 & rel_ind60 q2 2019-2022        4     2       1 -0.358431
                        range_compress q2 & rel_ind60 q2 2023-2026        4     1       0 -0.607833
                            range_compress q2 & vol20 q2 2023-2026        2     0       0  0.640277
                       range_compress q2 & vol_surge1 q1 2023-2026        4     1       1  0.531590
                       range_compress q2 & vol_surge5 q3 2019-2022        1     0       1 -0.518500
                       range_compress q2 & vol_surge5 q3 2023-2026        4     2       0  1.414664
                       range_compress q3 & vol_surge5 q1 2015-2018        4     0       2 -0.224991
                       range_compress q3 & vol_surge5 q1 2019-2022        4     1       2  0.053131
                       range_compress q3 & vol_surge5 q1 2023-2026        4     0       1  0.385317
                                       range_compress q4 2015-2018        3     1       0 -0.644639
                                       range_compress q4 2019-2022        4     1       2  0.189492
                                       range_compress q4 2023-2026        4     0       2 -0.058365
                        range_compress q4 & rel_ind60 q3 2015-2018        3     1       2 -0.320054
                        range_compress q4 & rel_ind60 q3 2019-2022        4     1       2  0.268976
                        range_compress q4 & rel_ind60 q3 2023-2026        4     0       1 -0.212785
                           range_compress q4 & skew60 q0 2015-2018        4     1       1  0.870063
                           range_compress q4 & skew60 q0 2019-2022        4     1       1  0.246209
                           range_compress q4 & skew60 q0 2023-2026        4     0       2 -0.166268
                             rel_ind20 q1 & vol_ratio q4 2023-2026        1     0       0 -0.760459
                                            rel_ind20 q2 2019-2022        3     1       1  0.405941
                                            rel_ind20 q2 2023-2026        4     0       1  0.279725
                                            rel_ind20 q3 2015-2018        3     1       2 -0.370698
                                            rel_ind20 q3 2019-2022        4     2       1 -0.837473
                                            rel_ind20 q3 2023-2026        4     0       2  0.188230
                                rel_ind20 q3 & skew60 q2 2015-2018        1     0       1 -0.134570
                                rel_ind20 q3 & skew60 q2 2019-2022        4     0       2 -0.634328
                                rel_ind20 q3 & skew60 q2 2023-2026        4     2       0  1.396007
                            rel_ind20 q3 & vol_surge1 q3 2015-2018        2     1       0  0.835303
                            rel_ind20 q3 & vol_surge1 q3 2019-2022        4     1       2 -0.036952
                            rel_ind20 q3 & vol_surge1 q3 2023-2026        4     1       3 -0.278491
                                            rel_ind20 q4 2015-2018        2     0       1 -0.301332
                                            rel_ind20 q4 2019-2022        4     2       0 -0.732893
                                            rel_ind20 q4 2023-2026        4     1       2 -0.083211
                            rel_ind20 q4 & vol_surge5 q0 2023-2026        4     0       2 -0.115724
                                            rel_ind60 q1 2015-2018        4     1       3 -0.054985
                                            rel_ind60 q1 2019-2022        4     1       2  0.588639
                                            rel_ind60 q1 2023-2026        4     2       0  1.442176
                                            rel_ind60 q2 2015-2018        1     1       0  3.519611
                                            rel_ind60 q2 2019-2022        4     2       1  0.996585
                                            rel_ind60 q2 2023-2026        4     0       2 -0.113266
                                 rel_ind60 q2 & vol20 q2 2023-2026        2     0       1  0.025657
                             rel_ind60 q2 & vol_ratio q2 2015-2018        2     1       0  0.853347
                             rel_ind60 q2 & vol_ratio q2 2019-2022        4     1       1  0.189912
                             rel_ind60 q2 & vol_ratio q2 2023-2026        4     0       3 -0.442277
                            rel_ind60 q2 & vol_surge1 q3 2019-2022        3     3       0  1.437471
                            rel_ind60 q2 & vol_surge1 q3 2023-2026        4     0       3 -0.426562
        rel_ind60 q2 & vol_surge5 q2 unless close_loc q4 2019-2022        1     0       1  0.123893
        rel_ind60 q2 & vol_surge5 q2 unless close_loc q4 2023-2026        4     2       2 -0.351452
        rel_ind60 q2 & vol_surge5 q2 unless ind_mom20 q4 2019-2022        1     0       0 -0.535382
        rel_ind60 q2 & vol_surge5 q2 unless ind_mom20 q4 2023-2026        4     0       1 -0.144353
                            rel_ind60 q2 & vol_surge5 q3 2019-2022        1     1       0  2.108557
                            rel_ind60 q2 & vol_surge5 q3 2023-2026        4     1       2  0.278506
                                            rel_ind60 q3 2015-2018        3     0       2 -0.492339
                                            rel_ind60 q3 2019-2022        4     1       2  0.402293
                                            rel_ind60 q3 2023-2026        4     0       2  0.377060
                                            rel_ind60 q4 2015-2018        2     0       2  0.501633
                                            rel_ind60 q4 2019-2022        4     1       0 -0.651837
                                            rel_ind60 q4 2023-2026        4     1       2 -0.377481
                                               skew60 q1 2019-2022        1     0       0  0.509164
                                               skew60 q1 2023-2026        4     2       0  0.890025
                               skew60 q1 & vol_surge1 q1 2023-2026        4     2       1  0.588482
                                               skew60 q2 2015-2018        2     1       0  0.619572
                                               skew60 q2 2019-2022        4     1       3 -0.074371
                                               skew60 q2 2023-2026        4     2       0  1.264799
                                               skew60 q3 2019-2022        1     0       0 -0.392043
                                               skew60 q3 2023-2026        4     0       2 -0.042638
                                               skew60 q4 2019-2022        1     0       0 -0.851174
                                               skew60 q4 2023-2026        4     1       1 -0.841583
                               skew60 q4 & vol_surge1 q0 2023-2026        1     0       0 -0.624348
                               skew60 q4 & vol_surge5 q0 2023-2026        1     1       0 -1.288270
                                                vol20 q0 2015-2018        3     0       1 -0.134967
                                                vol20 q0 2019-2022        4     1       2 -0.245553
                                                vol20 q0 2023-2026        4     1       3 -0.277132
                                vol20 q0 & vol_surge1 q4 2015-2018        3     0       0 -0.513407
                                vol20 q0 & vol_surge1 q4 2019-2022        4     1       2 -0.480956
                                vol20 q0 & vol_surge1 q4 2023-2026        4     0       2  0.194135
                                                vol20 q2 2023-2026        4     3       0  1.419460
                                 vol20 q2 & vol_ratio q2 2023-2026        2     2       0  1.237166
                                 vol20 q2 & vol_ratio q3 2023-2026        2     0       0 -0.464914
                                                vol20 q3 2019-2022        4     2       2  0.287389
                                                vol20 q3 2023-2026        4     2       0  1.663168
                                 vol20 q3 & vol_ratio q1 2019-2022        4     1       3 -0.222055
                                 vol20 q3 & vol_ratio q1 2023-2026        4     1       0  1.030234
                                 vol20 q3 & vol_ratio q3 2023-2026        3     1       2  0.043941
                                 vol20 q3 & vol_ratio q4 2023-2026        3     1       1  0.398331
                                vol20 q3 & vol_surge1 q2 2019-2022        3     1       1  0.556009
                                vol20 q3 & vol_surge1 q2 2023-2026        4     0       3 -0.207255
                                                vol20 q4 2023-2026        4     1       0 -0.890735
                                vol20 q4 & vol_surge1 q0 2023-2026        1     0       0 -0.604818
                                vol20 q4 & vol_surge5 q0 2023-2026        1     0       0 -0.458363
                                            vol_ratio q0 2015-2018        1     0       0  0.333551
                                            vol_ratio q0 2019-2022        4     0       0  0.557621
                                            vol_ratio q0 2023-2026        4     1       2 -0.152796
                                            vol_ratio q3 2019-2022        2     0       1  0.322752
                                            vol_ratio q3 2023-2026        4     1       1  0.621017
                                            vol_ratio q4 2015-2018        2     1       1  0.085958
                                            vol_ratio q4 2019-2022        4     1       1 -0.692085
                                            vol_ratio q4 2023-2026        4     1       2 -0.057238
                           vol_surge1 q0 & vol_surge5 q0 2023-2026        1     0       0 -0.780563
                                           vol_surge1 q1 2015-2018        2     0       0  0.307051
                                           vol_surge1 q1 2019-2022        4     1       0  0.500512
                                           vol_surge1 q1 2023-2026        4     2       1  0.652038
                                           vol_surge1 q2 2019-2022        2     1       0  0.846326
                                           vol_surge1 q2 2023-2026        4     0       3 -0.126891
                                           vol_surge1 q3 2015-2018        2     1       0  0.985729
                                           vol_surge1 q3 2019-2022        4     2       2  0.210659
                                           vol_surge1 q3 2023-2026        4     0       3  0.133962
                                           vol_surge5 q0 2019-2022        1     1       0 -2.518397
                                           vol_surge5 q0 2023-2026        4     2       1 -0.484380
                                           vol_surge5 q1 2019-2022        3     0       1  0.307038
                                           vol_surge5 q1 2023-2026        4     1       0  0.933850
                                           vol_surge5 q2 2023-2026        3     0       1  0.041416
                                           vol_surge5 q3 2019-2022        3     2       1  1.511630
                                           vol_surge5 q3 2023-2026        4     2       0  0.815760

