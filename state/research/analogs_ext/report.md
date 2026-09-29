# Phase 8 analog engines on real caches

- seed 7, 250 sampled tickers, refit every 1008 sessions, peak RSS 451 MB
- survivor universe: sector/stock outcomes are optimistic, compare levels only against controls on the same rows
- audit dirty rows (must be 0): market 0, sector 0, stock 0
- causality offenders (must be empty): 0
- weight-learner false acceptance: real vs shuffled outcomes, IC {'accept_real': 0.25, 'accept_null_mean': 0.5416666666666666, 'accept_null_max': 0.75, 'n_shuffles': 3}, descent {'accept_real': 0.3333333333333333, 'accept_null_mean': 0.5, 'accept_null_max': 0.5, 'n_shuffles': 2}

## Market ablation (fwd 21d return, vs no-analog)

                n     mse     mae    corr     hit  skill_ctl   t_ctl   p_ctl  skill_ref   t_ref   p_ref
none        621.0  0.0022  0.0332 -0.0419  0.6184     0.0744  4.9797  0.0005        NaN     NaN     NaN
unweighted  621.0  0.0026  0.0363  0.0308  0.5298    -0.0943 -2.2790  0.9775    -0.1824 -4.3888  1.0000
weighted    621.0  0.0025  0.0362  0.0336  0.5330    -0.0856 -2.0755  0.9630    -0.1729 -4.2150  1.0000
random      621.0  0.0023  0.0351 -0.0438  0.5282        NaN     NaN     NaN    -0.0804 -4.9797  1.0000
shuffled    621.0  0.0022  0.0346  0.0557  0.5411     0.0421  1.4800  0.0515    -0.0349 -1.5377  0.9655
nn_random   621.0  0.0025  0.0355  0.0026  0.5298    -0.0777 -1.8793  0.9545    -0.1644 -4.0362  1.0000

## Market per-decade skill

mode   nn_random  none  random  shuffled  unweighted  weighted
era                                                           
1970s    -0.2833   NaN -0.1393   -0.1453     -0.2745   -0.2745
1980s    -0.0915   NaN -0.0541   -0.0048     -0.1251   -0.0880
1990s    -0.1337   NaN -0.0715   -0.1051     -0.1778   -0.1778
2000s    -0.2588   NaN -0.0465    0.0079     -0.2527   -0.2141
2010s    -0.1632   NaN -0.2111   -0.1697     -0.3166   -0.4150
2020s    -0.1116   NaN -0.0747    0.0352     -0.0708   -0.0708

## Market confidence calibration

              n  confidence     mae     hit
confidence                                 
0           156      0.0307  0.0367  0.4679
1           155      0.1641  0.0322  0.4968
2           155      0.3577  0.0396  0.5806
3           155      0.6334  0.0362  0.5871

## Market metric/k sweep

                   n     mse    corr     hit  skill_ref   p_ref
euclid_k3      311.0  0.0027  0.0590  0.5113    -0.2847  0.9995
euclid_k5      311.0  0.0024  0.0586  0.5145    -0.1622  0.9985
euclid_k10     311.0  0.0023  0.0385  0.5627    -0.1063  0.9785
manhattan_k3   311.0  0.0027  0.0690  0.5145    -0.2864  1.0000
manhattan_k5   311.0  0.0025  0.0596  0.5370    -0.1944  0.9985
manhattan_k10  311.0  0.0023  0.0126  0.5209    -0.1207  0.9975
cosine_k3      311.0  0.0028  0.0258  0.5177    -0.3351  1.0000
cosine_k5      311.0  0.0025  0.0491  0.5048    -0.1913  1.0000
cosine_k10     311.0  0.0023  0.0317  0.5498    -0.1185  0.9885

## Market feature weights

                      mean_weight  share_above_uniform  stability
dispersion                 2.2431                 0.75     0.0423
ret_1m                     1.5212                 0.50     0.0423
top_sector_share_chg       1.4901                 0.75     0.0423
breadth                    1.3731                 0.50     0.0423
ret_3m                     1.1822                 0.75     0.0423
vol_ratio                  1.1174                 0.50     0.0423
accel                      0.9409                 0.25     0.0423
dist_ma200                 0.9332                 0.25     0.0423
vol_1m                     0.7215                 0.25     0.0423
sector_crowding            0.5666                 0.25     0.0423
drawdown                   0.4745                 0.25     0.0423
ret_6m                     0.3842                 0.00     0.0423
rate_chg_1y                0.3789                 0.00     0.0423
ret_12m                    0.3371                 0.00     0.0423
vix                           NaN                 0.00     0.0423
vix_term                      NaN                 0.00     0.0423
credit_chg_3m                 NaN                 0.00     0.0423

## Market weight refits

         date  accepted  n_train  n_val  val_uniform  val_learned
0  1963-12-31      True       60     40       3.5333       3.2369
1  1967-12-29     False       60     40       1.4282       1.4869
2  1972-02-04     False       60     40       2.9000       2.8684
3  1976-02-04     False       60     40       1.8536       2.1262
4  1980-01-31      True       60     40       1.0611       1.0227
5  1984-01-26     False       60     40       1.0076       1.3394
6  1988-01-22     False       60     40       2.3207       2.3355
7  1992-01-17     False       60     40       0.6699       0.8288
8  1996-01-12     False       60     40       0.6530       0.9105
9  2000-01-10     False       60     40       0.5392       0.5836
10 2004-01-15      True       60     40       1.6561       1.4666
11 2008-01-17     False       60     40       1.5291       1.5166
12 2012-01-18      True       60     40       1.1540       1.1249
13 2016-01-21     False       60     40       0.6791       0.8759
14 2020-01-23     False       60     40       0.9260       1.0286
15 2024-01-25     False       60     40       1.4695       1.4939

## Market weight learners (uniform vs descent vs IC)

                     n     mse     mae    corr     hit  skill_ctl   t_ctl   p_ctl  skill_ref   t_ref  p_ref
none             621.0  0.0022  0.0332 -0.0419  0.6184     0.1542  4.3888  0.0005        NaN     NaN    NaN
uniform          621.0  0.0026  0.0363  0.0308  0.5298        NaN     NaN     NaN    -0.1824 -4.3888    1.0
learned_descent  621.0  0.0025  0.0362  0.0336  0.5330     0.0080  0.8115  0.2284    -0.1729 -4.2150    1.0
learned_ic       621.0  0.0026  0.0364  0.0374  0.5169     0.0007  0.0787  0.4813    -0.1815 -4.3757    1.0

## Fingerprint coverage (Bible 8.1)

                                              columns  present  non_nan      first       last
item                                                                                         
market drawdown                              drawdown     True   0.9964 1962-03-27 2026-09-28
1m return                                      ret_1m     True   0.9987 1962-01-31 2026-09-28
3m return                                      ret_3m     True   0.9961 1962-04-02 2026-09-28
6m return                                      ret_6m     True   0.9923 1962-07-02 2026-09-28
12m return                                    ret_12m     True   0.9845 1963-01-02 2026-09-28
acceleration                                    accel     True   0.9845 1963-01-02 2026-09-28
distance from MA200                        dist_ma200     True   0.9939 1962-05-23 2026-09-28
volatility                                     vol_1m     True   0.9991 1962-01-23 2026-09-28
volatility ratio                            vol_ratio     True   0.9939 1962-05-24 2026-09-28
VIX                                               vix     True   0.5679 1990-01-02 2026-09-28
VIX term structure                           vix_term     True   0.3119 2006-07-17 2026-09-28
breadth                                       breadth     True   0.9999 1962-01-02 2026-09-25
dispersion                                 dispersion     True   0.9994 1962-01-16 2026-09-28
sector crowding                       sector_crowding     True   1.0000 1962-01-02 2026-09-28
top-sector concentration change  top_sector_share_chg     True   0.9956 1962-04-13 2026-09-28
FRED data                                                False   0.0000        NaT        NaT
10Y yield change                          rate_chg_1y     True   0.9845 1963-01-03 2026-09-28
credit-spread change                    credit_chg_3m     True   0.0422 2024-01-02 2026-09-28

## Sector ablation by sector

        n_names  n_eval  mse_none  mse_unweighted  mse_weighted  skill_unweighted  skill_weighted  p_weighted_vs_none  skill_random  skill_shuffled  skill_nn_random  beats_random  beats_unweighted
sector                                                                                                                                                                                              
28           34     409    0.0061          0.0070        0.0069           -0.1553         -0.1442              0.9955       -0.0908         -0.0464          -0.0668         False             False
35            7     322    0.0114          0.0127        0.0131           -0.1103         -0.1478              0.9975       -0.0576         -0.0586          -0.1109         False             False
36            9     191    0.0079          0.0087        0.0091           -0.1126         -0.1594              0.9155       -0.0536         -0.0378          -0.1196         False             False
37           11     621    0.0030          0.0036        0.0036           -0.1728         -0.1842              1.0000       -0.0629         -0.0782          -0.1086         False             False
38           20     324    0.0037          0.0044        0.0044           -0.1855         -0.1815              0.9885       -0.0665         -0.2098          -0.0895         False             False
60           22     492    0.0025          0.0028        0.0028           -0.1349         -0.1355              1.0000       -0.0680         -0.0614          -0.0865         False             False
67           11     340    0.0038          0.0045        0.0045           -0.2004         -0.1853              1.0000       -0.0488         -0.0578          -0.0656         False             False
73           33     440    0.0060          0.0065        0.0068           -0.0828         -0.1291              0.9970       -0.0421         -0.1037          -0.0586         False             False

## Sector weighted skill by decade (mean over sectors)

              n  skill_ref
era                       
1970s   61.0000    -0.1272
1980s   86.0000    -0.1311
1990s   67.8571    -0.1997
2000s  120.0000    -0.2695
2010s  118.8750    -0.2638
2020s   80.0000    -0.0343

## Stock own-history ablation (fwd 5d return)

        n_eval  mse_none  skill_unweighted  p_unweighted  skill_random  skill_shuffled  skill_nn_random
ticker                                                                                                 
GT         267    0.0031           -0.3215        1.0000        0.0057         -0.0387          -0.1272
HON        267    0.0014           -0.3861        1.0000       -0.0294         -0.0654          -0.1387
MSI        267    0.0023           -0.1730        0.9870       -0.0662         -0.1471          -0.0763
MCD        270    0.0014           -0.1945        0.9500       -0.1091         -0.0428          -0.3748
F          268    0.0026           -0.0859        0.9410       -0.0276         -0.0813          -0.0031
DE         268    0.0021           -0.1642        0.9755       -0.0545         -0.0558          -0.0683
TXNM       264    0.0010           -0.3391        1.0000       -0.0802         -0.0655          -0.2195
RGR        263    0.0026           -0.3244        1.0000       -0.0354         -0.0521          -0.1622
AVT        263    0.0033           -0.0994        0.9350       -0.0272         -0.0346          -0.0028
DCO        263    0.0043           -0.1877        0.9945       -0.0083         -0.0771          -0.0792
OII        251    0.0050           -0.3425        1.0000       -0.1057         -0.1720          -0.1712
HELE       275    0.0047           -0.4392        1.0000       -0.1078         -0.0941          -0.2477

## Pooled stock analog rank IC by half-decade

        mean  count
2005  0.0116     45
2010 -0.0209     60
2015  0.0486     60
2020  0.0119     60
2025 -0.0137     20
