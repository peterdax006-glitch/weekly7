# timeline / objective / basis-search on real weekly baskets (seed 0)

600 sampled tickers, 26 windows of 52 weeks. Proxy portfolio, see caveats in JSON.

## share of weeks in the 5-10% band, by basket and era

| basket | overall | 2000-2008 | 2009-2015 | 2016-2026 |
|---|---|---|---|---|
| k1_pool0.5 | 23% | 21% | 25% | 24% |
| k1_pool0.9 | 23% | 21% | 23% | 24% |
| k2_pool0.5 | 23% | 24% | 23% | 23% |
| k2_pool0.9 | 29% | 29% | 26% | 31% |
| k3_pool0.5 | 24% | 22% | 24% | 25% |
| k3_pool0.9 | 28% | 26% | 29% | 28% |
| k5_pool0.5 | 21% | 19% | 23% | 22% |
| k5_pool0.9 | 29% | 31% | 28% | 28% |
| k10_pool0.5 | 17% | 16% | 13% | 19% |
| k10_pool0.9 | 28% | 28% | 27% | 27% |
| k25_pool0.5 | 15% | 13% | 14% | 16% |
| k25_pool0.9 | 23% | 22% | 18% | 27% |

tier1-vs-tier2 rank correlation across baskets: -0.69

## dial gate (train early, test later)

- k1_pool0.9: FAIL: no reliable improvement (better on tier1 (weeks in band); bootstrap lower bound -0.011) (in-band share on test, base -> dial: 24% -> 16%)
- k3_pool0.9: FAIL: no reliable improvement (worse on tier1 (weeks in band); bootstrap lower bound -0.040); train gain +0.013 did not carry to test -0.005; not stable across eras (per-era gain [-0.055, 0.019, 0.029]) (in-band share on test, base -> dial: 28% -> 22%)
- k10_pool0.5: FAIL: no reliable improvement (better on tier1 (weeks in band); bootstrap lower bound -0.020); train gain +0.024 did not carry to test +0.007; not stable across eras (per-era gain [0.022, -0.01, 0.01]) (in-band share on test, base -> dial: 19% -> 21%)

## basis search (as_of 2012)

- adopted: True (ADOPTED: eligible: better on tier1 (weeks in band)); cfg {'k': 25, 'pool_q': 0.7}
- out of sample in-band share: incumbent 12% -> adopted 17%
- risk: -0.171 -> -0.193
