# timeline / objective / basis-search on real weekly baskets (seed 1)

600 sampled tickers, 26 windows of 52 weeks. Proxy portfolio, see caveats in JSON.

## share of weeks in the 5-10% band, by basket and era

| basket | overall | 2000-2008 | 2009-2015 | 2016-2026 |
|---|---|---|---|---|
| k1_pool0.5 | 24% | 21% | 24% | 25% |
| k1_pool0.9 | 25% | 26% | 24% | 25% |
| k2_pool0.5 | 25% | 26% | 24% | 24% |
| k2_pool0.9 | 28% | 27% | 29% | 28% |
| k3_pool0.5 | 23% | 22% | 19% | 26% |
| k3_pool0.9 | 29% | 30% | 26% | 31% |
| k5_pool0.5 | 22% | 18% | 25% | 24% |
| k5_pool0.9 | 29% | 28% | 27% | 30% |
| k10_pool0.5 | 20% | 15% | 19% | 23% |
| k10_pool0.9 | 25% | 24% | 21% | 29% |
| k25_pool0.5 | 15% | 13% | 15% | 16% |
| k25_pool0.9 | 25% | 23% | 21% | 30% |

tier1-vs-tier2 rank correlation across baskets: -0.66

## dial gate (train early, test later)

- k1_pool0.9: FAIL: no reliable improvement (no measurable gain on any tier; bootstrap lower bound +0.000); train gain +0.019 did not carry to test +0.000; not stable across eras (per-era gain [0.0, 0.0, 0.0]) (in-band share on test, base -> dial: 25% -> 18%)
- k3_pool0.9: FAIL: no reliable improvement (worse on tier1 (weeks in band); bootstrap lower bound -0.052); train gain +0.021 did not carry to test -0.011; not stable across eras (per-era gain [-0.094, 0.067, -0.006]) (in-band share on test, base -> dial: 31% -> 22%)
- k10_pool0.5: FAIL: no reliable improvement (worse on tier1 (weeks in band); bootstrap lower bound -0.056); train gain +0.026 did not carry to test -0.032; not stable across eras (per-era gain [-0.046, -0.005, -0.051]) (in-band share on test, base -> dial: 23% -> 19%)

## basis search (as_of 2012)

- adopted: True (ADOPTED: eligible: better on tier1 (weeks in band)); cfg {'k': 10, 'pool_q': 0.7}
- out of sample in-band share: incumbent 13% -> adopted 23%
- risk: -0.172 -> -0.268
