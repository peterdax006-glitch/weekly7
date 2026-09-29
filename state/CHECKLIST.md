# Weekly7 checkoff list (canon C32; priority C36: Algorithm > Find volatility > minimal Test changes > Live)

Work through every item; check it off only with evidence. Use all the RAM; don't stop until the list is done.

## Live
- [ ] L1 Decide the live upgrade (owner decision, once Test or Volatility has a proven improvement)

## Test
- [ ] T1 Rebuild the 39 archived windows with the move-likelihood signal
- [ ] T2 Resume the self-learning loop
- [ ] T3 Reach a +7% weekly average (7% first, accuracy second)
- [~] T4 Make weekly self-adjustment actually fire: now fires (4 switches/reverts on a test window) via the factor-weighted memory (C34); the loop tunes how aggressively
- [x] T5 Insider leak fixed: routine rule keyed per company; parity 0.0 on 4 windows x 8 days with insider data on
- [ ] T6 Restart the SEC filings refresh and the 13D attribution fix
- [ ] T7 Refresh the Pattern Explorer and Sensitivity pages

## Finding volatility
- [ ] V1 10 weekly ±10% movers at 95% in every era
- [ ] V2 Direction: bet only at >=80% confidence; per-stock-type indicator trust
- [ ] V3 Exact exit
- [ ] V4 Exact stop: losers never worse than -20% (aim -10% to -15%)
- [ ] V5 8 of 10 finish +10%, consistently in every era, without cheating

## Added along the way
- [x] C33 No cheating / no after-hours or weekend trading: simulations fill at the next open; live verified
- [x] C34 Advanced memory: recency, market similarity, reliability, shock detection, long-term memory (earlier windows only); deterministic; scramble gate passes

## Algorithm (highest priority, C35/C36)
- [x] A1 Multi-timeframe candle features (daily/weekly/monthly; reversals, inside/outside, engulfing, streaks, gap fills)
- [x] A2 Pattern miner: singles, pairs, "unless" exceptions; relevance-weighted memory; FDR coincidence control; walk-forward confirmation; shrunk effect sizes; death detection and cause search
- [ ] A3 Heavy test: patterns learned before a cut must predict unseen later data (several cuts / eras)
- [ ] A4 Long-term pattern memory bank (earlier windows only) re-tested in each new window
- [ ] A5 Feed the pattern score into Test and Find volatility (minimal Test change: one input)
- [ ] A6 Minute candles: forward collector (free history does not exist for past decades)
- [ ] A7 Algorithm tunes itself: its own settings (memory half-life, context width, FDR level, shrinkage) searched on held-out eras
