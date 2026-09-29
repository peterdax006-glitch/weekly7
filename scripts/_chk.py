import pathlib
p=pathlib.Path("state/CHECKLIST.md"); s=p.read_text(encoding="utf-8")
s = s.replace("# Weekly7 checkoff list (canon C32)", "# Weekly7 checkoff list (canon C32; priority C36: Algorithm > Find volatility > minimal Test changes > Live)")
s += """
## Algorithm (highest priority, C35/C36)
- [x] A1 Multi-timeframe candle features (daily/weekly/monthly; reversals, inside/outside, engulfing, streaks, gap fills)
- [x] A2 Pattern miner: singles, pairs, "unless" exceptions; relevance-weighted memory; FDR coincidence control; walk-forward confirmation; shrunk effect sizes; death detection and cause search
- [ ] A3 Heavy test: patterns learned before a cut must predict unseen later data (several cuts / eras)
- [ ] A4 Long-term pattern memory bank (earlier windows only) re-tested in each new window
- [ ] A5 Feed the pattern score into Test and Find volatility (minimal Test change: one input)
- [ ] A6 Minute candles: forward collector (free history does not exist for past decades)
- [ ] A7 Algorithm tunes itself: its own settings (memory half-life, context width, FDR level, shrinkage) searched on held-out eras
"""
p.write_text(s,encoding="utf-8",newline="\n")
