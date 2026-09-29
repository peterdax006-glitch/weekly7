"""Canon C33 patch: simulated orders decided after a close fill at the NEXT session's open."""
import pathlib

p = pathlib.Path("engine/adaptive.py"); s = p.read_text(encoding="utf-8")
old = '''    def on_day(self, day, px, closes_to_now, next_is_new_week, snap=None):
        if closes_to_now is not None and closes_to_now.index.max() > day:
            raise TimeFence(f"session handed prices after {day}")
        val = self.equity(px)
        wr = val / self.week_start - 1
        if snap is not None:
            held = list(self.pos)
            if self.adapter is not None:
                self.cfg = self.adapter.step(day, snap, closes_to_now, self.divs, held)
                det = self.adapter.detector_scores(snap)
            else:
                det = None
            target = pick(snap, self.cfg, held, self.divs, det)
            self._trade(target, px, val, day, "rebalance" if held else "initial build")
            self.decisions.append((str(day.date()), sorted(target.index)))
        elif not self.capped and self.cfg.get("brake") and wr <= -self.cfg["brake"]:
            target = pd.Series({t: q * px[t] / val for t, q in self.pos.items()}) * self.P.TOPK["brake_exposure"]
            self._trade(target, px, val, day, f"weekly brake ({wr:.1%})")
            self.capped = True'''
new = '''    def on_day(self, day, px, closes_to_now, next_is_new_week, snap=None, px_open=None):
        """Canon C33: decisions use information up to today's close, so they are executed at the NEXT session's
        open (regular hours only; never after-hours or weekends). px_open = today's opening prices."""
        if closes_to_now is not None and closes_to_now.index.max() > day:
            raise TimeFence(f"session handed prices after {day}")
        # 1) morning: fill yesterday's decision at today's open
        if self.pending is not None:
            target, reason, as_weights = self.pending
            fill = px_open if px_open is not None else px
            val_open = self.equity(fill)
            if not as_weights:                          # brake: scale the current holdings by a factor
                target = pd.Series({t: q * fill.get(t, np.nan) / val_open for t, q in self.pos.items()}) * target
            self._trade(target, fill, val_open, day, reason)
            self.pending = None
        # 2) after the close: look at the day and decide for tomorrow's open
        val = self.equity(px)
        wr = val / self.week_start - 1
        if snap is not None:
            held = list(self.pos)
            if self.adapter is not None:
                self.cfg = self.adapter.step(day, snap, closes_to_now, self.divs, held)
                det = self.adapter.detector_scores(snap)
            else:
                det = None
            target = pick(snap, self.cfg, held, self.divs, det)
            self.pending = (target, "rebalance" if held else "initial build", True)
            self.decisions.append((str(day.date()), sorted(target.index)))
        elif not self.capped and self.cfg.get("brake") and wr <= -self.cfg["brake"]:
            self.pending = (self.P.TOPK["brake_exposure"], f"weekly brake ({wr:.1%})", False)
            self.capped = True'''
assert old in s; s = s.replace(old, new)
s = s.replace('''        self.days, self.weeks, self.decisions, self.orders, self.week_rows = [], [], [], [], []''',
              '''        self.days, self.weeks, self.decisions, self.orders, self.week_rows = [], [], [], [], []
        self.pending = None''')
old = '''    def needs_snapshot(self, next_is_new_week):
        return (next_is_new_week and self.wk % self.cfg.get("rebalance_weeks", 1) == 0) or not self.pos'''
assert old in s
s = s.replace(old, '''    def needs_snapshot(self, next_is_new_week):
        return (next_is_new_week and self.wk % self.cfg.get("rebalance_weeks", 1) == 0) or (not self.pos and self.pending is None)''')
old = '''def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0):'''
assert old in s
s = s.replace(old, '''def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0, opens=None):''')
old = '''        closes.loc[m] = closes.loc[m].values * np.exp(rng.normal(0, 0.2, closes.loc[m].shape))'''
assert old in s
s = s.replace(old, '''        noise = np.exp(rng.normal(0, 0.2, closes.loc[m].shape))
        closes.loc[m] = closes.loc[m].values * noise
        if opens is not None:
            opens = opens.astype("float64").copy()
            opens.loc[m] = opens.loc[m].values * noise''')
old = '''        S.on_day(d, closes.loc[d], closes.loc[:d], nxt_new, snap)'''
assert old in s
s = s.replace(old, '''        S.on_day(d, closes.loc[d], closes.loc[:d], nxt_new, snap,
                 opens.loc[d] if opens is not None and d in opens.index else None)''')
p.write_text(s, encoding="utf-8", newline="\n")

p = pathlib.Path("engine/livesim.py"); s = p.read_text(encoding="utf-8")
old = '''        S.on_day(now, self.feed.prices(), closes_to_now, week_end, snap if S.needs_snapshot(week_end) else None)'''
assert old in s
s = s.replace(old, '''        S.on_day(now, self.feed.prices(), closes_to_now, week_end, snap if S.needs_snapshot(week_end) else None,
                 self.feed._stocks["Open"].iloc[self.feed.i])''')
p.write_text(s, encoding="utf-8", newline="\n")

p = pathlib.Path("scripts/livesim_loop2.py"); s = p.read_text(encoding="utf-8")
old = '''    closes = pd.read_parquet(a / ("closes_v2.parquet" if (a / "closes_v2.parquet").exists() else "closes.parquet"))'''
assert old in s
s = s.replace(old, old + '''
    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None''')
s = s.replace('''    return {"id": a.name, "snaps": ws, "closes": closes,''', '''    return {"id": a.name, "snaps": ws, "closes": closes, "opens": opens,''')
s = s.replace('''    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta)''',
              '''    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta, opens=w["opens"])''')
s = s.replace('''        S1 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"])''',
              '''        S1 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"], opens=w["opens"])''')
s = s.replace('''                      scramble_after=cut, seed=len(r))''', '''                      scramble_after=cut, seed=len(r), opens=w["opens"])''')
s = s.replace('''    feed._stocks["Close"].loc[feed.first_live:].to_parquet(a / "closes_v2.parquet")
    for k, v in trader.snaps.items():''', '''    feed._stocks["Close"].loc[feed.first_live:].to_parquet(a / "closes_v2.parquet")
    feed._stocks["Open"].loc[feed.first_live:].to_parquet(a / "opens_v2.parquet")
    for k, v in trader.snaps.items():''')
assert s.count("opens") >= 6, s.count("opens")
p.write_text(s, encoding="utf-8", newline="\n")

p = pathlib.Path("scripts/regen_weekly_snaps.py"); s = p.read_text(encoding="utf-8")
old = '''    feed._stocks["Close"].loc[feed.first_live:].to_parquet(a / "closes_v2.parquet")   # prices on today's data'''
assert old in s
s = s.replace(old, old + '''
    feed._stocks["Open"].loc[feed.first_live:].to_parquet(a / "opens_v2.parquet")     # C33: fills at the next open''')
p.write_text(s, encoding="utf-8", newline="\n")
print("patched")
