"""Blind live-clock simulation of a random hidden year (canon C11).

SealedYear  draws the year (1965-2025) and seals it; nothing reads it until reveal().
Feed        owns the real data. Shows the trader a disguised world: dates shifted by a random
            whole number of weeks into the 2100s (weekdays and holidays intact, year unknowable),
            tickers replaced by code names, filings released only once public. It is also the clock:
            one trading session per tick, and the next tick is released the instant the trader
            acknowledges the last one (lockstep - as fast as the trader can keep up, never ahead).
SimBroker   fills at the current session's close with era costs; never sees the future.
BlindTrader the live system (features -> model -> policy) driven only through the feed."""
import json, secrets, threading, queue, time
import numpy as np
import pandas as pd

from . import config as K, data, features, model, policy

DIR = K.STATE / "livesim"
DIR.mkdir(parents=True, exist_ok=True)
FIRST_YEAR, LAST_YEAR = 1965, 2025


class SealedYear:
    def __init__(self, run_id):
        self.path = DIR / f"sealed_{run_id}.json"
        if not self.path.exists():
            used = {json.loads(f.read_text())["year"] for f in DIR.glob("sealed_*.json")}
            pool = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in used] or list(range(FIRST_YEAR, LAST_YEAR + 1))
            y = pool[secrets.randbelow(len(pool))]              # no repeats until every year has been played
            weeks = 8000 + secrets.randbelow(3000)            # 2100s-2150s, a multiple of 7 days keeps weekdays
            self.path.write_text(json.dumps({"year": y, "shift_days": 7 * weeks}))

    def _read(self):
        return json.loads(self.path.read_text())

    def reveal(self):
        return self._read()["year"]


class Feed:
    """The only door between the real past and the trader."""

    def __init__(self, sealed: SealedYear, warmup_years=6):
        s = sealed._read()                                     # the feed may know; the trader never does
        Y, self._shift = s["year"], pd.Timedelta(days=s["shift_days"])
        from .replay import _all_prices
        stocks, market = _all_prices()
        w = min(warmup_years, Y - 1962)
        lo, hi = pd.Timestamp(f"{Y - w}-01-01"), pd.Timestamp(f"{Y}-12-31")
        C = stocks["Close"].loc[lo:hi]
        live_cols = C.columns[C.loc[f"{Y}"].notna().any()]      # names that trade in the hidden year
        rng = np.random.default_rng(secrets.randbits(64))
        codes = [f"S{n:04d}" for n in rng.permutation(len(live_cols))]
        self._map = dict(zip(live_cols, codes))
        self._stocks = {f: self._disguise(v.loc[lo:hi, live_cols]) for f, v in stocks.items()}
        mk = {f: v.loc[lo:hi, [c for c in ("SPY", "^VIX", "^VIX3M") if c in v]] for f, v in market.items()}
        self._market = {f: self._shift_index(v) for f, v in mk.items()}
        ev = pd.read_parquet(K.CACHE / "events.parquet")
        ev = ev[ev["ticker"].isin(self._map) & (ev["accepted"] >= lo.tz_localize("UTC")) & (ev["accepted"] <= hi.tz_localize("UTC"))].copy()
        ev["ticker"] = ev["ticker"].map(self._map)
        ev["accepted"] = ev["accepted"] + self._shift
        ev = ev[~ev["kind"].isin(["ACTIVIST", "ACTIVIST_AMEND"])]   # 13D attribution under repair
        self._events = ev.sort_values("accepted")
        ins = pd.read_parquet(K.CACHE / "insider.parquet")
        ins = ins[ins["symbol"].isin(self._map) & (ins["filed"] >= lo) & (ins["filed"] <= hi)].copy()
        ins["symbol"] = ins["symbol"].map(self._map)
        for c in ("filed", "tdate"):
            ins[c] = ins[c] + self._shift
        self._insider = ins
        sic = pd.read_parquet(K.CACHE / "sic.parquet")
        sic = sic[sic["ticker"].isin(self._map)].copy()
        sic["ticker"] = sic["ticker"].map(self._map)
        self.sic = sic[["ticker", "sic"]]                        # industry codes are timeless
        days = self._stocks["Close"].index
        self.sessions = days
        self.first_live = days[days.searchsorted(pd.Timestamp(f"{Y}-01-01") + self._shift)]
        self.i = days.get_loc(self.first_live) - 1               # clock starts at the end of the warm-up
        self.cost_bps = 40 if Y < 1997 else 20 if Y < 2001 else 10
        self._q_tick, self._q_ack = queue.Queue(1), queue.Queue(1)
        self.ticks, self.t_start = 0, None

    def _shift_index(self, df):
        df = df.copy()
        df.index = df.index + self._shift
        return df

    def _disguise(self, df):
        df = self._shift_index(df)
        df.columns = [self._map[c] for c in df.columns]
        return df

    # ---- feature service: computed once, served one session at a time (proven equal to live by parity_test) ----
    def precompute_features(self):
        ev, ins = self._events, self._insider
        X, atr = features.build(self._stocks, self._market, ev, ins, self.sic,
                                start=str(self.sessions[0].date()), relative=True)
        self._X, self._atr = X, atr

    def features_today(self):
        """Only the current session's row: exactly what a live data vendor would serve today."""
        d = self.now
        return self._X.xs(d, level=0, drop_level=False) if d in self._X.index.get_level_values(0) else self._X.iloc[:0]

    def features_until_now(self):
        """Warm-up training data: rows up to the current session only."""
        return self._X[self._X.index.get_level_values(0) <= self.now], self._atr.loc[:self.now]

    # ---- what the trader may see ----
    @property
    def now(self):
        return self.sessions[self.i]

    def history(self, lookback=None):
        lo = 0 if lookback is None else max(0, self.i - lookback)
        cut = lambda d: {f: v.iloc[lo:self.i + 1] for f, v in d.items()}
        return cut(self._stocks), cut(self._market)

    def filings(self):
        t = self.now.tz_localize("UTC") + pd.Timedelta(hours=20, minutes=30)    # public by this session's close
        return self._events[self._events["accepted"] <= t], self._insider[self._insider["filed"] <= self.now]

    def price(self, code):
        return float(self._stocks["Close"].iloc[self.i].get(code, np.nan))

    def prices(self):
        return self._stocks["Close"].iloc[self.i]

    def next_session_is_new_week(self):
        if self.i + 1 >= len(self.sessions):
            return True
        return self.sessions[self.i + 1].isocalendar().week != self.now.isocalendar().week

    def done(self):
        return self.i + 1 >= len(self.sessions)

    # ---- the clock: lockstep, never ahead of the trader ----
    def run_clock(self):
        self.t_start = time.perf_counter()
        while not self.done():
            self.i += 1
            self.ticks += 1
            self._q_tick.put(self.now)
            self._q_ack.get()                                  # wait for the trader, then tick again at once
        self._q_tick.put(None)

    def wait_tick(self):
        return self._q_tick.get()

    def ack(self):
        self._q_ack.put(True)


class SimBroker:
    def __init__(self, feed, cash=K.START_CASH):
        self.feed, self.cash, self.pos, self.log = feed, cash, {}, []

    def equity(self):
        px = self.feed.prices()
        return self.cash + sum(q * px[t] for t, q in self.pos.items() if np.isfinite(px.get(t, np.nan)))

    def order_to(self, code, dollars_target, reason):
        p = self.feed.price(code)
        if not np.isfinite(p):
            return
        dv = dollars_target - self.pos.get(code, 0.0) * p
        if abs(dv) < 1.0:
            return
        self.cash -= dv + abs(dv) * self.feed.cost_bps / 1e4
        self.pos[code] = self.pos.get(code, 0.0) + dv / p
        self.log.append({"session": str(self.feed.now.date()), "code": code, "dollars": round(dv, 2),
                         "price": round(p, 4), "reason": reason})
        if abs(self.pos[code]) * p < 0.5:
            self.pos.pop(code)


class BlindTrader:
    def __init__(self, feed, cfg, fast=True):
        self.feed, self.cfg, self.fast = feed, cfg, fast
        self.broker = SimBroker(feed)
        self.m = None
        self.week_start, self.capped, self.week_count = K.START_CASH, False, 0
        self.days, self.weeks, self.picks, self.snaps = [], [], [], {}
        self.t_model = self.t_features = 0.0

    def train(self):
        t = time.perf_counter()
        stocks, market = self.feed.history()
        if self.fast:
            X, atr = self.feed.features_until_now()
        else:
            ev, ins = self.feed.filings()
            X, atr = features.build(stocks, market, ev, ins, self.feed.sic, start=str(stocks["Close"].index[0].date()), relative=True)
        yb, fw = features.labels(stocks, atr)
        d = X.index.get_level_values(0)
        ud = pd.DatetimeIndex(sorted(d.unique()))
        tr = ud[: max(1, len(ud) - model.EMBARGO)][::2]        # labels of the last warm-up days would need the future
        Xt = X[d.isin(tr)]
        y = yb.stack(future_stack=True).reindex(Xt.index)
        f = fw.stack(future_stack=True).reindex(Xt.index)
        ok = y.notna().values & f.notna().values
        self.m = model.fit_models(model.normalise(Xt)[ok], y[ok], f[ok], fast=self.fast)
        self.train_rows = int(ok.sum())
        self.t_model = time.perf_counter() - t

    def decide(self):
        t = time.perf_counter()
        if self.fast:
            X = self.feed.features_today()
        else:
            stocks, market = self.feed.history()
            ev, ins = self.feed.filings()
            X, _ = features.build(stocks, market, ev, ins, self.feed.sic, start=str(self.feed.now.date()), relative=True)
        self.t_features += time.perf_counter() - t
        if X.empty:
            return pd.Series(dtype=float), None
        xr = X.xs(X.index.get_level_values(0).max(), level=0)
        R = model.normalise(X).xs(X.index.get_level_values(0).max(), level=0).reindex(columns=self.m["cols"])
        ew = {k: v for k, v in model.EVIDENCE.items() if k != "ev_activist"}
        p = pd.DataFrame({"mu_raw": self.m["reg"].predict(R), "evidence": model.evidence_score(R, ew)}, index=R.index)
        s = policy.score(p, self.cfg["w_model"])
        ok = ~((xr["ev_red_flag"] > 0) | ((xr["ev_offering"] > 0) & (xr["log_dv"].rank(pct=True) < 0.5)))
        if self.cfg["vol_filter"]:
            ok &= ~((xr["vol20"].rank(pct=True) > 0.9) | (xr["max20"].rank(pct=True) > 0.9))
        ok &= xr["log_dv"].rank(pct=True) >= self.cfg["liq_q"]
        divs = {t: policy.sic_division(c) for t, c in zip(self.feed.sic["ticker"], self.feed.sic["sic"])}
        target = policy.topk_targets(s[ok.reindex(s.index).fillna(False)], list(self.broker.pos), self.cfg["k"],
                                     self.cfg["exit_q"], divs if self.cfg["max_per_sector"] else None,
                                     self.cfg["max_per_sector"], pick=self.cfg["pick"], vol=xr["vol20"],
                                     pool_q=self.cfg["pool_q"])
        snap = pd.DataFrame({"mu_raw": p["mu_raw"], "evidence": p["evidence"]}).join(
            xr[["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering", "r5"]])
        snap["score"] = s
        self.snaps[str(self.feed.now.date())] = snap
        return target, snap

    def on_tick(self):
        b = self.broker
        val = b.equity()
        wr = val / self.week_start - 1
        week_end = self.feed.next_session_is_new_week()
        rebalance_week = week_end and self.week_count % self.cfg.get("rebalance_weeks", 1) == 0
        if rebalance_week or not b.pos:
            target, snap = self.decide()
            val = b.equity()
            for code in sorted(set(b.pos) | set(target.index), key=lambda c: target.get(c, 0.0)):
                b.order_to(code, target.get(code, 0.0) * 0.985 * val, "rebalance" if b.pos else "initial build")
            if snap is not None:
                self.picks.append({"session": str(self.feed.now.date()), "names": list(target.index),
                                   "scores": snap.loc[list(target.index), "score"].round(4).tolist()})
        elif not self.capped and self.cfg["brake"] and wr <= -self.cfg["brake"]:
            for code, q in list(b.pos.items()):
                b.order_to(code, q * self.feed.price(code) * policy.TOPK["brake_exposure"], f"weekly brake ({wr:.1%})")
            self.capped = True
        val = b.equity()
        self.days.append({"session": str(self.feed.now.date()), "equity": val, "holdings": sorted(b.pos)})
        if week_end:
            self.weeks.append({"week_end": str(self.feed.now.date()), "ret": val / self.week_start - 1,
                               "brake": self.capped, "holdings": sorted(b.pos)})
            self.week_start, self.capped = val, False
            self.week_count += 1


def parity_test(feed, n_days=2, seed=None):
    """Leakage guard for the fast path: recompute features the slow, strictly-live way (history up to that
    day only) on random days and require them to equal the precomputed rows. Any mismatch = abort."""
    rng = np.random.default_rng(seed)
    days = feed.sessions[feed.sessions >= feed.first_live]
    keep_i = feed.i
    worst = 0.0
    for d in rng.choice(days, size=min(n_days, len(days)), replace=False):
        feed.i = feed.sessions.get_loc(d)
        stocks, market = feed.history()
        ev, ins = feed.filings()
        slow, _ = features.build(stocks, market, ev, ins, feed.sic, start=str(feed.now.date()), relative=True)
        a = slow.xs(feed.now, level=0).sort_index()
        b = feed.features_today().xs(feed.now, level=0).reindex(index=a.index, columns=a.columns)
        diff = (a - b).abs().where(~(a.isna() & b.isna()), 0.0)
        worst = max(worst, float(np.nanmax(diff.values)) if diff.size else 0.0)
        if (a.isna() != b.isna()).values.any():
            worst = max(worst, 1.0)
    feed.i = keep_i
    if worst > 1e-4:
        raise RuntimeError(f"PARITY FAIL: fast features differ from live-computed ones (max diff {worst:.3g}) - possible leakage")
    return worst


def run(cfg, run_id, log=print, check_parity=True):
    sealed = SealedYear(run_id)
    feed = Feed(sealed)
    t = time.perf_counter()
    feed.precompute_features()
    log(f"  feature service ready in {time.perf_counter() - t:.0f}s")
    if check_parity:
        t = time.perf_counter()
        w = parity_test(feed)
        log(f"  parity test passed (fast path == live path on random days, max diff {w:.1e}) in {time.perf_counter() - t:.0f}s")
    trader = BlindTrader(feed, cfg)
    log(f"  warm-up: {len(feed.sessions)} sessions visible, {feed.i + 1} of them before the hidden year; training ...")
    trader.train()
    log(f"  model ready ({trader.train_rows:,} rows, {trader.t_model:.0f}s). Clock starts at {feed.now.date()} (disguised).")
    clock = threading.Thread(target=feed.run_clock, daemon=True)
    clock.start()
    while True:
        t = feed.wait_tick()
        if t is None:
            break
        trader.on_tick()
        feed.ack()
    wall = time.perf_counter() - feed.t_start
    return feed, trader, sealed, wall
