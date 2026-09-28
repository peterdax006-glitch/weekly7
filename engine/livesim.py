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
    """Seals a random 12-month window (canon C19): a start month from Jan 1965 to Sep 2025, then 12 consecutive
    months. Older seals from the calendar-year design ({"year": Y}) are read as a January start."""

    def __init__(self, run_id):
        self.path = DIR / f"sealed_{run_id}.json"
        if not self.path.exists():
            used = [self.start_of(json.loads(f.read_text())) for f in DIR.glob("sealed_*.json")]
            months = pd.date_range(f"{FIRST_YEAR}-01-01", "2025-09-01", freq="MS")
            # prefer windows that overlap no played window by more than half (keeps coverage spread out)
            fresh = [m for m in months if all(abs((m - u).days) > 183 for u in used)] or list(months)
            start = fresh[secrets.randbelow(len(fresh))]
            weeks = 8000 + secrets.randbelow(3000)            # 2100s-2150s, a multiple of 7 days keeps weekdays
            self.path.write_text(json.dumps({"start": str(start.date()), "shift_days": 7 * weeks}))

    @staticmethod
    def start_of(d):
        return pd.Timestamp(d["start"]) if "start" in d else pd.Timestamp(f"{d['year']}-01-01")

    def _read(self):
        return json.loads(self.path.read_text())

    def reveal(self):
        st = self.start_of(self._read())
        end = st + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        return f"{st:%b %Y} - {end:%b %Y}"


class Feed:
    """The only door between the real past and the trader."""

    def __init__(self, sealed: SealedYear, warmup_years=6):
        s = sealed._read()                                     # the feed may know; the trader never does
        start, self._shift = SealedYear.start_of(s), pd.Timedelta(days=s["shift_days"])
        Y = start.year
        from .replay import _all_prices
        stocks, market = _all_prices()
        w = min(warmup_years, (start - pd.Timestamp("1962-01-01")).days // 365)
        lo, hi = start - pd.DateOffset(years=w), start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        C = stocks["Close"].loc[lo:hi]
        live_cols = C.columns[C.loc[start:hi].notna().any()]    # names that trade in the hidden window
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
        # C18 fail-safe (28 Sep 2026): the fast path sometimes counted an insider filing the strictly-live path could
        # not yet see (parity test, window w01c). Until that is fixed and proven, blind simulations run WITHOUT
        # insider data; features that depend on it are zero in both paths, so parity holds by construction.
        self._insider = ins.iloc[0:0]
        sic = pd.read_parquet(K.CACHE / "sic.parquet")
        sic = sic[sic["ticker"].isin(self._map)].copy()
        sic["ticker"] = sic["ticker"].map(self._map)
        self.sic = sic[["ticker", "sic"]]                        # industry codes are timeless
        days = self._stocks["Close"].index
        self.sessions = days
        self.first_live = days[days.searchsorted(start + self._shift)]
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
    """The live system inside the blind feed. All trading logic lives in adaptive.Session (shared with the
    re-tester); this class only turns what the feed shows today into a snapshot and hands it over."""

    def __init__(self, feed, cfg, fast=True, adaptive=False, meta=None):
        from . import adaptive as A
        self.A = A
        self.feed, self.cfg, self.fast, self.adaptive, self.meta = feed, dict(cfg), fast, adaptive, meta
        self.m = None
        self.snaps, self.warm_snaps = {}, {}
        self.t_model = self.t_features = 0.0
        self.divs = {t: policy.sic_division(c) for t, c in zip(feed.sic["ticker"], feed.sic["sic"])}
        self.session = None
        self.preseason = None

    def _fit(self, X, stocks, atr, until_idx):
        yb, fw = features.labels(stocks, atr)
        d = X.index.get_level_values(0)
        ud = pd.DatetimeIndex(sorted(d.unique()))
        ud = ud[ud <= until_idx]
        tr = ud[: max(1, len(ud) - model.EMBARGO)][::2]        # labels of the last days would need the future
        Xt = X[d.isin(tr)]
        y = yb.stack(future_stack=True).reindex(Xt.index)
        f = fw.stack(future_stack=True).reindex(Xt.index)
        ok = y.notna().values & f.notna().values
        return model.fit_models(model.normalise(Xt)[ok], y[ok], f[ok], fast=True), int(ok.sum())

    def snapshot_from(self, m, X, day):
        Xd = X.xs(day, level=0, drop_level=False)
        if Xd.empty:
            return None
        xr = Xd.xs(day, level=0)
        R = model.normalise(Xd).xs(day, level=0).reindex(columns=m["cols"])
        p = pd.DataFrame({"mu_raw": m["reg"].predict(R)}, index=R.index)
        for c in policy.EVIDENCE_FEATS:
            if c in R:
                p[f"e_{c}"] = R[c].values
        p["evidence"] = policy.evidence_from(p, policy.default_evidence_weights())
        return p.join(xr[["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering", "r5"] +
                         [c for c in xr.columns if c.startswith("m_")]])

    def train(self):
        t = time.perf_counter()
        stocks, market = self.feed.history()
        X, atr = self.feed.features_until_now()
        now = self.feed.now
        self.m, self.train_rows = self._fit(X, stocks, atr, now)
        # pre-season study (C17): a second model that stops 18 months earlier, so the last 18 warm-up months
        # are out-of-sample for it; candidate settings are judged there, quarter by quarter
        if self.adaptive:
            cut = now - pd.DateOffset(months=18)
            m2, _ = self._fit(X, stocks, atr, cut)
            closes = stocks["Close"]
            wdays = closes.loc[cut:now].index
            for i, d in enumerate(wdays[:-1]):
                if wdays[i + 1].isocalendar().week != d.isocalendar().week:
                    sn = self.snapshot_from(m2, X, d)
                    if sn is not None:
                        self.warm_snaps[str(d.date())] = sn
            wclose = closes.loc[cut:now]
            cands = [{**self.cfg, k: v} for k, v in self.A.neighbours(self.cfg, self.A.META_DEFAULT["adaptive_knobs"])]
            run = lambda cfg, sn, cl, bps, dv: self.A.replay(cfg, sn, cl, bps, dv).result()
            chosen, info = self.A.choose_default(self.warm_snaps, wclose, self.divs, self.cfg, self.feed.cost_bps, run, cands)
            self.preseason = {"prior": self.cfg, "chosen": chosen, **info}
            self.cfg = chosen
        self.session = self.A.Session(self.cfg, self.divs, self.feed.cost_bps, adaptive=self.adaptive, meta=self.meta)
        self.t_model = time.perf_counter() - t

    def on_tick(self):
        S, now = self.session, self.feed.now
        week_end = self.feed.next_session_is_new_week()
        snap = None
        if week_end or not S.pos:                          # archive a snapshot every week (and on the first day)
            t = time.perf_counter()
            snap = self.snapshot_from(self.m, self.feed.features_today(), now)
            self.t_features += time.perf_counter() - t
            if snap is not None:
                self.snaps[str(now.date())] = snap
        closes_to_now = self.feed.history()[0]["Close"]
        S.on_day(now, self.feed.prices(), closes_to_now, week_end, snap if S.needs_snapshot(week_end) else None)

    # views for the diagnosis code
    @property
    def days(self):
        return [{"session": d, "equity": v} for d, v in self.session.days]

    @property
    def weeks(self):
        return self.session.week_rows

    @property
    def picks(self):
        return [{"session": d, "names": n} for d, n in self.session.decisions]


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


def run(cfg, run_id, log=print, check_parity=True, adaptive=False, meta=None):
    sealed = SealedYear(run_id)
    feed = Feed(sealed)
    t = time.perf_counter()
    feed.precompute_features()
    log(f"  feature service ready in {time.perf_counter() - t:.0f}s")
    if check_parity:
        t = time.perf_counter()
        w = parity_test(feed)
        log(f"  parity test passed (fast path == live path on random days, max diff {w:.1e}) in {time.perf_counter() - t:.0f}s")
    trader = BlindTrader(feed, cfg, adaptive=adaptive, meta=meta)
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
