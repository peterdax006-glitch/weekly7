"""Blind live-clock simulation of a random hidden year (canon C11).

SealedYear  draws the year (1965-2025) and seals it; nothing reads it until reveal().
Feed        owns the real data. Shows the trader a disguised world: dates shifted by a random
            whole number of weeks into the 2100s (weekdays and holidays intact, year unknowable),
            tickers replaced by code names, filings released only once public. It is also the clock:
            one trading session per tick, and the next tick is released the instant the trader
            acknowledges the last one (lockstep - as fast as the trader can keep up, never ahead).
SimBroker   fills at the current session's close with era costs; never sees the future.
BlindTrader the live system (features -> model -> policy) driven only through the feed.

Phases 21-22 (engine/blind_gates.py) are enforced here, not just available: the seal is drawn and digested by
blind_gates.seal_window and audited when a Feed is built; the Feed refuses to start when the disguise (constant
week-multiple shift, bijective code names, no real ticker or date visible, unchanged holiday calendar) fails its gates;
every served filing and memory row is checked as public before use; and the lockstep clock is a BlindClock that
requires one complete pass per session and writes a hash-chained InformationLedger entry (what information existed,
and when it came from) for every tick. Gate failures raise BlindGateError; nothing is silently repaired."""
import json, os, secrets, threading, queue, time
import numpy as np
import pandas as pd

from . import config as K, data, features, model, policy, blind_gates as BG

DIR = K.STATE / "livesim"
DIR.mkdir(parents=True, exist_ok=True)
FIRST_YEAR, LAST_YEAR = 1965, 2025
CLOSE_UTC = pd.Timedelta(hours=20, minutes=30)      # a filing accepted by this time is public at the session close


class BlindGateError(RuntimeError):
    """A Phase 21/22 gate failed: the window was not honestly blind, so no result from it may be used."""


class SealedYear:
    """Seals a random 12-month window (canon C19): a start month from Jan 1965 to Sep 2025, then 12 consecutive
    months. Older seals from the calendar-year design ({"year": Y}) are read as a January start."""

    def __init__(self, run_id):
        self.path = DIR / f"sealed_{run_id}.json"
        if not self.path.exists():
            used = [self.start_of(json.loads(f.read_text())) for f in DIR.glob("sealed_*.json")]
            # random start month, 12 consecutive months, preferring windows that overlap no played window by more than
            # half a year; shift of 8000-11000 whole weeks (2100s-2150s, weekdays kept). The seed is secret entropy.
            rec = BG.seal_window(used, secrets.randbits(63), str(pd.Timestamp.now()), tag=run_id)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, self.path)                        # the seal appears whole or not at all

    @staticmethod
    def start_of(d):
        return pd.Timestamp(d["start"]) if "start" in d else pd.Timestamp(f"{d['year']}-01-01")

    def _read(self):
        return json.loads(self.path.read_text())

    def audit(self, first_worker_start=None):
        """Phase 21 findings for this seal: 12 consecutive months, digest unaltered, sealed before the first worker,
        overlap with other windows. Seals written before the digest existed are legacy: their sealing order cannot be
        proven, which is reported as a warning rather than blocking every older window. Overlap is a warning too
        (seal_window falls back to the least-overlapping months when the calendar is crowded)."""
        rec = self._read()
        if "digest" not in rec:
            return [BG.Finding("seal", "warn", "legacy seal (no digest or sealed_at): sealing order is not provable")]
        others = [self.start_of(json.loads(f.read_text())) for f in DIR.glob("sealed_*.json") if f != self.path]
        return [BG.Finding(f.gate, "warn" if "overlaps" in f.message else f.severity, f.message)
                for f in BG.check_seal(rec, others, first_worker_start)]

    def reveal(self, gate=None):
        """The true period. With a `blind_gates.RevealGate` the answer is refused until adjustments are locked, all
        predictions recorded, all trades completed and all learning finalized (Phase 21)."""
        if gate is not None:
            gate.reveal()
        st = self.start_of(self._read())
        end = st + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        return f"{st:%b %Y} - {end:%b %Y}"


class Feed:
    """The only door between the real past and the trader."""

    def __init__(self, sealed: SealedYear, warmup_years=6, use_insider=True, data=None, enforce=True, drop_crypto=True,
                 tradable_rule="split_invariant"):
        """`data` = (stocks, market, events, insider, sic) replaces the real caches (synthetic windows in tests).
        `enforce=False` skips the Phase 21/22 gates and the ledger; it exists only so a test can prove that a clean
        run is bit-identical with and without them. `drop_crypto` removes canon-C11 crypto tickers from the universe BEFORE
        disguising (real tickers at load). `tradable_rule` is passed to features.build: the Test path uses the split-invariant
        rule (canon C56: adjusted price levels encode later splits)."""
        s = sealed._read()                                     # the feed may know; the trader never does
        start, self._shift = SealedYear.start_of(s), pd.Timedelta(days=s["shift_days"])
        Y = start.year
        self.enforce, self.gate_findings, self._sealed = enforce, [], sealed
        self.tradable_rule = tradable_rule
        if enforce:
            self.gate_findings += sealed.audit(first_worker_start=pd.Timestamp.now())
            self._raise_on_fail(self.gate_findings)
        if data is None:
            from .replay import _all_prices
            stocks, market = _all_prices()
            ev_all = pd.read_parquet(K.CACHE / "events.parquet")
            ins_all = pd.read_parquet(K.CACHE / "insider.parquet")
            sic_all = pd.read_parquet(K.CACHE / "sic.parquet")
        else:
            stocks, market, ev_all, ins_all, sic_all = data
        w = min(warmup_years, (start - pd.Timestamp("1962-01-01")).days // 365)
        lo, hi = start - pd.DateOffset(years=w), start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        C = stocks["Close"].loc[lo:hi]
        live_cols = C.columns[C.loc[start:hi].notna().any()]    # names that trade in the hidden window
        crypto = policy.crypto_set() if drop_crypto else set()
        self.dropped_crypto = sorted(c for c in live_cols if c in crypto)
        live_cols = live_cols[~live_cols.isin(crypto)]           # C11: crypto never reaches the disguise (real tickers here)
        rng = np.random.default_rng(s["shift_days"] * 7919 + 17)   # code names fixed per sealed window (reproducible)
        codes = [f"S{n:04d}" for n in rng.permutation(len(live_cols))]
        self._map = dict(zip(live_cols, codes))
        self._stocks = {f: self._disguise(v.loc[lo:hi, live_cols]) for f, v in stocks.items()}
        mk = {f: v.loc[lo:hi, [c for c in ("SPY", "^VIX", "^VIX3M") if c in v]] for f, v in market.items()}
        self._market = {f: self._shift_index(v) for f, v in mk.items()}
        ev = ev_all
        ev = ev[ev["ticker"].isin(self._map) & (ev["accepted"] >= lo.tz_localize("UTC")) & (ev["accepted"] <= hi.tz_localize("UTC"))].copy()
        ev["ticker"] = ev["ticker"].map(self._map)
        ev["accepted"] = ev["accepted"] + self._shift
        ev = ev[~ev["kind"].isin(["ACTIVIST", "ACTIVIST_AMEND"])]   # 13D attribution under repair
        self._events = ev.sort_values("accepted")
        ins = ins_all
        ins = ins[ins["symbol"].isin(self._map) & (ins["filed"] >= lo) & (ins["filed"] <= hi)].copy()
        ins["symbol"] = ins["symbol"].map(self._map)
        for c in ("filed", "tdate"):
            ins[c] = ins[c] + self._shift
        # Insider data re-enabled 28 Sep 2026: the parity leak was the routine-insider rule keyed per insider instead
        # of per insider-and-company (fixed in features.py; parity proven on 4 windows x 8 days).
        self._insider = ins if use_insider else ins.iloc[0:0]
        sic = sic_all
        sic = sic[sic["ticker"].isin(self._map)].copy()
        sic["ticker"] = sic["ticker"].map(self._map)
        self.sic = sic[["ticker", "sic"]]                        # industry codes are timeless
        days = self._stocks["Close"].index
        self.sessions = days
        self.first_live = days[days.searchsorted(start + self._shift)]
        self.i = days.get_loc(self.first_live) - 1               # clock starts at the end of the warm-up
        self.cost_bps = 40 if Y < 1997 else 20 if Y < 2001 else 10
        self._q_tick, self._q_ack = queue.Queue(1), queue.Queue(1)
        self.ticks, self.acks, self.t_start = 0, 0, None
        self.error = None                                        # a clock-thread failure, re-raised in the trader thread
        self._processed = False
        # Phase 22: the clock covers the sessions the trader will be ticked through, one full pass each
        self.clock = BG.BlindClock(days[self.i + 1:])
        self.ledger = BG.InformationLedger()
        if enforce:
            self._gate_disguise(stocks["Close"].loc[lo:hi, live_cols].index, days)

    def _shift_index(self, df):
        df = df.copy()
        df.index = df.index + self._shift
        return df

    def _disguise(self, df):
        df = self._shift_index(df)
        df.columns = [self._map[c] for c in df.columns]
        return df

    def _gate_disguise(self, real_index, shown_index):
        """Phase 21 disguise gates on what the trader will be shown. Fatal on any failure except duplicate price paths
        (real data can hold two listings with identical history; that is reported, not fatal)."""
        f = BG.check_shift(self._shift.days, real_index, shown_index)
        f += BG.check_ticker_map(self._map)
        f += BG.check_disguised_frame(self._stocks["Close"], self._map, real_dates_known=real_index)
        f += BG.check_calendar(real_index, shown_index)
        f += [BG.Finding(x.gate, "warn", x.message) for x in BG.check_disguise_signature(self._stocks["Close"])]
        self.gate_findings += f
        self._raise_on_fail(f)

    @staticmethod
    def _raise_on_fail(findings):
        bad = [x for x in findings if x.severity == "fail"]
        if bad:
            raise BlindGateError("; ".join(str(x) for x in bad[:5]))

    # ---- feature service: computed once, served one session at a time (proven equal to live by parity_test) ----
    def precompute_features(self, rel_q=(0.2, 0.4)):
        ev, ins = self._events, self._insider
        X, atr = features.build(self._stocks, self._market, ev, ins, self.sic,
                                start=str(self.sessions[0].date()), relative=True, rel_q=rel_q, tradable_rule=self.tradable_rule)
        self._X, self._atr = X, atr

    def features_today(self):
        """Only the current session's row: exactly what a live data vendor would serve today."""
        d = self.now
        return self._X.xs(d, level=0, drop_level=False) if d in self._X.index.get_level_values(0) else self._X.iloc[:0]

    def features_until_now(self):
        """Warm-up training data: rows up to the current session only."""
        return self._X[self._X.index.get_level_values(0) <= self.now], self._atr.loc[:self.now]

    # ---- long-term memory (C34): only lessons from windows that ENDED before this one began ----
    def long_term_memory(self):
        bank = DIR / "memory_bank.parquet"
        if not bank.exists():
            return None
        b = pd.read_parquet(bank)
        start = self.first_live - self._shift                    # real start date, known only to the feed
        b = b[pd.to_datetime(b["real_end"]) < start]
        if self.enforce:                                         # C34 causality, checked on what is actually returned
            self._raise_on_fail(BG.check_memory_bank_causality(b, start))
        return b[["arm", "ctx", "outcome"]].reset_index(drop=True) if len(b) else None

    def real_end(self):
        return str((self.sessions[-1] - self._shift).date())

    # ---- what the trader may see ----
    @property
    def now(self):
        return self.sessions[self.i]

    def history(self, lookback=None):
        lo = 0 if lookback is None else max(0, self.i - lookback)
        cut = lambda d: {f: v.iloc[lo:self.i + 1] for f, v in d.items()}
        return cut(self._stocks), cut(self._market)

    def filings(self):
        t = self.now.tz_localize("UTC") + CLOSE_UTC                              # public by this session's close
        ev, ins = self._events[self._events["accepted"] <= t], self._insider[self._insider["filed"] <= self.now]
        if self.enforce:
            self._raise_on_fail(BG.check_public_release(ev, t) + BG.check_public_release(ins, self.now, "filed"))
        return ev, ins

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
        """Lockstep: tick n+1 is released only after the trader acknowledged tick n. A clock failure (ledger, order,
        gate) is stored and re-raised in the trader thread by wait_tick/ack; a daemon thread that just died would
        leave the trader blocked forever."""
        self.t_start = time.perf_counter()
        try:
            while not self.done():
                if self.ticks != self.acks:
                    raise BG.ClockViolation(f"clock ahead of the trader: tick {self.ticks + 1} before ack {self.ticks}")
                self.i += 1
                self.ticks += 1
                if self.enforce:
                    self.clock.begin_session()
                    self._processed = False
                self._q_tick.put(self.now)
                self._q_ack.get()                              # wait for the trader, then tick again at once
                if self.error is not None:
                    return
        except Exception as e:                                 # noqa: BLE001 - relayed to the trader thread
            self.error = e
        finally:
            self._q_tick.put(None)

    def wait_tick(self):
        t = self._q_tick.get()
        if self.error is not None:
            raise self.error
        return t

    def mark_processed(self):
        """The trader calls this once it has handled the session it was ticked for."""
        self._processed = True

    def ack(self):
        """Acknowledge the current tick. Under enforcement this closes the session: the trader must have processed it,
        every applicable clock step is then completed in order (steps 1-7 run inside adaptive.Session.on_day; the
        clock verifies the pass is whole and the week steps happen only on a week's last session) and the exact
        information available is written to the ledger (step 8). A violation is raised to the trader AND stops the
        clock."""
        try:
            if self.enforce:
                if not self._processed:
                    raise BG.ClockViolation(f"session {self.now.date()} acknowledged before it was processed")
                for name in BG.STEPS[:-1]:
                    if name in BG.WEEK_ONLY and not self.clock.is_week_end():
                        continue
                    self.clock.step(name)
                self.ledger.record(self.now + CLOSE_UTC, self._information_sources(), {"week_end": bool(self.clock.is_week_end())})
                self.clock.step("record_information")
        except Exception as e:                                 # noqa: BLE001 - stop the clock, then raise here too
            self.error = e
            self._q_ack.put(True)
            raise
        self.acks += 1
        self._q_ack.put(True)

    def _information_sources(self):
        """Latest source timestamp of everything the trader can see right now, taken from the served data itself
        (independent of filings(), which filters the same way): prices, market series, filings, insider forms."""
        now = self.now
        lo = max(0, self.i - 1)
        src = {"close": self._stocks["Close"].index[lo:self.i + 1].max(), "open": self._stocks["Open"].index[lo:self.i + 1].max()}
        for f, v in self._market.items():
            src[f"market_{f.lower()}"] = v.index[lo:self.i + 1].max()
        acc = self._events["accepted"]                          # sorted by accepted
        k = acc.searchsorted(now.tz_localize("UTC") + CLOSE_UTC, side="right")
        src["filing"] = acc.iloc[k - 1].tz_convert(None) if k else None
        filed = self._insider["filed"]
        vis = filed[filed <= now]
        src["insider"] = vis.max() if len(vis) else None
        return src

    def audit(self):
        """Every Phase 21/22 finding for the finished run: seal, disguise, ledger chain and coverage of decisions.
        Callers must treat any 'fail' as an excluded result."""
        f = list(self.gate_findings)
        f += BG.verify_ledger(self.ledger.entries)
        n_sessions = len(self.clock.sessions)
        if len(self.ledger.entries) != n_sessions:
            f.append(BG.Finding("ledger-coverage", "fail", f"{len(self.ledger.entries)} ledger entries for {n_sessions} sessions"))
        return f


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
        # C33: the model target is measured from the next session's open, where every decision fills. "close" (the old
        # target, which also rewards the overnight gap) is kept so the loop can compare; it is a knob of the trader,
        # not of the policy, so it is taken out of cfg before cfg reaches adaptive.Session.
        self.label_entry = self.cfg.pop("label_entry", None) or (meta or {}).get("label_entry") or "open"
        if self.label_entry not in ("open", "close"):
            raise ValueError(f"label_entry must be 'open' or 'close', not {self.label_entry!r}")
        self.m = None
        self.snaps, self.warm_snaps = {}, {}
        self.t_model = self.t_features = 0.0
        self.divs = {t: policy.sic_division(c) for t, c in zip(feed.sic["ticker"], feed.sic["sic"])}
        self.session = None
        self.preseason = None

    def _fit(self, X, stocks, atr, until_idx):
        yb, fw = features.labels(stocks, atr, entry=self.label_entry)
        d = X.index.get_level_values(0)
        ud = pd.DatetimeIndex(sorted(d.unique()))
        ud = ud[ud <= until_idx]
        tr = ud[: max(1, len(ud) - model.EMBARGO)][::2]        # labels of the last days would need the future
        Xt = X[d.isin(tr)]
        y = yb.stack(future_stack=True).reindex(Xt.index)
        f = fw.stack(future_stack=True).reindex(Xt.index)
        ok = y.notna().values & f.notna().values
        return model.fit_models(model.normalise(Xt)[ok], y[ok], f[ok], fast=True), int(ok.sum())

    def _fit_mover(self, X, stocks, until_idx):
        """C30: the volatility finder inside Test - P(stock touches +/-10% within 5 sessions), trained only on
        warm-up weeks whose label window closed before until_idx."""
        import lightgbm as lgb
        C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
        entry = stocks["Open"].shift(-1)                                   # C33: bought at the next open
        hi = pd.concat([Hh.shift(-k) for k in range(1, 6)]).groupby(level=0).max()
        lo = pd.concat([L.shift(-k) for k in range(1, 6)]).groupby(level=0).min()
        touch = ((hi / entry - 1 >= 0.10) | (lo / entry - 1 <= -0.10)).astype(float)
        ud = pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
        ud = ud[ud <= until_idx]
        ud = ud[: max(1, len(ud) - 6)]
        wk = [d for i, d in enumerate(ud[:-1]) if ud[i + 1].isocalendar().week != d.isocalendar().week]
        cols = [c for c in X.columns if not c.startswith(("ev_activist",))]
        Xt = X[X.index.get_level_values(0).isin(wk)][cols]
        y = touch.stack(future_stack=True).reindex(Xt.index)
        ok = y.notna().values
        Rt = Xt[ok].groupby(level=0).rank(pct=True)
        for c in cols:
            if c.startswith("m_"):
                Rt[c] = Xt[ok][c]
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=300, num_leaves=31, min_child_samples=200,
                                 learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                 random_state=7, verbose=-1)
        clf.fit(Rt, y[ok])
        return {"clf": clf, "cols": cols}

    def _p_move(self, mv, Xd, day):
        xr = Xd.xs(day, level=0)[mv["cols"]]
        R = xr.rank(pct=True)
        for c in mv["cols"]:
            if c.startswith("m_"):
                R[c] = xr[c]
        return mv["clf"].predict_proba(R)[:, 1]

    def snapshot_from(self, m, X, day, mv=None):
        Xd = X.xs(day, level=0, drop_level=False)
        if Xd.empty:
            return None
        xr = Xd.xs(day, level=0)
        R = model.normalise(Xd).xs(day, level=0).reindex(columns=m["cols"])
        p = pd.DataFrame({"mu_raw": m["reg"].predict(R)}, index=R.index)
        mv = mv if mv is not None else getattr(self, "mv", None)
        if mv is not None:
            p["p_move"] = self._p_move(mv, Xd, day)
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
        self.mv = self._fit_mover(X, stocks, now)
        # pre-season study (C17): a second model that stops 18 months earlier, so the last 18 warm-up months
        # are out-of-sample for it; candidate settings are judged there, quarter by quarter
        if self.adaptive:
            cut = now - pd.DateOffset(months=18)
            m2, _ = self._fit(X, stocks, atr, cut)
            mv2 = self._fit_mover(X, stocks, cut)
            closes = stocks["Close"]
            wdays = closes.loc[cut:now].index
            for i, d in enumerate(wdays[:-1]):
                if wdays[i + 1].isocalendar().week != d.isocalendar().week:
                    sn = self.snapshot_from(m2, X, d, mv=mv2)
                    if sn is not None:
                        self.warm_snaps[str(d.date())] = sn
            wclose = closes.loc[cut:now]
            cands = [{**self.cfg, k: v} for k, v in self.A.neighbours(self.cfg, self.A.META_DEFAULT["adaptive_knobs"])]
            run = lambda cfg, sn, cl, bps, dv: self.A.replay(cfg, sn, cl, bps, dv).result()
            chosen, info = self.A.choose_default(self.warm_snaps, wclose, self.divs, self.cfg, self.feed.cost_bps, run, cands)
            self.preseason = {"prior": self.cfg, "chosen": chosen, **info}
            self.cfg = chosen
        self.long_term = self.feed.long_term_memory() if self.adaptive else None
        self.session = self.A.Session(self.cfg, self.divs, self.feed.cost_bps, adaptive=self.adaptive, meta=self.meta,
                                      long_term=self.long_term)
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
        S.on_day(now, self.feed.prices(), closes_to_now, week_end, snap if S.needs_snapshot(week_end) else None,
                 self.feed._stocks["Open"].iloc[self.feed.i])
        self.feed.mark_processed()

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


def order_preserving_codes(names, seed, prefix="Q", width=5):
    """Fresh code names for `names` that keep their sorted order: the k-th smallest name gets the k-th smallest of a
    random draw of distinct numbers, zero-padded so string order equals numeric order. Any tie-break that sorts by
    ticker (Session._trade sorts by (weight, code)) then resolves exactly as it did under the previous names, so a
    rerun differs from the first run only through what the system learned (canon C54/C55)."""
    names = sorted(names)
    if len(names) > 10 ** width:
        raise ValueError(f"{len(names)} names do not fit {width}-digit codes")
    nums = np.sort(np.random.default_rng(seed).choice(10 ** width, size=len(names), replace=False))
    return {n: f"{prefix}{int(k):0{width}d}" for n, k in zip(names, nums)}


def reseal_window(src_id, new_id, seed, revealed, sealed_at=None, shift_weeks=(8000, 11000)):
    """Seal an already REVEALED window again under a new disguise (canon C55): same real months, a new secret shift
    (so new dates and, through Feed, new code names) and a new digest. `revealed` must be True: the caller asserts the
    source window's true period has been revealed through the RevealGate; an unrevealed source is refused, so this can
    never expose a live seal. The new shift differs from the source's by at least 8 whole weeks. Returns the record."""
    if not revealed:
        raise PermissionError(f"{src_id} is not revealed; a live seal is never re-sealed")
    src = json.loads((DIR / f"sealed_{src_id}.json").read_text())
    path = DIR / f"sealed_{new_id}.json"
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; a seal is never overwritten")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        shift = 7 * int(rng.integers(*shift_weeks))
        if abs(shift - src["shift_days"]) >= 56:
            break
    else:
        raise RuntimeError("could not draw a shift far enough from the source shift")
    start = SealedYear.start_of(src)
    rec = {"start": str(start.date()), "shift_days": shift, "sealed_at": str(pd.Timestamp(sealed_at or pd.Timestamp.now())),
           "seed_tag": new_id, "resealed_from_revealed": True}
    rec["digest"] = BG.seal_digest(rec)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return rec


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
        slow, _ = features.build(stocks, market, ev, ins, feed.sic, start=str(feed.now.date()), relative=True,
                                 tradable_rule=feed.tradable_rule)
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


def drive(feed, on_tick):
    """Run the lockstep clock against `on_tick` until the window ends; returns wall seconds. A gate or clock failure
    raised in either thread surfaces here."""
    clock = threading.Thread(target=feed.run_clock, daemon=True)
    clock.start()
    while True:
        t = feed.wait_tick()
        if t is None:
            break
        on_tick()
        feed.ack()
    clock.join(timeout=5)
    if feed.error is not None:
        raise feed.error
    return time.perf_counter() - feed.t_start


def blind_feed_class(hardened=True):
    """The Feed class blind runs use: the hardened one (columns sorted by code, market prices rebased, names hidden until
    they list - engine.leak_audit, canon C56) unless a caller asks for the plain one."""
    if hardened:
        from .leak_audit import hardened_feed_class
        return hardened_feed_class()
    return Feed


def run(cfg, run_id, log=print, check_parity=True, adaptive=False, meta=None, hardened=True):
    sealed = SealedYear(run_id)
    feed = blind_feed_class(hardened)(sealed)
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
    wall = drive(feed, trader.on_tick)
    return feed, trader, sealed, wall
