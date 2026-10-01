"""The research loop's data feed: ONE builder that turns daily OHLCV bars (plus optional events / insider / macro / sector tables)
into EVERY loop stage's inputs, per simulated day (contract C66 sections 3, 35, 47 'the full cycle must actually carry data', 53;
RESEARCH_MAPPING rule 27; canon C56, C63, C64, C66, C67). IMPLEMENTED - NOT VALIDATED (C63: code and planted-world tests only; the
real-cache adapter below is code that has never been run on the real files).

Why this exists: W01 proved every stage is statically reachable, but on its planted feed 24 stages reported SKIPPED_NO_INPUT, i.e. the
loop was wired but no data flowed ('wired means reachable AND data flows'). This module is the data.

Design
  World          the raw tables (bars, events, insider, macro, sectors, SIC codes, market proxy) for one span; `before(now)` is the
                 point-in-time slice (strictly before `now`).
  Source         where Worlds come from, one calendar year at a time (rule 27: stream year by year, never the whole market at once):
                 InMemorySource (the planted world) and RealCacheSource (engine.research.episodes.load_bars + engine/edgar tables).
  FrameStore     the volatility-lab research frame (engine.fv_pipeline.build_panel -> volatility_lab.frame_from_panel ->
                 with_event_inputs) built per year; a look reads the rolling window of the last `frame_weeks` (F16).
  WorldFeed      the loop's Feed: observe(now) returns the Observation (matured panel, point-in-time decision day) and a LAZY builder
                 per stage. A builder runs INSIDE its stage (engine.research.loop.Ctx.extra), so it can read what earlier stages of the
                 same cycle produced (the two-stage decisions feed observer, autopsy, frontier, symmetry, targets), and a future leak
                 it carries is refused AT that stage (REFUSED_LEAK), never upstream where it would be misattributed.
  audit          every builder's output passes `audit_stage_input` before the module sees it: any date-like field at/after `now`
                 raises FirewallBreach. A builder that has nothing new raises NoInput with the reason.
  planted world  `planted_world` hides a GENUINE volatility mechanism (a persistent per-name volatility state, so trailing volatility
                 predicts next week's +-10% touch), scheduled earnings shocks (knowable before the event), a transient early
                 COINCIDENCE (cheap names jump only in the first third of the sample: a noise pattern that must never be promoted),
                 noise insider filings and a macro series; `truth` names all of it so tests score against exact ground truth.
  C68            the prediction-error inputs are built by engine.research.error_loop's registered builder `c68.world` (P06); the
                 earlier `c68_inputs` slot here was an unused duplicate and was retired (29 Sep, C69 section 6).

Public entries: `WorldFeed(source, cfg)` (the loop Feed), `planted_feed(...)`, `real_cache_feed(...)`, `input_table(reports)`."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from engine.research.core import FirewallBreach, as_date, stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
BAR_FIELDS = ("Open", "High", "Low", "Close", "Volume")
MARKET_COLS = ("SPY", "^VIX", "^VIX3M")
FEED_STAGES = ("observe.observer", "observe.autopsy", "evaluate.frontier", "evaluate.symmetry", "surprises.cross_section",
               "surprises.multiscale", "missed.knowability", "missed.counterfactual", "breaks.break_research", "questions.discovery",
               "questions.interactions", "questions.precursors", "questions.targets")
_DATE_KEYS = ("published_at", "resolved_at", "matured_at", "matured_through", "evidence_through", "end_date", "event_end", "filed",
              "accepted", "decided_at")


class NoInput(Exception):
    """Raised by a builder with nothing new for its stage (the loop maps it to SKIPPED_NO_INPUT with this reason)."""


# ================================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class FeedConfig:
    """How the feed slices the world. Units: sessions unless named otherwise."""
    horizon: int = 5                       # holding / outcome horizon (matches the volatility lab and the two-stage chain)
    move: float = 0.10                     # the +-move a 'mover' touches inside the horizon
    daily_mover: float = 0.05              # a single-session |close-to-close| move that counts as a mover episode
    history_years: int = 2                 # streamed bar-world years kept besides the current one (rule 27); the research frame's span
                                           # only when frame_weeks = 0 (the pre-F16 calendar window)
    frame_weeks: int = 156                 # F16: the research frame is the rolling window (now - frame_weeks, now] of decision dates
    bar_lookback: int = 420                # sessions of bars a builder may see (the rest is streamed away)
    first_decision: str | None = None      # loop clock start (None = after warm-up)
    last_decision: str | None = None
    warm_weeks: int = 70                   # decision dates skipped at the start (features and history must exist first)
    min_price: float = 3.0
    min_dollar_vol: float = 2e6
    max_knowability_moves: int = 10        # new moves classified per cycle (the module's history is immutable, so each move once)
    max_counterfactual_events: int = 3     # counterfactual costs ~0.3 s per event and is not incremental
    multiscale_lookback: int = 130        # sessions the multiscale ledger re-reads each call (its cost grows with the window)
    cs_min_history: int = 15               # cross-section lab history before it speaks (sessions)
    frontier_boot: int = 60
    symmetry_min_n: int = 20
    symmetry_min_calls: int = 30
    discovery_min_weeks: int = 20
    precursor_years_back: int = 2
    probe_features: tuple = ("lv20", "price_low")   # patterns always followed by break research / symmetry besides released knowledge
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if not 1 <= self.horizon <= 20 or not 0 < self.move < 1 or not 0 < self.daily_mover < 1:
            errs.append("horizon in [1,20], move and daily_mover in (0,1) required")
        if self.history_years < 0 or self.bar_lookback < 80 or self.warm_weeks < 0:
            errs.append("history_years >= 0, bar_lookback >= 80 and warm_weeks >= 0 required")
        if self.frame_weeks != 0 and not 104 <= self.frame_weeks <= 520:
            errs.append("frame_weeks must be 0 (calendar window) or in [104, 520]: the gate needs two unseen years after a train window")
        if self.max_knowability_moves < 1 or self.max_counterfactual_events < 1:
            errs.append("per-cycle caps must be >= 1")
        if self.frontier_boot < 50:
            errs.append("frontier_boot must be >= 50 (engine.research.frontier refuses fewer)")
        return errs


@dataclasses.dataclass(frozen=True)
class PlantConfig:
    """The planted world. Every mechanism is named in World.truth."""
    n_names: int = 48
    n_days: int = 1010                     # about four years of sessions
    start: str = "2016-01-04"
    seed: int = 0
    n_sectors: int = 6
    vol_persistence: float = 0.99          # AR(1) of the latent log-volatility state: the GENUINE mechanism
    vol_state_sd: float = 0.10             # innovation sd of that state
    base_sigma: tuple = (0.010, 0.020)
    earnings_every: int = 63               # sessions between scheduled earnings (jittered per name)
    earnings_shock: float = 0.07           # sd of the event-day return
    coincidence_share: float = 0.33        # the NOISE pattern lives only in this early share of the sample
    coincidence_rate: float = 0.05         # extra jump probability per cheap name-day inside it
    insider_rate: float = 0.01
    price_range: tuple = (6.0, 160.0)
    # F23: the band-eligible mechanism (a knowable-in-advance 5-10% weekly move for a minority of names). A name whose latent
    # volatility state is HOT (h > trend_hot_sd x the state's stationary sd) may start a directional run (hazard trend_rate per
    # session, direction a fair coin, geometric length with mean trend_len); during the run the name drifts trend_drift per session
    # (~7% a week at the default). The run is invisible on its first days and then shows in r5 / r20 (the precursor): the edge is
    # real but modest because a run ends without warning and most high-r20 names are noise. No hot state (vol_state_sd = 0, the null
    # world) means no run at all. Drawn from its own seeded stream, so every earlier mechanism of a seed is unchanged.
    trend_rate: float = 0.03            # F23 (30 Sep): shorter, more frequent runs; 40-session runs moved prices so far that the
    trend_len: float = 25.0             # planted price_low coincidence became informative late (AUC 0.43) and lv20 drew down -50%
    trend_drift: float = 0.020
    trend_hot_sd: float = 0.5

    def validate(self) -> list[str]:
        errs = []
        if not 0.0 <= self.trend_rate < 1.0 or self.trend_len < 1.0 or not 0.0 <= self.trend_drift < 0.05:
            errs.append("trend_rate in [0,1), trend_len >= 1 and trend_drift in [0, 0.05) required")
        if not 0.0 <= self.vol_persistence < 1.0 or self.vol_state_sd < 0.0:
            errs.append("vol_persistence in [0,1) and vol_state_sd >= 0 required")
        return errs


NULL_PLANT = {"vol_state_sd": 0.0, "base_sigma": (0.015, 0.015), "coincidence_rate": 0.0, "earnings_shock": 0.0}   # nothing predicts


@dataclasses.dataclass(frozen=True)
class LeakPlant:
    """A planted future leak: from `from_date` on, the builder of `stage` is fed one item dated `days_ahead` calendar days AFTER now
    (as a buggy adapter would). The audit must refuse it at that stage."""
    stage: str
    from_date: str
    days_ahead: int = 3


# ================================================================================================================ the world
@dataclasses.dataclass
class World:
    bars: dict                                   # field -> DataFrame (sessions x tickers)
    events: pd.DataFrame | None = None           # ticker, kind, accepted (UTC); engine/edgar events.parquet layout
    insider: pd.DataFrame | None = None          # symbol, filed; engine/edgar insider.parquet layout
    macro: pd.DataFrame | None = None            # date-indexed, one column per series, already dated by publication
    sectors: Mapping[str, str] = dataclasses.field(default_factory=dict)
    sic: Mapping[str, int] = dataclasses.field(default_factory=dict)
    market: dict | None = None                   # field -> DataFrame with MARKET_COLS (SPY, ^VIX, ^VIX3M)
    truth: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.bars["Close"].index)

    @property
    def tickers(self) -> list[str]:
        return [str(c) for c in self.bars["Close"].columns]

    def validate(self) -> list[str]:
        errs = []
        miss = [f for f in BAR_FIELDS if f not in self.bars]
        if miss:
            return [f"bars missing {miss}"]
        c = self.bars["Close"]
        for f in BAR_FIELDS:
            b = self.bars[f]
            if not b.index.equals(c.index) or list(b.columns) != list(c.columns):
                errs.append(f"{f} is not aligned with Close")
        if len(c.index) and not c.index.is_monotonic_increasing:
            errs.append("sessions are not increasing")
        if len(c) and (self.bars["High"] < self.bars["Low"] - 1e-9).any().any():
            errs.append("a bar has high < low")
        if self.events is not None and len(self.events) and not {"ticker", "kind", "accepted"} <= set(self.events.columns):
            errs.append("events need ticker, kind, accepted")
        if self.insider is not None and len(self.insider) and not {"symbol", "filed"} <= set(self.insider.columns):
            errs.append("insider needs symbol, filed")
        return errs

    def before(self, now, lookback: int | None = None) -> "World":
        """Everything known strictly before `now` (a session's bar is known at its close; events at their acceptance time;
        insider filings the day after filing; macro rows by their publication date index)."""
        n = pd.Timestamp(as_date(now))
        idx = self.sessions
        keep = idx < n
        if lookback is not None:
            pos = np.flatnonzero(keep)
            keep = np.zeros(len(idx), bool)
            keep[pos[-lookback:]] = True
        bars = {f: self.bars[f].loc[keep] for f in BAR_FIELDS}
        mk = {f: v.loc[v.index < n] for f, v in self.market.items()} if self.market else None
        ev = None if self.events is None else self.events[_utc(self.events["accepted"]) < _utc_ts(n)]
        ins = None if self.insider is None else self.insider[pd.to_datetime(self.insider["filed"]) + pd.Timedelta(days=1) < n]
        mac = None if self.macro is None else self.macro.loc[self.macro.index < n]
        return World(bars, ev, ins, mac, self.sectors, self.sic, mk, self.truth)

    def window(self, start, end) -> "World":
        a, b = pd.Timestamp(as_date(start)), pd.Timestamp(as_date(end))
        bars = {f: self.bars[f].loc[a:b] for f in BAR_FIELDS}
        mk = {f: v.loc[a:b] for f, v in self.market.items()} if self.market else None
        ev = None if self.events is None else self.events[(_utc(self.events["accepted"]) >= _utc_ts(a)) & (_utc(self.events["accepted"]) <= _utc_ts(b + pd.Timedelta(days=1)))]
        ins = None if self.insider is None else self.insider[(pd.to_datetime(self.insider["filed"]) >= a) & (pd.to_datetime(self.insider["filed"]) <= b)]
        mac = None if self.macro is None else self.macro.loc[a:b]
        return World(bars, ev, ins, mac, self.sectors, self.sic, mk, self.truth)


def _utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s)
    return t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")


def _utc_ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def market_proxy(bars: Mapping[str, pd.DataFrame]) -> dict:
    """An equal-weight index (SPY stand-in) and realised-volatility VIX / VIX3M proxies, in the layout pit.PITStore expects.
    Used for the planted world and whenever the real market cache is absent (the real adapter prefers market_*.parquet)."""
    C = bars["Close"]
    r = C.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    spy = 100.0 * (1.0 + r).cumprod()
    vix = (r.rolling(20, min_periods=5).std() * math.sqrt(252) * 100).fillna(15.0)          # no bfill: it copied later values into the warm-up (F09 leak)
    vix3 = (r.rolling(63, min_periods=10).std() * math.sqrt(252) * 100).fillna(17.0)
    base = pd.DataFrame({"SPY": spy, "^VIX": vix, "^VIX3M": vix3}, index=C.index)
    out = {f: base.copy() for f in ("Open", "High", "Low", "Close")}
    out["Volume"] = pd.DataFrame({"SPY": 1e8, "^VIX": 0.0, "^VIX3M": 0.0}, index=C.index)
    return out


GENUINE_VOL = ("lv20", "latr", "xs_vol_rank", "xs_atr_rank", "xs_range_rank", "vol_over_mkt", "sector_rel_vol", "lv20_x_stress",
               "max5_ratio", "vol_ratio_short")


def planted_world(pc: PlantConfig = PlantConfig()) -> World:
    """Seeded OHLCV world with KNOWN mechanisms (see the module docstring). truth keys: genuine (derived features that carry the
    volatility state), genuine_event, noise (the early coincidence), coincidence_until, earnings (list of (session, ticker))."""
    rng = np.random.default_rng(pc.seed)
    n, T = pc.n_names, pc.n_days
    if n < 8 or T < 300:
        raise ValueError("planted world needs >= 8 names and >= 300 sessions")
    errs = pc.validate()
    if errs:
        raise ValueError("; ".join(errs))
    dates = pd.bdate_range(pc.start, periods=T)
    tick = [f"W{j:03d}" for j in range(n)]
    base = rng.uniform(pc.base_sigma[0], pc.base_sigma[1], n)
    h = np.zeros((T, n))
    h[0] = rng.normal(0, pc.vol_state_sd / math.sqrt(1 - pc.vol_persistence ** 2), n)
    for t in range(1, T):
        h[t] = pc.vol_persistence * h[t - 1] + rng.normal(0, pc.vol_state_sd, n)
    sig = base * np.exp(h)
    r = np.clip(rng.standard_t(5, (T, n)) * sig / math.sqrt(5 / 3), -0.25, 0.25)
    V = np.exp(rng.normal(14.0, 0.5, n)) * np.exp(rng.normal(0, 0.25, (T, n)))
    p0 = np.exp(rng.uniform(math.log(pc.price_range[0]), math.log(pc.price_range[1]), n))
    cheap = p0 < np.median(p0)
    until = int(T * pc.coincidence_share)
    jumps = (rng.random((until, n)) < pc.coincidence_rate) & cheap[None, :]
    r[:until] += np.where(jumps, rng.choice([-1.0, 1.0], (until, n)) * rng.uniform(0.06, 0.12, (until, n)), 0.0)
    earn = []
    phase = rng.integers(5, pc.earnings_every, n)
    for j in range(n):
        t = int(phase[j])
        while t < T:
            earn.append((t, j))
            r[t, j] += rng.normal(0, pc.earnings_shock)
            V[t, j] *= 3.0
            t += int(pc.earnings_every + rng.integers(-3, 4))
    drift, runs = trend_runs(h, pc)
    r = r + drift                      # no re-clip: with the mechanism off the world is bit-identical to the pre-F23 world
    C = p0 * np.exp(np.cumsum(np.log1p(r), axis=0))
    prev = np.vstack([C[:1] / (1 + r[:1]), C[:-1]])
    O = prev * (1.0 + r * rng.uniform(0.0, 0.5, (T, n)))
    wick = np.abs(rng.normal(0, 1, (2, T, n))) * sig * 0.6
    H = np.maximum(O, C) * (1 + wick[0])
    L = np.minimum(O, C) * (1 - np.minimum(wick[1], 0.5))
    bars = {k: pd.DataFrame(v, index=dates, columns=tick) for k, v in zip(BAR_FIELDS, (O, H, L, C, V))}
    ev = pd.DataFrame({"ticker": [tick[j] for _, j in earn], "kind": "EARN", "form": "8-K",
                       "accepted": [(dates[t] - pd.Timedelta(hours=10)).tz_localize("UTC") for t, _ in earn]})
    ev = ev.sort_values("accepted").reset_index(drop=True)          # released before the open of the event session (public that day)
    ins_mask = rng.random((T, n)) < pc.insider_rate
    it, ij = np.nonzero(ins_mask)
    ins = pd.DataFrame({"symbol": [tick[j] for j in ij], "filed": dates[it], "value": rng.uniform(1e4, 1e6, len(it))})
    rate = np.cumsum(rng.normal(0, 0.02, T)) + 2.0
    macro = pd.DataFrame({"rate": rate}, index=dates + pd.Timedelta(days=1))    # published the day after it is measured
    sectors = {t: f"SEC{j % pc.n_sectors}" for j, t in enumerate(tick)}
    sic = {t: int((1311, 2834, 3571, 4911, 5411, 6021, 7372, 3674)[j % 8]) for j, t in enumerate(tick)}
    truth: dict[str, Any] = {"genuine": GENUINE_VOL,
             "genuine_event": ("ev_soon", "ev_days"), "noise": ("price_low",), "noise_family": ("price_low", "dv_level", "dv_rel",
                                                                                                 "insider_recent"),
             "coincidence_until": str(dates[until - 1].date()), "earnings": [(str(dates[t].date()), tick[j]) for t, j in earn],
             "cheap": [tick[j] for j in np.flatnonzero(cheap)], "seed": pc.seed}
    if runs:
        # a run makes |r20| a genuine volatility predictor and r5 / r20 a genuine (modest) direction precursor
        truth["genuine"] = GENUINE_VOL + ("abs_rel_r20",)
        truth["genuine_direction"] = ("rel_r20", "rel_r5")
    truth["trend_runs"] = [(str(dates[a].date()), tick[j], int(s), int(k)) for a, j, s, k in runs]
    truth["trend_share"] = float((drift != 0).mean())
    return World(bars, ev, ins, macro, sectors, sic, market_proxy(bars), truth)


def trend_runs(h: np.ndarray, pc: PlantConfig) -> tuple[np.ndarray, list]:
    """F23: the per-session drift of the directional runs and the runs themselves as (start session, name index, sign, length).
    A run can start only while the name's volatility state is hot, and only when it is not already running; its length is drawn at
    the start (geometric) and nothing about it is visible before it moves the price. Seeded on its own stream [seed, 0x7E1D]."""
    T, n = h.shape
    drift = np.zeros((T, n))
    runs: list = []
    sd_h = pc.vol_state_sd / math.sqrt(1.0 - pc.vol_persistence ** 2) if pc.vol_state_sd > 0 else 0.0
    if pc.trend_rate <= 0 or pc.trend_drift <= 0 or sd_h <= 0:
        return drift, runs
    rg = np.random.default_rng([int(pc.seed), 0x7E1D])
    hot = h > pc.trend_hot_sd * sd_h
    left = np.zeros(n, int)
    sign = np.zeros(n)
    for t in range(T):
        u, dirs, lens = rg.random(n), np.asarray(rg.choice([-1.0, 1.0], n), float), np.asarray(rg.geometric(1.0 / pc.trend_len, n), int)
        start = (left == 0) & hot[t] & (u < pc.trend_rate)
        for j in np.flatnonzero(start):
            runs.append((t, int(j), int(dirs[j]), int(lens[j])))
        sign = np.where(start, dirs, sign)
        left = np.where(start, lens, left)
        drift[t] = sign * pc.trend_drift
        left = np.maximum(left - 1, 0)
        sign = np.where(left == 0, 0.0, sign)
    return drift, runs


# ================================================================================================================ sources
class Source(Protocol):
    def years(self) -> list[int]: ...

    def load(self, year: int, warm_days: int, future_days: int) -> World: ...


class InMemorySource:
    """A World held in memory (the planted world). load(year) returns the year's sessions plus `warm_days` before and `future_days`
    after, exactly the slicing the real adapter does, so the rest of the feed cannot tell them apart."""

    def __init__(self, world: World):
        errs = world.validate()
        if errs:
            raise ValueError("invalid world: " + "; ".join(errs))
        self.world = world

    def years(self) -> list[int]:
        return sorted(set(self.world.sessions.year))

    def load(self, year: int, warm_days: int, future_days: int) -> World:
        a = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(warm_days * 1.5) + 10)
        b = pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=int(future_days * 1.5) + 7)
        return self.world.window(a, b)


class RealCacheSource:
    """The real-cache adapter (C63: code only, never run on the real files in this wave). Bars stream a year at a time through
    engine.research.episodes.load_bars (pre-2000 + modern files; SURVIVOR-ONLY panel, see the canon); events / insider / macro / SIC
    come from engine/edgar's parquet layout, read with a date filter so only the year's rows are held. Tickers can be a seeded sample
    (rule 10: RAM is shared)."""

    def __init__(self, years: Sequence[int], tickers: Sequence[str] | None = None, cache_dir=None, sample: int | None = None,
                 seed: int = 0, with_tables: bool = True):
        from engine import config as K
        self.cache = Path(cache_dir) if cache_dir is not None else Path(K.CACHE)
        self._years = sorted(int(y) for y in years)
        self.tickers = list(tickers) if tickers is not None else None
        self.sample, self.seed, self.with_tables = sample, seed, with_tables

    def years(self) -> list[int]:
        return list(self._years)

    def _names(self) -> list[str] | None:
        if self.tickers is not None or self.sample is None:
            return self.tickers
        from engine.research import episodes as EP
        all_t = EP.cache_tickers(self.cache)
        rng = np.random.default_rng(self.seed)
        return sorted(rng.choice(all_t, size=min(self.sample, len(all_t)), replace=False).tolist())

    def _read(self, name: str, date_col: str | None, a, b) -> pd.DataFrame | None:
        p = self.cache / name
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if date_col is not None and date_col in df:
            d = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
            df = df[(d >= a) & (d <= b)]
        return df

    def load(self, year: int, warm_days: int, future_days: int) -> World:
        from engine.research import episodes as EP
        a = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(warm_days * 1.5) + 10)
        b = pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=int(future_days * 1.5) + 7)
        bars = EP.load_bars(a, b, self._names(), self.cache)
        bars = {f: bars[f].astype("float32") for f in BAR_FIELDS}
        ev = ins = mac = None
        sic: dict = {}
        mk = None
        if self.with_tables:
            ev = self._read("events.parquet", "accepted", a, b)
            ins = self._read("insider.parquet", "filed", a, b)
            if ins is not None:
                from engine.features import clean_insider
                ins = clean_insider(ins)
            m = self._read("macro.parquet", None, a, b)
            mac = m.loc[(m.index >= a) & (m.index <= b)] if m is not None and isinstance(m.index, pd.DatetimeIndex) else None
            s = self._read("sic.parquet", None, a, b)
            if s is not None and {"ticker", "sic"} <= set(s.columns):
                sic = {str(t): int(v) for t, v in zip(s["ticker"], s["sic"]) if pd.notna(v)}
            parts = {f: v for f in BAR_FIELDS if (v := self._read(f"market_{f.lower()}.parquet", None, a, b)) is not None}
            if len(parts) == len(BAR_FIELDS) and set(MARKET_COLS) <= set(parts["Close"].columns):
                mk = {f: v.loc[a:b, list(MARKET_COLS)] for f, v in parts.items()}
        sectors = {t: f"SIC{v // 100:02d}" for t, v in sic.items()}
        return World(bars, ev, ins, mac, sectors, sic, mk or market_proxy(bars), {"source": "real_cache", "survivor_only": True})


# ================================================================================================================ research frame
def research_frame(world: World, cfg: FeedConfig = FeedConfig(), year: int | None = None) -> pd.DataFrame:
    """The volatility-lab frame (index (date, ticker); point-in-time features; outcomes touch/absmove/tday/close/up/end) built by the
    repo's own pipeline: fv_pipeline.build_panel (features at week-end closes, outcome of buying the NEXT open and holding) ->
    volatility_lab.frame_from_panel -> with_event_inputs (event features from records public strictly before each decision)."""
    from engine import fv_pipeline as FV
    from engine.research import volatility_lab as VL
    if len(world.sessions) < 80:
        return pd.DataFrame()
    fvc = FV.FVConfig(move=cfg.move, min_price=cfg.min_price, min_dollar_vol=cfg.min_dollar_vol)
    P = FV.build_panel({f: world.bars[f] for f in BAR_FIELDS}, fvc)
    F = VL.frame_from_panel(P, VL.LabConfig(move=cfg.move), sector=dict(world.sectors) or None)
    if year is not None:
        F = F[np.asarray(pd.to_datetime(F.index.get_level_values(0)).year == year)]
    ev = world.events.rename(columns={}) if world.events is not None and len(world.events) else None
    F = VL.with_event_inputs(F, ev, world.insider if world.insider is not None and len(world.insider) else None)
    F = F[np.isfinite(F["touch"].to_numpy(float)) | pd.isna(F["touch"]).to_numpy()]
    F.attrs["survivor_free"] = bool(world.truth.get("survivor_free", "genuine" in world.truth))
    return F


class FrameStore:
    """Year-by-year research frames, keeping only the rows `observe` may still read (rule 27: stream, never hold the market).

    F16 (C69 sections 12-14, 21; follow-up to F14): `upto(now)` used to return whole calendar years now.year - history_years .. now.year,
    so in the first weeks of a year the frame had just dropped its oldest year and held no matured row of the new one: it spanned two
    calendar years and the gate (two unseen years required) could see only one - every January look was starved. The frame is now the
    ROLLING window (now - frame_weeks, now] of decision dates, the same length of history in every month. Frames are still built one
    calendar year at a time (a year's features are computed once, with its warm-up, exactly as before); the oldest held year is trimmed
    to the window and rebuilt only if an earlier `now` later needs the trimmed rows (observe stays a pure function of now)."""

    def __init__(self, source: Source, cfg: FeedConfig):
        self.source, self.cfg = source, cfg
        self.frames: dict[int, pd.DataFrame] = {}
        self.trimmed: dict[int, pd.Timestamp] = {}          # year -> rows dated at/before this were dropped from the held frame
        self.worlds: dict[int, World] = {}
        self.built = 0
        self.rebuilt = 0

    def world(self, year: int) -> World:
        if year not in self.worlds:
            self.worlds[year] = self.source.load(year, max(self.cfg.bar_lookback, 90), self.cfg.horizon + 5)
            for y in [y for y in self.worlds if y < year - self.cfg.history_years - 1]:
                del self.worlds[y]
        return self.worlds[year]

    def frame(self, year: int, since: pd.Timestamp | None = None) -> pd.DataFrame:
        """The research frame of calendar year `year`. `since` = the caller needs every row dated after it; a held frame trimmed
        past that point is rebuilt (never served short)."""
        cut = self.trimmed.get(year)
        if year in self.frames and cut is not None and (since is None or since < cut):
            del self.frames[year], self.trimmed[year]
            self.rebuilt += 1
        if year not in self.frames:
            self.frames[year] = research_frame(self.world(year), self.cfg, year)
            self.built += 1
        return self.frames[year]

    def span(self, now) -> tuple[pd.Timestamp | None, pd.Timestamp, list[int]]:
        """(lower bound exclusive or None for the calendar window, now, the calendar years the frame reads)."""
        n = pd.Timestamp(as_date(now))
        if self.cfg.frame_weeks:
            lo = n - pd.Timedelta(weeks=self.cfg.frame_weeks)
            return lo, n, [k for k in self.source.years() if lo.year <= k <= n.year]
        return None, n, [k for k in self.source.years() if n.year - self.cfg.history_years <= k <= n.year]

    def _release(self, lo: pd.Timestamp | None, years: Sequence[int]) -> None:
        """Drop held years the window has left and trim the oldest held year to the window (rows dated <= lo are never read again
        by a forward run; an earlier `now` triggers a rebuild through `frame(since=...)`)."""
        first = min(years) if years else None
        for y in [y for y in self.frames if first is None or y < first]:
            del self.frames[y]
            self.trimmed.pop(y, None)
        if lo is None or first is None or first not in self.frames or first != lo.year:
            return
        F = self.frames[first]
        if len(F):
            keep = np.asarray(pd.to_datetime(F.index.get_level_values(0)) > lo)
            if not keep.all():
                self.frames[first] = F[keep]
                self.trimmed[first] = max(lo, self.trimmed.get(first, lo))

    def upto(self, now) -> pd.DataFrame:
        """The research frame for a look at `now`: rows dated in (now - frame_weeks, now] (calendar years now.year - history_years ..
        when frame_weeks = 0). Never a row dated after `now`; outcomes of rows near `now` are still immature (observe keeps end < now)."""
        lo, n, years = self.span(now)
        parts = [self.frame(k, lo if lo is not None and k == lo.year else None) for k in years]
        self._release(lo, years)
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame()
        F = pd.concat(parts).sort_index()
        d = pd.to_datetime(F.index.get_level_values(0))
        keep = d <= n if lo is None else (d > lo) & (d <= n)
        return F[np.asarray(keep)]

    def bars_before(self, now, lookback: int | None = None) -> World:
        """The point-in-time world for builders: this year's and last year's streamed worlds merged, strictly before `now`."""
        y = as_date(now).year
        ws = [self.world(k) for k in (y - 1, y) if k in self.source.years()]
        if not ws:
            raise NoInput(f"no bars for {y}")
        w = _merge_worlds(ws)
        return w.before(now, lookback or self.cfg.bar_lookback)


def _merge_worlds(ws: Sequence[World]) -> World:
    if len(ws) == 1:
        return ws[0]

    def cat(frames):
        f = pd.concat(frames)
        return f[~f.index.duplicated(keep="last")].sort_index()
    bars = {f: cat([w.bars[f] for w in ws]) for f in BAR_FIELDS}
    mk = {f: cat([m[f] for w in ws if (m := w.market) is not None]) for f in BAR_FIELDS} if all(w.market for w in ws) else None
    tabs = []
    for attr in ("events", "insider"):
        parts = [getattr(w, attr) for w in ws if getattr(w, attr) is not None]
        tabs.append(pd.concat(parts).drop_duplicates().reset_index(drop=True) if parts else None)
    mac = [w.macro for w in ws if w.macro is not None]
    return World(bars, tabs[0], tabs[1], cat(mac) if mac else None, ws[-1].sectors, ws[-1].sic, mk, ws[-1].truth)


# ================================================================================================================ the input audit
def _max_date(x) -> pd.Timestamp | None:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        if len(x) == 0:
            return None
        idx = x.index
        if isinstance(idx, pd.MultiIndex):
            idx = idx.get_level_values(0)
        if isinstance(idx, pd.DatetimeIndex):
            m = idx.max()
            return None if pd.isna(m) else pd.Timestamp(m).tz_localize(None) if m.tzinfo else pd.Timestamp(m)
        return None
    return None


def audit_stage_input(stage: str, payload: Any, now, depth: int = 0) -> int:
    """Fail closed on anything a stage input carries that is dated at/after `now`: DatetimeIndex maxima of frames, and the dated
    fields named in _DATE_KEYS of mappings / dataclasses / frames (columns). Returns how many dated values were checked (a check that
    looked at nothing is visible). Raises engine.research.core.FirewallBreach naming the stage and the field."""
    n = pd.Timestamp(as_date(now))
    checked = 0
    if depth > 4 or payload is None:
        return 0
    if isinstance(payload, (pd.Timestamp, dt.datetime, dt.date, np.datetime64)):
        d = pd.Timestamp(payload)
        d = d.tz_convert("UTC").tz_localize(None) if d.tzinfo else d
        if d.normalize() >= n:
            raise FirewallBreach(f"{stage}: a dated value {d.date()} is at/after now {n.date()}")
        return 1
    m = _max_date(payload)
    if m is not None:
        checked += 1
        if m.normalize() >= n:
            raise FirewallBreach(f"{stage}: input frame reaches {m.date()} >= now {n.date()}")
    if isinstance(payload, pd.DataFrame):
        for k in _DATE_KEYS:
            if k in payload.columns and len(payload):
                v = pd.to_datetime(payload[k], utc=True, errors="coerce").dt.tz_localize(None)
                checked += 1
                if (v.dt.normalize() >= n).any():
                    raise FirewallBreach(f"{stage}: column {k} holds {int((v.dt.normalize() >= n).sum())} value(s) at/after now {n.date()}")
        return checked
    items: Iterable[tuple[Any, Any]]
    if isinstance(payload, Mapping):
        items = payload.items()
    elif dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        items = ((f.name, getattr(payload, f.name)) for f in dataclasses.fields(payload))
    elif isinstance(payload, (list, tuple)):
        for x in payload[:2000]:
            checked += audit_stage_input(stage, x, now, depth + 1)
        return checked
    else:
        return checked
    for k, v in items:
        if k in _DATE_KEYS and v not in (None, ""):
            try:
                d = pd.Timestamp(v)
            except (ValueError, TypeError):
                continue
            d = d.tz_convert("UTC").tz_localize(None) if d.tzinfo else d
            checked += 1
            if d.normalize() >= n:
                raise FirewallBreach(f"{stage}: {k}={d.date()} is at/after now {n.date()}")
        elif isinstance(v, (pd.DataFrame, pd.Series, Mapping, list, tuple)) or (dataclasses.is_dataclass(v) and not isinstance(v, type)):
            checked += audit_stage_input(stage, v, now, depth + 1)
    return checked


# ================================================================================================================ builder helpers
def _memo(ctx, key: str) -> dict:
    """Per-builder bookkeeping kept in the loop state (checkpointed with it), so a resumed loop feeds exactly what the killed one
    would have: nothing twice, nothing skipped."""
    return ctx.state.memo.setdefault("feed:" + key, {})


def _decisions(ctx) -> list:
    """Two-stage decisions (this loop's own outputs), oldest first."""
    return list(getattr(ctx.state, "decisions", []) or [])


def _resolved(dec, sessions: pd.DatetimeIndex, horizon: int) -> tuple[int, int] | None:
    """(entry position, last position) of a decision's holding window when it resolved strictly inside `sessions`, else None."""
    d = pd.Timestamp(as_date(dec.decided_at))
    pos = int(sessions.searchsorted(d, side="right"))
    last = pos + horizon - 1
    if pos <= 0 or last >= len(sessions):
        return None
    return pos, last


def _sector_code(sectors: Mapping[str, str], tickers: Sequence[str]) -> np.ndarray:
    labels = sorted(set(sectors.values()))
    code = {s: i for i, s in enumerate(labels)}
    return np.array([code.get(sectors.get(t, ""), -1) for t in tickers], np.int64)


def _by_ticker(table: pd.DataFrame) -> pd.DataFrame:
    t = table.droplevel(0) if isinstance(table.index, pd.MultiIndex) else table
    t.index = t.index.astype(str)
    return t


# ================================================================================================================ stage builders
def b_observer_pair(feed: "WorldFeed", ctx, key: str) -> dict:
    """observer / autopsy: the newest two-stage decision whose holding window resolved strictly before now, as a DecisionSnapshot
    (what the model considered: score = P(volatility), dir_prob = P(up | mover), picked = a position) and a DayOutcome (entry at the
    next open, horizon high / low / close; day_* = the fill session)."""
    from engine.research import observer as OB
    memo = _memo(ctx, key)
    done = memo.setdefault("done", [])
    w = feed.store.bars_before(ctx.now)
    S = w.sessions
    H = feed.cfg.horizon
    for dec in reversed(_decisions(ctx)):
        if dec.decided_at in done or len(dec.table) == 0:
            continue
        rr = _resolved(dec, S, H)
        if rr is None:
            continue
        pos, last = rr
        tb = _by_ticker(dec.table)
        tk = [t for t in tb.index if t in w.bars["Close"].columns]
        tb = tb.loc[tk]
        C, O, Hh, L, V = (w.bars[f][tk] for f in ("Close", "Open", "High", "Low", "Volume"))
        d0 = S[pos - 1]
        side = tb["side"].to_numpy(float)
        elig = tb["eligible"].to_numpy(bool)
        conf = np.clip(np.nan_to_num(tb["p_move"].to_numpy(float), nan=0.5), 0.0, 1.0)
        rets = C.pct_change(fill_method=None)
        feats = {"vol20": rets.iloc[max(0, pos - 20):pos].std().to_numpy(float),
                 "r5": (C.iloc[pos - 1] / C.iloc[max(0, pos - 6)] - 1).to_numpy(float)}
        ev_known = np.zeros(len(tk), bool)
        if w.events is not None and len(w.events):
            acc = _utc(w.events["accepted"])
            recent = w.events[(acc < _utc_ts(S[pos])) & (acc >= _utc_ts(d0 - pd.Timedelta(days=5)))]
            ev_known = np.isin(tk, recent["ticker"].astype(str).to_numpy())
        snap = OB.DecisionSnapshot.make(str(d0.date()), tk, C.iloc[pos - 1].to_numpy(float), eligible=elig,
                                        score=tb["p_move"].to_numpy(float), confidence=conf, dir_prob=tb["p_up"].to_numpy(float),
                                        picked=(side != 0) & elig, sector=_sector_code(w.sectors, tk), features=feats, event_known=ev_known)
        vr = (V.iloc[pos] / V.iloc[max(0, pos - 20):pos].mean()).to_numpy(float)
        out = OB.DayOutcome.make(str(S[last].date()), tk, O.iloc[pos].to_numpy(float), Hh.iloc[pos:last + 1].max().to_numpy(float),
                                 L.iloc[pos:last + 1].min().to_numpy(float), C.iloc[last].to_numpy(float), volume_ratio=vr,
                                 day_hi=Hh.iloc[pos].to_numpy(float), day_lo=L.iloc[pos].to_numpy(float), day_close=C.iloc[pos].to_numpy(float))
        errs = snap.validate() + out.validate()
        if errs:
            raise ValueError(f"{key}: built an invalid snapshot/outcome: {errs[:3]}")
        done.append(dec.decided_at)
        return {"snap": snap, "out": out, "ctx": None}
    raise NoInput(f"no two-stage decision has resolved ({H} sessions) strictly before now and not been observed yet")


def b_frontier(feed: "WorldFeed", ctx) -> dict:
    """Conditional-accuracy frontier rows from the two-stage chain: P(up | predicted mover) of its decisions when the direction model
    exists, else the chain's own walk-forward calibration block (research side). Only rows whose outcome matured before now."""
    from engine.research import frontier as FR
    from engine.research import two_stage as TS
    from engine.research import vol_hypotheses as VH
    memo = _memo(ctx, "frontier")
    seen = memo.setdefault("dates", [])
    M = ctx.obs.matured
    rows = []
    for dec in _decisions(ctx):
        t = dec.table
        if dec.decided_at in seen or len(t) == 0:
            continue
        mv = t[t["mover"].to_numpy(bool) & np.isfinite(t["p_up"].to_numpy(float))]
        j = mv.join(M[["up", "close", "end"]], how="inner")
        j = j[j["up"].notna()]
        if len(j):
            rows.append(pd.DataFrame({"date": j.index.get_level_values(0), "ticker": j.index.get_level_values(1).astype(str),
                                      "p": j["p_up"].clip(0.001, 0.999).to_numpy(float), "up": j["up"].to_numpy(float),
                                      "matured_at": pd.to_datetime(j["end"]).to_numpy(), "fwd": j["close"].to_numpy(float)}))
            seen.append(dec.decided_at)
    pipe = getattr(ctx.rt, "pipe", None)
    if not rows and pipe is not None and pipe.dir_model is not None and pipe.calibrator is not None and pipe.last_oos is not None:
        oos = pipe.last_oos
        pool = oos[TS.predicted_movers(oos["p_TS"], pipe.cfg).to_numpy()]
        pdd = pd.to_datetime(pool.index.get_level_values(0))
        cd = np.array(sorted(pd.unique(pdd)))
        if len(cd) >= 3:
            cut = cd[int(len(cd) * (1 - pipe.cfg.calib_frac))]
            fed = set(seen)
            keep = np.asarray(pdd >= cut) & ~np.isin(pdd.strftime("%Y-%m-%d"), list(fed))
            blk = pool[keep]
            blk = blk[np.asarray(pd.to_datetime(blk["end"]) < pd.Timestamp(as_date(ctx.now))) & blk["up"].notna().to_numpy()]
            if len(blk):
                p = pipe.calibrator(pipe.dir_model.prob(VH.derive(blk, pipe.dir_cols).to_numpy(float)))
                rows.append(pd.DataFrame({"date": blk.index.get_level_values(0), "ticker": blk.index.get_level_values(1).astype(str),
                                          "p": np.clip(p, 0.001, 0.999), "up": blk["up"].to_numpy(float),
                                          "matured_at": pd.to_datetime(blk["end"]).to_numpy(), "fwd": blk["close"].to_numpy(float)}))
                seen.extend(sorted(set(pd.to_datetime(blk.index.get_level_values(0)).strftime("%Y-%m-%d"))))
    if not rows:
        raise NoInput("no matured P(up | predicted mover) rows from the two-stage chain yet")
    R = pd.concat(rows, ignore_index=True).dropna(subset=["p", "up", "fwd"]).drop_duplicates(["date", "ticker"])
    fc = FR.FrontierConfig(n_boot=feed.cfg.frontier_boot, n_sim=feed.cfg.frontier_boot)
    return {"rows": R, "make": lambda: FR.FrontierState(fc)}


def _pattern_calls(M: pd.DataFrame, feature: str, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """(call, margin) of a released-knowledge / probe pattern on a universe frame: it calls a WINNER (+1) on its top decile when the
    sign is positive, a LOSER (-1) when negative; silent (0) elsewhere; margin = rank - 0.9."""
    from engine.research import vol_hypotheses as VH
    r = (sign * VH.derive(M, (feature,))[feature]).groupby(level=0).rank(pct=True).to_numpy(float)
    call = np.where(r >= 0.9, 1 if sign >= 0 else -1, 0)
    return call, np.nan_to_num(r - 0.9, nan=-1.0)


def _patterns(feed: "WorldFeed", ctx, columns) -> list[tuple[str, str, float]]:
    """(pattern id, feature, sign): every released-knowledge feature plus the configured probes (identity-free ids)."""
    from engine.research import vol_hypotheses as VH
    out = {}
    for _, k in sorted(getattr(ctx.state, "knowledge", {}).items()):
        out[f"k_{k['feature']}"] = (k["feature"], float(k.get("sign", 1.0)))
    for f in feed.cfg.probe_features:
        out.setdefault(f"p_{f}", (f, 1.0))
    return [(pid, f, s) for pid, (f, s) in sorted(out.items()) if f in VH.DERIVED and not VH.missing_columns((f,), columns)]


def b_symmetry(feed: "WorldFeed", ctx) -> dict:
    """Winner/loser symmetry rows: every pattern's call on the WHOLE matured universe of each new decision date (silent rows
    included, so missed winners and losers are measurable)."""
    from engine.research import symmetry as SY
    memo = _memo(ctx, "symmetry")
    seen = set(memo.setdefault("dates", []))
    M = ctx.obs.matured
    if len(M) == 0:
        raise NoInput("no matured universe yet")
    ds = pd.to_datetime(M.index.get_level_values(0)).strftime("%Y-%m-%d")
    new = M[~np.isin(ds, list(seen)) & M["close"].notna().to_numpy()]
    pats = _patterns(feed, ctx, M.columns)
    if len(new) == 0 or not pats:
        raise NoInput("no new matured decision date for the symmetry engine")
    med = float(np.nanmedian(M["m_vol"])) if "m_vol" in M else 0.0
    mvol = new["m_vol"].to_numpy(float) if "m_vol" in new else np.zeros(len(new))
    ev = new["days_to_event"].between(0, 5).to_numpy(bool) if "days_to_event" in new else np.zeros(len(new), bool)
    rows = []
    for pid, f, s in pats:
        call, margin = _pattern_calls(new, f, s)
        rows.append(pd.DataFrame({"pattern_id": pid, "date": new.index.get_level_values(0), "ticker": new.index.get_level_values(1).astype(str),
                                  "call": call, "lean": 1 if s >= 0 else -1, "margin": margin, "fwd": new["close"].to_numpy(float),
                                  "matured_at": pd.to_datetime(new["end"]).to_numpy(), "vol": new["vol20"].to_numpy(float),
                                  "dollar_volume": np.exp(new["log_dv"].to_numpy(float)) if "log_dv" in new else np.nan,
                                  "gap": new["gap"].to_numpy(float) if "gap" in new else 0.0,
                                  "regime": np.where(mvol > med, "high_vol", "low_vol"), "regime_transition": False, "external_event": ev}))
    memo["dates"] = sorted(seen | set(pd.to_datetime(new.index.get_level_values(0)).strftime("%Y-%m-%d")))
    cfg = SY.SymmetryConfig(min_n=feed.cfg.symmetry_min_n, min_calls=feed.cfg.symmetry_min_calls)
    return {"rows": pd.concat(rows, ignore_index=True), "make": lambda: SY.SymmetryState(cfg)}


def b_cross_section(feed: "WorldFeed", ctx) -> dict:
    """Cross-sectional day frames (ret, sic, vol20, log_dv, mom20 per name) for every session since the last one fed, strictly
    before now, each with the settlement of the session two back (its next-session return, matured the session before)."""
    from engine.research import cross_section as CS
    memo = _memo(ctx, "cross_section")
    last = memo.get("last")
    w = feed.store.bars_before(ctx.now)
    C, V = w.bars["Close"], w.bars["Volume"]
    S = w.sessions
    ret = C.pct_change(fill_method=None)
    start = int(S.searchsorted(pd.Timestamp(last), side="right")) if last else max(21, len(S) - 5)
    fed = memo.setdefault("fed", [])
    names = [str(c) for c in C.columns]
    days = []
    for t in range(max(start, 21), len(S)):
        fr = pd.DataFrame({"ret": ret.iloc[t].to_numpy(float), "sic": [w.sic.get(c, -1) for c in names],
                           "vol20": ret.iloc[t - 19:t + 1].std().to_numpy(float),
                           "log_dv": np.log((C.iloc[t] * V.iloc[t]).clip(lower=1.0)).to_numpy(float),
                           "mom20": (C.iloc[t] / C.iloc[t - 20] - 1).to_numpy(float)}, index=names)
        settle: tuple[tuple[Any, pd.Series, Any], ...] = ()
        if str(S[t - 2].date()) in fed:
            fwd = pd.Series(ret.iloc[t - 1].to_numpy(float), index=names)
            settle = ((S[t - 2], fwd, S[t - 1]),)
        days.append((S[t], fr, settle))
        fed.append(str(S[t].date()))
    memo["fed"] = fed[-10:]
    if not days:
        raise NoInput("no new session for the cross-section lab")
    memo["last"] = str(S[-1].date())
    cfg = CS.CrossConfig(min_history=feed.cfg.cs_min_history, persist_window=5, min_names=min(30, max(8, C.shape[1] // 2)))
    return {"days": days, "make": lambda: CS.CrossSectionLab(cfg), "null_seed": feed.cfg.seed}


def b_multiscale(feed: "WorldFeed", ctx) -> dict:
    """Multiscale inputs: pattern flags (single-session movers, volume spikes) as (date, ticker) booleans, and the close / open
    panels, all strictly before now."""
    w = feed.store.bars_before(ctx.now, min(feed.cfg.bar_lookback, feed.cfg.multiscale_lookback))
    C, O, V = w.bars["Close"], w.bars["Open"], w.bars["Volume"]
    if len(C) < 40:
        raise NoInput("fewer than 40 sessions of bars")
    mover = C.pct_change(fill_method=None).abs() >= feed.cfg.daily_mover
    spike = (V / V.rolling(20, min_periods=10).mean()) >= 2.5
    flags = {}
    for name, f in (("mover_day", mover), ("volume_spike", spike)):
        s = f.stack()
        s.index.names = ["date", "ticker"]
        flags[name] = s.astype(bool)
    # each flag's DESIGN horizon is declared (next session): left undeclared, the ledger declares the measured home, which can move
    # between calls and then refuses the re-declaration (observed on the planted run after ~50 cycles)
    return {"flags": flags, "close": C, "open": O, "declared": {k: "1d" for k in flags}}


def _info_items(w: World, ticker: str, lo: pd.Timestamp, hi: pd.Timestamp, now) -> list:
    """Knowability InfoItems for one name in [lo, hi]: EDGAR-style events (public at acceptance) and insider filings (public the day
    after filing). Only items PUBLISHED strictly before now enter (the rest do not exist yet)."""
    from engine.research import knowability as KB
    n = pd.Timestamp(as_date(now))
    items = []
    if w.events is not None and len(w.events):
        e = w.events[w.events["ticker"].astype(str) == ticker]
        acc = _utc(e["accepted"]).dt.tz_localize(None)
        e = e[((acc >= lo) & (acc <= hi) & (acc < n)).to_numpy()]
        for i, a in enumerate(_utc(e["accepted"]).dt.tz_localize(None)):
            items.append(KB.InfoItem(f"ev-{ticker}-{a.date()}-{i}", KB.InfoKind.EVENT, ticker, str(a.date()), str(a.date()), "edgar",
                                     strength=0.7, detail="earnings"))
    if w.insider is not None and len(w.insider):
        s = w.insider[w.insider["symbol"].astype(str) == ticker]
        f = pd.to_datetime(s["filed"])
        for i, fd in enumerate(f[((f >= lo) & (f <= hi) & (f + pd.Timedelta(days=1) < n)).to_numpy()]):
            items.append(KB.InfoItem(f"in-{ticker}-{fd.date()}-{i}", KB.InfoKind.INSIDER, ticker, str(fd.date()),
                                     str((fd + pd.Timedelta(days=1)).date()), "form4", strength=0.3))
    return items


def b_knowability(feed: "WorldFeed", ctx) -> dict:
    """'Could I have known?' inputs: each single-session mover (|close-to-close| >= daily_mover) becomes a MoveInputs with a FIXED
    bar window (the module's history is immutable, so a move is submitted once, only after its window closed before now), the market
    return series and the information items published before now. At most max_knowability_moves per cycle, oldest first."""
    from engine.research import knowability as KB
    memo = _memo(ctx, "knowability")
    done = set(memo.setdefault("done", []))
    w = feed.store.bars_before(ctx.now)
    C, O, Hh, L, V = (w.bars[f] for f in ("Close", "Open", "High", "Low", "Volume"))
    S = w.sessions
    c2c = C.pct_change(fill_method=None)
    mret = c2c.mean(axis=1).fillna(0.0)
    h = 3
    out = []
    ti, tj = np.nonzero((c2c.abs() >= feed.cfg.daily_mover).to_numpy())
    for k in np.lexsort((tj, ti)):
        t, j = int(ti[k]), int(tj[k])
        if t < 60 or t + h + 3 >= len(S):
            continue
        tk = str(C.columns[j])
        mid = f"M-{tk}-{S[t].date()}"
        if mid in done:
            continue
        cut, end = S[t + h + 3], S[t + h - 1]
        b = pd.DataFrame({"open": O[tk], "high": Hh[tk], "low": L[tk], "close": C[tk], "volume": V[tk]}).loc[S[t - 60]:cut]
        mv = KB.MoveEvent(mid, tk, str(S[t - 1].date()), str(S[t].date()), str(end.date()), float(C[tk].iloc[t + h - 1] / O[tk].iloc[t] - 1), h,
                          sector=str(w.sectors.get(tk, "")))
        out.append(KB.MoveInputs(mv, b, market=mret.loc[S[t - 60]:cut], items=tuple(_info_items(w, tk, S[t - 60], cut, ctx.now))))
        done.add(mid)
        if len(out) >= feed.cfg.max_knowability_moves:
            break
    memo["done"] = sorted(done)
    if not out:
        raise NoInput("no new single-session mover whose window closed before now")
    return {"inputs": out, "calendar": None}


def b_counterfactual(feed: "WorldFeed", ctx) -> dict:
    """Counterfactual inputs: a point-in-time pit.PITStore over the bars, market proxy and events known before now, and the newest
    single-session movers (at least 10 sessions old, so the module's settle window is past) as EventSpecs, max_counterfactual_events
    per cycle (the module re-assesses everything it is given, so it is only given what is new)."""
    from engine.research import counterfactual as CFM
    memo = _memo(ctx, "counterfactual")
    done = set(memo.setdefault("done", []))
    w = feed.store.bars_before(ctx.now, 300)
    C, O = w.bars["Close"], w.bars["Open"]
    S = w.sessions
    ti, tj = np.nonzero((C.pct_change(fill_method=None).abs() >= feed.cfg.daily_mover).to_numpy())
    cand = []
    for t, j in sorted(zip(ti.tolist(), tj.tolist()), reverse=True):
        if t < 30 or t + 2 >= len(S) - 10:
            continue
        tk = str(C.columns[j])
        key = f"{tk}|{S[t].date()}"
        if key in done:
            continue
        r = float(C[tk].iloc[t + 2] / O[tk].iloc[t] - 1)
        if np.isfinite(r) and r != 0:
            cand.append((t, tk, r, key))
        if len(cand) >= feed.cfg.max_counterfactual_events:
            break
    if not cand:
        raise NoInput("no new matured mover episode for the counterfactual lab")
    ev = w.events if w.events is not None and len(w.events) else None
    store = CFM.store_from_feed_data((w.bars, w.market, ev, None, None))
    specs = []
    for t, tk, r, key in cand:
        peers = tuple(str(c) for c in C.columns if str(c) != tk and w.sectors.get(str(c)) == w.sectors.get(tk))[:6]
        specs.append(CFM.EventSpec.make(tk, S[t - 1], S[t + 2], 1 if r > 0 else -1, store.cal, category="mover", realized_return=r,
                                        peers=peers))
        done.add(key)
    memo["done"] = sorted(done)
    return {"store": store, "events": specs, "providers": None}


def feature_week_series(M: pd.DataFrame, feature: str, sign: float) -> pd.Series:
    """Per decision date: touch rate of the pattern's top quintile minus the date's base rate (the signed outcome a pattern that
    selects on this feature earned that week)."""
    from engine.research import vol_hypotheses as VH
    r = (sign * VH.derive(M, (feature,))[feature]).groupby(level=0).rank(pct=True)
    top = M["touch"].where(r >= 0.8)
    return (top.groupby(level=0).mean() - M["touch"].groupby(level=0).mean()).astype(float)


def b_break_research(feed: "WorldFeed", ctx) -> dict:
    """Pattern item series for break research: each released-knowledge feature and each probe as a weekly signed-outcome series on the
    matured panel, with market context columns (volatility, breadth, trend, liquidity) known at each decision."""
    from engine.learning import break_detection as BD
    M = ctx.obs.matured
    M = M[M["touch"].notna()] if len(M) else M
    if len(M) == 0:
        raise NoInput("no matured panel")
    per = M.groupby(level=0)
    ctxf = pd.DataFrame({"m_vol": per["vol20"].median(), "m_breadth": per["r5"].apply(lambda s: float((s > 0).mean())),
                         "m_trend": per["r20"].median(), "m_liq": per["log_dv"].median() if "log_dv" in M else per["vol20"].count()})
    ctxf.index = pd.DatetimeIndex(ctxf.index)
    cols = (BD.ContextColumn("m_vol", "volatility"), BD.ContextColumn("m_breadth", "breadth"), BD.ContextColumn("m_trend", "trend"),
            BD.ContextColumn("m_liq", "liquidity"))
    items = {}
    for pid, f, s in _patterns(feed, ctx, M.columns):
        v = feature_week_series(M, f, s).rename("value")
        v.index = pd.DatetimeIndex(v.index)
        fr = pd.concat([v, ctxf], axis=1).dropna()
        if len(fr) >= 30:
            items[pid] = BD.ItemSeries(pid, fr, cols)
    if not items:
        raise NoInput("fewer than 30 matured weeks for every pattern")
    return {"items": items, "health": None}


def b_discovery(feed: "WorldFeed", ctx) -> dict:
    """Discovery SourceInputs via discovery.inputs_from_wide over the bars before now, plus earnings / filings / insider / macro tables
    in its own layout; `engine_factory` builds the small-panel DiscoveryEngine the stage creates once."""
    from engine.research import discovery as DI
    from engine.research import discovery_sources as DS
    w = feed.store.bars_before(ctx.now)
    tabs: dict = {}
    if w.events is not None and len(w.events):
        acc = _utc(w.events["accepted"]).dt.tz_localize(None).dt.normalize()
        e = pd.DataFrame({"ticker": w.events["ticker"].astype(str).to_numpy(), "date": acc.to_numpy(), "kind": w.events["kind"].to_numpy()})
        tabs["earnings"] = e[e["kind"] == "EARN"][["ticker", "date"]].reset_index(drop=True)
        tabs["filings"] = pd.DataFrame({"ticker": e["ticker"], "filed_at": e["date"]})
    if w.insider is not None and len(w.insider):
        tabs["insiders"] = pd.DataFrame({"ticker": w.insider["symbol"].astype(str).to_numpy(),
                                         "filed_at": pd.to_datetime(w.insider["filed"]).to_numpy(),
                                         "value": w.insider["value"].to_numpy(float) if "value" in w.insider else 1.0})
    if w.macro is not None and len(w.macro):
        tabs["macro"] = w.macro
    inp = DI.inputs_from_wide(w.bars, sectors=dict(w.sectors) or None, **tabs)
    errs = inp.validate()
    if errs:
        raise ValueError(f"discovery inputs invalid: {errs[:3]}")
    dcfg = DI.DiscoveryConfig(min_weeks=feed.cfg.discovery_min_weeks, min_rows=100, min_active_weeks=6, context_min_weeks=6, max_pairs=200,
                              max_unless=20, unless_top_pairs=4, top_singles=10, n_val_blocks=2, null_reps=2, min_independent=3, holdout_frac=0.25)
    return {"inputs": inp, "engine_factory": lambda: DI.DiscoveryEngine(dcfg, DS.SourceConfig(warmup=25), audit=False, families_per_step=6)}


_IA_ROLES = {"vol_surge": "volume_ratio", "atr": "atr_pct", "r20": "mom_20", "gap": "gap_pct", "vol20": "vol_10", "log_dv": "liq_dollar",
             "m_vol": "m_vix"}


def b_interactions(feed: "WorldFeed", ctx) -> dict:
    """Interaction inputs from the matured panel: features renamed to the module's role prefixes, y = the horizon return, pattern
    masks (released knowledge and probes, top decile), an event column (earnings within 5 days) and a volatility regime per date."""
    from engine.research import interactions as IA
    M = ctx.obs.matured
    M = M[M["close"].notna()] if len(M) else M
    if len(M) < 200:
        raise NoInput(f"{len(M)} matured rows (< 200) for interaction search")
    X = pd.DataFrame({new: M[old].to_numpy(float) for old, new in _IA_ROLES.items() if old in M}, index=M.index)
    if "r5" in M:
        X["rev_5"] = -M["r5"].to_numpy(float)
    masks = pd.DataFrame(index=M.index)
    for pid, f, s in _patterns(feed, ctx, M.columns):
        call, _ = _pattern_calls(M, f, s)
        masks["pat_" + pid] = call != 0
    events = pd.DataFrame({"ev_earnings": M["days_to_event"].between(0, 5).to_numpy(bool)}, index=M.index) if "days_to_event" in M else None
    mv = M["m_vol"].groupby(level=0).median() if "m_vol" in M else M["vol20"].groupby(level=0).median()
    regimes = pd.Series(np.where(mv > mv.median(), "high", "low"), index=pd.DatetimeIndex(mv.index))
    return {"inputs": IA.InteractionInputs(X, M["close"].astype(float), feed.cfg.horizon, masks=masks if masks.shape[1] else None,
                                           events=events, regimes=regimes)}


def b_precursors(feed: "WorldFeed", ctx) -> dict:
    """R21 precursor inputs: an in-memory loader over the bars known before now and the completed calendar years to cover."""
    y = as_date(ctx.now).year
    years = [k for k in feed.source.years() if y - feed.cfg.precursor_years_back <= k < y]
    if not years:
        raise NoInput("no completed calendar year before now to sweep")
    return {"loader": feed.pit_loader(ctx.now), "years": years, "cfg": feed.precursor_cfg(), "registry": feed.precursor_registry()}


def b_targets(feed: "WorldFeed", ctx) -> dict:
    """Daily research-target sources from the matured decisions: every eligible name of the newest matured decision day is a
    PredictionRow of the two-stage 'predicted mover' call (correct = the call matched the realised touch), earlier matured days are
    the pattern history, this cycle's screen events are the surprises and sector coverage counts what has been studied."""
    from engine.research import targets as TG
    M = ctx.obs.matured
    memo = _memo(ctx, "targets")
    cols = [c for c in ("touch", "close", "m_vol", "m_breadth", "sector") if c in M]
    rows_by_day = []
    for dec in _decisions(ctx):
        t = dec.table
        if len(t) == 0:
            continue
        j = t[t["eligible"].to_numpy(bool)].join(M[cols], how="inner")
        j = j[j["touch"].notna()]
        if len(j) < 6:
            continue
        rows = [TG.PredictionRow("two_stage_mover", bool(bool(r["mover"]) == (r["touch"] >= 0.5)),
                                 {"m_vol": float(r.get("m_vol", 0.0)), "breadth": float(r.get("m_breadth", 0.5))},
                                 group=str(r.get("sector", "all")), move=float(r["close"]), predicted_move=bool(r["mover"]))
                for _, r in j.iterrows()]
        rows_by_day.append((dec.decided_at, rows))
    if not rows_by_day:
        raise NoInput("no matured decision day with at least 6 eligible names")
    day, preds = rows_by_day[-1]
    if memo.get("last") == day:
        raise NoInput(f"the newest matured decision day ({day}) was already turned into targets")
    memo["last"] = day
    hist = [r for _, rs in rows_by_day[:-1] for r in rs]
    surprises = [{"subject": str(e.subject)[:80], "z": float(3.0 * e.magnitude)} for e in (ctx.bus.get("events") or [])][:10]
    cov = {"sector": {str(k): int(v) for k, v in M["sector"].value_counts().items()}} if "sector" in M else {}
    return {"day": TG.DayInput(ctx.obs.evidence_through, preds, surprises=surprises, pattern_history=hist, coverage=cov)}


BUILDERS: dict[str, Callable] = {
    "observe.observer": lambda f, c: b_observer_pair(f, c, "observer"),
    "observe.autopsy": lambda f, c: b_observer_pair(f, c, "autopsy"),
    "evaluate.frontier": b_frontier,
    "evaluate.symmetry": b_symmetry,
    "surprises.cross_section": b_cross_section,
    "surprises.multiscale": b_multiscale,
    "missed.knowability": b_knowability,
    "missed.counterfactual": b_counterfactual,
    "breaks.break_research": b_break_research,
    "questions.discovery": b_discovery,
    "questions.interactions": b_interactions,
    "questions.precursors": b_precursors,
    "questions.targets": b_targets,
}


def register_builder(key: str, fn: Callable, replace: bool = False) -> None:
    """PUBLIC. Add a builder `fn(feed, ctx) -> dict` under a stage name or a namespaced key (e.g. 'c68.errors') so a registered loop
    stage (engine.research.loop.register_stage) gets its per-day inputs from this feed without editing it. Its output passes the same
    fail-closed audit; raise feeds.NoInput for 'nothing new'."""
    if not key or not callable(fn):
        raise ValueError("register_builder needs a key and a callable fn(feed, ctx) -> dict")
    if key in BUILDERS and not replace and BUILDERS[key] is not fn:
        raise ValueError(f"a builder for {key!r} exists; pass replace=True to swap it")
    BUILDERS[key] = fn


# ================================================================================================================ planted leaks
def plant_leak(stage: str, payload: dict, now, days_ahead: int) -> dict:
    """Inject ONE future-dated item into a real input of `stage` (what a buggy adapter would do). Supported: knowability (an
    information item published after now), observer / autopsy (an outcome resolved after now), multiscale (a close row after now);
    any other stage gets a dated probe record the audit must refuse the same way."""
    fut = pd.Timestamp(as_date(now)) + pd.Timedelta(days=days_ahead)
    out = dict(payload)
    if stage == "missed.knowability" and out.get("inputs"):
        from engine.research import knowability as KB
        first = out["inputs"][0]
        leak = KB.InfoItem("leak-probe", KB.InfoKind.NEWS_EXTERNAL, first.move.ticker, str(fut.date()), str(fut.date()), "planted", strength=1.0)
        out["inputs"] = [dataclasses.replace(first, items=tuple(first.items) + (leak,))] + list(out["inputs"][1:])
    elif stage in ("observe.observer", "observe.autopsy") and "out" in out:
        out["out"] = dataclasses.replace(out["out"], resolved_at=str(fut.date()))
    elif stage == "surprises.multiscale" and "close" in out:
        c = out["close"]
        row = c.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex([fut])
        out["close"] = pd.concat([c, row])
    else:
        out["leak_probe"] = {"published_at": str(fut.date())}
    return out


# ================================================================================================================ the feed
class WorldFeed:
    """The loop Feed over a Source. `observe(now)` is a pure function of (source, cfg, now): a resumed loop sees exactly what the
    killed one saw; builders keep their bookkeeping in the loop state, not here."""

    def __init__(self, source: Source, cfg: FeedConfig = FeedConfig(), leak: LeakPlant | None = None, sessions: Sequence | None = None):
        errs = cfg.validate()
        if errs:
            raise ValueError("invalid FeedConfig: " + "; ".join(errs))
        self.source, self.cfg, self.leak = source, cfg, leak
        self.store = FrameStore(source, cfg)
        self._dates: list[str] | None = None
        self._sessions = pd.DatetimeIndex(sessions) if sessions is not None else None
        self.audit_counts: dict[str, int] = {}
        self._pc = None

    # ---- clock
    def all_sessions(self) -> pd.DatetimeIndex:
        if self._sessions is None:
            if isinstance(self.source, InMemorySource):
                self._sessions = self.source.world.sessions
            else:
                parts = [self.store.world(y).sessions for y in self.source.years()]
                self._sessions = pd.DatetimeIndex(sorted(set().union(*[set(p) for p in parts]))) if parts else pd.DatetimeIndex([])
        return self._sessions

    def dates(self) -> list[str]:
        """Decision dates: the last session of each ISO week (the fv_pipeline convention), after `warm_weeks`, within the config's
        first/last bounds, and never the final week of the data (its outcome cannot mature inside it)."""
        if self._dates is None:
            from engine import fv_pipeline as FV
            S = self.all_sessions()
            if len(S) < 10:
                self._dates = []
                return []
            wk = S[FV.week_end_sessions(S)]
            d = [str(x.date()) for x in wk[self.cfg.warm_weeks:-2]]
            if self.cfg.first_decision:
                d = [x for x in d if x >= str(as_date(self.cfg.first_decision))]
            if self.cfg.last_decision:
                d = [x for x in d if x <= str(as_date(self.cfg.last_decision))]
            self._dates = d
        return list(self._dates)

    # ---- the observation
    def observe(self, now):
        from engine.research import loop as LP
        from engine.research import two_stage as TS
        n = pd.Timestamp(as_date(now))
        F = self.store.upto(now)
        if len(F) == 0:
            return LP.Observation(str(as_date(now)), F, F, {}, "")
        ends = pd.to_datetime(F["end"])
        mat = F[np.asarray(ends < n)]
        pit = TS.point_in_time_state(F, now)
        through = str(ends[ends < n].max()) if len(mat) else ""
        dh = stable_hash({"n": len(mat), "through": through, "cols": list(mat.columns), "world": str(self.source.years())}, 16)
        extras = {stage: _Lazy(self, stage) for stage in BUILDERS}
        return LP.Observation(str(as_date(now)), mat, pit.frame, extras, dh)

    def build(self, stage: str, ctx) -> dict:
        """Run one stage's builder inside the stage, plant the configured leak if due, and audit the result fail-closed."""
        payload = BUILDERS[stage](self, ctx)
        if self.leak is not None and self.leak.stage == stage and as_date(ctx.now) >= as_date(self.leak.from_date):
            payload = plant_leak(stage, payload, ctx.now, self.leak.days_ahead)
        audit_ms = {k: v for k, v in payload.items() if k not in ("make", "engine_factory", "loader", "registry", "cfg")}
        self.audit_counts[stage] = self.audit_counts.get(stage, 0) + audit_stage_input(stage, audit_ms, ctx.now)
        return payload

    # ---- R21 precursor plumbing
    def pit_loader(self, now):
        """A precursor loader over the bars known strictly before `now` (the module itself refuses any session at/after now)."""
        from engine.research import episodes as EP

        def load(unit, ecfg, extra_warm: int = 0, extra_future: int = 0):
            ys = [y for y in self.source.years() if y <= as_date(now).year]
            w = _merge_worlds([self.store.world(y) for y in ys[-3:]]).before(now)
            return EP.year_bars(w.bars, unit.year, ecfg, extra_warm, extra_future)
        return load

    def precursor_cfg(self):
        from engine.research import precursors as PC
        return PC.LabConfig(n_slices=1, lenses=("c2c",), n_perm=3, cluster_months=1, max_controls=20, seed=self.cfg.seed + 1)

    def precursor_registry(self):
        from engine.research import precursors as PC
        return PC.PrecursorRegistry([s for s in PC.default_registry(audit=False).specs.values() if s.family in ("volume", "gap")], audit=False)

    def sweeps(self) -> list:
        """C67 always-on sweep over this feed: R21's precursor sweep with a point-in-time loader bound to each call's `now`."""
        from engine.research import loop as LP
        from engine.research import precursors as PC
        feed = self

        def factory(root: Path):
            years = [y for y in feed.source.years()][:-1] or feed.source.years()
            st, store = PC.open_state(Path(root) / "sweeps" / "precursors_feed", years, feed.precursor_cfg(), registry=feed.precursor_registry())

            def run(n: int, now) -> int:
                ready = [y for y in feed.source.years() if y < as_date(now).year]     # F29: years become sweepable as they complete
                if not ready:
                    return 0
                st.book.extend_years(ready)
                return len(PC.step(st, now, feed.pit_loader(now), max_units=n, store=store).done)
            return run
        return [LP.SweepSpec("sweep.precursors_feed", "NEW_REPRESENTATION", factory)]


class _Lazy:
    """A stage input that is built when (and only when) its stage asks for it: engine.research.loop.Ctx.extra calls it with the
    stage context. Picklable-free by design: it lives on the transient Observation, never in the checkpointed state."""

    def __init__(self, feed: WorldFeed, stage: str):
        self.feed, self.stage = feed, stage

    def __call__(self, ctx) -> dict:
        try:
            return self.feed.build(self.stage, ctx)
        except NoInput as e:
            from engine.research import loop as LP
            raise LP.NoInput(str(e)) from None


def planted_feed(pc: PlantConfig = PlantConfig(), cfg: FeedConfig = FeedConfig(), leak: LeakPlant | None = None) -> WorldFeed:
    return WorldFeed(InMemorySource(planted_world(pc)), cfg, leak)


def real_cache_feed(years: Sequence[int], cfg: FeedConfig = FeedConfig(), tickers: Sequence[str] | None = None, sample: int | None = 300,
                    cache_dir=None, seed: int = 0) -> WorldFeed:
    """The real-cache feed (C63: code only; not run in this wave). A seeded ticker sample keeps RAM bounded (rule 10)."""
    return WorldFeed(RealCacheSource(years, tickers, cache_dir, sample, seed), cfg)


# ================================================================================================================ reports
def input_table(reports: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Per stage over a run: cycles OK / skipped for no input / refused / failed / other, the first OK cycle and the last skip reason.
    The section-47 question 'did data reach every stage?' answered from the loop's own stage records."""
    rows: dict[str, dict] = {}
    for rep in reports:
        for s in rep.get("stages", []):
            r = rows.setdefault(s["stage"], {"stage": s["stage"], "ok": 0, "no_input": 0, "refused": 0, "failed": 0, "other": 0,
                                             "first_ok": None, "n_in": 0, "last_skip": ""})
            st = s["status"]
            if st == "OK":
                r["ok"] += 1
                r["n_in"] += int(s.get("n_in", 0))
                if r["first_ok"] is None:
                    r["first_ok"] = rep["cycle"]
            elif st == "SKIPPED_NO_INPUT":
                r["no_input"] += 1
                r["last_skip"] = s.get("reason", "")[:90]
            elif st == "REFUSED_LEAK":
                r["refused"] += 1
            elif st == "FAILED":
                r["failed"] += 1
                r["last_skip"] = s.get("reason", "")[:90]
            else:
                r["other"] += 1
    return pd.DataFrame(list(rows.values()), columns=["stage", "ok", "no_input", "refused", "failed", "other", "first_ok", "n_in", "last_skip"])


def starved_stages(reports: Sequence[Mapping[str, Any]], exempt: Iterable[str] = ()) -> list[str]:
    """Stages that never once ran OK over the run (data never reached them), excluding `exempt` (disabled ones)."""
    t = input_table(reports)
    ex = set(exempt)
    return sorted(s for s, ok in zip(t["stage"], t["ok"]) if ok == 0 and s not in ex)
