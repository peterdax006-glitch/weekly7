"""Feature future-leak audit by truncation (C69 ledger section 5: future-leak concern 2 / data-quality problem 6; C69 sections 21-22;
canon C56 "only live information").

Every production frame builder is run twice: on the full inputs, and on the inputs truncated at a cut date T (bars through T, filings
accepted by the end of T, insider reports filed by T - `engine.parity.Inputs.truncate`, the same point-in-time cut the parity firewall
uses). Every value dated <= T must be identical. A difference means the feature read the future: a panel-wide mean/std/rank/quantile or
winsorisation over all dates, a full-sample fitted scaler, a backfill, a centred window, a negative shift, a universe chosen by later
listings. The comparison is `engine.parity.compare_panels` (one comparator for the whole repo).

    python scripts/feature_leak_audit.py [--source planted|real] [--tickers 40] [--since 2012-01-01] [--cuts 5] [--seed 0]
                                         [--only NAME ...] [--out state/research/feature_leak] [--fail-on leak|never]

Output: <out>/report.md (one row per feature: CLEAN / LEAK / UNTESTED, max abs diff, offending operation file:line) and
<out>/results_<source>.json with provenance. Exit 0 = every production feature CLEAN (or UNTESTED with a reason), 1 = a LEAK
(with --fail-on leak), 2 = crash, 3 = not enough free RAM. A planted leaking builder runs in every audit as a canary: if the audit
fails to flag it, the audit itself is broken and the run fails (a check that cannot fail is worthless).

Truncation cannot see a leak that is baked into the stored data (back-adjusted prices that encode later splits/dividends, a survivor-
only universe, a revised macro vintage). Those channels are audited elsewhere (engine.leak_audit, parity_suite's publication audit,
the survivorship gate) and are listed in the report as out of scope, never as CLEAN."""
from __future__ import annotations

import argparse
import dataclasses as dc
import importlib.util
import inspect
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine import config as K
from engine import parity as P

KEY = "_"                                      # ticker slot for date-level (market-wide) builders
TOL = P.GATE_TOL
ET = "America/New_York"
MIN_FREE_GB = 2.5


# =====================================================================================================================================
# inputs: a planted world (tests, CI) and a small real-cache slice
# =====================================================================================================================================
def planted_inputs(n_days: int = 420, n_tickers: int = 24, seed: int = 0, start: str = "2015-01-02") -> P.Inputs:
    """Seeded OHLCV + market + filings + insider + SIC world with the awkward cases truncation must survive: a late listing, a delisting,
    a ticker whose FIRST filing arrives late in the sample, prices straddling the tradability floor, filings after 15:30 and on weekends,
    quarterly earnings cadences, insider buys filed late."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tick = [f"T{i:02d}" for i in range(n_tickers)]
    sig = rng.uniform(0.01, 0.04, n_tickers)
    ret = rng.standard_t(4, (n_days, n_tickers)) * sig / np.sqrt(2) + 0.0002
    p0 = np.exp(rng.uniform(np.log(2.0), np.log(120.0), n_tickers))
    C = pd.DataFrame(p0 * np.exp(np.cumsum(ret, 0)), index=dates, columns=tick)
    O = C.shift(1).fillna(C.iloc[0]) * np.exp(rng.normal(0, 0.006, C.shape))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.008, C.shape)))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.008, C.shape)))
    V = pd.DataFrame(np.exp(rng.normal(15.5, 0.8, n_tickers)) * np.exp(rng.normal(0, 0.4, C.shape)), index=dates, columns=tick)
    late, gone = tick[-1], tick[-2]
    for f in (O, H, L, C, V):
        f.loc[dates[: n_days // 3], late] = np.nan              # lists a third of the way in
        f.loc[dates[-n_days // 5:], gone] = np.nan              # delists before the end
    stocks = dict(Open=O, High=H, Low=L, Close=C, Volume=V)
    mret = ret.mean(1) + rng.normal(0, 0.004, n_days)
    spy = 200 * np.exp(np.cumsum(mret))
    vix = np.clip(18 + 60 * pd.Series(np.abs(mret)).rolling(10, min_periods=1).mean().to_numpy() * 10 + rng.normal(0, 1, n_days), 9, 80)
    m = pd.DataFrame({"SPY": spy, "^VIX": vix, "^VIX3M": vix * rng.uniform(1.0, 1.15, n_days)}, index=dates)
    market = {k: m.copy() for k in ("Open", "High", "Low", "Close")}
    rows = []
    for j, tk in enumerate(tick):
        first = 5 + int(rng.integers(0, 60)) if j != 3 else int(n_days * 0.7)       # T03's first filing is late
        for i in range(first, n_days, 63):
            hour = [8, 12, 16, 17][int(rng.integers(0, 4))]
            rows.append((tk, "EARN", "8-K", dates[i] + pd.Timedelta(hours=hour)))
        for i in rng.integers(first, n_days, 3):
            kind = ["SHELF", "OFFERING", "AGREEMENT", "ACTIVIST", "LATE_FILING"][int(rng.integers(0, 5))]
            form = {"OFFERING": "424B5", "SHELF": "S-3", "ACTIVIST": "SC 13D"}.get(kind, "8-K")
            rows.append((tk, kind, form, dates[int(i)] + pd.Timedelta(hours=int(rng.integers(7, 20)))))
    rows.append((tick[5], "EARN", "8-K", dates[200] + pd.Timedelta(days=1, hours=10)))   # a Saturday-ish filing (calendar day off-session)
    ev = pd.DataFrame(rows, columns=["ticker", "kind", "form", "accepted"])
    ev["accepted"] = ev["accepted"].dt.tz_localize(ET)
    ins_rows = []
    for k in range(60):
        tk = tick[int(rng.integers(0, n_tickers - 2))]
        td = dates[int(rng.integers(20, n_days - 5))]
        lag = int(rng.choice([1, 2, 3, 30]))
        ins_rows.append(dict(symbol=tk, owner_cik=int(rng.integers(1, 12)), tdate=td, filed=td + pd.Timedelta(days=lag),
                             value=float(rng.uniform(2e4, 2e6)), relation="officer" if k % 3 == 0 else "director", title=""))
    ins = pd.DataFrame(ins_rows)
    sic = pd.DataFrame({"ticker": tick, "sic": [f"{[2834, 3674, 6022, 7372][i % 4]}" for i in range(n_tickers)]})
    return P.Inputs(stocks, market, ev, ins, sic, None)


def _run_parity_module():
    spec = importlib.util.spec_from_file_location("_w7_run_parity", ROOT / "scripts" / "run_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_inputs(n_tickers: int, seed: int, since: str) -> tuple[P.Inputs, list[str]]:
    """The parity firewall's seeded real-cache slice (scripts/run_parity.load_real: column subsets only), with the full market frame
    set the fv/feeds builders expect."""
    inp, pick = _run_parity_module().load_real(n_tickers, seed, since)
    mc = inp.market["Close"]
    return P.Inputs(inp.stocks, {k: mc for k in ("Open", "High", "Low", "Close")}, inp.ev, inp.ins, inp.sic, None), pick


# =====================================================================================================================================
# the production builders, each as (Inputs, T) -> long frame indexed (date, ticker)
# =====================================================================================================================================
@dc.dataclass(frozen=True)
class Builder:
    name: str
    fn: Callable[[P.Inputs, pd.Timestamp], pd.DataFrame]
    sources: tuple[str, ...]                 # "module:qualname" of the code that computes the columns (for the locator)
    owner: str                               # who may change the source (C69 builder briefs)
    exclude: tuple[str, ...] = ()            # label/outcome columns: forward by definition, reported as LABEL, never audited
    planted: bool = False                    # the canary: MUST come out LEAK
    note: str = ""


def _long(frames: Mapping[str, pd.DataFrame], start=None) -> pd.DataFrame:
    cols = []
    for name, w in frames.items():
        w = w if start is None else w.loc[pd.Timestamp(start):]
        cols.append(w.stack(future_stack=True).rename(name).astype("float64"))
    if not cols:
        return pd.DataFrame()
    X = pd.concat(cols, axis=1)
    X.index.names = ["date", "ticker"]
    return X.dropna(how="all")


def _date_level(df: pd.DataFrame) -> pd.DataFrame:
    d = df.astype("float64").copy()
    d.index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(d.index), [KEY] * len(d)], names=["date", "ticker"])
    return d


def _start(inp: P.Inputs) -> str:
    return str(inp.dates[min(len(inp.dates) - 1, 260 if len(inp.dates) > 400 else 60)].date())


def b_features(mode: str):
    from engine import features as F

    def fn(inp, T, start):
        kw = {"default": {}, "relative": {"relative": True}, "split_invariant": {"relative": True, "tradable_rule": "split_invariant"}}[mode]
        X, _ = F.build(inp.stocks, inp.market, inp.ev, inp.ins, inp.sic, start=start, **kw)
        return X.astype("float64")
    return fn


def b_candles(inp, T, start):
    from engine import candles
    return _long(candles.build(inp.stocks), start)


def _fv_panel(inp, start):
    from engine import fv_pipeline as FV
    bars = {k: inp.stocks[k] for k in ("Open", "High", "Low", "Close", "Volume")}
    return FV.build_panel(bars, FV.FVConfig(), start=start)


def b_fv_X(inp, T, start):
    return _fv_panel(inp, start).X.astype("float64")


def b_fv_XR(inp, T, start):
    return _fv_panel(inp, start).XR.astype("float64")


def b_vol_lab_frame(inp, T, start):
    from engine.research import volatility_lab as VL
    sector = dict(zip(inp.sic["ticker"], inp.sic["sic"].astype(str).str[:2]))
    F = VL.frame_from_panel(_fv_panel(inp, start), VL.LabConfig(), sector=sector)
    return _numeric(F.drop(columns=[c for c in ("sector",) if c in F.columns]))


def _numeric(F: pd.DataFrame) -> pd.DataFrame:
    """float64 copy; datetime columns become days since the epoch (NaT -> NaN) so a date that moves is a diff like any other."""
    G = pd.DataFrame(index=F.index)
    for c in F.columns:
        s = F[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            v = s.to_numpy("datetime64[ns]").astype("int64").astype("float64") / 8.64e13
            G[c] = np.where(s.isna().to_numpy(), np.nan, v)
        else:
            G[c] = pd.to_numeric(s, errors="coerce").astype("float64")
    return G


def b_event_inputs(inp, T, start):
    from engine.research import volatility_lab as VL
    X = b_features("default")(inp, T, start)
    return VL.event_inputs(X.index, inp.ev, inp.ins).astype("float64")


def _pool(X: pd.DataFrame, sic: pd.DataFrame) -> pd.DataFrame:
    sec = dict(zip(sic["ticker"], sic["sic"].astype(str).str[:2]))
    t = X.index.get_level_values(1)
    return pd.DataFrame({"date": X.index.get_level_values(0), "sector": [sec.get(k, "na") for k in t]}, index=X.index)


def b_direction_features(inp, T, start):
    from engine import direction_features as DF
    X = b_features("relative")(inp, T, start)
    D = DF.add_derived(X)
    cols = [c for c in ("r5", "r20", "ear", "dist_52wh", "close_loc", "m_vix") if c in D.columns]
    R = DF.xs_rank(D, cols).add_prefix("xsr_")
    new = [c for c in D.columns if c not in X.columns]
    return pd.concat([D[new], R], axis=1).astype("float64")


def b_direction_lab(inp, T, start):
    from engine.research import direction_lab as DL
    X = b_features("relative")(inp, T, start)
    out = DL.derive_features(X, _pool(X, inp.sic))
    return _numeric(out[[c for c in out.columns if c not in X.columns]])


def b_market_proxy(inp, T, start):
    from engine.research import feeds
    return _date_level(feeds.market_proxy(inp.stocks)["Close"])


def b_regime_frame(inp, T, start):
    from engine import features as F
    C = inp.stocks["Close"]
    return _date_level(F.regime_frame(inp.market, C, np.log(C / C.shift(1))))


def b_episodes(inp, T, start):
    from engine.research import episodes as EPI
    g = EPI.build_grid({k: inp.stocks[k] for k in ("Open", "High", "Low", "Close", "Volume")}, EPI.EpisodeConfig())
    e = EPI.episode_frame(g)
    if e.empty:
        return pd.DataFrame()
    return _numeric(e.set_index(["date", "ticker"]).drop(columns=["ti", "nj"]))


def b_precursors(inp, T, start):
    from engine.research import episodes as EPI
    from engine.research import precursors as PR
    bars = {k: inp.stocks[k] for k in ("Open", "High", "Low", "Close", "Volume")}
    g = EPI.build_grid(bars, EPI.EpisodeConfig())
    sec = dict(zip(inp.sic["ticker"], inp.sic["sic"].astype(str).str[:2]))
    codes = pd.factorize(pd.Series([sec.get(t, "na") for t in g.tickers]))[0]
    ctx = PR.Ctx(g, sector_codes=np.asarray(codes, int))
    frames = {}
    for spec in PR.default_registry(audit=False).specs.values():
        base = np.asarray(spec.fn(ctx), dtype=np.float64)
        for lag in spec.lags:
            frames[f"{spec.name}@{lag}"] = pd.DataFrame(PR.shift_down(base, lag), index=g.dates, columns=g.tickers)
    return _long(frames, start)


def b_ohlc_structure(inp, T, start):
    from engine.research import multiscale as MS
    s = inp.stocks
    return MS.ohlc_day_structure(s["Open"], s["High"], s["Low"], s["Close"], T, dtype="float64")


def b_boundary(inp, T, start):
    from engine.learning import boundary as BD
    mret = np.log(inp.market["Close"]["SPY"]).diff()
    reg = pd.Series(np.where(inp.market["Close"]["SPY"] > inp.market["Close"]["SPY"].rolling(50, min_periods=20).mean(), "up", "down"),
                    index=mret.index)
    parts = [BD.shock_features(mret), BD.regime_transition_features(reg), BD.confidence_deterioration(mret)]
    return _date_level(pd.concat(parts, axis=1))


def b_ts_quintile(inp, T, start):
    from engine import candidates as CA
    vix = inp.market["Close"]["^VIX"]
    return _date_level(pd.DataFrame({"vix_ts_quintile": CA.ts_quintile_series(vix, 60).astype(float)}))


def b_planted(inp, T, start):
    """The canary. Honest columns must come out CLEAN, every leak_* column LEAK."""
    C = inp.stocks["Close"].loc[pd.Timestamp(start) - pd.Timedelta(days=120):]
    r1 = np.log(C / C.shift(1))
    r5 = np.log(C / C.shift(5))
    stack = r1.stack()
    lo, hi = r5.stack().quantile([0.01, 0.99])
    f = {
        "honest_r5": r5,
        "honest_xs_rank": r5.rank(axis=1, pct=True),
        "honest_expanding_z": (r1 - r1.expanding(20).mean()) / r1.expanding(20).std(),
        "leak_panel_zscore": (r1 - stack.mean()) / stack.std(),
        "leak_winsor_fullsample": r5.clip(lo, hi),
        "leak_ts_rank_fullhistory": r5.rank(pct=True),
        "leak_minmax_scaler": (r5 - r5.min()) / (r5.max() - r5.min()),
        "leak_bfill": r5.where(np.repeat((np.arange(len(r5)) % 10 == 0)[:, None], r5.shape[1], 1)).bfill(),
        "leak_centred_window": r1.rolling(5, center=True, min_periods=1).mean(),
        "leak_negative_shift": r1.shift(-1),
    }
    return _long(f, start)


PRODUCTION: tuple[Builder, ...] = (
    Builder("features.build[default]", b_features("default"), ("engine.features:build", "engine.features:_block",
            "engine.features:_insider_features", "engine.features:regime_frame", "engine.features:_event_calendar"), "F09",
            note="Live and parity path"),
    Builder("features.build[relative]", b_features("relative"), ("engine.features:build", "engine.features:_block"), "F09",
            note="replay / any-era path (price and dollar-volume percentile tradability)"),
    Builder("features.build[split_invariant]", b_features("split_invariant"), ("engine.features:build", "engine.features:_block",
            "engine.leak_audit:split_invariant_tradable"), "F09", note="Test (livesim) path"),
    Builder("features.regime_frame", b_regime_frame, ("engine.features:regime_frame",), "F09"),
    Builder("candles.build", b_candles, ("engine.candles:build", "engine.candles:_candle"), "F09"),
    Builder("fv_pipeline.build_panel.X", b_fv_X, ("engine.fv_pipeline:price_features", "engine.fv_pipeline:build_panel"), "F09"),
    Builder("fv_pipeline.rank_features", b_fv_XR, ("engine.fv_pipeline:rank_features",), "F09"),
    Builder("volatility_lab.frame_from_panel", b_vol_lab_frame, ("engine.research.volatility_lab:frame_from_panel",), "F09",
            exclude=("absmove", "touch", "tday", "up", "close", "end")),
    Builder("volatility_lab.event_inputs", b_event_inputs, ("engine.research.volatility_lab:event_inputs",), "R07 (volatility_lab)"),
    Builder("direction_features.add_derived+xs_rank", b_direction_features, ("engine.direction_features:add_derived",
            "engine.direction_features:xs_rank", "engine.direction_features:event_type"), "F09"),
    Builder("direction_lab.derive_features", b_direction_lab, ("engine.research.direction_lab:derive_features",
            "engine.research.direction_lab:_leave_one_out_mean"), "R08 (direction_lab)"),
    Builder("feeds.market_proxy", b_market_proxy, ("engine.research.feeds:market_proxy",), "W02 (feeds.py)"),
    Builder("episodes.episode_frame", b_episodes, ("engine.research.episodes:build_grid", "engine.research.episodes:episode_frame"),
            "R21 (episodes)"),
    Builder("precursors.default_registry", b_precursors, ("engine.research.precursors",), "R21 (precursors)"),
    Builder("multiscale.ohlc_day_structure", b_ohlc_structure, ("engine.research.multiscale:ohlc_day_structure",), "R16 (multiscale)"),
    Builder("learning.boundary", b_boundary, ("engine.learning.boundary:shock_features", "engine.learning.boundary:regime_transition_features",
            "engine.learning.boundary:confidence_deterioration"), "S-series (boundary)"),
    Builder("candidates.ts_quintile_series", b_ts_quintile, ("engine.candidates:ts_quintile_series",), "F09"),
)
CANARY = Builder("planted.canary", b_planted, ("scripts.feature_leak_audit:b_planted",), "F09", planted=True,
                 note="planted leaks; must be flagged")

# Builders covered by another truncation harness, named here so the inventory is complete and nothing is silently assumed clean.
COVERED_ELSEWHERE = {
    "analogs.fingerprints / macro (mac_*)": "engine.parity_suite future-invariance + macro publication audit (scripts/run_parity.py)",
    "live.todays_features": "engine.parity_suite live-vs-research",
    "PatternMiner.quantile_matrix context levels": "candidates.ts_quintile_series (audited here) + tests/test_patterns*",
    "learning.* per-week context (curator / trader_view)": "engine.learning.future_firewall + curator firewalls",
}
OUT_OF_SCOPE = (
    "back-adjusted prices encode later splits and dividends (a stored-data leak; truncation keeps the adjusted history). "
    "Mitigated on the Test path by split-invariant tradability; audited by engine.leak_audit.",
    "the price panel is survivor-only (delisted names missing): a universe leak no feature truncation can see; see scripts/survivorship_gate.py.",
    "feature SELECTION on the full sample (e.g. direction_lab.screen_feature_leakage runs over every pooled date) chooses columns, not "
    "values; it is a model-selection question for the walk-forward harnesses, listed as a finding, not audited here.",
)


# =====================================================================================================================================
# the audit
# =====================================================================================================================================
@dc.dataclass
class FeatureResult:
    builder: str
    feature: str
    status: str                              # CLEAN / LEAK / UNTESTED / LABEL
    max_abs: float = 0.0
    n_fail: int = 0
    n_nan_mismatch: int = 0
    n_finite: int = 0
    first_bad: str | None = None             # earliest (date, ticker) whose value moved
    cut: str | None = None                   # the cut that exposed it
    where: str = ""                          # offending operation file:line (LEAK only)
    reason: str = ""

    def as_dict(self) -> dict:
        return dc.asdict(self)


@dc.dataclass
class BuilderResult:
    builder: str
    owner: str
    status: str                              # OK / LEAK / ERROR / EMPTY
    cuts: list[str]
    features: list[FeatureResult]
    rows_leaked: int = 0                     # rows whose EXISTENCE at a date < T depends on data after T
    rows_boundary: int = 0                   # full-panel rows dated exactly T that the truncated panel cannot emit yet (not a leak)
    error: str = ""
    seconds: float = 0.0
    planted: bool = False
    note: str = ""

    @property
    def leaks(self) -> list[FeatureResult]:
        return [f for f in self.features if f.status == "LEAK"]

    def as_dict(self) -> dict:
        d = dc.asdict(self)
        d["features"] = [f.as_dict() for f in self.features]
        return d


def choose_cuts(dates: pd.DatetimeIndex, n: int, start) -> list[pd.Timestamp]:
    """`n` cut dates spread over the WHOLE history (a backfill or an ever-seen universe shows up at the start of a series, before any
    builder's first emitted date) plus one a few sessions after the builder start (warm-up edges are where expanding windows and
    min_periods misbehave). Never the final session (nothing would be removed). Deterministic."""
    d = pd.DatetimeIndex(dates)
    if len(d) < 4 or n <= 0:
        return []
    pos = {min(len(d) - 2, max(2, int(round(q * (len(d) - 1))))) for q in np.linspace(0.0, 0.9, n)}
    after = np.flatnonzero(d >= pd.Timestamp(start))
    if len(after) > 3:
        pos.add(min(len(d) - 2, int(after[2])))
    return [d[p] for p in sorted(pos)]


def _truncate(inp: P.Inputs, T: pd.Timestamp) -> P.Inputs:
    """engine.parity.Inputs.truncate: bars and market through T, filings accepted by the end of T, insider reports filed by T."""
    return inp.truncate(T)


def _upto(X: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Rows dated <= T; a frame without a (date, ticker) index is treated as no rows (a builder asked for nothing yet)."""
    if X is None or len(X) == 0 or not isinstance(X.index, pd.MultiIndex):
        return pd.DataFrame(columns=getattr(X, "columns", []))
    return X[X.index.get_level_values(0) <= T]


def _cmp(full: pd.DataFrame, part: pd.DataFrame, T: pd.Timestamp):
    a, b = _upto(full, T), _upto(part, T)
    rep = P.compare_panels(a, b, TOL, "truncation", [T]) if len(a) and len(b) else None
    return a, b, rep


def audit_builder(b: Builder, inp: P.Inputs, cuts: Sequence[pd.Timestamp], start: str | None = None) -> BuilderResult:
    """Full build once, one truncated build per cut, per-feature verdicts. `start` (the first emitted date) is fixed from the FULL inputs
    so the truncated build emits the same rows. A crash is an ERROR result, never a silent pass."""
    start = start or _start(inp)
    t0 = time.time()
    res = BuilderResult(b.name, b.owner, "OK", [str(pd.Timestamp(c).date()) for c in cuts], [], planted=b.planted, note=b.note)
    try:
        full = b.fn(inp, inp.dates[-1], start)
    except Exception as e:                                           # pragma: no cover - exercised by the error test
        res.status, res.error = "ERROR", f"{type(e).__name__}: {e}"
        res.seconds = round(time.time() - t0, 2)
        return res
    if full is None or len(full) == 0:
        res.status, res.error = "EMPTY", "builder produced no rows on these inputs: nothing was compared (never CLEAN)"
        res.seconds = round(time.time() - t0, 2)
        return res
    labels = [c for c in full.columns if c in b.exclude]
    cols = [c for c in full.columns if c not in b.exclude]
    acc = {c: FeatureResult(b.name, c, "UNTESTED") for c in cols}
    skipped: list[str] = []
    for c in labels:
        res.features.append(FeatureResult(b.name, c, "LABEL", reason="outcome column: forward by definition, never a model input"))
    for T in cuts:
        T = pd.Timestamp(T)
        try:
            part = b.fn(_truncate(inp, T), T, start)
        except Exception as e:
            if T < pd.Timestamp(start):                   # data ends before the builder's first date: nothing to emit, nothing to compare
                skipped.append(str(T.date()))
                continue
            res.status, res.error = "ERROR", f"truncated build at {T.date()} raised {type(e).__name__}: {e}"
            break
        a, bb, rep = _cmp(full[cols], part[[c for c in cols if c in part.columns]] if len(part) else part, T)
        if rep is None:
            if len(a) and not len(_upto(part, T)):        # the full panel has rows <= T that the truncated panel cannot emit
                on_t = int((a.index.get_level_values(0) >= T).sum())
                res.rows_boundary += on_t
                res.rows_leaked += len(a) - on_t
            continue
        for r in rep.missing_rows:
            if pd.Timestamp(r[0]) >= T:
                res.rows_boundary += 1
            else:
                res.rows_leaked += 1
        res.rows_leaked += len(rep.extra_rows)
        common = a.index.intersection(bb.index)
        for c in cols:
            fr = acc[c]
            if c not in bb.columns:
                if fr.status != "LEAK":
                    fr.status, fr.reason = "LEAK", f"column absent when the data ends at {T.date()}"
                    fr.cut = str(T.date())
                continue
            d = rep.per_feature.get(c)
            if d is None:
                continue
            x, y = a.loc[common, c].to_numpy(float), bb.loc[common, c].to_numpy(float)
            fr.n_finite += int((np.isfinite(x) & np.isfinite(y)).sum())
            bad = d.n_fail + d.n_nan_mismatch + d.n_inf_mismatch
            if bad:
                diff = np.where(np.isfinite(x) & np.isfinite(y), np.abs(x - y), np.where(np.isnan(x) == np.isnan(y), 0.0, np.inf))
                moved = np.flatnonzero(diff > TOL)
                first = common[moved[0]] if len(moved) else d.worst
                if fr.status != "LEAK" or (d.max_abs > fr.max_abs):
                    fr.cut, fr.first_bad = str(T.date()), str(tuple(str(v) for v in first)) if first is not None else None
                fr.status = "LEAK"
                fr.max_abs = max(fr.max_abs, float(d.max_abs if np.isfinite(d.max_abs) else 0.0))
                fr.n_fail += d.n_fail
                fr.n_nan_mismatch += d.n_nan_mismatch
    for c, fr in acc.items():
        if fr.status == "UNTESTED" and fr.n_finite > 0:
            fr.status = "CLEAN"
        elif fr.status == "UNTESTED":
            fr.reason = "no finite value was compared (all-NaN or no rows before any cut)"
        if fr.status == "LEAK" and fr.n_nan_mismatch and not fr.n_fail:
            fr.reason = fr.reason or "missingness (NaN vs value) changes with data after the row date"
        if fr.status == "LEAK":
            fr.where = locate(b, c)
        res.features.append(fr)
    if skipped and not res.error:
        res.note = (res.note + "; " if res.note else "") + f"cuts before the builder start not comparable: {', '.join(skipped)}"
    if res.status == "OK" and (any(f.status == "LEAK" for f in res.features) or res.rows_leaked):
        res.status = "LEAK"
    res.seconds = round(time.time() - t0, 2)
    return res


# =====================================================================================================================================
# locating the offending operation
# =====================================================================================================================================
SUSPECT_OPS: tuple[tuple[str, str], ...] = (
    (r"\.bfill\(|backfill|fillna\(method=['\"]bfill", "backfill"),
    (r"center\s*=\s*True", "centred window"),
    (r"\.shift\(\s*-", "negative shift"),
    (r"\.(mean|std|median|min|max|sum|var)\(\s*\)", "whole-sample statistic"),
    (r"\.quantile\(", "quantile over the frame"),
    (r"\.rank\((?![^)]*axis\s*=\s*1)(?![^)]*groupby)", "rank without axis=1 (whole-column rank)"),
    (r"\.fit\(|Scaler\(", "fitted transform"),
    (r"\.any\(\s*\)|\.all\(\s*\)", "any/all over the whole history"),
    (r"\.unique\(\)", "set of values over the whole history"),
)
SAFE_CONTEXT = re.compile(r"rolling|expanding|groupby|axis\s*=\s*1|ewm|level\s*=\s*0")


def _resolve(src: str):
    mod, _, qual = src.partition(":")
    if mod.startswith("scripts."):
        m = sys.modules.get(__name__)
    else:
        m = importlib.import_module(mod)
    obj = m
    for part in [p for p in qual.split(".") if p]:
        obj = getattr(obj, part)
    return obj


def _source_lines(src: str) -> tuple[str, int, list[str]]:
    obj = _resolve(src)
    lines, first = inspect.getsourcelines(obj)
    path = Path(inspect.getsourcefile(obj)).resolve()
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    return rel, max(first, 1), lines


def suspect_lines(src: str) -> list[tuple[str, int, str, str]]:
    """(file, line, operation, code) for every line of `src` that matches a future-reading pattern outside a rolling/expanding/per-date
    context. A static hint for the report, never a verdict (the truncation diff is the verdict)."""
    try:
        rel, first, lines = _source_lines(src)
    except (TypeError, OSError, AttributeError, ImportError):
        return []
    out, in_doc = [], False
    for i, ln in enumerate(lines):
        code = ln.split("#", 1)[0]
        n_triple = code.count('"""') + code.count("'''")
        if in_doc or n_triple:
            in_doc = in_doc ^ (n_triple % 2 == 1)
            continue                                      # docstring text describes operations, it does not run them
        if not code.strip():
            continue
        for pat, op in SUSPECT_OPS:
            if re.search(pat, code) and not (op in ("whole-sample statistic", "quantile over the frame") and SAFE_CONTEXT.search(code)):
                out.append((rel, first + i, op, code.strip()[:120]))
                break
    return out


def locate(b: Builder, feature: str) -> str:
    """file:line of the code that assigns `feature`, plus the suspicious operations nearest to it (the assignment line itself first). Lag
    suffixes (name@k) and derived prefixes are stripped so the defining line is found. A hint for the report; the diff is the verdict."""
    base = feature.split("@")[0]
    names = [base] + [base[len(p):] for p in ("xs_", "xsr_", "leak_") if base.startswith(p)]
    hits: list[tuple[str, int]] = []
    sus: list[tuple[str, int, str]] = []
    for src in b.sources:
        try:
            rel, first, lines = _source_lines(src)
        except (TypeError, OSError, AttributeError, ImportError):
            continue
        for i, ln in enumerate(lines):
            if any(re.search(r"[\"']" + re.escape(n) + r"[\"']", ln) for n in names):
                hits.append((rel, first + i))
        sus += [(r, n, op) for r, n, op, _ in suspect_lines(src)]
    if hits:
        f0, l0 = hits[0]
        sus.sort(key=lambda t: (t[0] != f0, abs(t[1] - l0)))
    parts = []
    if hits:
        parts.append("assigned at " + ", ".join(dict.fromkeys(f"{f}:{n}" for f, n in hits[:2])))
    if sus:
        parts.append("suspect " + "; ".join(f"{f}:{n} {op}" for f, n, op in sus[:2]))
    return " | ".join(parts) or "not located statically (see the first moved row)"


# =====================================================================================================================================
# running and reporting
# =====================================================================================================================================
@dc.dataclass
class AuditRun:
    source: str
    cuts: list[str]
    results: list[BuilderResult]
    canary_ok: bool
    started: str
    seconds: float = 0.0
    inputs: dict = dc.field(default_factory=dict)
    provenance: dict = dc.field(default_factory=dict)

    @property
    def production(self) -> list[BuilderResult]:
        return [r for r in self.results if not r.planted]

    @property
    def leaks(self) -> list[FeatureResult]:
        return [f for r in self.production for f in r.leaks]

    @property
    def errors(self) -> list[BuilderResult]:
        return [r for r in self.production if r.status in ("ERROR", "EMPTY")]

    @property
    def passed(self) -> bool:
        return self.canary_ok and not self.leaks and not any(r.rows_leaked for r in self.production) and not self.errors

    def counts(self) -> dict:
        c: dict = {}
        for r in self.production:
            for f in r.features:
                c[f.status] = c.get(f.status, 0) + 1
        return c

    def as_dict(self) -> dict:
        return {"source": self.source, "cuts": self.cuts, "canary_ok": self.canary_ok, "passed": self.passed, "counts": self.counts(),
                "started": self.started, "seconds": self.seconds, "inputs": self.inputs, "provenance": self.provenance,
                "results": [r.as_dict() for r in self.results]}


def canary_verdict(r: BuilderResult) -> tuple[bool, list[str]]:
    """The planted builder must flag every leak_* column and clear every honest_* column; anything else means the audit is blind."""
    why = []
    st = {f.feature: f.status for f in r.features}
    if r.status == "ERROR":
        return False, [f"canary crashed: {r.error}"]
    for c, s in st.items():
        if c.startswith("leak_") and s != "LEAK":
            why.append(f"planted {c} came out {s}")
        if c.startswith("honest_") and s != "CLEAN":
            why.append(f"honest {c} came out {s}")
    if not any(c.startswith("leak_") for c in st):
        why.append("canary produced no planted columns")
    return not why, why


def run_audit(inp: P.Inputs, builders: Sequence[Builder] = PRODUCTION, n_cuts: int = 4, source: str = "planted",
              canary: Builder | None = CANARY, log: Callable[[str], None] | None = None) -> AuditRun:
    t0 = time.time()
    start = _start(inp)
    cuts = choose_cuts(inp.dates, n_cuts, start)
    run = AuditRun(source, [str(c.date()) for c in cuts], [], canary_ok=canary is None,
                   started=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.inputs = {"sessions": len(inp.dates), "tickers": int(inp.stocks["Close"].shape[1]), "first": str(inp.dates[0].date()),
                  "last": str(inp.dates[-1].date()), "events": 0 if inp.ev is None else len(inp.ev),
                  "insider": 0 if inp.ins is None else len(inp.ins)}
    for b in ([canary] if canary is not None else []) + list(builders):
        r = audit_builder(b, inp, cuts, start)
        run.results.append(r)
        if b.planted:
            run.canary_ok, why = canary_verdict(r)
            r.error = r.error or "; ".join(why)
        if log:
            log(f"{b.name:42s} {r.status:6s} {len(r.leaks):3d} leak / {len(r.features):3d} cols  {r.seconds:6.1f}s {r.error[:90]}")
    run.seconds = round(time.time() - t0, 1)
    return run


def _fmt(x: float) -> str:
    return "" if not x else ("inf" if not np.isfinite(x) else f"{x:.3g}")


def render_report(runs: Sequence[AuditRun], before: Mapping[str, str] | None = None) -> str:
    """Markdown: verdict, per-builder summary, one row per feature per source, the known fixes (before/after) and what truncation
    cannot see. `before` maps 'builder/feature' -> the pre-fix status (from FIXED)."""
    before = dict(before or {})
    L = ["# Feature future-leak audit (truncation)", "",
         "C69 ledger section 5 (future-leak 2, data-quality 6); canon C56. Generated by `scripts/feature_leak_audit.py` - do not edit by hand.",
         "Method: every production frame builder is built on the full inputs and on the inputs truncated at each cut date T; every value "
         f"dated <= T must match (tolerance {TOL:g}, NaN vs value is a mismatch). The planted canary must be flagged or the run fails.", ""]
    for run in runs:
        L += [f"## Source: {run.source}", "",
              f"- verdict: **{'PASS' if run.passed else 'FAIL'}**; canary {'caught every planted leak' if run.canary_ok else 'NOT CAUGHT - audit blind'}",
              f"- inputs: {run.inputs}", f"- cuts: {', '.join(run.cuts)}", f"- feature verdicts: {run.counts()}",
              f"- code: {run.provenance.get('code_hash', '?')} commit {run.provenance.get('git_commit', '?')}; {run.seconds}s", "",
              "| builder | owner | status | features | LEAK | rows leaked | boundary rows | seconds | note |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in run.results:
            L.append(f"| {r.builder}{' (canary)' if r.planted else ''} | {r.owner} | {r.status} | {len(r.features)} | {len(r.leaks)} | "
                     f"{r.rows_leaked} | {r.rows_boundary} | {r.seconds} | {(r.error or r.note)[:80]} |")
        L += ["", "| builder | feature | status | max abs diff | cells moved | NaN flips | first moved row | cut | offending operation |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in run.results:
            for f in sorted(r.features, key=lambda f: (f.status != "LEAK", f.feature)):
                L.append(f"| {r.builder} | {f.feature} | {f.status} | {_fmt(f.max_abs)} | {f.n_fail or ''} | {f.n_nan_mismatch or ''} | "
                         f"{f.first_bad or ''} | {f.cut or ''} | {f.where or f.reason} |")
        L.append("")
    if before:
        L += ["## Before / after (fixed at the source)", "", "| builder / feature | before | after | fix | meaning changed? |", "|---|---|---|---|---|"]
        after = {f"{r.builder}/{f.feature}": f.status for run in runs for r in run.results for f in r.features}
        for k, (st, fix, meaning) in FIXED.items():
            L.append(f"| {k} | {st} | {after.get(k, 'not run')} | {fix} | {meaning} |")
        L.append("")
    L += ["## Covered by another truncation harness", ""] + [f"- {k}: {v}" for k, v in COVERED_ELSEWHERE.items()]
    L += ["", "## What truncation cannot see (out of scope, never counted CLEAN)", ""] + [f"- {s}" for s in OUT_OF_SCOPE]
    return "\n".join(L) + "\n"


# builder/feature -> (status before the fix, the fix, whether the feature's meaning changed). Filled from the audit's own findings.
FIXED: dict[str, tuple[str, str, str]] = {}


def free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("inf")


def wait_for_ram(need: float = MIN_FREE_GB, poll: int = 60, give_up_s: int = 1200, log=print) -> bool:
    t0 = time.time()
    while free_gb() < need:
        if time.time() - t0 > give_up_s:
            return False
        log(f"free RAM {free_gb():.2f} GB < {need} GB: waiting {poll}s")
        time.sleep(poll)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("planted", "real", "both"), default="planted")
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--since", default="2012-01-01")
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="builder names (substring match)")
    ap.add_argument("--out", default=str(K.STATE / "research" / "feature_leak"))
    ap.add_argument("--fail-on", choices=("leak", "never"), default="leak")
    ap.add_argument("--no-report", action="store_true", help="print only (CI)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    builders = [b for b in PRODUCTION if not a.only or any(s in b.name for s in a.only)]
    sources = ("planted", "real") if a.source == "both" else (a.source,)
    runs = []
    try:
        from engine import provenance
        for src in sources:
            if src == "real":
                if not wait_for_ram():
                    print(f"not enough free RAM (< {MIN_FREE_GB} GB) after 20 min: real-cache audit not run", flush=True)
                    return 3
                inp, pick = real_inputs(a.tickers, a.seed, a.since)
            else:
                inp, pick = planted_inputs(seed=a.seed), None
            cfg = {"source": src, "tickers": a.tickers, "since": a.since, "cuts": a.cuts, "only": a.only}
            stamp = provenance.stamp(cfg, a.seed)          # BEFORE the run: an edit made while it runs must show up as stale
            run = run_audit(inp, builders, a.cuts, src, log=lambda s: print(s, flush=True))
            run.provenance = dict(stamp, code_stale_at_end=bool(provenance.stale(stamp)))
            if pick is not None:
                run.inputs["sample"] = pick
            runs.append(run)
            print(f"{src}: {'PASS' if run.passed else 'FAIL'} canary_ok={run.canary_ok} {run.counts()} leaks="
                  f"{[(f.builder, f.feature) for f in run.leaks][:12]}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASH {type(e).__name__}: {e}", flush=True)
        return 2
    if not a.no_report:
        out.mkdir(parents=True, exist_ok=True)
        for run in runs:
            (out / f"results_{run.source}.json").write_text(json.dumps(run.as_dict(), indent=1, default=str), encoding="utf-8")
        prev = []
        for s in ("planted", "real"):
            p = out / f"results_{s}.json"
            if s not in sources and p.exists():
                prev.append(load_run(p))
        (out / "report.md").write_text(render_report(sorted(runs + prev, key=lambda r: r.source), FIXED), encoding="utf-8")
        print(f"report: {out / 'report.md'}", flush=True)
    ok = all(r.passed for r in runs)
    if not all(r.canary_ok for r in runs):
        return 1
    return 0 if ok or a.fail_on == "never" else 1


def load_run(path: Path) -> AuditRun:
    """Re-hydrate a saved run (so one report can hold the planted and the real-cache results from separate invocations)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    res = []
    for r in d["results"]:
        feats = [FeatureResult(**f) for f in r.pop("features")]
        res.append(BuilderResult(features=feats, **r))
    return AuditRun(d["source"], d["cuts"], res, d["canary_ok"], d["started"], d.get("seconds", 0.0), d.get("inputs", {}),
                    d.get("provenance", {}))


if __name__ == "__main__":
    sys.exit(main())
