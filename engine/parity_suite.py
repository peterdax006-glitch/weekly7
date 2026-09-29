"""Bible Phase 2 (parity firewall), all feature families. `engine.parity` is the harness; this module points it at
every place a model input is made, each through the REAL production code:

  features     engine.features.build           per-stock panel (returns, events, insider, market context)
  candles      engine.candles.build            multi-timeframe candles and small-print signals
  market       engine.features.regime_frame    the m_* context columns
  fingerprints engine.analogs.fingerprints     market-state rows the analog engine matches on
  macro        the mac_* / rate / credit columns of the fingerprints (publication-lagged FRED series)
  analogs      engine.analogs.Analogs.find     analog days, distances, uniqueness, prediction
  live path    engine.live.todays_features     what the trader actually sees, vs the research path

Every family gets the same battery: strict incremental-vs-batch parity, future-invariance (replace everything after t
with junk; the row at t must not move), and, where it applies, a live-window run. On top of that: universe invariance
(a stock's value may not depend on which other stocks were loaded - the insider leak of 28 Sep), gappy-data parity
(NaN bars, late listings, delistings), and a macro PUBLICATION audit. Truncation cannot see a missing publication lag
(an observation dated t is inside any window ending at t), so the audit recomputes each mac_* value from the raw series
and the lag table and checks that table against how late each release really is."""
from __future__ import annotations

import fnmatch
import tempfile
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from . import parity as P
from . import candles as _candles
from . import features as _features

KEY = "_"                              # ticker slot for market-wide (date-only) families


# ------------------------------------------------------------------ family builders: (Inputs, start) -> panel
def _panelise(frames: dict, start, keep=None) -> pd.DataFrame:
    """dict of wide (date x ticker) frames -> long float32 panel, dates >= start, all-NaN rows dropped."""
    st = pd.Timestamp(start)
    cols = []
    for name, w in frames.items():
        if keep is not None and name not in keep:
            continue
        cols.append(w.loc[st:].stack(future_stack=True).rename(name).astype("float32"))
    if not cols:
        return pd.DataFrame()
    X = pd.concat(cols, axis=1)
    X.index.names = ["date", "ticker"]
    return X.dropna(how="all")


def _date_panel(df: pd.DataFrame, start) -> pd.DataFrame:
    d = df.loc[pd.Timestamp(start):].astype("float32")
    d.index = pd.MultiIndex.from_arrays([d.index, [KEY] * len(d)], names=["date", "ticker"])
    return d


def features_builder(inp, start):
    return P.default_builder(inp, start)


def candles_builder(inp, start):
    return _panelise(_candles.build(inp.stocks), start)


def market_builder(inp, start):
    C = inp.stocks["Close"]
    return _date_panel(_features.regime_frame(inp.market, C, np.log(C / C.shift(1))), start)


@contextmanager
def analogs_env(inp):
    """Run engine.analogs against `inp` instead of the on-disk caches: a data.load shim (pre-2000 and index history
    absent, which analogs already handles) and a temp cache dir holding the sic and macro tables."""
    from . import analogs
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        inp.sic.to_parquet(cache / "sic.parquet")
        if inp.macro is not None:
            inp.macro.to_parquet(cache / "macro.parquet")

        def load(name):
            if name == "market":
                return inp.market
            if name == "stocks":
                return inp.stocks
            raise FileNotFoundError(name)
        with mock.patch.object(analogs, "data", types.SimpleNamespace(load=load)), \
                mock.patch.object(analogs, "K", types.SimpleNamespace(CACHE=cache)):
            yield analogs


def fingerprints_builder(inp, start):
    with analogs_env(inp) as an:
        F, _ = an.fingerprints()
    return _date_panel(F, start)


def macro_builder(inp, start):
    X = fingerprints_builder(inp, start)
    return X[[c for c in X.columns if c.startswith("mac_") or c in ("rate_chg_1y", "credit_chg_3m")]]


# name -> (builder, needs warmup sessions before the first comparable date, cross-sectional columns to ignore in the
# universe-invariance check, run in live-window mode?)
FAMILIES = {
    "features": (features_builder, 260, ("ind_mom*", "rel_ind*", "m_*"), True),
    "candles": (candles_builder, 30, (), True),
    "market": (market_builder, 260, ("m_breadth", "m_dispersion"), True),
    "fingerprints": (fingerprints_builder, 300, ("breadth", "dispersion", "sector_crowding", "top_sector_share_chg"),
                     False),          # analogs are research-only and use the whole history by design
    "macro": (macro_builder, 300, (), False),
}


# ------------------------------------------------------------------ generic checks
def future_invariance(builder, inp, fast, dates, seed=0, tol=P.GATE_TOL, name="") -> P.ParityReport:
    """For each date t: rebuild with everything after t replaced by junk and compare the row at t with `fast`.
    Catches any read of t+1 or later regardless of how it is disguised (shift(-1), full-sample statistics, centred
    windows, a join on a later date)."""
    parts = []
    for i, t in enumerate(dates):
        X = builder(inp.perturb_future(t, seed + i), t)
        parts.append(X[X.index.get_level_values(0) == pd.Timestamp(t)])
    got = pd.concat([p for p in parts if len(p)]).sort_index() if any(len(p) for p in parts) else pd.DataFrame()
    ref = fast[fast.index.get_level_values(0).isin(list(dates))]
    rep = P.compare_panels(ref, got, tol, "future-invariance", list(dates))
    rep.fingerprint = P.feature_code_fingerprint()
    return rep


def universe_invariance(builder, inp, as_of, frac=0.5, seed=0, ignore=(), tol=P.GATE_TOL) -> P.ParityReport:
    """A stock's per-stock features must not depend on which other stocks are loaded. Builds the row at `as_of` for
    the full universe and for a random `frac` subset and compares the shared rows. Columns matching `ignore`
    (cross-sectional by construction: industry and market breadth) are excluded and named in the report error-free."""
    cols = list(inp.stocks["Close"].columns)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(cols), max(2, int(len(cols) * frac)), replace=False).tolist())
    sub = inp.subset([cols[i] for i in pick])
    t = pd.Timestamp(as_of)
    a = builder(inp.truncate(t), t)
    b = builder(sub.truncate(t), t)
    keep = [c for c in a.columns if not any(fnmatch.fnmatch(c, g) for g in ignore)]
    if not len(a) or not len(b):
        return P.compare_panels(a, b, tol, "universe-invariance", [t])
    a = a[a.index.get_level_values(0) == t][keep]
    b = b[b.index.get_level_values(0) == t][[c for c in keep if c in b.columns]]
    a = a[a.index.isin(b.index)]                        # only stocks present in both universes
    return P.compare_panels(a, b, tol, "universe-invariance", [t])


def gappy_parity(builder, inp, warmup, n_dates=3, seed=0, tol=P.GATE_TOL) -> P.ParityReport:
    """Parity on data with holes: random NaN bars, a late listing and a delisting."""
    n = len(inp.dates)
    g = inp.with_gaps(seed=seed, frac=0.01, late_ipo=warmup // 2 + 40, delist=n - 25)
    r = P.run_parity(g, builder, None, n_dates, seed, warmup, tol, "strict")
    r.mode = "gappy"
    return r


# ------------------------------------------------------------------ macro publication audit
# Earliest a release can be public after the observation's date, in calendar days (conservative floors).
RELEASE_FLOOR_DAYS = {"monthly": 28, "weekly": 4, "daily": 1}


def cadence(s: pd.Series) -> str:
    gaps = s.dropna().index.to_series().diff().dt.days.dropna()
    if gaps.empty:
        return "daily"
    m = float(gaps.median())
    return "monthly" if m >= 25 else "weekly" if m >= 5 else "daily"


def audit_macro_publication(macro: pd.DataFrame, idx: pd.DatetimeIndex, lags: dict, dates, F: pd.DataFrame | None = None,
                            tol=P.GATE_TOL) -> dict:
    """Independent check that no macro value is used before it was public.
    lag_table : each series' lag (sessions) must cover its cadence's release floor (a monthly series with no
                declared lag, or a lag shorter than the floor, is used before publication).
    values    : for sampled dates, recompute mac_<series> observation-first (obs date -> first session it could be
                public = position + lag) and compare with the fingerprint column."""
    out = {"lag_table": [], "values": [], "passed": True}
    for c in macro.columns:
        s = macro[c].dropna()
        cad = cadence(s)
        floor_sessions = RELEASE_FLOOR_DAYS[cad] * 5 / 7
        lag = lags.get(c, 1)
        if lag < floor_sessions:
            out["lag_table"].append(f"{c}: {cad} series with lag {lag} sessions < {floor_sessions:.0f} needed"
                                    + ("" if c in lags else " (no LAG entry, default 1)"))
    if F is not None:
        for c in macro.columns:
            col = f"mac_{c}"
            if col not in F.columns:
                continue
            s = macro[c].dropna()
            pub = idx.searchsorted(s.index.values) + lags.get(c, 1)         # first session position it is public
            for t in dates:
                i = idx.get_loc(pd.Timestamp(t))
                ok = pub <= i
                want = s.iloc[np.flatnonzero(ok)[-1]] if ok.any() else np.nan
                got = F[col].iloc[i]
                if not ((np.isnan(want) and np.isnan(got)) or abs(want - got) <= tol * max(1.0, abs(want))):
                    out["values"].append(f"{col} @ {pd.Timestamp(t).date()}: used {got:.6g}, public value was {want:.6g}")
    out["passed"] = not out["lag_table"] and not out["values"]
    return out


# ------------------------------------------------------------------ analog find parity
def _find_vector(res) -> dict:
    """Flatten an Analogs.find() result to floats so it can be compared like a feature row."""
    if res is None:
        return {"none": 1.0}
    v = {"none": 0.0, "uniqueness": res["uniqueness"], "n_close": float(res["n_close"])}
    for k, x in res["prediction"].items():
        v[f"pred_{k}"] = x
    for j, r in res["analogs"].reset_index(drop=True).iterrows():
        v[f"date{j}"] = float(pd.Timestamp(r["date"]).toordinal())
        v[f"dist{j}"] = float(r["distance"])
    return v


def analog_find_parity(inp, dates, tol=P.GATE_TOL, k=5) -> P.ParityReport:
    """Analogs found for day t from the full history vs from fingerprints built ONLY from data through t. The chosen
    days, their distances, the uniqueness score and the outcome prediction must agree: an analog engine that has
    seen the future of t (or of its analogs beyond the `gap`) cannot reproduce itself on truncated data."""
    from .analogs import Analogs
    with analogs_env(inp) as an:
        F, O = an.fingerprints()
    full = Analogs(F, O)
    fast, strict = {}, {}
    for t in dates:
        t = pd.Timestamp(t)
        fast[(t, KEY)] = _find_vector(full.find(t, k))
        with analogs_env(inp.truncate(t)) as an:
            Ft, Ot = an.fingerprints()
        strict[(t, KEY)] = _find_vector(Analogs(Ft, Ot).find(t, k))
    mk = lambda d: pd.DataFrame.from_dict(d, orient="index").rename_axis(["date", "ticker"]) \
        if False else _dict_panel(d)
    rep = P.compare_panels(mk(fast), mk(strict), tol, "analog-find", list(dates))
    rep.fingerprint = P.feature_code_fingerprint()
    return rep


def _dict_panel(d: dict) -> pd.DataFrame:
    X = pd.DataFrame.from_dict(d, orient="index")
    X.index = pd.MultiIndex.from_tuples(list(d), names=["date", "ticker"])
    return X.astype("float64")


# ------------------------------------------------------------------ research path vs live path
def live_rows(inp) -> pd.DataFrame:
    """Run the ACTUAL `engine.live.todays_features` on `inp` (already truncated to the decision moment and, if wanted,
    to the live lookback). It reads events/insider/sic from K.CACHE, so those are pointed at a temp dir holding the
    truncated tables; nothing on disk is read or written."""
    from . import live
    from . import config as K
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        inp.sic.to_parquet(cache / "sic.parquet")
        (inp.ev if inp.ev is not None else pd.DataFrame({"kind": [], "accepted": [], "ticker": [], "form": []})
         ).to_parquet(cache / "events.parquet")
        if inp.ins is not None:
            inp.ins.to_parquet(cache / "insider.parquet")
        shim = types.SimpleNamespace(**{k: v for k, v in vars(K).items() if not k.startswith("__")})
        shim.CACHE = cache
        with mock.patch.object(live, "K", shim):
            X, _ = live.todays_features(inp.stocks, inp.market)
    return X


def live_vs_research(inp, dates, fast=None, tol=P.GATE_TOL, lookback_days=420) -> dict:
    """Two assertions per date. (1) same-inputs: live.todays_features vs the research builder on the SAME truncated
    frames must be identical (tolerance 1e-9): they are meant to be one code path. (2) research-vs-live: the live row
    on the lookback window vs the row from the full-history research panel `fast` at the gate tolerance."""
    same, vs_fast = [], []
    for t in dates:
        t = pd.Timestamp(t)
        win = inp.truncate(t, lookback_days)
        L = live_rows(win)
        R = P.strict_row(P.default_builder, win, t)
        same.append((L, R))
        if fast is not None:
            vs_fast.append((L, fast[fast.index.get_level_values(0) == t]))
    cat = lambda pairs, i: pd.concat([p[i] for p in pairs]).sort_index() if pairs else pd.DataFrame()
    out = {"same_inputs": P.compare_panels(cat(same, 0), cat(same, 1), 1e-9, "live-same-inputs", list(dates))}
    if fast is not None:
        out["research_vs_live"] = P.compare_panels(cat(vs_fast, 0), cat(vs_fast, 1), tol, "research-vs-live", list(dates))
    return out


# ------------------------------------------------------------------ the suite
@dataclass
class Check:
    family: str
    name: str
    passed: bool
    detail: str = ""
    report: dict | None = None
    status: str = "run"              # run | skipped


@dataclass
class SuiteReport:
    checks: list = field(default_factory=list)
    seed: int = 0
    tol: float = P.GATE_TOL
    fingerprint: str = ""

    @property
    def passed(self):
        return bool(self.checks) and all(c.passed for c in self.checks if c.status == "run")

    @property
    def failed(self):
        return [f"{c.family}/{c.name}" for c in self.checks if c.status == "run" and not c.passed]

    @property
    def skipped(self):
        return [f"{c.family}/{c.name}: {c.detail}" for c in self.checks if c.status == "skipped"]

    def summary(self) -> str:
        lines = [f"PARITY SUITE {'PASS' if self.passed else 'FAIL'}: {len(self.checks)} checks, "
                 f"{len(self.failed)} failed, {len(self.skipped)} skipped, tol={self.tol:g}"]
        for c in self.checks:
            tag = "skip" if c.status == "skipped" else ("ok  " if c.passed else "FAIL")
            lines.append(f"  [{tag}] {c.family:13s} {c.name:22s} {c.detail}")
        return "\n".join(lines)

    def to_dict(self):
        return {"passed": self.passed, "failed": self.failed, "skipped": self.skipped, "seed": self.seed, "tol": self.tol,
                "fingerprint": self.fingerprint,
                "checks": [{"family": c.family, "name": c.name, "passed": c.passed, "status": c.status,
                            "detail": c.detail, "report": c.report} for c in self.checks]}


def _pr(fam, name, rep: P.ParityReport) -> Check:
    return Check(fam, name, rep.passed, f"compared={rep.n_compared} max_abs={rep.max_abs:.2g}"
                 + (f" FAIL {rep.failing_features[:4]}" if rep.failing_features else "") + (f" {rep.error}" if rep.error else ""),
                 rep.to_dict())


def run_suite(inp: P.Inputs, families=None, n_dates=6, seed=0, tol=P.GATE_TOL, live=True, macro_lags=None,
              builders: dict | None = None) -> SuiteReport:
    """Every check for every family. `builders` overrides/extends FAMILIES (name -> builder) for planted-defect tests.
    A check that crashes is a failure with the exception recorded, never a silent pass."""
    from . import analogs as _an
    rep = SuiteReport(seed=seed, tol=tol, fingerprint=P.feature_code_fingerprint())
    reg = dict(FAMILIES)
    for n, b in (builders or {}).items():
        reg[n] = (b,) + reg.get(n, (None, 260, (), False))[1:]
    names = families or list(reg)

    def guard(fam, name, fn):
        try:
            r = fn()
            for c in (r if isinstance(r, list) else [r]):
                rep.checks.append(c)
        except Exception as e:
            rep.checks.append(Check(fam, name, False, f"{type(e).__name__}: {e}"))

    for fam in names:
        builder, warm, ignore, live_mode = reg[fam]
        n = len(inp.dates)
        w = min(warm, max(n - 30, 1))
        first = inp.dates[w]
        fast_holder = {}

        def fastp():
            if "x" not in fast_holder:
                fast_holder["x"] = builder(inp, first)
            return fast_holder["x"]

        def strict():
            r = P.run_parity(inp, builder, fastp(), n_dates, seed, w, tol, "strict")
            fast_holder["days"] = r.dates
            return _pr(fam, "strict-vs-batch", r)
        guard(fam, "strict-vs-batch", strict)
        if live_mode and live:
            guard(fam, "live-window", lambda: _pr(fam, "live-window",
                                                  P.run_parity(inp, builder, fastp(), max(n_dates // 2, 2), seed + 1, w, tol, "live")))
        guard(fam, "future-invariance", lambda: _pr(fam, "future-invariance", future_invariance(
            builder, inp, fastp(), fast_holder.get("days") or P.sample_dates(fastp().index.get_level_values(0).unique(), 3, seed),
            seed, tol)))
        if len(inp.stocks["Close"].columns) >= 4 and fam in ("features", "candles"):
            guard(fam, "universe-invariance", lambda: _pr(fam, "universe-invariance", universe_invariance(
                builder, inp, inp.dates[-1], 0.5, seed, ignore, tol)))
        guard(fam, "gappy-data", lambda: _pr(fam, "gappy-data", gappy_parity(builder, inp, w, max(n_dates // 3, 2), seed, tol)))

    if "macro" in names and inp.macro is not None and len(inp.macro.columns):
        def macro_audit():
            with analogs_env(inp) as an:
                F, _ = an.fingerprints()
            days = P.sample_dates(F.index, n_dates, seed, warmup=300)
            a = audit_macro_publication(inp.macro, F.index, macro_lags if macro_lags is not None else _an.LAG, days, F, tol)
            return Check("macro", "publication-lag", a["passed"], "; ".join((a["lag_table"] + a["values"])[:3]) or
                         f"{len(days)} dates x {inp.macro.shape[1]} series public-only", a)
        guard("macro", "publication-lag", macro_audit)
    if "fingerprints" in names:
        def find():
            days = P.sample_dates(inp.dates, min(n_dates, 4), seed, warmup=340)
            return _pr("analogs", "find-vs-truncated", analog_find_parity(inp, days, tol))
        guard("analogs", "find-vs-truncated", find)
    if live and "features" in names:
        def livepath():
            fast = fastp_features(inp, reg, seed)
            days = P.sample_dates(fast.index.get_level_values(0).unique(), max(n_dates // 3, 2), seed + 2)
            r = live_vs_research(inp, days, fast, tol)
            return [_pr("live-path", k, v) for k, v in r.items()]
        try:
            import engine.live  # noqa: F401
        except Exception as e:
            rep.checks.append(Check("live-path", "import", False, f"engine.live not importable: {e}", status="skipped"))
        else:
            guard("live-path", "live-vs-research", livepath)
    return rep


def fastp_features(inp, reg, seed):
    b, warm = reg["features"][0], reg["features"][1]
    return b(inp, inp.dates[min(warm, max(len(inp.dates) - 30, 1))])
