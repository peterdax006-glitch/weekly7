"""Bible Phase 2: feature/parity firewall (canon: no look-ahead; the fast path must reproduce the strict live path).

Three independent checks, all fail-closed:
  1. PARITY  - the fast (batch) panel is compared with a STRICT recomputation for randomly sampled dates. Strict means
     every input (bars, filings, insider reports) is truncated to what existed at that date, the feature code is run
     from scratch, and only that date's row is kept. A feature that peeks forward, or whose value depends on how much
     history/universe happens to be loaded, cannot reproduce itself and is caught. The same harness runs the LIVE
     window (lookback frames, as `live.todays_features` sees them) against the research batch.
  2. CONTRACTS - schema, dtype, NaN policy, finite-ness, range and per-date-constancy (market columns) per feature.
  3. DRIFT - reference vs recent distribution per feature (PSI, KS, NaN-share shift).

Gate: max difference <= 1e-4. `require_parity` raises ParityFailure (abort the dependent experiment) and refuses a
looser tolerance - the threshold is never negotiable, only the code that fails it."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as _features

GATE_TOL = 1e-4                 # Bible Phase 2 gate; require_parity refuses anything looser
ET = "America/New_York"


class ParityFailure(RuntimeError):
    """Raised to abort any experiment that depends on a feature panel which failed the firewall."""


# ------------------------------------------------------------------ inputs and point-in-time truncation
@dataclass
class Inputs:
    """Everything the feature code reads. `ev`: filings with `accepted` (tz-aware), `kind`, `ticker`, `form`;
    `ins`: insider trades with `filed`, `tdate`, `symbol`, ...; either may be None/empty."""
    stocks: dict
    market: dict
    ev: pd.DataFrame
    ins: pd.DataFrame | None
    sic: pd.DataFrame
    macro: pd.DataFrame | None = None       # raw FRED observations dated by their OBSERVATION date (data/cache/macro.parquet)

    def truncate(self, as_of, lookback_days=None) -> "Inputs":
        """What was knowable at the close of `as_of`: bars through that date and filings accepted by the end of it
        (the feature code itself defers filings after 15:30 ET). Optional `lookback_days` mimics live.fresh_frames."""
        t = pd.Timestamp(as_of).normalize()
        cut = None if lookback_days is None else t - pd.Timedelta(days=lookback_days)
        cutf = lambda v: v.loc[cut:t] if cut is not None else v.loc[:t]
        stocks = {k: cutf(v) for k, v in self.stocks.items()}
        market = {k: cutf(v) for k, v in self.market.items()}
        ev = self.ev
        if ev is not None and len(ev):
            end = t.tz_localize(ET) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            acc = ev["accepted"]
            acc = acc.dt.tz_localize(ET) if acc.dt.tz is None else acc
            ev = ev[acc <= end]
        ins = self.ins
        if ins is not None and len(ins):
            ins = ins[ins["filed"] < t + pd.Timedelta(days=1)]
        macro = None if self.macro is None else self.macro.loc[:t]
        return Inputs(stocks, market, ev, ins, self.sic, macro)

    def subset(self, tickers) -> "Inputs":
        """Same data restricted to `tickers` (universe-invariance checks)."""
        keep = [t for t in self.stocks["Close"].columns if t in set(tickers)]
        ev = self.ev if self.ev is None or not len(self.ev) else self.ev[self.ev["ticker"].isin(keep)]
        ins = self.ins if self.ins is None or not len(self.ins) else self.ins[self.ins["symbol"].isin(keep)]
        return Inputs({k: v[keep] for k, v in self.stocks.items()}, self.market, ev, ins,
                      self.sic[self.sic["ticker"].isin(keep)], self.macro)

    def perturb_future(self, as_of, seed=0) -> "Inputs":
        """Copy in which everything AFTER `as_of` is replaced by junk (bars rescaled by random factors, macro
        randomised, later filings re-labelled, later insider values scaled). A feature row for `as_of` computed from
        this copy must equal the one from the real data; if not, that feature read the future."""
        rng = np.random.default_rng(seed)
        t = pd.Timestamp(as_of).normalize()
        fut = self.dates > t

        def junk(v, sd, mask):
            w = v.astype("float64")
            w.loc[mask] = w.loc[mask].to_numpy() * np.exp(rng.normal(0, sd, (int(mask.sum()), w.shape[1])))
            return w.astype(v.dtypes)
        stocks = {k: junk(v, 0.25, fut) for k, v in self.stocks.items()}
        # keep High/Low consistent-ish so range-based features stay finite
        arr = [stocks[k].to_numpy("float64") for k in ("Open", "High", "Low", "Close")]
        for k, f in (("High", np.maximum.reduce), ("Low", np.minimum.reduce)):
            stocks[k] = pd.DataFrame(f(arr), index=self.dates, columns=stocks[k].columns).astype(stocks[k].dtypes)
        market = {k: junk(v, 0.25, v.index > t) for k, v in self.market.items()}
        ev = self.ev
        if ev is not None and len(ev):
            ev = ev.copy()
            end = t.tz_localize(ET) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            acc = ev["accepted"].dt.tz_localize(ET) if ev["accepted"].dt.tz is None else ev["accepted"]
            later = (acc > end).to_numpy()
            kinds = np.array(sorted(ev["kind"].unique()))
            ev.loc[later, "kind"] = rng.choice(kinds, int(later.sum()))
        ins = self.ins
        if ins is not None and len(ins):
            ins = ins.copy()
            later = (ins["filed"] >= t + pd.Timedelta(days=1)).to_numpy()
            ins.loc[later, "value"] = ins.loc[later, "value"] * 7.0
        macro = self.macro
        if macro is not None and len(macro):
            macro = junk(macro, 0.5, macro.index > t)
        return Inputs(stocks, market, ev, ins, self.sic, macro)

    def with_gaps(self, seed=0, frac=0.01, late_ipo=None, delist=None) -> "Inputs":
        """Copy with missing-data patterns real caches have: random NaN bars (all fields on the same cell), one ticker
        that lists late (NaN before position `late_ipo`) and one that delists (NaN from position `delist`)."""
        rng = np.random.default_rng(seed)
        C = self.stocks["Close"]
        hole = rng.random(C.shape) < frac
        if late_ipo is not None:
            hole[:late_ipo, 0] = True
        if delist is not None:
            hole[delist:, -1] = True
        stocks = {k: v.mask(hole) for k, v in self.stocks.items()}
        return Inputs(stocks, self.market, self.ev, self.ins, self.sic, self.macro)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.stocks["Close"].index


def default_builder(inp: Inputs, start) -> pd.DataFrame:
    """The production feature code, `engine.features.build`, as a (Inputs, start) -> panel callable."""
    X, _ = _features.build(inp.stocks, inp.market, inp.ev, inp.ins, inp.sic, start=str(pd.Timestamp(start).date()))
    return X


def feature_code_fingerprint(module=_features) -> str:
    """Hash of the feature module's source, so a cached parity pass is void the moment the code changes."""
    return hashlib.sha256(inspect.getsource(module).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ sampling
def sample_dates(dates, n=8, seed=0, warmup=0, must_include_last=True) -> list[pd.Timestamp]:
    """Seeded random sample of decision dates after `warmup` sessions (features need history). The last date is always
    included: it is the one the live path actually trades on."""
    dates = pd.DatetimeIndex(dates)
    pool = dates[warmup:]
    if len(pool) == 0:
        return []
    rng = np.random.default_rng(seed)
    k = min(n, len(pool))
    pick = set(rng.choice(len(pool), size=k, replace=False).tolist())
    if must_include_last:
        pick.add(len(pool) - 1)
        if len(pick) > k:                                    # keep the budget: drop a random non-last pick
            pick.discard(int(rng.choice([i for i in pick if i != len(pool) - 1])))
    return [pool[i] for i in sorted(pick)]


# ------------------------------------------------------------------ recomputation
def strict_row(builder, inp: Inputs, as_of, lookback_days=None) -> pd.DataFrame:
    """Strict recomputation: truncate inputs to `as_of`, run the feature code from scratch, keep only that date."""
    t = pd.Timestamp(as_of).normalize()
    X = builder(inp.truncate(t, lookback_days), t)
    if X is None or len(X) == 0:
        return X.iloc[0:0] if X is not None else pd.DataFrame()
    d = X.index.get_level_values(0)
    return X[d == t]


def incremental_panel(builder, inp: Inputs, dates, lookback_days=None) -> pd.DataFrame:
    """Day-by-day (strict) panel over `dates`, to be compared with the batch panel."""
    parts = [strict_row(builder, inp, d, lookback_days) for d in dates]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


# ------------------------------------------------------------------ exact comparison
@dataclass
class FeatureDiff:
    name: str
    n_compared: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    n_fail: int = 0             # elements where BOTH abs and rel error exceed tol
    n_nan_mismatch: int = 0     # NaN in exactly one of the two
    n_inf_mismatch: int = 0
    worst: tuple | None = None  # (date, ticker) of the largest absolute error

    @property
    def ok(self):
        return self.n_fail == 0 and self.n_nan_mismatch == 0 and self.n_inf_mismatch == 0


@dataclass
class ParityReport:
    tol: float
    mode: str
    dates: list
    fast_rows: int
    strict_rows: int
    missing_rows: list = field(default_factory=list)      # in fast but not strict: [(date, ticker)]
    extra_rows: list = field(default_factory=list)        # in strict but not fast
    missing_cols: list = field(default_factory=list)      # in fast panel, absent from strict
    extra_cols: list = field(default_factory=list)
    per_feature: dict = field(default_factory=dict)
    per_date_max: dict = field(default_factory=dict)
    fingerprint: str = ""
    error: str = ""

    @property
    def max_abs(self):
        return max((d.max_abs for d in self.per_feature.values()), default=0.0)

    @property
    def max_rel(self):
        return max((d.max_rel for d in self.per_feature.values()), default=0.0)

    @property
    def n_compared(self):
        return sum(d.n_compared for d in self.per_feature.values())

    @property
    def failing_features(self):
        return sorted(k for k, d in self.per_feature.items() if not d.ok)

    @property
    def passed(self):
        """Fail closed: nothing compared, a missing row/column or any single failing element is a failure."""
        return (not self.error and self.n_compared > 0 and not self.failing_features and not self.missing_rows
                and not self.extra_rows and not self.missing_cols and not self.extra_cols)

    def summary(self) -> str:
        head = (f"PARITY {'PASS' if self.passed else 'FAIL'} [{self.mode}] tol={self.tol:g} dates={len(self.dates)} "
                f"compared={self.n_compared} max_abs={self.max_abs:.3g} max_rel={self.max_rel:.3g}")
        lines = [head]
        if self.error:
            lines.append(f"  error: {self.error}")
        if self.missing_cols or self.extra_cols:
            lines.append(f"  columns only in fast: {self.missing_cols}; only in strict: {self.extra_cols}")
        if self.missing_rows or self.extra_rows:
            lines.append(f"  rows only in fast: {len(self.missing_rows)}; only in strict: {len(self.extra_rows)}")
        for name in self.failing_features:
            d = self.per_feature[name]
            lines.append(f"  {name}: fail={d.n_fail} nan_mismatch={d.n_nan_mismatch} inf_mismatch={d.n_inf_mismatch} "
                         f"max_abs={d.max_abs:.3g} worst={d.worst}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dates"] = [str(x)[:10] for x in self.dates]
        d["per_date_max"] = {str(k)[:10]: v for k, v in self.per_date_max.items()}
        d["missing_rows"] = [[str(a)[:10], b] for a, b in self.missing_rows[:50]]
        d["extra_rows"] = [[str(a)[:10], b] for a, b in self.extra_rows[:50]]
        for v in d["per_feature"].values():
            v["worst"] = None if v["worst"] is None else [str(v["worst"][0])[:10], v["worst"][1]]
        d.update(passed=self.passed, max_abs=self.max_abs, max_rel=self.max_rel, n_compared=self.n_compared)
        return d


def compare_panels(fast: pd.DataFrame, strict: pd.DataFrame, tol=GATE_TOL, mode="strict", dates=None) -> ParityReport:
    """Exact comparison over the union of rows. Per element: match if |a-b|<=tol or |a-b|/max(|a|,|b|)<=tol (the
    relative arm only matters for large magnitudes such as days_since_earn or log_dv). NaN vs value is always a
    mismatch, whatever the value; so is a row present in only one panel (missing-data mismatch)."""
    dates = list(dates) if dates is not None else sorted(set(fast.index.get_level_values(0))) if len(fast) else []
    rep = ParityReport(tol=tol, mode=mode, dates=dates, fast_rows=len(fast), strict_rows=len(strict))
    if len(fast) == 0 or len(strict) == 0:
        rep.error = f"empty panel (fast={len(fast)}, strict={len(strict)}): nothing to compare"
        return rep
    rep.missing_cols = sorted(set(fast.columns) - set(strict.columns))
    rep.extra_cols = sorted(set(strict.columns) - set(fast.columns))
    fi, si = fast.index, strict.index
    rep.missing_rows = list(fi.difference(si))
    rep.extra_rows = list(si.difference(fi))
    common = fi.intersection(si)
    cols = [c for c in fast.columns if c in strict.columns]
    a = fast.loc[common, cols].astype("float64")
    b = strict.loc[common, cols].astype("float64")
    day_max = {}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        d = FeatureDiff(c)
        nx, ny = np.isnan(x), np.isnan(y)
        d.n_nan_mismatch = int((nx != ny).sum())
        fx, fy = np.isinf(x), np.isinf(y)
        d.n_inf_mismatch = int(((fx | fy) & ~((x == y) & (fx | fy))).sum())
        ok = ~(nx | ny | fx | fy)
        d.n_compared = int(ok.sum() + (nx & ny).sum())
        if ok.any():
            err = np.abs(x[ok] - y[ok])
            rel = err / np.maximum(np.maximum(np.abs(x[ok]), np.abs(y[ok])), 1e-300)
            rel = np.where(err == 0, 0.0, rel)
            d.max_abs, d.max_rel = float(err.max()), float(rel.max())
            d.n_fail = int(((err > tol) & (rel > tol)).sum())
            if d.max_abs > 0:
                d.worst = common[np.flatnonzero(ok)[int(err.argmax())]]
            dts = common.get_level_values(0)[ok]
            for t, e in pd.Series(err, index=dts).groupby(level=0).max().items():
                day_max[t] = max(day_max.get(t, 0.0), float(e))
        rep.per_feature[c] = d
    rep.per_date_max = dict(sorted(day_max.items()))
    return rep


def run_parity(inp: Inputs, builder=default_builder, fast: pd.DataFrame | None = None, n_dates=8, seed=0, warmup=260,
               tol=GATE_TOL, mode="strict", live_lookback_days=None) -> ParityReport:
    """Random-date parity. `fast` is the batch panel (built here from the full inputs when not supplied).
    mode='strict': truncated-to-date recomputation over the same history. mode='live': the same, but on the lookback
    window the live path loads (`live_lookback_days`, default 420 calendar days like live.fresh_frames)."""
    if mode not in ("strict", "live"):
        raise ValueError(f"mode must be 'strict' or 'live', got {mode!r}")
    lb = (live_lookback_days or 420) if mode == "live" else None
    try:
        first = inp.dates[min(warmup, len(inp.dates) - 1)] if len(inp.dates) else pd.Timestamp("1970-01-01")
        if fast is None:
            fast = builder(inp, first)
        avail = pd.DatetimeIndex(sorted(set(fast.index.get_level_values(0)))) if len(fast) else pd.DatetimeIndex([])
        days = sample_dates(avail, n_dates, seed)
        strict = incremental_panel(builder, inp, days, lb)
        rep = compare_panels(fast[fast.index.get_level_values(0).isin(days)], strict, tol, mode, days)
    except Exception as e:                                 # a crash is a failed gate, never a silent pass
        rep = ParityReport(tol=tol, mode=mode, dates=[], fast_rows=0, strict_rows=0, error=f"{type(e).__name__}: {e}")
    rep.fingerprint = feature_code_fingerprint()
    return rep


def require_parity(report: ParityReport, tol=GATE_TOL, label="experiment"):
    """Abort a dependent experiment unless parity passed at the Bible tolerance. Refuses a loosened `tol`."""
    if tol > GATE_TOL:
        raise ValueError(f"parity tolerance {tol:g} is looser than the Bible gate {GATE_TOL:g}; fix the code, not the threshold")
    if report.tol > GATE_TOL:
        raise ParityFailure(f"{label} aborted: parity was measured at a loosened tolerance {report.tol:g}")
    if not report.passed:
        raise ParityFailure(f"{label} aborted:\n{report.summary()}")
    return report


def save_report(report: ParityReport, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=1, default=str), encoding="utf-8")
    return p


def cached_pass(path, tol=GATE_TOL) -> bool:
    """True only if a saved report passed, at the gate tolerance, for the CURRENT feature code."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(d.get("passed")) and d.get("tol", 1) <= min(tol, GATE_TOL) and d.get("fingerprint") == feature_code_fingerprint()


# ------------------------------------------------------------------ contracts
@dataclass
class FeatureContract:
    name: str
    dtype: str = "float32"
    max_nan_frac: float = 1.0
    lo: float = -np.inf
    hi: float = np.inf
    per_date_constant: bool = False        # market-context (m_*) columns are one value per date

    def to_dict(self):
        d = asdict(self)
        d["lo"], d["hi"] = (None if not np.isfinite(v) else v for v in (self.lo, self.hi))
        return d


def infer_contracts(ref: pd.DataFrame, slack=0.5, nan_slack=0.02) -> dict[str, FeatureContract]:
    """Learn a contract per feature from a reference panel: observed dtype, range widened by `slack` x range, and a NaN
    ceiling of 1.5x the observed share (+ `nan_slack`). Run once on the accepted research panel and store the result."""
    out = {}
    dates = ref.index.get_level_values(0)
    for c in ref.columns:
        s = ref[c].astype("float64").replace([np.inf, -np.inf], np.nan)
        lo, hi = (s.min(), s.max()) if s.notna().any() else (0.0, 0.0)
        span = hi - lo
        out[c] = FeatureContract(c, str(ref[c].dtype), min(1.0, float(ref[c].isna().mean()) * 1.5 + nan_slack),
                                 float(lo - slack * span), float(hi + slack * span),
                                 per_date_constant=bool(c.startswith("m_") and
                                                        (ref[c].groupby(dates).nunique(dropna=False) <= 1).all()))
    return out


def check_contracts(X: pd.DataFrame, contracts: dict[str, FeatureContract]) -> list[str]:
    """Return human-readable violations (empty list = clean). Checks index shape, columns, dtype, NaN share,
    infinities, range and per-date constancy."""
    v = []
    if not isinstance(X.index, pd.MultiIndex) or list(X.index.names) != ["date", "ticker"]:
        return [f"index must be MultiIndex named ['date','ticker'], got {list(X.index.names)}"]
    if X.index.has_duplicates:
        v.append(f"{int(X.index.duplicated().sum())} duplicate (date, ticker) rows")
    if not X.index.is_monotonic_increasing:
        v.append("index not sorted")
    for c in sorted(set(contracts) - set(X.columns)):
        v.append(f"{c}: contracted feature missing from panel")
    for c in sorted(set(X.columns) - set(contracts)):
        v.append(f"{c}: feature has no contract")
    dates = X.index.get_level_values(0)
    for c, k in contracts.items():
        if c not in X.columns:
            continue
        s = X[c]
        if str(s.dtype) != k.dtype:
            v.append(f"{c}: dtype {s.dtype} != {k.dtype}")
            if not pd.api.types.is_numeric_dtype(s):
                continue
        f = s.astype("float64")
        n = max(len(f), 1)
        ninf = int(np.isinf(f).sum())
        if ninf:
            v.append(f"{c}: {ninf} infinite values")
        nf = float(f.isna().sum()) / n
        if nf > k.max_nan_frac:
            v.append(f"{c}: NaN share {nf:.3f} > {k.max_nan_frac:.3f}")
        g = f[np.isfinite(f)]
        if len(g) and (g.min() < k.lo or g.max() > k.hi):
            v.append(f"{c}: range [{g.min():.4g}, {g.max():.4g}] outside [{k.lo:.4g}, {k.hi:.4g}]")
        if k.per_date_constant and (f.groupby(dates).nunique(dropna=False) > 1).any():
            v.append(f"{c}: not constant within a date")
    return v


# ------------------------------------------------------------------ drift
def psi(ref: np.ndarray, cur: np.ndarray, bins=10) -> float:
    """Population stability index with quantile bins from the reference. Degenerate reference (constant) -> 0 if the
    current sample sits on the same value, else a large sentinel (a constant feature that moved is drift)."""
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(cur == ref[0]) else 10.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(((q - p) * np.log(q / p)).sum())


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, psi_warn=0.10, psi_alarm=0.25, nan_shift=0.10, max_n=50_000,
                 seed=0) -> pd.DataFrame:
    """Per-feature drift, worst first. status: ok / warn (PSI>=warn) / alarm (PSI>=alarm or NaN share moved by more than
    `nan_shift`) / missing (feature absent from one panel)."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    rows = []
    for c in sorted(set(ref.columns) | set(cur.columns)):
        if c not in ref.columns or c not in cur.columns:
            rows.append({"feature": c, "psi": np.nan, "ks": np.nan, "nan_ref": np.nan, "nan_cur": np.nan,
                         "mean_shift_sd": np.nan, "status": "missing"})
            continue
        a = ref[c].to_numpy("float64")
        b = cur[c].to_numpy("float64")
        af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(af) > max_n:
            af = rng.choice(af, max_n, replace=False)
        if len(bf) > max_n:
            bf = rng.choice(bf, max_n, replace=False)
        p = psi(af, bf)
        ks = float(ks_2samp(af, bf).statistic) if len(af) and len(bf) else np.nan
        na, nb = float(np.isnan(a).mean()), float(np.isnan(b).mean())
        sd = float(af.std()) if len(af) else 0.0
        shift = float((bf.mean() - af.mean()) / sd) if len(af) and len(bf) and sd > 0 else np.nan
        status = "ok"
        if not np.isfinite(p) or abs(nb - na) > nan_shift or p >= psi_alarm:
            status = "alarm"
        elif p >= psi_warn:
            status = "warn"
        rows.append({"feature": c, "psi": p, "ks": ks, "nan_ref": na, "nan_cur": nb, "mean_shift_sd": shift,
                     "status": status})
    out = pd.DataFrame(rows, columns=["feature", "psi", "ks", "nan_ref", "nan_cur", "mean_shift_sd", "status"])
    return out.sort_values("psi", ascending=False, na_position="first").reset_index(drop=True)


# ------------------------------------------------------------------ one-call firewall
def firewall(inp: Inputs, builder=default_builder, contracts=None, n_dates=8, seed=0, warmup=260,
             tol=GATE_TOL, live=True, ref_frac=0.5) -> dict:
    """Run everything: strict parity, live-window parity, contracts on the batch panel (inferred from its first
    `ref_frac` when none supplied, then checked on the rest) and a drift table between the two halves. Returns
    {'strict','live','contract_violations','drift','passed'}; contracts/drift are advisory, parity decides `passed`."""
    first = inp.dates[min(warmup, len(inp.dates) - 1)]
    fast = builder(inp, first)
    out = {"strict": run_parity(inp, builder, fast, n_dates, seed, warmup, tol, "strict")}
    out["live"] = run_parity(inp, builder, fast, n_dates, seed + 1, warmup, tol, "live") if live else None
    days = fast.index.get_level_values(0)
    cut = days.unique().sort_values()
    cut = cut[int(len(cut) * ref_frac)] if len(cut) else None
    if cut is not None:
        ref, cur = fast[days < cut], fast[days >= cut]
        ct = contracts or infer_contracts(ref)
        out["contract_violations"] = check_contracts(cur if contracts is None else fast, ct)
        out["drift"] = drift_report(ref, cur)
    else:
        out["contract_violations"], out["drift"] = ["empty panel"], pd.DataFrame()
    out["passed"] = out["strict"].passed and (out["live"] is None or out["live"].passed)
    return out
