"""Data expansion adapters (Bible PHASE 27): sector ETF history, backup price source, delisted-company registry.

Every source passes the same gate (`validate_prices`): schema, dates, duplicates, missing data, point-in-time,
provenance. Network code lives only in `fetch_*` functions that adapters receive as an injected callable, so tests
use fakes and nothing here touches the network at import or under test.

Canon: "do not add data simply because it exists" -> `incremental_information` measures whether a new feature adds
out-of-sample rank information beyond the current set, against a seeded permutation null.

Long price format: columns date, ticker, open, high, low, close, volume."""
from __future__ import annotations

import hashlib, json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
XLRE_START, XLC_START = pd.Timestamp("2015-10-08"), pd.Timestamp("2018-06-19")      # first trading days
ETF_FIRST_DATE = {"XLRE": XLRE_START, "XLC": XLC_START}


# ------------------------------------------------------------------ validation
@dataclass
class ValidationReport:
    source: str
    rows_in: int = 0
    rows_out: int = 0
    errors: list = field(default_factory=list)       # fatal: frame refused
    warnings: list = field(default_factory=list)     # repaired or flagged
    dropped: dict = field(default_factory=dict)      # reason -> row count
    provenance: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"source": self.source, "ok": self.ok, "rows_in": self.rows_in, "rows_out": self.rows_out,
                "errors": self.errors, "warnings": self.warnings, "dropped": self.dropped,
                "provenance": self.provenance}


def frame_hash(df: pd.DataFrame) -> str:
    """Order-independent content hash (sorted by date/ticker) so a re-fetch of identical data hashes identically."""
    d = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:16]


def validate_prices(df: pd.DataFrame, source: str, as_of, fetched_at=None, first_date: dict | None = None,
                    max_gap_days: int = 6, max_daily_move: float = 0.6) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a long price frame. `as_of` is the latest date allowed to exist (point-in-time): rows
    dated after it are dropped and counted, never passed on. Returns (clean_frame, report); the clean frame is
    empty when the report has errors.

    Checks: required columns; numeric coercion; parseable dates; no weekend dates; date <= as_of; per-ticker
    date ranges not before `first_date[ticker]` (a fund cannot trade before it exists); exact duplicate rows
    collapsed, conflicting duplicates (same date+ticker, different values) are an ERROR; OHLC consistency
    (low <= open,close <= high, positive prices, non-negative volume); gaps longer than `max_gap_days` calendar days
    and single-day moves above `max_daily_move` are warnings; NaN close rows dropped."""
    rep = ValidationReport(source=source, rows_in=len(df))
    empty = pd.DataFrame(columns=PRICE_COLS)
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return empty, rep
    if df.empty:
        rep.errors.append("empty frame")
        return empty, rep
    d = df[PRICE_COLS].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in PRICE_COLS[2:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()

    def drop(mask, why):
        n = int(mask.sum())
        if n:
            rep.dropped[why] = rep.dropped.get(why, 0) + n
        return d[~mask]
    d = drop(d["date"].isna(), "unparseable_date")
    d = drop(d["ticker"].isin(["", "NAN", "NONE"]), "blank_ticker")
    d["date"] = d["date"].dt.normalize()
    d = drop(d["date"].dt.weekday >= 5, "weekend_date")
    d = drop(d["date"] > pd.Timestamp(as_of), "after_as_of")
    for t, first in (first_date or {}).items():
        d = drop((d["ticker"] == t) & (d["date"] < first), f"before_inception:{t}")
    d = drop(d["close"].isna(), "missing_close")
    d = drop((d[["open", "high", "low", "close"]] <= 0).any(axis=1), "non_positive_price")
    d = drop(d["volume"].fillna(0) < 0, "negative_volume")
    d = d.assign(volume=d["volume"].fillna(0.0))
    bad_ohlc = (d["high"] < d[["open", "close", "low"]].max(axis=1) - 1e-9) | \
               (d["low"] > d[["open", "close", "high"]].min(axis=1) + 1e-9)
    d = drop(bad_ohlc.fillna(False), "ohlc_inconsistent")
    key = d.duplicated(["date", "ticker"], keep=False)
    if key.any():
        exact = d.duplicated(keep="first")
        conflict = d[key].drop_duplicates().duplicated(["date", "ticker"], keep=False)
        if conflict.any():
            rep.errors.append(f"{int(conflict.sum())} rows conflict on the same (date, ticker)")
        d = drop(exact, "exact_duplicate")
    if rep.errors:
        return empty, rep
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    for t, g in d.groupby("ticker"):
        gaps = g["date"].diff().dt.days
        if (gaps > max_gap_days).any():
            rep.warnings.append(f"{t}: {int((gaps > max_gap_days).sum())} gaps > {max_gap_days} days "
                                f"(largest {int(gaps.max())})")
        mv = g["close"].pct_change().abs()
        if (mv > max_daily_move).any():
            rep.warnings.append(f"{t}: {int((mv > max_daily_move).sum())} moves > {max_daily_move:.0%} "
                                f"(possible unadjusted split)")
    from .isolation import nyse_holidays
    hol = {pd.Timestamp(h) for y in d["date"].dt.year.unique() for h in nyse_holidays(int(y))}
    n_hol = int(d["date"].isin(hol).sum())
    if n_hol:                                   # warn only: real closures (9/11, Sandy) differ from the rule set
        rep.warnings.append(f"{n_hol} rows dated on an NYSE holiday (vendor filler or calendar exception)")
    rep.rows_out = len(d)
    rep.provenance = {"source": source, "fetched_at": str(fetched_at) if fetched_at is not None else None,
                      "as_of": str(pd.Timestamp(as_of).date()), "content_hash": frame_hash(d) if len(d) else None,
                      "tickers": int(d["ticker"].nunique()),
                      "first": str(d["date"].min().date()) if len(d) else None,
                      "last": str(d["date"].max().date()) if len(d) else None}
    if not len(d):
        rep.errors.append("no rows survived validation")
    return d, rep


def to_wide(df: pd.DataFrame, field_: str = "close") -> pd.DataFrame:
    """Long -> wide (date x ticker), the shape engine.data stores."""
    return df.pivot(index="date", columns="ticker", values=field_).sort_index()


def wide_to_long(frames: dict) -> pd.DataFrame:
    """engine.data-style {field: wide frame} -> long frame."""
    parts = {f.lower(): frames[f].stack() for f in ("Open", "High", "Low", "Close", "Volume") if f in frames}
    out = pd.DataFrame(parts)
    out = out.dropna(how="all")                  # pre-listing / post-delisting NaN cells are not rows
    out.index.names = ["date", "ticker"]
    return out.reset_index()[PRICE_COLS]


# ------------------------------------------------------------------ adapters (network behind injected fetchers)
Fetcher = Callable[[list, str, str], pd.DataFrame]       # (tickers, start, end) -> long frame


def fetch_yfinance(tickers, start, end) -> pd.DataFrame:      # pragma: no cover - network
    """Default primary fetcher. Imported lazily so this module loads without yfinance or a network."""
    import yfinance as yf
    d = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    frames = {f: d[f] for f in ("Open", "High", "Low", "Close", "Volume")}
    return wide_to_long(frames)


class SectorETFAdapter:
    """Sector-ETF history through the shared validation gate. Knows each ETF's inception (XLRE, XLC) so a
    backfilled pre-inception row is refused rather than trusted."""

    def __init__(self, fetch: Fetcher = fetch_yfinance, tickers=SECTOR_ETFS):
        self.fetch, self.tickers = fetch, list(tickers)

    def load(self, start, end, as_of, fetched_at=None):
        if pd.Timestamp(end) > pd.Timestamp(as_of):
            end = as_of                                      # never ask for data past the as-of date
        raw = self.fetch(self.tickers, str(start), str(end))
        df, rep = validate_prices(raw, "sector_etf", as_of, fetched_at, first_date=ETF_FIRST_DATE)
        absent = [t for t in self.tickers if len(df) and t not in set(df["ticker"])]
        if absent:
            rep.warnings.append(f"tickers with no rows: {absent}")
        return df, rep


def sector_relative(df: pd.DataFrame, bench: pd.Series, window: int = 20) -> pd.DataFrame:
    """Point-in-time sector momentum relative to a benchmark close series (indexed by date). Uses only data up to
    each date (trailing window), so it is safe as a feature."""
    w = to_wide(df)
    b = bench.reindex(w.index).ffill()
    return (w.pct_change(window) - b.pct_change(window).values[:, None]).dropna(how="all")


# ------------------------------------------------------------------ backup source reconciliation
def reconcile(primary: pd.DataFrame, backup: pd.DataFrame, tol: float = 0.005, split_tol: float = 0.02) -> dict:
    """Compare backup to primary closes on the overlap. Returns
      coverage        share of primary (date,ticker) pairs the backup also has
      max_rel_diff / share_within_tol
      disagreements   rows where |backup/primary - 1| > tol, with the ratio
      split_suspects  tickers whose ratio is a stable non-1 constant near 2,3,4,5,10 or 1/n (adjustment mismatch)
      only_backup     rows the backup has that primary lacks (candidate gap fills)
    Both frames must already be validated."""
    m = primary[["date", "ticker", "close"]].merge(backup[["date", "ticker", "close"]], on=["date", "ticker"],
                                                   how="outer", suffixes=("_p", "_b"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    only_b = m[m["_merge"] == "right_only"][["date", "ticker", "close_b"]]
    n_p = int((m["_merge"] != "right_only").sum())
    out = {"coverage": len(both) / n_p if n_p else 0.0, "overlap": len(both), "only_backup": only_b,
           "max_rel_diff": 0.0, "share_within_tol": float("nan"), "disagreements": both.iloc[0:0],
           "split_suspects": []}
    if not len(both):
        return out
    both["ratio"] = both["close_b"] / both["close_p"]
    rel = (both["ratio"] - 1).abs()
    out["max_rel_diff"], out["share_within_tol"] = float(rel.max()), float((rel <= tol).mean())
    out["disagreements"] = both[rel > tol][["date", "ticker", "close_p", "close_b", "ratio"]].reset_index(drop=True)
    factors = np.array([2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10])
    for t, g in both.groupby("ticker"):
        r = g["ratio"]
        if len(g) >= 5 and abs(r.median() - 1) > tol and r.std() / r.median() < split_tol:
            if np.min(np.abs(factors / r.median() - 1)) < 0.05:
                out["split_suspects"].append(t)
    return out


def merge_with_backup(primary: pd.DataFrame, backup: pd.DataFrame, rec: dict | None = None,
                      max_fill_frac: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Primary wins. Backup rows fill (date,ticker) gaps only, for tickers not flagged as split suspects, and only
    if fills stay under `max_fill_frac` of the merged frame (a large fill means the primary is broken, not gappy:
    then refuse and report). Filled rows carry source='backup'."""
    rec = rec or reconcile(primary, backup)
    p = primary.assign(source="primary")
    keys = rec["only_backup"][["date", "ticker"]]
    fill = backup.merge(keys, on=["date", "ticker"])
    fill = fill[~fill["ticker"].isin(rec["split_suspects"])].assign(source="backup")
    info = {"filled": len(fill), "refused_split": sorted(set(rec["only_backup"]["ticker"]) &
                                                         set(rec["split_suspects"])), "applied": True}
    if len(fill) > max_fill_frac * (len(p) + len(fill)):
        info["applied"] = False
        return p, info
    return pd.concat([p, fill], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True), info


def parity(a: pd.DataFrame, b: pd.DataFrame, cols=("open", "high", "low", "close", "volume"), tol=1e-6) -> dict:
    """Exact-schema parity between two loads of the same source (e.g. cached vs re-fetched): same key set and
    values within tol. Returns mismatch counts per column."""
    m = a.merge(b, on=["date", "ticker"], how="outer", suffixes=("_a", "_b"), indicator=True)
    res = {"only_a": int((m["_merge"] == "left_only").sum()), "only_b": int((m["_merge"] == "right_only").sum())}
    both = m[m["_merge"] == "both"]
    for c in cols:
        x, y = both[f"{c}_a"], both[f"{c}_b"]
        res[c] = int((~np.isclose(x, y, rtol=tol, atol=tol)).sum())
    res["ok"] = all(v == 0 for v in res.values())
    return res


# ------------------------------------------------------------------ delisted-company registry
REASONS = {"bankruptcy", "acquired", "merger", "going_private", "exchange_removal", "voluntary", "other"}


class DelistedRegistry:
    """Point-in-time registry of delisted companies. Each record: ticker, delist_date (last trading day),
    announced (date the market could first know), reason, last_close, optional terminal_return (the return
    realised by holders after the last close, e.g. -1.0 for bankruptcy wipe-out, buyout premium otherwise).

    `alive_at(as_of)` and `delistings_known_at(as_of)` never reveal a delisting before it was announced, so a
    backtest cannot avoid a stock because it "knew" the future. Survivorship-free universes come from
    `universe_at`."""

    def __init__(self, records: list[dict] | None = None):
        self.rows: dict[str, list[dict]] = {}
        self.rejected: list[tuple[dict, str]] = []
        for r in records or []:
            self.add(r)

    def add(self, r: dict) -> bool:
        try:
            t = str(r["ticker"]).strip().upper()
            dd, an = pd.Timestamp(r["delist_date"]), pd.Timestamp(r.get("announced", r["delist_date"]))
        except (KeyError, ValueError, TypeError):
            self.rejected.append((r, "schema"))
            return False
        why = None
        if not t or pd.isna(dd) or pd.isna(an):
            why = "schema"
        elif an > dd:
            why = "announced after delist date"
        elif r.get("reason", "other") not in REASONS:
            why = "unknown reason"
        elif r.get("last_close") is not None and not (r["last_close"] > 0):
            why = "bad last_close"
        elif r.get("terminal_return") is not None and r["terminal_return"] < -1.0:
            why = "terminal return below -100%"
        elif any(x["delist_date"] == dd for x in self.rows.get(t, [])):
            why = "duplicate"
        if why:
            self.rejected.append((r, why))
            return False
        rec = {"ticker": t, "delist_date": dd, "announced": an, "reason": r.get("reason", "other"),
               "last_close": r.get("last_close"), "terminal_return": r.get("terminal_return"),
               "source": r.get("source", "unknown")}
        self.rows.setdefault(t, []).append(rec)          # tickers can be reused: keep every listing episode
        self.rows[t].sort(key=lambda x: x["delist_date"])
        return True

    def frame(self) -> pd.DataFrame:
        recs = [x for v in self.rows.values() for x in v]
        return pd.DataFrame(recs).sort_values(["delist_date", "ticker"]).reset_index(drop=True) if recs \
            else pd.DataFrame(columns=["ticker", "delist_date", "announced", "reason", "last_close",
                                       "terminal_return", "source"])

    def delistings_known_at(self, as_of) -> pd.DataFrame:
        f, a = self.frame(), pd.Timestamp(as_of)
        return f[f["announced"] <= a].reset_index(drop=True)

    def is_delisted(self, ticker: str, as_of) -> bool:
        """True only once the last trading day has passed AND it was announced by as_of."""
        a = pd.Timestamp(as_of)
        return any(x["delist_date"] < a and x["announced"] <= a for x in self.rows.get(ticker.upper(), []))

    def universe_at(self, listed: dict[str, tuple], as_of) -> list[str]:
        """listed: {ticker: (first_date, last_date_or_None)} from a price source. Returns tickers tradable at
        as_of: started on or before it, and not yet delisted. Delisted names remain in the universe until their
        delist_date - the survivorship-free set."""
        a = pd.Timestamp(as_of)
        out = []
        for t, (first, last) in listed.items():
            if pd.Timestamp(first) > a:
                continue
            if last is not None and pd.Timestamp(last) < a and not self.rows.get(t.upper()):
                continue                                   # stopped trading, unexplained: exclude, do not guess
            if self.is_delisted(t, a):
                continue
            out.append(t)
        return sorted(out)

    def terminal_return(self, ticker: str, as_of) -> float | None:
        """Realised return after the last close for a ticker delisted by as_of; None if unknown or still live."""
        for x in self.rows.get(ticker.upper(), []):
            if x["delist_date"] < pd.Timestamp(as_of) and x["announced"] <= pd.Timestamp(as_of):
                return x["terminal_return"]
        return None

    def survivorship_gap(self, listed: dict, start, end) -> dict:
        """How much of the window's true universe a survivors-only file would miss: names that delisted inside
        [start, end] and share of all names alive at start."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        f = self.frame()
        gone = f[(f["delist_date"] >= s) & (f["delist_date"] <= e)]
        alive0 = self.universe_at(listed, s)
        return {"delisted_in_window": sorted(gone["ticker"].unique()),
                "share_of_start_universe": len(set(gone["ticker"]) & set(alive0)) / max(1, len(alive0))}

    def save(self, path: Path):
        f = self.frame()
        for c in ("delist_date", "announced"):
            f[c] = f[c].dt.strftime("%Y-%m-%d")
        Path(path).write_text(json.dumps(f.to_dict("records"), indent=1, default=float))

    @classmethod
    def load(cls, path: Path) -> "DelistedRegistry":
        p = Path(path)
        return cls(json.loads(p.read_text()) if p.exists() else [])


# ------------------------------------------------------------------ incremental information value
def _rank_ic_by_date(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    return df.groupby(level=0).apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 4 else np.nan).dropna()


def _walk_forward_pred(X: pd.DataFrame, y: pd.Series, n_folds: int, ridge: float) -> pd.Series:
    """Expanding-window ridge (train strictly before test dates). Predictions for all but the first block."""
    dates = X.index.get_level_values(0).unique().sort_values()
    edges = np.linspace(0, len(dates), n_folds + 2).astype(int)
    out = []
    for i in range(1, n_folds + 1):
        tr_d, te_d = dates[:edges[i]], dates[edges[i]:edges[i + 1]]
        tr = X.index.get_level_values(0).isin(tr_d)
        te = X.index.get_level_values(0).isin(te_d)
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(), Xt.std().replace(0, 1)
        A = ((Xt - mu) / sd).fillna(0).values
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ (yt.values - yt.mean()))
        out.append(pd.Series((((X[te] - mu) / sd).fillna(0).values @ w), index=X.index[te]))
    return pd.concat(out)


def incremental_information(base: pd.DataFrame, new: pd.DataFrame, y: pd.Series, seed: int, n_perm: int = 50,
                            n_folds: int = 4, ridge: float = 10.0) -> dict:
    """Does `new` add out-of-sample rank information beyond `base`? Metric: mean daily rank IC of an expanding
    ridge fit, with vs without the new columns, on identical test rows. Null: shuffle the new columns' rows within
    each date (seeded), recompute the gain n_perm times. Reports gain, null mean/sd, and one-sided p. A source
    that does not beat its own shuffled version has not earned a place in the data set."""
    idx = base.index.intersection(new.index).intersection(y.dropna().index)
    b, n, yy = base.loc[idx], new.loc[idx], y.loc[idx]

    def gain(nw):
        both = pd.concat([b, nw], axis=1)
        p0, p1 = _walk_forward_pred(b, yy, n_folds, ridge), _walk_forward_pred(both, yy, n_folds, ridge)
        return _rank_ic_by_date(p1, yy).mean() - _rank_ic_by_date(p0, yy).mean()
    g = float(gain(n))
    rng = np.random.default_rng(seed)
    dates = idx.get_level_values(0)
    order = pd.Series(np.arange(len(idx)), index=idx).groupby(level=0)
    groups = [v.values for _, v in order]
    null = []
    for _ in range(n_perm):
        perm = np.arange(len(idx))
        for ix in groups:
            perm[ix] = rng.permutation(ix)
        null.append(float(gain(pd.DataFrame(n.values[perm], index=idx, columns=n.columns))))
    null = np.array(null)
    return {"gain": g, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
            "p_value": float((1 + (null >= g).sum()) / (1 + len(null))), "n_dates": int(dates.nunique()),
            "adds_value": bool(g > 0 and (1 + (null >= g).sum()) / (1 + len(null)) < 0.05)}


# ------------------------------------------------------------------ point-in-time frames for non-price datasets
def point_in_time_filter(df: pd.DataFrame, as_of, available_col: str = "available", event_col: str = "date"):
    """For datasets with a publication/availability date (filings, ratings): keep rows whose availability date is
    <= as_of. Rows whose availability precedes their event date are impossible and are dropped and counted.
    Returns (frame, {'future': n, 'impossible': n})."""
    for c in (available_col, event_col):
        if c not in df.columns:
            raise KeyError(f"missing column {c}")
    a, e = pd.to_datetime(df[available_col], errors="coerce"), pd.to_datetime(df[event_col], errors="coerce")
    impossible = (a < e) | a.isna() | e.isna()
    future = ~impossible & (a > pd.Timestamp(as_of))
    return df[~impossible & ~future].copy(), {"future": int(future.sum()), "impossible": int(impossible.sum())}


# ------------------------------------------------------------------ session calendar coverage
def expected_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] from the independent calendar in engine.isolation (weekdays minus holidays)."""
    from .isolation import nyse_holidays
    days = pd.bdate_range(start, end)
    hol = set()
    for y in range(days[0].year, days[-1].year + 1) if len(days) else []:
        hol |= {pd.Timestamp(d) for d in nyse_holidays(y)}
    return days[~days.isin(list(hol))]


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: first/last date, sessions observed vs expected in that span, share missing, longest run of
    missing sessions. Uses the NYSE calendar, so a holiday is not counted as missing."""
    rows = []
    for t, g in df.groupby("ticker"):
        exp = expected_sessions(g["date"].min(), g["date"].max())
        have = pd.DatetimeIndex(g["date"].unique())
        miss = exp[~exp.isin(have)]
        longest = 0
        if len(miss):
            pos = exp.get_indexer(miss)
            run = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
            longest = max(len(r) for r in run)
        rows.append({"ticker": t, "first": g["date"].min(), "last": g["date"].max(), "observed": len(have),
                     "expected": len(exp), "missing": len(miss), "missing_share": len(miss) / max(1, len(exp)),
                     "longest_missing_run": longest})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ------------------------------------------------------------------ data-quality mechanisms
SPLIT_RATIOS = np.array([2, 3, 4, 5, 10, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 1 / 20, 3 / 2, 2 / 3])


def detect_splits(close: pd.DataFrame, volume: pd.DataFrame | None = None, tol: float = 0.06) -> pd.DataFrame:
    """Suspected UNADJUSTED corporate actions in wide (date x ticker) frames: a one-day close ratio within `tol`
    of a common split ratio. With volume, also report the volume ratio the same day (a real split moves volume
    roughly opposite to price; a crash or a data error usually does not) and `supported` = volume moved the right way."""
    ratio = (close / close.shift(1)).stack()
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    nearest = SPLIT_RATIOS[np.abs(np.log(ratio.values[:, None] / SPLIT_RATIOS[None, :])).argmin(axis=1)]
    hit = np.abs(ratio.values / nearest - 1) < tol
    out = pd.DataFrame({"close_ratio": ratio.values[hit], "split_ratio": nearest[hit]}, index=ratio.index[hit])
    out.index.names = ["date", "ticker"]
    if volume is not None and len(out):
        v = (volume / volume.shift(1).replace(0, np.nan)).stack()
        out["vol_ratio"] = v.reindex(out.index).replace([0, np.inf], np.nan).values
        out["supported"] = (np.log(out["vol_ratio"]) * np.log(out["split_ratio"]) < 0).fillna(False).values
    return out.reset_index()


def stale_runs(df: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Flat-lined feeds: runs of >= min_run identical closes, or of zero volume, per ticker. Dead data that still
    looks like data (a delisted name forward-filled, a vendor repeating the last print)."""
    rows = []
    for t, g in df.sort_values("date").groupby("ticker"):
        for kind, s in (("same_close", g["close"].diff().eq(0)), ("zero_volume", g["volume"].eq(0))):
            grp = (~s).cumsum()
            for _, run in s[s].groupby(grp[s]):
                n = len(run) + (1 if kind == "same_close" else 0)
                if n >= min_run:
                    rows.append({"ticker": t, "kind": kind, "start": g.loc[run.index[0], "date"],
                                 "end": g.loc[run.index[-1], "date"], "length": n})
    return pd.DataFrame(rows, columns=["ticker", "kind", "start", "end", "length"])


ERAS = [("pre-1990", None, "1989-12-31"), ("1990s", "1990-01-01", "1999-12-31"), ("2000s", "2000-01-01", "2009-12-31"),
        ("2010s", "2010-01-01", "2019-12-31"), ("2020s", "2020-01-01", None)]


def era_of(ts: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=ts.index, dtype=object)
    for name, lo, hi in ERAS:
        m = (ts >= pd.Timestamp(lo)) if lo else pd.Series(True, index=ts.index)
        if hi:
            m &= ts <= pd.Timestamp(hi)
        out[m] = name
    return out


def defect_census(df: pd.DataFrame, big_move: float = 0.4) -> pd.DataFrame:
    """Vectorised per-era count of every defect class on a RAW long frame (before cleaning), plus rows and defect
    rate. This is how a source's quality is compared across eras: old data is usually dirtier."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values(["ticker", "date"])
    d["era"] = era_of(d["date"])
    ohlc_bad = (d["high"] < d[["open", "close", "low"]].max(axis=1) - 1e-9) | \
               (d["low"] > d[["open", "close", "high"]].min(axis=1) + 1e-9)
    move = d.groupby("ticker")["close"].pct_change().abs()
    flags = pd.DataFrame({"era": d["era"], "weekend": d["date"].dt.weekday >= 5, "missing_close": d["close"].isna(),
                          "non_positive": (d[["open", "high", "low", "close"]] <= 0).any(axis=1),
                          "ohlc_inconsistent": ohlc_bad, "duplicate_key": d.duplicated(["date", "ticker"]),
                          "zero_volume": d["volume"].fillna(0).eq(0), "big_move": move > big_move})
    g = flags.groupby("era")
    out = g.sum().astype(int)
    out.insert(0, "rows", g.size())
    out["defect_rate"] = out.drop(columns=["rows", "zero_volume"]).sum(axis=1) / out["rows"]
    return out.reindex([e[0] for e in ERAS]).dropna(how="all")


def unexplained_stops(close: pd.DataFrame, as_of, registry: DelistedRegistry, min_lag_sessions: int = 10) -> pd.DataFrame:
    """Tickers that stopped trading more than `min_lag_sessions` sessions before as_of and that the registry does not
    explain: candidates for the delisted registry. Two kinds: `no_data` (series ends) and `flat_tail` (the cache
    forward-filled a dead name: an unbroken run of identical closes at the end counts as the stop, dated at the
    start of the run). `guess` is a heuristic label only (sub-$1 last close after a >50% final-month fall ->
    bankruptcy-like; otherwise acquired-like); it is never written into the registry."""
    cut = pd.Timestamp(as_of)
    rows = []
    for t in close.columns:
        s = close[t].dropna()
        s = s[s.index <= cut]
        if s.empty or registry.rows.get(str(t).upper()):
            continue
        kind, end = "no_data", len(s) - 1
        moved = (s.diff().fillna(1) != 0).values
        run = int(np.argmax(moved[::-1])) if moved.any() else len(s)      # trailing identical closes after the last move
        if run >= min_lag_sessions and len(s) > run:
            kind, end = "flat_tail", len(s) - 1 - run
        lag = len(expected_sessions(s.index[end], cut)) - 1
        if lag <= min_lag_sessions:
            continue
        fall = s.iloc[end] / s.iloc[max(0, end - 20)] - 1
        rows.append({"ticker": t, "kind": kind, "last_date": s.index[end], "last_close": float(s.iloc[end]),
                     "lag_sessions": lag, "final_month_return": float(fall),
                     "guess": "bankruptcy-like" if s.iloc[end] < 1 and fall < -0.5 else "acquired-like"})
    return pd.DataFrame(rows, columns=["ticker", "kind", "last_date", "last_close", "lag_sessions",
                                       "final_month_return", "guess"]).sort_values("last_date").reset_index(drop=True)


# ------------------------------------------------------------------ provenance ledger
class ProvenanceLedger:
    """Append-only JSONL of every ingest (source, hash, counts, as_of), hash-chained so an edited or deleted line is
    detected by `verify`. A source cannot be silently swapped for another that happens to have the same name."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    @staticmethod
    def _digest(rec: dict, prev: str) -> str:
        body = json.dumps({k: v for k, v in rec.items() if k != "chain"}, sort_keys=True, default=str)
        return hashlib.sha256((prev + body).encode()).hexdigest()[:24]

    def append(self, report: ValidationReport, note: str = "") -> dict:
        if not report.ok:
            raise ValueError(f"refusing to record a failed ingest for {report.source}: {report.errors}")
        prev = self._lines()[-1]["chain"] if self._lines() else "genesis"
        rec = {**report.provenance, "rows_in": report.rows_in, "rows_out": report.rows_out,
               "dropped": report.dropped, "note": note}
        rec["chain"] = self._digest(rec, prev)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def verify(self) -> dict:
        prev, bad = "genesis", []
        recs = self._lines()
        for i, r in enumerate(recs):
            if self._digest(r, prev) != r.get("chain"):
                bad.append(i)
            prev = r.get("chain", "")
        return {"ok": not bad, "entries": len(recs), "broken_at": bad}

    def latest(self, source: str) -> dict | None:
        hits = [r for r in self._lines() if r.get("source") == source]
        return hits[-1] if hits else None


# ------------------------------------------------------------------ EDGAR-derived delisting events (pure logic; the
# network fetch lives in scripts/fetch_delisted.py so tests never touch it)
F25_FORMS = ("25", "25-NSE", "25-NSE/A")
F15_PREFIX = ("15-12B", "15-12G", "15-15D", "15F")
ANNUAL_PREFIX = ("10-K", "10K")
DELIST_LAG_DAYS = 10          # exchange delistings take effect ~10 days after the Form 25 is filed (Rule 12d2-2)


def parse_form_idx(text: str) -> pd.DataFrame:
    """Parse an EDGAR full-index form.idx / form.gz body into columns form, company, cik, date, path. Column
    offsets come from the header line, so a shifted layout is read correctly; unparseable rows are dropped."""
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("Form Type") and "CIK" in l), None)
    if hdr is None:
        return pd.DataFrame(columns=["form", "company", "cik", "date", "path"])
    h = lines[hdr]
    c1, c2, c3, c4 = h.index("Company Name"), h.index("CIK"), h.index("Date Filed"), h.index("File Name")
    rows = []
    for l in lines[hdr + 2:]:
        if len(l) < c4:
            continue
        rows.append((l[:c1].strip(), l[c1:c2].strip(), l[c2:c3].strip(), l[c3:c4].strip(), l[c4:].strip()))
    df = pd.DataFrame(rows, columns=["form", "company", "cik", "date", "path"])
    df["cik"] = pd.to_numeric(df["cik"].where(df["cik"].str.fullmatch(r"\d{1,10}")), errors="coerce")  # "inf" is not a CIK
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["cik", "date"]).astype({"cik": "int64"}).reset_index(drop=True)


def keep_relevant_forms(df: pd.DataFrame) -> pd.DataFrame:
    """Only the forms the delisting logic needs: Form 25 family, Form 15 family, annual reports."""
    f = df["form"]
    m = f.isin(F25_FORMS) | f.str.startswith(F15_PREFIX) | f.str.startswith(ANNUAL_PREFIX)
    return df[m].reset_index(drop=True)


def classify_delistings(events: pd.DataFrame, as_of, window_days: int = 400) -> pd.DataFrame:
    """One row per CIK that ever filed a Form 25, evaluated at its LAST Form 25:
      continuing     an annual report is filed 60+ days after the last exit filing (a transfer between exchanges or
                     a partial-class delisting: the company is still a reporting issuer)
      deregistered   a Form 15 within [-30, +window_days] days and no later annual report
      went_dark      no Form 15, no later annual report, and the Form 25 is older than window_days
      too_recent     Form 25 within window_days of as_of with nothing yet to judge it by
    `terminal` is True for deregistered and went_dark. Point-in-time: only events dated <= as_of are read. Columns:
    cik, company, f25_date, f15_date, last_10k_after, status, terminal, announced, delist_date (Form 25 + 10 days,
    an approximation stated here on purpose)."""
    a = pd.Timestamp(as_of)
    e = events[events["date"] <= a]
    f25 = e[e["form"].isin(F25_FORMS)].sort_values("date").groupby("cik").tail(1)
    f15 = e[e["form"].str.startswith(F15_PREFIX)]
    ann = e[e["form"].str.startswith(ANNUAL_PREFIX)]
    rows = []
    for r in f25.itertuples():
        d15 = f15[(f15["cik"] == r.cik) & (f15["date"] >= r.date - pd.Timedelta(days=30)) &
                  (f15["date"] <= r.date + pd.Timedelta(days=window_days))]["date"]
        ref = d15.max() if len(d15) else r.date            # an annual report 60+ days after the LAST exit filing
        later = ann[(ann["cik"] == r.cik) & (ann["date"] > ref + pd.Timedelta(days=60))]["date"]
        if r.date > a - pd.Timedelta(days=window_days) and not len(d15) and not len(later):
            status = "too_recent"
        elif len(later):
            status = "continuing"
        elif len(d15):
            status = "deregistered"
        elif r.date > a - pd.Timedelta(days=window_days):
            status = "too_recent"
        else:
            status = "went_dark"
        rows.append({"cik": r.cik, "company": r.company, "f25_date": r.date,
                     "f15_date": d15.min() if len(d15) else pd.NaT,
                     "last_10k_after": later.max() if len(later) else pd.NaT, "status": status,
                     "terminal": status in ("deregistered", "went_dark"), "announced": r.date,
                     "delist_date": r.date + pd.Timedelta(days=DELIST_LAG_DAYS)})
    return pd.DataFrame(rows, columns=["cik", "company", "f25_date", "f15_date", "last_10k_after", "status",
                                       "terminal", "announced", "delist_date"])


_SUFFIX = ("INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "LLC", "LP", "PLC", "HOLDINGS", "GROUP",
           "THE", "DE", "NEW", "TRUST")


def clean_name(n: str) -> str:
    import re
    n = re.sub(r"[^A-Z0-9 ]", " ", str(n).upper().replace("&", " AND "))
    toks = [t for t in n.split() if t not in _SUFFIX]
    return " ".join(toks)


def name_similarity(a: str, b: str) -> float:
    """0..1 similarity of two company names after cleaning; vendor names are truncated ('Lehman Brothers Holdings
    Capita'), so the shorter is compared with the same-length prefix of the longer."""
    from difflib import SequenceMatcher
    x, y = clean_name(a), clean_name(b)
    if not x or not y:
        return 0.0
    n = min(len(x), len(y))
    return SequenceMatcher(None, x[:n], y[:n]).ratio() * (0.5 + 0.5 * n / max(len(x), len(y)))


def pick_symbol(company: str, quotes: list[dict], min_sim: float = 0.75) -> dict | None:
    """Choose the search hit whose name best matches `company` (equities only). None when nothing clears min_sim:
    an unresolved ticker is honest, a wrong ticker would attach another company's prices."""
    best, score = None, 0.0
    for q in quotes:
        if q.get("quoteType", "EQUITY") not in ("EQUITY", None):
            continue
        s = name_similarity(company, q.get("longname") or q.get("shortname") or "")
        if s > score:
            best, score = q, s
    return {"symbol": best["symbol"], "sim": score, "name": best.get("shortname")} if best and score >= min_sim else None


def to_registry(classified: pd.DataFrame, tickers: dict[int, str], prices_last: dict[str, float] | None = None) -> DelistedRegistry:
    """Terminal delistings with a resolved ticker -> DelistedRegistry (source 'edgar_form25')."""
    reg = DelistedRegistry()
    for r in classified[classified["terminal"]].itertuples():
        t = tickers.get(r.cik)
        if not t:
            continue
        reg.add({"ticker": t, "delist_date": r.delist_date, "announced": r.announced, "reason": "other",
                 "last_close": (prices_last or {}).get(t), "source": f"edgar_form25:{r.status}"})
    return reg


def attrition_coverage(classified: pd.DataFrame, resolved: set[int], priced: set[int], alive_by_year: pd.Series,
                       lo: float = 0.03, hi: float = 0.06) -> pd.DataFrame:
    """Per delisting year: terminal Form-25 events found, how many got a ticker, how many got prices, and the
    expected count band (lo..hi x names alive that year, the usual 3-6%/yr attrition). `found_vs_low/high` are
    events/expected; a value far below 1 means the source is missing whole classes of exits, far above means the
    Form 25 family is counting non-equity securities."""
    t = classified[classified["terminal"]].copy()
    t["year"] = t["delist_date"].dt.year
    g = t.groupby("year")
    out = pd.DataFrame({"terminal_events": g.size(), "resolved": g.apply(lambda x: int(x["cik"].isin(resolved).sum())),
                        "priced": g.apply(lambda x: int(x["cik"].isin(priced).sum()))})
    out["alive"] = alive_by_year.reindex(out.index)
    out["expected_low"], out["expected_high"] = out["alive"] * lo, out["alive"] * hi
    out["found_vs_low"], out["found_vs_high"] = out["terminal_events"] / out["expected_low"], \
        out["terminal_events"] / out["expected_high"]
    out["priced_vs_low"] = out["priced"] / out["expected_low"]
    return out
