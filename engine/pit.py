"""Bible PHASE 1 - point-in-time data firewall (canon: no look-ahead; decide at a close, fill at the NEXT open; a fence
violation is an exception, never a warning; anything that changes when the future changes FAILS CLOSED).

Layers (each answers one Bible clause):
  1.1 date integrity   Calendar, Wide/Record sources with effective + availability dates and publication lags,
                       restatement vintages, PITStore.validate(), label_close_dates / verify_training_rows /
                       verify_feature_availability
  1.2 time fence       Guard: an object bound to one `as_of`; every request beyond it raises LookAheadError, every
                       access (allowed or denied) lands in a hash-chained AuditLog; only named live fields pass
  1.3 future scramble  future_invariance (a feature fn, truncate vs real future vs scrambled future),
                       future_scramble / future_scramble_store (a whole pipeline), implausible_ic (a feature that
                       "knows" the label); ScrambleReport.require() raises FailClosed
  1.4 fill audit       Executor (only fills at the strict next session's open, logged) and audit_fills (post-hoc
                       proof: decision after close, fill at next open, no same-close price, no next-day inputs)
Survivorship: Listings (membership as of a date, delistings only known once they happen) and survivorship_report.

Conventions: `as_of` is a decision CLOSE (date granularity, normalised). A daily bar dated d is available at as_of >= d.
Frames are wide (date x ticker, one per field) for prices, long records for events / insider / fundamentals / macro.
Python cannot stop deliberate reflection into a Guard's private state; the fence stops honest mistakes and every
access is provable from the log, which is what the scramble tests then verify end to end.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class PITError(Exception):
    """Base class for every firewall violation."""


class LookAheadError(PITError):
    """Something at or after a timestamp beyond `as_of` was requested or found."""


class LabelLeakError(PITError):
    """A training row's label had not closed when the prediction was made."""


class IntegrityError(PITError):
    """A source violates date integrity (missing / impossible dates, ambiguous vintages)."""


class SurvivorshipError(PITError):
    """The panel or membership list is survivor-biased or internally impossible."""


class FillTimingError(PITError):
    """A decision or fill breaks decide-at-close / fill-at-next-open."""


class FailClosed(PITError):
    """A future-scramble or invariance test found a dependence on the future."""


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


# --------------------------------------------------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------------------------------------------------
class Calendar:
    """Session calendar. Given real sessions, weekday gaps become holidays; beyond the covered span it falls back to
    plain weekdays. Without sessions it is a Mon-Fri calendar."""

    def __init__(self, sessions=None):
        if sessions is None or len(sessions) == 0:
            self.sessions = None
            self._hol = np.array([], dtype="datetime64[D]")
            return
        s = pd.DatetimeIndex(sessions).normalize().unique().sort_values()
        self.sessions = s
        bd = pd.bdate_range(s[0], s[-1])
        self._hol = np.asarray(bd.difference(s).values, dtype="datetime64[D]")

    def _d(self, dates) -> tuple[np.ndarray, np.ndarray]:
        idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(np.asarray(dates)))).normalize()
        ok = ~np.asarray(idx.isna())
        return idx, ok

    def _apply(self, dates, n, roll):
        idx, ok = self._d(dates)
        out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
        if ok.any():
            d = idx[ok].values.astype("datetime64[D]")
            out[ok] = np.busday_offset(d, n, roll=roll, holidays=self._hol).astype("datetime64[ns]")
        return pd.DatetimeIndex(out)

    def is_session(self, dates) -> np.ndarray:
        idx, ok = self._d(dates)
        res = np.zeros(len(idx), dtype=bool)
        if ok.any():
            res[ok] = np.is_busday(idx[ok].values.astype("datetime64[D]"), holidays=self._hol)
        return res

    def on_or_after(self, dates) -> pd.DatetimeIndex:
        """First session on or after each date (a Saturday filing is seen Monday)."""
        return self._apply(dates, 0, "forward")

    def strict_next(self, dates) -> pd.DatetimeIndex:
        """First session strictly after each date, whatever the date is (a Friday decision fills Monday)."""
        return self._apply(dates, 1, "backward")

    def shift(self, dates, n: int) -> pd.DatetimeIndex:
        """`n` sessions after (n>0) / before (n<0) each date, first rolling the date onto a session."""
        return self._apply(dates, int(n), "forward" if n >= 0 else "backward")


# --------------------------------------------------------------------------------------------------------------------
# Findings / reports (shared by every checker)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Finding:
    severity: str            # "error" | "warn"
    code: str
    subject: str
    detail: str = ""


@dataclass
class Report:
    name: str
    findings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, severity, code, subject, detail=""):
        self.findings.append(Finding(severity, code, str(subject), detail))

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set:
        return {f.code for f in self.findings}

    def error_codes(self) -> set:
        return {f.code for f in self.errors}

    def summary(self) -> str:
        head = f"{self.name}: {'OK' if self.ok else 'FAIL'} ({len(self.errors)} errors, " \
               f"{len(self.findings) - len(self.errors)} warnings)"
        lines = [f"  [{f.severity}] {f.code} {f.subject} {f.detail}".rstrip() for f in self.findings[:20]]
        return "\n".join([head] + lines)

    def require(self, exc=PITError):
        if not self.ok:
            raise exc(self.summary())
        return self


# --------------------------------------------------------------------------------------------------------------------
# Audit log: every access, allowed or denied, hash-chained so an edit after the fact is detectable
# --------------------------------------------------------------------------------------------------------------------
class AuditLog:
    def __init__(self):
        self.records: list[dict] = []
        self._digests: list[str] = []

    @staticmethod
    def _digest(prev: str, rec: dict) -> str:
        return hashlib.sha256((prev + json.dumps(rec, sort_keys=True, default=str)).encode()).hexdigest()

    def record(self, **rec) -> dict:
        rec["seq"] = len(self.records)
        prev = self._digests[-1] if self._digests else ""
        self._digests.append(self._digest(prev, rec))
        self.records.append(rec)
        return rec

    def __len__(self):
        return len(self.records)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def verify(self) -> bool:
        prev = ""
        for rec, dg in zip(self.records, self._digests):
            prev = self._digest(prev, rec)
            if prev != dg:
                return False
        return len(self.records) == len(self._digests)

    def denied(self) -> list[dict]:
        return [r for r in self.records if r.get("verdict") == "denied"]

    def leaks(self) -> list[dict]:
        """Allowed data accesses whose returned rows were available after the guard's as_of. Must always be empty;
        it is an independent re-check of the filter, so a bug in the filter cannot hide itself."""
        out = []
        for r in self.records:
            if r.get("verdict") == "ok" and r.get("op") != "fill" and r.get("max_avail") is not None:
                if pd.Timestamp(r["max_avail"]) > pd.Timestamp(r["as_of"]):
                    out.append(r)
        return out

    def max_avail_touched(self, as_of=None) -> pd.Timestamp | None:
        vals = [pd.Timestamp(r["max_avail"]) for r in self.records
                if r.get("max_avail") is not None and r.get("verdict") == "ok" and r.get("op") != "fill"
                and (as_of is None or pd.Timestamp(r["as_of"]) == _ts(as_of))]
        return max(vals) if vals else None

    def sources_touched(self) -> set:
        return {r["source"] for r in self.records if r.get("verdict") == "ok"}


# --------------------------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------------------------
class WideSource:
    """Daily bars: {field: DataFrame(date x ticker)}. A bar is available `lag` sessions after its own date (0 = at that
    day's close)."""
    kind = "wide"

    def __init__(self, name, frames, calendar: Calendar, lag: int = 0):
        if isinstance(frames, pd.DataFrame):
            frames = {"value": frames}
        self.name, self.cal, self.lag = name, calendar, int(lag)
        self.frames = {k: v.copy() for k, v in frames.items()}
        self._avail = {k: self._avail_index(v.index) for k, v in self.frames.items()}

    def _avail_index(self, index) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(index).normalize()
        return self.cal.shift(idx, self.lag) if self.lag else idx

    def fields(self):
        return list(self.frames)

    def tickers(self):
        cols = set()
        for f in self.frames.values():
            cols |= set(f.columns)
        return sorted(cols)

    def asof(self, as_of, field_=None, tickers=None, start=None, end=None, lookback=None):
        """Visible rows only. Returns (frame, max_effective, max_avail)."""
        as_of = _ts(as_of)
        fld = field_ if field_ is not None else next(iter(self.frames))
        if fld not in self.frames:
            raise KeyError(f"{self.name}: no field {fld!r} (have {self.fields()})")
        df, av = self.frames[fld], self._avail[fld]
        keep = np.asarray(av <= as_of)
        idx = pd.DatetimeIndex(df.index).normalize()
        if start is not None:
            keep &= np.asarray(idx >= start)
        if end is not None:
            keep &= np.asarray(idx <= end)
        sub = df[keep]
        if tickers is not None:
            sub = sub[[t for t in sub.columns if t in set(tickers)]]
        if lookback is not None:
            sub = sub.iloc[-int(lookback):] if lookback > 0 else sub.iloc[:0]
        if len(sub) == 0:
            return sub.copy(), None, None
        pos = df.index.get_indexer(sub.index)
        return sub.copy(), pd.Timestamp(sub.index.max()), pd.Timestamp(av[pos].max())

    def last_visible(self, as_of) -> pd.Timestamp | None:
        av = next(iter(self._avail.values()))
        vis = av[av <= as_of]
        return pd.Timestamp(vis.max()) if len(vis) else None

    def scrambled(self, as_of, rng, n_extra=3) -> "WideSource":
        out = {}
        for k, df in self.frames.items():
            av = self._avail[k]
            fut = np.asarray(av > as_of)
            new = df.copy()
            if fut.any():
                new = _scramble_rows(new, fut, rng)
            if n_extra > 0 and len(df):
                extra_idx = self.cal.shift([df.index.max()], 1)
                extra_idx = pd.DatetimeIndex(
                    pd.bdate_range(extra_idx[0], periods=n_extra))
                pool = new.iloc[-min(len(new), 20):]
                take = rng.integers(0, len(pool), n_extra)
                ext = pool.iloc[take].copy()
                ext.index = extra_idx
                num = ext.select_dtypes("number").columns
                ext[num] = ext[num].values * (1 + 0.2 * rng.standard_normal((n_extra, len(num))))
                new = pd.concat([new, ext])
            out[k] = new
        return WideSource(self.name, out, self.cal, self.lag)

    def describe(self):
        f = next(iter(self.frames.values()))
        return {"kind": "wide", "fields": self.fields(), "rows": len(f), "cols": f.shape[1], "lag": self.lag}


class RecordSource:
    """Long records (events, insider, fundamentals, macro). `effective` is what the record describes (event date,
    trade date, period end, observation date); availability is when the market could know it: an explicit column, or
    derived from effective + timing + `lag` sessions. `period` marks records that restate one another (fundamentals
    per fiscal period, macro per observation): as-of the latest visible vintage wins."""

    def __init__(self, name, df, effective, avail=None, lag=0, key="ticker", period=None, timing=None,
                 calendar: Calendar | None = None, kind="events"):
        self.name, self.kind, self.effective, self.key, self.period = name, kind, effective, key, period
        self.cal = calendar or Calendar()
        self.lag, self.timing = int(lag), timing
        d = df.reset_index(drop=True).copy()
        if effective not in d.columns:
            raise IntegrityError(f"{name}: effective-date column {effective!r} missing")
        d[effective] = pd.to_datetime(d[effective]).dt.normalize()
        if avail is not None:
            if avail not in d.columns:
                raise IntegrityError(f"{name}: availability column {avail!r} missing")
            av = pd.DatetimeIndex(pd.to_datetime(d[avail]).dt.normalize())
            self._avail_col = avail
        else:
            av = self._derive(d, effective, timing)
            self._avail_col = None
        d["available"] = av
        order = np.argsort(d["available"].values.astype("datetime64[ns]"), kind="stable")   # NaT sorts last
        self.df = d.iloc[order].reset_index(drop=True)
        self._av = self.df["available"].values.astype("datetime64[ns]")

    def _derive(self, d, effective, timing) -> pd.DatetimeIndex:
        eff = pd.DatetimeIndex(d[effective])
        if timing and timing in d.columns:
            same_day = d[timing].astype(str).str.lower().isin(["bmo", "dmh", "intraday"]).values
            base = pd.DatetimeIndex(np.where(same_day, self.cal.on_or_after(eff).values,
                                             self.cal.strict_next(eff).values))
        else:
            base = self.cal.on_or_after(eff)
        return self.cal.shift(base, self.lag) if self.lag else base

    def _visible(self, as_of):
        as_of = _ts(as_of)
        n = int(np.searchsorted(self._av, as_of.to_datetime64(), side="right"))
        return self.df.iloc[:n]

    def _latest(self, sub):
        cols = [c for c in (self.key, self.period) if c and c in sub.columns]
        if not cols:
            return sub
        sub = sub.copy()
        sub["n_versions"] = sub.groupby(cols, dropna=False)[cols[0]].transform("size")
        return sub.drop_duplicates(cols, keep="last")

    def asof(self, as_of, tickers=None, start=None, end=None, latest=False, lookback_days=None):
        as_of = _ts(as_of)
        sub = self._visible(as_of)
        m = np.ones(len(sub), dtype=bool)
        if tickers is not None and self.key in sub.columns:
            m &= sub[self.key].isin(list(tickers)).values
        eff = sub[self.effective]
        if start is not None:
            m &= (eff >= start).values
        if lookback_days is not None:
            m &= (eff > as_of - pd.Timedelta(days=int(lookback_days))).values
        if end is not None:
            m &= (eff <= end).values
        sub = sub[m]
        if latest:
            sub = self._latest(sub)
        if len(sub) == 0:
            return sub.copy(), None, None
        return sub.copy(), pd.Timestamp(sub[self.effective].max()), pd.Timestamp(sub["available"].max())

    def versions(self, as_of, value_cols) -> pd.DataFrame:
        """Visible (key, period) groups that were restated, with the first and latest visible values."""
        sub = self._visible(as_of)
        cols = [c for c in (self.key, self.period) if c and c in sub.columns]
        if not cols or not len(sub):
            return pd.DataFrame(columns=cols + ["n_versions"])
        g = sub.groupby(cols, dropna=False)
        out = g.size().rename("n_versions").reset_index()
        for v in value_cols:
            out[f"{v}_first"] = g[v].first().values
            out[f"{v}_last"] = g[v].last().values
        return out[out["n_versions"] > 1].reset_index(drop=True)

    def restatement_exposure(self, as_of, value_cols) -> dict:
        """AUDIT ONLY (reads the future by design): how different would the visible periods look if a frame kept just
        the final vintage. This is the size of the leak a naive 'latest values' join would inject."""
        vis = self._latest(self._visible(as_of))
        cols = [c for c in (self.key, self.period) if c and c in vis.columns]
        if not cols or not len(vis):
            return {"periods": 0, "restated": 0, "mean_abs_rel_diff": 0.0}
        fin = self._latest(self.df[self.df["available"].notna()])
        mrg = vis.merge(fin[cols + list(value_cols)], on=cols, suffixes=("", "_final"))
        rel = []
        for v in value_cols:
            a, b = mrg[v].astype(float), mrg[f"{v}_final"].astype(float)
            rel.append(((a - b).abs() / b.abs().replace(0, np.nan)).fillna(0.0))
        r = pd.concat(rel, axis=1).max(axis=1) if rel else pd.Series(dtype=float)
        return {"periods": int(len(mrg)), "restated": int((r > 1e-12).sum()),
                "mean_abs_rel_diff": float(r.mean()) if len(r) else 0.0}

    def scrambled(self, as_of, rng, n_extra=3) -> "RecordSource":
        d = self.df.copy()
        fut = (d["available"] > as_of).values
        protect = {self.effective, "available", self.key, self.period, self.timing, self._avail_col}
        cols = [c for c in d.columns if c not in protect]
        if fut.any() and cols:
            sub = _scramble_rows(d.loc[fut, cols].reset_index(drop=True), np.ones(int(fut.sum()), bool), rng)
            for c in cols:
                if sub[c].dtype != d[c].dtype:
                    d[c] = d[c].astype(sub[c].dtype)
                d.loc[fut, c] = sub[c].values
        if n_extra > 0 and len(d):
            tail = d.iloc[rng.integers(0, len(d), n_extra)].copy()
            bump = pd.Timedelta(days=400 + int(rng.integers(1, 30)))
            tail[self.effective] = tail[self.effective] + bump
            if self._avail_col:
                tail[self._avail_col] = pd.to_datetime(tail[self._avail_col]) + bump
            d = pd.concat([d, tail], ignore_index=True)
        d = d.drop(columns=["available"])
        return RecordSource(self.name, d, self.effective, avail=self._avail_col, lag=self.lag, key=self.key,
                            period=self.period, timing=self.timing, calendar=self.cal, kind=self.kind)

    def describe(self):
        return {"kind": self.kind, "rows": len(self.df), "lag": self.lag, "key": self.key, "period": self.period}


def _scramble_rows(df: pd.DataFrame, mask: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Replace the masked rows' contents with a permutation of themselves plus noise; index is left alone."""
    out = df.copy()
    pos = np.flatnonzero(mask)
    if len(pos) == 0:
        return out
    perm = rng.permutation(len(pos))
    for j in range(out.shape[1]):
        col = out.iloc[:, j]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            vals = col.values[pos][perm]
            sd = np.nanstd(col.values.astype(float)) or 1.0
            new = col.astype(float)
            if np.nanmin(col.values.astype(float)) > 0:        # prices / volumes stay positive so log() features run
                new.iloc[pos] = vals.astype(float) * np.exp(0.3 * rng.standard_normal(len(pos)))
            else:
                new.iloc[pos] = vals.astype(float) + sd * (0.5 * rng.standard_normal(len(pos)) + 0.25)
        else:                                          # .array keeps timezone-aware / categorical dtypes intact
            new = col.copy()
            new.iloc[pos] = col.iloc[pos].iloc[perm].array
        out.isetitem(j, new)
    return out


class Listings:
    """Listing table: ticker, list_date, delist_date (NaT while alive), reason. A delisting is only KNOWN on or after
    its date, so membership as of T never uses a later delisting."""

    def __init__(self, df: pd.DataFrame):
        d = df.copy()
        need = {"ticker", "list_date", "delist_date"}
        if not need <= set(d.columns):
            raise IntegrityError(f"listings need columns {sorted(need)}")
        d["list_date"] = pd.to_datetime(d["list_date"]).dt.normalize()
        d["delist_date"] = pd.to_datetime(d["delist_date"]).dt.normalize()
        if "reason" not in d:
            d["reason"] = ""
        self.df = d.reset_index(drop=True)

    def members(self, as_of) -> list[str]:
        d = self.df
        alive = (d["list_date"] <= as_of) & (d["delist_date"].isna() | (d["delist_date"] > as_of))
        return sorted(d.loc[alive, "ticker"])

    def known_delisted(self, as_of) -> pd.DataFrame:
        d = self.df
        return d[d["delist_date"].notna() & (d["delist_date"] <= as_of)].copy()

    def scrambled(self, as_of, rng) -> "Listings":
        d = self.df.copy()
        fut = (d["delist_date"] > as_of) | (d["list_date"] > as_of)
        d.loc[fut & d["delist_date"].notna(), "reason"] = "scrambled"
        return Listings(d)


# --------------------------------------------------------------------------------------------------------------------
# Store, guard, executor
# --------------------------------------------------------------------------------------------------------------------
class PITStore:
    """Holds every source. Models never receive the store, only a Guard from `view()`."""

    def __init__(self, calendar: Calendar | None = None):
        self.cal = calendar or Calendar()
        self._src: dict = {}
        self.listings: Listings | None = None
        self.log = AuditLog()

    # -- registration
    def add_prices(self, name, frames, lag=0):
        self._register(WideSource(name, frames, self.cal, lag))
        return self

    def add_events(self, name, df, effective="date", timing="timing", lag=0, key="ticker"):
        self._register(RecordSource(name, df, effective, lag=lag, key=key, timing=timing if timing in df.columns
                                    else None, calendar=self.cal, kind="events"))
        return self

    def add_insider(self, name, df, effective="trade_date", avail="filing_date", lag=2, key="ticker"):
        """Form-4 style: an explicit filing date wins; otherwise trade date + `lag` sessions."""
        a = avail if avail in df.columns else None
        self._register(RecordSource(name, df, effective, avail=a, lag=lag, key=key, calendar=self.cal,
                                    kind="insider"))
        return self

    def add_fundamentals(self, name, df, period="period_end", avail="filed", key="ticker"):
        """One row per filing version; `period` groups restatements of the same fiscal period."""
        self._register(RecordSource(name, df, period, avail=avail, key=key, period=period, calendar=self.cal,
                                    kind="fundamentals"))
        return self

    def add_macro(self, name, df, date="date", series="series", lag=1, vintage=None):
        """`lag` is one int or {series: sessions}. A `vintage` column (release date of that value) overrides lag and
        lets revised numbers coexist as separate vintages."""
        d = df.copy()
        if vintage is None:
            lags = lag if isinstance(lag, dict) else {s: lag for s in d[series].unique()}
            miss = set(d[series].unique()) - set(lags)
            if miss:
                raise IntegrityError(f"{name}: no publication lag declared for series {sorted(miss)}")
            eff = pd.to_datetime(d[date]).dt.normalize()
            av = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
            for s, n in lags.items():
                m = (d[series] == s).values
                av[m] = self.cal.shift(eff[m], n).values
            d["vintage"] = av
            vintage = "vintage"
        self._register(RecordSource(name, d, date, avail=vintage, key=series, period=date, calendar=self.cal,
                                    kind="macro"))
        return self

    def add_listings(self, df):
        self.listings = Listings(df)
        return self

    def _register(self, src):
        if src.name in self._src:
            raise IntegrityError(f"source {src.name!r} already registered")
        self._src[src.name] = src

    def source(self, name):
        if name not in self._src:
            raise KeyError(f"unknown source {name!r}; have {sorted(self._src)}")
        return self._src[name]

    def names(self):
        return sorted(self._src)

    # -- views
    def view(self, as_of, live=None, live_fields=(), log: AuditLog | None = None) -> "Guard":
        return Guard(self, as_of, log if log is not None else self.log, live, live_fields)

    def executor(self, decision_asof, prices="prices", open_field="Open", log: AuditLog | None = None):
        return Executor(self, decision_asof, prices, open_field, log if log is not None else self.log)

    def scrambled(self, as_of, seed=0, n_extra=3) -> "PITStore":
        """A copy in which everything not yet available at `as_of` is garbage (values shuffled + noised) and a few
        fabricated later rows are appended. A correct pipeline cannot tell it from the real store."""
        rng = np.random.default_rng(seed)
        as_of = _ts(as_of)
        new = PITStore(self.cal)
        for n, s in self._src.items():
            new._src[n] = s.scrambled(as_of, rng, n_extra)
        new.listings = self.listings.scrambled(as_of, rng) if self.listings is not None else None
        return new

    # -- 1.1 integrity
    def validate(self) -> Report:
        rep = Report("pit.validate")
        for n, s in self._src.items():
            if isinstance(s, WideSource):
                self._validate_wide(rep, n, s)
            else:
                self._validate_records(rep, n, s)
        if self.listings is not None:
            d = self.listings.df
            bad = d[d["delist_date"].notna() & (d["delist_date"] < d["list_date"])]
            for t in bad["ticker"]:
                rep.add("error", "delist_before_list", t)
            if d["ticker"].duplicated().any():
                rep.add("warn", "relisted_ticker", "listings", "ticker appears more than once")
        rep.stats["sources"] = {n: s.describe() for n, s in self._src.items()}
        return rep

    @staticmethod
    def _validate_wide(rep, n, s: WideSource):
        for fld, df in s.frames.items():
            idx = pd.DatetimeIndex(df.index)
            if not idx.is_monotonic_increasing:
                rep.add("error", "unsorted_index", f"{n}.{fld}")
            if idx.has_duplicates:
                rep.add("error", "duplicate_dates", f"{n}.{fld}")
            if idx.isna().any():
                rep.add("error", "nat_index", f"{n}.{fld}")
            if s.lag < 0:
                rep.add("error", "negative_lag", n)
        cols = {tuple(f.columns) for f in s.frames.values()}
        if len(cols) > 1:
            rep.add("warn", "ragged_fields", n, "fields disagree on the ticker set")

    @staticmethod
    def _validate_records(rep, n, s: RecordSource):
        d = s.df
        eff = d[s.effective]
        nat_av = d["available"].isna()
        if nat_av.any():
            rep.add("error", "missing_availability", n, f"{int(nat_av.sum())} rows have no availability date")
        if eff.isna().any():
            rep.add("error", "missing_effective", n, f"{int(eff.isna().sum())} rows have no effective date")
        early = (d["available"] < eff) & ~nat_av
        if early.any():
            rep.add("error", "available_before_effective", n, f"{int(early.sum())} rows known before they happened")
        if s.kind in ("fundamentals", "macro") and len(d):
            same = float(((d["available"] == eff) & ~nat_av).mean())
            if same > 0.5:
                rep.add("error", "no_publication_lag", n,
                        f"{same:.0%} of {s.kind} rows are available on their own effective date")
        cols = [c for c in (s.key, s.period) if c and c in d.columns]
        if s.period and cols:
            dup = d.duplicated(cols + ["available"], keep=False)
            if dup.any():
                rep.add("error", "ambiguous_vintage", n,
                        f"{int(dup.sum())} rows share (key, period, availability)")


class Guard:
    """The time fence. Bound to one `as_of` for life; there is no way to widen it. Everything goes through here, and
    anything after as_of raises LookAheadError (after being logged as denied)."""
    __slots__ = ("_store", "_as_of", "_log", "_live", "_live_ok")

    def __init__(self, store: PITStore, as_of, log: AuditLog, live=None, live_fields=()):
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_as_of", _ts(as_of))
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_live", dict(live or {}))
        object.__setattr__(self, "_live_ok", frozenset(live_fields))

    def __setattr__(self, k, v):
        raise AttributeError("Guard is immutable: the fence cannot be moved")

    @property
    def as_of(self) -> pd.Timestamp:
        return self._as_of

    @property
    def log(self) -> AuditLog:
        return self._log

    # -- fence primitives
    def _deny(self, source, op, requested, reason):
        self._log.record(op=op, source=source, as_of=str(self._as_of.date()), requested=str(requested),
                         verdict="denied", reason=reason, rows=0, max_effective=None, max_avail=None)
        raise LookAheadError(f"{source}.{op}: {reason} (as_of {self._as_of.date()}, requested {requested})")

    def fence(self, ts, what="timestamp"):
        """Raise unless ts <= as_of. Use for any time a caller computed itself."""
        t = _ts(ts)
        if t > self._as_of:
            self._deny("-", "fence", t.date(), f"{what} is after as_of")
        return t

    def _bounds(self, source, op, start, end):
        s = _ts(start) if start is not None else None
        e = _ts(end) if end is not None else None
        for label, v in (("start", s), ("end", e)):
            if v is not None and v > self._as_of:
                self._deny(source, op, v.date(), f"{label} is after as_of")
        if s is not None and e is not None and s > e:
            raise ValueError(f"{source}: start {s.date()} after end {e.date()}")
        return s, (e if e is not None else self._as_of)

    def _emit(self, source, op, requested, frame_rows, max_eff, max_av):
        # independent re-check of the filter that produced the rows
        if max_av is not None and max_av > self._as_of:
            self._log.record(op=op, source=source, as_of=str(self._as_of.date()), requested=str(requested),
                             verdict="ok", rows=frame_rows, max_effective=str(max_eff), max_avail=str(max_av))
            raise LookAheadError(f"{source}.{op}: filter returned rows available {max_av.date()} > as_of")
        self._log.record(op=op, source=source, as_of=str(self._as_of.date()), requested=str(requested),
                         verdict="ok", rows=int(frame_rows),
                         max_effective=None if max_eff is None else str(max_eff.date()),
                         max_avail=None if max_av is None else str(max_av.date()))

    # -- data access
    def wide(self, name, field_=None, tickers=None, start=None, end=None, lookback=None) -> pd.DataFrame:
        src = self._store.source(name)
        if not isinstance(src, WideSource):
            raise TypeError(f"{name} is a record source; use .records()")
        s, e = self._bounds(name, "wide", start, end)
        df, me, ma = src.asof(self._as_of, field_, tickers, s, e, lookback)
        self._emit(name, "wide", f"{field_}[{s.date() if s is not None else ''}:{e.date()}]", len(df), me, ma)
        return df

    def last_row(self, name, field_=None, tickers=None) -> pd.Series:
        df = self.wide(name, field_, tickers, lookback=1)
        return df.iloc[-1] if len(df) else pd.Series(dtype=float)

    def records(self, name, tickers=None, start=None, end=None, latest=False, lookback_days=None) -> pd.DataFrame:
        src = self._store.source(name)
        if isinstance(src, WideSource):
            raise TypeError(f"{name} is a wide source; use .wide()")
        s, e = self._bounds(name, "records", start, end)
        df, me, ma = src.asof(self._as_of, tickers, s, e, latest, lookback_days)
        self._emit(name, "records", f"latest={latest}[{s.date() if s is not None else ''}:{e.date()}]", len(df), me, ma)
        return df

    def macro_wide(self, name, value="value", start=None) -> pd.DataFrame:
        """Latest visible vintage of each observation, pivoted date x series."""
        src = self._store.source(name)
        df = self.records(name, start=start, latest=True)
        if not len(df):
            return pd.DataFrame()
        return df.pivot(index=src.effective, columns=src.key, values=value).sort_index()

    def members(self) -> list[str]:
        lst = self._store.listings
        if lst is None:
            raise KeyError("no listings registered")
        m = lst.members(self._as_of)
        self._log.record(op="members", source="listings", as_of=str(self._as_of.date()), requested="as_of",
                         verdict="ok", rows=len(m), max_effective=str(self._as_of.date()), max_avail=None)
        return m

    def live(self, field_):
        """The only sanctioned window onto 'now': explicitly permitted live snapshot fields."""
        if field_ not in self._live_ok or field_ not in self._live:
            self._deny("live", "live", field_, "not a permitted live snapshot field")
        self._log.record(op="live", source="live", as_of=str(self._as_of.date()), requested=field_,
                         verdict="ok", rows=1, max_effective=None, max_avail=None)
        return self._live[field_]

    def assert_frame_clean(self, df: pd.DataFrame, what="frame"):
        """For frames a model assembled itself: no date-level index value may lie after as_of."""
        dates = _index_dates(df)
        if dates is not None and len(dates) and dates.max() > self._as_of:
            self._deny(what, "assert_frame_clean", dates.max().date(), "frame contains rows after as_of")
        return df


def _index_dates(df) -> pd.DatetimeIndex | None:
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    return idx.normalize()


class Executor:
    """The simulator's side of decide-at-close / fill-at-next-open. Given a decision made at `decision_asof`'s close it
    only ever prices at the strict next session's Open. No data for that session -> no fill (never a substitute)."""

    def __init__(self, store: PITStore, decision_asof, prices, open_field, log: AuditLog):
        self.store, self.log = store, log
        self.decision = _ts(decision_asof)
        if not bool(store.cal.is_session([self.decision])[0]):
            raise FillTimingError(f"decision date {self.decision.date()} is not a session: no weekend/holiday decisions")
        self.src = store.source(prices)
        self.open_field = open_field
        self.fill_date = store.cal.strict_next([self.decision])[0]

    def fill_next_open(self, ticker, shares, order_id=None) -> dict:
        df = self.src.frames.get(self.open_field)
        if df is None or ticker not in df.columns or self.fill_date not in df.index:
            raise FillTimingError(f"no {self.open_field} for {ticker} on {self.fill_date.date()}: cannot fill")
        px = df.at[self.fill_date, ticker]
        if not np.isfinite(px):
            raise FillTimingError(f"{ticker} has no open on {self.fill_date.date()}: cannot fill")
        oid = order_id if order_id is not None else f"{ticker}-{self.decision.date()}-{len(self.log)}"
        rec = dict(order_id=oid, ticker=ticker, decision_date=self.decision, fill_date=self.fill_date,
                   fill_price=float(px), shares=float(shares))
        self.log.record(op="fill", source=self.src.name, as_of=str(self.decision.date()), requested=ticker,
                        verdict="ok", rows=1, max_effective=str(self.fill_date.date()),
                        max_avail=str(self.fill_date.date()))
        return rec


# --------------------------------------------------------------------------------------------------------------------
# 1.1 labels and feature stamps
# --------------------------------------------------------------------------------------------------------------------
def label_close_dates(dates, horizon: int, calendar: Calendar, entry_lag: int = 1) -> pd.DatetimeIndex:
    """When each row's forward label is fully realised. Decide at close t, enter at the open `entry_lag` sessions later,
    hold `horizon` sessions: the label closes at the close of session t + entry_lag + horizon - 1."""
    if horizon < 1 or entry_lag < 1:
        raise ValueError("horizon >= 1 and entry_lag >= 1 (a same-close entry is forbidden)")
    return calendar.shift(dates, entry_lag + horizon - 1)


def verify_training_rows(row_dates, label_close, as_of):
    """Every training row must have a label closed on or before as_of."""
    rd = pd.DatetimeIndex(row_dates)
    lc = pd.DatetimeIndex(label_close)
    if len(rd) != len(lc):
        raise ValueError("row_dates and label_close differ in length")
    if len(lc) == 0:
        return
    if lc.isna().any():
        raise LabelLeakError(f"{int(lc.isna().sum())} training rows have no label close date")
    if (lc < rd).any():
        raise LabelLeakError("label closes before its own decision date (corrupt label dates)")
    late = lc > _ts(as_of)
    if late.any():
        raise LabelLeakError(f"{int(late.sum())} training rows have labels closing after {_ts(as_of).date()} "
                             f"(latest {lc.max().date()})")


def purged_training_set(X: pd.DataFrame, y: pd.Series, as_of, horizon: int, calendar: Calendar,
                        entry_lag: int = 1, embargo: int = 0):
    """Rows of a (date, ticker) panel whose label closed `embargo` sessions before as_of. Verified before return."""
    dates = X.index.get_level_values(0)
    lc = label_close_dates(dates, horizon, calendar, entry_lag)
    cutoff = calendar.shift([_ts(as_of)], -embargo)[0] if embargo else _ts(as_of)
    keep = np.asarray(lc <= cutoff)
    Xs, ys = X[keep], y.reindex(X.index)[keep]
    verify_training_rows(dates[keep], lc[keep], cutoff)
    return Xs, ys


def verify_feature_availability(row_dates, avail):
    """`avail` (Series/DataFrame of the availability date of each feature's latest input, aligned to the rows) must
    never exceed the row's decision date."""
    rd = pd.DatetimeIndex(row_dates)
    a = avail if isinstance(avail, pd.DataFrame) else avail.to_frame("feature")
    if len(a) != len(rd):
        raise ValueError("avail and row_dates differ in length")
    for c in a.columns:
        v = pd.DatetimeIndex(pd.to_datetime(a[c]))
        bad = np.asarray(v > rd)
        if bad.any():
            raise LookAheadError(f"feature {c!r}: {int(bad.sum())} rows built from data not yet available "
                                 f"(worst {int((v - rd).max().days)} days early)")
        if v.isna().any():
            raise LookAheadError(f"feature {c!r}: {int(v.isna().sum())} rows carry no availability stamp")


# --------------------------------------------------------------------------------------------------------------------
# Survivorship
# --------------------------------------------------------------------------------------------------------------------
def survivorship_report(prices: pd.DataFrame, listings: Listings, as_of, start=None,
                        expected_annual_attrition: float = 0.03, min_names: int = 30) -> Report:
    """Compare a wide close panel to the listing table.
    errors: ghost_prices (bars after a delisting), missing_history (a member at `start` absent from the panel),
            truncated_history (delisted name whose prices stop long before delisting), survivor_only_panel.
    The last one is the classic bias: if the panel holds only names alive at the end, measured attrition is ~0 while a
    real universe loses a few percent a year."""
    rep = Report("pit.survivorship")
    as_of = _ts(as_of)
    px = prices[prices.index <= as_of]
    start = _ts(start) if start is not None else (_ts(px.index.min()) if len(px) else as_of)
    d = listings.df
    have = set(px.columns)
    members0 = set(listings.members(start))
    last_valid = {t: px[t].last_valid_index() for t in px.columns}
    first_valid = {t: px[t].first_valid_index() for t in px.columns}

    for _, r in d.iterrows():
        t, dl = r["ticker"], r["delist_date"]
        if t in have and pd.notna(dl):
            after = px[t][px.index > dl].dropna()
            if len(after):
                rep.add("error", "ghost_prices", t, f"{len(after)} bars after delisting {dl.date()}")
            lv = last_valid[t]
            if lv is not None and dl <= as_of and (dl - lv).days > 10:
                rep.add("error", "truncated_history", t, f"prices end {lv.date()}, delisted {dl.date()}")
        if t in have and first_valid[t] is not None and first_valid[t] < r["list_date"] - pd.Timedelta(days=5):
            rep.add("warn", "prices_before_listing", t, f"first bar {first_valid[t].date()} before {r['list_date'].date()}")
    for t in sorted(members0 - have):
        rep.add("error", "missing_history", t, "listed at start but absent from the price panel")

    present0 = members0 & have
    end_alive = {t for t in present0 if last_valid[t] is not None and last_valid[t] >= px.index[-1] - pd.Timedelta(days=7)} \
        if len(px) else set()
    years = max((as_of - start).days / 365.25, 1e-9)
    attr = 1.0 - len(end_alive) / max(len(present0), 1)
    annual = 1.0 - (1.0 - attr) ** (1.0 / years) if attr < 1 else 1.0
    rep.stats.update(names_at_start=len(present0), alive_at_end=len(end_alive), attrition_total=attr,
                     attrition_annual=annual, years=years)
    if len(present0) >= min_names and years >= 2 and annual < expected_annual_attrition * 0.2:
        rep.add("error", "survivor_only_panel", "panel",
                f"annual attrition {annual:.2%} vs expected about {expected_annual_attrition:.1%}: "
                "delisted names look dropped")
    return rep


# --------------------------------------------------------------------------------------------------------------------
# 1.4 fill audit
# --------------------------------------------------------------------------------------------------------------------
def audit_fills(decisions: pd.DataFrame, fills: pd.DataFrame, opens: pd.DataFrame, closes: pd.DataFrame | None,
                calendar: Calendar, price_tol: float = 1e-6) -> Report:
    """decisions: order_id, ticker, decision_date [, decision_time (hour float or Timestamp), inputs_max_avail]
    fills:     order_id, ticker, fill_date, fill_price
    opens/closes: wide date x ticker. Proves after-close decisions, next-session-open fills, no same-close price,
    no inputs published after the decision date."""
    rep = Report("pit.audit_fills")
    dec = decisions.copy()
    dec["decision_date"] = pd.to_datetime(dec["decision_date"]).dt.normalize()
    fl = fills.copy()
    fl["fill_date"] = pd.to_datetime(fl["fill_date"]).dt.normalize()
    if dec["order_id"].duplicated().any():
        rep.add("error", "duplicate_order_id", "decisions")
    if fl["order_id"].duplicated().any():
        rep.add("warn", "multiple_fills", "fills", "an order has several fills; each is checked")
    ns = set(fl["order_id"]) - set(dec["order_id"])
    for o in sorted(ns, key=str):
        rep.add("error", "fill_without_decision", o)
    for o in sorted(set(dec["order_id"]) - set(fl["order_id"]), key=str):
        rep.add("warn", "unfilled_decision", o)
    # decision-level checks run on every decision, filled or not
    for _, r in dec.iterrows():
        oid, dd = r["order_id"], r["decision_date"]
        if not calendar.is_session([dd])[0]:
            rep.add("error", "decision_not_session", oid, str(dd.date()))
        if "decision_time" in dec.columns and pd.notna(r["decision_time"]):
            h = r["decision_time"]
            hour = h.hour + h.minute / 60 if isinstance(h, pd.Timestamp) else float(h)
            if hour < 16.0:
                rep.add("error", "decision_before_close", oid, f"decided at {hour:.2f}h")
        if "inputs_max_avail" in dec.columns and pd.notna(r["inputs_max_avail"]):
            if pd.Timestamp(r["inputs_max_avail"]).normalize() > dd:
                rep.add("error", "next_day_information", oid,
                        f"inputs available {pd.Timestamp(r['inputs_max_avail']).date()} > decision {dd.date()}")
    m = fl.drop(columns=["decision_date"], errors="ignore").merge(dec[["order_id", "decision_date"]], on="order_id")
    nxt_cache: dict = {}
    for _, r in m.iterrows():
        oid, t, dd, fd = r["order_id"], r["ticker"], r["decision_date"], r["fill_date"]
        if not calendar.is_session([fd])[0]:
            rep.add("error", "fill_not_session", oid, f"{fd.date()} is a weekend/holiday")
        if fd == dd:
            rep.add("error", "same_close_fill", oid, "filled on the decision date")
        elif fd < dd:
            rep.add("error", "fill_before_decision", oid, f"{fd.date()} < {dd.date()}")
        else:
            nxt = nxt_cache.setdefault(dd, calendar.strict_next([dd])[0])
            if fd > nxt:
                rep.add("warn", "late_fill", oid, f"filled {fd.date()}, next session was {nxt.date()}")
        px = r["fill_price"]
        if t not in opens.columns or fd not in opens.index or not np.isfinite(opens.at[fd, t]):
            rep.add("error", "no_open_data", oid, f"no open for {t} on {fd.date()}")
        elif abs(px - opens.at[fd, t]) > price_tol * max(1.0, abs(px)):
            hit_close = closes is not None and t in closes.columns and dd in closes.index \
                and abs(px - closes.at[dd, t]) <= price_tol * max(1.0, abs(px))
            rep.add("error", "same_close_price" if hit_close else "not_open_price", oid,
                    f"fill {px:.4f} vs open {opens.at[fd, t]:.4f}")
    rep.stats.update(decisions=len(dec), fills=len(fl))
    return rep


def audit_log_against_decisions(log: AuditLog, decisions: pd.DataFrame) -> Report:
    """Every decision date must show no access beyond it in the audit log, and no denied access went unremarked."""
    rep = Report("pit.audit_log")
    if not log.verify():
        rep.add("error", "log_tampered", "audit_log", "hash chain does not verify")
    for r in log.leaks():
        rep.add("error", "leaked_access", r["source"], f"{r['op']} touched {r['max_avail']} at as_of {r['as_of']}")
    for r in log.denied():
        rep.add("warn", "denied_access", r["source"], f"{r['op']}: {r['reason']}")
    seen = {r["as_of"] for r in log.records}
    for d in pd.to_datetime(decisions["decision_date"]).dt.normalize().unique():
        if str(pd.Timestamp(d).date()) not in seen:
            rep.add("warn", "decision_without_access", str(pd.Timestamp(d).date()), "no logged data access")
    return rep


# --------------------------------------------------------------------------------------------------------------------
# 1.3 leak detectors
# --------------------------------------------------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """Stable hash of frames / arrays / dicts / scalars: for asserting opaque state (Memory, adaptive decisions)
    is byte-identical across a scramble."""
    h = hashlib.sha256()

    def feed(o):
        if isinstance(o, pd.DataFrame):
            h.update(b"DF"); h.update(repr((list(map(str, o.columns)), o.shape)).encode())
            h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
        elif isinstance(o, pd.Series):
            h.update(b"S"); h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
        elif isinstance(o, np.ndarray):
            h.update(b"A"); h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                h.update(str(k).encode()); feed(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"L")
            for v in o:
                feed(v)
        else:
            h.update(repr(o).encode())
    feed(obj)
    return h.hexdigest()


def _compare(a, b, path="out", rtol=1e-7, atol=1e-9, out=None) -> list[str]:
    """Structural diff of two pipeline outputs; empty list means identical within tolerance."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}[{k!r}] present in one run only")
            else:
                _compare(a[k], b[k], f"{path}[{k!r}]", rtol, atol, out)
    elif isinstance(a, (pd.DataFrame, pd.Series)) and isinstance(b, type(a)):
        fa = a.to_frame() if isinstance(a, pd.Series) else a
        fb = b.to_frame() if isinstance(b, pd.Series) else b
        if fa.shape != fb.shape or not fa.index.equals(fb.index) or list(fa.columns) != list(fb.columns):
            out.append(f"{path}: shape/index/columns differ {fa.shape} vs {fb.shape}")
        else:
            for c in fa.columns:
                x, y = fa[c], fb[c]
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    xv, yv = x.values.astype(float), y.values.astype(float)
                    ok = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
                    if not ok.all():
                        out.append(f"{path}[{c}]: {int((~ok).sum())} values differ, max abs "
                                   f"{np.nanmax(np.abs(xv - yv)):.3g}")
                elif not x.equals(y):
                    out.append(f"{path}[{c}]: non-numeric values differ")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            if not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol, equal_nan=True):
                out.append(f"{path}: arrays differ")
        except (ValueError, TypeError):
            out.append(f"{path}: arrays not comparable")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _compare(x, y, f"{path}[{i}]", rtol, atol, out)
    elif isinstance(a, float) or isinstance(b, float):
        if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            out.append(f"{path}: {a!r} vs {b!r}")
    else:
        try:
            same = bool(a == b)
        except (ValueError, TypeError):
            same = repr(a) == repr(b)
        if not same:
            out.append(f"{path}: {a!r} vs {b!r}")
    return out


@dataclass
class ScrambleReport:
    name: str
    variants: dict = field(default_factory=dict)     # variant -> list of diff strings
    errors: dict = field(default_factory=dict)       # variant -> exception text
    detail: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors and not any(self.variants.values())

    def summary(self) -> str:
        lines = [f"{self.name}: {'PASS' if self.passed else 'FAIL CLOSED'}"]
        for v, d in self.variants.items():
            lines += [f"  {v}: {x}" for x in d[:10]]
        lines += [f"  {v} raised: {e}" for v, e in self.errors.items()]
        return "\n".join(lines)

    def require(self):
        if not self.passed:
            raise FailClosed(self.summary())
        return self


def _future_mask(df: pd.DataFrame, as_of, mask_fn=None) -> np.ndarray:
    if mask_fn is not None:
        return np.asarray(mask_fn(df, as_of), dtype=bool)
    dates = _index_dates(df)
    if dates is not None:
        return np.asarray(dates > as_of)
    for c in ("available", "avail", "available_date", "filed", "filing_date", "date"):
        if c in df.columns:
            return np.asarray(pd.to_datetime(df[c]) > as_of)
    raise ValueError("cannot tell which rows are in the future: pass mask_fns")


def scramble_after(data, as_of, rng: np.random.Generator, mask_fn=None):
    """Return `data` (frame or dict of frames) with every row after as_of replaced by shuffled + noised values."""
    as_of = _ts(as_of)
    if isinstance(data, dict):
        return {k: scramble_after(v, as_of, rng, (mask_fn or {}).get(k) if isinstance(mask_fn, dict) else mask_fn)
                for k, v in data.items()}
    m = _future_mask(data, as_of, mask_fn)
    return _scramble_rows(data, m, rng) if m.any() else data.copy()


def truncate_after(data, as_of, mask_fn=None):
    as_of = _ts(as_of)
    if isinstance(data, dict):
        return {k: truncate_after(v, as_of, (mask_fn or {}).get(k) if isinstance(mask_fn, dict) else mask_fn)
                for k, v in data.items()}
    return data[~_future_mask(data, as_of, mask_fn)].copy()


def future_scramble(pipeline, frames: dict, as_of, seed: int = 0, mask_fns=None, **cmp_kw) -> ScrambleReport:
    """pipeline(frames, as_of) -> outputs (predictions, scores, selected patterns, memory fingerprint, adaptive
    decisions...). Runs it on the real frames, on frames with the future removed and on frames with the future
    scrambled; all three must agree. An exception in any variant is a failure (fail closed)."""
    as_of = _ts(as_of)
    rng = np.random.default_rng(seed)
    variants = {
        "real": frames,
        "truncated": truncate_after(frames, as_of, mask_fns),
        "scrambled": scramble_after(frames, as_of, rng, mask_fns),
    }
    return _run_variants("future_scramble", variants, lambda fr: pipeline(fr, as_of), cmp_kw)


def future_scramble_store(pipeline, store: PITStore, as_of, seed: int = 0, n_extra: int = 3, **cmp_kw) -> ScrambleReport:
    """pipeline(guard) -> outputs. Same test through the fence: real store vs scrambled-future store (with fabricated
    later rows). The fence should make the two indistinguishable."""
    as_of = _ts(as_of)
    stores = {"real": store, "scrambled": store.scrambled(as_of, seed, n_extra)}
    rep = _run_variants("future_scramble_store", stores,
                        lambda s: pipeline(s.view(as_of, log=AuditLog())), cmp_kw)
    return rep


def _run_variants(name, variants: dict, call, cmp_kw) -> ScrambleReport:
    rep = ScrambleReport(name)
    outs, base = {}, None
    for v, arg in variants.items():
        try:
            outs[v] = call(arg)
        except Exception as e:                       # noqa: BLE001 - a crash under scramble is a dependence too
            rep.errors[v] = f"{type(e).__name__}: {e}"
    base_name = next(iter(variants))
    if base_name not in outs:
        rep.errors.setdefault(base_name, "baseline failed")
        return rep
    base = outs[base_name]
    for v, o in outs.items():
        if v != base_name:
            rep.variants[v] = _compare(base, o, **cmp_kw)
    rep.detail["fingerprints"] = {v: fingerprint(o) for v, o in outs.items()}
    return rep


@dataclass
class InvarianceReport:
    columns: dict = field(default_factory=dict)      # column -> {n_bad, max_abs_diff, first_bad_date, mode}
    cuts: list = field(default_factory=list)
    rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.columns

    def summary(self) -> str:
        if self.passed:
            return f"invariant at {len(self.cuts)} cuts ({self.rows_checked} rows compared)"
        return "LEAK: " + "; ".join(f"{c}: {v['n_bad']} rows differ ({v['mode']}, first {v['first_bad_date']})"
                                    for c, v in self.columns.items())

    def require(self):
        if not self.passed:
            raise FailClosed(self.summary())
        return self


def _as_frame(o):
    if isinstance(o, pd.Series):
        return o.to_frame(o.name if o.name is not None else "value")
    return o


def future_invariance(fn, data, cuts=None, n_cuts=4, seed=0, atol=1e-9, rtol=1e-7, modes=("real", "scrambled"),
                      mask_fns=None) -> InvarianceReport:
    """fn(data) -> DataFrame/Series indexed by date (or (date, ticker)); `mask_fns` {name: f(df, as_of) -> bool mask of
    the rows that are in the future} for frames without a date index. For each cut date the feature computed on
    data[:cut] must equal, at every date <= cut, the feature computed with the real future present and with the future
    scrambled. This catches negative shifts, centred windows, full-sample normalisation and any other peeking."""
    frames = data if isinstance(data, dict) else {"_": data}
    dates = None
    for f in frames.values():
        d = _index_dates(f)
        if d is not None and len(d):
            dates = d.unique().sort_values() if dates is None else dates.union(d.unique())
    rep = InvarianceReport()
    if dates is None or len(dates) < 3:
        return rep
    if cuts is None:
        qs = np.linspace(0.3, 0.9, n_cuts)
        cuts = sorted({dates[int(q * (len(dates) - 1))] for q in qs})
    cuts = [_ts(c) for c in cuts]
    rep.cuts = cuts
    call = (lambda d: fn(d["_"])) if not isinstance(data, dict) else fn
    full = _as_frame(call(frames))
    rng = np.random.default_rng(seed)
    for cut in cuts:
        clipped = _as_frame(call(truncate_after(frames, cut, mask_fns)))
        cand = {"real": full}
        if "scrambled" in modes:
            cand["scrambled"] = _as_frame(call(scramble_after(frames, cut, rng, mask_fns)))
        cand = {k: v for k, v in cand.items() if k in modes}
        cd = _index_dates(clipped)
        rep.rows_checked += len(clipped)
        for mode, other in cand.items():
            other_c = other.reindex(clipped.index)
            for col in clipped.columns:
                if col not in other_c.columns:
                    rep.columns.setdefault(str(col), dict(n_bad=len(clipped), max_abs_diff=np.inf,
                                                          first_bad_date=None, mode=mode))
                    continue
                a, b = clipped[col], other_c[col]
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    av, bv = a.values.astype(float), b.values.astype(float)
                    bad = ~np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
                    md = float(np.nanmax(np.abs(av - bv))) if bad.any() and np.isfinite(av - bv).any() else 0.0
                else:
                    bad = (a.values != b.values) & ~(pd.isna(a.values) & pd.isna(b.values))
                    md = 0.0
                if bad.any():
                    first = cd[np.flatnonzero(bad)[0]] if cd is not None else None
                    cur = rep.columns.get(str(col))
                    n = int(bad.sum())
                    if cur is None or n > cur["n_bad"]:
                        rep.columns[str(col)] = dict(n_bad=n, max_abs_diff=md, mode=mode,
                                                     first_bad_date=None if first is None else str(first.date()))
    return rep


def implausible_ic(X: pd.DataFrame, y: pd.Series, cap: float = 0.15, min_dates: int = 20, min_names: int = 10,
                   z: float = 2.33) -> pd.DataFrame:
    """Per-feature mean daily rank-IC against the label. Honest single features on liquid stocks sit far below 0.1; a
    mean IC above `cap` is a feature that already knows the answer. Returns every feature with mean_ic, t, n, ic_lo and flag.
    The flag needs |mean IC| to exceed `cap` BEYOND sampling noise (ic_lo = |mean| - z * se > cap, one-sided 1%): on a small
    panel (40 dates x 20 names, se ~0.036) an honest IC of 0.1 crossed 0.15 by chance on 3 of 60 seeds (C75 section 4,
    'reference-context IC false alarms'). A real leak (IC 0.3+ with se of a few hundredths, or a copy with se 0) still flags."""
    yy = y.reindex(X.index)
    rk = X.groupby(level=0).rank(pct=True)
    ry = yy.groupby(level=0).rank(pct=True)
    rows = []
    for c in X.columns:
        df = pd.DataFrame({"a": rk[c], "b": ry}).dropna()
        g = df.groupby(level=0)
        cnt = g.size()
        good = cnt[cnt >= min_names].index
        if len(good) == 0:
            rows.append(dict(feature=c, mean_ic=np.nan, t=np.nan, n_dates=0, ic_lo=np.nan, flag=False))
            continue
        ic = g.apply(lambda d: d["a"].corr(d["b"])).loc[good].dropna()
        sd = ic.std(ddof=1) if len(ic) > 1 else np.nan
        t = ic.mean() / (sd / np.sqrt(len(ic))) if sd and sd > 0 else (np.inf if len(ic) > 1 and ic.mean() != 0 else np.nan)
        se = sd / np.sqrt(len(ic)) if len(ic) > 1 and np.isfinite(sd) else np.nan
        lo = abs(float(ic.mean())) - z * se if np.isfinite(se) else np.nan
        rows.append(dict(feature=c, mean_ic=float(ic.mean()), t=float(t) if np.isfinite(t) else t,
                         n_dates=int(len(ic)), ic_lo=float(lo),
                         flag=bool(len(ic) >= min_dates and abs(ic.mean()) > cap and np.isfinite(lo) and lo > cap)))
    return pd.DataFrame(rows).set_index("feature")


# --------------------------------------------------------------------------------------------------------------------
# Diagnostics used on real caches (era / type breakdowns)
# --------------------------------------------------------------------------------------------------------------------
def lag_profile(effective, avail, calendar: Calendar, by=None, long_lag: int = 2) -> pd.DataFrame:
    """Publication-lag distribution in sessions between when something happened (`effective`) and when it became
    public (`avail`), broken down by any label (era, form type...). `negative` counts records known BEFORE they happened
    (impossible), `late` those published more than `long_lag` sessions after: those are what a join on the effective
    date leaks. Columns: n, negative, negative_share, median, p95, late_share, mean_calendar_days."""
    cols = ["n", "negative", "negative_share", "median", "p95", "late_share", "mean_calendar_days"]
    e = pd.Series(pd.DatetimeIndex(pd.to_datetime(effective)).normalize())
    a = pd.Series(pd.DatetimeIndex(pd.to_datetime(avail)).normalize())
    if len(e) != len(a):
        raise ValueError("effective and avail differ in length")
    grp = pd.Series(np.asarray(by) if by is not None else ["all"] * len(e))
    if len(grp) != len(e):
        raise ValueError("by must match the length of effective")
    ok = (e.notna() & a.notna()).values
    e, a, grp = e[ok].reset_index(drop=True), a[ok].reset_index(drop=True), grp[ok].reset_index(drop=True)
    if len(e) == 0:
        return pd.DataFrame(columns=cols)
    ev, av = e.values.astype("datetime64[D]"), a.values.astype("datetime64[D]")
    hol = calendar._hol
    fwd = av >= ev
    sess = np.zeros(len(e))
    sess[fwd] = np.busday_count(ev[fwd], av[fwd], holidays=hol)
    sess[~fwd] = -np.busday_count(av[~fwd], ev[~fwd], holidays=hol)
    cal = (a - e).dt.days.values
    df = pd.DataFrame({"g": grp, "s": sess, "cal": cal, "neg": cal < 0})
    out = df.groupby("g").agg(n=("s", "size"), negative=("neg", "sum"),
                              median=("s", "median"), p95=("s", lambda x: float(np.percentile(x, 95))),
                              late_share=("s", lambda x: float((x > long_lag).mean())),
                              mean_calendar_days=("cal", "mean"))
    out["negative"] = out["negative"].astype(int)
    out["negative_share"] = out["negative"] / out["n"]
    return out[cols]


def visibility_gap(system_visible, pit_visible, by=None, calendar: Calendar | None = None) -> pd.DataFrame:
    """Compare when a pipeline THINKS each record is usable (`system_visible`) with when it really was (`pit_visible`).
    early = the system saw it before it existed publicly (a leak); early_sessions_mean is how many sessions early on
    average. late = conservative. Rows with NaT on either side are skipped."""
    cols = ["n", "early", "early_share", "late", "early_sessions_mean", "early_days_max"]
    s = pd.Series(pd.DatetimeIndex(pd.to_datetime(system_visible)).normalize())
    p = pd.Series(pd.DatetimeIndex(pd.to_datetime(pit_visible)).normalize())
    if len(s) != len(p):
        raise ValueError("system_visible and pit_visible differ in length")
    grp = pd.Series(np.asarray(by) if by is not None else ["all"] * len(s))
    ok = (s.notna() & p.notna()).values
    s, p, grp = s[ok].reset_index(drop=True), p[ok].reset_index(drop=True), grp[ok].reset_index(drop=True)
    if len(s) == 0:
        return pd.DataFrame(columns=cols)
    hol = calendar._hol if calendar is not None else np.array([], dtype="datetime64[D]")
    sv, pv = s.values.astype("datetime64[D]"), p.values.astype("datetime64[D]")
    early = sv < pv
    gap = np.zeros(len(s))
    gap[early] = np.busday_count(sv[early], pv[early], holidays=hol)
    df = pd.DataFrame({"g": grp, "early": early, "late": sv > pv, "gap": gap,
                       "days": (p - s).dt.days.clip(lower=0).values})
    out = df.groupby("g").agg(n=("early", "size"), early=("early", "sum"), late=("late", "sum"),
                              early_sessions_mean=("gap", lambda x: float(x[x > 0].mean()) if (x > 0).any() else 0.0),
                              early_days_max=("days", "max"))
    out["early_share"] = out["early"] / out["n"]
    return out[cols]


def entry_gap_profile(opens: pd.DataFrame, closes: pd.DataFrame, by_year: bool = True) -> pd.DataFrame:
    """A label that enters at close t credits the holder with nothing between close t and open t+1, but a decision
    made at close t can only fill at open t+1. The overnight gap log(O[t+1]/C[t]) is the return a close-entry label
    ignores. Per year (or overall): mean, mean absolute, std, and share of |gap| > 1%, across all names/dates."""
    cols = ["n", "mean", "mean_abs", "std", "share_abs_gt_1pct"]
    gap = np.log(opens.shift(-1) / closes).replace([np.inf, -np.inf], np.nan)
    if len(gap) == 0:
        return pd.DataFrame(columns=cols)
    key = gap.index.year if by_year else np.zeros(len(gap), dtype=int)
    rows = {}
    for yr, sub in gap.groupby(key):
        v = sub.values[np.isfinite(sub.values)]
        if len(v):
            rows[yr] = dict(n=int(len(v)), mean=float(v.mean()), mean_abs=float(np.abs(v).mean()),
                            std=float(v.std()), share_abs_gt_1pct=float((np.abs(v) > 0.01).mean()))
    return pd.DataFrame(rows, index=cols).T if rows else pd.DataFrame(columns=cols)


def full_check(store: PITStore, as_of, pipeline, seed: int = 0) -> Report:
    """One call for the Bible's Phase-1 gate on a store: integrity + fence tripwires + scramble. Returns a Report."""
    rep = store.validate()
    rep.name = "pit.full_check"
    sc = future_scramble_store(pipeline, store, as_of, seed)
    if not sc.passed:
        rep.add("error", "future_scramble", "pipeline", sc.summary())
    for r in store.log.leaks():
        rep.add("error", "leaked_access", r["source"], f"{r['op']}")
    if not store.log.verify():
        rep.add("error", "log_tampered", "audit_log")
    return rep
