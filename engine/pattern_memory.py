"""Bible phases 4, 5, 9 - TIMELINE PATTERN MEMORY (canon C55, C56, C57, C58, C59, C60).

What it is: an append-only, hash-chained store of dated evidence about identity-free patterns (feature / quintile /
context conditions - never tickers) that only ever accumulates, never forgets, and can never show the trader anything
that had not yet happened.

  store      every observation = (pattern key, obs_date, mature_date = obs + label horizon in sessions, effect, n, t,
             context vector, source run id). Runs and candidate "tries" are chained into the same log. Nothing is edited.
  view       `view(real_now, ctx_now)` is the ONLY trader-facing read. It uses observations with mature_date < real_now
             (strictly) and returns pattern keys, pooled statistics and a relevance weight - never a date (C55/C58).
  relevance  C59: a pattern that stayed consistent across many DISTINCT periods (same sign, adequate t in most, no
             significant break, not failing lately) is UNIVERSAL: weight independent of era distance. A pattern that did
             not stay consistent is either LOCAL (weight from time distance, context similarity, recency of confirmation,
             number of periods it held) or DISREGARDED for this year (weight 0: broke, lapsed, or went stale). Nothing is
             deleted: disregarded patterns stay in the store and requalify automatically if matured evidence restores them.
  intra-year C60: relevance also moves WITHIN a year. The latest few matured observations (whatever their calendar
             year) scale the weight down continuously as evidence of a break arrives, and switch the pattern off when the
             recent run is significantly against it - a pattern found in January that stops working by June fades by June.
  gate       `view(..., gate=f)`: f(key, ctx_now, TimelineSummary) -> multiplier in [0, 1], applied after the consistency
             rule (B27's reliability model: switch a pattern off in conditions where it is predicted unreliable, without
             deleting it). The hook sees only a date-free `TimelineSummary` built from matured observations, and every
             input is scanned before the call.
  tries      every candidate ever tried is counted in a cumulative ledger; the multiple-testing correction in `view`
             uses that cumulative count, so re-searching the same data cannot promote best-of-many noise (C57).
  analytics  `Analytics` (owner-side, dated, still bounded by `as_of`) shows when a pattern was first noticed, where it
             held and failed along real time, and its relevance to a given year.
  guards     `check_no_future` raises LeakError if anything with mature_date >= real_now reaches the trader;
             `scan_for_dates` raises if a date is inside anything handed to the trader; ingestion refuses observations
             whose labels were not yet realised at the run date, or that carry a stock identity.

The same evidence seen twice (same key, same obs_date, e.g. the same year rerun 500 times) is de-duplicated to the latest
record: repetition does not inflate n or t. What grows with reruns is the set of candidates explored and the timeline."""
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from math import erfc, exp, sqrt, tanh

import numpy as np
import pandas as pd

from .pattern_bank import file_lock

SCHEMA = 1
PARAMS = {
    "horizon": 5,              # label horizon in sessions (mature = obs + horizon sessions)
    "n_universal": 6,          # distinct periods (calendar years) needed to be universal
    "held_share": 0.75,        # share of periods in which the pattern held in its pooled direction
    "t_hold": 1.0,             # a period "held" if signed period t >= this
    "t_break": 2.0,            # a period "broke" if opposite-signed with |t| >= this
    "t_fail": 1.5,             # recent evidence opposing the pattern this strongly disregards it
    "recent_k": 2,             # how many latest periods count as "recent"
    "t_scale": 3.0,            # strength = tanh(pooled_t / t_scale)
    "t_min_local": 1.5,        # local patterns need at least this pooled t
    "tau_time_years": 3.0,     # e-folding time distance for local patterns
    "ctx_bandwidth": 1.0,      # gaussian bandwidth in standardised context units
    "sim_fail": 0.6,           # context similarity above which a contrary observation counts as a failure "here"
    "min_weight": 0.02,        # local weights under this are stale -> disregarded
    "fdr_alpha": 0.10,         # BH level over the cumulative try count
    "use_fdr": True,
    "intra_k": 3,              # latest matured observations that define the within-year state of a pattern
    "lock_timeout_s": 30.0,
}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR = re.compile(r"(?<![A-Za-z0-9_.])(?:19|20)\d{2}(?![A-Za-z0-9_])")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class MemoryError_(RuntimeError):
    pass


class LeakError(MemoryError_):
    """Something dated at or after real_now (or a date itself) was about to reach the trader, or a write claimed evidence
    from a date that had not occurred at the run's own date."""


class ChainCorrupt(MemoryError_):
    pass


class KeyError_(MemoryError_):
    """A pattern key that is not identity-free (ticker / date inside)."""


# ---------------------------------------------------------------- small pure helpers
def _day(x):
    return pd.Timestamp(x).normalize()


def _iso(x):
    return _day(x).strftime("%Y-%m-%d")


def add_sessions(d, n, holidays=None):
    """Session date `n` sessions after `d` (weekdays; optional holiday list). n >= 1."""
    hol = np.array(sorted({_iso(h) for h in (holidays or [])}), dtype="datetime64[D]")
    return pd.Timestamp(np.busday_offset(np.datetime64(_iso(d)), int(n), roll="forward", holidays=hol))


def p_two_sided(t):
    return float(erfc(abs(float(t)) / sqrt(2.0)))


def bh_adjust(p, m_total):
    """Benjamini-Hochberg q-values when `m_total` >= len(p) hypotheses were tried in all: the untested-in-this-batch
    tries are conservatively counted as failures (p = 1), which is what best-of-many noise is."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    m = max(int(m_total), p.size)
    order = np.argsort(p, kind="mergesort")
    q = p[order] * m / (np.arange(p.size) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def expected_best_null_t(m):
    """Approximate largest |t| among m independent null candidates - the bar a 'best of many' has to beat."""
    return float(sqrt(2.0 * np.log(max(int(m), 2))))


def validate_key(key, forbidden=()):
    """A key is a condition string over features (e.g. 'ret_20d_q5 & m_vix_q4'). Reject anything that names a stock or a
    date - the memory is about patterns, never identities (C57) and never about calendar (C55)."""
    if not isinstance(key, str) or not key.strip():
        raise KeyError_("empty pattern key")
    if _ISO.search(key) or _YEAR.search(key):
        raise KeyError_(f"pattern key contains a date: {key!r}")
    bad = {str(f).upper() for f in forbidden}
    for tok in _TOKEN.findall(key):
        if tok.upper() in bad and tok.upper() not in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            raise KeyError_(f"pattern key names a forbidden identity {tok!r}: {key!r}")
        if tok.isalpha() and tok.isupper() and len(tok) <= 5:
            raise KeyError_(f"pattern key looks like a ticker {tok!r}: {key!r}")
    return key


def scan_for_dates(obj, _path="view"):
    """Raise LeakError if any date-like value sits anywhere inside `obj` (dataclasses, dicts, lists, frames, strings)."""
    if obj is None or isinstance(obj, (bool, int, float, np.integer, np.floating, np.bool_)):
        return
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64, pd.Timedelta)):
        raise LeakError(f"date object at {_path}")
    if isinstance(obj, str):
        if _ISO.search(obj) or _YEAR.search(obj):
            raise LeakError(f"date-like string at {_path}: {obj!r}")
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            scan_for_dates(getattr(obj, f.name), f"{_path}.{f.name}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan_for_dates(k, f"{_path}.<key>")
            scan_for_dates(v, f"{_path}[{k!r}]")
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            scan_for_dates(v, f"{_path}[{i}]")
        return
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.DatetimeIndex) or any(str(t).startswith("datetime") for t in obj.dtypes.astype(str)):
            raise LeakError(f"datetime data at {_path}")
        scan_for_dates(obj.columns.tolist(), _path + ".columns")
        scan_for_dates(obj.astype(object).values.ravel().tolist(), _path + ".values")
        return
    if isinstance(obj, pd.Series):
        scan_for_dates(obj.to_frame(), _path)
        return
    if isinstance(obj, np.ndarray):
        scan_for_dates(obj.ravel().tolist(), _path)
        return
    raise LeakError(f"unscannable type {type(obj).__name__} at {_path}")


def check_no_future(records, real_now):
    """The enforcement point: every record that reaches the trader must have matured strictly before `real_now`."""
    now = _day(real_now)
    late = [r for r in records if pd.Timestamp(r["mature_date"]) >= now]
    if late:
        raise LeakError(f"{len(late)} observation(s) with mature_date >= real_now reached the trader "
                        f"(first: key={late[0]['key']!r})")
    return True


# ---------------------------------------------------------------- trader-facing result types
@dataclass(frozen=True)
class PatternWeight:
    key: str
    weight: float                 # 0 when disregarded; otherwise relevance in (0, 1]
    direction: int                # +1 / -1: sign of the pooled effect
    mode: str                     # universal | local | disregarded
    reason: str                   # short cause; never contains a date
    pooled_effect: float
    pooled_t: float
    n_periods: int                # distinct periods that carried evidence
    n_held: int                   # periods in which it held in its pooled direction
    n_obs: int
    q_value: float
    components: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineSummary:
    """Date-free matured timeline of one pattern for another module (B27's reliability model). Observation arrays are in
    chronological order; `period` maps each observation to an ordinal calendar-period index (0 = oldest period seen).
    `ctx` holds the market context of each observation (NaN where unrecorded), columns named by `ctx_cols`. Effects and t
    are in the pattern's own pooled direction (positive = it worked). No date, run id or count of runs is present."""
    key: str
    direction: int
    effect: np.ndarray
    t: np.ndarray
    n: np.ndarray
    period: np.ndarray
    ctx_cols: tuple
    ctx: np.ndarray
    period_t: np.ndarray          # signed Stouffer t per period, oldest first
    period_effect: np.ndarray
    period_ctx: np.ndarray        # mean context per period (periods x ctx_cols)
    pooled_t: float


@dataclass(frozen=True)
class PatternView:
    weights: dict                 # key -> PatternWeight

    def active(self):
        return {k: w for k, w in self.weights.items() if w.weight > 0}

    def signed_weights(self):
        return {k: w.direction * w.weight for k, w in self.weights.items() if w.weight > 0}

    def __len__(self):
        return len(self.weights)

    def frame(self):
        cols = ["key", "weight", "direction", "mode", "reason", "pooled_effect", "pooled_t", "n_periods", "n_held", "n_obs",
                "q_value"]
        return pd.DataFrame([[getattr(w, c) for c in cols] for w in self.weights.values()], columns=cols)


# ---------------------------------------------------------------- the store
def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(prev, body):
    return hashlib.sha256((prev + _canon(body)).encode()).hexdigest()


def _num(x):
    x = float(x)
    if not np.isfinite(x):
        raise MemoryError_("non-finite number in a memory record")
    return x


class PatternMemory:
    """Append-only, hash-chained timeline store. Records are lines of `chain.jsonl` under `root`:
    {seq, prev, kind: obs|run, body, hash}. Open verifies the whole chain and fails closed (ChainCorrupt)."""

    def __init__(self, root, params=None, forbidden_identities=(), holidays=None):
        self.root = os.fspath(root)
        self.p = {**PARAMS, **(params or {})}
        self.forbidden = tuple(forbidden_identities)
        self.holidays = list(holidays or [])
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "chain.jsonl")
        self._lock = os.path.join(self.root, "chain.lock")
        self._recs = []              # parsed records in order
        self._offset = 0
        self._last = "0" * 64
        self.torn_tail = False
        self.last_audit = {}
        self._sync()

    # ----- persistence
    def _sync(self):
        """Read lines appended since our offset (by us or another process), verifying each link."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        if not data:
            return
        lines = data.split(b"\n")
        tail = lines.pop()                       # text after the last newline: empty when the file ends cleanly
        self.torn_tail = bool(tail)
        for raw in lines:
            if not raw.strip():
                self._offset += len(raw) + 1
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
                body, prev, seq, h = rec["body"], rec["prev"], rec["seq"], rec["hash"]
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                raise ChainCorrupt(f"unreadable line at seq {len(self._recs)}: {e}") from e
            if seq != len(self._recs) or prev != self._last or _hash(prev, {"kind": rec["kind"], **body}) != h:
                raise ChainCorrupt(f"chain broken at seq {seq}")
            self._recs.append(rec)
            self._last = h
            self._offset += len(raw) + 1

    def _append(self, kind, bodies):
        """Append several bodies under one lock (atomic per call: one write, one fsync)."""
        with file_lock(self._lock, self.p["lock_timeout_s"]):
            self._sync()
            if self.torn_tail:
                raise ChainCorrupt("torn tail (a writer died mid-line); repair the file before appending")
            lines, seq, prev = [], len(self._recs), self._last
            for b in bodies:
                h = _hash(prev, {"kind": kind, **b})
                lines.append(_canon({"seq": seq, "prev": prev, "kind": kind, "body": b, "hash": h}))
                seq, prev = seq + 1, h
            if lines:
                with open(self.path, "ab") as fh:
                    fh.write(("\n".join(lines) + "\n").encode())
                    fh.flush()
                    os.fsync(fh.fileno())
            self._sync()

    def verify(self):
        """Re-read the file from byte 0 and check every hash link, independent of the in-memory copy."""
        prev, n = "0" * 64, 0
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                raw = fh.read()
            for line in raw.split(b"\n")[:-1]:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                    ok = (rec["seq"] == n and rec["prev"] == prev and _hash(prev, {"kind": rec["kind"], **rec["body"]}) == rec["hash"])
                except (ValueError, KeyError, UnicodeDecodeError):
                    ok = False
                if not ok:
                    return {"ok": False, "records": n, "first_bad_seq": n}
                prev, n = rec["hash"], n + 1
        return {"ok": True, "records": n, "first_bad_seq": None, "head": prev}

    # ----- writing
    def add_observations(self, run_id, run_now, obs, n_tried=None):
        """Record a run's evidence. Each obs: key, obs_date, effect, n, t, optional ctx {m_col: value}, optional
        horizon (sessions). mature_date is computed here (obs + horizon sessions) and may never be later than run_now:
        a run cannot have used labels that had not been realised on the day it ran. `n_tried` (default = number of
        observations) is the number of candidates this run examined, feeding the cumulative multiple-testing count."""
        rn = _day(run_now)
        bodies = []
        for o in obs:
            key = validate_key(o["key"], self.forbidden)
            od = _day(o["obs_date"])
            hz = int(o.get("horizon", self.p["horizon"]))
            if hz < 1:
                raise MemoryError_("label horizon must be >= 1 session")
            md = add_sessions(od, hz, self.holidays)
            if o.get("mature_date") is not None:
                claimed = _day(o["mature_date"])
                if claimed < md:
                    raise LeakError(f"claimed mature_date precedes obs_date + {hz} sessions for {key!r}")
                md = claimed
            if md > rn:
                raise LeakError(f"observation for {key!r} uses labels that mature after the run date")
            ctx = {str(k): _num(v) for k, v in (o.get("ctx") or {}).items() if v is not None and np.isfinite(v)}
            bodies.append({"key": key, "obs_date": _iso(od), "mature_date": _iso(md), "horizon": hz,
                           "effect": _num(o["effect"]), "n": int(o["n"]), "t": _num(o["t"]), "ctx": ctx,
                           "run_id": str(run_id)})
        if bodies:
            self._append("obs", bodies)
        self._append("run", [{"run_id": str(run_id), "run_now": _iso(rn), "n_obs": len(bodies),
                              "n_tried": int(len(bodies) if n_tried is None else n_tried),
                              "keys": sorted({b["key"] for b in bodies})}])
        return len(bodies)

    def record_tries(self, run_id, run_now, n_tried, keys=()):
        """Count candidates that were tried without producing an observation (the losers). Never subtracted."""
        for k in keys:
            validate_key(k, self.forbidden)
        self._append("run", [{"run_id": str(run_id), "run_now": _iso(run_now), "n_obs": 0, "n_tried": int(n_tried),
                              "keys": sorted(set(keys))}])

    def ingest_miner(self, frame, run_id, run_now, ctx=None, horizon=None, n_tried=None, statuses=("active", "rescoped")):
        """Fold a PatternMiner.patterns frame in as one observation per surviving row, dated at the miner's `now`.
        Uses t_conf (the held-out confirmation half) as the run's t. The frame's full length is the try count unless
        `n_tried` says the miner examined more (its report['tested'])."""
        if frame is None or len(frame) == 0:
            self.record_tries(run_id, run_now, n_tried or 0)
            return 0
        rows = frame[frame["status"].isin(statuses)] if "status" in frame else frame
        hz = int(horizon or self.p["horizon"])
        # the miner's window ended `horizon` sessions before run_now at the latest, so its labels are mature by run_now
        od = _day(run_now) - pd.tseries.offsets.BDay(hz)
        obs = [{"key": r["key_named"], "obs_date": od, "effect": float(r["effect"]), "n": int(r.get("n_eff", 1) or 1),
                "t": float(r["t_conf"]), "ctx": ctx, "horizon": hz}
               for _, r in rows.iterrows() if np.isfinite(r.get("t_conf", np.nan)) and np.isfinite(r.get("effect", np.nan))]
        return self.add_observations(run_id, run_now, obs, n_tried=len(frame) if n_tried is None else n_tried)

    def observe_panel(self, key, mask, y, run_id, run_now, ctx=None, freq="M", horizon=None, min_names=5, min_dates=3):
        """Turn one candidate's mask over a (date, ticker) panel into per-period observations and store them.
        Per date the effect is mean(y | mask) - mean(y); dates are thinned to every `horizon`-th so the labels do not
        overlap; a period's t is the t-stat of those per-date effects. Only dates with date + horizon <= run_now are used."""
        hz = int(horizon or self.p["horizon"])
        rn = _day(run_now)
        d = y.index.get_level_values(0)
        rows = pd.DataFrame({"y": y.values, "m": mask.reindex(y.index).fillna(False).values.astype(bool)}, index=d)
        per = []
        for date, g in rows.groupby(level=0):
            if add_sessions(date, hz, self.holidays) > rn:
                continue
            sel = g["y"][g["m"]]
            if len(sel) >= min_names:
                per.append((date, float(sel.mean() - g["y"].mean())))
        if not per:
            return 0
        s = pd.Series({d_: v for d_, v in per}).sort_index()
        obs = []
        for _, grp in s.groupby(s.index.to_period(freq)):
            grp = grp.iloc[::hz]
            if len(grp) < min_dates or grp.std(ddof=1) == 0 or not np.isfinite(grp.std(ddof=1)):
                continue
            t = float(grp.mean() / (grp.std(ddof=1) / sqrt(len(grp))))
            c = None
            if ctx is not None:
                cc = ctx.loc[ctx.index.isin(grp.index)]
                c = {k: float(v) for k, v in cc.mean().items() if np.isfinite(v)}
            obs.append({"key": key, "obs_date": grp.index[-1], "effect": float(grp.mean()), "n": len(grp), "t": t,
                        "ctx": c, "horizon": hz})
        return self.add_observations(run_id, run_now, obs)

    # ----- raw reads (owner side)
    def refresh(self):
        self._sync()

    def records(self, kind="obs"):
        return [dict(r["body"]) for r in self._recs if r["kind"] == kind]

    def keys(self):
        return sorted({r["key"] for r in self.records("obs")})

    def __len__(self):
        return len(self._recs)

    def _matured(self, real_now):
        """Observations with mature_date < real_now (strict), de-duplicated to the latest record per (key, obs_date)."""
        now = _day(real_now)
        latest = {}
        for r in self._recs:
            if r["kind"] == "obs" and pd.Timestamp(r["body"]["mature_date"]) < now:
                latest[(r["body"]["key"], r["body"]["obs_date"])] = r["body"]
        return list(latest.values())

    # ----- cumulative try accounting (C57)
    def cumulative_tries(self):
        runs = [r["body"] for r in self._recs if r["kind"] == "run"]
        seen, growth, tot = set(), [], 0
        for b in runs:
            tot += b["n_tried"]
            seen |= set(b["keys"])
            growth.append({"run_id": b["run_id"], "n_tried": b["n_tried"], "cumulative": tot, "distinct_keys": len(seen)})
        return {"total_tries": tot, "distinct_keys": len(seen), "runs": len(runs),
                "expected_best_null_t": expected_best_null_t(max(tot, 2)), "growth": growth}

    # ----- the trader-facing read
    def timeline_summary(self, key, real_now):
        """Date-free matured timeline of one pattern (see TimelineSummary); None if nothing has matured for it."""
        self._sync()
        mat = self._matured(real_now)
        check_no_future(mat, real_now)
        obs = [m for m in mat if m["key"] == key]
        return _summary(key, obs) if obs else None

    def view(self, real_now, ctx_now=None, gate=None):
        """Aggregated per-pattern relevance using only observations that matured strictly before `real_now`.
        `gate(key, ctx_now, TimelineSummary) -> multiplier in [0, 1]` is applied after the consistency rule to every
        pattern that still has weight; a 0 marks it 'gated' (kept, switched off here), never deleted."""
        self._sync()
        now = _day(real_now)
        mat = self._matured(now)
        check_no_future(mat, now)
        self.last_audit = {"n_used": len(mat), "max_mature": max((m["mature_date"] for m in mat), default=None),
                           "real_now": _iso(now)}
        by_key = {}
        for m in mat:
            by_key.setdefault(m["key"], []).append(m)
        scale = self._ctx_scale(mat)
        stats = {k: _pool(k, v, now, ctx_now, scale, self.p) for k, v in by_key.items()}
        keys = sorted(stats)
        pv = [p_two_sided(stats[k]["pooled_t"] * sqrt(1.0)) for k in keys]
        qv = bh_adjust(pv, self.cumulative_tries()["total_tries"]) if self.p["use_fdr"] else np.zeros(len(keys))
        if gate is not None:
            scan_for_dates(ctx_now if ctx_now is not None else {}, "ctx_now")
        out, gate_calls = {}, 0
        for k, q in zip(keys, qv):
            s = stats[k]
            mode, reason, w, comp = s["mode"], s["reason"], s["weight"], dict(s["components"])
            if self.p["use_fdr"] and mode != "disregarded" and q > self.p["fdr_alpha"]:
                mode, reason, w = "disregarded", "not_significant_after_all_tries", 0.0
            if gate is not None and w > 0:
                summ = _summary(k, by_key[k])
                check_no_future([{"key": k, "mature_date": o["mature_date"]} for o in by_key[k]], now)
                scan_for_dates(summ, f"summary[{k}]")
                g = gate(k, dict(ctx_now) if ctx_now else {}, summ)
                gate_calls += 1
                if isinstance(g, (bool, np.bool_)) or not isinstance(g, (int, float, np.floating, np.integer))                         or not np.isfinite(g) or not 0.0 <= float(g) <= 1.0:
                    raise MemoryError_(f"gate for {k!r} returned {g!r}, not a multiplier in [0, 1]")
                comp["gate"] = float(g)
                w *= float(g)
                if g == 0:
                    mode, reason = "gated", "predicted_unreliable_here"
            out[k] = PatternWeight(k, float(w), s["direction"], mode, reason, s["pooled_effect"], s["pooled_t"],
                                   s["n_periods"], s["n_held"], s["n_obs"], float(q), comp)
        self.last_audit["gate_calls"] = gate_calls
        pv_ = PatternView(out)
        scan_for_dates(pv_)                       # C55: nothing dated may leave
        return pv_

    def _ctx_scale(self, mat):
        cols = {}
        for m in mat:
            for c, v in m["ctx"].items():
                cols.setdefault(c, []).append(v)
        return {c: (float(np.std(v)) if len(v) > 1 and np.std(v) > 1e-12 else 1.0) for c, v in cols.items()}


# ---------------------------------------------------------------- pooling and the C59 relevance rule
def _period_of(iso):
    return int(iso[:4])


def _ctx_sim(ctx_obs, ctx_now, scale, bw):
    if not ctx_now or not ctx_obs:
        return 0.5
    shared = [c for c in ctx_obs if c in ctx_now and np.isfinite(ctx_now[c])]
    if not shared:
        return 0.5
    d2 = np.mean([((ctx_obs[c] - ctx_now[c]) / scale.get(c, 1.0)) ** 2 for c in shared])
    return float(exp(-d2 / (2 * bw * bw)))


def _summary(key, obs):
    """Build the date-free TimelineSummary from matured observations of one key."""
    obs = sorted(obs, key=lambda o: o["obs_date"])
    per = periods_of(obs)
    yrs = sorted(per)
    direction = 1 if sum(per[y]["t"] for y in yrs) >= 0 else -1
    cols = tuple(sorted({c for o in obs for c in o["ctx"]}))
    ctx = np.array([[o["ctx"].get(c, np.nan) for c in cols] for o in obs], float).reshape(len(obs), len(cols))
    pidx = np.array([yrs.index(_period_of(o["obs_date"])) for o in obs], int)
    pctx = np.full((len(yrs), len(cols)), np.nan)
    for i in range(len(yrs)):
        blk = ctx[pidx == i]
        if blk.size and (~np.isnan(blk)).any(axis=0).any():
            with np.errstate(all="ignore"):
                pctx[i] = np.nanmean(blk, axis=0)
    pt = np.array([direction * per[y]["t"] for y in yrs])
    return TimelineSummary(key, direction, direction * np.array([o["effect"] for o in obs]),
                           direction * np.array([o["t"] for o in obs]), np.array([o["n"] for o in obs], int), pidx, cols,
                           ctx, pt, direction * np.array([per[y]["effect"] for y in yrs]), pctx,
                           float(abs(sum(per[y]["t"] for y in yrs)) / sqrt(len(yrs))))


def periods_of(obs):
    """Group observations by calendar year of obs_date -> {year: {t, effect, n, k}} (Stouffer within a period with
    sqrt(n) weights; effect is the n-weighted mean)."""
    g = {}
    for o in obs:
        g.setdefault(_period_of(o["obs_date"]), []).append(o)
    out = {}
    for y, v in sorted(g.items()):
        w = np.array([sqrt(max(o["n"], 1)) for o in v])
        t = np.array([o["t"] for o in v])
        e = np.array([o["effect"] for o in v])
        nn = np.array([max(o["n"], 1) for o in v], float)
        out[y] = {"t": float((w * t).sum() / sqrt((w * w).sum())), "effect": float((nn * e).sum() / nn.sum()),
                  "n": int(nn.sum()), "k": len(v)}
    return out


def _pool(key, obs, now, ctx_now, scale, P):
    """Pooled statistics and the C59 decision for one pattern from matured observations only."""
    per = periods_of(obs)
    yrs = sorted(per)
    ts = np.array([per[y]["t"] for y in yrs])
    npd = len(yrs)
    signed_sum = float(ts.sum())
    direction = 1 if signed_sum >= 0 else -1
    st = direction * ts                                        # signed period t: positive = held
    pooled_t = float(abs(signed_sum) / sqrt(npd))
    nn = np.array([per[y]["n"] for y in yrs], float)
    pooled_eff = float((nn * np.array([per[y]["effect"] for y in yrs])).sum() / nn.sum())
    held = st >= P["t_hold"]
    broke = st <= -P["t_break"]
    n_held = int(held.sum())
    held_share = n_held / npd
    kk = min(P["recent_k"], npd)
    recent_t = float(st[-kk:].sum() / sqrt(kk))
    base = {"direction": direction, "pooled_effect": pooled_eff, "pooled_t": pooled_t, "n_periods": npd,
            "n_held": n_held, "n_obs": len(obs), "components": {"held_share": float(held_share), "recent_t": recent_t}}

    def out(mode, reason, weight, **comp):
        return {**base, "mode": mode, "reason": reason, "weight": float(weight),
                "components": {**base["components"], **{k: float(v) for k, v in comp.items()}}}

    now_year = now.year
    sims = np.array([_ctx_sim(o["ctx"], ctx_now, scale, P["ctx_bandwidth"]) for o in obs])
    contrary = np.array([direction * o["t"] <= -P["t_fail"] for o in obs])
    # --- disregard rules (consistency lost) - checked on matured evidence only
    if pooled_t < P["t_min_local"]:
        return out("disregarded", "weak_or_inconsistent", 0.0)
    if npd >= 2 and recent_t <= -P["t_fail"]:
        return out("disregarded", "failed_recently", 0.0)
    if bool((contrary & (sims >= P["sim_fail"])).any()) and ctx_now:
        return out("disregarded", "failed_in_similar_context", 0.0)
    # --- within-year state (C60): the latest k matured observations, whatever calendar year they fall in
    recent_obs = sorted(obs, key=lambda o: o["obs_date"])[-P["intra_k"]:]
    intra_t = float(direction * sum(o["t"] for o in recent_obs) / sqrt(len(recent_obs)))
    if len(obs) >= P["intra_k"] and intra_t <= -P["t_fail"]:
        return out("disregarded", "broke_within_year", 0.0, intra_t=intra_t)
    fresh = 0.5 + 0.5 * tanh(intra_t / P["t_scale"]) if len(obs) >= P["intra_k"] else 1.0
    strength = tanh(pooled_t / P["t_scale"])
    universal = (npd >= P["n_universal"] and held_share >= P["held_share"] and not bool(broke.any()) and recent_t > 0
                 and n_held >= P["n_universal"])
    if universal:
        w = strength * held_share * min(1.0, npd / (2.0 * P["n_universal"]) + 0.5) * fresh
        return out("universal", "consistent_across_periods", min(1.0, w), strength=strength, fresh=fresh, intra_t=intra_t)
    if npd >= 2 and recent_t < P["t_hold"] and (st[-kk:] < P["t_hold"]).all():
        return out("disregarded", "lapsed", 0.0)
    if bool(broke.any()) and npd >= 2 and n_held / npd < P["held_share"] and int(held[-1]) == 0:
        return out("disregarded", "broke_and_not_restored", 0.0)
    # --- local relevance: time distance, context similarity, recency of confirmation, number of periods held
    ev = np.clip(np.array([direction * o["t"] for o in obs]), 0.0, 4.0)
    age = np.array([max((now - pd.Timestamp(o["obs_date"])).days, 0) / 365.25 for o in obs])
    tau = P["tau_time_years"]
    if ev.sum() <= 0:
        return out("disregarded", "no_supporting_evidence", 0.0)
    time_k = float((ev * np.exp(-age / tau)).sum() / ev.sum())
    ctx_k = float((ev * sims).sum() / ev.sum())
    conf_age = float(age[ev >= P["t_hold"]].min()) if bool((ev >= P["t_hold"]).any()) else float(age.min())
    rec_k = float(exp(-conf_age / tau))
    persist = 1.0 - exp(-max(n_held, 0) / 3.0)
    w = strength * time_k * (0.5 + 0.5 * ctx_k) * (0.5 + 0.5 * rec_k) * persist * fresh
    if w < P["min_weight"]:
        return out("disregarded", "stale_for_this_period", 0.0, time=time_k, ctx=ctx_k, recency=rec_k)
    return out("local", "held_in_similar_periods", w, strength=strength, time=time_k, ctx=ctx_k, recency=rec_k,
               persist=persist, fresh=fresh, intra_t=intra_t)


# ---------------------------------------------------------------- owner-side analytics (dated; still bounded by as_of)
class Analytics:
    """Timeline analytics. These return dates, so they are for the owner's reports, never for the trader; they are still
    bounded by `as_of` so an analysis of an earlier moment cannot read later evidence either."""

    def __init__(self, mem):
        self.mem = mem

    def _obs(self, key, as_of):
        return sorted([o for o in self.mem._matured(as_of) if o["key"] == key], key=lambda o: o["obs_date"])

    def first_noticed(self, key, as_of):
        obs = self._obs(key, as_of)
        if not obs:
            return None
        runs = {}
        for r in self.mem._recs:
            if r["kind"] == "obs" and r["body"]["key"] == key:
                runs.setdefault(r["body"]["run_id"], r["seq"])
        first_run = min(runs, key=runs.get) if runs else None
        return {"first_obs_date": obs[0]["obs_date"], "first_mature_date": obs[0]["mature_date"], "first_run": first_run}

    def timeline(self, key, as_of):
        """Per calendar year: n observations, pooled effect and t, and whether it held / broke / was flat."""
        obs = self._obs(key, as_of)
        if not obs:
            return pd.DataFrame(columns=["year", "k", "n", "effect", "t", "status"])
        per = periods_of(obs)
        sign = 1 if sum(v["t"] for v in per.values()) >= 0 else -1
        P = self.mem.p
        rows = []
        for y, v in per.items():
            st = sign * v["t"]
            rows.append({"year": y, "k": v["k"], "n": v["n"], "effect": v["effect"], "t": v["t"],
                         "status": "held" if st >= P["t_hold"] else "broke" if st <= -P["t_break"] else
                         "opposed" if st < 0 else "flat"})
        return pd.DataFrame(rows)

    def held_failed(self, key, as_of):
        tl = self.timeline(key, as_of)
        return {"held_years": tl.loc[tl["status"] == "held", "year"].tolist() if len(tl) else [],
                "failed_years": tl.loc[tl["status"].isin(["broke", "opposed"]), "year"].tolist() if len(tl) else []}

    def relevance_to_year(self, key, year, ctx_now=None, when="start"):
        """The weight the trader would see for `key` at the start (or, with when='end', end) of `year`."""
        real_now = pd.Timestamp(year, 1, 1) if when == "start" else pd.Timestamp(year, 12, 31)
        v = self.mem.view(real_now, ctx_now).weights.get(key)
        return None if v is None else v.weight

    def report(self, as_of, ctx_now=None):
        """One row per pattern as of `as_of`: mode, weight, first noticed, years held / failed."""
        v = self.mem.view(as_of, ctx_now).weights
        rows = []
        for k, w in v.items():
            fn = self.first_noticed(k, as_of)
            hf = self.held_failed(k, as_of)
            rows.append({"key": k, "mode": w.mode, "reason": w.reason, "weight": w.weight, "pooled_t": w.pooled_t,
                         "n_periods": w.n_periods, "first_noticed": fn["first_obs_date"] if fn else None,
                         "held_years": hf["held_years"], "failed_years": hf["failed_years"]})
        return pd.DataFrame(rows).sort_values(["weight", "key"], ascending=[False, True]).reset_index(drop=True) \
            if rows else pd.DataFrame(columns=["key", "mode", "reason", "weight", "pooled_t", "n_periods", "first_noticed",
                                               "held_years", "failed_years"])


def default_root():
    from . import config as K
    return os.path.join(K.STATE, "pattern_memory")
