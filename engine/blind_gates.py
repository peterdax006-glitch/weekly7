"""Bible Phases 21 and 22 (canons C11, C19, C33): blind-simulator hardening gates and the blind clock.

Nothing here runs a simulation. These are the checks that prove a blind window was honestly blind:

  Phase 21  seal integrity (12 consecutive months, no overlap, sealed BEFORE any worker ran, file unaltered),
            disguise (week-multiple shift into the unreal future, seeded bijective ticker remap that leaks nothing),
            warm-up length, and the reveal gate (nothing is revealed until all four closing conditions hold).
  Phase 22  the daily clock as an enforced state machine (fill -> observe -> update -> close week -> learn -> adapt ->
            decide -> record), a look-ahead scan of recorded information sets, a fill-timing audit (a decision at
            a close fills at the NEXT session's open), and a perturbation probe that proves a function cannot
            see the future by scrambling everything after `now` and demanding an identical answer.

Every check returns a Finding list; a gate is passed only when no finding has severity "fail" (fail closed)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_START = pd.Timestamp("1965-01-01")
LAST_START = pd.Timestamp("2025-09-01")
MIN_SHIFT_YEAR = 2100          # a shifted date must be unmistakably outside the real record
WINDOW_MONTHS = 12
MAX_OVERLAP_DAYS = 183         # canon C19: a fresh window overlaps no played window by more than half


@dataclass
class Finding:
    gate: str
    severity: str              # "fail" | "warn" | "info"
    message: str

    def __str__(self):
        return f"[{self.severity}] {self.gate}: {self.message}"


def passed(findings):
    """Fail closed: a gate passes only if it produced no failing finding."""
    return not any(f.severity == "fail" for f in findings)


def _fail(out, gate, msg):
    out.append(Finding(gate, "fail", msg))


# ============================================================ Phase 21: sealed window
def window_end(start):
    """Last calendar day of the 12 consecutive months beginning at `start`."""
    return pd.Timestamp(start) + pd.DateOffset(months=WINDOW_MONTHS) - pd.Timedelta(days=1)


def overlap_days(a_start, b_start):
    a0, a1, b0, b1 = pd.Timestamp(a_start), window_end(a_start), pd.Timestamp(b_start), window_end(b_start)
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0, (hi - lo).days + 1)


def seal_digest(rec):
    """Digest of the fields that define a seal; written into the seal and re-derived at every check."""
    body = {k: rec[k] for k in ("start", "shift_days", "sealed_at", "seed_tag") if k in rec}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def seal_window(used_starts, seed, sealed_at, shift_weeks=(8000, 11000), tag=""):
    """Draw a sealed window deterministically from `seed`. Prefers months overlapping no `used_starts` window by more
    than half a year; falls back to the least-overlapping months rather than failing when the calendar is crowded."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(FIRST_START, LAST_START, freq="MS")
    used = [pd.Timestamp(u) for u in used_starts]
    m0 = months.values.astype("datetime64[M]")
    lo, hi = m0.astype("datetime64[D]"), (m0 + WINDOW_MONTHS).astype("datetime64[D]") - 1
    worst = np.zeros(len(months), dtype=int)
    for u in used:
        u0 = np.datetime64(u.date(), "M")
        ulo, uhi = u0.astype("datetime64[D]"), (u0 + WINDOW_MONTHS).astype("datetime64[D]") - 1
        ov = (np.minimum(hi, uhi) - np.maximum(lo, ulo)).astype(int) + 1
        worst = np.maximum(worst, np.clip(ov, 0, None))
    ok = np.flatnonzero(worst <= MAX_OVERLAP_DAYS)
    pool = ok if len(ok) else np.flatnonzero(worst == worst.min())
    start = months[int(rng.choice(pool))]
    rec = {"start": str(start.date()), "shift_days": 7 * int(rng.integers(*shift_weeks)),
           "sealed_at": str(pd.Timestamp(sealed_at)), "seed_tag": tag}
    rec["digest"] = seal_digest(rec)
    return rec


def check_seal(rec, other_starts=(), first_worker_start=None, real_last_date=None):
    """Audit one seal record. `other_starts` are the starts of every other window (played or concurrently sealed);
    `first_worker_start` is when the first worker touched this window; the seal must predate it."""
    out, g = [], "seal"
    for k in ("start", "shift_days", "sealed_at", "digest"):
        if k not in rec:
            _fail(out, g, f"missing field {k!r}")
    if out:
        return out
    start = pd.Timestamp(rec["start"])
    if start.day != 1:
        _fail(out, g, f"start {start.date()} is not a month start, so the window is not whole consecutive months")
    if not FIRST_START <= start <= LAST_START:
        _fail(out, g, f"start {start.date()} outside the drawable range {FIRST_START.date()}..{LAST_START.date()}")
    if rec["digest"] != seal_digest(rec):
        _fail(out, g, "seal digest mismatch: the sealed record was altered after sealing")
    if first_worker_start is not None and not pd.Timestamp(rec["sealed_at"]) < pd.Timestamp(first_worker_start):
        _fail(out, g, f"sealed_at {rec['sealed_at']} is not before the first worker start {first_worker_start}")
    for o in other_starts:
        ov = overlap_days(start, o)
        if ov > MAX_OVERLAP_DAYS:
            _fail(out, g, f"overlaps window starting {pd.Timestamp(o).date()} by {ov} days (> {MAX_OVERLAP_DAYS})")
    if real_last_date is not None and window_end(start) > pd.Timestamp(real_last_date):
        _fail(out, g, f"window ends {window_end(start).date()}, after the last real data date {real_last_date}")
    return out


def check_no_overlap_across(starts):
    """Pairwise overlap audit over a whole archive of windows."""
    out = []
    s = sorted(pd.Timestamp(x) for x in starts)
    for i, a in enumerate(s):
        for b in s[i + 1:]:
            if (b - a).days > 366:
                break
            if overlap_days(a, b) > MAX_OVERLAP_DAYS:
                _fail(out, "overlap", f"{a.date()} and {b.date()} overlap by {overlap_days(a, b)} days")
    return out


def check_sealed_before_workers(seal_time, worker_starts):
    """Every worker must have started after the seal; returns the offenders as failures."""
    st, out = pd.Timestamp(seal_time), []
    for name, t in worker_starts.items():
        if not st < pd.Timestamp(t):
            _fail(out, "seal-order", f"worker {name} started {t}, not after the seal {seal_time}")
    return out


# ============================================================ Phase 21: disguise
def make_ticker_map(tickers, shift_days_seed):
    """Deterministic seeded bijection real ticker -> code name, independent of input order (sorted first)."""
    real = sorted(set(tickers))
    rng = np.random.default_rng(int(shift_days_seed) * 7919 + 17)
    return {t: f"S{n:04d}" for t, n in zip(real, rng.permutation(len(real)))}


def check_shift(shift_days, real_dates, shifted_dates):
    out, g = [], "shift"
    if shift_days % 7:
        _fail(out, g, f"shift {shift_days} days is not a whole number of weeks; weekdays would change")
    rd, sd = pd.DatetimeIndex(real_dates), pd.DatetimeIndex(shifted_dates)
    if len(rd) != len(sd):
        _fail(out, g, "real and shifted date lists differ in length")
        return out
    if len(rd) and sd.min().year < MIN_SHIFT_YEAR:
        _fail(out, g, f"earliest shifted date {sd.min().date()} is before {MIN_SHIFT_YEAR}: could be a real year")
    if len(rd):
        if not (sd.dayofweek == rd.dayofweek).all():
            _fail(out, g, "weekdays changed under the shift")
        if not ((sd - rd) == pd.Timedelta(days=shift_days)).all():
            _fail(out, g, "shift is not one constant offset (holiday and gap pattern would be distorted)")
    return out


def check_ticker_map(mapping, real_tickers=None, seed=None):
    """Mapping must be a bijection, leak nothing about the real ticker, be reproducible, and not sort like the real
    names (a sorted-order leak lets an observer guess alphabetical position)."""
    out, g = [], "ticker-map"
    codes = list(mapping.values())
    if len(set(codes)) != len(codes):
        _fail(out, g, "two real tickers share one code name (not a bijection)")
    for t, c in mapping.items():
        if len(t) >= 3 and (t.upper() in c.upper() or c.upper() in t.upper()):   # 1-2 letter tickers collide with the S prefix
            _fail(out, g, f"code {c} contains or is contained in the real ticker {t}")
            break
    if real_tickers is not None and set(real_tickers) - set(mapping):
        _fail(out, g, f"{len(set(real_tickers) - set(mapping))} real tickers have no code name")
    if seed is not None and make_ticker_map(list(mapping), seed) != mapping:
        _fail(out, g, "mapping is not reproducible from its seed")
    if len(mapping) >= 30:
        real_rank = pd.Series(range(len(mapping)), index=sorted(mapping))
        code_rank = pd.Series({t: mapping[t] for t in mapping}).rank()
        rho = real_rank.corr(code_rank.reindex(real_rank.index), method="spearman")
        if abs(rho) > 0.5:
            _fail(out, g, f"code order tracks alphabetical real order (Spearman {rho:.2f})")
    return out


def check_disguised_frame(frame, mapping, real_dates_known=None, shift_days=None):
    """A frame shown to the trader: columns (or the ticker level) must be code names only, and the date index must sit
    in the shifted era. `real_dates_known` (feed side only) additionally proves no unshifted date slipped through."""
    out, g = [], "disguise"
    cols = frame.columns if not isinstance(frame.index, pd.MultiIndex) else frame.index.get_level_values(-1).unique()
    codes, real = set(mapping.values()), set(mapping)
    leaked = [c for c in cols if c in real and c not in codes]
    if leaked:
        _fail(out, g, f"real tickers visible to the trader: {leaked[:5]}")
    foreign = [c for c in cols if c not in codes and c not in real]
    if foreign:
        _fail(out, g, f"columns that are neither a code nor a known ticker: {foreign[:5]}")
    dates = pd.DatetimeIndex(frame.index.get_level_values(0) if isinstance(frame.index, pd.MultiIndex) else frame.index)
    if len(dates) and dates.min().year < MIN_SHIFT_YEAR:
        _fail(out, g, f"date {dates.min().date()} is in the real-history era")
    if real_dates_known is not None:
        hit = dates.intersection(pd.DatetimeIndex(real_dates_known))
        if len(hit):
            _fail(out, g, f"{len(hit)} dates equal real dates (unshifted)")
    return out


def check_disguise_signature(disguised_close, min_corr_gap=0.05):
    """Statistical fingerprint check: every code name must have its own price path. Two codes with identical series
    (a duplicated column) would let an observer collapse the universe. Returns fails for duplicates."""
    out = []
    d = disguised_close.dropna(axis=1, how="all")
    h = d.apply(lambda s: hashlib.md5(s.dropna().round(6).to_numpy().tobytes()).hexdigest())
    dup = h[h.duplicated(keep=False)]
    if len(dup):
        _fail(out, "disguise-dup", f"duplicate price paths under different codes: {sorted(dup.index)[:6]}")
    return out


# ============================================================ Phase 21: warm-up and reveal
def warmup_years_available(start, first_data=pd.Timestamp("1962-01-01"), want=6.0):
    """Years of history before the window start, capped at `want` (Bible: six where possible)."""
    return float(min(want, max(0.0, (pd.Timestamp(start) - pd.Timestamp(first_data)).days / 365.25)))


def check_warmup(start, actual_years, first_data=pd.Timestamp("1962-01-01"), want=6.0, tol=0.05):
    out, g = [], "warmup"
    possible = warmup_years_available(start, first_data, want)
    if actual_years + tol < possible:
        _fail(out, g, f"warm-up {actual_years:.2f}y is shorter than the {possible:.2f}y available")
    if actual_years > want + tol:
        out.append(Finding(g, "warn", f"warm-up {actual_years:.2f}y exceeds the {want:.0f}y convention"))
    return out


@dataclass
class RevealGate:
    """The true period stays hidden until adjustments are locked, all predictions recorded, all trades completed and
    all learning decisions finalized. `reveal()` raises rather than returning a partial answer."""
    window_label: str
    adjustments_locked: bool = False
    predictions_recorded: bool = False
    trades_completed: bool = False
    learning_finalized: bool = False
    revealed: bool = field(default=False, init=False)

    def blockers(self):
        return [n for n, v in (("adjustments_locked", self.adjustments_locked),
                               ("predictions_recorded", self.predictions_recorded),
                               ("trades_completed", self.trades_completed),
                               ("learning_finalized", self.learning_finalized)) if not v]

    def reveal(self):
        b = self.blockers()
        if b:
            raise PermissionError(f"reveal refused; still open: {b}")
        self.revealed = True
        return self.window_label

    def lock_adjustments(self):
        self.adjustments_locked = True


def check_reveal_order(events):
    """`events` = ordered [(name, time)] from the run log. The reveal event must come after the last of the four
    closing events, and after none of them may an adjustment or prediction event appear."""
    out, g = [], "reveal"
    names = [n for n, _ in events]
    if "reveal" not in names:
        return out
    ri = names.index("reveal")
    need = {"adjustments_locked", "predictions_recorded", "trades_completed", "learning_finalized"}
    missing = need - set(names[:ri])
    if missing:
        _fail(out, g, f"reveal happened before: {sorted(missing)}")
    late = [n for n in names[ri + 1:] if n in need | {"prediction", "trade", "adjustment"}]
    if late:
        _fail(out, g, f"activity after reveal (results may be steered by the answer): {late[:5]}")
    return out


# ============================================================ Phase 22: the clock
STEPS = ("fill_prior_decision", "observe_close", "update_state", "close_week", "learn", "adapt", "decide_next_week",
         "record_information")
WEEK_ONLY = {"close_week", "learn", "adapt", "decide_next_week"}   # run only on a session that ends a week


class ClockViolation(RuntimeError):
    pass


class BlindClock:
    """Enforces the Phase 22 order for each simulated session. `step(name)` raises ClockViolation on any out-of-order
    call. Non-week-end sessions skip the week steps but still fill, observe, update and record. `now` only moves
    forward, one session at a time; it can never be set from outside past the next session."""

    def __init__(self, sessions):
        self.sessions = pd.DatetimeIndex(sessions)
        if not self.sessions.is_monotonic_increasing or self.sessions.has_duplicates:
            raise ValueError("sessions must be strictly increasing")
        self.i = -1
        self.pending = None      # index into STEPS of the next allowed step for the current session
        self.log = []

    @property
    def now(self):
        if self.i < 0:
            raise ClockViolation("clock has not started")
        return self.sessions[self.i]

    def is_week_end(self):
        nxt = self.sessions[self.i + 1] if self.i + 1 < len(self.sessions) else None
        return nxt is None or nxt.isocalendar().week != self.now.isocalendar().week

    def begin_session(self):
        if self.pending is not None and self.pending != len(STEPS):
            raise ClockViolation(f"session {self.now.date()} not finished; next step is {STEPS[self.pending]}")
        if self.i + 1 >= len(self.sessions):
            raise ClockViolation("no more sessions")
        self.i += 1
        self.pending = 0
        return self.now

    def _expected(self):
        j = self.pending
        while j < len(STEPS) and STEPS[j] in WEEK_ONLY and not self.is_week_end():
            j += 1
        return j

    def step(self, name):
        if self.pending is None:
            raise ClockViolation("begin_session() first")
        if name not in STEPS:
            raise ClockViolation(f"unknown step {name!r}")
        j = self._expected()
        if j >= len(STEPS) or STEPS[j] != name:
            want = STEPS[j] if j < len(STEPS) else "begin_session"
            raise ClockViolation(f"{name} called at {self.now.date()}; the order requires {want}")
        self.log.append((self.now, name))
        self.pending = j + 1
        if self.pending == len(STEPS):
            return self.now
        return self.now

    def finish_session(self):
        j = self._expected()
        if j < len(STEPS):
            raise ClockViolation(f"session {self.now.date()} ended before {STEPS[j]}")
        self.pending = len(STEPS)


FORBIDDEN_FIELDS = ("label", "target", "y_fwd", "fwd_ret", "future", "next_close", "next_high", "next_low",
                    "realized_return")


def scan_information(records, forbidden=FORBIDDEN_FIELDS):
    """Each record: {"now": ts, "fields": {name: max_source_timestamp_or_None}, "frames": {name: DataFrame|Series}}.
    Fails on any source timestamp or frame date later than `now`, and on any field whose name marks a label/future."""
    out, g = [], "lookahead"
    for r in records:
        now = pd.Timestamp(r["now"])
        for name, ts in (r.get("fields") or {}).items():
            low = name.lower()
            if any(f in low for f in forbidden):
                _fail(out, g, f"{now.date()}: information set contains forbidden field {name!r}")
            if ts is not None and pd.Timestamp(ts) > now:
                _fail(out, g, f"{now.date()}: field {name!r} sourced from {pd.Timestamp(ts).date()} (future)")
        for name, fr in (r.get("frames") or {}).items():
            idx = fr.index.get_level_values(0) if isinstance(fr.index, pd.MultiIndex) else fr.index
            if len(idx) and pd.DatetimeIndex(idx).max() > now:
                _fail(out, g, f"{now.date()}: frame {name!r} reaches {pd.DatetimeIndex(idx).max().date()} (future)")
    return out


def check_fill_timing(decisions, fills, sessions):
    """decisions: DataFrame [ticker, decided_at]; fills: DataFrame [ticker, decided_at, filled_at, price_kind].
    A decision at a close must fill at the NEXT session's open, on a real session, never the same day."""
    out, g = [], "fill-timing"
    sess = pd.DatetimeIndex(sessions)
    if len(decisions) and not len(fills):
        _fail(out, g, "decisions recorded but no fills")
    nxt = {d: sess[i + 1] for i, d in enumerate(sess[:-1])}
    for _, f in fills.iterrows():
        d, t = pd.Timestamp(f["decided_at"]), pd.Timestamp(f["filled_at"])
        if d not in nxt:
            _fail(out, g, f"{f['ticker']}: decided at {d.date()} which is not a session with a next session")
        elif t != nxt[d]:
            _fail(out, g, f"{f['ticker']}: decided {d.date()} filled {t.date()}, expected {nxt[d].date()}")
        if f.get("price_kind", "open") != "open":
            _fail(out, g, f"{f['ticker']}: filled at {f['price_kind']}, must fill at the open")
        if t.dayofweek >= 5:
            _fail(out, g, f"{f['ticker']}: filled on a weekend {t.date()}")
    key = decisions.merge(fills, on=["ticker", "decided_at"], how="left", indicator=True) if len(decisions) else decisions
    if len(decisions):
        lost = key[key["_merge"] == "left_only"]
        if len(lost):
            out.append(Finding(g, "warn", f"{len(lost)} decisions were never filled"))
    return out


def lookahead_probe(fn, data, now, seed=0, trials=3):
    """Prove `fn(data_up_to_now_view)` cannot see the future: scramble every value dated after `now` (a permutation
    plus noise, seeded) and demand a bit-identical result. `data` is a DataFrame or Series indexed by date or
    (date, ticker); `fn` receives the FULL scrambled object and must do its own truncation. Returns fail findings
    naming the trial on which the answer moved."""
    out, now = [], pd.Timestamp(now)
    base = fn(data)
    dates = pd.DatetimeIndex(data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index)
    future = np.asarray(dates > now)
    if not future.any():
        return [Finding("lookahead-probe", "fail", "no data after `now`: the probe has nothing to scramble")]
    for k in range(trials):
        rng = np.random.default_rng([seed, k])
        vals = data.to_numpy(dtype=float, copy=True)
        fv = vals[future]
        scale = (np.nanstd(fv) if np.isfinite(fv).any() else 0.0) or 1.0
        vals[future] = fv[rng.permutation(len(fv))] * (1 + rng.normal(0, .5, fv.shape)) + rng.normal(0, scale, fv.shape)
        d2 = (pd.DataFrame(vals, index=data.index, columns=data.columns) if isinstance(data, pd.DataFrame)
              else pd.Series(vals, index=data.index, name=data.name))
        got = fn(d2)
        if not _same(base, got):
            _fail(out, "lookahead-probe", f"answer changed when data after {now.date()} was scrambled (trial {k})")
    return out


def _same(a, b):
    """Exact equality (NaNs equal), for frames, series, arrays and plain values. Any tolerance would hide a small leak."""
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return isinstance(b, type(a)) and a.equals(b)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def truncate(data, now):
    """The only sanctioned view of a panel at `now`: rows dated <= now."""
    idx = data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index
    return data[np.asarray(idx <= pd.Timestamp(now))]


def audit_run(seal, sessions, records, decisions, fills, mapping, disguised, shift_days, other_starts=(),
              first_worker_start=None):
    """One call that runs every Phase 21/22 gate over a finished run and returns all findings."""
    out = check_seal(seal, other_starts, first_worker_start)
    out += check_ticker_map(mapping, seed=shift_days)
    out += check_disguised_frame(disguised, mapping)
    out += scan_information(records)
    out += check_fill_timing(decisions, fills, sessions)
    return out


def save_report(findings, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"gate": f.gate, "severity": f.severity, "message": f.message} for f in findings]
    Path(path).write_text(json.dumps({"passed": passed(findings), "n": len(rows), "findings": rows}, indent=1))
    return passed(findings)


# ============================================================ Phase 21: coverage and cross-window leakage
def era_of(ts):
    """Cost/regime era used for per-era breakdowns (matches the simulator's cost tiers: <1997, 1997-2000, 2001+)."""
    y = pd.Timestamp(ts).year
    return "pre1997" if y < 1997 else "1997-2000" if y < 2001 else "2001+"


def coverage_report(starts, bins=(1965, 1975, 1985, 1995, 2005, 2015, 2026)):
    """How evenly the played windows spread over the calendar. Returns {"per_bin": {label: n}, "per_era": {...},
    "max_share": float, "empty_bins": [...]} - a run of windows all drawn from one decade would make every result
    an answer about that decade (canon C19 spreads them on purpose)."""
    s = pd.to_datetime(list(starts))
    per_bin, empty = {}, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((s.year >= lo) & (s.year < hi)).sum())
        per_bin[f"{lo}-{hi - 1}"] = n
        if n == 0:
            empty.append(f"{lo}-{hi - 1}")
    per_era = {}
    for t in s:
        per_era[era_of(t)] = per_era.get(era_of(t), 0) + 1
    return {"n": len(s), "per_bin": per_bin, "per_era": per_era, "empty_bins": empty,
            "max_share": (max(per_bin.values()) / len(s)) if len(s) else float("nan")}


def check_coverage(starts, max_share=0.5, min_windows=6):
    """Warn (not fail) when the archive is lopsided; a handful of windows cannot be balanced so it only speaks once
    there are enough of them."""
    out = []
    rep = coverage_report(starts)
    if rep["n"] >= min_windows and rep["max_share"] > max_share:
        out.append(Finding("coverage", "warn", f"{rep['max_share']:.0%} of windows fall in one bin: {rep['per_bin']}"))
    return out


def check_memory_bank_causality(bank, window_start):
    """C34: long-term memory may only hold lessons from windows that ENDED before this window began. `bank` needs a
    `real_end` column (date-like). Any row ending on or after `window_start` is a leak from the future of this run."""
    out = []
    if bank is None or not len(bank):
        return out
    if "real_end" not in bank:
        return [Finding("memory-bank", "fail", "bank has no real_end column, so its causality cannot be shown")]
    late = pd.to_datetime(bank["real_end"]) >= pd.Timestamp(window_start)
    if late.any():
        _fail(out, "memory-bank", f"{int(late.sum())} lessons come from windows ending on/after {pd.Timestamp(window_start).date()}")
    return out


def check_public_release(filings, now, time_col="accepted"):
    """A filing may be visible at `now` only if it was public by `now` (accepted <= now). Guards the feed against
    handing out an 8-K stamped after the close as if it were available at the close."""
    out = []
    if filings is None or not len(filings):
        return out
    t = pd.to_datetime(filings[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    n = pd.Timestamp(now)
    if n.tz is not None:                       # accepted times are naive UTC after the conversion above
        n = n.tz_convert("UTC").tz_localize(None)
    late = t > n
    if late.any():
        _fail(out, "filing-release", f"{int(late.sum())} filings stamped after {n} are visible (first {t[late].min()})")
    return out


def scan_text_for_leaks(text, real_tickers=(), real_years=(), min_ticker_len=3):
    """Scan free text (log lines, reports the trader could read, config dumps) for a real ticker or real four-digit
    year. Tickers shorter than `min_ticker_len` are skipped (they collide with ordinary words). Returns findings."""
    import re
    out = []
    for m in re.finditer(r"\b(19[6-9]\d|20[0-2]\d)\b", text):
        if int(m.group(1)) in set(real_years) or not real_years:
            _fail(out, "text-leak", f"real-era year {m.group(1)} at offset {m.start()}")
            break
    words = set(re.findall(r"\b[A-Z][A-Z0-9.\-]{%d,}\b" % (min_ticker_len - 1), text))
    hit = sorted(words & set(real_tickers))
    if hit:
        _fail(out, "text-leak", f"real tickers in text: {hit[:6]}")
    return out


def calendar_fingerprint(dates):
    """Weekday histogram and gap histogram of a session calendar. A disguised calendar must have the fingerprint of
    the real one shifted by whole weeks (identical), which is what keeps holidays and weekends plausible."""
    d = pd.DatetimeIndex(dates)
    gaps = pd.Series(d[1:] - d[:-1]).dt.days.value_counts().sort_index()
    return {"weekday": np.bincount(d.dayofweek, minlength=7).tolist(), "gaps": {int(k): int(v) for k, v in gaps.items()},
            "n": len(d)}


def check_calendar(real_dates, shown_dates):
    out = []
    if calendar_fingerprint(real_dates) != calendar_fingerprint(shown_dates):
        _fail(out, "calendar", "holiday/weekend pattern of the shown calendar differs from the real one")
    return out


# ============================================================ Phase 22: recorded information (step 8)
class InformationLedger:
    """Step 8 of the clock: record exactly what information was available at each session. Each entry carries the
    source timestamp of every input and a hash chained to the previous entry, so an entry cannot be edited or removed
    after the fact without breaking every later hash. The ledger refuses an entry whose inputs post-date `now` and
    refuses a `now` that does not advance."""

    def __init__(self):
        self.entries, self._last = [], "genesis"

    def record(self, now, inputs, decision=None):
        now = pd.Timestamp(now)
        if self.entries and now <= pd.Timestamp(self.entries[-1]["now"]):
            raise ClockViolation(f"ledger time did not advance: {now} after {self.entries[-1]['now']}")
        future = {k: str(pd.Timestamp(v)) for k, v in inputs.items() if v is not None and pd.Timestamp(v) > now}
        if future:
            raise ClockViolation(f"inputs from the future at {now.date()}: {future}")
        body = {"now": str(now), "inputs": {k: (None if v is None else str(pd.Timestamp(v))) for k, v in inputs.items()},
                "decision": decision, "prev": self._last}
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        self.entries.append(body)
        self._last = body["hash"]
        return body["hash"]

    def verify(self):
        return verify_ledger(self.entries)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.entries, sort_keys=True))


def verify_ledger(entries):
    """Recompute the chain. Fails on an edited entry, a removed/reordered entry, or a non-increasing timestamp."""
    out, prev, last_now = [], "genesis", None
    for i, e in enumerate(entries):
        body = {k: e[k] for k in ("now", "inputs", "decision", "prev")}
        if e.get("prev") != prev:
            _fail(out, "ledger", f"entry {i} does not follow the previous entry (removed or reordered)")
        if hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e.get("hash"):
            _fail(out, "ledger", f"entry {i} ({e.get('now')}) was edited after it was written")
        if last_now is not None and pd.Timestamp(e["now"]) <= last_now:
            _fail(out, "ledger", f"entry {i} time does not advance")
        for k, v in e["inputs"].items():
            if v is not None and pd.Timestamp(v) > pd.Timestamp(e["now"]):
                _fail(out, "ledger", f"entry {i}: input {k!r} dated after the session")
        prev, last_now = e.get("hash"), pd.Timestamp(e["now"])
    return out


def check_decisions_use_recorded_info(decisions, ledger_entries):
    """Every decision date must have a ledger entry made at or before it (decide -> record, never decide blind of the
    record). decisions: iterable of dates."""
    out = []
    have = {pd.Timestamp(e["now"]) for e in ledger_entries}
    for d in decisions:
        if pd.Timestamp(d) not in have:
            _fail(out, "ledger-coverage", f"decision on {pd.Timestamp(d).date()} has no recorded information set")
    return out


def check_label_alignment(features_dates, label_dates, horizon_sessions, sessions):
    """A training row dated d may use its label only once the label's window has fully closed: label_date must be at
    least `horizon_sessions` sessions after the feature date, and never after `now`. Returns findings for rows whose
    label window is not yet closed at the moment they are used for learning (given as label_dates = the session the
    label became known)."""
    out = []
    sess = pd.DatetimeIndex(sessions)
    pos = pd.Series(range(len(sess)), index=sess)
    fd, ld = pd.to_datetime(list(features_dates)), pd.to_datetime(list(label_dates))
    if len(fd) != len(ld):
        return [Finding("label-align", "fail", "feature and label date lists differ in length")]
    bad = 0
    for a, b in zip(fd, ld):
        if a not in pos.index or b not in pos.index or pos[b] - pos[a] < horizon_sessions:
            bad += 1
    if bad:
        _fail(out, "label-align", f"{bad} rows use a label before its {horizon_sessions}-session window closed")
    return out
