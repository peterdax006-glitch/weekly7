"""Multi-scale learning (research contract C66 section 23; supports 8, 13, 14, 31, 48; canon C56, C63, C64, C66).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; C63: no real-data run was made).

The contract: research at intraday, daily, weekly, multi-week, monthly, regime and multi-year scales; never assume a pattern
that works at one scale works at another; every pattern carries an explicit horizon; horizons must match the data that exists.

What lives here
  * the horizon table (HORIZON_SPECS) built ON engine.research.core.Horizon plus three long horizons defined here, the
    Scale each horizon belongs to, and available_horizons(): which horizons the data can honestly test (intraday only where
    the minute collector has sessions; long horizons only market-wide because per-stock windows would be few and overlapping);
  * label maturity: forward returns are computed from data strictly before `now`, filled at the NEXT session's open
    (rule 3), and a decision-date label exists only once its horizon-end close is behind `now`;
  * measure_effect(): date-clustered, HAC (Newey-West) standard errors sized to the label overlap, so a 21-session label is not
    counted as 21 independent observations;
  * effect_profile(): one pattern flag measured at every horizon, its shape (persistent / decaying / peaked / reversing /
    flat), peak horizon and decay half-life;
  * transfer_verdict() + ScaleLedger + enforce_horizons(): transfer between scales is a MEASURED fact with four outcomes
    (TRANSFERS / NOT_TRANSFERRED / REVERSES / UNTESTED). Using a pattern at a horizon other than its home horizon without a
    recorded TRANSFERS verdict is a violation and raises UnprovenTransfer (a FirewallBreach);
  * scale diagnostics: Lo-MacKinlay variance ratios and an aggregated-variance Hurst exponent (is the market trending or
    mean-reverting at each scale - regimes.py reads this), an overnight/intraday split that separates the capturable part
    of an effect from the gap nobody can fill, and per-session features from the minute collector's files;
  * step(): the ONE public entry the research loop calls - measure every registered pattern at every testable horizon, test
    every pair for transfer, record, and return the violations.

Research filed under a real year is never released while that year is replayed in disguise (rule 27): every read of the ledger
takes replay_years and drops records whose evidence year is being replayed.

Built on: engine.research.core (Horizon, MaturedRecord, Namespace), engine.learning.core (hashing, Provenance, require_past,
FirewallBreach), engine.learning.context.welch_t, engine.learning.situation.horizon_state/clean_number."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import config as K
from engine.learning.context import welch_t
from engine.learning.core import (FirewallBreach, Provenance, Unknown, _StrEnum, as_date, current_code_hash, require_past,
                                  stable_hash)
from engine.learning.situation import clean_number, horizon_state
from engine.research.core import Horizon, MaturedRecord, Namespace

SCHEMA_VERSION = "multiscale.v1"
T_BAR = 2.0                                   # |t| a horizon effect needs before it counts as established
MIN_DATES = 20                                # decision dates needed before any effect is measured
MIN_ROWS_PER_SIDE = 5                         # flagged (and unflagged) names on a date for the date to contribute


class UnprovenTransfer(FirewallBreach):
    """A pattern was used at a horizon it was never shown to transfer to."""


class Scale(_StrEnum):                        # section 23 list, in order
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MULTI_WEEK = "multi_week"
    MONTHLY = "monthly"
    REGIME = "regime"
    MULTI_YEAR = "multi_year"


SCALE_ORDER = tuple(Scale)


class TransferVerdict(_StrEnum):
    TRANSFERS = "TRANSFERS"                   # same sign, established at the target scale
    NOT_TRANSFERRED = "NOT_TRANSFERRED"       # source established, target indistinguishable from zero
    REVERSES = "REVERSES"                     # established at the target with the OPPOSITE sign
    UNTESTED = "UNTESTED"                     # one side lacked data or the source was never established
    EQUIVALENT = "EQUIVALENT"                 # same number of sessions under two names (5d == 1w)


class ProfileShape(_StrEnum):
    PERSISTENT = "PERSISTENT"                 # per-session effect holds as the horizon grows
    DECAYING = "DECAYING"                     # per-session effect fades with the horizon
    PEAKED = "PEAKED"                         # established only in the middle of the horizon range
    DELAYED = "DELAYED"                       # absent at short horizons, established only at the longer ones
    REVERSING = "REVERSING"                   # established with opposite signs at different horizons
    FLAT = "FLAT"                             # nothing established anywhere
    UNKNOWN = "UNKNOWN"                       # fewer than two horizons could be measured


# ------------------------------------------------------------------------------------------------ horizon table

@dataclasses.dataclass(frozen=True)
class HorizonSpec:
    """One explicit horizon. sessions is the label length in trading sessions (0 for intraday horizons, which use minutes)."""
    key: str
    scale: Scale
    sessions: int
    minutes: int = 0
    market_only: bool = False                 # too few independent per-stock windows: measured on market/cohort series only
    source: str = "daily"                     # daily | daily_ohlc (intraday structure read off the daily bar) | minute_bars
    coverage_limited: bool = False            # True where history is short by construction (minute collector: months only)

    @property
    def is_intraday(self) -> bool:
        return self.sessions == 0

    def embargo(self) -> int:
        """Sessions to keep between a training row and a test row so their labels cannot overlap."""
        return max(self.sessions - 1, 0)

    def check(self) -> list[str]:
        errs = []
        if self.is_intraday and self.minutes <= 0:
            errs.append(f"{self.key}: intraday horizon without minutes")
        if not self.is_intraday and self.sessions < 1:
            errs.append(f"{self.key}: sessions must be >= 1")
        if not self.key:
            errs.append("horizon key missing")
        return errs


OHLC_DAY_KEY = "ohlc_day"                     # canon C67: the day's own intraday structure read from the daily O/H/L/C bar
SESSION_MINUTES = 390

# keys beyond engine.research.core.Horizon (section 23 lists regime and multi-year scales but the enum stops at 1 month)
LONG_HORIZONS = {"3m": HorizonSpec("3m", Scale.REGIME, 63, market_only=True),
                 "1y": HorizonSpec("1y", Scale.MULTI_YEAR, 252, market_only=True),
                 "3y": HorizonSpec("3y", Scale.MULTI_YEAR, 756, market_only=True)}

HORIZON_SPECS: dict[str, HorizonSpec] = {
    Horizon.MIN5.value: HorizonSpec("5min", Scale.INTRADAY, 0, 5, source="minute_bars", coverage_limited=True),
    Horizon.MIN30.value: HorizonSpec("30min", Scale.INTRADAY, 0, 30, source="minute_bars", coverage_limited=True),
    Horizon.HOUR1.value: HorizonSpec("1h", Scale.INTRADAY, 0, 60, source="minute_bars", coverage_limited=True),
    OHLC_DAY_KEY: HorizonSpec(OHLC_DAY_KEY, Scale.INTRADAY, 0, SESSION_MINUTES, source="daily_ohlc"),
    Horizon.DAY1.value: HorizonSpec("1d", Scale.DAILY, 1),
    "2d": HorizonSpec("2d", Scale.DAILY, 2),
    Horizon.DAY3.value: HorizonSpec("3d", Scale.DAILY, 3),
    Horizon.DAY5.value: HorizonSpec("5d", Scale.WEEKLY, 5),
    Horizon.WEEK1.value: HorizonSpec("1w", Scale.WEEKLY, 5),
    Horizon.WEEK2.value: HorizonSpec("2w", Scale.MULTI_WEEK, 10),
    Horizon.MONTH1.value: HorizonSpec("1m", Scale.MONTHLY, 21),
    **LONG_HORIZONS,
}
DAILY_LADDER = ("1d", "2d", "3d", "5d", "2w", "1m")   # the per-stock ladder used by default (5d and 1w are the same length)


def parse_horizon(h: Any) -> HorizonSpec:
    """Accept a Horizon, a key ('5d'), a HorizonSpec, or a session count (int -> '<n>d'-style spec). Unknown keys raise:
    a pattern with an unrecognised horizon is a pattern with NO horizon."""
    if isinstance(h, HorizonSpec):
        return h
    if isinstance(h, Horizon):
        return HORIZON_SPECS[h.value]
    if isinstance(h, bool):
        raise ValueError("a bool is not a horizon")
    if isinstance(h, (int, np.integer)):
        if h < 1:
            raise ValueError(f"horizon sessions must be >= 1, got {h}")
        for spec in HORIZON_SPECS.values():
            if spec.sessions == int(h) and spec.key.endswith("d"):
                return spec
        return HorizonSpec(f"{int(h)}s", scale_of_sessions(int(h)), int(h), market_only=int(h) > 126)
    s = str(h).strip().lower()
    if s in HORIZON_SPECS:
        return HORIZON_SPECS[s]
    m = re.fullmatch(r"(\d+)\s*(s|sess|sessions)", s)
    if m:
        return parse_horizon(int(m.group(1)))
    raise ValueError(f"unknown horizon {h!r}")


def scale_of_sessions(n: int) -> Scale:
    """Scale a session count belongs to (used for horizons that are not in the named table)."""
    if n <= 0:
        return Scale.INTRADAY
    if n <= 3:
        return Scale.DAILY
    if n <= 5:
        return Scale.WEEKLY
    if n <= 10:
        return Scale.MULTI_WEEK
    if n <= 42:
        return Scale.MONTHLY
    if n <= 126:
        return Scale.REGIME
    return Scale.MULTI_YEAR


def same_length(a: Any, b: Any) -> bool:
    """5d and 1w are two names for one horizon; a transfer test between them is meaningless."""
    sa, sb = parse_horizon(a), parse_horizon(b)
    return sa.sessions == sb.sessions and sa.minutes == sb.minutes


def horizon_state_of(h: Any) -> str | None:
    """The coarse label the Situation representation uses for a horizon (short / week / multi_week)."""
    spec = parse_horizon(h)
    return None if spec.is_intraday else horizon_state(spec.sessions)


def validate_table() -> list[str]:
    """Structural check of the horizon table: every entry valid, sessions monotone with scale order, keys unique."""
    errs = [e for spec in HORIZON_SPECS.values() for e in spec.check()]
    seen = {}
    for k, spec in HORIZON_SPECS.items():
        if spec.key != k and k not in (Horizon.MIN5.value, Horizon.MIN30.value, Horizon.HOUR1.value):
            errs.append(f"key {k!r} maps to spec {spec.key!r}")
        seen.setdefault((spec.sessions, spec.minutes), []).append(k)
    ranks = [SCALE_ORDER.index(s.scale) for s in sorted(HORIZON_SPECS.values(), key=lambda s: (s.sessions, s.minutes))]
    if ranks != sorted(ranks):
        errs.append("scale order is not monotone in horizon length")
    return errs


# ------------------------------------------------------------------------------------------------ data profile

@dataclasses.dataclass(frozen=True)
class DataProfile:
    """What history exists. intraday maps a bar interval ('1m', '5m') to the count of COMPLETED sessions on disk."""
    n_sessions: int = 0                                  # daily sessions available before `now`
    n_names: int = 0
    intraday: Mapping[str, int] = dataclasses.field(default_factory=dict)

    def check(self) -> list[str]:
        errs = []
        if self.n_sessions < 0 or self.n_names < 0:
            errs.append("negative counts")
        for k, v in self.intraday.items():
            if bar_minutes(k) is None:
                errs.append(f"intraday interval {k!r} unparseable")
            if v < 0:
                errs.append(f"intraday sessions for {k} negative")
        return errs


@dataclasses.dataclass(frozen=True)
class HorizonAvailability:
    horizon: str
    scale: Scale
    testable: bool
    windows: int                                         # non-overlapping windows the data holds
    reason: str


def bar_minutes(interval: str) -> int | None:
    m = re.fullmatch(r"(\d+)\s*(m|min|h)", str(interval).strip().lower())
    if not m:
        return None
    return int(m.group(1)) * (60 if m.group(2) == "h" else 1)


def intraday_inventory(now, root: Path | str | None = None) -> dict[str, int]:
    """Completed intraday sessions on disk per bar interval, counting only files dated strictly before `now`: a session file
    for `now` itself may be written after the decision, so it never counts. Missing folder or unreadable names -> {}."""
    base = Path(root) if root is not None else Path(K.DATA) / "intraday"
    now_d = as_date(now)
    out: dict[str, int] = {}
    if not base.is_dir():
        return out
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        n = 0
        for f in sub.glob("*.parquet"):
            try:
                if as_date(f.stem) < now_d:
                    n += 1
            except ValueError:
                continue
        out[sub.name] = n
    return out


def available_horizons(profile: DataProfile, min_windows: int = 30, min_market_windows: int = 8,
                       min_intraday_sessions: int = 20) -> list[HorizonAvailability]:
    """Which horizons the data can honestly test (section 23: 'must match available data').
    intraday horizon h needs a bar interval that divides it and >= min_intraday_sessions sessions of that interval;
    a session horizon needs >= min_windows NON-OVERLAPPING windows (market_only horizons need only min_market_windows and are
    then measured on market or cohort series, never per stock)."""
    errs = profile.check()
    if errs:
        raise ValueError("bad DataProfile: " + "; ".join(errs))
    out = []
    for key, spec in HORIZON_SPECS.items():
        if spec.key != key and key not in ("5min", "30min", "1h"):
            continue
        if spec.source == "daily_ohlc":                  # needs only the daily bars: decades of history, no coverage limit
            ok = profile.n_sessions >= min_windows
            out.append(HorizonAvailability(key, spec.scale, ok, profile.n_sessions,
                                           "ok" if ok else f"{profile.n_sessions} daily sessions (< {min_windows})"))
            continue
        if spec.is_intraday:
            best = 0
            for interval, sessions in profile.intraday.items():
                bm = bar_minutes(interval)
                if bm and spec.minutes % bm == 0 and sessions > best:
                    best = sessions
            ok = best >= min_intraday_sessions
            reason = "ok" if ok else (f"only {best} intraday sessions of a suitable bar size (< {min_intraday_sessions})"
                                      if best else "no minute collector sessions with a suitable bar size")
            out.append(HorizonAvailability(key, spec.scale, ok, best, reason))
            continue
        windows = profile.n_sessions // spec.sessions
        need = min_market_windows if spec.market_only else min_windows
        ok = windows >= need and profile.n_sessions > 0
        out.append(HorizonAvailability(key, spec.scale, ok, windows,
                                       "ok" if ok else f"{windows} non-overlapping windows (< {need})"))
    return out


def testable_keys(profile: DataProfile, per_stock: bool = True, **kw) -> tuple[str, ...]:
    """Keys of testable horizons, shortest first, one key per distinct length (5d wins over 1w)."""
    seen, keys = set(), []
    for a in sorted(available_horizons(profile, **kw), key=lambda a: (HORIZON_SPECS[a.horizon].sessions,
                                                                      HORIZON_SPECS[a.horizon].minutes)):
        spec = HORIZON_SPECS[a.horizon]
        sig = (spec.sessions, spec.minutes)
        if not a.testable or sig in seen or (per_stock and spec.market_only):
            continue
        seen.add(sig)
        keys.append(a.horizon)
    return tuple(keys)


# ------------------------------------------------------------------------------------------------ label maturity

def _sessions(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def past_only(frame: pd.DataFrame | pd.Series, now, inclusive: bool = False):
    """Rows of a date-indexed (or (date, x)-indexed) object dated before `now` (or up to and including it if inclusive).
    Used so a function never even READS data after its `now`."""
    idx = frame.index.get_level_values(0) if isinstance(frame.index, pd.MultiIndex) else frame.index
    d = _sessions(idx)
    cut = pd.Timestamp(as_date(now))
    return frame[(d <= cut) if inclusive else (d < cut)]


def maturity_date(dates: pd.DatetimeIndex, position: int, sessions: int):
    """Date on which the label of decision `position` is fully realised (the horizon-end close), or None if not in range."""
    j = position + max(sessions, 1)
    return None if j >= len(dates) else dates[j]


def mature_count(dates: pd.DatetimeIndex, sessions: int, now) -> int:
    """How many leading decision dates have a label that is fully behind `now` (strictly)."""
    d = _sessions(dates)
    cut = pd.Timestamp(as_date(now))
    upto = int((d < cut).sum())
    return max(upto - max(sessions, 1), 0)


def forward_log_returns(close: pd.DataFrame, sessions: int, now, open_: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forward log return for a decision at each close: entry at the NEXT session's open (open_ given; else the decision
    close), exit at the close `sessions` later. Built only from rows dated strictly before `now`, so a row whose horizon end
    is not yet behind `now` is NaN. (date x ticker)."""
    if sessions < 1:
        raise ValueError("forward_log_returns needs sessions >= 1 (intraday horizons use intraday_forward_returns)")
    C = past_only(close, now)
    if C.empty:
        return C.astype("float64")
    if open_ is not None:
        O = past_only(open_, now).reindex(index=C.index, columns=C.columns)
        entry = O.shift(-1)
    else:
        entry = C
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = np.log(C.shift(-sessions) / entry)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)
    return fwd


def forward_map(close: pd.DataFrame, keys: Iterable[Any], now, open_: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """{horizon key: stacked (date, ticker) forward-return Series} for every session horizon in keys."""
    out: dict[str, pd.Series] = {}
    for k in keys:
        spec = parse_horizon(k)
        if spec.is_intraday:
            continue
        f = forward_log_returns(close, spec.sessions, now, open_)
        s = f.stack()
        s.index.names = ["date", "ticker"]
        out[spec.key] = s.astype("float64")
    return out


# ------------------------------------------------------------------------------------------------ effect at one horizon

def hac_se(x: Sequence[float], lags: int) -> float:
    """Newey-West (Bartlett) standard error of the mean of x. NaN entries are missing days: a lagged product is used only
    where both ends exist, and every autocovariance is divided by the count of present days, so gaps do not shrink the error.
    lags = label length - 1 covers the overlap of forward labels built every session."""
    a = np.asarray(x, dtype="float64")
    ok = np.isfinite(a)
    n = int(ok.sum())
    if n < 2:
        return float("nan")
    m = float(a[ok].mean())
    xm = np.where(ok, a - m, 0.0)
    s = float(xm @ xm) / n
    for lag in range(1, min(int(lags), len(a) - 1) + 1):
        both = ok[lag:] & ok[:-lag]
        if not both.any():
            continue
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(xm[lag:][both] @ xm[:-lag][both]) / n
    return math.sqrt(max(s, 0.0) / n)


def _dates_of(s: pd.Series) -> pd.DatetimeIndex:
    return _sessions(s.index.get_level_values(0))


@dataclasses.dataclass(frozen=True)
class ScaleEffect:
    """What a pattern flag does to the forward return at ONE horizon, measured on matured labels only.
    effect is the mean over decision dates of (mean fwd of flagged names - mean fwd of unflagged names), with each date's
    cross-section demeaned first (the market's drift is not the pattern's effect). se is HAC-corrected for label overlap;
    naive_t is the row-pooled Welch t that IGNORES overlap - the ratio naive_t / t is the overlap inflation a careless test has."""
    horizon: str
    sessions: int
    n_rows: int
    n_dates: int
    n_eff: float
    effect: float | None
    se: float | None
    t: float | None
    mean_in: float | None
    mean_out: float | None
    hit_rate: float | None
    naive_t: float | None
    matured_through: str
    status: str = "OK"
    evidence_years: tuple[int, ...] = ()       # calendar years of the decision dates used (rule 27: same-year rerun leak)

    def established(self, t_bar: float = T_BAR) -> bool:
        return self.status == "OK" and self.t is not None and abs(self.t) >= t_bar

    @property
    def per_session(self) -> float | None:
        return None if self.effect is None else self.effect / max(self.sessions, 1)

    @property
    def sign(self) -> int:
        return 0 if not self.effect else (1 if self.effect > 0 else -1)

    @property
    def overlap_inflation(self) -> float | None:
        if self.naive_t is None or not self.t:
            return None
        return abs(self.naive_t) / abs(self.t)

    def check(self) -> list[str]:
        errs = []
        if self.n_dates < 0 or self.n_rows < 0 or self.n_eff < 0:
            errs.append("negative counts")
        if self.status == "OK" and (self.effect is None or self.se is None or self.t is None):
            errs.append("status OK but effect/se/t missing")
        if self.se is not None and self.se < 0:
            errs.append("se < 0")
        if not self.matured_through:
            errs.append("matured_through missing")
        return errs


def _insufficient(spec: HorizonSpec, n_rows: int, n_dates: int, through: str, status: str,
                  years: tuple[int, ...] = ()) -> ScaleEffect:
    return ScaleEffect(spec.key, spec.sessions, n_rows, n_dates, n_dates / max(spec.sessions, 1), None, None, None, None, None,
                       None, None, through, status, years)


def measure_effect(flag: pd.Series, fwd: pd.Series, horizon: Any, now, calendar: pd.DatetimeIndex | None = None,
                   min_dates: int = MIN_DATES, min_side: int = 1, demean: bool = True) -> ScaleEffect:
    """Effect of `flag` on `fwd` at one horizon. Both are (date, ticker)-indexed. `fwd` must hold matured labels only:
    with a `calendar` (the session dates) a row whose horizon-end close is not strictly before `now` is refused
    (FirewallBreach); without one, any row dated at/after `now` is refused."""
    spec = parse_horizon(horizon)
    if spec.is_intraday:
        raise ValueError(f"{spec.key} is an intraday horizon; measure it with measure_intraday_effect")
    now_d = pd.Timestamp(as_date(now))
    f = fwd.dropna()
    if f.empty:
        return _insufficient(spec, 0, 0, str(now_d.date()), Unknown.INSUFFICIENT_DATA.value)
    fd = _dates_of(f)
    if calendar is not None:
        cal = _sessions(calendar)
        pos = cal.searchsorted(fd)
        end = np.minimum(pos + spec.sessions, len(cal) - 1)
        late = (pos + spec.sessions >= len(cal)) | (cal[end] >= now_d)
        if late.any():
            raise FirewallBreach(f"{int(late.sum())} label(s) at {spec.key} are not matured before now={now_d.date()}")
    elif (fd >= now_d).any():
        raise FirewallBreach(f"forward labels dated at/after now={now_d.date()}")
    through = str(fd.max().date())
    years = tuple(sorted({int(y) for y in fd.year.unique()}))
    g = flag.reindex(f.index)
    frame = pd.DataFrame({"f": f.to_numpy(dtype="float64"), "g": g.fillna(False).astype(bool).to_numpy(),
                          "d": fd.to_numpy()}, index=f.index)
    if demean:
        frame["f"] = frame["f"] - frame.groupby("d")["f"].transform("mean")
    grp = frame.groupby(["d", "g"])["f"].agg(["mean", "count"]).unstack("g")
    if True not in grp["mean"].columns or False not in grp["mean"].columns:
        return _insufficient(spec, len(frame), 0, through, Unknown.INSUFFICIENT_DATA.value, years)
    m_in, m_out = grp["mean"][True], grp["mean"][False]
    c_in, c_out = grp["count"][True], grp["count"][False]
    good = (c_in >= min_side) & (c_out >= min_side)
    spread = (m_in - m_out)[good]
    n_dates = int(spread.notna().sum())
    if n_dates < min_dates:
        return _insufficient(spec, len(frame), n_dates, through, Unknown.INSUFFICIENT_DATA.value, years)
    full = spread.reindex(pd.DatetimeIndex(sorted(frame["d"].unique())))
    eff = float(spread.mean())
    se = hac_se(full.to_numpy(), spec.sessions - 1)
    inn = frame[frame["g"]]["f"]
    out = frame[~frame["g"]]["f"]
    naive = welch_t(float(inn.mean()), float(inn.var(ddof=1)) if len(inn) > 1 else None, len(inn),
                    float(out.mean()), float(out.var(ddof=1)) if len(out) > 1 else None, len(out))
    if not math.isfinite(se) or se <= 1e-15:
        return ScaleEffect(spec.key, spec.sessions, len(frame), n_dates, n_dates / max(spec.sessions, 1), eff, None, None,
                           float(m_in[good].mean()), float(m_out[good].mean()), float((inn > 0).mean()), naive, through,
                           "DEGENERATE", years)
    return ScaleEffect(spec.key, spec.sessions, len(frame), n_dates, n_dates / max(spec.sessions, 1), eff, se, eff / se,
                       float(m_in[good].mean()), float(m_out[good].mean()), float((inn > 0).mean()), naive, through, "OK", years)


def block_permutation_p(flag: pd.Series, fwd: pd.Series, horizon: Any, now, n_perm: int = 200, seed: int = 0,
                        calendar: pd.DatetimeIndex | None = None, min_dates: int = MIN_DATES) -> float | None:
    """Two-sided permutation p-value of the horizon effect. Whole date-blocks of flags are shuffled ACROSS dates (preserving
    the within-date flag pattern and the label overlap), so autocorrelation cannot manufacture significance. None when the
    effect cannot be measured. Deterministic in `seed`."""
    spec = parse_horizon(horizon)
    base = measure_effect(flag, fwd, spec, now, calendar, min_dates)
    if base.status != "OK":
        return None
    rng = np.random.default_rng(seed)
    f = fwd.dropna()
    dates = _dates_of(f)
    uniq = np.array(sorted(set(dates)))
    g = flag.reindex(f.index).fillna(False).astype(bool)
    by_date = {d: g[dates == d].to_numpy() for d in uniq}
    sizes = {d: len(v) for d, v in by_date.items()}
    hits = 0
    extreme = abs(base.effect)
    for _ in range(int(n_perm)):
        order = rng.permutation(len(uniq))
        fake = np.empty(len(g), dtype=bool)
        for d, j in zip(uniq, order):
            src = by_date[uniq[j]]
            tgt = np.flatnonzero(dates == d)
            fake[tgt] = np.resize(src, sizes[d])            # equal-size dates take the block whole; unequal ones tile it
        e = measure_effect(pd.Series(fake, index=f.index), f, spec, now, calendar, min_dates)
        if e.effect is not None and abs(e.effect) >= extreme:
            hits += 1
    return (hits + 1.0) / (n_perm + 1.0)


# ------------------------------------------------------------------------------------------------ profile across horizons

@dataclasses.dataclass(frozen=True)
class EffectProfile:
    """One flag measured at every measurable horizon. Immutable; the shape is derived only from the tuple of effects."""
    effects: tuple[ScaleEffect, ...]
    shape: ProfileShape
    peak: str | None
    half_life_sessions: float | None
    t_bar: float = T_BAR

    def measurable(self) -> tuple[ScaleEffect, ...]:
        return tuple(e for e in self.effects if e.status == "OK")

    def established(self) -> tuple[ScaleEffect, ...]:
        return tuple(e for e in self.effects if e.established(self.t_bar))

    def by_key(self) -> dict[str, ScaleEffect]:
        return {e.horizon: e for e in self.effects}

    def home(self) -> str | None:
        """The horizon where the effect is strongest (largest |t|; ties go to the shorter horizon)."""
        est = self.established()
        if not est:
            return None
        return max(est, key=lambda e: (abs(e.t), -e.sessions)).horizon


def _decay_half_life(est: Sequence[ScaleEffect]) -> float | None:
    """Half-life, in sessions, of the per-session effect across established horizons (log-linear fit in the horizon length);
    None if fewer than two horizons or the per-session effect does not shrink."""
    pts = [(e.sessions, abs(e.per_session)) for e in est if e.per_session]
    if len(pts) < 2:
        return None
    h = np.array([p[0] for p in pts], dtype=float)
    v = np.log(np.array([p[1] for p in pts]))
    if np.ptp(h) == 0:
        return None
    slope = float(np.polyfit(h, v, 1)[0])
    return None if slope >= -1e-9 else float(math.log(2.0) / -slope)


def classify_profile(effects: Sequence[ScaleEffect], t_bar: float = T_BAR) -> tuple[ProfileShape, str | None, float | None]:
    ok = sorted((e for e in effects if e.status == "OK"), key=lambda e: e.sessions)
    if len(ok) < 2:
        return ProfileShape.UNKNOWN, None, None
    est = [e for e in ok if e.established(t_bar)]
    if not est:
        return ProfileShape.FLAT, None, None
    peak = max(est, key=lambda e: abs(e.t)).horizon
    if {e.sign for e in est} == {1, -1}:
        return ProfileShape.REVERSING, peak, None
    idx = [ok.index(e) for e in est]
    hl = _decay_half_life(est)
    if len(est) == len(ok):
        per = [abs(e.per_session) for e in ok]
        hs = np.log([e.sessions for e in ok])
        slope = float(np.polyfit(hs, np.log(np.maximum(per, 1e-12)), 1)[0]) if len(ok) >= 2 else 0.0
        return (ProfileShape.PERSISTENT if slope > -0.35 else ProfileShape.DECAYING), peak, hl
    contiguous = idx == list(range(idx[0], idx[-1] + 1))
    if idx[0] == 0 and contiguous:
        return ProfileShape.DECAYING, peak, hl
    if idx[-1] == len(ok) - 1 and contiguous:
        return ProfileShape.DELAYED, peak, hl
    return ProfileShape.PEAKED, peak, hl


def effect_profile(flag: pd.Series, fwd_map: Mapping[str, pd.Series], now, calendar: pd.DatetimeIndex | None = None,
                   t_bar: float = T_BAR, min_dates: int = MIN_DATES) -> EffectProfile:
    """Measure `flag` at every horizon in fwd_map (key -> matured forward-return Series)."""
    effects = tuple(sorted((measure_effect(flag, s, k, now, calendar, min_dates) for k, s in fwd_map.items()),
                           key=lambda e: e.sessions))
    shape, peak, hl = classify_profile(effects, t_bar)
    return EffectProfile(effects, shape, peak, hl, t_bar)


# ------------------------------------------------------------------------------------------------ transfer between scales

@dataclasses.dataclass(frozen=True)
class TransferRecord:
    """Whether a pattern established at `source` shows up at `target`. A measured fact with evidence, never an assumption."""
    pattern_id: str
    source: str
    target: str
    verdict: TransferVerdict
    source_effect: float | None
    target_effect: float | None
    target_t: float | None
    per_session_ratio: float | None             # target per-session effect / source per-session effect
    matured_through: str

    def check(self) -> list[str]:
        errs = []
        if self.source == self.target:
            errs.append("a horizon cannot transfer to itself")
        if self.verdict in (TransferVerdict.TRANSFERS, TransferVerdict.REVERSES, TransferVerdict.NOT_TRANSFERRED) \
                and (self.source_effect is None or self.target_effect is None):
            errs.append(f"{self.verdict} without both effects")
        return errs


MIN_PER_SESSION_RATIO = 0.25                  # a longer-horizon return CONTAINS the short one; the effect must not merely be carried along


def transfer_verdict(src: ScaleEffect, tgt: ScaleEffect, t_bar: float = T_BAR, min_ratio: float = MIN_PER_SESSION_RATIO) -> TransferVerdict:
    """Cumulative forward returns nest (a 21-session return contains the first session), so significance at the target is not
    enough: a 1-day blip stays 'significant' in a month-long return while being diluted 21-fold. TRANSFERS therefore also needs the
    per-session effect at the target to be at least min_ratio of the source's (the effect is not just carried along); a diluted
    same-sign effect is NOT_TRANSFERRED. An opposite-sign established effect is REVERSES whatever its size."""
    if same_length(src.horizon, tgt.horizon):
        return TransferVerdict.EQUIVALENT
    if src.status != "OK" or tgt.status != "OK" or not src.established(t_bar):
        return TransferVerdict.UNTESTED
    if tgt.established(t_bar):
        if tgt.sign != src.sign:
            return TransferVerdict.REVERSES
        ratio = abs(tgt.per_session) / abs(src.per_session) if src.per_session else 0.0
        return TransferVerdict.TRANSFERS if ratio >= min_ratio else TransferVerdict.NOT_TRANSFERRED
    return TransferVerdict.NOT_TRANSFERRED


def transfer_records(pattern_id: str, profile: EffectProfile) -> list[TransferRecord]:
    """Every ordered (source, target) pair where the source is established, tested against every other measured horizon."""
    out = []
    for s in profile.effects:
        if not s.established(profile.t_bar):
            continue
        for t in profile.effects:
            if t.horizon == s.horizon:
                continue
            v = transfer_verdict(s, t, profile.t_bar)
            ratio = None
            if s.per_session and t.per_session is not None:
                ratio = t.per_session / s.per_session
            out.append(TransferRecord(pattern_id, s.horizon, t.horizon, v, s.effect, t.effect, t.t, ratio,
                                      min(s.matured_through, t.matured_through)))
    return out


# ------------------------------------------------------------------------------------------------ minute coverage (limit declared)

@dataclasses.dataclass(frozen=True)
class MinuteCoverage:
    """How much minute history a minute-derived pattern rests on. The collector keeps ~7 days of 1m and ~60 of 5m bars, so a
    pattern learned from them has seen months at most and may never claim more."""
    interval: str
    sessions: int
    first: str = ""                               # trusted-side only
    last: str = ""

    @property
    def max_supported_years(self) -> float:
        return self.sessions / 252.0

    def supports_span(self, years: float) -> bool:
        return years <= self.max_supported_years + 1e-9

    def check(self) -> list[str]:
        errs = []
        if bar_minutes(self.interval) is None:
            errs.append(f"interval {self.interval!r} unparseable")
        if self.sessions < 0:
            errs.append("negative sessions")
        return errs


def minute_coverage(now, root: Path | str | None = None) -> dict[str, MinuteCoverage]:
    """Coverage per bar interval from the collector's files dated strictly before `now`."""
    base = Path(root) if root is not None else Path(K.DATA) / "intraday"
    now_d = as_date(now)
    out: dict[str, MinuteCoverage] = {}
    if not base.is_dir():
        return out
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        days = []
        for f in sub.glob("*.parquet"):
            try:
                d = as_date(f.stem)
            except ValueError:
                continue
            if d < now_d:
                days.append(d)
        if days:
            out[sub.name] = MinuteCoverage(sub.name, len(days), min(days).isoformat(), max(days).isoformat())
    return out


# ------------------------------------------------------------------------------------------------ the ledger

def _visible(evidence_years: Sequence[int], matured_through: str, now, replay_years: Iterable[int]) -> bool:
    """A research record may be read at `now` only if its newest evidence is strictly behind `now` AND none of its evidence
    years is being replayed in disguise right now (rule 27: the same-year rerun leak)."""
    require_past(matured_through, now, "scale record")
    return not (set(int(y) for y in evidence_years) & set(int(y) for y in replay_years))


@dataclasses.dataclass(frozen=True)
class Declaration:
    pattern_id: str
    horizon: str
    declared_by: str                        # 'design' (the horizon the pattern was mined for) | 'measured'
    declared_at: str
    version: int = 1
    intraday_source: str = ""               # '' | 'daily_ohlc' | 'minute_bars'

    def check(self) -> list[str]:
        errs = []
        try:
            spec = parse_horizon(self.horizon)
        except ValueError as e:
            return [str(e)]
        if self.intraday_source and self.intraday_source not in ("daily_ohlc", "minute_bars"):
            errs.append(f"intraday_source {self.intraday_source!r} unknown")
        if spec.source == "minute_bars" and self.intraday_source != "minute_bars":
            errs.append("a minute-scale horizon must declare intraday_source='minute_bars'")
        return errs


class ScaleLedger:
    """Append-only record of what each pattern does at each horizon and whether it transfers (research world:
    MATURED_RESEARCH_STATE). Nothing is overwritten; a re-declared horizon is a NEW version. Reads take `now` and drop anything
    not yet matured, and take `replay_years` to drop evidence from a year currently being replayed in disguise."""

    def __init__(self) -> None:
        self._decl: dict[str, list[Declaration]] = {}
        self._effects: list[tuple[str, ScaleEffect]] = []
        self._transfers: list[tuple[TransferRecord, tuple[int, ...]]] = []
        self._coverage: dict[str, MinuteCoverage] = {}

    # ---- writes
    def declare(self, pattern_id: str, horizon: Any, now, declared_by: str = "design", supersede: bool = False,
                intraday_source: str = "") -> Declaration:
        spec = parse_horizon(horizon)
        cur = self._decl.get(pattern_id, [])
        if cur and cur[-1].horizon != spec.key and not supersede:
            raise ValueError(f"{pattern_id} already declared at {cur[-1].horizon}; pass supersede=True for a new version")
        if cur and cur[-1].horizon == spec.key:
            return cur[-1]
        src = intraday_source or ("minute_bars" if spec.source == "minute_bars" else "daily_ohlc" if spec.source == "daily_ohlc" else "")
        d = Declaration(pattern_id, spec.key, declared_by, str(as_date(now)), len(cur) + 1, src)
        errs = d.check()
        if errs:
            raise ValueError("; ".join(errs))
        self._decl.setdefault(pattern_id, []).append(d)
        return d

    def declare_minute_coverage(self, pattern_id: str, coverage: MinuteCoverage) -> None:
        """A minute-derived pattern must state how much history it rests on (canon C67)."""
        errs = coverage.check()
        if errs:
            raise ValueError("; ".join(errs))
        self._coverage[pattern_id] = coverage

    def record_effect(self, pattern_id: str, effect: ScaleEffect, now) -> None:
        errs = effect.check()
        if errs:
            raise ValueError("; ".join(errs))
        require_past(effect.matured_through, now, f"scale effect {pattern_id}@{effect.horizon}")
        self._effects.append((pattern_id, effect))

    def record_transfer(self, rec: TransferRecord, now, evidence_years: Sequence[int] = ()) -> None:
        errs = rec.check()
        if errs:
            raise ValueError("; ".join(errs))
        require_past(rec.matured_through, now, f"transfer {rec.pattern_id}")
        self._transfers.append((rec, tuple(int(y) for y in evidence_years)))

    # ---- reads
    def patterns(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._decl) | {p for p, _ in self._effects} | {r.pattern_id for r, _ in self._transfers}))

    def declaration(self, pattern_id: str) -> Declaration | None:
        cur = self._decl.get(pattern_id)
        return cur[-1] if cur else None

    def coverage(self, pattern_id: str) -> MinuteCoverage | None:
        return self._coverage.get(pattern_id)

    def home(self, pattern_id: str) -> str | None:
        d = self.declaration(pattern_id)
        return d.horizon if d else None

    def effects(self, pattern_id: str, now, replay_years: Iterable[int] = ()) -> list[ScaleEffect]:
        now_d = as_date(now)
        ys = tuple(replay_years)
        return [e for p, e in self._effects if p == pattern_id and as_date(e.matured_through) < now_d
                and _visible(e.evidence_years, e.matured_through, now, ys)]

    def latest_effect(self, pattern_id: str, horizon: Any, now, replay_years: Iterable[int] = ()) -> ScaleEffect | None:
        key = parse_horizon(horizon).key
        es = [e for e in self.effects(pattern_id, now, replay_years) if e.horizon == key]
        return max(es, key=lambda e: e.matured_through) if es else None

    def transfers(self, pattern_id: str, now, replay_years: Iterable[int] = ()) -> list[TransferRecord]:
        now_d = as_date(now)
        ys = tuple(replay_years)
        return [r for r, ey in self._transfers if r.pattern_id == pattern_id and as_date(r.matured_through) < now_d
                and _visible(ey, r.matured_through, now, ys)]

    def transfer_between(self, pattern_id: str, source: Any, target: Any, now, replay_years: Iterable[int] = ()) -> TransferRecord | None:
        s, t = parse_horizon(source).key, parse_horizon(target).key
        rs = [r for r in self.transfers(pattern_id, now, replay_years) if r.source == s and r.target == t]
        return max(rs, key=lambda r: r.matured_through) if rs else None

    def transfer_table(self, pattern_id: str, now, replay_years: Iterable[int] = ()) -> pd.DataFrame:
        """Source x target matrix of verdict strings ('' where never tested)."""
        rs = self.transfers(pattern_id, now, replay_years)
        keys = sorted({r.source for r in rs} | {r.target for r in rs}, key=lambda k: (parse_horizon(k).sessions, parse_horizon(k).minutes))
        tab = pd.DataFrame("", index=keys, columns=keys)
        for r in sorted(rs, key=lambda r: r.matured_through):
            tab.loc[r.source, r.target] = r.verdict.value
        return tab

    def transfer_rate(self, now, replay_years: Iterable[int] = ()) -> dict[str, Any]:
        """Across all patterns: how often does an established effect carry to another scale? Measured, never assumed."""
        counts = {v.value: 0 for v in TransferVerdict}
        for p in self.patterns():
            latest: dict[tuple[str, str], TransferRecord] = {}
            for r in self.transfers(p, now, replay_years):
                k = (r.source, r.target)
                if k not in latest or r.matured_through > latest[k].matured_through:
                    latest[k] = r
            for r in latest.values():
                counts[r.verdict.value] += 1
        tested = counts["TRANSFERS"] + counts["NOT_TRANSFERRED"] + counts["REVERSES"]
        return {"counts": counts, "tested": tested,
                "transfer_share": None if tested == 0 else counts["TRANSFERS"] / tested,
                "reversal_share": None if tested == 0 else counts["REVERSES"] / tested}

    def matured_records(self, now, replay_years: Iterable[int] = ()) -> list[MaturedRecord]:
        """Identity-free MaturedRecords of every visible effect (no ticker, no date, no year in the payload)."""
        out = []
        for p in self.patterns():
            for e in self.effects(p, now, replay_years):
                payload = {"pattern": p, "horizon": e.horizon, "sessions": e.sessions, "effect": clean_number(e.effect),
                           "t": clean_number(e.t), "n_eff": clean_number(e.n_eff), "status": e.status}
                prov = Provenance(created_real=str(as_date(now)), learned_at=e.matured_through, code_hash=current_code_hash(),
                                  outcomes_seen_through=e.matured_through)
                out.append(MaturedRecord("MS" + stable_hash([p, e.horizon, e.matured_through], 10), e.matured_through, payload, prov,
                                         Namespace.MATURED_RESEARCH))
        return out

    # ---- persistence
    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION,
                "decl": {p: [dataclasses.asdict(d) for d in v] for p, v in self._decl.items()},
                "effects": [[p, dataclasses.asdict(e)] for p, e in self._effects],
                "transfers": [[dict(dataclasses.asdict(r), verdict=r.verdict.value), list(ey)] for r, ey in self._transfers],
                "coverage": {p: dataclasses.asdict(c) for p, c in self._coverage.items()}}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "ScaleLedger":
        if st.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"ledger schema {st.get('schema')!r} != {SCHEMA_VERSION}")
        led = cls()
        led._decl = {p: [Declaration(**d) for d in v] for p, v in st["decl"].items()}
        led._effects = [(p, ScaleEffect(**dict(e, evidence_years=tuple(e.get("evidence_years", ()))))) for p, e in st["effects"]]
        led._transfers = [(TransferRecord(**dict(r, verdict=TransferVerdict(r["verdict"]))), tuple(ey)) for r, ey in st["transfers"]]
        led._coverage = {p: MinuteCoverage(**c) for p, c in st["coverage"].items()}
        return led

    def content_hash(self) -> str:
        return stable_hash(self.state(), 16)


# ------------------------------------------------------------------------------------------------ enforcement

@dataclasses.dataclass(frozen=True)
class ScaleViolation:
    pattern_id: str
    used: str
    home: str | None
    kind: str                                 # NO_HORIZON | UNPROVEN_TRANSFER | REVERSES_AT_TARGET | COVERAGE_UNDECLARED | COVERAGE_OVERCLAIM
    detail: str


def enforce_horizons(uses: Iterable[tuple[str, Any]], ledger: ScaleLedger, now, replay_years: Iterable[int] = (),
                     claimed_years: Mapping[str, float] | None = None, raise_on_violation: bool = False) -> list[ScaleViolation]:
    """Audit (pattern_id, horizon_used) pairs. A use is legitimate only if the pattern has a declared horizon and the use is
    at that horizon (or an equal-length alias), or a recorded TRANSFERS verdict from the home horizon covers it. Minute-derived
    patterns must have declared coverage, and may not claim a span longer than the coverage supports.
    Raises UnprovenTransfer (a FirewallBreach) when raise_on_violation and anything is wrong."""
    out: list[ScaleViolation] = []
    claimed = dict(claimed_years or {})
    for pid, h in uses:
        spec = parse_horizon(h)
        home = ledger.home(pid)
        if home is None:
            out.append(ScaleViolation(pid, spec.key, None, "NO_HORIZON", "pattern carries no explicit horizon"))
            continue
        if not same_length(home, spec):
            rec = ledger.transfer_between(pid, home, spec, now, replay_years)
            if rec is None or rec.verdict == TransferVerdict.UNTESTED:
                out.append(ScaleViolation(pid, spec.key, home, "UNPROVEN_TRANSFER", f"never tested from {home} to {spec.key}"))
            elif rec.verdict == TransferVerdict.REVERSES:
                out.append(ScaleViolation(pid, spec.key, home, "REVERSES_AT_TARGET", f"effect flips sign at {spec.key}"))
            elif rec.verdict == TransferVerdict.NOT_TRANSFERRED:
                out.append(ScaleViolation(pid, spec.key, home, "UNPROVEN_TRANSFER", f"no effect at {spec.key}"))
        decl = ledger.declaration(pid)
        if spec.source == "minute_bars" or (decl and decl.intraday_source == "minute_bars"):
            cov = ledger.coverage(pid)
            if cov is None:
                out.append(ScaleViolation(pid, spec.key, home, "COVERAGE_UNDECLARED", "minute-derived pattern with no coverage limit"))
            elif pid in claimed and not cov.supports_span(claimed[pid]):
                out.append(ScaleViolation(pid, spec.key, home, "COVERAGE_OVERCLAIM",
                                          f"claims {claimed[pid]:.2f} years, minute data supports {cov.max_supported_years:.2f}"))
    if out and raise_on_violation:
        raise UnprovenTransfer(f"{len(out)} scale violation(s): " + "; ".join(f"{v.pattern_id}:{v.kind}" for v in out[:4]))
    return out


def audit_pattern_horizons(patterns: Iterable[Any]) -> list[str]:
    """Pattern ids (or reprs) that carry no readable explicit horizon. Duck-typed: a dict / Series with 'horizon', an object
    with .horizon, or a KnowledgeObject-like with .effect.horizon_days."""
    missing = []
    for p in patterns:
        h = None
        if isinstance(p, Mapping):
            h = p.get("horizon", p.get("horizon_days"))
        elif hasattr(p, "horizon"):
            h = p.horizon
        elif hasattr(getattr(p, "effect", None), "horizon_days"):
            h = p.effect.horizon_days
        ident = str(p.get("key_named", p.get("pattern_id", p))) if isinstance(p, Mapping) else \
            str(getattr(p, "knowledge_id", getattr(p, "pattern_id", p)))
        try:
            if h is None or (isinstance(h, float) and math.isnan(h)):
                raise ValueError("none")
            parse_horizon(h)
        except (ValueError, KeyError):
            missing.append(ident)
    return missing


# ------------------------------------------------------------------------------------------------ canon C67: OHLC intraday scale

def episode_path_horizons() -> tuple[tuple[int, ...], str]:
    """The next-path horizons in sessions (1, 2, 3, 5) SHARED with the mover-episode lab (engine.research.episodes, R21).
    R21 owns the definition: if its module exposes PATH_HORIZONS / NEXT_PATH_HORIZONS / NEXT_HORIZONS it is used and the source
    says so; otherwise the canon C67 fallback (1, 2, 3, 5) is used and the source says 'fallback'."""
    try:
        import importlib
        mod = importlib.import_module("engine.research.episodes")
    except Exception as e:                                   # not written yet, or broken: never let it stop research
        return FALLBACK_PATH_HORIZONS, f"fallback:{type(e).__name__}"
    for name in ("PATH_HORIZONS", "NEXT_PATH_HORIZONS", "NEXT_HORIZONS"):
        v = getattr(mod, name, None)
        if v:
            try:
                hs = tuple(sorted({int(x) for x in v}))
            except (TypeError, ValueError):
                continue
            if hs and all(h >= 1 for h in hs):
                return hs, f"engine.research.episodes.{name}"
    return FALLBACK_PATH_HORIZONS, "fallback:no_horizon_attribute"


FALLBACK_PATH_HORIZONS = (1, 2, 3, 5)


def path_horizon_keys() -> tuple[str, ...]:
    """Horizon keys of the next-path horizons, always resolvable in HORIZON_SPECS."""
    hs, _ = episode_path_horizons()
    return tuple(parse_horizon(h).key for h in hs)


def _wide(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    return past_only(frame, as_of, inclusive=True).astype("float64")


def ohlc_day_structure(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, as_of,
                       window: int = 20, dtype: str = "float32") -> pd.DataFrame:
    """The day's own intraday structure read off the daily bar - the only intraday scale with decades of history.
    Point in time: row t uses bars up to and including t (the decision is at the close), never later. Columns:
      range_pct       (H-L)/prev close                 gap            O/prev close - 1
      oc              C/O - 1                          cc             C/prev close - 1
      close_loc       (C-L)/(H-L) in [0,1]             body_share     |C-O|/(H-L)
      upper_wick, lower_wick  shares of the range      gap_follow     (C-O)*sign(gap)/prev close: >0 gap-and-go, <0 gap-and-fade
      range_dod       range_t / range_{t-1}            range_exp      range_t / mean(range over the previous `window` days)
      move_z          cc / trailing std of cc (previous `window` days)
    Zero-range days give NaN location shares (never 0 or 0.5). Stacked (date, ticker)."""
    O, H, L, C = (_wide(x, as_of) for x in (open_, high, low, close))
    idx, cols = C.index, C.columns
    O, H, L = O.reindex(index=idx, columns=cols), H.reindex(index=idx, columns=cols), L.reindex(index=idx, columns=cols)
    prev = C.shift(1)
    rng = (H - L)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng_safe = rng.where(rng > 0)
        rpct = rng / prev
        cc = C / prev - 1
        gap = O / prev - 1
        body_top, body_bot = np.maximum(O, C), np.minimum(O, C)
        feats = {
            "range_pct": rpct, "gap": gap, "oc": C / O - 1, "cc": cc,
            "close_loc": (C - L) / rng_safe, "body_share": (C - O).abs() / rng_safe,
            "upper_wick": (H - body_top) / rng_safe, "lower_wick": (body_bot - L) / rng_safe,
            "gap_follow": (C - O) * np.sign(gap) / prev,
            "range_dod": rpct / rpct.shift(1).where(rpct.shift(1) > 0),
            "range_exp": rpct / rpct.shift(1).rolling(window, min_periods=max(window // 2, 3)).mean().where(lambda x: x > 0),
            "move_z": cc / cc.shift(1).rolling(window, min_periods=max(window // 2, 3)).std().where(lambda x: x > 0),
        }
    stacked = {k: v.replace([np.inf, -np.inf], np.nan).stack(future_stack=True) for k, v in feats.items()}
    out = pd.DataFrame(stacked).astype(dtype)
    out.index.names = ["date", "ticker"]
    return out


def next_path_outcomes(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, horizons: Iterable[int],
                       now, dtype: str = "float32") -> pd.DataFrame:
    """What the price did over the next h sessions after each decision close - the RESEARCH-world labels of canon C67
    (MATURED_RESEARCH_STATE: they use the future by definition). Built from bars strictly before `now`, so a row whose h-th
    next close is not yet behind `now` is NaN. For each h:
      nxt_ret_h        log(C[t+h]/C[t])                       nxt_open_ret_h   log(C[t+h]/O[t+1])  (what a next-open fill could keep)
      nxt_cont_h       nxt_ret_h * sign(cc[t])                (>0: continued the day's move, <0: reversed it)
      nxt_range_h      mean range over t+1..t+h / range[t]    (>1: the volatility expanded, <1: it consolidated)
      nxt_ext_h, nxt_draw_h   max high / min low over t+1..t+h relative to C[t]
      nxt_gap_1        O[t+1]/C[t] - 1
    Stacked (date, ticker)."""
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1:
        raise ValueError("next_path_outcomes needs horizons >= 1")
    O, H, L, C = (past_only(x, now).astype("float64") for x in (open_, high, low, close))
    if C.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    O, H, L = O.reindex(index=C.index, columns=C.columns), H.reindex(index=C.index, columns=C.columns), L.reindex(index=C.index, columns=C.columns)
    rng = (H - L) / C.shift(1)
    rng0 = rng.where(rng > 0)
    cc = C / C.shift(1) - 1
    feats: dict[str, pd.DataFrame] = {"nxt_gap_1": O.shift(-1) / C - 1}
    with np.errstate(divide="ignore", invalid="ignore"):
        for h in hs:
            ret = np.log(C.shift(-h) / C)
            feats[f"nxt_ret_{h}"] = ret
            feats[f"nxt_open_ret_{h}"] = np.log(C.shift(-h) / O.shift(-1))
            feats[f"nxt_cont_{h}"] = ret * np.sign(cc)
            avg_next = rng.shift(-1).rolling(h).mean().shift(-(h - 1))
            feats[f"nxt_range_{h}"] = avg_next / rng0
            feats[f"nxt_ext_{h}"] = H.shift(-1).rolling(h).max().shift(-(h - 1)) / C - 1
            feats[f"nxt_draw_{h}"] = L.shift(-1).rolling(h).min().shift(-(h - 1)) / C - 1
    stacked = {k: v.replace([np.inf, -np.inf], np.nan).stack(future_stack=True) for k, v in feats.items()}
    out = pd.DataFrame(stacked).astype(dtype)
    out.index.names = ["date", "ticker"]
    return out


def bars_to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Collapse one session of minute bars ((ts, ticker) index; Open/High/Low/Close/Volume) to a daily O/H/L/C/V per ticker."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    b = bars.sort_index(level=0)
    g = b.groupby(level=1)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(), "Close": g["Close"].last(),
                         "Volume": g["Volume"].sum()})


def minute_session_features(bars: pd.DataFrame, session, now, interval_minutes: int = 5, open_minute: int = 570) -> pd.DataFrame:
    """Per-ticker features of ONE completed session of minute bars, for a session dated strictly before `now`.
      rv            sum of squared log bar returns             first30   return over the first 30 minutes from the open
      last60        return over the last 60 minutes            open_range (max high - min low in first 30 min)/open
      vol_first30   share of the day's volume in the first 30 minutes
    Minute data exist only for recent months (collector limit): the result carries no claim beyond that coverage."""
    require_past(session, now, "minute session")
    cols = ["rv", "first30", "last60", "open_range", "vol_first30"]
    if bars is None or bars.empty:
        return pd.DataFrame(columns=cols, dtype="float64")
    b = bars.sort_index(level=0).copy()
    ts = pd.DatetimeIndex(b.index.get_level_values(0))
    day = pd.Timestamp(as_date(session))
    if ts.normalize().min() != day or ts.normalize().max() != day:
        raise FirewallBreach("minute bars span more than the declared session")
    b["mins"] = ts.hour * 60 + ts.minute - open_minute
    rows = {}
    for tkr, g in b.groupby(level=1):
        g = g.droplevel(1)
        c = g["Close"].to_numpy(dtype="float64")
        if len(c) < 6 or not np.isfinite(c).all() or (c <= 0).any():
            continue
        r = np.diff(np.log(c))
        first = g[g["mins"] < 30]
        last = g[g["mins"] >= g["mins"].max() - 59]
        o0 = float(g["Open"].iloc[0])
        tot_v = float(g["Volume"].sum())
        rows[tkr] = {"rv": float((r ** 2).sum()),
                     "first30": float(first["Close"].iloc[-1] / o0 - 1) if len(first) else np.nan,
                     "last60": float(c[-1] / last["Open"].iloc[0] - 1) if len(last) else np.nan,
                     "open_range": float((first["High"].max() - first["Low"].min()) / o0) if len(first) else np.nan,
                     "vol_first30": float(first["Volume"].sum() / tot_v) if tot_v > 0 and len(first) else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def ohlc_vs_minute_agreement(daily_row: pd.DataFrame, minute_daily: pd.DataFrame) -> dict[str, Any]:
    """Do the daily bar and the day rebuilt from minute bars describe the same session? Both are ticker-indexed with
    Open/High/Low/Close. The OHLC-derived intraday scale is only trusted where this agrees on the overlap (the two sources
    differ in feed, adjustment and extended hours). Returns median absolute range difference and close-location correlation."""
    common = daily_row.index.intersection(minute_daily.index)
    if len(common) < 5:
        return {"n": int(len(common)), "corr_close_loc": None, "median_abs_range_diff": None, "agrees": None}
    a, b = daily_row.loc[common], minute_daily.loc[common]
    with np.errstate(divide="ignore", invalid="ignore"):
        la = (a["Close"] - a["Low"]) / (a["High"] - a["Low"]).where(a["High"] > a["Low"])
        lb = (b["Close"] - b["Low"]) / (b["High"] - b["Low"]).where(b["High"] > b["Low"])
        ra = (a["High"] - a["Low"]) / a["Open"]
        rb = (b["High"] - b["Low"]) / b["Open"]
    ok = la.notna() & lb.notna()
    corr = float(np.corrcoef(la[ok], lb[ok])[0, 1]) if ok.sum() >= 5 and la[ok].std() > 0 and lb[ok].std() > 0 else None
    diff = float((ra - rb).abs().median())
    return {"n": int(len(common)), "corr_close_loc": corr, "median_abs_range_diff": diff,
            "agrees": None if corr is None else bool(corr > 0.8 and diff < 0.01)}


def intraday_scale_report(pattern_ids: Iterable[str], ledger: ScaleLedger, now) -> list[dict[str, Any]]:
    """For every intraday-scale pattern: its source (daily_ohlc vs minute_bars) and, for minute patterns, the coverage limit
    it declared. Undeclared coverage is reported as such - never as a default."""
    rows = []
    for pid in pattern_ids:
        d = ledger.declaration(pid)
        if d is None or not parse_horizon(d.horizon).is_intraday:
            continue
        cov = ledger.coverage(pid)
        rows.append({"pattern": pid, "horizon": d.horizon, "source": d.intraday_source or "unknown",
                     "coverage_declared": cov is not None,
                     "coverage_sessions": None if cov is None else cov.sessions,
                     "max_supported_years": None if cov is None else round(cov.max_supported_years, 3)})
    return rows


# ------------------------------------------------------------------------------------------------ scale diagnostics

def variance_ratio(returns: Sequence[float], q: int) -> tuple[float | None, float | None]:
    """Lo-MacKinlay variance ratio VR(q) and its heteroskedasticity-robust z*. VR > 1: returns trend at horizon q (positive
    autocorrelation); VR < 1: they mean-revert. (None, None) when there are too few observations or zero variance."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if q < 2 or n < 4 * q or n < 20:
        return None, None
    mu = r.mean()
    x = r - mu
    s1 = float((x ** 2).sum()) / (n - 1)
    if s1 <= 1e-18:
        return None, None
    cum = np.cumsum(np.insert(r, 0, 0.0))
    rq = cum[q:] - cum[:-q]
    m = q * (n - q + 1) * (1.0 - q / n)
    sq = float(((rq - q * mu) ** 2).sum()) / m
    vr = sq / s1
    denom = float((x ** 2).sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * float((x[j:] ** 2 * x[:-j] ** 2).sum()) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta
    if theta <= 0:
        return vr, None
    return float(vr), float(math.sqrt(n) * (vr - 1.0) / math.sqrt(theta))


def hurst_exponent(returns: Sequence[float], scales: Sequence[int] = (1, 2, 4, 8, 16), min_blocks: int = 8) -> float | None:
    """Aggregated-variance Hurst exponent: std of non-overlapping m-sums scales as m^H. H > 0.5 persistent, < 0.5 anti-persistent.
    None when the sample cannot give >= min_blocks blocks at the largest scale or fewer than three usable scales."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    xs, ys = [], []
    for m in scales:
        k = len(r) // m
        if k < min_blocks:
            continue
        s = r[:k * m].reshape(k, m).sum(axis=1).std(ddof=1)
        if s > 0:
            xs.append(math.log(m))
            ys.append(math.log(s))
    if len(xs) < 3:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


class ScaleRegime(_StrEnum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    RANDOM = "RANDOM"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class MarketScaleState:
    """How the market's own returns behave at each scale, from data up to `as_of`. regimes.py reads `persistence` for its
    trend / mean-reversion axis; disagreement between scales (trending at 2 sessions, reverting at 21) is kept, not averaged away."""
    by_q: Mapping[int, ScaleRegime]
    vr: Mapping[int, float | None]
    z: Mapping[int, float | None]
    hurst: float | None
    persistence: float | None                 # mean of (VR - 1) over scales that could be computed; None = unknown
    n_obs: int

    def dominant(self) -> ScaleRegime:
        vals = [v for v in self.by_q.values() if v != ScaleRegime.UNKNOWN]
        if not vals:
            return ScaleRegime.UNKNOWN
        for r in (ScaleRegime.TRENDING, ScaleRegime.MEAN_REVERTING):
            if sum(v == r for v in vals) > len(vals) / 2:
                return r
        return ScaleRegime.RANDOM

    def scales_disagree(self) -> bool:
        s = {v for v in self.by_q.values() if v in (ScaleRegime.TRENDING, ScaleRegime.MEAN_REVERTING)}
        return len(s) == 2


def market_scale_state(returns: pd.Series, as_of, qs: Sequence[int] = (2, 5, 10, 21), z_bar: float = 2.0, window: int = 756,
                       min_obs: int = 120) -> MarketScaleState:
    """Scale behaviour of a market return series using rows dated <= as_of (the last `window`). Fewer than `min_obs` -> UNKNOWN."""
    r = past_only(returns.dropna(), as_of, inclusive=True).iloc[-window:]
    n = len(r)
    by_q, vrs, zs = {}, {}, {}
    for q in qs:
        vr, z = variance_ratio(r.to_numpy(), q) if n >= min_obs else (None, None)
        vrs[q], zs[q] = vr, z
        if vr is None or z is None:
            by_q[q] = ScaleRegime.UNKNOWN
        elif z >= z_bar and vr > 1:
            by_q[q] = ScaleRegime.TRENDING
        elif z <= -z_bar and vr < 1:
            by_q[q] = ScaleRegime.MEAN_REVERTING
        else:
            by_q[q] = ScaleRegime.RANDOM
    known = [v - 1.0 for v in vrs.values() if v is not None]
    return MarketScaleState(by_q, vrs, zs, hurst_exponent(r.to_numpy()) if n >= min_obs else None,
                            float(np.mean(known)) if known else None, n)


def scaling_table(states: Mapping[str, MarketScaleState]) -> pd.DataFrame:
    """One row per label (e.g. an era or a series name): VR and regime at every scale, for the report."""
    rows = []
    for name, st in states.items():
        row: dict[str, Any] = {"series": name, "hurst": st.hurst, "persistence": st.persistence, "n": st.n_obs}
        for q in st.by_q:
            row[f"vr{q}"] = st.vr[q]
            row[f"reg{q}"] = st.by_q[q].value
        rows.append(row)
    return pd.DataFrame(rows).set_index("series") if rows else pd.DataFrame()


def session_components(open_: pd.DataFrame, close: pd.DataFrame, as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overnight, intraday) log returns per session up to as_of: overnight = log(O_t / C_{t-1}), intraday = log(C_t / O_t).
    The overnight gap cannot be captured by a decision at the previous close (fills are at the next open), the intraday leg can."""
    O, C = _wide(open_, as_of), _wide(close, as_of)
    O = O.reindex(index=C.index, columns=C.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(O / C.shift(1)), np.log(C / O)


def capturable_split(flag: pd.Series, open_: pd.DataFrame, close: pd.DataFrame, now, min_dates: int = MIN_DATES) -> dict[str, Any]:
    """Split a flag's 1-session effect into the OVERNIGHT gap to the next open (not capturable by a close decision) and the
    INTRADAY leg from that open to the next close (capturable). capturable_share = intraday / (overnight + intraday), reported
    only when both legs share a sign; a pattern whose whole effect is the overnight gap has an edge nobody can fill."""
    O, C = past_only(open_, now).astype("float64"), past_only(close, now).astype("float64")
    O = O.reindex(index=C.index, columns=C.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        over = np.log(O.shift(-1) / C)
        intra = np.log(C.shift(-1) / O.shift(-1))
    res: dict[str, Any] = {}
    for name, w in (("overnight", over), ("intraday", intra)):
        s = w.replace([np.inf, -np.inf], np.nan).stack(future_stack=True).dropna()
        s.index.names = ["date", "ticker"]
        res[name] = measure_effect(flag, s, "1d", now, C.index, min_dates)
    o, i = res["overnight"], res["intraday"]
    share = None
    if o.effect is not None and i.effect is not None and o.effect * i.effect > 0:
        share = i.effect / (o.effect + i.effect)
    res["capturable_share"] = share
    res["all_overnight"] = bool(o.established() and not i.established())
    return res


# ------------------------------------------------------------------------------------------------ the public entry

@dataclasses.dataclass(frozen=True)
class ScaleStepResult:
    n_patterns: int
    n_effects: int
    n_transfers: int
    horizons: tuple[str, ...]
    shapes: Mapping[str, str]
    homes: Mapping[str, str | None]
    undeclared: tuple[str, ...]
    skipped: Mapping[str, str]
    violations: tuple[ScaleViolation, ...]
    ledger_hash: str

    def ok(self) -> bool:
        return not self.violations and not self.undeclared


def step(ledger: ScaleLedger, now, flags: Mapping[str, pd.Series], close: pd.DataFrame, open_: pd.DataFrame | None = None,
         declared: Mapping[str, Any] | None = None, uses: Iterable[tuple[str, Any]] = (), replay_years: Iterable[int] = (),
         horizons: Sequence[Any] | None = None, profile: DataProfile | None = None, min_dates: int = MIN_DATES,
         t_bar: float = T_BAR, intraday_root: Path | str | None = None) -> ScaleStepResult:
    """ONE research-loop step at `now`: measure every registered pattern flag at every testable session horizon on matured
    labels only, record the effects and every pairwise transfer verdict, make sure each pattern carries an explicit horizon
    (its design horizon from `declared`, else the measured home), and audit `uses` for un-earned transfers.
    flags: {pattern_id: (date, ticker) bool Series}. Nothing dated at/after `now` is read."""
    C = past_only(close, now)
    prof = profile or DataProfile(len(C), C.shape[1], intraday_inventory(now, intraday_root))
    keys = tuple(parse_horizon(h).key for h in horizons) if horizons is not None else testable_keys(prof)
    keys = tuple(k for k in keys if not parse_horizon(k).is_intraday)
    fwd = forward_map(close, keys, now, open_) if keys else {}
    cal = C.index
    skipped: dict[str, str] = {}
    shapes: dict[str, str] = {}
    homes: dict[str, str | None] = {}
    undeclared: list[str] = []
    n_eff = n_tr = 0
    for pid in sorted(flags):
        if not fwd:
            skipped[pid] = "no testable session horizon in the data"
            continue
        prof_p = effect_profile(flags[pid], fwd, now, cal, t_bar, min_dates)
        shapes[pid] = prof_p.shape.value
        homes[pid] = prof_p.home()
        for e in prof_p.effects:
            if e.status == "OK":
                ledger.record_effect(pid, e, now)
                n_eff += 1
        for rec in transfer_records(pid, prof_p):
            ys = sorted({y for e in prof_p.effects if e.horizon in (rec.source, rec.target) for y in e.evidence_years})
            ledger.record_transfer(rec, now, ys)
            n_tr += 1
        want = (declared or {}).get(pid) or homes[pid]
        if want is not None:
            ledger.declare(pid, want, now, declared_by="design" if (declared or {}).get(pid) else "measured")
        if ledger.home(pid) is None:
            undeclared.append(pid)
    viol = enforce_horizons(uses, ledger, now, replay_years)
    return ScaleStepResult(len(flags), n_eff, n_tr, keys, shapes, homes, tuple(undeclared), skipped, tuple(viol),
                           ledger.content_hash())


def render_report(ledger: ScaleLedger, now, replay_years: Iterable[int] = ()) -> str:
    """Plain-text scale report: per pattern the declared horizon, the effect at each measured horizon and the transfer table;
    then the cross-pattern transfer rate. Identity-free (no tickers)."""
    lines = [f"MULTI-SCALE LEDGER  ({SCHEMA_VERSION})  hash {ledger.content_hash()}"]
    for pid in ledger.patterns():
        d = ledger.declaration(pid)
        lines.append(f"- {pid}  home={d.horizon if d else 'NONE'}"
                     + (f"  source={d.intraday_source}" if d and d.intraday_source else ""))
        latest: dict[str, ScaleEffect] = {}
        for e in ledger.effects(pid, now, replay_years):
            if e.horizon not in latest or e.matured_through > latest[e.horizon].matured_through:
                latest[e.horizon] = e
        for e in sorted(latest.values(), key=lambda e: e.sessions):
            lines.append(f"    {e.horizon:>5}  effect {e.effect:+.5f}  t {e.t if e.t is None else round(e.t, 2)}  "
                         f"n_eff {e.n_eff:.1f}  {'ESTABLISHED' if e.established() else 'not established'}")
        tab = ledger.transfer_table(pid, now, replay_years)
        if not tab.empty:
            lines.append("    transfer (source rows -> target columns):")
            lines.extend("      " + ln for ln in tab.to_string().splitlines())
    rate = ledger.transfer_rate(now, replay_years)
    lines.append(f"transfer share {rate['transfer_share']}  reversal share {rate['reversal_share']}  counts {rate['counts']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ multiplicity, folds, eras

def t_to_p(t: float | None) -> float | None:
    """Two-sided normal p-value of a t statistic (None stays None: an unmeasured horizon has no p-value)."""
    if t is None or not math.isfinite(t):
        return None
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def benjamini_hochberg(pvals: Sequence[float | None]) -> list[float | None]:
    """BH-adjusted q-values; None entries stay None and do not count toward the number of tests."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out: list[float | None] = [None] * len(pvals)
    if not idx:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        out[i] = prev
    return out


def profile_fdr(profile: EffectProfile, q: float = 0.05) -> dict[str, dict[str, Any]]:
    """Measuring one pattern at K horizons is K tests. Returns {horizon: {p, q_value, survives}} with BH across the horizons
    (a horizon 'survives' when its q-value <= q AND its |t| clears the profile's bar)."""
    effs = list(profile.effects)
    p = [t_to_p(e.t) if e.status == "OK" else None for e in effs]
    qv = benjamini_hochberg(p)
    return {e.horizon: {"p": pp, "q_value": qq, "survives": bool(qq is not None and qq <= q and e.established(profile.t_bar))}
            for e, pp, qq in zip(effs, p, qv)}


def embargoed_folds(n_dates: int, sessions: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged, embargoed walk-forward-style folds over decision-date positions 0..n_dates-1. Each test block is contiguous;
    training positions within `sessions` before the block (their labels reach into it) and within `sessions` after it (their
    features straddle its labels) are removed, so no training label overlaps a test label. Returns [(train, test), ...]."""
    if n_folds < 2 or n_dates < n_folds * 2:
        raise ValueError("need n_folds >= 2 and at least two dates per fold")
    edges = np.linspace(0, n_dates, n_folds + 1).astype(int)
    pos = np.arange(n_dates)
    folds = []
    gap = max(int(sessions), 1)
    for a, b in zip(edges[:-1], edges[1:]):
        test = pos[a:b]
        keep = (pos < a - gap) | (pos >= b + gap)
        folds.append((pos[keep], test))
    return folds


def labels_overlap(train_pos: np.ndarray, test_pos: np.ndarray, sessions: int) -> bool:
    """True if any training label window [t, t+sessions] intersects a test decision position - the check embargoed_folds must pass."""
    if len(train_pos) == 0 or len(test_pos) == 0:
        return False
    lo, hi = int(test_pos.min()), int(test_pos.max())
    ends = train_pos + max(int(sessions), 1)
    return bool(((train_pos <= hi) & (ends >= lo)).any())


def era_profile(flag: pd.Series, fwd_map: Mapping[str, pd.Series], now, era_edges: Sequence[Any],
                calendar: pd.DatetimeIndex | None = None, min_dates: int = MIN_DATES) -> dict[str, dict[str, ScaleEffect]]:
    """The horizon profile measured separately in each era (edges are dates: [e0, e1), [e1, e2), ...). A horizon effect that
    holds only in one era is era-bound, not a property of the scale. Returns {era_label: {horizon: ScaleEffect}}."""
    edges = [pd.Timestamp(as_date(e)) for e in era_edges]
    out: dict[str, dict[str, ScaleEffect]] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        label = f"era{len(out)}"
        row = {}
        for k, s in fwd_map.items():
            d = _dates_of(s)
            sub = s[(d >= a) & (d < b)]
            row[k] = measure_effect(flag, sub, k, now, calendar, min_dates) if len(sub) else \
                _insufficient(parse_horizon(k), 0, 0, str(as_date(now)), Unknown.INSUFFICIENT_DATA.value)
        out[label] = row
    return out


def era_consistency(profiles: Mapping[str, Mapping[str, ScaleEffect]], t_bar: float = T_BAR) -> dict[str, Any]:
    """Per horizon: in how many eras is the effect established, and do the established eras agree on sign?"""
    horizons = sorted({k for r in profiles.values() for k in r}, key=lambda k: parse_horizon(k).sessions)
    out = {}
    for k in horizons:
        es = [r[k] for r in profiles.values() if k in r and r[k].status == "OK"]
        est = [e for e in es if e.established(t_bar)]
        signs = {e.sign for e in est}
        out[k] = {"eras_measured": len(es), "eras_established": len(est), "sign_agrees": len(signs) <= 1,
                  "universal": bool(len(es) >= 2 and len(est) == len(es) and len(signs) == 1)}
    return out


# ------------------------------------------------------------------------------------------------ canon C67 path tables

def _path_horizon(col: str) -> int | None:
    m = re.search(r"_(\d+)$", col)
    return int(m.group(1)) if m else None


def path_effect_table(flag: pd.Series, outcomes: pd.DataFrame, now, calendar: pd.DatetimeIndex | None = None,
                      min_dates: int = MIN_DATES, min_side: int = 1) -> pd.DataFrame:
    """Effect of a flag (e.g. 'a 5-10% mover today') on every next-path outcome column from next_path_outcomes(), at the
    horizon the column names (1, 2, 3, 5 sessions), each with its own overlap-corrected standard error. One row per column."""
    rows = []
    for col in outcomes.columns:
        h = _path_horizon(col)
        if h is None:
            continue
        e = measure_effect(flag, outcomes[col].astype("float64").dropna(), h, now, calendar, min_dates, min_side)
        rows.append({"outcome": col, "horizon": e.horizon, "sessions": h, "effect": e.effect, "se": e.se, "t": e.t,
                     "n_dates": e.n_dates, "established": e.established(), "status": e.status})
    cols = ["outcome", "horizon", "sessions", "effect", "se", "t", "n_dates", "established", "status"]
    return pd.DataFrame(rows, columns=cols)


def horizon_move_scale(close: pd.DataFrame, sessions: Sequence[int], now) -> pd.DataFrame:
    """How large is a typical move at each horizon, and does it grow like sqrt(h)? Per horizon: the cross-sectional median
    of |log return| pooled over dates, its ratio to sqrt(h) times the 1-session figure (1.0 = diffusive; >1 trending; <1
    mean-reverting), and the share of name-windows beyond 5% and 10% (the canon C67 mover bands). Uses closes before `now`."""
    C = past_only(close, now).astype("float64")
    rows = []
    base = None
    for h in sorted({int(x) for x in sessions}):
        if h < 1:
            raise ValueError("horizon sessions must be >= 1")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(C.shift(-h) / C).to_numpy().ravel()
        r = r[np.isfinite(r)]
        if len(r) == 0:
            rows.append({"sessions": h, "median_abs": None, "sqrt_ratio": None, "share_5pct": None, "share_10pct": None, "n": 0})
            continue
        med = float(np.median(np.abs(r)))
        if h == 1 or base is None:
            base = med / math.sqrt(h) if base is None else base
        rows.append({"sessions": h, "median_abs": med, "sqrt_ratio": med / (base * math.sqrt(h)) if base else None,
                     "share_5pct": float((np.abs(r) >= 0.05).mean()), "share_10pct": float((np.abs(r) >= 0.10).mean()), "n": int(len(r))})
    return pd.DataFrame(rows).set_index("sessions") if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ what to research next

def untested_transfers(ledger: ScaleLedger, now, replay_years: Iterable[int] = (),
                       targets: Sequence[Any] = DAILY_LADDER) -> list[tuple[str, str, str]]:
    """(pattern, home, target) triples where the pattern is established at home but never tested at a target scale."""
    out = []
    for pid in ledger.patterns():
        home = ledger.home(pid)
        if home is None or parse_horizon(home).is_intraday:
            continue
        e = ledger.latest_effect(pid, home, now, replay_years)
        if e is None or not e.established():
            continue
        for t in targets:
            key = parse_horizon(t).key
            if same_length(home, key) or ledger.transfer_between(pid, home, key, now, replay_years) is not None:
                continue
            out.append((pid, home, key))
    return out


def transfer_questions(ledger: ScaleLedger, now, created_real: str, evidence_through: str, replay_years: Iterable[int] = (),
                       limit: int = 20):
    """Turn every untested transfer into an identity-free ResearchQuestion (section 40). Ordered by the home effect's |t|, so
    the strongest established patterns are asked about first."""
    from engine.research.core import Problem, ResearchQuestion
    ranked = []
    for pid, home, tgt in untested_transfers(ledger, now, replay_years):
        e = ledger.latest_effect(pid, home, now, replay_years)
        ranked.append((abs(e.t), pid, home, tgt))
    ranked.sort(key=lambda r: (-r[0], r[1], r[3]))
    return [ResearchQuestion.make(f"Does pattern {pid} established at {home} hold at {tgt}?", "scale_transfer", Problem.VOLATILITY,
                                  created_real, evidence_through,
                                  f"same-sign effect with |t| >= {T_BAR} at {tgt}", f"|t| < {T_BAR} or opposite sign at {tgt}")
            for _, pid, home, tgt in ranked[:limit]]


def ledger_health(ledger: ScaleLedger, now, replay_years: Iterable[int] = ()) -> dict[str, list[str]]:
    """Invariant checks on the ledger. Empty lists mean healthy; a non-empty list names the offending pattern ids."""
    issues: dict[str, list[str]] = {"no_horizon": [], "home_unmeasured": [], "transfer_without_source": [],
                                    "minute_without_coverage": [], "bad_record": []}
    for pid in ledger.patterns():
        d = ledger.declaration(pid)
        if d is None:
            issues["no_horizon"].append(pid)
            continue
        errs = d.check()
        if errs:
            issues["bad_record"].append(pid)
        spec = parse_horizon(d.horizon)
        if not spec.is_intraday and ledger.latest_effect(pid, d.horizon, now, replay_years) is None \
                and d.declared_by == "measured":
            issues["home_unmeasured"].append(pid)
        for r in ledger.transfers(pid, now, replay_years):
            src = ledger.latest_effect(pid, r.source, now, replay_years)
            if src is None or (r.verdict in (TransferVerdict.TRANSFERS, TransferVerdict.REVERSES) and not src.established()):
                issues["transfer_without_source"].append(pid)
                break
        if (spec.source == "minute_bars" or d.intraday_source == "minute_bars") and ledger.coverage(pid) is None:
            issues["minute_without_coverage"].append(pid)
    return issues
