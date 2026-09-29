"""Regime-aware research (research contract C66 section 25; supports 13, 14, 26, 31, 48; canon C56, C58, C63, C64, C66, C67).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; C63: no real-data run was made).

The contract: continually monitor bull / bear, high / low volatility, high / low dispersion, high / low liquidity, trend /
mean-reversion and event-heavy / event-light; do not hard-code them as final truths; discover new regimes; a pattern that works in
one regime is never universal; learn pattern x regime, not pattern alone.

What lives here
  * RegimeMonitor (streaming, one day at a time): six named axes, each a past-only indicator classified against the PAST
    distribution of that same indicator (quantile bands, hysteresis so a threshold graze does not flip the state, UNKNOWN until
    enough history). The 'trend vs mean-reversion' axis is measured (variance ratio of market returns, engine.research.multiscale),
    not assumed. Nothing here is a fixed level: thresholds come from the past and move.
  * DiscoveredRegimes: unsupervised, past-only clustering of the market state vector. The number of clusters is CHOSEN by silhouette
    (one cluster = 'no structure found' is a legal answer), ids stay stable across refits by centroid matching, and a cluster only
    counts as a regime if its label sequence persists far longer than the same labels shuffled in time (dwell_ratio): clusters
    of pure noise switch every day and are rejected as REJECTED_NOISE.
  * PatternRegimeBook: for each pattern, the daily effect series split by the state of every axis (and by discovered regime).
    Per axis it tests every state against zero and against the other states with overlap-corrected errors, Benjamini-Hochberg
    across all pattern x axis contrasts, and issues a verdict: UNIVERSAL, REGIME_BOUND, REGIME_REVERSING, NOT_ESTABLISHED or
    INSUFFICIENT_DATA. regime_gate() turns the verdict into allowed / not allowed / abstain for the current regime.
  * step(): the ONE public entry the research loop calls.

Firewall: pattern x regime evidence is MATURED_RESEARCH_STATE. Rows are added only once matured strictly before `now`, dropped while
their year is being replayed in disguise (rule 27), and leave through MaturedRecord (identity-free payload). The trader sees only
ordinal state codes (trader_regime_row, checked by engine.learning.trader_view.assert_trader_safe).

Built on: engine.learning.situation.classify_regime (the composite label the Situation representation already uses),
engine.research.multiscale (variance_ratio, hac_se, benjamini_hochberg, t_to_p, past_only), engine.research.cross_section (kmeans,
silhouette, robust_scale), engine.research.core, engine.learning.core."""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, current_code_hash, require_past, stable_hash)
from engine.learning.situation import clean_number, classify_regime
from engine.learning.trader_view import assert_trader_safe
from engine.research.core import MaturedRecord, Namespace, Problem, ResearchQuestion
from engine.research.cross_section import kmeans, robust_scale, silhouette
from engine.research.multiscale import benjamini_hochberg, hac_se, t_to_p, variance_ratio

SCHEMA_VERSION = "regimes.v1"
UNKNOWN_STATE = "unknown"
NEUTRAL = "neutral"
T_BAR = 2.0


@dataclasses.dataclass(frozen=True)
class AxisSpec:
    """One regime axis. mode 'quantile': hi_state when the indicator is above its past q_hi quantile, lo_state below q_lo;
    mode 'signed': hi_state above +band robust sigmas of the past, lo_state below -band (centred on zero, e.g. trend)."""
    name: str
    indicator: str
    hi_state: str
    lo_state: str
    mode: str = "quantile"

    def states(self) -> tuple[str, ...]:
        return (self.hi_state, NEUTRAL, self.lo_state)


AXES: tuple[AxisSpec, ...] = (
    AxisSpec("direction", "trend200", "bull", "bear", "signed"),
    AxisSpec("volatility", "vol", "high_vol", "low_vol"),
    AxisSpec("dispersion", "dispersion", "high_dispersion", "low_dispersion"),
    AxisSpec("liquidity", "log_dv", "high_liquidity", "low_liquidity"),
    AxisSpec("character", "persistence", "trend", "mean_reversion", "signed"),
    AxisSpec("events", "event_share", "event_heavy", "event_light"),
)
AXIS_BY_NAME = {a.name: a for a in AXES}
STATE_CODE = {"bull": 1, "high_vol": 1, "high_dispersion": 1, "high_liquidity": 1, "trend": 1, "event_heavy": 1,
              "bear": -1, "low_vol": -1, "low_dispersion": -1, "low_liquidity": -1, "mean_reversion": -1, "event_light": -1,
              NEUTRAL: 0}


@dataclasses.dataclass(frozen=True)
class RegimeConfig:
    q_lo: float = 0.33
    q_hi: float = 0.67
    band: float = 0.35                        # signed axes: robust sigmas of the past either side of zero
    hysteresis: float = 0.15                  # share of the neutral band that must be crossed to leave the previous state
    min_history: int = 60
    history_days: int = 1000
    vol_window: int = 20
    persistence_window: int = 120
    persistence_q: int = 5
    ma_window: int = 200
    dwell_ratio_min: float = 2.0              # discovered regime persistence vs the same labels shuffled
    refit_every: int = 40
    fit_window: int = 500
    k_max: int = 5
    min_silhouette: float = 0.25
    min_cluster_share: float = 0.05

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.q_lo < self.q_hi < 1:
            errs.append("need 0 < q_lo < q_hi < 1")
        if self.band <= 0 or not 0 <= self.hysteresis < 1:
            errs.append("band > 0 and 0 <= hysteresis < 1 required")
        if self.min_history < 30 or self.history_days < self.min_history:
            errs.append("min_history >= 30 and history_days >= min_history required")
        if self.persistence_window < 4 * self.persistence_q:
            errs.append("persistence_window must be >= 4 * persistence_q")
        if self.k_max < 2 or self.dwell_ratio_min <= 1:
            errs.append("k_max >= 2 and dwell_ratio_min > 1 required")
        return errs


INDICATORS = tuple(a.indicator for a in AXES)
RAW_FIELDS = ("ret", "vix", "dispersion", "dollar_volume", "event_share", "breadth", "spy_ma200")


def validate_market_row(row: Mapping[str, Any]) -> list[str]:
    """Problems with one day's market summary. 'ret' is required; everything else may be missing (that axis is then UNKNOWN)."""
    errs = []
    if row is None or "ret" not in row or clean_number(row.get("ret")) is None:
        return ["market return 'ret' missing or non-finite"]
    if abs(float(row["ret"])) > 0.5:
        errs.append(f"market return {row['ret']} beyond +-50%: units are not fractions")
    for k in ("dispersion", "dollar_volume"):
        v = clean_number(row.get(k))
        if v is not None and v < 0:
            errs.append(f"{k} is negative")
    ev = clean_number(row.get("event_share"))
    if ev is not None and not 0 <= ev <= 1:
        errs.append("event_share outside [0, 1]")
    return errs


@dataclasses.dataclass(frozen=True)
class RegimeState:
    """The monitor's reading for one day. states: axis -> state; values: the indicator readings behind them (None = missing)."""
    date: str
    states: Mapping[str, str]
    values: Mapping[str, float | None]
    dwell: Mapping[str, int]                  # consecutive days the axis has been in its current state
    composite: str                            # engine.learning.situation.classify_regime label, or 'unknown'
    discovered: str = UNKNOWN_STATE

    def known(self) -> dict[str, str]:
        return {a: s for a, s in self.states.items() if s != UNKNOWN_STATE}

    def code(self, axis: str) -> int | None:
        s = self.states.get(axis, UNKNOWN_STATE)
        return None if s == UNKNOWN_STATE else STATE_CODE[s]


class IndicatorHistory:
    """Bounded past of the six indicator values plus the raw market returns. Classification always uses days strictly before
    today; push() happens after."""

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self.days: deque[tuple[str, dict[str, float | None]]] = deque(maxlen=cfg.history_days)
        self.rets: deque[float] = deque(maxlen=max(cfg.history_days, cfg.persistence_window + 5))

    def __len__(self) -> int:
        return len(self.days)

    def last_date(self) -> str | None:
        return self.days[-1][0] if self.days else None

    def series(self, name: str) -> np.ndarray:
        a = np.array([v.get(name) if v.get(name) is not None else np.nan for _, v in self.days], dtype="float64")
        return a[np.isfinite(a)]

    def push(self, date, values: Mapping[str, float | None], ret: float) -> None:
        d = as_date(date).isoformat()
        if self.days and d <= self.days[-1][0]:
            raise FirewallBreach(f"indicator history push {d} is not after {self.days[-1][0]}")
        self.days.append((d, dict(values)))
        self.rets.append(float(ret))


def compute_indicators(row: Mapping[str, Any], hist: IndicatorHistory) -> dict[str, float | None]:
    """The six indicators for today from today's row and PAST returns. vol: the VIX when given, else the trailing realised
    volatility of market returns (annualised) including today. persistence: variance ratio VR(q)-1 of the last
    persistence_window returns including today (>0 trending, <0 mean-reverting). trend200: the SPY/MA200 fraction, else
    the trailing cumulative return over ma_window days."""
    cfg = hist.cfg
    ret = float(row["ret"])
    rets = np.array(list(hist.rets) + [ret], dtype="float64")
    vix = clean_number(row.get("vix"))
    if vix is not None and vix > 0:
        vol = vix
    elif len(rets) >= cfg.vol_window:
        vol = float(rets[-cfg.vol_window:].std(ddof=1) * math.sqrt(252) * 100.0)
    else:
        vol = None
    ma = clean_number(row.get("spy_ma200"))
    if ma is None and len(rets) >= cfg.ma_window:
        ma = float(np.expm1(np.log1p(np.clip(rets[-cfg.ma_window:], -0.99, None)).sum()))
    pers = None
    if len(rets) >= cfg.persistence_window:
        vr, _ = variance_ratio(rets[-cfg.persistence_window:], cfg.persistence_q)
        pers = None if vr is None else vr - 1.0
    dv = clean_number(row.get("dollar_volume"))
    return {"trend200": ma, "vol": vol, "dispersion": clean_number(row.get("dispersion")),
            "log_dv": math.log(dv) if dv is not None and dv > 0 else None, "persistence": pers,
            "event_share": clean_number(row.get("event_share"))}


def classify_axis(spec: AxisSpec, value: float | None, past: np.ndarray, prev_state: str | None, cfg: RegimeConfig) -> str:
    """State of one axis given today's indicator and its past distribution. UNKNOWN when the value is missing or fewer than
    min_history past values exist. Hysteresis: when the value sits within `hysteresis` of the neutral band's width of a
    boundary AND the previous state is the neighbouring state, the previous state is kept."""
    if value is None or not math.isfinite(value) or len(past) < cfg.min_history:
        return UNKNOWN_STATE
    if spec.mode == "signed":
        s = robust_scale(past)
        if s <= 0:
            return UNKNOWN_STATE
        lo_b, hi_b = -cfg.band * s, cfg.band * s
    else:
        lo_b, hi_b = float(np.quantile(past, cfg.q_lo)), float(np.quantile(past, cfg.q_hi))
        if hi_b <= lo_b:
            return UNKNOWN_STATE
    margin = cfg.hysteresis * (hi_b - lo_b)
    if prev_state == spec.hi_state and value > hi_b - margin:
        return spec.hi_state
    if prev_state == spec.lo_state and value < lo_b + margin:
        return spec.lo_state
    if value > hi_b:
        return spec.hi_state
    if value < lo_b:
        return spec.lo_state
    return NEUTRAL


def composite_label(row: Mapping[str, Any], values: Mapping[str, float | None]) -> str:
    """The single-label regime the Situation representation already uses (bull_calm ... stress), so the two vocabularies agree."""
    ma = values.get("trend200")
    vix = clean_number(row.get("vix"))
    return classify_regime(ma, vix if vix is not None else values.get("vol"), clean_number(row.get("vix_term")))[0]


class RegimeMonitor:
    """Streaming regime monitor. process(date, row) classifies the day from the past, then learns from it."""

    def __init__(self, cfg: RegimeConfig | None = None, discovery: "DiscoveredRegimes | None" = None):
        self.cfg = cfg or RegimeConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("bad RegimeConfig: " + "; ".join(errs))
        self.history = IndicatorHistory(self.cfg)
        self.discovery = discovery if discovery is not None else DiscoveredRegimes(self.cfg)
        self._prev: dict[str, str] = {}
        self._dwell: dict[str, int] = {}
        self.states: list[RegimeState] = []
        self.transitions: dict[str, dict[tuple[str, str], int]] = {a.name: {} for a in AXES}
        self.rejected_days = 0

    def process(self, date, row: Mapping[str, Any], now=None) -> RegimeState:
        d = as_date(date)
        if now is not None and d > as_date(now):
            raise FirewallBreach(f"regime day {d} is after now={as_date(now)}")
        if self.history.last_date() is not None and d.isoformat() <= self.history.last_date():
            raise FirewallBreach(f"regime day {d} is not after the last processed day {self.history.last_date()}")
        errs = validate_market_row(row)
        if errs:
            self.rejected_days += 1
            raise ValueError("bad market row: " + "; ".join(errs))
        values = compute_indicators(row, self.history)
        states, dwell = {}, {}
        for spec in AXES:
            st = classify_axis(spec, values.get(spec.indicator), self.history.series(spec.indicator), self._prev.get(spec.name), self.cfg)
            prev = self._prev.get(spec.name)
            if prev is not None and prev != st:
                tr = self.transitions[spec.name]
                tr[(prev, st)] = tr.get((prev, st), 0) + 1
            self._dwell[spec.name] = self._dwell.get(spec.name, 0) + 1 if prev == st else 1
            states[spec.name], dwell[spec.name] = st, self._dwell[spec.name]
            self._prev[spec.name] = st
        vec = self.discovery.vector(row, values, self.history)
        disc = self.discovery.assign(vec)
        self.discovery.observe(d, vec, self.history)
        self.history.push(d, values, float(row["ret"]))
        rs = RegimeState(d.isoformat(), states, values, dwell, composite_label(row, values), disc)
        self.states.append(rs)
        return rs

    def frame(self) -> pd.DataFrame:
        """Date-indexed table of states (one column per axis + composite + discovered)."""
        if not self.states:
            return pd.DataFrame()
        rows = [dict(s.states, composite=s.composite, discovered=s.discovered, date=s.date) for s in self.states]
        return pd.DataFrame(rows).set_index("date")

    def occupancy(self) -> pd.DataFrame:
        """Share of days in each state per axis, and the mean dwell (run length) of each state."""
        f = self.frame()
        rows = []
        for spec in AXES:
            if f.empty:
                continue
            s = f[spec.name]
            runs = (s != s.shift()).cumsum()
            lengths = s.groupby(runs).agg(["first", "size"])
            for st in spec.states() + (UNKNOWN_STATE,):
                m = lengths[lengths["first"] == st]["size"]
                rows.append({"axis": spec.name, "state": st, "share": float((s == st).mean()),
                             "mean_dwell": float(m.mean()) if len(m) else None, "n_runs": int(len(m))})
        return pd.DataFrame(rows)

    def label_map(self, axis: str) -> dict[str, str]:
        """{date: state} for one axis (or 'composite' / 'discovered'), for ShareLedger.by_label and similar consumers."""
        f = self.frame()
        return {} if f.empty else {d: str(v) for d, v in f[axis].items() if v != UNKNOWN_STATE}

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "cfg": dataclasses.asdict(self.cfg), "days": [[d, v] for d, v in self.history.days],
                "rets": list(self.history.rets), "states": [dataclasses.asdict(s) for s in self.states], "discovery": self.discovery.state()}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "RegimeMonitor":
        if st.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"monitor schema {st.get('schema')!r} != {SCHEMA_VERSION}")
        cfg = RegimeConfig(**st["cfg"])
        mon = cls(cfg, DiscoveredRegimes.from_state(cfg, st["discovery"]))
        for d, v in st["days"]:
            mon.history.days.append((d, v))
        mon.history.rets.extend(st["rets"])
        mon.states = [RegimeState(**s) for s in st["states"]]
        if mon.states:
            last = mon.states[-1]
            mon._prev, mon._dwell = dict(last.states), dict(last.dwell)
        return mon

    def content_hash(self) -> str:
        return stable_hash(self.state(), 16)


# ------------------------------------------------------------------------------------------------ discovery

def run_lengths(labels: Sequence[Any]) -> list[int]:
    out, run = [], 0
    for i, x in enumerate(labels):
        run = run + 1 if i > 0 and x == labels[i - 1] else 1
        if i == len(labels) - 1 or labels[i + 1] != x:
            out.append(run)
    return out


def dwell_ratio(labels: Sequence[Any], seed: int, n_shuffles: int = 100) -> tuple[float | None, float | None]:
    """(observed mean run length / mean run length of the same labels shuffled in time, z of the observed against the shuffles).
    A regime persists: ratio >> 1. Labels that switch at random (noise clusters) give a ratio near 1. None below 20 labels or one label."""
    lab = list(labels)
    if len(lab) < 20 or len(set(lab)) < 2:
        return None, None
    obs = float(np.mean(run_lengths(lab)))
    rng = np.random.default_rng(seed)
    arr = np.array(lab, dtype=object)
    sims = np.array([np.mean(run_lengths(list(arr[rng.permutation(len(arr))]))) for _ in range(int(n_shuffles))])
    sd = float(sims.std(ddof=1))
    return obs / float(sims.mean()), (obs - float(sims.mean())) / sd if sd > 0 else None


@dataclasses.dataclass(frozen=True)
class DiscoveryFit:
    """One refit. status: ACCEPTED (persistent clusters), NO_STRUCTURE (silhouette too low for any k >= 2), REJECTED_NOISE (clusters
    exist but do not persist in time), TOO_LITTLE_DATA."""
    fitted_through: str
    k: int
    status: str
    silhouette: float | None
    dwell_ratio: float | None
    sizes: tuple[int, ...]
    ids: tuple[str, ...]


class DiscoveredRegimes:
    """Unsupervised regimes from the market state vector, past-only. observe() stores today's vector AFTER assign() used the
    centroids fitted before today, and refits every refit_every days on the stored past."""

    FEATURES = ("ret_mean20", "vol", "dispersion", "log_dv_chg", "event_share", "breadth", "persistence")

    def __init__(self, cfg: RegimeConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.vectors: deque[tuple[str, np.ndarray]] = deque(maxlen=cfg.fit_window)
        self.centroids: np.ndarray | None = None
        self.ids: tuple[str, ...] = ()
        self.center = np.zeros(len(self.FEATURES))
        self.scale = np.ones(len(self.FEATURES))
        self.fits: list[DiscoveryFit] = []
        self._next_id = 0
        self._since = 0

    def vector(self, row: Mapping[str, Any], values: Mapping[str, float | None], hist: IndicatorHistory) -> np.ndarray | None:
        rets = list(hist.rets)[-19:] + [float(row["ret"])]
        dvs = hist.series("log_dv")
        dv, vol = values.get("log_dv"), values.get("vol")
        parts = [float(np.mean(rets)) if len(rets) >= 5 else None, vol, values.get("dispersion"),
                 (dv - float(np.median(dvs[-60:]))) if dv is not None and len(dvs) >= 20 else None,
                 values.get("event_share"), clean_number(row.get("breadth")), values.get("persistence")]
        if any(p is None for p in parts):
            return None
        return np.array(parts, dtype="float64")

    def assign(self, vec: np.ndarray | None) -> str:
        if vec is None or self.centroids is None:
            return UNKNOWN_STATE
        z = (vec - self.center) / self.scale
        return self.ids[int(((self.centroids - z) ** 2).sum(axis=1).argmin())]

    def observe(self, date, vec: np.ndarray | None, hist: IndicatorHistory) -> None:
        if vec is None:
            return
        self.vectors.append((as_date(date).isoformat(), vec))
        self._since += 1
        if self._since >= self.cfg.refit_every and len(self.vectors) >= self.cfg.min_history:
            self._since = 0
            self.refit()

    def refit(self) -> DiscoveryFit:
        """Choose k by silhouette on standardised past vectors; accept only persistent, non-trivial clusters."""
        d0 = self.vectors[-1][0]
        X = np.array([v for _, v in self.vectors])
        if len(X) < self.cfg.min_history:
            fit = DiscoveryFit(d0, 0, "TOO_LITTLE_DATA", None, None, (), ())
            self.fits.append(fit)
            return fit
        center = np.median(X, axis=0)
        scale = np.array([robust_scale(X[:, j]) or float(X[:, j].std()) or 1.0 for j in range(X.shape[1])])
        Z = (X - center) / scale
        cands: list = []
        for k in range(2, self.cfg.k_max + 1):
            lab, cent, _ = kmeans(Z, k, self.seed, n_init=8)
            sizes = np.bincount(lab, minlength=k)
            if sizes.min() < self.cfg.min_cluster_share * len(Z):
                continue
            sil = silhouette(Z, lab, seed=self.seed)
            if sil is not None:
                cands.append((sil, k, lab, cent))
        # parsimony: the smallest k whose silhouette is within 10% of the best (extra clusters that add little are noise-splitting)
        best = None
        if cands:
            top = max(c[0] for c in cands)
            best = min((c for c in cands if c[0] >= 0.9 * top), key=lambda c: c[1])
        if best is None or best[0] < self.cfg.min_silhouette:
            fit = DiscoveryFit(d0, 1, "NO_STRUCTURE", None if best is None else best[0], None, (len(Z),), ())
            self.centroids, self.ids = None, ()
            self.fits.append(fit)
            return fit
        sil, k, lab, cent = best
        ratio, _z = dwell_ratio(list(lab), self.seed)
        if ratio is None or ratio < self.cfg.dwell_ratio_min:
            fit = DiscoveryFit(d0, k, "REJECTED_NOISE", sil, ratio, tuple(int(s) for s in np.bincount(lab, minlength=k)), ())
            self.centroids, self.ids = None, ()
            self.fits.append(fit)
            return fit
        ids = self._match(cent, center, scale)
        self.centroids, self.ids, self.center, self.scale = cent, ids, center, scale
        fit = DiscoveryFit(d0, k, "ACCEPTED", sil, ratio, tuple(int(s) for s in np.bincount(lab, minlength=k)), ids)
        self.fits.append(fit)
        return fit

    def _match(self, cent: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[str, ...]:
        """Keep an old regime's id when a new centroid (compared in raw feature space) is the nearest unclaimed one; new ones get new ids."""
        raw_new = cent * scale + center
        ids: list[str | None] = [None] * len(cent)
        if self.centroids is not None and self.ids:
            raw_old = self.centroids * self.scale + self.center
            pairs = sorted(((float((((raw_new[i] - raw_old[j]) / scale) ** 2).sum()), i, j) for i in range(len(raw_new)) for j in range(len(raw_old))))
            used_old = set()
            for dist, i, j in pairs:
                if ids[i] is None and j not in used_old and dist < 4.0 * len(scale):
                    ids[i] = self.ids[j]
                    used_old.add(j)
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = f"disc_{self._next_id}"
                self._next_id += 1
        return tuple(ids)

    def status(self) -> str:
        return self.fits[-1].status if self.fits else "NOT_FITTED"

    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "vectors": [[d, v.tolist()] for d, v in self.vectors], "next_id": self._next_id,
                "centroids": None if self.centroids is None else self.centroids.tolist(), "ids": list(self.ids),
                "center": self.center.tolist(), "scale": self.scale.tolist(), "since": self._since,
                "fits": [dataclasses.asdict(f) for f in self.fits]}

    @classmethod
    def from_state(cls, cfg: RegimeConfig, st: Mapping[str, Any]) -> "DiscoveredRegimes":
        d = cls(cfg, st["seed"])
        d.vectors.extend((dd, np.array(v)) for dd, v in st["vectors"])
        d._next_id, d._since = st["next_id"], st["since"]
        d.centroids = None if st["centroids"] is None else np.array(st["centroids"])
        d.ids, d.center, d.scale = tuple(st["ids"]), np.array(st["center"]), np.array(st["scale"])
        d.fits = [DiscoveryFit(**{**f, "sizes": tuple(f["sizes"]), "ids": tuple(f["ids"])}) for f in st["fits"]]
        return d


# ------------------------------------------------------------------------------------------------ pattern x regime

class Verdict(_StrEnum):
    UNIVERSAL = "UNIVERSAL"
    REGIME_BOUND = "REGIME_BOUND"
    REGIME_REVERSING = "REGIME_REVERSING"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclasses.dataclass(frozen=True)
class StateEffect:
    axis: str
    state: str
    n_days: int
    effect: float | None
    se: float | None
    t: float | None

    def established(self, t_bar: float = T_BAR) -> bool:
        return self.t is not None and abs(self.t) >= t_bar

    @property
    def sign(self) -> int:
        return 0 if not self.effect else (1 if self.effect > 0 else -1)


@dataclasses.dataclass(frozen=True)
class Contrast:
    axis: str
    a: str
    b: str
    diff: float | None
    t: float | None
    q_value: float | None


@dataclasses.dataclass(frozen=True)
class PatternRegimeReport:
    pattern_id: str
    verdict: Verdict
    overall: StateEffect
    by_state: tuple[StateEffect, ...]
    contrasts: tuple[Contrast, ...]
    bound_axes: tuple[str, ...]
    good_states: Mapping[str, tuple[str, ...]]        # axis -> states where the effect is established with the overall sign
    bad_states: Mapping[str, tuple[str, ...]]         # axis -> states where it is absent or opposite (measured, enough data)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(e) for e in self.by_state])


class PatternRegimeBook:
    """Daily pattern effect series with the regime state at each decision date. Only matured, identity-free numbers are kept."""

    def __init__(self, sessions: int = 5, min_days: int = 30):
        self.sessions = int(sessions)
        self.min_days = int(min_days)
        self._rows: dict[str, list[tuple]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._rows.values())

    def patterns(self) -> tuple[str, ...]:
        return tuple(sorted(self._rows))

    def add(self, pattern_id: str, date, effect: float, matured_at, now, state: RegimeState) -> bool:
        """One decision day's pattern effect (flagged minus unflagged forward return). Refused unless the label matured strictly
        before `now` and after the decision date; a non-finite effect is skipped (False), never stored as 0."""
        require_past(matured_at, now, f"pattern effect {pattern_id}")
        d = as_date(date)
        if as_date(matured_at) <= d:
            raise FirewallBreach(f"effect matures {matured_at} on/before its decision date {d}")
        if effect is None or not math.isfinite(effect):
            return False
        rows = self._rows.setdefault(pattern_id, [])
        if rows and d.isoformat() <= rows[-1][0]:
            raise FirewallBreach(f"{pattern_id}: {d} is not after {rows[-1][0]}")
        rows.append((d.isoformat(), float(effect), as_date(matured_at).isoformat(), d.year, dict(state.states), state.discovered))
        return True

    def frame(self, pattern_id: str, now, replay_years: Iterable[int] = ()) -> pd.DataFrame:
        ys = {int(y) for y in replay_years}
        rows = [r for r in self._rows.get(pattern_id, []) if as_date(r[2]) < as_date(now) and r[3] not in ys]
        if not rows:
            return pd.DataFrame(columns=["date", "effect"])
        df = pd.DataFrame({"date": [r[0] for r in rows], "effect": [r[1] for r in rows], "disc": [r[5] for r in rows]})
        for spec in AXES:
            df[spec.name] = [r[4].get(spec.name, UNKNOWN_STATE) for r in rows]
        df["discovered"] = df.pop("disc")
        return df

    def _effect(self, df: pd.DataFrame, axis: str, state: str, all_days: pd.Index) -> StateEffect:
        sel = df[df[axis] == state] if axis != "all" else df
        if len(sel) < self.min_days:
            return StateEffect(axis, state, int(len(sel)), float(sel["effect"].mean()) if len(sel) else None, None, None)
        s = sel.set_index("date")["effect"]
        se = hac_se(s.reindex(all_days).to_numpy(), self.sessions - 1)
        m = float(s.mean())
        return StateEffect(axis, state, int(len(s)), m, se, m / se if se and math.isfinite(se) and se > 1e-15 else None)

    def report(self, pattern_id: str, now, replay_years: Iterable[int] = (), t_bar: float = T_BAR, q_bar: float = 0.1) -> PatternRegimeReport:
        df = self.frame(pattern_id, now, replay_years)
        all_days = pd.Index(sorted(df["date"].unique())) if len(df) else pd.Index([])
        overall = self._effect(df, "all", "all", all_days) if len(df) else StateEffect("all", "all", 0, None, None, None)
        if overall.t is None:
            return PatternRegimeReport(pattern_id, Verdict.INSUFFICIENT_DATA, overall, (), (), (), {}, {})
        axes = [a.name for a in AXES] + ["discovered"]
        by_state: list[StateEffect] = []
        contrasts: list[Contrast] = []
        for axis in axes:
            if axis not in df.columns:
                continue
            for st in sorted(x for x in df[axis].unique() if x != UNKNOWN_STATE):
                by_state.append(self._effect(df, axis, st, all_days))
            meas = [e for e in by_state if e.axis == axis and e.t is not None]
            for i in range(len(meas)):
                for j in range(i + 1, len(meas)):
                    a, b = meas[i], meas[j]
                    se = math.sqrt(a.se ** 2 + b.se ** 2)
                    diff = a.effect - b.effect
                    contrasts.append(Contrast(axis, a.state, b.state, diff, diff / se if se > 1e-15 else None, None))
        q = benjamini_hochberg([t_to_p(c.t) for c in contrasts])
        contrasts = [dataclasses.replace(c, q_value=qq) for c, qq in zip(contrasts, q)]
        sig = [c for c in contrasts if c.t is not None and abs(c.t) >= t_bar and c.q_value is not None and c.q_value <= q_bar]
        bound = tuple(sorted({c.axis for c in sig}))
        good: dict[str, tuple[str, ...]] = {}
        bad: dict[str, tuple[str, ...]] = {}
        osign = overall.sign
        for axis in axes:
            ms = [e for e in by_state if e.axis == axis and e.t is not None]
            g = tuple(e.state for e in ms if e.established(t_bar) and e.sign == osign)
            b = tuple(e.state for e in ms if not e.established(t_bar) or e.sign != osign)
            if g:
                good[axis] = g
            if b:
                bad[axis] = b
        opposite = any(e.established(t_bar) and e.sign == -osign for e in by_state if e.t is not None) and osign != 0
        n_meas_axes = len({e.axis for e in by_state if e.t is not None})
        if opposite and bound:
            verdict = Verdict.REGIME_REVERSING
        elif bound:
            verdict = Verdict.REGIME_BOUND
        elif not overall.established(t_bar):
            verdict = Verdict.NOT_ESTABLISHED             # a lone significant state with no significant contrast is not regime evidence
        elif n_meas_axes >= 2 and overall.established(t_bar):
            verdict = Verdict.UNIVERSAL
        else:
            verdict = Verdict.INSUFFICIENT_DATA         # established once, but too few regimes measured to call it universal
        return PatternRegimeReport(pattern_id, verdict, overall, tuple(by_state), tuple(contrasts), bound, good, bad)

    def regime_gate(self, pattern_id: str, state: RegimeState, now, replay_years: Iterable[int] = ()) -> dict[str, Any]:
        """Should the pattern be used in TODAY's regime? allowed True / False / None (None = abstain: unmeasured or unknown).
        UNIVERSAL -> True. REGIME_BOUND / REVERSING -> True only if every bound axis is currently in a state where the effect
        holds, False if in a state where it does not, None if that axis's state is unknown. Everything else -> None."""
        rep = self.report(pattern_id, now, replay_years)
        if rep.verdict == Verdict.UNIVERSAL:
            return {"allowed": True, "verdict": rep.verdict.value, "reason": "established in every measured regime"}
        if rep.verdict in (Verdict.REGIME_BOUND, Verdict.REGIME_REVERSING):
            for axis in rep.bound_axes:
                cur = state.discovered if axis == "discovered" else state.states.get(axis, UNKNOWN_STATE)
                if cur == UNKNOWN_STATE:
                    return {"allowed": None, "verdict": rep.verdict.value, "reason": f"{axis} state unknown"}
                if cur in rep.bad_states.get(axis, ()):
                    return {"allowed": False, "verdict": rep.verdict.value, "reason": f"effect absent in {axis}={cur}"}
                if cur not in rep.good_states.get(axis, ()):
                    return {"allowed": None, "verdict": rep.verdict.value, "reason": f"{axis}={cur} not measured"}
            return {"allowed": True, "verdict": rep.verdict.value, "reason": "current regime is one where the effect holds"}
        return {"allowed": None, "verdict": rep.verdict.value, "reason": "no established effect"}

    def matured_records(self, now, replay_years: Iterable[int] = ()) -> list[MaturedRecord]:
        recs = []
        for pid in self.patterns():
            rep = self.report(pid, now, replay_years)
            df = self.frame(pid, now, replay_years)
            if rep.verdict in (Verdict.INSUFFICIENT_DATA,) or df.empty:
                continue
            through = max(r[2] for r in self._rows[pid] if as_date(r[2]) < as_date(now))
            prov = Provenance(created_real=str(as_date(now)), learned_at=through, code_hash=current_code_hash(), outcomes_seen_through=through)
            payload = {"pattern": pid, "verdict": rep.verdict.value, "bound_axes": list(rep.bound_axes),
                       "good": {k: list(v) for k, v in rep.good_states.items()}, "bad": {k: list(v) for k, v in rep.bad_states.items()},
                       "effect": clean_number(rep.overall.effect), "t": clean_number(rep.overall.t)}
            recs.append(MaturedRecord("RG" + stable_hash([pid, rep.verdict.value, through], 10), through, payload, prov, Namespace.MATURED_RESEARCH))
        return recs

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "sessions": self.sessions, "min_days": self.min_days,
                "rows": {p: [list(r) for r in v] for p, v in self._rows.items()}}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "PatternRegimeBook":
        if st.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"book schema {st.get('schema')!r} != {SCHEMA_VERSION}")
        b = cls(st["sessions"], st["min_days"])
        b._rows = {p: [tuple(r) for r in v] for p, v in st["rows"].items()}
        return b


# ------------------------------------------------------------------------------------------------ changes, trader view, step

@dataclasses.dataclass(frozen=True)
class RegimeChange:
    axis: str
    before: str
    after: str
    persisted_days: int


def detect_changes(states: Sequence[RegimeState], min_persist: int = 3) -> list[RegimeChange]:
    """Regime changes that lasted at least min_persist days (a one-day flicker is not a change). Only days already in `states`
    (the past) are read, so the newest change is reported once it has persisted."""
    out = []
    for spec in AXES:
        seq = [s.states.get(spec.name, UNKNOWN_STATE) for s in states]
        i = 0
        runs = []
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1] == seq[i]:
                j += 1
            runs.append((seq[i], j - i + 1))
            i = j + 1
        for (a, la), (b, lb) in zip(runs, runs[1:]):
            if UNKNOWN_STATE not in (a, b) and lb >= min_persist and la >= min_persist:
                out.append(RegimeChange(spec.name, a, b, lb))
    return out


def regime_questions(changes: Sequence[RegimeChange], created_real: str, evidence_through: str, limit: int = 10) -> list[ResearchQuestion]:
    """A regime change is a research target (section 40): which patterns changed behaviour across it? Identity-free text."""
    return [ResearchQuestion.make(f"Which patterns changed behaviour when {c.axis} moved from {c.before} to {c.after}?", "regime",
                                  Problem.VOLATILITY, created_real, evidence_through,
                                  "a pattern verdict differs between the two states with q <= 0.1",
                                  "no pattern shows a significant state difference") for c in changes[:limit]]


def trader_regime_row(state: RegimeState) -> dict[str, float]:
    """What the blind trader may see of the regime: an ordinal code per axis (-1, 0, +1), unknown axes omitted. No dates, no names."""
    row = {f"rg_{a}": float(c) for a in AXIS_BY_NAME if (c := state.code(a)) is not None}
    assert_trader_safe(row, "regime trader row")
    return row


@dataclasses.dataclass(frozen=True)
class RegimeStepResult:
    date: str
    states: Mapping[str, str]
    composite: str
    discovered: str
    discovery_status: str
    n_added: int
    verdicts: Mapping[str, str]
    gates: Mapping[str, Any]
    changes: tuple[RegimeChange, ...]


def step(monitor: RegimeMonitor, book: PatternRegimeBook, now, market_row: Mapping[str, Any],
         pattern_effects: Iterable[tuple[str, Any, float, Any]] = (), replay_years: Iterable[int] = (),
         gate_patterns: Iterable[str] = ()) -> RegimeStepResult:
    """ONE research-loop step at `now`: file each matured pattern effect under the regime that held on ITS decision date
    (pattern_effects = [(pattern_id, decision_date, effect, matured_at)]), classify today's regime from the past, refresh verdicts
    and gates for gate_patterns, and report persisted regime changes. Nothing dated at/after `now` is read."""
    by_date = {s.date: s for s in monitor.states}
    added = 0
    for pid, ddate, eff, matured in pattern_effects:
        st = by_date.get(as_date(ddate).isoformat())
        if st is not None and book.add(pid, ddate, eff, matured, now, st):
            added += 1
    today = monitor.process(now, market_row, now)
    verdicts = {p: book.report(p, now, replay_years).verdict.value for p in book.patterns()}
    gates = {p: book.regime_gate(p, today, now, replay_years) for p in gate_patterns}
    return RegimeStepResult(today.date, today.states, today.composite, today.discovered, monitor.discovery.status(), added,
                            verdicts, gates, tuple(detect_changes(monitor.states)))


def render_report(monitor: RegimeMonitor, book: PatternRegimeBook, now, replay_years: Iterable[int] = ()) -> str:
    """Plain-text report: state occupancy and dwell per axis, the discovery status, and each pattern's verdict with its bound axes."""
    lines = [f"REGIMES ({SCHEMA_VERSION})  days {len(monitor.states)}  discovery {monitor.discovery.status()}"]
    occ = monitor.occupancy()
    if not occ.empty:
        lines.extend("  " + ln for ln in occ.round(3).to_string(index=False).splitlines())
    for pid in book.patterns():
        rep = book.report(pid, now, replay_years)
        lines.append(f"- {pid}: {rep.verdict.value}  overall {rep.overall.effect if rep.overall.effect is None else round(rep.overall.effect, 6)}"
                     f"  bound {list(rep.bound_axes)}  good {dict(rep.good_states)}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ regime dynamics

def transition_matrix(monitor: RegimeMonitor, axis: str) -> pd.DataFrame:
    """Row-normalised day-to-day transition probabilities between the KNOWN states of one axis, from the monitor's own past.
    Rows with no observations are omitted (not filled with a uniform guess)."""
    seq = [s.states.get(axis, UNKNOWN_STATE) for s in monitor.states]
    counts: dict[tuple[str, str], int] = {}
    for a, b in zip(seq, seq[1:]):
        if UNKNOWN_STATE not in (a, b):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        return pd.DataFrame()
    m = pd.Series(counts).unstack(fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


def expected_dwell(monitor: RegimeMonitor, axis: str) -> dict[str, float]:
    """Mean run length in days of each known state that has occurred."""
    seq = [s.states.get(axis, UNKNOWN_STATE) for s in monitor.states]
    runs: dict[str, list[int]] = {}
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        if seq[i] != UNKNOWN_STATE:
            runs.setdefault(seq[i], []).append(j - i + 1)
        i = j + 1
    return {st: float(np.mean(v)) for st, v in runs.items()}


def stay_probability(monitor: RegimeMonitor, axis: str, state: str) -> float | None:
    tm = transition_matrix(monitor, axis)
    if tm.empty or state not in tm.index or state not in tm.columns:
        return None
    return float(tm.loc[state, state])


def state_return_profile(monitor: RegimeMonitor, returns: pd.Series, axis: str, now) -> pd.DataFrame:
    """Market-return behaviour by state, next-day: for days classified into a state, the FOLLOWING day's return (so the profile
    is what the state predicted, not what defined it). Rows: state; columns: n, mean, std, worst, best, hit rate. Only days whose
    following return is strictly before `now` are used."""
    f = monitor.frame()
    if f.empty or axis not in f.columns:
        return pd.DataFrame()
    r = returns.copy()
    r.index = pd.DatetimeIndex(r.index).strftime("%Y-%m-%d")
    nxt = r.shift(-1)
    end = pd.Series(list(r.index[1:]) + [None], index=r.index)
    df = pd.DataFrame({"state": f[axis], "nxt": nxt.reindex(f.index), "end": end.reindex(f.index)}).dropna()
    df = df[df["state"] != UNKNOWN_STATE]
    df = df[df["end"].map(lambda d: as_date(d) < as_date(now))]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("state")["nxt"]
    out = g.agg(["count", "mean", "std", "min", "max"])
    out["hit_rate"] = g.apply(lambda x: float((x > 0).mean()))
    return out.rename(columns={"count": "n", "min": "worst", "max": "best"})


def cusum_break(values: Sequence[float], threshold: float = 5.0, drift: float = 0.5) -> list[int]:
    """Two-sided CUSUM change points on robust-standardised values (scale from the first half only, so a late shift cannot
    inflate its own yardstick). Returns positions where the cumulative deviation first exceeded `threshold` sigmas after allowing
    `drift`; the statistic resets after each detection. Empty for fewer than 20 values or zero spread."""
    a = np.asarray(values, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return []
    ref = a[: len(a) // 2]
    s = robust_scale(ref)
    if s <= 0:
        return []
    z = (a - np.median(ref)) / s
    hi = lo = 0.0
    out = []
    for i, x in enumerate(z):
        hi, lo = max(0.0, hi + x - drift), min(0.0, lo + x + drift)
        if hi > threshold or lo < -threshold:
            out.append(i)
            hi = lo = 0.0
    return out


def indicator_breaks(monitor: RegimeMonitor) -> dict[str, list[str]]:
    """Dates where each indicator's level shifted (CUSUM on the stored past), independent of the named-state thresholds: the
    thresholds might be wrong, a level shift is not. Trusted-side output (dates); never handed to the trader."""
    out = {}
    for name in INDICATORS:
        ok = [(d, v.get(name)) for d, v in monitor.history.days if v.get(name) is not None and math.isfinite(v.get(name))]
        out[name] = [ok[i][0] for i in cusum_break([x for _, x in ok])]
    return out


# ------------------------------------------------------------------------------------------------ are the axes honest?

def threshold_sensitivity(monitor: RegimeMonitor, q_grid: Sequence[tuple[float, float]] = ((0.25, 0.75), (0.33, 0.67), (0.4, 0.6))) -> pd.DataFrame:
    """How much do the named states depend on the (arbitrary) quantile cut-offs? For each quantile-mode axis, the share of days
    whose state differs between the first and last cut-off pair. A high share means the axis label is mostly a definition, not a
    discovery. Replays the stored indicator history past-only per day, without hysteresis, to isolate the cut-off."""
    rows = []
    for spec in AXES:
        if spec.mode != "quantile":
            continue
        hist = [v.get(spec.indicator) for _, v in monitor.history.days]
        per_q = {}
        for q in (q_grid[0], q_grid[-1]):
            cfg = dataclasses.replace(monitor.cfg, q_lo=q[0], q_hi=q[1], hysteresis=0.0)
            past: list[float] = []
            states = []
            for x in hist:
                states.append(classify_axis(spec, x, np.array(past), None, cfg))
                if x is not None and math.isfinite(x):
                    past.append(x)
            per_q[q] = states
        known = [(a, b) for a, b in zip(per_q[q_grid[0]], per_q[q_grid[-1]]) if UNKNOWN_STATE not in (a, b)]
        rows.append({"axis": spec.name, "days": len(known), "changed_share": float(np.mean([a != b for a, b in known])) if known else None})
    return pd.DataFrame(rows)


def axis_association(monitor: RegimeMonitor) -> pd.DataFrame:
    """Spearman correlation between the ordinal state codes of every pair of axes (known days only). Two axes that always agree
    are one regime, not two; the table lets the research loop stop counting them twice."""
    f = monitor.frame()
    if f.empty:
        return pd.DataFrame()
    codes = pd.DataFrame({a.name: f[a.name].map(lambda s: np.nan if s == UNKNOWN_STATE else STATE_CODE[s]) for a in AXES})
    return codes.corr(method="spearman", min_periods=30)


def discovered_vs_named(monitor: RegimeMonitor) -> dict[str, float | None]:
    """Cramer's V between the discovered regime label and each named axis (days where both are known). High V for an axis means
    the unsupervised clusters mostly rediscovered that axis; low V everywhere means they found something the named ones miss."""
    f = monitor.frame()
    out: dict[str, float | None] = {}
    if f.empty:
        return out
    for spec in AXES:
        sub = f[(f["discovered"] != UNKNOWN_STATE) & (f[spec.name] != UNKNOWN_STATE)]
        if len(sub) < 30 or sub["discovered"].nunique() < 2 or sub[spec.name].nunique() < 2:
            out[spec.name] = None
            continue
        tab = pd.crosstab(sub["discovered"], sub[spec.name]).to_numpy(dtype=float)
        exp = tab.sum(axis=1)[:, None] * tab.sum(axis=0)[None, :] / tab.sum()
        chi2 = float(((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
        out[spec.name] = math.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    return out


# ------------------------------------------------------------------------------------------------ transfer to unseen regimes

def unseen_regime_transfer(book: PatternRegimeBook, pattern_id: str, axis: str, now, replay_years: Iterable[int] = (),
                           t_bar: float = T_BAR) -> list[dict[str, Any]]:
    """Leave-one-regime-out (contract section 48: unseen regimes). For each known state s of `axis`: measure the pattern in all the
    OTHER states (train) and in s (held out). It transfers if both are established with the same sign. A pattern that only ever
    works in the states it was found in fails here, whatever its full-sample t says."""
    df = book.frame(pattern_id, now, replay_years)
    if df.empty or axis not in df.columns:
        return []
    days = pd.Index(sorted(df["date"].unique()))
    out = []
    for st in sorted(x for x in df[axis].unique() if x != UNKNOWN_STATE):
        train = df[(df[axis] != st) & (df[axis] != UNKNOWN_STATE)].assign(_all="all")
        test = df[df[axis] == st].assign(_all="all")
        a = book._effect(train, "_all", "all", days)
        b = book._effect(test, "_all", "all", days)
        ok = a.established(t_bar) and b.established(t_bar) and a.sign == b.sign
        out.append({"held_out": st, "train_effect": a.effect, "train_t": a.t, "test_effect": b.effect, "test_t": b.t,
                    "n_test": b.n_days, "transfers": bool(ok),
                    "verdict": "TRANSFERS" if ok else "UNTESTED" if a.t is None or b.t is None else "FAILS"})
    return out


def coverage_gaps(book: PatternRegimeBook, now, replay_years: Iterable[int] = (), min_days: int | None = None) -> list[tuple[str, str, str, int]]:
    """(pattern, axis, state, days) for every pole state with fewer than min_days of evidence for a pattern: where the next
    experiment buys the most (a verdict is only as wide as the regimes it has been seen in), least-covered first."""
    need = min_days if min_days is not None else book.min_days
    gaps = []
    for pid in book.patterns():
        df = book.frame(pid, now, replay_years)
        if df.empty:
            continue
        for spec in AXES:
            counts = df[spec.name].value_counts()
            for st in (spec.hi_state, spec.lo_state):
                n = int(counts.get(st, 0))
                if n < need:
                    gaps.append((pid, spec.name, st, n))
    return sorted(gaps, key=lambda g: (g[3], g[0], g[1], g[2]))


def coverage_questions(gaps: Sequence[tuple[str, str, str, int]], created_real: str, evidence_through: str, limit: int = 10) -> list[ResearchQuestion]:
    return [ResearchQuestion.make(f"Does pattern {pid} hold when {axis} is {st}? Only {n} days observed.", "regime", Problem.VOLATILITY,
                                  created_real, evidence_through, "effect measured with >= min_days in the state and a verdict either way",
                                  "still fewer than min_days after further data") for pid, axis, st, n in gaps[:limit]]


def regime_weights(book: PatternRegimeBook, state: RegimeState, now, patterns: Iterable[str] | None = None,
                   replay_years: Iterable[int] = ()) -> dict[str, float | None]:
    """Weight per pattern for today's regime, for the curator (never the trader directly): 1.0 allowed, 0.0 disallowed,
    None abstain (the consumer must treat None as unknown, not as 0)."""
    out: dict[str, float | None] = {}
    for pid in (patterns if patterns is not None else book.patterns()):
        g = book.regime_gate(pid, state, now, replay_years)["allowed"]
        out[pid] = None if g is None else (1.0 if g else 0.0)
    return out


def regime_share_labels(monitor: RegimeMonitor, axis: str = "composite") -> dict[str, str]:
    """{date: label} for engine.research.cross_section.ShareLedger.by_label: does the split of moves between sector, style and
    stock-specific change with the regime?"""
    return monitor.label_map(axis)


def scope_effect_by_regime(outcomes, monitor: RegimeMonitor, axis: str, now, replay_years: Iterable[int] = (), min_days: int = 20) -> pd.DataFrame:
    """Cross-sectional scope outcomes (engine.research.cross_section.ScopeOutcomes, duck-typed on .frame(now, replay_years)) split
    by regime state: mean next-h continuation (fwd x move sign) and day count per scope and state. States with fewer than
    min_days days report NaN."""
    df = outcomes.frame(now, replay_years)
    if df.empty:
        return pd.DataFrame()
    df = df.assign(state=df["date"].map(monitor.label_map(axis))).dropna(subset=["state"])
    if df.empty:
        return pd.DataFrame()
    df["cont"] = df["fwd"] * df["sign"]
    per_day = df.groupby(["scope", "state", "date"])["cont"].mean().reset_index()
    g = per_day.groupby(["scope", "state"])["cont"].agg(["mean", "count"])
    g.loc[g["count"] < min_days, "mean"] = np.nan
    return g.rename(columns={"count": "days"})


# ------------------------------------------------------------------------------------------------ mover rates and shrinkage by regime

def mover_rate_by_regime(monitor: RegimeMonitor, mover_counts: Mapping[str, tuple[int, int]], axis: str, now) -> pd.DataFrame:
    """Canon C67 link: how many 5-10% (and >10%) movers a day, by regime state. mover_counts = {date: (n_5_10, n_over_10)} per
    processed day (each day's count is known at its close), n_names = the day's universe size is NOT needed because the
    counts are compared as shares only via `n_days`. Rows: state; columns: days, mean count per band and the ratio to the
    all-days mean (>1: this regime produces more movers). Days on/after `now` are ignored."""
    lab = monitor.label_map(axis)
    rows = [(lab[d], c[0], c[1]) for d, c in mover_counts.items() if d in lab and as_date(d) < as_date(now)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["state", "movers_5_10", "movers_over_10"])
    g = df.groupby("state").agg(days=("state", "size"), movers_5_10=("movers_5_10", "mean"), movers_over_10=("movers_over_10", "mean"))
    g["ratio_5_10"] = g["movers_5_10"] / df["movers_5_10"].mean() if df["movers_5_10"].mean() > 0 else np.nan
    g["ratio_over_10"] = g["movers_over_10"] / df["movers_over_10"].mean() if df["movers_over_10"].mean() > 0 else np.nan
    return g


def shrunk_state_effects(report: PatternRegimeReport, axis: str, strength: float = 4.0) -> dict[str, float | None]:
    """Empirical-Bayes shrinkage of a pattern's per-state effects toward its overall effect: e_s' = w e_s + (1 - w) e_all with
    w = precision_s / (precision_s + strength x precision_all... implemented as w = 1 / (1 + strength x se_s^2 / se_all^2).
    A state with few days (large se) is pulled hard toward the overall effect, so a thin regime cannot claim a large effect. States
    that could not be measured are omitted."""
    if report.overall.effect is None or report.overall.se is None or report.overall.se <= 0:
        return {}
    out = {}
    for e in report.by_state:
        if e.axis != axis or e.effect is None or e.se is None or e.se <= 0:
            continue
        w = 1.0 / (1.0 + strength * (e.se ** 2) / (report.overall.se ** 2))
        out[e.state] = w * e.effect + (1.0 - w) * report.overall.effect
    return out


def best_axis(book: PatternRegimeBook, pattern_id: str, now, replay_years: Iterable[int] = ()) -> tuple[str, float] | None:
    """The axis along which the pattern's effect differs most (largest |t| among BH-surviving contrasts), or None: which regime
    dimension the pattern is actually conditional on. Ties break alphabetically."""
    rep = book.report(pattern_id, now, replay_years)
    ok = [c for c in rep.contrasts if c.t is not None and c.q_value is not None and c.q_value <= 0.1]
    if not ok:
        return None
    top = max(ok, key=lambda c: (abs(c.t), c.axis))
    return top.axis, abs(top.t)


class RegimeForecast:
    """One-step-ahead state probabilities from the monitor's own transition counts (Laplace-smoothed toward the marginal so
    unseen transitions are neither 0 nor 1). Scored with a running Brier skill against 'tomorrow = today', so a forecast that
    does not beat persistence is visible as such."""

    def __init__(self, monitor: RegimeMonitor, axis: str, smoothing: float = 1.0):
        self.monitor, self.axis, self.smoothing = monitor, axis, smoothing

    def next_probs(self) -> dict[str, float] | None:
        tm = transition_matrix(self.monitor, self.axis)
        if tm.empty:
            return None
        cur = self.monitor.states[-1].states.get(self.axis, UNKNOWN_STATE)
        if cur == UNKNOWN_STATE or cur not in tm.index:
            return None
        seq = [s.states.get(self.axis, UNKNOWN_STATE) for s in self.monitor.states]
        counts = pd.Series([x for x in seq if x != UNKNOWN_STATE]).value_counts(normalize=True)
        n_from = sum(1 for a, b in zip(seq, seq[1:]) if a == cur and UNKNOWN_STATE not in (a, b))
        w = n_from / (n_from + self.smoothing)
        return {st: float(w * tm.loc[cur].get(st, 0.0) + (1 - w) * counts.get(st, 0.0)) for st in counts.index}

    def brier_skill(self, min_days: int = 60) -> float | None:
        """1 - Brier(model) / Brier(persistence) evaluated walk-forward over the stored states (each day forecast with counts
        from earlier days only). >0: beats 'same as today'. None with fewer than min_days scored transitions."""
        seq = [s.states.get(self.axis, UNKNOWN_STATE) for s in self.monitor.states]
        states = sorted({x for x in seq if x != UNKNOWN_STATE})
        if len(states) < 2:
            return None
        counts: dict[tuple[str, str], int] = {}
        b_model = b_pers = 0.0
        n = 0
        for a, b in zip(seq, seq[1:]):
            if UNKNOWN_STATE in (a, b):
                continue
            tot = sum(counts.get((a, s), 0) for s in states)
            if tot >= 10:
                p = {s: (counts.get((a, s), 0) + self.smoothing / len(states)) / (tot + self.smoothing) for s in states}
                b_model += sum((p[s] - (s == b)) ** 2 for s in states)
                b_pers += sum(((s == a) - (s == b)) ** 2 for s in states)
                n += 1
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return None if n < min_days or b_pers <= 0 else 1.0 - b_model / b_pers
