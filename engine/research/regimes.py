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
    clip_z: float = 6.0                       # standardised discovery features are clipped so one spike cannot own a cluster
    min_stability: float = 0.6                # adjusted Rand agreement of bootstrap refits with the full fit
    stability_boots: int = 6
    stability_tol: float = 0.05               # take the smallest k within this of the best stability
    max_match: float = 1.5                    # centroid distance (robust sigmas per feature) beyond which an old id is not inherited

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
        last = self.history.last_date()
        if last is not None and d.isoformat() <= last:
            raise FirewallBreach(f"regime day {d} is not after the last processed day {last}")
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


def cluster_stability(Z: np.ndarray, ref_labels: np.ndarray, k: int, seed: int, n_boot: int = 6, frac: float = 0.7) -> float:
    """Mean adjusted Rand index between the reference clustering and clusterings refit on random 70% subsamples (every point then
    assigned to its nearest subsample centroid). 1.0: the same partition however the days are sampled; near 0: the partition is
    an accident of the sample."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed + 7919)
    n = len(Z)
    scores = []
    for _ in range(int(n_boot)):
        keep = rng.choice(n, max(k * 5, int(frac * n)), replace=False)
        _lab, cent, _ = kmeans(Z[keep], k, int(rng.integers(1 << 30)), n_init=3)
        assigned = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        scores.append(adjusted_rand_score(ref_labels, assigned))
    return float(np.mean(scores)) if scores else 0.0


@dataclasses.dataclass(frozen=True)
class DiscoveryFit:
    """One refit. status: ACCEPTED (stable, persistent clusters), NO_STRUCTURE (silhouette too low for any k >= 2), UNSTABLE (clusters
    change with the sample), REJECTED_NOISE (clusters exist but do not persist in time), TOO_LITTLE_DATA."""
    fitted_through: str
    k: int
    status: str
    silhouette: float | None
    dwell_ratio: float | None
    sizes: tuple[int, ...]
    ids: tuple[str, ...]
    stability: float | None = None
    match_cost: float | None = None


class DiscoveredRegimes:
    """Unsupervised regimes from the market state vector, past-only. observe() stores today's vector AFTER assign() used the
    centroids fitted before today, and refits every refit_every days on the stored past."""

    # ret_mean20 and persistence are left out on purpose: drift is noise at this scale and the variance-ratio is a slow, heteroskedasticity-
    # sensitive statistic that (measured) pulled k-means onto partitions unrelated to the volatility / dispersion / event / breadth regimes.
    # Trend and mean-reversion have their own axes.
    FEATURES = ("vol", "dispersion", "log_dv_chg", "event_share", "breadth")

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
        dvs = hist.series("log_dv")
        dv = values.get("log_dv")
        parts = [values.get("vol"), values.get("dispersion"),
                 (dv - float(np.median(dvs[-60:]))) if dv is not None and len(dvs) >= 20 else None,
                 values.get("event_share"), clean_number(row.get("breadth"))]
        if any(p is None for p in parts):
            return None
        return np.array(parts, dtype="float64")

    def assign(self, vec: np.ndarray | None) -> str:
        if vec is None or self.centroids is None:
            return UNKNOWN_STATE
        z = np.clip((vec - self.center) / self.scale, -self.cfg.clip_z, self.cfg.clip_z)
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
        """Choose k by BOOTSTRAP STABILITY (not raw fit), gate on silhouette and persistence, then inherit ids by Hungarian
        matching of centroids to the previous fit. A cluster count that changes when 30% of the days are dropped is not a regime count."""
        d0 = self.vectors[-1][0]
        X = np.array([v for _, v in self.vectors])
        if len(X) < self.cfg.min_history:
            fit = DiscoveryFit(d0, 0, "TOO_LITTLE_DATA", None, None, (), ())
            self.fits.append(fit)
            return fit
        center = np.median(X, axis=0)
        # scale = std of the 1st-99th percentile winsorised column: a MAD collapses when one regime holds most days (the other regime is
        # then 15 'sigmas' away and clipping would flatten it), while winsorising still stops a single spike from owning the scale
        lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
        scale = np.array([float(np.clip(X[:, j], lo[j], hi[j]).std()) or 1.0 for j in range(X.shape[1])])
        Z = np.clip((X - center) / scale, -self.cfg.clip_z, self.cfg.clip_z)
        cands = []
        for k in range(2, self.cfg.k_max + 1):
            lab, cent, _ = kmeans(Z, k, self.seed, n_init=8)
            if np.bincount(lab, minlength=k).min() < self.cfg.min_cluster_share * len(Z):
                continue
            sil = silhouette(Z, lab, seed=self.seed)
            if sil is not None and sil >= self.cfg.min_silhouette:
                cands.append((cluster_stability(Z, lab, k, self.seed, self.cfg.stability_boots), sil, k, lab, cent))
        if not cands:
            self.centroids, self.ids = None, ()
            fit = DiscoveryFit(d0, 1, "NO_STRUCTURE", None, None, (len(Z),), ())
            self.fits.append(fit)
            return fit
        top = max(c[0] for c in cands)
        if top < self.cfg.min_stability:
            self.centroids, self.ids = None, ()
            fit = DiscoveryFit(d0, 1, "UNSTABLE", max(c[1] for c in cands), None, (len(Z),), (), top)
            self.fits.append(fit)
            return fit
        stab, sil, k, lab, cent = min((c for c in cands if c[0] >= top - self.cfg.stability_tol), key=lambda c: c[2])
        ratio, _z = dwell_ratio(list(lab), self.seed)
        sizes = tuple(int(x) for x in np.bincount(lab, minlength=k))
        if ratio is None or ratio < self.cfg.dwell_ratio_min:
            self.centroids, self.ids = None, ()
            fit = DiscoveryFit(d0, k, "REJECTED_NOISE", sil, ratio, sizes, (), stab)
            self.fits.append(fit)
            return fit
        ids, cost = self._match(cent, center, scale)
        self.centroids, self.ids, self.center, self.scale = cent, ids, center, scale
        fit = DiscoveryFit(d0, k, "ACCEPTED", sil, ratio, sizes, ids, stab, cost)
        self.fits.append(fit)
        return fit

    def _match(self, cent: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[tuple[str, ...], float | None]:
        """Optimal (Hungarian) one-to-one matching of the new centroids to the previous fit's, compared in raw feature space in
        units of the new scale. A pair closer than 3/4 of the old fit's smallest inter-centroid distance (max_match when the old fit
        had one cluster) inherits the old id; the rest get new ids.
        Returns (ids, mean matched distance or None when nothing was matched)."""
        from scipy.optimize import linear_sum_assignment
        raw_new = cent * scale + center
        ids: list[str | None] = [None] * len(cent)
        costs = []
        if self.centroids is not None and self.ids:
            raw_old = self.centroids * self.scale + self.center
            dist = np.sqrt((((raw_new[:, None, :] - raw_old[None, :, :]) / scale) ** 2).mean(axis=2))
            old_d = np.sqrt((((raw_old[:, None, :] - raw_old[None, :, :]) / scale) ** 2).mean(axis=2))
            # a centroid may drift, but never farther than 3/4 of the way to its nearest sibling (else it could be the sibling)
            limit = 0.75 * float(old_d[old_d > 0].min()) if (old_d > 0).any() else self.cfg.max_match
            for i, j in zip(*linear_sum_assignment(dist)):
                if dist[i, j] <= limit:
                    ids[i] = self.ids[j]
                    costs.append(float(dist[i, j]))
        final: list[str] = []
        for known in ids:
            if known is None:
                known = f"disc_{self._next_id}"
                self._next_id += 1
            final.append(known)
        return tuple(final), (float(np.mean(costs)) if costs else None)

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

    def effect_se(self) -> tuple[float, float]:
        """(effect, se) of a measured state; a state with a t-statistic always has both."""
        if self.effect is None or self.se is None:
            raise ValueError(f"state {self.axis}={self.state} was not measured")
        return self.effect, self.se


@dataclasses.dataclass(frozen=True)
class Contrast:
    axis: str
    a: str
    b: str
    diff: float | None
    t: float | None
    q_value: float | None


@dataclasses.dataclass(frozen=True)
class PooledState:
    """One state's effect after shrinking toward the axis's pooled effect (random-effects, DerSimonian-Laird). weight is the
    share of the state's OWN estimate kept (1 = no shrinkage); se is the posterior standard error."""
    axis: str
    state: str
    raw_effect: float
    raw_se: float
    effect: float
    se: float
    weight: float

    @property
    def t(self) -> float | None:
        return self.effect / self.se if self.se > 1e-15 else None

    def established(self, t_bar: float = T_BAR) -> bool:
        return self.t is not None and abs(self.t) >= t_bar


def pool_states(effects: Sequence[StateEffect]) -> tuple[list[PooledState], float, float | None]:
    """Random-effects pooling of one axis's per-state effects. Returns (pooled states, tau2, pooled mean). tau2 is the
    method-of-moments between-state variance: 0 when the states agree within their errors (everything is pulled to the pooled
    mean), large when they truly differ (little shrinkage). A thin state (large se) always moves furthest. Fewer than two
    measured states: nothing to pool, ([], 0.0, None)."""
    ms = [(e, float(e.effect), float(e.se)) for e in effects if e.effect is not None and e.se is not None and e.se > 0]
    if len(ms) < 2:
        return [], 0.0, None
    w = np.array([1.0 / s ** 2 for _, _, s in ms])
    y = np.array([eff for _, eff, _ in ms])
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum())
    denom = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (len(ms) - 1)) / denom) if denom > 0 else 0.0
    var_mu = 1.0 / float(w.sum())
    out = []
    for e, eff, s in ms:
        b = tau2 / (tau2 + s ** 2)
        out.append(PooledState(e.axis, e.state, eff, s, mu + b * (eff - mu),
                               math.sqrt(b * s ** 2 + (1 - b) ** 2 * var_mu), b))
    return out, tau2, mu


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
    pooled: Mapping[str, tuple["PooledState", ...]] = dataclasses.field(default_factory=dict)   # empirical-Bayes view per axis
    tau2: Mapping[str, float] = dataclasses.field(default_factory=dict)       # between-state variance per axis

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
                    (ea, sa), (eb, sb) = a.effect_se(), b.effect_se()
                    se = math.sqrt(sa ** 2 + sb ** 2)
                    diff = ea - eb
                    contrasts.append(Contrast(axis, a.state, b.state, diff, diff / se if se > 1e-15 else None, None))
        q = benjamini_hochberg([t_to_p(c.t) for c in contrasts])
        contrasts = [dataclasses.replace(c, q_value=qq) for c, qq in zip(contrasts, q)]
        sig = [c for c in contrasts if c.t is not None and abs(c.t) >= t_bar and c.q_value is not None and c.q_value <= q_bar]
        bound = tuple(sorted({c.axis for c in sig}))
        good: dict[str, tuple[str, ...]] = {}
        bad: dict[str, tuple[str, ...]] = {}
        pooled: dict[str, tuple[PooledState, ...]] = {}
        tau2: dict[str, float] = {}
        osign = overall.sign
        for axis in axes:
            ms = [e for e in by_state if e.axis == axis and e.t is not None]
            ps, tau2[axis], _mu = pool_states([e for e in by_state if e.axis == axis])
            if ps:
                pooled[axis] = tuple(ps)
                # judged on the SHRUNK effects: a thin state cannot be 'good' on a lucky estimate or 'bad' on an unlucky one
                g = tuple(p.state for p in ps if p.established(t_bar) and (p.effect > 0) == (osign > 0))
                bd = tuple(p.state for p in ps if not (p.established(t_bar) and (p.effect > 0) == (osign > 0)))
            else:
                g = tuple(e.state for e in ms if e.established(t_bar) and e.sign == osign)
                bd = tuple(e.state for e in ms if not e.established(t_bar) or e.sign != osign)
            if g:
                good[axis] = g
            if bd:
                bad[axis] = bd
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
        return PatternRegimeReport(pattern_id, verdict, overall, tuple(by_state), tuple(contrasts), bound, good, bad, pooled, tau2)

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
    active: tuple[str, ...] = ()                       # the named regimes (of the twelve) active today
    warnings: tuple["EarlyWarning", ...] = ()          # axes under pressure toward a pole they are not yet in
    unhealthy: tuple[str, ...] = ()                    # axes failing axis_health(): distrust verdicts that lean on them
    discovery_stability: float | None = None


def step(monitor: RegimeMonitor, book: PatternRegimeBook, now, market_row: Mapping[str, Any],
         pattern_effects: Iterable[tuple[str, Any, float, Any]] = (), replay_years: Iterable[int] = (),
         gate_patterns: Iterable[str] = (), named: "NamedRegimeMonitor | None" = None) -> RegimeStepResult:
    """ONE research-loop step at `now`: file each matured pattern effect under the regime that held on ITS decision date
    (pattern_effects = [(pattern_id, decision_date, effect, matured_at)]), classify today's regime from the past, refresh verdicts
    and gates for gate_patterns, and report persisted regime changes, the active named regimes (via `named`, built on demand),
    live early warnings and unhealthy axes. Nothing dated at/after `now` is read."""
    by_date = {s.date: s for s in monitor.states}
    added = 0
    for pid, ddate, eff, matured in pattern_effects:
        st = by_date.get(as_date(ddate).isoformat())
        if st is not None and book.add(pid, ddate, eff, matured, now, st):
            added += 1
    today = monitor.process(now, market_row, now)
    verdicts = {p: book.report(p, now, replay_years).verdict.value for p in book.patterns()}
    gates = {p: book.regime_gate(p, today, now, replay_years) for p in gate_patterns}
    named = named or NamedRegimeMonitor(monitor)
    named.update()
    fits = [f for f in monitor.discovery.fits if f.status == "ACCEPTED"]
    return RegimeStepResult(today.date, today.states, today.composite, today.discovered, monitor.discovery.status(), added,
                            verdicts, gates, tuple(detect_changes(monitor.states)), tuple(named.active()), tuple(all_warnings(monitor)),
                            tuple(sorted(unhealthy_axes(monitor))), fits[-1].stability if fits else None)


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
    out: dict[str, list[str]] = {}
    for name in INDICATORS:
        ok = [(d, float(x)) for d, v in monitor.history.days if (x := v.get(name)) is not None and math.isfinite(x)]
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
    out: dict[str, float | None] = {}
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
    ok = [(c.axis, abs(c.t)) for c in rep.contrasts if c.t is not None and c.q_value is not None and c.q_value <= 0.1]
    if not ok:
        return None
    return max(ok, key=lambda at: (at[1], at[0]))


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


# ------------------------------------------------------------------------------------------------ every section-25 regime, explicitly

@dataclasses.dataclass(frozen=True)
class RegimeDef:
    """One of the twelve regimes the contract names, with its own definition. The definition is relative to the indicator's own PAST
    (a quantile band or a robust-sigma band), never a fixed level, so it moves when the market does."""
    name: str
    axis: str
    state: str
    indicator: str
    meaning: str


def _defs() -> tuple[RegimeDef, ...]:
    text = {
        "bull": "SPY/200-day-average above +band robust sigmas of its own past (market above its long trend)",
        "bear": "SPY/200-day-average below -band robust sigmas of its own past (market below its long trend)",
        "high_vol": "VIX (else trailing realised volatility) above its own past 67th percentile",
        "low_vol": "VIX (else trailing realised volatility) below its own past 33rd percentile",
        "high_dispersion": "cross-sectional return spread above its own past 67th percentile",
        "low_dispersion": "cross-sectional return spread below its own past 33rd percentile",
        "high_liquidity": "market dollar volume above its own past 67th percentile",
        "low_liquidity": "market dollar volume below its own past 33rd percentile",
        "trend": "variance ratio VR(5)-1 of recent market returns above +band robust sigmas of its own past (returns persist)",
        "mean_reversion": "variance ratio VR(5)-1 of recent market returns below -band robust sigmas of its own past (returns revert)",
        "event_heavy": "share of names with a live event above its own past 67th percentile",
        "event_light": "share of names with a live event below its own past 33rd percentile",
    }
    out = []
    for a in AXES:
        for st in (a.hi_state, a.lo_state):
            out.append(RegimeDef(st, a.name, st, a.indicator, text[st]))
    return tuple(out)


NAMED_REGIMES: tuple[RegimeDef, ...] = _defs()
NAMED_BY_NAME = {r.name: r for r in NAMED_REGIMES}
assert len(NAMED_REGIMES) == 12


def regime_flags(state: RegimeState) -> dict[str, bool | None]:
    """The twelve regimes as booleans for one day. None = that axis is UNKNOWN (never False: not-known is not not-in-it). NEUTRAL
    on an axis makes both of its regimes False."""
    out: dict[str, bool | None] = {}
    for r in NAMED_REGIMES:
        st = state.states.get(r.axis, UNKNOWN_STATE)
        out[r.name] = None if st == UNKNOWN_STATE else st == r.state
    return out


@dataclasses.dataclass(frozen=True)
class RegimeEpisode:
    name: str
    start: str
    end: str | None                            # None while the regime is still active
    days: int


class NamedRegimeMonitor:
    """Explicit monitor of every one of the twelve regimes on top of a RegimeMonitor: which are active now, when each began,
    every past episode, and how often and how long each occurs. update() is incremental and idempotent; it reads only states the
    monitor has already classified from the past."""

    def __init__(self, monitor: RegimeMonitor):
        self.monitor = monitor
        self._seen = 0
        self._open: dict[str, tuple[str, int]] = {}
        self._episodes: dict[str, list[RegimeEpisode]] = {r.name: [] for r in NAMED_REGIMES}
        self._active_days: dict[str, int] = {r.name: 0 for r in NAMED_REGIMES}
        self._known_days: dict[str, int] = {r.name: 0 for r in NAMED_REGIMES}
        self._last_end: dict[str, str] = {}

    def update(self) -> int:
        n = 0
        for st in self.monitor.states[self._seen:]:
            for name, flag in regime_flags(st).items():
                if flag is not None:
                    self._known_days[name] += 1
                if flag:
                    self._active_days[name] += 1
                    if name not in self._open:
                        self._open[name] = (st.date, 0)
                    self._open[name] = (self._open[name][0], self._open[name][1] + 1)
                elif name in self._open:
                    start, days = self._open.pop(name)
                    self._episodes[name].append(RegimeEpisode(name, start, st.date, days))
                    self._last_end[name] = st.date
            n += 1
        self._seen = len(self.monitor.states)
        return n

    def process(self, date, row: Mapping[str, Any], now=None) -> dict[str, bool | None]:
        st = self.monitor.process(date, row, now)
        self.update()
        return regime_flags(st)

    def active(self) -> list[str]:
        return sorted(self._open)

    def episodes(self, name: str, include_open: bool = True) -> list[RegimeEpisode]:
        eps = list(self._episodes[name])
        if include_open and name in self._open:
            eps.append(RegimeEpisode(name, self._open[name][0], None, self._open[name][1]))
        return eps

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in NAMED_REGIMES:
            eps = self.episodes(r.name)
            lens = [e.days for e in eps]
            known = self._known_days[r.name]
            rows.append({"regime": r.name, "axis": r.axis, "known_days": known, "active_days": self._active_days[r.name],
                         "share": self._active_days[r.name] / known if known else None, "episodes": len(eps),
                         "mean_len": float(np.mean(lens)) if lens else None, "longest": max(lens) if lens else 0,
                         "active_now": r.name in self._open})
        return pd.DataFrame(rows).set_index("regime")

    def exclusive_violations(self) -> list[tuple[str, str, str]]:
        """Days on which both regimes of an axis were active: (date, hi, lo). Must always be empty; a non-empty list is a bug."""
        out = []
        for st in self.monitor.states:
            fl = regime_flags(st)
            for a in AXES:
                if fl[a.hi_state] and fl[a.lo_state]:
                    out.append((st.date, a.hi_state, a.lo_state))
        return out

    def boundary(self, name: str) -> tuple[float, float] | None:
        """(low, high) cut-offs of the regime's axis as they stand now, computed from the monitor's own past. None until there is
        enough history. Answers 'what would it take to enter this regime today'."""
        r = NAMED_BY_NAME[name]
        spec = AXIS_BY_NAME[r.axis]
        past = self.monitor.history.series(spec.indicator)
        cfg = self.monitor.cfg
        if len(past) < cfg.min_history:
            return None
        if spec.mode == "signed":
            s = robust_scale(past)
            return (-cfg.band * s, cfg.band * s) if s > 0 else None
        return float(np.quantile(past, cfg.q_lo)), float(np.quantile(past, cfg.q_hi))

    def distance_to_entry(self, name: str) -> float | None:
        """Signed distance of the latest indicator value from the regime's entry cut-off, in the indicator's units: <= 0 means
        already inside (a hi regime: value above the cut; a lo regime: below it)."""
        r = NAMED_BY_NAME[name]
        b = self.boundary(name)
        if b is None or not self.monitor.history.days:
            return None
        v = self.monitor.history.days[-1][1].get(r.indicator)
        if v is None or not math.isfinite(v):
            return None
        hi_regime = r.state == AXIS_BY_NAME[r.axis].hi_state
        return (b[1] - v) if hi_regime else (v - b[0])


# ------------------------------------------------------------------------------------------------ transition detection and lead time

@dataclasses.dataclass(frozen=True)
class TransitionEvent:
    """A persisted move of one axis into a pole state, and whether pressure on the indicator warned of it beforehand."""
    axis: str
    flip_index: int
    flip_date: str
    before: str
    after: str
    warned_index: int | None
    lead_days: int | None                     # flip_index - warned_index; None = no advance warning (or the warning came from the flip itself)


@dataclasses.dataclass(frozen=True)
class EarlyWarning:
    axis: str
    date: str
    toward: str
    z: float


def indicator_pressure(values: np.ndarray, fast: int = 5, slow: int = 30) -> np.ndarray:
    """Shift statistic per day: mean of the last `fast` values minus mean of the `slow` values before them, over its standard
    error (a two-sample z with the slow window's spread). Row t uses values[: t + 1] only. NaN until fast + slow values exist or
    when the slow window is constant."""
    v = np.asarray(values, dtype="float64")
    out = np.full(len(v), np.nan)
    for t in range(fast + slow - 1, len(v)):
        f, s = v[t - fast + 1: t + 1], v[t - fast - slow + 1: t - fast + 1]
        if not (np.isfinite(f).all() and np.isfinite(s).all()):
            continue
        sd = s.std(ddof=1)
        if sd <= 1e-12:
            continue
        out[t] = (f.mean() - s.mean()) / (sd * math.sqrt(1.0 / fast + 1.0 / slow))
    return out


def _axis_arrays(monitor: RegimeMonitor, axis: str) -> tuple[np.ndarray, list[str], list[str]]:
    spec = AXIS_BY_NAME[axis]
    by_date = {d: v for d, v in monitor.history.days}
    dates = [s.date for s in monitor.states if s.date in by_date]
    vals = np.array([by_date[d].get(spec.indicator) if by_date[d].get(spec.indicator) is not None else np.nan for d in dates], dtype="float64")
    states = [monitor.states[i].states.get(axis, UNKNOWN_STATE) for i, s in enumerate(monitor.states) if s.date in by_date]
    return vals, states, dates


def current_warning(monitor: RegimeMonitor, axis: str, fast: int = 5, slow: int = 30, z_bar: float = 3.0) -> EarlyWarning | None:
    """Is the axis's indicator being pushed toward a pole it is not yet in, as of the latest processed day? Past-only."""
    spec = AXIS_BY_NAME[axis]
    vals, states, dates = _axis_arrays(monitor, axis)
    if len(vals) < fast + slow or states[-1] == UNKNOWN_STATE:
        return None
    z = indicator_pressure(vals, fast, slow)[-1]
    if not np.isfinite(z) or abs(z) < z_bar:
        return None
    toward = spec.hi_state if z > 0 else spec.lo_state
    return None if states[-1] == toward else EarlyWarning(axis, dates[-1], toward, float(z))


def transition_events(monitor: RegimeMonitor, axis: str, fast: int = 5, slow: int = 30, z_bar: float = 3.0,
                      lookback: int = 25, min_persist: int = 3) -> list[TransitionEvent]:
    """Retrospective (research-side) evaluation: every persisted move into a pole state, and how many days before the flip the
    indicator pressure toward that pole began and stayed on (a run of consecutive warning days ending at or before the flip).
    lead_days = 0 or None means the flip was not foreshadowed."""
    spec = AXIS_BY_NAME[axis]
    vals, states, dates = _axis_arrays(monitor, axis)
    press = indicator_pressure(vals, fast, slow)
    events = []
    for i in range(1, len(states)):
        after, before = states[i], states[i - 1]
        if UNKNOWN_STATE in (after, before) or after == before or after not in (spec.hi_state, spec.lo_state):
            continue
        if any(s != after for s in states[i: i + min_persist]) or len(states[i: i + min_persist]) < min_persist:
            continue
        sign = 1.0 if after == spec.hi_state else -1.0
        j = i - 1
        first = None
        while j >= max(i - lookback, 0) and np.isfinite(press[j]) and sign * press[j] >= z_bar and states[j] != after:
            first = j
            j -= 1
        events.append(TransitionEvent(axis, i, dates[i], before, after, first, None if first is None else i - first))
    return events


def false_alarm_rate(monitor: RegimeMonitor, axis: str, fast: int = 5, slow: int = 30, z_bar: float = 3.0, horizon: int = 25) -> float | None:
    """Share of warning episodes (a run of consecutive pressure days toward a pole the axis is not in) NOT followed by entry into
    that pole within `horizon` days. None when there were no warnings at all."""
    spec = AXIS_BY_NAME[axis]
    vals, states, _ = _axis_arrays(monitor, axis)
    press = indicator_pressure(vals, fast, slow)
    runs = []
    i = 0
    while i < len(press):
        if np.isfinite(press[i]) and abs(press[i]) >= z_bar and states[i] != UNKNOWN_STATE:
            tgt = spec.hi_state if press[i] > 0 else spec.lo_state
            j = i
            while j + 1 < len(press) and np.isfinite(press[j + 1]) and (press[j + 1] > 0) == (press[i] > 0) and abs(press[j + 1]) >= z_bar:
                j += 1
            if states[i] != tgt:
                runs.append((i, j, tgt))
            i = j + 1
        else:
            i += 1
    if not runs:
        return None
    misses = sum(1 for i, j, tgt in runs if not any(s == tgt for s in states[i: j + horizon + 1]))
    return misses / len(runs)


@dataclasses.dataclass(frozen=True)
class LeadTimeEstimate:
    axis: str
    n_events: int
    n_warned: int
    hit_rate: float | None
    median_lead: float | None
    lead_iqr: tuple[float, float] | None
    false_alarm_rate: float | None


def lead_time_estimate(monitor: RegimeMonitor, axis: str, fast: int = 5, slow: int = 30, z_bar: float = 3.0, lookback: int = 25) -> LeadTimeEstimate:
    """How much notice does indicator pressure give before this axis flips into a pole? Median lead over the warned events, the
    share of events warned at all, and the false-alarm rate of the warning rule. The estimate is only as good as the number of
    events: fewer than 3 warned events report no median (None), never a guess."""
    ev = transition_events(monitor, axis, fast, slow, z_bar, lookback)
    warned = [e.lead_days for e in ev if e.lead_days]
    med = float(np.median(warned)) if len(warned) >= 3 else None
    iqr = (float(np.percentile(warned, 25)), float(np.percentile(warned, 75))) if len(warned) >= 3 else None
    return LeadTimeEstimate(axis, len(ev), len(warned), len(warned) / len(ev) if ev else None, med, iqr,
                            false_alarm_rate(monitor, axis, fast, slow, z_bar))


def all_warnings(monitor: RegimeMonitor, **kw) -> list[EarlyWarning]:
    """Current early warnings across all axes (past-only, latest day)."""
    return [w for a in AXES if (w := current_warning(monitor, a.name, **kw)) is not None]


# ------------------------------------------------------------------------------------------------ regime features for the models

COMPOSITE_CODES = {"unknown": 0, "bull_calm": 1, "bull_volatile": 2, "correction_calm": 3, "bear_volatile": 4, "stress": 5}


def regime_feature_frame(monitor: RegimeMonitor, fast: int = 5, slow: int = 30, z_bar: float = 3.0) -> pd.DataFrame:
    """Date-indexed feature table of the whole regime picture, point in time (row t uses only days <= t): one 0/1/NaN column per
    named regime (NaN = axis unknown), the days the axis has been in its state, the composite label code, and per axis a
    pressure code (+1 pushed toward the hi pole, -1 toward the lo pole, 0 none). For the curator / models, never handed to the
    trader raw (use trader_regime_row)."""
    if not monitor.states:
        return pd.DataFrame()
    rows = {}
    for st in monitor.states:
        r = {n: (np.nan if f is None else float(f)) for n, f in regime_flags(st).items()}
        r.update({f"dwell_{a}": float(d) for a, d in st.dwell.items()})
        r["composite_code"] = float(COMPOSITE_CODES.get(st.composite, 0))
        rows[st.date] = r
    df = pd.DataFrame.from_dict(rows, orient="index")
    for spec in AXES:
        vals, states, dates = _axis_arrays(monitor, spec.name)
        press = indicator_pressure(vals, fast, slow)
        code = np.where(np.isfinite(press) & (np.abs(press) >= z_bar), np.sign(press), 0.0)
        toward = np.where(code > 0, spec.hi_state, np.where(code < 0, spec.lo_state, ""))
        already = np.array([s == t for s, t in zip(states, toward)])
        df[f"pressure_{spec.name}"] = pd.Series(np.where(already, 0.0, code), index=dates)
    return df


# ------------------------------------------------------------------------------------------------ two-way pattern x regime

def two_way_effects(book: PatternRegimeBook, pattern_id: str, axis_a: str, axis_b: str, now, replay_years: Iterable[int] = (),
                    t_bar: float = T_BAR) -> list[dict[str, Any]]:
    """Joint-cell effects of a pattern (state of axis A x state of axis B) against the additive prediction from the two one-way
    effects: interaction = cell - (overall + (a - overall) + (b - overall)). The interactions are shrunk toward zero with a
    random-effects estimate of their spread, so a thin cell cannot claim a large interaction. This is where 'works only in
    high volatility AND low liquidity' shows up when neither one-way effect explains it."""
    df = book.frame(pattern_id, now, replay_years)
    if df.empty or axis_a not in df.columns or axis_b not in df.columns:
        return []
    days = pd.Index(sorted(df["date"].unique()))
    overall = book._effect(df, "all", "all", days)
    if overall.effect is None:
        return []
    one = {}
    for ax in (axis_a, axis_b):
        for st in df[ax].unique():
            if st != UNKNOWN_STATE:
                one[(ax, st)] = book._effect(df, ax, st, days)
    cells = []
    for sa in sorted(x for x in df[axis_a].unique() if x != UNKNOWN_STATE):
        for sb in sorted(x for x in df[axis_b].unique() if x != UNKNOWN_STATE):
            sub = df[(df[axis_a] == sa) & (df[axis_b] == sb)].assign(_all="all")
            e = book._effect(sub, "_all", "all", days)
            ea, eb = one[(axis_a, sa)], one[(axis_b, sb)]
            if e.effect is None or e.se is None or ea.effect is None or eb.effect is None:
                cells.append({"a": sa, "b": sb, "n_days": e.n_days, "effect": e.effect, "additive": None, "interaction": None,
                              "shrunk_interaction": None, "t": None, "established": False})
                continue
            add = overall.effect + (ea.effect - overall.effect) + (eb.effect - overall.effect)
            cells.append({"a": sa, "b": sb, "n_days": e.n_days, "effect": e.effect, "additive": add, "interaction": e.effect - add,
                          "se": e.se, "shrunk_interaction": None, "t": None, "established": False})
    meas = [c for c in cells if c["interaction"] is not None]
    if len(meas) >= 2:
        y = np.array([c["interaction"] for c in meas])
        w = np.array([1.0 / c["se"] ** 2 for c in meas])
        mu0 = 0.0
        q = float((w * (y - mu0) ** 2).sum())
        tau2 = max(0.0, (q - len(meas)) / float(w.sum()) * len(meas)) if w.sum() > 0 else 0.0
        for c in meas:
            b = tau2 / (tau2 + c["se"] ** 2)
            c["shrunk_interaction"] = b * c["interaction"]
            se_post = math.sqrt(b * c["se"] ** 2)
            c["t"] = c["shrunk_interaction"] / se_post if se_post > 1e-15 else 0.0
            c["established"] = abs(c["t"]) >= t_bar
    return cells


# ------------------------------------------------------------------------------------------------ patterns break at regime changes?

def effect_after_transitions(book: PatternRegimeBook, monitor: RegimeMonitor, pattern_id: str, axis: str, now, window: int = 10,
                             replay_years: Iterable[int] = (), min_days: int = 20) -> dict[str, Any]:
    """Contract section 14 x 25: does a pattern weaken right after the market changes state on `axis`? Days within `window` days
    after a persisted transition versus all other days: mean effect in each, and the difference with an overlap-corrected error.
    A pattern that only fails in the weeks after regime changes is a candidate for 'gate off during transitions'."""
    df = book.frame(pattern_id, now, replay_years)
    ev = transition_events(monitor, axis)
    if df.empty or not ev:
        return {"n_post": 0, "n_other": len(df), "diff": None, "t": None, "verdict": "INSUFFICIENT_DATA"}
    dates = [s.date for s in monitor.states]
    post = set()
    for e in ev:
        post.update(dates[e.flip_index: e.flip_index + window])
    flag = df["date"].isin(post)
    if flag.sum() < min_days or (~flag).sum() < min_days:
        return {"n_post": int(flag.sum()), "n_other": int((~flag).sum()), "diff": None, "t": None, "verdict": "INSUFFICIENT_DATA"}
    days = pd.Index(sorted(df["date"].unique()))
    a = book._effect(df[flag].assign(_all="all"), "_all", "all", days)
    b = book._effect(df[~flag].assign(_all="all"), "_all", "all", days)
    if a.se is None or b.se is None:
        return {"n_post": a.n_days, "n_other": b.n_days, "diff": None, "t": None, "verdict": "INSUFFICIENT_DATA"}
    (ea, sa), (eb, sb) = a.effect_se(), b.effect_se()
    diff = ea - eb
    t = diff / math.sqrt(sa ** 2 + sb ** 2)
    verdict = "WEAKER_AFTER_TRANSITIONS" if t <= -T_BAR else "STRONGER_AFTER_TRANSITIONS" if t >= T_BAR else "NO_DIFFERENCE"
    return {"n_post": a.n_days, "n_other": b.n_days, "post": a.effect, "other": b.effect, "diff": diff, "t": t, "verdict": verdict}


def verdict_by_era(book: PatternRegimeBook, pattern_id: str, now, era_edges: Sequence[Any], replay_years: Iterable[int] = ()) -> dict[str, str]:
    """The pattern x regime verdict re-derived inside each era ([e0, e1), [e1, e2) ...). A pattern whose verdict changes from era to
    era (BOUND in one, UNIVERSAL in another) is not settled: the regime dependence itself is unstable."""
    rows = book._rows.get(pattern_id, [])
    edges = [as_date(e).isoformat() for e in era_edges]
    out = {}
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sub = PatternRegimeBook(book.sessions, book.min_days)
        sub._rows = {pattern_id: [r for r in rows if a <= r[0] < b]}
        out[f"era{i}"] = sub.report(pattern_id, now, replay_years).verdict.value
    return out


def regime_dependence_stable(verdicts: Mapping[str, str]) -> bool | None:
    """True if every era with a verdict agrees; None if fewer than two eras were decidable (INSUFFICIENT_DATA is not a verdict)."""
    v = [x for x in verdicts.values() if x != Verdict.INSUFFICIENT_DATA.value]
    return None if len(v) < 2 else len(set(v)) == 1


# ------------------------------------------------------------------------------------------------ monitor self-checks

@dataclasses.dataclass(frozen=True)
class AxisHealth:
    axis: str
    unknown_share: float
    flips_per_100: float                      # state changes per 100 known days: high = flickering axis
    dominant_share: float                     # share of known days in the single most common state: ~1 = a stuck axis
    issues: tuple[str, ...]


def axis_health(monitor: RegimeMonitor, max_flips: float = 12.0, max_dominant: float = 0.9, max_unknown: float = 0.5) -> list[AxisHealth]:
    """Is each axis behaving like a regime? An axis that flips every few days is measuring noise; one that sits in a single state is
    measuring nothing (its thresholds are wrong or the indicator is dead); one that is mostly UNKNOWN has a broken input. Reported
    per axis; the research loop should distrust any pattern verdict built on an unhealthy axis."""
    out = []
    n = len(monitor.states)
    for spec in AXES:
        seq = [s.states.get(spec.name, UNKNOWN_STATE) for s in monitor.states]
        known = [x for x in seq if x != UNKNOWN_STATE]
        unk = 1.0 - len(known) / n if n else 1.0
        flips = sum(1 for a, b in zip(known, known[1:]) if a != b)
        fl = 100.0 * flips / max(len(known) - 1, 1)
        dom = max((known.count(x) for x in set(known)), default=0) / len(known) if known else 0.0
        issues = []
        if unk > max_unknown:
            issues.append("MOSTLY_UNKNOWN")
        if known and fl > max_flips:
            issues.append("FLICKERING")
        if known and dom > max_dominant:
            issues.append("STUCK")
        out.append(AxisHealth(spec.name, unk, fl, dom, tuple(issues)))
    return out


def unhealthy_axes(monitor: RegimeMonitor, **kw) -> set[str]:
    return {h.axis for h in axis_health(monitor, **kw) if h.issues}


def definition_agreement(monitor: RegimeMonitor) -> pd.DataFrame:
    """The named states are one definition among many. For each quantile-mode axis, re-classify every day with an alternative rule
    (past mean +- 0.5 past standard deviations instead of past terciles) and report the share of days on which the two rules
    agree about hi / neutral / lo. Low agreement means the axis is a matter of definition and its patterns should be distrusted."""
    rows = []
    for spec in AXES:
        if spec.mode != "quantile":
            continue
        past: list[float] = []
        same = tot = 0
        base = [s.states.get(spec.name, UNKNOWN_STATE) for s in monitor.states]
        for (d, v), st in zip(monitor.history.days, base):
            x = v.get(spec.indicator)
            if x is not None and math.isfinite(x) and len(past) >= monitor.cfg.min_history and st != UNKNOWN_STATE:
                mu, sd = float(np.mean(past)), float(np.std(past))
                alt = spec.hi_state if x > mu + 0.5 * sd else spec.lo_state if x < mu - 0.5 * sd else NEUTRAL
                same += alt == st
                tot += 1
            if x is not None and math.isfinite(x):
                past.append(x)
        rows.append({"axis": spec.name, "days": tot, "agreement": same / tot if tot else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ discovered regimes: meaning and history

def describe_discovered(discovery: DiscoveredRegimes) -> dict[str, dict[str, str]]:
    """Plain-language reading of each discovered regime: for every feature the cluster centre versus the middle of all centres
    ('high' / 'low' / 'mid' by half a fit-scale unit). A discovered regime is only useful if a person can say what it is."""
    if discovery.centroids is None or not discovery.ids:
        return {}
    raw = discovery.centroids * discovery.scale + discovery.center
    mid = raw.mean(axis=0)
    out = {}
    for i, rid in enumerate(discovery.ids):
        d = {}
        for j, f in enumerate(discovery.FEATURES):
            z = (raw[i, j] - mid[j]) / discovery.scale[j]
            d[f] = "high" if z > 0.5 else "low" if z < -0.5 else "mid"
        out[rid] = d
    return out


def refit_history(discovery: DiscoveredRegimes) -> dict[str, Any]:
    """Stability of the discovery over its refits: status counts, the sequence of k among ACCEPTED fits, whether that k ever
    changed, how many distinct ids appeared beyond the largest k (id churn), and the mean centroid drift of inherited ids."""
    fits = discovery.fits
    acc = [f for f in fits if f.status == "ACCEPTED"]
    ids = {i for f in acc for i in f.ids}
    ks = [f.k for f in acc]
    costs = [f.match_cost for f in acc if f.match_cost is not None]
    return {"n_fits": len(fits), "status": dict(pd.Series([f.status for f in fits]).value_counts()) if fits else {},
            "k_sequence": ks, "k_changed": len(set(ks)) > 1, "id_churn": max(len(ids) - max(ks, default=0), 0),
            "mean_drift": float(np.mean(costs)) if costs else None,
            "mean_stability": float(np.mean([f.stability for f in acc])) if acc else None}


def discovered_transition_matrix(monitor: RegimeMonitor) -> pd.DataFrame:
    """Row-normalised day-to-day transitions between discovered regimes (known days only)."""
    seq = [s.discovered for s in monitor.states]
    counts: dict[tuple[str, str], int] = {}
    for a, b in zip(seq, seq[1:]):
        if UNKNOWN_STATE not in (a, b):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        return pd.DataFrame()
    m = pd.Series(counts).unstack(fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


# ------------------------------------------------------------------------------------------------ pattern x regime: prediction and ranking

def regime_adjusted_effect(book: PatternRegimeBook, pattern_id: str, state: RegimeState, now, replay_years: Iterable[int] = ()) -> dict[str, Any]:
    """Predicted effect of the pattern in TODAY's full regime: the overall effect plus, for every axis whose state is known, the
    pooled (shrunk) deviation of that state from the overall effect. Additive by construction; use two_way_effects to check for
    interactions. Axes whose state is unknown or unmeasured contribute nothing and are listed as unused."""
    rep = book.report(pattern_id, now, replay_years)
    if rep.overall.effect is None:
        return {"effect": None, "used": [], "unused": [], "verdict": rep.verdict.value}
    total, used, unused = rep.overall.effect, [], []
    for spec in AXES:
        cur = state.states.get(spec.name, UNKNOWN_STATE)
        p = next((x for x in rep.pooled.get(spec.name, ()) if x.state == cur), None)
        if cur == UNKNOWN_STATE or p is None:
            unused.append(spec.name)
            continue
        total += p.effect - rep.overall.effect
        used.append(spec.name)
    return {"effect": total, "used": used, "unused": unused, "verdict": rep.verdict.value}


def rank_by_regime_dependence(book: PatternRegimeBook, now, replay_years: Iterable[int] = ()) -> pd.DataFrame:
    """One row per pattern: verdict, its most discriminating axis, the between-state spread (tau) on that axis, the share of held-out
    volatility regimes it transfers to (leave-one-regime-out) and its overall t. The table the research loop reads to decide which
    patterns must be gated by regime and which may be treated as universal."""
    rows = []
    for pid in book.patterns():
        rep = book.report(pid, now, replay_years)
        ba = best_axis(book, pid, now, replay_years)
        loro = unseen_regime_transfer(book, pid, "volatility", now, replay_years)
        rows.append({"pattern": pid, "verdict": rep.verdict.value, "best_axis": ba[0] if ba else None,
                     "axis_t": ba[1] if ba else None, "tau": math.sqrt(rep.tau2[ba[0]]) if ba and ba[0] in rep.tau2 else None,
                     "transfer_share": (sum(r["transfers"] for r in loro) / len(loro)) if loro else None, "overall_t": rep.overall.t})
    return pd.DataFrame(rows).set_index("pattern") if rows else pd.DataFrame()


def universal_claims_audit(book: PatternRegimeBook, claims: Iterable[str], now, replay_years: Iterable[int] = ()) -> list[str]:
    """Patterns someone is treating as universal (claims) that the book does NOT support as UNIVERSAL. The contract: a pattern that
    works in one regime must not be treated as universal. Returns the offending ids, sorted."""
    return sorted(p for p in claims if book.report(p, now, replay_years).verdict != Verdict.UNIVERSAL)


# ------------------------------------------------------------------------------------------------ how long do regimes last?

def survival_curve(named: NamedRegimeMonitor, name: str, horizon: int = 120) -> pd.Series:
    """Kaplan-Meier probability that an episode of the regime lasts at least d days, d = 1..horizon, from the monitor's own past
    episodes. The still-open episode is right-censored, not counted as finished. Empty when the regime never occurred."""
    eps = named.episodes(name)
    if not eps:
        return pd.Series(dtype="float64")
    durations = np.array([e.days for e in eps])
    observed = np.array([e.end is not None for e in eps])
    surv, s = {}, 1.0
    for d in range(1, horizon + 1):
        at_risk = int((durations >= d).sum())
        ended = int(((durations == d) & observed).sum())
        if at_risk > 0:
            s *= 1.0 - ended / at_risk
        surv[d] = s
    return pd.Series(surv)


def expected_remaining(named: NamedRegimeMonitor, name: str, age: int, horizon: int = 250) -> float | None:
    """Expected further days in the regime given it has already lasted `age` days: the area under S(d)/S(age) beyond age from the
    Kaplan-Meier curve. None when fewer than 3 episodes or when the curve never falls below the age (all episodes longer)."""
    if len(named.episodes(name)) < 3:
        return None
    s = survival_curve(named, name, horizon)
    if s.empty or age < 1 or age > horizon or s.loc[age] <= 0:
        return None
    tail = s.loc[age + 1:] / s.loc[age]
    return float(tail.sum()) if len(tail) else 0.0


def axis_persistence_test(monitor: RegimeMonitor, seed: int = 0, n_shuffles: int = 100) -> dict[str, float | None]:
    """Does each axis behave like a REGIME (states persist) or like a coin? Observed mean run length over the same states shuffled
    in time, per axis. Near 1: the labels are noise. The contract's 'do not hard-code' cuts both ways: an axis that does not
    persist is not a regime and should not be conditioned on."""
    out = {}
    for spec in AXES:
        seq = [s.states.get(spec.name, UNKNOWN_STATE) for s in monitor.states]
        out[spec.name] = dwell_ratio([x for x in seq if x != UNKNOWN_STATE], seed, n_shuffles)[0]
    return out


def shock_days(monitor: RegimeMonitor, axis: str, z_bar: float = 5.0) -> list[str]:
    """Days on which the axis's indicator jumped by more than z_bar robust sigmas of its own day-to-day changes: a shock, not a
    drift. Shocks are where regimes can change without any advance pressure (lead time zero)."""
    vals, _states, dates = _axis_arrays(monitor, axis)
    d = np.diff(vals)
    s = robust_scale(d[np.isfinite(d)])
    if s <= 0:
        return []
    return [dates[i + 1] for i in range(len(d)) if np.isfinite(d[i]) and abs(d[i]) > z_bar * s]


# ------------------------------------------------------------------------------------------------ combinations of regimes

def cooccurrence(monitor: RegimeMonitor, regime_a: str, regime_b: str) -> dict[str, float | None]:
    """How often two named regimes are active together: P(b | a), P(b) and the lift P(b|a) / P(b) over days where both axes are
    known. Lift near 1: independent; well above 1: they travel together (bear and high_vol), so conditioning on both double
    counts one thing."""
    a, b = NAMED_BY_NAME[regime_a], NAMED_BY_NAME[regime_b]
    both = na = nb = n = 0
    for st in monitor.states:
        fa, fb = regime_flags(st)[regime_a], regime_flags(st)[regime_b]
        if fa is None or fb is None:
            continue
        n += 1
        na += fa
        nb += fb
        both += fa and fb
    if n == 0 or na == 0 or nb == 0:
        return {"n": n, "p_b_given_a": None, "p_b": None, "lift": None}
    p_b, p_ba = nb / n, both / na
    return {"n": n, "p_b_given_a": p_ba, "p_b": p_b, "lift": p_ba / p_b}


def joint_state_table(monitor: RegimeMonitor, axis_a: str = "direction", axis_b: str = "volatility") -> pd.DataFrame:
    """Days in every (state of A, state of B) cell, known days only: which combined regimes the history actually contains. A cell
    with few days cannot carry a verdict for any pattern, whatever the one-way tables say."""
    f = monitor.frame()
    if f.empty:
        return pd.DataFrame()
    sub = f[(f[axis_a] != UNKNOWN_STATE) & (f[axis_b] != UNKNOWN_STATE)]
    return pd.crosstab(sub[axis_a], sub[axis_b]) if len(sub) else pd.DataFrame()


def regime_report_card(monitor: RegimeMonitor, named: NamedRegimeMonitor | None = None) -> str:
    """Plain-text card: axis health, persistence ratios, twelve-regime occupancy, discovery history and any live early warnings.
    Everything is a diagnostic of the monitor itself, so a reader can see when it is not to be trusted."""
    named = named or NamedRegimeMonitor(monitor)
    named.update()
    lines = [f"REGIME CARD  days {len(monitor.states)}  discovery {monitor.discovery.status()}"]
    for h in axis_health(monitor):
        lines.append(f"  axis {h.axis:<11} unknown {h.unknown_share:.2f}  flips/100 {h.flips_per_100:.1f}  dominant {h.dominant_share:.2f}  {list(h.issues)}")
    lines.append("  persistence ratio: " + ", ".join(f"{k}:{'n/a' if v is None else round(v, 2)}" for k, v in axis_persistence_test(monitor).items()))
    lines.extend("  " + ln for ln in named.summary().round(3).to_string().splitlines())
    lines.extend(f"  WARNING {w.axis} -> {w.toward} (z {w.z:.1f})" for w in all_warnings(monitor))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ not final truths: tune and test the definitions

def _replay_states(monitor: RegimeMonitor, spec: AxisSpec, cfg: RegimeConfig) -> list[str]:
    """Re-classify the stored indicator history for one axis under a given config, past-only per day, hysteresis on."""
    past: list[float] = []
    prev, out = None, []
    for _, v in monitor.history.days:
        x = v.get(spec.indicator)
        st = classify_axis(spec, x, np.array(past), prev, cfg)
        out.append(st)
        prev = st if st != UNKNOWN_STATE else prev
        if x is not None and math.isfinite(x):
            past.append(x)
    return out


def tune_quantiles(monitor: RegimeMonitor, axis: str, grid: Sequence[tuple[float, float]] = ((0.2, 0.8), (0.25, 0.75), (0.33, 0.67), (0.4, 0.6)),
                   seed: int = 0, min_state_share: float = 0.1) -> pd.DataFrame:
    """The 33/67 cut-offs are a default, not a truth. For each candidate pair on a quantile-mode axis: the share of days in each pole,
    the persistence ratio of the resulting labels (observed run length over shuffled) and a score = ratio when both poles hold at
    least min_state_share of the days, else 0 (a regime that almost never happens cannot be studied). The pair with the best score
    is the data's preference; the research loop may adopt it through RegimeConfig, never silently."""
    spec = AXIS_BY_NAME[axis]
    if spec.mode != "quantile":
        raise ValueError(f"{axis} is a signed axis; tune its band instead")
    rows = []
    for q_lo, q_hi in grid:
        states = [x for x in _replay_states(monitor, spec, dataclasses.replace(monitor.cfg, q_lo=q_lo, q_hi=q_hi)) if x != UNKNOWN_STATE]
        if len(states) < 30:
            rows.append({"q_lo": q_lo, "q_hi": q_hi, "n": len(states), "hi_share": None, "lo_share": None, "persistence": None, "score": 0.0})
            continue
        hi, lo = states.count(spec.hi_state) / len(states), states.count(spec.lo_state) / len(states)
        ratio = dwell_ratio(states, seed, 60)[0]
        rows.append({"q_lo": q_lo, "q_hi": q_hi, "n": len(states), "hi_share": hi, "lo_share": lo, "persistence": ratio,
                     "score": float(ratio) if ratio is not None and min(hi, lo) >= min_state_share else 0.0})
    return pd.DataFrame(rows)


def axis_information(monitor: RegimeMonitor, axis: str, returns: pd.Series, now) -> dict[str, float | None]:
    """Does the axis's state today say anything about tomorrow's market? Spearman correlation between the ordinal state code (-1 lo,
    0 neutral, +1 hi) and the NEXT day's absolute return (volatility information) and signed return (direction information), with
    the number of days. Only days whose next return is strictly before `now`. An axis that informs neither is a label, not a regime."""
    f = monitor.frame()
    if f.empty or axis not in f.columns:
        return {"n": 0, "abs_ic": None, "signed_ic": None}
    r = returns.copy()
    r.index = pd.DatetimeIndex(r.index).strftime("%Y-%m-%d")
    nxt = r.shift(-1)
    end = pd.Series(list(r.index[1:]) + [None], index=r.index)
    code = f[axis].map(lambda s: np.nan if s == UNKNOWN_STATE else STATE_CODE[s])
    df = pd.DataFrame({"code": code, "nxt": nxt.reindex(f.index), "end": end.reindex(f.index)}).dropna()
    df = df[df["end"].map(lambda d: as_date(d) < as_date(now))]
    if len(df) < 30 or df["code"].nunique() < 2:
        return {"n": int(len(df)), "abs_ic": None, "signed_ic": None}
    return {"n": int(len(df)), "abs_ic": float(df["code"].corr(df["nxt"].abs(), method="spearman")),
            "signed_ic": float(df["code"].corr(df["nxt"], method="spearman"))}


def regime_diagnostics(monitor: RegimeMonitor, returns: pd.Series, now) -> pd.DataFrame:
    """One row per axis: health issues, persistence ratio, definition agreement and next-day information. The table that says which
    of the six axes are worth conditioning patterns on. Identity-free."""
    health = {h.axis: h for h in axis_health(monitor)}
    pers = axis_persistence_test(monitor)
    agree = agreement_by_axis(monitor)
    rows = []
    for spec in AXES:
        info = axis_information(monitor, spec.name, returns, now)
        rows.append({"axis": spec.name, "issues": ",".join(health[spec.name].issues), "persistence": pers[spec.name],
                     "agreement": agree.get(spec.name), "abs_ic": info["abs_ic"], "signed_ic": info["signed_ic"], "n": info["n"]})
    return pd.DataFrame(rows).set_index("axis")


def agreement_by_axis(monitor: RegimeMonitor) -> dict[str, float | None]:
    """definition_agreement as a plain dict (signed axes have no alternative definition here and are absent)."""
    da = definition_agreement(monitor)
    return {r["axis"]: r["agreement"] for _, r in da.iterrows()} if not da.empty else {}


# ------------------------------------------------------------------------------------------------ knowledge about the regimes themselves

def monitor_records(monitor: RegimeMonitor, now, discovery_only: bool = False) -> list[MaturedRecord]:
    """Identity-free MaturedRecords of what the monitor has learned about its own definitions (axis persistence, health, discovery
    stability), for the curator. No date, year or ticker in a payload; maturity is the last processed day, which must be strictly
    before `now`. Records built on an unhealthy axis say so in their payload instead of being left out."""
    if not monitor.states:
        return []
    through = monitor.states[-1].date
    require_past(through, now, "regime monitor")
    prov = Provenance(created_real=str(as_date(now)), learned_at=through, code_hash=current_code_hash(), outcomes_seen_through=through)
    recs = []
    if not discovery_only:
        pers, health = axis_persistence_test(monitor), {h.axis: h for h in axis_health(monitor)}
        for spec in AXES:
            payload = {"axis": spec.name, "persistence": clean_number(pers[spec.name]), "issues": list(health[spec.name].issues),
                       "flips_per_100": clean_number(health[spec.name].flips_per_100)}
            recs.append(MaturedRecord("RM" + stable_hash([spec.name, through], 10), through, payload, prov, Namespace.MATURED_RESEARCH))
    hist = refit_history(monitor.discovery)
    dpay = {"axis": "discovered", "status": monitor.discovery.status(), "n_fits": hist["n_fits"], "k_changed": hist["k_changed"],
            "id_churn": hist["id_churn"], "mean_stability": clean_number(hist["mean_stability"])}
    recs.append(MaturedRecord("RM" + stable_hash(["discovered", through], 10), through, dpay, prov, Namespace.MATURED_RESEARCH))
    return recs


def rebuild_named(monitor: RegimeMonitor) -> NamedRegimeMonitor:
    """A NamedRegimeMonitor reconstructed from a monitor's stored states (e.g. after RegimeMonitor.from_state), so the twelve-regime
    episode history never has to be persisted separately: it is a pure function of the states."""
    named = NamedRegimeMonitor(monitor)
    named.update()
    return named


def adopt_tuned(cfg: RegimeConfig, tuning: pd.DataFrame, min_gain: float = 0.2) -> tuple[RegimeConfig, dict[str, Any]]:
    """Turn a tune_quantiles table into a NEW config, only if the best pair beats the config's current pair by min_gain (relative)
    on the persistence score; otherwise return cfg unchanged. Returns (config, decision) - the decision records the old and new
    pairs, their scores and why, so a change of definition is an explicit, logged act, never a silent drift. One quantile pair
    applies to every quantile-mode axis, so tune on the axis you care about and check the others with tune_quantiles too."""
    if tuning.empty or tuning["score"].max() <= 0:
        return cfg, {"adopted": False, "reason": "no candidate with populated, persistent poles"}
    cur = tuning[(tuning["q_lo"].round(6) == round(cfg.q_lo, 6)) & (tuning["q_hi"].round(6) == round(cfg.q_hi, 6))]
    cur_score = float(cur["score"].iloc[0]) if len(cur) else 0.0
    best = tuning.loc[tuning["score"].idxmax()]
    if cur_score > 0 and best["score"] < (1.0 + min_gain) * cur_score:
        return cfg, {"adopted": False, "reason": f"best {best['score']:.2f} is not {min_gain:.0%} above current {cur_score:.2f}",
                     "current": (cfg.q_lo, cfg.q_hi)}
    new = dataclasses.replace(cfg, q_lo=float(best["q_lo"]), q_hi=float(best["q_hi"]))
    errs = new.validate()
    if errs:
        return cfg, {"adopted": False, "reason": "; ".join(errs)}
    return new, {"adopted": True, "from": (cfg.q_lo, cfg.q_hi), "to": (new.q_lo, new.q_hi), "old_score": cur_score, "new_score": float(best["score"]),
                 "config_hash": stable_hash(dataclasses.asdict(new), 12)}


def cohort_shares_by_regime(cohort_ledger, monitor: RegimeMonitor, axis: str, now, min_days: int = 15) -> pd.DataFrame:
    """Section 24 x 25: which cohort explains the day's moves, by regime state (engine.research.cross_section.CohortAttributionLedger,
    duck-typed on .frame(now)). Rows: state; columns: mean sequential share per cohort plus the day count. States with fewer than
    min_days days are left out. 'Sector moves matter in calm markets, the market itself in stress' would show up here."""
    df = cohort_ledger.frame(now)
    if df.empty:
        return pd.DataFrame()
    df = df.assign(state=df["date"].map(monitor.label_map(axis))).dropna(subset=["state"])
    cols = [c for c in df.columns if c.startswith("seq_")]
    g = df.groupby("state")[cols].mean().assign(days=df.groupby("state").size())
    return g[g["days"] >= min_days]


def active_regimes(state: RegimeState) -> tuple[str, ...]:
    """Names of the named regimes active on one day, sorted (unknown axes contribute nothing): the compact 'what is the market
    like today' answer used in reports and as a stable key for grouping days."""
    return tuple(sorted(n for n, f in regime_flags(state).items() if f))
