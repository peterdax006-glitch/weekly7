"""Change-point detection and early regime warning (PREDICTION_ERROR_ADDITION checklists I, R and X; canon C68, C66 sections 13/25/26/48, C63).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; no real-data run was made).

Checklist I: a streaming detector for every relationship the checklist names - feature -> outcome, pattern -> outcome, volatility,
direction, sector relationships, correlations, breadth, momentum persistence, reversal, holding periods, optimal exits, confidence
calibration and prediction-error distributions (Target, build_streams). Each stream is standardised against a past-only reference
(median / MAD, clipped so one print cannot own the statistic) and watched by a two-sided CUSUM whose threshold is CALIBRATED by
simulation of the whole procedure (warm-up estimation included) to a stated false-alarm probability over a stated horizon - so
'as early as statistically justified' has a number attached (calibrate_h, latency_curve), not a hand-picked constant. On an alarm
the detector re-baselines on the new level after a settling period instead of alarming forever.

Checklist X (no retrospective cheating) is enforced by construction: a detector/monitor sees one value at a time, refuses a date not
after the last, and its state at t is a function of values up to t only. run_forward() streams data exactly as history would have;
check_no_lookahead() PROVES it by scrambling everything after t in several ways and demanding a bit-identical state digest (exact
float hex, not rounded) and identical decisions up to t. A Detection carries the information cut-off it used; claiming a change was
'detectable earlier' than its alarm is refused (detectable_at / earlier_claim_valid).

Checklist R: EarlyWarningSystem watches market-level, sector-level and single-stock streams together. Alarms are counted against the
rate chance alone would produce (binomial), so a single name alarming is a SINGLE_STOCK anomaly, one sector alarming above chance is a
SECTOR change, and many sectors alarming (or several market-level streams) is a MARKET_WIDE regime change (classify_scope). The eleven
checklist-R precursors are named signals; regime-layer pressure (engine.research.regimes) joins them as further evidence.

Firewall: everything here is a function of values up to `now`; outputs are MATURED_RESEARCH_STATE until they pass a gate.
Built on: engine.learning.calibration.change_scan (location refinement), engine.research.regimes (RegimeMonitor history and
all_warnings feed the same board), engine.research.multiscale (benjamini_hochberg), engine.research.core, engine.learning.core."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from engine.learning.calibration import change_scan
from engine.learning.core import FirewallBreach, _StrEnum, as_date, stable_hash
from engine.research import regimes as RG
from engine.research.multiscale import benjamini_hochberg

SCHEMA_VERSION = "change_points.v1"


class Target(_StrEnum):
    """The thirteen checklist-I relationships, plus two precursor-only streams named in checklist R."""
    FEATURE_OUTCOME = "FEATURE_OUTCOME"
    PATTERN_OUTCOME = "PATTERN_OUTCOME"
    VOLATILITY = "VOLATILITY"
    DIRECTION = "DIRECTION"
    SECTOR = "SECTOR"
    CORRELATION = "CORRELATION"
    BREADTH = "BREADTH"
    MOMENTUM = "MOMENTUM"
    REVERSAL = "REVERSAL"
    HOLDING_PERIOD = "HOLDING_PERIOD"
    OPTIMAL_EXIT = "OPTIMAL_EXIT"
    CALIBRATION = "CALIBRATION"
    ERROR_DISTRIBUTION = "ERROR_DISTRIBUTION"
    DISPERSION = "DISPERSION"                    # checklist R: abnormal cross-sectional dispersion
    CONFIDENT_FAILURES = "CONFIDENT_FAILURES"    # checklist R: repeated confident failures


CHECKLIST_I = tuple(t for t in Target if t not in (Target.DISPERSION, Target.CONFIDENT_FAILURES))
PRECURSORS = ("prediction_error_shift", "pattern_degradation", "pattern_strengthening", "volatility_change", "breadth_change",
              "correlation_change", "sector_leadership_change", "holding_period_change", "optimal_exit_change", "abnormal_dispersion",
              "repeated_confident_failures")
PRECURSOR_OF: dict[Target, str] = {
    Target.CALIBRATION: "prediction_error_shift", Target.ERROR_DISTRIBUTION: "prediction_error_shift", Target.VOLATILITY: "volatility_change",
    Target.BREADTH: "breadth_change", Target.CORRELATION: "correlation_change", Target.SECTOR: "sector_leadership_change",
    Target.HOLDING_PERIOD: "holding_period_change", Target.OPTIMAL_EXIT: "optimal_exit_change", Target.DISPERSION: "abnormal_dispersion",
    Target.CONFIDENT_FAILURES: "repeated_confident_failures"}


# ------------------------------------------------------------------------------------------------ exact hashing

def _exact(obj: Any) -> Any:
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, float) or (hasattr(obj, "item") and isinstance(obj.item(), float)):
        f = float(obj)
        return f.hex() if math.isfinite(f) else str(f)
    if hasattr(obj, "item") and callable(obj.item):
        return _exact(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _exact(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, deque)):
        return [_exact(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _exact(dataclasses.asdict(obj))
    return str(obj)


def exact_digest(obj: Any) -> str:
    """Bit-exact content hash: floats enter as their hexadecimal form, so a difference in the last bit changes the digest (the
    rounding in learning.core.stable_hash would hide exactly the leak checklist X hunts for)."""
    return hashlib.sha256(json.dumps(_exact(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]


# ------------------------------------------------------------------------------------------------ configuration and calibration

@dataclasses.dataclass(frozen=True)
class ChangeConfig:
    k: float = 0.5                               # CUSUM allowance in reference sigmas (detects shifts of ~1 sigma and up)
    alpha: float = 0.05                          # false-alarm probability per stream over `horizon` null steps
    horizon: int = 250
    warmup: int = 40                             # values used to fix the reference; no alarms before it is set
    clip: float = 4.0
    settle: int = 25                             # post-alarm values used to re-baseline; alarms are muted meanwhile
    scan_window: int = 120                       # recent standardised values searched to locate the change
    scan_threshold: float = 3.0
    n_sim: int = 400
    sim_seed: int = 20260929
    scope_window: int = 10                       # steps over which alarms count as 'simultaneous' for scope classification
    scope_alpha: float = 0.01
    min_sector_units: int = 5
    max_single: int = 3
    min_market_share: float = 0.10
    min_market_sectors: int = 2

    def validate(self) -> list[str]:
        e = []
        if self.k <= 0 or not 0 < self.alpha < 0.5 or self.horizon < 20:
            e.append("k > 0, 0 < alpha < 0.5 and horizon >= 20 required")
        if self.warmup < 20 or self.clip < 2 or self.settle < 10:
            e.append("warmup >= 20, clip >= 2 and settle >= 10 required")
        if self.n_sim < 100:
            e.append("n_sim >= 100 required for a stable threshold")
        if self.scope_window < 2 or not 0 < self.scope_alpha < 0.5:
            e.append("scope_window >= 2 and 0 < scope_alpha < 0.5 required")
        return e


def reference_of(v: np.ndarray) -> tuple[Any, Any]:
    """(centre, scale) of a reference sample along the last axis: median and 1.4826 MAD, falling back to the standard deviation
    when the MAD is zero, floored so a constant reference cannot make the next value infinitely surprising."""
    v = np.asarray(v, dtype="float64")
    c = np.median(v, axis=-1)
    s = 1.4826 * np.median(np.abs(v - np.expand_dims(c, -1)), axis=-1)
    s = np.where(s > 0, s, np.std(v, axis=-1))
    return c, np.maximum(s, 1e-6 * np.maximum(1.0, np.abs(c)))


_H_CACHE: dict[tuple, float] = {}


def calibrate_h(cfg: ChangeConfig) -> float:
    """The CUSUM threshold h for which the WHOLE procedure (reference estimated from `warmup` null values, clipped, two-sided) raises
    a false alarm within `horizon` null steps with probability alpha. Simulated with a fixed seed: same config, same h."""
    key = (cfg.k, cfg.alpha, cfg.horizon, cfg.warmup, cfg.clip, cfg.n_sim, cfg.sim_seed)
    if key in _H_CACHE:
        return _H_CACHE[key]
    rng = np.random.default_rng(cfg.sim_seed)
    x = rng.standard_normal((cfg.n_sim, cfg.warmup + cfg.horizon))
    c, s = reference_of(x[:, : cfg.warmup])
    z = np.clip((x[:, cfg.warmup:] - c[:, None]) / s[:, None], -cfg.clip, cfg.clip)
    hi = lo = np.zeros(cfg.n_sim)
    peak = np.zeros(cfg.n_sim)
    for t in range(cfg.horizon):
        hi = np.maximum(0.0, hi + z[:, t] - cfg.k)
        lo = np.minimum(0.0, lo + z[:, t] + cfg.k)
        peak = np.maximum(peak, np.maximum(hi, -lo))
    _H_CACHE[key] = float(np.quantile(peak, 1.0 - cfg.alpha))
    return _H_CACHE[key]


def null_alarm_rate(cfg: ChangeConfig, steps: int) -> float:
    """Probability that one unchanged stream alarms within `steps` steps, from the calibrated per-horizon rate (chance level for
    the scope test: how many alarms would appear with nothing wrong)."""
    return float(1.0 - (1.0 - cfg.alpha) ** (steps / cfg.horizon))


def _day(i: int) -> str:
    return (dt.date(2000, 1, 3) + dt.timedelta(days=i)).isoformat()


@dataclasses.dataclass(frozen=True)
class LatencyPoint:
    shift: float
    detect_prob: float
    median_delay: float | None
    p90_delay: float | None


def latency_curve(cfg: ChangeConfig, shifts: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0), max_steps: int = 120, n_sim: int = 200,
                  seed: int = 0) -> list[LatencyPoint]:
    """How early is early? For a step of `shift` reference sigmas at a random time, the probability of an alarm within max_steps
    and the delay distribution - the honest statement of the price of the false-alarm guarantee, by simulation."""
    rng = np.random.default_rng(seed)
    out = []
    for sh in shifts:
        delays = []
        for _ in range(n_sim):
            det = StreamDetector("sim", Target.VOLATILITY, cfg)
            for i in range(cfg.warmup + 10):
                det.update(_day(i), float(rng.standard_normal()))
            for i in range(max_steps):
                if det.update(_day(cfg.warmup + 10 + i), float(rng.standard_normal() + sh)) is not None:
                    delays.append(i + 1)
                    break
        out.append(LatencyPoint(float(sh), len(delays) / n_sim, float(np.median(delays)) if delays else None,
                                float(np.quantile(delays, 0.9)) if delays else None))
    return out


# ------------------------------------------------------------------------------------------------ detections and the detector

@dataclasses.dataclass(frozen=True)
class Detection:
    """A declared change and exactly what it knew when it was declared."""
    stream: str
    target: Target
    unit: str | None
    sector: str | None
    alarm_index: int
    alarm_date: str
    change_index: int                            # estimated start of the new level, from values up to alarm_index only
    change_date: str
    direction: int                               # +1 the stream moved up, -1 down
    stat: float
    threshold: float
    old_center: float
    old_scale: float
    information_through: str                     # the last date whose value the decision could have used (== alarm_date)

    @property
    def latency(self) -> int:
        return self.alarm_index - self.change_index

    def validate(self) -> list[str]:
        e = []
        if self.change_index > self.alarm_index or as_date(self.change_date) > as_date(self.alarm_date):
            e.append("change located after its own alarm")
        if self.information_through != self.alarm_date:
            e.append("information cut-off is not the alarm date")
        if self.direction not in (-1, 1):
            e.append("direction must be +-1")
        return e


_WARMUP, _MONITOR, _SETTLE = "warmup", "monitor", "settle"


class StreamDetector:
    """Streaming two-sided CUSUM on one stream. update(date, value) sees values in date order and nothing else; its whole state is
    plain data (state()/from_state()), so a checkpoint restores it exactly."""

    def __init__(self, stream: str, target: Target, cfg: ChangeConfig | None = None, unit: str | None = None, sector: str | None = None):
        self.stream, self.target, self.unit, self.sector = stream, target, unit, sector
        self.cfg = cfg or ChangeConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("bad ChangeConfig: " + "; ".join(errs))
        self.h = calibrate_h(self.cfg)
        self.phase = _WARMUP
        self.n_seen = 0
        self.n_missing = 0
        self.warm: list[float] = []
        self.settle_buf: list[float] = []
        self.center = 0.0
        self.scale = 1.0
        self.hi = 0.0
        self.lo = 0.0
        self.hi_start = 0
        self.lo_start = 0
        self.monitor_start = 0
        self.zbuf: deque[float] = deque(maxlen=self.cfg.scan_window)
        self.dates: deque[str] = deque(maxlen=max(self.cfg.scan_window, 8))
        self.last_date: str | None = None
        self.n_alarms = 0

    def update(self, date, value) -> Detection | None:
        d = as_date(date).isoformat()
        if self.last_date is not None and d <= self.last_date:
            raise FirewallBreach(f"{self.stream}: value dated {d} is not after {self.last_date}")
        idx = self.n_seen
        self.n_seen += 1
        self.last_date = d
        self.dates.append(d)
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not math.isfinite(v):
            self.n_missing += 1
            self.zbuf.append(0.0)                # a missing day carries no evidence; the buffer stays aligned to dates
            return None
        if self.phase == _WARMUP:
            self.warm.append(v)
            self.zbuf.append(0.0)
            if len(self.warm) >= self.cfg.warmup:
                self.center, self.scale = (float(a) for a in reference_of(np.array(self.warm)))
                self.phase, self.monitor_start = _MONITOR, idx + 1
            return None
        if self.phase == _SETTLE:
            self.settle_buf.append(v)
            self.zbuf.append(0.0)
            if len(self.settle_buf) >= self.cfg.settle:
                self.center, self.scale = (float(a) for a in reference_of(np.array(self.settle_buf)))
                self.phase, self.hi, self.lo, self.monitor_start = _MONITOR, 0.0, 0.0, idx + 1
                self.zbuf.clear()
                self.settle_buf = []
            return None
        z = float(np.clip((v - self.center) / self.scale, -self.cfg.clip, self.cfg.clip))
        self.zbuf.append(z)
        nhi, nlo = max(0.0, self.hi + z - self.cfg.k), min(0.0, self.lo + z + self.cfg.k)
        if self.hi == 0.0 and nhi > 0.0:
            self.hi_start = idx
        if self.lo == 0.0 and nlo < 0.0:
            self.lo_start = idx
        self.hi, self.lo = nhi, nlo
        if max(self.hi, -self.lo) <= self.h:
            return None
        direction = 1 if self.hi >= -self.lo else -1
        stat = self.hi if direction == 1 else -self.lo
        start = max(self.hi_start if direction == 1 else self.lo_start, self.monitor_start)
        ci = self._locate(idx, start)
        det = Detection(self.stream, self.target, self.unit, self.sector, idx, d, ci, self.dates[-1 - (idx - ci)] if idx - ci < len(self.dates) else d,
                        direction, float(stat), self.h, self.center, self.scale, d)
        self.n_alarms += 1
        self.phase, self.settle_buf, self.hi, self.lo = _SETTLE, [], 0.0, 0.0
        return det

    def _locate(self, idx: int, cusum_start: int) -> int:
        """Where did the new level begin? The CUSUM run start, refined by the mean-shift scan of the buffered values. Both use
        only values already seen; the answer is clamped to [monitor start, now]."""
        n = min(idx - self.monitor_start + 1, len(self.zbuf))
        z = np.array(list(self.zbuf)[-n:], dtype="float64")
        first = idx - n + 1
        _, k = change_scan(z, threshold=self.cfg.scan_threshold)
        est = first + k if k is not None else cusum_start
        return int(min(max(est, self.monitor_start, first), idx))

    def state(self) -> dict[str, Any]:
        return {"phase": self.phase, "n_seen": self.n_seen, "n_missing": self.n_missing, "warm": list(self.warm),
                "settle": list(self.settle_buf), "center": self.center, "scale": self.scale, "hi": self.hi, "lo": self.lo,
                "hi_start": self.hi_start, "lo_start": self.lo_start, "monitor_start": self.monitor_start, "zbuf": list(self.zbuf),
                "dates": list(self.dates), "last_date": self.last_date, "n_alarms": self.n_alarms, "h": self.h}

    @classmethod
    def from_state(cls, stream: str, target: Target, cfg: ChangeConfig, st: Mapping[str, Any], unit=None, sector=None) -> "StreamDetector":
        d = cls(stream, target, cfg, unit, sector)
        d.phase, d.n_seen, d.n_missing = st["phase"], st["n_seen"], st["n_missing"]
        d.warm, d.settle_buf = list(st["warm"]), list(st["settle"])
        d.center, d.scale, d.hi, d.lo = st["center"], st["scale"], st["hi"], st["lo"]
        d.hi_start, d.lo_start, d.monitor_start = st["hi_start"], st["lo_start"], st["monitor_start"]
        d.zbuf.extend(st["zbuf"])
        d.dates.extend(st["dates"])
        d.last_date, d.n_alarms = st["last_date"], st["n_alarms"]
        return d

    def digest(self) -> str:
        return exact_digest(self.state())


class ChangeMonitor:
    """A set of stream detectors advanced together, one date at a time. The digest after step t depends on values up to t only."""

    def __init__(self, cfg: ChangeConfig | None = None):
        self.cfg = cfg or ChangeConfig()
        self.detectors: dict[str, StreamDetector] = {}
        self.detections: list[Detection] = []
        self.last_date: str | None = None
        self.n_steps = 0

    def register(self, stream: str, target: Target, unit: str | None = None, sector: str | None = None) -> StreamDetector:
        if stream in self.detectors:
            return self.detectors[stream]
        d = StreamDetector(stream, target, self.cfg, unit, sector)
        d.n_seen = self.n_steps                      # a stream that appears late is aligned to the monitor's clock
        self.detectors[stream] = d
        return d

    def update(self, date, values: Mapping[str, float | None], now=None, targets: Mapping[str, Target] | None = None,
               units: Mapping[str, tuple[str | None, str | None]] | None = None) -> list[Detection]:
        d = as_date(date).isoformat()
        if now is not None and as_date(d) > as_date(now):
            raise FirewallBreach(f"step {d} is after now={as_date(now)}")
        if self.last_date is not None and d <= self.last_date:
            raise FirewallBreach(f"step {d} is not after the last step {self.last_date}")
        for s in values:
            if s not in self.detectors:
                if targets is None or s not in targets:
                    raise ValueError(f"stream {s!r} is not registered and no target was given")
                unit, sector = (units or {}).get(s, (None, None))
                self.register(s, targets[s], unit, sector)
        out = []
        for s, det in self.detectors.items():
            hit = det.update(d, values.get(s))
            if hit is not None:
                out.append(hit)
        self.n_steps += 1
        self.last_date = d
        self.detections.extend(out)
        return out

    def digest(self) -> str:
        return exact_digest({"streams": {s: d.state() for s, d in self.detectors.items()}, "last": self.last_date, "n": self.n_steps,
                             "det": [dataclasses.asdict(x) for x in self.detections]})

    def recent(self, window: int | None = None) -> list[Detection]:
        w = window or self.cfg.scope_window
        return [d for d in self.detections if d.alarm_index >= self.n_steps - w]

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "cfg": dataclasses.asdict(self.cfg), "last_date": self.last_date, "n_steps": self.n_steps,
                "streams": {s: {"target": d.target.value, "unit": d.unit, "sector": d.sector, "st": d.state()} for s, d in self.detectors.items()},
                "detections": [dataclasses.asdict(x) for x in self.detections]}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "ChangeMonitor":
        if st.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"monitor schema {st.get('schema')!r} != {SCHEMA_VERSION}")
        m = cls(ChangeConfig(**st["cfg"]))
        m.last_date, m.n_steps = st["last_date"], st["n_steps"]
        for s, e in st["streams"].items():
            m.detectors[s] = StreamDetector.from_state(s, Target(e["target"]), m.cfg, e["st"], e["unit"], e["sector"])
        m.detections = [Detection(**{**x, "target": Target(x["target"])}) for x in st["detections"]]
        return m


# ------------------------------------------------------------------------------------------------ forward-only running and the proof

@dataclasses.dataclass(frozen=True)
class ForwardRun:
    dates: tuple[str, ...]
    digests: tuple[str, ...]                     # monitor digest after each step
    detections: tuple[Detection, ...]
    monitor: ChangeMonitor

    def decisions_through(self, t: int) -> list[tuple[str, int, int]]:
        return [(d.stream, d.alarm_index, d.direction) for d in self.detections if d.alarm_index <= t]


def run_forward(dates: Sequence[Any], streams: Mapping[str, Sequence[float]], targets: Mapping[str, Target],
                cfg: ChangeConfig | None = None, upto: int | None = None) -> ForwardRun:
    """Feed the streams through a fresh monitor one date at a time, exactly as history would have delivered them. `upto` stops the
    run early (inclusive index); values after it are never touched."""
    n = len(dates) if upto is None else min(len(dates), upto + 1)
    for s, v in streams.items():
        if len(v) < n:
            raise ValueError(f"stream {s} shorter than the dates")
    mon = ChangeMonitor(cfg)
    for s, tg in targets.items():
        mon.register(s, tg)
    digests = []
    for i in range(n):
        mon.update(dates[i], {s: streams[s][i] for s in streams}, now=dates[i])
        digests.append(mon.digest())
    return ForwardRun(tuple(as_date(d).isoformat() for d in dates[:n]), tuple(digests), tuple(mon.detections), mon)


SCRAMBLES = ("permute", "level_shift", "huge_noise", "reverse", "nan")


def scramble_after(streams: Mapping[str, Sequence[float]], t: int, mode: str, seed: int = 0) -> dict[str, np.ndarray]:
    """Copy of the streams with every value AFTER index t replaced: shuffled, level-shifted by 50 sigma, replaced by huge noise,
    reversed, or blanked. Values up to and including t are untouched."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, v in streams.items():
        a = np.array(v, dtype="float64")
        tail = a[t + 1:].copy()
        sd = float(np.nanstd(a[: t + 1])) or 1.0
        if mode == "permute":
            tail = rng.permutation(tail)
        elif mode == "level_shift":
            tail = tail + 50 * sd
        elif mode == "huge_noise":
            tail = rng.standard_normal(len(tail)) * 1e6
        elif mode == "reverse":
            tail = tail[::-1]
        elif mode == "nan":
            tail = np.full(len(tail), np.nan)
        else:
            raise ValueError(f"unknown scramble {mode}")
        a[t + 1:] = tail
        out[s] = a
    return out


def check_no_lookahead(dates: Sequence[Any], streams: Mapping[str, Sequence[float]], targets: Mapping[str, Target],
                       cfg: ChangeConfig | None = None, t: int | None = None, seed: int = 0, modes: Sequence[str] = SCRAMBLES,
                       runner=None) -> list[str]:
    """Checklist X as a runnable proof. The reference run is over the real data; each variant scrambles everything after t. The
    monitor's exact state digest after every step <= t and every decision with alarm_index <= t must be identical. Returns the
    violations (empty = the detector's state at t depends on nothing after t)."""
    t = len(dates) // 2 if t is None else t
    run = runner or run_forward
    base = run(dates, streams, targets, cfg)
    bad = []
    for mode in modes:
        alt = run(dates, scramble_after(streams, t, mode, seed), targets, cfg)
        for i in range(t + 1):
            if base.digests[i] != alt.digests[i]:
                bad.append(f"{mode}: state digest differs at step {i} (<= t={t})")
                break
        if base.decisions_through(t) != alt.decisions_through(t):
            bad.append(f"{mode}: decisions up to t differ")
    return bad


def detectable_at(detections: Sequence[Detection], stream: str, index: int) -> bool:
    """Could this stream's change have been known by step `index`? Only if the detector had ALREADY alarmed by then."""
    return any(d.stream == stream and d.alarm_index <= index for d in detections)


def earlier_claim_valid(det: Detection, claimed_index: int) -> bool:
    """A change may be called 'detectable at claimed_index' only if the alarm itself came at or before it. A claim built from where
    the change is now known to have started is retrospective and is refused."""
    return det.alarm_index <= claimed_index and not det.validate()


@dataclasses.dataclass(frozen=True)
class DetectionScore:
    stream: str
    planted: int | None
    detected: bool
    latency: int | None                          # alarm_index - planted change index
    false_alarms: int
    early_location_error: int | None             # |estimated change - planted change|


def evaluate_detections(detections: Sequence[Detection], planted: Mapping[str, int | None], slack: int = 3) -> list[DetectionScore]:
    """Score alarms against changes we planted (a synthetic world). An alarm before planted - slack, or any alarm on a stream with
    no planted change, is a false alarm; the first alarm at/after the plant is the detection. Latency is measured to the ALARM."""
    out = []
    for s, p in planted.items():
        mine = sorted((d for d in detections if d.stream == s), key=lambda d: d.alarm_index)
        if p is None:
            out.append(DetectionScore(s, None, False, None, len(mine), None))
            continue
        fa = [d for d in mine if d.alarm_index < p - slack]
        hit = next((d for d in mine if d.alarm_index >= p - slack), None)
        out.append(DetectionScore(s, p, hit is not None, None if hit is None else hit.alarm_index - p, len(fa),
                                  None if hit is None else abs(hit.change_index - p)))
    return out


# ------------------------------------------------------------------------------------------------ the thirteen relationships

def _rank(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, method="average")


def daily_ic(x: np.ndarray, y: np.ndarray, min_n: int = 12) -> float:
    """Rank correlation of a feature with the outcome across one day's cross-section (feature -> outcome relationship)."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x[ok]), _rank(y[ok]))[0, 1])


def pattern_t(y: np.ndarray, fired: np.ndarray, min_n: int = 5) -> float:
    """Welch t of the outcome on names where a pattern fired against names where it did not, for one day (pattern -> outcome)."""
    y, f = np.asarray(y, dtype="float64"), np.asarray(fired, dtype=bool)
    ok = np.isfinite(y)
    a, b = y[ok & f], y[ok & ~f]
    if len(a) < min_n or len(b) < min_n:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def mean_pair_correlation(returns: np.ndarray, min_periods: int = 10) -> float:
    """Average off-diagonal correlation of a names x periods return window (correlation structure)."""
    r = np.asarray(returns, dtype="float64")
    if r.ndim != 2 or r.shape[0] < 3 or r.shape[1] < min_periods or not np.isfinite(r).all():
        return float("nan")
    sd = r.std(axis=1)
    keep = sd > 1e-12
    if keep.sum() < 3:
        return float("nan")
    c = np.corrcoef(r[keep])
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def calibration_residual(p: np.ndarray, y: np.ndarray) -> float:
    """Mean standardised confidence residual of one day's binary predictions: > 0 overconfident when p is the stated probability of
    the predicted side and y the hit (confidence calibration)."""
    p, y = np.asarray(p, dtype="float64"), np.asarray(y, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    pc = np.clip(p[ok], 0.02, 0.98)
    return float(np.mean((pc - y[ok]) / np.sqrt(pc * (1 - pc))))


@dataclasses.dataclass(frozen=True)
class DayInputs:
    """One day's raw material for the thirteen streams. Every field is optional; a stream is emitted only when its inputs exist
    (an absent input is a missing stream, never a zero)."""
    outcome: np.ndarray | None = None                          # per-name forward outcome
    features: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
    patterns: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)      # name -> boolean fired mask
    sector: np.ndarray | None = None                           # per-name sector code
    returns_window: np.ndarray | None = None                   # names x periods, for correlations
    day_abs_move: float | None = None                          # cross-sectional mean |return| (volatility behaviour)
    dispersion: float | None = None
    breadth_up: float | None = None
    direction_hits: int | None = None
    direction_n: int | None = None
    past_long: np.ndarray | None = None                        # long-horizon past return per name (momentum)
    past_short: np.ndarray | None = None                       # short-horizon past return per name (reversal)
    holding_days: float | None = None
    optimal_exit_days: float | None = None
    confidence_p: np.ndarray | None = None
    confidence_hit: np.ndarray | None = None
    error_z: np.ndarray | None = None                          # standardised prediction errors that matured today
    confident_failures: int | None = None
    confident_n: int | None = None


def build_streams(day: DayInputs) -> tuple[dict[str, float | None], dict[str, Target], dict[str, tuple[str | None, str | None]]]:
    """(values, targets, units) for one day: a stream per feature, pattern and sector plus each aggregate relationship. Units give
    (unit, sector) for streams that describe a single group, so the scope classifier can count them."""
    v: dict[str, float | None] = {}
    tg: dict[str, Target] = {}
    un: dict[str, tuple[str | None, str | None]] = {}

    def put(name: str, value: float | None, target: Target, unit=None, sector=None):
        v[name], tg[name], un[name] = (None if value is None or not math.isfinite(value) else float(value)), target, (unit, sector)
    if day.outcome is not None:
        y = np.asarray(day.outcome, dtype="float64")
        for n, x in day.features.items():
            put(f"feature:{n}", daily_ic(x, y), Target.FEATURE_OUTCOME)
        for n, f in day.patterns.items():
            put(f"pattern:{n}", pattern_t(y, f), Target.PATTERN_OUTCOME)
        if day.sector is not None:
            sec = np.asarray(day.sector)
            mu, sd = np.nanmean(y), np.nanstd(y)
            for c in np.unique(sec[sec >= 0]) if sec.dtype.kind in "iuf" else np.unique(sec):
                m = sec == c
                if m.sum() >= 5 and sd > 0:
                    put(f"sector:{c}", (np.nanmean(y[m]) - mu) / (sd / math.sqrt(m.sum())), Target.SECTOR, str(c), str(c))
        if day.past_long is not None:
            put("momentum", daily_ic(day.past_long, y), Target.MOMENTUM)
        if day.past_short is not None:
            put("reversal", daily_ic(-np.asarray(day.past_short, dtype="float64"), y), Target.REVERSAL)
    if day.returns_window is not None:
        put("correlation", mean_pair_correlation(day.returns_window), Target.CORRELATION)
    if day.day_abs_move is not None and day.day_abs_move > 0:
        put("volatility", math.log(day.day_abs_move), Target.VOLATILITY)
    if day.dispersion is not None and day.dispersion > 0:
        put("dispersion", math.log(day.dispersion), Target.DISPERSION)
    if day.breadth_up is not None:
        put("breadth", day.breadth_up, Target.BREADTH)
    if day.direction_n:
        put("direction", (day.direction_hits - day.direction_n / 2) / math.sqrt(day.direction_n / 4), Target.DIRECTION)
    if day.holding_days is not None:
        put("holding_period", day.holding_days, Target.HOLDING_PERIOD)
    if day.optimal_exit_days is not None:
        put("optimal_exit", day.optimal_exit_days, Target.OPTIMAL_EXIT)
    if day.confidence_p is not None and day.confidence_hit is not None:
        put("calibration", calibration_residual(day.confidence_p, day.confidence_hit), Target.CALIBRATION)
    if day.error_z is not None and len(day.error_z):
        z = np.asarray(day.error_z, dtype="float64")
        z = z[np.isfinite(z)]
        if len(z):
            put("error_level", float(z.mean()), Target.ERROR_DISTRIBUTION)
            put("error_spread", float(np.log(np.abs(z) + 0.1).mean()), Target.ERROR_DISTRIBUTION)
    if day.confident_n:
        put("confident_failures", day.confident_failures / day.confident_n, Target.CONFIDENT_FAILURES)
    return v, tg, un


def coverage_of_checklist_i(targets: Mapping[str, Target]) -> dict[str, bool]:
    """Which checklist-I relationships have at least one stream watching them (a missing relationship is a gap to report)."""
    seen = set(targets.values())
    return {t.value: t in seen for t in CHECKLIST_I}


# ------------------------------------------------------------------------------------------------ scope: stock, sector or market?

class Scope(_StrEnum):
    NONE = "NONE"
    SINGLE_STOCK = "SINGLE_STOCK_ANOMALY"
    SECTOR = "SECTOR_CHANGE"
    MARKET_WIDE = "MARKET_WIDE_REGIME_CHANGE"
    UNCLEAR = "UNCLEAR"


@dataclasses.dataclass(frozen=True)
class ScopeVerdict:
    scope: Scope
    n_units: int
    n_alarmed: int
    sectors: tuple[str, ...]
    units: tuple[str, ...]
    p_overall: float | None
    market_alarms: int
    reason: str


def _binom_sf(k: int, n: int, p: float) -> float:
    return float(sps.binom.sf(k - 1, n, min(max(p, 1e-12), 1 - 1e-12))) if n > 0 else 1.0


def classify_scope(alarmed: Mapping[str, str | None], universe: Mapping[str, str | None], null_rate: float, cfg: ChangeConfig,
                   market_alarms: int = 0, market_streams: int = 0) -> ScopeVerdict:
    """Single-stock anomaly, sector change or market-wide regime change? `alarmed` are the units (unit -> sector) whose streams alarmed
    inside the scope window, `universe` all watched units. Each level is judged against the alarms chance alone would produce
    (`null_rate` per unit): sectors by a binomial test, BH-corrected across sectors; the market by the overall count and by how many
    sectors are individually above chance, with market-level streams (`market_alarms` of `market_streams`) as separate evidence.
    A lone alarm is never promoted to a regime; unresolved patterns are UNCLEAR, not forced into a class."""
    n, k = len(universe), len(alarmed)
    if k == 0 and market_alarms == 0:
        return ScopeVerdict(Scope.NONE, n, 0, (), (), None, 0, "no alarms")
    p_all = _binom_sf(k, n, null_rate) if n else 1.0
    by_sector: dict[str, list[str]] = {}
    for u, s in universe.items():
        by_sector.setdefault(str(s), []).append(u)
    ps, names = [], []
    for s, members in by_sector.items():
        if len(members) >= cfg.min_sector_units:
            ka = sum(1 for u in members if u in alarmed)
            ps.append(_binom_sf(ka, len(members), null_rate))
            names.append(s)
    qs = benjamini_hochberg(ps)
    hot = tuple(s for s, q in zip(names, qs) if q is not None and q <= cfg.scope_alpha)
    mk_p = _binom_sf(market_alarms, market_streams, null_rate) if market_streams else 1.0
    market_sig = market_streams >= 3 and market_alarms >= 2 and mk_p < cfg.scope_alpha
    share = k / n if n else 0.0
    units = tuple(sorted(alarmed))
    if market_sig and (len(hot) >= 1 or p_all < cfg.scope_alpha or k == 0):
        return ScopeVerdict(Scope.MARKET_WIDE, n, k, hot, units, p_all, market_alarms, f"{market_alarms}/{market_streams} market streams alarmed above chance")
    if len(hot) >= cfg.min_market_sectors and p_all < cfg.scope_alpha:
        return ScopeVerdict(Scope.MARKET_WIDE, n, k, hot, units, p_all, market_alarms, f"{len(hot)} sectors alarmed above chance")
    if len(hot) == 1:
        rest = [u for u in alarmed if str(alarmed[u]) not in hot]
        rest_n = sum(1 for u, s in universe.items() if str(s) not in hot)
        if _binom_sf(len(rest), rest_n, null_rate) >= cfg.scope_alpha:
            return ScopeVerdict(Scope.SECTOR, n, k, hot, units, p_all, market_alarms, f"sector {hot[0]} above chance, the rest at chance")
        return ScopeVerdict(Scope.MARKET_WIDE, n, k, hot, units, p_all, market_alarms, "one hot sector and alarms elsewhere above chance")
    if p_all < cfg.scope_alpha:
        if share >= cfg.min_market_share:
            return ScopeVerdict(Scope.MARKET_WIDE, n, k, hot, units, p_all, market_alarms, f"{share:.0%} of units alarmed, no single sector responsible")
        return ScopeVerdict(Scope.UNCLEAR, n, k, hot, units, p_all, market_alarms, "more alarms than chance but neither concentrated nor broad")
    if 0 < k <= cfg.max_single:
        return ScopeVerdict(Scope.SINGLE_STOCK, n, k, hot, units, p_all, market_alarms, "isolated alarm(s), no more than chance would give")
    return ScopeVerdict(Scope.UNCLEAR, n, k, hot, units, p_all, market_alarms, "alarm count fits no single level")


# ------------------------------------------------------------------------------------------------ early warning (checklist R)

class Level(_StrEnum):
    NONE = "NONE"
    WATCH = "WATCH"
    WARNING = "WARNING"
    ALERT = "ALERT"


@dataclasses.dataclass(frozen=True)
class EarlyWarning:
    date: str
    level: Level
    scope: ScopeVerdict
    precursors: Mapping[str, int]                # checklist-R precursor -> number of alarms inside the window
    pattern_degrading: int
    pattern_strengthening: int
    regime_pressure: tuple[str, ...]             # precursors raised by the regime layer, marked separately as second-hand evidence
    detections: tuple[Detection, ...]

    def fired(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, n in self.precursors.items() if n > 0))


def regime_precursors(monitor: RG.RegimeMonitor) -> tuple[str, ...]:
    """Precursors implied by the regime layer's current early warnings (engine.research.regimes.all_warnings, past-only)."""
    m = {"volatility": "volatility_change", "dispersion": "abnormal_dispersion", "direction": "breadth_change",
         "character": "prediction_error_shift", "liquidity": "correlation_change", "events": "abnormal_dispersion"}
    return tuple(sorted({m[w.axis] for w in RG.all_warnings(monitor) if w.axis in m}))


def streams_from_regime_monitor(monitor: RG.RegimeMonitor) -> tuple[list[str], dict[str, list[float]], dict[str, Target]]:
    """The regime monitor's stored indicator history as change-detection streams (dates, series, targets). Only stored past values
    are read; a missing indicator day stays NaN."""
    tg = {"vol": Target.VOLATILITY, "dispersion": Target.DISPERSION, "persistence": Target.MOMENTUM, "trend200": Target.DIRECTION,
          "log_dv": Target.CORRELATION, "event_share": Target.BREADTH}
    dates = [d for d, _ in monitor.history.days]
    return dates, {n: [float("nan") if v.get(n) is None else float(v[n]) for _, v in monitor.history.days] for n in tg}, tg


class EarlyWarningSystem:
    """Market-level streams, sector streams and single-name streams watched together. step() advances everything by one date and
    returns the EarlyWarning: which precursors fired, at what scope, and how loud."""

    def __init__(self, cfg: ChangeConfig | None = None):
        self.cfg = cfg or ChangeConfig()
        self.market = ChangeMonitor(self.cfg)
        self.units = ChangeMonitor(self.cfg)
        self.unit_sector: dict[str, str | None] = {}
        self.history: list[EarlyWarning] = []

    def step(self, now, market_values: Mapping[str, float | None], market_targets: Mapping[str, Target],
             unit_values: Mapping[str, float | None] | None = None, unit_sector: Mapping[str, str | None] | None = None,
             unit_target: Target = Target.PATTERN_OUTCOME, regime_monitor: RG.RegimeMonitor | None = None) -> EarlyWarning:
        self.market.update(now, market_values, now, market_targets)
        if unit_values:
            self.unit_sector.update(unit_sector or {})
            self.units.update(now, unit_values, now, {u: unit_target for u in unit_values},
                              {u: (u, self.unit_sector.get(u)) for u in unit_values})
        win = self.cfg.scope_window
        recent_m, recent_u = self.market.recent(win), self.units.recent(win)
        counts = {p: 0 for p in PRECURSORS}
        for d in recent_m:
            if d.target in PRECURSOR_OF:
                counts[PRECURSOR_OF[d.target]] += 1
        deg = sum(1 for d in recent_m + recent_u if d.target == Target.PATTERN_OUTCOME and d.direction < 0)
        stren = sum(1 for d in recent_m + recent_u if d.target == Target.PATTERN_OUTCOME and d.direction > 0)
        n_pat = sum(1 for det in list(self.market.detectors.values()) + list(self.units.detectors.values()) if det.target == Target.PATTERN_OUTCOME)
        p0 = null_alarm_rate(self.cfg, win)
        if n_pat and _binom_sf(deg, n_pat, p0 / 2) < self.cfg.scope_alpha:
            counts["pattern_degradation"] = deg
        if n_pat and _binom_sf(stren, n_pat, p0 / 2) < self.cfg.scope_alpha:
            counts["pattern_strengthening"] = stren
        alarmed = {d.unit: d.sector for d in recent_u if d.unit is not None}
        scope = classify_scope(alarmed, self.unit_sector if self.unit_sector else {u: None for u in self.units.detectors}, p0, self.cfg,
                               len({d.stream for d in recent_m}), len(self.market.detectors))
        reg = regime_precursors(regime_monitor) if regime_monitor is not None else ()
        n_kinds = len([k for k, n in counts.items() if n > 0])
        if n_kinds == 0 and not reg and scope.scope in (Scope.NONE, Scope.SINGLE_STOCK):
            level = Level.NONE
        elif scope.scope == Scope.MARKET_WIDE and n_kinds >= 2:
            level = Level.ALERT
        elif n_kinds >= 2 or scope.scope in (Scope.SECTOR, Scope.MARKET_WIDE):
            level = Level.WARNING
        else:
            level = Level.WATCH
        ew = EarlyWarning(as_date(now).isoformat(), level, scope, counts, deg, stren, reg, tuple(recent_m + recent_u))
        self.history.append(ew)
        return ew

    def digest(self) -> str:
        return exact_digest([self.market.digest(), self.units.digest(), [(e.date, e.level.value, e.scope.scope.value) for e in self.history]])


def step(system: EarlyWarningSystem, now, market_values: Mapping[str, float | None], market_targets: Mapping[str, Target], **kw) -> EarlyWarning:
    """The ONE public entry the research loop calls: advance every stream by one date and report the early warning."""
    return system.step(now, market_values, market_targets, **kw)


def summarize(detections: Sequence[Detection]) -> dict[str, dict[str, float]]:
    """Per-target count, mean latency and share of upward changes - the table the health report prints."""
    out: dict[str, dict[str, float]] = {}
    for t in {d.target for d in detections}:
        ds = [d for d in detections if d.target == t]
        out[t.value] = {"n": float(len(ds)), "mean_latency": float(np.mean([d.latency for d in ds])), "up_share": float(np.mean([d.direction > 0 for d in ds]))}
    return out
