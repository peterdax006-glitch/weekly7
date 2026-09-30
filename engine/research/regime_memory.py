"""Regime-change memory (PREDICTION_ERROR_ADDITION checklists W and X; canon C68, C66 sections 13/25/48, C58-C61, C63).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; no real-data run was made).

When simultaneous change-point alarms (engine.research.change_points) reach a quorum, the change is written down with every
checklist-W field: the old regime and the new one (quantity profiles, plus the regime-layer labels when a RegimeMonitor is given),
the earliest detectable evidence, the patterns affected / strengthened / weakened (Welch test of each pattern's own effect before vs
after, BH across patterns), the prediction errors that piled up before detection, the detection latency, the signals that could have
detected it earlier, the signals that misled (alarms that reverted, or pointed the other way), the conditions for a return to the old
regime (a band per quantity, held for several steps) and a confidence in the classification with its parts.

Memory is append-only and hash-chained: a status change (CANDIDATE -> CONFIRMED / FALSE_ALARM -> RETURNED) is a new event, never an
edit, and verify() finds any rewrite. From confirmed and false-alarmed records the memory learns which signals actually lead
(signal_reliability) and turns that into the sentence the checklist wants - 'when I start seeing X, the market is beginning to behave
differently and these patterns should be treated differently' (advice) - with the false-alarm history of the same signals attached
and UNKNOWN whenever fewer than min_support remembered changes match.

Checklist X: a record is built only from evidence dated at/before `now`; its detection facts (detected_at, earliest evidence,
latency) come from the detector's own alarms; anything learned later is stored in the separate `hindsight` field and never enters a
detection claim. A regime alarm alone never disables a pattern: pattern_guard() decides from the pattern's OWN post-change evidence,
so a false regime change leaves a still-working pattern untouched.

Firewall: records are MATURED_RESEARCH_STATE; they leave only through matured_records() as identity-free MaturedRecords (no dates,
units or tickers) and are released by MaturedRecord.gate(now).
Built on: engine.research.change_points (Detection, Scope, ScopeVerdict, PRECURSOR_OF, exact_digest), engine.research.regimes
(RegimeMonitor labels), engine.research.multiscale (benjamini_hochberg, t_to_p), engine.research.core, engine.learning.core."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np

from engine.learning.core import FirewallBreach, Provenance, _StrEnum, as_date, current_code_hash, stable_hash
from engine.research import regimes as RG
from engine.research.change_points import PRECURSOR_OF, Detection, Scope, ScopeVerdict, exact_digest
from engine.research.core import MaturedRecord, Namespace
from engine.research.expectations import SealedLane
from engine.research.multiscale import benjamini_hochberg, t_to_p

SCHEMA_VERSION = "regime_memory.v1"
_CODE_HASH: list[str] = []


def _code_hash() -> str:
    if not _CODE_HASH:
        _CODE_HASH.append(current_code_hash())
    return _CODE_HASH[0]


class RegimeStatus(_StrEnum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    FALSE_ALARM = "FALSE_ALARM"
    RETURNED = "RETURNED"                        # the old regime came back after a confirmed change


class PatternAction(_StrEnum):
    KEEP = "KEEP"
    REDUCE = "REDUCE"
    SUSPEND = "SUSPEND"


class SignalRole(_StrEnum):
    LEADING = "LEADING"                          # alarmed before the change was declared and belongs to the change
    CONCURRENT = "CONCURRENT"
    MISLEADING = "MISLEADING"


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    tol: int = 8                                 # steps: alarms whose estimated change lies this close belong to one change
    window: int = 25                             # steps after the first alarm during which more alarms join the cluster
    min_streams: int = 3                         # quorum: distinct streams alarming before a change is declared
    min_targets: int = 2                         # ... spanning this many kinds of relationship (or twice the quorum of units)
    profile_window: int = 60
    min_old: int = 20
    min_new: int = 5
    lookback: int = 40                           # steps before the declaration searched for misleading alarms
    revert_window: int = 8                       # the latest steps before the declaration that decide whether an alarm held
    revert_frac: float = 0.5                     # an alarm whose stream sits within this many old sigmas of its old centre later = reverted
    q_fdr: float = 0.10
    confirm_steps: int = 30                      # post-declaration steps needed to confirm or refute a candidate
    confirm_sigma: float = 2.0
    return_persist: int = 5
    min_support: int = 2
    min_overlap: float = 0.5
    guard_alpha: float = 0.05
    guard_reduce_alpha: float = 0.01
    min_after: int = 15

    def validate(self) -> list[str]:
        e = []
        if self.tol < 1 or self.window < self.tol or self.min_streams < 2 or self.min_targets < 1:
            e.append("tol >= 1, window >= tol, min_streams >= 2 and min_targets >= 1 required")
        if self.min_old < 10 or self.profile_window < self.min_old or self.min_new < 3:
            e.append("min_old >= 10, profile_window >= min_old and min_new >= 3 required")
        if not 0 < self.q_fdr < 1 or self.confirm_steps < 5:
            e.append("0 < q_fdr < 1 and confirm_steps >= 5 required")
        return e


@dataclasses.dataclass(frozen=True)
class QuantityProfile:
    name: str
    mean: float
    sd: float
    lo: float                                    # 10th percentile
    hi: float                                    # 90th percentile
    n: int


@dataclasses.dataclass(frozen=True)
class RegimeProfile:
    quantities: tuple[QuantityProfile, ...]
    labels: tuple[tuple[str, str], ...]          # regime-layer (axis, state) at the profile's end, when a monitor was supplied
    n: int
    partial: bool                                # fewer points than the window asked for

    def get(self, name: str) -> QuantityProfile | None:
        return next((q for q in self.quantities if q.name == name), None)


@dataclasses.dataclass(frozen=True)
class Signal:
    stream: str
    precursor: str
    target: str
    alarm_date: str
    alarm_step: int
    direction: int
    lead: int                                    # steps between this alarm and the declaration (positive = earlier)
    role: SignalRole


@dataclasses.dataclass(frozen=True)
class PatternImpact:
    pattern: str
    before: float
    after: float
    t: float
    q: float | None
    kind: str                                    # 'strengthened' | 'weakened' | 'unchanged'


@dataclasses.dataclass(frozen=True)
class ErrorsBefore:
    n: int
    mean_z: float | None
    mean_abs_z: float | None
    share_big: float | None                      # share of |z| >= 2.5 (the surprise investigation trigger)
    first_big_step: int | None


@dataclasses.dataclass(frozen=True)
class ReturnCondition:
    kind: str                                    # 'quantity' | 'pattern'
    name: str
    lo: float
    hi: float
    persist: int


@dataclasses.dataclass(frozen=True)
class RegimeChangeRecord:
    """Checklist W in one immutable object. Detection facts (detected_at .. misleading) come only from what the detectors knew;
    `hindsight` holds anything learned afterwards and is never read by a detection claim."""
    record_id: str
    scope: str
    detected_at: str
    change_date: str
    old_regime: RegimeProfile
    new_regime: RegimeProfile
    earliest_evidence: Signal
    detection_latency: int                       # steps from the estimated change to the declaration
    signals: tuple[Signal, ...]
    earlier_signals: tuple[Signal, ...]          # alarms available before the declaration
    misleading_signals: tuple[Signal, ...]
    patterns_affected: tuple[str, ...]
    patterns_strengthened: tuple[str, ...]
    patterns_weakened: tuple[str, ...]
    pattern_impacts: tuple[PatternImpact, ...]
    errors_before_detection: ErrorsBefore
    return_conditions: tuple[ReturnCondition, ...]
    confidence: float
    confidence_parts: tuple[tuple[str, float], ...]
    information_through: str
    hindsight: tuple[tuple[str, str], ...] = ()

    def validate(self) -> list[str]:
        e = []
        if as_date(self.change_date) > as_date(self.detected_at):
            e.append("change dated after its own detection")
        if self.information_through != self.detected_at:
            e.append("information cut-off is not the detection date")
        if any(as_date(s.alarm_date) > as_date(self.detected_at) for s in self.earlier_signals):
            e.append("an 'earlier' signal is dated after the declaration")
        if as_date(self.earliest_evidence.alarm_date) > as_date(self.detected_at):
            e.append("earliest evidence is dated after the declaration")
        if self.detection_latency < 0:
            e.append("negative detection latency")
        if not 0.0 <= self.confidence <= 1.0:
            e.append("confidence outside [0, 1]")
        if set(self.patterns_affected) != set(self.patterns_strengthened) | set(self.patterns_weakened):
            e.append("patterns_affected must be the union of strengthened and weakened")
        return e

    def precursors(self) -> tuple[str, ...]:
        return tuple(sorted({s.precursor for s in self.signals if s.role != SignalRole.MISLEADING}))


@dataclasses.dataclass(frozen=True)
class RegimeEvidence:
    """Everything known at `now`, aligned by position: dates[i] <-> quantities[q][i] <-> pattern_effects[p][i] <-> error_z[i]."""
    dates: Sequence[str]
    detections: Sequence[Detection]
    quantities: Mapping[str, Sequence[float]]
    pattern_effects: Mapping[str, Sequence[float]] = dataclasses.field(default_factory=dict)
    error_z: Sequence[float] = ()

    def validate(self, now) -> list[str]:
        e = []
        n = len(self.dates)
        if n and as_date(self.dates[-1]) > as_date(now):
            e.append("evidence contains dates after now")
        for name, s in list(self.quantities.items()) + list(self.pattern_effects.items()):
            if len(s) != n:
                e.append(f"{name}: length {len(s)} != {n} dates")
        if len(self.error_z) not in (0, n):
            e.append("error_z length != dates")
        if any(as_date(d.alarm_date) > as_date(now) for d in self.detections):
            e.append("a detection is dated after now")
        return e


# ------------------------------------------------------------------------------------------------ clustering and profiles

def cluster_detections(dets: Sequence[Detection], pos: Mapping[str, int], cfg: MemoryConfig) -> list[list[Detection]]:
    """Group alarms that describe ONE change: their estimated change steps lie within `tol` of the anchor's and they alarmed within
    `window` steps of it. A cluster stands only with a quorum of distinct streams across enough kinds of relationship (or twice the
    quorum of single-unit streams): one relationship moving is a relationship change, not a regime change."""
    ds = sorted((d for d in dets if d.alarm_date in pos and d.change_date in pos), key=lambda d: (pos[d.alarm_date], d.stream))
    used: set[int] = set()
    out = []
    for i, a in enumerate(ds):
        if i in used:
            continue
        mem = [j for j, d in enumerate(ds) if j not in used and abs(pos[d.change_date] - pos[a.change_date]) <= cfg.tol
               and 0 <= pos[d.alarm_date] - pos[a.alarm_date] <= cfg.window]
        used.update(mem)
        group = [ds[j] for j in mem]
        streams = {d.stream for d in group}
        kinds = {d.target for d in group}
        if len(streams) >= cfg.min_streams and (len(kinds) >= cfg.min_targets or len(streams) >= 2 * cfg.min_streams):
            out.append(group)
    return out


def _profile(evidence: RegimeEvidence, lo: int, hi: int, want: int, labels: tuple[tuple[str, str], ...] = ()) -> RegimeProfile:
    qs = []
    for name, s in evidence.quantities.items():
        v = np.asarray(s[lo:hi], dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            qs.append(QuantityProfile(name, float(v.mean()), float(v.std(ddof=1)), float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9)), len(v)))
    return RegimeProfile(tuple(qs), labels, max(hi - lo, 0), (hi - lo) < want)


def _labels_at(monitor: RG.RegimeMonitor | None, date: str) -> tuple[tuple[str, str], ...]:
    if monitor is None:
        return ()
    upto = [s for s in monitor.states if s.date <= date]
    return tuple(sorted(max(upto, key=lambda s: s.date).known().items())) if upto else ()


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float | None]:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), None
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se <= 1e-12:
        return float("nan"), None
    t = float((b.mean() - a.mean()) / se)
    return t, t_to_p(t)


def pattern_impacts(evidence: RegimeEvidence, cp: int, end: int, cfg: MemoryConfig) -> list[PatternImpact]:
    """Each pattern's own effect in the window before the change against the window since it (Welch, BH across patterns): which
    patterns strengthened, weakened, or did not move. Positive effect = the pattern works."""
    names, raw = [], []
    for p, s in evidence.pattern_effects.items():
        v = np.asarray(s, dtype="float64")
        a, b = v[max(cp - cfg.profile_window, 0):cp], v[cp:end]
        t, pv = _welch(a, b)
        names.append((p, float(np.nanmean(a)) if np.isfinite(a).any() else float("nan"), float(np.nanmean(b)) if np.isfinite(b).any() else float("nan"), t))
        raw.append(pv)
    q = benjamini_hochberg(raw)
    out = []
    for (p, before, after, t), qi in zip(names, q):
        kind = "unchanged" if qi is None or qi > cfg.q_fdr else ("strengthened" if after > before else "weakened")
        out.append(PatternImpact(p, before, after, t, qi, kind))
    return out


def errors_before(evidence: RegimeEvidence, cp: int, declared: int) -> ErrorsBefore:
    """What the undetected change cost: the standardised prediction errors between the estimated change and the declaration."""
    if not len(evidence.error_z):
        return ErrorsBefore(0, None, None, None, None)
    z = np.asarray(evidence.error_z[cp:declared + 1], dtype="float64")
    ok = np.isfinite(z)
    if not ok.any():
        return ErrorsBefore(0, None, None, None, None)
    big = np.flatnonzero(ok & (np.abs(np.where(ok, z, 0.0)) >= 2.5))
    return ErrorsBefore(int(ok.sum()), float(z[ok].mean()), float(np.abs(z[ok]).mean()), float((np.abs(z[ok]) >= 2.5).mean()),
                        int(cp + big[0]) if len(big) else None)


def _signal(d: Detection, declared: int, pos: Mapping[str, int], role: SignalRole) -> Signal:
    return Signal(d.stream, PRECURSOR_OF.get(d.target, d.target.value.lower()), d.target.value, d.alarm_date, pos[d.alarm_date], d.direction,
                  declared - pos[d.alarm_date], role)


def misleading_alarms(evidence: RegimeEvidence, cluster: Sequence[Detection], declared: int, pos: Mapping[str, int], cfg: MemoryConfig) -> list[Signal]:
    """Alarms in the lookback window that were NOT part of the change and did not hold up: the stream went back inside its old
    band by the declaration, or (same kind of relationship as the cluster) moved the other way. Uses only steps up to the
    declaration, so what counts as misleading is itself decided with the information the detector had."""
    members = {d.stream for d in cluster}
    kinds = {d.target: d.direction for d in cluster}
    out = []
    for d in evidence.detections:
        if d.stream in members or d.alarm_date not in pos or not (declared - cfg.lookback <= pos[d.alarm_date] <= declared):
            continue
        s = evidence.quantities.get(d.stream)
        reverted = False
        if s is not None:
            tail = np.asarray(s[max(pos[d.alarm_date], declared - cfg.revert_window): declared + 1], dtype="float64")
            tail = tail[np.isfinite(tail)]
            reverted = len(tail) >= 3 and abs(tail.mean() - d.old_center) < cfg.revert_frac * d.old_scale
        opposite = d.target in kinds and d.direction == -kinds[d.target]
        if reverted or opposite:
            out.append(_signal(d, declared, pos, SignalRole.MISLEADING))
    return out


def return_conditions(old: RegimeProfile, new: RegimeProfile, impacts: Sequence[PatternImpact], cfg: MemoryConfig) -> tuple[ReturnCondition, ...]:
    """When has the old regime come back? Each quantity that left its old 10-90% band must be back inside it for `return_persist`
    steps; each weakened pattern must have recovered to within one old spread of its former effect."""
    out = []
    for q in old.quantities:
        nq = new.get(q.name)
        if nq is not None and not (q.lo <= nq.mean <= q.hi):
            out.append(ReturnCondition("quantity", q.name, q.lo, q.hi, cfg.return_persist))
    for i in impacts:
        if i.kind == "weakened":
            out.append(ReturnCondition("pattern", i.pattern, i.before - abs(i.before) * 0.25, float("inf"), cfg.return_persist))
    return tuple(out)


def returned(rec: RegimeChangeRecord, recent: Mapping[str, Sequence[float]]) -> bool:
    """Have ALL of the record's return conditions held over the latest `persist` values of `recent` (name -> series through now)?
    A record with no conditions can never be declared returned (there is nothing to check)."""
    if not rec.return_conditions:
        return False
    for c in rec.return_conditions:
        v = np.asarray(recent.get(c.name, []), dtype="float64")[-c.persist:]
        if len(v) < c.persist or not np.isfinite(v).all() or not ((v >= c.lo) & (v <= c.hi)).all():
            return False
    return True


def confidence_of(cluster: Sequence[Detection], old: RegimeProfile, new: RegimeProfile, impacts: Sequence[PatternImpact],
                  scope: ScopeVerdict | None) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Confidence in 'this is a regime change', from four visible parts (breadth of relationships, size of the move in old sigmas,
    pattern corroboration, scope) shrunk by how little of the new regime has been seen. Every part is stored so the number can
    be audited, and the total never reaches 1."""
    kinds = len({d.target for d in cluster})
    breadth = min(1.0, kinds / 4.0)
    shifts = []
    for d in cluster:
        o, n = old.get(d.stream), new.get(d.stream)
        if o and n and o.sd > 1e-12:
            shifts.append(abs(n.mean - o.mean) / o.sd)
    size = min(1.0, (float(np.median(shifts)) if shifts else 0.0) / 3.0)
    corro = min(1.0, sum(1 for i in impacts if i.kind != "unchanged") / max(2.0, 0.3 * max(len(impacts), 1))) if impacts else 0.0
    scope_c = {Scope.MARKET_WIDE: 1.0, Scope.SECTOR: 0.6, Scope.UNCLEAR: 0.4, Scope.SINGLE_STOCK: 0.2, Scope.NONE: 0.3}[scope.scope] if scope else 0.5
    seen = new.n / (new.n + 20.0)
    parts = (("breadth", breadth), ("size", size), ("pattern_corroboration", corro), ("scope", scope_c), ("new_regime_seen", seen))
    return float(min(0.95, float(np.mean([breadth, size, corro, scope_c])) * seen)), parts


def build_record(evidence: RegimeEvidence, now, cfg: MemoryConfig | None = None, scope: ScopeVerdict | None = None,
                 monitor: RG.RegimeMonitor | None = None) -> list[RegimeChangeRecord]:
    """Every regime change the alarms support as of `now`: cluster, declare at the quorum alarm, then describe old vs new regime,
    affected patterns, errors before detection, signals (leading / concurrent / misleading), return conditions and confidence.
    Raises FirewallBreach if the evidence contains anything dated after `now`."""
    cfg = cfg or MemoryConfig()
    errs = cfg.validate()
    if errs:
        raise ValueError("bad MemoryConfig: " + "; ".join(errs))
    bad = evidence.validate(now)
    if bad:
        raise FirewallBreach("regime evidence rejected: " + "; ".join(bad))
    pos = {as_date(d).isoformat(): i for i, d in enumerate(evidence.dates)}
    end = len(evidence.dates)
    recs = []
    for group in cluster_detections(evidence.detections, pos, cfg):
        ordered = sorted(group, key=lambda d: (pos[d.alarm_date], d.stream))
        seen: set[str] = set()
        quorum = None
        for d in ordered:
            seen.add(d.stream)
            if len(seen) >= cfg.min_streams:
                quorum = d
                break
        if quorum is None:                      # cluster_detections only returns groups with enough independent streams
            continue
        declared, first = pos[quorum.alarm_date], ordered[0]
        cp = int(np.median([pos[d.change_date] for d in ordered if pos[d.alarm_date] <= declared]))
        cp = min(cp, declared)
        old = _profile(evidence, max(cp - cfg.profile_window, 0), cp, cfg.profile_window, _labels_at(monitor, evidence.dates[max(cp - 1, 0)]))
        new = _profile(evidence, cp, end, cfg.profile_window, _labels_at(monitor, evidence.dates[-1]))
        if old.n < cfg.min_old or new.n < cfg.min_new:
            continue
        impacts = pattern_impacts(evidence, cp, end, cfg)
        sigs = tuple(_signal(d, declared, pos, SignalRole.LEADING if pos[d.alarm_date] < declared else SignalRole.CONCURRENT) for d in ordered)
        mis = tuple(misleading_alarms(evidence, ordered, declared, pos, cfg))
        conf, parts = confidence_of(ordered, old, new, impacts, scope)
        strong = tuple(i.pattern for i in impacts if i.kind == "strengthened")
        weak = tuple(i.pattern for i in impacts if i.kind == "weakened")
        rec = RegimeChangeRecord(
            "R" + stable_hash([sorted(d.stream for d in ordered[: ordered.index(quorum) + 1]), evidence.dates[cp]], 12),
            scope.scope.value if scope else Scope.UNCLEAR.value,
            quorum.alarm_date, evidence.dates[cp], old, new, _signal(first, declared, pos, SignalRole.LEADING if pos[first.alarm_date] < declared else SignalRole.CONCURRENT),
            declared - cp, sigs, tuple(s for s in sigs if s.role == SignalRole.LEADING), mis, tuple(sorted(strong + weak)), strong, weak, tuple(impacts),
            errors_before(evidence, cp, declared), return_conditions(old, new, impacts, cfg), conf, parts, quorum.alarm_date)
        errs = rec.validate()
        if errs:
            raise FirewallBreach("built an invalid regime record: " + "; ".join(errs))
        recs.append(rec)
    return recs


# ------------------------------------------------------------------------------------------------ status and pattern guard

def resolve_status(rec: RegimeChangeRecord, quantities: Mapping[str, Sequence[float]], dates: Sequence[str], now, cfg: MemoryConfig | None = None) -> RegimeStatus:
    """CANDIDATE -> CONFIRMED / FALSE_ALARM once `confirm_steps` steps have passed since the declaration: the change is confirmed if
    a majority of the alarming streams are still `confirm_sigma` old sigmas from their old mean, a false alarm otherwise. Values
    dated after `now` are refused; too little post-declaration data leaves it a CANDIDATE."""
    cfg = cfg or MemoryConfig()
    if any(as_date(d) > as_date(now) for d in dates):
        raise FirewallBreach("status evidence dated after now")
    post = [i for i, d in enumerate(dates) if as_date(d) > as_date(rec.detected_at)]
    if len(post) < cfg.confirm_steps:
        return RegimeStatus.CANDIDATE
    holds, tested = 0, 0
    for s in rec.signals:
        o = rec.old_regime.get(s.stream)
        series = quantities.get(s.stream)
        if o is None or series is None or o.sd <= 1e-12:
            continue
        v = np.asarray([series[i] for i in post], dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) < cfg.confirm_steps // 2:
            continue
        tested += 1
        holds += int(abs(v.mean() - o.mean) >= cfg.confirm_sigma * o.sd / math.sqrt(max(1.0, len(v) ** 0.5)) and abs(v.mean() - o.mean) >= 0.5 * o.sd)
    if tested == 0:
        return RegimeStatus.CANDIDATE
    return RegimeStatus.CONFIRMED if holds > tested / 2 else RegimeStatus.FALSE_ALARM


@dataclasses.dataclass(frozen=True)
class GuardDecision:
    pattern: str
    action: PatternAction
    reason: str
    p_worse: float | None


def pattern_guard(pattern: str, status: RegimeStatus | None, before: Sequence[float], after: Sequence[float], advice_weakens: bool = False,
                  cfg: MemoryConfig | None = None) -> GuardDecision:
    """What to do with a pattern when a regime change is on the table. The pattern's OWN post-change evidence decides: a regime
    alarm - candidate, false or confirmed - never switches off a pattern that is still working. SUSPEND needs a confirmed change AND
    the pattern significantly worse (or reversed); on a candidate or false alarm the worst outcome is REDUCE, and only on strong
    evidence (guard_reduce_alpha). Too little post-change data on a confirmed change with memory advice to weaken: REDUCE, never
    SUSPEND. Otherwise KEEP."""
    cfg = cfg or MemoryConfig()
    b, a = np.asarray(before, dtype="float64"), np.asarray(after, dtype="float64")
    b, a = b[np.isfinite(b)], a[np.isfinite(a)]
    if len(a) < cfg.min_after or len(b) < cfg.min_after:
        if status == RegimeStatus.CONFIRMED and advice_weakens:
            return GuardDecision(pattern, PatternAction.REDUCE, "confirmed change, memory says this pattern weakens, own evidence still thin", None)
        return GuardDecision(pattern, PatternAction.KEEP, "too little post-change evidence to act on a regime alarm", None)
    t, _ = _welch(b, a)
    p_worse = float(0.5 * math.erfc(-t / math.sqrt(2.0))) if math.isfinite(t) else None      # one-sided: chance of a fall this large
    much_worse = a.mean() <= 0.5 * b.mean() if b.mean() > 0 else a.mean() < b.mean()
    reversed_ = b.mean() > 0 > a.mean()
    if p_worse is None:
        return GuardDecision(pattern, PatternAction.KEEP, "pattern evidence degenerate; no action on a regime alarm alone", None)
    if status == RegimeStatus.CONFIRMED and p_worse < cfg.guard_alpha and (much_worse or reversed_):
        return GuardDecision(pattern, PatternAction.SUSPEND, "confirmed change and the pattern's own effect fell significantly", p_worse)
    if p_worse < cfg.guard_reduce_alpha and much_worse:
        return GuardDecision(pattern, PatternAction.REDUCE, "the pattern's own effect fell sharply", p_worse)
    if status == RegimeStatus.CONFIRMED and advice_weakens and p_worse < 0.2:
        return GuardDecision(pattern, PatternAction.REDUCE, "confirmed change, memory advice and a hint of decline in its own evidence", p_worse)
    return GuardDecision(pattern, PatternAction.KEEP, "its own evidence is intact; a regime alarm alone does not switch a pattern off", p_worse)


# ------------------------------------------------------------------------------------------------ the memory

@dataclasses.dataclass(frozen=True)
class MemoryEvent:
    kind: str                                    # 'DETECTED' | 'STATUS'
    record_id: str
    at: str
    status: str
    payload_digest: str
    prev: str
    digest: str = ""

    def compute(self) -> str:
        return exact_digest([self.kind, self.record_id, self.at, self.status, self.payload_digest, self.prev])


@dataclasses.dataclass(frozen=True)
class SignalReliability:
    precursor: str
    n_leading: int
    n_misleading: int
    precision: float | None
    median_lead: float | None
    false_alarm_records: int


@dataclasses.dataclass(frozen=True)
class RegimeAdvice:
    known: bool
    support: int
    false_alarm_rate: float | None
    weaken: tuple[str, ...]
    strengthen: tuple[str, ...]
    confidence: float
    text: str


LANE_REGIME = "rgm68"


class RegimeMemory:
    """Append-only store of regime-change records whose status EVENTS live on an ARCHIVE CHAIN LANE (engine.research.expectations.
    SealedLane over engine.learning.archive.ChainFile, kind 'rgm68' - P06 de-duplication: no private hash chain). An event's `prev` is
    the lane head it extends; `verify` re-reads the lane and compares every cached event with the body the chain holds."""

    def __init__(self, cfg: MemoryConfig | None = None, root=None):
        self.cfg = cfg or MemoryConfig()
        self._records: dict[str, RegimeChangeRecord] = {}
        self._events: list[MemoryEvent] = []
        self.lane = SealedLane(root, LANE_REGIME)

    def __len__(self) -> int:
        return len(self._records)

    def _head(self) -> str:
        return self.lane.head

    def _push(self, kind: str, rid: str, at: str, status: str, payload_digest: str) -> MemoryEvent:
        ev = MemoryEvent(kind, rid, at, status, payload_digest, self._head())
        ev = dataclasses.replace(ev, digest=ev.compute())
        self.lane.append({"kind": kind, "record_id": rid, "at": at, "status": status, "payload_digest": payload_digest, "digest": ev.digest})
        self._events.append(ev)
        return ev

    def add(self, rec: RegimeChangeRecord, now) -> bool:
        """File a newly detected change. Refused if it carries information from at/after `now`; a record already known is left alone
        (returns False) - nothing is ever overwritten."""
        errs = rec.validate()
        if errs:
            raise ValueError("invalid record: " + "; ".join(errs))
        if as_date(rec.information_through) > as_date(now):
            raise FirewallBreach(f"record {rec.record_id} carries information from {rec.information_through}, after now={as_date(now)}")
        if rec.record_id in self._records:
            return False
        self._records[rec.record_id] = rec
        self._push("DETECTED", rec.record_id, rec.detected_at, RegimeStatus.CANDIDATE.value, exact_digest(rec))
        return True

    def set_status(self, rid: str, status: RegimeStatus, now) -> bool:
        """Record a status transition as a new event (CANDIDATE -> CONFIRMED / FALSE_ALARM -> RETURNED). A repeated status or a
        transition dated before the declaration is refused."""
        if rid not in self._records:
            raise KeyError(rid)
        if as_date(now) <= as_date(self._records[rid].detected_at):
            raise FirewallBreach("a status cannot be set at or before the declaration date")
        if self.status(rid) == status:
            return False
        if self.status(rid) in (RegimeStatus.FALSE_ALARM, RegimeStatus.RETURNED):
            raise ValueError(f"{rid} is already closed as {self.status(rid).value}")
        self._push("STATUS", rid, as_date(now).isoformat(), status.value, exact_digest([rid, status.value]))
        return True

    def status(self, rid: str) -> RegimeStatus:
        st = [e for e in self._events if e.record_id == rid]
        return RegimeStatus(st[-1].status) if st else RegimeStatus.CANDIDATE

    def get(self, rid: str) -> RegimeChangeRecord:
        return self._records[rid]

    def records(self, now=None, status: RegimeStatus | None = None) -> list[RegimeChangeRecord]:
        """Records declared strictly before `now` (None = all), optionally of one status, in declaration order."""
        out = [r for r in self._records.values() if (now is None or as_date(r.detected_at) < as_date(now)) and (status is None or self.status(r.record_id) == status)]
        return sorted(out, key=lambda r: (r.detected_at, r.record_id))

    def events(self) -> tuple[MemoryEvent, ...]:
        return tuple(self._events)

    def verify(self) -> list[int]:
        """Event indices whose content differs from its digest or from the body on the archive lane, or that do not extend the lane
        position they were appended at; -1 if the lane itself is broken or a stored record no longer matches its DETECTED digest."""
        rep = self.lane.verify()
        lines = self.lane.lines()
        bad = [] if rep["ok"] and len(lines) == len(self._events) else [-1]
        for i, (e, ln) in enumerate(zip(self._events, lines)):
            b = ln["body"]
            same = (b.get("kind"), b.get("record_id"), b.get("at"), b.get("status"), b.get("payload_digest"), b.get("digest")) == \
                (e.kind, e.record_id, e.at, e.status, e.payload_digest, e.digest)
            if not same or e.prev != ln["prev"] or e.digest != e.compute():
                bad.append(i)
        for e in self._events:
            if e.kind == "DETECTED" and (e.record_id not in self._records or exact_digest(self._records[e.record_id]) != e.payload_digest):
                bad.append(-1)
        return bad

    def signal_reliability(self, now) -> list[SignalReliability]:
        """Which signals actually lead? Per precursor, over records declared before `now`: times it alarmed early inside a CONFIRMED
        change, times it misled (listed as misleading, or belonged to a FALSE_ALARM), precision, median lead, and false-alarm records."""
        lead: dict[str, list[int]] = {}
        mis: dict[str, int] = {}
        fa: dict[str, int] = {}
        for r in self.records(now):
            st = self.status(r.record_id)
            if st == RegimeStatus.FALSE_ALARM:
                for p in {s.precursor for s in r.signals}:
                    mis[p] = mis.get(p, 0) + 1
                    fa[p] = fa.get(p, 0) + 1
                continue
            for s in r.earlier_signals:
                if st in (RegimeStatus.CONFIRMED, RegimeStatus.RETURNED):
                    lead.setdefault(s.precursor, []).append(s.lead)
            for s in r.misleading_signals:
                mis[s.precursor] = mis.get(s.precursor, 0) + 1
        out = []
        for p in sorted(set(lead) | set(mis)):
            nl, nm = len(lead.get(p, [])), mis.get(p, 0)
            out.append(SignalReliability(p, nl, nm, nl / (nl + nm) if nl + nm >= 3 else None,
                                         float(np.median(lead[p])) if p in lead else None, fa.get(p, 0)))
        return out

    def advice(self, active: Sequence[str], now) -> RegimeAdvice:
        """'When I start seeing X, the market is beginning to behave differently, and these patterns should be treated
        differently.' Matches the active precursors to remembered changes (Jaccard overlap of precursor sets >= min_overlap);
        needs `min_support` matches, else UNKNOWN. Pattern advice comes from CONFIRMED matches only, and the share of matches that
        turned out FALSE_ALARM is reported alongside."""
        act = set(active)
        if not act:
            return RegimeAdvice(False, 0, None, (), (), 0.0, "no active signals")
        match = []
        for r in self.records(now):
            sig = set(r.precursors())
            j = len(act & sig) / len(act | sig) if act | sig else 0.0
            if j >= self.cfg.min_overlap:
                match.append(r)
        if len(match) < self.cfg.min_support:
            return RegimeAdvice(False, len(match), None, (), (), 0.0, f"UNKNOWN: {len(match)} remembered change(s) resemble this, need {self.cfg.min_support}")
        conf = [r for r in match if self.status(r.record_id) in (RegimeStatus.CONFIRMED, RegimeStatus.RETURNED)]
        far = sum(1 for r in match if self.status(r.record_id) == RegimeStatus.FALSE_ALARM) / len(match)
        need = max(2, math.ceil(0.6 * len(conf)))
        wk: dict[str, int] = {}
        st: dict[str, int] = {}
        for r in conf:
            for p in r.patterns_weakened:
                wk[p] = wk.get(p, 0) + 1
            for p in r.patterns_strengthened:
                st[p] = st.get(p, 0) + 1
        weaken = tuple(sorted(p for p, n in wk.items() if n >= need))
        strengthen = tuple(sorted(p for p, n in st.items() if n >= need))
        c = float(np.mean([r.confidence for r in match])) * len(match) / (len(match) + 2.0) * (1.0 - far)
        text = (f"seen {len(match)}x before ({far:.0%} false alarms): treat {', '.join(weaken) or 'no pattern'} as weaker, "
                f"{', '.join(strengthen) or 'no pattern'} as stronger")
        return RegimeAdvice(True, len(match), far, weaken, strengthen, c, text)

    def matured_records(self, now, prov_created: str) -> list[MaturedRecord]:
        """Identity-free research-world facts about records declared before `now`: no dates, streams or units, only counts and
        precursor names. Each is released to the trader only by MaturedRecord.gate(now)."""
        out = []
        for r in self.records(now):
            prov = Provenance(created_real=prov_created, learned_at=r.detected_at, code_hash=_code_hash(), outcomes_seen_through=r.detected_at)
            payload = {"scope": r.scope, "status": self.status(r.record_id).value, "precursors": list(r.precursors()), "latency": r.detection_latency,
                       "confidence": r.confidence, "weakened": list(r.patterns_weakened), "strengthened": list(r.patterns_strengthened),
                       "n_misleading": len(r.misleading_signals)}
            out.append(MaturedRecord(stable_hash([r.record_id, "matured"], 16), r.detected_at, payload, prov, Namespace.MATURED_RESEARCH))
        return out

    def content_hash(self) -> str:
        return exact_digest([e.compute() for e in self._events])         # recomputed from content, so a rewritten event changes it


@dataclasses.dataclass(frozen=True)
class MemoryStep:
    new_records: tuple[str, ...]
    status_changes: tuple[tuple[str, str], ...]
    returned: tuple[str, ...]
    advice: RegimeAdvice | None


def step(memory: RegimeMemory, now, evidence: RegimeEvidence, scope: ScopeVerdict | None = None, monitor: RG.RegimeMonitor | None = None,
         active_precursors: Sequence[str] = ()) -> MemoryStep:
    """The ONE public entry the research loop calls: file any regime change the alarms now support, resolve pending candidates
    that have enough post-declaration data, note confirmed changes whose return conditions now hold, and (if signals are active)
    return the advice memory can give. Everything read is dated at/before `now`."""
    new = [r.record_id for r in build_record(evidence, now, memory.cfg, scope, monitor) if memory.add(r, now)]
    changes, back = [], []
    dates = [as_date(d).isoformat() for d in evidence.dates]
    for r in memory.records():
        st = memory.status(r.record_id)
        if st == RegimeStatus.CANDIDATE:
            new_st = resolve_status(r, evidence.quantities, dates, now, memory.cfg)
            if new_st != RegimeStatus.CANDIDATE and memory.set_status(r.record_id, new_st, now):
                changes.append((r.record_id, new_st.value))
        elif st == RegimeStatus.CONFIRMED and returned(r, {k: list(v) for k, v in evidence.quantities.items()}):
            if memory.set_status(r.record_id, RegimeStatus.RETURNED, now):
                back.append(r.record_id)
    adv = memory.advice(active_precursors, now) if active_precursors else None
    return MemoryStep(tuple(new), tuple(changes), tuple(back), adv)


def render_record(rec: RegimeChangeRecord) -> str:
    """Plain-text card for one remembered regime change (all checklist-W fields)."""
    lines = [f"REGIME CHANGE {rec.record_id} [{rec.scope}] confidence {rec.confidence:.2f}",
             f"  changed ~{rec.change_date}, declared {rec.detected_at} (latency {rec.detection_latency} steps)",
             f"  earliest evidence: {rec.earliest_evidence.stream} on {rec.earliest_evidence.alarm_date}",
             f"  earlier signals: {', '.join(s.stream for s in rec.earlier_signals) or 'none'}",
             f"  misleading signals: {', '.join(s.stream for s in rec.misleading_signals) or 'none'}",
             f"  patterns weakened: {', '.join(rec.patterns_weakened) or 'none'}; strengthened: {', '.join(rec.patterns_strengthened) or 'none'}",
             f"  errors before detection: n={rec.errors_before_detection.n} mean|z|={rec.errors_before_detection.mean_abs_z}",
             f"  return conditions: {len(rec.return_conditions)}"]
    for q in rec.new_regime.quantities:
        o = rec.old_regime.get(q.name)
        if o is not None:
            lines.append(f"    {q.name}: {o.mean:.4g} -> {q.mean:.4g}")
    return "\n".join(lines)
