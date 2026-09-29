"""The curator: the trusted, automatic, invisible side of the blind-trader design (canon C64; C55-C59; contract sections 15,
17, 28-30, 49, 55). IMPLEMENTED - NOT VALIDATED.

C64: while the system runs it must not know the year and must receive information day by day, only as it became available.
When memory is saved it is FILED under the real year it operated in. "Relative memory" (which memories matter now) is still
important, but it lives in a separate system that DOES know the year, so it can prioritise memory differently in different
eras without the trader ever seeing that this happens. That system is this module.

  trader side (blind)                        curator side (this file; knows the real date)
  -------------------                        ---------------------------------------------
  TraderSituation  <-- Curator.day() ------  clock: strictly increasing real dates; future_firewall on every daily input
  TraderRelease    <-- Curator.release() --  relevance = era similarity x C59 consistency x reliability x recency x C58 filter
  (weights, no dates, no reasons)            audit trail of every release stays HERE (trusted-side reports only)

Storage reuses archive.ChainFile - a typed lane on engine.pattern_memory's hash chain, the store every other layer uses - one
directory per real year (`y<year>/`), so filing by year is literal and each year verifies independently. There is no new
store. Freshness reuses future_firewall (timestamp + availability checks) and memory_firewall.could_exist_at (a memory must be
able to have existed at `real_now`); C58 is enforced twice, by the matured-strictly-before filter and by that firewall.

Channel 6 (a measurement switch for LATER testing, not wired as default): `relative_m_context` re-expresses each m_* input as
its position within its own trailing past-only history and drops absolute levels; `transform_report` measures, with the
leak_audit fingerprint probe (read-only), how much year-identifiability and how much signal that transform removes."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .archive import ChainFile
from .core import (FirewallBreach, Provenance, TemporalClass, as_date, canonical_json, current_code_hash, require_past,
                   stable_hash)
from .future_firewall import InputKind, LearningInput, check_availability, check_timestamp
from .memory_firewall import MemoryPolicy, could_exist_at
from .trader_view import (KINDS, TraderDay, compare_releases, TraderMemoryItem, TraderRelease, TraderSituation, ScrubLedger, find_violations,
                          opaque_token, release_year_hits, scrub, weight_entropy)

M_PREFIX = "m_"
LANE_KIND = "cur"
AUDIT_KIND = "aud"
ITEM_FIELDS = ("item_id", "kind", "weight", "features", "lean", "horizon")
HALF_LIFE_DAYS: dict[TemporalClass, float | None] = {
    TemporalClass.PERSISTENT: None, TemporalClass.SLOW_DECAY: 1460.0, TemporalClass.FAST_DECAY: 180.0,
    TemporalClass.EPISODIC: 365.0, TemporalClass.REGIME_BOUND: 730.0, TemporalClass.EVENT_BOUND: 365.0,
    TemporalClass.SEASONAL: None, TemporalClass.UNKNOWN: 730.0}


@dataclasses.dataclass(frozen=True)
class RelevanceConfig:
    """Every knob of the curator's prioritisation, in one frozen record so a change is a new config, never a mutation."""
    bandwidth: float = 1.0                  # era-similarity kernel width in robust standard deviations of the state
    universal_min_years: int = 3            # C59: consistency across at least this many DISTINCT real years ...
    universal_share: float = 0.8            # ... with this share of years agreeing in sign => universal
    universal_era_floor: float = 0.35       # a universal memory is only mildly gated by era
    min_priority: float = 0.02              # below this a memory is not worth handing over
    default_k: int = 8
    recency_floor: float = 0.05
    min_shared_state_keys: int = 1
    history_days: int = 1260                # trailing sessions used to express today's state relative to its own past
    min_history: int = 60
    z_clip: float = 6.0
    seasonal_width_days: float = 45.0       # SEASONAL memories: gaussian in circular day-of-year distance
    strength_edges: tuple = (0.2, 0.4, 0.6)  # S17a: era-free absolute strength (reliability x consistency) -> band 0..len(edges)
    single_year_max_band: int = 1           # a memory seen in one real year can never read as strong

    def check(self) -> list[str]:
        errs = []
        if self.bandwidth <= 0:
            errs.append("bandwidth must be > 0")
        if not 0 <= self.universal_share <= 1 or self.universal_min_years < 2:
            errs.append("universal rule needs share in [0,1] and >= 2 years")
        if not 0 <= self.universal_era_floor <= 1 or not 0 <= self.min_priority < 1 or not 0 < self.recency_floor <= 1:
            errs.append("floors must lie in [0,1]")
        if self.default_k < 1 or self.min_history < 2 or self.history_days < self.min_history:
            errs.append("k >= 1 and 2 <= min_history <= history_days")
        if list(self.strength_edges) != sorted(set(self.strength_edges)) or not all(0 < e < 1 for e in self.strength_edges):
            errs.append("strength_edges must be strictly increasing values inside (0,1)")
        return errs


# ------------------------------------------------------------------------------------------------ filed memory
def _iso(x) -> str:
    return as_date(x).isoformat()


def _clean_state(state: Mapping[str, Any] | None, what: str = "state") -> dict[str, float]:
    """m_* market state as plain finite floats. Anything else in the mapping is a mistake, not silently dropped."""
    out: dict[str, float] = {}
    for k, v in dict(state or {}).items():
        if not isinstance(k, str) or not k.startswith(M_PREFIX):
            raise FirewallBreach(f"{what}: key {k!r} is not an m_* market-context column")
        v = v.item() if hasattr(v, "item") and callable(v.item) else v
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise FirewallBreach(f"{what}: {k}={v!r} is not a finite number")
        out[k] = float(v)
    return out


@dataclasses.dataclass(frozen=True)
class FiledMemory:
    """One memory as the curator files it (trusted side: it carries the real year and dates). `key` is the trader-side opaque
    item id; `payload_json` is exactly what the trader may later be shown; `context_json` is the m_* state when it operated."""
    mem_id: str
    key: str
    kind: str
    real_year: int
    filed_date: str                  # the real day the memory operated
    matured_at: str                  # the real day its outcome became knowable
    payload_json: str                # {"features": {...}, "lean": x, "horizon": h}
    context_json: str
    reliability: float
    calibration_error: float
    temporal: TemporalClass
    code_hash: str
    created_real: str

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    @property
    def context(self) -> dict[str, float]:
        return json.loads(self.context_json)

    @property
    def lean(self) -> float:
        return float(self.payload["lean"])

    def content(self) -> dict:
        return {"key": self.key, "kind": self.kind, "real_year": self.real_year, "filed_date": self.filed_date,
                "matured_at": self.matured_at, "payload": self.payload, "context": self.context,
                "reliability": self.reliability, "calibration_error": self.calibration_error,
                "temporal": self.temporal.value, "code_hash": self.code_hash}

    def body(self) -> dict:
        return {**self.content(), "mem_id": self.mem_id, "created_real": self.created_real}

    @classmethod
    def from_body(cls, b: Mapping) -> "FiledMemory":
        return cls(mem_id=b["mem_id"], key=b["key"], kind=b["kind"], real_year=int(b["real_year"]), filed_date=b["filed_date"],
                   matured_at=b["matured_at"], payload_json=canonical_json(b["payload"]), context_json=canonical_json(b["context"]),
                   reliability=float(b["reliability"]), calibration_error=float(b["calibration_error"]),
                   temporal=TemporalClass.parse(b["temporal"]), code_hash=b["code_hash"], created_real=b.get("created_real", ""))

    def validate(self) -> list[str]:
        errs = []
        try:
            f, m = as_date(self.filed_date), as_date(self.matured_at)
        except (ValueError, TypeError):
            return ["filed_date/matured_at are not dates"]
        if f.year != self.real_year:
            errs.append(f"real_year {self.real_year} does not match the day it operated ({f})")
        if m < f:
            errs.append("outcome matured before the memory operated")
        if self.kind not in KINDS:
            errs.append(f"kind {self.kind!r} not in {sorted(KINDS)}")
        for name in ("reliability", "calibration_error"):
            v = getattr(self, name)
            if not (isinstance(v, float) and 0.0 <= v <= 1.0):
                errs.append(f"{name}={v!r} outside [0,1]")
        try:
            p = self.payload
            TraderMemoryItem.make(self.key, self.kind, 0.0, p["features"], p["lean"], p["horizon"])
        except (FirewallBreach, KeyError, TypeError, ValueError) as e:
            errs.append(f"payload is not a valid trader item: {e}")
        if not self.context:
            errs.append("no m_* context: the memory cannot be placed in an era")
        return errs

    def provenance(self) -> Provenance:
        return Provenance(created_real=self.created_real or "1970-01-01T00:00:00", learned_at=self.matured_at,
                          code_hash=self.code_hash, outcomes_seen_through=self.matured_at)


class CuratorStore:
    """Year-filed persistence: one hash-chained lane per real year, `<root>/y<year>/chain.jsonl` (archive.ChainFile on
    pattern_memory's chain). `root=None` keeps the same structure in memory. Opening verifies every chain and fails closed."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = os.fspath(root) if root is not None else None
        self._lanes: dict[int, ChainFile] = {}
        self._by_year: dict[int, list[FiledMemory]] = defaultdict(list)
        self._ids: dict[str, FiledMemory] = {}
        if self.root is not None and os.path.isdir(self.root):
            for name in sorted(os.listdir(self.root)):
                if re.fullmatch(r"y\d{4}", name):
                    self._ingest(int(name[1:]))

    def lane(self, year: int) -> ChainFile:
        if year not in self._lanes:
            self._lanes[year] = ChainFile(os.path.join(self.root, f"y{year}") if self.root else None, LANE_KIND)
        return self._lanes[year]

    def _ingest(self, year: int) -> None:
        for line in self.lane(year).take_new():
            m = FiledMemory.from_body(line["body"])
            if m.real_year != year:
                raise FirewallBreach(f"memory {m.mem_id} is filed in y{year} but operated in {m.real_year}")
            if stable_hash(m.content(), 24) != m.mem_id:
                raise FirewallBreach(f"memory {m.mem_id}: content does not match its id (store edited)")
            if m.mem_id not in self._ids:
                self._ids[m.mem_id] = m
                self._by_year[year].append(m)

    def refresh(self) -> None:
        """Absorb what other writers appended to known year lanes."""
        for y, lane in list(self._lanes.items()):
            lane.sync()
            self._ingest(y)

    def add(self, m: FiledMemory) -> tuple[FiledMemory, bool]:
        errs = m.validate()
        if errs:
            raise FirewallBreach("memory refused: " + "; ".join(errs))
        if m.mem_id in self._ids:
            return self._ids[m.mem_id], False
        self.lane(m.real_year).append_many([m.body()])
        self._ingest(m.real_year)
        return self._ids[m.mem_id], True

    def __len__(self) -> int:
        return len(self._ids)

    def years(self) -> list[int]:
        return sorted(y for y, v in self._by_year.items() if v)

    def memories(self, matured_before=None) -> list[FiledMemory]:
        """All memories, optionally only those whose outcome matured strictly before `matured_before` (C58)."""
        out = [m for y in sorted(self._by_year) for m in self._by_year[y]]
        if matured_before is None:
            return out
        n = as_date(matured_before)
        return [m for m in out if as_date(m.matured_at) < n]

    def counts_by_year(self) -> dict[int, int]:
        return {y: len(v) for y, v in sorted(self._by_year.items()) if v}

    def verify(self) -> dict:
        """Re-verify every year's chain from scratch; ok only if all are intact."""
        per = {y: self.lane(y).verify() for y in sorted(self._lanes)}
        return {"ok": all(v["ok"] for v in per.values()), "years": per}


# ------------------------------------------------------------------------------------------------ priorities (trusted)
@dataclasses.dataclass(frozen=True)
class Priority:
    """Why one filed memory ranks where it does - the explanation the trader never gets."""
    mem_id: str
    key: str
    real_year: int
    era: float
    era_known: bool
    consistency: float
    n_years: int
    universal: bool
    reliability: float
    recency: float
    seasonal: float
    score: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReleaseAudit:
    """Trusted-side record of one release. Kept for reports; never passed to the trader."""
    real_now: str
    step: int
    k: int
    released: tuple[dict, ...]
    candidates: int
    excluded_unmatured: int
    excluded_firewall: int
    excluded_low: int
    state_digest: str

    def to_dict(self) -> dict:
        return {"real_now": self.real_now, "step": self.step, "k": self.k, "released": list(self.released),
                "candidates": self.candidates, "excluded_unmatured": self.excluded_unmatured,
                "excluded_firewall": self.excluded_firewall, "excluded_low": self.excluded_low, "state_digest": self.state_digest}


def robust_scale(values: Sequence[float], floor: float = 1e-6) -> float:
    a = np.asarray([v for v in values if math.isfinite(v)], dtype="float64")
    if len(a) < 2:
        return 1.0
    mad = float(np.median(np.abs(a - np.median(a))))
    s = 1.4826 * mad if mad > 0 else float(a.std(ddof=1))
    return s if s > floor else 1.0


def era_similarity(ctx: Mapping[str, float], state: Mapping[str, float], scales: Mapping[str, float], bandwidth: float = 1.0,
                   min_shared: int = 1) -> tuple[float, bool]:
    """Gaussian kernel of the distance between a memory's filed market state and today's, in robust units. (similarity, known):
    when the two share fewer than `min_shared` keys the era is unknown and reads a neutral 0.5 rather than 0 or 1."""
    keys = [k for k in ctx if k in state]
    if len(keys) < min_shared:
        return 0.5, False
    d2 = float(np.mean([((state[k] - ctx[k]) / (bandwidth * scales.get(k, 1.0))) ** 2 for k in keys]))
    return math.exp(-0.5 * d2), True


def consistency_across_years(leans_by_year: Mapping[int, Sequence[float]], cfg: RelevanceConfig) -> tuple[float, int, bool]:
    """C59 on filed memory: each DISTINCT real year gets one sign (its mean lean); consistency is the smoothed share of years that
    agree with the majority sign; a memory is universal when enough distinct years agree strongly. A single year proves nothing,
    so its smoothed consistency is only 2/3."""
    signs = []
    for y, xs in leans_by_year.items():
        m = float(np.mean(xs)) if len(xs) else 0.0
        signs.append(0 if abs(m) < 1e-12 else 1 if m > 0 else -1)
    n = len(signs)
    if n == 0:
        return 0.0, 0, False
    top = max(signs.count(1), signs.count(-1), signs.count(0))
    cons = (top + 1.0) / (n + 2.0)
    return cons, n, n >= cfg.universal_min_years and top / n >= cfg.universal_share


def strength_band(reliability: float, consistency: float, n_years: int, cfg: RelevanceConfig) -> float:
    """Coarse ABSOLUTE strength of one memory in [0,1] (band / n_bands), for the trader. The release weights sum to 1, so a lone weak
    memory would otherwise read as weight 1.0. Era-free on purpose: only calibrated reliability (already shrunk by the
    calibration error) and cross-year consistency enter; era match, recency and season are excluded because they would tell the
    trader when the memory is from. Coarse on purpose: a fine number would be a fingerprint of the year."""
    strength = float(reliability) * float(consistency)
    band = sum(strength >= e for e in cfg.strength_edges)
    if n_years < 2:
        band = min(band, cfg.single_year_max_band)
    return band / len(cfg.strength_edges)


def recency_factor(age_days: int, temporal: TemporalClass, cfg: RelevanceConfig) -> float:
    hl = HALF_LIFE_DAYS.get(temporal, 730.0)
    if hl is None:
        return 1.0
    return max(cfg.recency_floor, 0.5 ** (max(age_days, 0) / hl))


def seasonal_factor(filed: dt.date, now: dt.date, temporal: TemporalClass, cfg: RelevanceConfig) -> float:
    if temporal != TemporalClass.SEASONAL:
        return 1.0
    d = abs(filed.timetuple().tm_yday - now.timetuple().tm_yday)
    d = min(d, 365 - d)
    return max(cfg.recency_floor, math.exp(-0.5 * (d / cfg.seasonal_width_days) ** 2))


# ------------------------------------------------------------------------------------------------ relative state (trader-bound)
def relative_position(history: Sequence[Mapping[str, float]], state: Mapping[str, float], cfg: RelevanceConfig) -> tuple[dict, bool]:
    """Today's m_* state as z-scores against its own trailing past (today excluded), clipped. Returns (features, warm). Names
    lose the m_ prefix and gain 'rel_' so nothing absolute or year-identifying crosses to the trader."""
    out: dict[str, float] = {}
    warm = len(history) >= cfg.min_history
    for k, v in sorted(state.items()):
        past = np.asarray([h[k] for h in history[-cfg.history_days:] if k in h], dtype="float64")
        if len(past) >= cfg.min_history and past.std(ddof=1) > 1e-12:
            z = (v - past.mean()) / past.std(ddof=1)
            out["rel_" + k[len(M_PREFIX):]] = float(np.clip(z, -cfg.z_clip, cfg.z_clip))
        else:
            out["rel_" + k[len(M_PREFIX):]] = 0.0
            warm = False
    return out, warm


class Curator:
    """The trusted side. One instance per simulated run; the clock is driven by `day`, which is the only way time moves."""

    def __init__(self, store_root: str | os.PathLike | None = None, config: RelevanceConfig | None = None,
                 code_hash: str | None = None, created_real: str | None = None, store: CuratorStore | None = None):
        self.cfg = config or RelevanceConfig()
        errs = self.cfg.check()
        if errs:
            raise ValueError("RelevanceConfig: " + "; ".join(errs))
        self.store = store or CuratorStore(store_root)
        self._code_hash = code_hash
        self._created_real = created_real or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        self.now: dt.date | None = None
        self.step = -1
        self._history: list[dict[str, float]] = []
        self.audit: list[ReleaseAudit] = []
        self._audit_lane = ChainFile(os.path.join(os.fspath(store_root), "audit") if store_root is not None else None, AUDIT_KIND)
        self.ledger = ScrubLedger()
        self.policy = MemoryPolicy(require_data_hash=False, require_experiment_id=False)

    @property
    def code_hash(self) -> str:
        if self._code_hash is None:
            self._code_hash = current_code_hash()
        return self._code_hash

    # ---- filing --------------------------------------------------------------------------------------------------
    def _coerce_item(self, item, on_dirty: str) -> TraderMemoryItem:
        """A trader item from an object or a mapping. Refuse mode raises on the first date-like thing; scrub mode strips it,
        counts it in the ledger and re-derives an opaque id from the cleaned content when the given id is not clean."""
        if isinstance(item, TraderMemoryItem):
            return item
        if not isinstance(item, Mapping):
            raise FirewallBreach(f"cannot file {type(item).__name__}: need a TraderMemoryItem or a mapping")
        extra = sorted(set(map(str, item)) - set(ITEM_FIELDS))
        if on_dirty == "refuse":
            v = find_violations(dict(item))
            if v or extra:
                raise FirewallBreach("memory refused: " + "; ".join([str(x) for x in v[:3]] + ([f"unknown fields {extra}"] if extra else [])))
            body = dict(item)
        else:
            body = self.ledger.scrub({k: v for k, v in dict(item).items() if k in ITEM_FIELDS}) or {}
            if extra:
                self.ledger.counts["unknown_field"] += len(extra)
        feats = body.get("features") or {}
        if not isinstance(feats, Mapping) or not feats:
            raise FirewallBreach("memory has no features left to file")
        lean: Any = body.get("lean")
        horizon: Any = body.get("horizon")
        kind = body.get("kind", "pattern")
        given = item.get("item_id") if isinstance(item, Mapping) else None
        clean_id = body.get("item_id")
        if not (isinstance(clean_id, str) and clean_id == given and re.fullmatch(r"[a-z]{6,40}", clean_id)):
            if on_dirty == "refuse" and given is not None:
                raise FirewallBreach(f"item_id {given!r} is not an opaque letters-only token")
            clean_id = opaque_token({"kind": kind, "features": feats, "h": horizon})
        return TraderMemoryItem.make(clean_id, kind, float(body.get("weight", 1.0)), feats, lean, horizon)

    def file(self, item, real_date, real_year: int, context: Mapping[str, float], matured_at=None, reliability: float = 0.5,
             calibration_error: float = 0.0, temporal: TemporalClass | str = TemporalClass.UNKNOWN,
             on_dirty: str = "refuse") -> FiledMemory:
        """File one memory under the real year it operated in. `item` is what the trader may later see; `context` is the m_*
        market state when it operated. If the clock is running the outcome must already have matured strictly before it: you
        cannot file what has not happened yet (FirewallBreach). Idempotent: refiling identical content returns the record."""
        if on_dirty not in ("refuse", "scrub"):
            raise ValueError("on_dirty is 'refuse' or 'scrub'")
        ti = self._coerce_item(item, on_dirty)
        filed = as_date(real_date)
        if int(real_year) != filed.year:
            raise FirewallBreach(f"real_year {real_year} does not match the day the memory operated ({filed})")
        matured = as_date(matured_at) if matured_at is not None else filed
        if self.now is not None:
            require_past(matured, self.now, "memory being filed")
        for name, v in (("reliability", reliability), ("calibration_error", calibration_error)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                raise FirewallBreach(f"{name}={v!r} must be a number in [0, 1]")
        payload = {"features": ti.features, "lean": ti.lean, "horizon": ti.horizon}
        m = FiledMemory(mem_id="", key=ti.item_id, kind=ti.kind, real_year=filed.year, filed_date=filed.isoformat(),
                        matured_at=matured.isoformat(), payload_json=canonical_json(payload),
                        context_json=canonical_json(_clean_state(context, "context")), reliability=float(reliability),
                        calibration_error=float(calibration_error), temporal=TemporalClass.parse(temporal),
                        code_hash=self.code_hash, created_real=self._created_real)
        m = dataclasses.replace(m, mem_id=stable_hash(m.content(), 24))
        return self.store.add(m)[0]

    # ---- the daily clock ------------------------------------------------------------------------------------------
    def day(self, real_date, market_state: Mapping[str, float] | None = None, info: Iterable[Mapping] = ()) -> TraderSituation:
        """Advance the trusted clock by one simulated day and return what the trader may know today. Time only moves forward
        (a repeated or earlier date is a FirewallBreach) and every input dated or published after today is refused before
        it is used. The trader receives the run-relative step and today's state relative to its own past - never the date."""
        today = as_date(real_date)
        if self.now is not None and today <= self.now:
            raise FirewallBreach(f"clock must move forward: {today} is not after {self.now}")
        state = _clean_state(market_state, "market_state")
        inputs = [LearningInput(name="market_state", kind=InputKind.FEATURE, timestamp=today)] if state else []
        for row in info:
            if not isinstance(row, Mapping) or "name" not in row or "timestamp" not in row:
                raise FirewallBreach(f"daily input {row!r} needs at least a name and a timestamp")
            inputs.append(LearningInput(name=str(row["name"]), kind=row.get("kind", InputKind.FEATURE), timestamp=row["timestamp"],
                                        available_at=row.get("available_at")))
        for chk in (check_timestamp(inputs, today), check_availability(inputs, today)):
            bad = [f for f in chk.findings if f.is_fail]
            if bad:
                raise FirewallBreach(f"day {today}: {chk.check} check failed: {bad[0].subject}: {bad[0].message}")
        feats, warm = relative_position(self._history, state, self.cfg)
        self.now = today
        self.step += 1
        if state:
            self._history.append(state)
        return TraderSituation.make(self.step, feats, warm)

    def visible_count(self, real_now=None) -> int:
        """How many filed memories are knowable at `real_now` (default: today). Grows day by day as outcomes mature."""
        return len(self.store.memories(matured_before=real_now if real_now is not None else self._need_clock()))

    def _need_clock(self) -> dt.date:
        if self.now is None:
            raise FirewallBreach("the clock has not started: call day() first")
        return self.now

    # ---- relevance --------------------------------------------------------------------------------------------------
    def _existence_ok(self, m: FiledMemory, now: dt.date) -> bool:
        item = SimpleNamespace(knowledge_id=m.mem_id, version=1, provenance=m.provenance(), contexts={}, anti_contexts={},
                               payload=m.payload["features"])
        return could_exist_at(item, now, policy=self.policy).could_exist

    def relevance(self, real_now, market_state: Mapping[str, float]) -> tuple[list[Priority], dict]:
        """Score every filed memory that could exist at `real_now`. Returns (record-level priorities, exclusion counts). Past-only
        by construction: unmatured memories are removed before anything (scales, consistency, ranking) is computed."""
        now = as_date(real_now)
        state = _clean_state(market_state, "market_state")
        allm = self.store.memories()
        eligible = [m for m in allm if as_date(m.matured_at) < now]
        unmatured = len(allm) - len(eligible)
        ok = [m for m in eligible if self._existence_ok(m, now)]
        excl = {"candidates": len(allm), "unmatured": unmatured, "firewall": len(eligible) - len(ok)}
        if not ok:
            return [], excl
        scales = {k: robust_scale([m.context[k] for m in ok if k in m.context] + [state.get(k, math.nan)]) for k in state}
        by_key: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for m in ok:
            by_key[m.key][m.real_year].append(m.lean)
        cons = {k: consistency_across_years(v, self.cfg) for k, v in by_key.items()}
        out = []
        for m in ok:
            era, known = era_similarity(m.context, state, scales, self.cfg.bandwidth, self.cfg.min_shared_state_keys)
            c, n_years, universal = cons[m.key]
            era_term = self.cfg.universal_era_floor + (1 - self.cfg.universal_era_floor) * era if universal else era
            rel = m.reliability * (1.0 - m.calibration_error)
            rec = recency_factor((now - as_date(m.matured_at)).days, m.temporal, self.cfg)
            sea = seasonal_factor(as_date(m.filed_date), now, m.temporal, self.cfg)
            out.append(Priority(m.mem_id, m.key, m.real_year, era, known, c, n_years, universal, rel, rec, sea,
                                float(rel * c * era_term * rec * sea)))
        return out, excl

    def rank_keys(self, prios: Sequence[Priority]) -> list[dict]:
        """Collapse record priorities to one entry per trader-side item: the best record's score, and a score-weighted lean."""
        mems = {m.mem_id: m for m in self.store.memories()}
        groups: dict[str, list[Priority]] = defaultdict(list)
        for p in prios:
            groups[p.key].append(p)
        rows = []
        for key, ps in groups.items():
            best = max(ps, key=lambda p: (p.score, p.mem_id))
            w = np.asarray([p.score for p in ps], dtype="float64")
            leans = np.asarray([mems[p.mem_id].lean for p in ps], dtype="float64")
            lean = float((w * leans).sum() / w.sum()) if w.sum() > 0 else float(leans.mean())
            rows.append({"key": key, "score": best.score, "lean": lean, "best": best.mem_id, "years": sorted({p.real_year for p in ps}),
                         "n_records": len(ps), "consistency": best.consistency, "n_years": best.n_years,
                         "universal": best.universal, "era": best.era, "reliability": best.reliability, "recency": best.recency,
                         "seasonal": best.seasonal,
                         "band": strength_band(best.reliability, best.consistency, best.n_years, self.cfg)})
        return sorted(rows, key=lambda r: (-r["score"], r["key"]))

    def release(self, real_now, market_state: Mapping[str, float], k: int | None = None) -> TraderRelease:
        """Hand the trader its weighted, anonymised, date-free memory for today. The clock must have reached `real_now` (a release
        from a day that has not happened is a FirewallBreach). Weights sum to 1 and items are ordered by opaque id, so neither the
        numbers nor the order say why one memory outranks another; the audit trail stays on this side."""
        now = as_date(real_now)
        if self.now is None or now > self.now:
            raise FirewallBreach(f"release for {now} but the clock is at {self.now}: information cannot be served before its day")
        k = self.cfg.default_k if k is None else int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        prios, excl = self.relevance(now, market_state)
        ranked = self.rank_keys(prios)
        kept = [r for r in ranked if r["score"] >= self.cfg.min_priority][:k]
        low = len([r for r in ranked if r["score"] < self.cfg.min_priority])
        mems = {m.mem_id: m for m in self.store.memories()}
        total = sum(r["score"] for r in kept)
        items = []
        for r in sorted(kept, key=lambda r: r["key"]):
            best = mems[r["best"]]
            p = best.payload
            items.append(TraderMemoryItem.make(r["key"], best.kind, r["score"] / total, {**p["features"], "strength_band": r["band"]},
                                               float(np.clip(r["lean"], -1, 1)), p["horizon"]))
        rel = TraderRelease(max(self.step, 0), tuple(items))
        hits = release_year_hits(rel)
        if hits:
            raise FirewallBreach(f"release would carry date-like content to the trader: {hits}")
        entry = ReleaseAudit(now.isoformat(), max(self.step, 0), k, tuple(kept), excl["candidates"], excl["unmatured"],
                             excl["firewall"], low, stable_hash(_clean_state(market_state, "market_state")))
        self.audit.append(entry)
        self._audit_lane.append_many([entry.to_dict()])
        return rel

    def run_day(self, real_date, market_state: Mapping[str, float], k: int | None = None, info: Iterable[Mapping] = ()):
        """The trusted clock's once-per-day call: advance, then serve the situation and the release. Returns both trader records."""
        sit = self.day(real_date, market_state, info)
        return sit, self.release(real_date, market_state, k)

    # ---- reports (trusted side only) ---------------------------------------------------------------------------------
    def audit_frame(self) -> pd.DataFrame:
        """One row per released item per day, with the reasons - for reports, never for the trader."""
        rows = []
        for a in self.audit:
            for r in a.released:
                rows.append({"real_now": a.real_now, "step": a.step, **{k: v for k, v in r.items() if k != "years"},
                             "years": ",".join(str(y) for y in r["years"])})
        return pd.DataFrame(rows)

    def audit_verify(self) -> dict:
        return self._audit_lane.verify()

    def year_share(self) -> pd.Series:
        """Share of released weight-mass by the real year a memory came from (a diagnostic: is the curator leaning on one era?)."""
        df = self.audit_frame()
        if df.empty:
            return pd.Series(dtype="float64")
        s: dict[int, float] = defaultdict(float)
        for _, r in df.iterrows():
            ys = [int(y) for y in str(r["years"]).split(",") if y]
            for y in ys:
                s[y] += float(r["score"]) / max(len(ys), 1)
        tot = sum(s.values())
        return pd.Series({y: v / tot for y, v in sorted(s.items())})

    def truncated(self, real_now) -> "Curator":
        """A twin holding only memories whose outcomes matured strictly before `real_now` - what a run stopped at that day would have."""
        twin = Curator(None, self.cfg, self.code_hash, self._created_real)
        for m in self.store.memories(matured_before=real_now):
            twin.store.add(m)
        return twin

    def report_markdown(self) -> str:
        v = self.store.verify()
        lines = [f"# Curator report", f"- clock: {self.now} (step {self.step})", f"- filed memories: {len(self.store)} across years "
                 f"{self.store.years()}", f"- store chains intact: {v['ok']}", f"- releases audited: {len(self.audit)}",
                 f"- scrub ledger: {self.ledger.summary()}"]
        ys = self.year_share()
        if len(ys):
            lines.append("- release mass by source year: " + ", ".join(f"{y}: {s:.0%}" for y, s in ys.items()))
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ invariants
def prefix_invariance(cur: Curator, nows: Sequence, state_of: Callable[[Any], Mapping[str, float]], k: int = 8) -> list[str]:
    """For each date, the release from the full store must equal the release from a store holding only memories that had
    matured by then. Any difference means a later-matured memory influenced an earlier day. Returns mismatch descriptions."""
    bad = []
    for d in nows:
        full, _ = cur.relevance(d, state_of(d))
        twin = cur.truncated(d)
        part, _ = twin.relevance(d, state_of(d))
        a = [(r["key"], round(r["score"], 9)) for r in cur.rank_keys(full)[:k]]
        b = [(r["key"], round(r["score"], 9)) for r in twin.rank_keys(part)[:k]]
        if a != b:
            bad.append(f"{_iso(d)}: release differs when later-matured memories are removed")
    return bad


def priorities_differ(a: Sequence[Priority], b: Sequence[Priority], tol: float = 1e-9) -> bool:
    """True when two priority lists (for the same trader-side keys) assign different scores - the curator saw two eras."""
    sa, sb = {p.key: p.score for p in a}, {p.key: p.score for p in b}
    return sa.keys() != sb.keys() or any(abs(sa[k] - sb[k]) > tol for k in sa)


# ------------------------------------------------------------------------------------------------ channel 6: relative m_* context
def m_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith(M_PREFIX)]


def _past_percentile(x: np.ndarray, lookback: int, min_periods: int) -> np.ndarray:
    """Percentile rank of x[i] within the strictly-earlier window x[i-lookback:i] (ties count half). NaN until min_periods."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not math.isfinite(x[i]):
            continue
        w = x[max(0, i - lookback):i]
        w = w[np.isfinite(w)]
        if len(w) >= min_periods:
            out[i] = float((w < x[i]).mean() + 0.5 * (w == x[i]).mean())
    return out


def relative_m_context(frame: pd.DataFrame, lookback: int = 1260, min_periods: int = 60, mode: str = "both",
                       z_clip: float = 6.0) -> pd.DataFrame:
    """Channel-6 measurement switch: replace each m_* column by its position relative to its OWN trailing history (z-score and/or
    percentile over the previous `lookback` sessions, today excluded) and drop the absolute levels entirely. Past-only: row i
    depends on rows < i and row i, so truncating the frame never changes an earlier row. Names: m_vix -> rel_vix / pct_vix."""
    if mode not in ("z", "pct", "both"):
        raise ValueError("mode is 'z', 'pct' or 'both'")
    cols = m_columns(frame)
    out = {}
    for c in cols:
        s = pd.to_numeric(frame[c], errors="coerce").astype("float64")
        stem = str(c)[len(M_PREFIX):]
        if mode in ("z", "both"):
            past = s.shift(1)
            mu = past.rolling(lookback, min_periods=min_periods).mean()
            sd = past.rolling(lookback, min_periods=min_periods).std()
            z = ((s - mu) / sd.where(sd > 1e-12)).clip(-z_clip, z_clip)
            out["rel_" + stem] = z
        if mode in ("pct", "both"):
            out["pct_" + stem] = pd.Series(_past_percentile(s.to_numpy(), lookback, min_periods), index=frame.index)
    return pd.DataFrame(out, index=frame.index)


def window_summary(frame: pd.DataFrame, starts: pd.DatetimeIndex, months: int = 12) -> pd.DataFrame:
    """Per-window mean and sd of every column over [start, start + months): the feature vector the fingerprint probe sees."""
    rows = {}
    for s in starts:
        w = frame.loc[(frame.index >= s) & (frame.index < s + pd.DateOffset(months=months))]
        rec = {}
        for c in frame.columns:
            rec[f"{c}_mean"] = float(w[c].mean()) if w[c].notna().any() else math.nan
            rec[f"{c}_sd"] = float(w[c].std()) if w[c].notna().sum() > 1 else math.nan
        rows[s] = rec
    return pd.DataFrame.from_dict(rows, orient="index")


def _mean_abs_ic(frame: pd.DataFrame, target: pd.Series) -> float:
    ics = []
    for c in frame.columns:
        pair = pd.concat([frame[c], target], axis=1, join="inner").dropna()
        if len(pair) >= 30 and pair.iloc[:, 0].nunique() > 1:
            ics.append(abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))))
    return float(np.mean(ics)) if ics else math.nan


def transform_report(frame: pd.DataFrame, target: pd.Series, lookback: int = 1260, months: int = 12, seed: int = 0,
                     n_trees: int = 100) -> dict:
    """How much year-identifiability and how much signal does `relative_m_context` remove? Identifiability is the leak_audit
    FingerprintProbe (read-only) on window summaries of the raw m_* columns versus the relative ones: skill 0 = cannot tell the
    year, >= 0.25 = identifiable. Signal is the mean |Spearman IC| of each column against `target` (a caller-supplied future
    outcome series, used here only for measurement). A transform that removes the identifiability AND keeps the signal is the
    one to adopt; one that removes both has just thrown information away."""
    from engine.leak_audit import FingerprintProbe, fingerprint_verdict
    raw = frame[m_columns(frame)]
    rel = relative_m_context(frame, lookback=lookback, mode="z")
    last = frame.index.max() - pd.DateOffset(months=months)
    starts = pd.DatetimeIndex(sorted({d.replace(day=1) for d in frame.index if d <= last}))
    probe = FingerprintProbe(seed=seed, n_trees=n_trees)
    res: dict[str, dict[str, Any]] = {}
    for name, fr in (("raw", raw), ("relative", rel.dropna(how="all"))):
        st = starts[starts >= fr.index.min()] if len(fr) else starts
        F = window_summary(fr, st, months)
        sc = probe.score(F, st, list(F.columns)) if len(st) else {"skill": math.nan}
        res[name] = {"skill": sc.get("skill", math.nan), "mae_years": sc.get("mae_years", math.nan), "n_train": sc.get("n_train", 0),
                     "n_test": sc.get("n_test", 0), "verdict": fingerprint_verdict(sc) if sc.get("n_train") else "not measured"}
    ic_raw, ic_rel = _mean_abs_ic(raw, target), _mean_abs_ic(rel, target)
    s_raw, s_rel = res["raw"]["skill"], res["relative"]["skill"]
    return {"identifiability": res,
            "id_skill_removed": float(s_raw - s_rel) if math.isfinite(s_raw) and math.isfinite(s_rel) else math.nan,
            "signal": {"raw_abs_ic": ic_raw, "relative_abs_ic": ic_rel,
                       "signal_removed": float(1 - ic_rel / ic_raw) if math.isfinite(ic_raw) and ic_raw > 0 and math.isfinite(ic_rel) else math.nan},
            "n_windows": int(len(starts)), "columns": m_columns(frame)}


# ------------------------------------------------------------------------------------------------ trusted-side diagnostics
def key_history(store: CuratorStore, key: str, real_now) -> pd.DataFrame:
    """Per real year, how one trader-side item has behaved (matured evidence only): the C59 table the curator reasons from."""
    rows = defaultdict(list)
    for m in store.memories(matured_before=real_now):
        if m.key == key:
            rows[m.real_year].append(m)
    out = []
    for y, ms in sorted(rows.items()):
        leans = np.asarray([m.lean for m in ms], dtype="float64")
        out.append({"real_year": y, "n": len(ms), "mean_lean": float(leans.mean()), "sign": int(np.sign(leans.mean())),
                    "mean_reliability": float(np.mean([m.reliability for m in ms])), "last_matured": max(m.matured_at for m in ms)})
    return pd.DataFrame(out, columns=["real_year", "n", "mean_lean", "sign", "mean_reliability", "last_matured"])


def year_summary(store: CuratorStore, real_now=None) -> pd.DataFrame:
    """One row per filed year: memories, distinct keys, how many are already knowable at `real_now`, and the mean state they
    were filed under (the era each year contributes)."""
    rows = []
    for y in store.years():
        ms = [m for m in store.memories() if m.real_year == y]
        known = [m for m in ms if real_now is None or as_date(m.matured_at) < as_date(real_now)]
        state = pd.DataFrame([m.context for m in ms]).mean().to_dict()
        rows.append({"real_year": y, "memories": len(ms), "keys": len({m.key for m in ms}), "knowable": len(known),
                     **{f"mean_{k}": v for k, v in sorted(state.items())}})
    return pd.DataFrame(rows)


def explain(cur: Curator, key: str, real_now, market_state: Mapping[str, float]) -> dict:
    """Trusted-side answer to 'why does this item rank where it does today?' - the answer the trader is never given."""
    prios, excl = cur.relevance(real_now, market_state)
    mine = [p for p in prios if p.key == key]
    if not mine:
        return {"key": key, "eligible": False, "excluded": excl}
    best = max(mine, key=lambda p: (p.score, p.mem_id))
    ranked = cur.rank_keys(prios)
    pos = [r["key"] for r in ranked].index(key)
    return {"key": key, "eligible": True, "rank": pos + 1, "of": len(ranked), "best": best.to_dict(), "records": len(mine),
            "history": key_history(cur.store, key, real_now).to_dict("records")}


def era_sensitivity(cur: Curator, real_now, states: Mapping[str, Mapping[str, float]], k: int = 5) -> pd.DataFrame:
    """Top-k trader-side keys under each named market state at the same real date, and their overlap with the first state. A
    curator whose ranking never moves with the state is not doing relative memory at all."""
    rows, base = [], None
    for label, st in states.items():
        prios, _ = cur.relevance(real_now, st)
        top = [r["key"] for r in cur.rank_keys(prios)[:k]]
        base = top if base is None else base
        union = set(top) | set(base)
        rows.append({"state": label, "top": top, "overlap_with_first": len(set(top) & set(base)) / len(union) if union else 1.0})
    return pd.DataFrame(rows)


def turnover_series(cur: Curator) -> pd.Series:
    """Day-to-day change in the released set (1 - Jaccard of consecutive audits); a curator that thrashes is unstable."""
    out, prev = {}, None
    for a in cur.audit:
        ids = {r["key"] for r in a.released}
        if prev is not None:
            u = ids | prev
            out[a.real_now] = 1.0 - (len(ids & prev) / len(u) if u else 1.0)
        prev = ids
    return pd.Series(out, dtype="float64")


def release_window_features(releases: Sequence[TraderRelease], situations: Sequence[TraderSituation]) -> dict[str, float]:
    """Summary of what the trader was shown over one window: item count, weight entropy, mean lean, mean absolute weighted lean,
    and the mean and spread of each relative state feature. This is the trader's whole view of the era, compressed for the probe."""
    items = [i for r in releases for i in r.items]
    feats: dict[str, list[float]] = defaultdict(list)
    for s in situations:
        for k, v in s.features.items():
            feats[k].append(v)
    out = {"n_items": float(np.mean([len(r) for r in releases])) if releases else 0.0,
           "entropy": float(np.mean([weight_entropy(r) for r in releases])) if releases else 0.0,
           "mean_lean": float(np.mean([i.lean * i.weight for i in items])) if items else 0.0,
           "abs_lean": float(np.mean([abs(i.lean) * i.weight for i in items])) if items else 0.0}
    out.update({f"{k}_mean": float(np.mean(v)) for k, v in sorted(feats.items())})
    out.update({f"{k}_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in sorted(feats.items())})
    return out


def trader_view_identifiability(window_rows: pd.DataFrame, starts: pd.DatetimeIndex, seed: int = 0, n_trees: int = 100) -> dict:
    """Can the real year be read off the trader's own view? Runs the leak_audit FingerprintProbe (read-only) on one row per
    window of `release_window_features`, keyed by the window's real start. Skill near 0 = the two-sided design holds; a
    skill at or above 0.25 means the curator's output still carries the era."""
    from engine.leak_audit import FingerprintProbe, fingerprint_verdict
    if len(window_rows) != len(starts):
        raise ValueError("one feature row per window start")
    sc = FingerprintProbe(seed=seed, n_trees=n_trees).score(window_rows.set_axis(starts), starts, list(window_rows.columns))
    sc["verdict"] = fingerprint_verdict(sc) if sc.get("n_train") else "not measured"
    return sc


def store_manifest(cur: Curator) -> dict:
    """Trusted-side inventory for provenance: chains, counts by year, config hash, code hash, clock."""
    v = cur.store.verify()
    return {"years": cur.store.counts_by_year(), "chains_ok": v["ok"], "head_by_year": {y: c["head"] for y, c in v["years"].items()},
            "config_hash": stable_hash(dataclasses.asdict(cur.cfg)), "code_hash": cur.code_hash, "clock": str(cur.now),
            "releases": len(cur.audit), "scrub": cur.ledger.summary()}


# ------------------------------------------------------------------------------------------------ the trusted driver
def file_batch(cur: Curator, records: Iterable[Mapping], on_dirty: str = "refuse") -> dict:
    """File many memories, never stopping on a bad one: each record is a mapping of `Curator.file` arguments. Returns how many
    were filed, and for every refusal the index and reason - a refusal is data for the report, not a reason to lose the batch."""
    filed, refused = 0, []
    for i, rec in enumerate(records):
        try:
            cur.file(on_dirty=on_dirty, **rec)
            filed += 1
        except (FirewallBreach, KeyError, TypeError, ValueError) as e:
            refused.append({"index": i, "reason": type(e).__name__ + ": " + str(e)[:200]})
    return {"filed": filed, "refused": refused, "store_size": len(cur.store)}


def run_window(cur: Curator, states: pd.DataFrame, k: int | None = None, info_of: Callable[[Any], Sequence[Mapping]] | None = None) -> list[TraderDay]:
    """The trusted clock: one call to `run_day` per row of `states` (index = real dates, m_* columns), in order. Every day's
    trader-bound output is re-scanned for date-like content before it is returned; a hit aborts the run (fail closed)."""
    days = []
    cols = m_columns(states)
    for d, row in states[cols].iterrows():
        state = {c: float(v) for c, v in row.items() if math.isfinite(float(v))}
        sit, rel = cur.run_day(d, state, k, info_of(d) if info_of else ())
        day = TraderDay(sit, rel)
        hits = find_violations(day.to_dict())
        if hits:
            raise FirewallBreach(f"trader-bound output for step {sit.step} carries dates: {hits[0]}")
        days.append(day)
    return days


def replay_matches(make: Callable[[], Curator], states: pd.DataFrame, k: int | None = None) -> bool:
    """Two independent runs from the same store and states must hand the trader byte-identical days (determinism, section 62)."""
    a = [d.json() for d in run_window(make(), states, k)]
    b = [d.json() for d in run_window(make(), states, k)]
    return a == b


def blind_equivalence(make: Callable[[], Curator], states: pd.DataFrame, shift_days: int, k: int | None = None) -> dict:
    """Shift the real calendar by `shift_days` (the same market, a different year label) and compare what the trader sees.
    The curator's memories are filed by real year, so a shifted run legitimately sees different priorities; what must NOT change
    is the SHAPE of the trader's view (steps, field names, item counts under an empty store). Returns the comparison the
    report shows: identical situations, and how much the releases differ."""
    base = run_window(make(), states, k)
    moved = states.copy()
    moved.index = moved.index + pd.Timedelta(days=shift_days)
    other = run_window(make(), moved, k)
    same_steps = [a.situation.step for a in base] == [b.situation.step for b in other]
    same_sit = all(a.situation.features == b.situation.features for a, b in zip(base, other))
    diffs = [compare_releases(a.release, b.release)["weight_l1"] for a, b in zip(base, other)]
    return {"days": len(base), "same_steps": same_steps, "same_situations": same_sit,
            "mean_release_l1": float(np.mean(diffs)) if diffs else 0.0}
