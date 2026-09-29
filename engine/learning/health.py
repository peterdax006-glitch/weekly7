"""Knowledge health monitor (contract C62 section 46; checklist K05, D11/D12 support; canon C60/C61).
Bible phase serving: pattern health / reliability (9/10), the C61 "every pattern re-checked every period" rule, generalised from
patterns to any knowledge item.

Continuously track nine states - HEALTHY, DEGRADING, BROKEN, CONTRADICTED, DORMANT, RECOVERING, UNSTABLE, INSUFFICIENT_EVIDENCE,
UNKNOWN - for every item, and export the dashboard the owner reads: what is trusted, what is losing trust, WHY, what evidence
caused the change, and which research is investigating it. The export is plain JSON with a schema check, so a machine (and a
different account) can read it.

Sources combined per item, all strictly before `as_of`:
  engine.pattern_reliability.health_monitor  sequential CUSUM against the item's own established effect + a discounted posterior;
                                             a phantom (effect never established) is BROKEN, not merely quiet
  lifecycle.trace                            stage (DECAY / FAILURE / RECOVERY / DORMANT ...)
  the retirement ledger's state              DEGRADED / DORMANT / RETIRED (passed in as a string)
  contradictions                             open, unresolved contradictions with a strength (passed in)
  research assignments                       who is looking at it (passed in)

A state is decided by fixed precedence, but every condition that holds is kept in `flags`, so a BROKEN item that is also
contradicted shows both. BROKEN, CONTRADICTED, DORMANT, UNKNOWN and INSUFFICIENT_EVIDENCE items carry zero trust weight and
`assert_untrusted_silent` fails closed if any caller gives them influence. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import pattern_reliability as PR

from . import lifecycle as LC
from .core import Epistemic, FirewallBreach, Health, Lifecycle, as_date, current_code_hash, stable_hash

SCHEMA_VERSION = "health-dashboard/1"
SEVERITY = {Health.HEALTHY: 0, Health.RECOVERING: 1, Health.DEGRADING: 2, Health.UNSTABLE: 3, Health.INSUFFICIENT_EVIDENCE: 3,
            Health.UNKNOWN: 4, Health.CONTRADICTED: 5, Health.DORMANT: 5, Health.BROKEN: 6}
TRUST_WEIGHT = {Health.HEALTHY: 1.0, Health.RECOVERING: 0.5, Health.DEGRADING: 0.5, Health.UNSTABLE: 0.25,
                Health.BROKEN: 0.0, Health.CONTRADICTED: 0.0, Health.DORMANT: 0.0, Health.INSUFFICIENT_EVIDENCE: 0.0,
                Health.UNKNOWN: 0.0}
EPISTEMIC_FOR = {Health.HEALTHY: Epistemic.SUPPORTED, Health.RECOVERING: Epistemic.DEGRADED, Health.DEGRADING: Epistemic.DEGRADED,
                 Health.UNSTABLE: Epistemic.GATED, Health.BROKEN: Epistemic.GATED, Health.CONTRADICTED: Epistemic.CONTRADICTED,
                 Health.DORMANT: Epistemic.GATED, Health.INSUFFICIENT_EVIDENCE: Epistemic.HYPOTHESIS, Health.UNKNOWN: Epistemic.UNKNOWN}
SILENT = (Health.BROKEN, Health.CONTRADICTED, Health.DORMANT, Health.INSUFFICIENT_EVIDENCE, Health.UNKNOWN)
LOSING = (Health.DEGRADING, Health.UNSTABLE, Health.BROKEN, Health.CONTRADICTED)

PARAMS = {
    "min_n": 38,                    # finite outcomes needed before the sequential monitor is meaningful (burn-in 26 + 12)
    "stale_days": 45,               # newest outcome older than this while the item should be firing = UNKNOWN
    "contradiction_bar": 0.5,       # open contradiction strength at or above this = CONTRADICTED
    "unstable_lookback": 52, "unstable_events": 2, "flip_window": 26, "flip_count": 4,
    "recovery_window": 26,          # rows after a broken episode ends during which the item is RECOVERING
    "p_degrade": 0.5,               # discounted P(still works) under this = DEGRADING
    "change_tol": 0.10,             # relative move of an evidence number that counts as "caused the change"
    "arl0": 500, "half_life": 13, "est_win": 26,
}


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


# ------------------------------------------------------------------------------------------------- inputs

@dataclasses.dataclass(frozen=True)
class ContradictionRef:
    """An open contradiction against an item (produced by the contradiction graph; section 18)."""
    contradiction_id: str
    strength: float                # 0..1
    resolved: bool = False
    since: str = ""

    def check(self) -> list[str]:
        errs = []
        if not 0.0 <= self.strength <= 1.0:
            errs.append(f"{self.contradiction_id}: strength outside [0,1]")
        return errs


@dataclasses.dataclass(frozen=True)
class ResearchAssignment:
    research_id: str
    question: str
    item_ids: tuple
    status: str = "OPEN"           # OPEN | RUNNING | DONE

    def active(self) -> bool:
        return self.status in ("OPEN", "RUNNING")


@dataclasses.dataclass
class HealthInput:
    """What the monitor needs about one item. `values` is a Series of signed outcomes indexed by the date each MATURED (so a row
    dated at or after `as_of` is not yet known and is ignored, never used)."""
    knowledge_id: str
    values: pd.Series
    exposure: pd.Series | None = None          # bool per row: did the item fire (aligned with `values`)
    contradictions: tuple = ()
    retirement_state: str | None = None        # ACTIVE | DEGRADED | DORMANT | RETIRED
    expected_effect: float | None = None
    should_fire: bool = True                   # False = the item is intentionally quiet (no staleness alarm)

    def validate(self) -> list[str]:
        errs = []
        if not isinstance(self.values, pd.Series):
            return ["values must be a pandas Series"]
        if not self.values.index.is_monotonic_increasing or not self.values.index.is_unique:
            errs.append("values index must be strictly increasing")
        if np.isinf(self.values.astype(float).values).any():
            errs.append("values contain infinities")
        if self.retirement_state not in (None, "ACTIVE", "DEGRADED", "DORMANT", "RETIRED"):
            errs.append(f"unknown retirement_state {self.retirement_state!r}")
        for c in self.contradictions:
            errs.extend(c.check())
        if self.exposure is not None and not self.exposure.index.equals(self.values.index):
            errs.append("exposure must share the values index")
        return errs


# ------------------------------------------------------------------------------------------------- records

@dataclasses.dataclass(frozen=True)
class HealthRecord:
    knowledge_id: str
    as_of: str
    state: Health
    flags: tuple
    reasons: tuple
    evidence: Mapping[str, Any]
    trusted: bool
    trust_weight: float
    investigating: tuple = ()
    prev_state: Health | None = None
    since: str = ""
    code_hash: str = ""

    @property
    def record_id(self) -> str:
        return stable_hash([self.knowledge_id, self.as_of, self.state.value, self.flags, dict(self.evidence)], 16)

    @property
    def epistemic(self) -> Epistemic:
        return EPISTEMIC_FOR[self.state]

    def as_dict(self) -> dict:
        return {"record_id": self.record_id, "knowledge_id": self.knowledge_id, "as_of": self.as_of, "state": self.state.value,
                "flags": list(self.flags), "reasons": list(self.reasons), "evidence": _clean(dict(self.evidence)),
                "trusted": self.trusted, "trust_weight": self.trust_weight, "investigating": list(self.investigating),
                "prev_state": None if self.prev_state is None else self.prev_state.value, "since": self.since,
                "epistemic": self.epistemic.value, "code_hash": self.code_hash}


def _clean(o: Any) -> Any:
    """JSON-safe copy: NaN / inf become None, numpy scalars become Python numbers."""
    if isinstance(o, Mapping):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ------------------------------------------------------------------------------------------------- one snapshot

def _wide(inputs: Sequence[HealthInput], as_of) -> tuple[PR.Timelines, dict]:
    """One shared-index Timelines of every item's matured outcomes plus an all-NaN final row at `as_of`, so the monitor's row for
    `as_of` uses only outcomes dated before it. Column names are opaque so an id that looks like a date cannot trip the
    pattern-key check."""
    cut = pd.Timestamp(as_of)
    series, names = {}, {}
    for k, inp in enumerate(inputs):
        s = inp.values.astype(float)
        s = s[[pd.Timestamp(i) < cut for i in s.index]]
        series[f"k{k}"] = s
        names[f"k{k}"] = inp.knowledge_id
    idx = sorted(set().union(*[set(s.index) for s in series.values()]) | {cut}) if series else [cut]
    df = pd.DataFrame({c: s.reindex(idx) for c, s in series.items()}, index=pd.DatetimeIndex(idx))
    if not len(df.columns):
        df["k_none"] = np.nan
    tl = PR.Timelines(df, pd.DataFrame({"m_dummy": np.zeros(len(df))}, index=df.index))
    return tl, names


def _episodes(events: pd.DataFrame, col: str) -> pd.DataFrame:
    return events[events["pattern"] == col] if len(events) else events


def assess(inputs: Sequence[HealthInput], as_of, cfg=None, research: Sequence[ResearchAssignment] = ()) -> list[HealthRecord]:
    """Health record of every item at `as_of` (only outcomes dated strictly before it are used). See the module docstring for
    the sources and the precedence: DORMANT > UNKNOWN > INSUFFICIENT_EVIDENCE > BROKEN > CONTRADICTED > RECOVERING > UNSTABLE >
    DEGRADING > HEALTHY. Every condition that holds is kept in `flags`."""
    P = _cfg(cfg)
    for inp in inputs:
        errs = inp.validate()
        if errs:
            raise ValueError(f"{inp.knowledge_id}: " + "; ".join(errs))
    tl, names = _wide(inputs, as_of)
    hl = PR.health_monitor(tl, {"arl0": P["arl0"], "half_life": P["half_life"], "est_win": P["est_win"]},
                           expected={c: inp.expected_effect for c, inp in zip(names, inputs) if inp.expected_effect is not None})
    last = len(tl.rets) - 1
    by_item = {}
    for r in research:
        if r.active():
            for k in r.item_ids:
                by_item.setdefault(k, []).append(r.research_id)
    code_hash = current_code_hash()
    out = []
    for col, inp in zip(names, inputs):
        out.append(_assess_one(inp, col, tl, hl, last, as_of, P, tuple(sorted(by_item.get(inp.knowledge_id, ()))), code_hash))
    return out


def _assess_one(inp: HealthInput, col: str, tl, hl, last: int, as_of, P, investigating, code_hash) -> HealthRecord:
    cut = pd.Timestamp(as_of)
    s = inp.values.astype(float)
    keep = [pd.Timestamp(i) < cut for i in s.index]
    s = s[keep]
    expo = None if inp.exposure is None else inp.exposure[keep].astype(bool).values
    fin = s.dropna()
    n = len(fin)
    j = list(tl.rets.columns).index(col)
    code = int(hl.codes.iat[last, j])
    ev: dict[str, Any] = {"n_obs": n, "monitor_code": code, "cusum": hl.cusum.iat[last, j], "cusum_alarm": hl.h.iloc[j],
                          "p_still_works": hl.pwork.iat[last, j], "recent_mean": hl.recent.iat[last, j],
                          "expected_effect": hl.mu0.iloc[j], "retirement_state": inp.retirement_state,
                          "open_contradictions": [c.contradiction_id for c in inp.contradictions
                                                  if not c.resolved and c.strength >= P["contradiction_bar"]]}
    flags: list[str] = []
    reasons: list[str] = []
    tr = LC.trace(s.values, exposure=expo)
    stage = tr.current if len(tr) else Lifecycle.BIRTH
    ev["lifecycle_stage"] = stage.value
    ev["t_trail"] = tr.t_trail[-1] if len(tr) else float("nan")
    epis = _episodes(hl.events, col)
    phantom = bool(len(epis) and epis["phantom"].astype(bool).any() and code == PR.BROKEN)
    recent_ep = epis[(epis["detect"] > last - P["unstable_lookback"])] if len(epis) else epis
    ev["episodes_recent"] = int(len(recent_ep))
    ev["phantom"] = phantom
    just_recovered = bool(len(epis) and (epis["recover"].dropna() > last - P["recovery_window"]).any() and code != PR.BROKEN)
    flips = _flip_count(hl.codes.iloc[max(0, last - P["flip_window"]):last + 1, j].values)
    ev["flips_recent"] = flips
    fresh_gap = (cut - pd.Timestamp(fin.index[-1])).days if n else None
    ev["days_since_outcome"] = fresh_gap

    if inp.retirement_state in ("DORMANT", "RETIRED"):
        flags.append("retired" if inp.retirement_state == "RETIRED" else "parked")
    if stage == Lifecycle.DORMANT:
        flags.append("stopped_firing")
    if inp.retirement_state == "DEGRADED":
        flags.append("ledger_degraded")
    if ev["open_contradictions"]:
        flags.append("contradicted")
    if phantom:
        flags.append("phantom")
    if code == PR.BROKEN:
        flags.append("monitor_broken")
    if code == PR.SUSPECT:
        flags.append("suspect")
    if stage == Lifecycle.DECAY:
        flags.append("decaying")
    if stage == Lifecycle.FAILURE:
        flags.append("lifecycle_failure")
    if stage == Lifecycle.RECOVERY or just_recovered:
        flags.append("recovering")
    if len(recent_ep) >= P["unstable_events"] or flips >= P["flip_count"]:
        flags.append("unstable")
    pw = ev["p_still_works"]
    if isinstance(pw, float) and np.isfinite(pw) and pw < P["p_degrade"] and code not in (0, PR.BROKEN):
        flags.append("posterior_low")

    if "retired" in flags or "parked" in flags or "stopped_firing" in flags:
        state = Health.DORMANT
        reasons.append("retired or parked by the ledger" if ("retired" in flags or "parked" in flags)
                       else "the item has stopped firing; it must not influence live decisions")
    elif n == 0 or (inp.should_fire and fresh_gap is not None and fresh_gap > P["stale_days"]):
        state = Health.UNKNOWN
        reasons.append("no outcomes at all" if n == 0 else f"newest outcome is {fresh_gap} days old while the item should be firing")
    elif n < P["min_n"] or code == 0:
        state = Health.INSUFFICIENT_EVIDENCE
        reasons.append(f"{n} outcomes; the sequential monitor needs {P['min_n']} and an established effect")
    elif "monitor_broken" in flags or "lifecycle_failure" in flags:
        state = Health.BROKEN
        reasons.append("effect was never established (phantom)" if phantom else
                       f"evidence has turned against it (CUSUM {ev['cusum']:.1f} vs alarm {ev['cusum_alarm']:.1f})"
                       if "monitor_broken" in flags else "lifecycle trace is in FAILURE")
    elif "contradicted" in flags:
        state = Health.CONTRADICTED
        reasons.append(f"open contradiction(s): {', '.join(ev['open_contradictions'])}")
    elif "recovering" in flags:
        state = Health.RECOVERING
        reasons.append("recent evidence is positive again after a break; on probation")
    elif "unstable" in flags:
        state = Health.UNSTABLE
        reasons.append(f"{len(recent_ep)} breaks in the last {P['unstable_lookback']} rows and {flips} status flips recently")
    elif {"suspect", "decaying", "ledger_degraded", "posterior_low"} & set(flags):
        state = Health.DEGRADING
        why = {"suspect": "the CUSUM is above the suspect share of its alarm", "decaying": "trailing effect has decayed from its peak",
               "ledger_degraded": "the retirement ledger has it DEGRADED", "posterior_low": f"P(still works)={pw:.2f}"}
        reasons.extend(why[f] for f in ("suspect", "decaying", "ledger_degraded", "posterior_low") if f in flags)
    else:
        state = Health.HEALTHY
        reasons.append("monitored, established and within its expected range")
    if state != Health.CONTRADICTED and "contradicted" in flags:
        reasons.append("also carries an open contradiction")
    weight = TRUST_WEIGHT[state]
    return HealthRecord(inp.knowledge_id, str(as_date(as_of)), state, tuple(flags), tuple(reasons), ev, weight >= 1.0, weight,
                        investigating, None, str(as_date(as_of)), code_hash)


def _flip_count(codes: np.ndarray) -> int:
    c = [int(x) for x in codes if int(x) != 0]
    return int(sum(1 for a, b in zip(c[:-1], c[1:]) if a != b))


# ------------------------------------------------------------------------------------------------- change and explanation

def explain_change(prev: HealthRecord | None, cur: HealthRecord, tol: float = 0.10) -> list[str]:
    """What evidence caused the move from `prev` to `cur`: the numbers that changed by more than `tol` (relative), new or
    cleared contradictions, new flags. Empty when nothing moved."""
    if prev is None:
        return [f"first assessment: {cur.state.value}"]
    out = []
    if prev.state != cur.state:
        out.append(f"state {prev.state.value} -> {cur.state.value}")
    for key, label in (("cusum", "CUSUM"), ("p_still_works", "P(still works)"), ("recent_mean", "recent mean"), ("n_obs", "outcomes")):
        a, b = prev.evidence.get(key), cur.evidence.get(key)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        if abs(b - a) > tol * max(abs(a), 1e-9) and abs(b - a) > 1e-9:
            out.append(f"{label} {a:.4g} -> {b:.4g}")
    new_c = set(cur.evidence.get("open_contradictions", [])) - set(prev.evidence.get("open_contradictions", []))
    gone_c = set(prev.evidence.get("open_contradictions", [])) - set(cur.evidence.get("open_contradictions", []))
    if new_c:
        out.append(f"new contradiction(s): {sorted(new_c)}")
    if gone_c:
        out.append(f"contradiction(s) resolved: {sorted(gone_c)}")
    for f in sorted(set(cur.flags) - set(prev.flags)):
        out.append(f"new flag: {f}")
    for f in sorted(set(prev.flags) - set(cur.flags)):
        out.append(f"flag cleared: {f}")
    if prev.evidence.get("lifecycle_stage") != cur.evidence.get("lifecycle_stage"):
        out.append(f"lifecycle {prev.evidence.get('lifecycle_stage')} -> {cur.evidence.get('lifecycle_stage')}")
    return out


# ------------------------------------------------------------------------------------------------- the book

class HealthBook:
    """Append-only history of health records with a hash chain. `record` fills in prev_state and `since` from the item's own
    history, so the dashboard can say how long an item has been in its state and what it was before."""

    def __init__(self):
        self._by_item: dict[str, list[HealthRecord]] = {}
        self._chain: list[dict] = []

    def __len__(self):
        return len(self._chain)

    def record(self, rec: HealthRecord) -> HealthRecord:
        hist = self._by_item.setdefault(rec.knowledge_id, [])
        if hist and as_date(rec.as_of) <= as_date(hist[-1].as_of):
            raise ValueError(f"{rec.knowledge_id}: records must move forward in time ({rec.as_of} <= {hist[-1].as_of})")
        prev = hist[-1] if hist else None
        since = rec.as_of if prev is None or prev.state != rec.state else prev.since
        rec = dataclasses.replace(rec, prev_state=None if prev is None else prev.state, since=since)
        hist.append(rec)
        p = self._chain[-1]["chain"] if self._chain else "genesis"
        self._chain.append({"record_id": rec.record_id, "prev": p, "chain": stable_hash([p, rec.record_id], 20)})
        return rec

    def latest(self, kid: str, as_of=None) -> HealthRecord | None:
        hist = self._by_item.get(kid, [])
        if as_of is not None:
            hist = [r for r in hist if as_date(r.as_of) <= as_date(as_of)]
        return hist[-1] if hist else None

    def previous(self, kid: str, as_of=None) -> HealthRecord | None:
        hist = self._by_item.get(kid, [])
        if as_of is not None:
            hist = [r for r in hist if as_date(r.as_of) <= as_date(as_of)]
        return hist[-2] if len(hist) >= 2 else None

    def history(self, kid: str) -> list[HealthRecord]:
        return list(self._by_item.get(kid, []))

    def items(self) -> list[str]:
        return sorted(self._by_item)

    def verify(self) -> list[str]:
        errs, prev, i = [], "genesis", 0
        for row in self._chain:
            if row["prev"] != prev or row["chain"] != stable_hash([prev, row["record_id"]], 20):
                errs.append(f"chain broken at {i}")
            prev = row["chain"]
            i += 1
        flat = [r.record_id for k in self._by_item for r in self._by_item[k]]
        if sorted(flat) != sorted(r["record_id"] for r in self._chain):
            errs.append("records and chain disagree")
        return errs

    def snapshot(self, as_of=None) -> list[HealthRecord]:
        return [r for r in (self.latest(k, as_of) for k in self.items()) if r is not None]

    def counts(self, as_of=None) -> dict:
        out = {h.value: 0 for h in Health}
        for r in self.snapshot(as_of):
            out[r.state.value] += 1
        return out

    def state_history_table(self) -> pd.DataFrame:
        rows = [{"knowledge_id": r.knowledge_id, "as_of": r.as_of, "state": r.state.value} for k in self.items() for r in self._by_item[k]]
        return pd.DataFrame(rows, columns=["knowledge_id", "as_of", "state"])

    def transition_counts(self) -> pd.DataFrame:
        counts: dict[tuple, int] = {}
        for k in self.items():
            h = self._by_item[k]
            for a, b in zip(h[:-1], h[1:]):
                if a.state != b.state:
                    counts[(a.state.value, b.state.value)] = counts.get((a.state.value, b.state.value), 0) + 1
        return pd.DataFrame([{"from": a, "to": b, "n": n} for (a, b), n in sorted(counts.items())], columns=["from", "to", "n"])


class HealthMonitor:
    """Runs `assess` at successive dates and records into a HealthBook, so health is tracked continuously rather than on demand."""

    def __init__(self, cfg=None):
        self.cfg = _cfg(cfg)
        self.book = HealthBook()

    def step(self, inputs: Sequence[HealthInput], as_of, research: Sequence[ResearchAssignment] = ()) -> list[HealthRecord]:
        recs = assess(inputs, as_of, self.cfg, research)
        return [self.book.record(r) for r in recs]

    def run(self, inputs: Sequence[HealthInput], dates: Sequence[Any], research: Sequence[ResearchAssignment] = ()) -> HealthBook:
        for d in dates:
            self.step(inputs, d, research)
        return self.book


# ------------------------------------------------------------------------------------------------- weights and silence

def apply_trust(weights: Mapping[str, float], records: Sequence[HealthRecord]) -> dict[str, float]:
    """Scale each proposed weight by the item's trust weight; an item with no record at all gets zero (unknown is not free)."""
    tw = {r.knowledge_id: r.trust_weight for r in records}
    return {k: float(w) * tw.get(k, 0.0) for k, w in weights.items()}


def assert_untrusted_silent(weights: Mapping[str, float], records: Sequence[HealthRecord]) -> None:
    """Fail closed if a BROKEN / CONTRADICTED / DORMANT / UNKNOWN / INSUFFICIENT_EVIDENCE item carries live weight."""
    state = {r.knowledge_id: r.state for r in records}
    bad = sorted(k for k, w in weights.items() if abs(w) > 1e-12 and (k not in state or state[k] in SILENT))
    if bad:
        raise FirewallBreach(f"untrusted knowledge carries live weight: {bad[:8]}")


# ------------------------------------------------------------------------------------------------- dashboard

def _entry(rec: HealthRecord, prev: HealthRecord | None, tol: float) -> dict:
    return {"knowledge_id": rec.knowledge_id, "state": rec.state.value, "since": rec.since,
            "prev_state": None if rec.prev_state is None else rec.prev_state.value, "why": list(rec.reasons),
            "flags": list(rec.flags), "evidence": _clean({k: rec.evidence.get(k) for k in
                                                         ("n_obs", "cusum", "cusum_alarm", "p_still_works", "recent_mean",
                                                          "expected_effect", "lifecycle_stage", "open_contradictions", "phantom")}),
            "evidence_changed": explain_change(prev, rec, tol), "investigating": list(rec.investigating),
            "trust_weight": rec.trust_weight}


def build_dashboard(book: HealthBook, as_of, research: Sequence[ResearchAssignment] = (), cfg=None) -> dict:
    """The machine-readable dashboard at `as_of`. Sections are disjoint in item terms except `losing_trust`, which is a view
    across states (items that moved to a worse state since the previous assessment, or currently sit in DEGRADING / UNSTABLE / BROKEN / CONTRADICTED)."""
    P = _cfg(cfg)
    recs = book.snapshot(as_of)
    prev = {r.knowledge_id: book.previous(r.knowledge_id, as_of) for r in recs}
    entries = {r.knowledge_id: _entry(r, prev[r.knowledge_id], P["change_tol"]) for r in recs}
    sec = {name: [] for name in ("trusted", "recovering", "degrading", "unstable", "broken", "contradicted", "dormant",
                                 "insufficient_evidence", "unknown")}
    key = {Health.HEALTHY: "trusted", Health.RECOVERING: "recovering", Health.DEGRADING: "degrading", Health.UNSTABLE: "unstable",
           Health.BROKEN: "broken", Health.CONTRADICTED: "contradicted", Health.DORMANT: "dormant",
           Health.INSUFFICIENT_EVIDENCE: "insufficient_evidence", Health.UNKNOWN: "unknown"}
    for r in recs:
        sec[key[r.state]].append(entries[r.knowledge_id])
    losing = []
    for r in recs:
        p = prev[r.knowledge_id]
        worse = p is not None and SEVERITY[r.state] > SEVERITY[p.state]
        if worse or r.state in LOSING:
            losing.append(entries[r.knowledge_id])
    losing.sort(key=lambda e: (-SEVERITY[Health(e["state"])], e["knowledge_id"]))
    failing = [e for e in entries.values() if Health(e["state"]) in LOSING]
    unattended = [e["knowledge_id"] for e in failing if not e["investigating"]]
    return {"schema": SCHEMA_VERSION, "as_of": str(as_date(as_of)), "code_hash": current_code_hash(), "n_items": len(recs),
            "counts": book.counts(as_of), "sections": sec, "losing_trust": losing, "unattended_failures": sorted(unattended),
            "research": [{"research_id": r.research_id, "question": r.question, "items": list(r.item_ids), "status": r.status}
                         for r in research],
            "status": "IMPLEMENTED - NOT VALIDATED"}


def validate_dashboard(d: Mapping[str, Any]) -> list[str]:
    """Schema and consistency check of an exported dashboard. Empty list = valid."""
    errs = []
    for k in ("schema", "as_of", "n_items", "counts", "sections", "losing_trust", "unattended_failures", "research"):
        if k not in d:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if d["schema"] != SCHEMA_VERSION:
        errs.append(f"unknown schema {d['schema']}")
    seen = []
    for name, lst in d["sections"].items():
        for e in lst:
            seen.append(e["knowledge_id"])
            for req in ("state", "why", "evidence", "investigating", "since"):
                if req not in e:
                    errs.append(f"{e.get('knowledge_id')}: entry lacks {req}")
            if not e.get("why"):
                errs.append(f"{e.get('knowledge_id')}: no reason given")
    if len(seen) != len(set(seen)):
        errs.append("an item appears in two sections")
    if len(seen) != d["n_items"]:
        errs.append(f"sections hold {len(seen)} items but n_items={d['n_items']}")
    if sum(d["counts"].values()) != d["n_items"]:
        errs.append("counts do not sum to n_items")
    known = {r["research_id"] for r in d["research"]}
    for name, lst in d["sections"].items():
        for e in lst:
            for rid in e.get("investigating", []):
                if rid not in known:
                    errs.append(f"{e['knowledge_id']}: research {rid} is not in the research list")
    try:
        json.dumps(d, allow_nan=False)
    except ValueError as err:
        errs.append(f"not JSON-safe: {err}")
    return errs


def write_dashboard(path, dash: Mapping[str, Any]) -> Path:
    """Write the dashboard as strict JSON. Refuses to write an invalid one."""
    errs = validate_dashboard(dash)
    if errs:
        raise ValueError("invalid dashboard: " + "; ".join(errs[:5]))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dash, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return p


def render_text(dash: Mapping[str, Any]) -> str:
    """The dashboard as a plain-language page: trusted, losing trust (with why, evidence, research), the rest by state."""
    lines = [f"# Knowledge health as of {dash['as_of']}  ({dash['status']})", "",
             "counts: " + ", ".join(f"{k}={v}" for k, v in dash["counts"].items() if v)]
    lines += ["", f"## Trusted ({len(dash['sections']['trusted'])})"]
    lines += [f"- {e['knowledge_id']}: {'; '.join(e['why'])}" for e in dash["sections"]["trusted"]]
    lines += ["", f"## Losing trust ({len(dash['losing_trust'])})"]
    for e in dash["losing_trust"]:
        rs = ", ".join(e["investigating"]) or "NOBODY"
        lines.append(f"- {e['knowledge_id']} [{e['state']}] since {e['since']}: {'; '.join(e['why'])}")
        lines.append(f"    evidence changed: {'; '.join(e['evidence_changed']) or 'n/a'}")
        lines.append(f"    research investigating: {rs}")
    for name in ("recovering", "broken", "contradicted", "dormant", "insufficient_evidence", "unknown"):
        lst = dash["sections"][name]
        if lst:
            lines += ["", f"## {name.replace('_', ' ').title()} ({len(lst)})"]
            lines += [f"- {e['knowledge_id']}: {'; '.join(e['why'])}" for e in lst]
    if dash["unattended_failures"]:
        lines += ["", "## Failing with no research assigned", *[f"- {k}" for k in dash["unattended_failures"]]]
    return "\n".join(lines)


def phantom_items(recs: Sequence[HealthRecord]) -> list[str]:
    """Items the monitor caught as phantoms: an effect that was never established, so it was never trusted."""
    return sorted(r.knowledge_id for r in recs if r.evidence.get("phantom"))


def false_alarm_expectation(records: Sequence[HealthRecord], n_periods: int, arl0: int | None = None) -> float:
    """Expected number of false BROKEN alarms if every monitored item were healthy: monitored item-periods / ARL0."""
    a = arl0 or PARAMS["arl0"]
    monitored = sum(1 for r in records if r.evidence.get("monitor_code", 0) != 0)
    return monitored * n_periods / float(a)
