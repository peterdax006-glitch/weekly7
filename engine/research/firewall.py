"""The research-side / trader-side firewall (contract C66 sections 29, 30, 31; canon C56, C64, C66; checklist D, RT07-RT09).
IMPLEMENTED — NOT VALIDATED (C63: code + unit tests only; no real-data run was made).

RESEARCH SIDE (engine.research.namespaces.ResearchStore) may study matured outcomes, real years, tickers and causes. TRADER SIDE
may only see point-in-time data and matured knowledge whose maturity precedes the decision. This module is the ONE road between
them: `ResearchTraderFirewall.release` (public entry for the wave-2 loop: `run_day`). Every research object asked for is checked
on every leak channel of section 29/30 and the whole release FAILS CLOSED on any breach:

  filing / provenance   the object must be filed, unmodified, in MATURED_RESEARCH_STATE, with complete provenance; memory_firewall
                        .could_exist_at judges existence and ancestry (a child of a future parent is future)
  maturity              the object AND its whole lineage must be first-usable on/before `now`, and core.MaturedRecord.gate(now)
                        must pass (evidence strictly before now); the channel named is that of the kind that set the date
                        (future price, label, event, filing, pattern status, experiment result, learned state, research result)
  same-year rerun       research filed under a real year that is being replayed in disguise is never released (C55/C64 leak)
  projection            only `features`/`lean`/`horizon` cross, as a trader_view.TraderMemoryItem with a content token for an id;
                        audit tags, rationale and real identities stay behind. trader_view.find_violations names year leaks
  identity/fingerprint  tickers, real names of the replayed window, exact numeric fingerprints of the window's hidden values,
                        dataset structure (universe size, session count) and training-run markers are refused
  cache / shared state  cache keys that name a year or the research store; payloads that reach a namespace store
  hindsight             knowability / unknown-cause classes may reach TRAINING only through `TrainingGate`, never a decision

Also here: the static and runtime import guards (the trader path never imports engine.research), the section-30 disguise audits
(year blindness under a whole-week calendar shift, run-index invariance, text leaks), a RESEARCH layer for
engine.learning.firewalls.LearningFirewallGate, a hash-chained trusted-side audit trail, and the planted-leak suite.

Built on: trader_view (find_violations, TraderMemoryItem/TraderRelease, opaque_token, release_year_hits, path_violations,
trader_path_violations/closure, indistinguishable), memory_firewall (could_exist_at, views_as_of, hidden_label_overlap,
identity_key_share), firewalls (Finding, FirewallLayer, GateContext, default_layers), future_firewall.text_leak_findings,
research.core (MaturedRecord, Knowability, Availability). No new firewall family: one research LAYER plus the namespace road."""
from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import importlib.abc
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import firewalls as FW
from engine.learning import memory_firewall as MF
from engine.learning import trader_view as TV
from engine.learning.core import FirewallBreach, Provenance, _StrEnum, as_date, stable_hash
from engine.research.core import Availability, Knowability, Namespace
from engine.research.namespaces import (InfoKind, InfoObject, LiveStore, NamespacePair, ResearchStore, SharedStateError,
                                        _ISO_INSIDE, moment, reachable_stores, sign_ticket, ticket_valid)

LABEL = "IMPLEMENTED — NOT VALIDATED"


class LeakChannel(_StrEnum):               # sections 29-31 (+ the section-30 disguise channels)
    FUTURE_PRICE = "FUTURE_PRICE"
    FUTURE_VOLUME = "FUTURE_VOLUME"
    FUTURE_FILING = "FUTURE_FILING"
    FUTURE_EARNINGS = "FUTURE_EARNINGS"
    FUTURE_LABEL = "FUTURE_LABEL"
    FUTURE_METADATA = "FUTURE_METADATA"
    FUTURE_CORPORATE_ACTION = "FUTURE_CORPORATE_ACTION"
    FUTURE_MARKET_STATE = "FUTURE_MARKET_STATE"
    FUTURE_EVENT = "FUTURE_EVENT"
    FUTURE_LEARNED_STATE = "FUTURE_LEARNED_STATE"
    FUTURE_RESEARCH_RESULT = "FUTURE_RESEARCH_RESULT"
    FUTURE_PATTERN_STATUS = "FUTURE_PATTERN_STATUS"
    FUTURE_PATTERN_HEALTH = "FUTURE_PATTERN_HEALTH"
    FUTURE_EXPERIMENT_RESULT = "FUTURE_EXPERIMENT_RESULT"
    YEAR_IDENTITY = "YEAR_IDENTITY"
    SAME_YEAR_RERUN = "SAME_YEAR_RERUN"
    TICKER_IDENTITY = "TICKER_IDENTITY"
    NUMERIC_FINGERPRINT = "NUMERIC_FINGERPRINT"
    DATASET_STRUCTURE = "DATASET_STRUCTURE"
    TRAINING_RUN_MARKER = "TRAINING_RUN_MARKER"
    HIDDEN_CACHE = "HIDDEN_CACHE"
    IMPLICIT_SHARED_STATE = "IMPLICIT_SHARED_STATE"
    CONVENIENCE_IMPORT = "CONVENIENCE_IMPORT"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    TIMESTAMP_AMBIGUITY = "TIMESTAMP_AMBIGUITY"
    HINDSIGHT_LABEL = "HINDSIGHT_LABEL"
    RESEARCH_ONLY_KNOWLEDGE = "RESEARCH_ONLY_KNOWLEDGE"
    MALFORMED = "MALFORMED"


# the six leaks section 31 names, plus the two the brief adds (future learned state, future research results)
REQUIRED_CHANNELS = (LeakChannel.FUTURE_PRICE, LeakChannel.FUTURE_LABEL, LeakChannel.FUTURE_EVENT, LeakChannel.YEAR_IDENTITY,
                     LeakChannel.FUTURE_PATTERN_STATUS, LeakChannel.FUTURE_EXPERIMENT_RESULT, LeakChannel.FUTURE_LEARNED_STATE,
                     LeakChannel.FUTURE_RESEARCH_RESULT)

KIND_CHANNEL = types.MappingProxyType({
    InfoKind.PRICE: LeakChannel.FUTURE_PRICE, InfoKind.VOLUME: LeakChannel.FUTURE_VOLUME, InfoKind.FILING: LeakChannel.FUTURE_FILING,
    InfoKind.EARNINGS: LeakChannel.FUTURE_EARNINGS, InfoKind.LABEL: LeakChannel.FUTURE_LABEL,
    InfoKind.METADATA: LeakChannel.FUTURE_METADATA, InfoKind.CORPORATE_ACTION: LeakChannel.FUTURE_CORPORATE_ACTION,
    InfoKind.MARKET_STATE: LeakChannel.FUTURE_MARKET_STATE, InfoKind.EVENT: LeakChannel.FUTURE_EVENT,
    InfoKind.FEATURE: LeakChannel.FUTURE_MARKET_STATE, InfoKind.CONFIG: LeakChannel.FUTURE_METADATA,
    InfoKind.LEARNED_STATE: LeakChannel.FUTURE_LEARNED_STATE, InfoKind.RESEARCH_RESULT: LeakChannel.FUTURE_RESEARCH_RESULT,
    InfoKind.PATTERN_STATUS: LeakChannel.FUTURE_PATTERN_STATUS, InfoKind.PATTERN_HEALTH: LeakChannel.FUTURE_PATTERN_HEALTH,
    InfoKind.EXPERIMENT_RESULT: LeakChannel.FUTURE_EXPERIMENT_RESULT, InfoKind.YEAR_IDENTITY: LeakChannel.YEAR_IDENTITY})

_L = FW.LayerName
CHANNEL_LAYER = types.MappingProxyType({
    **{c: _L.TIME for c in (LeakChannel.FUTURE_PRICE, LeakChannel.FUTURE_VOLUME, LeakChannel.FUTURE_FILING,
                            LeakChannel.FUTURE_EARNINGS, LeakChannel.FUTURE_LABEL, LeakChannel.FUTURE_METADATA,
                            LeakChannel.FUTURE_CORPORATE_ACTION, LeakChannel.FUTURE_MARKET_STATE, LeakChannel.FUTURE_EVENT)},
    **{c: _L.MEMORY for c in (LeakChannel.FUTURE_LEARNED_STATE, LeakChannel.FUTURE_RESEARCH_RESULT, LeakChannel.FUTURE_PATTERN_STATUS,
                              LeakChannel.FUTURE_PATTERN_HEALTH, LeakChannel.FUTURE_EXPERIMENT_RESULT, LeakChannel.HINDSIGHT_LABEL,
                              LeakChannel.RESEARCH_ONLY_KNOWLEDGE)},
    **{c: _L.IDENTITY for c in (LeakChannel.YEAR_IDENTITY, LeakChannel.SAME_YEAR_RERUN, LeakChannel.TICKER_IDENTITY,
                                LeakChannel.NUMERIC_FINGERPRINT, LeakChannel.DATASET_STRUCTURE, LeakChannel.TRAINING_RUN_MARKER)},
    **{c: _L.PROVENANCE for c in (LeakChannel.MISSING_PROVENANCE, LeakChannel.TIMESTAMP_AMBIGUITY)},
    **{c: _L.DATA for c in (LeakChannel.HIDDEN_CACHE, LeakChannel.IMPLICIT_SHARED_STATE, LeakChannel.MALFORMED)},
    LeakChannel.CONVENIENCE_IMPORT: _L.CODE_VERSION})

TIME_TOKENS = frozenset({"year", "years", "yr", "date", "dates", "datetime", "day_of_year", "era", "eras", "epoch", "calendar",
                         "month", "quarter", "timestamp", "ts", "asof", "filed", "matured", "occurred", "vintage", "regime_year"})
STRUCTURE_TOKENS = frozenset({"universe", "names", "tickers", "sessions", "session", "rows", "row", "columns", "column", "cols",
                              "shape", "position", "index", "idx", "count", "counts", "window", "windows", "panel", "dataset"})
RUN_TOKENS = frozenset({"run", "runs", "rerun", "reruns", "replay", "attempt", "seed", "training", "train", "is_training",
                        "first_run", "pass", "fold", "trial", "episode", "iteration", "seen_before", "repeat"})
HINDSIGHT_TOKENS = frozenset({"knowability", "hindsight", "external_cause", "unpredictable", "known_only_after", "could_have_known",
                              "unknown_cause", "post_event", "after_event", "cause"})
HINDSIGHT_VALUES = frozenset({str(k) for k in Knowability} | {str(a) for a in Availability} - {str(Availability.KNOWN_BEFORE_EVENT)})
CACHE_MARKERS = ("matured_research", "research_store", "curator_store", "curator_audit", "hindsight", "outcomes")
PROJECTED_FIELDS = ("features", "lean", "horizon", "trader_kind")
_TICKER = MF._TICKER                           # one ticker shape for the whole repo


# ------------------------------------------------------------------------------------------------ records
@dataclasses.dataclass(frozen=True)
class ReplayContext:
    """What the TRUSTED side knows about the window the trader is replaying (never handed to the trader). `hidden_values` are
    audit fingerprints of the window (its daily market returns, a few prices) that no released number may equal exactly."""
    window_id: str
    real_start: str
    real_end: str
    run_index: int = 0
    step: int = 0
    session_count: int | None = None
    universe_size: int | None = None
    hidden_values: tuple[float, ...] = ()
    real_tickers: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        errs = []
        try:
            s, e = moment(self.real_start, "replay.real_start"), moment(self.real_end, "replay.real_end")
            if e < s:
                errs.append("replay window ends before it starts")
        except FirewallBreach as err:
            errs.append(str(err))
        if not self.window_id:
            errs.append("replay window_id missing")
        if self.run_index < 0 or self.step < 0:
            errs.append("run_index and step must be non-negative")
        return errs

    def bounds(self) -> tuple[dt.date, dt.date]:
        return moment(self.real_start, "replay.real_start"), moment(self.real_end, "replay.real_end")

    def years(self) -> frozenset:
        a, b = self.bounds()
        return frozenset(range(a.year, b.year + 1))


@dataclasses.dataclass(frozen=True)
class Breach:
    channel: LeakChannel
    check: str
    object_id: str
    message: str
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict, compare=False, hash=False)

    def __str__(self):
        return f"[{self.channel}] {self.check} ({self.object_id}): {self.message}"

    def to_finding(self) -> FW.Finding:
        return FW.fail(CHANNEL_LAYER[self.channel], f"research.{self.channel.value.lower()}.{self.check}", self.object_id,
                       self.message, channel=self.channel.value, **dict(self.evidence))

    def to_dict(self) -> dict:
        return {"channel": self.channel.value, "check": self.check, "object_id": self.object_id, "message": self.message}


@dataclasses.dataclass(frozen=True)
class Decision:
    """The firewall's answer for one object at one `now`. `admitted` is True only when no check found a breach."""
    object_id: str
    now: str
    breaches: tuple[Breach, ...]
    n_checks: int

    @property
    def admitted(self) -> bool:
        return not self.breaches and self.n_checks > 0

    @property
    def channels(self) -> frozenset:
        return frozenset(b.channel for b in self.breaches)

    def require(self) -> "Decision":
        if not self.admitted:
            raise FirewallBreach(f"research object {self.object_id} refused at {self.now}: "
                                 + "; ".join(str(b) for b in self.breaches[:3]) + (f" (+{len(self.breaches) - 3})" if len(self.breaches) > 3 else ""))
        return self

    def digest(self) -> str:
        return stable_hash({"id": self.object_id, "now": self.now, "b": [b.to_dict() for b in self.breaches], "n": self.n_checks})


@dataclasses.dataclass(frozen=True)
class FirewallPolicy:
    fingerprint_decimals: int = 8          # an exact match to this many decimals is not a coincidence ...
    fingerprint_min_digits: int = 4        # ... when the number has at least this many significant digits
    overlap_min_numbers: int = 8           # hidden_label_overlap on the projection: minimum numbers before a share is judged
    overlap_max_share: float = 0.3
    structure_min_count: int = 10          # integers equal to the window's universe/session count, at least this big
    max_identity_key_share: float = 0.0    # projected feature keys that look like tickers/dates
    strict: bool = True                    # one refused object refuses the whole release (fail closed)
    memory_policy: MF.MemoryPolicy = MF.MemoryPolicy(require_data_hash=False, require_experiment_id=False,
                                                     forbid_label_payload=False, forbid_identity_contexts=True)

    def validate(self) -> list[str]:
        errs = []
        if not 1 <= self.fingerprint_decimals <= 15:
            errs.append("fingerprint_decimals must be in 1..15")
        if self.fingerprint_min_digits < 1 or self.overlap_min_numbers < 1 or self.structure_min_count < 1:
            errs.append("minimum counts must be >= 1")
        if not 0.0 <= self.overlap_max_share <= 1.0 or not 0.0 <= self.max_identity_key_share <= 1.0:
            errs.append("shares must be in [0, 1]")
        return errs


# ------------------------------------------------------------------------------------------------ the trader projection
def projection_dict(obj: InfoObject) -> dict:
    """The ONLY part of a research object that may cross: its features, lean, horizon and trader kind. Everything else in the
    payload (hit rates, sample sizes, notes, rationale) and every tag stays on the research side."""
    p = obj.payload
    if not isinstance(p, Mapping) or not isinstance(p.get("features"), Mapping):
        raise FirewallBreach(f"{obj.object_id}: payload has no `features` mapping; nothing trader-shaped can be projected")
    return {"features": dict(p["features"]), "lean": p.get("lean", 0.0), "horizon": p.get("horizon", 5),
            "kind": p.get("trader_kind", "pattern")}


def project(obj: InfoObject, weight: float = 1.0) -> TV.TraderMemoryItem:
    d = projection_dict(obj)
    token = TV.opaque_token({"f": d["features"], "l": d["lean"], "h": d["horizon"], "k": d["kind"]})
    return TV.TraderMemoryItem.make(token, d["kind"], weight, d["features"], d["lean"], d["horizon"])


def _numbers(obj: Any) -> list[float]:
    return [float(v) for v in MF.payload_numbers(obj).tolist()]


def _sig_digits(x: float, decimals: int) -> int:
    s = f"{abs(x):.{decimals}f}".replace(".", "").lstrip("0").rstrip("0")
    return len(s)


def _key_tokens(k: str) -> list[str]:
    return TV._key_tokens(k)


def _walk_keys_and_strings(obj: Any, depth: int = 0) -> Iterable[tuple[str, str]]:
    """(role, text) for every key and string value in a nested payload."""
    if depth > 8:
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield "key", str(k)
            yield from _walk_keys_and_strings(v, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in list(obj)[:2000]:
            yield from _walk_keys_and_strings(v, depth + 1)
    elif isinstance(obj, (str, _StrEnum)):
        yield "value", str(obj)


# ------------------------------------------------------------------------------------------------ checks
@dataclasses.dataclass
class _Ctx:
    obj: InfoObject
    now: dt.date
    replay: ReplayContext | None
    store: ResearchStore
    policy: FirewallPolicy
    filed: bool


def _validation_channel(obj: InfoObject, err: str) -> LeakChannel:
    if err.startswith("timestamp:"):
        return LeakChannel.TIMESTAMP_AMBIGUITY
    if err.startswith("provenance") or "parent" in err:
        return LeakChannel.MISSING_PROVENANCE
    if "smuggles" in err or "physically possible" in err:
        return KIND_CHANNEL.get(obj.kind, LeakChannel.FUTURE_RESEARCH_RESULT)
    if "audit tags" in err or "may exist only" in err:
        return LeakChannel.YEAR_IDENTITY
    return LeakChannel.MALFORMED


def check_filing(c: _Ctx) -> list[Breach]:
    """Filed, unmodified, in the research namespace, and valid. An object handed over outside the store has no audited lineage."""
    o, out = c.obj, []
    if o.namespace != Namespace.MATURED_RESEARCH:
        out.append(Breach(LeakChannel.MISSING_PROVENANCE, "wrong-namespace", o.object_id,
                          f"only {Namespace.MATURED_RESEARCH} objects cross this firewall (got {o.namespace})"))
    if not c.filed:
        out.append(Breach(LeakChannel.MISSING_PROVENANCE, "not-filed", o.object_id,
                          "not filed (or modified since filing) in the research store: provenance and lineage cannot be audited"))
    out += [Breach(_validation_channel(o, e), "invalid", o.object_id, e) for e in o.validate()]
    if o.kind == InfoKind.YEAR_IDENTITY:
        out.append(Breach(LeakChannel.YEAR_IDENTITY, "year-identity-kind", o.object_id, "a year-identity record never reaches the trader"))
    return out


def _memory_item(o: InfoObject) -> dict:
    return {"knowledge_id": o.object_id, "version": 0, "payload": None, "contexts": {}, "anti_contexts": {},
            "provenance": dataclasses.replace(o.provenance, parents=o.all_parents())}


_MF_CHANNEL = types.MappingProxyType({
    "no-provenance": LeakChannel.MISSING_PROVENANCE, "provenance-invalid": LeakChannel.MISSING_PROVENANCE,
    "learned-at-missing": LeakChannel.MISSING_PROVENANCE, "code_hash-missing": LeakChannel.MISSING_PROVENANCE,
    "data_hash-missing": LeakChannel.MISSING_PROVENANCE, "experiment_id-missing": LeakChannel.MISSING_PROVENANCE,
    "parent-missing": LeakChannel.MISSING_PROVENANCE, "lineage-cycle": LeakChannel.MISSING_PROVENANCE,
    "child-predates-parent": LeakChannel.MISSING_PROVENANCE, "sealed-window-unparseable": LeakChannel.TIMESTAMP_AMBIGUITY,
    "saw-sealed-window": LeakChannel.SAME_YEAR_RERUN, "not-before-eval-window": LeakChannel.SAME_YEAR_RERUN,
    "identity-context": LeakChannel.TICKER_IDENTITY, "code-unregistered": LeakChannel.MISSING_PROVENANCE,
    "code-postdates-record": LeakChannel.MISSING_PROVENANCE})


def check_existence(c: _Ctx) -> list[Breach]:
    """memory_firewall.could_exist_at on the object and its lineage, with the replayed window as the evaluation window."""
    o = c.obj
    pool = [_memory_item(x) for x in c.store.objects()]
    if not c.filed:
        pool.append(_memory_item(o))
    views = MF.views_as_of(pool, c.now)
    env = MF.MemoryEnvironment(eval_window=(c.replay.real_start, c.replay.real_end) if c.replay else None)
    ex = MF.could_exist_at(_memory_item(o), c.now, views, env, (), c.policy.memory_policy)
    out = []
    for f in ex.findings:
        if not f.is_fail:
            continue
        ch = _MF_CHANNEL.get(f.check)
        if ch is None and f.check == "tainted-by-parent":
            anc = f.evidence.get("ancestor")
            ch = KIND_CHANNEL.get(c.store.get(anc).kind) if anc in c.store else LeakChannel.MISSING_PROVENANCE
        out.append(Breach(ch or KIND_CHANNEL.get(o.kind, LeakChannel.FUTURE_RESEARCH_RESULT), f"exist.{f.check}", o.object_id, f.message))
    return out


def check_maturity(c: _Ctx) -> list[Breach]:
    """First-usable date of the object and every ancestor <= now, and MaturedRecord.gate(now) (evidence strictly before now)."""
    o, out = c.obj, []
    if c.filed:
        k, who = c.store.effective_knowable(o.object_id)
        rec = c.store.matured_record(o.object_id)
        setter = c.store.get(who)
    else:
        k, who, setter, rec = o.knowable_at(), o.object_id, o, None
    if k > c.now:
        out.append(Breach(KIND_CHANNEL.get(setter.kind, LeakChannel.FUTURE_RESEARCH_RESULT), "not-yet-knowable", o.object_id,
                          f"first usable {k} (set by {who}, a {setter.kind}), after now {c.now}", {"set_by": who, "knowable": str(k)}))
    if rec is not None:
        try:
            rec.gate(c.now)
        except FirewallBreach as e:
            if not out:
                out.append(Breach(KIND_CHANNEL.get(setter.kind, LeakChannel.FUTURE_RESEARCH_RESULT), "not-matured", o.object_id, str(e)))
    late = [d for d in (o.evidence_dates()) if d >= c.now]
    if late and not out:
        out.append(Breach(KIND_CHANNEL.get(o.kind, LeakChannel.FUTURE_RESEARCH_RESULT), "evidence-at-or-after-now", o.object_id,
                          f"rests on evidence dated {max(late)}, not strictly before now {c.now}"))
    return out


def check_same_year(c: _Ctx) -> list[Breach]:
    """Research filed under a real year that is being replayed may not be released while that replay runs (C55/C64)."""
    if c.replay is None:
        return []
    o = c.obj
    years = set(c.store.filed_years(o.object_id)) if c.filed else set(range(o.evidence_from().year, o.evidence_through().year + 1))
    tagged = o.tags.get("real_year") if isinstance(o.tags, Mapping) else None
    if isinstance(tagged, (int, np.integer)):
        years.add(int(tagged))
    hit = sorted(years & c.replay.years())
    out = []
    if hit:
        out.append(Breach(LeakChannel.SAME_YEAR_RERUN, "filed-under-replayed-year", o.object_id,
                          f"filed under real year(s) {hit}, which the trader is replaying in disguise (window {c.replay.window_id})",
                          {"years": hit}))
    if c.replay.window_id and c.replay.window_id in tuple(o.provenance.sealed_windows):
        out.append(Breach(LeakChannel.SAME_YEAR_RERUN, "saw-replayed-window", o.object_id,
                          f"provenance lists the replayed window {c.replay.window_id} as seen"))
    return out


def _violation_channel(v: TV.Violation) -> LeakChannel:
    cat = v.category
    if cat in ("oversized_number",):
        return LeakChannel.NUMERIC_FINGERPRINT
    if cat in ("non_finite", "unsupported_type", "non_string_key", "too_large", "too_deep"):
        return LeakChannel.MALFORMED
    if cat.endswith("forbidden_key"):
        toks = set(_key_tokens(v.detail))
        return LeakChannel.YEAR_IDENTITY if toks & TIME_TOKENS or "_at" in v.detail else LeakChannel.RESEARCH_ONLY_KNOWLEDGE
    return LeakChannel.YEAR_IDENTITY


def check_projection(c: _Ctx) -> list[Breach]:
    """What would cross must be trader-safe (trader_view.find_violations) and constructible as a TraderMemoryItem."""
    o = c.obj
    try:
        d = projection_dict(o)
    except FirewallBreach as e:
        return [Breach(LeakChannel.MALFORMED, "unprojectable", o.object_id, str(e))]
    out = [Breach(_violation_channel(v), "trader-unsafe", o.object_id, str(v)) for v in TV.find_violations(d)]
    if not out:
        try:
            project(o)
        except FirewallBreach as e:
            out.append(Breach(LeakChannel.MALFORMED, "item-refused", o.object_id, str(e)))
    return out


def check_identity(c: _Ctx) -> list[Breach]:
    """Tickers (by shape or by the replayed window's real names) anywhere in what would cross."""
    o = c.obj
    try:
        d = projection_dict(o)
    except FirewallBreach:
        return []
    out = []
    real = {t.lower() for t in (c.replay.real_tickers if c.replay else ())}
    for role, text in _walk_keys_and_strings(d):
        if _TICKER.match(text) and text not in ("pattern", "lesson", "context"):
            out.append(Breach(LeakChannel.TICKER_IDENTITY, f"ticker-{role}", o.object_id, f"{role} {text!r} has the shape of a ticker"))
        toks = set(_key_tokens(text)) | {text.lower()}
        hit = sorted(toks & real)
        if hit:
            out.append(Breach(LeakChannel.TICKER_IDENTITY, f"real-name-{role}", o.object_id,
                              f"{role} {text!r} names a real ticker of the replayed window {hit}"))
    share, n = MF.identity_key_share(d["features"])
    if n and share > c.policy.max_identity_key_share and not out:
        out.append(Breach(LeakChannel.TICKER_IDENTITY, "identity-keys", o.object_id, f"{share:.0%} of {n} feature keys are identities"))
    return out


def check_fingerprint(c: _Ctx) -> list[Breach]:
    """No projected number may equal a hidden value of the replayed window to `fingerprint_decimals` (unless it is a short, round
    number such as 0.5), and the projection as a whole may not overlap the window's values (memory_firewall.hidden_label_overlap)."""
    if c.replay is None or not c.replay.hidden_values:
        return []
    o, pol = c.obj, c.policy
    try:
        d = projection_dict(o)
    except FirewallBreach:
        return []
    hidden = {round(float(h), pol.fingerprint_decimals) for h in c.replay.hidden_values if math.isfinite(float(h))}
    out = []
    for x in _numbers({"f": d["features"], "l": d["lean"]}):
        r = round(x, pol.fingerprint_decimals)
        if r in hidden and _sig_digits(x, pol.fingerprint_decimals) >= pol.fingerprint_min_digits:
            out.append(Breach(LeakChannel.NUMERIC_FINGERPRINT, "exact-hidden-value", o.object_id,
                              f"projected number {x!r} equals a hidden value of the replayed window to {pol.fingerprint_decimals} decimals"))
    f = MF.hidden_label_overlap(d, list(c.replay.hidden_values), pol.fingerprint_decimals, pol.overlap_min_numbers, pol.overlap_max_share)
    if f is not None:
        out.append(Breach(LeakChannel.NUMERIC_FINGERPRINT, "hidden-overlap", o.object_id, f.message))
    return out


def check_structure(c: _Ctx) -> list[Breach]:
    """Dataset structure (universe size, session count, positions) and training-run markers must not cross (section 30)."""
    o = c.obj
    try:
        d = projection_dict(o)
    except FirewallBreach:
        return []
    out = []
    for k in d["features"]:
        toks = set(_key_tokens(str(k)))
        if toks & RUN_TOKENS:
            out.append(Breach(LeakChannel.TRAINING_RUN_MARKER, "run-marker-key", o.object_id,
                              f"feature {k!r} describes the run itself ({sorted(toks & RUN_TOKENS)})"))
        elif toks & STRUCTURE_TOKENS:
            out.append(Breach(LeakChannel.DATASET_STRUCTURE, "structure-key", o.object_id,
                              f"feature {k!r} describes the dataset's structure ({sorted(toks & STRUCTURE_TOKENS)})"))
    if c.replay is not None:
        marks = {n: v for n, v in (("universe_size", c.replay.universe_size), ("session_count", c.replay.session_count))
                 if v is not None and v >= c.policy.structure_min_count}
        for x in _numbers({"f": d["features"], "l": d["lean"]}) + [float(d["horizon"]) if isinstance(d["horizon"], (int, float)) else 0.0]:
            for n, v in marks.items():
                if float(x).is_integer() and int(x) == int(v):
                    out.append(Breach(LeakChannel.DATASET_STRUCTURE, f"equals-{n}", o.object_id,
                                      f"projected number {x!r} equals the replayed window's {n}"))
    return out


def check_cache_and_state(c: _Ctx) -> list[Breach]:
    """A cache key that names a year or a research store, and a payload that reaches a namespace store, are refused."""
    o, out = c.obj, []
    if o.cache_key:
        out += [Breach(LeakChannel.HIDDEN_CACHE, "cache-key-dated", o.object_id, f"cache key {o.cache_key!r}: {v}")
                for v in TV.path_violations(o.cache_key)]
        low = o.cache_key.lower()
        out += [Breach(LeakChannel.HIDDEN_CACHE, "cache-key-marker", o.object_id, f"cache key {o.cache_key!r} names {m!r}")
                for m in CACHE_MARKERS if m in low]
    for ref in reachable_stores(o.payload) + reachable_stores(o.tags):
        out.append(Breach(LeakChannel.IMPLICIT_SHARED_STATE, "reaches-store", o.object_id, str(ref)))
    return out


def check_hindsight(c: _Ctx) -> list[Breach]:
    """Hindsight classes (knowability, availability-after-event, unknown cause) and research-only knowledge (decision effect NONE)
    never reach a decision; hindsight may reach training only through TrainingGate, dated at maturity."""
    o, out = c.obj, []
    p = o.payload if isinstance(o.payload, Mapping) else {}
    eff = p.get("decision_effect")
    effs = [eff] if isinstance(eff, (str, _StrEnum)) else list(eff) if isinstance(eff, (list, tuple)) else []
    if any(str(e) == "NONE" for e in effs):
        out.append(Breach(LeakChannel.RESEARCH_ONLY_KNOWLEDGE, "decision-effect-none", o.object_id,
                          "marked research knowledge only (decision effect NONE): it may not reach a decision"))
    for role, text in _walk_keys_and_strings(p):
        toks = set(_key_tokens(text)) | {text.lower()}
        if role == "key" and toks & HINDSIGHT_TOKENS:
            out.append(Breach(LeakChannel.HINDSIGHT_LABEL, "hindsight-key", o.object_id, f"payload key {text!r} is a hindsight class"))
        elif role == "value" and text in HINDSIGHT_VALUES:
            out.append(Breach(LeakChannel.HINDSIGHT_LABEL, "hindsight-value", o.object_id, f"payload value {text!r} is a hindsight class"))
    return out


CHECKS: tuple[tuple[str, Callable[[_Ctx], list[Breach]]], ...] = (
    ("filing", check_filing), ("existence", check_existence), ("maturity", check_maturity), ("same_year", check_same_year),
    ("projection", check_projection), ("identity", check_identity), ("fingerprint", check_fingerprint),
    ("structure", check_structure), ("cache_state", check_cache_and_state), ("hindsight", check_hindsight))
CHECK_CHANNELS = types.MappingProxyType({
    "filing": frozenset({LeakChannel.MISSING_PROVENANCE, LeakChannel.TIMESTAMP_AMBIGUITY, LeakChannel.YEAR_IDENTITY,
                         LeakChannel.MALFORMED} | set(KIND_CHANNEL.values())),
    "existence": frozenset({LeakChannel.MISSING_PROVENANCE, LeakChannel.SAME_YEAR_RERUN, LeakChannel.TICKER_IDENTITY,
                            LeakChannel.TIMESTAMP_AMBIGUITY} | set(KIND_CHANNEL.values())),
    "maturity": frozenset(KIND_CHANNEL.values()), "same_year": frozenset({LeakChannel.SAME_YEAR_RERUN}),
    "projection": frozenset({LeakChannel.YEAR_IDENTITY, LeakChannel.NUMERIC_FINGERPRINT, LeakChannel.MALFORMED,
                             LeakChannel.RESEARCH_ONLY_KNOWLEDGE}),
    "identity": frozenset({LeakChannel.TICKER_IDENTITY}), "fingerprint": frozenset({LeakChannel.NUMERIC_FINGERPRINT}),
    "structure": frozenset({LeakChannel.DATASET_STRUCTURE, LeakChannel.TRAINING_RUN_MARKER}),
    "cache_state": frozenset({LeakChannel.HIDDEN_CACHE, LeakChannel.IMPLICIT_SHARED_STATE}),
    "hindsight": frozenset({LeakChannel.HINDSIGHT_LABEL, LeakChannel.RESEARCH_ONLY_KNOWLEDGE})})


# ------------------------------------------------------------------------------------------------ audit trail (trusted side)
@dataclasses.dataclass(frozen=True)
class AuditEntry:
    seq: int
    now: str
    release_id: str
    admitted: tuple[str, ...]
    refused: Mapping[str, tuple[str, ...]]      # object id -> channels
    outcome: str                                 # RELEASED | REFUSED | PARTIAL | EMPTY
    release_digest: str
    prev: str
    link: str

    def to_dict(self) -> dict:
        return {"seq": self.seq, "now": self.now, "release_id": self.release_id, "admitted": list(self.admitted),
                "refused": {k: list(v) for k, v in self.refused.items()}, "outcome": self.outcome,
                "release_digest": self.release_digest, "prev": self.prev, "link": self.link}


class FirewallAudit:
    """Append-only, hash-chained log of every release decision. It holds real dates and ids, so it lives on the trusted side only
    and is never handed to the trader."""

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def append(self, now, release_id: str, admitted: Sequence[str], refused: Mapping[str, Iterable], outcome: str, digest: str) -> AuditEntry:
        prev = self.entries[-1].link if self.entries else "genesis"
        ref = {k: tuple(sorted(str(c) for c in v)) for k, v in sorted(refused.items())}
        body = {"seq": len(self.entries), "now": str(as_date(now)), "r": release_id, "a": sorted(admitted), "x": ref, "o": outcome,
                "d": digest, "prev": prev}
        e = AuditEntry(len(self.entries), str(as_date(now)), release_id, tuple(sorted(admitted)), ref, outcome, digest, prev,
                       stable_hash(body, 24))
        self.entries.append(e)
        return e

    def verify(self) -> list[str]:
        errs, prev = [], "genesis"
        for e in self.entries:
            body = {"seq": e.seq, "now": e.now, "r": e.release_id, "a": sorted(e.admitted), "x": dict(e.refused), "o": e.outcome,
                    "d": e.release_digest, "prev": prev}
            if e.prev != prev or stable_hash(body, 24) != e.link:
                errs.append(f"audit entry {e.seq} altered or out of order")
            prev = e.link
        return errs

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"seq": e.seq, "now": e.now, "release_id": e.release_id, "n_admitted": len(e.admitted),
                              "n_refused": len(e.refused), "outcome": e.outcome} for e in self.entries],
                            columns=["seq", "now", "release_id", "n_admitted", "n_refused", "outcome"])

    def channel_counts(self) -> pd.Series:
        c: dict[str, int] = {}
        for e in self.entries:
            for chans in e.refused.values():
                for ch in chans:
                    c[ch] = c.get(ch, 0) + 1
        return pd.Series(c, dtype="int64").sort_values(ascending=False) if c else pd.Series(dtype="int64")


# ------------------------------------------------------------------------------------------------ the firewall
@dataclasses.dataclass(frozen=True)
class ReleaseResult:
    """What a release produced. `release` is the only thing the trader receives; `live_objects` + `tickets` let the trusted runner
    file the released items into the live namespace (LiveStore.accept); `decisions` and `audit` stay on the trusted side."""
    release: TV.TraderRelease
    decisions: tuple[Decision, ...]
    live_objects: tuple[InfoObject, ...]
    tickets: tuple
    release_id: str

    @property
    def refused(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if not d.admitted)


class ResearchTraderFirewall:
    """The one road from MATURED_RESEARCH_STATE to the trader. Holds the research store it guards and the secret that signs its
    crossing tickets (a LiveStore built by `pair()` verifies them). Every check runs on every object (one failing check never
    hides another); a check that crashes is a MALFORMED breach (fail closed)."""

    def __init__(self, store: ResearchStore, secret: str, policy: FirewallPolicy | None = None,
                 checks: Sequence[tuple[str, Callable[[_Ctx], list[Breach]]]] = CHECKS):
        if not isinstance(store, ResearchStore):
            raise FirewallBreach("the firewall guards a ResearchStore (MATURED_RESEARCH_STATE) only")
        self.store, self._secret = store, str(secret)
        self.policy = policy or FirewallPolicy()
        errs = self.policy.validate()
        if errs or not self._secret:
            raise ValueError("; ".join(errs) or "firewall needs a non-empty secret")
        self.checks = tuple(checks)
        if not self.checks:
            raise ValueError("a firewall with no checks cannot fail and so cannot pass")
        self.audit = FirewallAudit()

    def verifier(self) -> Callable:
        secret = self._secret
        return lambda t: ticket_valid(secret, t)

    def live_store(self, name: str = "live") -> LiveStore:
        return LiveStore(name, self.verifier())

    def inspect(self, obj: InfoObject | str, now, replay: ReplayContext | None = None) -> Decision:
        if now is None:
            raise FirewallBreach("the research firewall needs an explicit `now` (no implicit clock)")
        nowd = moment(now, "now")
        oid = obj if isinstance(obj, str) else getattr(obj, "object_id", "?")
        if isinstance(obj, str):
            if obj not in self.store:
                return Decision(obj, str(nowd), (Breach(LeakChannel.MISSING_PROVENANCE, "unknown-id", obj, "no such research object"),), 1)
            obj = self.store.get(obj)
        if not isinstance(obj, InfoObject):
            return Decision(str(oid), str(nowd), (Breach(LeakChannel.MALFORMED, "not-info-object", str(oid),
                                                         f"{type(obj).__name__} is not an InfoObject"),), 1)
        if replay is not None and replay.validate():
            return Decision(obj.object_id, str(nowd), (Breach(LeakChannel.TIMESTAMP_AMBIGUITY, "replay-invalid", obj.object_id,
                                                              "; ".join(replay.validate())),), 1)
        try:
            filed = self.store.holds(obj)
        except FirewallBreach:
            filed = False
        ctx = _Ctx(obj, nowd, replay, self.store, self.policy, filed)
        out: list[Breach] = []
        for name, fn in self.checks:
            try:
                out += fn(ctx)
            except SharedStateError as e:
                out.append(Breach(LeakChannel.IMPLICIT_SHARED_STATE, f"{name}-shared-state", obj.object_id, str(e)))
            except Exception as e:                             # noqa: BLE001 - a check that cannot run must refuse, not pass
                ch = LeakChannel.TIMESTAMP_AMBIGUITY if "timestamp" in str(e).lower() or "does not name one day" in str(e) \
                    else LeakChannel.MALFORMED
                out.append(Breach(ch, f"{name}-error", obj.object_id, f"{type(e).__name__}: {e}"))
        if not filed and not any(b.check == "not-filed" for b in out):     # structural: no check list can waive filing
            out.append(Breach(LeakChannel.MISSING_PROVENANCE, "not-filed", obj.object_id, "not filed in the research store"))
        return Decision(obj.object_id, str(nowd), tuple(dict.fromkeys(out)), len(self.checks))

    def _live_form(self, obj: InfoObject, item: TV.TraderMemoryItem) -> InfoObject:
        rec = self.store.matured_record(obj.object_id)
        return InfoObject(f"L{item.item_id}", obj.kind, Namespace.LIVE_POINT_IN_TIME, rec.matured_at, item.to_dict(),
                          dataclasses.replace(obj.provenance, parents=()), origin="engine.research.firewall")

    def release(self, ids: Sequence[str | InfoObject], now, replay: ReplayContext | None = None, step: int | None = None,
                weights: Mapping[str, float] | None = None) -> ReleaseResult:
        """Inspect every requested object; in strict mode any breach refuses the whole release (FirewallBreach, audit logged).
        Admitted objects are projected, weighted (given weights or uniform; normalised to 1), merged by content token and ordered
        by token, and the TraderRelease is re-scanned for date-like content before it leaves."""
        nowd = moment(now, "now")
        step = int(replay.step if step is None and replay is not None else (step or 0))
        decisions = tuple(self.inspect(i, nowd, replay) for i in ids)
        rid = "R" + stable_hash({"now": str(nowd), "d": [d.digest() for d in decisions], "w": dict(weights or {})}, 16)
        refused = {d.object_id: d.channels for d in decisions if not d.admitted}
        admitted = [d for d in decisions if d.admitted]
        if refused and self.policy.strict:
            self.audit.append(nowd, rid, [d.object_id for d in admitted], refused, "REFUSED", "")
            first = next(d for d in decisions if not d.admitted)
            raise FirewallBreach(f"release refused at {nowd} ({len(refused)}/{len(decisions)} objects breached "
                                 f"{sorted({str(c) for v in refused.values() for c in v})}); first: {first.breaches[0]}")
        merged: dict[str, list] = {}
        for d in admitted:
            obj = self.store.get(d.object_id)
            w = float((weights or {}).get(d.object_id, 1.0))
            if not math.isfinite(w) or w < 0:
                raise FirewallBreach(f"weight for {d.object_id} must be a finite non-negative number")
            item = project(obj)
            merged.setdefault(item.item_id, [item, 0.0, []])
            merged[item.item_id][1] += w
            merged[item.item_id][2].append(obj)
        total = sum(v[1] for v in merged.values())
        items, lives, tickets = [], [], []
        for tok in sorted(merged):
            item, w, objs = merged[tok]
            item = item.with_weight(w / total if total > 0 else 1.0 / len(merged))
            items.append(item)
            live = self._live_form(objs[0], item)
            lives.append(live)
            tickets.append(sign_ticket(self._secret, live.object_id, live.digest(), nowd, rid))
        rel = TV.TraderRelease(step, tuple(items))
        hits = TV.release_year_hits(rel)
        if hits:
            self.audit.append(nowd, rid, [], {d.object_id: {LeakChannel.YEAR_IDENTITY} for d in admitted}, "REFUSED", rel.digest())
            raise FirewallBreach(f"release would carry date-like content to the trader: {hits}")
        outcome = "EMPTY" if not decisions else "PARTIAL" if refused else "RELEASED"
        self.audit.append(nowd, rid, [d.object_id for d in admitted], refused, outcome, rel.digest())
        return ReleaseResult(rel, decisions, tuple(lives), tuple(tickets), rid)

    def admit_to_live(self, result: ReleaseResult, live: LiveStore) -> int:
        """File a release's items into live state with their tickets (the trusted runner's step after `release`)."""
        for obj, t in zip(result.live_objects, result.tickets):
            live.accept(obj, t)
        return len(result.live_objects)


def pair(run_name: str, secret: str, policy: FirewallPolicy | None = None) -> tuple[NamespacePair, ResearchTraderFirewall]:
    """A run's two namespaces and the firewall between them, wired so that the live store verifies this firewall's tickets."""
    p = NamespacePair(run_name, lambda t: ticket_valid(str(secret), t))
    return p, ResearchTraderFirewall(p.research, secret, policy)


def run_day(fw: ResearchTraderFirewall, now, candidate_ids: Sequence[str] | None = None, replay: ReplayContext | None = None,
            weights: Mapping[str, float] | None = None, live: LiveStore | None = None) -> ReleaseResult:
    """PUBLIC ENTRY (wave-2 research loop, once per simulated session): release the candidate research objects (default: every
    object in the research store) that pass the firewall at `now`, strictly fail-closed, and file them into live state if a
    LiveStore is given (its clock is advanced to `now` first). Non-strict policies return the admitted subset instead of raising."""
    ids = list(candidate_ids) if candidate_ids is not None else list(fw.store.ids())
    res = fw.release(ids, now, replay, weights=weights)
    if live is not None:
        live.advance(now)
        fw.admit_to_live(res, live)
    return res


# ------------------------------------------------------------------------------------------------ training gate (hindsight labels)
@dataclasses.dataclass(frozen=True)
class TrainingGateReport:
    n_in: int
    n_admitted: int
    n_unmatured: int
    n_same_year: int
    now: str
    label: str = LABEL

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class TrainingGate:
    """The explicit point-in-time gate for research-made LABELS (including hindsight classes such as knowability or unknown cause)
    entering training state. Each row must carry the real date its label matured; rows whose label had not matured strictly before
    `now`, or whose rows or labels fall in a real year being replayed, are held back. A row without a maturity date, or whose label
    claims to have matured before the row it labels, refuses the whole batch (fail closed)."""

    def __init__(self, date_col: str = "date", matured_col: str = "matured_at"):
        self.date_col, self.matured_col = date_col, matured_col

    def admit(self, frame: pd.DataFrame, now, replay: ReplayContext | None = None) -> tuple[pd.DataFrame, TrainingGateReport]:
        nowd = pd.Timestamp(moment(now, "now"))
        for c in (self.date_col, self.matured_col):
            if c not in frame.columns:
                raise FirewallBreach(f"training labels need a {c!r} column: maturity cannot be shown")
        if frame.empty:
            return frame.copy(), TrainingGateReport(0, 0, 0, 0, str(nowd.date()))
        rows = pd.to_datetime(frame[self.date_col], errors="coerce").dt.normalize()
        mat = pd.to_datetime(frame[self.matured_col], errors="coerce").dt.normalize()
        if rows.isna().any() or mat.isna().any():
            raise FirewallBreach(f"{int(rows.isna().sum() + mat.isna().sum())} training rows have unreadable dates (timestamp ambiguity)")
        if (mat < rows).any():
            raise FirewallBreach(f"{int((mat < rows).sum())} labels claim to have matured before the row they label")
        matured = mat < nowd
        same = pd.Series(False, index=frame.index)
        if replay is not None:
            ys = replay.years()
            same = rows.dt.year.isin(ys) | mat.dt.year.isin(ys)
        keep = matured & ~same
        rep = TrainingGateReport(int(len(frame)), int(keep.sum()), int((~matured).sum()), int((matured & same).sum()), str(nowd.date()))
        return frame.loc[keep].copy(), rep

    def blind_rows(self, admitted: pd.DataFrame, keep: Sequence[str]) -> pd.DataFrame:
        """The admitted rows as the blind learner may hold them: only `keep` columns, none of which may name time."""
        bad = [c for c in keep if TV.key_reason(str(c)) or TV.string_reasons(str(c))]
        if bad or self.date_col in keep or self.matured_col in keep:
            raise FirewallBreach(f"columns {bad or [self.date_col, self.matured_col]} would carry dates into the blind learner")
        return admitted.loc[:, list(keep)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ import guards
RESEARCH_PACKAGE = "engine.research"
RESEARCH_SYMBOLS = frozenset({"ResearchStore", "MaturedRecord", "InfoObject", "NamespacePair", "ResearchTraderFirewall",
                              "CrossingTicket", "FirewallAudit", "TrainingGate"})
RESEARCH_STORE_MARKERS = ("matured_research", "research_store")


def research_modules(root=None) -> frozenset:
    """Every dotted module of the research package under `root` (the package itself included)."""
    r = TV._root(root)
    pkg = r / "engine" / "research"
    mods = {RESEARCH_PACKAGE} if (pkg / "__init__.py").exists() else set()
    mods |= {f"{RESEARCH_PACKAGE}.{p.stem}" for p in pkg.glob("*.py") if p.stem != "__init__"}
    return frozenset(mods)


def trader_research_violations(entries: Sequence[str] | None = None, root=None) -> list[TV.PathViolation]:
    """Static guard: the trader path's import closure (top-level and lazy imports) may reach neither the curator nor any research
    module, may not name a research symbol, and may not name the research store (trader_view does the walking)."""
    r = TV._root(root)
    forbidden = TV.CURATOR_MODULES | research_modules(r)
    found = list(TV.trader_path_violations(entries, r, forbidden))
    for mod in TV.trader_closure(entries, r):
        if mod == RESEARCH_PACKAGE or mod.startswith(RESEARCH_PACKAGE + "."):
            continue
        path = TV.module_file(mod, r)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(r).as_posix()
        for n in ast.walk(tree):
            name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
            if name in RESEARCH_SYMBOLS:
                found.append(TV.PathViolation(rel, n.lineno, "symbol", name, mod))
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                for mk in RESEARCH_STORE_MARKERS:
                    if mk in n.value.lower():
                        found.append(TV.PathViolation(rel, n.lineno, "store_read", mk, mod))
    return sorted(set(found), key=lambda v: (v.file, v.line, v.kind, v.target))


def assert_trader_research_clean(entries: Sequence[str] | None = None, root=None) -> int:
    v = trader_research_violations(entries, root)
    if v:
        raise FirewallBreach("trader path reaches research/curator state: " + "; ".join(str(x) for x in v[:3]))
    n = len(TV.trader_closure(entries, root))
    if not n:
        raise FirewallBreach("trader import guard inspected no modules: it cannot pass on nothing")
    return n


class ResearchImportBlocked(FirewallBreach, ImportError):
    """Raised when trader-side code imports the research package while the runtime guard is active."""


class ResearchImportBlocker(importlib.abc.MetaPathFinder):
    """Runtime guard (section 29: no convenience imports). While active, ANY import of engine.research.* raises
    ResearchImportBlocked - including modules already loaded, which are hidden from sys.modules for the duration and restored
    after. Python cannot stop reflection on objects already held; the static guard covers source, this covers execution."""

    def __init__(self, package: str = RESEARCH_PACKAGE):
        self.package = package
        self.attempts: list[str] = []
        self._stash: dict[str, types.ModuleType] = {}
        self._attr = None

    def _match(self, name: str) -> bool:
        return name == self.package or name.startswith(self.package + ".")

    def find_spec(self, fullname, path=None, target=None):
        if self._match(fullname):
            self.attempts.append(fullname)
            raise ResearchImportBlocked(f"trader-side import of {fullname} blocked by the research firewall")
        return None

    def __enter__(self) -> "ResearchImportBlocker":
        self._stash = {k: sys.modules.pop(k) for k in list(sys.modules) if self._match(k)}
        parent, _, leaf = self.package.rpartition(".")
        pmod = sys.modules.get(parent)
        if pmod is not None and hasattr(pmod, leaf):
            self._attr = (pmod, leaf, getattr(pmod, leaf))
            delattr(pmod, leaf)
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc) -> bool:
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for k in [k for k in sys.modules if self._match(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(self._stash)
        if self._attr is not None:
            setattr(*self._attr)
        self._stash, self._attr = {}, None
        return False


# ------------------------------------------------------------------------------------------------ section 30: disguise audits
def _shift_value(x: Any, days: int) -> Any:
    delta = dt.timedelta(days=days)
    if isinstance(x, pd.Timestamp):
        return x + delta
    if isinstance(x, dt.datetime) or isinstance(x, dt.date):
        return x + delta
    if isinstance(x, str):
        return _ISO_INSIDE.sub(lambda m: (dt.date(int(m[1]), int(m[2]), int(m[3])) + delta).isoformat(), x)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        y = x.copy()
        if isinstance(y.index, pd.DatetimeIndex):
            y.index = y.index + delta
        if isinstance(y, pd.DataFrame):
            for c in y.columns:
                if pd.api.types.is_datetime64_any_dtype(y[c]):
                    y[c] = y[c] + delta
        return y
    if isinstance(x, Mapping):
        return {k: _shift_value(v, days) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_shift_value(v, days) for v in x)
    return x


def shift_object(obj: InfoObject, days: int) -> InfoObject:
    """The same research object in another calendar: every date (fields, provenance, payload, tags) moved by `days`. Used by the
    year-blindness probe; a whole number of weeks keeps weekdays."""
    p = obj.provenance
    prov = dataclasses.replace(p, learned_at=_shift_value(p.learned_at, days) if p.learned_at else p.learned_at,
                               outcomes_seen_through=_shift_value(p.outcomes_seen_through, days) if p.outcomes_seen_through else "")
    tags = dict(obj.tags)
    if isinstance(tags.get("real_year"), (int, np.integer)):
        tags["real_year"] = (dt.date(int(tags["real_year"]), 7, 1) + dt.timedelta(days=days)).year
    return dataclasses.replace(obj, observed_at=_shift_value(obj.observed_at, days),
                               available_at=_shift_value(obj.available_at, days) if obj.available_at is not None else None,
                               payload=_shift_value(dict(obj.payload), days), provenance=prov, tags=tags)


def shift_replay(replay: ReplayContext, days: int) -> ReplayContext:
    d = dt.timedelta(days=days)
    return dataclasses.replace(replay, real_start=(moment(replay.real_start) + d).isoformat(),
                               real_end=(moment(replay.real_end) + d).isoformat())


@dataclasses.dataclass(frozen=True)
class BlindnessResult:
    shifts: tuple[int, ...]
    indistinguishable: Mapping[int, bool]
    digests: Mapping[int, str]

    @property
    def passed(self) -> bool:
        return bool(self.indistinguishable) and all(self.indistinguishable.values())


def year_blindness_probe(producer: Callable[[int], Sequence[InfoObject]], now, replay: ReplayContext | None, secret: str,
                         shifts: Sequence[int] = (7 * 52 * 7, -7 * 52 * 11), policy: FirewallPolicy | None = None) -> BlindnessResult:
    """Section 30 'this is 2008': the research producer is run on the same world placed in other calendars (whole-week shifts),
    each through a fresh store and firewall. The trader's releases must be indistinguishable (trader_view.indistinguishable); a
    producer or projection that encodes the calendar is caught here even when no single release looks dated."""
    def one(off: int) -> TV.TraderRelease:
        st = ResearchStore(f"blind{off}")
        objs = list(producer(off))
        st.put_many(objs)
        fw = ResearchTraderFirewall(st, secret, policy)
        rp = shift_replay(replay, off) if replay is not None else None
        return fw.release([o.object_id for o in objs], moment(now) + dt.timedelta(days=off), rp).release
    base = one(0)
    ind: dict[int, bool] = {}
    digs: dict[int, str] = {0: base.digest()}
    for s in shifts:
        r = one(int(s))
        ind[int(s)] = TV.indistinguishable(base, r)
        digs[int(s)] = r.digest()
    return BlindnessResult(tuple(int(s) for s in shifts), ind, digs)


def run_invariance(fw: ResearchTraderFirewall, ids: Sequence[str], now, replay: ReplayContext, run_indices: Sequence[int] = (0, 1, 5)) -> dict:
    """Section 30 'this is the same year I saw before' / 'this is the training run': with the same research state, the same day
    and the same window, the release must not depend on WHICH run it is. Returns digests per run and whether they agree."""
    digs = {int(k): fw.release(ids, now, dataclasses.replace(replay, run_index=int(k))).release.digest() for k in run_indices}
    return {"digests": digs, "invariant": len(set(digs.values())) == 1, "n_runs": len(digs)}


def release_text_findings(releases: Sequence[TV.TraderRelease], real_tickers: Iterable[str] = (), real_years: Iterable[int] = ()) -> list[FW.Finding]:
    """Serialised releases scanned for real tickers and years (future_firewall.text_leak_findings, i.e. engine.blind_gates)."""
    from engine.learning.future_firewall import text_leak_findings
    texts = [TV.blind_json(r) for r in releases]
    return text_leak_findings(texts, tuple(real_tickers), tuple(real_years), name="release")


def cross_run_overlap(a: Sequence[TV.TraderRelease], b: Sequence[TV.TraderRelease]) -> dict:
    """How much two runs' releases share (trusted-side diagnostic). A same-year rerun that shares far more than a different-year
    run is recognisable from memory content alone; the number is reported, not judged (it depends on the world)."""
    ia = {i.item_id for r in a for i in r.items}
    ib = {i.item_id for r in b for i in r.items}
    u = ia | ib
    return {"n_a": len(ia), "n_b": len(ib), "shared": len(ia & ib), "jaccard": (len(ia & ib) / len(u)) if u else float("nan")}


# ------------------------------------------------------------------------------------------------ RESEARCH layer for the 8-layer gate
class ResearchLayerName(_StrEnum):
    RESEARCH = "RESEARCH"


@dataclasses.dataclass
class ResearchGateContext(FW.GateContext):
    research_ids: Sequence[str] | None = None
    replay: ReplayContext | None = None
    firewall: ResearchTraderFirewall | None = None


class ResearchReleaseLayer(FW.FirewallLayer):
    """A ninth LAYER (not a ninth firewall) for engine.learning.firewalls.LearningFirewallGate: every research object in play
    must pass this module's checks at ctx.now. Findings carry the mapped standard layer name so they validate unchanged."""
    name = ResearchLayerName.RESEARCH

    def missing(self, ctx, what: str):
        return [FW.fail(_L.PROVENANCE, "research-input-missing", ctx.subject,
                        f"{what} not supplied: the research layer cannot demonstrate the release is clean")], 0

    def inspect(self, ctx):
        if not isinstance(ctx, ResearchGateContext) or ctx.firewall is None:
            return self.missing(ctx, "research firewall")
        if ctx.research_ids is None:
            return self.missing(ctx, "research object ids")
        out: list[FW.Finding] = []
        for oid in ctx.research_ids:
            d = ctx.firewall.inspect(oid, ctx.now, ctx.replay)
            out += [b.to_finding() for b in d.breaches]
        if not ctx.research_ids:
            out.append(FW.info(_L.PROVENANCE, "research-empty", ctx.subject, "no research objects in play"))
        return out, len(ctx.research_ids)


def research_gate() -> FW.LearningFirewallGate:
    return FW.LearningFirewallGate(FW.default_layers() + [ResearchReleaseLayer()])


# ------------------------------------------------------------------------------------------------ planted-leak suite
REF_NOW = "2020-06-01"


def _prov(learned: str, seen: str | None = None, code: str = "c0de", parents: tuple = ()) -> Provenance:
    return Provenance(created_real="2026-09-29T00:00:00", learned_at=learned, code_hash=code, data_hash="d0", config_hash="k0",
                      experiment_id="e0", outcomes_seen_through=seen or learned, parents=parents)


def clean_object(oid: str = "clean", learned: str = "2019-11-29", features: Mapping[str, float] | None = None, **payload) -> InfoObject:
    """A legitimate research result: matured months before REF_NOW, outside the replayed year, trader-shaped."""
    p = {"features": dict(features or {"vol_z": 1.5, "r20": -0.02}), "lean": 0.3, "horizon": 5, "trader_kind": "pattern",
         "hit_rate": 0.56, "n_obs": 212, **payload}
    return InfoObject(oid, InfoKind.RESEARCH_RESULT, Namespace.MATURED_RESEARCH, learned, p, _prov(learned),
                      origin="engine.research.planted", tags={"real_year": int(learned[:4]), "sample_ticker": "AAPL"})


def reference_replay(now: str = REF_NOW) -> ReplayContext:
    y = moment(now).year
    return ReplayContext("w-ref", f"{y}-01-01", f"{y}-12-31", run_index=1, step=97, session_count=253, universe_size=2873,
                         hidden_values=(0.0173421937, -0.0291055812, 0.0048812201), real_tickers=("ZQXW", "MSFT"))


def _frame(start: str, end: str, col: str = "close") -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({col: np.linspace(1.0, 1.1, len(idx))}, index=idx)


def _kind(oid, kind, observed, learned, available=None, seen=None, parents=(), **payload):
    base = {"features": {"vol_z": 0.8}, "lean": 0.1, "horizon": 5}
    base.update(payload)
    return InfoObject(oid, kind, Namespace.MATURED_RESEARCH, observed, base, _prov(learned, seen), available_at=available,
                      parents=tuple(parents), origin="engine.research.planted")


def planted_leaks() -> dict[LeakChannel, Callable[[], list[InfoObject]]]:
    """Channel -> builder of the objects to file; the LAST object is the one the research side tries to release. Each is a
    research-side attempt to leak exactly one thing; `run_planted_suite` asks that the release is refused naming that channel."""
    return {
        LeakChannel.FUTURE_PRICE: lambda: [_kind("px", InfoKind.PRICE, "2019-12-31", "2019-12-31",
                                                 bars=_frame("2019-12-02", "2020-06-10"))],
        LeakChannel.FUTURE_VOLUME: lambda: [_kind("vol", InfoKind.VOLUME, "2020-06-03", "2020-06-03")],
        LeakChannel.FUTURE_FILING: lambda: [_kind("fil", InfoKind.FILING, "2020-05-29", "2020-05-29", available="2020-05-29")],
        LeakChannel.FUTURE_EARNINGS: lambda: [_kind("eps", InfoKind.EARNINGS, "2019-09-30", "2019-11-05",
                                                    note="surprise confirmed 2020-07-28")],
        LeakChannel.FUTURE_LABEL: lambda: [_kind("lab", InfoKind.LABEL, REF_NOW, "2019-10-01")],
        LeakChannel.FUTURE_METADATA: lambda: [_kind("meta", InfoKind.METADATA, "2019-06-03", "2019-06-03",
                                                    sector_as_of="2021-01-04")],
        LeakChannel.FUTURE_CORPORATE_ACTION: lambda: [_kind("split", InfoKind.CORPORATE_ACTION, "2020-08-31", "2019-06-03")],
        LeakChannel.FUTURE_MARKET_STATE: lambda: [_kind("mkt", InfoKind.MARKET_STATE, "2020-06-05", "2020-06-05")],
        LeakChannel.FUTURE_EVENT: lambda: [_kind("evt", InfoKind.EVENT, "2020-05-20", "2019-06-03", available="2020-06-02")],
        LeakChannel.FUTURE_LEARNED_STATE: lambda: [_kind("state", InfoKind.LEARNED_STATE, "2019-10-01", "2019-10-01",
                                                         seen="2020-06-15")],
        LeakChannel.FUTURE_RESEARCH_RESULT: lambda: [_kind("res", InfoKind.RESEARCH_RESULT, "2019-10-01", "2019-10-01",
                                                           note={"confirmed": "held again through 2020-07-01"})],
        LeakChannel.FUTURE_PATTERN_STATUS: lambda: [_kind("stat", InfoKind.PATTERN_STATUS, "2019-11-01", "2020-07-01")],
        LeakChannel.FUTURE_PATTERN_HEALTH: lambda: [_kind("hlth", InfoKind.PATTERN_HEALTH, "2019-11-01", "2019-11-01",
                                                          seen="2020-09-30")],
        LeakChannel.FUTURE_EXPERIMENT_RESULT: lambda: [_kind("exp", InfoKind.EXPERIMENT_RESULT, "2020-08-14", "2020-08-14"),
                                                       _kind("child", InfoKind.RESEARCH_RESULT, "2019-10-01", "2019-10-01",
                                                             parents=("exp",))],
        LeakChannel.YEAR_IDENTITY: lambda: [clean_object("yr", features={"vol_z": 1.2, "regime_2008_like": 1.0})],
        LeakChannel.SAME_YEAR_RERUN: lambda: [clean_object("same", learned="2020-02-14")],
        LeakChannel.TICKER_IDENTITY: lambda: [clean_object("tkr", features={"vol_z": 1.1, "AAPL": 0.4})],
        LeakChannel.NUMERIC_FINGERPRINT: lambda: [clean_object("fp", features={"vol_z": 0.0173421937})],
        LeakChannel.DATASET_STRUCTURE: lambda: [clean_object("dim", features={"vol_z": 1.0, "breadth": 2873.0})],
        LeakChannel.TRAINING_RUN_MARKER: lambda: [clean_object("mark", features={"vol_z": 1.0, "rerun": 1.0})],
        LeakChannel.HIDDEN_CACHE: lambda: [dataclasses.replace(clean_object("cache"),
                                                               cache_key="cache/research/y2008_matured_research.parquet")],
        LeakChannel.IMPLICIT_SHARED_STATE: lambda: [clean_object("shared", backdoor=ResearchStore("backdoor"))],
        LeakChannel.MISSING_PROVENANCE: lambda: [dataclasses.replace(clean_object("noprov"),
                                                                     provenance=dataclasses.replace(_prov("2019-11-29"), code_hash=""))],
        LeakChannel.TIMESTAMP_AMBIGUITY: lambda: [dataclasses.replace(clean_object("ambig"), observed_at="2019-11")],
        LeakChannel.HINDSIGHT_LABEL: lambda: [clean_object("hind", knowability=str(Knowability.EXTERNALLY_CAUSED))],
        LeakChannel.RESEARCH_ONLY_KNOWLEDGE: lambda: [clean_object("ronly", decision_effect="NONE")],
        LeakChannel.MALFORMED: lambda: [dataclasses.replace(clean_object("malf"), payload={"lean": 0.2, "horizon": 5})],
    }


@dataclasses.dataclass(frozen=True)
class PlantResult:
    channel: LeakChannel
    refused: bool                  # the release raised (the trader got nothing)
    named: bool                    # the breach named the planted channel
    filed: bool                    # the research store accepted the object (else refused at filing too)
    channels: tuple[str, ...]

    @property
    def caught(self) -> bool:
        return self.refused and self.named


@dataclasses.dataclass(frozen=True)
class SuiteResult:
    results: Mapping[LeakChannel, PlantResult]
    clean_released: bool
    import_guard_caught: bool | None

    @property
    def void(self) -> bool:
        """A suite whose clean control is refused proves nothing (a firewall that refuses everything 'catches' every leak)."""
        return not self.clean_released

    @property
    def missed(self) -> list[str]:
        return [str(c) for c, r in self.results.items() if not r.caught]

    @property
    def passed(self) -> bool:
        return not self.void and not self.missed and all(c in self.results for c in REQUIRED_CHANNELS) \
            and self.import_guard_caught is not False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"channel": str(c), "refused": r.refused, "named": r.named, "filed": r.filed,
                              "channels": ",".join(r.channels)} for c, r in self.results.items()],
                            columns=["channel", "refused", "named", "filed", "channels"])


def plant_import_leak(root: Path) -> list[TV.PathViolation]:
    """Write a minimal trader entry that lazily imports the research store under `root` and return what the static guard finds."""
    (root / "engine" / "research").mkdir(parents=True, exist_ok=True)
    (root / "engine" / "__init__.py").write_text("", encoding="utf-8")
    (root / "engine" / "research" / "__init__.py").write_text("", encoding="utf-8")
    (root / "engine" / "research" / "namespaces.py").write_text("X = 1\n", encoding="utf-8")
    (root / "engine" / "livesim.py").write_text("def decide():\n    from engine.research.namespaces import X\n    return X\n",
                                                encoding="utf-8")
    return trader_research_violations(("engine/livesim.py",), root)


_REFERENCE = "reference"


def run_planted_suite(now: str = REF_NOW, secret: str = "planted-suite", import_root: Path | None = None,
                      channels: Iterable[LeakChannel] | None = None, replay: ReplayContext | None | str = _REFERENCE,
                      checks: Sequence[tuple[str, Callable[[_Ctx], list[Breach]]]] = CHECKS) -> SuiteResult:
    """Every planted leak through a fresh store + firewall, by default with the reference replay (the trader is replaying
    REF_NOW's year). A leak counts as caught only if the release is REFUSED and the breach NAMES the planted channel. `checks`
    lets a test remove one check and show which channels only that check catches (ablation)."""
    replay = reference_replay(now) if isinstance(replay, str) else replay
    plants = planted_leaks()
    wanted = list(channels) if channels is not None else list(plants)
    st = ResearchStore("control")
    st.put(clean_object())
    clean_ok = True
    try:
        ResearchTraderFirewall(st, secret, checks=checks).release(["clean"], now, replay)
    except FirewallBreach:
        clean_ok = False
    results: dict[LeakChannel, PlantResult] = {}
    for ch in wanted:
        store = ResearchStore(f"plant-{ch.value.lower()}")
        store.put(clean_object())
        objs = plants[ch]()
        filed = True
        for o in objs:
            try:
                store.put(o)
            except FirewallBreach:
                filed = False
        fw = ResearchTraderFirewall(store, secret, checks=checks)
        target = objs[-1]
        dec = fw.inspect(target if not filed or target.object_id not in store else target.object_id, now, replay)
        refused = False
        try:
            fw.release(["clean", target if target.object_id not in store else target.object_id], now, replay)
        except FirewallBreach:
            refused = True
        results[ch] = PlantResult(ch, refused, ch in dec.channels, filed, tuple(sorted(str(c) for c in dec.channels)))
    guard = None
    if import_root is not None:
        guard = any(v.target.startswith(RESEARCH_PACKAGE) for v in plant_import_leak(Path(import_root)))
    return SuiteResult(results, clean_ok, guard)


def coverage(suite: SuiteResult) -> dict:
    """Coverage in both directions: every channel planted, and every channel some check can produce."""
    planted = set(suite.results)
    producible = set().union(*CHECK_CHANNELS.values()) | {LeakChannel.CONVENIENCE_IMPORT}
    return {"unplanted": sorted(str(c) for c in set(LeakChannel) - planted - {LeakChannel.CONVENIENCE_IMPORT}),
            "unproducible": sorted(str(c) for c in set(LeakChannel) - producible),
            "import_planted": suite.import_guard_caught is not None, "missed": suite.missed}


def suite_markdown(suite: SuiteResult) -> str:
    lines = [f"# Research/trader firewall planted-leak suite ({LABEL})",
             f"clean control released: {suite.clean_released}   void: {suite.void}   passed: {suite.passed}", "",
             "| channel | refused | named | filed | channels reported |", "|---|---|---|---|---|"]
    for c, r in suite.results.items():
        lines.append(f"| {c} | {r.refused} | {r.named} | {r.filed} | {', '.join(r.channels)} |")
    if suite.import_guard_caught is not None:
        lines.append(f"| {LeakChannel.CONVENIENCE_IMPORT} | {suite.import_guard_caught} | {suite.import_guard_caught} | - | static guard |")
    return "\n".join(lines)


def decision_frame(decisions: Iterable[Decision]) -> pd.DataFrame:
    """One row per breach (trusted-side report)."""
    rows = [{"object_id": d.object_id, "now": d.now, "channel": str(b.channel), "check": b.check, "message": b.message}
            for d in decisions for b in d.breaches]
    return pd.DataFrame(rows, columns=["object_id", "now", "channel", "check", "message"])


# ------------------------------------------------------------------------------------------------ the trader's only reader
TRADER_OBSERVABLE = frozenset({InfoKind.PRICE, InfoKind.VOLUME, InfoKind.MARKET_STATE, InfoKind.FEATURE, InfoKind.EVENT,
                               InfoKind.FILING, InfoKind.EARNINGS, InfoKind.METADATA, InfoKind.CORPORATE_ACTION})


class TraderGateway:
    """Section 31 'the trader may ONLY observe point-in-time data, access matured knowledge whose maturity precedes the decision,
    retrieve permitted historical patterns'. The trusted runner calls this on the trader's behalf; what comes back is blind:
    no dates, no ids that carry dates, no research tags. Every read is logged here (trusted side) with what was withheld."""

    def __init__(self, live: LiveStore, observable: Iterable[InfoKind] = TRADER_OBSERVABLE):
        if not isinstance(live, LiveStore):
            raise FirewallBreach("the trader gateway reads LIVE_POINT_IN_TIME_STATE only")
        self._live = live
        self.observable = frozenset(InfoKind.parse(k) for k in observable)
        self.log: list[dict] = []

    def _record(self, now, method: str, served: int, withheld: int, reasons: Mapping[str, int]) -> None:
        self.log.append({"now": str(as_date(now)), "method": method, "served": served, "withheld": withheld, "reasons": dict(reasons)})

    def observe(self, now, kinds: Iterable[InfoKind] | None = None) -> list[dict]:
        """Numeric scalar values of the point-in-time objects knowable at `now`, one blind record per object. Anything that is not
        a plain finite number (frames, strings, nested maps) stays behind: turning raw series into relative features is the
        curator's job, not a shortcut through this door."""
        kinds = self.observable if kinds is None else frozenset(InfoKind.parse(k) for k in kinds)
        if not kinds <= self.observable:
            raise FirewallBreach(f"kinds {sorted(str(k) for k in kinds - self.observable)} are not observable by the trader")
        out, why = [], {}
        for o in self._live.view(now):
            if o.kind not in kinds or o.origin.startswith(LiveStore.RESEARCH_ORIGIN):
                continue
            vals = {str(k): float(v) for k, v in o.payload.items()
                    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and math.isfinite(float(v))}
            dropped = len(o.payload) - len(vals)
            if dropped:
                why["non_scalar"] = why.get("non_scalar", 0) + dropped
            rec = {"kind": str(o.kind).lower(), "values": vals}
            if TV.find_violations(rec):
                why["date_like"] = why.get("date_like", 0) + 1
                continue
            out.append(rec)
        self._record(now, "observe", len(out), sum(why.values()), why)
        return sorted(out, key=TV.opaque_token)

    def memory(self, now, step: int) -> TV.TraderRelease:
        """Released research knowledge filed in live state and knowable at `now`, re-weighted to sum to 1."""
        items, why = [], {}
        for o in self._live.view(now):
            if not o.origin.startswith(LiveStore.RESEARCH_ORIGIN):
                continue
            p = o.payload
            try:
                items.append(TV.TraderMemoryItem.make(p["item_id"], p["kind"], float(p["weight"]), dict(p["features"]),
                                                      float(p["lean"]), int(p["horizon"])))
            except (KeyError, TypeError, ValueError, FirewallBreach):
                why["malformed"] = why.get("malformed", 0) + 1
        uniq = {i.item_id: i for i in items}
        total = sum(i.weight for i in uniq.values())
        norm = [i.with_weight(i.weight / total if total > 0 else 1.0 / len(uniq)) for i in uniq.values()]
        rel = TV.TraderRelease(int(step), tuple(sorted(norm, key=lambda i: i.item_id)))
        self._record(now, "memory", len(rel), sum(why.values()), why)
        return rel

    def access_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["now", "method", "served", "withheld", "reasons"])


# ------------------------------------------------------------------------------------------------ audit persistence
def save_audit(audit: FirewallAudit, path) -> int:
    """Write the chained audit as JSON lines (trusted-side state/ path only). Returns entries written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for e in audit.entries:
            fh.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")
    return len(audit.entries)


def load_audit(path) -> FirewallAudit:
    """Read an audit back and re-verify its chain; a tampered or reordered file raises FirewallBreach. A missing file is an
    empty audit (nothing was released yet)."""
    a = FirewallAudit()
    p = Path(path)
    if not p.exists():
        return a
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        a.entries.append(AuditEntry(int(d["seq"]), d["now"], d["release_id"], tuple(d["admitted"]),
                                    {k: tuple(v) for k, v in d["refused"].items()}, d["outcome"], d["release_digest"],
                                    d["prev"], d["link"]))
    errs = a.verify()
    if errs:
        raise FirewallBreach(f"audit file {p.name} fails verification: {errs[0]}")
    return a


# ------------------------------------------------------------------------------------------------ census and release schedule
def release_census(fw: ResearchTraderFirewall, now, replay: ReplayContext | None = None) -> pd.DataFrame:
    """Every object in the research store judged at `now`: kind, filed years, first usable date, admitted, channels. The
    trusted-side view of how much research the trader could use today and what holds the rest back."""
    rows = []
    for oid in fw.store.ids():
        o = fw.store.get(oid)
        d = fw.inspect(oid, now, replay)
        try:
            k, who = fw.store.effective_knowable(oid)
            years = fw.store.filed_years(oid)
        except FirewallBreach:
            k, who, years = None, "", ()
        rows.append({"object_id": oid, "kind": str(o.kind), "first_year": min(years) if years else None,
                     "last_year": max(years) if years else None, "knowable_at": k, "set_by": who, "admitted": d.admitted,
                     "channels": ",".join(sorted(str(c) for c in d.channels))})
    return pd.DataFrame(rows, columns=["object_id", "kind", "first_year", "last_year", "knowable_at", "set_by", "admitted", "channels"])


def census_summary(census: pd.DataFrame) -> dict:
    """Admitted share overall, by kind and by last filed year, and refusal counts by channel. Empty census -> n=0, no shares."""
    if census.empty:
        return {"n": 0, "admitted_share": None, "by_kind": {}, "by_year": {}, "by_channel": {}}
    ch: dict[str, int] = {}
    for s in census["channels"]:
        for c in filter(None, str(s).split(",")):
            ch[c] = ch.get(c, 0) + 1
    return {"n": int(len(census)), "admitted_share": float(census["admitted"].mean()),
            "by_kind": {str(k): float(v) for k, v in census.groupby("kind")["admitted"].mean().items()},
            "by_year": {int(k): float(v) for k, v in census.dropna(subset=["last_year"]).groupby("last_year")["admitted"].mean().items()},
            "by_channel": dict(sorted(ch.items(), key=lambda kv: -kv[1]))}


def first_release_date(fw: ResearchTraderFirewall, object_id: str, replay: ReplayContext | None = None) -> dt.date | None:
    """The first session at which the object could cross, by time alone: its lineage's first usable date. None when the object
    can never cross during this replay (filed under a replayed year) or its lineage cannot be audited."""
    try:
        k, _ = fw.store.effective_knowable(object_id)
        if replay is not None and set(fw.store.filed_years(object_id)) & replay.years():
            return None
    except FirewallBreach:
        return None
    return k


def release_schedule(fw: ResearchTraderFirewall, replay: ReplayContext | None = None) -> pd.DataFrame:
    rows = [{"object_id": oid, "kind": str(fw.store.get(oid).kind), "first_release": first_release_date(fw, oid, replay)}
            for oid in fw.store.ids()]
    df = pd.DataFrame(rows, columns=["object_id", "kind", "first_release"])
    df["_never"] = df["first_release"].isna()
    df["_key"] = df["first_release"].map(lambda d: d.isoformat() if d is not None else "")
    return df.sort_values(["_never", "_key", "object_id"], ignore_index=True).drop(columns=["_never", "_key"])


# ------------------------------------------------------------------------------------------------ boundary sweep
def boundary_sweep(obj: InfoObject, now, offsets: Sequence[int] = tuple(range(-4, 5)), secret: str = "sweep",
                   replay: ReplayContext | None = None) -> pd.DataFrame:
    """Move a clean object's whole calendar so its newest evidence lands `offset` days from `now` and ask the firewall each time.
    The exact rule is: admitted iff the object is first usable on/before now AND its evidence is strictly before now. The
    table shows the observed and expected answer per offset; `boundary_ok` checks they agree and admission never returns
    once refused."""
    base = obj.evidence_through()
    nowd = moment(now, "now")
    rows = []
    for off in offsets:
        shift = (nowd + dt.timedelta(days=int(off)) - base).days
        o = dataclasses.replace(shift_object(obj, shift), object_id=f"{obj.object_id}_{'m' if off < 0 else 'p'}{abs(int(off))}")
        st = ResearchStore(f"sweep{off}")
        filed = True
        try:
            st.put(o)
        except FirewallBreach:
            filed = False
        fw = ResearchTraderFirewall(st, secret)
        d = fw.inspect(o.object_id if filed else o, nowd, replay)
        rows.append({"offset": int(off), "evidence_through": o.evidence_through(), "knowable_at": o.knowable_at(),
                     "expected": o.knowable_at() <= nowd and o.evidence_through() < nowd, "admitted": d.admitted,
                     "channels": ",".join(sorted(str(c) for c in d.channels))})
    return pd.DataFrame(rows, columns=["offset", "evidence_through", "knowable_at", "expected", "admitted", "channels"])


def boundary_ok(sweep: pd.DataFrame) -> bool:
    """True when every offset's answer equals the exact rule and admission is monotone (never admitted after a refusal)."""
    if sweep.empty:
        return False
    s = sweep.sort_values("offset")
    adm = [bool(x) for x in s["admitted"]]
    first_refusal = next((i for i, a in enumerate(adm) if not a), len(adm))
    return bool((s["expected"] == s["admitted"]).all()) and not any(adm[first_refusal:])
