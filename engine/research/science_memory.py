"""Scientific long-term memory (contract C66 section 39) - IMPLEMENTED, NOT VALIDATED.

Memory that holds not "pattern worked" but the whole story: why it was proposed, why and how it was tested, what it predicted,
where it worked, where and why it failed, whether the failure transferred, whether the explanation survived, and what replaced
it. From that record it must answer two questions for any item:
    why_believe(item, now)      -> "Why do you currently believe this pattern is useful?"
    what_would_stop(item, now)  -> "What evidence would cause you to stop trusting it?"
Design: an append-only, hash-chained ledger of `Entry` records (one per step of the story, dated by when it became KNOWN, never by
when it is queried). Nothing is edited: a correction is a later entry. Every query takes `now` and sees only entries known strictly
before it (C56/C58). The ledger is MATURED_RESEARCH_STATE; it leaves only through `release`, a MaturedRecord the trader must gate.
It EXTENDS the existing stores rather than duplicating them: evidence is read from engine.research.research_graph.ResearchGraph
(`ingest_graph`), engine.learning.archive.Archive failure records (`ingest_archive`) and engine.research.decision_bridge entries
(`ingest_bridge`); the chain lane is engine.learning.archive.ChainFile, so a persistent ledger shares the repo's one tamper-evident
chain. Falsifiers are declared at proposal time and evaluated against new observations (`evaluate_stops`), so "what would stop you"
is a testable rule, not prose. Public entry: `step(mem, now)`."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from engine.learning.archive import ChainFile, _identity_errors
from engine.learning.core import (FailureCause, FirewallBreach, Provenance, _StrEnum, as_date, canonical_json,
                                  current_code_hash, require_past, stable_hash)
from engine.learning.knowledge_graph import wilson_lower
from engine.research.core import MaturedRecord, Namespace


class Stage(_StrEnum):
    PROPOSED = "PROPOSED"                  # why it was proposed (origin, motivating observation, falsifier)
    TESTED = "TESTED"                      # how it was tested (method, controls, sample) and the result
    PREDICTION = "PREDICTION"              # what it predicted, before the outcome
    WORKED = "WORKED"                      # where it worked
    FAILED = "FAILED"                      # where it failed
    FAILURE_EXPLAINED = "FAILURE_EXPLAINED"  # why it failed (cause, or UNKNOWN)
    TRANSFER = "TRANSFER"                  # whether the failure (or the success) transferred to another setting
    EXPLANATION_CHECK = "EXPLANATION_CHECK"  # whether an explanation survived a second look
    REPLACED = "REPLACED"                  # what replaced it
    RETIRED = "RETIRED"                    # withdrawn, with the reason
    REINSTATED = "REINSTATED"              # brought back, with the new evidence
    NOTE = "NOTE"                          # anything else worth keeping, never load-bearing


S = Stage
REQUIRED: dict[Stage, tuple[str, ...]] = {
    S.PROPOSED: ("why", "origin", "falsifier"),
    S.TESTED: ("method", "result", "n"),
    S.PREDICTION: ("claim", "horizon"),
    S.WORKED: ("context",),
    S.FAILED: ("context",),
    S.FAILURE_EXPLAINED: ("failure", "cause"),
    S.TRANSFER: ("failure_or_success", "from_context", "to_context", "transferred"),
    S.EXPLANATION_CHECK: ("failure", "survived", "how"),
    S.REPLACED: ("replacement", "why"),
    S.RETIRED: ("reason",),
    S.REINSTATED: ("reason", "evidence"),
    S.NOTE: ("text",)}
TEXT_FIELDS = frozenset({"why", "origin", "falsifier", "method", "result", "claim", "context", "failure", "how", "replacement",
                         "reason", "text", "from_context", "to_context", "evidence", "horizon", "cause", "failure_or_success"})
ORDER = {s: i for i, s in enumerate(Stage)}


class MemoryError_(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Falsifier:
    """A machine-checkable reason to stop trusting an item, declared when it is proposed. `kind` selects the test in
    `evaluate_stops`; `threshold` is its trigger level."""
    kind: str                                  # failures_in_gate / effect_below / transfer_lower_below / replicated_fail / explanation_fails / stale_days / contradicted_by
    threshold: float
    text: str = ""

    KINDS = ("failures_in_gate", "effect_below", "transfer_lower_below", "replicated_fail", "explanation_fails", "stale_days",
             "contradicted_by")

    def check(self) -> list[str]:
        errs = []
        if self.kind not in self.KINDS:
            errs.append(f"unknown falsifier kind {self.kind!r}; expected one of {self.KINDS}")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or math.isnan(self.threshold):
            errs.append("falsifier threshold must be a number")
        elif self.kind in ("transfer_lower_below",) and not 0 <= self.threshold <= 1:
            errs.append("transfer_lower_below threshold must be in [0,1]")
        elif self.kind in ("failures_in_gate", "replicated_fail", "stale_days", "contradicted_by") and self.threshold < 1:
            errs.append(f"{self.kind} threshold must be >= 1")
        return errs

    def to_dict(self) -> dict:
        return {"kind": self.kind, "threshold": float(self.threshold), "text": self.text}


@dataclasses.dataclass(frozen=True)
class Entry:
    """One dated step in the scientific history of one item. `known_at` is when the step became knowable."""
    entry_id: str
    item: str
    stage: Stage
    known_at: str
    payload_json: str
    evidence: tuple[str, ...] = ()             # ids of graph nodes / archive records / experiments this step rests on
    source: str = ""
    prev: str = ""
    hash: str = ""

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    def body(self) -> dict:
        return {"entry_id": self.entry_id, "item": self.item, "stage": self.stage.value, "known_at": self.known_at,
                "payload": self.payload, "evidence": list(self.evidence), "source": self.source}

    @staticmethod
    def make(item: str, stage: Stage | str, known_at, payload: Mapping[str, Any], evidence: Sequence[str] = (), source: str = "") -> "Entry":
        st = Stage.parse(stage)
        pj = canonical_json(dict(payload))
        d = str(as_date(known_at))
        return Entry(stable_hash([item, st.value, d, pj], 14), item, st, d, pj, tuple(evidence), source)

    def check(self) -> list[str]:
        errs = []
        if not self.item or not self.item.strip():
            errs.append("entry has no item")
        p = self.payload
        for f in REQUIRED[self.stage]:
            if f not in p or p[f] in (None, "") or (isinstance(p[f], str) and not p[f].strip()):
                errs.append(f"{self.stage.value}: payload.{f} required")
        if self.stage == S.PROPOSED and isinstance(p.get("falsifier"), Mapping):
            errs += Falsifier(**{k: v for k, v in p["falsifier"].items() if k in ("kind", "threshold", "text")}).check() \
                if {"kind", "threshold"} <= set(p["falsifier"]) else ["PROPOSED: falsifier needs kind and threshold"]
        elif self.stage == S.PROPOSED and "falsifier" in p:
            errs.append("PROPOSED: falsifier must be an object {kind, threshold}")
        if self.stage == S.FAILURE_EXPLAINED and p.get("cause") not in {c.value for c in FailureCause}:
            errs.append(f"FAILURE_EXPLAINED: unknown cause {p.get('cause')!r}")
        if self.stage == S.FAILURE_EXPLAINED and p.get("cause") not in (None, "UNKNOWN") and not (p.get("context") or self.evidence):
            errs.append("FAILURE_EXPLAINED: a named cause needs the explaining context or evidence (UNKNOWN needs neither)")
        if self.stage == S.TESTED and isinstance(p.get("n"), (int, float)) and p["n"] < 0:
            errs.append("TESTED: n < 0")
        if self.stage == S.TRANSFER and not isinstance(p.get("transferred"), bool):
            errs.append("TRANSFER: transferred must be true or false")
        if self.stage == S.EXPLANATION_CHECK and not isinstance(p.get("survived"), bool):
            errs.append("EXPLANATION_CHECK: survived must be true or false")
        if self.stage == S.REPLACED and p.get("replacement") == self.item:
            errs.append("REPLACED: an item cannot replace itself")
        errs += _identity_errors({k: v for k, v in p.items() if k in TEXT_FIELDS or isinstance(v, (Mapping, list))}, frozenset(), "payload")
        errs += _identity_errors({"item": self.item, "evidence": list(self.evidence)}, frozenset(), "entry")
        return errs


class Trust(_StrEnum):
    UNPROVEN = "UNPROVEN"                  # proposed, not yet tested
    SUPPORTED = "SUPPORTED"
    CONDITIONAL = "CONDITIONAL"            # works, but only where a recorded context says so / has unexplained failures
    DOUBTED = "DOUBTED"                    # a falsifier fired or an explanation collapsed
    REPLACED = "REPLACED"
    RETIRED = "RETIRED"
    UNKNOWN = "UNKNOWN"                    # no history at all


@dataclasses.dataclass(frozen=True)
class Standing:
    """Everything derivable from the ledger for one item at one date."""
    item: str
    now: str
    trust: Trust
    proposed: Mapping[str, Any] | None
    tests: tuple[Mapping[str, Any], ...]
    worked: tuple[str, ...]
    failed: tuple[str, ...]
    explained: Mapping[str, str]               # failure -> cause
    unexplained: tuple[str, ...]
    transfers: tuple[int, int]                 # (transferred, did not)
    transfer_lower: float | None
    checks: tuple[int, int]                    # (survived, collapsed)
    replaced_by: str
    retired: str
    last_evidence_at: str
    stages_seen: tuple[str, ...]
    reasons: tuple[str, ...]                   # why `trust` is what it is


@dataclasses.dataclass(frozen=True)
class Answer:
    """An answer to a scientific-history question, with every sentence tied to the entries it rests on."""
    item: str
    question: str
    lines: tuple[tuple[str, tuple[str, ...]], ...]      # (sentence, entry ids)
    unanswered: tuple[str, ...]                          # parts of the question the record cannot answer

    def text(self) -> str:
        out = [f"{self.question} [{self.item}]"] + [f"- {s}" for s, _ in self.lines]
        out += [f"- UNKNOWN: {u}" for u in self.unanswered]
        return "\n".join(out)

    @property
    def complete(self) -> bool:
        return not self.unanswered

    def cited(self) -> tuple[str, ...]:
        return tuple(sorted({e for _, es in self.lines for e in es}))


@dataclasses.dataclass(frozen=True)
class StopCondition:
    """One concrete thing that would end trust in an item, with how far away it is."""
    kind: str
    text: str
    threshold: float
    current: float | None
    declared_by: str                           # 'proposal' (the researchers' own falsifier) or 'derived' (from the record)
    distance: float | None                     # >0 = still trusted; <=0 = already met
    entry_ids: tuple[str, ...] = ()

    @property
    def met(self) -> bool:
        return self.distance is not None and self.distance <= 0


class ScienceMemory:
    """Append-only, hash-chained scientific ledger. `root=None` keeps it in memory."""

    def __init__(self, root=None, forbidden_identities: Iterable[str] = ()):
        self._chain = ChainFile(root, "sci")
        self.forbidden = frozenset(str(x).upper() for x in forbidden_identities)
        self._entries: list[Entry] = []
        self._by_item: dict[str, list[Entry]] = defaultdict(list)
        self._ids: set[str] = set()
        self._replay()

    def _replay(self) -> None:
        for line in self._chain.take_new():
            b = line["body"]
            e = Entry(b["entry_id"], b["item"], Stage(b["stage"]), b["known_at"], canonical_json(b["payload"]),
                      tuple(b["evidence"]), b["source"], line["prev"], line["hash"])
            self._index(e)

    def _index(self, e: Entry) -> None:
        self._entries.append(e)
        self._by_item[e.item].append(e)
        self._ids.add(e.entry_id)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self, now=None) -> list[str]:
        return sorted(i for i, es in self._by_item.items() if now is None or any(as_date(e.known_at) < as_date(now) for e in es))

    def verify(self) -> dict:
        return self._chain.verify()

    # ------------------------------------------------------------------ writing
    def append(self, e: Entry) -> Entry:
        """Validate and record one step. Idempotent for identical content. Ordering rules keep the story causal: a step cannot be
        known before the item was proposed; a REPLACED step must name an item that has been proposed and was not known later."""
        errs = e.check()
        if self.forbidden:
            errs += _identity_errors(e.payload, self.forbidden, "payload")
        if errs:
            raise MemoryError_(f"{e.item}/{e.stage.value}: " + "; ".join(errs))
        if e.entry_id in self._ids:
            return next(x for x in self._by_item[e.item] if x.entry_id == e.entry_id)
        have = self._by_item.get(e.item, [])
        prop = [x for x in have if x.stage == S.PROPOSED]
        if e.stage == S.PROPOSED and prop:
            raise MemoryError_(f"{e.item} was already proposed on {prop[0].known_at}; record a NOTE or a new item")
        if e.stage != S.PROPOSED:
            if not prop:
                raise MemoryError_(f"{e.item}: {e.stage.value} recorded before the item was ever PROPOSED")
            if as_date(e.known_at) < as_date(prop[0].known_at):
                raise MemoryError_(f"{e.item}: {e.stage.value} known {e.known_at} before it was proposed {prop[0].known_at}")
        if e.stage == S.REPLACED:
            rep = e.payload["replacement"]
            first = self._by_item.get(rep)
            if not first or not any(x.stage == S.PROPOSED for x in first):
                raise MemoryError_(f"{e.item}: replacement {rep} has no history")
            if as_date(first[0].known_at) > as_date(e.known_at):
                raise MemoryError_(f"{e.item}: replacement {rep} was proposed after the replacement was recorded")
            if self._replaces(rep, e.item):
                raise MemoryError_(f"{e.item}: replacement chain would loop through {rep}")
        if e.stage in (S.WORKED, S.FAILED, S.FAILURE_EXPLAINED, S.TRANSFER, S.EXPLANATION_CHECK) and not any(
                x.stage == S.TESTED for x in have) and e.stage != S.FAILURE_EXPLAINED:
            raise MemoryError_(f"{e.item}: {e.stage.value} recorded on an item that was never TESTED")
        if e.stage == S.FAILURE_EXPLAINED:
            fails = {x.payload["context"] for x in have if x.stage == S.FAILED}
            fid = {x.payload.get("failure") for x in have if x.stage == S.FAILED}
            if e.payload["failure"] not in fid and e.payload["failure"] not in fails:
                raise MemoryError_(f"{e.item}: explains failure {e.payload['failure']!r} that was never recorded as FAILED")
        if e.stage == S.EXPLANATION_CHECK and not any(x.stage == S.FAILURE_EXPLAINED and x.payload["failure"] == e.payload["failure"]
                                                      for x in have):
            raise MemoryError_(f"{e.item}: checks an explanation of {e.payload['failure']!r} that does not exist")
        last = max((as_date(x.known_at) for x in have), default=None)
        if last is not None and as_date(e.known_at) < last and e.stage in (S.RETIRED, S.REPLACED, S.REINSTATED):
            raise MemoryError_(f"{e.item}: {e.stage.value} dated {e.known_at} is before the latest entry ({last}); history is not rewritten")
        self._chain.append_many([e.body()])
        self._replay()
        return next(x for x in self._by_item[e.item] if x.entry_id == e.entry_id)

    def _replaces(self, start: str, target: str) -> bool:
        seen, cur = set(), start
        while cur not in seen:
            seen.add(cur)
            nxt = next((x.payload["replacement"] for x in self._by_item.get(cur, []) if x.stage == S.REPLACED), None)
            if nxt is None:
                return False
            if nxt == target:
                return True
            cur = nxt
        return False

    # ------------------------------------------------------------------ typed writers
    def propose(self, item: str, known_at, why: str, origin: str, falsifier: Falsifier, source: str = "", evidence: Sequence[str] = (),
                **extra) -> Entry:
        return self.append(Entry.make(item, S.PROPOSED, known_at, {"why": why, "origin": origin, "falsifier": falsifier.to_dict(), **extra},
                                      evidence, source))

    def tested(self, item: str, known_at, method: str, result: str, n: int, effect: float | None = None, se: float | None = None,
               controls: Sequence[str] = (), holdout: bool = False, evidence: Sequence[str] = (), source: str = "") -> Entry:
        p: dict[str, Any] = {"method": method, "result": result, "n": int(n), "controls": list(controls), "holdout": bool(holdout)}
        if effect is not None:
            p["effect"] = float(effect)
        if se is not None:
            p["se"] = float(se)
        return self.append(Entry.make(item, S.TESTED, known_at, p, evidence, source))

    def predicted(self, item: str, known_at, claim: str, horizon: str, evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.PREDICTION, known_at, {"claim": claim, "horizon": horizon}, evidence))

    def worked(self, item: str, known_at, context: str, evidence: Sequence[str] = (), n: int | None = None) -> Entry:
        return self.append(Entry.make(item, S.WORKED, known_at, {"context": context, **({"n": n} if n is not None else {})}, evidence))

    def failed(self, item: str, known_at, context: str, failure: str | None = None, evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.FAILED, known_at, {"context": context, "failure": failure or context}, evidence))

    def explained(self, item: str, known_at, failure: str, cause: FailureCause | str, context: str = "", note: str = "",
                  evidence: Sequence[str] = ()) -> Entry:
        p = {"failure": failure, "cause": FailureCause.parse(cause).value, "note": note}
        if context:
            p["context"] = context
        return self.append(Entry.make(item, S.FAILURE_EXPLAINED, known_at, p, evidence))

    def transferred(self, item: str, known_at, subject: str, from_context: str, to_context: str, transferred: bool,
                    evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.TRANSFER, known_at, {"failure_or_success": subject, "from_context": from_context,
                                                                   "to_context": to_context, "transferred": bool(transferred)}, evidence))

    def checked(self, item: str, known_at, failure: str, survived: bool, how: str, evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.EXPLANATION_CHECK, known_at, {"failure": failure, "survived": bool(survived), "how": how}, evidence))

    def replaced(self, item: str, known_at, replacement: str, why: str, evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.REPLACED, known_at, {"replacement": replacement, "why": why}, evidence))

    def retired(self, item: str, known_at, reason: str, evidence: Sequence[str] = ()) -> Entry:
        return self.append(Entry.make(item, S.RETIRED, known_at, {"reason": reason}, evidence))

    def reinstated(self, item: str, known_at, reason: str, evidence: str) -> Entry:
        return self.append(Entry.make(item, S.REINSTATED, known_at, {"reason": reason, "evidence": evidence}, ()))

    def note(self, item: str, known_at, text: str) -> Entry:
        return self.append(Entry.make(item, S.NOTE, known_at, {"text": text}))

    # ------------------------------------------------------------------ reading
    def history(self, item: str, now) -> list[Entry]:
        """The item's entries known strictly before `now`, in the order they became known (ties by story stage)."""
        n = as_date(now)
        return sorted((e for e in self._by_item.get(item, []) if as_date(e.known_at) < n),
                      key=lambda e: (as_date(e.known_at), ORDER[e.stage], e.entry_id))

    def entry(self, entry_id: str) -> Entry:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        raise MemoryError_(f"unknown entry {entry_id}")

    def chain_ok(self) -> bool:
        return bool(self._chain.verify()["ok"])


# ====================================================================================================== standing

def standing(mem: ScienceMemory, item: str, now) -> Standing:
    """Reduce the item's history to what is currently true of it. RETIRED / REPLACED stand until a later REINSTATED; a fired
    falsifier or a collapsed explanation makes it DOUBTED; an item that works but has failures with no explanation, or that failed to
    transfer, is CONDITIONAL; SUPPORTED needs a positive test, at least one WORKED context and nothing above; no test is UNPROVEN.
    Each verdict carries its reasons."""
    hist = mem.history(item, now)
    if not hist:
        return Standing(item, str(as_date(now)), Trust.UNKNOWN, None, (), (), (), {}, (), (0, 0), None, (0, 0), "", "", "", (), ("no history",))
    proposed = next((e.payload for e in hist if e.stage == S.PROPOSED), None)
    tests = tuple(e.payload for e in hist if e.stage == S.TESTED)
    worked, failed = [], []
    explained: dict[str, str] = {}
    for e in hist:
        if e.stage == S.WORKED:
            worked.append(e.payload["context"])
        elif e.stage == S.FAILED:
            failed.append(e.payload["context"])
        elif e.stage == S.FAILURE_EXPLAINED:
            explained[e.payload["failure"]] = e.payload["cause"]
    failure_ids = {e.payload.get("failure", e.payload["context"]) for e in hist if e.stage == S.FAILED}
    unexplained = tuple(sorted(f for f in failure_ids if explained.get(f, "UNKNOWN") == "UNKNOWN"))
    tr = [e.payload["transferred"] for e in hist if e.stage == S.TRANSFER]
    checks = [e.payload["survived"] for e in hist if e.stage == S.EXPLANATION_CHECK]
    replaced_by = retired = ""
    for e in (e for e in hist if e.stage in (S.RETIRED, S.REPLACED, S.REINSTATED)):
        if e.stage == S.REINSTATED:
            replaced_by = retired = ""
        elif e.stage == S.RETIRED:
            retired = e.payload["reason"]
        else:
            replaced_by = e.payload["replacement"]
    reasons: list[str] = []
    if retired:
        trust = Trust.RETIRED
        reasons.append(f"retired: {retired}")
    elif replaced_by:
        trust = Trust.REPLACED
        reasons.append(f"replaced by {replaced_by}")
    else:
        fired = [c for c in evaluate_stops(mem, item, now) if c.met and c.declared_by == "proposal"]
        collapsed = checks.count(False)
        positive = [t for t in tests if str(t.get("result", "")).lower() in POSITIVE]
        if fired or (checks and collapsed > len(checks) - collapsed):
            trust = Trust.DOUBTED
            reasons += [f"falsifier met: {c.text}" for c in fired]
            if collapsed:
                reasons.append(f"{collapsed} of {len(checks)} explanation checks collapsed")
        elif not tests:
            trust = Trust.UNPROVEN
            reasons.append("proposed but never tested")
        elif positive and worked and not (unexplained or tr.count(False) > tr.count(True)):
            trust = Trust.SUPPORTED
            reasons.append(f"{len(positive)} positive test(s), works in {len(set(worked))} context(s), every failure explained")
        elif positive or worked:
            trust = Trust.CONDITIONAL
            if unexplained:
                reasons.append(f"{len(unexplained)} failure(s) with no explanation")
            if tr.count(False) > tr.count(True):
                reasons.append("failed to transfer more often than it transferred")
            if not worked:
                reasons.append("no context in which it is known to work")
            if not positive:
                reasons.append("no positive test on record")
        else:
            trust = Trust.DOUBTED
            reasons.append("tested with no positive result and no working context")
    dates = [as_date(e.known_at) for e in hist if e.stage in (S.TESTED, S.WORKED, S.FAILED, S.TRANSFER, S.EXPLANATION_CHECK)]
    return Standing(item, str(as_date(now)), trust, proposed, tests, tuple(sorted(set(worked))), tuple(sorted(set(failed))), dict(explained),
                    unexplained, (tr.count(True), tr.count(False)), wilson_lower(tr.count(True), len(tr)),
                    (checks.count(True), checks.count(False)), replaced_by, retired,
                    max(dates).isoformat() if dates else "", tuple(sorted({e.stage.value for e in hist})), tuple(reasons))


POSITIVE = ("positive", "supported", "pass", "passed", "improved")


def _effect_now(hist: Sequence[Entry]) -> tuple[float | None, float | None, list[str]]:
    """Inverse-variance pooled effect over tests that carried an effect and a standard error (else the latest effect alone)."""
    rows = [(e.payload["effect"], e.payload.get("se"), e.entry_id) for e in hist if e.stage == S.TESTED and "effect" in e.payload]
    if not rows:
        return None, None, []
    w = [(1.0 / (se * se), eff, eid) for eff, se, eid in rows if se and se > 0]
    if w:
        tot = sum(x for x, _, _ in w)
        return sum(x * eff for x, eff, _ in w) / tot, math.sqrt(1.0 / tot), [eid for _, _, eid in w]
    return rows[-1][0], None, [rows[-1][2]]


def evaluate_stops(mem: ScienceMemory, item: str, now, observations: Mapping[str, Any] | None = None) -> list[StopCondition]:
    """Every condition that would end trust in the item, and how close each is. `observations` tests hypothetical or fresh evidence
    without writing it: keys failures_in_gate, effect, transfer_lower, replicated_fail, explanation_fails, age_days, contradicted_by,
    unexplained_failures. The proposal's own falsifier comes first (declared_by='proposal'); derived ones come from the record."""
    hist = mem.history(item, now)
    obs = dict(observations or {})
    out: list[StopCondition] = []
    prop = next((e for e in hist if e.stage == S.PROPOSED), None)
    st_fails = [e for e in hist if e.stage == S.FAILED]
    eff, se, eff_ids = _effect_now(hist)
    tr = [e.payload["transferred"] for e in hist if e.stage == S.TRANSFER]
    checks = [e for e in hist if e.stage == S.EXPLANATION_CHECK]
    dates = [as_date(e.known_at) for e in hist if e.stage in (S.TESTED, S.WORKED, S.TRANSFER)]
    age = (as_date(now) - max(dates)).days if dates else None

    def add(kind, text, thr, cur, by, higher_is_worse=True, ids=()):
        dist = None if cur is None else (thr - cur if higher_is_worse else cur - thr)
        out.append(StopCondition(kind, text, float(thr), None if cur is None else float(cur), by, None if dist is None else float(dist), tuple(ids)))

    if prop is not None:
        f = prop.payload["falsifier"]
        k, thr = f["kind"], float(f["threshold"])
        txt = f.get("text") or k
        if k == "failures_in_gate":
            add(k, f"{txt}: failures inside its own gate reach {thr:g}", thr, obs.get(k, len(st_fails)), "proposal", True, [e.entry_id for e in st_fails])
        elif k == "effect_below":
            add(k, f"{txt}: pooled effect falls to {thr:g} or below", thr, obs.get("effect", eff), "proposal", False, eff_ids)
        elif k == "transfer_lower_below":
            add(k, f"{txt}: transfer lower bound below {thr:g}", thr, obs.get("transfer_lower", wilson_lower(tr.count(True), len(tr))), "proposal", False)
        elif k == "replicated_fail":
            cur = obs.get(k, sum(1 for e in hist if e.stage == S.TRANSFER and not e.payload["transferred"]))
            add(k, f"{txt}: {thr:g} independent replication failure(s)", thr, cur, "proposal", True)
        elif k == "explanation_fails":
            cur = obs.get(k, sum(1 for e in checks if not e.payload["survived"]))
            add(k, f"{txt}: {thr:g} explanation check(s) collapse", thr, cur, "proposal", True, [e.entry_id for e in checks])
        elif k == "stale_days":
            add(k, f"{txt}: no fresh evidence for {thr:g} days", thr, obs.get("age_days", age), "proposal", True)
        elif k == "contradicted_by":
            add(k, f"{txt}: contradicted by {thr:g} independent item(s)", thr, obs.get(k, 0), "proposal", True)
    if eff is not None and se:
        edge = eff - 1.96 * se if eff > 0 else -(eff + 1.96 * se)
        add("effect_interval_touches_zero", "the pooled effect's 95% interval reaches zero", 0.0, obs.get("interval_edge", edge), "derived", False, eff_ids)
    if st_fails:
        unexpl = [e for e in st_fails if not any(x.stage == S.FAILURE_EXPLAINED and x.payload["cause"] != "UNKNOWN"
                                                 and x.payload["failure"] == e.payload.get("failure", e.payload["context"]) for x in hist)]
        add("unexplained_failures", "unexplained failures accumulate to three (a pattern that fails for no known reason is not trusted)",
            3.0, obs.get("unexplained_failures", len(unexpl)), "derived", True, [e.entry_id for e in unexpl])
    if tr:
        add("transfer_majority_fails", "more failed transfers than successful ones", 0.0, tr.count(True) - tr.count(False), "derived", False)
    if age is not None:
        add("silence", "no supporting evidence for a year", 365.0, obs.get("age_days", age), "derived", True)
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def why_believe(mem: ScienceMemory, item: str, now) -> Answer:
    """'Why do you currently believe this pattern is useful?' Built only from recorded steps; each sentence cites its entries; every
    part of the story that was never recorded is listed as UNKNOWN rather than filled in."""
    hist = mem.history(item, now)
    q = "Why do you currently believe this is useful?"
    if not hist:
        return Answer(item, q, (), ("nothing is recorded about this item",))
    sd = standing(mem, item, now)
    by: dict[Stage, list[Entry]] = defaultdict(list)
    for e in hist:
        by[e.stage].append(e)
    lines: list[tuple[str, tuple[str, ...]]] = []
    miss: list[str] = []
    if sd.trust in (Trust.RETIRED, Trust.REPLACED):
        e = by[S.RETIRED][-1] if sd.trust == Trust.RETIRED else by[S.REPLACED][-1]
        lines.append((f"I do NOT currently believe it: it is {sd.trust.value.lower()} ({sd.reasons[0]}).", (e.entry_id,)))
    else:
        lines.append((f"Current standing: {sd.trust.value} - {'; '.join(sd.reasons)}.", ()))
    p = by[S.PROPOSED][0]
    lines.append((f"It was proposed because {p.payload['why']} (origin: {p.payload['origin']}).", (p.entry_id,)))
    if by[S.TESTED]:
        for e in by[S.TESTED][-3:]:
            pl = e.payload
            eff = (f", effect {pl['effect']:+.4g}" + (f" +/- {pl['se']:.2g}" if pl.get("se") else "")) if "effect" in pl else ""
            lines.append((f"Tested by {pl['method']} on n={pl['n']}{' (held out)' if pl.get('holdout') else ''}: {pl['result']}{eff}.", (e.entry_id,)))
        if not any(e.payload.get("holdout") for e in by[S.TESTED]):
            miss.append("no test on held-out data is recorded")
    else:
        miss.append("how it was tested: never tested")
    if by[S.PREDICTION]:
        e = by[S.PREDICTION][-1]
        lines.append((f"It predicts: {e.payload['claim']} over {e.payload['horizon']}.", (e.entry_id,)))
    else:
        miss.append("what it predicted: no prediction recorded")
    if sd.worked:
        lines.append((f"It has worked in {', '.join(sd.worked)}.", tuple(e.entry_id for e in by[S.WORKED])))
    else:
        miss.append("where it worked: no context recorded")
    if sd.failed:
        lines.append((f"It has failed in {', '.join(sd.failed)}.", tuple(e.entry_id for e in by[S.FAILED])))
        for f, cause in sorted(sd.explained.items()):
            ids = tuple(e.entry_id for e in by[S.FAILURE_EXPLAINED] if e.payload["failure"] == f)
            lines.append((f"Failure {f}: " + ("cause unknown" if cause == "UNKNOWN" else f"explained as {cause.lower().replace('_', ' ')}"), ids))
        for f in sd.unexplained:
            if f not in sd.explained:
                miss.append(f"why it failed in {f}: no explanation recorded")
    if sd.transfers != (0, 0):
        lines.append((f"Transfer: {sd.transfers[0]} settings transferred, {sd.transfers[1]} did not (lower bound {_fmt(sd.transfer_lower)}).",
                      tuple(e.entry_id for e in by[S.TRANSFER])))
    else:
        miss.append("whether it transfers: never tried elsewhere")
    if sum(sd.checks):
        lines.append((f"Explanations re-checked: {sd.checks[0]} survived, {sd.checks[1]} collapsed.", tuple(e.entry_id for e in by[S.EXPLANATION_CHECK])))
    elif any(c != "UNKNOWN" for c in sd.explained.values()):
        miss.append("whether its failure explanation survived a second look")
    if sd.replaced_by:
        lines.append((f"It was replaced by {sd.replaced_by}.", tuple(e.entry_id for e in by[S.REPLACED])))
    if by[S.REINSTATED]:
        lines.append((f"It was reinstated: {by[S.REINSTATED][-1].payload['reason']}.", (by[S.REINSTATED][-1].entry_id,)))
    return Answer(item, q, tuple(lines), tuple(miss))


def what_would_stop(mem: ScienceMemory, item: str, now, observations: Mapping[str, Any] | None = None) -> Answer:
    """'What evidence would cause you to stop trusting it?' The researchers' own falsifier first, then what the record implies, each
    with its current distance. A missing falsifier is stated: an unfalsifiable claim is itself a finding."""
    q = "What evidence would make me stop trusting this?"
    if not mem.history(item, now):
        return Answer(item, q, (), ("nothing is recorded about this item",))
    conds = evaluate_stops(mem, item, now, observations)
    lines, miss = [], []
    if not any(c.declared_by == "proposal" for c in conds):
        miss.append("no falsifier was declared at proposal: nothing has ever been stated that would make me stop trusting it")
    for c in conds:
        state = "ALREADY MET" if c.met else ("distance unknown (no evidence yet)" if c.distance is None else f"{c.distance:.3g} away")
        lines.append((f"[{c.declared_by}] {c.text} (now {_fmt(c.current)}; {state}).", c.entry_ids))
    return Answer(item, q, tuple(lines), tuple(miss))


def succession(mem: ScienceMemory, item: str, now) -> list[str]:
    """What replaced it, and what replaced that: the chain of items from `item` forward as known before `now`."""
    chain, seen, cur = [item], {item}, item
    while True:
        h = mem.history(cur, now)
        rep = [e for e in h if e.stage == S.REPLACED]
        back = [e for e in h if e.stage == S.REINSTATED]
        if not rep or (back and ORDER_KEY(back[-1]) > ORDER_KEY(rep[-1])):
            return chain
        cur = rep[-1].payload["replacement"]
        if cur in seen:
            return chain
        seen.add(cur)
        chain.append(cur)


def ORDER_KEY(e: Entry) -> tuple:
    return (as_date(e.known_at), ORDER[e.stage])


def predecessors(mem: ScienceMemory, item: str, now) -> list[str]:
    """Items that this one replaced."""
    return sorted(e.item for e in mem._entries if e.stage == S.REPLACED and e.payload["replacement"] == item
                  and as_date(e.known_at) < as_date(now))


# ====================================================================================================== ingestion from the other stores

def ingest_graph(mem: ScienceMemory, graph, now, source: str = "research_graph", falsifier: Falsifier | None = None) -> dict[str, int]:
    """Write the graph's record of every belief into the ledger as dated steps. Contexts, failures, explanations, transfers and
    experiments keep the dates the GRAPH knew them. A belief with no proposal in the ledger gets one whose `why` says exactly
    that its origin was not recorded ('imported from the graph'), so it is visibly less documented than a proposed one, and its
    falsifier is the default. Idempotent (entries are content-addressed). Returns counts per stage."""
    from engine.research import research_graph as rg
    fal = falsifier or Falsifier("failures_in_gate", 3.0, "default: three failures inside its own gate")
    counts: Counter = Counter()

    def put(fn, *a, **k):
        before = len(mem)
        fn(*a, **k)
        return len(mem) - before

    for p in rg._active_beliefs(graph, now) + [n.node_id for n in graph.nodes(now) if graph.kind_of(n.node_id) in rg.BELIEF_KINDS
                                                and str(n.attrs.get("status", "")).lower() in rg.RETIRED_STATUSES]:
        nd = graph.node_at(p, now)
        counts["proposed"] += put(mem.propose, p, nd.known_at, "imported from the research graph: origin not recorded", "graph:" + source, fal, source)
        for e in graph.by_role(p, rg.R.VALIDATED_BY_EXP, now) + graph.by_role(p, rg.R.REFUTED_BY_EXP, now):
            exp, edge = e
            pos = rg.R.VALIDATED_BY_EXP in graph.roles_of(edge)
            en = graph.node_at(exp, now)
            counts["tested"] += put(mem.tested, p, edge.known_at, en.attrs.get("method") or "graph experiment", "positive" if pos else "negative",
                                    int(en.attrs.get("n", 1) or 1), en.attrs.get("effect"), None, (), False, (exp,), source)
        for c, v in graph.contexts_of(p, now).items():
            if v["works"] and any(x.stage == S.TESTED for x in mem.history(p, now)):
                counts["worked"] += put(mem.worked, p, _edge_date(graph, p, c, now), c, (c,))
            if v["fails"] and any(x.stage == S.TESTED for x in mem.history(p, now)):
                counts["failed"] += put(mem.failed, p, _edge_date(graph, p, c, now), c, c, (c,))
        for f in graph.failures_of(p, now):
            fn = graph.node_at(f, now)
            ex = graph.targets(f, rg.R.EXPLAINED_BY, now)
            fd = fn.known_at
            if any(x.stage == S.TESTED for x in mem.history(p, now)) and not any(x.stage == S.FAILED and x.payload.get("failure") == f
                                                                                 for x in mem.history(p, now)):
                counts["failed"] += put(mem.failed, p, fd, ex[0] if ex else f, f, (f,))
            if any(x.stage == S.FAILED and x.payload.get("failure") == f for x in mem.history(p, now)):
                cause = fn.attrs.get("cause", "UNKNOWN")
                counts["explained"] += put(mem.explained, p, fd, f, cause, ex[0] if ex and cause != "UNKNOWN" else "", fn.attrs.get("note", ""), (f,))
        for bed, e in graph.by_role(p, rg.R.TRANSFERS_TO, now):
            a = graph.role_attrs(e, rg.R.TRANSFERS_TO)
            if any(x.stage == S.TESTED for x in mem.history(p, now)):
                counts["transfer"] += put(mem.transferred, p, e.known_at, "result", a.get("axis") or "origin", bed, bool(a.get("success", True)), (bed,))
    return dict(counts)


def _edge_date(graph, pattern: str, context: str, now) -> str:
    dates = [e.known_at for _, e in graph.neighbors(pattern, now, None, "both") if _ == context]
    return min(dates, key=as_date) if dates else graph.node_at(pattern, now).known_at


def ingest_archive(mem: ScienceMemory, archive, now) -> dict[str, int]:
    """Archive `failure` records for items the ledger already tracks become FAILED (+ FAILURE_EXPLAINED with the archive's cause; UNKNOWN
    stays UNKNOWN). Records for items with no ledger history are counted as `orphans`, never invented into a story."""
    counts: Counter = Counter()
    for r in sorted(archive.failures(now), key=lambda r: (r.matured_at, r.seq)):
        if not mem.history(r.subject, now) or not any(x.stage == S.TESTED for x in mem.history(r.subject, now)):
            counts["orphans"] += 1
            continue
        ctx = ",".join(f"{k}={v}" for k, v in r.contexts) or "unrecorded_context"
        mem.failed(r.subject, r.matured_at, ctx, r.rec_id, (r.rec_id,))
        cause = r.payload["cause"]
        mem.explained(r.subject, r.matured_at, r.rec_id, cause, ctx if cause != "UNKNOWN" and r.contexts else "", "" if cause == "UNKNOWN" or r.contexts
                      else "archived without context", (r.rec_id,) if cause != "UNKNOWN" and not r.contexts else ())
        counts["failures"] += 1
    return dict(counts)


def ingest_bridge(mem: ScienceMemory, bridge, now) -> dict[str, int]:
    """Bridge dispositions as NOTE entries on the items a discovery is about: what decision each finding was allowed to change (or that
    it was informational). Notes are never load-bearing for trust; they let the history answer 'and what did we do with it?'."""
    counts: Counter = Counter()
    for e in bridge.entries():
        d = bridge.discovery(e.discovery_id)
        for s in d.subjects:
            if mem.history(s, dt.date.max) and as_date(e.recorded_at) < as_date(now):
                mem.note(s, e.recorded_at, f"bridge {e.disposition.value}: {e.answer[:180]}")
                counts[e.disposition.value] += 1
    return dict(counts)


# ====================================================================================================== audit

@dataclasses.dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    item: str
    detail: str


def audit(mem: ScienceMemory, now) -> list[Finding]:
    """Integrity and honesty checks over the whole ledger as known at `now`."""
    out: list[Finding] = []
    if not mem.chain_ok():
        out.append(Finding("CHAIN_BROKEN", "error", "*", "ledger hash chain failed verification"))
    for item in mem.items(now):
        h = mem.history(item, now)
        stages = [e.stage for e in h]
        if S.PROPOSED not in stages:
            out.append(Finding("NO_PROPOSAL", "error", item, "history without a proposal"))
            continue
        prop = next(e for e in h if e.stage == S.PROPOSED)
        if str(prop.payload["why"]).startswith("imported"):
            out.append(Finding("ORIGIN_UNRECORDED", "warn", item, "why it was proposed was never recorded"))
        for e in h:
            if e.stage != S.PROPOSED and as_date(e.known_at) < as_date(prop.known_at):
                out.append(Finding("BEFORE_PROPOSAL", "error", item, f"{e.stage.value} known before the proposal"))
        tested = [e for e in h if e.stage == S.TESTED]
        if any(s in (S.WORKED, S.FAILED) for s in stages) and not tested:
            out.append(Finding("CLAIM_WITHOUT_TEST", "error", item, "worked/failed recorded with no test"))
        if tested and not any(e.payload.get("holdout") for e in tested) and not any(s == S.TRANSFER for s in stages):
            out.append(Finding("NEVER_OUT_OF_SAMPLE", "warn", item, "no held-out test and no transfer"))
        for e in h:
            if e.stage == S.FAILURE_EXPLAINED and e.payload["cause"] != "UNKNOWN":
                if not any(c.stage == S.EXPLANATION_CHECK and c.payload["failure"] == e.payload["failure"] for c in h):
                    days = (as_date(now) - as_date(e.known_at)).days
                    if days > 90:
                        out.append(Finding("EXPLANATION_UNCHECKED", "warn", item, f"explanation of {e.payload['failure']} unchecked for {days} days"))
        sd = standing(mem, item, now)
        if sd.trust in (Trust.SUPPORTED, Trust.CONDITIONAL) and sd.replaced_by:
            out.append(Finding("REPLACED_BUT_TRUSTED", "error", item, "replaced yet still trusted"))
        for c in evaluate_stops(mem, item, now):
            if c.met and c.declared_by == "proposal" and sd.trust not in (Trust.DOUBTED, Trust.RETIRED, Trust.REPLACED):
                out.append(Finding("FALSIFIER_IGNORED", "error", item, f"falsifier met but item is {sd.trust.value}: {c.text}"))
        if sd.trust == Trust.SUPPORTED and sd.last_evidence_at and (as_date(now) - as_date(sd.last_evidence_at)).days > 365:
            out.append(Finding("SUPPORT_GONE_STALE", "warn", item, "supported on evidence older than a year"))
        if S.REPLACED in stages:
            rep = [e for e in h if e.stage == S.REPLACED][-1].payload["replacement"]
            rs = standing(mem, rep, now)
            if rs.trust in (Trust.RETIRED, Trust.DOUBTED, Trust.UNKNOWN):
                out.append(Finding("REPLACEMENT_NOT_BETTER", "warn", item, f"replaced by {rep}, which is now {rs.trust.value}"))
    return out


def story_completeness(mem: ScienceMemory, now) -> dict[str, Any]:
    """Across every item: how many can answer each of the section-39 questions (why proposed, why/how tested, what predicted, where worked,
    where failed, why failed, whether failure transferred, whether explanation survived, what replaced it where replaced)."""
    items = mem.items(now)
    qs = {"why_proposed": 0, "how_tested": 0, "predicted": 0, "where_worked": 0, "where_failed": 0, "why_failed": 0,
          "failure_transferred": 0, "explanation_checked": 0}
    with_failure = with_expl = 0
    for i in items:
        h = mem.history(i, now)
        st = {e.stage for e in h}
        prop = next((e for e in h if e.stage == S.PROPOSED), None)
        qs["why_proposed"] += bool(prop) and not str(prop.payload["why"]).startswith("imported")
        qs["how_tested"] += S.TESTED in st
        qs["predicted"] += S.PREDICTION in st
        qs["where_worked"] += S.WORKED in st
        qs["where_failed"] += S.FAILED in st
        if S.FAILED in st:
            with_failure += 1
            qs["why_failed"] += any(e.stage == S.FAILURE_EXPLAINED and e.payload["cause"] != "UNKNOWN" for e in h)
            qs["failure_transferred"] += S.TRANSFER in st
            if any(e.stage == S.FAILURE_EXPLAINED and e.payload["cause"] != "UNKNOWN" for e in h):
                with_expl += 1
                qs["explanation_checked"] += S.EXPLANATION_CHECK in st
    n = len(items)
    denom = {"why_failed": with_failure, "failure_transferred": with_failure, "explanation_checked": with_expl}
    return {"items": n, "answerable": {k: v for k, v in qs.items()}, "of": {k: denom.get(k, n) for k in qs},
            "share": {k: (v / denom.get(k, n)) if denom.get(k, n) else None for k, v in qs.items()}}


def trust_table(mem: ScienceMemory, now) -> dict[str, int]:
    return dict(sorted(Counter(standing(mem, i, now).trust.value for i in mem.items(now)).items()))


def unfalsifiable(mem: ScienceMemory, now) -> list[str]:
    """Items whose declared falsifier can never fire on the evidence the ledger can hold (threshold already unreachable or trivially
    met at proposal): a falsifier that cannot fail protects nothing."""
    bad = []
    for i in mem.items(now):
        prop = next((e for e in mem.history(i, now) if e.stage == S.PROPOSED), None)
        if prop is None:
            continue
        f = prop.payload["falsifier"]
        k, t = f["kind"], float(f["threshold"])
        if (k == "transfer_lower_below" and t <= 0) or (k == "stale_days" and t > 3650) or (k == "failures_in_gate" and t > 1000) \
                or (k == "effect_below" and t < -1.0):
            bad.append(i)
    return sorted(bad)


# ====================================================================================================== change over time

@dataclasses.dataclass(frozen=True)
class TrustChange:
    item: str
    before: str
    after: str
    why: str


def diff_standing(mem: ScienceMemory, t0, t1) -> list[TrustChange]:
    """Items whose trust changed between two dates, with the first reason at the later date."""
    if as_date(t1) <= as_date(t0):
        raise FirewallBreach("diff_standing needs t1 after t0")
    out = []
    for i in mem.items(t1):
        a, b = standing(mem, i, t0), standing(mem, i, t1)
        if a.trust != b.trust:
            out.append(TrustChange(i, a.trust.value, b.trust.value, b.reasons[0] if b.reasons else ""))
    return sorted(out, key=lambda c: (c.after, c.item))


def timeline(mem: ScienceMemory, item: str, dates: Sequence) -> list[tuple[str, str]]:
    """(date, trust) at each date: how belief in an item moved as the evidence arrived."""
    return [(str(as_date(d)), standing(mem, item, d).trust.value) for d in dates]


def lessons_repeated(mem: ScienceMemory, now, min_items: int = 3) -> list[tuple[str, int, list[str]]]:
    """Failure causes and contexts that recur across DIFFERENT items: (cause, items, item ids). A mistake the system keeps making in
    different clothes is the lesson it has not yet learned."""
    by: dict[str, set] = defaultdict(set)
    for i in mem.items(now):
        for e in mem.history(i, now):
            if e.stage == S.FAILURE_EXPLAINED and e.payload["cause"] != "UNKNOWN":
                by[e.payload["cause"]].add(i)
                if e.payload.get("context"):
                    by["context:" + e.payload["context"]].add(i)
    return sorted(((k, len(v), sorted(v)) for k, v in by.items() if len(v) >= min_items), key=lambda t: (-t[1], t[0]))


def explanation_survival(mem: ScienceMemory, now) -> dict[str, dict[str, float | int | None]]:
    """Per failure cause: how many explanations were checked and how many survived. A cause whose explanations keep collapsing is a
    label the system likes more than it earns."""
    by: dict[str, list[bool]] = defaultdict(list)
    for i in mem.items(now):
        h = mem.history(i, now)
        cause = {e.payload["failure"]: e.payload["cause"] for e in h if e.stage == S.FAILURE_EXPLAINED}
        for e in h:
            if e.stage == S.EXPLANATION_CHECK and e.payload["failure"] in cause:
                by[cause[e.payload["failure"]]].append(e.payload["survived"])
    return {c: {"checked": len(v), "survived": sum(v), "rate": sum(v) / len(v), "lower": wilson_lower(sum(v), len(v))}
            for c, v in sorted(by.items())}


def replacement_effect(mem: ScienceMemory, now) -> list[dict[str, Any]]:
    """For every replacement: did the replacement do better where the old one failed? Compared on recorded contexts."""
    rows = []
    for e in mem._entries:
        if e.stage != S.REPLACED or as_date(e.known_at) >= as_date(now):
            continue
        old, new = standing(mem, e.item, now), standing(mem, e.payload["replacement"], now)
        fixed = sorted(set(old.failed) & set(new.worked))
        broke = sorted(set(old.worked) & set(new.failed))
        rows.append({"old": e.item, "new": e.payload["replacement"], "why": e.payload["why"], "fixed": fixed, "broke": broke,
                     "new_trust": new.trust.value, "verdict": "BETTER" if fixed and not broke else "WORSE" if broke and not fixed else
                     "MIXED" if fixed and broke else "UNTESTED"})
    return sorted(rows, key=lambda r: (r["old"], r["new"]))


def next_experiments(mem: ScienceMemory, now, top: int = 10) -> list[tuple[str, str, float]]:
    """(item, what the record most needs, priority) - the questions the history itself leaves open, weighted by how much trust is
    currently resting on the item. Priority = trust weight x how many of the story's steps are missing."""
    w = {Trust.SUPPORTED: 1.0, Trust.CONDITIONAL: 0.8, Trust.UNPROVEN: 0.4, Trust.DOUBTED: 0.6}
    rows = []
    for i in mem.items(now):
        sd = standing(mem, i, now)
        if sd.trust not in w:
            continue
        need = []
        if not sd.tests:
            need.append("a first test")
        elif not any(t.get("holdout") for t in sd.tests):
            need.append("a held-out test")
        if sd.unexplained:
            need.append("an explanation for " + sd.unexplained[0])
        if sd.transfers == (0, 0) and sd.tests:
            need.append("a transfer test")
        if any(c != "UNKNOWN" for c in sd.explained.values()) and not sum(sd.checks):
            need.append("a check of its failure explanation")
        if need:
            rows.append((i, need[0], w[sd.trust] * len(need) / 4.0))
    return sorted(rows, key=lambda r: (-r[2], r[0]))[:top]


# ====================================================================================================== dossiers and release

@dataclasses.dataclass(frozen=True)
class Dossier:
    item: str
    now: str
    standing: Standing
    why: Answer
    stop: Answer
    lineage: tuple[str, ...]
    predecessors: tuple[str, ...]
    open_needs: tuple[str, ...]
    digest: str

    def markdown(self) -> str:
        sd = self.standing
        lines = [f"# {self.item} as of {self.now}", "", f"**Trust: {sd.trust.value}**", "", "## Why do you believe this?", ""]
        lines += [f"- {s}" for s, _ in self.why.lines] + [f"- *UNKNOWN:* {u}" for u in self.why.unanswered]
        lines += ["", "## What would make you stop?", ""]
        lines += [f"- {s}" for s, _ in self.stop.lines] + [f"- *UNKNOWN:* {u}" for u in self.stop.unanswered]
        if len(self.lineage) > 1:
            lines += ["", "Succession: " + " -> ".join(self.lineage)]
        if self.predecessors:
            lines += ["", "Replaced: " + ", ".join(self.predecessors)]
        if self.open_needs:
            lines += ["", "Still needed: " + "; ".join(self.open_needs)]
        return "\n".join(lines)


def dossier(mem: ScienceMemory, item: str, now) -> Dossier:
    """Everything the ledger can say about one item, in the two answers plus its succession and what the record still lacks."""
    sd = standing(mem, item, now)
    needs = tuple(r[1] for r in next_experiments(mem, now, 1000) if r[0] == item)
    h = mem.history(item, now)
    return Dossier(item, str(as_date(now)), sd, why_believe(mem, item, now), what_would_stop(mem, item, now),
                   tuple(succession(mem, item, now)), tuple(predecessors(mem, item, now)), needs, stable_hash([e.entry_id for e in h], 16))


def release(mem: ScienceMemory, item: str, now, created_real: str, replaying: Iterable[int] = ()) -> MaturedRecord:
    """Hand the trader side an identity-free summary of one item. The record matures on the newest entry it summarises, so the trader can
    open it only with `.gate(its_now)` after that date. Refused for an item that is not currently SUPPORTED or CONDITIONAL, and - the
    same-year rerun leak - when any real year the history spans is being replayed in disguise right now."""
    from engine.learning import trader_view
    h = mem.history(item, now)
    if not h:
        raise FirewallBreach(f"{item}: no history before {as_date(now)}")
    sd = standing(mem, item, now)
    if sd.trust not in (Trust.SUPPORTED, Trust.CONDITIONAL):
        raise FirewallBreach(f"{item} is {sd.trust.value}; only SUPPORTED or CONDITIONAL knowledge may be released")
    years = {as_date(e.known_at).year for e in h}
    clash = years & {int(y) for y in replaying}
    if clash:
        raise FirewallBreach(f"{item}: history spans year(s) {sorted(clash)} being replayed; releasing it would leak the answer into the rerun")
    matured = max(h, key=lambda e: as_date(e.known_at)).known_at
    payload = {"trust": sd.trust.value, "worked_in": list(sd.worked), "failed_in": list(sd.failed),
               "transfer": {"ok": sd.transfers[0], "failed": sd.transfers[1]}, "unexplained_failures": len(sd.unexplained),
               "stop_conditions": [{"kind": c.kind, "distance": c.distance} for c in evaluate_stops(mem, item, now)]}
    trader_view.assert_trader_safe(payload, f"release of {item}")
    prov = Provenance(created_real=created_real, learned_at=matured, code_hash=current_code_hash(), outcomes_seen_through=matured)
    return MaturedRecord(stable_hash([item, payload], 16), matured, payload, prov, Namespace.MATURED_RESEARCH)


# ====================================================================================================== persistence

def to_records(mem: ScienceMemory) -> dict:
    return {"entries": [e.body() for e in mem._entries]}


def from_records(rec: Mapping, forbidden_identities: Iterable[str] = ()) -> ScienceMemory:
    """Rebuild a ledger by replaying its entries through `append` (all validation runs again). The ids must reproduce."""
    mem = ScienceMemory(None, forbidden_identities)
    for b in rec["entries"]:
        e = mem.append(Entry.make(b["item"], b["stage"], b["known_at"], b["payload"], b["evidence"], b["source"]))
        if e.entry_id != b["entry_id"]:
            raise MemoryError_(f"entry {b['entry_id']} does not reproduce its id: the record was altered")
    return mem


def digest(mem: ScienceMemory, now) -> str:
    """Content hash of everything known before `now` (for proving the past did not change)."""
    return stable_hash([e.entry_id for i in mem.items(now) for e in mem.history(i, now)], 24)


def prefix_invariant(mem: ScienceMemory, dates: Sequence) -> list[str]:
    """Adding later entries must never change what an earlier date could see: every date's digest computed now must equal the one
    computed from the entries known before it alone. Returns the dates that fail (empty = the ledger is time-consistent)."""
    bad = []
    for d in dates:
        keep = [e for e in mem._entries if as_date(e.known_at) < as_date(d)]
        again = ScienceMemory(None)
        for e in sorted(keep, key=lambda e: (as_date(e.known_at), ORDER[e.stage], e.entry_id)):
            again.append(Entry.make(e.item, e.stage, e.known_at, e.payload, e.evidence, e.source))
        if digest(again, d) != digest(mem, d):
            bad.append(str(as_date(d)))
    return bad


# ====================================================================================================== entry

@dataclasses.dataclass(frozen=True)
class MemoryStepReport:
    now: str
    items: int
    trust: Mapping[str, int]
    findings: tuple[Finding, ...]
    completeness: Mapping[str, Any]
    unfalsifiable: tuple[str, ...]
    repeated_lessons: tuple[tuple[str, int, list[str]], ...]
    next: tuple[tuple[str, str, float], ...]
    changes: tuple[TrustChange, ...]
    ingested: Mapping[str, Mapping[str, int]]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")


def step(mem: ScienceMemory, now, graph=None, archive=None, bridge=None, previous=None, falsifier: Falsifier | None = None) -> MemoryStepReport:
    """PUBLIC ENTRY for the research loop: fold the graph, archive and bridge into the ledger as of `now`, then audit it, report which
    items lost or gained trust since `previous`, what lessons repeat, and what the record most needs next."""
    ing: dict[str, dict[str, int]] = {}
    if graph is not None:
        ing["graph"] = ingest_graph(mem, graph, now, falsifier=falsifier)
    if archive is not None:
        ing["archive"] = ingest_archive(mem, archive, now)
    if bridge is not None:
        ing["bridge"] = ingest_bridge(mem, bridge, now)
    ch = tuple(diff_standing(mem, previous, now)) if previous is not None and as_date(previous) < as_date(now) else ()
    return MemoryStepReport(str(as_date(now)), len(mem.items(now)), trust_table(mem, now), tuple(audit(mem, now)), story_completeness(mem, now),
                            tuple(unfalsifiable(mem, now)), tuple(lessons_repeated(mem, now)), tuple(next_experiments(mem, now)), ch, ing)


def memory_markdown(mem: ScienceMemory, now, top: int = 10) -> str:
    sc = story_completeness(mem, now)
    lines = [f"# Scientific memory as of {as_date(now)}", "", f"{sc['items']} items.", "", "| trust | items |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in trust_table(mem, now).items()]
    lines += ["", "| can answer | items | of |", "|---|---:|---:|"]
    lines += [f"| {k} | {sc['answerable'][k]} | {sc['of'][k]} |" for k in sc["answerable"]]
    lines += ["", "## What the record needs next", ""] + [f"- {i}: {n} (priority {p:.2f})" for i, n, p in next_experiments(mem, now, top)]
    fs = audit(mem, now)
    lines += ["", f"audit: {sum(f.severity == 'error' for f in fs)} errors, {sum(f.severity == 'warn' for f in fs)} warnings"]
    return "\n".join(lines)


# ====================================================================================================== quantitative strength of the case

@dataclasses.dataclass(frozen=True)
class Strength:
    """The evidential weight behind an item, kept as separate numbers (never one score)."""
    item: str
    tests: int
    positive: int
    holdout_tests: int
    pooled_effect: float | None
    pooled_se: float | None
    z: float | None
    contexts_worked: int
    contexts_failed: int
    replications: int                          # transfers that succeeded
    replication_lower: float | None
    failure_explained_share: float | None
    evidence_age_days: int | None
    independent_sources: int


def strength(mem: ScienceMemory, item: str, now) -> Strength:
    h = mem.history(item, now)
    tests = [e for e in h if e.stage == S.TESTED]
    eff, se, _ = _effect_now(h)
    sd = standing(mem, item, now)
    fails = {e.payload.get("failure", e.payload["context"]) for e in h if e.stage == S.FAILED}
    ex = {f for f, c in sd.explained.items() if c != "UNKNOWN"}
    age = (as_date(now) - as_date(sd.last_evidence_at)).days if sd.last_evidence_at else None
    return Strength(item, len(tests), sum(str(e.payload["result"]).lower() in POSITIVE for e in tests),
                    sum(bool(e.payload.get("holdout")) for e in tests), eff, se, (eff / se) if eff is not None and se else None,
                    len(sd.worked), len(sd.failed), sd.transfers[0], sd.transfer_lower, (len(ex & fails) / len(fails)) if fails else None, age,
                    len({e.source for e in tests if e.source}))


def compare(mem: ScienceMemory, a: str, b: str, now) -> dict[str, Any]:
    """Which of two items is better supported, dimension by dimension. It refuses to name an overall winner when the dimensions disagree."""
    sa, sb = strength(mem, a, now), strength(mem, b, now)
    dims = {"holdout_tests": (sa.holdout_tests, sb.holdout_tests, True), "replications": (sa.replications, sb.replications, True),
            "contexts_worked": (sa.contexts_worked, sb.contexts_worked, True), "contexts_failed": (sa.contexts_failed, sb.contexts_failed, False),
            "z": (sa.z, sb.z, True), "explained_share": (sa.failure_explained_share, sb.failure_explained_share, True)}
    verdict: dict[str, str] = {}
    for k, (x, y, higher) in dims.items():
        if x is None or y is None or x == y:
            verdict[k] = "tie" if x == y else "unknown"
        else:
            verdict[k] = a if (x > y) == higher else b
    wins = Counter(v for v in verdict.values() if v in (a, b))
    overall = "undecided"
    if wins[a] and not wins[b]:
        overall = a
    elif wins[b] and not wins[a]:
        overall = b
    return {"a": a, "b": b, "by_dimension": verdict, "overall": overall, "strength": (sa, sb)}


# ====================================================================================================== search

def find(mem: ScienceMemory, now, trust: Iterable[Trust | str] | None = None, worked_in: str | None = None, failed_in: str | None = None,
         cause: FailureCause | str | None = None, text: str | None = None, min_tests: int = 0) -> list[str]:
    """Items matching every given constraint: trust level, a context they worked / failed in, an explained failure cause, a word in
    the proposal's why/origin, a minimum number of tests."""
    want = {Trust.parse(t) for t in trust} if trust is not None else None
    cz = FailureCause.parse(cause).value if cause is not None else None
    out = []
    for i in mem.items(now):
        sd = standing(mem, i, now)
        if want is not None and sd.trust not in want:
            continue
        if worked_in and worked_in not in sd.worked:
            continue
        if failed_in and failed_in not in sd.failed:
            continue
        if cz and cz not in sd.explained.values():
            continue
        if len(sd.tests) < min_tests:
            continue
        if text:
            p = sd.proposed or {}
            hay = f"{p.get('why', '')} {p.get('origin', '')}".lower()
            if not all(w in hay for w in text.lower().split()):
                continue
        out.append(i)
    return out


def context_history(mem: ScienceMemory, context: str, now) -> dict[str, list[str]]:
    """Reverse index: every item that worked or failed in a context, and every failure attributed to it."""
    out: dict[str, list[str]] = {"worked": [], "failed": [], "explains": []}
    for i in mem.items(now):
        for e in mem.history(i, now):
            if e.stage == S.WORKED and e.payload["context"] == context:
                out["worked"].append(i)
            elif e.stage == S.FAILED and e.payload["context"] == context:
                out["failed"].append(i)
            elif e.stage == S.FAILURE_EXPLAINED and e.payload.get("context") == context:
                out["explains"].append(i)
    return {k: sorted(set(v)) for k, v in out.items()}


def context_reliability(mem: ScienceMemory, now, min_items: int = 3) -> list[dict[str, Any]]:
    """Per context: how many items worked vs failed there, with a Wilson bound; THIN under `min_items`."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i in mem.items(now):
        sd = standing(mem, i, now)
        for c in sd.worked:
            tally[c][0] += 1
        for c in sd.failed:
            tally[c][1] += 1
    rows = []
    for c, (w, f) in sorted(tally.items()):
        n = w + f
        rows.append({"context": c, "worked": w, "failed": f, "lower_work": wilson_lower(w, n),
                     "verdict": "THIN" if n < min_items else "HOSTILE" if f / n >= 0.66 else "FRIENDLY" if w / n >= 0.66 else "MIXED"})
    return rows


# ====================================================================================================== timing of learning

def explanation_lags(mem: ScienceMemory, now) -> dict[str, Any]:
    """Days from each failure to its explanation, and failures still unexplained: how quickly does the system understand its mistakes?"""
    lags, open_ = [], []
    for i in mem.items(now):
        h = mem.history(i, now)
        expl = {e.payload["failure"]: e for e in h if e.stage == S.FAILURE_EXPLAINED}
        for e in h:
            if e.stage == S.FAILED:
                key = e.payload.get("failure", e.payload["context"])
                x = expl.get(key)
                if x is None or x.payload["cause"] == "UNKNOWN":
                    open_.append((i, key, (as_date(now) - as_date(e.known_at)).days))
                else:
                    lags.append((as_date(x.known_at) - as_date(e.known_at)).days)
    lags.sort()
    med = lags[len(lags) // 2] if lags else None
    return {"explained": len(lags), "median_lag_days": med, "max_lag_days": lags[-1] if lags else None,
            "open": sorted(open_, key=lambda t: (-t[2], t[0])), "oldest_open_days": max((o[2] for o in open_), default=None)}


def evidence_age_profile(mem: ScienceMemory, now, buckets: Sequence[int] = (30, 90, 180, 365)) -> dict[str, int]:
    """How stale the evidence behind live (non-retired) items is: counts of items by age of their newest evidence."""
    edges = list(buckets)
    prof: Counter = Counter()
    for i in mem.items(now):
        sd = standing(mem, i, now)
        if sd.trust in (Trust.RETIRED, Trust.REPLACED, Trust.UNKNOWN) or not sd.last_evidence_at:
            continue
        age = (as_date(now) - as_date(sd.last_evidence_at)).days
        label = next((f"<={b}d" for b in edges if age <= b), f">{edges[-1]}d")
        prof[label] += 1
    return dict(sorted(prof.items(), key=lambda kv: (len(kv[0]), kv[0])))


def falsifier_record(mem: ScienceMemory, now) -> dict[str, Any]:
    """Did the declared falsifiers do their job? Of the items that were retired or replaced, how many had a falsifier that had fired
    BEFORE the withdrawal (good: the rule worked) versus none (withdrawn on judgment, or the falsifier was blind)."""
    fired_first = judged = 0
    for i in mem.items(now):
        h = mem.history(i, now)
        w = next((e for e in h if e.stage in (S.RETIRED, S.REPLACED)), None)
        if w is None:
            continue
        before = ScienceMemory(None)
        for e in h:
            if as_date(e.known_at) < as_date(w.known_at):
                before.append(Entry.make(e.item, e.stage, e.known_at, e.payload, e.evidence, e.source))
        conds = evaluate_stops(before, i, w.known_at)
        if any(c.met and c.declared_by == "proposal" for c in conds):
            fired_first += 1
        else:
            judged += 1
    n = fired_first + judged
    return {"withdrawn": n, "falsifier_fired_first": fired_first, "withdrawn_on_judgment": judged,
            "falsifier_rate": (fired_first / n) if n else None}


def reinstatement_audit(mem: ScienceMemory, now) -> list[Finding]:
    """A reinstatement must rest on evidence known AFTER the withdrawal it reverses; otherwise the same evidence is being asked to
    give two opposite answers."""
    out = []
    for i in mem.items(now):
        h = mem.history(i, now)
        last_out = None
        for e in h:
            if e.stage in (S.RETIRED, S.REPLACED):
                last_out = e
            elif e.stage == S.REINSTATED:
                if last_out is None:
                    out.append(Finding("REINSTATED_NEVER_WITHDRAWN", "error", i, "reinstated but never retired or replaced"))
                    continue
                fresh = [x for x in h if x.stage in (S.TESTED, S.WORKED, S.TRANSFER) and as_date(x.known_at) > as_date(last_out.known_at)
                         and as_date(x.known_at) <= as_date(e.known_at)]
                if not fresh:
                    out.append(Finding("REINSTATED_WITHOUT_NEW_EVIDENCE", "error", i, "no test, success or transfer between the withdrawal and the reinstatement"))
                last_out = None
    return out


def narrative(mem: ScienceMemory, item: str, now) -> str:
    """The item's history in order, one line per step, dated: the scientific diary of one belief."""
    lines = []
    for e in mem.history(item, now):
        p = e.payload
        t = {S.PROPOSED: lambda: f"proposed: {p['why']} (falsifier: {p['falsifier']['kind']} {p['falsifier']['threshold']:g})",
             S.TESTED: lambda: f"tested by {p['method']}, n={p['n']}: {p['result']}",
             S.PREDICTION: lambda: f"predicted {p['claim']} over {p['horizon']}",
             S.WORKED: lambda: f"worked in {p['context']}",
             S.FAILED: lambda: f"failed in {p['context']}",
             S.FAILURE_EXPLAINED: lambda: f"failure {p['failure']} explained as {p['cause']}",
             S.TRANSFER: lambda: f"{p['failure_or_success']} {'transferred' if p['transferred'] else 'did not transfer'} from {p['from_context']} to {p['to_context']}",
             S.EXPLANATION_CHECK: lambda: f"explanation of {p['failure']} {'survived' if p['survived'] else 'collapsed'} ({p['how']})",
             S.REPLACED: lambda: f"replaced by {p['replacement']}: {p['why']}",
             S.RETIRED: lambda: f"retired: {p['reason']}",
             S.REINSTATED: lambda: f"reinstated: {p['reason']}",
             S.NOTE: lambda: f"note: {p['text']}"}[e.stage]()
        lines.append(f"{e.known_at}  {t}")
    return "\n".join(lines)


def answers_everything(mem: ScienceMemory, now) -> list[str]:
    """Live items that cannot answer BOTH section-39 questions completely. A ledger is only memory of the process if every live
    item can say why it is believed and what would end that belief."""
    bad = []
    for i in mem.items(now):
        if standing(mem, i, now).trust in (Trust.RETIRED, Trust.REPLACED):
            continue
        if not (why_believe(mem, i, now).complete and what_would_stop(mem, i, now).complete):
            bad.append(i)
    return sorted(bad)


def forgotten(mem: ScienceMemory, graph, now) -> list[str]:
    """Beliefs in the research graph that the ledger has no history for: knowledge that exists without a scientific record."""
    from engine.research import research_graph as rg
    have = set(mem.items(now))
    return sorted(p for p in rg._active_beliefs(graph, now) if p not in have)


def disagreements_with_graph(mem: ScienceMemory, graph, now) -> list[str]:
    """Items where the ledger and the graph tell different stories: the graph shows a failure or refutation the ledger has not
    recorded, or the ledger has retired an item the graph still lists as active."""
    from engine.research import research_graph as rg
    out = []
    active = set(rg._active_beliefs(graph, now))
    for i in mem.items(now):
        sd = standing(mem, i, now)
        if sd.trust in (Trust.RETIRED, Trust.REPLACED) and i in active:
            out.append(f"{i}: ledger says {sd.trust.value.lower()}, graph still lists it active")
        if graph.has_node(i, now):
            gf = len(graph.failures_of(i, now))
            if gf > len(sd.failed) and sd.trust != Trust.UNKNOWN:
                out.append(f"{i}: graph holds {gf} failure(s), ledger records {len(sd.failed)}")
    return sorted(out)
