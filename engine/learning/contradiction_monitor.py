"""Contradiction monitor (contract C62 sections 18, 46; checklist J09) - IMPLEMENTED, NOT VALIDATED.

Runs every period over the knowledge graph (knowledge_graph.py CONTRADICTS edges) and the investigation ledger
(contradiction.py). For every contradiction it tracks: when it was born, how old it is, whether it has been investigated, whether
a context resolved it, and whether it is unresolved. It flags contradictions that are GROWING in evidence and ones left
UNINVESTIGATED too long, escalates the worst, raises research questions in research_priority.Signal shape, and exports rows for
reports.health_dashboard. Two contradicting items are never averaged: the only sanctioned read is `estimate`, which answers per
context or says UNKNOWN. Every query takes `now` and sees only what was known strictly before it (C56/C58)."""
from __future__ import annotations

import dataclasses
import enum
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .contradiction import ContradictionError, ContradictionLedger, Status
from .core import Edge, Epistemic, FirewallBreach, Health, Unknown, _StrEnum, as_date, require_past, stable_hash
from .knowledge_graph import EdgeProposal, GraphError, KnowledgeGraph


class Flag(_StrEnum):
    GROWING = "GROWING"                          # weight / evidence is rising: the disagreement is getting louder
    STALE_UNINVESTIGATED = "STALE_UNINVESTIGATED"
    REOPEN = "REOPEN"                            # materially newer evidence than the last investigation saw
    ESCALATED = "ESCALATED"
    STUCK = "STUCK"                              # investigated repeatedly, still unresolved


class Phase(_StrEnum):
    UNINVESTIGATED = "UNINVESTIGATED"
    UNRESOLVED = "UNRESOLVED"
    PARTLY_RESOLVED = "PARTLY_RESOLVED"
    NEEDS_DATA = "NEEDS_DATA"
    RESOLVED_BY_CONTEXT = "RESOLVED_BY_CONTEXT"
    CLOSED = "CLOSED"                            # investigation found no disagreement
    RETRACTED = "RETRACTED"                      # the edge was withdrawn (history kept)


_PHASE_OF = {Status.UNRESOLVED_KEEP_BOTH: Phase.UNRESOLVED, Status.PARTLY_RESOLVED: Phase.PARTLY_RESOLVED,
             Status.NEEDS_DATA: Phase.NEEDS_DATA, Status.RESOLVED: Phase.RESOLVED_BY_CONTEXT, Status.CLOSED: Phase.CLOSED,
             Status.OPEN: Phase.UNINVESTIGATED}
_OPEN_PHASES = frozenset({Phase.UNINVESTIGATED, Phase.UNRESOLVED, Phase.PARTLY_RESOLVED, Phase.NEEDS_DATA})


@dataclasses.dataclass(frozen=True)
class MonitorConfig:
    stale_days: int = 45                # uninvestigated this long -> STALE_UNINVESTIGATED
    escalate_days: int = 90             # uninvestigated this long, or stale AND growing -> ESCALATED
    growth_ratio: float = 1.5           # weight_now / weight_born at or above this is growth
    growth_min_gain: float = 0.15       # ... and the absolute gain must also reach this (tiny weights double by noise)
    evidence_gain: int = 2              # or this many extra evidence ids since birth
    reopen_days: int = 60               # newer evidence than the last investigation by this many days
    stuck_tries: int = 3                # investigations that never resolved
    max_questions: int = 20

    def validate(self) -> "MonitorConfig":
        errs = []
        if not (0 < self.stale_days <= self.escalate_days):
            errs.append("need 0 < stale_days <= escalate_days")
        if self.growth_ratio <= 1.0 or not (0.0 < self.growth_min_gain <= 1.0):
            errs.append("growth_ratio must exceed 1 and growth_min_gain lie in (0,1]")
        if self.evidence_gain < 1 or self.stuck_tries < 1 or self.reopen_days < 1 or self.max_questions < 1:
            errs.append("evidence_gain, stuck_tries, reopen_days, max_questions must be >= 1")
        if errs:
            raise ValueError("MonitorConfig: " + "; ".join(errs))
        return self


@dataclasses.dataclass(frozen=True)
class TrackedContradiction:
    pair: tuple[str, str]
    born: str
    age_days: int
    phase: Phase
    weight_born: float
    weight_now: float
    weight_peak: float
    evidence_born: int
    evidence_now: int
    evidence_through: str                        # date of the newest edge version seen
    last_investigated: str
    days_since_investigation: int | None
    tries: int
    resolved_dimension: str
    flags: tuple[Flag, ...]
    priority: float

    @property
    def is_open(self) -> bool:
        return self.phase in _OPEN_PHASES

    def health(self) -> Health:
        if self.phase in (Phase.RESOLVED_BY_CONTEXT, Phase.CLOSED):
            return Health.HEALTHY
        if self.phase == Phase.RETRACTED:
            return Health.UNKNOWN
        return Health.CONTRADICTED

    def why(self) -> str:
        head = {Phase.UNINVESTIGATED: f"contradiction born {self.born} never investigated (age {self.age_days}d)",
                Phase.UNRESOLVED: f"investigated {self.tries}x, still unresolved (age {self.age_days}d)",
                Phase.PARTLY_RESOLVED: "context explains part of it; residual disagreement remains",
                Phase.NEEDS_DATA: "investigation lacked independent dates; needs more data",
                Phase.RESOLVED_BY_CONTEXT: f"resolved by context {self.resolved_dimension or '?'}",
                Phase.CLOSED: "investigation found no real disagreement",
                Phase.RETRACTED: "edge retracted"}[self.phase]
        return head + (" [" + ", ".join(f.value for f in self.flags) + "]" if self.flags else "")


@dataclasses.dataclass(frozen=True)
class MonitorReport:
    now: str
    items: tuple[TrackedContradiction, ...]
    hidden_future: int                           # edge versions at/after `now` that were ignored, never read
    graph_head: str

    def open_items(self) -> list[TrackedContradiction]:
        return [t for t in self.items if t.is_open]

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(t.phase.value for t in self.items).items()))

    def flagged(self, flag: Flag) -> list[TrackedContradiction]:
        return [t for t in self.items if flag in t.flags]

    def to_record(self) -> dict:
        rec = {"section": 46, "job": "J09", "now": self.now, "graph_head": self.graph_head, "counts": self.counts(),
               "open": len(self.open_items()), "hidden_future": self.hidden_future,
               "items": [{**dataclasses.asdict(t), "phase": t.phase.value, "flags": [f.value for f in t.flags]} for t in self.items]}
        rec["report_id"] = stable_hash(rec)
        return rec

    def markdown(self) -> str:
        lines = [f"# Contradiction monitor @ {self.now}", "",
                 f"{len(self.items)} tracked, {len(self.open_items())} open; " + (", ".join(f"{k}={v}" for k, v in self.counts().items()) or "none"),
                 "", "| pair | phase | age d | weight | flags | priority |", "|---|---|---:|---|---|---:|"]
        for t in sorted(self.items, key=lambda x: (-x.priority, x.pair)):
            lines.append(f"| {t.pair[0]} vs {t.pair[1]} | {t.phase.value} | {t.age_days} | {t.weight_born:.2f}->{t.weight_now:.2f} | "
                         f"{', '.join(f.value for f in t.flags) or '-'} | {t.priority:.2f} |")
        return "\n".join(lines)


def _ordered(pair: Sequence[str]) -> tuple[str, str]:
    a, b = sorted(pair)
    return a, b


class ContradictionMonitor:
    """Read-only over the graph and the ledger, except `ingest`, which is the one guarded way to add a contradiction."""

    def __init__(self, graph: KnowledgeGraph, ledger: ContradictionLedger | None = None, cfg: MonitorConfig | None = None):
        self.graph = graph
        self.ledger = ledger if ledger is not None else ContradictionLedger()
        self.cfg = (cfg or MonitorConfig()).validate()
        self._periods: list[dict] = []           # one compact record per run_period, in run order

    # ------------------------------------------------------------------ intake
    def ingest(self, proposals: Iterable[EdgeProposal], known_at, now) -> list:
        """Record CONTRADICTS proposals. A proposal known at/after `now` is refused (FirewallBreach); so is any other relation."""
        props = list(proposals)
        for p in props:
            if str(getattr(p.rel, "value", p.rel)) != Edge.CONTRADICTS.value:
                raise GraphError(f"monitor ingests CONTRADICTS only, got {p.rel!r}")
        require_past(known_at, now, "contradiction edge")
        return self.graph.apply(props, known_at)

    # ------------------------------------------------------------------ per-pair history
    def _versions(self, pair: tuple[str, str], now) -> tuple[list, int]:
        """Edge versions of the pair known strictly before `now`, and how many were hidden for being at/after it."""
        hist = self.graph.edge_history(pair[0], pair[1], Edge.CONTRADICTS)
        n = as_date(now)
        seen = [v for v in hist if as_date(v.known_at) < n]
        return seen, len(hist) - len(seen)

    def _all_pairs(self) -> list[tuple[str, str]]:
        return sorted({(k[0], k[1]) for k in self.graph._edges if k[2] == Edge.CONTRADICTS.value})

    def track(self, pair: tuple[str, str], now) -> tuple[TrackedContradiction | None, int]:
        pair = _ordered(pair)
        seen, hidden = self._versions(pair, now)
        if not seen:
            return None, hidden
        for nid in pair:                         # a node not yet known at `now` makes the edge unreadable
            if self.graph.node_at(nid, now) is None:
                return None, hidden + len(seen)
        cfg, n = self.cfg, as_date(now)
        live = [v for v in seen if not v.retracted]
        born = as_date(seen[0].known_at)
        retracted = seen[-1].retracted
        weights = [v.weight for v in live] or [0.0]
        ev_now = len(seen[-1].evidence) if not retracted else len(live[-1].evidence) if live else 0
        entries = self.ledger.entries(pair, now)
        status = self.ledger.status(pair, now)
        phase = Phase.RETRACTED if retracted else _PHASE_OF[status]
        last_inv = entries[-1].at if entries else ""
        since = (n - as_date(last_inv)).days if last_inv else None
        age = (n - born).days
        through = seen[-1].known_at
        gain = weights[-1] - weights[0]
        growing = (not retracted and weights[0] > 0 and weights[-1] / weights[0] >= cfg.growth_ratio and gain >= cfg.growth_min_gain) \
            or (not retracted and ev_now - len(live[0].evidence) >= cfg.evidence_gain)
        flags: list[Flag] = []
        if growing:
            flags.append(Flag.GROWING)
        if phase == Phase.UNINVESTIGATED and age >= cfg.stale_days:
            flags.append(Flag.STALE_UNINVESTIGATED)
        if phase in _OPEN_PHASES and entries and self.ledger.needs_reinvestigation(pair, now, through, cfg.reopen_days):
            flags.append(Flag.REOPEN)
        if phase in (Phase.UNRESOLVED, Phase.PARTLY_RESOLVED) and self.ledger.tries(pair, now) >= cfg.stuck_tries:
            flags.append(Flag.STUCK)
        if phase == Phase.UNINVESTIGATED and (age >= cfg.escalate_days or (Flag.STALE_UNINVESTIGATED in flags and growing)) \
                or (Flag.STUCK in flags and growing):
            flags.append(Flag.ESCALATED)
        dossier = self.ledger.dossier(pair, now) if entries else None
        dim = "x".join(dossier.dimension) if dossier is not None and phase == Phase.RESOLVED_BY_CONTEXT else ""
        t = TrackedContradiction(pair, born.isoformat(), age, phase, weights[0], weights[-1], max(weights), len(live[0].evidence) if live else 0,
                                 ev_now, through, last_inv, since, self.ledger.tries(pair, now), dim, tuple(flags), 0.0)
        return dataclasses.replace(t, priority=self._priority(t)), hidden

    def _priority(self, t: TrackedContradiction) -> float:
        if not t.is_open:
            return 0.0
        cfg = self.cfg
        p = 0.35 * t.weight_now + 0.25 * min(1.0, t.age_days / max(1, cfg.escalate_days))
        weight = {Flag.GROWING: 0.15, Flag.STALE_UNINVESTIGATED: 0.1, Flag.REOPEN: 0.1, Flag.STUCK: 0.05, Flag.ESCALATED: 0.2}
        p += sum(weight[f] for f in t.flags)
        return round(min(1.0, p), 6)

    # ------------------------------------------------------------------ the period scan
    def scan(self, now, strict: bool = False) -> MonitorReport:
        """Track every contradiction visible before `now`. strict=True raises if the graph holds anything at/after `now`
        (the mode for a live period run); otherwise such versions are ignored and counted in `hidden_future`."""
        items, hidden = [], 0
        for pair in self._all_pairs():
            t, h = self.track(pair, now)
            hidden += h
            if t is not None:
                items.append(t)
        if strict and hidden:
            raise FirewallBreach(f"{hidden} contradiction edge version(s) dated at/after now={as_date(now)}")
        items.sort(key=lambda t: (-t.priority, t.pair))
        return MonitorReport(as_date(now).isoformat(), tuple(items), hidden, self.graph.head)

    def run_period(self, now) -> MonitorReport:
        """One scheduled run: scan strictly, remember the compact result, and refuse to run out of order."""
        if self._periods and as_date(now) <= as_date(self._periods[-1]["now"]):
            raise FirewallBreach(f"period {as_date(now)} is not after the previous run {self._periods[-1]['now']}")
        rep = self.scan(now, strict=True)
        self._periods.append({"now": rep.now, "open": len(rep.open_items()), "counts": rep.counts(),
                              "escalated": [t.pair for t in rep.flagged(Flag.ESCALATED)]})
        return rep

    def open_series(self, dates: Sequence) -> list[dict]:
        """Open / born / resolved counts at each date, recomputed from the graph (not from remembered runs)."""
        out: list[dict] = []
        prev: set[tuple[str, str]] = set()
        for d in sorted(dates, key=as_date):
            rep = self.scan(d)
            now_open = {t.pair for t in rep.open_items()}
            out.append({"date": rep.now, "open": len(now_open), "newly_open": len(now_open - prev), "newly_closed": len(prev - now_open),
                        "oldest_open_days": max((t.age_days for t in rep.open_items()), default=0)})
            prev = now_open
        return out

    def timeline(self, pair: Sequence[str], now) -> list[dict]:
        """Dated events for one pair: birth, weight changes, investigations, retraction."""
        seen, _ = self._versions(_ordered(pair), now)
        ev = []
        for i, v in enumerate(seen):
            kind = "retracted" if v.retracted else "born" if i == 0 else "reweighted"
            ev.append({"date": v.known_at, "event": kind, "weight": v.weight, "evidence": len(v.evidence)})
        ev += [{"date": e.at, "event": "investigated:" + e.verdict, "weight": None, "evidence": None} for e in self.ledger.entries(pair, now)]
        return sorted(ev, key=lambda e: (e["date"], e["event"]))

    # ------------------------------------------------------------------ never average
    def estimate(self, a: str, b: str, context: Mapping[str, Any], now) -> dict:
        """The only sanctioned combined read of two contradicting items: what each says INSIDE `context`, else UNKNOWN."""
        pair = _ordered((a, b))
        t, _ = self.track(pair, now)
        if t is None:
            return {"state": Unknown.UNKNOWN.value, "reason": "no contradiction between these items is known yet"}
        dossier = self.ledger.dossier(pair, now)
        if dossier is None:
            return {"state": Unknown.CONFLICTED.value, "reason": "contradiction not investigated; refusing to average"}
        return dossier.estimate_in_context(context)

    def pooled(self, a: str, b: str, now) -> float:
        """Deliberately unavailable while the pair is contradicted (mirrors Investigation.pooled)."""
        t, _ = self.track((a, b), now)
        if t is not None and t.phase != Phase.CLOSED:
            raise ContradictionError(f"{a} and {b} contradict ({t.phase.value}): refusing to average")
        dossier = self.ledger.dossier(tuple(sorted((a, b))), now)
        if dossier is None:
            raise ContradictionError(f"{a} and {b}: nothing known to pool")
        return dossier.pooled()

    # ------------------------------------------------------------------ outputs
    def research_questions(self, report: MonitorReport) -> list[dict]:
        """research_priority.Signal-shaped records (kind CONTRADICTION) for the open contradictions, most urgent first."""
        out = []
        for t in report.open_items()[: self.cfg.max_questions]:
            when = as_date(t.evidence_through).isoformat()
            magnitude = round(min(1.0, t.weight_now + (0.25 if Flag.GROWING in t.flags else 0.0)), 6)
            action = ("investigate for the first time" if t.phase == Phase.UNINVESTIGATED else
                      "re-investigate with newer evidence" if Flag.REOPEN in t.flags else "collect stratified data for")
            out.append({"kind": "CONTRADICTION", "when": when, "subject": t.pair[0], "counterpart": t.pair[1], "magnitude": magnitude,
                        "stake": round(min(1.0, 0.3 + 0.7 * t.priority), 6), "n_obs": max(t.evidence_now, t.tries),
                        "evidence": (f"monitor:{report.now}",), "question": f"Why do {t.pair[0]} and {t.pair[1]} disagree? Next: {action} this pair.",
                        "escalated": Flag.ESCALATED in t.flags})
        return out

    def signals(self, report: MonitorReport) -> list:
        """The same questions as real research_priority.Signal objects (imported lazily to keep this module standalone)."""
        from .research_priority import make_signal
        sigs = []
        for q in self.research_questions(report):
            s = make_signal(q["kind"], q["when"], q["subject"], q["magnitude"], counterpart=q["counterpart"], stake=q["stake"],
                            n_obs=q["n_obs"], evidence=q["evidence"])
            errs = s.check()
            if errs:
                raise FirewallBreach("; ".join(errs))
            sigs.append(s)
        return sigs

    def dashboard_rows(self, report: MonitorReport) -> dict:
        """Rows for reports.health_dashboard: one per knowledge id touched by a contradiction (the worst pair wins)."""
        rows: dict[str, dict] = {}
        for t in report.items:
            for kid, other in ((t.pair[0], t.pair[1]), (t.pair[1], t.pair[0])):
                row = {"knowledge_id": kid, "health": t.health().value, "why": t.why(), "evidence": f"contradicts {other}",
                       "sample_size": t.evidence_now, "research": "monitor:" + t.phase.value.lower(),
                       "epistemic": (Epistemic.CONDITIONAL if t.phase == Phase.RESOLVED_BY_CONTEXT else
                                     Epistemic.CONTRADICTED if t.is_open else Epistemic.SUPPORTED).value,
                       "priority": t.priority}
                if kid not in rows or row["priority"] > rows[kid]["priority"]:
                    rows[kid] = row
        return {"section": 46, "job": "J09", "now": report.now, "counts": report.counts(), "escalated": len(report.flagged(Flag.ESCALATED)),
                "rows": [rows[k] for k in sorted(rows)]}

    def dumps(self, report: MonitorReport) -> str:
        return json.dumps(report.to_record(), sort_keys=True, indent=1)
