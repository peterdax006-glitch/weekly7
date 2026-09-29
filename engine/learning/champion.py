"""Champion / challenger / shadow / retired knowledge (contract C62 section 44; checklist J01-J03). IMPLEMENTED - NOT VALIDATED.

Production knowledge is never replaced in one step. Every knowledge version lives in exactly one role -
RESEARCH, SHADOW, CHALLENGER, CHAMPION or RETIRED (engine.learning.core.Promotion) - inside a decision SLOT (what decision it
changes, for which subsystem and scope). Only a CHAMPION influences a decision; shadows and challengers are observed
counterfactually and get weight zero. A challenger replaces the incumbent only when (1) the ten-gate promotion gate in
engine.learning.promotion passes (OOS, transfer, risk, anti-memorization, stability, reproducibility, ...) and (2) its paired
shadow record beats the incumbent's. The replaced champion is RETIRED, never deleted (section 13), and a post-promotion watch
rolls the promotion back if the new champion underperforms the retired one live.

The board is EVENT-SOURCED on engine.champion.Ledger (append-only, hash-chained): state is the fold of the ledger, so live
state and replayed state are the same code path, a tampered or reordered ledger is detected by `verify`, and a crash between
two events leaves a consistent board. Time only moves forward: an event dated before the previous one is a FirewallBreach."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engine.champion import Ledger

from .core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, KnowledgeLike, Promotion, Subsystem, as_date,
                   require_past, stable_hash)
from .promotion import PromotionDecision, PromotionEvidence, PromotionGate, write_rejection_report

P = Promotion
LEGAL: frozenset[tuple[Promotion, Promotion]] = frozenset({
    (P.RESEARCH, P.SHADOW), (P.SHADOW, P.CHALLENGER), (P.CHALLENGER, P.CHAMPION),
    (P.CHALLENGER, P.SHADOW),                                        # demotion: evidence weakened while waiting
    (P.RESEARCH, P.RETIRED), (P.SHADOW, P.RETIRED), (P.CHALLENGER, P.RETIRED), (P.CHAMPION, P.RETIRED),
    (P.RETIRED, P.SHADOW),                                           # recovery re-enters through shadow, never straight to production
})
ROLLBACK_EDGE = (P.RETIRED, P.CHAMPION)                              # legal ONLY through a rollback event


class BoardError(RuntimeError):
    """An illegal request (bad transition, unknown member, duplicate). Refused requests change nothing."""


# ------------------------------------------------------------------------------------------------ types
@dataclass(frozen=True)
class Slot:
    """A place in the decision pipeline that at most one champion may occupy."""
    effect: DecisionEffect
    subsystem: Subsystem | None = None
    scope: str = "global"

    @property
    def key(self) -> str:
        return f"{self.effect.value}|{self.subsystem.value if self.subsystem else '-'}|{self.scope}"

    @classmethod
    def from_key(cls, key: str) -> "Slot":
        e, s, sc = key.split("|", 2)
        return cls(DecisionEffect(e), None if s == "-" else Subsystem(s), sc)


@dataclass(frozen=True)
class BoardPolicy:
    min_shadow_sessions: int = 20
    max_challengers_per_slot: int = 3
    shadow_harm_t: float = -1.28                 # a shadow whose paired t is below this is refused as a challenger
    head_to_head_t: float = 1.64                 # vs the incumbent, paired, one-sided
    head_to_head_min_gain: float = 0.0
    watch_sessions: int = 10
    watch_t_stop: float = -1.28
    watch_min_gap: float = 0.0005

    def validate(self) -> list[str]:
        errs = []
        if self.min_shadow_sessions < 5:
            errs.append("min_shadow_sessions below 5 cannot separate a challenger from noise")
        if self.max_challengers_per_slot < 1:
            errs.append("max_challengers_per_slot must be >= 1")
        if self.watch_sessions < 3:
            errs.append("watch_sessions must be >= 3")
        if self.watch_t_stop >= 0 or self.shadow_harm_t >= 0:
            errs.append("stop thresholds must be negative t values")
        return errs


@dataclass
class Member:
    """Mutable ONLY through Board._apply (which replays the ledger); callers get frozen `MemberView`s."""
    mid: str
    knowledge_id: str
    version: int
    slot: str
    effects: tuple[str, ...]
    role: Promotion
    since: str
    reason: str = ""
    cause: str = ""
    path: list[tuple[str, str]] = field(default_factory=list)       # (date, role) history, append-only
    shadow: list[tuple[str, float, float]] = field(default_factory=list)   # (date, candidate value, champion/baseline value)
    watch: dict | None = None


@dataclass(frozen=True)
class MemberView:
    mid: str
    knowledge_id: str
    version: int
    slot: str
    role: Promotion
    since: str
    reason: str
    cause: str
    n_shadow: int
    path: tuple[tuple[str, str], ...]


def member_id(k: Any) -> str:
    return f"{k.knowledge_id}@v{int(k.version)}"


class _RoleView:
    """Read-only proxy of a knowledge object whose `promotion` is the BOARD's role, not whatever the (possibly stale) object
    claims: the board is the single source of truth for who is champion."""

    def __init__(self, k: Any, role: Promotion):
        self._k, self.promotion = k, role

    def __getattr__(self, name):
        return getattr(self._k, name)


def paired_stats(diffs: Sequence[float]) -> dict:
    """Mean, t and n of paired differences (candidate minus incumbent). n < 2 gives t=0 (no evidence either way)."""
    d = np.asarray(list(diffs), float)
    if d.size and not np.isfinite(d).all():
        raise BoardError("paired differences contain non-finite values")
    if d.size < 2:
        return {"n": int(d.size), "mean": float(d.mean()) if d.size else 0.0, "t": 0.0}
    sd = float(d.std(ddof=1))
    m = float(d.mean())
    t = 0.0 if sd == 0 and m == 0 else (math.copysign(math.inf, m) if sd == 0 else m / (sd / math.sqrt(d.size)))
    return {"n": int(d.size), "mean": m, "t": float(t)}


# ------------------------------------------------------------------------------------------------ the board
class KnowledgeBoard:
    def __init__(self, ledger_path: str | Path, gate: PromotionGate | None = None, policy: BoardPolicy | None = None,
                 report_dir: str | Path | None = None):
        self.policy = policy or BoardPolicy()
        errs = self.policy.validate()
        if errs:
            raise BoardError(f"invalid board policy: {errs}")
        self.ledger = Ledger(ledger_path)
        self.gate = gate or PromotionGate()
        self.report_dir = Path(report_dir) if report_dir else None
        self.members: dict[str, Member] = {}
        self.champions: dict[str, str] = {}                       # slot key -> member id
        self.decisions: list[dict] = []                           # promotion decisions in order (from the ledger)
        self.last_date: str | None = None
        for row in self.ledger.rows():                            # crash recovery = replay
            self._apply(row["event"], row["id"], row["t"], row["detail"])

    # ---- event application: the ONLY place state changes ----
    def _apply(self, event: str, mid: str, t: str, d: Mapping[str, Any]) -> None:
        self.last_date = t
        m = self.members.get(mid)
        if event == "registered":
            self.members[mid] = Member(mid, d["knowledge_id"], int(d["version"]), d["slot"], tuple(d["effects"]),
                                       P.RESEARCH, t, path=[(t, P.RESEARCH.value)])
            return
        if m is None:
            raise BoardError(f"ledger event {event} names unknown member {mid}")
        if event in ("to_shadow", "reinstated", "demoted"):
            self._move(m, P.SHADOW, t, d.get("reason", ""))
        elif event == "shadow_obs":
            m.shadow.append((t, float(d["candidate"]), float(d["baseline"])))
        elif event == "to_challenger":
            self._move(m, P.CHALLENGER, t, d.get("reason", ""))
        elif event == "promoted":
            old = self.champions.get(m.slot)
            if old:
                self._move(self.members[old], P.RETIRED, t, f"replaced by {mid}", cause="REPLACED")
            self._move(m, P.CHAMPION, t, d.get("reason", "passed all gates"))
            self.champions[m.slot] = mid
            m.watch = {"predecessor": old, "status": "pending" if old else "no_predecessor", "obs": []}
            self.decisions.append({"t": t, "id": mid, **{k: d[k] for k in ("decision_id", "verdict") if k in d}})
        elif event == "promotion_refused":
            self.decisions.append({"t": t, "id": mid, **{k: d[k] for k in ("decision_id", "verdict") if k in d}})
        elif event == "retired":
            if self.champions.get(m.slot) == mid:
                del self.champions[m.slot]
            self._move(m, P.RETIRED, t, d.get("reason", ""), cause=d.get("cause", ""))
        elif event == "watch_obs":
            m.watch["obs"].append((t, float(d["new"]), float(d["old"])))
        elif event == "watch_confirmed":
            m.watch["status"] = "confirmed"
        elif event == "rolled_back":
            pred = self.members[d["restored"]]
            self._move(m, P.RETIRED, t, f"rolled back: {d['reason']}", cause="ROLLED_BACK")
            self._move(pred, P.CHAMPION, t, f"restored after rollback of {mid}", rollback=True)
            self.champions[m.slot] = pred.mid
            m.watch["status"] = "rolled_back"
        else:
            raise BoardError(f"unknown ledger event {event!r}")

    def _move(self, m: Member, to: Promotion, t: str, reason: str, cause: str = "", rollback: bool = False) -> None:
        edge = (m.role, to)
        if edge not in LEGAL and not (rollback and edge == ROLLBACK_EDGE):
            raise BoardError(f"{m.mid}: illegal transition {m.role} -> {to}")
        m.role, m.since, m.reason, m.cause = to, t, reason, cause
        m.path.append((t, to.value))

    def _emit(self, event: str, mid: str, now, **detail) -> None:
        t = str(as_date(now))
        if self.last_date is not None and as_date(t) < as_date(self.last_date):
            raise FirewallBreach(f"board event dated {t} precedes the last recorded event {self.last_date}")
        row = self.ledger.append(event, mid, t, **detail)
        self._apply(row["event"], row["id"], row["t"], row["detail"])

    def _get(self, mid: str, *roles: Promotion) -> Member:
        m = self.members.get(mid)
        if m is None:
            raise BoardError(f"unknown member {mid}")
        if roles and m.role not in roles:
            raise BoardError(f"{mid} is {m.role}; this action needs {[str(r) for r in roles]}")
        return m

    # ---- registration and shadow ----
    def register(self, k: Any, slot: Slot, now) -> str:
        bad = KnowledgeLike.conforms(k)
        if bad:
            raise BoardError(f"not a knowledge object: {bad}")
        mid = member_id(k)
        if mid in self.members:
            raise BoardError(f"{mid} already registered; a changed item must be a new version")
        effects = tuple(DecisionEffect.parse(e) for e in k.decision_effect)
        if slot.effect not in effects:
            raise BoardError(f"slot effect {slot.effect} is not among the knowledge's decision effects {[str(e) for e in effects]}")
        if slot.effect == DecisionEffect.NONE:
            raise BoardError("research-only knowledge (DecisionEffect.NONE) cannot occupy a decision slot")
        if Promotion.parse(k.promotion) not in (P.RESEARCH, P.SHADOW):
            raise BoardError(f"new knowledge enters as RESEARCH or SHADOW, not {k.promotion}")
        self._emit("registered", mid, now, knowledge_id=k.knowledge_id, version=int(k.version), slot=slot.key,
                   effects=[e.value for e in effects])
        return mid

    def to_shadow(self, mid: str, now, reason: str = "") -> None:
        m = self._get(mid, P.RESEARCH)
        self._emit("to_shadow", mid, now, reason=reason or "begin counterfactual observation")

    def reinstate(self, mid: str, now, reason: str) -> None:
        """Recovery of retired knowledge (section 13): back into SHADOW only, with a stated reason; it must re-earn production."""
        if not reason:
            raise BoardError("reinstating retired knowledge needs a reason")
        self._get(mid, P.RETIRED)
        self._emit("reinstated", mid, now, reason=reason)

    def record_shadow(self, mid: str, candidate_value: float, baseline_value: float, now) -> None:
        """One session's counterfactual: what the shadow WOULD have delivered vs what the incumbent (or general rule) did.
        Stored, never acted on. Non-finite values are refused so a missing session is not counted as zero."""
        self._get(mid, P.SHADOW, P.CHALLENGER)
        if not (math.isfinite(candidate_value) and math.isfinite(baseline_value)):
            raise BoardError("shadow observation must be finite")
        m = self.members[mid]
        if m.shadow and as_date(m.shadow[-1][0]) == as_date(now):
            raise BoardError(f"{mid}: a shadow observation for {as_date(now)} already exists")
        self._emit("shadow_obs", mid, now, candidate=float(candidate_value), baseline=float(baseline_value))

    def shadow_record(self, mid: str, now=None) -> dict:
        m = self._get(mid)
        obs = [o for o in m.shadow if now is None or as_date(o[0]) < as_date(now)]
        return paired_stats([c - b for _, c, b in obs])

    # ---- challenger ----
    def open_challenge(self, mid: str, now) -> dict:
        m = self._get(mid, P.SHADOW)
        st = self.shadow_record(mid, now)
        pol = self.policy
        if st["n"] < pol.min_shadow_sessions:
            raise BoardError(f"{mid}: {st['n']} shadow sessions < {pol.min_shadow_sessions}")
        if st["t"] < pol.shadow_harm_t:
            raise BoardError(f"{mid}: shadow record shows harm (paired t={st['t']:.2f})")
        open_here = [x for x in self.members.values() if x.slot == m.slot and x.role == P.CHALLENGER]
        if len(open_here) >= pol.max_challengers_per_slot:
            raise BoardError(f"slot {m.slot} already has {len(open_here)} challengers")
        self._emit("to_challenger", mid, now, reason=f"{st['n']} shadow sessions, paired mean {st['mean']:+.5f}")
        return st

    def demote(self, mid: str, now, reason: str) -> None:
        if not reason:
            raise BoardError("a demotion needs a reason")
        self._get(mid, P.CHALLENGER)
        self._emit("demoted", mid, now, reason=reason)

    # ---- promotion ----
    def head_to_head(self, mid: str, now) -> dict:
        """Paired shadow record of the challenger against the incumbent. With no incumbent the general rule was the baseline
        and beating it is not required beyond the gate (there is nothing to displace)."""
        m = self._get(mid)
        st = self.shadow_record(mid, now)
        has_inc = m.slot in self.champions
        ok = (not has_inc) or (st["n"] >= self.policy.min_shadow_sessions and st["t"] >= self.policy.head_to_head_t
                               and st["mean"] > self.policy.head_to_head_min_gain)
        return {**st, "incumbent": self.champions.get(m.slot), "ok": bool(ok)}

    def attempt_promotion(self, k: Any, evidence: PromotionEvidence, now) -> dict:
        """Run the gate and the head-to-head; promote only if both pass. Always returns the decision; a refusal changes
        nothing but the ledger (a written rejection report is saved when a report directory is configured)."""
        mid = member_id(k)
        m = self._get(mid, P.CHALLENGER)
        if int(k.version) != m.version:
            raise BoardError(f"{mid}: version mismatch between object and board")
        dec = self.gate.evaluate(_RoleView(k, m.role), evidence, now)
        h2h = self.head_to_head(mid, now)
        promote = dec.promote and h2h["ok"]
        out = {"promoted": promote, "decision": dec, "head_to_head": h2h, "report": None}
        if self.report_dir is not None and not promote:
            out["report"] = write_rejection_report(self.report_dir, dec)
        if not promote:
            why = list(dec.critical_failures) + [f.gate for f in dec.failed if not f.critical]
            if not h2h["ok"]:
                why.append("head_to_head")
            self._emit("promotion_refused", mid, now, decision_id=dec.decision_id, verdict="BLOCK", failed=why,
                       preconditions=list(dec.preconditions))
            return out
        self._emit("promoted", mid, now, decision_id=dec.decision_id, verdict="PROMOTE", replaced=self.champions.get(m.slot),
                   reason="passed all applicable gates and beat the incumbent head-to-head")
        return out

    # ---- retirement and post-promotion watch ----
    def retire(self, mid: str, cause: FailureCause, reason: str, now) -> None:
        """Retire, do not delete. A retired champion leaves its slot EMPTY: decisions fall back to the general rule."""
        if not reason:
            raise BoardError("retirement needs a reason (section 13)")
        m = self._get(mid)
        if m.role == P.RETIRED:
            raise BoardError(f"{mid} is already retired")
        self._emit("retired", mid, now, cause=FailureCause.parse(cause).value, reason=reason)

    def record_watch(self, slot: Slot, new_value: float, old_value: float, now) -> None:
        mid = self.champions.get(slot.key)
        if mid is None:
            raise BoardError(f"slot {slot.key} has no champion")
        w = self.members[mid].watch
        if not w or w["status"] != "pending":
            raise BoardError("no pending post-promotion watch")
        if not (math.isfinite(new_value) and math.isfinite(old_value)):
            raise BoardError("watch observation must be finite")
        self._emit("watch_obs", mid, now, new=float(new_value), old=float(old_value))

    def review_watch(self, slot: Slot, now) -> dict:
        """10-session tripwire: a promoted champion that underperforms the retired predecessor's shadow is rolled back.
        Live P&L can only trigger a ROLLBACK; it can never promote."""
        mid = self.champions.get(slot.key)
        w = self.members[mid].watch if mid else None
        if not w or w["status"] != "pending":
            return {"status": (w or {}).get("status", "none")}
        pol = self.policy
        st = paired_stats([n - o for _, n, o in w["obs"]])
        if st["n"] < pol.watch_sessions:
            return {"status": "pending", **st, "needed": pol.watch_sessions}
        if st["mean"] < -pol.watch_min_gap and st["t"] < pol.watch_t_stop:
            reason = f"underperformed predecessor: mean {st['mean']:+.5f}, t {st['t']:.2f}"
            self._emit("rolled_back", mid, now, restored=w["predecessor"], reason=reason)
            return {"status": "rolled_back", "restored": w["predecessor"], **st}
        self._emit("watch_confirmed", mid, now, **{k: v for k, v in st.items() if math.isfinite(v)})
        return {"status": "confirmed", **st}

    # ---- queries ----
    def weight(self, mid: str) -> float:
        """Decision weight: 1 for a champion, exactly 0 for everything else. Shadows and challengers never affect decisions."""
        m = self.members.get(mid)
        return 1.0 if m is not None and m.role == P.CHAMPION else 0.0

    def champion(self, slot: Slot) -> str | None:
        return self.champions.get(slot.key)

    def production_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.champions.values()))

    def view(self, mid: str) -> MemberView:
        m = self._get(mid)
        return MemberView(m.mid, m.knowledge_id, m.version, m.slot, m.role, m.since, m.reason, m.cause, len(m.shadow), tuple(m.path))

    def roles(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {str(r): [] for r in Promotion}
        for mid, m in sorted(self.members.items()):
            out[str(m.role)].append(mid)
        return out

    def history(self, knowledge_id: str) -> list[MemberView]:
        return [self.view(mid) for mid, m in sorted(self.members.items()) if m.knowledge_id == knowledge_id]

    def digest(self) -> str:
        """State fingerprint: identical ledgers give identical digests on any machine."""
        return stable_hash({mid: [m.slot, m.role.value, m.path] for mid, m in sorted(self.members.items())})

    def invariants(self) -> list[str]:
        """Structural problems; an empty list means the board is consistent. Run after every replay and in tests."""
        probs = []
        v = self.ledger.verify()
        if not v["ok"]:
            probs += [f"ledger: {p}" for p in v["problems"]]
        seen: dict[str, str] = {}
        for mid, m in self.members.items():
            if m.role == P.CHAMPION:
                if m.slot in seen:
                    probs.append(f"slot {m.slot} has two champions: {seen[m.slot]} and {mid}")
                seen[m.slot] = mid
                if self.champions.get(m.slot) != mid:
                    probs.append(f"{mid} is CHAMPION but not indexed")
                if DecisionEffect.NONE.value in m.effects and len(m.effects) == 1:
                    probs.append(f"{mid} has no decision effect but is in production")
            if m.role == P.RETIRED and not m.reason:
                probs.append(f"{mid} retired without a reason")
            for (_, a), (_, b) in zip(m.path, m.path[1:]):
                if (Promotion(a), Promotion(b)) not in LEGAL and (Promotion(a), Promotion(b)) != ROLLBACK_EDGE:
                    probs.append(f"{mid}: illegal path step {a}->{b}")
        for slot, mid in self.champions.items():
            if self.members[mid].role != P.CHAMPION:
                probs.append(f"index names {mid} champion of {slot} but it is {self.members[mid].role}")
        return probs

    def report(self) -> str:
        lines = [f"# Knowledge board ({len(self.members)} versions, digest {self.digest()})",
                 "Status: IMPLEMENTED - NOT VALIDATED", ""]
        for r, mids in self.roles().items():
            lines.append(f"## {r} ({len(mids)})")
            for mid in mids:
                m = self.members[mid]
                extra = f", {len(m.shadow)} shadow sessions" if m.shadow else ""
                lines.append(f"- {mid} slot {m.slot} since {m.since}{extra}" + (f" - {m.reason}" if m.reason else ""))
        prob = self.invariants()
        lines += ["", "## Invariants", "clean" if not prob else "\n".join(f"- {p}" for p in prob)]
        return "\n".join(lines) + "\n"
