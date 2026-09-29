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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from engine import pattern_stats as PS
from engine.champion import ChallengerQueue, Ledger

from .core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, Health, KnowledgeLike, Promotion, Subsystem, as_date,
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
                 report_dir: str | Path | None = None, tried=None, as_of=None):
        self.policy = policy or BoardPolicy()
        errs = self.policy.validate()
        if errs:
            raise BoardError(f"invalid board policy: {errs}")
        if as_of is None:
            Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(ledger_path)
        self.gate = gate or PromotionGate()
        self.report_dir = Path(report_dir) if report_dir else None
        # waiting list for the challenger slot: engine.champion.ChallengerQueue (priority = PRE-REGISTERED expected gain, never
        # realised P&L; `tried` is an optional engine.experiment_memory.TriedIndex that refuses repeats and known-bad ideas)
        self.queue = ChallengerQueue(Path(ledger_path).with_suffix(".queue.json"), tried)
        self.members: dict[str, Member] = {}
        self.champions: dict[str, str] = {}                       # slot key -> member id
        self.decisions: list[dict] = []                           # promotion decisions in order (from the ledger)
        self.last_date: str | None = None
        self.frozen = as_of is not None                           # a historical view can be read but never written
        for row in self.ledger.rows():                            # crash recovery = replay
            if as_of is not None and as_date(row["t"]) >= as_date(as_of):
                break                                             # events are date-ordered, so nothing later can precede
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
        if self.frozen:
            raise BoardError("this board is a read-only historical view (board_as_of); it cannot record events")
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

    # ---- waiting list (reuses engine.champion.ChallengerQueue) ----
    def enqueue(self, mid: str, cfg: Mapping[str, Any], expected_gain: float, now) -> None:
        """Queue a shadow for the (limited) challenger slots. The expected gain must be stated up front."""
        self._get(mid, P.SHADOW)
        self.queue.submit(mid, dict(cfg), expected_gain, str(as_date(now)))

    def next_queued(self, slot: "Slot", now) -> str | None:
        """Open a challenge for the highest-expected-gain queued shadow in `slot` that qualifies; skip (and keep) those that
        do not yet. Returns the member id moved to CHALLENGER, or None."""
        for item in self.queue.order():
            m = self.members.get(item["id"])
            if m is None or m.slot != slot.key or m.role != P.SHADOW:
                continue
            try:
                self.open_challenge(m.mid, now)
            except BoardError:
                continue
            self.queue.items = [i for i in self.queue.items if i["id"] != m.mid]
            self.queue._save()
            return m.mid
        return None

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


# ------------------------------------------------------------------------------------------------ analytics over the board
@dataclass(frozen=True)
class ShadowSummary:
    mid: str
    n: int
    mean: float
    sd: float
    t: float
    p_one_sided: float
    hit_rate: float
    cumulative: float
    max_relative_drawdown: float       # worst peak-to-trough of the cumulative (candidate - baseline) series
    recent_mean: float                 # last 10 sessions: is the edge fading?
    first: str | None
    last: str | None


def _p_one_sided(t: float) -> float:
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    p2 = float(PS.t_to_p(t))
    return p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0


def shadow_summary(board: KnowledgeBoard, mid: str, now=None) -> ShadowSummary:
    """Everything the shadow record says, using only sessions strictly before `now`."""
    m = board._get(mid)
    obs = [o for o in m.shadow if now is None or as_date(o[0]) < as_date(now)]
    d = np.array([c - b for _, c, b in obs], float)
    if d.size == 0:
        return ShadowSummary(mid, 0, 0.0, 0.0, 0.0, 1.0, float("nan"), 0.0, 0.0, 0.0, None, None)
    st = paired_stats(d)
    cum = np.cumsum(d)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return ShadowSummary(mid, st["n"], st["mean"], float(d.std(ddof=1)) if d.size > 1 else 0.0, st["t"], _p_one_sided(st["t"]),
                         float((d > 0).mean()), float(cum[-1]), float((cum - peak).min()), float(d[-10:].mean()),
                         obs[0][0], obs[-1][0])


def sessions_needed(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Sessions of paired shadow data needed to detect a mean paired difference `effect` at one-sided `alpha` with `power`:
    n = ((z_alpha + z_power) * sd / effect)^2. Infinite when the effect is not positive - no amount of data confirms it."""
    from statistics import NormalDist
    if not (effect > 0 and sd > 0 and math.isfinite(effect) and math.isfinite(sd)):
        return math.inf
    nd = NormalDist()
    return float(((nd.inv_cdf(1 - alpha) + nd.inv_cdf(power)) * sd / effect) ** 2)


def rank_challengers(board: KnowledgeBoard, slot: Slot, now, alpha: float = 0.05) -> list[dict]:
    """Challengers of one slot ranked by their shadow record against the incumbent. With k challengers the best of k will look
    good by luck, so p-values are Bonferroni-adjusted across the k (engine.pattern_stats.bonferroni) before any is 'eligible'."""
    ch = sorted(mid for mid, m in board.members.items() if m.slot == slot.key and m.role == P.CHALLENGER)
    sums = [shadow_summary(board, mid, now) for mid in ch]
    if not sums:
        return []
    adj = PS.bonferroni([s.p_one_sided for s in sums], len(sums))
    rows = [{"mid": s.mid, "n": s.n, "mean": s.mean, "t": s.t, "p": s.p_one_sided, "p_adj": float(a),
             "eligible": bool(a <= alpha and s.mean > 0 and s.n >= board.policy.min_shadow_sessions),
             "sessions_needed": sessions_needed(s.mean, s.sd, alpha)} for s, a in zip(sums, adj)]
    return sorted(rows, key=lambda r: (r["p_adj"], -r["mean"], r["mid"]))


def effective_champion(board: KnowledgeBoard, effect: DecisionEffect, subsystem: Subsystem | None,
                       scope_chain: Sequence[str]) -> str | None:
    """The knowledge that decides for a request: the champion of the MOST SPECIFIC scope in `scope_chain` (ordered specific
    to general, e.g. ['small_cap', 'equities', 'global']). No champion anywhere -> None, and the caller uses its general rule."""
    for sc in scope_chain:
        mid = board.champions.get(Slot(effect, subsystem, sc).key)
        if mid is not None:
            return mid
    return None


def scope_conflicts(board: KnowledgeBoard) -> list[tuple[str, str]]:
    """Pairs (specific champion, general champion) sharing effect and subsystem across different scopes. Legal - the specific
    one wins in `effective_champion` - but worth a look when the specific one is newer than the general one it overrides."""
    by: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, mid in board.champions.items():
        sl = Slot.from_key(key)
        by.setdefault((sl.effect.value, sl.subsystem.value if sl.subsystem else "-"), []).append((sl.scope, mid))
    out = []
    for grp in by.values():
        general = [mid for sc, mid in grp if sc == "global"]
        out += [(mid, g) for sc, mid in grp if sc != "global" for g in general]
    return sorted(out)


def review_due(board: KnowledgeBoard, now, max_age_days: int = 90) -> list[dict]:
    """Champions that have held their slot for more than `max_age_days` since they last (re)entered the role: production
    knowledge is re-examined on a schedule, not only when it visibly breaks."""
    out = []
    for mid in board.production_ids():
        m = board.members[mid]
        age = (as_date(now) - as_date(m.since)).days
        if age > max_age_days:
            out.append({"mid": mid, "slot": m.slot, "age_days": age, "watch": (m.watch or {}).get("status")})
    return sorted(out, key=lambda r: -r["age_days"])


HEALTH_ACTIONS = {
    Health.HEALTHY: "none", Health.RECOVERING: "none", Health.DORMANT: "hold",
    Health.DEGRADING: "review", Health.UNSTABLE: "review",
    Health.INSUFFICIENT_EVIDENCE: "abstain", Health.UNKNOWN: "abstain",
    Health.BROKEN: "retire", Health.CONTRADICTED: "retire",
}
HEALTH_CAUSE = {Health.BROKEN: FailureCause.WEAKENING_EFFECT, Health.CONTRADICTED: FailureCause.REVERSAL}


def apply_health(board: KnowledgeBoard, mid: str, health: Health, now, detail: str = "") -> dict:
    """React to a knowledge-health verdict (section 46). Only BROKEN and CONTRADICTED change the board: a champion is retired,
    a challenger demoted to shadow, a shadow retired. DORMANT is deliberately not a failure (section 12 - the conditions may
    return), DEGRADING and UNSTABLE only flag for review, and UNKNOWN/INSUFFICIENT_EVIDENCE never act (unknown must not become
    confidence OR condemnation). Returns what was decided."""
    health = Health.parse(health)
    m = board._get(mid)
    action = HEALTH_ACTIONS[health]
    out = {"mid": mid, "health": health.value, "action": action, "applied": False, "role": m.role.value}
    if action != "retire" or m.role in (P.RETIRED, P.RESEARCH):
        return out
    why = f"health {health.value}" + (f": {detail}" if detail else "")
    if m.role == P.CHALLENGER:
        board.demote(mid, now, why)
    else:
        board.retire(mid, HEALTH_CAUSE[health], why, now)
    out.update(applied=True, role=board.members[mid].role.value)
    return out


def slot_status(board: KnowledgeBoard, now) -> list[dict]:
    """One row per slot ever used: who holds it, for how long, and how many are waiting or gone. Empty slots are listed
    (decisions there use the general rule) - they are not errors."""
    rows: dict[str, dict] = {}
    for mid, m in sorted(board.members.items()):
        r = rows.setdefault(m.slot, {"slot": m.slot, "champion": None, "tenure_days": None, "challengers": 0, "shadows": 0,
                                     "research": 0, "retired": 0})
        key = {P.CHALLENGER: "challengers", P.SHADOW: "shadows", P.RESEARCH: "research", P.RETIRED: "retired"}.get(m.role)
        if key:
            r[key] += 1
    for slot, mid in board.champions.items():
        rows[slot]["champion"] = mid
        rows[slot]["tenure_days"] = (as_date(now) - as_date(board.members[mid].since)).days
    return [rows[k] for k in sorted(rows)]


def explain_member(board: KnowledgeBoard, mid: str) -> list[str]:
    """Human-readable life story of one knowledge version, straight from the hash-chained ledger (why it moved, when, on
    whose decision id). Shadow observations are summarised, not listed."""
    board._get(mid)
    lines, n_obs = [], 0
    for r in board.ledger.rows():
        if r["id"] != mid:
            continue
        d = r["detail"]
        if r["event"] in ("shadow_obs", "watch_obs"):
            n_obs += 1
            continue
        if n_obs:
            lines.append(f"  ... {n_obs} observations")
            n_obs = 0
        extra = d.get("reason") or (f"failed {d['failed']}" if d.get("failed") else "") or d.get("decision_id", "")
        lines.append(f"{r['t']} {r['event']}" + (f": {extra}" if extra else ""))
    if n_obs:
        lines.append(f"  ... {n_obs} observations")
    return lines


def promotion_funnel(board: KnowledgeBoard) -> dict:
    """Counts across the board's life: registered -> shadow -> challenger -> promoted, plus WHY promotions were refused."""
    ev: dict[str, int] = {}
    why: dict[str, int] = {}
    for r in board.ledger.rows():
        ev[r["event"]] = ev.get(r["event"], 0) + 1
        if r["event"] == "promotion_refused":
            for g in r["detail"].get("failed", []):
                why[g] = why.get(g, 0) + 1
    return {"events": ev, "refusal_reasons": dict(sorted(why.items(), key=lambda kv: -kv[1])),
            "attempts": ev.get("promoted", 0) + ev.get("promotion_refused", 0)}


def to_snapshot(board: KnowledgeBoard) -> dict:
    """What a compute worker may know about production knowledge: which versions are champions of which slots, and the board
    digest. Pass to engine.learning.compute.SnapshotStore.create(..., as_of=board.last_date). Shadows/challengers are absent
    on purpose - an experiment must not be conditioned on knowledge that has no production weight."""
    return {"champions": dict(sorted(board.champions.items())), "digest": board.digest(), "as_of": board.last_date}


# ------------------------------------------------------------------------------------------------ sequential monitoring
@dataclass(frozen=True)
class SequentialVerdict:
    action: str                 # CONTINUE | READY | ABANDON
    n: int
    fraction: float             # information fraction n / max_sessions
    z: float
    boundary: float             # efficacy boundary at this fraction
    reason: str


def obrien_fleming_bound(fraction: float, alpha: float = 0.05) -> float:
    """One-sided O'Brien-Fleming efficacy boundary z_(1-alpha) / sqrt(fraction). Looking at a shadow record every day and
    promoting at the first t > 1.64 promotes lucky noise about a third of the time; an alpha-spending boundary makes early
    looks demand far stronger evidence (about 3.3 at 25% information) and costs almost nothing at the final look."""
    from statistics import NormalDist
    if not 0.0 < fraction <= 1.0:
        raise ValueError("information fraction must be in (0, 1]")
    return NormalDist().inv_cdf(1.0 - alpha) / math.sqrt(fraction)


def sequential_verdict(board: KnowledgeBoard, mid: str, now, max_sessions: int = 60, alpha: float = 0.05) -> SequentialVerdict:
    """Peek-safe reading of a shadow record. READY only when the paired t crosses the O'Brien-Fleming boundary for the
    information collected so far; ABANDON when harm is already evident or the record is flat past half the budget; otherwise
    CONTINUE. Uses only sessions strictly before `now`."""
    if max_sessions < board.policy.min_shadow_sessions:
        raise ValueError("max_sessions below min_shadow_sessions can never trigger")
    st = board.shadow_record(mid, now)
    frac = min(1.0, st["n"] / max_sessions)
    if st["n"] < board.policy.min_shadow_sessions:
        return SequentialVerdict("CONTINUE", st["n"], frac, st["t"], math.inf, "too few sessions to look yet")
    bound = obrien_fleming_bound(frac, alpha)
    if st["t"] >= bound and st["mean"] > 0:
        return SequentialVerdict("READY", st["n"], frac, st["t"], bound, "crossed the alpha-spending boundary")
    if st["t"] < board.policy.shadow_harm_t:
        return SequentialVerdict("ABANDON", st["n"], frac, st["t"], bound, "shadow record shows harm")
    if frac >= 0.5 and st["t"] < 0.0:
        return SequentialVerdict("ABANDON", st["n"], frac, st["t"], bound, "half the budget used and the record is not positive")
    if frac >= 1.0:
        return SequentialVerdict("ABANDON", st["n"], frac, st["t"], bound, "full budget used without crossing the boundary")
    return SequentialVerdict("CONTINUE", st["n"], frac, st["t"], bound, "inconclusive")


# ------------------------------------------------------------------------------------------------ context breakdown
def shadow_by_context(board: KnowledgeBoard, mid: str, labels: Mapping[str, str], now=None, min_n: int = 5) -> dict[str, dict]:
    """Paired shadow result split by a context label per session date (e.g. regime, volatility bucket). `labels` maps ISO date
    -> label; sessions with no label are grouped under 'unlabelled' rather than dropped. Groups below `min_n` are reported
    but marked unreliable."""
    m = board._get(mid)
    groups: dict[str, list[float]] = {}
    for t, c, b in m.shadow:
        if now is not None and as_date(t) >= as_date(now):
            continue
        groups.setdefault(labels.get(str(as_date(t)), "unlabelled"), []).append(c - b)
    out = {}
    for lab, d in sorted(groups.items()):
        st = paired_stats(d)
        out[lab] = {**st, "sum": float(np.sum(d)), "reliable": st["n"] >= min_n}
    return out


def context_consistency(breakdown: Mapping[str, Mapping[str, float]], concentration: float = 0.6) -> dict:
    """Is the edge spread across contexts or supplied by one? A challenger whose whole gain comes from a single regime is a
    regime bet, not general knowledge (it may still be valid as conditional knowledge, but must be scoped as such)."""
    rel = {k: v for k, v in breakdown.items() if v.get("reliable")}
    if not rel:
        return {"contexts": 0, "positive_share": float("nan"), "worst_context": None, "worst_mean": float("nan"), "concentrated": False}
    pos_sum = sum(v["sum"] for v in rel.values() if v["sum"] > 0)
    top = max(rel, key=lambda k: rel[k]["sum"])
    worst = min(rel, key=lambda k: rel[k]["mean"])
    return {"contexts": len(rel), "positive_share": sum(v["mean"] > 0 for v in rel.values()) / len(rel),
            "worst_context": worst, "worst_mean": float(rel[worst]["mean"]),
            "concentrated": bool(len(rel) > 1 and pos_sum > 0 and rel[top]["sum"] / pos_sum > concentration), "top_context": top}


# ------------------------------------------------------------------------------------------------ recovery
RECOVERABLE_CAUSES = frozenset({FailureCause.TEMPORARY_INACTIVITY.value, FailureCause.WRONG_CONTEXT.value,
                                FailureCause.REGIME_CHANGE.value, FailureCause.INSUFFICIENT_EVIDENCE.value, "REPLACED"})


def recovery_candidates(board: KnowledgeBoard, conditions_returned: Callable[[Member], bool]) -> list[str]:
    """Retired knowledge whose retirement cause was situational (inactivity, wrong context, regime change, or replacement) and
    for which the caller's predicate says the enabling conditions are back. A pattern retired as a FALSE_PATTERN or as a
    REVERSAL is never a candidate: recovery is for knowledge that was true but out of season."""
    return sorted(mid for mid, m in board.members.items()
                  if m.role == P.RETIRED and m.cause in RECOVERABLE_CAUSES and conditions_returned(m))


def reinstate_recovered(board: KnowledgeBoard, conditions_returned: Callable[[Member], bool], now) -> list[str]:
    """Move every recovery candidate back to SHADOW (never straight to production - it must re-earn CHALLENGER)."""
    done = []
    for mid in recovery_candidates(board, conditions_returned):
        board.reinstate(mid, now, f"conditions for cause {board.members[mid].cause} have returned")
        done.append(mid)
    return done


# ------------------------------------------------------------------------------------------------ time travel
def board_as_of(ledger_path: str | Path, as_of, gate: PromotionGate | None = None) -> KnowledgeBoard:
    """A read-only board rebuilt from the ledger using ONLY events dated strictly before `as_of`: what was in production on that
    date. This is the future-memory audit's question - 'could this knowledge have been active when this decision was made?'
    - answered from the hash-chained record instead of from memory."""
    return KnowledgeBoard(ledger_path, gate, as_of=as_of)


def production_at(ledger_path: str | Path, as_of) -> tuple[str, ...]:
    return board_as_of(ledger_path, as_of).production_ids()


def diff_boards(old: KnowledgeBoard, new: KnowledgeBoard) -> dict:
    """What changed in production knowledge between two views (typically two dates of the same ledger)."""
    a, b = dict(old.champions), dict(new.champions)
    return {"added": sorted(set(b.values()) - set(a.values())), "removed": sorted(set(a.values()) - set(b.values())),
            "replaced": sorted((a[s], b[s]) for s in a if s in b and a[s] != b[s]),
            "new_members": sorted(set(new.members) - set(old.members))}


# ------------------------------------------------------------------------------------------------ housekeeping
def last_activity(m: Member) -> str:
    """Date of the newest thing that happened to a member: its last shadow session or its last role change."""
    return max([m.since] + [o[0] for o in m.shadow], key=as_date)


def expire_idle(board: KnowledgeBoard, now, max_idle_days: int = 180) -> list[str]:
    """Nothing waits in limbo forever: RESEARCH and SHADOW members with no activity for `max_idle_days` are RETIRED as
    INSUFFICIENT_EVIDENCE. That is a retirement, not a deletion - the record stays and `recovery_candidates` may reinstate it
    when new evidence arrives."""
    out = []
    for mid, m in sorted(board.members.items()):
        if m.role in (P.RESEARCH, P.SHADOW) and (as_date(now) - as_date(last_activity(m))).days > max_idle_days:
            board.retire(mid, FailureCause.INSUFFICIENT_EVIDENCE,
                         f"no shadow or research activity for more than {max_idle_days} days", now)
            out.append(mid)
    return out


def challenger_capacity(board: KnowledgeBoard) -> dict[str, int]:
    """Free challenger seats per slot (max_challengers_per_slot minus the seats in use)."""
    used: dict[str, int] = {}
    for m in board.members.values():
        used.setdefault(m.slot, 0)
        used[m.slot] += m.role == P.CHALLENGER
    return {slot: board.policy.max_challengers_per_slot - n for slot, n in sorted(used.items())}


def scorecard(board: KnowledgeBoard, now) -> dict:
    """One-glance state of the knowledge board for reports: counts by role, empty slots, oldest champion, slots with no free
    challenger seat, and any invariant problems (which must be empty)."""
    roles = {k: len(v) for k, v in board.roles().items()}
    rows = slot_status(board, now)
    oldest = max(rows, key=lambda r: r["tenure_days"] or -1, default=None)
    return {"members": len(board.members), "roles": roles, "empty_slots": [r["slot"] for r in rows if r["champion"] is None],
            "oldest_champion": None if not oldest or oldest["champion"] is None else
            {"mid": oldest["champion"], "tenure_days": oldest["tenure_days"]},
            "full_slots": [s for s, free in challenger_capacity(board).items() if free <= 0],
            "problems": board.invariants(), "digest": board.digest()}


def audit_decision_sources(board: KnowledgeBoard, used: Mapping[str, Sequence[str]]) -> list[dict]:
    """Firewall check for a finished decision run: `used` maps a decision id to the knowledge ids that influenced it. Only
    champions may influence a decision; anything else (a shadow, a challenger, a retired version, an id the board never
    heard of) is a violation - a shadow that leaked into production would make its own shadow record meaningless."""
    out = []
    for decision, mids in sorted(used.items()):
        for mid in mids:
            m = board.members.get(mid)
            if m is None:
                out.append({"decision": decision, "mid": mid, "problem": "unknown to the board"})
            elif m.role != P.CHAMPION:
                out.append({"decision": decision, "mid": mid, "problem": f"was {m.role.value}, not CHAMPION"})
    return out


def slot_history(board: KnowledgeBoard, slot: Slot) -> list[dict]:
    """Who held a slot and when, from the ledger: [{'mid', 'from', 'to', 'how_ended'}], the open tenure having to=None. Answers
    'what was deciding this on date D' without replaying anything (see `board_as_of` for the full historical board)."""
    tenures: list[dict] = []
    for r in board.ledger.rows():
        m = board.members.get(r["id"])
        if m is None or m.slot != slot.key:
            continue
        if r["event"] == "promoted":
            for ten in tenures:
                if ten["mid"] == r["detail"].get("replaced") and ten["to"] is None:
                    ten.update({"to": r["t"], "how_ended": "replaced"})
            tenures.append({"mid": r["id"], "from": r["t"], "to": None, "how_ended": None})
        elif r["event"] in ("retired", "rolled_back"):
            for ten in reversed(tenures):
                if ten["mid"] == r["id"] and ten["to"] is None:
                    ten.update({"to": r["t"], "how_ended": r["event"]})
                    break
            if r["event"] == "rolled_back":                       # the predecessor is restored the same day
                tenures.append({"mid": r["detail"]["restored"], "from": r["t"], "to": None, "how_ended": None})
    return tenures


# ------------------------------------------------------------------------------------------------ shadow scoring on identical decisions
@dataclass(frozen=True)
class DecisionPair:
    """One decision evaluated twice: what the shadow/challenger WOULD have delivered and what the live champion (or the general
    rule) actually delivered, on the SAME decision id. A comparison across different decisions is not a paired comparison."""
    date: str
    decision_id: str
    candidate: float
    champion: float


def score_on_identical_decisions(board: KnowledgeBoard, mid: str, pairs: Sequence[DecisionPair], now) -> dict:
    """Turn per-decision pairs into board shadow sessions (one session per date, the mean over that date's decisions). Every
    pair must be dated strictly before `now` (an outcome not yet matured is a FirewallBreach), be finite, and appear once;
    a date already recorded is skipped, so replaying the same batch after a crash is safe."""
    m = board._get(mid, P.SHADOW, P.CHALLENGER)
    by_date: dict[str, list[DecisionPair]] = {}
    seen: set[tuple[str, str]] = set()
    for p in pairs:
        require_past(p.date, now, f"shadow decision {p.decision_id}")
        if not (math.isfinite(p.candidate) and math.isfinite(p.champion)):
            raise BoardError(f"decision {p.decision_id}: non-finite value")
        key = (str(as_date(p.date)), p.decision_id)
        if key in seen:
            raise BoardError(f"decision {p.decision_id} on {key[0]} supplied twice")
        seen.add(key)
        by_date.setdefault(key[0], []).append(p)
    recorded, skipped = [], []
    for d in sorted(by_date):
        if any(as_date(o[0]) == as_date(d) for o in m.shadow):
            skipped.append(d)
            continue
        rows = by_date[d]
        board.record_shadow(mid, float(np.mean([r.candidate for r in rows])), float(np.mean([r.champion for r in rows])), d)
        recorded.append(d)
    return {"recorded": recorded, "skipped_existing": skipped, "n_decisions": len(seen)}


# ------------------------------------------------------------------------------------------------ degradation hand-off
def check_degradation(board: KnowledgeBoard, mid: str, live_diffs: Sequence[float], now, min_n: int = 10,
                      t_stop: float = -1.28, min_gap: float = 0.0005) -> dict:
    """Live degradation test for a CHAMPION from recent paired differences (champion minus the general rule, same decisions).
    Enough sessions, a negative mean beyond `min_gap` and a t below `t_stop` hands the champion to retirement (cause
    WEAKENING_EFFECT: the slot is left empty and decisions fall back to the general rule) and lists the eligible challengers
    that could replace it - which must still pass the promotion gate. Anything less leaves the board untouched."""
    m = board._get(mid, P.CHAMPION)
    st = paired_stats(live_diffs)
    degraded = st["n"] >= min_n and st["mean"] < -min_gap and st["t"] < t_stop
    out = {"degraded": bool(degraded), **st, "replacements": [], "action": "none"}
    if degraded:
        board.retire(mid, FailureCause.WEAKENING_EFFECT, f"live record degraded: mean {st['mean']:+.5f}, t {st['t']:.2f}", now)
        out["replacements"] = [r["mid"] for r in rank_challengers(board, Slot.from_key(m.slot), now) if r["eligible"]]
        out["action"] = "retired; slot empty until a replacement passes the promotion gate"
    return out


# ------------------------------------------------------------------------------------------------ head-to-head record
def head_to_head_record(board: KnowledgeBoard, family_of: Callable[[Member], str] | None = None) -> dict[str, dict]:
    """Who has beaten whom, per knowledge family (default family = the decision slot). A win is a promotion over an incumbent
    or a rollback that restored the predecessor; the ledger is the only source, so the record cannot be edited. Each family
    reports pairwise wins and a standings table (wins minus losses)."""
    fam = family_of or (lambda m: m.slot)
    out: dict[str, dict] = {}

    def bump(winner: str, loser: str, t: str, how: str) -> None:
        f = fam(board.members[winner])
        rec = out.setdefault(f, {"pairs": {}, "standings": {}, "contests": 0})
        pair = rec["pairs"].setdefault(f"{winner} > {loser}", {"wins": 0, "last": t, "how": []})
        pair["wins"] += 1
        pair["last"] = t
        pair["how"].append(how)
        rec["contests"] += 1
        rec["standings"][winner] = rec["standings"].get(winner, 0) + 1
        rec["standings"][loser] = rec["standings"].get(loser, 0) - 1

    for r in board.ledger.rows():
        if r["event"] == "promoted" and r["detail"].get("replaced"):
            bump(r["id"], r["detail"]["replaced"], r["t"], "promotion")
        elif r["event"] == "rolled_back":
            bump(r["detail"]["restored"], r["id"], r["t"], "rollback")
    return out


def readiness(board: KnowledgeBoard, mid: str, now, max_sessions: int = 60) -> dict:
    """Where a shadow or challenger stands and what it needs next: its shadow summary, the peek-safe sequential verdict, how
    many more sessions a confirmation would take at the observed effect, and the single next step the board would allow."""
    m = board._get(mid)
    s = shadow_summary(board, mid, now)
    seq = sequential_verdict(board, mid, now, max_sessions)
    need = sessions_needed(s.mean, s.sd)
    free = challenger_capacity(board).get(m.slot, board.policy.max_challengers_per_slot)
    if m.role == P.RETIRED:
        step = "retired: reinstate to SHADOW only if the enabling conditions have returned"
    elif m.role == P.CHAMPION:
        step = "in production: keep the post-promotion watch and live degradation check running"
    elif seq.action == "ABANDON":
        step = "retire or demote: the shadow record does not support it"
    elif m.role == P.CHALLENGER:
        step = "submit evidence to the promotion gate" if seq.action == "READY" else "keep collecting shadow sessions"
    elif s.n < board.policy.min_shadow_sessions:
        step = f"collect {board.policy.min_shadow_sessions - s.n} more shadow sessions"
    elif free <= 0:
        step = "wait: the slot has no free challenger seat"
    else:
        step = "open a challenge"
    return {"mid": mid, "role": m.role.value, "n": s.n, "t": s.t, "verdict": seq.action, "boundary": seq.boundary,
            "sessions_still_needed": None if not math.isfinite(need) else max(0.0, need - s.n), "next_step": step}
