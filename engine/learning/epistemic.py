"""Contract C62 section 6 (A02): epistemic states with ENFORCED definitions, legal transitions and scoped states.

A learned item does not have one truth value. It has a state per SCOPE: historical (over everything ever seen), by context
(e.g. regime=bear) and current (the recent window). "Worked historically, fails in the current regime" is therefore not
"false": it is SUPPORTED historically + CONDITIONAL by regime + DEGRADED currently, and the profile keeps all three.

Nothing here reads market data. Callers hand in an `EvidenceSummary` measured on data that matured before `now`; every state
change is appended to a hash-chained transition log so the history of belief is auditable and cannot be rewritten (section 49).
Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Mapping

from engine.learning.core import (Epistemic, FirewallBreach, as_date, stable_hash)

HISTORICAL, CONTEXT, CURRENT = "HISTORICAL", "CONTEXT", "CURRENT"
SCOPE_KINDS = (HISTORICAL, CONTEXT, CURRENT)


class EpistemicError(ValueError):
    """A state was asserted that its definition or the transition table does not permit."""


class IllegalTransition(EpistemicError):
    pass


# ---------------------------------------------------------------- evidence and thresholds

@dataclasses.dataclass(frozen=True)
class EvidenceSummary:
    """Everything the definitions may look at, measured by the caller on data that matured before `now`.
    None means NOT MEASURED - which is never the same as zero."""
    n_events: int = 0
    n_eff: float = 0.0                       # effective sample size (overlap/cluster adjusted)
    t_discovery: float | None = None         # in-sample statistic
    t_confirm: float | None = None           # out-of-sample statistic (sign = agreement with the claimed direction)
    p_real: float | None = None              # P(real) from the pattern-truth machinery
    contradiction_rate: float | None = None  # share of independent evidence pointing the other way
    recent_ratio: float | None = None        # recent effect / long-run effect (1 = unchanged, <0 = reversed)
    reversal_t: float | None = None          # significance of an OPPOSITE-sign effect in the recent window
    works_in: tuple[str, ...] = ()           # context keys where the effect is confirmed
    fails_in: tuple[str, ...] = ()           # context keys where it is confirmed absent/reversed
    has_hypothesis: bool = False
    has_prediction: bool = False             # a testable prediction was stated BEFORE the test
    gate_reason: str = ""
    retire_reason: str = ""
    prior_supported: bool = False            # was SUPPORTED/CONDITIONAL in some earlier scope/version

    def check(self) -> list[str]:
        errs = []
        if self.n_events < 0 or self.n_eff < 0 or math.isnan(self.n_eff):
            errs.append("sample sizes must be finite and >= 0")
        if self.n_eff > self.n_events + 1e-9:
            errs.append("n_eff exceeds n_events")
        for name in ("p_real", "contradiction_rate"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                errs.append(f"{name}={v} outside [0,1]")
        for name in ("t_discovery", "t_confirm", "recent_ratio", "reversal_t"):
            v = getattr(self, name)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                errs.append(f"{name} not finite")
        if set(self.works_in) & set(self.fails_in):
            errs.append(f"context both works and fails: {sorted(set(self.works_in) & set(self.fails_in))}")
        return errs


@dataclasses.dataclass(frozen=True)
class Thresholds:
    min_n_eff: float = 30.0
    min_t_confirm: float = 2.0
    min_p_real: float = 0.70
    max_contradiction: float = 0.30
    degraded_ratio: float = 0.50            # recent effect below half the long-run effect -> degraded
    contradicted_t: float = 2.0
    contradicted_rate: float = 0.50
    degraded_weight: float = 0.25           # decision weight multiplier while DEGRADED (section 11: reduce, not drop)

    def check(self) -> list[str]:
        errs = []
        if self.min_n_eff <= 0 or self.min_t_confirm <= 0 or self.contradicted_t <= 0:
            errs.append("thresholds must be positive")
        for n in ("min_p_real", "max_contradiction", "contradicted_rate", "degraded_weight"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                errs.append(f"{n} outside [0,1]")
        if self.max_contradiction >= self.contradicted_rate:
            errs.append("max_contradiction must be below contradicted_rate or SUPPORTED and CONTRADICTED overlap")
        return errs


DEFAULT_THRESHOLDS = Thresholds()


def _confirmed(ev: EvidenceSummary, th: Thresholds) -> bool:
    return (ev.t_confirm is not None and ev.t_confirm >= th.min_t_confirm and ev.n_eff >= th.min_n_eff
            and ev.p_real is not None and ev.p_real >= th.min_p_real)


def _contradicted(ev: EvidenceSummary, th: Thresholds) -> bool:
    if ev.reversal_t is not None and ev.reversal_t >= th.contradicted_t:
        return True
    return (ev.contradiction_rate is not None and ev.contradiction_rate >= th.contradicted_rate
            and ev.n_eff >= th.min_n_eff)


def _degraded(ev: EvidenceSummary, th: Thresholds) -> bool:
    return (ev.prior_supported and ev.recent_ratio is not None and ev.recent_ratio < th.degraded_ratio
            and not _contradicted(ev, th))


# ---------------------------------------------------------------- definitions (enforced)

@dataclasses.dataclass(frozen=True)
class Definition:
    state: Epistemic
    meaning: str
    check: Callable[[EvidenceSummary, str, Thresholds], list[str]]


def _def_observed(ev, scope, th):
    out = []
    if ev.n_events < 1:
        out.append("OBSERVED needs at least one recorded event")
    if ev.has_hypothesis:
        out.append("OBSERVED is an observation only; a stated hypothesis makes it HYPOTHESIS")
    return out


def _def_hypothesis(ev, scope, th):
    out = []
    if not ev.has_hypothesis:
        out.append("HYPOTHESIS needs a stated hypothesis")
    if not ev.has_prediction:
        out.append("HYPOTHESIS needs a testable prediction stated before the test")
    if _confirmed(ev, th) and not _contradicted(ev, th):
        out.append("confirmed evidence exists: state must be SUPPORTED/CONDITIONAL, not HYPOTHESIS")
    return out


def _def_supported(ev, scope, th):
    out = []
    if ev.n_eff < th.min_n_eff:
        out.append(f"SUPPORTED needs n_eff >= {th.min_n_eff}, has {ev.n_eff}")
    if ev.t_confirm is None:
        out.append("SUPPORTED needs out-of-sample confirmation (t_confirm not measured)")
    elif ev.t_confirm < th.min_t_confirm:
        out.append(f"t_confirm {ev.t_confirm:.2f} below {th.min_t_confirm}")
    if ev.p_real is None or ev.p_real < th.min_p_real:
        out.append(f"p_real {ev.p_real} below {th.min_p_real}")
    if ev.contradiction_rate is not None and ev.contradiction_rate > th.max_contradiction:
        out.append(f"contradiction_rate {ev.contradiction_rate:.2f} above {th.max_contradiction}")
    if ev.fails_in:
        out.append(f"fails in {list(ev.fails_in)}: that is CONDITIONAL, not unconditionally SUPPORTED")
    if scope == CURRENT and (ev.recent_ratio is None or ev.recent_ratio < th.degraded_ratio):
        out.append("CURRENT SUPPORTED needs a measured recent effect at least half the long-run effect")
    return out


def _def_conditional(ev, scope, th):
    out = []
    if not ev.works_in:
        out.append("CONDITIONAL needs at least one context where it is confirmed")
    if scope != CONTEXT and not ev.fails_in:
        out.append("CONDITIONAL needs a context where it fails; otherwise it is SUPPORTED")
    if not _confirmed(ev, th):
        out.append("CONDITIONAL needs confirmed evidence inside the working context")
    return out


def _def_degraded(ev, scope, th):
    out = []
    if scope == HISTORICAL:
        out.append("DEGRADED is a statement about the recent window; not valid for the HISTORICAL scope")
    if not ev.prior_supported:
        out.append("DEGRADED requires the item to have been SUPPORTED/CONDITIONAL before")
    if ev.recent_ratio is None:
        out.append("DEGRADED needs a measured recent_ratio")
    elif ev.recent_ratio >= th.degraded_ratio:
        out.append(f"recent_ratio {ev.recent_ratio:.2f} not below {th.degraded_ratio}")
    if _contradicted(ev, th):
        out.append("significant opposite-sign evidence: that is CONTRADICTED, not DEGRADED")
    return out


def _def_contradicted(ev, scope, th):
    if _contradicted(ev, th):
        return []
    return ["CONTRADICTED needs significant opposite evidence (reversal_t or contradiction_rate with enough n_eff)"]


def _def_gated(ev, scope, th):
    return [] if ev.gate_reason else ["GATED needs the name of the gate that holds it"]


def _def_retired(ev, scope, th):
    return [] if ev.retire_reason else ["RETIRED needs an explicit reason (retire is never silent deletion)"]


def _def_unknown(ev, scope, th):
    out = []
    if _confirmed(ev, th) or _contradicted(ev, th):
        out.append("evidence is sufficient to make a call; UNKNOWN would hide it")
    return out


DEFINITIONS: dict[Epistemic, Definition] = {
    Epistemic.OBSERVED: Definition(Epistemic.OBSERVED, "something happened; no claim about why or whether it recurs", _def_observed),
    Epistemic.HYPOTHESIS: Definition(Epistemic.HYPOTHESIS, "a stated, testable claim not yet confirmed out of sample", _def_hypothesis),
    Epistemic.SUPPORTED: Definition(Epistemic.SUPPORTED, "confirmed out of sample, enough effective data, no context failures", _def_supported),
    Epistemic.CONDITIONAL: Definition(Epistemic.CONDITIONAL, "confirmed only inside named contexts, fails elsewhere", _def_conditional),
    Epistemic.DEGRADED: Definition(Epistemic.DEGRADED, "was supported; recent effect is a fraction of the long-run effect", _def_degraded),
    Epistemic.CONTRADICTED: Definition(Epistemic.CONTRADICTED, "significant evidence of the opposite effect", _def_contradicted),
    Epistemic.GATED: Definition(Epistemic.GATED, "held out of use by a named gate regardless of its statistics", _def_gated),
    Epistemic.RETIRED: Definition(Epistemic.RETIRED, "withdrawn with a reason; kept, never deleted", _def_retired),
    Epistemic.UNKNOWN: Definition(Epistemic.UNKNOWN, "evidence too thin to call; must not be read as confidence", _def_unknown),
}


def violations(state: Epistemic, ev: EvidenceSummary, scope: str = HISTORICAL,
               th: Thresholds = DEFAULT_THRESHOLDS) -> list[str]:
    """Why `ev` does NOT justify `state` in `scope` (empty list = the definition holds)."""
    state = Epistemic.parse(state)
    if scope not in SCOPE_KINDS:
        raise EpistemicError(f"unknown scope kind {scope!r}")
    errs = list(ev.check())
    if state not in SCOPE_STATES[scope]:
        errs.append(f"{state.value} is not a valid state for scope {scope}")
    errs.extend(DEFINITIONS[state].check(ev, scope, th))
    return errs


def derive_state(ev: EvidenceSummary, scope: str = HISTORICAL, th: Thresholds = DEFAULT_THRESHOLDS) -> Epistemic:
    """The strongest state the evidence justifies, in precedence order. Always satisfies its own definition (tested)."""
    bad = ev.check()
    if bad:
        raise EpistemicError("; ".join(bad))
    if ev.retire_reason:
        return Epistemic.RETIRED
    if ev.gate_reason:
        return Epistemic.GATED
    if _contradicted(ev, th):
        return Epistemic.CONTRADICTED
    if scope != HISTORICAL and _degraded(ev, th):
        return Epistemic.DEGRADED
    if ev.works_in and ev.fails_in and _confirmed(ev, th):
        return Epistemic.CONDITIONAL
    if _confirmed(ev, th) and not ev.fails_in and (ev.contradiction_rate or 0.0) <= th.max_contradiction:
        if scope != CURRENT or (ev.recent_ratio is not None and ev.recent_ratio >= th.degraded_ratio):
            return Epistemic.SUPPORTED
    if scope == CONTEXT and ev.works_in and _confirmed(ev, th):
        return Epistemic.CONDITIONAL
    if ev.has_hypothesis and ev.has_prediction:
        return Epistemic.HYPOTHESIS
    if ev.n_events >= 1 and scope == HISTORICAL and not ev.has_hypothesis and ev.n_eff < th.min_n_eff:
        return Epistemic.OBSERVED
    return Epistemic.UNKNOWN


# ---------------------------------------------------------------- scopes and transitions

SCOPE_STATES: dict[str, frozenset[Epistemic]] = {
    HISTORICAL: frozenset({Epistemic.OBSERVED, Epistemic.HYPOTHESIS, Epistemic.SUPPORTED, Epistemic.CONDITIONAL,
                           Epistemic.CONTRADICTED, Epistemic.GATED, Epistemic.RETIRED, Epistemic.UNKNOWN}),
    CONTEXT: frozenset({Epistemic.HYPOTHESIS, Epistemic.SUPPORTED, Epistemic.CONDITIONAL, Epistemic.DEGRADED,
                        Epistemic.CONTRADICTED, Epistemic.UNKNOWN, Epistemic.GATED}),
    CURRENT: frozenset({Epistemic.HYPOTHESIS, Epistemic.SUPPORTED, Epistemic.CONDITIONAL, Epistemic.DEGRADED,
                        Epistemic.CONTRADICTED, Epistemic.GATED, Epistemic.RETIRED, Epistemic.UNKNOWN}),
}

_E = Epistemic
ALLOWED: dict[Epistemic | None, frozenset[Epistemic]] = {
    None: frozenset({_E.OBSERVED, _E.HYPOTHESIS, _E.UNKNOWN}),
    _E.OBSERVED: frozenset({_E.HYPOTHESIS, _E.UNKNOWN, _E.RETIRED}),
    _E.HYPOTHESIS: frozenset({_E.SUPPORTED, _E.CONDITIONAL, _E.CONTRADICTED, _E.UNKNOWN, _E.GATED, _E.RETIRED}),
    _E.SUPPORTED: frozenset({_E.CONDITIONAL, _E.DEGRADED, _E.CONTRADICTED, _E.GATED, _E.RETIRED}),
    _E.CONDITIONAL: frozenset({_E.SUPPORTED, _E.DEGRADED, _E.CONTRADICTED, _E.GATED, _E.RETIRED}),
    _E.DEGRADED: frozenset({_E.SUPPORTED, _E.CONDITIONAL, _E.CONTRADICTED, _E.GATED, _E.RETIRED}),
    _E.CONTRADICTED: frozenset({_E.HYPOTHESIS, _E.CONDITIONAL, _E.GATED, _E.RETIRED}),
    _E.GATED: frozenset({_E.SUPPORTED, _E.CONDITIONAL, _E.DEGRADED, _E.HYPOTHESIS, _E.UNKNOWN, _E.RETIRED}),
    _E.RETIRED: frozenset({_E.HYPOTHESIS}),          # revival only, and only with new evidence (revive=True)
    _E.UNKNOWN: frozenset({_E.OBSERVED, _E.HYPOTHESIS, _E.SUPPORTED, _E.CONDITIONAL, _E.CONTRADICTED, _E.GATED,
                           _E.RETIRED}),
}


def legal(frm: Epistemic | None, to: Epistemic) -> bool:
    return Epistemic.parse(to) in ALLOWED[Epistemic.parse(frm) if frm is not None else None]


def transition_graph() -> dict[str, list[str]]:
    """The full legal-move table as plain strings (for reports and for tests that walk every edge)."""
    return {("START" if k is None else k.value): sorted(x.value for x in v) for k, v in ALLOWED.items()}


@dataclasses.dataclass(frozen=True)
class ScopedState:
    kind: str
    key: str                     # "" for HISTORICAL/CURRENT, the context label (e.g. "regime=bear") for CONTEXT
    state: Epistemic
    since: str
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class TransitionRecord:
    kind: str
    key: str
    frm: Epistemic | None
    to: Epistemic
    at: str
    reason: str
    evidence_hash: str
    revive: bool
    prev_hash: str
    hash: str = ""

    def body_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("hash")
        return stable_hash(d, 20)


@dataclasses.dataclass(frozen=True)
class Usage:
    weight: float
    reason: str


@dataclasses.dataclass(frozen=True)
class EpistemicProfile:
    """Immutable scoped belief. `apply` returns a NEW profile; the old one is untouched."""
    scopes: tuple[ScopedState, ...] = ()
    history: tuple[TransitionRecord, ...] = ()

    # -- lookup
    def state(self, kind: str, key: str = "") -> Epistemic | None:
        for s in self.scopes:
            if s.kind == kind and s.key == key:
                return s.state
        return None

    @property
    def historical(self) -> Epistemic | None:
        return self.state(HISTORICAL)

    @property
    def current(self) -> Epistemic | None:
        return self.state(CURRENT)

    def contexts(self) -> dict[str, Epistemic]:
        return {s.key: s.state for s in self.scopes if s.kind == CONTEXT}

    def is_empty(self) -> bool:
        return not self.scopes

    # -- change
    def apply(self, kind: str, key: str, new: Epistemic, ev: EvidenceSummary, now, reason: str = "",
              revive: bool = False, th: Thresholds = DEFAULT_THRESHOLDS) -> "EpistemicProfile":
        new = Epistemic.parse(new)
        if kind not in SCOPE_KINDS:
            raise EpistemicError(f"unknown scope kind {kind!r}")
        if (kind == CONTEXT) != bool(key):
            raise EpistemicError("CONTEXT scope needs a key; HISTORICAL/CURRENT must not have one")
        old = self.state(kind, key)
        if old == new:
            raise IllegalTransition(f"{kind}/{key or '-'} is already {new.value}")
        first_look = old is None and kind != HISTORICAL          # a new scope is assessed once, from whatever evidence says
        if not first_look and not legal(old, new):
            raise IllegalTransition(f"{kind}/{key or '-'}: {old.value if old else 'START'} -> {new.value} is not a legal move")
        if old == Epistemic.RETIRED and not revive:
            raise IllegalTransition("a RETIRED item can only return as HYPOTHESIS with revive=True and new evidence")
        if old == Epistemic.RETIRED and (ev.n_events < 1):
            raise IllegalTransition("revival needs new evidence")
        if self.history and as_date(now) < as_date(self.history[-1].at):
            raise FirewallBreach(f"transition dated {now} precedes the previous one {self.history[-1].at}")
        why = violations(new, ev, kind, th)
        if new == Epistemic.RETIRED and old == Epistemic.RETIRED:
            why = []
        if why:
            raise EpistemicError(f"{new.value} not justified in {kind}: " + "; ".join(why))
        rec = TransitionRecord(kind, key, old, new, str(as_date(now)), reason, stable_hash(ev), bool(revive),
                               self.history[-1].hash if self.history else "")
        rec = dataclasses.replace(rec, hash=rec.body_hash())
        kept = tuple(s for s in self.scopes if not (s.kind == kind and s.key == key))
        scopes = tuple(sorted(kept + (ScopedState(kind, key, new, rec.at, reason),), key=lambda s: (s.kind, s.key)))
        out = EpistemicProfile(scopes, self.history + (rec,))
        bad = out.coherence()
        if bad:
            raise EpistemicError("incoherent profile: " + "; ".join(bad))
        return out

    # -- audit
    def verify_history(self) -> list[str]:
        errs, prev = [], ""
        for i, r in enumerate(self.history):
            if r.prev_hash != prev:
                errs.append(f"transition {i}: prev_hash broken")
            if r.hash != r.body_hash():
                errs.append(f"transition {i}: body altered")
            prev = r.hash
        return errs

    def coherence(self) -> list[str]:
        """Rules across scopes: the same item cannot be currently strong if it was never historically strong."""
        errs = []
        hist, cur = self.historical, self.current
        strong = {Epistemic.SUPPORTED, Epistemic.CONDITIONAL}
        if cur in strong and hist is not None and hist not in strong:
            errs.append(f"currently {cur.value} but historically {hist.value}")
        if cur == Epistemic.DEGRADED and hist is not None and hist not in strong:
            errs.append("DEGRADED currently requires SUPPORTED/CONDITIONAL history")
        if hist == Epistemic.RETIRED and cur not in (None, Epistemic.RETIRED):
            errs.append("historically RETIRED but currently not retired")
        if hist == Epistemic.CONDITIONAL and not any(s in strong for s in self.contexts().values()):
            errs.append("historically CONDITIONAL but no context scope is SUPPORTED/CONDITIONAL")
        for s in self.scopes:
            if s.state not in SCOPE_STATES[s.kind]:
                errs.append(f"{s.state.value} invalid in scope {s.kind}")
        return errs

    def headline(self) -> Epistemic:
        """One label for dashboards; the scoped states remain the truth. Precedence favours the cautious reading."""
        hist, cur, ctx = self.historical, self.current, set(self.contexts().values())
        if self.is_empty():
            return Epistemic.UNKNOWN
        if hist == Epistemic.RETIRED or cur == Epistemic.RETIRED:
            return Epistemic.RETIRED
        if cur == Epistemic.GATED or hist == Epistemic.GATED:
            return Epistemic.GATED
        if cur is not None:
            if cur == Epistemic.SUPPORTED and (Epistemic.CONTRADICTED in ctx or Epistemic.DEGRADED in ctx):
                return Epistemic.CONDITIONAL
            return cur
        if hist == Epistemic.SUPPORTED and (Epistemic.CONTRADICTED in ctx or Epistemic.DEGRADED in ctx):
            return Epistemic.CONDITIONAL
        return hist if hist is not None else Epistemic.CONDITIONAL if ctx & {Epistemic.SUPPORTED, Epistemic.CONDITIONAL} else Epistemic.UNKNOWN

    def summary(self) -> str:
        parts = []
        if self.historical:
            parts.append(f"{self.historical.value} historically")
        for k, v in sorted(self.contexts().items()):
            parts.append(f"{v.value} at {k}")
        if self.current:
            parts.append(f"{self.current.value} currently")
        return "; ".join(parts) or "UNKNOWN (no scoped evidence)"

    def usage(self, active_contexts: tuple[str, ...] = (), th: Thresholds = DEFAULT_THRESHOLDS) -> Usage:
        """Decision-weight multiplier in [0,1]. Only the CURRENT scope (falling back to HISTORICAL when the recent window has
        not been assessed) can grant weight, and an active context that is contradicted or degraded vetoes it."""
        cur = self.current if self.current is not None else self.historical
        if cur is None:
            return Usage(0.0, "no epistemic evidence")
        ctx = self.contexts()
        for k in active_contexts:
            if ctx.get(k) in (Epistemic.CONTRADICTED, Epistemic.DEGRADED, Epistemic.GATED):
                return Usage(0.0, f"active context {k} is {ctx[k].value}")
        if cur == Epistemic.SUPPORTED:
            return Usage(1.0, "supported")
        if cur == Epistemic.CONDITIONAL:
            ok = [k for k in active_contexts if ctx.get(k) in (Epistemic.SUPPORTED, Epistemic.CONDITIONAL)]
            return Usage(1.0, f"conditional; active in {ok}") if ok else Usage(0.0, "conditional and no working context is active")
        if cur == Epistemic.DEGRADED:
            return Usage(th.degraded_weight, "degraded: reduced weight, not removed")
        return Usage(0.0, f"{cur.value} carries no decision weight")


def build_profile(now, historical: EvidenceSummary, per_context: Mapping[str, EvidenceSummary] | None = None,
                  current: EvidenceSummary | None = None, th: Thresholds = DEFAULT_THRESHOLDS) -> EpistemicProfile:
    """Derive a complete profile from evidence in three scopes. Context scopes go first so a CONDITIONAL history is coherent."""
    prof = EpistemicProfile()
    for key, ev in sorted((per_context or {}).items()):
        prof = prof.apply(CONTEXT, key, derive_state(ev, CONTEXT, th), ev, now, "derived", th=th)
    target = derive_state(historical, HISTORICAL, th)
    if not legal(None, target):
        # belief forms through a stated hypothesis before it is confirmed, retired or gated
        pre = EvidenceSummary(n_events=max(historical.n_events, 1), has_hypothesis=True, has_prediction=True)
        prof = prof.apply(HISTORICAL, "", Epistemic.HYPOTHESIS, pre, now, "entered as hypothesis", th=th)
    prof = prof.apply(HISTORICAL, "", target, historical, now, "derived", th=th)
    if current is not None:
        prof = prof.apply(CURRENT, "", derive_state(current, CURRENT, th), current, now, "derived", th=th)
    return prof
