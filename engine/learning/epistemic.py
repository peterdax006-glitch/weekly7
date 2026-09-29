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
from typing import Callable, Iterable, Mapping

import numpy as np

from engine.learning.core import (Epistemic, FirewallBreach, Health, Lifecycle, as_date, stable_hash)

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


def _callable(ev: EvidenceSummary, scope: str, th: Thresholds) -> bool:
    """True if the evidence justifies SUPPORTED, CONDITIONAL, CONTRADICTED or DEGRADED in this scope."""
    return (not _def_supported(ev, scope, th) or (bool(ev.works_in) and bool(ev.fails_in) and not _def_conditional(ev, scope, th))
            or _contradicted(ev, th) or (scope != HISTORICAL and _degraded(ev, th)))


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
    if _callable(ev, HISTORICAL if scope == HISTORICAL else scope, th):
        out.append("evidence already justifies a definite state; HYPOTHESIS would understate it")
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
    if not ev.fails_in:
        out.append("CONDITIONAL needs a context where it fails; otherwise it is SUPPORTED")
    if scope == CONTEXT:
        if not _confirmed(ev, th):
            out.append("CONDITIONAL inside a context still needs confirmed evidence there")
    else:
        # the whole-sample mean is a blend of working and failing contexts, so it is NOT expected to be confirmed;
        # what is required is enough data and P(real): the effect is real somewhere.
        if ev.n_eff < th.min_n_eff:
            out.append(f"CONDITIONAL needs n_eff >= {th.min_n_eff}, has {ev.n_eff}")
        if ev.p_real is None or ev.p_real < th.min_p_real:
            out.append(f"CONDITIONAL needs p_real >= {th.min_p_real}, has {ev.p_real}")
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
    if _callable(ev, scope, th):
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
    if ev.retire_reason and scope != CONTEXT:               # retirement is a fact about the item, not about one context
        return Epistemic.RETIRED
    if ev.gate_reason:
        return Epistemic.GATED
    if ev.works_in and ev.fails_in and not _def_conditional(ev, scope, th):
        return Epistemic.CONDITIONAL                         # contradiction explained by a context is not a contradiction
    if _contradicted(ev, th):
        return Epistemic.CONTRADICTED
    if scope != HISTORICAL and _degraded(ev, th):
        return Epistemic.DEGRADED
    if _confirmed(ev, th) and not ev.fails_in and (ev.contradiction_rate or 0.0) <= th.max_contradiction:
        if scope != CURRENT or (ev.recent_ratio is not None and ev.recent_ratio >= th.degraded_ratio):
            return Epistemic.SUPPORTED
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


# ---------------------------------------------------------------- turning per-period results into evidence

def _t_stat(x: "np.ndarray") -> float:
    n = len(x)
    if n < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return 0.0 if sd == 0 else float(np.mean(x) / (sd / math.sqrt(n)))


def _eff_n(x: "np.ndarray") -> float:
    """Effective sample size under AR(1) autocorrelation: n * (1 - r) / (1 + r), clipped to [1, n]. Overlapping or
    clustered periods otherwise make thin evidence look thick."""
    n = len(x)
    if n < 3 or float(np.std(x)) == 0:
        return float(n)
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if math.isnan(r):
        return float(n)
    r = max(0.0, min(0.95, r))
    return float(max(1.0, min(n, n * (1 - r) / (1 + r))))


def _block_disagreement(x: "np.ndarray", max_blocks: int = 8, min_block: int = 10) -> float | None:
    """Share of chronological blocks whose mean disagrees with the claimed direction (x is sign-adjusted so positive
    agrees). Period-by-period disagreement is not used: even a real effect loses individual periods, but it should not lose
    whole blocks. None when there is too little data for at least two blocks."""
    k = min(max_blocks, len(x) // min_block)
    if k < 2:
        return None
    return float(np.mean([b.mean() < 0 for b in np.array_split(x, k)]))


def summarize_periods(effects: Iterable[float], p_real: float | None = None, direction: int = 1, recent: int = 12,
                      confirm_from: int | None = None, prior_supported: bool = False, has_hypothesis: bool = True,
                      has_prediction: bool = True) -> EvidenceSummary:
    """Build an EvidenceSummary from a time-ordered series of per-period effect estimates (already past `now`).

    `direction` (+1/-1) is the claimed sign; t-statistics are signed so that a positive t always AGREES with the claim.
    `confirm_from` is the index where out-of-sample evidence starts (t_confirm is measured only on x[confirm_from:]);
    None means no out-of-sample split exists and t_confirm stays None (never measured, not zero).
    recent_ratio compares the last `recent` periods with everything before them; reversal_t is the t of the recent
    window in the OPPOSITE direction; contradiction_rate is the share of periods whose sign disagrees with the claim."""
    if direction not in (-1, 1):
        raise EpistemicError("direction must be +1 or -1")
    x = np.asarray([float(v) for v in effects], dtype=float)
    if len(x) and not np.all(np.isfinite(x)):
        raise EpistemicError("non-finite period effect")
    x = x * direction
    n = len(x)
    if n == 0:
        return EvidenceSummary(has_hypothesis=has_hypothesis, has_prediction=has_prediction, prior_supported=prior_supported)
    t_disc = _t_stat(x if confirm_from is None else x[:confirm_from])
    t_conf = None
    if confirm_from is not None and 0 < confirm_from < n - 1:
        t_conf = _t_stat(x[confirm_from:])
    ratio = rev = None
    if n > recent + 3 and recent >= 3:
        older, rec = x[:-recent], x[-recent:]
        base = float(np.mean(older))
        ratio = float(np.mean(rec) / base) if base > 0 else None
        rev = -_t_stat(rec)
    return EvidenceSummary(
        n_events=n, n_eff=_eff_n(x), t_discovery=t_disc, t_confirm=t_conf, p_real=p_real,
        contradiction_rate=_block_disagreement(x), recent_ratio=ratio, reversal_t=rev,
        has_hypothesis=has_hypothesis, has_prediction=has_prediction, prior_supported=prior_supported)


def summarize_by_context(effects: Iterable[float], labels: Iterable[str], p_real: float | None = None, direction: int = 1,
                         th: Thresholds = DEFAULT_THRESHOLDS) -> tuple[EvidenceSummary, dict[str, EvidenceSummary]]:
    """Whole-sample evidence plus one EvidenceSummary per context label, with works_in/fails_in filled on the whole-sample
    summary: a context works if its own evidence would be confirmed, fails if it would be contradicted or points the
    wrong way with enough data. Labels with too little data land in neither list (unknown, not failing)."""
    x = np.asarray([float(v) for v in effects], dtype=float)
    lab = np.asarray(list(labels))
    if len(x) != len(lab):
        raise EpistemicError("effects and labels differ in length")
    per, works, fails = {}, [], []
    for c in sorted(set(lab.tolist())):
        sub = x[lab == c]
        ev = summarize_periods(sub, p_real, direction, recent=3, confirm_from=len(sub) // 2 if len(sub) >= 8 else None)
        ev = dataclasses.replace(ev, works_in=(), fails_in=())
        per[c] = ev
        if ev.n_eff < th.min_n_eff / 2:
            continue
        if _confirmed(ev, th):
            works.append(c)
        elif (ev.t_discovery or 0.0) <= -th.contradicted_t or (ev.contradiction_rate or 0.0) >= th.contradicted_rate:
            fails.append(c)
    whole = summarize_periods(x, p_real, direction, confirm_from=len(x) // 2 if len(x) >= 8 else None)
    return dataclasses.replace(whole, works_in=tuple(works), fails_in=tuple(fails)), per


# ---------------------------------------------------------------- diagnostics and replay

def diagnose(ev: EvidenceSummary, scope: str = HISTORICAL, th: Thresholds = DEFAULT_THRESHOLDS) -> dict[str, list[str]]:
    """For every state: why the evidence does or does not justify it (empty list = justified). Used in reports."""
    return {s.value: violations(s, ev, scope, th) for s in Epistemic if s in SCOPE_STATES[scope]}


def justified_states(ev: EvidenceSummary, scope: str = HISTORICAL, th: Thresholds = DEFAULT_THRESHOLDS) -> list[Epistemic]:
    return [Epistemic(s) for s, why in diagnose(ev, scope, th).items() if not why]


def replay(history: Iterable[TransitionRecord]) -> EpistemicProfile:
    """Rebuild the scoped state from the transition log alone; equality with the live profile proves the log is complete."""
    scopes: dict[tuple[str, str], ScopedState] = {}
    for r in history:
        scopes[(r.kind, r.key)] = ScopedState(r.kind, r.key, r.to, r.at, r.reason)
    return EpistemicProfile(tuple(sorted(scopes.values(), key=lambda s: (s.kind, s.key))), tuple(history))


def render_history(profile: EpistemicProfile) -> str:
    rows = []
    for r in profile.history:
        where = r.kind + (f"[{r.key}]" if r.key else "")
        rows.append(f"{r.at}  {where:<22} {(r.frm.value if r.frm else 'START'):>12} -> {r.to.value:<12} {r.reason}")
    return "\n".join(rows) if rows else "(no transitions)"


def weight_over_time(profile: EpistemicProfile, active_contexts: tuple[str, ...] = (),
                     th: Thresholds = DEFAULT_THRESHOLDS) -> list[tuple[str, float]]:
    """Decision weight the profile would have granted after each recorded transition (a state trajectory for plotting)."""
    return [(r.at, replay(profile.history[:i + 1]).usage(active_contexts, th).weight) for i, r in enumerate(profile.history)]


# ---------------------------------------------------------------- the four existing state vocabularies -> contract states

@dataclasses.dataclass(frozen=True)
class StateMapping:
    """How one state of an existing module reads in the contract's terms. `scope` says which scope the source state
    speaks about (a health verdict is about the recent window, a lifecycle state about the whole history)."""
    epistemic: Epistemic
    lifecycle: Lifecycle | None
    scope: str
    health: Health | None = None
    gate: str = ""                 # for GATED: the gate that holds it
    retire_reason: str = ""        # for RETIRED
    note: str = ""


_S = StateMapping
_L = Lifecycle

# engine.pattern_lifecycle.STATES (== engine.pattern_identity.STATES). 'active' already passed that module's long-run /
# discovery / confirmation / recent tests, which is what SUPPORTED demands; nothing else is promoted here.
PATTERN_LIFECYCLE: dict[str, StateMapping] = {
    "candidate": _S(Epistemic.HYPOTHESIS, _L.BIRTH, HISTORICAL, note="mined, untested"),
    "rejected": _S(Epistemic.RETIRED, _L.RETIRED, HISTORICAL, retire_reason="failed the candidate tests (noise, not a regime loss)"),
    "duplicate": _S(Epistemic.GATED, _L.DORMANT, HISTORICAL, gate="redundant_with_stronger_pattern"),
    "no_gain": _S(Epistemic.GATED, _L.DORMANT, HISTORICAL, gate="no_incremental_gain",
                  note="may still be real (P(real) can be 1.0): the gain gate is order dependent"),
    "active": _S(Epistemic.SUPPORTED, _L.ACTIVE, HISTORICAL),
    "watch": _S(Epistemic.DEGRADED, _L.DECAY, CURRENT, note="under observation after a weak window"),
    "failed": _S(Epistemic.DEGRADED, _L.FAILURE, CURRENT, note="transient: must proceed to cause_search"),
    "cause_search": _S(Epistemic.DEGRADED, _L.FAILURE, CURRENT, note="cause of failure being searched"),
    "rescoped": _S(Epistemic.CONDITIONAL, _L.RECOVERY, HISTORICAL, note="works inside a named context only"),
    "discarded": _S(Epistemic.RETIRED, _L.RETIRED, HISTORICAL, retire_reason="failed and no context explained it"),
}

# engine.pattern_memory PatternWeight.mode
PATTERN_MEMORY_MODES: dict[str, StateMapping] = {
    "universal": _S(Epistemic.SUPPORTED, _L.ACTIVE, HISTORICAL, note="held across many independent periods"),
    "local": _S(Epistemic.CONDITIONAL, _L.ACTIVE, HISTORICAL, note="holds in some periods only"),
    "disregarded": _S(Epistemic.UNKNOWN, _L.DORMANT, HISTORICAL, note="no evidence it is real after all tries: not a claim it is false"),
    "gated": _S(Epistemic.GATED, _L.DORMANT, CURRENT, gate="predicted_unreliable_here", note="kept, switched off in this state"),
}

# engine.pattern_reliability.VERDICTS (C61: every break ends in a verdict) and HealthLedger status names
PATTERN_RELIABILITY_VERDICTS: dict[str, StateMapping] = {
    "EXPLAINED_AND_GATED": _S(Epistemic.CONDITIONAL, _L.RECOVERY, CURRENT, note="break explained by a driver; used outside its bad zone"),
    "DISCARDED_UNPREDICTABLE": _S(Epistemic.RETIRED, _L.RETIRED, HISTORICAL, retire_reason="broke and no predictor of the break survived"),
    "PHANTOM_DISCARDED": _S(Epistemic.RETIRED, _L.RETIRED, HISTORICAL, retire_reason="effect was never established (phantom)"),
    "OPEN": _S(Epistemic.DEGRADED, _L.FAILURE, CURRENT, note="investigation unresolved: cause UNKNOWN is a valid outcome"),
}
PATTERN_RELIABILITY_HEALTH: dict[str, StateMapping] = {
    "unmonitored": _S(Epistemic.UNKNOWN, _L.BIRTH, CURRENT, Health.INSUFFICIENT_EVIDENCE, note="burn-in: no health opinion yet"),
    "healthy": _S(Epistemic.SUPPORTED, _L.ACTIVE, CURRENT, Health.HEALTHY),
    "suspect": _S(Epistemic.DEGRADED, _L.DECAY, CURRENT, Health.DEGRADING),
    "broken": _S(Epistemic.DEGRADED, _L.FAILURE, CURRENT, Health.BROKEN, note="alarm raised; contradicted only if reversal evidence exists"),
}

# engine.pattern_bank.PRIOR_STATES: a subset of the lifecycle vocabulary, reused as-is
PATTERN_BANK_PRIOR: dict[str, StateMapping] = {s: PATTERN_LIFECYCLE[s] for s in ("active", "watch", "rescoped", "discarded")}

VOCABULARIES: dict[str, dict[str, StateMapping]] = {
    "pattern_lifecycle.STATES": PATTERN_LIFECYCLE,
    "pattern_memory.mode": PATTERN_MEMORY_MODES,
    "pattern_reliability.VERDICTS": PATTERN_RELIABILITY_VERDICTS,
    "pattern_reliability.STATUS_NAMES": PATTERN_RELIABILITY_HEALTH,
    "pattern_bank.PRIOR_STATES": PATTERN_BANK_PRIOR,
}


def map_state(vocabulary: str, state: str) -> StateMapping:
    """Translate an existing module's state. Unknown states raise: a new state added elsewhere must be mapped on purpose."""
    try:
        table = VOCABULARIES[vocabulary]
    except KeyError:
        raise EpistemicError(f"unknown vocabulary {vocabulary!r}; known: {sorted(VOCABULARIES)}") from None
    key = str(state)
    if key not in table:
        raise EpistemicError(f"{vocabulary} state {state!r} has no contract mapping; add it to epistemic.VOCABULARIES")
    return table[key]


def mapping_problems() -> list[str]:
    """Self-consistency of the mapping tables: every mapped state must be legal in its scope and carry the reason its
    contract state requires (GATED needs a gate, RETIRED needs a reason)."""
    errs = []
    for vname, table in VOCABULARIES.items():
        for name, m in table.items():
            if m.epistemic not in SCOPE_STATES[m.scope]:
                errs.append(f"{vname}:{name}: {m.epistemic.value} is not valid in scope {m.scope}")
            if m.epistemic == Epistemic.GATED and not m.gate:
                errs.append(f"{vname}:{name}: GATED without a gate name")
            if m.epistemic == Epistemic.RETIRED and not m.retire_reason:
                errs.append(f"{vname}:{name}: RETIRED without a reason")
            if (m.lifecycle == Lifecycle.RETIRED) != (m.epistemic == Epistemic.RETIRED):
                errs.append(f"{vname}:{name}: lifecycle RETIRED and epistemic RETIRED must go together")
    return errs


def unmapped_states() -> dict[str, list[str]]:
    """Import the live modules and list states they define that the tables above do not cover (empty dict = complete).
    Lazy imports keep this module light; a module that cannot be imported is reported, never silently skipped."""
    found: dict[str, list[str]] = {}

    def gap(vocab: str, names) -> None:
        miss = sorted(str(n) for n in names if str(n) not in VOCABULARIES[vocab])
        if miss:
            found[vocab] = miss

    try:
        from engine import pattern_lifecycle
        gap("pattern_lifecycle.STATES", pattern_lifecycle.STATES)
    except Exception as e:                                      # pragma: no cover - environment specific
        found["pattern_lifecycle.STATES"] = [f"import failed: {e}"]
    try:
        from engine import pattern_bank
        gap("pattern_bank.PRIOR_STATES", pattern_bank.PRIOR_STATES)
    except Exception as e:                                      # pragma: no cover
        found["pattern_bank.PRIOR_STATES"] = [f"import failed: {e}"]
    try:
        from engine import pattern_reliability as pr
        gap("pattern_reliability.VERDICTS", pr.VERDICTS)
        gap("pattern_reliability.STATUS_NAMES", pr.STATUS_NAMES.values())
    except Exception as e:                                      # pragma: no cover
        found["pattern_reliability.VERDICTS"] = [f"import failed: {e}"]
    return found
