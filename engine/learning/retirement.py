"""Retirement and recovery without deletion (contract C62 sections 12-13; checklist B11 recovery index, J05 retirement gate,
J06 recovery gate). Bible phase serving: pattern lifecycle (Phase 4) generalised to any knowledge item.

Four states - ACTIVE, DEGRADED, DORMANT, RETIRED - and one rule: nothing is ever deleted. A ledger of immutable, hash-chained
transitions records every move with its reason, cause and the numbers behind it; the state at any date is a replay of the
transitions that were decided STRICTLY BEFORE that date (a decision taken at the close of day d is known from d+1).
DORMANT and RETIRED items carry zero influence and `assert_clean` raises FirewallBreach if any caller tries to give them
weight. Recovery is never automatic: it needs NEW evidence (window starting after the item went inactive), more of it and
stronger than the evidence that degraded it (hysteresis), a probation stop in DEGRADED, and - when the item carries stated
recovery conditions - a current context that satisfies one of them.

Built on: engine.learning.core vocabulary; the state names of engine/pattern_lifecycle.py are mapped by `adapt_pattern_state`
(that module keeps owning pattern-level break detection, B27 owns pattern health). This module works at the generic
knowledge level and consumes an `Evidence` record that those modules' tests can populate."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from .core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, Health, Lifecycle, as_date,
                   canonical_json, require_past, stable_hash)


class State(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    DORMANT = "DORMANT"
    RETIRED = "RETIRED"

    def __str__(self):
        return self.value

    @property
    def lifecycle(self) -> Lifecycle:
        return Lifecycle(self.value)

    @property
    def epistemic(self) -> Epistemic | None:
        """The epistemic label a state implies (None = leave the item's own label alone)."""
        return {"DEGRADED": Epistemic.DEGRADED, "DORMANT": Epistemic.GATED, "RETIRED": Epistemic.RETIRED}.get(self.value)


ALLOWED: dict[State, frozenset[State]] = {
    State.ACTIVE: frozenset({State.DEGRADED, State.DORMANT, State.RETIRED}),
    State.DEGRADED: frozenset({State.ACTIVE, State.DORMANT, State.RETIRED}),
    State.DORMANT: frozenset({State.DEGRADED, State.RETIRED}),      # recovery passes through probation, never straight to ACTIVE
    State.RETIRED: frozenset({State.DEGRADED}),                     # revival is possible, but only via the stricter revival gate
}

KINDS = ("REGISTER", "DEGRADE", "DORMANT", "RETIRE", "RECOVER_PROBATION", "RECOVER_FULL", "REVIVE", "REVERT")

# causes that mean the pattern was never real / is redundant: retire outright instead of parking it
TERMINAL_CAUSES = frozenset({FailureCause.FALSE_PATTERN, FailureCause.MEASUREMENT_ERROR, FailureCause.REDUNDANCY})
# causes that mean it may come back: park it as DORMANT with recovery conditions
PARKING_CAUSES = frozenset({FailureCause.TEMPORARY_INACTIVITY, FailureCause.REGIME_CHANGE, FailureCause.WRONG_CONTEXT})

# engine/pattern_lifecycle.STATES -> generic states (None = not yet a live item). The vocabulary is OWNED by pattern_lifecycle;
# this table only maps it, and `pattern_states_covered()` fails closed if that vocabulary gains or loses a state.
PATTERN_STATE_MAP = {"candidate": None, "rejected": State.RETIRED, "duplicate": State.RETIRED, "no_gain": State.RETIRED,
                     "active": State.ACTIVE, "watch": State.DEGRADED, "failed": State.DEGRADED, "cause_search": State.DEGRADED,
                     "rescoped": State.DEGRADED, "discarded": State.RETIRED}


def pattern_states_covered() -> list[str]:
    """Differences between pattern_lifecycle.STATES and the map (empty = in sync). Imported lazily so this module still loads
    while pattern_lifecycle is being edited by another builder."""
    from engine import pattern_lifecycle as PL
    return sorted(set(PL.STATES) ^ set(PATTERN_STATE_MAP))


def adapt_pattern_state(name: str) -> State | None:
    """Map a pattern_lifecycle state to the generic four-state view (no deletion: 'discarded' is RETIRED, kept)."""
    drift = pattern_states_covered()
    if drift:
        raise FirewallBreach(f"pattern_lifecycle vocabulary changed; retirement map out of date for {drift}")
    if name not in PATTERN_STATE_MAP:
        raise ValueError(f"unknown pattern lifecycle state {name!r}")
    return PATTERN_STATE_MAP[name]


# ------------------------------------------------------------------------------------------------- evidence

@dataclasses.dataclass(frozen=True)
class Evidence:
    """A window of NEW outcomes summarised for the gates. `effect` is signed so that positive = the item worked in the
    direction it claims; `se` its standard error. `window_start`/`window_end` are the first/last outcome dates used."""
    n: int
    effect: float
    se: float
    window_start: str
    window_end: str
    reliability: float | None = None       # current_reliability from section 11, if the caller has it
    source: str = ""

    def validate(self) -> list[str]:
        errs = []
        if self.n < 0:
            errs.append("n<0")
        if not (math.isfinite(self.effect) and math.isfinite(self.se)) or self.se < 0:
            errs.append("effect/se not finite or se<0")
        if self.reliability is not None and not 0.0 <= self.reliability <= 1.0:
            errs.append("reliability outside [0,1]")
        try:
            if as_date(self.window_start) > as_date(self.window_end):
                errs.append("window_start after window_end")
        except ValueError:
            errs.append("window dates unparsable")
        return errs

    @property
    def t(self) -> float:
        if self.n < 2:
            return 0.0
        if self.se <= 0:
            return math.copysign(math.inf, self.effect) if self.effect else 0.0
        return self.effect / self.se


def series_evidence(dates: Sequence[Any], values: Sequence[float], since, now, source: str = "") -> Evidence:
    """Summarise raw signed outcomes into Evidence using only dates in (since, now): strictly after the moment the item went
    inactive and strictly before `now`. Anything at/after `now` is ignored (it is not yet known), not an error, because a
    live caller legitimately holds a series that already contains today's row."""
    lo, hi = as_date(since), as_date(now)
    keep = [(as_date(d), float(v)) for d, v in zip(dates, values) if lo < as_date(d) < hi and math.isfinite(float(v))]
    if not keep:
        return Evidence(0, 0.0, 0.0, str(lo), str(lo), None, source)
    keep.sort()
    vals = [v for _, v in keep]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return Evidence(n, mean, math.sqrt(var / n) if n > 1 else 0.0, keep[0][0].isoformat(), keep[-1][0].isoformat(),
                    None, source)


# ------------------------------------------------------------------------------------------------- conditions

@dataclasses.dataclass(frozen=True)
class RecoveryCondition:
    """A learned condition under which a parked item is worth re-testing (section 14: 'conditions for recovery').
    Exactly one of `equals` or a (lo, hi) range is used; `key` names the entry of the caller's context snapshot."""
    key: str
    description: str = ""
    equals: str | None = None
    lo: float | None = None
    hi: float | None = None

    def validate(self) -> list[str]:
        errs = []
        if not self.key:
            errs.append("condition key empty")
        if self.equals is None and self.lo is None and self.hi is None:
            errs.append("condition has neither equals nor a range")
        if self.equals is not None and (self.lo is not None or self.hi is not None):
            errs.append("condition mixes equals and range")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            errs.append("lo>hi")
        return errs

    def satisfied_by(self, snapshot: Mapping[str, Any]) -> bool:
        if self.key not in snapshot or snapshot[self.key] is None:
            return False                                   # unknown context never satisfies a condition
        v = snapshot[self.key]
        if self.equals is not None:
            return str(v) == self.equals
        try:
            x = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(x):
            return False
        return (self.lo is None or x >= self.lo) and (self.hi is None or x <= self.hi)


# ------------------------------------------------------------------------------------------------- policy

@dataclasses.dataclass(frozen=True)
class RetirementPolicy:
    """Gate thresholds. They are conventions to be TUNED on real data in the validation wave; hysteresis is structural:
    recover_t > degrade_t so an item cannot flap across one boundary."""
    min_n: int = 20                        # below this no state change is ever made from the numbers alone
    degrade_t: float = 1.0                 # evidence t below this (or reliability below degrade_reliability) -> DEGRADED
    degrade_reliability: float = 0.4
    contradict_t: float = -2.0             # significantly reversed
    recover_t: float = 2.0                 # t needed for recovery evidence
    recover_min_n: int = 30
    revive_t_mult: float = 1.5             # RETIRED needs stricter evidence than DORMANT
    revive_n_mult: float = 2.0
    unstated_condition_t_bonus: float = 0.5  # recovery outside every stated condition needs extra t
    degraded_influence: float = 0.5        # influence multiplier while DEGRADED (scaled by reliability if given)
    grace_days: int = 60                   # DEGRADED this long while still bad -> DORMANT
    retire_after_days: int = 730           # DORMANT this long with no recovery attempt worth logging -> RETIRE proposal
    min_days_between: int = 5              # no two transitions of one item closer than this (anti-flapping)

    def validate(self) -> list[str]:
        errs = []
        if self.recover_t <= self.degrade_t:
            errs.append("recover_t must exceed degrade_t (hysteresis)")
        if self.recover_min_n < self.min_n:
            errs.append("recover_min_n below min_n")
        if not 0.0 <= self.degraded_influence <= 1.0:
            errs.append("degraded_influence outside [0,1]")
        if self.contradict_t >= 0:
            errs.append("contradict_t must be negative")
        if self.revive_t_mult < 1 or self.revive_n_mult < 1:
            errs.append("revival must not be easier than recovery")
        return errs


# ------------------------------------------------------------------------------------------------- ledger records

@dataclasses.dataclass(frozen=True)
class Transition:
    """One immutable move. `prev_hash` chains records so an edited history is detectable (`verify_chain`)."""
    seq: int
    knowledge_id: str
    kind: str
    from_state: str | None
    to_state: str
    at: str                                # decision date; effective for decisions strictly after it
    reason: str
    cause: str = FailureCause.UNKNOWN.value
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    conditions: tuple[RecoveryCondition, ...] = ()
    prev_hash: str = ""
    id: str = ""

    def body(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("id")
        return d

    def compute_id(self) -> str:
        return stable_hash(self.body(), 20)

    def to_json(self) -> str:
        return canonical_json(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Transition":
        d = json.loads(s)
        d["conditions"] = tuple(RecoveryCondition(**c) for c in d.get("conditions", ()))
        return cls(**d)


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What a gate concluded. `to_state is None` means 'no change', with the reason why (never silent)."""
    knowledge_id: str
    to_state: State | None
    kind: str | None
    reason: str
    numbers: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def changes(self) -> bool:
        return self.to_state is not None


# ------------------------------------------------------------------------------------------------- ledger

class RetirementLedger:
    """Append-only lifecycle ledger for many knowledge items."""

    def __init__(self, policy: RetirementPolicy | None = None):
        self.policy = policy or RetirementPolicy()
        bad = self.policy.validate()
        if bad:
            raise ValueError("invalid RetirementPolicy: " + "; ".join(bad))
        self._log: list[Transition] = []
        self._by_id: dict[str, list[int]] = {}

    # ---- basic access
    def __len__(self) -> int:
        return len(self._log)

    def known(self, kid: str) -> bool:
        return kid in self._by_id

    def history(self, kid: str) -> tuple[Transition, ...]:
        """The complete history, including retired items (retirement is never deletion)."""
        return tuple(self._log[i] for i in self._by_id.get(kid, ()))

    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._log)

    def _visible(self, kid: str, as_of) -> list[Transition]:
        cut = as_date(as_of)
        return [t for t in self.history(kid) if as_date(t.at) < cut]

    def state(self, kid: str, as_of) -> State | None:
        """State in force for decisions on `as_of`; None if the item was not yet registered then."""
        vis = self._visible(kid, as_of)
        return State(vis[-1].to_state) if vis else None

    def last_transition(self, kid: str, as_of) -> Transition | None:
        vis = self._visible(kid, as_of)
        return vis[-1] if vis else None

    def since(self, kid: str, as_of) -> str | None:
        t = self.last_transition(kid, as_of)
        return t.at if t else None

    def conditions(self, kid: str, as_of) -> tuple[RecoveryCondition, ...]:
        """Recovery conditions in force: the most recent non-empty set stated on a transition."""
        for t in reversed(self._visible(kid, as_of)):
            if t.conditions:
                return t.conditions
        return ()

    # ---- writes
    def _append(self, kid, kind, frm, to, at, reason, cause, evidence, conditions) -> Transition:
        prev = self._log[-1].id if self._log else ""
        t = Transition(len(self._log), kid, kind, frm.value if frm else None, to.value, as_date(at).isoformat(), reason,
                       cause.value, dict(evidence), tuple(conditions), prev)
        t = dataclasses.replace(t, id=t.compute_id())
        self._log.append(t)
        self._by_id.setdefault(kid, []).append(t.seq)
        return t

    def register(self, kid: str, now, state: State = State.ACTIVE, reason: str = "registered",
                 conditions: Sequence[RecoveryCondition] = ()) -> Transition:
        if self.known(kid):
            raise ValueError(f"{kid} already registered; a second registration would rewrite history")
        for c in conditions:
            if c.validate():
                raise ValueError(f"bad recovery condition {c}: {c.validate()}")
        return self._append(kid, "REGISTER", None, state, now, reason, FailureCause.UNKNOWN, {}, conditions)

    def transition(self, kid: str, to: State, now, kind: str, reason: str, cause: FailureCause = FailureCause.UNKNOWN,
                   evidence: Mapping[str, Any] | None = None, conditions: Sequence[RecoveryCondition] = ()) -> Transition:
        """The only door for state changes. Refuses illegal moves, unknown kinds, and anything dated before the last record."""
        if kind not in KINDS:
            raise ValueError(f"unknown transition kind {kind}")
        cur = self.state(kid, dt.date.max)
        if cur is None:
            raise KeyError(f"{kid} was never registered")
        if to not in ALLOWED[cur]:
            raise ValueError(f"{kid}: {cur} -> {to} is not an allowed move")
        last = self.history(kid)[-1]
        if as_date(now) < as_date(last.at):
            raise FirewallBreach(f"{kid}: transition dated {now} is before the previous record {last.at}")
        if (as_date(now) - as_date(last.at)).days < self.policy.min_days_between and last.kind != "REGISTER":
            raise ValueError(f"{kid}: transitions closer than {self.policy.min_days_between} days (flapping guard)")
        for c in conditions:
            if c.validate():
                raise ValueError(f"bad recovery condition {c}: {c.validate()}")
        return self._append(kid, kind, cur, to, now, reason, cause, evidence or {}, conditions)

    # ---- influence
    def influence(self, kid: str, now, reliability: float | None = None) -> float:
        """Multiplier in [0, 1] applied to a knowledge item's weight in a live decision. Unregistered items get 0:
        knowledge the ledger has never heard of does not silently influence anything."""
        s = self.state(kid, now)
        if s is None or s in (State.DORMANT, State.RETIRED):
            return 0.0
        if s is State.ACTIVE:
            return 1.0
        base = self.policy.degraded_influence
        return base if reliability is None else base * max(0.0, min(1.0, float(reliability)))

    def apply_influence(self, weights: Mapping[str, float], now) -> dict[str, float]:
        """Return weights with the ledger's multiplier applied; dormant/retired/unknown ids are dropped to 0.0 (kept as keys
        so a report shows what was silenced)."""
        return {k: float(w) * self.influence(k, now) for k, w in weights.items()}

    def assert_clean(self, weights: Mapping[str, float], now) -> None:
        """Firewall: a dormant or retired item must never carry live weight."""
        bad = [k for k, w in weights.items() if abs(float(w)) > 0 and self.influence(k, now) == 0.0]
        if bad:
            raise FirewallBreach(f"inactive knowledge carries live weight at {now}: {sorted(bad)[:8]}")

    # ---- gates
    def evaluate(self, kid: str, ev: Evidence, now, cause: FailureCause = FailureCause.UNKNOWN,
                 p_real: float | None = None, apply: bool = False,
                 conditions: Sequence[RecoveryCondition] = ()) -> Verdict:
        """Retirement gate (J05): should this item lose influence? Never acts on thin evidence; never lowers an item that is
        already at least as low as the numbers justify. With apply=True the verdict is written to the ledger."""
        pol, s = self.policy, self.state(kid, now)
        if s is None:
            return Verdict(kid, None, None, "unregistered")
        errs = ev.validate()
        if errs:
            raise ValueError("bad evidence: " + "; ".join(errs))
        require_past(ev.window_end, now, "retirement evidence window_end") if ev.n else None
        numbers = {"n": ev.n, "effect": ev.effect, "se": ev.se, "t": ev.t, "reliability": ev.reliability, "state": s.value,
                   "cause": cause.value, "p_real": p_real}
        last = self.last_transition(kid, now)
        age = (as_date(now) - as_date(last.at)).days if last else 0
        if s is State.RETIRED:
            return Verdict(kid, None, None, "already retired; only the revival gate can act", numbers)
        if ev.n < pol.min_n:
            return Verdict(kid, None, None, f"insufficient evidence (n={ev.n} < {pol.min_n})", numbers)
        bad_rel = ev.reliability is not None and ev.reliability < pol.degrade_reliability
        weak = ev.t < pol.degrade_t or bad_rel
        reversed_ = ev.t <= pol.contradict_t
        v: tuple[State, str, str] | None = None
        if cause in TERMINAL_CAUSES and (weak or reversed_):
            v = (State.RETIRED, "RETIRE", f"terminal cause {cause.value} with weak evidence")
        elif p_real is not None and p_real < 0.05 and reversed_:
            v = (State.RETIRED, "RETIRE", f"p_real={p_real:.3f} and evidence reversed")
        elif s is State.ACTIVE and (weak or reversed_):
            park = reversed_ and cause in PARKING_CAUSES
            v = (State.DORMANT, "DORMANT", f"reversed while {cause.value}: parked with recovery conditions") if park else \
                (State.DEGRADED, "DEGRADE", f"t={ev.t:.2f} < {pol.degrade_t} or reliability low")
        elif s is State.DEGRADED and (weak or reversed_) and last is not None and last.kind in ("DEGRADE", "REVERT") \
                and age >= pol.grace_days:
            v = (State.DORMANT, "DORMANT", f"still weak after {age} days degraded")
        elif s is State.DEGRADED and reversed_ and cause in PARKING_CAUSES:
            v = (State.DORMANT, "DORMANT", f"reversed while {cause.value}")
        elif s is State.DORMANT and age >= pol.retire_after_days and weak:
            v = (State.RETIRED, "RETIRE", f"dormant {age} days with no recovery")
        if v is None:
            return Verdict(kid, None, None, "no gate condition met", numbers)
        verdict = Verdict(kid, v[0], v[1], v[2], numbers)
        if apply:
            self.transition(kid, v[0], now, v[1], v[2], cause, numbers, conditions)
        return verdict

    def attempt_recovery(self, kid: str, ev: Evidence, now, context: Mapping[str, Any] | None = None,
                         apply: bool = False) -> Verdict:
        """Recovery gate (J06). DORMANT/RETIRED -> DEGRADED (probation) and DEGRADED-in-probation -> ACTIVE. Needs evidence
        that is (a) entirely after the item went inactive, (b) large enough, (c) strong enough, (d) consistent with a stated
        recovery condition or else held to a stricter bar. Failing the bar is a reasoned no-change, never an error."""
        pol, s = self.policy, self.state(kid, now)
        if s is None:
            return Verdict(kid, None, None, "unregistered")
        errs = ev.validate()
        if errs:
            raise ValueError("bad evidence: " + "; ".join(errs))
        numbers = {"n": ev.n, "effect": ev.effect, "se": ev.se, "t": ev.t, "state": s.value}
        if s is State.ACTIVE:
            return Verdict(kid, None, None, "already active", numbers)
        last = self.last_transition(kid, now)
        if ev.n:
            require_past(ev.window_end, now, "recovery evidence window_end")
        if last is not None and ev.n and as_date(ev.window_start) <= as_date(last.at):
            return Verdict(kid, None, None,
                           f"evidence starts {ev.window_start}, not after the item went inactive on {last.at}: "
                           "old evidence cannot recover it", numbers)
        need_n, need_t = pol.recover_min_n, pol.recover_t
        if s is State.RETIRED:
            need_n, need_t = math.ceil(need_n * pol.revive_n_mult), need_t * pol.revive_t_mult
        conds = self.conditions(kid, now)
        matched = [c.key for c in conds if context is not None and c.satisfied_by(context)]
        if conds and not matched:
            need_t += pol.unstated_condition_t_bonus
        numbers.update(need_n=need_n, need_t=need_t, conditions_matched=matched, conditions_stated=len(conds))
        if ev.n < need_n:
            return Verdict(kid, None, None, f"insufficient recovery evidence (n={ev.n} < {need_n})", numbers)
        if ev.t < need_t:
            return Verdict(kid, None, None, f"recovery evidence too weak (t={ev.t:.2f} < {need_t:.2f})", numbers)
        if s is State.DEGRADED:
            if last is None or last.kind not in ("RECOVER_PROBATION", "REVIVE"):
                return Verdict(kid, None, None, "degraded but not on recovery probation; use the retirement gate", numbers)
            out = (State.ACTIVE, "RECOVER_FULL", "second independent window confirmed recovery")
        elif s is State.DORMANT:
            out = (State.DEGRADED, "RECOVER_PROBATION", "recovery evidence accepted; on probation at reduced influence")
        else:
            out = (State.DEGRADED, "REVIVE", "revival evidence accepted; on probation at reduced influence")
        verdict = Verdict(kid, out[0], out[1], out[2], numbers)
        if apply:
            self.transition(kid, out[0], now, out[1], out[2], FailureCause.UNKNOWN, numbers)
        return verdict

    def probation_failed(self, kid: str, ev: Evidence, now, apply: bool = False) -> Verdict:
        """An item on recovery probation whose next window is weak again goes back to DORMANT (kind REVERT)."""
        s, last = self.state(kid, now), self.last_transition(kid, now)
        numbers = {"n": ev.n, "t": ev.t}
        if s is not State.DEGRADED or last is None or last.kind not in ("RECOVER_PROBATION", "REVIVE"):
            return Verdict(kid, None, None, "not on recovery probation", numbers)
        if ev.n < self.policy.min_n or ev.t >= self.policy.degrade_t:
            return Verdict(kid, None, None, "probation window not (yet) failed", numbers)
        verdict = Verdict(kid, State.DORMANT, "REVERT", f"probation failed (t={ev.t:.2f})", numbers)
        if apply:
            self.transition(kid, State.DORMANT, now, "REVERT", verdict.reason, FailureCause.UNKNOWN, numbers)
        return verdict

    # ---- views
    def states_at(self, now) -> dict[str, State]:
        out = {}
        for kid in self._by_id:
            s = self.state(kid, now)
            if s is not None:
                out[kid] = s
        return out

    def live_ids(self, now) -> tuple[str, ...]:
        return tuple(sorted(k for k, s in self.states_at(now).items() if s in (State.ACTIVE, State.DEGRADED)))

    def counts(self, now) -> dict[str, int]:
        c = {s.value: 0 for s in State}
        for s in self.states_at(now).values():
            c[s.value] += 1
        return c

    def health(self, now) -> Health:
        """Coarse portfolio-of-knowledge health for J11: mostly-inactive stores are flagged rather than hidden."""
        c = self.counts(now)
        total = sum(c.values())
        if total == 0:
            return Health.INSUFFICIENT_EVIDENCE
        live = (c["ACTIVE"] + c["DEGRADED"]) / total
        if c["ACTIVE"] == 0:
            return Health.BROKEN
        return Health.HEALTHY if live >= 0.6 and c["DEGRADED"] <= c["ACTIVE"] else Health.DEGRADING

    def dormant_review(self, now) -> list[str]:
        """Ids dormant longer than retire_after_days: candidates for a RETIRE decision (they are only listed here)."""
        out = []
        for kid, s in self.states_at(now).items():
            last = self.last_transition(kid, now)
            if s is State.DORMANT and last and (as_date(now) - as_date(last.at)).days >= self.policy.retire_after_days:
                out.append(kid)
        return sorted(out)

    def verify_chain(self) -> list[str]:
        """Tamper check: every id recomputes and every prev_hash matches the record before it."""
        errs, prev = [], ""
        for i, t in enumerate(self._log):
            if t.seq != i:
                errs.append(f"seq {t.seq} at position {i}")
            if t.prev_hash != prev:
                errs.append(f"record {i}: prev_hash mismatch")
            if t.id != t.compute_id():
                errs.append(f"record {i}: id does not match content")
            prev = t.id
        return errs

    # ---- persistence
    def dump(self, path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(t.to_json() for t in self._log) + ("\n" if self._log else ""), encoding="utf-8")
        return len(self._log)

    @classmethod
    def load(cls, path, policy: RetirementPolicy | None = None) -> "RetirementLedger":
        led = cls(policy)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = Transition.from_json(line)
                    led._log.append(t)
                    led._by_id.setdefault(t.knowledge_id, []).append(t.seq)
        errs = led.verify_chain()
        if errs:
            raise FirewallBreach("ledger failed integrity check: " + "; ".join(errs[:3]))
        return led

    def report(self, now) -> str:
        lines = [f"RETIREMENT LEDGER as of {as_date(now)} - IMPLEMENTED - NOT VALIDATED", f"counts: {self.counts(now)}",
                 f"health: {self.health(now)}"]
        for kid, s in sorted(self.states_at(now).items()):
            last = cast(Transition, self.last_transition(kid, now))
            lines.append(f"  {kid}: {s} since {last.at} ({last.kind}: {last.reason})")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- recovery index (B11)

@dataclasses.dataclass(frozen=True)
class RecoveryCandidate:
    knowledge_id: str
    state: str
    days_inactive: int
    matched: tuple[str, ...]
    stated: int
    priority: float


class RecoveryIndex:
    """B11: which parked items are worth re-testing now. Reads the ledger; changes nothing. An item is a candidate only when
    at least one stated recovery condition holds in the current context snapshot. Items with no stated condition are listed
    separately (`unconditioned`) because nothing tells us when to look, so they get periodic review, not context triggers."""

    def __init__(self, ledger: RetirementLedger):
        self.ledger = ledger

    def candidates(self, context: Mapping[str, Any], now) -> list[RecoveryCandidate]:
        out = []
        for kid, s in self.ledger.states_at(now).items():
            if s not in (State.DORMANT, State.RETIRED):
                continue
            conds = self.ledger.conditions(kid, now)
            matched = tuple(c.key for c in conds if c.satisfied_by(context))
            if not matched:
                continue
            last = cast(Transition, self.ledger.last_transition(kid, now))
            days = (as_date(now) - as_date(last.at)).days
            frac = len(matched) / len(conds)
            prio = frac * (0.5 if s is State.RETIRED else 1.0)
            out.append(RecoveryCandidate(kid, s.value, days, matched, len(conds), prio))
        return sorted(out, key=lambda c: (-c.priority, -c.days_inactive, c.knowledge_id))

    def unconditioned(self, now) -> list[str]:
        return sorted(k for k, s in self.ledger.states_at(now).items()
                      if s in (State.DORMANT, State.RETIRED) and not self.ledger.conditions(k, now))

    def by_condition_key(self, now) -> dict[str, list[str]]:
        """Inverse index: context key -> parked ids that would wake on it (so a regime change can query one key)."""
        inv: dict[str, list[str]] = {}
        for kid, s in self.ledger.states_at(now).items():
            if s in (State.DORMANT, State.RETIRED):
                for c in self.ledger.conditions(kid, now):
                    inv.setdefault(c.key, []).append(kid)
        return {k: sorted(v) for k, v in sorted(inv.items())}


def decision_effect_allowed(state: State, effect: DecisionEffect) -> bool:
    """Section 43 x 13: what a state may still influence. DEGRADED items may rank but may not size or exit;
    DORMANT and RETIRED items may only feed research priority (never a live decision)."""
    if state is State.ACTIVE:
        return True
    if state is State.DEGRADED:
        return effect in (DecisionEffect.RANKING, DecisionEffect.PATTERN_WEIGHTING, DecisionEffect.CONFIDENCE,
                          DecisionEffect.RESEARCH_PRIORITY, DecisionEffect.NONE)
    return effect in (DecisionEffect.RESEARCH_PRIORITY, DecisionEffect.NONE)


def batch_evaluate(ledger: RetirementLedger, evidence: Mapping[str, Evidence], now, causes: Mapping[str, FailureCause] | None = None,
                   apply: bool = False) -> list[Verdict]:
    """Run the retirement gate over many items in sorted-id order (deterministic) and return every verdict, including the
    no-change ones, so a report can say why each item stayed where it was."""
    causes = causes or {}
    return [ledger.evaluate(k, evidence[k], now, causes.get(k, FailureCause.UNKNOWN), apply=apply) for k in sorted(evidence)]


def states_summary(verdicts: Iterable[Verdict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in verdicts:
        key = v.to_state.value if v.to_state else "NO_CHANGE"
        out[key] = out.get(key, 0) + 1
    return out
