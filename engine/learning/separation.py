"""Selection / timing / direction / risk / exit separation (contract C62 section 23; checklist C09 blame assignment).

"Never let one failure teach the wrong subsystem." A stock can be correctly selected and badly timed; that loss must
teach TIMING, not "stock selection is bad". This module settles who is to blame NUMERICALLY, not by label:

  decompose()   walks one trade through a chain of counterfactual stages, each a stricter version of reality, so the
                per-subsystem contributions add up EXACTLY to (pnl - promise):
                    promise   = what selection said the stock would do (exp_move, right direction, ideal entry/exit)
                    selection = realised absolute move - promise            (did it move as much as promised?)
                    direction = side-adjusted move - absolute move          (0 if the side was right, -2*move if wrong)
                    timing    = side-adjusted fill-to-end - decision-to-end (the overnight gap: decisions fill next open)
                    exit      = realised exit - hold-to-horizon-end         (stops, targets, give-back)
                    risk      = stop slippage (gap through the stop)        (carved out of the exit stage)
                    cost      = -cost
                  plus a residual that must be ~0; a large residual is a measurement fault, reported not hidden.
  attribute()   turns the contributions into blame shares, names the primary subsystem, lists the subsystems that
                demonstrably did their job (they are PROTECTED: a loss may not teach them) and reconciles the numeric
                answer with the detectors' subsystem votes.
  route()       turns (cause, attribution) into per-subsystem teaching weights; a cause label that points at a
                protected subsystem is overridden by the arithmetic.
  SubsystemLedger + misattribution_audit() count how many lessons a naive "blame the decision-maker" rule would have
                taught the wrong subsystem, and whether direction "failures" are more than coin-flip noise.

Nothing here reads outcomes after `now`; it takes resolved TradeRecords. Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.core import FailureCause, Subsystem, clip01, require_past, stable_hash
from engine.learning.failure import Classification, TradeRecord, _f

FC = FailureCause
SUBSYSTEMS = tuple(Subsystem)
NL = chr(10)

# which subsystem a failure CAUSE normally teaches when the arithmetic does not object. Pattern/context/regime causes
# describe knowledge used by the decider, so they teach whoever decided (route() substitutes trade.decided_by).
CAUSE_SUBSYSTEM: dict[FailureCause, Subsystem | None] = {
    FC.SELECTION_ERROR: Subsystem.SELECTION, FC.TIMING_ERROR: Subsystem.TIMING, FC.RISK_ERROR: Subsystem.RISK,
    FC.MEASUREMENT_ERROR: None, FC.UNKNOWN: None, FC.INSUFFICIENT_EVIDENCE: None}
KNOWLEDGE_CAUSES = (FC.FALSE_PATTERN, FC.TEMPORARY_INACTIVITY, FC.WRONG_CONTEXT, FC.REGIME_CHANGE, FC.WEAKENING_EFFECT,
                    FC.REVERSAL, FC.REDUNDANCY, FC.INTERACTION_FAILURE)


@dataclass(frozen=True)
class SeparationParams:
    min_blame: float = 0.25              # a subsystem is taught only if it carries at least this share of the blame
    tol: float = 0.002                   # contributions smaller than this (fraction of position) are zero
    residual_tol: float = 0.01           # a decomposition residual above this is a measurement fault
    ambiguous_gap: float = 0.15          # top-two blame shares closer than this -> ambiguous
    vote_conflict: float = 0.3           # detectors' subsystem vote must exceed this to contradict the arithmetic
    min_direction_n: int = 30            # below this many direction calls, a hit rate says nothing
    noise_alpha: float = 0.10            # direction failures are 'noise' unless the hit rate differs from claim at this level
    boot: int = 500


# ==================================================================================================================
# additive decomposition
# ==================================================================================================================
@dataclass(frozen=True)
class Decomposition:
    rid: str
    promise: float
    pnl: float
    selection: float
    direction: float
    timing: float
    exit: float
    risk: float
    cost: float
    residual: float
    sizing_excess: float                  # extra PORTFOLIO-unit loss from oversizing: reported, not part of the sum
    selection_known: bool
    direction_owner: Subsystem
    notes: tuple[str, ...] = ()

    @property
    def parts(self) -> dict[str, float]:
        return {"selection": self.selection, "direction": self.direction, "timing": self.timing, "exit": self.exit,
                "risk": self.risk, "cost": self.cost}

    @property
    def total_vs_promise(self) -> float:
        return self.pnl - self.promise

    def checksum(self) -> float:
        """promise + parts + residual - pnl: identically zero by construction; a nonzero value is a code fault."""
        return self.promise + sum(self.parts.values()) + self.residual - self.pnl

    def contribution(self, s: Subsystem) -> float:
        """Signed contribution of a subsystem (negative = it cost money). Direction is routed to its owner: the
        direction model when one was used, otherwise whoever chose the side."""
        v = {Subsystem.SELECTION: self.selection, Subsystem.TIMING: self.timing, Subsystem.EXIT: self.exit,
             Subsystem.RISK: self.risk, Subsystem.DIRECTION: 0.0}[s]
        if s == self.direction_owner:
            v += self.direction
        return float(v)


def decompose(t: TradeRecord, p: SeparationParams | None = None, order: str = "selection_first") -> Decomposition:
    """Additive counterfactual decomposition of one trade. Works for wins too (a subsystem error rate needs the
    denominator of trades that went right)."""
    p = p or SeparationParams()
    t.require_valid()
    notes = []
    sig = _f(t.signal_ret)
    if sig is None:
        raise ValueError(f"trade {t.rid}: signal_ret is required to decompose")
    gap = _f(t.entry_gap)
    end = _f(t.end_ret_from_fill)
    if end is None:
        if gap is None:
            gap, end = 0.0, sig
            notes.append("entry_gap and end_ret_from_fill absent: timing stage assumed zero")
        else:
            end = (1 + sig) / (1 + gap) - 1
            notes.append("end_ret_from_fill derived from signal_ret and entry_gap")
    a = abs(sig)
    selection_known = _f(t.exp_move) is not None
    promise = float(t.exp_move) if selection_known else a
    v2 = t.side * sig
    if order == "selection_first":                        # movement first, then whether the side was right
        sel = a - promise
        dirn = v2 - a
    elif order == "direction_first":                      # side first (at the promised size), then how big the move was
        ideal = promise if v2 >= 0 else -promise
        dirn = ideal - promise
        sel = v2 - ideal
    else:
        raise ValueError(f"unknown chain order {order!r}")
    v3 = t.side * end
    timing = v3 - v2
    exit_ret = _f(t.exit_ret)
    gross_exit = t.side * exit_ret if exit_ret is not None else v3
    if exit_ret is None:
        notes.append("exit_ret absent: exit stage assumed to hold to the horizon end")
    exit_total = gross_exit - v3
    risk = 0.0
    if t.stop_hit and _f(t.stop) is not None and _f(t.stop_fill_ret) is not None:
        risk = min(0.0, float(t.stop_fill_ret) + float(t.stop))          # side-adjusted fill worse than the stop level
    exit_only = exit_total - risk
    cost = -float(t.cost)
    residual = float(t.pnl) - (promise + sel + dirn + timing + exit_only + risk + cost)
    if abs(residual) > p.residual_tol:
        notes.append(f"residual {residual:.4f} exceeds tolerance: the books do not reconcile (measurement fault)")
    sizing = 0.0
    if _f(t.weight) is not None and _f(t.target_weight) and t.target_weight > 0 and t.pnl < 0:
        sizing = float(t.pnl) * max(0.0, float(t.weight) / float(t.target_weight) - 1.0)
    owner = Subsystem.DIRECTION if _f(t.dir_prob) is not None else Subsystem.SELECTION
    return Decomposition(t.rid, promise, float(t.pnl), sel, dirn, timing, exit_only, risk, cost, residual, sizing,
                         selection_known, owner, tuple(notes))


# ==================================================================================================================
# blame assignment
# ==================================================================================================================
@dataclass(frozen=True)
class Attribution:
    rid: str
    blame: Mapping[str, float]            # Subsystem value -> share of the total NEGATIVE contribution (sums to 1)
    primary: Subsystem | None
    ambiguous: bool
    teach: tuple[Subsystem, ...]          # subsystems this loss may teach
    protected: tuple[Subsystem, ...]      # subsystems that did their job: must not be taught by this loss
    confidence: float
    basis: str                            # decomposition | reconciled | insufficient | no_loss
    conflict: str = ""                    # set when detectors' votes contradict the arithmetic
    magnitude: float = 0.0                # total negative contribution the blame is a share of

    def share(self, s: Subsystem) -> float:
        return float(self.blame.get(s.value, 0.0))


def blame_vector(d: Decomposition, p: SeparationParams) -> tuple[dict[str, float], float]:
    """Shares of the total negative contribution per subsystem (cost is never blamed on a subsystem)."""
    neg = {}
    for s in SUBSYSTEMS:
        c = d.contribution(s)
        if c < -p.tol:
            neg[s.value] = -c
    if d.sizing_excess < -p.tol:
        neg[Subsystem.RISK.value] = neg.get(Subsystem.RISK.value, 0.0) - d.sizing_excess
    tot = sum(neg.values())
    if tot <= 0:
        return {}, 0.0
    return {k: v / tot for k, v in neg.items()}, float(tot)


def attribute(t: TradeRecord, cls: Classification | None = None, p: SeparationParams | None = None,
              d: Decomposition | None = None, order_check: bool = True) -> Attribution:
    """Blame for one trade. `cls` (optional) is the classifier's output: its subsystem votes are compared with the
    arithmetic and a disagreement is reported, never silently resolved in favour of the label."""
    p = p or SeparationParams()
    d = d or decompose(t, p)
    if abs(d.residual) > p.residual_tol * 3:
        return Attribution(t.rid, {}, None, True, (), (), 0.0, "insufficient",
                           f"decomposition residual {d.residual:.4f}: books do not reconcile", 0.0)
    blame, mag = blame_vector(d, p)
    protected = tuple(s for s in SUBSYSTEMS if d.contribution(s) >= -p.tol)
    if not blame:
        return Attribution(t.rid, {}, None, False, (), protected, 0.0, "no_loss", "", 0.0)
    order = sorted(blame.items(), key=lambda kv: (-kv[1], kv[0]))
    primary = Subsystem(order[0][0])
    gap = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
    ambiguous = gap < p.ambiguous_gap and len(order) > 1
    teach = tuple(Subsystem(k) for k, v in order if v >= p.min_blame)
    conf = clip01(0.5 + 0.5 * gap) * (1.0 if not d.notes else 0.85)
    basis, conflict = "decomposition", ""
    if order_check and d.selection_known:
        alt_blame, _ = blame_vector(decompose(t, p, "direction_first"), p)
        if alt_blame and max(alt_blame.items(), key=lambda kv: (kv[1], kv[0]))[0] != primary.value:
            ambiguous, conf = True, conf * 0.8              # the answer depends on an arbitrary ordering of the chain
            conflict = "primary subsystem changes with the order of the decomposition chain"
    if cls is not None and cls.subsystem_votes:
        vote = max(cls.subsystem_votes.items(), key=lambda kv: kv[1])
        if vote[1] >= p.vote_conflict and Subsystem(vote[0]) != primary:
            conflict = f"detectors point at {vote[0]}, arithmetic at {primary.value}"
            basis, conf = "reconciled", conf * 0.7
            vs = Subsystem(vote[0])
            if vs not in teach and vs not in protected:
                teach = teach + (vs,)
        elif Subsystem(vote[0]) == primary:
            conf = clip01(conf + 0.1)
    teach = tuple(s for s in teach if s not in protected)
    return Attribution(t.rid, blame, primary, ambiguous, teach, protected, clip01(conf), basis, conflict, mag)


# ==================================================================================================================
# lesson routing: cause label + arithmetic -> who is taught, how much
# ==================================================================================================================
@dataclass(frozen=True)
class TeachingSignal:
    rid: str
    weights: Mapping[str, float]          # Subsystem value -> weight in [0,1]; only these may learn from this loss
    blocked: tuple[str, ...]              # subsystems explicitly barred (they were right)
    overridden: bool                      # the cause label pointed at a subsystem the arithmetic protects
    reason: str

    def weight(self, s: Subsystem) -> float:
        return float(self.weights.get(s.value, 0.0))


def route(t: TradeRecord, cls: Classification, att: Attribution) -> TeachingSignal:
    """Decide who learns from a classified loss. Rules, in order:
      1. an unnamed cause (UNKNOWN / INSUFFICIENT_EVIDENCE) teaches nobody except through the arithmetic, at half weight;
      2. a named cause maps to a subsystem (knowledge causes map to the decider);
      3. if the arithmetic PROTECTS that subsystem, the label is overridden and the arithmetic's teach-set is used;
      4. a measurement fault teaches no strategy subsystem at all."""
    protected = {s.value for s in att.protected}
    if cls.cause == FC.MEASUREMENT_ERROR:
        return TeachingSignal(t.rid, {}, tuple(sorted(protected | {s.value for s in SUBSYSTEMS})), False,
                              "measurement fault: repair the data, teach no strategy subsystem")
    if not cls.named:
        w = {s.value: 0.5 * att.blame.get(s.value, 0.0) for s in att.teach}
        return TeachingSignal(t.rid, w, tuple(sorted(protected)), False,
                              "no cause named; arithmetic-only teaching at half weight" if w else "no cause named and no blame")
    target = CAUSE_SUBSYSTEM.get(cls.cause) if cls.cause in CAUSE_SUBSYSTEM else t.decided_by
    if cls.cause in KNOWLEDGE_CAUSES:
        target = t.decided_by
    if target is None:
        return TeachingSignal(t.rid, {}, tuple(sorted(protected)), False, "cause has no subsystem owner")
    if target.value in protected:
        w = {s.value: att.blame.get(s.value, 0.0) for s in att.teach}
        return TeachingSignal(t.rid, w, tuple(sorted(protected)), True,
                              f"cause {cls.cause.value} points at {target.value}, which the arithmetic shows did its job")
    w = {target.value: clip01(cls.confidence)}
    for s in att.teach:
        if s != target:
            w[s.value] = 0.5 * att.blame.get(s.value, 0.0)
    return TeachingSignal(t.rid, w, tuple(sorted(protected)), False, f"cause {cls.cause.value} teaches {target.value}")


# ==================================================================================================================
# audits
# ==================================================================================================================
def naive_primary(t: TradeRecord) -> Subsystem:
    """What a system without separation would blame: whoever made the decision."""
    return t.decided_by


def misattribution_audit(trades: Sequence[TradeRecord], p: SeparationParams | None = None) -> dict[str, Any]:
    """Over the LOSING trades: how often would the naive rule have taught a different subsystem than the arithmetic,
    and how many were 'right idea, wrong timing/exit' (selection and direction fine, loss from entry or exit)."""
    p = p or SeparationParams()
    n = wrong = rightidea = 0
    cross: dict[str, dict[str, int]] = {}
    for t in trades:
        if t.pnl >= 0:
            continue
        d = decompose(t, p)
        a = attribute(t, None, p, d)
        if a.primary is None:
            continue
        n += 1
        nv = naive_primary(t)
        cross.setdefault(nv.value, {}).setdefault(a.primary.value, 0)
        cross[nv.value][a.primary.value] += 1
        if nv != a.primary:
            wrong += 1
        if d.selection >= -p.tol and d.direction >= -p.tol and (d.timing < -p.tol or d.exit + d.risk < -p.tol):
            rightidea += 1
    return {"n_losses": n, "misattributed": wrong, "misattribution_rate": wrong / n if n else float("nan"),
            "right_idea_wrong_execution": rightidea, "crosstab": cross}


def direction_noise_check(trades: Sequence[TradeRecord], p: SeparationParams | None = None) -> dict[str, Any]:
    """Is the direction model better than a coin? If its hit rate is indistinguishable from 50% (or from what its own
    probabilities promised), individual direction 'failures' are noise and must not be treated as lessons."""
    p = p or SeparationParams()
    hits, probs = [], []
    for t in trades:
        s, dp = _f(t.signal_ret), _f(t.dir_prob)
        if s is None or dp is None or abs(s) < 1e-9:
            continue
        went_up = s > 0
        hits.append(1.0 if (t.side > 0) == went_up else 0.0)
        probs.append(dp if t.side > 0 else 1.0 - dp)          # probability the model gave to ITS side
    n = len(hits)
    if n < p.min_direction_n:
        return {"n": n, "hit_rate": float("nan"), "verdict": "INSUFFICIENT_DATA", "noise": True}
    h = float(np.mean(hits))
    pr = float(np.mean(probs))
    se = math.sqrt(max(1e-12, 0.25 / n))
    z_vs_coin = (h - 0.5) / se
    z_vs_claim = (h - pr) / math.sqrt(max(1e-12, pr * (1 - pr) / n))
    from engine import pattern_stats as PS
    has_edge = z_vs_coin > 0 and PS.t_to_p(z_vs_coin) / 2.0 <= p.noise_alpha
    overclaims = z_vs_claim < 0 and PS.t_to_p(z_vs_claim) / 2.0 <= p.noise_alpha
    verdict = "EDGE" if has_edge and not overclaims else "OVERCLAIMS" if overclaims else "NO_EDGE"
    return {"n": n, "hit_rate": h, "claimed": pr, "z_vs_coin": float(z_vs_coin), "z_vs_claim": float(z_vs_claim),
            "verdict": verdict, "noise": not has_edge}


def counterfactual_pnl(d: Decomposition, fixed: Subsystem) -> float:
    """The pnl this trade would have booked had `fixed` done no harm (its negative contribution removed)."""
    c = d.contribution(fixed)
    return float(d.pnl - min(0.0, c))


# ==================================================================================================================
# ledger
# ==================================================================================================================
class SubsystemLedger:
    """Accumulates decompositions and attributions of ALL trades (wins included: error RATES need denominators)."""

    def __init__(self, params: SeparationParams | None = None):
        self.p = params or SeparationParams()
        self._d: list[Decomposition] = []
        self._a: list[Attribution] = []
        self._tags: list[Mapping[str, str]] = []
        self._ids: set[str] = set()

    def __len__(self):
        return len(self._d)

    def add(self, t: TradeRecord, cls: Classification | None = None) -> Attribution:
        if t.rid in self._ids:
            raise ValueError(f"trade {t.rid} already recorded: history is append-only")
        d = decompose(t, self.p)
        a = attribute(t, cls, self.p, d)
        self._ids.add(t.rid)
        self._d.append(d)
        self._a.append(a)
        self._tags.append(dict(t.tags))
        return a

    def error_rates(self) -> dict[str, float]:
        """Share of ALL trades in which each subsystem contributed a loss beyond the tolerance."""
        if not self._d:
            return {}
        return {s.value: float(np.mean([d.contribution(s) < -self.p.tol for d in self._d])) for s in SUBSYSTEMS}

    def mean_contribution(self) -> dict[str, float]:
        if not self._d:
            return {}
        return {s.value: float(np.mean([d.contribution(s) for d in self._d])) for s in SUBSYSTEMS}

    def blame_totals(self) -> dict[str, float]:
        """Sum of blame MAGNITUDE (return units) per subsystem over losing trades."""
        out = {s.value: 0.0 for s in SUBSYSTEMS}
        for a in self._a:
            for k, v in a.blame.items():
                out[k] += v * a.magnitude
        return out

    def blame_share_ci(self, seed: int = 0, level: float = 0.9) -> dict[str, tuple[float, float, float]]:
        """Bootstrap (over trades) CI of each subsystem's share of total blame. A share whose interval is wide or
        straddles its neighbours is not evidence about where to work."""
        rng = np.random.default_rng(seed)
        rows = [(a.blame, a.magnitude) for a in self._a if a.blame]
        if len(rows) < 5:
            return {}
        mat = np.array([[b.get(s.value, 0.0) * m for s in SUBSYSTEMS] for b, m in rows])
        draws = []
        for _ in range(self.p.boot):
            idx = rng.integers(0, len(mat), len(mat))
            tot = mat[idx].sum(axis=0)
            draws.append(tot / tot.sum() if tot.sum() > 0 else np.zeros(len(SUBSYSTEMS)))
        draws = np.array(draws)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        pt = mat.sum(axis=0)
        pt = pt / pt.sum() if pt.sum() > 0 else pt
        return {s.value: (float(pt[i]), float(np.quantile(draws[:, i], lo)), float(np.quantile(draws[:, i], hi)))
                for i, s in enumerate(SUBSYSTEMS)}

    def by_tag(self, tag: str) -> dict[str, dict[str, float]]:
        """Blame magnitude per subsystem within each value of a tag (era, winner type ...)."""
        out: dict[str, dict[str, float]] = {}
        for a, tg in zip(self._a, self._tags):
            bucket = out.setdefault(tg.get(tag, ""), {s.value: 0.0 for s in SUBSYSTEMS})
            for k, v in a.blame.items():
                bucket[k] += v * a.magnitude
        return out

    def taught_counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in SUBSYSTEMS}
        for a in self._a:
            for s in a.teach:
                out[s.value] += 1
        return out

    def report(self) -> str:
        lines = [f"subsystem ledger: {len(self)} trades   IMPLEMENTED - NOT VALIDATED"]
        er, mc, bt = self.error_rates(), self.mean_contribution(), self.blame_totals()
        for s in SUBSYSTEMS:
            lines.append(f"  {s.value:<10} error rate {er.get(s.value, 0):.3f}  mean contribution {mc.get(s.value, 0):+.4f}"
                         f"  blame {bt.get(s.value, 0):.4f}")
        return NL.join(lines)


def content_id(t: TradeRecord, d: Decomposition) -> str:
    """Stable id of a decomposition, for audit trails."""
    return stable_hash({"rid": t.rid, "parts": d.parts, "promise": d.promise, "pnl": d.pnl})


# ==================================================================================================================
# order sensitivity: is the blame an artefact of the chain order?
# ==================================================================================================================
def order_sensitivity(trades: Sequence[TradeRecord], p: SeparationParams | None = None) -> dict[str, Any]:
    """The decomposition chain has an order (movement before side, or side before movement). Blame that flips when the order
    flips is not a finding. Reports, over losing trades with a stated promise, how often the two orders agree on the
    primary subsystem and where they disagree."""
    p = p or SeparationParams()
    agree = n = 0
    flips: dict[str, int] = {}
    for t in trades:
        if t.pnl >= 0 or _f(t.exp_move) is None or _f(t.signal_ret) is None:
            continue
        b1, _ = blame_vector(decompose(t, p, "selection_first"), p)
        b2, _ = blame_vector(decompose(t, p, "direction_first"), p)
        if not b1 or not b2:
            continue
        n += 1
        p1 = max(b1.items(), key=lambda kv: (kv[1], kv[0]))[0]
        p2 = max(b2.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if p1 == p2:
            agree += 1
        else:
            key = f"{p1}->{p2}"
            flips[key] = flips.get(key, 0) + 1
    return {"n": n, "agreement": agree / n if n else float("nan"), "flips": dict(sorted(flips.items(), key=lambda kv: -kv[1]))}


# ==================================================================================================================
# per-subsystem diagnostics (each answers "is THIS subsystem the problem?" from its own numbers)
# ==================================================================================================================
def selection_report(trades: Sequence[TradeRecord], bins: int = 5) -> dict[str, Any]:
    """Selection promises movement. By rank bucket: mean realised/expected move and the shortfall rate; and the slope of
    realised on expected move (1.0 = calibrated promise, well below 1.0 = selection over-promises)."""
    rows = [(float(t.rank_pct), abs(float(t.signal_ret)), float(t.exp_move)) for t in trades
            if _f(t.rank_pct) is not None and _f(t.signal_ret) is not None and _f(t.exp_move)]
    if len(rows) < 2 * bins:
        return {"n": len(rows), "verdict": "INSUFFICIENT_DATA"}
    a = np.array(rows)
    edges = np.quantile(a[:, 0], np.linspace(0, 1, bins + 1))
    table = []
    for i in range(bins):
        m = (a[:, 0] >= edges[i]) & ((a[:, 0] < edges[i + 1]) if i < bins - 1 else (a[:, 0] <= edges[i + 1]))
        if m.any():
            ratio = a[m, 1] / a[m, 2]
            table.append({"bucket": i, "n": int(m.sum()), "mean_ratio": float(ratio.mean()), "shortfall_rate": float((ratio < 0.6).mean())})
    x, y = a[:, 2], a[:, 1]
    slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x, ddof=1) > 1e-12 else float("nan")
    ratio_all = y / x
    verdict = "OVERPROMISES" if ratio_all.mean() < 0.75 else "CALIBRATED" if ratio_all.mean() <= 1.25 else "UNDERPROMISES"
    top, bot = table[-1]["mean_ratio"], table[0]["mean_ratio"]
    return {"n": len(rows), "table": table, "slope": slope, "mean_ratio": float(ratio_all.mean()), "verdict": verdict,
            "ranking_informative": bool(top > bot + 0.05)}


def direction_calibration(trades: Sequence[TradeRecord], bins: int = 5) -> dict[str, Any]:
    """Reliability of the direction model's stated probability: bin P(own side) and compare with how often that side was
    right. Also the hit rate of the calls the model was most sure of (a model with no edge has no confident hits)."""
    rows = []
    for t in trades:
        s, dp = _f(t.signal_ret), _f(t.dir_prob)
        if s is None or dp is None or abs(s) < 1e-9:
            continue
        rows.append((dp if t.side > 0 else 1.0 - dp, 1.0 if t.side * s > 0 else 0.0))
    if len(rows) < 2 * bins:
        return {"n": len(rows), "verdict": "INSUFFICIENT_DATA"}
    a = np.array(rows)
    edges = np.quantile(a[:, 0], np.linspace(0, 1, bins + 1))
    table = []
    for i in range(bins):
        m = (a[:, 0] >= edges[i]) & ((a[:, 0] < edges[i + 1]) if i < bins - 1 else (a[:, 0] <= edges[i + 1]))
        if m.any():
            table.append({"bucket": i, "n": int(m.sum()), "mean_prob": float(a[m, 0].mean()), "hit_rate": float(a[m, 1].mean())})
    gap = float(np.mean([abs(r["mean_prob"] - r["hit_rate"]) * r["n"] for r in table]) * len(table) / len(a))
    conf = [r for r in table[-1:]]
    return {"n": len(rows), "table": table, "calibration_gap": gap, "top_bucket_hit_rate": conf[0]["hit_rate"] if conf else float("nan"),
            "verdict": "MISCALIBRATED" if gap > 0.08 else "CALIBRATED"}


def timing_policy_curve(trades: Sequence[TradeRecord], gaps: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.05)) -> list[dict[str, Any]]:
    """What a gap filter would have done: skip any entry whose overnight gap went against the position by more than g. For
    each g: entries skipped, the P&L of the skipped trades (negative = losses avoided), P&L kept, and the net change. This
    is a diagnostic for the TIMING subsystem, not a rule: the filter is tuned on nothing and applied to nothing."""
    rows = [t for t in trades if _f(t.entry_gap) is not None]
    out = []
    total = float(sum(t.pnl for t in rows))
    for g in gaps:
        skipped = [t for t in rows if t.side * float(t.entry_gap) > g]
        kept = [t for t in rows if t.side * float(t.entry_gap) <= g]
        sp = float(sum(t.pnl for t in skipped))
        out.append({"gap": g, "n": len(rows), "skipped": len(skipped), "skipped_pnl": sp, "kept_pnl": float(sum(t.pnl for t in kept)),
                    "net_change": -sp, "winners_skipped": sum(t.pnl > 0 for t in skipped),
                    "mean_kept": float(np.mean([t.pnl for t in kept])) if kept else float("nan"), "total": total})
    return out


def stop_effectiveness(trades: Sequence[TradeRecord], slip_tol: float = 0.25) -> dict[str, Any]:
    """Did the stops do their job? Among stopped-out trades: mean slippage beyond the stop level, the share that gapped
    through it, and what holding to the horizon end would have earned instead (stop saved / stop cost)."""
    hit = [t for t in trades if t.stop_hit and _f(t.stop) is not None and _f(t.stop_fill_ret) is not None]
    if not hit:
        return {"n": 0, "verdict": "NO_STOPS_HIT"}
    slip = np.array([max(0.0, -float(t.stop) - float(t.stop_fill_ret)) for t in hit])
    gapped = slip > slip_tol * np.array([float(t.stop) for t in hit])
    held = [t.side * float(t.end_ret_from_fill) - t.cost for t in hit if _f(t.end_ret_from_fill) is not None]
    actual = [float(t.pnl) for t in hit if _f(t.end_ret_from_fill) is not None]
    diff = float(np.mean(np.array(actual) - np.array(held))) if held else float("nan")
    return {"n": len(hit), "mean_slip": float(slip.mean()), "gapped_share": float(gapped.mean()), "max_slip": float(slip.max()),
            "mean_stop_vs_hold": diff, "verdict": "STOPS_LEAK" if gapped.mean() > 0.3 else "STOPS_HOLD",
            "stop_helped": bool(diff > 0) if diff == diff else None}


def exit_report(trades: Sequence[TradeRecord], mfe_min: float = 0.03) -> dict[str, Any]:
    """Exit quality: among trades that were ever well ahead (mfe >= mfe_min), how much was given back and how many
    round-tripped into a loss. A high round-trip share with the ENTRY fine is an exit problem, not a selection one."""
    ahead = [t for t in trades if _f(t.mfe) is not None and t.mfe >= mfe_min]
    if not ahead:
        return {"n_ahead": 0, "verdict": "NEVER_AHEAD"}
    give = np.array([float(t.mfe) - float(t.pnl) for t in ahead])
    rt = np.array([t.pnl < 0 for t in ahead])
    return {"n_ahead": len(ahead), "mean_giveback": float(give.mean()), "median_giveback": float(np.median(give)),
            "round_trip_share": float(rt.mean()), "verdict": "LEAKY_EXITS" if rt.mean() > 0.25 else "EXITS_OK"}


def risk_report(trades: Sequence[TradeRecord], risk_z: float = 2.5) -> dict[str, Any]:
    """Risk quality in the units the risk model uses: losses in expected-sigma units, the share beyond `risk_z` (a well
    calibrated model puts about 1% there), oversizing frequency, and the worst loss."""
    z = np.array([-float(t.pnl) / float(t.exp_vol) for t in trades if _f(t.exp_vol) and t.pnl < 0])
    over = [float(t.weight) / float(t.target_weight) - 1.0 for t in trades if _f(t.weight) is not None and _f(t.target_weight)]
    if len(z) == 0 and not over:
        return {"n": 0, "verdict": "INSUFFICIENT_DATA"}
    tail = float((z >= risk_z).mean()) if len(z) else float("nan")
    return {"n_losses": int(len(z)), "loss_sigma_p50": float(np.median(z)) if len(z) else float("nan"),
            "loss_sigma_p95": float(np.quantile(z, 0.95)) if len(z) else float("nan"), "tail_share": tail,
            "oversized_share": float(np.mean([o > 0.25 for o in over])) if over else float("nan"),
            "worst_loss": float(max(-float(t.pnl) for t in trades)) if trades else float("nan"),
            "verdict": "TAILS_TOO_FAT" if tail == tail and tail > 0.05 else "TAILS_OK"}


def subsystem_scorecard(trades: Sequence[TradeRecord], p: SeparationParams | None = None) -> dict[str, Any]:
    """All five diagnostics and the ledger's error rates in one structure - what each subsystem is answerable for."""
    p = p or SeparationParams()
    led = SubsystemLedger(p)
    for t in trades:
        if _f(t.signal_ret) is not None:
            led.add(t)
    return {"n": len(led), "error_rates": led.error_rates(), "mean_contribution": led.mean_contribution(),
            "selection": selection_report(trades), "direction": direction_calibration(trades), "direction_edge": direction_noise_check(trades, p),
            "timing": timing_policy_curve(trades), "exit": exit_report(trades), "risk": risk_report(trades), "stops": stop_effectiveness(trades)}


# ==================================================================================================================
# accumulating teaching signals (nobody learns from one loss)
# ==================================================================================================================
class TeachingLedger:
    """Accumulates TeachingSignals per subsystem with exponential decay in the ledger's own tick count (not a date), and the
    number of DISTINCT periods that contributed. A subsystem is 'ready to be examined' only when enough independent
    periods have taught it; readiness authorises a hypothesis test, never a parameter change."""

    def __init__(self, decay: float = 0.98):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.tick = 0
        self._w: dict[str, float] = {s.value: 0.0 for s in SUBSYSTEMS}
        self._n: dict[str, int] = {s.value: 0 for s in SUBSYSTEMS}
        self._periods: dict[str, set] = {s.value: set() for s in SUBSYSTEMS}
        self._blocked: dict[str, int] = {s.value: 0 for s in SUBSYSTEMS}
        self._overrides = 0

    def add(self, sig: TeachingSignal, period: str) -> None:
        self.tick += 1
        for k in self._w:
            self._w[k] *= self.decay
        for k, w in sig.weights.items():
            self._w[k] += float(w)
            self._n[k] += 1
            self._periods[k].add(period)
        for k in sig.blocked:
            if k in self._blocked:
                self._blocked[k] += 1
        self._overrides += int(sig.overridden)

    def weight(self, s: Subsystem) -> float:
        return self._w[s.value]

    def ready(self, min_periods: int = 6, min_weight: float = 3.0) -> list[Subsystem]:
        return [s for s in SUBSYSTEMS if len(self._periods[s.value]) >= min_periods and self._w[s.value] >= min_weight]

    def snapshot(self) -> dict[str, Any]:
        return {"tick": self.tick, "weight": dict(self._w), "n": dict(self._n), "periods": {k: len(v) for k, v in self._periods.items()},
                "blocked": dict(self._blocked), "label_overrides": self._overrides}


def rolling_blame(trades: Sequence[TradeRecord], window: int = 30, p: SeparationParams | None = None) -> list[dict[str, float]]:
    """Blame share per subsystem over a sliding window of the last `window` LOSING trades (input order = time order)."""
    p = p or SeparationParams()
    rows = []
    for t in trades:
        if t.pnl < 0 and _f(t.signal_ret) is not None:
            b, m = blame_vector(decompose(t, p), p)
            if b:
                rows.append({k: v * m for k, v in b.items()})
    out = []
    for i in range(window, len(rows) + 1):
        tot: dict[str, float] = {}
        for r in rows[i - window:i]:
            for k, v in r.items():
                tot[k] = tot.get(k, 0.0) + v
        z = sum(tot.values())
        out.append({k: v / z for k, v in tot.items()} if z > 0 else {})
    return out
