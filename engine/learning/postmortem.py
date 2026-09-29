"""Structured loss postmortems (contract C62 section 24; checklist D01/D10 output side, C09 blame).

For every meaningful loss the learner writes a Postmortem with all thirteen section-24 fields:
  what was believed / why / supporting evidence / contradicting evidence / what happened / what was surprising /
  which subsystem decided / which knowledge influenced it / which knowledge should NOT have / what condition was
  missing / what condition invalidated the belief / what should change / confidence in the explanation.

Hard rule of the section: NO POSTMORTEM MAY DIRECTLY MODIFY PRODUCTION BEHAVIOUR. What "should change" is therefore
emitted only as Hypothesis records - epistemic HYPOTHESIS, decision_effect declared, a written test plan, and
`production_effect` permanently False. `assert_hypothesis_only()` is called on every write path, and PostmortemStore is
the only sink: an append-only, hash-chained JSONL log that has no method able to touch parameters, weights or
knowledge. Hypotheses recur across losses; HypothesisBook merges them by content id and reports how many DISTINCT periods
support each, because ten losses in one week are one observation, not ten.

Future-knowledge check: knowledge whose provenance says it could not have existed at the decision date is a leak, not a
lesson; build_postmortem raises FirewallBreach (fail closed) instead of writing a comforting story about it.

Identity: nothing here stores a ticker or a date-keyed lookup. Records carry an opaque `rid` and salted period hash.
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing import Any, Iterable, Mapping, Sequence

from engine.learning.core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, Promotion, Provenance, Subsystem,
                                  Unknown, as_date, canonical_json, clip01, current_code_hash, require_past, stable_hash)
from engine.learning.failure import (Classification, Evidence, FailureEnv, Hypothesis, LossClassifier, TradeRecord,
                                     evaluate_contexts, regime_distance)
from engine.learning.separation import Attribution, SeparationParams, TeachingSignal, attribute, decompose, route

FC = FailureCause
NL = chr(10)
UNKNOWN_MARK = "UNKNOWN"
NONE_FOUND = "NONE_FOUND"

# the thirteen section-24 fields; Postmortem.validate() insists every one is populated (or explicitly UNKNOWN/NONE_FOUND)
REQUIRED_FIELDS = ("believed", "why_believed", "supporting_evidence", "contradicting_evidence", "what_happened", "surprise",
                   "deciding_subsystem", "knowledge_influencing", "knowledge_should_not_have", "missing_condition",
                   "invalidating_condition", "what_should_change", "confidence_in_explanation")


class ProductionWrite(Exception):
    """A postmortem tried to carry a production-changing effect. Raised, never swallowed."""


# ==================================================================================================================
# inputs describing the decision as it was made
# ==================================================================================================================
@dataclass(frozen=True)
class KnowledgeUse:
    """One piece of knowledge that fed the decision. `knowledge` is any KnowledgeLike (duck typed)."""
    knowledge: Any
    role: str = "pattern"                 # pattern | context_rule | lesson | prior | ...
    weight: float = 1.0                   # how much of the decision it carried
    direction: int = 1                    # +1 supported the trade, -1 argued against it


@dataclass(frozen=True)
class EvidenceItem:
    source: str                           # knowledge id or data source name
    statement: str
    strength: float = 0.5
    supports: bool = True

    def validate(self) -> list[str]:
        errs = []
        if not self.statement:
            errs.append("evidence statement empty")
        if not 0.0 <= self.strength <= 1.0:
            errs.append("evidence strength outside [0,1]")
        return errs


# ==================================================================================================================
# outputs
# ==================================================================================================================
@dataclass(frozen=True)
class Belief:
    statement: str
    exp_ret: float | None
    exp_move: float | None
    dir_prob: float | None
    side: int
    confidence: float | None              # the system's own stated confidence at the time (rank percentile proxy)


@dataclass(frozen=True)
class Misuse:
    knowledge_id: str
    reason: str
    severity: str                         # WARN | BLOCK
    fact: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Postmortem:
    pid: str
    rid: str
    period: str                           # salted hash of the decision year-week: counts distinct periods, cannot be looked up
    believed: Belief
    why_believed: tuple[str, ...]
    supporting_evidence: tuple[EvidenceItem, ...]
    contradicting_evidence: tuple[EvidenceItem, ...]
    what_happened: Mapping[str, Any]
    surprise: Mapping[str, Any]
    deciding_subsystem: Subsystem
    knowledge_influencing: tuple[str, ...]
    knowledge_should_not_have: tuple[Misuse, ...]
    missing_condition: tuple[str, ...]
    invalidating_condition: tuple[str, ...]
    what_should_change: tuple[Hypothesis, ...]
    confidence_in_explanation: float
    cause: FailureCause
    unknown_state: str
    attribution_primary: str
    teach: Mapping[str, float]
    provenance: Provenance
    params_hash: str = ""
    coverage: float = 0.0
    inputs_hash: str = ""                  # content hash of the TradeRecord this was built from (replay_check compares it)

    def validate(self) -> list[str]:
        errs = []
        named = self.cause not in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE)
        for name in REQUIRED_FIELDS:
            v = getattr(self, name)
            if name == "what_should_change" and not named:
                continue                                   # an unexplained loss may honestly propose nothing
            if v is None or (isinstance(v, (tuple, list, dict, str)) and len(v) == 0):
                errs.append(f"section-24 field {name} is empty (use an explicit {UNKNOWN_MARK}/{NONE_FOUND} marker)")
        if not 0.0 <= self.confidence_in_explanation <= 1.0:
            errs.append("confidence_in_explanation outside [0,1]")
        for h in self.what_should_change:
            errs += h.validate()
        for e in self.supporting_evidence + self.contradicting_evidence:
            errs += e.validate()
        errs += self.provenance.check()
        if self.cause in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE) and self.confidence_in_explanation > 0:
            errs.append("an unnamed cause must carry zero confidence in the explanation")
        return errs

    def to_dict(self) -> dict:
        return json.loads(canonical_json(self))

    def render(self) -> str:
        b = self.believed
        lines = [f"POSTMORTEM {self.pid}  cause={self.cause.value}  confidence={self.confidence_in_explanation:.2f}",
                 f"  believed:      {b.statement}",
                 f"  why:           {'; '.join(self.why_believed) or UNKNOWN_MARK}",
                 f"  happened:      " + ", ".join(f"{k}={v}" for k, v in self.what_happened.items()),
                 f"  surprise:      " + ", ".join(f"{k}={v}" for k, v in self.surprise.items()),
                 f"  decided by:    {self.deciding_subsystem.value}",
                 f"  influenced by: {', '.join(self.knowledge_influencing) or NONE_FOUND}"]
        for m in self.knowledge_should_not_have:
            lines.append(f"  SHOULD NOT HAVE ({m.severity}): {m.knowledge_id} - {m.reason}")
        lines.append(f"  missing:       {'; '.join(self.missing_condition)}")
        lines.append(f"  invalidated:   {'; '.join(self.invalidating_condition)}")
        for h in self.what_should_change:
            lines.append(f"  HYPOTHESIS [{h.subsystem.value}/{h.decision_effect.value}] {h.statement}")
        return NL.join(lines)


# ==================================================================================================================
# the analyses behind the fields
# ==================================================================================================================
def _kid(k: Any) -> str:
    return str(getattr(k, "knowledge_id", k))


def audit_influence(uses: Sequence[KnowledgeUse], t: TradeRecord, now, min_reliability: float = 0.3,
                    max_failure_risk: float = 0.7) -> list[Misuse]:
    """Knowledge that should not have influenced the decision. BLOCK = production should not have used it at all."""
    out: list[Misuse] = []
    for u in uses:
        k = u.knowledge
        kid = _kid(k)
        ep = getattr(k, "epistemic", None)
        if ep in (Epistemic.GATED, Epistemic.RETIRED, Epistemic.CONTRADICTED, Epistemic.UNKNOWN):
            out.append(Misuse(kid, f"epistemic state {ep} may not drive decisions", "BLOCK", {"epistemic": str(ep)}))
        elif ep in (Epistemic.HYPOTHESIS, Epistemic.OBSERVED):
            out.append(Misuse(kid, f"{ep} knowledge is not validated", "BLOCK", {"epistemic": str(ep)}))
        elif ep == Epistemic.DEGRADED:
            out.append(Misuse(kid, "knowledge was flagged DEGRADED at decision time", "WARN", {"epistemic": str(ep)}))
        pr = getattr(k, "promotion", None)
        eff = tuple(getattr(k, "decision_effect", ()) or ())
        if pr in (Promotion.RESEARCH, Promotion.SHADOW, Promotion.RETIRED) and eff and any(e != DecisionEffect.NONE for e in eff) \
                and u.weight > 0:
            out.append(Misuse(kid, f"promotion {pr} knowledge carried decision weight {u.weight:.2f}", "BLOCK",
                              {"promotion": str(pr), "weight": u.weight}))
        conf = getattr(k, "confidence", None)
        cr = getattr(conf, "current_reliability", None)
        fr = getattr(conf, "failure_risk", None)
        if cr is not None and cr < min_reliability:
            out.append(Misuse(kid, f"current reliability {cr:.2f} below {min_reliability}", "WARN", {"current_reliability": cr}))
        if fr is not None and fr > max_failure_risk:
            out.append(Misuse(kid, f"failure risk {fr:.2f} above {max_failure_risk}", "WARN", {"failure_risk": fr}))
        v = evaluate_contexts(k, t.context)
        if v.anti_hit:
            out.append(Misuse(kid, f"anti-context active: {', '.join(v.anti_hit)}", "BLOCK", {"anti_hit": v.anti_hit}))
        elif v.failed:
            out.append(Misuse(kid, f"outside its contexts: {', '.join(v.failed)}", "WARN", {"failed": v.failed}))
        prov = getattr(k, "provenance", None)
        if prov is not None and hasattr(prov, "could_exist_at") and not prov.could_exist_at(t.decided_at):
            raise FirewallBreach(f"knowledge {kid} could not have existed at {t.decided_at} "
                                 f"(learned_at={getattr(prov, 'learned_at', '?')}): future knowledge influenced the decision")
    return out


def missing_conditions(t: TradeRecord, env: FailureEnv, uses: Sequence[KnowledgeUse], z_flag: float = 2.0) -> list[str]:
    """Conditions the knowledge did not have but, in hindsight of this loss, arguably should. Three sources:
    dimensions a rule needed but the trade did not record, dimensions the rule ignored although they were abnormal,
    and dimensions the rule constrained wrongly."""
    out: list[str] = []
    constrained: set[str] = set()
    for u in uses:
        v = evaluate_contexts(u.knowledge, t.context)
        constrained |= set(dict(getattr(u.knowledge, "contexts", {}) or {})) | set(dict(getattr(u.knowledge, "anti_contexts", {}) or {}))
        for d in v.unknown:
            out.append(f"{_kid(u.knowledge)} needs context {d}, which was not recorded for this trade")
        for d in v.failed:
            out.append(f"{_kid(u.knowledge)}: context {d} was outside its stated range")
    ref = env.context_ref
    if ref is not None:
        _, _, zs = regime_distance(t.context, ref)
        for d, z in sorted(zs.items(), key=lambda kv: -abs(kv[1])):
            if abs(z) >= z_flag and d not in constrained:
                out.append(f"no rule conditions on {d}, which stood {z:+.1f} sd from its reference")
    return out


def invalidating_conditions(cls: Classification, t: TradeRecord) -> list[str]:
    """What broke the belief, read off the evidence that supported the named cause."""
    out: list[str] = []
    for e in cls.evidence:
        f = e.facts
        c = e.cause
        if c == FC.REVERSAL and "pattern" in f:
            out.append(f"pattern {f['pattern']} reversed: recent effect {f.get('recent_mean', float('nan')):+.4f} vs long-run {f.get('base_mean', float('nan')):+.4f}")
        elif c == FC.REVERSAL:
            out.append(f"prior trend {f.get('prior_ret', 0):+.3f} turned against the position")
        elif c == FC.WEAKENING_EFFECT:
            out.append(f"pattern {f.get('pattern')} decayed to {f.get('recent_mean', float('nan')):+.4f} (trend t={f.get('trend_t', float('nan')):.1f})")
        elif c == FC.TEMPORARY_INACTIVITY:
            out.append(f"pattern {f.get('pattern')} went flat (t={f.get('recent_t', float('nan')):.1f}); {f.get('prior_recoveries', 0)} earlier flat spells recovered")
        elif c == FC.FALSE_PATTERN:
            out.append(f"pattern {f.get('pattern')} never confirmed: p_real={f.get('p_real')}, t_conf={f.get('t_conf')}")
        elif c == FC.WRONG_CONTEXT:
            dims = f.get("anti_hit") or f.get("failed") or f.get("unknown_dims") or ()
            out.append(f"context violated for {f.get('knowledge')}: {', '.join(map(str, dims))}")
        elif c == FC.REGIME_CHANGE:
            out.append(f"market context {f.get('rms_z', float('nan')):.1f} sd from its reference on {', '.join(map(str, f.get('worst_dims', ())))}"
                       + (" and the regime label changed" if f.get("label_shift") else ""))
        elif c == FC.REDUNDANCY:
            out.append(f"patterns {f.get('pair')} are {f.get('corr', float('nan')):.2f} correlated: one signal counted twice")
        elif c == FC.INTERACTION_FAILURE:
            out.append(f"patterns {f.get('pair')} jointly earn {f.get('joint', float('nan')):+.4f} vs {f.get('single_min', float('nan')):+.4f} alone")
        elif c == FC.SELECTION_ERROR:
            out.append(f"the stock moved {f.get('realised_move', float('nan')):.3f} against a promised {f.get('exp_move', float('nan')):.3f}")
        elif c == FC.TIMING_ERROR:
            out.append("entry/exit timing: " + e.note)
        elif c == FC.RISK_ERROR:
            out.append("risk control failed: " + e.note)
        elif c == FC.MEASUREMENT_ERROR:
            out.append("data integrity: " + e.note)
    return out or [UNKNOWN_MARK]


# ==================================================================================================================
# hypothesis generation (D01 output; never a change)
# ==================================================================================================================
_PLAN_COMMON = ("validate on later periods than the losses that suggested it (out of sample)",
                "compare against a shuffled-label control (the effect must vanish)",
                "require support from many distinct periods and names, not one episode")


def make_hypothesis(cause: FailureCause, subsystem: Subsystem, t: TradeRecord, cls: Classification) -> Hypothesis | None:
    """Turn one classified loss into (at most) one hypothesis. hid depends on WHAT is proposed, not on which loss
    proposed it, so the same idea from different losses merges in HypothesisBook."""
    ev = cls.evidence
    pat = next((e.facts.get("pattern") for e in ev if "pattern" in e.facts), None)
    kn = next((e.facts.get("knowledge") for e in ev if "knowledge" in e.facts), None)
    pair = next((e.facts.get("pair") for e in ev if "pair" in e.facts), None)
    target = str(pat or kn or (pair and "+".join(pair)) or "")
    S = Subsystem
    D = DecisionEffect
    spec: dict[FailureCause, tuple[str, Subsystem, DecisionEffect, tuple[str, ...]]] = {
        FC.SELECTION_ERROR: ("require a minimum realised-movement expectation: demote picks whose expected move is below the promise",
                             S.SELECTION, D.RANKING, ("measure movement shortfall rate by rank decile",)),
        FC.TIMING_ERROR: ("test entering on a delay / skipping entries when the overnight gap is adverse, and widening stops that shake out",
                          S.TIMING if not any(e.subsystem == S.EXIT for e in ev) else S.EXIT, D.TIMING,
                          ("replay entries with a 1-session delay and gap filter on held-out periods",)),
        FC.RISK_ERROR: ("cap position size by expected volatility and treat stops as non-binding across gaps",
                        S.RISK, D.POSITION_SIZE, ("measure worst-loss and gap-through-stop frequency under the cap",)),
        FC.FALSE_PATTERN: (f"demote pattern {target or '?'} to CONTRADICTED pending re-test", S.SELECTION, D.PATTERN_WEIGHTING,
                           ("re-run its confirmation on data it has never seen",)),
        FC.WEAKENING_EFFECT: (f"reduce the weight of pattern {target or '?'} and re-estimate its effect with decay", S.SELECTION,
                              D.PATTERN_WEIGHTING, ("fit an effect-decay model and compare forecast error with the static effect",)),
        FC.REVERSAL: (f"gate pattern {target or '?'} while its recent effect is significantly negative and test the inverse",
                      S.SELECTION, D.PATTERN_WEIGHTING, ("check whether the inverse effect persists out of sample",)),
        FC.TEMPORARY_INACTIVITY: (f"do not retire pattern {target or '?'}: abstain from weighting it while its effect is flat",
                                  S.SELECTION, D.ABSTENTION, ("measure recovery after previous flat spells",)),
        FC.WRONG_CONTEXT: (f"enforce the context and anti-context conditions of {target or '?'} at decision time", S.SELECTION,
                           D.SELECTION, ("count decisions taken outside stated contexts and their mean result",)),
        FC.REGIME_CHANGE: ("abstain or shrink positions when market context is far from its reference", S.SELECTION, D.CONFIDENCE,
                           ("evaluate shrinkage as a function of context distance out of sample",)),
        FC.REDUNDANCY: (f"collapse patterns {target or '?'} into a single vote", S.SELECTION, D.PATTERN_WEIGHTING,
                        ("compare portfolio result with and without the duplicate vote",)),
        FC.INTERACTION_FAILURE: (f"penalise joint activation of {target or '?'} or model the interaction explicitly", S.SELECTION,
                                 D.RANKING, ("estimate the joint effect on later periods",)),
        FC.MEASUREMENT_ERROR: ("add a data-integrity rule that quarantines suspect bars before they reach features", S.SELECTION,
                               D.NONE, ("replay the quarantine rule over history and count wrongly-dropped good rows",)),
    }
    if cause in spec:
        text, sub, eff, plan = spec[cause]
    elif cause in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE):
        miss = sorted({f"{d}:{m}" for d, ms in cls.missing.items() for m in ms})
        if not miss and cause == FC.UNKNOWN:
            return None                                    # nothing missing and nothing found: honest silence
        text = "collect the missing inputs so that losses like this can be explained: " + (", ".join(miss[:6]) or "none identified")
        sub, eff, plan = subsystem, D.RESEARCH_PRIORITY, ("count how many later losses become classifiable with the new inputs",)
    else:
        return None
    hid = stable_hash({"cause": cause.value, "sub": sub.value, "eff": eff.value, "target": target, "text": text})
    return Hypothesis(hid, text, sub, eff, cause, plan + _PLAN_COMMON, target=target)


def assert_hypothesis_only(obj: Any) -> None:
    """Every write path calls this. A postmortem or hypothesis that claims a production effect is a bug to surface."""
    hyps: Iterable[Hypothesis]
    if isinstance(obj, Postmortem):
        hyps = obj.what_should_change
    elif isinstance(obj, Hypothesis):
        hyps = (obj,)
    else:
        raise TypeError(f"only Postmortem or Hypothesis may be written, got {type(obj).__name__}")
    for h in hyps:
        if h.production_effect or h.epistemic != Epistemic.HYPOTHESIS:
            raise ProductionWrite(f"hypothesis {h.hid} is not hypothesis-only (production_effect={h.production_effect}, epistemic={h.epistemic})")


# ==================================================================================================================
# builder
# ==================================================================================================================
def _period_hash(decided_at: str, salt: str) -> str:
    d = as_date(decided_at)
    y, w, _ = d.isocalendar()
    return stable_hash({"salt": salt, "y": y, "w": w}, 10)


def _belief(t: TradeRecord) -> Belief:
    parts = ["long" if t.side > 0 else "short"]
    if t.exp_move is not None:
        parts.append(f"expecting a {t.exp_move:.3f} move")
    if t.exp_ret is not None:
        parts.append(f"expected return {t.exp_ret:+.4f}")
    if t.dir_prob is not None:
        parts.append(f"P(up)={t.dir_prob:.2f}")
    if t.rank_pct is not None:
        parts.append(f"rank percentile {t.rank_pct:.2f}")
    return Belief(", ".join(parts), t.exp_ret, t.exp_move, t.dir_prob, t.side, t.rank_pct)


def _what_happened(t: TradeRecord) -> dict[str, Any]:
    out: dict[str, Any] = {"pnl": round(float(t.pnl), 5)}
    for name in ("signal_ret", "entry_gap", "end_ret_from_fill", "exit_ret", "mfe", "mae"):
        v = getattr(t, name)
        if v is not None:
            out[name] = round(float(v), 5)
    if t.stop_hit:
        out["stop_hit"] = True
    return out


def _surprise(t: TradeRecord, cls: Classification) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if t.exp_vol:
        out["loss_sigma"] = round(t.loss / float(t.exp_vol), 3)
    if cls.surprise_bits is not None:
        out["direction_bits"] = round(cls.surprise_bits, 3)
    if t.exp_ret is not None:
        out["shortfall_vs_expected"] = round(float(t.pnl) - float(t.exp_ret), 5)
    return out or {"note": "no expectation recorded: surprise cannot be measured"}


def build_postmortem(t: TradeRecord, cls: Classification, att: Attribution, env: FailureEnv | None, now,
                     uses: Sequence[KnowledgeUse] = (), evidence: Sequence[EvidenceItem] = (), salt: str = "pm",
                     code_hash: str | None = None, learned_at: str | None = None) -> Postmortem:
    """One postmortem for one classified loss. Fails closed: the trade must have resolved before `now`, and
    influencing knowledge that could not have existed at the decision date raises FirewallBreach."""
    t.require_valid()
    require_past(t.resolved_at, now, f"postmortem for {t.rid}")
    if not cls.meaningful:
        raise ValueError(f"trade {t.rid} is not a meaningful loss ({cls.note}): no postmortem is written")
    if cls.rid != t.rid or att.rid != t.rid:
        raise ValueError("classification / attribution belong to a different trade")
    env = env or FailureEnv()
    misuse = audit_influence(uses, t, now)
    sig = route(t, cls, att)
    evidence = list(evidence) or evidence_from_uses(uses, t)
    support = tuple(e for e in evidence if e.supports)
    against = tuple(e for e in evidence if not e.supports)
    why = tuple(f"{_kid(u.knowledge)} ({u.role}, weight {u.weight:.2f}, {'for' if u.direction > 0 else 'against'})" for u in uses)
    if not why:
        why = (f"{UNKNOWN_MARK}: no knowledge use was logged for this decision",)
    miss = missing_conditions(t, env, uses) or [NONE_FOUND if cls.named else UNKNOWN_MARK]
    inval = invalidating_conditions(cls, t) if cls.named else [UNKNOWN_MARK]
    subs = att.primary or t.decided_by
    hyps: list[Hypothesis] = []
    h0 = make_hypothesis(cls.cause, subs, t, cls)
    if h0 is not None:
        cfs = counterfactuals(t)
        if cfs and cfs[0].delta > 0.0005:                  # size of the lever on this trade, kept beside the idea
            h0 = dataclasses.replace(h0, basis={"best_counterfactual": cfs[0].name, "delta": round(cfs[0].delta, 5),
                                                "assumption": cfs[0].assumption})
        hyps.append(h0)
    for m in misuse:
        if m.severity == "BLOCK":
            hid = stable_hash({"gate": m.knowledge_id, "reason": m.reason.split(":")[0]})
            hyps.append(Hypothesis(hid, f"gate knowledge {m.knowledge_id} from production use: {m.reason}", t.decided_by,
                                   DecisionEffect.SELECTION, cls.cause, ("count past decisions that used it in this state",) + _PLAN_COMMON,
                                   target=m.knowledge_id))
    conf = clip01(cls.confidence * (0.5 + 0.5 * att.confidence)) if cls.named else 0.0
    prov = Provenance(created_real=str(now), learned_at=str(learned_at or t.resolved_at), code_hash=code_hash or current_code_hash(),
                      outcomes_seen_through=str(learned_at or t.resolved_at), parents=tuple(_kid(u.knowledge) for u in uses))
    pm = Postmortem(
        pid=stable_hash({"rid": t.rid, "cause": cls.cause.value, "params": cls.params_hash}, 12), rid=t.rid,
        period=_period_hash(t.decided_at, salt), believed=_belief(t), why_believed=why, supporting_evidence=support or (
            EvidenceItem("decision_log", "no supporting evidence was logged", 0.0, True),),
        contradicting_evidence=against or (EvidenceItem("decision_log", "no contradicting evidence was logged", 0.0, False),),
        what_happened=_what_happened(t), surprise=_surprise(t, cls), deciding_subsystem=t.decided_by,
        knowledge_influencing=tuple(sorted({_kid(u.knowledge) for u in uses})) or (NONE_FOUND,),
        knowledge_should_not_have=tuple(misuse) or (Misuse(NONE_FOUND, "no knowledge audit finding", "INFO"),),
        missing_condition=tuple(miss), invalidating_condition=tuple(inval),
        what_should_change=tuple(hyps), confidence_in_explanation=conf, cause=cls.cause,
        unknown_state=cls.unknown_state.value if cls.unknown_state else "", attribution_primary=att.primary.value if att.primary else "",
        teach=dict(sig.weights), provenance=prov, params_hash=cls.params_hash, coverage=cls.coverage, inputs_hash=stable_hash(t))
    errs = pm.validate()
    if errs:
        raise ValueError(f"postmortem for {t.rid} invalid: " + "; ".join(errs))
    assert_hypothesis_only(pm)
    return pm


class PostmortemBuilder:
    """classify -> decompose -> attribute -> postmortem, for a batch, in one call. Skips (and counts) non-meaningful trades."""

    def __init__(self, clf: LossClassifier | None = None, sep: SeparationParams | None = None, salt: str = "pm",
                 code_hash: str | None = None):
        self.clf = clf or LossClassifier()
        self.sep = sep or SeparationParams()
        self.salt = salt
        self.code_hash = code_hash or current_code_hash()
        self.skipped: dict[str, int] = {}

    def build(self, t: TradeRecord, env: FailureEnv | None, now, uses: Sequence[KnowledgeUse] = (),
              evidence: Sequence[EvidenceItem] = ()) -> Postmortem | None:
        cls = self.clf.classify(t, env, now)
        if not cls.meaningful:
            self.skipped[cls.note] = self.skipped.get(cls.note, 0) + 1
            return None
        att = attribute(t, cls, self.sep)
        return build_postmortem(t, cls, att, env, now, uses, evidence, self.salt, self.code_hash)


# ==================================================================================================================
# sink: append-only, hash-chained, hypothesis-only
# ==================================================================================================================
class PostmortemStore:
    """The only place postmortems go. There is deliberately no method that reads a hypothesis back into behaviour."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._rows: list[dict] = []
        self._pms: list[Postmortem] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._rows.append(json.loads(line))
            self.verify_chain()

    def __len__(self):
        return len(self._rows)

    def append(self, pm: Postmortem) -> str:
        assert_hypothesis_only(pm)
        if any(r["pid"] == pm.pid for r in self._rows):
            raise ValueError(f"postmortem {pm.pid} already stored: history is immutable")
        prev = self._rows[-1]["chain"] if self._rows else "genesis"
        body = pm.to_dict()
        chain = stable_hash({"prev": prev, "body": body}, 20)
        row = {"pid": pm.pid, "prev": prev, "chain": chain, "body": body}
        self._rows.append(row)
        self._pms.append(pm)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as f:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + NL)
        return chain

    def verify_chain(self) -> None:
        prev = "genesis"
        for r in self._rows:
            if r["prev"] != prev or r["chain"] != stable_hash({"prev": prev, "body": r["body"]}, 20):
                raise FirewallBreach(f"postmortem chain broken at {r['pid']}: history was edited")
            prev = r["chain"]

    def bodies(self) -> list[dict]:
        return [r["body"] for r in self._rows]

    def audit_identity(self, terms: Iterable[str]) -> list[str]:
        """Terms (tickers, dates) that appear in the serialised store: must be empty."""
        text = json.dumps(self._rows, sort_keys=True)
        return sorted({str(t) for t in terms if str(t) and str(t) in text})

    def field_coverage(self) -> dict[str, float]:
        """Share of stored postmortems whose section-24 field carries real content, not an UNKNOWN/NONE marker."""
        if not self._rows:
            return {f: 0.0 for f in REQUIRED_FIELDS}
        out = {}
        for f in REQUIRED_FIELDS:
            good = 0
            for r in self._rows:
                v = r["body"].get(f)
                s = json.dumps(v)
                if v not in (None, "", [], {}) and UNKNOWN_MARK not in s and NONE_FOUND not in s:
                    good += 1
            out[f] = good / len(self._rows)
        return out


# ==================================================================================================================
# hypotheses across losses
# ==================================================================================================================
@dataclass
class HypothesisRecord:
    hypothesis: Hypothesis
    postmortems: list[str] = field(default_factory=list)
    periods: set = field(default_factory=set)
    loss_sum: float = 0.0

    @property
    def n(self) -> int:
        return len(self.postmortems)


class HypothesisBook:
    """Merges recurring hypotheses. Support is counted in DISTINCT periods: a hypothesis proposed by 30 losses that all
    happened in one week has period support 1 and stays untested."""

    def __init__(self):
        self._h: dict[str, HypothesisRecord] = {}

    def add(self, pm: Postmortem, loss: float = 0.0) -> int:
        assert_hypothesis_only(pm)
        n = 0
        for h in pm.what_should_change:
            r = self._h.setdefault(h.hid, HypothesisRecord(h))
            if pm.pid in r.postmortems:
                continue
            r.postmortems.append(pm.pid)
            r.periods.add(pm.period)
            r.loss_sum += float(loss)
            n += 1
        return n

    def ready(self) -> list[HypothesisRecord]:
        """Hypotheses with enough independent support to be worth an out-of-sample test (not to be applied)."""
        out = [r for r in self._h.values() if r.n >= r.hypothesis.min_support and len(r.periods) >= r.hypothesis.min_periods]
        return sorted(out, key=lambda r: (-r.loss_sum, r.hypothesis.hid))

    def ranked(self) -> list[HypothesisRecord]:
        return sorted(self._h.values(), key=lambda r: (-len(r.periods), -r.n, r.hypothesis.hid))

    def support(self, hid: str) -> tuple[int, int]:
        r = self._h.get(hid)
        return (r.n, len(r.periods)) if r else (0, 0)

    def __len__(self):
        return len(self._h)

    def summary(self) -> str:
        lines = [f"hypothesis book: {len(self)} hypotheses, {len(self.ready())} ready for out-of-sample testing (none applied)"]
        for r in self.ranked()[:10]:
            lines.append(f"  {r.hypothesis.hid[:8]} n={r.n:<4} periods={len(r.periods):<3} [{r.hypothesis.subsystem.value}] {r.hypothesis.statement[:90]}")
        return NL.join(lines)


# bridge to engine.lessons (which already owns episode kinds); we suggest a kind, we do not create episodes
def lesson_kind(pm: Postmortem) -> str | None:
    from engine import lessons
    if pm.cause == FC.REGIME_CHANGE:
        k = "regime_misread"
    elif pm.cause == FC.RISK_ERROR:
        k = "oversized_loser"
    elif pm.cause == FC.TIMING_ERROR and pm.attribution_primary == Subsystem.EXIT.value:
        k = "missed_exit"
    elif pm.cause in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE, FC.MEASUREMENT_ERROR):
        return None
    else:
        k = "bad_entry"
    return k if k in lessons.KINDS else None


# ==================================================================================================================
# counterfactuals: what the loss would have been had one thing gone differently (numbers behind "what should change")
# ==================================================================================================================
@dataclass(frozen=True)
class Counterfactual:
    name: str
    pnl_if: float                         # position-return if the change had been made
    delta: float                          # pnl_if - actual pnl (positive = the change would have helped)
    assumption: str

    def validate(self) -> list[str]:
        return [] if self.assumption else ["counterfactual without a stated assumption"]


def counterfactuals(t: TradeRecord) -> list[Counterfactual]:
    """Single-change replays of one trade, best first. Each states what it assumes; none is evidence of a rule, only of
    how large the lever was on THIS trade (a hypothesis needs many trades, see HypothesisBook)."""
    out: list[Counterfactual] = []
    pnl = float(t.pnl)

    def add(name, val, why):
        out.append(Counterfactual(name, float(val), float(val) - pnl, why))
    add("skip_trade", 0.0, "no position was opened; nothing else changes")
    if t.exit_ret is not None and t.end_ret_from_fill is not None:
        add("hold_to_horizon_end", t.side * float(t.end_ret_from_fill) - t.cost, "no stop or target; the position runs to the horizon end")
    if t.exit_ret is not None:
        add("flip_side", -t.side * float(t.exit_ret) - t.cost, "same fills on the opposite side (only meaningful if the direction call was wrong)")
    if t.entry_gap is not None and abs(float(t.entry_gap)) > 0:
        add("no_adverse_gap", pnl + max(0.0, t.side * float(t.entry_gap)), "the fill price equalled the decision close")
    if t.stop_hit and t.stop is not None:
        add("stop_honoured", -float(t.stop) - t.cost, "the stop filled at its level despite the gap")
    if t.weight is not None and t.target_weight and t.weight > t.target_weight:
        add("sized_to_plan", pnl * float(t.target_weight) / float(t.weight), "position scaled to the planned weight (portfolio-unit effect shown in return terms)")
    if t.mfe is not None and t.mfe >= 0.03:
        add("take_half_profit_at_mfe", 0.5 * float(t.mfe) + 0.5 * pnl - t.cost, "half the position exited at the best excursion, the rest as it was")
    return sorted(out, key=lambda c: (-c.delta, c.name))


def evidence_from_uses(uses: Sequence[KnowledgeUse], t: TradeRecord) -> list[EvidenceItem]:
    """What each piece of knowledge said at the time, as evidence items: a use that argued for the trade and whose contexts
    held is support; one that argued against, or whose contexts did not hold, is contradiction the decision overrode."""
    out = []
    for u in uses:
        k, kid = u.knowledge, _kid(u.knowledge)
        conf = getattr(k, "confidence", None)
        rel = getattr(conf, "current_reliability", None)
        v = evaluate_contexts(k, t.context)
        strength = clip01(u.weight * (rel if rel is not None else 0.5))
        if u.direction > 0 and not v.anti_hit and not v.failed:
            out.append(EvidenceItem(kid, f"{u.role} {kid} argued for the trade (reliability {'n/a' if rel is None else format(rel, '.2f')})", strength, True))
        elif u.direction > 0:
            dims = ", ".join(v.anti_hit + v.failed)
            out.append(EvidenceItem(kid, f"{u.role} {kid} argued for the trade but its contexts were violated ({dims})", strength, False))
        else:
            out.append(EvidenceItem(kid, f"{u.role} {kid} argued against the trade and was overruled", strength, False))
    return out


# ==================================================================================================================
# audits and reconstruction
# ==================================================================================================================
def audit_postmortem(pm: Postmortem, t: TradeRecord | None = None) -> list[str]:
    """Internal-consistency problems of a postmortem (empty list = consistent). Catches stories that do not hang together,
    not stories that are wrong - that needs later data."""
    errs = list(pm.validate())
    named = pm.cause not in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE)
    if named and list(pm.invalidating_condition) == [UNKNOWN_MARK]:
        errs.append("a named cause must say which condition invalidated the belief")
    if not named:
        for h in pm.what_should_change:
            if h.decision_effect != DecisionEffect.RESEARCH_PRIORITY and h.source == "":
                if not h.statement.startswith("gate knowledge"):
                    errs.append(f"unexplained loss proposes a behaviour change ({h.hid}); only research or gating is allowed")
    for k, w in pm.teach.items():
        if k not in {s.value for s in Subsystem}:
            errs.append(f"teach names unknown subsystem {k!r}")
        if not 0.0 <= float(w) <= 1.0:
            errs.append(f"teach weight for {k} outside [0,1]")
    if pm.coverage < 0.5 and pm.confidence_in_explanation > 0.6:
        errs.append("confidence is high although fewer than half the detectors could run")
    if any(m.severity == "BLOCK" and m.knowledge_id != NONE_FOUND for m in pm.knowledge_should_not_have) and not any(
            "gate knowledge" in h.statement for h in pm.what_should_change):
        errs.append("blocked knowledge influenced the decision but no gating hypothesis was proposed")
    if t is not None:
        if pm.rid != t.rid:
            errs.append("postmortem belongs to a different trade")
        if pm.deciding_subsystem != t.decided_by:
            errs.append("deciding subsystem does not match the trade record")
        if abs(float(pm.what_happened.get("pnl", 0.0)) - float(t.pnl)) > 1e-4:
            errs.append("recorded pnl does not match the trade")
    return errs


def load_postmortem(d: Mapping[str, Any]) -> Postmortem:
    """Rebuild a Postmortem from Postmortem.to_dict() output (or a PostmortemStore body). Enums are re-parsed, so a stored
    record with an unknown cause or subsystem fails loudly instead of loading as text."""
    def hyp(h):
        return Hypothesis(h["hid"], h["statement"], Subsystem.parse(h["subsystem"]), DecisionEffect.parse(h["decision_effect"]),
                          FC.parse(h["cause"]), tuple(h["test_plan"]), int(h["min_support"]), int(h["min_periods"]),
                          Epistemic.parse(h["epistemic"]), bool(h["production_effect"]), h.get("target", ""), h.get("source", ""),
                          dict(h.get("basis", {})))
    pv = dict(d["provenance"])
    for k in ("sealed_windows", "parents"):
        pv[k] = tuple(pv.get(k, ()))
    return Postmortem(
        pid=d["pid"], rid=d["rid"], period=d["period"], believed=Belief(**d["believed"]), why_believed=tuple(d["why_believed"]),
        supporting_evidence=tuple(EvidenceItem(**e) for e in d["supporting_evidence"]),
        contradicting_evidence=tuple(EvidenceItem(**e) for e in d["contradicting_evidence"]),
        what_happened=dict(d["what_happened"]), surprise=dict(d["surprise"]), deciding_subsystem=Subsystem.parse(d["deciding_subsystem"]),
        knowledge_influencing=tuple(d["knowledge_influencing"]), knowledge_should_not_have=tuple(Misuse(**m) for m in d["knowledge_should_not_have"]),
        missing_condition=tuple(d["missing_condition"]), invalidating_condition=tuple(d["invalidating_condition"]),
        what_should_change=tuple(hyp(h) for h in d["what_should_change"]), confidence_in_explanation=float(d["confidence_in_explanation"]),
        cause=FC.parse(d["cause"]), unknown_state=d.get("unknown_state", ""), attribution_primary=d.get("attribution_primary", ""),
        teach=dict(d.get("teach", {})), provenance=Provenance(**pv), params_hash=d.get("params_hash", ""), coverage=float(d.get("coverage", 0.0)),
        inputs_hash=d.get("inputs_hash", ""))


# ==================================================================================================================
# knowledge involvement in losses (a first-order credit signal; hypothesis-generating only)
# ==================================================================================================================
class KnowledgeInvolvement:
    """For each knowledge id: how many decisions it influenced and how many of those lost. Needs the winners too, otherwise
    every knowledge item looks guilty. A knowledge item whose loss rate is significantly above the overall loss rate (BH
    corrected across all items) is flagged for a hypothesis - it is not blamed."""

    def __init__(self):
        self.uses: dict[str, int] = {}
        self.losses: dict[str, int] = {}
        self.total = 0
        self.total_losses = 0

    def add(self, uses: Sequence[KnowledgeUse], lost: bool) -> None:
        self.total += 1
        self.total_losses += int(lost)
        for kid in {_kid(u.knowledge) for u in uses}:
            self.uses[kid] = self.uses.get(kid, 0) + 1
            self.losses[kid] = self.losses.get(kid, 0) + int(lost)

    def table(self, min_n: int = 20, q: float = 0.10) -> list[dict[str, Any]]:
        from engine import pattern_stats as PS
        if self.total == 0 or self.total_losses in (0, self.total):
            return []
        p0 = self.total_losses / self.total
        rows = [k for k, n in self.uses.items() if n >= min_n]
        if not rows:
            return []
        z = np.array([(self.losses[k] / self.uses[k] - p0) / math.sqrt(p0 * (1 - p0) / self.uses[k]) for k in rows])
        pv = np.array([PS.t_to_p(v) / 2.0 if v > 0 else 1.0 - PS.t_to_p(v) / 2.0 for v in z])       # one-sided: worse than average
        qv = PS.bh_qvalues(pv)
        out = [{"knowledge": k, "n": self.uses[k], "loss_rate": self.losses[k] / self.uses[k], "base_rate": p0, "z": float(z[i]),
                "q": float(qv[i]), "flag": bool(qv[i] <= q and z[i] > 0)} for i, k in enumerate(rows)]
        return sorted(out, key=lambda r: (-r["z"], r["knowledge"]))


# ==================================================================================================================
# aggregate report over many postmortems
# ==================================================================================================================
def _norm(s: str) -> str:
    """Strip digits so 'moved 0.012 against 0.070' and 'moved 0.030 against 0.070' count as the same finding."""
    return "".join("#" if ch.isdigit() else ch for ch in str(s))[:90]


def summarize(bodies: Sequence[Mapping[str, Any]], top: int = 5) -> dict[str, Any]:
    """Aggregate view of stored postmortem bodies: causes, who decided/who to teach, most frequent invalidating conditions
    and missing conditions, most frequent knowledge flags, mean confidence per cause."""
    if not bodies:
        return {"n": 0}
    causes: dict[str, list[float]] = {}
    cross: dict[str, dict[str, int]] = {}
    inval: dict[str, int] = {}
    miss: dict[str, int] = {}
    flagged: dict[str, int] = {}
    for b in bodies:
        c = b["cause"]
        causes.setdefault(c, []).append(float(b["confidence_in_explanation"]))
        cross.setdefault(c, {}).setdefault(b.get("attribution_primary") or "-", 0)
        cross[c][b.get("attribution_primary") or "-"] += 1
        for s in b["invalidating_condition"]:
            if s != UNKNOWN_MARK:
                inval[_norm(s)] = inval.get(_norm(s), 0) + 1
        for s in b["missing_condition"]:
            if s not in (UNKNOWN_MARK, NONE_FOUND):
                miss[_norm(s)] = miss.get(_norm(s), 0) + 1
        for m in b["knowledge_should_not_have"]:
            if m["knowledge_id"] != NONE_FOUND:
                flagged[m["knowledge_id"]] = flagged.get(m["knowledge_id"], 0) + 1
    order = lambda dct: sorted(dct.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return {"n": len(bodies), "causes": {c: {"n": len(v), "mean_confidence": float(np.mean(v))} for c, v in sorted(causes.items())},
            "cause_by_primary_subsystem": cross, "top_invalidating": order(inval), "top_missing": order(miss),
            "top_flagged_knowledge": order(flagged),
            "unnamed_share": float(sum(1 for b in bodies if b["cause"] in ("UNKNOWN", "INSUFFICIENT_EVIDENCE")) / len(bodies))}


def render_summary(s: Mapping[str, Any]) -> str:
    if not s.get("n"):
        return "no postmortems   IMPLEMENTED - NOT VALIDATED"
    lines = [f"{s['n']} postmortems, unnamed share {s['unnamed_share']:.2f}   IMPLEMENTED - NOT VALIDATED"]
    for c, v in s["causes"].items():
        lines.append(f"  {c:<22}{v['n']:>4}  mean confidence {v['mean_confidence']:.2f}")
    for title, key in (("invalidating conditions", "top_invalidating"), ("missing conditions", "top_missing"), ("flagged knowledge", "top_flagged_knowledge")):
        if s[key]:
            lines.append(f"  {title}:")
            lines += [f"    {n:>3}x {txt}" for txt, n in s[key]]
    return NL.join(lines)


# ==================================================================================================================
# which losses deserve a postmortem, which failure signatures repeat, what happened to each hypothesis
# ==================================================================================================================
def select_for_postmortem(trades: Sequence[TradeRecord], budget: int, seen_signatures: Mapping[str, int] | None = None) -> list[TradeRecord]:
    """When there are more losses than review capacity, spend the budget on the biggest and most novel: priority = loss size
    times a novelty factor 1/(1+times a coarse signature was already seen). Deterministic tie-break by rid. Winners and
    sub-threshold losses are never selected; the caller's classifier still filters noise."""
    seen = dict(seen_signatures or {})
    scored = []
    for t in trades:
        if t.pnl >= 0:
            continue
        sig = f"{t.decided_by.value}|{'|'.join(sorted(t.pattern_ids))}|{'stop' if t.stop_hit else 'open'}"
        scored.append((-(t.loss / (1.0 + seen.get(sig, 0))), t.rid, t))
        seen[sig] = seen.get(sig, 0) + 1
    scored.sort(key=lambda r: (r[0], r[1]))
    return [t for _, _, t in scored[:max(0, budget)]]


def failure_signature(pm: Postmortem) -> str:
    """Identity-free fingerprint of HOW a loss happened: cause, blamed subsystem and the normalised invalidating condition."""
    return stable_hash({"c": pm.cause.value, "s": pm.attribution_primary, "i": sorted(_norm(x) for x in pm.invalidating_condition)}, 10)


def recurrence(pms: Sequence[Postmortem], min_periods: int = 3) -> list[dict[str, Any]]:
    """Failure signatures that repeat across DISTINCT periods, longest streak of consecutive occurrences (input order = time).
    A signature that recurs in many periods is a systematic failure mode; one that recurs in one period is one event."""
    by: dict[str, dict[str, Any]] = {}
    for pm in pms:
        r = by.setdefault(failure_signature(pm), {"signature": failure_signature(pm), "cause": pm.cause.value, "primary": pm.attribution_primary,
                                                  "n": 0, "periods": set(), "streak": 0, "_cur": 0})
        r["n"] += 1
        r["periods"].add(pm.period)
    order = [failure_signature(pm) for pm in pms]
    for sig, r in by.items():
        cur = best = 0
        for s in order:
            cur = cur + 1 if s == sig else 0
            best = max(best, cur)
        r["streak"] = best
    out = [{**{k: v for k, v in r.items() if k not in ("periods", "_cur")}, "n_periods": len(r["periods"])} for r in by.values()
           if len(r["periods"]) >= min_periods]
    return sorted(out, key=lambda r: (-r["n_periods"], -r["n"], r["signature"]))


HYPOTHESIS_STATES = ("PROPOSED", "TESTED_PASS", "TESTED_FAIL", "TESTED_INCONCLUSIVE")


class HypothesisOutcomes:
    """What happened to each hypothesis after it was tested elsewhere. Records verdicts only: nothing here applies a
    hypothesis. A failed hypothesis is kept (failed learners are knowledge, section 38) and blocks re-proposing the same
    idea until new evidence is cited."""

    def __init__(self):
        self._state: dict[str, str] = {}
        self._notes: dict[str, list[str]] = {}

    def propose(self, h: Hypothesis) -> bool:
        """False (and no change) if this idea already failed its test - re-proposal needs `reopen`."""
        if self._state.get(h.hid) == "TESTED_FAIL":
            return False
        self._state.setdefault(h.hid, "PROPOSED")
        return True

    def record(self, hid: str, verdict: str, note: str) -> None:
        if hid not in self._state:
            raise KeyError(f"hypothesis {hid} was never proposed")
        mapped = {"PASS": "TESTED_PASS", "FAIL": "TESTED_FAIL", "INSUFFICIENT": "TESTED_INCONCLUSIVE"}.get(verdict)
        if mapped is None:
            raise ValueError(f"unknown verdict {verdict!r}")
        if not note:
            raise ValueError("a test outcome must carry a note saying what was tested")
        self._state[hid] = mapped
        self._notes.setdefault(hid, []).append(f"{verdict}: {note}")

    def reopen(self, hid: str, new_evidence: str) -> None:
        if self._state.get(hid) != "TESTED_FAIL" or not new_evidence:
            raise ValueError("only a failed hypothesis with new evidence can be reopened")
        self._state[hid] = "PROPOSED"
        self._notes[hid].append(f"REOPENED: {new_evidence}")

    def state(self, hid: str) -> str | None:
        return self._state.get(hid)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in HYPOTHESIS_STATES}
        for s in self._state.values():
            out[s] += 1
        return out

    def history(self, hid: str) -> list[str]:
        return list(self._notes.get(hid, []))


def export_markdown(pm: Postmortem) -> str:
    """Postmortem as a review page: one heading per section-24 field, in contract order."""
    b = pm.believed
    changes = [f"- [{h.subsystem.value}/{h.decision_effect.value}] {h.statement}" for h in pm.what_should_change] or ["- nothing proposed"]
    parts = [f"# Postmortem {pm.pid}", f"cause **{pm.cause.value}**, confidence {pm.confidence_in_explanation:.2f}, IMPLEMENTED - NOT VALIDATED", "",
             "## What was believed", b.statement, "## Why", *[f"- {w}" for w in pm.why_believed],
             "## Supporting evidence", *[f"- {e.statement} ({e.strength:.2f})" for e in pm.supporting_evidence],
             "## Contradicting evidence", *[f"- {e.statement} ({e.strength:.2f})" for e in pm.contradicting_evidence],
             "## What happened", *[f"- {k}: {v}" for k, v in pm.what_happened.items()],
             "## What was surprising", *[f"- {k}: {v}" for k, v in pm.surprise.items()],
             "## Deciding subsystem", pm.deciding_subsystem.value,
             "## Knowledge that influenced it", *[f"- {k}" for k in pm.knowledge_influencing],
             "## Knowledge that should not have", *[f"- {m.knowledge_id} [{m.severity}]: {m.reason}" for m in pm.knowledge_should_not_have],
             "## Missing condition", *[f"- {m}" for m in pm.missing_condition],
             "## Condition that invalidated the belief", *[f"- {m}" for m in pm.invalidating_condition],
             "## What should change (hypotheses only)", *changes, "## Confidence in this explanation", f"{pm.confidence_in_explanation:.2f}"]
    return NL.join(parts)


# ==================================================================================================================
# reproducibility and coverage of the postmortem process itself
# ==================================================================================================================
def replay_check(pm: Postmortem, t: TradeRecord, env: FailureEnv | None, now, uses: Sequence[KnowledgeUse] = (),
                 clf: LossClassifier | None = None, code_hash: str | None = None) -> list[str]:
    """Rebuild the postmortem from the same inputs and compare the parts that must be deterministic (pid, cause, hypotheses,
    teaching weights). A stored postmortem that cannot be reproduced was produced by something other than this code."""
    again = PostmortemBuilder(clf, code_hash=code_hash or pm.provenance.code_hash).build(t, env, now, uses)
    if again is None:
        return ["the trade is no longer a meaningful loss under these parameters"]
    diffs = []
    if again.inputs_hash != pm.inputs_hash:
        diffs.append("the trade record differs from the one this postmortem was built from")
    if again.pid != pm.pid:
        diffs.append(f"pid {pm.pid} != {again.pid}")
    if again.cause != pm.cause:
        diffs.append(f"cause {pm.cause.value} != {again.cause.value}")
    if [h.hid for h in again.what_should_change] != [h.hid for h in pm.what_should_change]:
        diffs.append("hypotheses differ")
    if {k: round(v, 9) for k, v in again.teach.items()} != {k: round(float(v), 9) for k, v in pm.teach.items()}:
        diffs.append("teaching weights differ")
    return diffs


def loss_coverage(trades: Sequence[TradeRecord], pms: Sequence[Postmortem], bins: int = 4) -> dict[str, Any]:
    """Share of meaningful losses that received a postmortem, overall and by loss-size quartile. A process that reviews the
    small losses and skips the large ones is doing the wrong half of the work."""
    have = {pm.rid for pm in pms}
    losses = sorted((t for t in trades if t.pnl < 0), key=lambda t: t.pnl)
    if not losses:
        return {"n_losses": 0, "coverage": float("nan")}
    out: dict[str, Any] = {"n_losses": len(losses), "coverage": sum(t.rid in have for t in losses) / len(losses), "by_size": []}
    step = max(1, len(losses) // bins)
    for i in range(0, len(losses), step):
        chunk = losses[i:i + step]
        out["by_size"].append({"rank_from": i, "worst_loss": float(-chunk[0].pnl), "coverage": sum(t.rid in have for t in chunk) / len(chunk)})
    return out


_OPPOSED = [({"demote", "reduce", "gate", "collapse", "penalise"}, {"promote", "increase", "raise", "enforce"})]


def hypothesis_conflicts(book: "HypothesisBook") -> list[tuple[str, str, str]]:
    """Pairs of recorded hypotheses about the SAME target that push in opposite directions (e.g. 'gate P' and 'promote P').
    Both stay hypotheses; the conflict is reported so it is resolved by a test and not by whichever arrived last."""
    by_target: dict[str, list[Hypothesis]] = {}
    for r in book._h.values():
        if r.hypothesis.target:
            by_target.setdefault(r.hypothesis.target, []).append(r.hypothesis)
    out = []
    for tgt, hs in sorted(by_target.items()):
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                a, b = hs[i].statement.lower().split(), hs[j].statement.lower().split()
                for neg, pos in _OPPOSED:
                    if (neg & set(a) and pos & set(b)) or (pos & set(a) and neg & set(b)):
                        out.append((tgt, hs[i].hid, hs[j].hid))
    return out


def query_store(store: "PostmortemStore", cause: str | None = None, subsystem: str | None = None, min_confidence: float = 0.0) -> list[dict]:
    """Filter stored postmortem bodies by cause, blamed subsystem and confidence. Read-only view of the log."""
    out = []
    for b in store.bodies():
        if cause is not None and b["cause"] != cause:
            continue
        if subsystem is not None and b.get("attribution_primary") != subsystem:
            continue
        if float(b["confidence_in_explanation"]) < min_confidence:
            continue
        out.append(b)
    return out


def hypotheses_in_store(store: "PostmortemStore") -> dict[str, dict[str, Any]]:
    """Every distinct hypothesis proposed anywhere in the log, with how many postmortems and distinct periods proposed it."""
    acc: dict[str, dict[str, Any]] = {}
    for b in store.bodies():
        for h in b["what_should_change"]:
            r = acc.setdefault(h["hid"], {"statement": h["statement"], "subsystem": h["subsystem"], "effect": h["decision_effect"], "n": 0, "periods": set()})
            r["n"] += 1
            r["periods"].add(b["period"])
    return {k: {**{a: v for a, v in r.items() if a != "periods"}, "n_periods": len(r["periods"])} for k, r in acc.items()}


def knowledge_flag_history(store: "PostmortemStore") -> dict[str, dict[str, Any]]:
    """For each knowledge id ever flagged as 'should not have influenced': flags by severity, distinct periods, and the
    reasons given. Knowledge flagged BLOCK in many periods is being used in a state the system says it should not be."""
    acc: dict[str, dict[str, Any]] = {}
    for b in store.bodies():
        for m in b["knowledge_should_not_have"]:
            if m["knowledge_id"] == NONE_FOUND:
                continue
            r = acc.setdefault(m["knowledge_id"], {"BLOCK": 0, "WARN": 0, "periods": set(), "reasons": set()})
            r[m["severity"]] = r.get(m["severity"], 0) + 1
            r["periods"].add(b["period"])
            r["reasons"].add(_norm(m["reason"]))
    return {k: {"BLOCK": v["BLOCK"], "WARN": v["WARN"], "n_periods": len(v["periods"]), "reasons": sorted(v["reasons"])} for k, v in acc.items()}


def dedupe_hypotheses(hyps: Sequence[Hypothesis]) -> list[Hypothesis]:
    """Collapse hypotheses with the same content id, keeping the first; order of first appearance preserved."""
    seen: dict[str, Hypothesis] = {}
    for h in hyps:
        seen.setdefault(h.hid, h)
    return list(seen.values())


def period_counts(store: "PostmortemStore") -> dict[str, int]:
    """Postmortems per period hash: a burst in one period is one event, not a trend."""
    out: dict[str, int] = {}
    for b in store.bodies():
        out[b["period"]] = out.get(b["period"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def confidence_by_cause(store: "PostmortemStore") -> dict[str, float]:
    """Mean confidence in the explanation per cause; a cause that is always named with low confidence is under-evidenced."""
    acc: dict[str, list[float]] = {}
    for b in store.bodies():
        acc.setdefault(b["cause"], []).append(float(b["confidence_in_explanation"]))
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def hypothesis_effect_mix(store: "PostmortemStore") -> dict[str, dict[str, int]]:
    """Which decision effects the stored hypotheses target, per subsystem: is the learner asking for ranking changes, size
    changes, exits? A log that only ever asks for one kind of change is telling you something about the classifier."""
    out: dict[str, dict[str, int]] = {}
    for b in store.bodies():
        for h in b["what_should_change"]:
            row = out.setdefault(h["subsystem"], {})
            row[h["decision_effect"]] = row.get(h["decision_effect"], 0) + 1
    return out


def severity_counts(store: "PostmortemStore") -> dict[str, int]:
    """How many BLOCK / WARN knowledge findings the log holds."""
    out = {"BLOCK": 0, "WARN": 0}
    for b in store.bodies():
        for m in b["knowledge_should_not_have"]:
            if m["severity"] in out:
                out[m["severity"]] += 1
    return out
