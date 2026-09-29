"""Loop hooks: every learning module on the production path of the section-4 loop (contract C62 sections 4, 60, 61, 83, 85;
canon C58, C61, C64).  STATUS: IMPLEMENTED — NOT VALIDATED (C63: code and planted-world tests only; no real-data run).

The integration audit of 29 Sep found the loop real but partly open: 17 learning modules were reachable only from tests and 39
queued hooks had no production call site.  `LoopHooks` is owned by the learner (learner.py) and is called from the stage that
owns each mechanism, so the one production path (scripts/livesim_loop2.py --learner legit -> test_path.PathRunner -> learner)
now runs all of them:

  DECIDE             decision_contract.policy_check (the contract), calibration.combined_influence (weight), hierarchy
                     (shrinkage before an expectation is used), disagreement.conflict_score per decision, champion.audit_decision_sources
  ASSIGN CREDIT      a knowledge-level credit ledger -> credit.update_proposals -> BeliefLedger; credit.masked_pairs
  INVESTIGATE FAIL   knowledge.with_failure on the item that failed
  LEARN CONDITIONS   hierarchy.HierarchyBank updates, boundary.to_knowledge shelf, competition.boundary_field (+ complexity priors)
  UPDATE RELIABILITY calibration cache, break_detection.BreakEngine, reliability.contexts_from_condition, lifecycle machines ->
                     lifecycle.apply_to_ledger, health.inputs_from_knowledge -> HealthMonitor -> epistemic_proposals,
                     temporal.to_evidence -> retirement gate
  UPDATE GRAPH       credit.credit_edges, redundancy (masked pairs) -> RedundancyReport edges, knowledge.with_relation,
                     ContradictionMonitor.run_period (via KnowledgeGraph.contradiction_keys)
  UPDATE META        questions.QuestionEngine.ask_many, interpretation.next_test, unknowns.rank_unknowns, complexity budget,
                     disagreement.DisagreementEngine, portfolio_value.decompose_value, scorecard.scorecard_for_learner -> store
  SELECT RESEARCH    signals_from_health / data_audit / contradiction monitor / unknowns / interpretation; failed_learners
                     check_proposal -> research_priority.propose_selected -> in-loop prospective experiment -> update_from_result
  (trusted side)     RunGuard: checkpoints.resume_verified / write_interruption and compute.job_for around a test-path run;
                     run_same_year_harness: the same_year / controls harness as a callable entry.

Everything learned is written under the learner's work directory (`<workdir>/loop/`), so it survives the process.  All ids that
reach the research engine pass `safe_token`: a knowledge hash can contain a year-like digit run, which the identity firewall
(correctly) refuses in a research subject."""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import belief as BL
from . import boundary as BD
from . import break_detection as BK
from . import calibration as CB
from . import champion as CH
from . import checkpoints as CK
from . import competition as CP
from . import complexity as CX
from . import compute as CO
from . import contradiction_monitor as CM
from . import credit as CR
from . import decision_contract as DC
from . import disagreement as DG
from . import experiment_memory as EM
from . import failed_learners as FLR
from . import health as HE
from . import hierarchy as HI
from . import identity_firewall as IDF
from . import interpretation as IN
from . import knowledge as KN
from . import knowledge_graph as KG
from . import learning_curve as LCV
from . import lifecycle as LC
from . import portfolio_value as PV
from . import questions as QU
from . import redundancy as RD
from . import reliability as RL
from . import reports as RE
from . import research_priority as RPR
from . import retirement as RT
from . import scorecard as SC
from . import temporal as TP
from . import unknowns as UK
from .core import (DecisionEffect, Edge, Epistemic, FailureCause, FirewallBreach, Health, Lifecycle, Promotion, Subsystem, as_date,
                   canonical_json, require_past, stable_hash)

LABEL = "IMPLEMENTED — NOT VALIDATED"
_LETTERS = str.maketrans("0123456789", "ghijklmnop")
HIER_DIMS = (("market", "regime.label"), ("sector", "sector.family"), ("stock_type", "stock_type.kind"),
             ("volatility", "volatility.state"), ("interaction", "breadth.state"))
BREAK_COLUMNS = (("vix", "volatility", "market.vix", "volatility"), ("breadth", "breadth", "breadth.breadth", "market"))
# per research-signal kind: (outcome when the item holds, when it fails, when a context split explains it, when it is noise)
GENERIC_MAP = {
    "WEAK_PATTERN": ("supports_context_dependence", "supports_leading", "supports_context_dependence", "supports_noise"),
    "RELIABLE_DRIFT": ("supports_leading", "supports_context_dependence", "supports_context_dependence", "supports_noise"),
    "CONTRADICTION": ("inconclusive", "supports_noise", "supports_leading", "supports_noise"),
    "UNKNOWN_AREA": ("supports_leading", "supports_noise", "supports_context_dependence", "supports_noise"),
}
HOLDS = ("supports_leading", "context_split_explains", "supports_context_dependence")
CHAMPION_EPISTEMIC = (Epistemic.SUPPORTED, Epistemic.CONDITIONAL, Epistemic.DEGRADED)


def safe_token(token: str) -> str:
    """Digits to letters, one-to-one: a hash id with a year-like digit run must not reach the identity firewall of the research
    engine or a trader-visible path."""
    return str(token).translate(_LETTERS)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
    os.replace(tmp, path)


def _t(vals: Sequence[float]) -> tuple[float, float, float]:
    """(mean, standard error, t) of a sample; se=inf and t=0 below two observations."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0), float("inf"), 0.0
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    m = float(v.mean())
    return m, se, (m / se if se > 0 else 0.0)


@dataclass(frozen=True)
class HookConfig:
    """Cadences (in learning ticks) and bounds of every hooked mechanism; part of the learner's config hash."""
    every_health: int = 4
    every_break: int = 8
    every_competition: int = 12
    every_questions: int = 8
    every_interpretation: int = 12
    every_redundancy: int = 16
    every_disagreement: int = 12
    every_value: int = 26
    every_persist: int = 8
    every_kcredit: int = 8
    kcredit_top: int = 4
    kcredit_min: int = 40
    redundancy_overlap: float = 0.6
    max_rows: int = 20000
    hierarchy: bool = True
    exp_wait: int = 4
    exp_max_wait: int = 26
    health_min_hold: int = 2
    health_min_n: int = 20
    boot: int = 100
    contract_policy: DC.ContractPolicy = DC.ContractPolicy()
    lifecycle_window: int = 26
    persist: bool = True

    def validate(self) -> list[str]:
        errs = [f"contract policy: {e}" for e in self.contract_policy.check()]
        for n in ("every_health", "every_break", "every_competition", "every_questions", "every_interpretation", "every_redundancy",
                  "every_disagreement", "every_value", "every_persist", "every_kcredit"):
            if getattr(self, n) < 1:
                errs.append(f"{n} < 1")
        if not 2 <= self.kcredit_top <= CR.EXACT_LIMIT - 1:
            errs.append(f"kcredit_top must lie in [2, {CR.EXACT_LIMIT - 1}] (exact Shapley, one slot kept for the rest)")
        if not 0 < self.redundancy_overlap <= 1:
            errs.append("redundancy_overlap outside (0, 1]")
        if self.exp_wait < 2 or self.exp_max_wait < self.exp_wait:
            errs.append("exp_wait >= 2 and exp_max_wait >= exp_wait")
        if self.health_min_hold < 1 or self.boot < 20 or self.lifecycle_window < 4 or self.max_rows < 100:
            errs.append("health_min_hold >= 1, boot >= 20, lifecycle_window >= 4, max_rows >= 100")
        return errs


@dataclass(frozen=True)
class ResolvedRow:
    """One matured decision row as the hooks keep it: identity-free (an opaque ident only) and dated by the feed's own clock."""
    decided: str
    matured: str
    ident: str
    raw_ret: float
    edge: float
    expected: float | None
    long: bool
    members: frozenset
    ctx: str
    vix: float
    breadth: float
    hier: tuple
    stance: tuple                             # ((kid, weight * sign(expected)), ...) of the items that carried the decision


@dataclass
class OpenExperiment:
    experiment_id: str
    cid: str
    predicted_bits: float
    created: str
    kind: str
    subjects: tuple


class _CalibrationCache:
    """Duck-typed monitor for calibration.combined_influence: the per-item assessment computed on the learning cadence, served to
    every later decision without recomputing the ECE null each time.  Only assessments computed at or before `now` are served, and
    each one read only records that matured strictly before its own date."""

    def __init__(self):
        self._by: dict[str, list] = {}

    def put(self, kid: str, as_of: str, assessment) -> None:
        self._by.setdefault(kid, []).append((as_of, assessment))

    def assess(self, now, seed: int, knowledge_id: str | None = None):
        rows = [a for d, a in self._by.get(str(knowledge_id), ()) if as_date(d) <= as_date(now)]
        return rows[-1] if rows else _NEUTRAL

    def __len__(self) -> int:
        return sum(len(v) for v in self._by.values())


@dataclass(frozen=True)
class _Neutral:
    influence: float = 1.0
    health: str = "INSUFFICIENT_EVIDENCE"


_NEUTRAL = _Neutral()


class LoopHooks:
    """Every hooked mechanism of the section-4 loop.  One instance per learner; see the module docstring for which stage calls what."""

    def __init__(self, learner, cfg: HookConfig | None = None, root: str | Path | None = None):
        self.L = learner
        self.cfg = cfg or HookConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid HookConfig: " + "; ".join(errs))
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        seed = learner.cfg.seed
        self.seed = seed
        self.fired: dict[str, int] = {}
        self.rows: list[ResolvedRow] = []
        self.safe: dict[str, str] = {}
        self.cal = _CalibrationCache()
        self.bank = HI.HierarchyBank()
        self._hier_cache: dict[tuple, Any] = {}
        self.boundary_shelf: dict[str, Any] = {}
        self.arenas: dict[str, dict] = {}
        self.breaks = BK.BreakEngine({"n_perm": 60, "boot": 60}, seed)
        self.break_applied: dict[str, str] = {}
        self.machines: dict[str, LC.LifecycleMachine] = {}
        self._lc_cursor: dict[str, int] = {}
        self.lifecycle_verdicts: list[dict] = []
        self.health = HE.HealthMonitor({"min_n": self.cfg.health_min_n})
        self.health_applied: list[dict] = []
        self.health_rows: list[dict] = []
        self.retire_verdicts: list[dict] = []
        self.kcredit_rows: list[tuple] = []
        self.kcredit_reports: list = []
        self.credit_seen: set[str] = set()
        self.proposals_applied: list[dict] = []
        self.masked: list = []
        self.redundancy_last: RD.RedundancyReport | None = None
        self.relations: set[tuple] = set()
        self.contra_monitor = CM.ContradictionMonitor(learner.graph)
        self.contra_last: CM.MonitorReport | None = None
        self.contra_signals: list = []
        self.contra_keys = 0
        self.answers = QU.AnswerLedger()
        self.qengine = QU.QuestionEngine()
        self.interpretations: dict[str, Any] = {}
        self.next_tests: dict[str, tuple] = {}
        self.unknowns = UK.UnknownLedger()
        self.unknown_rank: list = []
        self.complexity: dict[str, dict] = {}
        self.conflicts: dict[str, dict] = {}
        self.disagreement_last: DG.DisagreementReport | None = None
        self.value_last: PV.ValueDecomposition | None = None
        self.scorecards = SC.ScorecardStore(self.root / "scorecards.jsonl") if self.root is not None else None
        self.scorecard_ids: list[str] = []
        self.valid_card: SC.LearningScorecard | None = None
        self.staged = BL.BeliefLedger()
        self.staged_rows: list[dict] = []
        self.claims: dict[str, dict] = {}
        self.scorecard_refusals: list[dict] = []
        self.gate_warnings: dict[str, str] = {}
        self.failed = FLR.FailedLearnerRegistry(self.root / "failed_learners.jsonl" if self.root is not None else None)
        self.failed_seed = FLR.seed_registry(self.failed)
        self.dead_ends: list[dict] = []
        self.experiments = EM.ExperimentLedger(self.root / "experiments.jsonl" if self.root is not None else None)
        self.open_exps: dict[str, OpenExperiment] = {}
        self.exp_results: list[dict] = []
        self.readiness_log: list[dict] = []
        self.source_audits: list[dict] = []
        self.dashboard: dict | None = None
        self.persisted: dict[str, Any] = {}
        self._pending_signals: list = []

    # ------------------------------------------------------------------------------------------------ bookkeeping
    def _fire(self, name: str, n: int = 1) -> None:
        self.fired[name] = self.fired.get(name, 0) + n

    def token(self, kid: str) -> str:
        s = safe_token(kid)
        self.safe[s] = kid
        return s

    def _due(self, every: int) -> bool:
        return self.L._tick % every == 0

    def _kids(self) -> list[str]:
        return sorted(self.L._pid_of)

    def _learned(self, ep) -> str:
        return max(r.matured for r in ep.rows)

    def _signed_weekly(self, kid: str) -> tuple[list, list]:
        """(matured dates, signed weekly edges) of the item, one value per date (two episodes maturing on one date are averaged),
        oldest first: every consumer below needs a strictly increasing index."""
        s = self._series(kid)
        return [str(d.date()) for d in s.index], [float(v) for v in s.values]

    def _series(self, kid: str) -> pd.Series:
        pid = self.L._pid_of[kid]
        d, v = self.L._signed(pid, self.L._direction_of(kid))
        if not d:
            return pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        s = pd.Series(v, index=pd.DatetimeIndex([pd.Timestamp(x) for x in d]), dtype=float)
        return s.groupby(level=0).mean().sort_index()

    # ------------------------------------------------------------------------------------------------ DECIDE side
    def contract_allows(self, k, now) -> bool:
        """The decision contract under the configured policy (never looser than decision_contract.check)."""
        self._fire("policy_check")
        return DC.policy_check(k, now, policy=self.cfg.contract_policy).allowed

    def influence(self, kid: str, now, ctx_now: Mapping[str, Any] | None, reliability: float | None) -> float:
        """lifecycle x calibration x temporal (calibration.combined_influence) for a registered item."""
        prof = self.L.temporal.get(kid, now)
        self._fire("combined_influence")
        return CB.combined_influence(kid, now, self.L.retirement, self.cal, prof, ctx_now, self.seed, reliability).total  # type: ignore[arg-type]  # duck-typed monitor

    def _hier_context(self, sit) -> dict:
        b = sit.bins()
        return {dim: b.get(path, "na") for dim, path in HIER_DIMS}

    def shrunk_expectation(self, kid: str, sit) -> tuple[float, float] | None:
        """Hierarchy shrinkage before an expectation is used: the item's effect in this context, pulled toward its parents until
        the context has earned its own estimate.  None when the item has no hierarchy yet or the estimate is UNKNOWN."""
        if not self.cfg.hierarchy or str(kid) not in self.bank._h:
            return None
        ctx = self._hier_context(sit)
        key = (kid, tuple(sorted(ctx.items())), self.bank._h[str(kid)].rows_seen)
        if key not in self._hier_cache:
            if len(self._hier_cache) > 4096:
                self._hier_cache.clear()
            est = self.bank.get(kid).estimate(ctx)
            self._hier_cache[key] = (float(est.mean or 0.0), float(est.se or 0.0)) if est.is_known() else None
            self._fire("hierarchy_estimate")
        return self._hier_cache[key]

    def on_decide(self, ep) -> None:
        """Per-decision model disagreement and the champion-only firewall over what influenced the run."""
        used: dict[str, list[str]] = {}
        n_conf = 0
        scores = []
        for r in ep.rows:
            if not r.parts:
                continue
            used[f"{ep.eid}#{r.decision.slot}"] = [self.L._mid[p[0]] for p in r.parts if p[0] in self.L._mid]
            if len(r.parts) >= 2:
                S = np.array([[p[1] * np.sign(p[2]) for p in r.parts]], float)
                c = float(DG.conflict_score(S)[0])
                scores.append(c)
                n_conf += c >= DG.DisagreementConfig().conflict_min
        if not ep.track:
            return                                         # a dry probe at another date is audited by the contract alone
        bad = CH.audit_decision_sources(self.L.board, used)
        self._fire("audit_decision_sources")
        if bad:
            raise FirewallBreach(f"non-champion knowledge influenced a decision: {bad[:3]}")
        self.conflicts[ep.eid] = {"n_multi": len(scores), "mean_conflict": float(np.mean(scores)) if scores else 0.0,
                                  "n_conflicted": int(n_conf)}
        del_keys = sorted(self.conflicts)[:-400]
        for k in del_keys:
            del self.conflicts[k]
        self._fire("conflict_score")

    def note_gate(self, verdict, now) -> None:
        """Warnings from an admitted firewall pass become data-audit channels (UNVERIFIED) for the research engine."""
        for f in verdict.findings():
            if str(getattr(f, "severity", "")).upper() == "WARN":
                self.gate_warnings[f"{f.layer}_{f.check}"] = "UNVERIFIED"

    # ------------------------------------------------------------------------------------------------ ASSIGN CREDIT
    def after_credit(self, ep, now) -> None:
        L = self.L
        learned = self._learned(ep)
        for r in ep.rows:
            if r.expected is None or not r.parts:
                continue
            den = sum(w for _, w, *_ in r.parts) or 1.0
            self.kcredit_rows.append((f"{ep.eid}#{r.decision.slot}", str(as_date(ep.now)), r.matured,
                                      {kid: w * e / den for kid, w, e, *_ in r.parts}, float(r.edge),
                                      r.situation.bins().get(L.cfg.context_dim, "na")))
        del self.kcredit_rows[: max(0, len(self.kcredit_rows) - self.cfg.max_rows)]
        if L.credit_reports and L.credit_reports[-1].decision_hash not in self.credit_seen:
            rep = L.credit_reports[-1]
            self.credit_seen.add(rep.decision_hash)
            self.apply_credit(rep, (), learned, now)
        if self._due(self.cfg.every_kcredit) and len(self.kcredit_rows) >= self.cfg.kcredit_min:
            self.knowledge_credit(learned, now)

    def knowledge_credit(self, learned: str, now) -> None:
        """Shapley credit over the knowledge items themselves (top items by use, the rest pooled), so credit can name an item."""
        use: dict[str, float] = {}
        for *_, contrib, _o, _c in self.kcredit_rows:
            for k, v in contrib.items():
                use[k] = use.get(k, 0.0) + abs(v)
        top = sorted(use, key=lambda k: (-use[k], k))[: self.cfg.kcredit_top]
        if len(top) < 2:
            return
        comps = [self.token(k) for k in top] + ["rest"]
        ledger = CR.DecisionLedger()
        for did, asof, mat, contrib, out, ctx in self.kcredit_rows:
            if as_date(mat) >= as_date(now):
                continue
            sc = {self.token(k): float(contrib.get(k, 0.0)) for k in top}
            sc["rest"] = float(sum(v for k, v in contrib.items() if k not in top))
            ledger.add(CR.Decision(did, asof, mat, sc, out, {"ctx": ctx}, {c: {self.safe.get(c, c): 1.0} for c in comps if c != "rest"}))
        cfg = CR.CreditConfig(n_boot=self.cfg.boot, n_perm=20 * len(comps), min_decisions=self.cfg.kcredit_min, min_groups=6,   # BH-resolvable
                              neutral={c: 0.0 for c in comps}, seed=self.seed)
        eng = CR.CreditEngine(CR.WeightedSumCombiner({c: 1.0 for c in comps}), cfg)
        rep = eng.assess(ledger, now)
        self.kcredit_reports.append(rep)
        del self.kcredit_reports[:-4]
        self._fire("kcredit_assess")
        matured, _ = ledger.mature(now)
        if matured:
            frame = CR.DecisionFrame.build(matured, comps, cfg.neutral)
            self.masked = [p for p in CR.masked_pairs(frame, eng.raw_credit(frame)) if "rest" not in (p.a, p.b)]
            self._fire("masked_pairs")
        self.apply_credit(rep, (), learned, now)

    def apply_credit(self, rep, knowledge, learned: str, now) -> list[dict]:
        """credit.update_proposals -> BeliefLedger: each REINFORCE / WEAKEN / HOLD proposal with a finite interval is one piece of
        LIVE_OUTCOME evidence about 'does this component earn credit', weighted by the proposal's bounded step."""
        props = CR.update_proposals(rep, knowledge)
        self._fire("update_proposals")
        applied = []
        for u in props:
            if u.action not in (CR.UpdateAction.REINFORCE, CR.UpdateAction.WEAKEN, CR.UpdateAction.HOLD) or u.context != "all":
                continue
            if not (math.isfinite(u.lo) and math.isfinite(u.hi)) or u.hi <= u.lo or u.n < 2:
                continue
            subject = f"credit:{self.safe.get(u.target, u.target)}"
            se = (u.hi - u.lo) / (2 * 1.96)
            rel = 1.0 if u.action is CR.UpdateAction.HOLD else max(0.05, min(1.0, u.max_step / 0.25))
            if subject not in self.L.beliefs.subjects():
                self.L.beliefs.register(subject, 0.0, max(abs(u.evidence), se, 1e-6) * 4.0)
            st = self.L.beliefs.update(subject, BL.Evidence(subject, str(as_date(learned)), float(u.evidence), float(se), int(u.n),
                                                             BL.EvidenceKind.LIVE_OUTCOME, reliability=rel,
                                                             source=f"credit:{u.action.value}"), now)
            applied.append({"subject": subject, "action": u.action.value, "evidence": u.evidence, "mean": st.mean, "at": str(as_date(now))})
        self.proposals_applied += applied
        del self.proposals_applied[:-200]
        self._fire("credit_to_belief", len(applied))
        return applied

    # ------------------------------------------------------------------------------------------------ INVESTIGATE FAILURE
    def on_failure(self, trade, cls, pm, learned: str) -> None:
        """knowledge.with_failure on the item whose pick failed, once per new cause (never a version per loss)."""
        if not cls.meaningful or not trade.knowledge_ids:
            return
        kid = trade.knowledge_ids[0]
        cur = self.L.store.latest(kid)
        if cur is None or cur.promotion == Promotion.RETIRED:
            return
        last = cur.failure_explanations[-1] if cur.failure_explanations else None
        if last is not None and last.cause == cls.cause:
            return
        at = max(str(as_date(learned)), str(as_date(cur.updated_at)))
        sub = cls.top_subsystem()
        note = f"postmortem {pm.pid}: {cls.cause.value}" if cls.cause != FailureCause.UNKNOWN else ""
        self.L._put(_learned_version(KN.with_failure(cur, at, cls.cause, sub, note, evidence_id=pm.pid), at))
        self._fire("with_failure")

    # ------------------------------------------------------------------------------------------------ LEARN CONDITIONS
    def after_conditions(self, ep, now) -> None:
        self.record_rows(ep)
        L = self.L
        if self.cfg.hierarchy:
            by_kid: dict[str, list] = {}
            for r in ep.rows:
                ctx = self._hier_context(r.situation)
                for p in r.members:
                    kid = L._kid_of.get(p)
                    if kid is not None:
                        by_kid.setdefault(kid, []).append({"when": r.matured, "effect": float(r.edge), **ctx})
            for kid, rows in sorted(by_kid.items()):
                self.bank.update(kid, pd.DataFrame(rows), now)
                self._fire("hierarchy_update")
        for pid, sets in sorted(L.boundaries._sets.items()):
            bset = sets[-1]
            key = f"{pid}:{bset.set_id}"
            if key not in self.boundary_shelf:
                self.boundary_shelf[key] = BD.to_knowledge(bset, version=len(sets))
                self._fire("boundary_to_knowledge")
        if self._due(self.cfg.every_competition):
            self.run_competitions(now)

    def record_rows(self, ep) -> None:
        for r in ep.rows:
            stance = tuple((k, float(w * np.sign(e))) for k, w, e, *_ in r.parts) if r.parts else ()
            self.rows.append(ResolvedRow(str(as_date(ep.now)), r.matured, self.L._ident(r.key[1]), float(r.raw_ret), float(r.edge),
                                         r.expected, r.decision.action == "LONG", frozenset(r.members),
                                         r.situation.bins().get(self.L.cfg.context_dim, "na"),
                                         _num(r.situation.get("market.vix")), _num(r.situation.get("breadth.breadth")),
                                         tuple(self._hier_context(r.situation).values()), stance))
        del self.rows[: max(0, len(self.rows) - self.cfg.max_rows)]

    def _weekly_frame(self, kid: str, now) -> pd.DataFrame:
        """One row per matured week of the item: signed edge, the market state that week, and the item's own trailing mean
        (known before the week's outcome) as the competition's signal."""
        dates, vals = self._signed_weekly(kid)
        keep = [(d, v) for d, v in zip(dates, vals) if as_date(d) < as_date(now)]
        if not keep:
            return pd.DataFrame(columns=["y", "signal", "volatility", "breadth"])
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in keep])
        y = pd.Series([v for _, v in keep], index=idx, dtype=float)
        mk = pd.DataFrame(self.L._mkt).T if self.L._mkt else pd.DataFrame(columns=["vix", "breadth"])
        mk.index = pd.DatetimeIndex(mk.index)
        mk = mk.reindex(idx)
        return pd.DataFrame({"y": y.values, "signal": y.shift(1).expanding().mean().fillna(0.0).values,
                             "volatility": mk["vix"].ffill().bfill().fillna(0.0).values,
                             "breadth": mk["breadth"].ffill().bfill().fillna(0.0).values}, index=idx)

    def run_competitions(self, now) -> None:
        """competition.boundary_field for every item with learned boundaries: null vs 'persists' vs 'only inside the boundary'
        (each story charged its complexity prior), replayed from scratch so the arena only ever saw the past."""
        for kid in self._kids():
            pid = self.L._pid_of[kid]
            sets = self.L.boundaries.history(pid)
            df = self._weekly_frame(kid, now)
            if len(df) < CP.ArenaConfig().min_fit + 2:
                continue
            arena = CP.boundary_field("signal", df, sets[-1:], now, proxies=("volatility",), batch=4)
            spec_units = {s.hyp_id: CX.RuleSpec.units(s.rule_spec()) for s in arena.specs}
            self.arenas[kid] = {"leader": arena.leader(), "separated": arena.separated(), "weights": arena.weights(),
                                "units": spec_units, "n": arena.n_seen, "at": str(as_date(now))}
            self._fire("boundary_field")

    # ------------------------------------------------------------------------------------------------ UPDATE RELIABILITY
    def after_reliability(self, ep, now) -> None:
        L = self.L
        learned = self._learned(ep)
        if L.calibration_last is not None and self._due(L.cfg.discover_every):
            for kid in self._kids():
                self.cal.put(kid, str(as_date(now)), L.calibration.assess(now, self.seed, kid))
            self._fire("calibration_cache")
        self.lifecycle_step(now)
        if self._due(self.cfg.every_break):
            self.break_step(learned, now)
        if self._due(self.cfg.every_health):
            self.health_step(learned, now)

    def lifecycle_params(self) -> dict:
        """lifecycle.PARAMS rescaled to the configured window, keeping the default ratios (window 26 gives the defaults exactly)."""
        w = self.cfg.lifecycle_window
        return {"window": w, "est_n": w, "birth_n": max(4, w // 2), "recover_n": max(4, w // 2), "dormant_after": w,
                "hold": max(2, round(8 * w / 26)), "retire_after": 4 * w, "give_up": 4 * w}

    def lifecycle_step(self, now) -> None:
        """Push each item's new weekly outcomes through its lifecycle machine; the current stage goes through the retirement or
        recovery gate (lifecycle.apply_to_ledger), with the cause the break engine named when it named one."""
        for kid in self._kids():
            dates, vals = self._signed_weekly(kid)
            m = self.machines.setdefault(kid, LC.LifecycleMachine(self.lifecycle_params()))
            i0 = self._lc_cursor.get(kid, 0)
            for v in vals[i0:]:
                m.push(float(v))
            self._lc_cursor[kid] = len(vals)
            if not vals:
                continue
            cause = FailureCause(self.break_applied.get(kid, FailureCause.UNKNOWN.value))
            before = self.L.retirement.state(kid, now)
            v = LC.apply_to_ledger(self.L.retirement, kid, m.to_trace(), vals, dates, now, cause, apply=True)
            if v is not None:
                self._fire("apply_to_ledger")
                after = self.L.retirement.state(kid, pd.Timestamp(as_date(now)) + pd.Timedelta(days=1))   # written at now, in force after
                if after != before:
                    self.lifecycle_verdicts.append({"kid": kid, "stage": m.state.value, "from": str(before), "to": str(after),
                                                    "cause": cause.value, "at": str(as_date(now))})

    def break_step(self, learned: str, now) -> None:
        """break_detection on every item with enough history; an EXPLAINED break with a validated condition becomes contexts and
        anti-contexts (reliability.contexts_from_condition) written as a new knowledge version."""
        mk = pd.DataFrame(self.L._mkt).T if self.L._mkt else None
        for kid in self._kids():
            dates, vals = self._signed_weekly(kid)
            if len(vals) < 12 or mk is None:
                continue
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
            m = mk.copy()
            m.index = pd.DatetimeIndex(m.index)
            m = m.reindex(idx).ffill().bfill()
            fr = pd.DataFrame({"value": vals, "vix": m["vix"].fillna(0.0).values, "breadth": m["breadth"].fillna(0.0).values}, index=idx)
            item = BK.ItemSeries(kid, fr, tuple(BK.ContextColumn(c, dim) for c, dim, *_ in BREAK_COLUMNS))
            self.breaks.items[kid] = item
        for ex in self.breaks.run(now):
            self._fire("explain_break")
            if not ex.explained or ex.condition is None:
                continue
            self.break_applied[ex.item_id] = ex.cause.value
            ctx, anti = RL.contexts_from_condition(ex.condition)
            self._fire("contexts_from_condition")
            anti_set = _context_set(anti)
            cur = self.L.store.latest(ex.item_id)
            if anti_set is not None and cur is not None and cur.anti_contexts != anti_set and cur.promotion != Promotion.RETIRED:
                self.L._revise(ex.item_id, learned, f"break explained ({ex.cause.value}): anti-context from a validated condition",
                               anti_contexts=anti_set)

    def health_step(self, learned: str, now) -> None:
        """health.inputs_from_knowledge -> HealthMonitor -> debounced epistemic proposals written as new versions; BROKEN or
        DEGRADING items take their recent temporal series through the retirement gate (temporal.to_evidence)."""
        L = self.L
        items = [L.store.latest(k) for k in self._kids()]
        if not items:
            return
        series = {kid: self._series(kid) for kid in self._kids()}
        contra: dict[str, list] = {}
        for a, b in sorted(L._contradicts):
            w = float(L.index.contradictions(a).get(b, 0.0))
            for k in (a, b):
                contra.setdefault(k, []).append(HE.ContradictionRef(f"{a}|{b}", w, False, str(as_date(learned))))
        inputs = HE.inputs_from_knowledge(items, series, now, ledger=L.retirement, contradictions=contra)
        self._fire("inputs_from_knowledge")
        recs = self.health.step(inputs, now)
        self.health_rows = [{"knowledge_id": self.token(r.knowledge_id), "when": str(as_date(learned)), "health": r.state.value,
                             "subsystem": "SELECTION"} for r in recs]
        current = {k.knowledge_id: k.epistemic for k in items}
        props = HE.epistemic_proposals(recs, current)
        self._fire("epistemic_proposals")
        for p in props:
            kid = p["knowledge_id"]
            official = HE.debounced_state(self.health.book, kid, self.cfg.health_min_hold, now)
            cur = L.store.latest(kid)
            if official is None or official.value != p["because"] or cur.lifecycle == Lifecycle.RETIRED:
                continue
            to = Epistemic(p["to"])
            if official in (Health.UNKNOWN, Health.INSUFFICIENT_EVIDENCE, Health.CONTRADICTED):
                continue                   # no information is no reason to relabel; a contradiction is relabelled by its investigation
            if cur.promotion == Promotion.CHAMPION and to not in CHAMPION_EPISTEMIC:
                continue                   # a champion that broke loses influence through the retirement gate below, not a relabel
            if cur.epistemic == Epistemic.CONDITIONAL and to == Epistemic.SUPPORTED:
                continue                   # healthy inside its conditions says nothing new about outside them
            L._revise(kid, learned, f"health {official.value}: {'; '.join(p['reasons'])[:120]}", epistemic=to)
            self.health_applied.append({"kid": kid, "from": p["from"], "to": p["to"], "at": str(as_date(now))})
        for r in recs:
            if r.state in (Health.BROKEN, Health.DEGRADING) and L.retirement.state(r.knowledge_id, now) in (RT.State.ACTIVE, RT.State.DEGRADED):
                d, v = self._signed_weekly(r.knowledge_id)
                if len(d) < 4:
                    continue
                ser = TP.from_outcomes(d, v, now, period_days=7)
                since = as_date(d[max(0, len(d) - self.cfg.lifecycle_window)]) - pd.Timedelta(days=1)
                ev = TP.to_evidence(ser, since, now)
                self._fire("to_evidence")
                verdict = L.retirement.evaluate(r.knowledge_id, ev, now, FailureCause.WEAKENING_EFFECT, apply=True)
                self.retire_verdicts.append({"kid": r.knowledge_id, "health": r.state.value, "to": str(verdict.to_state),
                                             "n": ev.n, "at": str(as_date(now))})
        del self.retire_verdicts[:-200]

    # ------------------------------------------------------------------------------------------------ UPDATE GRAPH
    def after_graph(self, ep, now) -> None:
        self.graph_step(self._learned(ep), now)

    def graph_step(self, learned: str, now) -> None:
        """Credit edges, contradiction relations, redundancy on its cadence, and the contradiction monitor's period run."""
        g = self.L.graph
        for rep in self.kcredit_reports[-1:]:
            for a, rel, b, w in CR.credit_edges(rep):
                ka, kb = self.safe.get(a), self.safe.get(b)
                if ka and kb and g.has_node(ka) and g.has_node(kb):
                    g.add_edge(ka, kb, rel, learned, weight=float(min(1.0, max(0.0, w))), evidence=(f"credit:{rep.decision_hash[:12]}",))
                    self._fire("credit_edges")
        for a, b in sorted(self.L._contradicts):
            self._relate(a, "contradicting", b, learned, "contradiction triaged by the learner")
        if self._due(self.cfg.every_redundancy):
            self.redundancy_step(learned, now)
        report = self.contra_monitor.run_period(now)
        self.contra_last = report
        self._fire("run_period")
        keys = set(g.contradiction_keys(now))
        lost = sorted({tuple(t.pair) for t in report.items} - keys)
        if lost:
            raise FirewallBreach(f"contradiction monitor tracks pairs the graph never held before {as_date(now)}: {lost[:3]}")
        self.contra_keys = len(keys)
        self.contra_signals = []
        for q in self.contra_monitor.research_questions(report):
            s = RPR.make_signal(RPR.SignalKind.CONTRADICTION, q["when"], self.token(q["subject"]), q["magnitude"],
                                counterpart=self.token(q["counterpart"]), stake=q["stake"], n_obs=q["n_obs"], evidence=q["evidence"])
            if s.check():
                raise FirewallBreach("; ".join(s.check()))
            self.contra_signals.append(s)
        self._fire("contradiction_signals", len(self.contra_signals))

    def _relate(self, a: str, kind: str, b: str, learned: str, why: str) -> None:
        """knowledge.with_relation on both items, once per (item, kind, other)."""
        for x, y in ((a, b), (b, a)):
            key = (x, kind, y)
            cur = self.L.store.latest(x)
            if key in self.relations or cur is None or y in getattr(cur.relations, kind):
                self.relations.add(key)
                continue
            at = max(str(as_date(learned)), str(as_date(cur.updated_at)))
            self.L._put(_learned_version(KN.with_relation(cur, at, kind, y, why), at))
            self.relations.add(key)
            self._fire("with_relation")

    def redundancy_step(self, learned: str, now) -> None:
        """Candidate pairs: credit's masked pairs plus items whose activity overlaps heavily; redundancy's five-kind analysis
        decides, and its edges go to the graph and to both items' relations."""
        kids = self._kids()
        rows = [r for r in self.rows if as_date(r.matured) < as_date(now)]
        if len(kids) < 2 or len(rows) < 60:
            return
        act = {k: np.array([self.L._pid_of[k] in r.members for r in rows]) for k in kids}
        pairs = {tuple(sorted((self.safe.get(p.a, p.a), self.safe.get(p.b, p.b)))) for p in self.masked}
        for i, a in enumerate(kids):
            for b in kids[i + 1:]:
                inter, union = (act[a] & act[b]).sum(), (act[a] | act[b]).sum()
                if union and inter / union >= self.cfg.redundancy_overlap:
                    pairs.add((a, b))
        items = sorted({k for p in pairs for k in p if k in act})
        if len(items) < 2:
            return
        sig = pd.DataFrame({self.token(k): act[k].astype(float) * self.L._direction_of(k) for k in items})
        panel = RD.ObservationPanel.build(sig, [r.edge for r in rows], [pd.Timestamp(r.decided) for r in rows],
                                          [r.ctx for r in rows], horizon_days=self.L.cfg.horizon_days, now=now)
        profiles = [RD.ItemProfile(self.token(k), ("learned", "quantile-cell"), (self.L._pid_of[k].rsplit(":q", 1)[0],)) for k in items]
        rep = RD.RedundancyAnalyzer(RD.RedundancyConfig(seed=self.seed, n_boot=max(20, self.cfg.boot // 2))).analyze(panel, profiles, now)
        self.redundancy_last = rep
        self._fire("redundancy_analyze")
        for src, edge, dst, w, why in rep.edges():
            ka, kb = self.safe.get(src, src), self.safe.get(dst, dst)
            self.L.graph.add_edge(ka, kb, edge, learned, weight=float(min(1.0, max(0.0, w))), evidence=(why[:40],))
            self._relate(ka, "redundant" if edge == Edge.REDUNDANT_WITH else "complementary", kb, learned, why)
            self._fire("redundancy_edges")

    # ------------------------------------------------------------------------------------------------ UPDATE META
    def after_meta(self, ep, now) -> None:
        learned = self._learned(ep)
        if self._due(self.cfg.every_questions):
            self.questions_step(now)
            self.miner_step(now)
        if self._due(self.cfg.every_interpretation):
            self.interpretation_step(learned, now)
        self.unknowns_step(learned, now)
        self.complexity_step()
        if self._due(self.cfg.every_disagreement):
            self.disagreement_step(now)
        if self._due(self.cfg.every_value):
            self.value_step(now)

    def questions_step(self, now) -> None:
        """The three questions (real / useful / useful-now) for every stored item, family-wise controlled across the family."""
        rels = []
        for kid in self._kids():
            eff = self._series(kid)
            eff = eff[eff.index < pd.Timestamp(as_date(now))]
            if len(eff) < 3:
                continue
            rels.append(QU.RelationEvidence(self.token(kid), eff, n_trials=self.L._n_candidates(),
                                            oos_start=pd.Timestamp(self.L._birth_date[kid])))
        if not rels:
            return
        for t in self.qengine.ask_many(rels, now):
            self.answers.add(t)
        self._fire("ask_many", len(rels))

    def interpretation_step(self, learned: str, now) -> None:
        """An interpretation record per item (six competing explanations, the standard probes run on its resolved rows); the
        test with the highest expected information gain that has not been run yet goes to the research engine."""
        rows = [r for r in self.rows[-4000:] if as_date(r.matured) < as_date(now)]
        if len(rows) < 30:
            return
        y = np.array([r.edge for r in rows])
        vol = np.array([r.vix for r in rows])
        labels = sorted({r.ctx for r in rows})
        reg = np.array([labels.index(r.ctx) for r in rows], float)          # probes read numeric regime codes
        for kid in self._kids():
            pid = self.L._pid_of[kid]
            x = np.array([float(pid in r.members) for r in rows]) * self.L._direction_of(kid)
            if x.std() == 0:
                continue
            m, se, _ = _t(y[x != 0] * np.sign(x[x != 0]))
            obs = IN.Observation(f"rows in the item's cell earn a signed excess return", rows[0].decided, rows[-1].matured, int((x != 0).sum()),
                                 value=float(m))
            rel = IN.StatRelation(obs.observation_id, float(m), float(se if math.isfinite(se) else 1.0), float((x != 0).sum()),
                                  n_tests=self.L._n_candidates(), out_of_sample=True)
            probes = IN.run_standard_tests(y, x, vol=vol if np.isfinite(vol).all() else None, regime=reg, window=min(60, len(y) // 3))
            rec = IN.interpret(obs, rel, probes, str(as_date(now)), str(self.L._birth_date[kid]), subject=self.token(kid),
                               evidence_through=str(as_date(learned)))
            self.interpretations[kid] = rec
            nt = IN.next_test(rec.belief) if rec.belief is not None else None
            if nt is not None:
                self.next_tests[kid] = nt
            self._fire("next_test")

    def unknowns_step(self, learned: str, now) -> None:
        """What the learner does not know about each stored item, ranked by value at stake per cost (unknowns.rank_unknowns)."""
        usage = self.L.decision_log.usage_counts()
        n_ep = max(1, len(self.L.summaries))
        stakes = {}
        for kid in self._kids():
            st = self.L.beliefs.current(self.L._pid_of[kid])
            wk = self.L._weekly.get(self.L._pid_of[kid], [])
            ctxs = tuple(sorted(c for c, (m, n) in self.L.context_effects(self.L._pid_of[kid], self.L._direction_of(kid)).items() if n >= 5))
            p = float(st.prob_sign_right()) if st is not None else None
            facts = UK.UnknownFacts(n_events=len(wk), n_eff=float(len(wk)), min_n_eff=12.0, ever_tested=bool(wk), tested_contexts=ctxs,
                                    support=p, against=None if p is None else 1.0 - p)
            sub = self.token(kid)
            rec = UK.classify(sub, facts, str(as_date(now)), UK.Availability(can_collect_data=True, can_run_experiment=True))
            if rec is not None:
                self.unknowns.open(rec)
            elif any(r.subject == sub for r in self.unknowns.open_rows()):
                self.unknowns.resolve(sub, now, learned, "enough evidence to be judged on it")
            stakes[sub] = UK.Stake(sub, usage.get(kid, 0) / n_ep, abs(st.mean) if st is not None else 0.0, 1.0)
        self.unknown_rank = UK.rank_unknowns(self.unknowns.open_rows(), stakes, now)
        self._fire("rank_unknowns")

    def complexity_step(self) -> None:
        """Every stored item's scope charged in complexity units against the evidence it has (complexity.within_budget)."""
        for kid in self._kids():
            k = self.L.store.latest(kid)
            spec = CX.spec_from_knowledge(self.token(kid), dict(k.contexts.items()), dict(k.anti_contexts.items()))
            n_eff = float(len(self.L._weekly.get(self.L._pid_of[kid], [])))
            self.complexity[kid] = {"units": spec.units(), "within_budget": CX.within_budget(spec, max(n_eff, 1.0)), "n_eff": n_eff}
        self._fire("complexity_budget")

    def disagreement_step(self, now) -> None:
        """The items that voted on the same rows, as sources of a stance frame: when they disagree, who is right, and is the
        disagreement itself informative (disagreement.DisagreementEngine)."""
        rows = [r for r in self.rows[-4000:] if r.stance and as_date(r.matured) < as_date(now)]
        cnt: dict[str, int] = {}
        for r in rows:
            for k, _ in r.stance:
                cnt[k] = cnt.get(k, 0) + 1
        src = sorted(cnt, key=lambda k: (-cnt[k], k))[:8]
        if len(src) < 2 or len(rows) < 30:
            return
        st = pd.DataFrame([{self.token(k): dict(r.stance).get(k, np.nan) for k in src} for r in rows])
        sf = DG.StanceFrame.build(st, [pd.Timestamp(r.decided) for r in rows], [r.edge for r in rows], pd.DataFrame({"ctx": [r.ctx for r in rows]}),
                                  horizon_days=self.L.cfg.horizon_days, now=now)
        cfg = DG.DisagreementConfig(seed=self.seed, n_boot=max(20, self.cfg.boot // 2), n_perm=max(20, self.cfg.boot // 2))
        self.disagreement_last = DG.DisagreementEngine(cfg).assess(sf, now)
        self._fire("disagreement_assess")

    def value_step(self, now) -> None:
        """Did the learned knowledge change what a portfolio would have earned?  portfolio_value.decompose_value against an
        identity-free random ranking, then a scorecard with controls A-E appended to the hash-chained store."""
        rows = [r for r in self.rows if as_date(r.matured) < as_date(now)]
        if len({r.decided for r in rows}) < 4:
            return
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([pd.Timestamp(r.decided) for r in rows]), [r.ident for r in rows]], names=["date", "ticker"])
        base = [int(stable_hash([r.ident, r.decided, self.seed], 8), 16) / 16 ** 8 - 0.5 for r in rows]
        panel = pd.DataFrame({"fwd": [r.raw_ret for r in rows], "mature": [pd.Timestamp(r.matured) for r in rows], "score_base": base,
                              "score_new": [0.0 if r.expected is None else r.expected for r in rows]}, index=idx)
        panel = panel[~panel.index.duplicated()]
        spec = PV.ValueSpec(horizon_days=self.L.cfg.horizon_days, score=("score_base", "score_new"), move=None, direction=None, pick="score")
        self.value_last = PV.decompose_value(panel, spec, now, k=self.L.cfg.top_n, min_dates=4, seed=self.seed, n_boot=self.cfg.boot)
        self._fire("decompose_value")
        if self.scorecards is None:
            return
        units = pd.DataFrame({"date": [pd.Timestamp(r.decided) for r in rows], "mature": [pd.Timestamp(r.matured) for r in rows],
                              "ticker": [r.ident for r in rows], "base": 0.0, "alt": [r.edge for r in rows],
                              "signal": [0.0 if r.expected is None else r.expected for r in rows], "regime": [r.ctx for r in rows]})
        units = units.drop_duplicates(["ticker", "date"]).reset_index(drop=True)
        card = SC.scorecard_for_learner(units, trust_learned_signal, learner_version=f"{self.L.config_hash[:10]}-t{self.L._tick:05d}", now=now,
                                        code_hash=self.L.code_hash, seed=self.seed, n_boot=self.cfg.boot, min_units=20)
        errs = card.check()
        if errs:                                           # e.g. one simulated year gives no forward-year fold: the controls cannot run
            self.scorecard_refusals.append({"version": card.learner_version, "why": "; ".join(errs)[:200]})
            del self.scorecard_refusals[:-50]
            self._fire("scorecard_not_storable")
            return
        self.scorecard_ids.append(self.scorecards.append(card))
        self.valid_card = card
        self._fire("scorecard_append")

    def miner_rows(self, now) -> list[dict]:
        """The learner's own cell search written as PatternMiner rows: discovery = the weeks up to the item's birth (in-sample,
        charged for every cell searched), confirmation = the weeks after it (held out).  One row per stored item."""
        rows = []
        for kid in self._kids():
            s = self._series(kid)
            s = s[s.index < pd.Timestamp(as_date(now))]
            birth = pd.Timestamp(self.L._birth_date[kid])
            disc, conf = s[s.index <= birth], s[s.index > birth]
            md, sd, td = _t(disc.values)
            mc, sc, tc = _t(conf.values)
            rows.append({"key_named": self.token(kid), "effect": float(s.mean()) if len(s) else float("nan"),
                         "t_disc": td if len(disc) >= 3 else None, "t_conf": tc if len(conf) >= 3 else None,
                         "end_disc": str(disc.index.max().date()) if len(disc) else None,
                         "end_conf": str(conf.index.max().date()) if len(conf) else None, "n_disc": len(disc), "n_conf": len(conf)})
        return rows

    def miner_step(self, now) -> None:
        """Miner rows -> belief.evidence_from_pattern_row -> a staged BeliefLedger that keeps discovery (in-sample, multiplicity
        charged) apart from confirmation (out of sample); the learner's own weekly beliefs are not touched (no double counting)."""
        rows = self.miner_rows(now)
        for r in rows:
            ev = BL.evidence_from_pattern_row({k: v for k, v in r.items() if v is not None}, now, n_candidates=self.L._n_candidates(),
                                              n_disc=r["n_disc"] or None, n_conf=r["n_conf"] or None, source="learner-cells")
            self._fire("evidence_from_pattern_row")
            if ev:
                subject = ev[0].subject
                if subject not in self.staged.subjects():
                    self.staged.register(subject, 0.0, self.L.cfg.belief_prior_sd)
                self.staged.update(subject, ev, now)
        self.staged_rows = rows

    def claim_evidence(self, kid: str, now) -> dict:
        """Before a promotion attempt: the identity harness on the item's own rule over the recent panels, and the learner's latest
        valid scorecard, registered with the wiring hub so PromotionGate's learning-claim gate judges real evidence. Recomputed
        only when the item has a new version."""
        from . import wiring as W
        k = self.L.store.latest(kid)
        key = f"{kid}@{k.version}"
        if key in self.claims:
            return self.claims[key]
        out = {"identity": None, "scorecard": False}
        rep = self.identity_report(kid)
        if rep is not None:
            W.register_identity(kid, rep)
            out["identity"] = bool(rep.passed)
        if self.valid_card is not None:
            W.register_scorecard(kid, self.valid_card)
            out["scorecard"] = True
        self.claims[key] = out
        self._fire("claim_evidence")
        return out

    def identity_report(self, kid: str):
        """identity_firewall.IdentityHarness over the item's rule (sign x membership of its quantile cell, sign fitted on the
        training half only) on the learner's recent panels: an identity-free rule must survive every identity attack."""
        rec = self.L._recent
        if len(rec) < 4:
            return None
        pid = self.L._pid_of[kid]
        feat, lvl = pid.rsplit(":q", 1)
        X = pd.concat([x[[feat]] for x, _ in rec if feat in x.columns])
        y = pd.concat([e for x, e in rec if feat in x.columns])
        dates = sorted(set(X.index.get_level_values(0)))
        if len(dates) < 4:
            return None
        cut = dates[len(dates) // 2]
        tr = X.index.get_level_values(0) < cut
        n_q = self.L.cfg.n_quantiles

        def rule(Xt, yt, Xe, seed):
            def cell(Z):
                r = Z[feat].groupby(level=0).rank(method="first")
                n = Z[feat].groupby(level=0).transform("count")
                return (((r - 1) * n_q) // n).clip(0, n_q - 1) == int(lvl)
            m = cell(Xt).to_numpy()
            sign = 1.0 if float(np.nanmean(yt.to_numpy()[m])) >= 0 else -1.0 if m.any() else 0.0
            return pd.Series(sign * cell(Xe).astype(float).to_numpy(), index=Xe.index)

        h = IDF.IdentityHarness(rule, seed=self.seed, boot=max(50, self.cfg.boot), min_dates=2)
        self._fire("identity_harness")
        return h.run(X[tr], y[tr], X[~tr], y[~tr])

    # ------------------------------------------------------------------------------------------------ SELECT RESEARCH
    def research_signals(self, learned: str, now) -> list:
        """Signals from health, the data audit, the contradiction monitor, the ranked unknowns, interpretation's next test,
        the three questions, complexity and credit (INVESTIGATE proposals), for ResearchPriorityEngine.step."""
        sig: list = []
        if self.health_rows:
            sig += RPR.signals_from_health([r for r in self.health_rows if as_date(r["when"]) < as_date(now)], now)
            self._fire("signals_from_health")
        if self.gate_warnings:
            sig += RPR.signals_from_data_audit({safe_token(k): v for k, v in self.gate_warnings.items()}, learned, now)
            self._fire("signals_from_data_audit")
        sig += self.contra_signals
        for sub, score in self.unknown_rank[:3]:
            sig.append(RPR.make_signal(RPR.SignalKind.UNKNOWN_AREA, learned, sub, float(min(0.99, 0.2 + score)),
                                       contexts={"gap": f"unknown {sub}"}, stake=0.5))
        for kid, (test, eig) in sorted(self.next_tests.items())[:3]:
            sig.append(RPR.make_signal(RPR.SignalKind.UNKNOWN_AREA, learned, self.token(kid), float(min(0.99, eig / math.log(2))),
                                       contexts={"gap": f"{test} for {self.token(kid)}"}, stake=0.6))
        for rid in self.answers.relations():
            t = self.answers.latest(rid)
            if t is not None and (t.real.verdict == QU.Answer.NO or t.useful_now.verdict == QU.Answer.NO):
                sig.append(RPR.make_signal(RPR.SignalKind.WEAK_PATTERN, learned, rid, 0.5, stake=0.6))
        for kid, c in sorted(self.complexity.items()):
            if not c["within_budget"]:
                sig.append(RPR.make_signal(RPR.SignalKind.UNKNOWN_AREA, learned, self.token(kid), 0.4,
                                           contexts={"gap": f"scope of {self.token(kid)} exceeds its evidence"}, stake=0.4))
        for p in self.proposals_applied[-20:]:
            if p["action"] == CR.UpdateAction.WEAKEN.value:
                sig.append(RPR.make_signal(RPR.SignalKind.RELIABLE_DRIFT, learned, safe_token(p["subject"].split(":", 1)[1]), 0.5, stake=0.6))
        self._fire("hook_signals", len(sig))
        return sig

    def close_research(self, step, learned: str, now) -> dict:
        """failed_learners.check_proposal on every selected candidate -> propose_selected -> run the experiments whose prospective
        window has matured -> ExperimentLedger.record_result -> ResearchPriorityEngine.update_from_result."""
        L = self.L
        engine = L.research
        keep, blocked = [], {}
        for sc in step.plan.selection.selected:
            c = sc.candidate
            prop = FLR.LearnerProposal(c.cid, c.question, tuple(sorted({str(c.target.value).lower(), *c.family.split(":")})), c.family)
            v = self.failed.check_proposal(prop, now)
            self._fire("check_proposal")
            if v.blocked:
                blocked[c.cid] = v.message
                it = engine.queue.items.get(c.cid)
                if it is not None:
                    it.status = RPR.ItemStatus.OBSOLETE
                    it.note = f"dead end: {v.message}"[:200]
            else:
                keep.append(sc)
        self.dead_ends += [{"cid": k, "why": m, "at": str(as_date(now))} for k, m in blocked.items()]
        sel = dataclasses.replace(step.plan.selection, selected=tuple(keep))
        step2 = dataclasses.replace(step, plan=dataclasses.replace(step.plan, selection=sel))
        res = RPR.propose_selected(step2, engine, self.experiments, now, self.seed)
        self._fire("propose_selected")
        bits = {sc.candidate.cid: float(sc.eig_bits) for sc in keep}
        for eid in res["proposed"]:
            rec = self.experiments.get(eid, pd.Timestamp(as_date(now)) + pd.Timedelta(days=1))
            if rec is None:
                raise FirewallBreach(f"{eid} was proposed but the ledger cannot show it after {as_date(now)}")
            cid = eid[len("exp-"):] if eid.startswith("exp-") else eid
            cfgd = dict(rec.experiment.config)
            self.open_exps[eid] = OpenExperiment(eid, cid, bits.get(cid, 0.0), rec.created_at, str(cfgd.get("kind", "")),
                                                 tuple(cfgd.get("subjects", ())))
        done = []
        for eid in sorted(self.open_exps):
            ox = self.open_exps[eid]
            out = self.evaluate_experiment(ox, now)
            if out is None:
                continue
            result, learned_txt, kid, verdict = out
            rec = self.experiments.record_result(eid, result, now, learned_txt,
                                                 ["whether it persists beyond the test window"] if verdict else ["no cause established"],
                                                 "follow the queue's follow-ups" if verdict else "extend the sample")
            upd = engine.update_from_result(rec, ox.cid, ox.predicted_bits, now)
            self._fire("update_from_result")
            self._experiment_to_graph(eid, kid, result, now)
            self.exp_results.append({"experiment": eid, "outcome": result.outcome, "kind": result.kind, "bits": upd["realised_bits"],
                                     "follow_ups": len(upd["follow_ups"]), "at": str(as_date(now))})
            done.append(eid)
        for eid in done:
            del self.open_exps[eid]
        del self.exp_results[:-200]
        return {"proposed": res["proposed"], "refused": res["refused"], "blocked": blocked, "answered": done}

    def evaluate_experiment(self, ox: OpenExperiment, now):
        """The in-loop, prospective experiment: only outcomes that matured AFTER the experiment was registered (and before `now`)
        count.  Returns (result, learned, kid, decisive) or None while the window is still open."""
        kid = next((self.safe[s] for s in ox.subjects if s in self.safe and self.safe[s] in self.L._pid_of), None)
        created = as_date(ox.created)
        post, rows = [], []
        if kid is not None:
            d, v = self._signed_weekly(kid)
            post = [(a, b) for a, b in zip(d, v) if created < as_date(a) < as_date(now)]
            pid, dirn = self.L._pid_of[kid], self.L._direction_of(kid)
            rows = [r for r in self.rows if pid in r.members and created < as_date(r.matured) < as_date(now)]
        weeks = sorted({r.matured for r in self.rows if created < as_date(r.matured) < as_date(now)})
        if len(weeks) < self.cfg.exp_wait:
            return None
        outcome = None
        split = era = False
        m = se = t = 0.0
        if len(post) >= self.cfg.exp_wait:
            m, se, t = _t([b for _, b in post])
            by: dict[str, list] = {}
            for r in rows:
                by.setdefault(r.ctx, []).append(dirn * r.edge)
            ts = [_t(v)[2] for v in by.values() if len(v) >= 5]
            split = bool(ts) and max(ts) >= 2.0 and min(ts) <= -1.0
            half = len(post) // 2
            if half >= 2:
                a, b = _t([x for _, x in post[:half]]), _t([x for _, x in post[half:]])
                diff_se = math.sqrt(a[1] ** 2 + b[1] ** 2) if math.isfinite(a[1]) and math.isfinite(b[1]) else float("inf")
                era = diff_se > 0 and math.isfinite(diff_se) and abs(a[0] - b[0]) / diff_se >= 2.0
            enough = len(post) >= 2 * self.cfg.exp_wait
            if ox.kind in ("FAILURE", "SURPRISE"):
                outcome = ("sign_flips_out_of_sample" if t <= -2.0 else "context_split_explains" if split else "era_split_explains" if era
                           else "vanishes_everywhere" if abs(t) < 1.0 and enough else None)
            elif ox.kind in GENERIC_MAP:
                holds, fails, spl, noise = GENERIC_MAP[ox.kind]
                outcome = (spl if split else holds if t >= 2.0 else fails if t <= -1.0 and enough else noise if abs(t) < 1.0 and enough else None)
        if outcome is None and len(weeks) >= self.cfg.exp_max_wait:
            outcome = "inconclusive"
        if outcome is None:
            return None
        last = max([a for a, _ in post] + list(weeks)) if (post or weeks) else str(as_date(now))
        rec = self.experiments.get(ox.experiment_id, pd.Timestamp(as_date(now)) + pd.Timedelta(days=1))
        if rec is None:
            raise FirewallBreach(f"open experiment {ox.experiment_id} is missing from the ledger")
        lead = max(rec.competing_hypotheses, key=lambda h: float(h.prior)).hid
        probs = {e.outcome: float(e.probability) for e in rec.expected_outcomes if e.hid == lead}
        if outcome == "inconclusive":
            kind = EM.ResultKind.MIXED
        elif outcome in ("vanishes_everywhere", "supports_noise"):
            kind = EM.ResultKind.NULL
        else:
            kind = EM.ResultKind.CONFIRMED if probs and outcome == max(probs, key=lambda o: probs[o]) else EM.ResultKind.REFUTED
        n = len(post)
        note = (f"with n={n} weeks the test could detect a mean of {2.8 * se:.4f} per week" if n and math.isfinite(se)
                else "no evaluable subject: power undefined")
        result = EM.ExperimentResult(kind, outcome, {"mean": m, "t": t, "n_weeks": n, "context_split": split, "era_split": era}, n,
                                     power_note=note, observed_at=str(as_date(last)),
                                     summary=f"prospective check over {len(weeks)} matured weeks after registration")
        learned_txt = [f"{outcome} (prospective, {n} weeks of the subject)"]
        return result, learned_txt, kid, outcome != "inconclusive"

    def _experiment_to_graph(self, eid: str, kid: str | None, result, now) -> None:
        g = self.L.graph
        node = f"experiment:{eid}"
        at = result.observed_at
        g.add_node(node, KG.NodeType.EXPERIMENT, at, label=result.outcome)
        if kid is not None and g.has_node(kid) and result.outcome != "inconclusive":
            rel = KG.Link.VALIDATED_BY if result.outcome in HOLDS else KG.Link.REFUTED_BY
            g.add_edge(kid, node, rel, at)
            self._fire("experiment_edges")

    def readiness(self, k, now) -> dict:
        """decision_contract.readiness at the promotion gate: what production use would still need (diagnostic, recorded)."""
        r = DC.readiness(k, now)
        self.readiness_log.append({"kid": k.knowledge_id, "at": str(as_date(now)), "ready": r["ready"], "missing": r["missing"][:4]})
        del self.readiness_log[:-100]
        self._fire("readiness")
        return r

    # ------------------------------------------------------------------------------------------------ persistence and report
    def maybe_persist(self, now, force: bool = False) -> dict:
        if self.root is None or not self.cfg.persist or not (force or self._due(self.cfg.every_persist)):
            return {}
        return self.persist(now)

    def persist(self, now) -> dict:
        """Everything learned under `root`: knowledge versions, beliefs, the influence log, the research queue, health, breaks,
        unknowns, the graph, the contradiction monitor, the health dashboard and this report.  Experiments, scorecards and the
        failed-learner registry append to their own files as they happen."""
        if self.root is None:
            return {}
        L, r = self.L, self.root
        out: dict[str, Any] = {"knowledge": L.store.dump(r / "knowledge.jsonl")}
        tmp = r / "beliefs.jsonl.part"
        L.beliefs.save_jsonl(tmp)
        os.replace(tmp, r / "beliefs.jsonl")
        _atomic_write(r / "decision_log.json", canonical_json(DC.dump_log(L.decision_log)))
        _atomic_write(r / "research_queue.json", RPR.queue_snapshot(L.research.queue))
        _atomic_write(r / "health.json", json.dumps(HE.export_book(self.health.book), sort_keys=True))   # exact floats: ids re-derive
        _atomic_write(r / "breaks.json", canonical_json(self.breaks.ledger._rows))
        _atomic_write(r / "unknowns.json", canonical_json(self.unknowns.to_dict()))
        _atomic_write(r / "graph.json", canonical_json(L.graph.to_records(pd.Timestamp(as_date(now)) + pd.Timedelta(days=1))))
        if self.contra_last is not None:
            _atomic_write(r / "contradiction_monitor.json", self.contra_monitor.dumps(self.contra_last))
        _atomic_write(r / "boundary_knowledge.json", canonical_json({k: v for k, v in sorted(self.boundary_shelf.items())}))
        tmp = r / "staged_beliefs.jsonl.part"
        self.staged.save_jsonl(tmp)
        os.replace(tmp, r / "staged_beliefs.jsonl")
        self.dashboard = self.write_dashboard(now)
        _atomic_write(r / "loop_report.json", canonical_json(self.report()))
        out["files"] = len(list(r.glob("*")))
        self.persisted = {"at": str(as_date(now)), **out}
        self._fire("persist")
        return out

    def write_dashboard(self, now) -> dict:
        """reports.health_dashboard over the artefacts just written, merged with the contradiction monitor's dashboard rows."""
        if self.root is None:
            return {}
        rd = self.root / "reports"
        rd.mkdir(parents=True, exist_ok=True)
        self.L.store.dump(rd / RE.ARTEFACTS["knowledge"])
        rows: list[dict] = []
        if self.contra_last is not None:
            for t in self.contra_last.items:
                rows.append({"pair": list(t.pair), "verdict": "OPEN" if t.is_open else t.phase.value, "at": str(as_date(t.evidence_through))})
        _atomic_write(rd / RE.ARTEFACTS["contradictions"], "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        _atomic_write(rd / RE.ARTEFACTS["queue"], RPR.queue_snapshot(self.L.research.queue))
        ctx = RE.ReportContext(rd, pd.Timestamp(as_date(now)) + pd.Timedelta(days=1), checklist=None, code_hash=self.L.code_hash)
        dash = RE.health_dashboard(ctx)
        if self.contra_last is not None:
            dash["contradiction_rows"] = self.contra_monitor.dashboard_rows(self.contra_last)["rows"]
        _atomic_write(self.root / "dashboard.json", canonical_json(dash))
        self._fire("health_dashboard")
        return dash

    def report(self) -> dict:
        a = self.value_last
        return {"label": LABEL, "fired": dict(sorted(self.fired.items())), "rows": len(self.rows),
                "open_experiments": len(self.open_exps), "experiment_results": self.exp_results[-10:], "dead_ends": self.dead_ends[-10:],
                "failed_learners_seeded": self.failed_seed, "lifecycle_changes": self.lifecycle_verdicts[-10:],
                "health_relabels": self.health_applied[-10:], "retire_verdicts": self.retire_verdicts[-10:],
                "credit_beliefs": self.proposals_applied[-10:], "masked_pairs": len(self.masked),
                "redundancy_pairs": 0 if self.redundancy_last is None else len(self.redundancy_last.pairs),
                "contradictions_open": 0 if self.contra_last is None else len(self.contra_last.open_items()), "contradiction_pairs_ever": self.contra_keys,
                "unknowns_open": len(self.unknowns.open_rows()), "next_tests": {self.token(k): v[0] for k, v in sorted(self.next_tests.items())},
                "arenas": {self.token(k): {"leader": v["leader"], "separated": v["separated"]} for k, v in sorted(self.arenas.items())},
                "boundary_knowledge": len(self.boundary_shelf), "calibration_cache": len(self.cal),
                "value": None if a is None else {"portfolio_accept": a.portfolio_accept, "n_dates": a.n_dates},
                "staged_beliefs": len(self.staged.subjects()), "claims": dict(list(self.claims.items())[-5:]),
                "scorecards": list(self.scorecard_ids[-5:]), "scorecard_refusals": self.scorecard_refusals[-3:], "readiness": self.readiness_log[-5:],
                "conflict_last": list(self.conflicts.values())[-1] if self.conflicts else None, "persisted": dict(self.persisted)}

    def digest(self) -> str:
        return stable_hash({"fired": self.fired, "rows": len(self.rows), "lifecycle": self.lifecycle_verdicts, "health": self.health_applied,
                            "credit": [(p["subject"], round(p["mean"], 10)) for p in self.proposals_applied],
                            "experiments": self.exp_results, "unknowns": self.unknown_rank})


def _learned_version(k: KN.KnowledgeObject, at: str) -> KN.KnowledgeObject:
    """A version written from evidence that matured on `at` says so in its provenance (the archive refuses a record whose
    learned_at precedes the evidence it carries); the chain link to the parent is untouched."""
    la = max(str(as_date(at)), str(as_date(k.provenance.learned_at)))
    prov = dataclasses.replace(k.provenance, learned_at=la, outcomes_seen_through=max(la, k.provenance.outcomes_seen_through or la))
    return dataclasses.replace(k, provenance=prov).assert_valid()


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def _context_set(ranges: Mapping[str, tuple]) -> KN.ContextSet | None:
    """{'vix': (lo, hi)} from a validated break condition -> a knowledge ContextSet on the situation's own feature paths."""
    feat = {c: (path, dim) for c, _d, path, dim in BREAK_COLUMNS}
    conds = []
    for col, (lo, hi) in sorted(ranges.items()):
        if col not in feat:
            continue
        path, dim = feat[col]
        if lo is not None and hi is not None:
            conds.append(KN.Condition(dim, path, "between", (float(min(lo, hi)), float(max(lo, hi)))))
        elif lo is not None:
            conds.append(KN.Condition(dim, path, "ge", (float(lo),)))
        elif hi is not None:
            conds.append(KN.Condition(dim, path, "le", (float(hi),)))
    return KN.ContextSet(tuple(conds)) if conds else None


def trust_learned_signal(train: pd.DataFrame):
    """The scorecard's re-trainable learner B: take the loop's pick where its learned expectation was positive, but only if on the
    training units doing so earned more than abstaining.  Reads features only at prediction time (never alt / base)."""
    pos = train["signal"].to_numpy(float) > 0
    gain = (train["alt"].to_numpy(float) - train["base"].to_numpy(float))[pos]
    ok = bool(len(gain) >= 5 and gain.mean() > 0)
    return lambda t: ((t["signal"].to_numpy(float) > 0) & ok).astype(float)


# ---------------------------------------------------------------------------------------------------------------- trusted side
class RunGuard:
    """Checkpointing around one test-path run (section 58): `start` resumes only a checkpoint written by the same code
    (checkpoints.resume_verified; a stale one is discarded, never trusted), `tick` saves progress on a cadence, `interrupted`
    writes the section-58 interruption record, and `job` describes the run as a resources.Job (compute.job_for) so a scheduler
    can relaunch it in its own process.  Lives on the trusted side: nothing here is handed to the trader."""

    def __init__(self, root: str | Path, run_id: str, code_hash: str, every: int = 20):
        self.root = Path(root)
        self.run_id = safe_token(run_id)
        self.code_hash = code_hash
        self.every = max(1, int(every))
        self.store = CK.CheckpointStore(self.root / "checkpoints", self.run_id)
        self.plan: CK.ResumePlan | None = None
        self.stale = ""
        self.n = 0
        self.last: CK.ExecutionState | None = None
        self.spec: CO.ExperimentSpec | None = None
        self.job_obj: Any = None

    def start(self, now) -> dict:
        try:
            self.plan = CK.resume_verified(self.store, self.code_hash)
        except CK.StaleState as e:
            self.stale = str(e)[:300]
            self.plan = None
        prev = None if self.plan is None or self.plan.state is None else self.plan.state.notes.get("day", None)
        self.last = self.store.save(str(as_date(now)), "test_path", "run on_tick", self.code_hash,
                                    notes={"day": "0", "resumed_from": str(prev), "stale": self.stale[:120]})
        return {"resumed_from": prev, "stale": bool(self.stale)}

    def tick(self, now, counts: Mapping[str, int]) -> None:
        self.n += 1
        if self.n % self.every == 0:
            self.last = self.store.save(str(as_date(now)), "test_path", "run on_tick", self.code_hash,
                                        current_experiment=self.spec.key if self.spec else "",
                                        notes={"day": str(self.n), **{k: str(v) for k, v in sorted(counts.items())}})

    def interrupted(self, now, why: str) -> Path:
        st = self.store.save(str(as_date(now)), "test_path", f"resume on_tick after {as_date(now)}", self.code_hash,
                             current_experiment=self.spec.key if self.spec else "",
                             failures=(CK.FailureNote(str(as_date(now)), "on_tick", "ERROR", why[:200]),), notes={"day": str(self.n)})
        return CK.write_interruption(self.root / "INTERRUPTED.json", CK.InterruptionRecord.from_state(st, why[:200]))

    def run_harness_process(self, params: Mapping[str, Any], seed: int, as_of, timeout_s: float = 900.0, free_fn=None) -> dict:
        """compute.run_experiment_process: the same-year harness run as a REAL worker process under the compute ledger (the
        2.5 GB admission rule, its own logs, the ledger settled with whatever happened)."""
        import time as _time
        spec = CO.ExperimentSpec("same_year_harness", dict(params), int(seed), str(as_date(as_of)), est_gb=1.0)
        ledger = CO.ExperimentLedger(self.root / "compute_ledger.json")
        sub = ledger.submit(spec, _time.time(), self.code_hash)
        out = CO.run_experiment_process(spec, ledger, self.root / "runs", self.root / "work", _time.time(), timeout_s=timeout_s,
                                        modules=("engine.learning.loop_hooks",), free_fn=free_fn)
        return {"submitted": sub.action, **{k: v for k, v in out.items() if k != "result"}}

    def job(self, name: str, params: Mapping[str, Any], seed: int, as_of) -> Any:
        spec = CO.ExperimentSpec(name, dict(params), int(seed), str(as_date(as_of)))
        self.spec = spec
        self.job_obj = CO.job_for(spec, self.root / "specs", self.root / "runs", self.root / "compute_ledger.json")
        return self.job_obj


def append_curve_record(path: str | Path, rec: Mapping[str, Any]) -> list[dict]:
    """One run's record appended to the lineage's curve file (append-only JSON lines); returns every record so far."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "ab") as f:
        f.write((json.dumps(dict(rec), sort_keys=True, default=str) + "\n").encode("utf-8"))
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def curve_report(recs: Sequence[Mapping[str, Any]], tag: str = "legit") -> dict:
    """learning_curve.curve_from_play_records over the runs of one lineage and learning_curve.delta_from_records over any paired
    reruns among them. A chain of different windows has no same-situation pairs, so the delta says INSUFFICIENT until blind reruns
    of one year (C54/C55) add run1/run2 records: that is the honest verdict, not a gap."""
    curve = LCV.curve_from_play_records(list(recs), tag=tag, name=f"test-path:{tag}",
                                        knowledge_of=lambda r: int(r.get("knowledge", 0)))
    delta = LCV.delta_from_records(list(recs), min_pairs=3)
    gains = curve.column("same_year_gain") if len(curve) else np.array([])
    return {"points": len(curve), "last_same_year_gain": float(gains[-1]) if len(gains) else None, "fingerprint": curve.fingerprint(),
            "delta_pairs": delta.n_pairs, "delta_states": {k: str(v) for k, v in delta.states().items()}}


@CO.register_worker("same_year_harness")
def _same_year_worker(spec, ctx) -> dict:
    """The same-year harness as a compute worker (run in its own process by RunGuard.run_harness_process)."""
    from . import planted_world as PW
    p = dict(spec.params)
    world = PW.make_world(PW.standard_spec(int(p.get("weeks", 52)), int(p.get("stocks", 40))), int(p.get("world_seed", spec.seed)))
    return run_same_year_harness(world, n_runs=int(p.get("n_runs", 3)), seed=spec.seed, modes=tuple(p.get("modes", ("fresh_plain",))))


def run_same_year_harness(world, n_runs: int = 6, seed: int = 0, probe: bool = False, modes: Sequence[str] | None = None) -> dict:
    """The C54/C55 learning harness as a callable entry: the same year replayed under disguise with the five frozen controls
    (same_year.SameYearHarness over controls.standard_controls), judged, and summarised by same_year.record_of."""
    from . import controls as CT
    from . import same_year as SY
    h = SY.SameYearHarness(world, CT.standard_controls(), SY.HarnessConfig(n_runs=n_runs, seed=seed, probe=probe))
    for m in (modes or SY.MODES):
        h.run_mode(m)
    return SY.record_of(h)
