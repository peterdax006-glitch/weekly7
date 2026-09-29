"""The learning loop as one conductor (contract C62 sections 3, 4, 66, 86, 87; checklist C01-C17, L02).
STATUS: IMPLEMENTED - NOT VALIDATED (C63: unit-tested on a planted world only; no real-data run).

Section 4 draws the loop OBSERVE -> DESCRIBE SITUATION -> RETRIEVE -> ASSESS RELIABILITY -> FORM EXPECTATIONS -> DECIDE ->
OBSERVE OUTCOME -> MEASURE SURPRISE -> ASSIGN CREDIT/BLAME -> UPDATE BELIEFS -> INVESTIGATE FAILURE -> LEARN CONDITIONS ->
LEARN ANTI-CONDITIONS -> UPDATE RELIABILITY -> TEST TRANSFER -> STORE KNOWLEDGE -> UPDATE GRAPH -> UPDATE META-KNOWLEDGE ->
SELECT NEXT RESEARCH QUESTION.  This file is the conductor only: `LegitimateLearner` has one method per stage and every stage
calls the module that owns the mechanism (situation, retrieval, reliability, retirement, calibration, temporal, decision_contract,
surprise, credit, separation, belief, failure, postmortem, missed_winners, context, boundary, contradiction, transfer,
knowledge, archive, champion, promotion, knowledge_graph, meta_learning, research_priority, research_policy, firewalls).

Shape of one cycle (an EPISODE): stages 1-6 run when the decision is made (`decide_batch`); stages 7-19 run later, when the
outcomes have matured strictly before the new `now` (`resolve_and_learn`).  A stage that is called out of order raises
StageOrderError; a stage that would touch data or memory from the future raises FirewallBreach and the step stops (fail closed).

The section-3 protocol is `run_protocol`: BEFORE (decisions of a learner that has not had the experience, with the full record
demanded by section 3) -> EXPERIENCE -> LEARNING -> AFTER (the same situations under a new identity, learner frozen) ->
VALIDATION (did behaviour change, and only where the learner used knowledge that transfers?).  `run_acceptance` is the section-87
experiment in miniature: learn on world A, disguise, meet the same situation classes in a different episode under new identities,
and measure whether the decision changed and improved against a no-lesson learner and a learner that learned from shuffled outcomes.

A frozen learner (`freeze`) records its code and config hash and refuses to run if either changes (L02).  Everything is
deterministic given `LearnerConfig.seed`.  Rows never carry a ticker: decisions are keyed by a position in a hash-ordered list."""
from __future__ import annotations

import contextlib
import dataclasses
import enum
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import archive as AR
from . import belief as BL
from . import boundary as BD
from . import calibration as CB
from . import champion as CH
from . import context as CX
from . import contradiction as CT
from . import credit as CR
from . import decision_contract as DC
from . import failure as FL
from . import firewalls as FW
from . import knowledge as KN
from . import knowledge_graph as KG
from . import meta_learning as ML
from . import missed_winners as MW
from . import postmortem as PM
from . import promotion as PR
from . import reliability as RL
from . import research_policy as RP
from . import research_priority as RPR
from . import retirement as RT
from . import retrieval as RV
from . import separation as SP
from . import situation as ST
from . import surprise as SU
from . import temporal as TP
from . import transfer as TR
from .core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Subsystem, TemporalClass,
                   as_date, current_code_hash, require_past, stable_hash)

# ---------------------------------------------------------------------------------------------------------------- vocabulary


class Stage(str, enum.Enum):
    OBSERVE = "OBSERVE"
    DESCRIBE_SITUATION = "DESCRIBE_SITUATION"
    RETRIEVE = "RETRIEVE"
    ASSESS_RELIABILITY = "ASSESS_RELIABILITY"
    FORM_EXPECTATIONS = "FORM_EXPECTATIONS"
    DECIDE = "DECIDE"
    OBSERVE_OUTCOME = "OBSERVE_OUTCOME"
    MEASURE_SURPRISE = "MEASURE_SURPRISE"
    ASSIGN_CREDIT_BLAME = "ASSIGN_CREDIT_BLAME"
    UPDATE_BELIEFS = "UPDATE_BELIEFS"
    INVESTIGATE_FAILURE = "INVESTIGATE_FAILURE"
    LEARN_CONDITIONS = "LEARN_CONDITIONS"
    LEARN_ANTI_CONDITIONS = "LEARN_ANTI_CONDITIONS"
    UPDATE_RELIABILITY = "UPDATE_RELIABILITY"
    TEST_TRANSFER = "TEST_TRANSFER"
    STORE_KNOWLEDGE = "STORE_KNOWLEDGE"
    UPDATE_GRAPH = "UPDATE_GRAPH"
    UPDATE_META = "UPDATE_META"
    SELECT_RESEARCH = "SELECT_RESEARCH"

    def __str__(self):
        return self.value


STAGES: tuple[Stage, ...] = tuple(Stage)
DECISION_STAGES = STAGES[:6]
LEARNING_STAGES = STAGES[6:]
LEAK_PREFIXES = ("canary_", "future_", "label_", "target_")     # a feature column named like this is an outcome in disguise
LABEL = "IMPLEMENTED - NOT VALIDATED"


class StageOrderError(RuntimeError):
    """A stage was called out of the section-4 order (or for an episode that is not at that stage)."""


class FrozenLearnerError(RuntimeError):
    """A learning stage was requested on a frozen learner (frozen learners only decide)."""


class LearnerChanged(FirewallBreach):
    """The frozen learner's code or config hash differs from the one recorded at freeze time (L02)."""


@dataclass(frozen=True)
class ColumnMap:
    """target = clip(offset + scale * source): how a raw panel column becomes the situation input the builder expects."""
    target: str
    source: str
    scale: float = 1.0
    offset: float = 0.0
    lo: float | None = None
    hi: float | None = None

    def apply(self, panel: pd.DataFrame) -> pd.Series:
        if self.source not in panel.columns:
            raise KeyError(f"column map {self.target}: source column {self.source!r} is not in the panel")
        s = self.offset + self.scale * panel[self.source].astype(float)
        return s.clip(lower=self.lo, upper=self.hi)


@dataclass(frozen=True)
class LearnerConfig:
    """Every knob of the conductor in one hashable place; a changed config is a different learner (L02)."""
    seed: int = 0
    horizon_days: int = 7
    column_map: tuple[ColumnMap, ...] = ()
    min_coverage: float = 0.1
    min_cs_n: int = 20
    n_quantiles: int = 5
    candidate_levels: tuple[int, ...] = (0, 4)
    candidate_features: tuple[str, ...] = ()
    top_n: int = 5
    min_expected: float = 0.0
    min_active_rows: int = 3
    belief_prior_sd: float = 0.02
    min_weeks_belief: int = 6
    min_context_obs: int = 60
    discover_every: int = 6
    credit_every: int = 8
    meta_every: int = 10
    boundary_min_weeks: int = 26
    missed_every: int = 12
    winner_thr: float = 0.03
    loss_floor: float = 0.01
    cost: float = 0.0005
    max_history_rows: int = 30000
    skill_min_n: int = 10
    retrieval: RV.RetrievalConfig = RV.RetrievalConfig(novelty_check=False)
    promotion_policy: PR.PromotionPolicy | None = None
    board_policy: CH.BoardPolicy | None = None
    credit_cfg: CR.CreditConfig | None = None
    surprise_cfg: SU.SurpriseConfig | None = None
    missed_params: MW.MissedParams | None = None
    cpu_minutes: float = 30.0

    def validate(self) -> list[str]:
        errs = []
        if self.horizon_days < 1:
            errs.append("horizon_days < 1")
        if self.top_n < 1:
            errs.append("top_n < 1")
        if self.n_quantiles < 2 or any(not 0 <= lv < self.n_quantiles for lv in self.candidate_levels):
            errs.append("candidate_levels must lie inside 0..n_quantiles-1")
        if not self.candidate_levels:
            errs.append("no candidate levels")
        if self.belief_prior_sd <= 0:
            errs.append("belief_prior_sd must be positive")
        if self.min_active_rows < 2:
            errs.append("min_active_rows < 2 gives a weekly standard error of nothing")
        if self.skill_min_n < 10:
            errs.append("skill_min_n < 10 cannot support a skill claim")
        if not 0 < self.min_coverage <= 1:
            errs.append("min_coverage outside (0, 1]")
        for cm in self.column_map:
            if cm.scale == 0:
                errs.append(f"column map {cm.target}: zero scale")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class RowDecision:
    """The section-3 BEFORE record for one candidate: decision, confidence, prediction, expected outcome, risk, selected
    patterns, retrieved memories, abstentions, uncertainty, explanation.  No ticker and no date string: a slot, and the
    identity-free situation ids."""
    slot: int
    situation_id: str
    exact_id: str
    action: str                              # LONG | ABSTAIN
    size: float
    expected: float | None                   # prediction: expected edge over the horizon
    confidence: float | None
    risk: float | None                       # predicted downside scale (one standard error of the expectation)
    uncertainty: str                         # "" or the Unknown state that made the learner step back
    pattern_ids: tuple[str, ...]             # candidate patterns active in the situation
    knowledge_ids: tuple[str, ...]           # retrieved memories that carried weight
    abstention: str                          # why not LONG ("" when LONG)
    retrieval_id: str
    influence: bool
    explanation: str

    def behaviour_key(self) -> tuple:
        return (self.action, None if self.expected is None else round(self.expected, 9), self.knowledge_ids)


@dataclass(frozen=True)
class StageEvent:
    episode: str
    stage: Stage
    ok: bool
    n: int
    note: str = ""


@dataclass(frozen=True)
class EpisodeSummary:
    episode: str
    decided_on: str
    n_rows: int
    n_long: int
    n_abstain: int
    n_knowledge_used: int
    learned: bool
    matured_on: str | None = None
    mean_edge_long: float | None = None


@dataclass
class _Row:
    key: tuple                                # (date, ticker): private, never written into a record
    raw: Mapping[str, float]
    situation: ST.Situation | None = None
    members: tuple[str, ...] = ()             # candidate pattern ids active in this row
    retrieval: RV.Retrieval | None = None
    parts: list = field(default_factory=list)       # (kid, weight, expected, se)
    expected: float | None = None
    confidence: float | None = None
    risk: float | None = None
    decision: RowDecision | None = None
    edge: float | None = None
    raw_ret: float | None = None
    matured: str | None = None


@dataclass
class _Episode:
    eid: str
    now: Any
    rows: list
    cursor: int = 0
    learned: bool = False
    used: list = field(default_factory=list)      # knowledge objects that carried weight (for DECIDE bookkeeping)
    decided_kids: dict = field(default_factory=dict)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ord_hash(text: str, seed: int) -> int:
    return int(stable_hash([text, seed], 12), 16)


class LegitimateLearner:
    """One learner, one method per stage of the section-4 loop.  See the module docstring for the cycle."""

    def __init__(self, cfg: LearnerConfig | None = None, workdir: str | Path | None = None,
                 code_hash_fn: Callable[[], str] | None = None):
        self.cfg = cfg or LearnerConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid LearnerConfig: " + "; ".join(errs))
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="w7learner_")
            workdir = self._tmp.name
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._code_hash_fn = code_hash_fn or current_code_hash
        self.code_hash = self._code_hash_fn()
        self.config_hash = self.cfg.digest()
        self.frozen = False
        self._frozen_hashes: tuple[str, str] | None = None
        c = self.cfg
        # ---- perception and memory of what happened
        self.builder = ST.SituationBuilder(ST.SituationConfig(min_cs_n=c.min_cs_n, min_coverage=c.min_coverage))
        self.gate = FW.LearningFirewallGate()
        self.index = RV.KnowledgeIndex()
        self.monitor = RV.SkillMonitor(min_n=c.skill_min_n, seed=c.seed)
        self.retriever = RV.Retriever(self.index, config=c.retrieval, monitor=self.monitor)
        self.store = KN.KnowledgeStore()
        self.archive = AR.Archive(self.workdir / "archive")
        self.graph = KG.KnowledgeGraph()
        # ---- reliability, retirement, calibration, time
        self.tracker = RL.ReliabilityTracker()
        self.retirement = RT.RetirementLedger()
        self.calibration = CB.CalibrationMonitor()
        self.temporal = TP.TemporalMemory()
        # ---- learning
        self.surprise = SU.SurpriseTracker(c.surprise_cfg)
        self.beliefs = BL.BeliefLedger()
        self.context = CX.ContextModel(CX.ContextConfig(min_n=8, n_perm=100))
        self.rules = CX.RuleBook()
        self.boundaries = BD.BoundaryRegistry()
        self.contradictions = CT.ContradictionLedger()
        self.credit_ledger = CR.DecisionLedger()
        self.credit_reports: list = []
        self.subsystems = SP.SubsystemLedger()
        self.classifier = FL.LossClassifier()
        self.postmortems = PM.PostmortemStore()
        self.pm_builder = PM.PostmortemBuilder(self.classifier, code_hash=self.code_hash)
        self.hypotheses = PM.HypothesisBook()
        self.missed = MW.MissedLearningLedger(MW.RejectionAnalyzer(c.missed_params))
        self.missed_weeks: list = []
        self.transfer_ledger = TR.RuleTransferLedger()
        self.decision_log = DC.DecisionLog()
        self.meta = ML.MetaLearner(ML.MetaStore(), ML.MetaConfig(min_train=5, min_test=5, folds=2, min_family_n=3, min_group_n=2))
        self.meta_advice = None
        self.research = RPR.ResearchPriorityEngine()
        self.experiments = None
        self.next_questions: tuple = ()
        # ---- promotion machinery
        pol = c.promotion_policy or PR.PromotionPolicy()
        self.board = CH.KnowledgeBoard(self.workdir / "board.jsonl", gate=PR.PromotionGate(pol, code_hash=self.code_hash),
                                       policy=c.board_policy)
        # ---- bookkeeping
        self.trace: list[StageEvent] = []
        self.decisions: list[RowDecision] = []
        self.summaries: list[EpisodeSummary] = []
        self.pending: dict[str, _Episode] = {}
        self._n_episodes = 0
        self._weekly: dict[str, list] = {}                 # candidate pattern -> [(matured, mean_edge, se, n)]
        self._history: list = []                           # resolved rows kept for retro-fitting support to newborn knowledge
        self._kid_of: dict[str, str] = {}                  # pattern id -> knowledge id
        self._pid_of: dict[str, str] = {}
        self._registered_rids: set[str] = set()
        self._features: tuple[str, ...] = tuple(c.candidate_features)
        self.last_learned_on: Any = None
        self.counters: dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------ guards
    def freeze(self) -> "LegitimateLearner":
        """L02: from now on the learner only decides, and only while its code and config are what they were."""
        self.frozen = True
        self._frozen_hashes = (self._code_hash_fn(), self.cfg.digest())
        return self

    def verify_frozen(self) -> None:
        if not self.frozen:
            return
        code, cfg = self._code_hash_fn(), self.cfg.digest()
        if (code, cfg) != self._frozen_hashes:
            raise LearnerChanged(f"frozen learner changed: code {self._frozen_hashes[0]}->{code}, config {self._frozen_hashes[1]}->{cfg}")

    def _count(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    @contextlib.contextmanager
    def _stage(self, ep: _Episode, stage: Stage):
        """Enforces the section-4 order per episode and records one trace event whether or not the stage succeeds."""
        if ep.cursor >= len(STAGES) or STAGES[ep.cursor] != stage:
            want = STAGES[ep.cursor] if ep.cursor < len(STAGES) else "nothing (cycle complete)"
            raise StageOrderError(f"episode {ep.eid}: {stage} called but the loop is at {want}")
        if stage in LEARNING_STAGES and self.frozen:
            raise FrozenLearnerError(f"{stage} would learn; the learner is frozen")
        box = {"n": 0, "note": ""}
        try:
            yield box
        except BaseException as e:
            self.trace.append(StageEvent(ep.eid, stage, False, box["n"], f"{type(e).__name__}: {e}"[:200]))
            raise
        ep.cursor += 1
        self.trace.append(StageEvent(ep.eid, stage, True, box["n"], box["note"]))

    def _admit(self, now, subject: str, **kw) -> None:
        """The firewall gate (contract 55) at a stage that consumes data or memory; a failed layer stops the step."""
        ctx = FW.GateContext(now=now, subject=subject, **kw)
        self.gate.admit(ctx)

    # ------------------------------------------------------------------------------------------------ candidate patterns
    def _candidate_features(self, panel: pd.DataFrame) -> tuple[str, ...]:
        if not self._features:
            self._features = tuple(c for c in panel.columns if not str(c).startswith("m_") and pd.api.types.is_numeric_dtype(panel[c]))
        return self._features

    def _levels(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional quantile level (0 = lowest) of every candidate feature, per date: identity-free membership."""
        feats = self._candidate_features(panel)
        n_q = self.cfg.n_quantiles
        out = {}
        for f in feats:
            r = panel[f].groupby(level=0).rank(method="first")
            n = panel[f].groupby(level=0).transform("count")
            out[f] = (((r - 1) * n_q) // n).clip(0, n_q - 1)
        return pd.DataFrame(out, index=panel.index)

    def pattern_id(self, feature: str, level: int) -> str:
        return f"{feature}:q{int(level)}"

    def _members(self, lv_row: pd.Series) -> tuple[str, ...]:
        want = set(self.cfg.candidate_levels)
        return tuple(self.pattern_id(f, int(v)) for f, v in lv_row.items() if int(v) in want)

    def _direction(self, pid: str) -> int:
        st = self.beliefs.current(pid) if pid in self.beliefs.subjects() else None
        return 1 if st is None or st.mean >= 0 else -1

    # ------------------------------------------------------------------------------------------------ 1. OBSERVE
    def stage_observe(self, ep: _Episode, panel: pd.DataFrame) -> None:
        """C01. Take in the cross-section of `now`.  Anything dated after `now`, a column that is an outcome in disguise, or a
        panel the DATA/TIME firewalls reject stops the step here."""
        with self._stage(ep, Stage.OBSERVE) as box:
            now = ep.now
            bad = [c for c in panel.columns if str(c).lower().startswith(LEAK_PREFIXES)]
            if bad:
                raise FirewallBreach(f"observe: feature columns {bad} look like outcomes ({LEAK_PREFIXES}); refused")
            if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
                raise ValueError("observe: the panel must be indexed by (date, ticker)")
            dates = pd.DatetimeIndex(panel.index.get_level_values(0))
            if len(dates) and dates.max() > pd.Timestamp(as_date(now)):
                raise FirewallBreach(f"observe: rows dated {dates.max().date()} are after now={as_date(now)}")
            if len(dates) and dates.min() < pd.Timestamp(as_date(now)):
                raise ValueError("observe: a decision batch is the cross-section of `now` only")
            self._admit(now, "observe", X=panel, relevant=frozenset({FW.LayerName.DATA, FW.LayerName.TIME}))
            lv = self._levels(panel)
            for key, raw in zip(panel.index, panel.to_dict("records")):
                ep.rows.append(_Row(key, raw, members=self._members(lv.loc[key])))
            box["n"] = len(ep.rows)

    # ------------------------------------------------------------------------------------------------ 2. DESCRIBE SITUATION
    def _situation_input(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.column_map:
            return panel
        mapped = {cm.target: cm.apply(panel) for cm in self.cfg.column_map}
        return pd.DataFrame(mapped, index=panel.index)

    def stage_describe(self, ep: _Episode, panel: pd.DataFrame) -> None:
        """C02. A Situation per row: what is happening, never which stock or which date (section 15)."""
        with self._stage(ep, Stage.DESCRIBE_SITUATION) as box:
            known = {p: self._direction(p) for p in self._pid_of.values()}
            by_row = {r.key: [ST.ActivePattern(p, 1.0, known[p]) for p in r.members if p in known] for r in ep.rows}
            sits = self.builder.build_panel(self._situation_input(panel), ep.now, patterns_by_row=by_row)
            for r, sit in zip(ep.rows, sits.reindex([r.key for r in ep.rows]).tolist()):
                r.situation = sit
            unusable = sum(1 for r in ep.rows if not r.situation.usable(self.cfg.min_coverage))
            box["n"], box["note"] = len(ep.rows), f"{unusable} below coverage {self.cfg.min_coverage}"
            self._count("unusable_situations", unusable)

    # ------------------------------------------------------------------------------------------------ 3. RETRIEVE
    def stage_retrieve(self, ep: _Episode) -> None:
        """C03. Knowledge that could exist at `now`, ranked for each situation.  Abstains (influence=False) without proven skill."""
        with self._stage(ep, Stage.RETRIEVE) as box:
            visible = self.store.visible(ep.now)
            self._admit(ep.now, "retrieve", items=visible, all_items=self.store.visible(ep.now),
                        relevant=frozenset({FW.LayerName.MEMORY}))
            n_items = 0
            for r in ep.rows:
                if not r.situation.usable(self.cfg.min_coverage):
                    continue
                r.retrieval = self.retriever.retrieve(r.situation, ep.now)
                n_items += len(r.retrieval.items)
            box["n"], box["note"] = n_items, f"{len(visible)} visible knowledge objects"

    # ------------------------------------------------------------------------------------------------ 4. ASSESS RELIABILITY
    def item_weight(self, kid: str, now, ctx_now: Mapping[str, Any] | None = None) -> tuple[float, str]:
        """How much a retrieved item may count today: the five reliability dimensions (section 11) x its retirement state x its
        temporal profile.  Zero is a real answer (STANDBY / UNTESTED / retired) and is reported with its reason."""
        if not self.tracker.known(kid):
            return 0.0, "no reliability record"
        st = self.tracker.state(kid, now, ctx_now)
        dec = RL.decide(st)
        w = dec.weight
        if self.retirement.known(kid):
            w *= self.retirement.influence(kid, now, st.current_reliability)
        prof = self.temporal.get(kid, now)
        if prof is not None:
            w *= TP.expected_influence(prof, now)
        return float(max(0.0, min(1.0, w))), dec.action

    def stage_assess(self, ep: _Episode) -> None:
        """C11 (read side). Attach a weight to every retrieved item; weights come from evidence dated before `now` only."""
        with self._stage(ep, Stage.ASSESS_RELIABILITY) as box:
            cache: dict[str, tuple[float, str]] = {}
            n = 0
            for r in ep.rows:
                if r.retrieval is None:
                    continue
                ctx_now = self._ctx_now(r)
                for it in r.retrieval.items:
                    if it.knowledge_id not in cache or ctx_now:
                        cache[it.knowledge_id] = self.item_weight(it.knowledge_id, ep.now, ctx_now)
                    w, why = cache[it.knowledge_id]
                    r.parts.append([it.knowledge_id, w, it.expected_edge, why])
                    n += 1
            box["n"] = n

    def _ctx_now(self, r: _Row) -> dict[str, float]:
        vix = r.situation.get("market.vix")
        return {} if vix is None else {"market.vix": float(vix)}

    # ------------------------------------------------------------------------------------------------ 5. FORM EXPECTATIONS
    def stage_expectations(self, ep: _Episode) -> None:
        """C04. One expected edge per row: each retrieved pattern's context-aware estimate (context.py: rules first, then the
        pooled ladder), combined by reliability weight.  No weight -> no expectation (UNKNOWN is an answer, not a zero)."""
        with self._stage(ep, Stage.FORM_EXPECTATIONS) as box:
            formed = 0
            scale = self.surprise.scale_at(ep.now)[0]
            for r in ep.rows:
                num = den = 0.0
                var = 0.0
                kept = []
                for kid, w, sup_edge, why in r.parts:
                    if w <= 0:
                        continue
                    pid = self._pid_of[kid]
                    est = self.context.estimate(pid, r.situation, ep.now, self.cfg.seed) if self.context.n_obs(pid) >= self.cfg.min_context_obs else None
                    e = est.expected if est is not None and est.expected is not None else sup_edge
                    if e is None:
                        continue
                    se = est.se if est is not None and est.se is not None else scale
                    num += w * e
                    den += w
                    var += (w * se) ** 2
                    kept.append((kid, w, e, se))
                r.parts = kept
                if den > 0:
                    r.expected = num / den
                    r.risk = math.sqrt(var) / den
                    r.confidence = float(min(1.0, den / max(len(kept), 1)))
                    formed += 1
            box["n"] = formed

    # ------------------------------------------------------------------------------------------------ 6. DECIDE
    def stage_decide(self, ep: _Episode) -> list[RowDecision]:
        """C05. Only knowledge the decision contract lets act may act; the top `top_n` expectations above the floor go LONG,
        everything else ABSTAINS with the reason recorded (section 3 BEFORE record)."""
        with self._stage(ep, Stage.DECIDE) as box:
            for r in ep.rows:
                r.parts = [p for p in r.parts if self._contract_allows(p[0], ep.now)] if r.parts else []
                if not r.parts:
                    r.expected = r.confidence = r.risk = None
            cand = [r for r in ep.rows if r.expected is not None and r.retrieval is not None and r.retrieval.influence
                    and r.expected > self.cfg.min_expected]
            cand.sort(key=lambda r: (-r.expected, _ord_hash(r.situation.exact_id, self.cfg.seed)))
            chosen = {id(r) for r in cand[: self.cfg.top_n]}
            order = sorted(range(len(ep.rows)), key=lambda i: _ord_hash(ep.rows[i].situation.exact_id, self.cfg.seed))
            slot_of = {i: s for s, i in enumerate(order)}
            out = []
            for i, r in enumerate(ep.rows):
                out.append(self._row_decision(r, slot_of[i], id(r) in chosen, len(chosen)))
                r.decision = out[-1]
            self._log_influence(ep, out)
            ep.rows.sort(key=lambda r: r.decision.slot)
            self.decisions.extend(sorted(out, key=lambda d: d.slot))
            self.summaries.append(EpisodeSummary(ep.eid, str(as_date(ep.now)), len(out), sum(d.action == "LONG" for d in out),
                                                 sum(d.action != "LONG" for d in out), sum(bool(d.knowledge_ids) for d in out), False))
            self._register_predictions(ep)
            box["n"], box["note"] = len(out), f"{len(chosen)} long"
            return sorted(out, key=lambda d: d.slot)

    def _contract_allows(self, kid: str, now) -> bool:
        k = self.store.as_of(kid, now)
        return k is not None and DC.check(k, now).allowed

    def _row_decision(self, r: _Row, slot: int, long: bool, n_long: int) -> RowDecision:
        ret = r.retrieval
        kids = tuple(sorted({p[0] for p in r.parts}))
        if long:
            why, action, size = "", "LONG", 1.0 / max(n_long, 1)
        else:
            action, size = "ABSTAIN", 0.0
            if r.situation is None or not r.situation.usable(self.cfg.min_coverage):
                why = "situation below coverage"
            elif ret is None or not ret.items:
                why = f"no knowledge retrieved ({ret.unknown if ret is not None else 'unusable'})"
            elif not ret.influence:
                why = f"retrieval has not earned influence: {ret.withheld_reason}"
            elif r.expected is None:
                why = "retrieved knowledge carried no reliable weight"
            elif r.expected <= self.cfg.min_expected:
                why = f"expected edge {r.expected:+.4f} not above {self.cfg.min_expected}"
            else:
                why = "outside the top picks"
        unc = "" if r.expected is not None else str(ret.unknown or "UNKNOWN") if ret is not None else "INSUFFICIENT_DATA"
        return RowDecision(slot, r.situation.situation_id, r.situation.exact_id, action, size, r.expected, r.confidence, r.risk, unc,
                           r.members, kids, why, ret.retrieval_id if ret is not None else "", bool(ret.influence) if ret is not None else False,
                           ret.explain()[:240] if ret is not None else "")

    def _log_influence(self, ep: _Episode, decs: list) -> None:
        """Every knowledge item that carried weight is written to the hash-chained influence log (contract section 43)."""
        for r in ep.rows:
            if r.decision.knowledge_ids:
                items = [self.store.as_of(k, ep.now) for k in r.decision.knowledge_ids]
                self.decision_log.record_all(f"{ep.eid}#{r.decision.slot}", [i for i in items if i is not None], r.situation.bins(), ep.now)

    def _register_predictions(self, ep: _Episode) -> None:
        for r in ep.rows:
            if r.retrieval is None or not r.retrieval.items or r.retrieval.retrieval_id in self._registered_rids:
                continue
            self._registered_rids.add(r.retrieval.retrieval_id)
            RV.register_prediction(r.retrieval, self.monitor, ep.now, self.index)
            self._count("predictions")

    # ------------------------------------------------------------------------------------------------ 7. OBSERVE OUTCOME
    def stage_outcome(self, ep: _Episode, now, outcomes: pd.DataFrame) -> None:
        """C06. The realised return of every decision, accepted only if it matured strictly before `now` (require_past) and after
        the decision itself.  Excess edge = return minus the episode's cross-sectional mean (the benchmark is the same rows)."""
        with self._stage(ep, Stage.OBSERVE_OUTCOME) as box:
            need = {"ret", "matured"}
            if outcomes is None or not need <= set(outcomes.columns):
                raise ValueError(f"outcomes need columns {sorted(need)}")
            got = [r for r in ep.rows if r.key in outcomes.index]
            for r in got:
                o = outcomes.loc[r.key]
                require_past(o["matured"], now, f"outcome of a decision made {as_date(ep.now)}")
                if as_date(o["matured"]) <= as_date(ep.now):
                    raise FirewallBreach(f"outcome matured {as_date(o['matured'])} is not after its decision {as_date(ep.now)}")
                r.raw_ret, r.matured = float(o["ret"]), str(as_date(o["matured"]))
            if got:
                keys = pd.MultiIndex.from_tuples([r.key for r in got])
                self._admit(now, "observe_outcome", X=pd.DataFrame({"_": np.zeros(len(got))}, index=keys),
                            label_close=pd.DatetimeIndex([pd.Timestamp(r.matured) for r in got]), relevant=frozenset({FW.LayerName.TIME}))
            done = [r for r in got if r.raw_ret is not None and math.isfinite(r.raw_ret)]
            if len(done) < max(2, len(ep.rows) // 2):
                raise ValueError(f"only {len(done)} of {len(ep.rows)} outcomes present: refusing to learn from a biased remainder")
            mean = float(np.mean([r.raw_ret for r in done]))
            for r in done:
                r.edge = r.raw_ret - mean
            ep.rows = done
            box["n"], box["note"] = len(done), f"matured {max(r.matured for r in done)}"

    # ------------------------------------------------------------------------------------------------ 8. MEASURE SURPRISE
    def _cell(self, r: _Row) -> str:
        b = r.situation.bins()
        return f"ctx={b.get(self.cfg.context_dim, 'na')}|brd={b.get('breadth.state', 'na')}"

    def stage_surprise(self, ep: _Episode, now) -> None:
        """C07. Expected vs realised per situation cell, one aggregate per cell per week (rows of one week are not independent)."""
        with self._stage(ep, Stage.MEASURE_SURPRISE) as box:
            cells: dict[str, list] = {}
            for r in ep.rows:
                if r.expected is not None:
                    cells.setdefault(self._cell(r), []).append(r)
            n = 0
            for cell, rows in sorted(cells.items()):
                kids = sorted({k for r in rows for k, *_ in r.parts})
                self.surprise.observe(cell, float(np.mean([r.expected for r in rows])), float(np.mean([r.edge for r in rows])),
                                      ep.now, max(r.matured for r in rows), now, knowledge_ids=kids)
                n += 1
            box["n"] = n
            self._count("surprise_records", n)

    # ------------------------------------------------------------------------------------------------ 9. ASSIGN CREDIT / BLAME
    def _trade(self, ep: _Episode, r: _Row) -> FL.TradeRecord:
        kids = tuple(sorted(k for k, *_ in r.parts))
        return FL.TradeRecord(rid=f"{ep.eid}#{r.decision.slot}", decided_at=str(as_date(ep.now)), resolved_at=r.matured, side=1,
                              pnl=r.edge - self.cfg.cost, decided_by=Subsystem.SELECTION, cost=self.cfg.cost, signal_ret=r.raw_ret,
                              exp_ret=r.expected, exp_move=None if r.expected is None else abs(r.expected), exp_vol=r.risk,
                              universe_ret=r.raw_ret - r.edge, knowledge_ids=kids, pattern_ids=tuple(r.members),
                              tags={"ctx": self._cell(r)})

    def stage_credit(self, ep: _Episode, now) -> None:
        """C08/C09. Credit to the components of the decision (Shapley over the decision ledger) and blame to the subsystem that
        made the error (separation), so one failure never teaches the wrong subsystem."""
        with self._stage(ep, Stage.ASSIGN_CREDIT_BLAME) as box:
            n = 0
            for r in ep.rows:
                if r.expected is None:
                    continue
                kw = {kid: w for kid, w, *_ in r.parts}
                self.credit_ledger.add(CR.Decision(f"{ep.eid}#{r.decision.slot}", str(as_date(ep.now)), r.matured,
                                                   {"pattern": float(r.expected), "risk": 1.0}, float(r.edge),
                                                   {"ctx": r.situation.bins().get(self.cfg.context_dim, "na")}, {"pattern": kw}))
                n += 1
            blamed = 0
            for r in ep.rows:
                if r.decision.action == "LONG":
                    self.subsystems.add(self._trade(ep, r))
                    blamed += 1
            cc = self.cfg.credit_cfg or CR.CreditConfig(n_boot=80, n_perm=40, min_decisions=40, min_groups=6)
            if self._n_episodes % self.cfg.credit_every == 0 and len(self.credit_ledger) >= cc.min_decisions:
                eng = CR.CreditEngine(CR.WeightedSumCombiner({"pattern": 1.0}), cc)
                self.credit_reports.append(eng.assess(self.credit_ledger, now))
            box["n"], box["note"] = n, f"{blamed} trades attributed"

    # ------------------------------------------------------------------------------------------------ 10. UPDATE BELIEFS
    def _n_candidates(self) -> int:
        return max(1, len(self._features) * len(self.cfg.candidate_levels))

    def stage_beliefs(self, ep: _Episode, now) -> dict[str, Epistemic]:
        """C10. Every candidate pattern that fired gets one piece of evidence (this week's mean edge over its rows) and its belief
        is UPDATED, never overwritten (section 32).  The multiplicity burden n_trials is the number of patterns searched."""
        with self._stage(ep, Stage.UPDATE_BELIEFS) as box:
            edges = np.array([r.edge for r in ep.rows])
            sd_all = float(edges.std(ddof=1)) if len(edges) > 2 else 0.0
            matured = max(r.matured for r in ep.rows)
            by_pid: dict[str, list[float]] = {}
            for r in ep.rows:
                for p in r.members:
                    by_pid.setdefault(p, []).append(r.edge)
            out: dict[str, Epistemic] = {}
            for pid, vals in sorted(by_pid.items()):
                if len(vals) < self.cfg.min_active_rows or sd_all <= 0:
                    continue
                if pid not in self.beliefs.subjects():
                    self.beliefs.register(pid, 0.0, self.cfg.belief_prior_sd)
                est, se = float(np.mean(vals)), sd_all / math.sqrt(len(vals))
                self._weekly.setdefault(pid, []).append((matured, est, se, len(vals)))
                st = self.beliefs.update(pid, BL.Evidence(pid, matured, est, se, len(vals), BL.EvidenceKind.OOS_TEST,
                                                          n_trials=self._n_candidates(), source="weekly-cross-section"), now)
                out[pid] = BL.belief_status(st)[0]
            self._epistemic.update(out)
            box["n"] = len(out)
            box["note"] = f"{sum(v == Epistemic.SUPPORTED for v in out.values())} supported"
            return out

    # ------------------------------------------------------------------------------------------------ 11. INVESTIGATE FAILURE
    def _env(self, ep: _Episode) -> FL.FailureEnv:
        hist = {}
        for pid, wk in self._weekly.items():
            if len(wk) >= 3:
                hist[pid] = FL.PatternHistory(pid, tuple(w[1] for w in wk), tuple(w[0] for w in wk), tuple(w[3] for w in wk))
        know = {k.knowledge_id: k for k in self.store.visible(ep.now)}
        return FL.FailureEnv(patterns=hist, knowledge=know)

    def stage_failures(self, ep: _Episode, now) -> None:
        """D01-D14. Every meaningful loss of a knowledge-driven pick is classified, attributed and written as a postmortem that only
        proposes hypotheses (never changes behaviour); the week's candidates go to the missed-winner analysis."""
        with self._stage(ep, Stage.INVESTIGATE_FAILURE) as box:
            env, n_pm = self._env(ep), 0
            for r in ep.rows:
                if r.decision.action != "LONG" or r.edge >= -self.cfg.loss_floor:
                    continue
                t = self._trade(ep, r)
                cls = self.classifier.classify(t, env, now)
                self.failures.add(cls, t.tags)
                if not cls.meaningful:
                    continue
                uses = [PM.KnowledgeUse(env.knowledge[k], "pattern", w) for k, w, *_ in r.parts if k in env.knowledge]
                pm = PM.build_postmortem(t, cls, SP.attribute(t, cls), env, now, uses, salt="learner", code_hash=self.code_hash)
                self.postmortems.append(pm)
                self.hypotheses.add(pm, t.loss)
                self._failure_rows.append({"knowledge_id": t.knowledge_ids[0] if t.knowledge_ids else "none", "when": t.resolved_at,
                                           "loss_share": min(1.0, t.loss / 0.05), "subsystem": str(cls.top_subsystem() or ""),
                                           "cause": str(cls.cause), "explained": cls.named})
                n_pm += 1
            self._missed_week(ep, now)
            box["n"], box["note"] = n_pm, f"{len(self.missed_weeks)} weeks in the missed-winner ledger"

    def _missed_week(self, ep: _Episode, now) -> None:
        ranked = sorted([r for r in ep.rows if r.expected is not None], key=lambda r: -r.expected)
        rank = {id(r): i + 1 for i, r in enumerate(ranked)}
        cands = []
        for r in ep.rows:
            feats = {p: float(v) for p in _NUMERIC_PATHS if (v := r.situation.get(p)) is not None}
            cands.append(MW.Candidate(f"c{r.decision.slot}", feats, fwd=r.edge, picked=r.decision.action == "LONG", score=r.expected,
                                      rank=rank.get(id(r)), eligible=True, confidence=r.confidence))
        wk = MW.Week(ep.eid, str(as_date(ep.now)), max(r.matured for r in ep.rows), tuple(cands), k=self.cfg.top_n, thr=self.cfg.winner_thr)
        if wk.validate():
            raise ValueError("; ".join(wk.validate()))
        self.missed.add_week(wk)
        self.missed_weeks.append(wk)
        p = self.cfg.missed_params or MW.MissedParams(n_perm=60, n_boot=40, boot=200)
        if len(self.missed_weeks) >= self.cfg.missed_train_min + 12 and self._n_episodes % self.cfg.missed_every == 0:
            self.missed_report = MW.walk_forward_distinctions(self.missed_weeks, p, self.cfg.seed, train_min=self.cfg.missed_train_min,
                                                              fold_len=10, now=now)

    # ------------------------------------------------------------------------------------------------ 12. LEARN CONDITIONS
    def _knowledge_context(self, rule: CX.ContextRule) -> KN.ContextSet:
        conds = []
        for c in rule.spec.conditions:
            dim = c.path.split(".")[0]
            dim = dim if dim in KN.DIMENSIONS else "market"
            conds.append(KN.Condition(dim, c.path, "not_in" if c.negate else "in", labels=tuple(c.allowed)))
        return KN.ContextSet(tuple(conds))

    def stage_conditions(self, ep: _Episode, now) -> None:
        """C12. Where does each pattern work?  Outcomes go to the context model (both sides: inside AND outside the context are
        learned, section 8); at a fixed cadence confirmed CONTEXT rules and weekly boundaries are learned for the stored items."""
        with self._stage(ep, Stage.LEARN_CONDITIONS) as box:
            stored = set(self._pid_of.values())
            obs = [CX.Obs(p, r.situation, r.edge, r.matured) for r in ep.rows for p in r.members if p in stored]
            self.context.add_many(obs)
            vix = [v for r in ep.rows if (v := r.situation.get("market.vix")) is not None]
            brd = [v for r in ep.rows if (v := r.situation.get("breadth.breadth")) is not None]
            self._mkt[max(r.matured for r in ep.rows)] = {"vix": float(np.mean(vix)) if vix else np.nan,
                                                          "breadth": float(np.mean(brd)) if brd else np.nan}
            found = 0
            if self._n_episodes % self.cfg.discover_every == 0:
                for pid in sorted(stored):
                    if self.context.n_obs(pid) < self.cfg.min_context_obs:
                        continue
                    for rule in self.context.discover(pid, now, self.cfg.seed):
                        if rule.confirmed and rule.role == "CONTEXT" and self.rules.add(rule, now):
                            found += 1
                    found += self._learn_boundaries(pid, now)
            box["n"], box["note"] = len(obs), f"{found} new conditions"

    def _learn_boundaries(self, pid: str, now) -> int:
        wk = self._weekly.get(pid, [])
        if len(wk) < self.cfg.boundary_min_weeks:
            return 0
        idx = pd.DatetimeIndex([pd.Timestamp(w[0]) for w in wk])
        edge = pd.Series([w[1] for w in wk], index=idx)
        mk = pd.DataFrame(self._mkt).T
        mk.index = pd.DatetimeIndex(mk.index)
        mk = mk.reindex(idx)
        if mk["vix"].isna().all() and mk["breadth"].isna().all():
            return 0
        bset = BD.learn_pattern_boundaries(pid, edge, now, volatility=mk["vix"].ffill().bfill(), breadth=mk["breadth"].ffill().bfill())
        self.boundaries.add(bset)
        return len(bset.boundaries)

    # ------------------------------------------------------------------------------------------------ 13. LEARN ANTI-CONDITIONS
    def stage_anti_conditions(self, ep: _Episode, now) -> None:
        """C13. Where does each pattern FAIL?  Confirmed ANTI_CONTEXT rules; opposing-sign items whose scopes overlap are triaged
        as contradictions (never averaged, section 18) and the retrieval index is told so it stops counting both."""
        with self._stage(ep, Stage.LEARN_ANTI_CONDITIONS) as box:
            found = 0
            if self._tick % self.cfg.discover_every == 0:
                for pid in sorted(self._pid_of.values()):
                    if self.context.n_obs(pid) < self.cfg.min_context_obs:
                        continue
                    for rule in self.context.rules(pid, now, self.cfg.seed):
                        if rule.confirmed and rule.role == "ANTI_CONTEXT" and self.rules.add(rule, now):
                            found += 1
            claims = []
            for kid, pid in sorted(self._pid_of.items()):
                st = self.beliefs.current(pid)
                if st is not None and st.sd > 0:
                    claims.append(CT.Claim(kid, float(st.mean), float(st.sd), int(st.n_obs), ()))
            pairs = 0
            for d in CT.triage(claims):
                if d.kind.value.startswith("SIGN") or (d.p < 0.05 and d.scope_overlap):
                    strength = float(min(1.0, abs(d.z) / 6.0))
                    self.index.set_contradiction(d.a, d.b, strength)
                    self._contradicts.add(tuple(sorted((d.a, d.b))))
                    pairs += 1
            box["n"], box["note"] = found, f"{pairs} contradicting pairs"

    # ------------------------------------------------------------------------------------------------ 14. UPDATE RELIABILITY
    def _signed(self, pid: str, dirn: int) -> tuple[list, list]:
        wk = self._weekly.get(pid, [])
        return [w[0] for w in wk], [dirn * w[1] for w in wk]

    def _direction_of(self, kid: str) -> int:
        k = self.store.latest(kid)
        return 1 if k is None or k.effect.direction >= 0 else -1

    def stage_reliability(self, ep: _Episode, now) -> None:
        """C11 (write side). Reliability is re-derived from the new week (five dimensions, never one number); the retirement gate
        may degrade or park an item; the calibration monitor scores the probabilities the picks implied; temporal profiles are
        re-estimated on a cadence."""
        with self._stage(ep, Stage.UPDATE_RELIABILITY) as box:
            matured = max(r.matured for r in ep.rows)
            n = 0
            for kid, pid in sorted(self._pid_of.items()):
                act = [r.edge for r in ep.rows if pid in r.members]
                if len(act) < self.cfg.min_active_rows:
                    continue
                dirn = self._direction_of(kid)
                val = dirn * float(np.mean(act))
                last = self.tracker._frames[kid].index.max() if self.tracker.n_obs(kid) else None
                if last is None or pd.Timestamp(matured) > pd.Timestamp(last):
                    self.tracker.update(kid, pd.DataFrame({"value": [val]}, index=pd.DatetimeIndex([matured])), now)
                    n += 1
                self._retire_check(kid, pid, dirn, now)
            edge_sd = self.surprise.scale_at(now)[0]
            for r in ep.rows:
                if r.decision.action == "LONG" and r.expected is not None:
                    p = min(0.99, max(0.01, _phi(r.expected / max(edge_sd, 1e-6))))
                    self.calibration.add(p, int(r.edge > 0), ep.now, r.matured, now, context=self._cell(r),
                                         knowledge_id=r.decision.knowledge_ids[0] if r.decision.knowledge_ids else "")
            if self._tick % self.cfg.discover_every == 0:
                for kid, pid in sorted(self._pid_of.items()):
                    dates, vals = self._signed(pid, self._direction_of(kid))
                    ser = TP.from_outcomes(dates, vals, now, period_days=7)
                    if len(ser.dates) >= 4:
                        self.temporal.add(TP.estimate(kid, ser, now))
                if len(self.calibration.records(now)) >= self.calibration.policy.min_n:
                    self.calibration_last = self.calibration.assess(now, self.cfg.seed)
            box["n"] = n

    def _retire_check(self, kid: str, pid: str, dirn: int, now) -> None:
        dates, vals = self._signed(pid, dirn)
        w = self.cfg.retire_window
        if len(dates) < w:
            return
        ev = RT.series_evidence(dates[-w:], vals[-w:], as_date(dates[-w]) - pd.Timedelta(days=1), now, "recent-weeks")
        state = self.retirement.state(kid, now)
        if state in (RT.State.ACTIVE, RT.State.DEGRADED):
            self.retirement.evaluate(kid, ev, now, apply=True)
        elif state is RT.State.DORMANT:
            self.retirement.attempt_recovery(kid, ev, now, self._mkt_now(), apply=True)

    def _mkt_now(self) -> dict:
        if not self._mkt:
            return {}
        return {"market.vix": self._mkt[max(self._mkt)]["vix"]}

    # ------------------------------------------------------------------------------------------------ 15. TEST TRANSFER
    def context_effects(self, pid: str, dirn: int) -> dict[str, tuple[float, int]]:
        """Signed mean edge of the pattern in each value of the context dimension: where does it hold up? (section 26)."""
        by: dict[str, list[float]] = {}
        for h in self._history:
            if pid in h.members:
                by.setdefault(h.ctx, []).append(dirn * h.edge)
        return {c: (float(np.mean(v)), len(v)) for c, v in sorted(by.items())}

    def stage_transfer(self, ep: _Episode, now) -> None:
        """C14. Does each stored item's edge survive away from its core situations and across context values?  Result goes to the
        item's transfer confidence and to the rule-transfer ledger; a pattern that only works in one context is labelled so."""
        with self._stage(ep, Stage.TEST_TRANSFER) as box:
            n = 0
            for kid, pid in sorted(self._pid_of.items()):
                self.transfer_ledger.register(kid)
                te = RV.transfer_evidence(self.index, kid, now, self.retriever.sim_weights, self.cfg.min_transfer_cases)
                eff = self.context_effects(pid, self._direction_of(kid))
                good = [c for c, (m, k) in eff.items() if k >= 5 and m > 0]
                self._transfer[kid] = {"evidence": te, "contexts": eff, "share_positive": len(good) / max(len(eff), 1)}
                n += te["verdict"] != "UNTESTED"
                if te["verdict"] in ("TRANSFERS", "FAILS_TO_TRANSFER", "NARROW") and kid not in self._transfer_logged:
                    self._transfer_logged.add(kid)
                    self.meta.store.add(ML.TransferRecord(kid, self.cfg.context_dim, pid.split(":")[0], te["verdict"] == "TRANSFERS",
                                                          str(max(r.matured for r in ep.rows))))
            box["n"] = n

    # ------------------------------------------------------------------------------------------------ 16. STORE KNOWLEDGE
    def _new_object(self, pid: str, st: BL.BeliefState, learned: str) -> KN.KnowledgeObject:
        feat, lvl = pid.rsplit(":q", 1)
        wk = self._weekly[pid]
        obs = f"cross-sectional quantile {lvl} of {feat} earns an excess weekly return"
        series = [(w[0], round(w[1], 12)) for w in wk]
        prov = KN.make_provenance(learned, data=series, config=self.config_hash, experiment_id=f"learner-{self.cfg.seed}",
                                  run_id=f"{self.config_hash[:8]}-{len(self._kid_of)}", seed=self.cfg.seed, outcomes_seen_through=learned,
                                  code_hash=self.code_hash)
        dirn = 1 if st.mean >= 0 else -1
        k = KN.KnowledgeObject(
            knowledge_id=KN.derive_id(obs, KN.ContextSet(), source="learner"), created_at=learned, provenance=prov, observation=obs,
            hypothesis=f"{pid} predicts the next-week excess return", effect=KN.Effect(dirn, abs(st.mean), st.sd, "excess_return", self.cfg.horizon_days),
            confidence=Confidence(truth=float(st.prob_sign_right())), evidence=KN.Evidence(int(st.n_obs), float(len(wk)), 1.0, None, learned),
            mechanism_tags=("learned", "quantile-cell"), decision_effect=(DecisionEffect.RANKING,), epistemic=Epistemic.SUPPORTED,
            lifecycle=Lifecycle.BIRTH, promotion=Promotion.RESEARCH, temporal_class=TemporalClass.UNKNOWN)
        return k.assert_valid()

    def _put(self, k: KN.KnowledgeObject) -> KN.KnowledgeObject:
        self.store.add(k)
        self.index.add_item(_Indexed(k, self._pid_of.get(k.knowledge_id, "")))
        self.archive.put_knowledge(k, occurred_at=k.updated_at)
        return k

    def _revise(self, kid: str, learned: str, reason: str, **changes) -> KN.KnowledgeObject:
        cur = self.store.latest(kid)
        try:
            nxt = cur.new_version(learned, reason, learned_at=learned, **changes)
        except KN.SchemaError:
            return cur                                     # nothing changed: a no-op version is refused by design
        self._count("versions")
        return self._put(nxt)

    def _birth(self, pid: str, st: BL.BeliefState, now, learned: str) -> None:
        k = self._new_object(pid, st, learned)
        kid = k.knowledge_id
        if kid in self._pid_of:
            return
        self._pid_of[kid], self._kid_of[pid] = pid, kid
        self._put(k)
        for h in self._history:                              # retro-fit: every earlier resolved row where the pattern was active
            if pid in h.members:
                self.index.add_support(kid, h.situation, h.matured, outcome=h.edge, ref=h.ref)
        self.tracker.register(kid)
        dates, vals = self._signed(pid, k.effect.direction)
        self.tracker.update(kid, pd.DataFrame({"value": vals}, index=pd.DatetimeIndex(dates)), now)
        self.retirement.register(kid, now)
        self._mid[kid] = self.board.register(k, CH.Slot(DecisionEffect.RANKING, Subsystem.SELECTION, pid), now)
        self.board.to_shadow(self._mid[kid], now, "born from a supported belief")
        self._birth_date[kid], self._births.append(kid)
        self.graph.add_knowledge(k, learned, label=pid)

    def _shadow_value(self, ep: _Episode, pid: str, dirn: int) -> float | None:
        act = [r.edge for r in ep.rows if pid in r.members]
        return dirn * float(np.mean(act)) if len(act) >= self.cfg.min_active_rows else None

    def _advance(self, kid: str, ep: _Episode, now, learned: str) -> None:
        """Shadow -> challenger -> champion, one gate at a time (section 44/45).  Nothing here acts on money: only the board's
        CHAMPION role reaches the retrieval index as production knowledge."""
        pid, mid = self._pid_of[kid], self._mid[kid]
        m, pol = self.board.members[mid], self.board.policy
        if m.role in (Promotion.RETIRED, Promotion.CHAMPION):
            return
        val = self._shadow_value(ep, pid, self._direction_of(kid))
        if val is not None and as_date(learned) > as_date(self._birth_date[kid]):
            self.board.record_shadow(mid, val, 0.0, now)
        if m.role == Promotion.SHADOW and len(m.shadow) >= pol.min_shadow_sessions:
            try:
                self.board.open_challenge(mid, now)
            except CH.BoardError as e:
                self._refusals.append((kid, "open_challenge", str(e)[:120]))
                return
            self._revise(kid, learned, "shadow record supports a challenge", promotion=Promotion.CHALLENGER, lifecycle=Lifecycle.GROWTH)
        elif m.role == Promotion.CHALLENGER and len(m.shadow) >= self._min_sessions_for_gate() and self._tick % self.cfg.discover_every == 0:
            k = self.store.latest(kid)
            res = self.board.attempt_promotion(k, self._promotion_evidence(kid, pid, now, learned), now)
            self._gate_log.append((kid, res["promoted"], tuple(res["decision"].critical_failures)))
            if res["promoted"]:
                self._promote(kid, learned)
            else:
                self._refusals.append((kid, "promotion", ",".join(res["decision"].critical_failures)[:120]))

    def _min_sessions_for_gate(self) -> int:
        p = self.board.gate.policy
        return max(p.min_oos_periods, p.min_delta_periods, p.min_risk_periods, p.min_stability_periods, self.board.policy.min_shadow_sessions)

    def _promote(self, kid: str, learned: str) -> None:
        st = CH.shadow_summary(self.board, self._mid[kid])
        rel = self.tracker.state(kid, pd.Timestamp(learned) + pd.Timedelta(days=1))
        cur = self.store.latest(kid)
        conf = dataclasses.replace(cur.confidence, usefulness=float(max(0.0, min(1.0, 1.0 - st.p_one_sided))),
                                   current_reliability=rel.current_reliability, failure_risk=rel.failure_risk, transfer=rel.transfer,
                                   context=rel.context)
        self._revise(kid, learned, "passed every promotion gate and the head-to-head", promotion=Promotion.CHAMPION,
                     lifecycle=Lifecycle.ACTIVE, confidence=conf)
        self._resolved_meta[kid] = True

    def _refresh_confidence(self, kid: str, now, learned: str) -> None:
        """Keep the champion's dimensions current: a new version only when a dimension moved enough to matter."""
        cur = self.store.latest(kid)
        rel = self.tracker.state(kid, pd.Timestamp(learned) + pd.Timedelta(days=1))
        new = dataclasses.replace(cur.confidence, truth=rel.truth, current_reliability=rel.current_reliability,
                                  failure_risk=rel.failure_risk, transfer=rel.transfer, context=rel.context)
        moved = max((abs((getattr(new, f) or 0.0) - (getattr(cur.confidence, f) or 0.0)) for f in ("truth", "current_reliability", "failure_risk")))
        if moved > self.cfg.version_tol:
            self._revise(kid, learned, "reliability re-measured", confidence=new)

    def _apply_rules(self, kid: str, pid: str, learned: str) -> None:
        best = {"CONTEXT": None, "ANTI_CONTEXT": None}
        for rule in self.rules.active(pid):
            cur = best[rule.role]
            if cur is None or abs(rule.diff) > abs(cur.diff):
                best[rule.role] = rule
        cur = self.store.latest(kid)
        ctx = self._knowledge_context(best["CONTEXT"]) if best["CONTEXT"] is not None else cur.contexts
        anti = self._knowledge_context(best["ANTI_CONTEXT"]) if best["ANTI_CONTEXT"] is not None else cur.anti_contexts
        if ctx != cur.contexts or anti != cur.anti_contexts:
            self._revise(kid, learned, "context rule learned from outcomes", contexts=ctx, anti_contexts=anti)

    def _retire_object(self, kid: str, learned: str, now) -> None:
        cur = self.store.latest(kid)
        if cur.promotion == Promotion.RETIRED:
            return
        mid = self._mid[kid]
        if self.board.members[mid].role != Promotion.RETIRED:
            self.board.retire(mid, FL.FailureCause.WEAKENING_EFFECT if hasattr(FL, "FailureCause") else FailureCause.WEAKENING_EFFECT,
                              "retirement gate: evidence no longer supports the item", now)
        self._put(cur.retire(learned, "retirement gate: evidence no longer supports the item"))
        self._resolved_meta[kid] = False

    def stage_store(self, ep: _Episode, now) -> None:
        """C15. Everything learned becomes versioned, provenance-stamped knowledge: births from supported beliefs, shadow ->
        challenger -> champion through the board and the ten-gate promotion check, context rules and reliability re-measured into
        new versions, retirements written as versions (never deletions), and the archive fed."""
        with self._stage(ep, Stage.STORE_KNOWLEDGE) as box:
            learned = max(r.matured for r in ep.rows)
            for r in ep.rows:
                self._history.append(_Hist(r.matured, r.edge, frozenset(r.members), r.situation, self._ident(r.key[1]),
                                           r.situation.bins().get(self.cfg.context_dim, "na"), f"{ep.eid}#{r.decision.slot}"))
                for p in r.members:
                    kid = self._kid_of.get(p)
                    if kid is not None:
                        self.index.add_support(kid, r.situation, r.matured, outcome=r.edge, ref=f"{ep.eid}#{r.decision.slot}")
            del self._history[: max(0, len(self._history) - self.cfg.max_history_rows)]
            born = 0
            for pid, status in sorted(self._epistemic.items()):
                if pid in self._kid_of or status != Epistemic.SUPPORTED or len(self._weekly.get(pid, ())) < self.cfg.min_weeks_belief:
                    continue
                self._birth(pid, self.beliefs.current(pid), now, learned)
                born += 1
            for kid, pid in sorted(self._pid_of.items()):
                self._advance(kid, ep, now, learned)
                self._apply_rules(kid, pid, learned)
                if self.store.latest(kid).promotion == Promotion.CHAMPION and self._tick % self.cfg.discover_every == 0:
                    self._refresh_confidence(kid, now, learned)
                st = self.retirement.state(kid, now)
                if st is RT.State.RETIRED:
                    self._retire_object(kid, learned, now)
                elif st is RT.State.DEGRADED and self.store.latest(kid).lifecycle == Lifecycle.ACTIVE:
                    self._revise(kid, learned, "retirement gate: evidence weakened", lifecycle=Lifecycle.DEGRADED, epistemic=Epistemic.DEGRADED)
            self.archive.log_observation(f"episode:{ep.eid}", {"n_rows": len(ep.rows), "mean_abs_edge": float(np.mean([abs(r.edge) for r in ep.rows]))},
                                         ep.now, matured_at=learned)
            box["n"], box["note"] = born, f"{len(self._pid_of)} items, {len(self.production_ids())} in production"

    def _ident(self, ticker) -> str:
        return stable_hash([str(ticker), self.cfg.seed], 10)

    def production_ids(self) -> tuple[str, ...]:
        return tuple(k for k in self._pid_of if (o := self.store.latest(k)) is not None and o.promotion == Promotion.CHAMPION)

    # ---- evidence for the ten-gate promotion check: measured here from the learner's own history, never asserted
    def _pattern_effect(self, pid: str, X: pd.DataFrame, edge: pd.Series, dirn: int) -> float:
        feat, lvl = pid.rsplit(":q", 1)
        lv = self._levels(X[[feat]])[feat]
        hit = edge[(lv == int(lvl)).reindex(edge.index, fill_value=False).to_numpy()]
        if not len(hit):
            return float("nan")
        return dirn * float(hit.groupby(level=0).mean().mean())

    def disguised_retention(self, pid: str, dirn: int) -> tuple[float, float]:
        """(identity-shuffle retention, disguised-rerun retention): the pattern's edge over the most recent audited weeks, recomputed
        after (a) opaque re-labelling of every ticker with rows re-ordered and (b) the same plus a date shift.  A rule whose
        membership consults identity or calendar collapses; a rule defined on situation features keeps 100%."""
        if not self._recent:
            return float("nan"), float("nan")
        X = pd.concat([p for p, _ in self._recent])
        edge = pd.concat([e for _, e in self._recent])
        base = self._pattern_effect(pid, X, edge, dirn)
        X2, tmap = ST.scramble_tickers(X, self.cfg.seed + 11)
        e2 = pd.Series(edge.to_numpy(), index=pd.MultiIndex.from_arrays([edge.index.get_level_values(0), edge.index.get_level_values(1).map(tmap)]))
        shuffled = self._pattern_effect(pid, X2, e2.reindex(X2.index), dirn)
        X3 = ST.shift_dates(X2, 1100)
        e3 = e2.reindex(X2.index)
        e3.index = X3.index
        disguised = self._pattern_effect(pid, X3, e3, dirn)
        if not math.isfinite(base) or base <= 0:
            return float("nan"), float("nan")
        return shuffled / base, disguised / base

    def _identity_concentration(self, pid: str, dirn: int) -> tuple[float, int]:
        by: dict[str, float] = {}
        for h in self._history:
            if pid in h.members:
                by[h.ident] = by.get(h.ident, 0.0) + dirn * h.edge
        pos = sum(v for v in by.values() if v > 0)
        return (max(by.values()) / pos if pos > 0 else 1.0), len(by)

    def _boot_value(self, sig: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return float(np.mean([rng.choice(sig, size=len(sig)).mean() for _ in range(200)]))

    def _perturbed(self, pid: str, dirn: int, sig: np.ndarray) -> list[float]:
        top = max(range(1), default=0)
        ident_rows = [(h.ident, dirn * h.edge) for h in self._history if pid in h.members]
        share: dict[str, float] = {}
        for i, e in ident_rows:
            share[i] = share.get(i, 0.0) + e
        best = max(share, key=share.get) if share else None
        drop_top = [e for i, e in ident_rows if i != best]
        cut = np.sort(sig)[: max(1, int(len(sig) * 0.9))]
        half = len(sig) // 2
        lo, hi = np.percentile(sig, [5, 95])
        return [float(cut.mean()), float(np.mean(drop_top)) if drop_top else float("nan"), float(sig[:half].mean()),
                float(sig[half:].mean()), float(np.clip(sig, lo, hi).mean()), float(np.median(sig)) + top]

    def _promotion_evidence(self, kid: str, pid: str, now, learned: str) -> PR.PromotionEvidence:
        k = self.store.latest(kid)
        dirn, birth = k.effect.direction, self._birth_date[kid]
        pol = self.board.gate.policy
        wk = self._weekly[pid]
        dates = [w[0] for w in wk]
        sig = np.array([dirn * w[1] for w in wk])
        post = np.array([as_date(d) > as_date(birth) for d in dates])
        t = PR.t_stat(sig)
        stat = PR.StatisticalEvidence(float(sig.mean()), int(sum(w[3] for w in wk)), t, PR.one_sided_p(t), self._n_candidates(), float(len(sig)))
        mem = self.board.members[self._mid[kid]]
        inc = PR.IncrementalEvidence(tuple(c - b for _, c, b in mem.shadow), self.cfg.seed, None)
        oos_sig = sig[post]
        oos = PR.OOSEvidence(str(birth), tuple(np.array(dates)[post]), tuple(float(x) for x in oos_sig), float(sig[~post].mean()) if (~post).any() else None,
                             ("post-birth",), ("pre-birth",))
        eff = self.context_effects(pid, dirn)
        home = tuple(sorted(eff, key=lambda c: -eff[c][0])[:1])
        trn = PR.TransferEvidence({c: m for c, (m, n) in eff.items()}, {c: n for c, (m, n) in eff.items()}, home, eff[home[0]][0] if home else None)
        cum = np.cumsum(oos_sig) if oos_sig.size else np.zeros(1)
        low = np.sort(oos_sig)[: max(1, int(0.05 * len(oos_sig)))] if oos_sig.size else np.zeros(1)
        risk = PR.RiskEvidence(int(oos_sig.size), float(oos_sig.min()) if oos_sig.size else 0.0, float((cum - np.maximum.accumulate(cum)).min()),
                               float(low.mean()), int((oos_sig < pol.worst_period_floor).sum()))
        shuf, disg = self.disguised_retention(pid, dirn)
        top_share, n_ident = self._identity_concentration(pid, dirn)
        memo = PR.MemorizationEvidence(shuf, disg, top_share > 0.6 or n_ident < 5, top_share, n_ident, (pid.rsplit(":q", 1)[0],)) \
            if math.isfinite(shuf) else PR.MemorizationEvidence(None, None, None, top_share, n_ident, (pid.rsplit(":q", 1)[0],))
        fut = PR.FutureAudit(True, tuple(KN.audit_future(self.store, now)), str(max(dates)), (), sum(as_date(d) >= as_date(now) for d in dates))
        data_hash = k.provenance.data_hash
        reruns = tuple(PR.RerunRecord(self._boot_value(sig, s), s, self.code_hash, data_hash) for s in (1, 2, 1, 3))
        chunks = np.array_split(sig, max(pol.min_stability_periods, 6))
        stab = PR.StabilityEvidence(tuple(float(c.mean()) for c in chunks if len(c)), tuple(x for x in self._perturbed(pid, dirn, sig) if math.isfinite(x)),
                                    float(sig.mean()))
        return PR.PromotionEvidence(stat, inc, oos, trn, risk, memo, fut, PR.ReproEvidence(reruns, data_hash), stab)

    # <<CONTINUE>>
