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
from . import experiment_memory as EM
from . import failure as FL
from . import firewalls as FW
from . import knowledge as KN
from . import knowledge_graph as KG
from . import loop_hooks as LH
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
from . import similarity as SM
from . import situation as ST
from . import surprise as SU
from . import temporal as TP
from . import transfer as TR
from . import wiring as W
from .core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle, Promotion, Subsystem, TemporalClass,
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
# held knowledge whose predictions are scored in shadow (never RETIRED: a retired item has stopped predicting)
SHADOW_PROMOTIONS = (Promotion.RESEARCH, Promotion.SHADOW, Promotion.CHALLENGER, Promotion.CHAMPION)


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
    retrieval: RV.RetrievalConfig = RV.RetrievalConfig(novelty_check=False, require_active_pattern=True)
    promotion_policy: PR.PromotionPolicy | None = None
    board_policy: CH.BoardPolicy | None = None
    credit_cfg: CR.CreditConfig | None = None
    surprise_cfg: SU.SurpriseConfig | None = None
    missed_params: MW.MissedParams | None = None
    cpu_minutes: float = 30.0
    context_dim: str = "regime.label"
    missed_train_min: int = 20
    min_transfer_cases: int = 20
    retire_window: int = 16                   # F10: 8 weeks falsely degraded every planted true item at least once in 5 years
                                              # (retirement.degrade_study; state/research/acceptance_mini/f10_degrade_study.json)
    version_tol: float = 0.05
    audit_weeks: int = 12
    transfer_every: int = 16
    similarity: SM.SimilarityWeights | None = None
    hooks: LH.HookConfig = LH.HookConfig()
    learning_claim: str = "enforce"           # PromotionGate's learning-claim gate (scorecard + firewalls + identity)
    shadow_monitoring: bool = True            # F07: retrieval skill is also scored on held, not-yet-production knowledge
    claim_evidence: bool = True               # F10: register the evidence card (shadow book + portfolio card) for the learning claim
    evidence_boot: int = 200                  # bootstrap draws of the evidence card
    evidence_window_weeks: int = 13           # portfolio-card window (a quarter); the card needs >= evidence_min_windows of them
    evidence_min_windows: int = 4
    max_book_rows: int = 60000

    def validate(self) -> list[str]:
        errs = [f"hooks: {e}" for e in self.hooks.validate()]
        if self.learning_claim not in PR.LEARNING_CLAIM_MODES:
            errs.append(f"learning_claim must be one of {PR.LEARNING_CLAIM_MODES}")
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
        if self.evidence_boot < 50 or self.evidence_window_weeks < 4 or self.evidence_min_windows < 3:
            errs.append("evidence card needs >= 50 bootstrap draws, windows of >= 4 weeks and >= 3 windows")
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
    probe_id: str                            # the situation's own content without the learner's pattern annotations
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
        return (self.action, None if self.expected is None else round(self.expected, 4), self.knowledge_ids)


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
    shadow: RV.Retrieval | None = None        # F07: retrieval over every held item, scored by the skill monitor only
    parts: list = field(default_factory=list)       # (kid, weight, expected, se)
    expected: float | None = None
    confidence: float | None = None
    risk: float | None = None
    decision: RowDecision | None = None
    edge: float | None = None
    raw_ret: float | None = None
    matured: str | None = None
    booked: bool = False                      # F10: this row belongs to the learner's shadow book
    book_exp: float | None = None             # ex-ante expectation of every held item (shadow) - recorded before the outcome exists
    book_scale: float | None = None           # the learner's edge scale at decision time (turns the expectation into a probability)


@dataclass(frozen=True)
class _BookRow:
    """One resolved unit of the learner's shadow book (F10): what all held knowledge expected before the outcome, and what happened.
    `ident` is the opaque per-seed hash the hooks already use; no ticker, no row key."""
    decided: str
    matured: str
    ident: str
    ctx: str
    expected: float | None
    scale: float | None
    edge: float
    raw_ret: float


@dataclass
class _Episode:
    eid: str
    now: Any
    rows: list
    cursor: int = 0
    learned: bool = False
    track: bool = True


_LETTERS = str.maketrans("0123456789", "ghijklmnop")


def _safe(token: str) -> str:
    """Digits to letters: the research engine's identity firewall refuses any subject that contains a year-like digit run, and a
    random hash id sometimes does by chance.  The mapping is one-to-one, so the token stays unique."""
    return str(token).translate(_LETTERS)


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
        self.retriever = _CachedRetriever(self.index, config=c.retrieval, monitor=self.monitor,
                                          sim_weights=c.similarity or RV.DEFAULT_SIMILARITY)
        # F07 root cause (state/research/acceptance_mini/diagnosis.md): the walk-forward skill gate only ever saw predictions made
        # after an item became CHAMPION, so a late promotion left it UNPROVEN on 32 rows and the lesson never acted.  The shadow
        # retriever ranks the same index with the non-production roles admitted; its predictions feed the SAME monitor before
        # their outcomes exist.  It has no monitor of its own and never reaches a decision (stage_decide reads `retrieval` only).
        self.shadow_retriever = _CachedRetriever(
            self.index, config=dataclasses.replace(c.retrieval, allowed_promotions=SHADOW_PROMOTIONS), monitor=None,
            sim_weights=c.similarity or RV.DEFAULT_SIMILARITY) if c.shadow_monitoring else None
        self.store = KN.KnowledgeStore()
        self.archive = AR.Archive(self.workdir / "archive")
        self.graph = KG.KnowledgeGraph()
        # ---- reliability, retirement, calibration, time
        self.tracker = RL.ReliabilityTracker()
        # F10: the window check reads `retire_window` outcomes; the ledger-wide floor stays at 8 because the lifecycle machine
        # (loop_hooks.lifecycle_step) writes to the same ledger on its own, shorter evidence windows
        self.retirement = RT.RetirementLedger(RT.RetirementPolicy(min_n=min(8, c.retire_window), recover_min_n=2 * c.retire_window))
        self.calibration = CB.CalibrationMonitor()
        self.temporal = TP.TemporalMemory()
        # ---- learning
        self.surprise = SU.SurpriseTracker(c.surprise_cfg)
        self.beliefs = BL.BeliefLedger()
        self.context = CX.ContextModel(CX.ContextConfig(min_n=10, n_perm=100))
        self.rules = CX.RuleBook()
        self.boundaries = BD.BoundaryRegistry()
        self.credit_ledger = CR.DecisionLedger()
        self.credit_reports: list = []
        self.subsystems = SP.SubsystemLedger()
        self.classifier = FL.LossClassifier()
        self.postmortems = PM.PostmortemStore()
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
        self.board = CH.KnowledgeBoard(self.workdir / "board.jsonl", gate=PR.PromotionGate(pol, code_hash=self.code_hash, learning_claim=c.learning_claim),
                                       policy=c.board_policy)
        # ---- bookkeeping
        self.trace: list[StageEvent] = []
        self.decisions: list[RowDecision] = []
        self.summaries: list[EpisodeSummary] = []
        self.pending: dict[str, _Episode] = {}
        self._weekly: dict[str, list] = {}                 # candidate pattern -> [(matured, mean_edge, se, n)]
        self._history: list = []                           # resolved rows kept for retro-fitting support to newborn knowledge
        self._kid_of: dict[str, str] = {}                  # pattern id -> knowledge id
        self._pid_of: dict[str, str] = {}
        self._registered_rids: set[str] = set()
        self._features: tuple[str, ...] = tuple(c.candidate_features)
        self.last_learned_on: Any = None
        self._tick = 0
        self._allow_cache: dict[tuple, bool] = {}
        self._epistemic: dict[str, Epistemic] = {}
        self.failures = FL.FailureLedger()
        self._failure_rows: list = []
        self._failure_cursor = 0
        self._mkt: dict = {}
        self.missed_report = None
        self._mid: dict[str, str] = {}
        self._birth_date: dict[str, str] = {}
        self._births: list[str] = []
        self._resolved_meta: dict[str, bool] = {}
        self._meta_done: set[str] = set()
        self._contradicts: set[tuple] = set()
        self._graph_pairs: set[tuple] = set()
        self._graph_rules: set[str] = set()
        self._transfer: dict[str, dict] = {}
        self._transfer_logged: set[str] = set()
        self._refusals: list = []
        self._gate_log: list = []
        self._recent: list = []
        self.calibration_last = None
        self.meta_update = None
        self.research_step = None
        self.counters: dict[str, int] = {}
        self._book: list[_BookRow] = []                   # F10: the shadow book the evidence card is measured on
        self._evidence: tuple | None = None                # (tick, valid card or None, refusal reason)
        self.evidence_log: list[dict] = []
        # ---- every other learning module, each called from the stage that owns it (loop_hooks.py); persisted under workdir/loop
        self.hooks = LH.LoopHooks(self, c.hooks, self.workdir / "loop")
        self.experiments = self.hooks.experiments

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
        self.hooks.note_gate(self.gate.admit(ctx), now)

    # ------------------------------------------------------------------------------------------------ candidate patterns
    def _candidate_features(self, panel: pd.DataFrame) -> tuple[str, ...]:
        if not self._features:
            self._features = tuple(c for c in panel.columns if not str(c).startswith("m_") and pd.api.types.is_numeric_dtype(panel[c]))
        return self._features

    def _levels(self, panel: pd.DataFrame, feats: Sequence[str] | None = None) -> pd.DataFrame:
        """Cross-sectional quantile level (0 = lowest) of every candidate feature, per date: identity-free membership."""
        feats = tuple(feats) if feats is not None else self._candidate_features(panel)
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
            lineage = [k for kid in self.store.ids() for k in self.store.history(kid) if k.visible_at(ep.now)]
            self._admit(ep.now, "retrieve", items=list(visible), all_items=lineage, knowledge_store=self.store,
                        relevant=frozenset({FW.LayerName.MEMORY}))
            n_items = n_shadow = 0
            held = set(self._pid_of.values()) if self._scores_shadow(ep) else set()
            for r in ep.rows:
                if not r.situation.usable(self.cfg.min_coverage):
                    continue
                r.retrieval = self.retriever.retrieve(r.situation, ep.now)
                n_items += len(r.retrieval.items)
                if held and held.intersection(r.members):
                    r.shadow = self.shadow_retriever.retrieve(r.situation, ep.now)
                    n_shadow += len(r.shadow.items)
            box["n"], box["note"] = n_items, f"{len(visible)} visible knowledge objects, {n_shadow} shadow-scored items"

    def _scores_shadow(self, ep: _Episode) -> bool:
        """Shadow predictions are made only while learning: a probe or a frozen learner registers nothing."""
        return self.shadow_retriever is not None and ep.track and not self.frozen

    def _scored(self, r: _Row) -> RV.Retrieval | None:
        """The retrieval whose prediction the skill monitor scores for this row: the shadow one when shadow monitoring is on (it
        ranks a superset of what production retrieval sees, so registering both would count one row twice), else production's."""
        return r.shadow if self.shadow_retriever is not None else r.retrieval

    # ------------------------------------------------------------------------------------------------ 4. ASSESS RELIABILITY
    def item_weight(self, kid: str, now, ctx_now: Mapping[str, Any] | None = None) -> tuple[float, str]:
        """How much a retrieved item may count today: the five reliability dimensions (section 11) x its retirement state x its
        temporal profile.  Zero is a real answer (STANDBY / UNTESTED / retired) and is reported with its reason."""
        if not self.tracker.known(kid):
            return 0.0, "no reliability record"
        st = self.tracker.state(kid, now, ctx_now)
        dec = RL.decide(st)
        w = dec.weight
        if self.retirement.known(kid):                  # lifecycle x calibration x temporal (calibration.combined_influence)
            w *= self.hooks.influence(kid, now, ctx_now, st.current_reliability)
        else:
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
                    shr = self.hooks.shrunk_expectation(kid, r.situation) if est is None or est.expected is None else None
                    e = est.expected if est is not None and est.expected is not None else shr[0] if shr is not None else sup_edge
                    if e is None:
                        continue
                    se = (est.se if est is not None and est.se is not None else shr[1] if shr is not None and shr[1] > 0 else scale)
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
            ep.rows.sort(key=lambda r: r.decision.slot)
            self.hooks.on_decide(ep)
            if ep.track:
                self._log_influence(ep, out)
                self.decisions.extend(sorted(out, key=lambda d: d.slot))
                self.summaries.append(EpisodeSummary(ep.eid, str(as_date(ep.now)), len(out), sum(d.action == "LONG" for d in out),
                                                     sum(d.action != "LONG" for d in out), sum(bool(d.knowledge_ids) for d in out), False))
                self._register_predictions(ep)
                self._book_decisions(ep)
            box["n"], box["note"] = len(out), f"{len(chosen)} long"
            return sorted(out, key=lambda d: d.slot)

    def _contract_allows(self, kid: str, now) -> bool:
        k = self.store.as_of(kid, now)
        if k is None:
            return False
        key = (kid, k.version, str(as_date(now)))
        if key not in self._allow_cache:
            if len(self._allow_cache) > 512:
                self._allow_cache.clear()
            self._allow_cache[key] = self.hooks.contract_allows(k, now)
        return self._allow_cache[key]

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
        probe = stable_hash([b.as_dict() for b in r.situation.blocks if b.kind != "pattern_interaction"])
        return RowDecision(slot, r.situation.situation_id, r.situation.exact_id, probe, action, size, r.expected, r.confidence, r.risk, unc,
                           r.members, kids, why, ret.retrieval_id if ret is not None else "", bool(ret.influence) if ret is not None else False,
                           ret.explain()[:240] if ret is not None else "")

    def _log_influence(self, ep: _Episode, decs: list) -> None:
        """Every knowledge item that carried weight is written to the hash-chained influence log (contract section 43)."""
        for r in ep.rows:
            if r.decision.knowledge_ids:
                items = [self.store.as_of(k, ep.now) for k in r.decision.knowledge_ids]
                self.decision_log.record_all(f"{ep.eid}#{r.decision.slot}", [i for i in items if i is not None], r.situation.bins(), ep.now)

    def _book_decisions(self, ep: _Episode) -> None:
        """F10: the shadow book.  For every row of a learning episode, what ALL held knowledge (production and not yet) expected,
        written before the outcome exists; production-only when shadow monitoring is off.  It never reaches a decision: stage_decide
        has already run on `retrieval`.  The evidence card measures the learner's claim on this book, which breaks the F07 circle
        (no decisions until promotion, no promotion until the decisions are measured) without lowering any bar."""
        if self.frozen or not self.cfg.claim_evidence:
            return
        scale = float(self.surprise.scale_at(ep.now)[0])
        for r in ep.rows:
            r.booked, r.book_scale = True, scale
            live = self._book_items(r.shadow, ep.now) if r.shadow is not None else ()
            if live:
                r.book_exp = RV.combine(dataclasses.replace(r.shadow, items=live), self.index)["expected"]
            elif r.decision is not None and r.decision.knowledge_ids:
                r.book_exp = r.expected

    def _book_items(self, retrieval: RV.Retrieval, now) -> tuple:
        """The shadow items the book may count: those the learner's own retirement ledger still lets carry weight (influence > 0).
        F10 found on the 5-year runs that a parked (DORMANT) decayed item kept voting in the book; its early-year value against its
        dead later years then reads to the claim gate as a positive memorisation gap (reproduced on a synthetic book).  The book is
        what the learner would hold, and it would not hold a parked item."""
        return tuple(i for i in retrieval.items if self.retirement.influence(i.knowledge_id, now) > 0.0)

    def _register_predictions(self, ep: _Episode) -> None:
        for r in ep.rows:
            ret = self._scored(r)
            if ret is None or not ret.items or ret.retrieval_id in self._registered_rids:
                continue
            self._registered_rids.add(ret.retrieval_id)
            RV.register_prediction(ret, self.monitor, ep.now, self.index)
            self._count("predictions")
            if ret is r.shadow and not (r.retrieval is not None and r.retrieval.items):
                self._count("shadow_only_predictions")

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
                ret = self._scored(r)
                if ret is not None and ret.items:
                    RV.resolve_outcome(ret, self.monitor, r.matured, r.edge)             # walk-forward score of retrieval itself
                if r.booked:
                    self._book.append(_BookRow(str(as_date(ep.now)), r.matured, self._ident(r.key[1]),
                                               str(r.situation.bins().get(self.cfg.context_dim, "na")), r.book_exp, r.book_scale, r.edge, r.raw_ret))
            del self._book[: max(0, len(self._book) - self.cfg.max_book_rows)]
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
            if self._tick % self.cfg.credit_every == 0 and len(self.credit_ledger) >= cc.min_decisions:
                eng = CR.CreditEngine(CR.WeightedSumCombiner({"pattern": 1.0}), cc)
                self.credit_reports.append(eng.assess(self.credit_ledger, now))
            self.hooks.after_credit(ep, now)
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
                self.hooks.on_failure(t, cls, pm, max(x.matured for x in ep.rows))
                self._failure_rows.append({"knowledge_id": _safe(t.knowledge_ids[0]) if t.knowledge_ids else "none",
                                           "kid": t.knowledge_ids[0] if t.knowledge_ids else "", "when": t.resolved_at,
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
        if len(self.missed_weeks) >= self.cfg.missed_train_min + 12 and self._tick % self.cfg.missed_every == 0:
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
            if self._tick % self.cfg.discover_every == 0:
                for pid in sorted(stored):
                    if self.context.n_obs(pid) < self.cfg.min_context_obs:
                        continue
                    for rule in self.context.discover(pid, now, self.cfg.seed):
                        if rule.confirmed and rule.role == "CONTEXT" and self.rules.add(rule, now):
                            found += 1
                    found += self._learn_boundaries(pid, now)
            self.hooks.after_conditions(ep, now)
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
            self.hooks.after_reliability(ep, now)
            box["n"] = n

    def _retire_check(self, kid: str, pid: str, dirn: int, now) -> None:
        """The weekly retirement / recovery pass (retirement.window_check: one rule shared with the false-degrade study)."""
        dates, vals = self._signed(pid, dirn)
        last = self.retirement.last_transition(kid, now)
        probation = last is not None and last.kind in ("RECOVER_PROBATION", "REVIVE")
        v = RT.window_check(self.retirement, kid, dates, vals, now, self.cfg.retire_window, self._mkt_now())
        if v is not None and v.changes:
            self._count({"RECOVER_FULL": "recovered_after_probation" if probation else "recovered_from_degraded",
                         "RECOVER_PROBATION": "recovered_to_probation"}.get(v.kind, f"retirement_{str(v.kind).lower()}"))

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
                due = kid not in self._transfer or self._tick % self.cfg.transfer_every == 0
                if not due:
                    continue                                  # the near/far test is quadratic in support size: on a cadence
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
        self.archive.put_knowledge(_Indexed(k, self._pid_of.get(k.knowledge_id, "")), occurred_at=k.updated_at)
        return k

    def _revise(self, kid: str, learned: str, reason: str, **changes) -> KN.KnowledgeObject:
        cur = self.store.latest(kid)
        changes.setdefault("provenance", dataclasses.replace(cur.provenance, outcomes_seen_through=str(as_date(learned))))
        try:
            nxt = cur.new_version(learned, reason, learned_at=learned, **changes)
        except KN.SchemaError as e:
            if "no-op" not in str(e):
                raise
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
        self._birth_date[kid] = learned
        self._births.append(kid)
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
        elif m.role == Promotion.CHALLENGER and len(m.shadow) >= self._min_sessions_for_gate() and self._tick % self.cfg.discover_every == 0:
            ep_now = self.store.latest(kid).epistemic
            if ep_now not in LH.CHAMPION_EPISTEMIC:
                # F10 (found on the first 5-year run): health relabelled a challenger (e.g. CONTESTED); the board would promote it and
                # the knowledge schema would then refuse a CHAMPION with that label, leaving board and store disagreeing.  Not attempted.
                self._refusals.append((kid, "promotion", f"epistemic {ep_now} cannot be champion"[:120]))
                return
            k = self.store.get(kid, m.version)                # the board gates the exact version it registered
            self.hooks.readiness(self.store.latest(kid), now)
            ce = self.hooks.claim_evidence(kid, now)      # identity report + scorecard registered for the learning-claim gate
            if ce.get("identity") is None:
                W.HUB.identity_reports.pop(kid, None)     # F10: no report of this learner's -> none at all (not another learner's)
            self._register_evidence(kid, now)             # F10: the evidence card (shadow book + portfolio card) supersedes it
            res = self.board.attempt_promotion(k, self._promotion_evidence(kid, pid, now, learned), now)
            self._gate_log.append((kid, res["promoted"], tuple(res["decision"].critical_failures)))
            if res["promoted"]:
                self._promote(kid, learned)
            else:
                self._refusals.append((kid, "promotion", ",".join(res["decision"].critical_failures)[:120]))

    # ------------------------------------------------------------------------------------------------ F10: evidence for the claim
    def book_frame(self, now) -> pd.DataFrame:
        """The shadow book rows whose outcome matured strictly before `now` (the rest is not yet evidence), as a unit table."""
        cut = as_date(now)
        rows = [b for b in self._book if as_date(b.matured) < cut]
        cols = ["date", "mature", "ticker", "ret", "edge", "expected", "scale", "regime"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame({"date": pd.to_datetime([b.decided for b in rows]), "mature": pd.to_datetime([b.matured for b in rows]),
                             "ticker": [b.ident for b in rows], "ret": [b.raw_ret for b in rows], "edge": [b.edge for b in rows],
                             "expected": [np.nan if b.expected is None else b.expected for b in rows],
                             "scale": [np.nan if b.scale is None else b.scale for b in rows], "regime": [b.ctx for b in rows]}
                            ).drop_duplicates(["ticker", "date"]).reset_index(drop=True)

    def evidence_card(self, now):
        """The learner's scorecard measured on its shadow book (scorecard_for_learner: controls A-E over forward-YEAR folds, stock and
        regime folds, memorisation and identity gaps, leak probe) merged with the portfolio card (risk, drawdown, band share,
        calibration of the book the learner would hold against a no-knowledge book).  None when the book has fewer than 4 matured
        decision weeks.  Nothing here changes a gate: the card is evidence the unchanged claim gate then judges."""
        from . import scorecard as SC
        book = self.book_frame(now)
        if book["date"].nunique() < 4:
            return None
        units = pd.DataFrame({"date": book["date"], "mature": book["mature"], "ticker": book["ticker"], "base": 0.0, "alt": book["edge"],
                              "signal": book["expected"].fillna(0.0), "regime": book["regime"]})
        card = SC.scorecard_for_learner(units, LH.trust_learned_signal, learner_version=f"{self.config_hash[:10]}-e{self._tick:05d}", now=now,
                                        code_hash=self.code_hash, seed=self.cfg.seed, n_boot=self.cfg.evidence_boot, min_units=20)
        port = SC.portfolio_card(book, now, top_n=self.cfg.top_n, window_weeks=self.cfg.evidence_window_weeks,
                                 min_windows=self.cfg.evidence_min_windows, min_expected=self.cfg.min_expected, scale="scale",
                                 seed=self.cfg.seed, n_boot=self.cfg.evidence_boot)
        return SC.merge_portfolio(card, port)

    def _register_evidence(self, kid: str, now) -> None:
        """Once per learning tick: build the evidence card; a valid one is registered for `kid`'s learning claim (replacing the hooks'
        production-only card, which in a learner with nothing promoted measures a book of zeros).  An invalid card is logged with the
        reason and nothing is registered: a refusal, never a pass."""
        if not self.cfg.claim_evidence:
            return
        if self._evidence is None or self._evidence[0] != self._tick:
            from . import scorecard as SC
            try:
                card = self.evidence_card(now)
                errs = ["fewer than 4 matured decision weeks in the shadow book"] if card is None else card.check()
            except ValueError as e:                        # a malformed book is a refusal (fail closed); a FirewallBreach propagates
                card, errs = None, [f"evidence card could not be built: {e}"]
            dec = SC.gate_improvement_claim(card) if card is not None and not errs else None
            self._evidence = (self._tick, None if errs else card, "; ".join(errs)[:200])
            self._count("evidence_cards" if not errs else "evidence_cards_refused")
            self.evidence_log.append({"tick": self._tick, "now": str(as_date(now)), "valid": not errs, "why": self._evidence[2],
                                      "allowed": None if dec is None else dec.allowed, "blockers": [] if dec is None else
                                      [c.name for c in dec.checks if c.blocking and not c.ok],
                                      "untested": [] if card is None else card.untested_fields(),
                                      "gaps": None if card is None else {f: [getattr(card, f).value, getattr(card, f).lo, getattr(card, f).hi]
                                                                         for f in ("learning_gain", "memorization_gap", "identity_gap")}})
            del self.evidence_log[:-200]
        own = self._evidence[1] if self._evidence[1] is not None else self.hooks.valid_card
        if own is not None:
            W.register_scorecard(kid, own)
        else:
            # the hub is process-wide and knowledge ids are content hashes: another learner (a control, an earlier seed) may have
            # registered a card under the same id.  This learner is never judged on someone else's evidence.
            W.HUB.scorecards.pop(kid, None)

    def _min_sessions_for_gate(self) -> int:
        p = self.board.gate.policy
        return max(p.min_oos_periods, p.min_delta_periods, p.min_risk_periods, p.min_stability_periods, self.board.policy.min_shadow_sessions)

    def _transfer_score(self, kid: str) -> float | None:
        """Transfer confidence from what this learner measured: the near/far support test when it had enough cases, else the share of
        context values (n >= 5) in which the signed edge stayed positive.  None only if neither could be measured.  A champion
        must carry a recorded value: the retriever otherwise re-runs the quadratic near/far test for every row it scores."""
        t = self._transfer.get(kid)
        if t is None:
            return None
        if t["evidence"]["score"] is not None:
            return float(t["evidence"]["score"])
        usable = [m for m, n in t["contexts"].values() if n >= 5]
        return float(np.mean([m > 0 for m in usable])) if len(usable) >= 2 else None

    def _promote(self, kid: str, learned: str) -> None:
        st = CH.shadow_summary(self.board, self._mid[kid])
        rel = self.tracker.state(kid, pd.Timestamp(learned) + pd.Timedelta(days=1))
        cur = self.store.latest(kid)
        conf = dataclasses.replace(cur.confidence, usefulness=float(max(0.0, min(1.0, 1.0 - st.p_one_sided))),
                                   current_reliability=rel.current_reliability, failure_risk=rel.failure_risk,
                                   transfer=self._transfer_score(kid), context=rel.context)
        self._revise(kid, learned, "passed every promotion gate and the head-to-head", promotion=Promotion.CHAMPION,
                     lifecycle=Lifecycle.ACTIVE, confidence=conf)
        self._resolved_meta[kid] = True

    def _refresh_confidence(self, kid: str, now, learned: str) -> None:
        """Keep the champion's dimensions current: a new version only when a dimension moved enough to matter."""
        cur = self.store.latest(kid)
        rel = self.tracker.state(kid, pd.Timestamp(learned) + pd.Timedelta(days=1))
        new = dataclasses.replace(cur.confidence, truth=rel.truth, current_reliability=rel.current_reliability,
                                  failure_risk=rel.failure_risk, transfer=self._transfer_score(kid), context=rel.context)
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
            self.board.retire(mid, FailureCause.WEAKENING_EFFECT, "retirement gate: evidence no longer supports the item", now)
        # F10: through _revise, not KnowledgeObject.retire: retire() keeps the old provenance.learned_at, and the archive refuses a
        # version whose evidence (outcomes seen through `learned`) matured after it was "learned" - first reached on a 5-year run,
        # where a DORMANT item was retired after 730 days.
        why = "retirement gate: evidence no longer supports the item"
        self._revise(kid, learned, f"retired: {why}", epistemic=Epistemic.RETIRED, lifecycle=Lifecycle.RETIRED, promotion=Promotion.RETIRED,
                     failure_explanations=cur.failure_explanations + (KN.FailureExplanation(FailureCause.WEAKENING_EFFECT, str(as_date(learned)), None, why),),
                     epistemic_profile=KN.EpistemicProfile())
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
                elif st is RT.State.ACTIVE and self.store.latest(kid).lifecycle == Lifecycle.DEGRADED:
                    # F07: the gate restored the item; the object must follow or the decision contract refuses it for ever
                    back = Epistemic.SUPPORTED if self._epistemic.get(pid) == Epistemic.SUPPORTED else self.store.latest(kid).epistemic
                    self._revise(kid, learned, "retirement gate: recovery confirmed on evidence after the degrade", lifecycle=Lifecycle.ACTIVE,
                                 epistemic=back)
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
        lv = self._levels(X, (feat,))[feat]
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
                float(sig[half:].mean()), float(np.clip(sig, lo, hi).mean()), float(np.median(sig))]

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
        feats = (pid.rsplit(":q", 1)[0],)
        memo = PR.MemorizationEvidence(shuf if math.isfinite(shuf) else None, disg if math.isfinite(disg) else None, feats,
                                       top_share > 0.6 or n_ident < 5, top_share, n_ident)
        fut = PR.FutureAudit(True, tuple(KN.audit_future(self.store, now)), str(max(dates)), (), sum(as_date(d) >= as_date(now) for d in dates))
        data_hash = k.provenance.data_hash
        reruns = tuple(PR.RerunRecord(self._boot_value(sig, s), s, self.code_hash, data_hash) for s in (1, 2, 1, 3))
        chunks = np.array_split(sig, max(pol.min_stability_periods, 6))
        stab = PR.StabilityEvidence(tuple(float(c.mean()) for c in chunks if len(c)), tuple(x for x in self._perturbed(pid, dirn, sig) if math.isfinite(x)),
                                    float(sig.mean()))
        return PR.PromotionEvidence(stat, inc, oos, trn, risk, memo, fut, PR.ReproEvidence(reruns, data_hash), stab)

    # ------------------------------------------------------------------------------------------------ 17. UPDATE GRAPH
    def stage_graph(self, ep: _Episode, now) -> None:
        """F01-F11. Decision, outcome, the knowledge that carried the decision, the failures it caused and the conditions it
        learned are written as typed nodes and time-stamped edges, so 'what caused this decision?' stays answerable."""
        with self._stage(ep, Stage.UPDATE_GRAPH) as box:
            g, matured = self.graph, max(r.matured for r in ep.rows)
            longs = [r for r in ep.rows if r.decision.action == "LONG"]
            dn, on = f"decision:{ep.eid}", f"outcome:{ep.eid}"
            g.add_node(dn, KG.NodeType.DECISION, ep.now, label=ep.eid, attrs={"n_long": len(longs)})
            g.add_node(on, KG.NodeType.OUTCOME, matured, label=ep.eid,
                       attrs={"mean_edge_long": float(np.mean([r.edge for r in longs])) if longs else 0.0, "n": len(ep.rows)})
            g.add_edge(dn, on, KG.Link.RESULTED_IN, matured)
            used: dict[str, float] = {}
            for r in ep.rows:
                for kid, w, *_ in r.parts:
                    used[kid] = used.get(kid, 0.0) + w
            tot = sum(used.values()) or 1.0
            n_edges = 0
            for kid, w in sorted(used.items()):
                g.add_edge(kid, dn, KG.Link.USED_IN, ep.now, weight=float(w / tot))
                n_edges += 1
            while self._failure_cursor < len(self._failure_rows):
                fr = self._failure_rows[self._failure_cursor]
                self._failure_cursor += 1
                fid = f"failure:{ep.eid}:{self._failure_cursor}"
                g.add_node(fid, KG.NodeType.FAILURE, fr["when"], label=fr["cause"], attrs={"subsystem": fr["subsystem"]})
                if fr["kid"] in self._pid_of:
                    g.add_edge(fid, fr["kid"], KG.Edge.CAUSES_FAILURE_OF, fr["when"], weight=fr["loss_share"])
                    n_edges += 1
            for a, b in sorted(self._contradicts - self._graph_pairs):
                g.add_edge(a, b, KG.Edge.CONTRADICTS, max(self._birth_date[a], self._birth_date[b]))
                self._graph_pairs.add((a, b))
                n_edges += 1
            for rule in self.rules.active():
                if rule.rule_id in self._graph_rules or rule.pattern_id not in self._kid_of:
                    continue
                self._graph_rules.add(rule.rule_id)
                cn, kid = f"condition:{rule.rule_id}", self._kid_of[rule.pattern_id]
                g.add_node(cn, KG.NodeType.CONDITION, matured, label=rule.spec.describe())
                if rule.role == "CONTEXT":
                    g.add_edge(kid, cn, KG.Link.APPLIES_IN, matured)
                else:
                    g.add_edge(cn, kid, KG.Edge.CAUSES_FAILURE_OF, matured)
                n_edges += 1
            self.hooks.after_graph(ep, now)
            box["n"], box["note"] = n_edges, f"{len(g.nodes(now))} nodes visible"

    # ------------------------------------------------------------------------------------------------ 18. UPDATE META-KNOWLEDGE
    def stage_meta(self, ep: _Episode, now) -> None:
        """G12/C16. The learner learns about its own learning: which kinds of discovery survived, how fast reliability decays, which
        contexts transfer; refitted on a cadence, out-of-sample, and handed to the research stage as advice."""
        with self._stage(ep, Stage.UPDATE_META) as box:
            matured = max(r.matured for r in ep.rows)
            added = 0
            for kid, survived in sorted(self._resolved_meta.items()):
                if kid in self._meta_done or as_date(matured) <= as_date(self._birth_date[kid]):
                    continue
                pid = self._pid_of[kid]
                st = self.beliefs.current(pid)
                feats = {"t_disc": float(abs(st.mean) / max(st.sd, 1e-9)), "log_n": float(math.log(max(st.n_obs, 1))),
                         "effect": float(abs(st.mean)), "n_conditions": 0.0, "p_real": float(st.prob_sign_right())}
                self.meta.store.add(ML.DiscoveryRecord(kid, pid.rsplit(":q", 1)[0], self._birth_date[kid], feats, survived, matured,
                                                       temporal_class=str(TemporalClass.UNKNOWN)))
                self._meta_done.add(kid)
                added += 1
            if self._tick % self.cfg.discover_every == 0:
                for kid in sorted(self._pid_of):
                    h = [s for s in self.tracker.history(kid) if s.current_reliability]
                    age = (as_date(matured) - as_date(self._birth_date[kid])).days
                    if len(h) >= 2 and age > 0 and h[0].current_reliability > 0:
                        self.meta.store.add(ML.DecayRecord(kid, "pattern", float(age), float(max(h[-1].current_reliability, 1e-3) / h[0].current_reliability),
                                                           matured))
                        added += 1
            if self._tick % self.cfg.meta_every == 0:
                self.meta_update = self.meta.update(now, self.cfg.seed)
                self.meta_advice = self.meta_update.advice
            self.hooks.after_meta(ep, now)
            box["n"] = added

    # ------------------------------------------------------------------------------------------------ 19. SELECT NEXT RESEARCH QUESTION
    def stage_research(self, ep: _Episode, now) -> None:
        """C17/G02-G06. Surprises, failures and missed winners become signals; the research engine ranks the questions by expected
        information per unit cost, using what meta-learning says about which kinds of experiment have paid off."""
        with self._stage(ep, Stage.SELECT_RESEARCH) as box:
            if self.experiments is None:
                self.experiments = EM.ExperimentLedger()
            learned = max(r.matured for r in ep.rows)
            recent = [r for r in self.surprise.records(now) if abs(r.z) >= 2.0][-8:]
            sig = RPR.signals_from_surprise_rows(
                [{"subject": _safe(r.cell), "when": r.matured_at, "expected": r.expected, "observed": r.actual, "sd": r.scale,
                  "subsystem": "SELECTION"} for r in recent], now, z_min=2.0)
            new_fail = [f for f in self._failure_rows if as_date(f["when"]) == as_date(max(r.matured for r in ep.rows))]
            sig += RPR.signals_from_failure_rows(new_fail[:5], now)
            winners = [r for r in ep.rows if r.edge >= self.cfg.winner_thr and r.decision.action != "LONG"]
            if winners:
                key = stable_hash(sorted(w.situation.situation_id for w in winners), 10)
                sig += RPR.signals_from_missed_winners([{"situation_key": _safe(f"missed-{key}"), "when": max(r.matured for r in ep.rows),
                                                         "gain_share": min(1.0, len(winners) / len(ep.rows)), "n_obs": len(winners)}], now)
            sig += self.hooks.research_signals(learned, now)
            budget = RP.ComputeBudget(cpu_minutes=self.cfg.cpu_minutes, ram_gb_free=8.0)
            step = self.research.step(sig, self.experiments, budget, now, self.cfg.seed, meta=self.meta_advice)
            self.next_questions = tuple(q.text for q in step.questions[:5])
            self.research_step = step
            closed = self.hooks.close_research(step, learned, now)
            box["n"], box["note"] = len(sig), (f"{len(step.questions)} questions, {len(closed['proposed'])} proposed, "
                                               f"{len(closed['answered'])} answered, queue {step.queue_summary}")

    # ------------------------------------------------------------------------------------------------ orchestration
    def decide_batch(self, now, panel: pd.DataFrame, track: bool = True) -> _Episode:
        """Stages 1-6 for the cross-section of `now`.  track=False is a dry decision (an evaluation probe): nothing is logged,
        registered or remembered, so probing a learner never changes it."""
        self.verify_frozen()
        eid = f"E{len(self.summaries) + 1:05d}" if track else f"P{stable_hash([str(as_date(now)), self.config_hash], 6)}"
        ep = _Episode(eid, now, [], track=track)
        self.stage_observe(ep, panel)
        self.stage_describe(ep, panel)
        self.stage_retrieve(ep)
        self.stage_assess(ep)
        self.stage_expectations(ep)
        self.stage_decide(ep)
        if track:                                         # the hub audits which board members carried weight in this run
            W.note_use(ep.eid, sorted({self._mid[p[0]] for r in ep.rows for p in r.parts if p[0] in self._mid}))
            W.end_decision_run(ep.eid)
        if track and not self.frozen:
            self.pending[ep.eid] = ep
        return ep

    def resolve_and_learn(self, ep: _Episode, now, outcomes: pd.DataFrame) -> EpisodeSummary:
        """Stages 7-19 once the outcomes have matured strictly before `now`.  Any FirewallBreach stops the cycle where it happens
        and leaves the episode pending (nothing half-learned is committed past the stage that failed)."""
        self.verify_frozen()
        if self.frozen:
            raise FrozenLearnerError("a frozen learner does not learn")
        if ep.eid not in self.pending:
            raise KeyError(f"episode {ep.eid} is not pending")
        self._tick += 1
        self.stage_outcome(ep, now, outcomes)
        self.stage_surprise(ep, now)
        self.stage_credit(ep, now)
        self.stage_beliefs(ep, now)
        self.stage_failures(ep, now)
        self.stage_conditions(ep, now)
        self.stage_anti_conditions(ep, now)
        self.stage_reliability(ep, now)
        self.stage_transfer(ep, now)
        self._recent.append((pd.DataFrame([r.raw for r in ep.rows], index=pd.MultiIndex.from_tuples([r.key for r in ep.rows])),
                             pd.Series([r.edge for r in ep.rows], index=pd.MultiIndex.from_tuples([r.key for r in ep.rows]))))
        del self._recent[: max(0, len(self._recent) - self.cfg.audit_weeks)]
        self.stage_store(ep, now)
        self.stage_graph(ep, now)
        self.stage_meta(ep, now)
        self.stage_research(ep, now)
        ep.learned = True
        del self.pending[ep.eid]
        longs = [r.edge for r in ep.rows if r.decision.action == "LONG"]
        i = next(j for j, s in enumerate(self.summaries) if s.episode == ep.eid)
        self.summaries[i] = dataclasses.replace(self.summaries[i], learned=True, matured_on=max(r.matured for r in ep.rows),
                                                mean_edge_long=float(np.mean(longs)) if longs else None)
        self.last_learned_on = max(r.matured for r in ep.rows)
        self.hooks.maybe_persist(now)
        return self.summaries[i]

    def ready(self, ep: _Episode, now) -> bool:
        return (as_date(now) - as_date(ep.now)).days > self.cfg.horizon_days

    def step(self, now, panel: pd.DataFrame, outcomes: pd.DataFrame | None = None) -> tuple[list[EpisodeSummary], _Episode]:
        """One tick of the loop: learn from every pending episode whose horizon has passed (oldest first), then decide today's
        cross-section.  The outcomes frame may hold rows for episodes that are not yet ready; those are left untouched."""
        learned = []
        if not self.frozen and outcomes is not None:
            for eid in sorted(self.pending):
                ep = self.pending[eid]
                if self.ready(ep, now):
                    learned.append(self.resolve_and_learn(ep, now, outcomes))
        return learned, self.decide_batch(now, panel)

    def picks(self, ep: _Episode) -> list[tuple[tuple, RowDecision]]:
        """(private row key, decision) of every LONG row: for an external scorer only; nothing the learner stores uses the key."""
        return [(r.key, r.decision) for r in ep.rows if r.decision is not None and r.decision.action == "LONG"]

    # ------------------------------------------------------------------------------------------------ reports
    def trace_table(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(e) | {"stage": e.stage.value} for e in self.trace])

    def trace_digest(self) -> str:
        return stable_hash([(e.episode, e.stage.value, e.ok, e.n) for e in self.trace])

    def report(self, now=None) -> dict[str, Any]:
        """One dictionary a report can read: stage counts, knowledge by role, skill, gate history, credit, counters."""
        now = now if now is not None else (self.last_learned_on or "1900-01-01")
        roles = {}
        for kid in self._pid_of:
            o = self.store.latest(kid)
            roles[str(o.promotion)] = roles.get(str(o.promotion), 0) + 1
        tr = self.trace_table()
        by_stage = {} if tr.empty else tr.groupby("stage")["ok"].agg(["count", "sum"]).rename(columns={"sum": "ok"}).astype(int).to_dict("index")
        return {"label": LABEL, "code_hash": self.code_hash, "config_hash": self.config_hash, "frozen": self.frozen,
                "episodes": len(self.summaries), "learned": sum(s.learned for s in self.summaries), "stages": by_stage,
                "knowledge": {"items": len(self._pid_of), "by_role": roles, "production": list(self.production_ids())},
                "skill": self.monitor.status(now) if now != "1900-01-01" else {}, "gate_history": list(self._gate_log),
                "refusals": list(self._refusals)[-10:], "counters": dict(self.counters), "n_credit_reports": len(self.credit_reports),
                "n_postmortems": len(self.postmortems.bodies()), "open_hypotheses": len(self.hypotheses),
                "contradicting_pairs": len(self._contradicts), "questions": list(self.next_questions),
                "influence_log_ok": not self.decision_log.verify(), "hooks": self.hooks.report()}


# F07 added the DEGRADED -> ACTIVE door as a learner-side subclass; F10 adopted it natively (RetirementLedger.recover_degraded), so
# the old name is kept only as an alias for callers and tests that still construct it.
RecoveringLedger = RT.RetirementLedger


class _CachedRetriever(RV.Retriever):
    """Retriever whose configuration id is computed once: weights, similarity weights and config are immutable for its life, and
    hashing them on every one of thousands of retrievals was a measurable share of a run."""

    def _weights_id(self) -> str:
        if not hasattr(self, "_wid"):
            self._wid = super()._weights_id()
        return self._wid


class _Indexed:
    """A KnowledgeObject as the retrieval index holds it: identical (typed ContextSets included, which context.py and
    archive.put_knowledge now read directly), plus the pattern id the retriever's activity gate reads."""
    __slots__ = ("_k", "pattern_id")

    def __init__(self, k: KN.KnowledgeObject, pattern_id: str):
        self._k, self.pattern_id = k, pattern_id

    def __getattr__(self, name):
        return getattr(self._k, name)


@dataclass(frozen=True)
class _Hist:
    matured: str
    edge: float
    members: frozenset
    situation: ST.Situation
    ident: str                                # opaque hash used ONLY by the identity-concentration audit
    ctx: str
    ref: str


_NUMERIC_PATHS = ("volatility.vol_rank", "liquidity.dv_rank", "sector.strength20", "sector.strength60", "stock_type.lottery_rank")

# ---------------------------------------------------------------------------------------------------------------- feeds


@dataclass(frozen=True)
class StepInput:
    now: Any
    panel: pd.DataFrame
    outcomes: pd.DataFrame | None


class WorldFeed:
    """Feeds a planted world to a learner one week at a time: the cross-section of week t as features, and as outcomes only the
    weeks whose label closed strictly before `now`.  `shuffle_seed` permutes the outcomes inside each week (the control learner
    learns from outcomes that carry no information about the features)."""

    def __init__(self, world, shuffle_seed: int | None = None, include_canary: bool = False):
        self.world = world
        self.cols = world.feature_columns() + world.market_columns() + (world.canary_columns() if include_canary else [])
        y = world.y.copy()
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            for w in range(len(world.dates)):
                m = world.week_idx == w
                y.loc[m] = rng.permutation(y.loc[m].to_numpy())
        self.y = y
        self.n_weeks = len(world.dates)

    def week_rows(self, t: int) -> np.ndarray:
        return self.world.week_idx == t

    def panel(self, t: int) -> pd.DataFrame:
        return self.world.X.loc[self.week_rows(t), self.cols]

    def realised(self, t: int) -> pd.Series:
        """Excess return of week t's rows, for the external scorer only (never passed to a learner at week t)."""
        y = self.y.loc[self.week_rows(t)]
        return y - y.mean()

    def outcomes(self, t: int, back: int = 3) -> pd.DataFrame | None:
        frames = []
        for w in range(max(0, t - back), t - 1):                       # label of week w closes at the date of week w+1 < now
            y = self.y.loc[self.week_rows(w)]
            frames.append(pd.DataFrame({"ret": y, "matured": self.world.dates[w + 1]}, index=y.index))
        return pd.concat(frames) if frames else None

    def input(self, t: int) -> StepInput:
        return StepInput(self.world.dates[t], self.panel(t), self.outcomes(t))


def learn_from(learner: LegitimateLearner, feed: WorldFeed, weeks: Iterable[int]) -> LegitimateLearner:
    for t in weeks:
        inp = feed.input(t)
        learner.step(inp.now, inp.panel, inp.outcomes)
    return learner


# ---------------------------------------------------------------------------------------------------------------- section 3


@dataclass(frozen=True)
class DecisionScore:
    """What a (frozen) learner decided on a probe, with the full section-3 record, and how the picks then did."""
    label: str
    records: tuple                                    # (week, RowDecision)
    by_exact: Mapping                                 # (week, exact situation id) -> behaviour key
    weekly: pd.DataFrame
    n_long: int
    n_with_knowledge: int
    mean_edge: float | None
    t_stat: float | None
    hit_rate: float | None
    row_edges: Mapping = field(default_factory=dict)  # (week, slot) -> realised excess return, for the truth trace (F07)


def score_decisions(learner: LegitimateLearner, feed: WorldFeed, weeks: Iterable[int], label: str = "") -> DecisionScore:
    recs, by_exact, rows, edges, row_edges = [], {}, [], [], {}
    for t in weeks:
        inp = feed.input(t)
        ep = learner.decide_batch(inp.now, inp.panel, track=False)
        real = feed.realised(t)
        picks = learner.picks(ep)
        got = [float(real.loc[k]) for k, _ in picks if k in real.index]
        for r in ep.rows:
            recs.append((t, r.decision))
            by_exact[(t, r.decision.probe_id)] = r.decision.behaviour_key()
            if r.key in real.index:
                row_edges[(t, r.decision.slot)] = float(real.loc[r.key])
        edges += got
        rows.append({"week": t, "n_long": len(picks), "mean_edge": float(np.mean(got)) if got else np.nan,
                     "n_knowledge": sum(bool(r.decision.knowledge_ids) for r in ep.rows)})
    weekly = pd.DataFrame(rows)
    ok = weekly["mean_edge"].dropna()
    t_stat = float(ok.mean() / (ok.std(ddof=1) / math.sqrt(len(ok)))) if len(ok) > 2 and ok.std(ddof=1) > 0 else None
    return DecisionScore(label, tuple(recs), by_exact, weekly, int(weekly["n_long"].sum()), int(weekly["n_knowledge"].sum()),
                         float(np.mean(edges)) if edges else None, t_stat, float(np.mean(np.array(edges) > 0)) if edges else None, row_edges)


@dataclass(frozen=True)
class ProtocolResult:
    """Section 3: BEFORE / EXPERIENCE / LEARNING / AFTER / VALIDATION as one record."""
    before: DecisionScore
    after: DecisionScore
    n_situations: int
    n_changed: int
    n_changed_with_knowledge: int
    n_changed_without_knowledge: int
    identity_invariant: bool | None
    experience_weeks: int
    verdict: str
    improvement: float | None
    notes: tuple = ()

    def behaved_differently(self) -> bool:
        return self.n_changed > 0


def compare_scores(before: DecisionScore, after: DecisionScore) -> tuple[int, int, int]:
    """(changed, changed with knowledge behind the new decision, changed with none) over situations both learners saw."""
    changed = wk = wo = 0
    for key, b in before.by_exact.items():
        a = after.by_exact.get(key)
        if a is None or a == b:
            continue
        changed += 1
        if a[2]:
            wk += 1
        else:
            wo += 1
    return changed, wk, wo


def validate_protocol(before: DecisionScore, after: DecisionScore, disguised: DecisionScore | None, experience_weeks: int) -> ProtocolResult:
    changed, wk, wo = compare_scores(before, after)
    inv = None if disguised is None else disguised.by_exact == after.by_exact
    imp = None if after.mean_edge is None else after.mean_edge - (before.mean_edge or 0.0)
    notes = []
    if wo:
        verdict = "INVALID: behaviour changed with no transferable knowledge behind it"
    elif changed == 0:
        verdict = "NO CHANGE: the experience did not alter any decision"
    elif inv is False:
        verdict = "INVALID: the decisions changed when only the identities changed"
    elif imp is not None and imp > 0 and (after.t_stat or 0) > 0:
        verdict = "IMPROVED"
    else:
        verdict = "CHANGED, NOT IMPROVED"
    if after.n_long == 0:
        notes.append("the learner abstained everywhere after learning")
    return ProtocolResult(before, after, len(before.by_exact), changed, wk, wo, inv, experience_weeks, verdict, imp, tuple(notes))


def run_protocol(make_learner: Callable[[], LegitimateLearner], learn_feed: WorldFeed, learn_weeks: Sequence[int],
                 probe_feed: WorldFeed, probe_weeks: Sequence[int], disguised_feed: WorldFeed | None = None) -> ProtocolResult:
    """BEFORE: a learner that has not had the experience decides on the probe.  EXPERIENCE + LEARNING: an identical fresh learner
    lives through `learn_weeks`.  AFTER: it is frozen and decides on the same probe (and, if given, on a disguised copy).
    VALIDATION: did behaviour change, only where knowledge carried the change, and only for transferable reasons?"""
    before = score_decisions(make_learner().freeze(), probe_feed, probe_weeks, "before")
    learner = learn_from(make_learner(), learn_feed, learn_weeks).freeze()
    after = score_decisions(learner, probe_feed, probe_weeks, "after")
    disg = score_decisions(learner, disguised_feed, probe_weeks, "disguised") if disguised_feed is not None else None
    return validate_protocol(before, after, disg, len(learn_weeks))


# ---------------------------------------------------------------------------------------------------------------- section 87


@dataclass(frozen=True)
class AcceptanceReport:
    """The final acceptance experiment in miniature: learn in year A, then meet the same situation classes in a different
    episode, under new identities, with the learner frozen and nothing from the future."""
    label: str
    seed: int
    learned_weeks: int
    probe_weeks: int
    production: int
    items: int
    lesson: DecisionScore
    control: DecisionScore                            # learned from outcomes shuffled inside each week
    lesson_on_disguised_a: DecisionScore              # year A again, under new identities (same episode)
    protocol: ProtocolResult
    truth: Mapping
    improvement_vs_none: float | None
    improvement_vs_control: float | None
    identity_invariant: bool | None
    skill: Mapping
    notes: tuple = ()
    trace: tuple = ()                                 # truth_trace rows: where each planted item's signal was lost (F07)
    degrades: Mapping = field(default_factory=dict)   # degrade_audit of the lesson learner (F10)
    evidence: tuple = ()                              # the lesson learner's evidence-card log: validity and claim-gate blockers (F10)

    def lost_at(self) -> dict[str, str]:
        return {r["item"]: r["stage"] for r in self.trace}

    def improved(self) -> bool:
        return bool(self.improvement_vs_none and self.improvement_vs_none > 0 and (self.lesson.t_stat or 0) > 1.0
                    and (self.improvement_vs_control is None or self.improvement_vs_control > 0))

    def render(self) -> str:
        f = lambda x: "n/a" if x is None else f"{x:+.5f}"
        t = lambda x: "n/a" if x is None else f"{x:+.2f}"
        rows = [f"Section-87 acceptance (miniature)   [{self.label}]",
                f"  learned {self.learned_weeks} weeks -> {self.items} knowledge items, {self.production} in production",
                f"  probe: {self.probe_weeks} weeks of a different episode under new identities",
                f"  lesson learner : {self.lesson.n_long} picks, mean edge {f(self.lesson.mean_edge)}, weekly t {t(self.lesson.t_stat)}, "
                f"{self.lesson.n_with_knowledge} rows backed by knowledge",
                f"  control learner: {self.control.n_long} picks, mean edge {f(self.control.mean_edge)}, weekly t {t(self.control.t_stat)}",
                f"  decisions changed vs a learner without the lesson: {self.protocol.n_changed}/{self.protocol.n_situations} "
                f"({self.protocol.n_changed_without_knowledge} without knowledge)",
                f"  identity-invariant on year A under new identities: {self.identity_invariant}",
                f"  improvement vs no lesson {f(self.improvement_vs_none)}, vs control {f(self.improvement_vs_control)}",
                f"  planted truth: {dict(self.truth)}",
                f"  verdict: {'IMPROVED' if self.improved() else 'NOT DEMONSTRATED'}  ({self.protocol.verdict})"]
        rows += [f"  note: {n}" for n in self.notes]
        rows += [f"  trace {r['item']:<10} {r['pattern']:<8} sign {r['planted_sign']:+d}: {r['stage']}" + (f" ({r['why']})" if r["why"] else "")
                 for r in self.trace]
        return "\n".join(rows)


def truth_check(world, learner: LegitimateLearner, at_week: int) -> dict:
    """The learner's stored items scored against the planted ledger (what it holds that is real, what it holds that is not)."""
    from . import planted_world as PW
    claims = []
    for pid in sorted(learner._kid_of):
        st = learner.beliefs.current(pid)
        claims.append(PW.Claim(pid.replace(":q", " q"), float(st.mean), None))
    sc = PW.score_claims(world, claims, at_week)
    return {"claims": sc.n_claims, "true_positive": sc.tp, "false_positive": sc.fp, "recall": round(sc.recall, 3),
            "false_discovery_rate": round(sc.false_discovery_rate, 3)}


# ---------------------------------------------------------------------------------------------------------------- F07 truth trace

TRACE_STAGES = ("never found", "found, not admitted", "admitted, not usable at the probe", "usable, never carried a decision",
                "acted on, wrong side", "acted on, right side")


def truth_trace(world, learner: LegitimateLearner, probe: DecisionScore, probe_now) -> list[dict]:
    """F07: every planted item followed through store -> retrieval -> knowledge -> decision -> outcome, and the FIRST stage at which
    its signal was lost.  found = a belief that became knowledge; admitted = CHAMPION; usable = the decision contract allows it and
    retrieval skill is PROVEN at the probe's first date; acted = it carried weight on a probe row; side = the sign of the learner's
    expectation on those rows against the planted sign, and `outcome` = planted sign x mean realised excess return of those rows (> 0
    means the probe world paid the call).  Noise items are traced too: for them any stage past 'never found' is a false discovery."""
    rows = []
    skill = learner.monitor.status(probe_now)["status"]
    for it in world.spec.items:
        pids = [learner.pattern_id(f, q) for f, q in it.conds] if len(it.conds) == 1 and it.unless is None else []
        pid = pids[0] if pids else None
        sign = int(np.sign(it.effect)) if it.effect else 0
        kid = learner._kid_of.get(pid) if pid else None
        rec = {"item": it.item_id, "kind": it.kind, "pattern": pid or it.key_named, "planted_sign": sign, "belief_mean": None,
               "born": kid is not None, "role": None, "lifecycle": None, "contract": None, "skill": skill, "weighted_rows": 0,
               "long_rows": 0, "mean_expected": None, "outcome": None, "stage": "never found", "why": ""}
        if pid in learner.beliefs.subjects():
            rec["belief_mean"] = round(float(learner.beliefs.current(pid).mean), 5)
        if kid is None:
            rec["why"] = "multi-condition item: outside the learner's single-cell search" if pid is None else "no supported belief became knowledge"
            rows.append(rec)
            continue
        k = learner.store.latest(kid)
        rec.update(role=str(k.promotion), lifecycle=str(k.lifecycle))
        if k.promotion != Promotion.CHAMPION:
            rec.update(stage="found, not admitted", why="; ".join(r[2] for r in learner._refusals if r[0] == kid)[-160:] or "never reached the gate")
            rows.append(rec)
            continue
        allowed = learner._contract_allows(kid, probe_now)
        rec["contract"] = allowed
        if not allowed or skill != "PROVEN":
            rec.update(stage="admitted, not usable at the probe", why=f"contract allows={allowed}, retrieval skill {skill}, lifecycle {k.lifecycle}")
            rows.append(rec)
            continue
        used = [(t, d) for t, d in probe.records if kid in d.knowledge_ids]
        rec["weighted_rows"] = len(used)
        rec["long_rows"] = sum(d.action == "LONG" for _, d in used)
        if not used:
            rec.update(stage="usable, never carried a decision", why="retrieval never ranked it with weight on a probe row")
            rows.append(rec)
            continue
        exp = [d.expected for _, d in used if d.expected is not None]
        real = [probe.row_edges[(t, d.slot)] for t, d in used if (t, d.slot) in probe.row_edges]
        rec["mean_expected"] = round(float(np.mean(exp)), 5) if exp else None
        rec["outcome"] = round(sign * float(np.mean(real)), 5) if real and sign else None
        right = exp and sign and np.sign(np.mean(exp)) == sign
        rec["stage"] = "acted on, right side" if right else "acted on, wrong side"
        rows.append(rec)
    return rows


def degrade_audit(world, learner: LegitimateLearner) -> dict:
    """F10: every DEGRADE the retirement gate wrote, classified against the planted truth (planted_world.live_for_degrade on the
    trusted side): on a still-working planted item it was FALSE, on noise or a decayed item it was right.  Also counts weeks spent
    not ACTIVE per planted item.  Items the learner holds that match no planted single-cell item are 'unknown'."""
    from . import planted_world as PW
    item_of = {}
    for it in world.spec.items:
        if len(it.conds) == 1 and it.unless is None:
            kid = learner._kid_of.get(learner.pattern_id(*it.conds[0]))
            if kid is not None:
                item_of[kid] = it.item_id
    dates = [pd.Timestamp(d) for d in world.dates]

    def week_of(at) -> int:
        return max(0, int(np.searchsorted(np.array(dates, dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(at)), side="right")) - 1)

    def live(kid, at):
        return None if kid not in item_of else PW.live_for_degrade(world.spec, item_of[kid], week_of(at))

    aud = RT.audit_degrades(learner.retirement, live)
    per = {}
    for kid, iid in sorted(item_of.items(), key=lambda x: x[1]):
        hist = learner.retirement.history(kid)
        kinds = [t.kind for t in hist]
        not_active = sum(learner.retirement.state(kid, d + pd.Timedelta(days=1)) not in (None, RT.State.ACTIVE) for d in dates)
        per[iid] = {"kind": world.spec.item(iid).kind, "degrades": kinds.count("DEGRADE"), "false_degrades": aud.by_item.get(kid, (0, 0))[0],
                    "recoveries": kinds.count("RECOVER_FULL"), "dormant": kinds.count("DORMANT"), "weeks_not_active": int(not_active),
                    "registered_week": week_of(hist[0].at) if hist else None}
    return {"false_degrades": aud.on_live, "correct_degrades": aud.on_dead, "unknown_degrades": aud.unknown,
            "false_degrade_share": None if math.isnan(aud.false_degrade_share) else round(aud.false_degrade_share, 3), "items": per}


def run_acceptance(spec, seed: int, make_learner: Callable[[], LegitimateLearner], years_apart: int = 6, probe_from: int = 0,
                   probe_weeks: int | None = None) -> AcceptanceReport:
    """Year A is learned once (and once more from shuffled outcomes as a control).  Year B is a different episode of the same
    situation classes (fresh noise, other volatility, later dates) shown under new tickers, shuffled rows and shifted dates.
    Both learners are frozen before they see B; B's outcomes are used only to score their picks."""
    from . import planted_world as PW
    world_a = PW.make_world(spec, seed)
    world_b = PW.year_swap(world_a, seed + 1, years=years_apart)
    world_b_id = PW.reidentify(world_b, seed + 2, tickers=True, shift_years=1).world
    world_a_id = PW.reidentify(world_a, seed + 3, tickers=True, shift_years=years_apart).world
    fa, fb = WorldFeed(world_a), WorldFeed(world_b_id)
    n_a, n_b = len(world_a.dates), len(world_b.dates)
    lesson = learn_from(make_learner(), fa, range(n_a)).freeze()
    control = learn_from(make_learner(), WorldFeed(world_a, shuffle_seed=seed + 4), range(n_a)).freeze()
    pw = range(probe_from, n_b if probe_weeks is None else min(n_b, probe_from + probe_weeks))
    s_lesson = score_decisions(lesson, fb, pw, "lesson")
    s_control = score_decisions(control, fb, pw, "control")
    s_none = score_decisions(make_learner().freeze(), fb, pw, "none")
    a_other = PW.reidentify(world_a, seed + 5, tickers=True, shift_years=years_apart + 3).world      # year A on its own dates would be
    a_disguised = score_decisions(lesson, WorldFeed(world_a_id), range(n_a), "year A disguised")   # knowledge from the future: refused
    a_plain = score_decisions(lesson, WorldFeed(a_other), range(n_a), "year A, second disguise")
    proto = validate_protocol(s_none, s_lesson, None, n_a)
    inv = a_plain.by_exact == a_disguised.by_exact
    imp = None if s_lesson.mean_edge is None else s_lesson.mean_edge - (s_none.mean_edge or 0.0)
    imp_c = None if s_lesson.mean_edge is None else s_lesson.mean_edge - (s_control.mean_edge or 0.0)
    notes = []
    if not lesson.production_ids():
        notes.append("no knowledge reached production, so the frozen learner abstained: the lesson was not transferable enough to act on")
    trace = tuple(truth_trace(world_a, lesson, s_lesson, world_b_id.dates[0]))
    return AcceptanceReport(LABEL, seed, n_a, len(pw), len(lesson.production_ids()), len(lesson._pid_of), s_lesson, s_control, a_disguised,
                            proto, truth_check(world_a, lesson, n_a - 1), imp, imp_c, inv, lesson.monitor.status(world_b.dates[0]), tuple(notes),
                            trace, degrade_audit(world_a, lesson), tuple(lesson.evidence_log))

