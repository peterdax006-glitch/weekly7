"""The C68 prediction-error pipeline wired into the research loop (PREDICTION_ERROR_ADDITION checklists T, Y, Z and the COMPLETION
REQUIREMENT; canon C68, C69 sections 4, 12-18 and 31; EXECUTION_LEDGER work item W-02). IMPLEMENTED - NOT VALIDATED (C63: code and
planted-world tests only; nothing here has run on real data).

The thirteen C68 modules existed and were unreached (EXECUTION_LEDGER headline 4). This module is the wiring: every checklist-Y step is
a stage REGISTERED into engine.research.loop (`register_stage`, never an edit of loop.py) and fed by a feed builder registered into
engine.research.feeds (`register_builder('c68.world')`: the point-in-time world strictly before `now`, through the feed's fail-closed
input audit). One cycle, in loop order:

  UPDATE_KNOWLEDGE  c68.selection_policy   the learned exit (exit_research.LearnedExitRule, target-blind) is fitted on matured
                                           candidate paths; the realisable-gain model (selection_constraint) is fitted UNDER THAT exit;
                                           the PathModel for expectations likewise; the 5-10% BandGate is installed INSIDE the
                                           two-stage funnel (two_stage.TwoStage.band_gate) so no later stage can bypass it
  EVALUATE          c68.expectations       every position of today's two-stage decision gets an immutable checklist-A expectation
                                           (expectations.ExpectationLedger.record, BEFORE the next-open fill) and a calibration
                                           commitment (calibration_target.CommitmentBook.commit); a decision taken without the gate is
                                           re-gated (two_stage.apply_band_gate) before anything reads it
  SURPRISES         c68.market_regime      market expectation vs reality (market_expectations), change points and early warning
                                           (change_points, forward only, one session at a time), regime memory (regime_memory)
                    c68.pattern_change     pattern verdicts (pattern_change, which consumes break_research's detector) + the regime
                                           guard -> pattern INFLUENCE, released to the gate only through MaturedRecord.gate(now)
  FAILURES          c68.outcomes_errors    the LEARNED exit decides every exit (the +-1pp target is never an input); outcomes
                                           (outcomes.reconstruct -> OutcomeLedger), ten separate errors (prediction_error.ErrorEngine,
                                           feeding the shared SurpriseTracker), exit records for the honest calibration statistic
                    c68.what_changed       ten-level investigation, five-way knowability (knowability.five_way, the one mapping),
                                           hypothesis trees into the loop's own TreeForest, checklist-V claims
                    c68.error_research     intensity x confidence x repeatability x value x market significance, confident-wrong
                                           15-question investigations, error-pattern escalation and the eleven checklist-T self-research
                                           questions -> QuestionEvents / ResearchQuestions on the loop bus (no second scheduler)
  QUESTIONS         c68.research_depth     after questions.generate: the depth multipliers (unknowable / barren cells) are applied to
                                           the loop's OWN priority state, so tiny and unknowable errors stay cheap
  LEARN             c68.validate_promote   self_correct: each candidate fix tested ALONE out of sample and gated by the existing quality
                                           gate; only a PROMOTE verdict changes the production learner; promoted fixes are monitored
  PRIORITIES        c68.monitor_audit      every ledger verified against anchors held OUTSIDE it (a rewrite is REFUSED_LEAK), the honest
                                           +-1pp report (calibration_target.evaluate, the canonical statistic), exit-independence audit,
                                           identification curve, a persisted cycle report

Persistence and audit: the P01 ledgers, the commitment book and the pipeline ledger live on archive ChainFile lanes under
<loop root>/c68 (one chain.jsonl); `PipelineLedger` writes one event per checklist-Y step and every event carries the hash of the
same prediction's previous event, so `trail(pid)` walks Prediction -> ... -> Future monitoring and fails closed on a broken link or an
out-of-order step. Everything else is in the loop state (checkpointed after every stage).

Research-world rule (C64/C66): all of this is MATURED_RESEARCH_STATE. The only things that reach a decision are the fitted exit policy
and gain model (trained on paths that ended strictly before `now`) and pattern influences released through MaturedRecord.gate(now).
Public entries: `register()` (idempotent; runs at import), `configure(cfg)`, the stage functions `st_*`, `plant_world(...)` (the
planted C68 world with known mechanisms) and `stage_table(reports)`."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import exits as EX
from engine.learning import promotion as PR
from engine.learning.archive import GENESIS, ChainCorrupt
from engine.learning.core import Provenance, as_date, canonical_json, stable_hash
from engine.learning.surprise import SurpriseTracker
from engine.research import calibration_target as CT
from engine.research import change_points as CP
from engine.research import error_research as ER
from engine.research import exit_research as XR
from engine.research import expectations as XP
from engine.research import feeds as FD
from engine.research import knowability as KN
from engine.research import loop as LP
from engine.research import market_expectations as ME
from engine.research import outcomes as OC
from engine.research import pattern_change as PC
from engine.research import prediction_error as PE
from engine.research import regime_memory as RM
from engine.research import selection_constraint as SC
from engine.research import self_correct as SCX
from engine.research import two_stage as TS
from engine.research import what_changed as WC
from engine.research.core import FirewallBreach, MaturedRecord, Namespace, Problem

LABEL = "IMPLEMENTED - NOT VALIDATED"
NAMESPACE = Namespace.MATURED_RESEARCH
WORLD_KEY = "c68.world"
SUBDIR = "c68"
LANE_PIPE = "pipe68"


# ================================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class PatternRule:
    """An identity-free pattern: fires on a name whose cross-sectional rank of `feature` is at/above `quantile` (top) or at/below it
    (bottom); it predicts a move in direction `sign`. The feature is one the research frame and the bars both carry."""
    name: str
    feature: str
    quantile: float
    top: bool
    sign: int = 1

    def validate(self) -> list[str]:
        errs = []
        if self.feature not in SIGNALS:
            errs.append(f"{self.name}: unknown signal {self.feature!r} (one of {SIGNALS})")
        if not 0 < self.quantile < 1 or self.sign not in (-1, 1):
            errs.append(f"{self.name}: quantile in (0,1) and sign +-1 required")
        return errs


SIGNALS = ("r20", "r5", "vol20")
DEFAULT_PATTERNS = (PatternRule("mom_r20_top", "r20", 0.8, True), PatternRule("rev_r5_bottom", "r5", 0.2, False),
                    PatternRule("vol_top", "vol20", 0.8, True))
CLASS_INFLUENCE = {PC.ChangeClass.NOISE: 1.0, PC.ChangeClass.NORMAL_VARIANCE: 1.0, PC.ChangeClass.STRENGTHENING: 1.0,
                   PC.ChangeClass.RETURNING: 1.0, PC.ChangeClass.WEAKENING: 0.5, PC.ChangeClass.REGIME_SPECIFIC_FAILURE: 0.5,
                   PC.ChangeClass.STRUCTURAL_CHANGE: 0.0, PC.ChangeClass.OBSOLESCENCE: 0.0}
GUARD_INFLUENCE = {RM.PatternAction.KEEP: 1.0, RM.PatternAction.REDUCE: 0.5, RM.PatternAction.SUSPEND: 0.0}
PATTERN_ROWS = 260                  # daily pattern rows (about a year) behind a what-changed pattern level
UNINVESTIGATED_FLOOR = 0.5          # checklist H: a failing pattern is investigated before it may lose more than half its influence


@dataclasses.dataclass(frozen=True)
class C68Config:
    horizon: int = 5                                   # sessions of a position (= the feed / two-stage outcome horizon)
    features: tuple = ("vol20", "atr", "r20", "r5")    # entry features of the realisable-gain model (known at the deciding close)
    path_context: tuple = ("r20", "r5")               # extra PathModel inputs (never a prediction or target: exit_research refuses those)
    market_features: tuple = ("m_vol", "m_r20", "m_breadth")
    train_max: int = 4000                              # newest matured candidate paths used to fit the exit / gain / path models
    min_train: int = 80
    patterns: tuple = DEFAULT_PATTERNS
    strength_window: int = 20                          # daily pattern-effect rows behind an expected pattern strength
    min_fired: int = 3                                 # names a pattern must fire on for a daily effect row
    selection: SC.SelectionConfig = SC.SelectionConfig()
    exit_cfg: XR.ExitResearchConfig = XR.ExitResearchConfig()
    error_cfg: PE.ErrorConfig = PE.ErrorConfig()
    research_cfg: ER.ErrorConfig = ER.ErrorConfig(window=24)   # 'repeated' = among the newest ~6 weeks of positions, not all history
    change: CP.ChangeConfig = CP.ChangeConfig()
    memory: RM.MemoryConfig = RM.MemoryConfig()
    market: ME.ExpectationConfig = ME.ExpectationConfig()
    target: CT.Target = CT.DEFAULT_TARGET
    self_correct: SCX.SelfCorrectConfig = SCX.SelfCorrectConfig()
    pattern_cfg: Mapping = dataclasses.field(default_factory=dict)
    what_cfg: Mapping = dataclasses.field(default_factory=lambda: {"lead_step": 8})   # lead located to 8 rows (4 re-classifications)
    max_investigations: int = 6                        # what-changed cases per cycle (the rest wait in the queue, never dropped)
    max_events: int = 12                               # error-research question events per cycle (largest intensity first)
    post_exit: int = 5
    peers: int = 8
    history_days: int = 400                            # daily evidence kept for regime memory
    guard_window: int = 25                             # newest post-change pattern rows the regime guard judges the pattern on
    seed: int = 0
    fix_evidence: SCX.FixEvidenceConfig = SCX.FixEvidenceConfig(retirement_trigger=True)   # the full gate evidence of every fix (W-06);
                                                       # the retirement trigger is THIS loop's monitor (`_monitor`)
    monitor_weeks: int = 26                            # newest post-promotion weeks the monitor judges a promoted fix on
    monitor_min_weeks: int = 8
    rollback_t: float = 2.0                            # the fix's weekly |error| gain over the shadow incumbent at t <= -this: roll back

    def validate(self) -> list[str]:
        errs = []
        if not 2 <= self.horizon <= 20:
            errs.append("horizon in [2, 20] required (an exit needs at least two sessions)")
        if not self.features or len(set(self.features)) != len(self.features):
            errs.append("features must be a non-empty set")
        if self.min_train < max(self.selection.min_support, self.exit_cfg.min_train, 20):
            errs.append("min_train below the gain model's or the exit learner's own support floor")
        for p in self.patterns:
            errs += p.validate()
        if len({p.name for p in self.patterns}) != len(self.patterns):
            errs.append("pattern names must be unique")
        errs += list(self.selection.validate()) + list(self.exit_cfg.validate()) + list(self.error_cfg.validate())
        errs += list(self.research_cfg.check()) + list(self.change.validate()) + list(self.memory.validate()) + list(self.market.validate())
        errs += list(self.target.validate()) + list(self.self_correct.validate())
        if self.max_investigations < 1 or self.max_events < 1:
            errs.append("per-cycle caps must be >= 1")
        errs += list(self.fix_evidence.validate())
        if self.monitor_min_weeks < 4 or self.monitor_weeks < self.monitor_min_weeks or self.rollback_t <= 0:
            errs.append("monitor_weeks >= monitor_min_weeks >= 4 and rollback_t > 0 required")
        return errs


_CONFIG: list[C68Config] = [C68Config()]


def configure(cfg: C68Config) -> C68Config:
    """Set the configuration a NEW loop state starts with (an existing state keeps the one it was created with, so a resumed loop
    never changes its rules mid-run)."""
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid C68Config: " + "; ".join(errs))
    _CONFIG[0] = cfg
    return cfg


# ================================================================================================================ the pipeline ledger
Y_STEPS = ("PREDICTION", "EXPECTATION", "OUTCOME", "ERROR", "CLASSIFIED", "CAUSE", "KNOWABILITY", "PATTERN_REGIME", "HYPOTHESIS",
           "RESEARCH", "OOS_TEST", "MODEL_UPDATE", "VALIDATION", "PROMOTED", "REJECTED", "MONITORED")
_ORDER = {s: i for i, s in enumerate(Y_STEPS)}
_ORDER["REJECTED"] = _ORDER["PROMOTED"]                # alternatives: the same position in the chain


class PipelineBroken(FirewallBreach):
    """A pipeline trail whose links do not hold, or a step recorded out of the checklist-Y order."""


class PipelineLedger:
    """Checklist Y, persistent and auditable: one archive ChainFile lane ('pipe68'). Every event names its object (a prediction id or
    a fix / claim id), the step, the date, a payload and `prev_event` = the content digest of the SAME object's previous event (parents
    name the objects it was derived from); the chain's own hash links make every event immutable once written. An object's steps must
    follow Y_STEPS order - an outcome can never be filed before its expectation. Events are buffered and written with one append (one
    lock, one fsync) by `flush()`; every read flushes first, so nothing is ever read around the chain."""

    def __init__(self, root=None):
        self.lane = XP.SealedLane(root, LANE_PIPE)
        self._pending: list[dict] = []
        self._last: dict[str, tuple[str, str]] = {}          # key -> (digest of its newest event, its step)
        for ln in self.lane.lines():
            self._last[ln["body"]["key"]] = (event_digest(ln["body"]), ln["body"]["step"])

    def __len__(self) -> int:
        return len(self.lane) + len(self._pending)

    @property
    def head(self) -> str:
        self.flush()
        return self.lane.head

    def last_step(self, key: str) -> str | None:
        return self._last.get(key, (None, None))[1]

    def add(self, key: str, step: str, at, payload: Mapping[str, Any] | None = None, parents: Sequence[str] = ()) -> str:
        if step not in _ORDER:
            raise ValueError(f"unknown pipeline step {step!r}")
        prev = self._last.get(key)
        if prev is not None and _ORDER[step] < _ORDER[prev[1]]:
            raise PipelineBroken(f"{key}: step {step} after {prev[1]} breaks the checklist-Y order")
        if prev is None and step not in ("PREDICTION", "OOS_TEST", "HYPOTHESIS", "VALIDATION"):
            raise PipelineBroken(f"{key}: a trail must start at PREDICTION (predictions) or OOS_TEST / HYPOTHESIS / VALIDATION (research)")
        body = {"key": str(key), "step": step, "at": str(as_date(at)), "prev_event": prev[0] if prev else "",
                "parents": sorted(str(p) for p in parents), "payload": json.loads(canonical_json(dict(payload or {})))}
        d = event_digest(body)
        self._pending.append(body)
        self._last[key] = (d, step)
        return d

    def flush(self) -> int:
        n = len(self._pending)
        if n:
            self.lane.append_many(self._pending)
            self._pending = []
        return n

    def trail(self, key: str) -> list[dict]:
        """The object's events oldest first, each checked to link to its predecessor and to keep the Y order."""
        self.flush()
        evs = [ln for ln in self.lane.lines() if ln["body"]["key"] == key]
        prev, pos = "", -1
        for ln in evs:
            b = ln["body"]
            if b["prev_event"] != prev:
                raise PipelineBroken(f"{key}: event {b['step']} does not link to its predecessor")
            if _ORDER[b["step"]] < pos:
                raise PipelineBroken(f"{key}: {b['step']} out of order")
            prev, pos = event_digest(b), _ORDER[b["step"]]
        return [dict(ln["body"], hash=ln["hash"]) for ln in evs]

    def keys(self) -> list[str]:
        return sorted(self._last)

    def can_add(self, key: str, step: str) -> bool:
        prev = self._last.get(key)
        return prev is None or _ORDER[step] >= _ORDER[prev[1]]

    def verify(self, anchors: Sequence[str] = ()) -> dict:
        """Chain integrity (and anchors), then every object's trail in ONE pass: each event must link to that object's previous
        event and keep the checklist-Y order."""
        self.flush()
        rep = self.lane.verify(anchors)
        last: dict[str, tuple[str, int]] = {}
        probs = list(rep["problems"])
        for ln in self.lane.lines():
            b = ln["body"]
            prev, pos = last.get(b["key"], ("", -1))
            if b["prev_event"] != prev:
                probs.append(f"{b['key']}: event {b['step']} does not link to its predecessor")
            elif _ORDER.get(b["step"], -1) < pos:
                probs.append(f"{b['key']}: {b['step']} out of order")
            last[b["key"]] = (event_digest(b), _ORDER.get(b["step"], -1))
        return {**rep, "ok": not probs, "problems": probs}

    def furthest(self) -> dict[str, int]:
        """How far each prediction got: step -> number of objects whose newest step it is."""
        out: dict[str, int] = {}
        for _, s in self._last.values():
            out[s] = out.get(s, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: _ORDER[kv[0]]))


def event_digest(body: Mapping[str, Any]) -> str:
    return stable_hash(dict(body), 24)


def can_record(pipe: PipelineLedger, key: str, step: str) -> bool:
    return pipe.can_add(key, step)


# ================================================================================================================ state
@dataclasses.dataclass
class Position:
    pid: str
    ticker: str
    decided_at: str
    entry_at: str
    policy_id: str
    sector: str
    vol: float
    atr: float
    patterns: tuple


@dataclasses.dataclass
class C68State:
    """Everything the C68 stages remember between cycles (picklable: it rides in the loop checkpoint)."""
    cfg: C68Config
    tracker: SurpriseTracker
    er: ER.ErrorResearchState
    pcs: PC.PatternChangeState
    wcs: WC.WhatChangedState
    market: ME.MarketExpectationEngine
    ews: CP.EarlyWarningSystem
    memory: RM.RegimeMemory
    policies: dict = dataclasses.field(default_factory=dict)        # policy id -> the fitted exit rule trades were committed under
    intended: dict = dataclasses.field(default_factory=dict)        # decided_at -> policy id intended that day
    gate: Any = None
    path_model: Any = None
    trained_through: str = ""
    pending: dict = dataclasses.field(default_factory=dict)         # pid -> Position (open positions awaiting their outcome)
    forecasts: dict = dataclasses.field(default_factory=dict)       # decided_at -> {ticker: forecast row} (research-side, all names)
    realised: dict = dataclasses.field(default_factory=dict)        # decided_at -> {ticker: (net, exit date)}
    exit_records: list = dataclasses.field(default_factory=list)    # calibration_target.ExitRecord of every traded exit
    decided: list = dataclasses.field(default_factory=list)         # decision days already turned into expectations
    investigated: list = dataclasses.field(default_factory=list)    # pids already sent to what_changed
    wc_queue: list = dataclasses.field(default_factory=list)
    contexts: dict = dataclasses.field(default_factory=dict)        # pid -> error_research.InvestigationContext
    cells: dict = dataclasses.field(default_factory=dict)           # question subject -> error-research cell (depth multipliers)
    subject_pids: dict = dataclasses.field(default_factory=dict)    # question subject -> pids behind it
    influence: dict = dataclasses.field(default_factory=dict)       # pattern -> influence in [0, 1] (1 = full)
    influence_log: list = dataclasses.field(default_factory=list)
    verdicts: dict = dataclasses.field(default_factory=dict)        # pattern -> latest change class
    guard: dict = dataclasses.field(default_factory=dict)           # pattern -> latest regime guard action
    ews_through: str = ""
    market_through: str = ""
    daily: dict = dataclasses.field(default_factory=lambda: {"dates": [], "q": {}, "error_z": []})
    warnings: list = dataclasses.field(default_factory=list)        # (date, level, scope)
    active: tuple = ()                                              # checklist-R precursors firing at the last session
    last_scope: str = "NONE"
    exited: list = dataclasses.field(default_factory=list)          # (policy id, Paths) of the newest exits: the independence audit
    depth_of: dict = dataclasses.field(default_factory=dict)        # pid -> checklist-D research tier (NONE / CHEAP / STANDARD / DEEP)
    pending_err_z: list = dataclasses.field(default_factory=list)   # (date the error became known, z) awaiting the next session
    gate_log: list = dataclasses.field(default_factory=list)        # per decision: released influence and what it did to the forecasts
    production: dict = dataclasses.field(default_factory=lambda: {"name": "incumbent", "since": "", "window_share": 1.0,
                                                                   "extra_features": ()})
    corrections: list = dataclasses.field(default_factory=list)
    monitoring: list = dataclasses.field(default_factory=list)
    retired: dict = dataclasses.field(default_factory=dict)         # fix name -> date it was rolled back (re-tested only on newer rows)
    shadow: Any = None                                              # the incumbent's gate while a fix is in production (monitoring)
    anchors: list = dataclasses.field(default_factory=list)         # (cycle, ledger, head): stored OUTSIDE the ledgers
    calibration: list = dataclasses.field(default_factory=list)
    independence: list = dataclasses.field(default_factory=list)
    runs: list = dataclasses.field(default_factory=list)            # per-stage run table rows
    counters: dict = dataclasses.field(default_factory=dict)
    namespace: Namespace = NAMESPACE

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(n)


def new_state(cfg: C68Config | None = None) -> C68State:
    cfg = cfg or _CONFIG[0]
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid C68Config: " + "; ".join(errs))
    return C68State(cfg, SurpriseTracker(), ER.new_state(cfg.research_cfg), PC.PatternChangeState(), WC.WhatChangedState(),
                    ME.MarketExpectationEngine(cfg.market, cfg.seed), CP.EarlyWarningSystem(cfg.change), RM.RegimeMemory(cfg.memory),
                    influence={p.name: 1.0 for p in cfg.patterns})


@dataclasses.dataclass
class Ledgers:
    """The on-disk C68 ledgers of one loop root (transient handles; rebuilt from the chain after a restart)."""
    root: Path
    expectations: XP.ExpectationLedger
    outcomes: OC.OutcomeLedger
    errors: PE.ErrorEngine
    book: CT.CommitmentBook
    pipe: PipelineLedger

    def heads(self) -> dict:
        """Head hashes to keep OUTSIDE the ledgers. All C68 ledgers are lanes of ONE chain under <root>/c68, so each head proves the
        whole chain up to it; they are kept per ledger so a report can say which writer came last."""
        return {"expectations": self.expectations.anchor(), "outcomes": self.outcomes.head, "errors": self.errors.lane.head,
                "book": self.book._cf.head, "pipe": self.pipe.head}


def open_ledgers(root, tracker: SurpriseTracker | None, cfg: C68Config, code_hash: str) -> Ledgers:
    """Open (or reload) the C68 ledgers under <root>/c68. A chain that does not verify on opening was edited on the medium: that is
    tampering, and it is refused as such (FirewallBreach -> REFUSED_LEAK), never read around."""
    d = Path(root) / SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    try:
        exp = XP.ExpectationLedger(d, code_hash)
        return Ledgers(d, exp, OC.OutcomeLedger(exp, d), PE.ErrorEngine(cfg.error_cfg, tracker, d), CT.CommitmentBook(d), PipelineLedger(d))
    except ChainCorrupt as e:
        raise XP.LedgerTampered(f"the C68 chain under {d} does not verify on opening: {e}") from e


def _state(ctx: LP.Ctx) -> C68State:
    return ctx.mod_state("c68", new_state)


def _ledgers(ctx: LP.Ctx, st: C68State) -> Ledgers:
    return ctx.handle("c68.ledgers", lambda: open_ledgers(ctx.rt.root, st.tracker, st.cfg, ctx.rt.code_hash))


def _run(st: C68State, ctx: LP.Ctx, stage: str, n_in: int, n_out: int, note: str) -> tuple:
    led = ctx.rt.__dict__.get("handles", {}).get("c68.ledgers")
    if led is not None:
        led.pipe.flush()                                    # the stage's pipeline events reach the chain before the checkpoint
    st.runs.append({"cycle": ctx.cycle, "now": ctx.now, "stage": stage, "in": int(n_in), "out": int(n_out), "note": note[:240]})
    st.runs = st.runs[-2000:]
    return int(n_in), int(n_out), note


# ================================================================================================================ point-in-time bars
@dataclasses.dataclass
class BarView:
    """The builder's world as arrays (every session strictly before now). Rows = sessions, columns = tickers."""
    sessions: pd.DatetimeIndex
    tickers: tuple
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    market: np.ndarray
    sector: np.ndarray
    col: dict

    @property
    def T(self) -> int:
        return len(self.sessions)

    def pos_after(self, day) -> int:
        """Index of the first session strictly after `day` (the next-open fill of a decision taken at day's close)."""
        return int(self.sessions.searchsorted(pd.Timestamp(as_date(day)), side="right"))

    def returns(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.C[1:] / self.C[:-1] - 1.0
        return np.vstack([np.full((1, self.C.shape[1]), np.nan), r])

    def sector_index(self, sector: str) -> np.ndarray:
        m = self.sector == sector
        r = self.returns()[:, m]
        ok = np.isfinite(r)
        mr = np.where(ok.any(1), np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1), 0.0) if m.any() else np.zeros(self.T)
        return 100.0 * np.cumprod(1.0 + mr)


def bar_view(world: FD.World) -> BarView:
    C = world.bars["Close"]
    S = pd.DatetimeIndex(C.index)
    tick = tuple(str(c) for c in C.columns)
    arr = {f: world.bars[f].reindex(index=S, columns=list(C.columns)).to_numpy(float) for f in FD.BAR_FIELDS}
    if world.market and "Close" in world.market and "SPY" in world.market["Close"]:
        mk = world.market["Close"]["SPY"].reindex(S).to_numpy(float)
    else:
        with np.errstate(invalid="ignore"):
            r = np.nanmean(arr["Close"][1:] / arr["Close"][:-1] - 1.0, axis=1)
        mk = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.nan_to_num(r)])
    sec = np.array([str(world.sectors.get(t, "all")) for t in tick], object)
    return BarView(S, tick, arr["Open"], arr["High"], arr["Low"], arr["Close"], arr["Volume"], mk, sec, {t: j for j, t in enumerate(tick)})


def b_world(feed, ctx) -> dict:
    """Feed builder 'c68.world': the point-in-time world strictly before now (bars, market proxy, sectors). It passes the feed's
    fail-closed audit like every other stage input (a bar dated at/after now is REFUSED_LEAK at the stage that asked)."""
    return {"world": feed.store.bars_before(ctx.now)}


def _bars(ctx: LP.Ctx) -> BarView:
    cache = ctx.rt.__dict__.setdefault("_c68_bars", {})
    if cache.get("now") != ctx.now:
        w = ctx.namespace(WORLD_KEY)["world"]
        if len(w.sessions) < 60:
            raise LP.NoInput(f"only {len(w.sessions)} sessions of bars before now")
        cache.clear()
        cache.update(now=ctx.now, bv=bar_view(w), world=w)
    return cache["bv"]


def next_session(ctx: LP.Ctx, day) -> str:
    """The exchange session after `day` (the fill). The calendar is public information; without one the next business day."""
    sess = getattr(ctx.rt.feed, "all_sessions", None)
    d = pd.Timestamp(as_date(day))
    if callable(sess):
        S = sess()
        i = int(S.searchsorted(d, side="right"))
        if i < len(S):
            return str(S[i].date())
    return str((d + pd.offsets.BDay(1)).date())


def week_end_positions(ctx: LP.Ctx, bv: BarView) -> list[int]:
    """Positions of the sessions that closed a calendar week (the market-expectation period). The newest session before now counts
    only if the public calendar says the next session opens a new week."""
    S = bv.sessions
    out = [t for t in range(bv.T - 1) if S[t + 1].to_period("W") != S[t].to_period("W")]
    if bv.T and pd.Timestamp(next_session(ctx, S[-1])).to_period("W") != S[-1].to_period("W"):
        out.append(bv.T - 1)
    return out


def next_week_end(ctx: LP.Ctx, day) -> str:
    """The session that will close the NEXT calendar week (the period the next market expectation is for)."""
    d = next_session(ctx, day)
    while True:
        n = next_session(ctx, d)
        if pd.Timestamp(n).to_period("W") != pd.Timestamp(d).to_period("W"):
            return d
        d = n


def session_after(ctx: LP.Ctx, day, k: int) -> str:
    """The k-th session after `day` (k=1 is the fill session)."""
    d = str(as_date(day))
    for _ in range(k):
        d = next_session(ctx, d)
    return d


# ================================================================================================================ paths
def build_paths(bv: BarView, rows: pd.DataFrame, horizon: int, now, kind_col: str = "sector") -> tuple[EX.Paths, pd.DataFrame]:
    """Candidate positions as engine.exits.Paths: for each (decision date, ticker) row, entry at the next session's open and
    `horizon` sessions of bars - only when EVERY bar is strictly before now (a path still running does not exist yet). Returns the
    paths and the rows they came from (same order)."""
    if not len(rows):
        return _empty_paths(horizon), rows.iloc[0:0]
    dates = pd.to_datetime(rows.index.get_level_values(0))
    ticks = [str(t) for t in rows.index.get_level_values(-1)]
    pos = bv.sessions.searchsorted(dates, side="right")
    j = np.array([bv.col.get(t, -1) for t in ticks])
    last = pos + horizon - 1
    ok = (j >= 0) & (pos >= 1) & (last < bv.T)
    n_cut = pd.Timestamp(as_date(now))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return _empty_paths(horizon), rows.iloc[0:0]
    pp, jj = pos[idx], j[idx]
    steps = pp[:, None] + np.arange(horizon)[None, :]
    o, h, l, c = (A[steps, jj[:, None]] for A in (bv.O, bv.H, bv.L, bv.C))
    prev = bv.C[pp - 1, jj]
    good = np.isfinite(o).all(1) & np.isfinite(h).all(1) & np.isfinite(l).all(1) & np.isfinite(c).all(1) & np.isfinite(prev)
    good &= (o > 0).all(1) & (c > 0).all(1) & (prev > 0)
    ends = bv.sessions[last[idx]]
    good &= np.asarray(ends < n_cut)
    idx, o, h, l, c, prev, pp = idx[good], o[good], h[good], l[good], c[good], prev[good], pp[good]
    src = rows.iloc[idx]
    h = np.maximum(h, np.maximum(o, c))
    l = np.minimum(l, np.minimum(o, c))
    vol = _col(src, "vol20", 0.02)
    atr = _col(src, "atr", 0.02)
    kinds = src[kind_col].astype(str).to_numpy(object) if kind_col in src else np.full(len(src), "all", object)
    P = EX.Paths(o, h, l, c, prev, vol, atr, np.asarray(pd.to_datetime(src.index.get_level_values(0)).values, "datetime64[D]"),
                 np.asarray(bv.sessions[pp + horizon - 1].values, "datetime64[D]"), kinds, np.array([str(t) for t in src.index.get_level_values(-1)], object))
    return P, src


def _col(df: pd.DataFrame, name: str, fill: float) -> np.ndarray:
    v = df[name].to_numpy(float) if name in df else np.full(len(df), fill)
    return np.where(np.isfinite(v) & (v > 0), v, fill)


def _empty_paths(horizon: int) -> EX.Paths:
    z = np.zeros((0, horizon))
    e = np.zeros(0)
    return EX.Paths(z, z, z, z, e, e, e, np.zeros(0, "datetime64[D]"), np.zeros(0, "datetime64[D]"), np.zeros(0, object), np.zeros(0, object))


INTERACTION = "_x_"                  # '<feature>_x_<sector>': the feature inside one sector, 0 elsewhere (a promoted slope fix)


def feature_column(rows: pd.DataFrame, name: str) -> np.ndarray:
    """One entry feature by name; an interaction name is computed from its base feature and the row's sector (both known at the
    deciding close). Absent = NaN (the gain model then drops the row), never zero."""
    if name in rows:
        return rows[name].to_numpy(float)
    if INTERACTION in name:
        base, grp = name.split(INTERACTION, 1)
        if base in rows and "sector" in rows:
            return rows[base].to_numpy(float) * (rows["sector"].astype(str).to_numpy() == grp)
    return np.full(len(rows), np.nan)


def feature_matrix(rows: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    return np.stack([feature_column(rows, n) for n in names], 1) if len(rows) else np.zeros((0, len(names)))


# ================================================================================================================ patterns from bars
def signals(bv: BarView) -> dict[str, pd.DataFrame]:
    """The pattern signals at each session's close from bars up to that close only (r20, r5, vol20 - the research frame's names)."""
    C = pd.DataFrame(bv.C, index=bv.sessions, columns=list(bv.tickers))
    r = C.pct_change(fill_method=None)
    return {"r20": C / C.shift(20) - 1.0, "r5": C / C.shift(5) - 1.0, "vol20": r.rolling(20, min_periods=15).std()}


def fired_masks(sig: Mapping[str, pd.DataFrame], rules: Sequence[PatternRule]) -> dict[str, pd.DataFrame]:
    out = {}
    for rule in rules:
        rk = sig[rule.feature].rank(axis=1, pct=True)
        out[rule.name] = (rk >= rule.quantile) if rule.top else (rk <= rule.quantile)
    return out


def market_labels(bv: BarView, cfg: PE.ErrorConfig, window: int = 20) -> pd.Series:
    """The market regime label ('UP_CALM', ...) at each session from the trailing `window` sessions of the market proxy (the same
    vocabulary prediction_error scores the expected regime in)."""
    m = pd.Series(bv.market, index=bv.sessions)
    mr = m.pct_change(fill_method=None)
    mv = m / m.shift(window) - 1.0
    vv = mr.rolling(window, min_periods=10).std()
    lab = [PE.realized_regime(None if not math.isfinite(a) else float(a), None if not math.isfinite(b) else float(b), window, cfg) or "UNKNOWN"
           for a, b in zip(mv.to_numpy(float), vv.to_numpy(float))]
    return pd.Series(lab, index=bv.sessions)


def pattern_frames(bv: BarView, rules: Sequence[PatternRule], cfg: C68Config) -> tuple[dict, dict]:
    """Daily pattern outcomes for pattern_change: row dated at session t+1 (when its outcome is known) = the sign-adjusted mean
    next-session return of the names the pattern fired on at t's close. Also the pairwise combination series (interactions).
    Columns regime / volatility let pattern_change find regime-specific failures."""
    sig = signals(bv)
    fired = fired_masks(sig, rules)
    R = pd.DataFrame(bv.returns(), index=bv.sessions, columns=list(bv.tickers))
    nxt = R.shift(-1)
    lab = market_labels(bv, cfg.error_cfg)
    frames, combos = {}, {}
    sign = {r.name: r.sign for r in rules}

    def eff(mask: pd.DataFrame, s: int) -> pd.Series:
        m = mask.fillna(False).to_numpy(bool)
        x = nxt.to_numpy(float)
        cnt = (m & np.isfinite(x)).sum(1)
        with np.errstate(invalid="ignore"):
            v = np.where(cnt >= cfg.min_fired, np.nansum(np.where(m, x, 0.0), 1) / np.maximum(cnt, 1), np.nan) * s
        out = pd.Series(v, index=bv.sessions).shift(1)            # dated at the session whose close realised it
        return out.dropna()
    for rule in rules:
        e = eff(fired[rule.name], rule.sign)
        lb = lab.reindex(e.index)
        frames[rule.name] = pd.DataFrame({"effect": e, "regime": lb.to_numpy(object),
                                          "volatility": ["VOLATILE" if "VOLATILE" in str(x) else "CALM" for x in lb]}, index=e.index)
    names = [r.name for r in rules]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            combos[f"{a}|{b}"] = eff(fired[a] & fired[b], sign[a])
    return frames, combos


def _patterns(ctx: LP.Ctx, st: C68State) -> tuple[dict, dict]:
    cache = ctx.rt.__dict__.setdefault("_c68_patterns", {})
    if cache.get("now") != ctx.now:
        cache.clear()
        cache.update(now=ctx.now, v=pattern_frames(_bars(ctx), st.cfg.patterns, st.cfg))
    return cache["v"]


def fired_today(today: pd.DataFrame, rules: Sequence[PatternRule]) -> dict[str, tuple]:
    """Which patterns fire on each name of the decision day, from the research frame's own features at the deciding close."""
    out: dict[str, list] = {str(ix[-1]): [] for ix in today.index}
    for rule in rules:
        if rule.feature not in today or len(today) < 5:
            continue
        rk = today[rule.feature].astype(float).rank(pct=True)
        hit = (rk >= rule.quantile) if rule.top else (rk <= rule.quantile)
        for ix, h in zip(today.index, hit.to_numpy(bool)):
            if h:
                out[str(ix[-1])].append(rule.name)
    return {k: tuple(v) for k, v in out.items()}


# ================================================================================================================ the 5-10% gate
@dataclasses.dataclass
class BandGate:
    """Checklist L inside the two-stage funnel: a name is eligible only when the realisable gain its INTENDED exit policy is forecast
    to deliver lies in [+5%, +10%] (selection_constraint.select; forbidden bases and stale policies refused there). Pattern influence
    (released through MaturedRecord.gate) shrinks a degraded pattern's feature toward the gain model's training mean on the names the
    pattern fires on, so a suspended pattern cannot carry a name into the band."""
    model: SC.RealisableGainModel
    policy: str
    cfg: SC.SelectionConfig
    features: tuple
    influence: Mapping[str, float]
    pattern_feature: Mapping[str, str]
    fired: Mapping[str, tuple]
    last: dict = dataclasses.field(default_factory=dict)

    def adjusted(self, rows: pd.DataFrame) -> np.ndarray:
        F = feature_matrix(rows, self.features)
        if self.model.mu_ is None:
            return F
        for p, w in self.influence.items():
            f = self.pattern_feature.get(p)
            if f not in self.features or w >= 1.0:
                continue
            j = self.features.index(f)
            hit = np.array([p in self.fired.get(str(ix[-1]), ()) for ix in rows.index], bool)
            F[hit, j] = self.model.mu_[j] + float(w) * (F[hit, j] - self.model.mu_[j])
        return F

    def forecast(self, rows: pd.DataFrame, now) -> list:
        kinds = rows["sector"].astype(str).tolist() if "sector" in rows else ["all"] * len(rows)
        return self.model.forecast([str(ix[-1]) for ix in rows.index], self.adjusted(rows), kinds, now)

    def __call__(self, rows: pd.DataFrame, now) -> pd.DataFrame:
        fcs = self.forecast(rows, now)
        rep = SC.select(fcs, now, self.policy, self.cfg)
        by = {d.candidate: d for d in rep.decisions}
        self.last = {fc.candidate: fc for fc in fcs}
        out = pd.DataFrame(index=rows.index)
        out["eligible"] = [by[str(ix[-1])].eligible for ix in rows.index]
        out["reason"] = [by[str(ix[-1])].reason.value for ix in rows.index]
        out["point"] = [fc.median if math.isfinite(fc.mean) else float("nan") for fc in fcs]
        out["p_band"] = [fc.p_in_band for fc in fcs]
        return out


def influence_effect(gate: BandGate, today: pd.DataFrame, now) -> dict:
    """What the released pattern influence did to today's forecasts: per pattern, the names it fires on and the mean change of their
    forecast median against full influence (negative = a degraded pattern pulled its names down, toward and out of the band)."""
    out = {"now": str(as_date(now)), "influence": dict(gate.influence), "shift": {}, "n_fired": {}}
    if not len(today) or gate.model.beta_ is None:
        return out
    full = dataclasses.replace(gate, influence={p: 1.0 for p in gate.influence})
    a = np.array([f.median for f in gate.forecast(today, now)])
    b = np.array([f.median for f in full.forecast(today, now)])
    for p, w in gate.influence.items():
        hit = np.array([p in gate.fired.get(str(ix[-1]), ()) for ix in today.index], bool) & np.isfinite(a) & np.isfinite(b)
        out["n_fired"][p] = int(hit.sum())
        out["shift"][p] = float(np.mean(a[hit] - b[hit])) if hit.any() else 0.0
    return out


def abstain_gate(rows: pd.DataFrame, now) -> pd.DataFrame:
    """The gate when no realisable-gain model could be fitted: nobody is eligible (UNSUPPORTED), never waved through."""
    return pd.DataFrame({"eligible": False, "reason": SC.Reason.UNSUPPORTED.value, "point": np.nan, "p_band": 0.0}, index=rows.index)


def released_influence(st: C68State, now, created_real: str, code_hash: str) -> dict[str, float]:
    """Pattern influence as the decision path may see it: each value is a MaturedRecord passed through gate(now), so an influence
    decided at `now` itself (from evidence that includes today's research) only takes effect at the next decision."""
    out = {}
    for row in reversed(st.influence_log):
        p = row["pattern"]
        if p in out:
            continue
        rec = MaturedRecord("INF-" + stable_hash([p, row["decided"]], 10), row["decided"], {"pattern": p, "influence": row["influence"]},
                            Provenance(created_real or row["decided"], row["evidence"], code_hash or "c68", outcomes_seen_through=row["evidence"]))
        try:
            out[p] = float(rec.gate(now)["influence"])
        except FirewallBreach:
            continue
    return {p.name: out.get(p.name, 1.0) for p in st.cfg.patterns}


def _gain_rows(M: pd.DataFrame, st: C68State) -> pd.DataFrame:
    """Training rows for the production gain model: the newest matured candidate rows, restricted to the promoted fix's window."""
    rows = M.sort_index().iloc[-st.cfg.train_max:]
    share = float(st.production.get("window_share", 1.0))
    if share < 1.0 and len(rows):
        d = pd.to_datetime(rows.index.get_level_values(0))
        cut = d.sort_values()[int(len(d) * (1.0 - share))]
        rows = rows[np.asarray(d >= cut)]
    return rows


# ================================================================================================================ stage: policy and gate
def st_policy(ctx: LP.Ctx) -> tuple:
    """c68.selection_policy. Learn the exit (target-blind), fit the realisable-gain and path models UNDER it, install the band gate in
    the two-stage pipe. Training uses only candidate paths whose last bar is strictly before now."""
    st = _state(ctx)
    cfg = st.cfg
    bv = _bars(ctx)
    M = ctx.obs.matured
    if len(M) == 0:
        raise LP.NoInput("no matured research rows")
    rows = _gain_rows(M, st)
    P, src = build_paths(bv, rows, cfg.horizon, ctx.now)
    pipe = ctx.rt.pipe if ctx.rt.pipe is not None else TS.TwoStage(ctx.state.cfg.two_stage)
    ctx.rt.pipe = pipe
    if len(P) < cfg.min_train:
        st.gate, pipe.band_gate = None, abstain_gate
        return _run(st, ctx, "c68.selection_policy", len(P), 0,
                    f"only {len(P)} matured candidate paths (< {cfg.min_train}): the band gate abstains, no position is allowed")
    rule = XR.LearnedExitRule(cfg.exit_cfg).fit(P, ctx.now)
    pid = SC.policy_id(rule)
    st.policies[pid] = rule
    feats = tuple(cfg.features) + tuple(st.production.get("extra_features", ()))
    gm = SC.RealisableGainModel(rule, feats, cfg.selection).fit(P, feature_matrix(src, feats), ctx.now)
    ctxf = {k: np.nan_to_num(src[k].to_numpy(float)) for k in cfg.path_context if k in src}
    pm = XR.PathModel(rule).fit(P, ctx.now, ctxf) if len(P) >= 20 else None
    infl = released_influence(st, ctx.now, ctx.created_real(), ctx.rt.code_hash)
    today = ctx.obs.today
    pf, fired = {p.name: p.feature for p in cfg.patterns}, fired_today(today, cfg.patterns)
    gate = BandGate(gm, pid, cfg.selection, feats, infl, pf, fired)
    pipe.band_gate = gate
    st.gate, st.path_model = gate, pm
    st.shadow = None
    if st.production.get("name") != "incumbent":
        # while a promoted fix is in production the incumbent keeps forecasting in the shadow (same exit, same paths, its own inputs
        # and window): the monitor compares the two on the same matured names - a fix is judged against what it replaced
        Ps, srcs = build_paths(bv, M.sort_index().iloc[-cfg.train_max:], cfg.horizon, ctx.now)
        base = tuple(cfg.features)
        sm = SC.RealisableGainModel(rule, base, cfg.selection).fit(Ps, feature_matrix(srcs, base), ctx.now)
        st.shadow = BandGate(sm, pid, cfg.selection, base, infl, pf, fired)
    st.gate_log.append(influence_effect(gate, today, ctx.now))
    st.gate_log = st.gate_log[-500:]
    st.trained_through = str(pd.Timestamp(P.end.max()).date())
    st.intended[str(as_date(ctx.now))] = pid
    dg = gm.diagnostics
    return _run(st, ctx, "c68.selection_policy", len(P), 1,
                f"policy {pid[:9]} (threshold {rule.threshold:.4f}); gain model {dg.get('status')} n={dg.get('n')} base in-band "
                f"{dg.get('base_in_band', float('nan')):.3f}; influence {', '.join(f'{k}={v:.2f}' for k, v in infl.items())}")


# ================================================================================================================ stage: expectations
def regime_label(series: np.ndarray, cfg: PE.ErrorConfig, window: int = 20) -> str:
    s = np.asarray(series, float)[-(window + 1):]
    if len(s) < window // 2 + 1 or not np.isfinite(s[[0, -1]]).all() or s[0] <= 0:
        return "UNKNOWN"
    r = s[1:] / s[:-1] - 1.0
    r = r[np.isfinite(r)]
    return PE.realized_regime(float(s[-1] / s[0] - 1.0), float(np.std(r, ddof=1)) if len(r) > 2 else None, window, cfg) or "UNKNOWN"


def _strength(series: pd.Series | None, before, window: int) -> float:
    if series is None:
        return 0.0
    s = series[series.index < pd.Timestamp(as_date(before))].tail(window)
    return float(s.mean()) if len(s) else 0.0


def expectation_context(ctx: LP.Ctx, st: C68State, bv: BarView, day: str, subject: str, sector: str, fired: tuple, entry_at: str) -> dict:
    """Checklist-A context for one position: expected market and sector regime (the trailing state, carried forward), the patterns
    that fire on it with their expected strengths (trailing mean daily effect), the interactions among them."""
    frames, combos = _patterns(ctx, st)
    cfg = st.cfg
    pats = tuple(sorted(fired)) or ("no_pattern",)
    strengths = {p: _strength(frames[p]["effect"] if p in frames else None, entry_at, cfg.strength_window) for p in pats}
    inter = {}
    for i, a in enumerate(pats):
        for b in pats[i + 1:]:
            key = f"{a}|{b}" if f"{a}|{b}" in combos else f"{b}|{a}"
            if key in combos:
                inter[f"{a} x {b}"] = _strength(combos[key], entry_at, cfg.strength_window)
    return {"market_regime": regime_label(bv.market, cfg.error_cfg), "sector_regime": regime_label(bv.sector_index(sector), cfg.error_cfg),
            "patterns": pats, "pattern_strengths": strengths, "interactions": inter, "entry_at": entry_at,
            "timestamp": f"{as_date(day).isoformat()}T15:59:00"}


class ExpectationModel:
    """The `path_model(row, subject)` of expectations_from_day. The committed prediction IS the number that selected the name: the
    predicted realisable return and its distribution come from the band gate's forecast (the realisable-gain model's median and its
    split-conformal draws, with pattern influence applied); the trajectory fields the gain model does not forecast (time to peak, exit
    window, holding period, excursions, volatility, epistemic uncertainty) come from exit_research.PathModel under the same learned exit.
    One forecaster per quantity - never two numbers for the same return."""

    def __init__(self, pm: XR.PathModel, gate: BandGate, forecasts: Mapping[str, SC.GainForecast]):
        self.pm, self.gate, self.fc = pm, gate, dict(forecasts)

    def __call__(self, row: Any, subject: str) -> dict:
        tr = dict(self.pm(row, subject))
        fc = self.fc.get(str(subject))
        m = self.gate.model
        if fc is None or not math.isfinite(fc.mean) or m.resid_ is None:
            raise XP.ExpectationInvalid(f"{subject}: no band-gate forecast to commit to")
        draws = fc.mean + m.resid_
        ret = float(fc.median)
        qv = np.maximum.accumulate(np.quantile(draws, XR.QUANTS))
        clipped = np.clip(draws, XR.PROB_EDGES[0], XR.PROB_EDGES[-1] - 1e-9)
        probs = np.histogram(clipped, bins=XR.PROB_EDGES)[0].astype(float)
        probs /= probs.sum()
        probs[-1] = 1.0 - probs[:-1].sum()
        loss_p, over_p = float((draws < 0).mean()), float((draws > EX.BAND_HI).mean())
        alts = [{"hypothesis": "reversal_to_loss", "probability": loss_p, "predicted_return": float(draws[draws < 0].mean())}] if loss_p > 0 else []
        if over_p > 0:
            alts.append({"hypothesis": "overshoot_above_band", "probability": over_p, "predicted_return": float(draws[draws > EX.BAND_HI].mean())})
        tr.update(predicted_return=ret, distribution=tuple((float(q), float(v)) for q, v in zip(XR.QUANTS, qv)),
                  prob_distribution=tuple((XR.PROB_EDGES[i], XR.PROB_EDGES[i + 1], float(probs[i])) for i in range(len(probs))),
                  mfe=max(float(tr["mfe"]), ret, 0.0), mae=min(float(tr["mae"]), ret, 0.0),
                  uncertainty={"aleatoric": float(np.std(m.resid_)), "epistemic": float(tr["uncertainty"]["epistemic"])},
                  alternatives=tuple(alts) or ({"hypothesis": "flat", "probability": 0.0, "predicted_return": 0.0},))
        return tr


def _subday(dec: TS.DayDecision, ix) -> TS.DayDecision:
    t = dec.table
    keep = (t.index == ix) | ((t["side"] == TS.FLAT) & t["mover"].astype(bool)).to_numpy()
    return TS.DayDecision(dec.decided_at, t[keep], dec.funnel, dec.gate, dec.knowledge_digest)


def st_expect(ctx: LP.Ctx) -> tuple:
    """c68.expectations. Make sure today's decision is band-gated, then freeze a checklist-A expectation and a calibration commitment
    for every position BEFORE its next-open fill, and log the research-side forecast of every name (identification / self-correction)."""
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    if not ctx.state.decisions:
        raise LP.NoInput("no two-stage decision this cycle")
    dec = ctx.state.decisions[-1]
    if dec.decided_at in st.decided:
        raise LP.NoInput(f"decision {dec.decided_at} already has its expectations")
    today = ctx.obs.today
    gate = st.gate if st.gate is not None else abstain_gate
    gated = TS.apply_band_gate(dec, gate, today, ctx.now)
    if gated is not dec:
        ctx.state.decisions[-1] = gated
        st.count("decisions_regated")
    dec = gated
    st.decided.append(dec.decided_at)
    bv = _bars(ctx)
    if isinstance(gate, BandGate) and len(today):
        fcs = gate.forecast(today, ctx.now)
        el = SC.select(fcs, ctx.now, gate.policy, gate.cfg)
        ok = set(el.eligible)
        cols = [c for c in (*cfg.features, *cfg.market_features) if c in today]
        vals = today[cols].astype(float).to_numpy() if cols else np.zeros((len(today), 0))
        secs = today["sector"].astype(str).to_numpy() if "sector" in today else np.full(len(today), "all", object)
        shadow = getattr(st, "shadow", None)
        sh = {fc.candidate: fc.median for fc in shadow.forecast(today, ctx.now)} if isinstance(shadow, BandGate) else {}
        st.forecasts[dec.decided_at] = {fc.candidate: {"mean": fc.mean, "median": fc.median, "p_band": fc.p_in_band, "eligible": fc.candidate in ok,
                                                       "policy": gate.policy, **dict(zip(cols, map(float, vals[k]))), "sector": str(secs[k]),
                                                       "shadow": float(sh.get(fc.candidate, np.nan))}
                                        for k, fc in enumerate(fcs) if math.isfinite(fc.mean)}
    pos = dec.positions
    if not len(pos):
        return _run(st, ctx, "c68.expectations", len(dec.table), 0, f"no position on {dec.decided_at}: {dict(dec.reasons())}")
    if st.path_model is None or not isinstance(gate, BandGate):
        raise FirewallBreach(f"{len(pos)} position(s) on {dec.decided_at} without a fitted path model and band gate")
    entry_at = next_session(ctx, dec.decided_at)
    matures = session_after(ctx, dec.decided_at, cfg.horizon)
    fired = fired_today(today, cfg.patterns)
    entries = {}
    for ix, row in pos.iterrows():
        t = today.loc[ix]
        entries[str(ix[-1])] = {"vol": float(t["vol20"]) if "vol20" in t and math.isfinite(float(t["vol20"])) else 0.02,
                                "atr": float(t["atr"]) if "atr" in t and math.isfinite(float(t["atr"])) else 0.02,
                                "kind": str(t["sector"]) if "sector" in t else "all",
                                **{k: float(np.nan_to_num(float(t[k]))) for k in cfg.path_context if k in t}}
    check = gate(today.loc[pos.index], ctx.now)                   # the canonical gate re-asked about every position (C69 section 31)
    bypass = [str(ix[-1]) for ix in pos.index if not bool(check.loc[ix, "eligible"])]
    if bypass:
        st.count("band_bypass_refused")
        raise FirewallBreach(f"{len(bypass)} position(s) on {dec.decided_at} are not eligible under the fitted 5-10% gate: another "
                             f"selector bypassed it; no expectation is recorded for a decision the constraint did not make")
    pm = ExpectationModel(st.path_model.bind(entries, ctx.now), gate, dict(gate.last))
    gm = gate.model
    version = f"{gate.policy}|{gm.digest()[:10]}|{str(ctx.rt.code_hash)[:8]}"
    info = {"bars": str(bv.sessions[-1].date()), "decision_frame": dec.decided_at, "gain_model": min(st.trained_through, dec.decided_at),
            "knowledge": dec.decided_at}
    recorded = 0
    committed = {c.pred_id for c in led.book.commitments()}
    for ix, row in pos.iterrows():
        tk = str(ix[-1])
        ctxd = expectation_context(ctx, st, bv, dec.decided_at, tk, entries[tk]["kind"], fired.get(tk, ()), entry_at)
        exp = XP.expectations_from_day(_subday(dec, ix), pm, ctxd, ctx.now, model_version=version, information_set=info,
                                       feature_columns=list(cfg.features), today=today)[0]
        pid = led.expectations.record(exp, ctx.now)
        if pid not in committed:
            committed.add(pid)
            led.book.commit(pid, exp, gate.policy, st.trained_through, matures, ctx.now, cfg.target)
        st.pending[pid] = Position(pid, tk, dec.decided_at, entry_at, gate.policy, entries[tk]["kind"], entries[tk]["vol"],
                                   entries[tk]["atr"], tuple(ctxd["patterns"]))
        fc = gate.last.get(tk)
        if led.pipe.last_step(pid) is None:
            led.pipe.add(pid, "PREDICTION", ctx.now, {"side": int(row["side"]), "p_move": float(row["p_move"]), "p_up": float(row["p_up"]),
                                                      "gain_forecast": None if fc is None else fc.median,
                                                      "p_band": None if fc is None else fc.p_in_band, "policy": gate.policy})
            led.pipe.add(pid, "EXPECTATION", ctx.now, {"content_hash": exp.content_hash, "predicted_return": exp.predicted_return,
                                                       "confidence": exp.confidence, "entry_at": exp.entry_at, "committed": True})
        recorded += 1
    st.count("expectations", recorded)
    return _run(st, ctx, "c68.expectations", len(pos), recorded, f"{recorded} expectation(s) frozen before the {entry_at} fill")


# ================================================================================================================ stage: market and regime
def market_day(bv: BarView, t: int, err_z: Sequence[float]) -> CP.DayInputs:
    """One session's market-level inputs for change_points.build_streams (volatility, dispersion, breadth, correlation, errors)."""
    R = bv.returns()
    r = R[t]
    ok = np.isfinite(r)
    win = R[max(1, t - 19): t + 1].T
    win = win[np.isfinite(win).all(1)]
    ez = np.asarray([z for z in err_z if math.isfinite(z)], float)
    return CP.DayInputs(returns_window=win if win.shape[1] >= 10 else None, day_abs_move=float(np.mean(np.abs(r[ok]))) if ok.sum() >= 5 else None,
                        dispersion=float(np.std(r[ok])) if ok.sum() >= 5 else None, breadth_up=float(np.mean(r[ok] > 0)) if ok.sum() >= 5 else None,
                        error_z=ez if len(ez) else None)


def sector_volatility(bv: BarView, t: int, min_names: int = 4) -> dict:
    """Per sector, log mean |return| of the session: a market-wide volatility change moves every sector's stream at once (evidence
    for the MARKET-wide scope), a sector change moves one."""
    r = bv.returns()[t]
    out = {}
    for sec in sorted(set(bv.sector.tolist())):
        x = r[(bv.sector == sec) & np.isfinite(r)]
        if len(x) >= min_names and np.mean(np.abs(x)) > 0:
            out[str(sec)] = float(math.log(np.mean(np.abs(x))))
    return out


def unit_values(bv: BarView, t: int) -> dict:
    """Per-name stream: the size of the session's IDIOSYNCRATIC move, |return - market return| over its robust (MAD) sd in a reference
    window that ends 20 sessions before t (sessions before t only). Serially independent under the null, so the calibrated CUSUM keeps its false-alarm rate; a name that
    enters its own turbulent state (a single-stock anomaly) shifts it for as long as the state lasts."""
    R = bv.returns()
    with np.errstate(invalid="ignore", divide="ignore"):
        mr = np.r_[np.nan, bv.market[1:] / bv.market[:-1] - 1.0]
        E = R - mr[:, None]
        lo, hi = max(1, t - 80), t - 20                  # a LAGGED reference: a turbulent spell cannot quietly become its own baseline
        ref = E[lo:hi]
        sd = 1.4826 * np.nanmedian(np.abs(ref - np.nanmedian(ref, axis=0)), axis=0) if hi - lo >= 30 else np.full(R.shape[1], np.nan)
        a = np.abs(E[t]) / sd
    # |z| of a normal return is half-normal; its normal score Phi^-1(2 Phi(|z|) - 1) is N(0,1) again, which is what the CUSUM's
    # threshold was calibrated on (a raw |z| stream false-alarms several times too often)
    from scipy.stats import norm
    u = np.clip(2.0 * norm.cdf(np.where(np.isfinite(a), a, 0.0)) - 1.0, 1e-9, 1 - 1e-9)
    z = np.where(np.isfinite(a), norm.ppf(u), np.nan)
    return {unit_id(tk): (float(v) if math.isfinite(v) else None) for tk, v in zip(bv.tickers, z)}


def unit_id(ticker: str) -> str:
    """A stable, identity-free stream name for one instrument (the research world may know the name; the stream label need not)."""
    return "u:" + stable_hash(str(ticker), 10)


def market_obs(bv: BarView, t_end: int, horizon: int) -> ME.MarketObs | None:
    """The period (last `horizon` sessions ending at t_end) as a market_expectations observation: opportunity (share of names that
    moved >= 5%), volatility, breadth, dispersion, correlation, momentum persistence, reversal probability, gap behaviour, movers and
    qualifying 5-10% opportunities, holding period and the best exit day of the movers."""
    a = t_end - horizon
    if a < 25:
        return None
    C, O = bv.C, bv.O
    ret = C[t_end] / C[a] - 1.0
    ok = np.isfinite(ret)
    if ok.sum() < 10:
        return None
    r = ret[ok]
    prior = (C[a] / C[a - 20] - 1.0)[ok]
    prev_w = (C[a] / C[a - horizon] - 1.0)[ok]
    path = C[a + 1:t_end + 1][:, ok] / C[a][ok] - 1.0
    movers = np.abs(r) >= 0.05
    qual = (r >= 0.05) & (r <= 0.10)
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = np.abs(O[a + 1:t_end + 1] / C[a:t_end] - 1.0)
    R = bv.returns()[a + 1:t_end + 1][:, ok]
    cm = np.corrcoef(R.T) if R.shape[0] >= 3 else None
    corr = float(np.nanmean(cm[np.triu_indices_from(cm, 1)])) if cm is not None and np.isfinite(cm).any() else None
    big = np.abs(prev_w) >= 0.05
    rev = float(np.mean(np.sign(r[big]) != np.sign(prev_w[big]))) if big.sum() >= 3 else None
    pm = np.isfinite(prior) & np.isfinite(r)
    mom = float(pd.Series(prior[pm]).corr(pd.Series(r[pm]), method="spearman")) if pm.sum() >= 10 else None
    best = (np.nanargmax(np.abs(path[:, movers]), axis=0) + 1).astype(float) if movers.any() else np.array([])
    vals = {"opportunity": float(movers.mean()), "volatility": float(np.mean(np.abs(r))), "breadth": float(np.mean(r > 0)),
            "dispersion": float(np.std(r)), "correlation": corr, "momentum_persistence": mom, "reversal_probability": rev,
            "gap_behavior": float(np.nanmedian(gaps)) if np.isfinite(gaps).any() else None, "n_movers": float(movers.sum()),
            "n_qualifying": float(qual.sum()), "holding_period": float(horizon), "optimal_exit_days": float(best.mean()) if len(best) else None}
    return ME.MarketObs(str(bv.sessions[t_end].date()), {k: v for k, v in vals.items() if v is None or math.isfinite(v)})


def _err_z_by_day(st: C68State, days: Sequence[str]) -> dict[str, list]:
    """Standardised return errors placed on the first session at/after the day the SYSTEM learned them (the cycle that computed
    them), never on their earlier maturity date: the early-warning stream may only see an error once the error engine had it."""
    out: dict[str, list] = {}
    keep = []
    for known, z in st.pending_err_z:
        d = next((x for x in days if x >= known), None)
        if d is None:
            keep.append((known, z))
        else:
            out.setdefault(d, []).append(z)
    st.pending_err_z = keep
    return out


def st_market(ctx: LP.Ctx) -> tuple:
    """c68.market_regime. Advance, one session at a time and strictly in date order, the early-warning streams (market + single-name),
    the weekly market expectation (resolve the last one, investigate a big miss, expect the next) and the regime memory."""
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    bv = _bars(ctx)
    new_days, alarms = 0, 0
    start = 25
    if st.ews_through:
        start = max(start, int(bv.sessions.searchsorted(pd.Timestamp(st.ews_through), side="right")))
    ez = _err_z_by_day(st, [str(d.date()) for d in bv.sessions[start:]])
    unit_sector = {unit_id(t): str(s) for t, s in zip(bv.tickers, bv.sector)}
    ew = None
    for t in range(start, bv.T):
        day = str(bv.sessions[t].date())
        di = market_day(bv, t, ez.get(day, ()))
        vals, tg, _ = CP.build_streams(di)
        for sec, v in sector_volatility(bv, t).items():
            vals[f"sector_vol:{sec}"], tg[f"sector_vol:{sec}"] = v, CP.Target.SECTOR       # a sector's behaviour (checklist I)
        ew = CP.step(st.ews, day, vals, tg, unit_values=unit_values(bv, t), unit_sector=unit_sector, unit_target=CP.Target.VOLATILITY)
        st.warnings.append((day, ew.level.value, ew.scope.scope.value))
        d = st.daily
        d["dates"].append(day)
        for q in [k for k in vals if k in ("volatility", "dispersion", "breadth", "correlation") or k.startswith("sector_vol:")]:
            d["q"].setdefault(q, [np.nan] * (len(d["dates"]) - 1)).append(np.nan if vals.get(q) is None else float(vals[q]))
        for q in d["q"]:
            if len(d["q"][q]) < len(d["dates"]):
                d["q"][q].append(np.nan)
        z = ez.get(day, ())
        d["error_z"].append(float(np.mean(z)) if len(z) else np.nan)
        alarms += len([x for x in ew.detections if x.alarm_date == day])
        st.ews_through = day
        new_days += 1
    keep = cfg.history_days
    if len(st.daily["dates"]) > keep:
        st.daily = {"dates": st.daily["dates"][-keep:], "q": {k: v[-keep:] for k, v in st.daily["q"].items()},
                    "error_z": st.daily["error_z"][-keep:]}
    st.warnings = st.warnings[-keep:]
    n_market = 0
    frames, _ = _patterns(ctx, st)
    for t in week_end_positions(ctx, bv):
        day = str(bv.sessions[t].date())
        if st.market_through and day <= st.market_through:
            continue
        ob = market_obs(bv, t, cfg.horizon)
        if ob is None:
            continue
        pe = {p: f["effect"][f.index <= bv.sessions[t]].tail(60).to_numpy(float) for p, f in frames.items()}
        res = ME.step(st.market, ctx.now, ob, pattern_effects=pe, next_period=next_week_end(ctx, day))
        st.market.feed_tracker(st.tracker, res, ctx.now)
        st.market_through = day
        n_market += 1
    days = set(st.daily["dates"])
    dets = [x for x in st.ews.market.detections if x.alarm_date in days and x.change_date in days]
    ev = RM.RegimeEvidence(tuple(st.daily["dates"]), tuple(dets), {k: list(v) for k, v in st.daily["q"].items()},
                           {p: _aligned(f["effect"], st.daily["dates"]) for p, f in _patterns(ctx, st)[0].items()}, tuple(st.daily["error_z"]))
    if ew is not None:
        st.active = tuple(ew.fired())
        st.last_scope = ew.scope.scope.value
    ms = RM.step(st.memory, ctx.now, ev, scope=ew.scope if ew is not None else None, active_precursors=st.active)
    if not new_days and not n_market:
        raise LP.NoInput("no new session before now")
    return _run(st, ctx, "c68.market_regime", new_days + n_market, alarms + len(ms.new_records),
                f"{new_days} session(s), {alarms} alarm(s), warning {ew.level.value if ew else 'n/a'} ({ew.scope.scope.value if ew else 'n/a'}); "
                f"{n_market} market period(s); regime records +{len(ms.new_records)}, status changes {list(ms.status_changes)}")


def _aligned(s: pd.Series, dates: Sequence[str]) -> list:
    m = {str(k.date()): float(v) for k, v in s.items()}
    return [m.get(d, np.nan) for d in dates]


# ================================================================================================================ stage: pattern change + influence
def st_patterns(ctx: LP.Ctx) -> tuple:
    """c68.pattern_change. Classify every pattern (pattern_change -> break_research's detector), then combine the verdict with the
    regime memory's guard (a regime alarm alone never switches a working pattern off) into the pattern's influence. A failing pattern
    is never deleted; before it has been investigated it keeps at least half its influence."""
    st = _state(ctx)
    cfg = st.cfg
    frames, _ = _patterns(ctx, st)
    if not frames:
        raise LP.NoInput("no pattern outcomes yet")
    rep = PC.step(st.pcs, ctx.now, frames, dict(cfg.pattern_cfg))
    recs = st.memory.records(ctx.now)
    latest = recs[-1] if recs else None
    status = st.memory.status(latest.record_id) if latest is not None else None
    adv = st.memory.advice(st.active, ctx.now)
    changed = 0
    for v in rep.verdicts:
        p = v.pattern_id
        st.verdicts[p] = v.change.value
        base = CLASS_INFLUENCE.get(v.change)
        if base is None:
            base = st.influence.get(p, 1.0)
        elif v.change in PC.FAILING and p not in st.pcs.investigated:
            base = max(base, UNINVESTIGATED_FLOOR)
        g = RM.PatternAction.KEEP
        if latest is not None:
            eff = frames[p]["effect"]
            cd = pd.Timestamp(latest.change_date)
            # the pattern's OWN current evidence since the change decides (its newest `guard_window` rows): a pattern that has since
            # recovered is judged on the recovery, not on the whole spell it spent failing
            gd = RM.pattern_guard(p, status, eff[eff.index < cd].to_numpy(float), eff[eff.index >= cd].to_numpy(float)[-cfg.guard_window:],
                                  p in adv.weaken, st.memory.cfg)
            g = gd.action
        st.guard[p] = g.value
        new = float(min(base, GUARD_INFLUENCE[g]))
        if abs(new - st.influence.get(p, 1.0)) > 1e-12:
            changed += 1
            ctx.bus.setdefault("events", []).append(_pattern_event(ctx, p, v, st.influence.get(p, 1.0), new))
        st.influence[p] = new
        st.influence_log.append({"pattern": p, "influence": new, "decided": str(as_date(ctx.now)), "change": v.change.value,
                                 "guard": g.value, "evidence": str(frames[p].index[frames[p].index < pd.Timestamp(as_date(ctx.now))].max().date())
                                 if len(frames[p]) else str(as_date(ctx.now))})
    st.influence_log = st.influence_log[-5000:]
    for pid in rep.investigations_needed:
        ev = _pattern_event(ctx, pid, st.pcs.latest[pid], st.influence.get(pid, 1.0), st.influence.get(pid, 1.0))
        ctx.bus.setdefault("events", []).append(ev)
    return _run(st, ctx, "c68.pattern_change", len(frames), changed,
                f"verdicts {dict(rep.counts)}; influence {', '.join(f'{k}={v:.2f}' for k, v in st.influence.items())}; "
                f"regime {status.value if status else 'none'}; {len(rep.investigations_needed)} to investigate")


def _pattern_event(ctx: LP.Ctx, p: str, v, old: float, new: float):
    from engine.research import questions as Q
    through = str(v.profile.as_of) if v is not None else str(as_date(ctx.now))
    last = ctx.evidence_date() if ctx.rt.obs is not None and ctx.obs.evidence_through else through
    ev_through = min(str(as_date(last)), str((pd.Timestamp(as_date(ctx.now)) - pd.Timedelta(days=1)).date()))
    mag = float(min(1.0, abs(v.profile.z) / 6.0)) if v is not None and math.isfinite(v.profile.z) else 0.5
    return Q.QuestionEvent("pattern_break", f"pattern {p}", ev_through, mag, stake=float(min(1.0, max(0.1, 1.0 - new))),
                           problem=Problem.VOLATILITY, contexts={"change": v.change.value if v is not None else "?",
                                                                 "influence": f"{old:.2f}->{new:.2f}"},
                           detail=f"pattern {p} {v.change.value if v is not None else ''}; influence {old:.2f} -> {new:.2f}")


# ================================================================================================================ stage: outcomes and errors
def position_paths(bv: BarView, p: Position, horizon: int, now) -> tuple[EX.Paths, int] | None:
    rows = pd.DataFrame({"vol20": [p.vol], "atr": [p.atr], "sector": [p.sector]},
                        index=pd.MultiIndex.from_tuples([(pd.Timestamp(p.decided_at), p.ticker)]))
    P, _ = build_paths(bv, rows, horizon, now)
    if not len(P):
        return None
    return P, bv.pos_after(p.decided_at)


def path_data(bv: BarView, p: Position, pos0: int, exit_pos: int, exit_price: float, frames: Mapping, combos: Mapping,
              patterns: Sequence[str], interactions: Sequence[str], post: int, peers: int) -> OC.PathData:
    """The raw material outcomes.reconstruct needs, all strictly before now: bars from the fill through `post` sessions after the
    exit, the market proxy, the equal-weight sector index, same-sector peers, the patterns' daily effects over the hold."""
    j = bv.col[p.ticker]
    hi = min(bv.T, exit_pos + 1 + post)
    sl = slice(pos0, hi)
    dates = tuple(str(d.date()) for d in bv.sessions[sl])
    sec = bv.sector_index(p.sector)
    peer_j = [bv.col[t] for t, s in zip(bv.tickers, bv.sector) if s == p.sector and t != p.ticker][:peers]
    ser = {k: tuple(_aligned(frames[k]["effect"], dates)) for k in patterns if k in frames}
    hold_dates = dates[: exit_pos - pos0 + 1]
    real = {k: float(np.nanmean(v[: len(hold_dates)])) for k, v in ser.items() if np.isfinite(v[: len(hold_dates)]).any()}
    ireal = {}
    for key in interactions:
        a, b = [x.strip() for x in key.split(" x ")]
        c = combos.get(f"{a}|{b}", combos.get(f"{b}|{a}"))
        if c is not None:
            v = np.asarray(_aligned(c, hold_dates), float)
            if np.isfinite(v).any():
                ireal[key] = float(np.nanmean(v))
    return OC.PathData(dates, tuple(bv.O[sl, j]), tuple(bv.H[sl, j]), tuple(bv.L[sl, j]), tuple(bv.C[sl, j]), hold_dates[-1], exit_price,
                       tuple(bv.V[sl, j]), tuple(bv.market[sl]), float(bv.market[pos0 - 1]), tuple(sec[sl]), float(sec[pos0 - 1]),
                       {f"peer{i}": tuple(bv.C[sl, pj]) for i, pj in enumerate(peer_j)}, {f"peer{i}": float(bv.C[pos0 - 1, pj]) for i, pj in enumerate(peer_j)},
                       ser, real, ireal)


def run_exit(rule: EX.Rule, P: EX.Paths) -> EX.ExitResult:
    """One learned-policy exit on realised paths. Its inputs are the path and the rule - no prediction, expectation, target or
    tolerance exists in this call (checklist O)."""
    return rule.run(P)


def decide_exits(st: C68State, items: Sequence[tuple[str, EX.Paths]]) -> list[EX.ExitResult]:
    """THE exit decision of every C68 position: the policy each position was COMMITTED under, run on its realised path. The state it
    is given also holds the +-1pp evaluation target; `exit_independence` re-runs this very function under other targets and demands
    bit-identical exits, so a future edit that lets the target leak into an exit is caught."""
    return [run_exit(st.policies[pid], P) for pid, P in items]


def _concat(results: Sequence[EX.ExitResult]) -> EX.ExitResult:
    z = np.zeros(0)
    if not results:
        return EX.ExitResult(z, z, z.astype(int), z.astype(int), z, 0)
    cat = lambda f: np.concatenate([getattr(r, f) for r in results])      # noqa: E731
    return EX.ExitResult(cat("net"), cat("gross"), cat("days"), cat("reason"), cat("stop_overshoot"), results[0].D)


def st_outcomes(ctx: LP.Ctx) -> tuple:
    """c68.outcomes_errors. For every position whose path ended before now: the committed learned exit decides the exit, the whole path
    is reconstructed against the frozen expectation, the ten errors are computed (feeding the shared surprise tracker), and the exit is
    recorded for the honest +-1pp statistic. Also the realised gain of every name forecast on a resolved day (identification)."""
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    bv = _bars(ctx)
    frames, combos = _patterns(ctx, st)
    done = 0
    skipped: dict[str, str] = {}
    exited_now: list = []
    for pid in sorted(st.pending):
        p = st.pending[pid]
        got = position_paths(bv, p, cfg.horizon, ctx.now)
        if got is None:
            continue
        P, pos0 = got
        res = decide_exits(st, [(p.policy_id, P)])[0]
        exited_now.append((p.policy_id, P))
        days = int(res.days[0])
        exit_pos = pos0 + days - 1
        exp = led.expectations.get(pid)
        meta = led.expectations.meta(pid)
        price = float(P.o[0, 0] * (1.0 + res.net[0]))              # net of costs: the SAME realised return the +-1pp book grades
        pdata = path_data(bv, p, pos0, exit_pos, price, frames, combos, exp.patterns, tuple(exp.interactions), cfg.post_exit, cfg.peers)
        try:
            out = OC.reconstruct(exp, meta["content_hash"], pdata, ctx.now)
        except (OC.OutcomeUnavailable, FirewallBreach) as e:
            skipped[pid] = str(e)[:120]
            continue
        led.outcomes.add(out, ctx.now)
        st.exit_records.append(CT.exit_record_from_policy(pid, out.exit_at, float(res.net[0]), p.policy_id))
        led.pipe.add(pid, "OUTCOME", ctx.now, {"exit_at": out.exit_at, "exit_return": out.exit_return, "net": float(res.net[0]),
                                               "held": out.holding_days, "decided_by": CT.LEARNED_POLICY, "policy": p.policy_id,
                                               "mfe": out.mfe, "mae": out.mae, "regret": out.regret})
        del st.pending[pid]
        done += 1
    st.exited = exited_now
    reports = led.errors.step(led.outcomes, ctx.now)
    st.pending_err_z += [(str(as_date(ctx.now)), float(r["return"].z)) for r in reports if r["return"].z is not None and math.isfinite(r["return"].z)]
    for rep in reports:
        led.pipe.add(rep.prediction_id, "ERROR", ctx.now, {k: v for k, v in rep.vector().items()})
        led.pipe.add(rep.prediction_id, "CLASSIFIED", ctx.now, {"diagnosis": list(rep.diagnosis), "confident_wrong": rep.confident_wrong,
                                                                "severity": rep.severity(3)})
        st.wc_queue.append(rep.prediction_id)
    ident = _resolve_forecasts(st, bv, ctx.now)
    st.count("outcomes", done)
    st.count("errors", len(reports))
    if not done and not reports and not ident:
        raise LP.NoInput(f"{len(st.pending)} open position(s), none resolved before now")
    return _run(st, ctx, "c68.outcomes_errors", done + len(skipped), len(reports),
                f"{done} exit(s) by the learned policy, {len(reports)} error report(s), {len(skipped)} unreconstructable, "
                f"{ident} candidate gain(s) resolved")


def _resolve_forecasts(st: C68State, bv: BarView, now) -> int:
    """Realised gain, under the policy intended on that day, of every name the gate forecast (research side: the identification
    curve and the self-correction frame need the unselected names too)."""
    n = 0
    for day, fc in sorted(st.forecasts.items()):
        have = st.realised.setdefault(day, {})
        todo = [t for t in fc if t not in have]
        if not todo:
            continue
        pol = st.policies.get(fc[todo[0]]["policy"])
        if pol is None:
            continue
        rows = pd.DataFrame({"vol20": [fc[t].get("vol20", 0.02) for t in todo], "atr": [fc[t].get("atr", 0.02) for t in todo],
                             "sector": [fc[t]["sector"] for t in todo]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp(day), t) for t in todo]))
        P, src = build_paths(bv, rows, st.cfg.horizon, now)
        if not len(P):
            continue
        res = run_exit(pol, P)
        pos0 = bv.pos_after(day)
        for k, t in enumerate(src.index.get_level_values(-1)):
            have[str(t)] = (float(res.net[k]), str(bv.sessions[pos0 + int(res.days[k]) - 1].date()))
            n += 1
    return n


# ================================================================================================================ stage: what changed
def error_case(bv: BarView, st: C68State, pid: str, exp: XP.Expectation, out: OC.OutcomeReconstruction, frames: Mapping, combos: Mapping,
               now) -> tuple[WC.ErrorCase, Any]:
    """The checklist-J case of one matured error, from bars strictly before now: histories cut at the decision, the hold path, the
    post-maturity path (hindsight, used by the timing level only), peers, pattern histories and a knowability assessment."""
    j = bv.col[exp.subject]
    R = bv.returns()
    S = bv.sessions
    d0 = int(S.searchsorted(pd.Timestamp(exp.decided_at), side="right")) - 1
    e0 = d0 + 1
    e1 = int(S.searchsorted(pd.Timestamp(out.exit_at), side="left"))
    idx_h = S[max(1, d0 - 150): d0 + 1]
    hist = pd.Series(R[max(1, d0 - 150): d0 + 1, j], index=idx_h)
    path = pd.Series(R[e0: e1 + 1, j] * exp.direction, index=S[e0: e1 + 1]).fillna(0.0)
    mr = pd.Series(bv.market, index=S).pct_change(fill_method=None)
    sec = pd.Series(bv.sector_index(str(bv.sector[j])), index=S).pct_change(fill_method=None)
    peers_j = [k for k, s in enumerate(bv.sector) if s == bv.sector[j] and k != j]
    fc = st.forecasts.get(exp.decided_at, {})
    pr = []
    for k in peers_j:
        c0, c1 = bv.C[d0, k], bv.C[e1, k]
        if math.isfinite(c0) and math.isfinite(c1) and c0 > 0:
            f = fc.get(bv.tickers[k])
            pr.append({"realised": c1 / c0 - 1.0, **({"predicted": f["median"]} if f and math.isfinite(f.get("median", np.nan)) else {})})
    peers = pd.DataFrame(pr) if pr else None
    if peers is not None and "predicted" in peers and peers["predicted"].isna().any():
        peers = peers.drop(columns=["predicted"])
    after = pd.Series(R[e1 + 1: min(bv.T, e1 + 6), j] * exp.direction, index=S[e1 + 1: min(bv.T, e1 + 6)]).dropna()
    kn = None
    try:
        a = max(0, d0 - 60)
        b = pd.DataFrame({"open": bv.O[a:, j], "high": bv.H[a:, j], "low": bv.L[a:, j], "close": bv.C[a:, j], "volume": bv.V[a:, j]},
                         index=S[a:]).loc[: S[min(bv.T - 1, e1 + 3)]]
        mv = KN.MoveEvent("K" + pid[1:12].translate(_NO_DIGITS), exp.subject, exp.decided_at, exp.entry_at, out.exit_at, float(bv.C[e1, j] / bv.C[d0, j] - 1.0),
                          max(1, e1 - d0), sector=str(bv.sector[j]))
        kn = KN.classify_move(KN.MoveInputs(mv, b, market=mr.loc[b.index].fillna(0.0)))
    except (KN.KnowabilityError, KeyError, ValueError, IndexError):
        kn = None
    cut = pd.Timestamp(exp.decided_at)
    pf = {p: frames[p][["effect"]].loc[:cut].tail(PATTERN_ROWS) for p in exp.patterns if p in frames}
    cb = {k: v for k, v in combos.items() if all(x in exp.patterns for x in k.split("|"))}
    case = WC.ErrorCase(case_id(pid), exp.decided_at, out.matured_at, float(exp.predicted_return), float(out.exit_return), hist, path,
                        mr.loc[idx_h], mr.iloc[e0: e1 + 1], sec.loc[idx_h], sec.iloc[e0: e1 + 1], peers, 0.0, pf, cb,
                        after if len(after) else None, float(out.exit_return), kn, float(exp.confidence))
    return case, kn


_NO_DIGITS = str.maketrans("0123456789", "ghijklmnop")


def case_id(pid: str) -> str:
    """The what-changed case id of a prediction: its hash with the digits mapped to letters, so no run of digits in a hash can be read
    as a year by the identity firewall (a real collision: 'C482d0b705d2048')."""
    return "C" + str(pid)[1:].translate(_NO_DIGITS)


def investigation_context(st: C68State, exp: XP.Expectation, out: OC.OutcomeReconstruction, kn, gate: BandGate | None) -> ER.InvestigationContext:
    """What the research world measured about one error, for the checklist-E questions. Absent evidence stays None (UNANSWERED)."""
    shifts = None
    if gate is not None and gate.model.mu_ is not None:
        shifts = {}
        for i, f in enumerate(gate.features):
            v = exp.feature_state.get(f)
            if v is not None and math.isfinite(float(v)) and gate.model.sd_[i] > 0:
                shifts[f] = float((float(v) - gate.model.mu_[i]) / gate.model.sd_[i])
    top = max(exp.pattern_strengths, key=lambda k: (abs(exp.pattern_strengths[k]), k)) if exp.pattern_strengths else ""
    v = st.pcs.latest.get(top)
    before = v.profile.hist_reliability if v is not None else None
    after = v.profile.recent_reliability if v is not None else None
    fc = st.forecasts.get(exp.decided_at, {})
    rl = st.realised.get(exp.decided_at, {})
    sector = fc.get(exp.subject, {}).get("sector")
    miss = [np.sign(rl[t][0] - fc[t]["median"]) for t in fc
            if t in rl and t != exp.subject and fc[t].get("sector") == sector and math.isfinite(fc[t]["median"])]
    share = float(np.mean(np.array(miss) == np.sign(out.exit_return - exp.predicted_return))) if len(miss) >= 3 else None
    return ER.InvestigationContext(feature_shifts=shifts, pattern_reliability_before=before, pattern_reliability_after=after,
                                   overriding_pattern=None, interaction_change_z=None,
                                   vol_ratio=float(out.realized_vol / exp.predicted_volatility) if math.isfinite(out.realized_vol) and exp.predicted_volatility > 0 else None,
                                   corr_change=None, sector_error_share=share, timing_shift_days=float(out.t_max - exp.time_to_peak),
                                   exit_regret=float(out.regret / abs(exp.predicted_return)) if abs(exp.predicted_return) > 1e-9 else None,
                                   model_inputs=tuple(exp.feature_state), knowability=kn, precursor_hits=None)


def st_what_changed(ctx: LP.Ctx) -> tuple:
    """c68.what_changed. Investigate the newest matured errors (largest |z| first, at most max_investigations per cycle - the rest wait,
    none is dropped): ten levels, the five-way knowability class, the conclusion chain, a hypothesis tree in the loop's own forest and a
    checklist-V claim; also the checklist-E context error_research reads for confident-wrong predictions."""
    from engine.research import hypothesis_tree as HT
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    queue = [p for p in st.wc_queue if p not in st.investigated]
    if not queue:
        raise LP.NoInput("no newly matured error to investigate")
    bv = _bars(ctx)
    frames, combos = _patterns(ctx, st)
    zs = {p: abs(led.errors.report(p)["return"].z or 0.0) for p in queue}
    todo = sorted(queue, key=lambda p: (-zs[p], p))[: cfg.max_investigations]
    cases, kns = [], {}
    for pid in todo:
        exp, out = led.expectations.get(pid), led.outcomes.get(pid, ctx.now)
        try:
            case, kn = error_case(bv, st, pid, exp, out, frames, combos, ctx.now)
        except KeyError:
            st.investigated.append(pid)
            st.count("case_unbuildable")
            continue
        errs = case.validate()
        if errs:                                             # counted and reported, never silently dropped
            st.investigated.append(pid)
            st.count("case_invalid")
            continue
        cases.append(case)
        kns[case.case_id] = (pid, kn)
        st.contexts[pid] = investigation_context(st, exp, out, kn, st.gate if isinstance(st.gate, BandGate) else None)
    rep = WC.step(st.wcs, ctx.now, cases, dict(cfg.what_cfg))
    forest = ctx.mod_state("hypothesis_tree", HT.TreeForest)
    trees = 0
    for cid, (pid, _) in kns.items():
        st.investigated.append(pid)
        inv = st.wcs.investigations.get(cid)
        if inv is None:
            continue
        exp = led.expectations.get(pid)
        for p in exp.patterns:
            if p in frames and any(p in str(e) for f in inv.findings if f.level == WC.Level.PATTERN for e in f.evidence):
                st.pcs.mark_investigated(p)
        tid = inv.conclusion.test.tree_id if inv.conclusion and inv.conclusion.test.kind == "HYPOTHESIS_TREE" else ""
        tree = st.wcs.trees.get(tid) if tid else None
        text = tree.nodes[tree.root].text if tree is not None else ""
        if tree is not None and tid not in forest.trees and not forest.find_similar(text):
            try:
                forest.add(tree)
                trees += 1
            except HT.TreeError:
                st.count("trees_refused")
        if not can_record(led.pipe, pid, "CAUSE"):
            st.count("late_investigation")                   # its research was queued first; the investigation still counts
            continue
        fired = [f.level.value for f in inv.findings if f.fired]
        led.pipe.add(pid, "CAUSE", ctx.now, {"significant": inv.significant, "fired": fired, "explained": inv.explained_share,
                                             "cause": inv.conclusion.cause if inv.conclusion else "insignificant error"})
        if not inv.significant:
            continue
        led.pipe.add(pid, "KNOWABILITY", ctx.now, {"class": inv.knowability.klass.value, "basis": inv.knowability.basis,
                                                   "reasons": list(inv.knowability.reasons)})
        led.pipe.add(pid, "PATTERN_REGIME", ctx.now, {"patterns": {p: st.verdicts.get(p, "?") for p in exp.patterns},
                                                      "influence": {p: st.influence.get(p, 1.0) for p in exp.patterns},
                                                      "regime_records": len(st.memory.records(ctx.now)),
                                                      "warning": st.warnings[-1][1] if st.warnings else "NONE"})
        if tree is not None:
            led.pipe.add(pid, "HYPOTHESIS", ctx.now, {"tree": tid, "claim": next((c for c, cl in st.wcs.claims.items() if cid in cl.origin_cases), ""),
                                                      "question": text[:160]})
    st.count("investigations", len(rep.investigated))
    return _run(st, ctx, "c68.what_changed", len(todo), len(rep.investigated),
                f"classes {dict(rep.by_class)}; {len(rep.preserved_unknowable)} unknowable preserved; {trees} tree(s) into the loop forest; "
                f"{len(queue) - len(todo)} waiting")


# ================================================================================================================ stage: error research (D, E, S, T, U)
def _magnitude(it: ER.Intensity, cfg: ER.ErrorConfig) -> float:
    """Question magnitude in [0,1] from the checklist-D intensity: the DEEP cut maps to 0.75 so deeper errors outrank shallower ones
    while nothing saturates below a confident-wrong boost."""
    return float(min(1.0, 0.75 * it.value / cfg.tier_cut[2]))


def st_error_research(ctx: LP.Ctx) -> tuple:
    """c68.error_research. error_research.step over every matured error (tiny ones stay NONE/CHEAP), with the checklist-E contexts;
    each job becomes a QuestionEvent on the loop bus (questions.generate -> hypothesis trees -> priority -> compute: the EXISTING
    scheduler), investigation follow-ups and the eleven checklist-T self-research questions become ResearchQuestions, and the shared
    surprise tracker's repeated cells and the market investigations join them."""
    from engine.research import questions as Q
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    records = led.errors.records(led.outcomes, ctx.now)
    if not records and not st.market.reports:
        raise LP.NoInput("no matured prediction error yet")
    rep = ER.step(st.er, ctx.now, records, st.contexts, priority_state=None, created_real=ctx.created_real())
    events = ctx.bus.setdefault("events", [])
    rqs = ctx.bus.setdefault("research_questions", [])
    by_obs = dict(st.er.intensities)
    book = st.er.book.records(ctx.now)
    ranked = sorted(rep.items, key=lambda i: (-float(i.value.decision_value or 0.0), i.item_id))[: cfg.max_events]
    n_ev = 0
    for item in ranked:
        cell = st.er.item_cells.get(item.item_id, "all")
        subject = f"prediction error {cell}"[:150]
        obs = [o.obs_id for o in book if (o.cell() or "all") == cell or cell in o.groups()]
        its = [by_obs[o] for o in obs if o in by_obs]
        mag = max([_magnitude(i, cfg.research_cfg) for i in its], default=float(min(1.0, (item.value.decision_value or 0.0))))
        cw = any(i.confident_wrong for i in its) or item.family.endswith("confident_wrong")
        ev = Q.QuestionEvent("loss" if cw else "surprise", subject, str(item.created), float(min(1.0, max(0.0, mag))),
                             stake=float(min(1.0, 0.3 + mag)), problem=item.problem, n_obs=len(obs), detail=f"{item.family} {'/'.join(item.tags)}")
        events.append(ev)
        st.cells[subject] = cell
        st.subject_pids[subject] = sorted(set(st.subject_pids.get(subject, [])) | set(obs))
        n_ev += 1
    for inv in rep.investigations:
        rqs.extend(inv.follow_ups)
    rqs.extend(s.question for s in rep.self_questions)
    for pr in st.tracker.research_priority(ctx.now)[:3]:
        if pr.score > 0:
            events.append(Q.QuestionEvent("surprise", f"repeated surprise {pr.cell}"[:150], _last_matured(st, ctx.now),
                                          float(min(1.0, pr.score)), problem=Problem.VOLATILITY, detail=pr.reason[:150]))
    n_mq = 0
    for mr in st.market.reports[st.counters.get("market_reports_sent", 0):]:
        mq = ME.investigation_questions(mr, ctx.created_real(), mr.period)
        rqs.extend(mq)
        n_mq += len(mq)
    st.counters["market_reports_sent"] = len(st.market.reports)
    for o, it in st.er.intensities.items():
        st.depth_of[o] = it.tier.value
    n_out = n_ev + len(rep.self_questions) + n_mq + sum(len(i.follow_ups) for i in rep.investigations)
    if not rep.ingested and not n_out:
        raise LP.NoInput("no new matured error and no market investigation to research")
    return _run(st, ctx, "c68.error_research", rep.ingested, n_out,
                f"{rep.summary()}; {n_ev} question event(s), {len(rep.self_questions)} self-research question(s), {n_mq} market question(s)")


def _last_matured(st: C68State, now) -> str:
    rs = st.er.book.records(now)
    return rs[-1].matured_at if rs else str((pd.Timestamp(as_date(now)) - pd.Timedelta(days=1)).date())


# ================================================================================================================ stage: research depth into priority
def st_depth(ctx: LP.Ctx) -> tuple:
    """c68.research_depth (after questions.generate). The questions the loop generated from C68 events carry the checklist-U depth
    multiplier of their cell (unknowable causes and barren cells are deprioritised) on the loop's OWN priority state - no second
    scheduler. The pipeline records which question each prediction's research became."""
    from engine.research import priority as PRI
    st = _state(ctx)
    led = _ledgers(ctx, st)
    new = ctx.bus.get("new_questions", [])
    ours = [(q, ctx.state.questions[q]) for q in new if q in ctx.state.questions and ctx.state.questions[q].subject in st.cells]
    if not ours:
        raise LP.NoInput("no new question from a C68 event this cycle")
    pst = ctx.mod_state("priority", PRI.new_state)
    damped = 0
    for qid, qo in ours:
        cell = st.cells[qo.subject]
        m = st.er.depth.multiplier(cell)
        if m < 1.0:
            pst.external_multipliers["r_" + qid] = m
            damped += 1
        for pid in st.subject_pids.get(qo.subject, []):
            # research is recorded only once the prediction's own investigation is done (or it was too small to investigate), so
            # the trail reads error -> cause -> knowability -> pattern/regime -> hypothesis -> research, never the other way round
            if pid in st.investigated and led.pipe.last_step(pid) in ("CLASSIFIED", "CAUSE", "KNOWABILITY", "PATTERN_REGIME", "HYPOTHESIS"):
                cw = st.er.investigations.get(pid)
                led.pipe.add(pid, "RESEARCH", ctx.now, {"question": qid, "priority": float(qo.priority), "multiplier": m,
                                                        "tier": st.depth_of.get(pid, "?"),
                                                        "confident_wrong_answered": None if cw is None else cw.answered_share})
    return _run(st, ctx, "c68.research_depth", len(ours), damped, f"{len(ours)} C68 question(s) in the loop's queue, {damped} damped")


# ================================================================================================================ stage: self-correction, promotion, monitoring
def correction_frame(st: C68State, now) -> pd.DataFrame:
    """One row per forecast whose realised gain matured before now (research side): the self_correct frame contract."""
    rows = []
    for day, fc in st.forecasts.items():
        rl = st.realised.get(day, {})
        for t, f in fc.items():
            if t not in rl or pd.Timestamp(rl[t][1]) >= pd.Timestamp(as_date(now)):
                continue
            net = rl[t][0]
            rows.append({"date": day, "matured_at": rl[t][1], "ticker": str(t), "predicted": f["median"], "shadow": f.get("shadow", np.nan),
                         "realised": net, "selected": bool(f["eligible"]),
                         "p_in_band": f["p_band"], "in_band": bool(0.05 <= net <= 0.10), "sector": f["sector"],
                         **{f"f_{k}": f.get(k, np.nan) for k in st.cfg.features}, **{k: f.get(k, np.nan) for k in st.cfg.market_features}})
    if not rows:
        return pd.DataFrame(columns=list(SCX.REQUIRED))
    df = pd.DataFrame(rows).sort_values(["matured_at", "date"], kind="stable").reset_index(drop=True)
    num = [c for c in df.columns if c.startswith(("f_", "m_"))]
    df[num] = df[num].astype(float).fillna(df[num].astype(float).median())
    return df


def _ridge_fix(cols_of: Callable[[pd.DataFrame], list], window_share: float = 1.0, noise: bool = False, lam: float = 1.0):
    """A CandidateFix builder: ridge of realised gain on the chosen columns, trained on `train` only (optionally its newest share)."""
    def build(train: pd.DataFrame, seed: int):
        tr = train
        if window_share < 1.0 and len(tr) > 20:
            tr = tr.iloc[int(len(tr) * (1.0 - window_share)):]
        cols = cols_of(tr)

        def design(fr: pd.DataFrame, s: int) -> np.ndarray:
            X = fr[cols].to_numpy(float) if cols else np.zeros((len(fr), 0))
            if noise:
                X = np.hstack([X, np.random.default_rng(s).normal(size=(len(fr), 1))])
            return X
        X = design(tr, seed)
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
        pen = np.eye(A.shape[1]) * lam
        pen[0, 0] = 0.0
        beta = np.linalg.solve(A.T @ A + pen, A.T @ tr["realised"].to_numpy(float))

        def predict(fr: pd.DataFrame) -> np.ndarray:
            Z = design(fr, seed + 7)
            return np.hstack([np.ones((len(Z), 1)), (Z - mu) / sd]) @ beta
        return predict
    return build


def candidate_fixes(cfg: C68Config, n_groups: int = 1) -> list:
    """The checklist-Q candidate fixes, each tested ALONE: a recency refit (regime recognition), market conditioning, a sector slope
    (one feature's slope differs inside one stock type: the incumbent's pooled slope cannot express it; searched over features x
    sectors, `n_groups` of them) and a placebo (a pure-noise feature) that must never be promoted - the control that proves the gate
    can say no."""
    f = lambda fr: [c for c in fr.columns if c.startswith("f_")]              # noqa: E731
    fm = lambda fr: [c for c in fr.columns if c.startswith(("f_", "m_"))]     # noqa: E731
    I = SCX.FixInput
    return [SCX.CandidateFix("recency_refit", SCX.Component.REGIME_RECOGNITION, tuple(I(c) for c in cfg.features), _ridge_fix(f, 0.4)),
            SCX.CandidateFix("market_conditioning", SCX.Component.MARKET_CONDITIONING,
                             tuple(I(c) for c in (*cfg.features, *cfg.market_features)), _ridge_fix(fm)),
            SCX.slope_fix("sector_slope", [f"f_{c}" for c in cfg.features], "sector", n_groups),
            SCX.CandidateFix("placebo_noise", SCX.Component.MISSING_FEATURE, tuple(I(c) for c in cfg.features), _ridge_fix(f, noise=True))]


def fix_key(name: str, now) -> str:
    return f"FIX:{name}@{as_date(now)}"


FIX_EFFECT = {"recency_refit": {"window_share": 0.4, "extra_features": ()},
              "market_conditioning": {"window_share": 1.0, "extra_features": ("m_vol", "m_r20", "m_breadth")},
              "placebo_noise": None}


def fix_effect(name: str, detail: Mapping[str, Any] | None = None) -> dict | None:
    """What a promoted fix changes in the production learner (the realisable-gain model's next fit). The sector slope adds the
    interaction its own training chose ('<feature>_x_<sector>'); the gain model then estimates that slope itself on every refit.
    None = a control or a fix with nothing to apply (never applied)."""
    if name == "sector_slope":
        d = dict(detail or {})
        if not d.get("feature") or not d.get("group"):
            return None
        return {"window_share": 1.0, "extra_features": (f"{str(d['feature']).removeprefix('f_')}{INTERACTION}{d['group']}",)}
    return FIX_EFFECT.get(name)


def _retired(st: C68State) -> dict:
    return st.__dict__.setdefault("retired", {})


def st_validate(ctx: LP.Ctx) -> tuple:
    """c68.validate_promote. Checklist Q through the EXISTING validation: diagnose, test every candidate fix alone out of sample, gate
    each through engine.research.quality_gate on its FULL measured evidence (self_correct.full_fix_evidence: identity, leak audit,
    replication, calibration, risk, complexity, transfer, failure behaviour - W-06). Only a PROMOTE verdict changes the production
    learner (the realisable-gain model's training window / inputs from the next fit); a placebo can never be promoted by design of the
    gate. A promoted fix is monitored against the incumbent kept in the shadow and rolled back when it does significantly worse on newer
    outcomes (`_monitor`); a rolled-back fix is retired and re-tested only on rows decided after its rollback."""
    from engine.research import quality_gate as QG
    st = _state(ctx)
    led = _ledgers(ctx, st)
    cfg = st.cfg
    fr = SCX.as_of(correction_frame(st, ctx.now), ctx.now)
    if len(fr) < 2 * cfg.self_correct.min_rows:
        raise LP.NoInput(f"only {len(fr)} matured forecasts (< {2 * cfg.self_correct.min_rows}) for self-correction")
    n_groups = int(fr["sector"].nunique()) if "sector" in fr else 1
    retired = _retired(st)
    all_fixes = candidate_fixes(cfg, n_groups)
    n_searched = sum(max(1, x.n_variants) for x in all_fixes)
    through = str(fr["matured_at"].max())
    base = QG.QualityEvidence(provenance=Provenance(ctx.created_real(), through, ctx.rt.code_hash,
                                                    data_hash=stable_hash(fr[["date", "matured_at", "predicted", "realised"]].to_numpy().tolist(), 16),
                                                    config_hash=stable_hash(dataclasses.asdict(cfg.self_correct), 12),
                                                    experiment_id=f"c68.self_correct|{as_date(ctx.now)}", run_id=ctx.state.cfg.run_id,
                                                    seed=cfg.seed, outcomes_seen_through=through))
    pol = QG.QualityPolicy(code_hash=ctx.rt.code_hash)
    fec = getattr(cfg, "fix_evidence", SCX.FixEvidenceConfig(retirement_trigger=True))
    live = [x for x in all_fixes if x.name not in retired]
    try:
        rep = SCX.step(fr, live, ctx.now, base=base, policy=pol, cfg=cfg.self_correct, evidence=fec, n_searched=n_searched)
    except ValueError as e:
        raise LP.NoInput(f"self-correction could not split the matured forecasts: {e}") from None
    reports = [rep]
    for x in [x for x in all_fixes if x.name in retired]:          # re-tested only on evidence newer than its rollback
        fresh = fr[pd.to_datetime(fr["date"]) > pd.Timestamp(retired[x.name])]
        try:
            reports.append(SCX.step(fresh, [x], ctx.now, base=base, policy=pol, cfg=cfg.self_correct, evidence=fec, n_searched=n_searched))
        except ValueError:
            st.count("retired_awaiting_fresh_evidence")
    results = [r for rp in reports for r in rp.results]
    promoted = [p for rp in reports for p in rp.promoted]
    rejected = {k: v for rp in reports for k, v in rp.rejected.items()}
    bundles = {b.fix: b for rp in reports for b in rp.bundles}
    decisions = {d.subject_id: d for rp in reports for d in rp.decisions}
    st.corrections.append({"now": ctx.now, "deteriorated": bool(rep.diagnosis.deterioration.detected),
                           "implicated": [c.value for c in rep.diagnosis.implicated], "promoted": list(promoted), "rejected": dict(rejected),
                           "effects": {r.fix: (r.mean_effect, r.t, len(r.oos_effects)) for r in results},
                           "verdicts": {r.fix: decisions[f"fix:{r.fix}"].verdict.value for r in results if f"fix:{r.fix}" in decisions},
                           "blocking": {r.fix: list(decisions[f"fix:{r.fix}"].blocking) for r in results if f"fix:{r.fix}" in decisions},
                           "full_evidence": sorted(k for k, b in bundles.items() if b.full),
                           "detail": {r.fix: dict(r.detail) for r in results if r.detail}, "production": st.production.get("name")})
    st.corrections = st.corrections[-400:]
    for r in results:
        key = fix_key(r.fix, ctx.now)                      # one trail per test of a fix: OOS test -> update proposal -> gate -> verdict
        if led.pipe.last_step(key) is None:
            b = bundles.get(r.fix)
            led.pipe.add(key, "OOS_TEST", ctx.now, {"mean_effect": r.mean_effect, "t": r.t, "weeks": len(r.oos_effects), "split": r.split,
                                                    "detail": dict(r.detail)},
                         parents=[e.prediction_id for e in led.errors.reports(ctx.now)][-50:])
            led.pipe.add(key, "MODEL_UPDATE", ctx.now, {"proposal": fix_effect(r.fix, r.detail) or "none (control)", "applied": False})
            led.pipe.add(key, "VALIDATION", ctx.now, {"gate": rejected.get(r.fix, "PROMOTE"), "full_evidence": bool(b is not None and b.full),
                                                      "missing": dict(b.missing) if b is not None else {}})
            led.pipe.add(key, "PROMOTED" if r.fix in promoted else "REJECTED", ctx.now, {"gate": rejected.get(r.fix, "PROMOTE")})
    changed = _apply_promotions(st, promoted, {r.fix: r for r in results}, ctx.now)
    rolled = _monitor(st, fr, ctx.now, led)
    for cid, v in st.wcs.verdicts.items():
        key = f"CLAIM:{cid}"
        if led.pipe.last_step(key) is None:
            led.pipe.add(key, "VALIDATION", ctx.now, {"status": v.status.value, "failed_steps": list(v.failed_steps())})
    return _run(st, ctx, "c68.validate_promote", len(fr), len(results),
                f"{len(results)} fix(es) tested alone and gated; {rep.summary().splitlines()[0]}; promoted {list(promoted)} "
                f"({changed} applied); production {st.production['name']}; rolled back {rolled}")


def _apply_promotions(st: C68State, promoted: Sequence[str], results: Mapping[str, Any], now) -> int:
    """A PROMOTE verdict changes the production learner: the promoted fix with the largest out-of-sample gain is applied (one change
    per cycle, so the monitor can attribute what follows). A fix already in production is not re-applied; a newly promoted fix was
    measured against the CURRENT production forecasts, so its inputs are added to the ones already there (never swapped out)."""
    best, eff_best = None, None
    for name in sorted(promoted, key=lambda n: (-float(results[n].mean_effect) if n in results else 0.0, n)):
        eff = fix_effect(name, results[name].detail if name in results else None)
        if eff is None:
            st.count("placebo_promoted" if name == "placebo_noise" else "promoted_without_effect")   # never applied
            continue
        if name in str(st.production.get("name", "")).split("+"):
            continue
        best, eff_best = name, eff
        break
    if best is None:
        return 0
    cur = st.production
    extras = tuple(dict.fromkeys(tuple(cur.get("extra_features", ())) + tuple(eff_best["extra_features"])))
    name = best if cur.get("name") == "incumbent" else f"{cur['name']}+{best}"
    st.production = {"name": name, "since": str(as_date(now)), "key": fix_key(best, now),
                     "window_share": min(float(cur.get("window_share", 1.0)), float(eff_best["window_share"])), "extra_features": extras}
    st.monitoring.append({"fix": best, "since": str(as_date(now)), "production": name, "extra_features": list(extras)})
    st.count("fixes_applied")
    return 1


def monitor_gains(fr: pd.DataFrame, since, weeks: int) -> pd.Series:
    """Per decision week since the promotion, the promoted learner's |error| gain over the shadow incumbent on the same matured names
    (positive = the fix is better), newest `weeks` weeks."""
    if "shadow" not in fr or not len(fr):
        return pd.Series(dtype=float)
    rows = fr[(pd.to_datetime(fr["date"]) >= pd.Timestamp(as_date(since))) & fr["shadow"].notna()]
    if not len(rows):
        return pd.Series(dtype=float)
    g = (rows["realised"] - rows["shadow"]).abs() - (rows["realised"] - rows["predicted"]).abs()
    return SCX.weekly(rows.assign(gain=g.to_numpy(float)), "gain").tail(weeks)


def _monitor(st: C68State, fr: pd.DataFrame, now, led: Ledgers) -> int:
    """Future monitoring of the production learner (the fix's retirement trigger): the promoted learner and the incumbent it replaced
    (kept forecasting in the shadow) are compared on the SAME newer matured names, week by week; when the fix's weekly gain over the
    newest `monitor_weeks` weeks is significantly negative (t <= -rollback_t, >= monitor_min_weeks weeks) the incumbent is restored and
    the fix retired (it is re-tested only on rows decided after the rollback)."""
    cur = st.production
    if cur.get("name") == "incumbent" or not cur.get("since"):
        return 0
    cfg = st.cfg
    g = monitor_gains(fr, cur["since"], getattr(cfg, "monitor_weeks", 26))
    if len(g) < getattr(cfg, "monitor_min_weeks", 8):
        return 0
    t = float(PR.t_stat(g.to_numpy(float)))
    roll = bool(t <= -getattr(cfg, "rollback_t", 2.0))
    key = cur.get("key") or fix_key(cur["name"], cur["since"])
    if led.pipe.can_add(key, "MONITORED"):
        led.pipe.add(key, "MONITORED", now, {"t": t, "weeks": int(len(g)), "mean_gain": float(g.mean()), "rolled_back": roll})
    st.monitoring.append({"now": str(as_date(now)), "production": cur["name"], "t": t, "weeks": int(len(g)), "rolled_back": roll})
    if roll:
        for n in str(cur["name"]).split("+"):
            _retired(st)[n] = str(as_date(now))
        st.production = {"name": "incumbent", "since": str(as_date(now)), "window_share": 1.0, "extra_features": ()}
        st.count("rolled_back")
        return 1
    return 0


# ================================================================================================================ stage: audit and report
def verify_all(st: C68State, led: Ledgers, deep: bool = True) -> list[str]:
    """Re-verify the C68 chain from its medium against the anchors stored OUTSIDE it, and the P03 memories. Every cycle: the whole
    chain's hash links, every anchor still on it, and the expectation lane on disk equal to what this process recorded (any edit of
    any lane breaks the links; a consistent re-hash of the chain drops the anchors). `deep` additionally re-derives every stored
    body's content hash in every ledger and every pipeline trail."""
    probs = []
    every = [h for _, _, h in st.anchors]                      # all C68 ledgers are lanes of ONE chain: every anchor must still be on it
    if not deep:
        checks = {"chain": led.expectations.lane.verify(every)}
    else:
        checks = {"expectations+outcomes": led.outcomes.verify(every),      # OutcomeLedger.verify re-verifies the expectation lane too
                  "errors": led.errors.verify(), "pipe": led.pipe.verify()}
    for k, v in checks.items():
        probs += [f"{k}: {p}" for p in v.get("problems", [])]
    if deep:
        probs += [f"book: {p}" for p in led.book.verify()]
    if st.market.ledger.verify():
        probs.append(f"market expectations: entries {st.market.ledger.verify()} rewritten")
    if st.memory.verify():
        probs.append(f"regime memory: events {st.memory.verify()} rewritten")
    return probs


INDEPENDENCE_TARGETS = (CT.Target(0.05, 0.5, name="pm5pp_50"), CT.Target(0.002, 0.95, name="pm02pp_95"))


def exit_independence(st: C68State, decide: Callable = None) -> XR.IndependenceReport | None:
    """Checklist O on this loop's own exits: the newest exited positions re-decided by THE exit function (`decide_exits`, or a
    stand-in a test plants) under the configured +-1pp target and two very different ones. Any difference in an exit day, reason or
    return means the exit read the target."""
    if not st.exited:
        return None
    decide = decide or decide_exits
    targets = [st.cfg.target, *INDEPENDENCE_TARGETS]
    return XR.exit_independence_audit(lambda t: _concat(decide(dataclasses.replace(st, cfg=dataclasses.replace(st.cfg, target=t)), st.exited)),
                                      targets)


def st_audit(ctx: LP.Ctx) -> tuple:
    """c68.monitor_audit. Verify every ledger against the anchors held outside it (a rewrite is a FirewallBreach: REFUSED_LEAK), store
    new anchors, measure the +-1pp target honestly (calibration_target.evaluate: the canonical statistic; honest_tolerance is its
    adapter), audit exit independence on this cycle's exits, and persist the cycle report under <root>/c68."""
    st = _state(ctx)
    led = _ledgers(ctx, st)
    probs = verify_all(st, led, deep=ctx.cycle % 4 == 0)
    if probs:
        st.count("tamper_detected")
        raise XP.LedgerTampered("; ".join(probs[:6]))
    ind = exit_independence(st)
    if ind is not None:
        st.independence.append({"now": ctx.now, "identical": ind.identical, "n_targets": ind.n_targets, "differing": list(ind.differing)})
        if not ind.identical:
            st.count("exit_read_target")
            raise FirewallBreach(f"exit decisions changed with the evaluation target ({', '.join(ind.differing)}): checklist O violated")
    for k, h in led.heads().items():
        if h != GENESIS and not any(a[1] == k and a[2] == h for a in st.anchors[-20:]):
            st.anchors.append((ctx.cycle, k, h))
    st.anchors = st.anchors[-500:]
    comm = led.book.commitments(ctx.now)
    key = (len(st.exit_records), len(comm), sum(as_date(c.matures_by) < as_date(ctx.now) for c in comm))
    cache = ctx.rt.__dict__.setdefault("_c68_cal", {})
    if cache.get("key") != key:                  # the honest statistic changes only when the book or its matured part changes
        cache.update(key=key, cal=CT.evaluate(led.book, st.exit_records, ctx.now, st.cfg.target, st.cfg.seed))
    cal = cache["cal"]
    reps = led.errors.reports(ctx.now)
    adapter = PE.honest_tolerance(reps, 0, st.cfg.error_cfg)
    st.calibration.append({"now": ctx.now, "status": cal.status.value, "share": cal.all_.share, "n": cal.all_.n, "oos_n": cal.oos.n,
                           "abuses": [a.kind.value for a in cal.abuses], "adapter_share": adapter["rate"], "headline": cal.headline})
    rep = cycle_report(ctx, st, led, cal, with_identification=ctx.cycle % 4 == 0)
    path = led.root / f"cycle_{ctx.cycle:05d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, default=str, indent=1), encoding="utf-8")
    tmp.replace(path)
    n_checked = len(led.expectations) + len(led.outcomes) + len(led.errors) + len(led.pipe) + len(st.market.ledger) + len(st.memory.events())
    if not n_checked:
        raise LP.NoInput("nothing recorded yet: the (empty) ledgers verified, no report to write")
    return _run(st, ctx, "c68.monitor_audit", n_checked, 1,
                f"{n_checked} record(s) verified, report written; {cal.headline[:150]}")


def identification(st: C68State, now) -> pd.DataFrame:
    """selection_constraint.identification_curve over every matured research-side forecast: per quarter, the share of the names the
    band gate made eligible whose realised gain landed in [5%, 10%], against the base rate of all forecast names."""
    frame = correction_frame(st, now)
    if not len(frame):
        return pd.DataFrame()
    pol = "c68"
    fcs = [SC.GainForecast(f"{r.date}|{i}", str(r.date), pol, float(r.predicted), {0.5: float(r.predicted)}, float(r.p_in_band),
                           st.cfg.selection.min_support, "research") for i, r in frame.iterrows()]
    return SC.identification_curve(fcs, {f.candidate: float(frame.loc[i, "realised"]) for i, f in zip(frame.index, fcs)},
                                   {f.candidate: str(frame.loc[i, "matured_at"]) for i, f in zip(frame.index, fcs)}, now, pol, st.cfg.selection)


def cycle_report(ctx: LP.Ctx, st: C68State, led: Ledgers, cal: CT.CalibrationReport, with_identification: bool = True) -> dict:
    ident = None
    if with_identification:
        cur = identification(st, ctx.now)
        ident = {"periods": int(len(cur)), "trend": SC.improvement_trend(cur) if len(cur) else None,
                 "rows": cur.to_dict("records") if len(cur) else []}
    return {"cycle": ctx.cycle, "now": ctx.now, "label": LABEL, "counters": dict(st.counters),
            "ledgers": {"expectations": len(led.expectations), "outcomes": len(led.outcomes), "errors": len(led.errors),
                        "commitments": len(led.book.commitments()), "pipeline_events": len(led.pipe), "pipeline_furthest": led.pipe.furthest()},
            "calibration": cal.to_dict() | {"target": dataclasses.asdict(cal.target)}, "influence": dict(st.influence),
            "verdicts": dict(st.verdicts), "guard": dict(st.guard), "production": dict(st.production),
            "regime_records": [{"id": r.record_id, "scope": r.scope, "detected": r.detected_at, "status": st.memory.status(r.record_id).value,
                                "weakened": list(r.patterns_weakened)} for r in st.memory.records(ctx.now)],
            "identification": ident, "stage_runs": [r for r in st.runs if r["cycle"] == ctx.cycle]}


# ================================================================================================================ the planted C68 world
@dataclasses.dataclass(frozen=True)
class C68Plant:
    """A seeded world whose C68 mechanisms are KNOWN (World.truth names them), so the pipeline can be scored against exact truth:
      trend episodes   a name in a trend drifts +`drift` per session with higher volatility; its trailing 20-session return reveals it
                       (the genuine, knowable pattern 'mom_r20_top' and a realisable gain inside the 5-10% band)
      false alarm      a market-wide volatility burst of `burst_len` sessions at `false_alarm` of the sample while the trend pattern keeps
                       working: a regime alarm that must NOT switch the pattern off
      regime switch    from `switch` to `recover` the trend fades to `post_switch_drift` (its signature stays visible, its payoff shrinks:
                       'expected 8-10%, realised 3-6%') and market volatility is higher: a genuine market-wide change that must degrade the
                       pattern and produce repeated, similar over-predictions; from `recover` on the drift is back (volatility stays high):
                       the pattern must regain influence
                       detected forward in time and must degrade the pattern
      shock            one name gaps down `shock_size` at `shock` with no information item anywhere and stays turbulent for
                       `distress_len` sessions: an unknowable single-stock anomaly that must stay a SINGLE-STOCK finding
      weak sector      (F11, off by default: weak_sector = -1) in ONE stock type (sector `weak_sector`) momentum is crowded: every
                       session a name gives back `weak_revert` times its trailing 20-session return (until `weak_until` of the
                       sample, `weak_revert_after` from then on). A model with one pooled r20 slope over-predicts exactly the strong
                       names of that sector, in every year: a GENUINE, knowable, persistent, fixable prediction error (the r20 slope
                       inside that sector differs). The null twin is the same seed with weak_sector = -1 (identical draws, no error);
                       a negative weak_revert_after turns the give-back into acceleration later (a promoted fix that then degrades)
      regime_switch    False removes the switch / recovery (a clean multi-year world for the self-correction proofs)"""
    n_names: int = 40
    n_days: int = 440
    seed: int = 0
    n_sectors: int = 5
    drift: float = 0.016
    base_sigma: tuple = (0.008, 0.014)
    start_p: float = 1 / 60
    stop_p: float = 1 / 30
    switch: float = 0.72
    recover: float = 0.84
    false_alarm: float = 0.58
    burst_len: int = 12
    burst_mult: float = 2.5
    shock: float = 0.65
    shock_size: float = -0.18
    shock_name: int = 7
    distress_len: int = 25
    distress_mult: float = 4.0
    post_switch_drift: float = 0.002
    market_vol: float = 0.006
    post_switch_vol: float = 2.2
    regime_switch: bool = True
    weak_sector: int = -1
    weak_revert: float = 0.02
    weak_until: float = 1.0
    weak_revert_after: float = 0.02


def momentum_giveback(r: np.ndarray, names: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Each session, the chosen names give back k[t] times their trailing 20-session return up to the previous close (a negative
    k[t] = momentum that accelerates instead). Sequential, because the trailing return includes earlier give-backs; only the past
    enters each session's adjustment."""
    r = r.copy()
    lc = np.zeros((r.shape[0] + 1, r.shape[1]))              # log close relative to the start
    for t in range(r.shape[0]):
        if t >= 20:
            r20 = np.exp(lc[t] - lc[t - 20]) - 1.0
            r[t] = np.where(names, np.clip(r[t] - k[t] * r20, -0.3, 0.3), r[t])
        lc[t + 1] = lc[t] + np.log1p(r[t])
    return r


def plant_world(pc: C68Plant = C68Plant()) -> FD.World:
    rng = np.random.default_rng(pc.seed)
    T, n = pc.n_days, pc.n_names
    if n < 10 or T < 300:
        raise ValueError("the C68 world needs >= 10 names and >= 300 sessions")
    if pc.weak_sector >= pc.n_sectors:
        raise ValueError("weak_sector must be -1 or a sector index")
    dates = pd.bdate_range("2016-01-04", periods=T)
    tick = [f"W{j:03d}" for j in range(n)]
    base = rng.uniform(*pc.base_sigma, n)
    state = np.zeros((T, n), bool)
    s = rng.random(n) < 0.15
    for t in range(T):
        s = (s | ((~s) & (rng.random(n) < pc.start_p))) & ~(s & (rng.random(n) < pc.stop_p))
        state[t] = s
    sw, f0, ts = int(T * pc.switch) if pc.regime_switch else T, int(T * pc.false_alarm), int(T * pc.shock)
    mvol = np.full(T, pc.market_vol)
    mvol[sw:] *= pc.post_switch_vol
    mvol[f0:f0 + pc.burst_len] *= pc.burst_mult
    m = rng.normal(0, 1, T) * mvol
    sig = base[None, :] * np.where(state, 1.4, 1.0) * np.sqrt(mvol / pc.market_vol)[:, None]
    sig[ts:ts + pc.distress_len, pc.shock_name] *= pc.distress_mult
    mu = np.where(state, pc.drift, 0.0)
    rc = int(T * pc.recover) if pc.regime_switch else T
    mu[sw:rc] = np.where(state[sw:rc], pc.post_switch_drift, 0.0)
    wk = int(T * pc.weak_until)
    r = np.clip(mu + m[:, None] + rng.normal(0, 1, (T, n)) * sig, -0.3, 0.3)
    if pc.weak_sector >= 0:                    # arithmetic on the drawn returns: the null twin draws exactly the same random numbers
        r = momentum_giveback(r, (np.arange(n) % pc.n_sectors) == pc.weak_sector,
                              np.where(np.arange(T) < wk, pc.weak_revert, pc.weak_revert_after))
    r[ts, pc.shock_name] += pc.shock_size
    p0 = np.exp(rng.uniform(math.log(10), math.log(120), n))
    C = p0 * np.exp(np.cumsum(np.log1p(r), axis=0))
    prev = np.vstack([C[:1] / (1 + r[:1]), C[:-1]])
    gf = rng.uniform(0.0, 0.4, (T, n))
    gf[ts, pc.shock_name] = 1.0
    O = prev * (1.0 + r * gf)
    wick = np.abs(rng.normal(0, 1, (2, T, n))) * sig * 0.5
    H = np.maximum(O, C) * (1 + wick[0])
    L = np.minimum(O, C) * (1 - np.minimum(wick[1], 0.5))
    V = np.exp(rng.normal(15.0, 0.4, n)) * np.exp(rng.normal(0, 0.25, (T, n)))
    bars = {k: pd.DataFrame(v, index=dates, columns=tick) for k, v in zip(FD.BAR_FIELDS, (O, H, L, C, V))}
    sectors = {t: f"SEC{j % pc.n_sectors}" for j, t in enumerate(tick)}
    truth = {"pattern": "mom_r20_top", "switch": str(dates[sw].date()) if sw < T else None, "recover": str(dates[min(rc, T - 1)].date()) if sw < T else None,
             "false_alarm": (str(dates[f0].date()), str(dates[f0 + pc.burst_len - 1].date())),
             "shock": (tick[pc.shock_name], str(dates[ts].date())), "trend_share": float(state.mean()), "survivor_free": True, "seed": pc.seed,
             "weak_sector": f"SEC{pc.weak_sector}" if pc.weak_sector >= 0 else None,
             "weak_until": str(dates[wk].date()) if pc.weak_sector >= 0 and wk < T else None}
    return FD.World(bars, None, None, None, sectors, {}, FD.market_proxy(bars), truth)


# ================================================================================================================ registration
STAGES = (
    ("c68.selection_policy", st_policy, "update.firewall_release", LP.LoopPhase.UPDATE_KNOWLEDGE),
    ("c68.expectations", st_expect, "evaluate.two_stage", LP.LoopPhase.EVALUATE),
    ("c68.market_regime", st_market, "surprises.regimes", LP.LoopPhase.SURPRISES),
    ("c68.pattern_change", st_patterns, "c68.market_regime", LP.LoopPhase.SURPRISES),
    ("c68.outcomes_errors", st_outcomes, "failures.decision_losses", LP.LoopPhase.FAILURES),
    ("c68.what_changed", st_what_changed, "c68.outcomes_errors", LP.LoopPhase.FAILURES),
    ("c68.error_research", st_error_research, "c68.what_changed", LP.LoopPhase.FAILURES),
    ("c68.research_depth", st_depth, "questions.generate", LP.LoopPhase.QUESTIONS),
    ("c68.validate_promote", st_validate, "learn.quality_gate", LP.LoopPhase.LEARN),
    ("c68.monitor_audit", st_audit, "priorities.stale", LP.LoopPhase.PRIORITIES),
)
STAGE_NAMES = tuple(s[0] for s in STAGES)


def register() -> tuple:
    """Register the feed builder and every C68 stage into the loop (idempotent: a re-import registers nothing twice)."""
    FD.register_builder(WORLD_KEY, b_world, replace=True)
    out = []
    for name, fn, after, phase in STAGES:
        out.append(LP.register_stage(name, fn, after=after, phase=phase, module="error_loop"))
    return tuple(s.name for s in out)


def unregister() -> None:
    """Remove the C68 stages and builder (tests leave the loop's stage table as they found it)."""
    for name in reversed(STAGE_NAMES):
        if name in LP.registered_stages():
            LP.unregister_stage(name)
    FD.BUILDERS.pop(WORLD_KEY, None)


def stage_table(reports: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The per-stage run table of the C68 stages over loop cycle reports: cycle, stage, status, in, out, note."""
    rows = [{"cycle": r.get("cycle"), "stage": s["stage"], "status": s["status"], "in": s.get("n_in"), "out": s.get("n_out"),
             "note": str(s.get("reason", ""))[:160]} for r in reports for s in r.get("stages", []) if str(s.get("stage", "")).startswith("c68.")]
    return pd.DataFrame(rows, columns=["cycle", "stage", "status", "in", "out", "note"])


register()
