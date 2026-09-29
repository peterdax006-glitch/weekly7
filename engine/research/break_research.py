"""Pattern-break research (contract C66 section 14; canon C60/C61/C62 section 10; Bible phase 9/10). IMPLEMENTED - NOT VALIDATED.

A pattern that stops working is not merely down-weighted: it becomes a scheduled INVESTIGATION. Every detected break (and every
BROKEN health row) opens one, with a ResearchQuestion, a hypothesis set drawn from the section-14 cause list (regime change,
market structure change, liquidity change, sector shift, event environment, pattern crowding, feature relationship changed,
measurement error, sampling artifact, random variation, hidden interaction, external shock, unknown cause) and the six
section-14 questions as statistical tests on the item's own history:

    Q1 a PRE-EXISTING variable predicts the break (it leads the onset; family-wise bar over every causal column searched)
    Q2 the predictor TRANSFERS out of sample (rule frozen on the discovery window, scored on strictly later data)
    Q3 it predicts FUTURE breaks (walk-forward, only breaks knowable at each origin train it, and it must beat time-since-break)
    Q4 GATING the pattern IMPROVES performance (expectancy per exposure, shift-null + block bootstrap)
    Q5 gating REDUCES LOSSES (net loss avoided, CVaR, drawdown, worst period)
    Q6 the explanation BEATS RANDOM explanations (the same search re-run on shifted columns and on persistence-matched random gates)

Only when all six pass is the break EXPLAINED, and then only as a gate PROPOSAL for the research queue and a later fresh-holdout
replication. Anything short of that is UNKNOWN, with the reason (a failed test, or a test that lacked the power to fail) recorded:
UNKNOWN is the correct answer, not a gap. Nothing here reaches the blind trader: conclusions are MaturedRecords in
MATURED_RESEARCH_STATE and can leave only through MaturedRecord.gate(now) (and never while the year they were filed under is
being replayed in disguise - the same-year rerun leak).

EXTENDS, never copies: engine.learning.break_detection (ItemSeries, causal Design, episodes, family-wise contrasts, the
condition search, event studies, interaction scan, shared breaks, placebo), engine.pattern_reliability (holm, auc), and
engine.learning.research_priority (Signals and signals_from_health, so the queue and its health trigger have a caller).

Public entry: `step(state, now, items, ...)`; state from `new_state()`. Evidence-only helpers are pure and take `as_of`/`now`."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_reliability as PR
from engine.learning import break_detection as BD
from engine.learning import research_priority as RP
from engine.learning.core import (FailureCause, FirewallBreach, Provenance, Unknown, ValidationLabel, _StrEnum, as_date,
                                  current_code_hash, require_past, stable_hash)
from engine.research.core import (ExperimentValue, MaturedRecord, Namespace, Problem, ResearchQuestion, ResearchState, Stage,
                                  Knowability, GateVerdict)

LABEL = ValidationLabel.NOT_VALIDATED.value

PARAMS: dict[str, Any] = {
    "horizon": 8,                 # a pre-onset row is one of the `horizon` rows before an estimated onset
    "min_lead": 1,                # the predictor must be known at least this many rows before the onset
    "disc_frac": 0.55, "embargo": 4,
    "alpha": 0.10,                # family-wise bar for discovery (Q1)
    "oos_alpha": 0.05,            # one-sided bar for every out-of-sample question
    "auc_bar": 0.58,              # a predictor must also be worth something: AUC at least this
    "min_episodes": 2, "min_pre_rows": 12, "min_ref_rows": 24, "min_confirm": 40, "min_oos_pos": 8,
    "min_future_breaks": 2, "max_detectable_smd": 1.0,
    "n_perm": 300, "n_random": 200, "boot": 300, "boot_block": 6,
    "wf_step": 13, "wf_min_train": 60,
    "cover_lo": 0.05, "cover_hi": 0.60,
    "cut_quantiles": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "baseline_margin": 0.02,      # Q3: must beat the best naive baseline AUC by this much
    "max_deff": 8.0,
    "drawdown_slack": 0.0,
    "lead_window": 8,
    "excluded_sources": ("age",),  # a clock is not an explanation: a monotone column separates any two eras
    "bd": {},                     # overrides for engine.learning.break_detection.PARAMS
    # scheduler
    "max_per_step": 3, "minutes_budget": 30.0, "cool_off_rows": 8, "reopen_rows": 26, "max_attempts": 4,
    "cpu_min_per_cell": 4e-6, "min_priority": 0.0,
    "holdout_min_rows": 26,
}

CAUSE_LABEL = {
    "REGIME_CHANGE": "the market regime changed", "MARKET_STRUCTURE_CHANGE": "the market structure changed",
    "LIQUIDITY_CHANGE": "liquidity changed", "SECTOR_SHIFT": "the sector mix shifted",
    "EVENT_ENVIRONMENT": "an event environment recurs at the same time of year", "PATTERN_CROWDING": "the pattern became crowded",
    "FEATURE_RELATIONSHIP_CHANGED": "the relation between a feature and the outcome changed",
    "MEASUREMENT_ERROR": "the measurement is wrong", "SAMPLING_ARTIFACT": "the sample was too thin or concentrated",
    "RANDOM_VARIATION": "chance alone", "HIDDEN_INTERACTION": "two conditions must hold together",
    "EXTERNAL_SHOCK": "an outside shock", "UNKNOWN_CAUSE": "unknown"}


class BreakCause(_StrEnum):        # the section-14 list, verbatim, plus UNKNOWN_CAUSE as a first-class answer
    REGIME_CHANGE = "REGIME_CHANGE"
    MARKET_STRUCTURE_CHANGE = "MARKET_STRUCTURE_CHANGE"
    LIQUIDITY_CHANGE = "LIQUIDITY_CHANGE"
    SECTOR_SHIFT = "SECTOR_SHIFT"
    EVENT_ENVIRONMENT = "EVENT_ENVIRONMENT"
    PATTERN_CROWDING = "PATTERN_CROWDING"
    FEATURE_RELATIONSHIP_CHANGED = "FEATURE_RELATIONSHIP_CHANGED"
    MEASUREMENT_ERROR = "MEASUREMENT_ERROR"
    SAMPLING_ARTIFACT = "SAMPLING_ARTIFACT"
    RANDOM_VARIATION = "RANDOM_VARIATION"
    HIDDEN_INTERACTION = "HIDDEN_INTERACTION"
    EXTERNAL_SHOCK = "EXTERNAL_SHOCK"
    UNKNOWN_CAUSE = "UNKNOWN_CAUSE"


# how a section-14 cause is spoken in the C62 failure vocabulary (used when handing the result to engine.learning consumers)
FAILURE_CAUSE_OF = {
    BreakCause.REGIME_CHANGE: FailureCause.REGIME_CHANGE, BreakCause.MARKET_STRUCTURE_CHANGE: FailureCause.REGIME_CHANGE,
    BreakCause.LIQUIDITY_CHANGE: FailureCause.REGIME_CHANGE, BreakCause.SECTOR_SHIFT: FailureCause.WRONG_CONTEXT,
    BreakCause.EVENT_ENVIRONMENT: FailureCause.TEMPORARY_INACTIVITY, BreakCause.PATTERN_CROWDING: FailureCause.WEAKENING_EFFECT,
    BreakCause.FEATURE_RELATIONSHIP_CHANGED: FailureCause.REVERSAL, BreakCause.MEASUREMENT_ERROR: FailureCause.MEASUREMENT_ERROR,
    BreakCause.SAMPLING_ARTIFACT: FailureCause.INSUFFICIENT_EVIDENCE, BreakCause.RANDOM_VARIATION: FailureCause.FALSE_PATTERN,
    BreakCause.HIDDEN_INTERACTION: FailureCause.INTERACTION_FAILURE, BreakCause.EXTERNAL_SHOCK: FailureCause.RISK_ERROR,
    BreakCause.UNKNOWN_CAUSE: FailureCause.UNKNOWN}

# base rates of each explanation before looking at the data; UNKNOWN keeps a floor so it is never priced out
BASE_PRIOR = {BreakCause.REGIME_CHANGE: 0.20, BreakCause.MARKET_STRUCTURE_CHANGE: 0.06, BreakCause.LIQUIDITY_CHANGE: 0.05,
              BreakCause.SECTOR_SHIFT: 0.05, BreakCause.EVENT_ENVIRONMENT: 0.04, BreakCause.PATTERN_CROWDING: 0.08,
              BreakCause.FEATURE_RELATIONSHIP_CHANGED: 0.07, BreakCause.MEASUREMENT_ERROR: 0.05,
              BreakCause.SAMPLING_ARTIFACT: 0.06, BreakCause.RANDOM_VARIATION: 0.14, BreakCause.HIDDEN_INTERACTION: 0.05,
              BreakCause.EXTERNAL_SHOCK: 0.05, BreakCause.UNKNOWN_CAUSE: 0.10}
UNKNOWN_FLOOR = 0.10

DIMENSIONS_OF = {
    BreakCause.REGIME_CHANGE: ("regime", "trend", "breadth", "macro"),
    BreakCause.MARKET_STRUCTURE_CHANGE: ("volatility", "liquidity", "feature_distribution"),
    BreakCause.LIQUIDITY_CHANGE: ("liquidity",),
    BreakCause.SECTOR_SHIFT: ("sector_composition", "stock_type"),
    BreakCause.MEASUREMENT_ERROR: ("missingness", "data_quality"),
    BreakCause.SAMPLING_ARTIFACT: ("sample_size", "concentration"),
}


class QuestionId(_StrEnum):        # the six section-14 questions
    PREEXISTING_PREDICTOR = "Q1_PREEXISTING_PREDICTOR"
    TRANSFERS_OOS = "Q2_TRANSFERS_OOS"
    PREDICTS_FUTURE_BREAKS = "Q3_PREDICTS_FUTURE_BREAKS"
    GATING_IMPROVES = "Q4_GATING_IMPROVES_PERFORMANCE"
    GATING_REDUCES_LOSSES = "Q5_GATING_REDUCES_LOSSES"
    BEATS_RANDOM = "Q6_BEATS_RANDOM_EXPLANATIONS"


QUESTION_TEXT = {
    QuestionId.PREEXISTING_PREDICTOR: "Can a variable known BEFORE the break predict it?",
    QuestionId.TRANSFERS_OOS: "Does that variable still predict on data it was not fitted on?",
    QuestionId.PREDICTS_FUTURE_BREAKS: "Does it predict breaks that had not happened when the rule was fitted?",
    QuestionId.GATING_IMPROVES: "Does gating the pattern with it improve performance per exposure?",
    QuestionId.GATING_REDUCES_LOSSES: "Does the same gate reduce the losses, not just move them?",
    QuestionId.BEATS_RANDOM: "Does the explanation beat random explanations found by the same search?"}
QUESTION_ORDER = tuple(QuestionId)
STAGE_OF = {QuestionId.PREEXISTING_PREDICTOR: Stage.CHEAP_SCREEN, QuestionId.TRANSFERS_OOS: Stage.STRONGER_TESTS,
            QuestionId.PREDICTS_FUTURE_BREAKS: Stage.STRONGER_TESTS, QuestionId.GATING_IMPROVES: Stage.CROSS_YEAR,
            QuestionId.GATING_REDUCES_LOSSES: Stage.CROSS_YEAR, QuestionId.BEATS_RANDOM: Stage.FRESH_HOLDOUT}


class Outcome(_StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"                  # adequately powered and negative
    NOT_TESTABLE = "NOT_TESTABLE"      # the data could not answer (too few breaks/rows, or too little power to see a failure)


class Verdict(_StrEnum):
    EXPLAINED = "EXPLAINED"                        # all six passed: a gate PROPOSAL, still unproven on fresh data
    UNKNOWN = "UNKNOWN"                            # at least one question failed with adequate power
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # the data could not answer at least one question
    NO_BREAK = "NO_BREAK"
    DATA_FAILURE = "DATA_FAILURE"                  # the input itself failed a firewall


class BreakResearchError(ValueError):
    pass


def _cfg(cfg) -> dict:
    P = {**PARAMS, **(cfg or {})}
    return P


def _bd_cfg(P) -> dict:
    return {**BD.PARAMS, **P.get("bd", {})}


# ------------------------------------------------------------------------------------------------- records

@dataclasses.dataclass(frozen=True)
class BreakEvent:
    """One break of one item as knowable at `as_of`: rows are positions in the item's matured history."""
    item_key: str                       # anonymised id: hash of the item id, safe for anything handed on
    onset: int
    detect: int
    recover: int | None
    onset_at: str
    detect_at: str
    severity: float                     # drop in mean outcome from the 26 rows before the onset to the broken stretch, in sd units
    length: int
    n_matured: int
    source: str = "episode"             # episode | health

    @property
    def open(self) -> bool:
        return self.recover is None

    @property
    def event_id(self) -> str:
        return stable_hash([self.item_key, self.onset_at, self.source], 12)

    def validate(self) -> list[str]:
        errs = []
        if not (0 <= self.onset <= self.detect):
            errs.append(f"onset {self.onset} must lie in [0, detect={self.detect}]")
        if self.recover is not None and self.recover < self.detect:
            errs.append("recover before detect")
        if self.detect >= max(self.n_matured, 1):
            errs.append("break detected on a row that is not matured")
        if self.severity != self.severity:
            errs.append("severity is NaN")
        return errs


@dataclasses.dataclass(frozen=True)
class Trigger:
    source: str                         # episode | health | shared | manual
    weight: float
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class HypothesisEvidence:
    """What the data say about one section-14 cause. `supported` is a diagnostic (it steers follow-up work); a cause is only
    ever CLAIMED when all six questions pass."""
    cause: BreakCause
    supported: bool | None              # None = the data could not test it
    strength: float                     # 0..1
    p: float
    n: int
    detail: str
    columns: tuple = ()

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 <= self.strength <= 1.0):
            errs.append(f"{self.cause}: strength {self.strength} outside [0,1]")
        if not (0.0 <= self.p <= 1.0):
            errs.append(f"{self.cause}: p {self.p} outside [0,1]")
        return errs


@dataclasses.dataclass(frozen=True)
class BreakHypothesis:
    """A cause hypothesis with its prior, the posterior after looking, and what would settle it."""
    cause: BreakCause
    statement: str
    prior: float
    posterior: float
    evidence: HypothesisEvidence | None
    discriminator: str                  # the observation that separates it from its neighbours

    def validate(self) -> list[str]:
        errs = []
        for nm, v in (("prior", self.prior), ("posterior", self.posterior)):
            if not (0.0 <= v <= 1.0) or v != v:
                errs.append(f"{self.cause}: {nm} {v} outside [0,1]")
        return errs


@dataclasses.dataclass(frozen=True)
class QuestionResult:
    qid: QuestionId
    outcome: Outcome
    statistic: float                    # the headline number (AUC, expectancy gain, net loss saved, ...)
    p: float
    n: int
    detail: Mapping[str, Any]
    why: str

    @property
    def passed(self) -> bool:
        return self.outcome == Outcome.PASSED

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 <= self.p <= 1.0) and self.p == self.p:
            errs.append(f"{self.qid}: p {self.p} outside [0,1]")
        if not self.why:
            errs.append(f"{self.qid}: every result needs a stated reason (an unexplained UNKNOWN is a gap)")
        return errs

    def brief(self) -> dict:
        return {"question": self.qid.value, "outcome": self.outcome.value, "statistic": self.statistic, "p": self.p,
                "n": self.n, "why": self.why}


class GateRule:
    """A frozen decision rule: mask(X) is True where the pattern should be WITHHELD (predicted broken/about to break)."""
    kind = "abstract"

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def columns(self) -> tuple:
        return ()

    @property
    def key(self) -> str:
        return stable_hash([self.kind, self.describe()], 10)


@dataclasses.dataclass(frozen=True)
class ColumnRule(GateRule):
    """Withhold when direction * column >= threshold. Fitted only on discovery rows; never refitted on what it is scored on."""
    column: str
    direction: int
    threshold: float
    fit_auc: float = float("nan")
    fit_j: float = float("nan")
    kind = "column"

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            raise BreakResearchError(f"rule column {self.column!r} is not in the frame")
        x = X[self.column].values.astype(float)
        with np.errstate(invalid="ignore"):
            return self.direction * x >= self.threshold

    def describe(self) -> str:
        op = ">=" if self.direction > 0 else "<="
        thr = self.threshold * self.direction
        return f"withhold when {self.column} {op} {thr:.4g}"

    def columns(self) -> tuple:
        return (self.column,)

    def validate(self) -> list[str]:
        errs = []
        if self.direction not in (-1, 1):
            errs.append("direction must be +1 or -1")
        if not math.isfinite(self.threshold):
            errs.append("threshold must be finite")
        return errs


@dataclasses.dataclass(frozen=True)
class Investigation:
    inv_id: str
    item_id: str                        # trusted-side raw id; never placed in question text or records
    item_key: str
    event: BreakEvent
    state: ResearchState
    stage: Stage
    trigger: Trigger
    priority: float
    value: ExperimentValue
    hypotheses: tuple = ()
    results: tuple = ()
    verdict: Verdict | None = None
    cause: BreakCause = BreakCause.UNKNOWN_CAUSE
    rule: ColumnRule | None = None
    created_at: str = ""
    updated_at: str = ""
    evidence_rows: int = 0              # matured rows in the item when last run (a rerun needs new evidence)
    frozen_row: int = -1                # rows up to here fitted the rule; anything later is fresh holdout
    attempts: int = 0
    cost_minutes: float = 0.0
    note: str = ""
    holdout: tuple = ()
    extras: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def result(self, qid: QuestionId) -> QuestionResult | None:
        for r in self.results:
            if r.qid == qid:
                return r
        return None

    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)

    def validate(self) -> list[str]:
        errs = list(self.event.validate())
        for r in self.results:
            errs.extend(r.validate())
        for h in self.hypotheses:
            errs.extend(h.validate())
        if self.verdict == Verdict.EXPLAINED:
            if {r.qid for r in self.results if r.passed} != set(QUESTION_ORDER):
                errs.append("EXPLAINED without all six questions passed")
            if self.rule is None:
                errs.append("EXPLAINED without a frozen rule")
        if self.cause != BreakCause.UNKNOWN_CAUSE and self.verdict != Verdict.EXPLAINED:
            errs.append("a cause may only be claimed by an EXPLAINED investigation")
        if self.state == ResearchState.QUEUED and self.results:
            errs.append("a QUEUED investigation cannot already hold results")
        return errs

    @property
    def label(self) -> str:
        return f"break#{self.item_key[:6]}"


# ------------------------------------------------------------------------------------------------- numerical helpers

def deff_of(labels: np.ndarray, cap: float = 8.0) -> float:
    """Design effect of a 0/1 label series from its lag-1 autocorrelation: labelled rows come in stretches, so n rows are worth
    fewer independent observations (1 for independent labels, capped so one long stretch cannot zero the test)."""
    x = np.asarray(labels, float)
    if len(x) < 8 or x.std() < 1e-12:
        return 1.0
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not math.isfinite(r):
        return 1.0
    r = min(max(r, 0.0), 0.9)
    return float(min((1 + r) / (1 - r), cap))


def auc_p(auc: float, n1: int, n0: int, deff: float = 1.0) -> float:
    """One-sided normal-approximation p of an AUC above 0.5 with the design effect inflating the variance."""
    if n1 < 2 or n0 < 2 or not math.isfinite(auc):
        return 1.0
    var = (n1 + n0 + 1.0) / (12.0 * n1 * n0) * max(deff, 1.0)
    return float(sps.norm.sf((auc - 0.5) / math.sqrt(var)))


def fast_auc_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC (probability a positive row scores above a negative row, ties averaged) of every column of X at once."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0 or X.shape[0] != len(y):
        return np.full(X.shape[1] if X.ndim == 2 else 1, np.nan)
    r = sps.rankdata(X, axis=0)
    return (r[y].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def block_resample(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices (keeps the serial dependence the outcomes and the gate share)."""
    if n <= 0:
        return np.zeros(0, int)
    block = max(1, min(block, n))
    starts = rng.integers(0, max(n - block + 1, 1), size=int(math.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def shift_null(stat: Callable[[np.ndarray], float], gate: np.ndarray, n_perm: int, rng: np.random.Generator, lo_frac: float = 0.05) -> np.ndarray:
    """Statistic of the gate circularly shifted against the outcomes: keeps the gate's run structure, destroys any link."""
    n = len(gate)
    lo = max(int(lo_frac * n), 1)
    hi = max(n - lo, lo + 1)
    return np.array([stat(np.roll(gate, int(rng.integers(lo, hi)))) for _ in range(n_perm)])


def plus_one_p(obs: float, null: np.ndarray) -> float:
    """(1 + #{null >= obs}) / (n + 1): never exactly zero."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null):
        return 1.0
    return float((1.0 + (null >= obs).sum()) / (len(null) + 1.0))


def cvar(v: np.ndarray, q: float = 0.10) -> float:
    """Mean of the worst q share of outcomes (the expected shortfall); NaN when there are none."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    k = max(int(math.ceil(q * len(v))), 1)
    return float(np.sort(v)[:k].mean())


def max_drawdown(v: np.ndarray) -> float:
    """Deepest peak-to-trough fall of the cumulative sum of outcomes (a positive number; 0 = never fell)."""
    v = np.nan_to_num(np.asarray(v, float))
    if not len(v):
        return 0.0
    c = np.cumsum(v)
    peak = np.maximum.accumulate(np.r_[0.0, c])[1:]
    return float((peak - c).max())


def runs(mask: np.ndarray) -> list[int]:
    """Lengths of the consecutive True runs."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    return [int(b - a) for a, b in zip(edges[0::2], edges[1::2])]


def markov_gate(cover: float, mean_run: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random two-state gate with the given coverage and mean run length: the persistence-matched null explanation."""
    cover = min(max(cover, 1e-3), 1 - 1e-3)
    mean_run = max(mean_run, 1.0)
    stay = 1.0 - 1.0 / mean_run
    enter = min((cover / (1.0 - cover)) * (1.0 / mean_run), 1.0)
    g = np.zeros(n, bool)
    g[0] = rng.random() < cover
    u = rng.random(n)
    for i in range(1, n):
        g[i] = (u[i] < stay) if g[i - 1] else (u[i] < enter)
    return g


def anon(item_id: str) -> str:
    return stable_hash(["break-item", item_id], 10)


def safe_text(text: str) -> str:
    leak = RP.identity_leak(text)
    if leak:
        raise FirewallBreach(f"identity firewall: text {leak}: {text[:60]!r}")
    return text


# ------------------------------------------------------------------------------------------------- the analysis frame

@dataclasses.dataclass
class AnalysisFrame:
    """Everything the six questions need about one item at one `as_of`, built once (the design and episodes are the costly part).
    All arrays are aligned to the matured rows 0..m-1; nothing later than `as_of` is in here."""
    item: BD.ItemSeries
    item_key: str
    as_of: Any
    P: dict
    design: BD.Design
    cols: list
    X: pd.DataFrame
    v: np.ndarray
    m: int
    eps: list
    states: np.ndarray
    n_disc: int
    lab: np.ndarray                     # hindsight window labels over ALL matured rows (evaluation side only)
    pre: np.ndarray                     # hindsight pre-onset rows over all episodes (evaluation side only)
    stretch: np.ndarray                 # rows inside any broken stretch
    near: np.ndarray                    # rows inside a stretch or within 2*horizon before an onset
    bk: list = dataclasses.field(default_factory=list)   # detector episodes + label episodes (evaluation side, hindsight)

    @property
    def disc_rows(self) -> np.ndarray:
        return np.arange(0, self.n_disc)

    @property
    def conf_rows(self) -> np.ndarray:
        return np.arange(min(self.n_disc + self.P["embargo"], self.m), self.m)

    def index_at(self, row: int) -> str:
        return str(self.item.frame.index[min(max(row, 0), len(self.item.frame) - 1)])[:10]


def pre_onset_mask(eps: Sequence[BD.Episode], m: int, horizon: int, min_lead: int, upto: int | None = None) -> np.ndarray:
    """Rows in [onset - horizon, onset - min_lead] for every episode whose onset lies below `upto`."""
    out = np.zeros(m, bool)
    for e in eps:
        if upto is not None and e.onset >= upto:
            continue
        lo, hi = max(e.onset - horizon, 0), max(e.onset - min_lead + 1, 0)
        out[lo:min(hi, m)] = True
    return out


def stretch_mask(eps: Sequence[BD.Episode], m: int) -> np.ndarray:
    out = np.zeros(m, bool)
    for e in eps:
        out[e.onset:e.end(m)] = True
    return out


def near_mask(eps: Sequence[BD.Episode], m: int, horizon: int) -> np.ndarray:
    out = stretch_mask(eps, m)
    for e in eps:
        out[max(e.onset - 2 * horizon, 0):e.onset] = True
    return out


def label_episodes(lab: np.ndarray, limit: int, half: int, min_len: int = 5) -> list:
    """Episodes recovered from the hindsight window labels: each run of 'failing' labels (>= min_len rows) inside rows < limit is a
    break whose onset is the run's first row. The detector misses breaks that start before it has re-proved the pattern; the
    labels do not, and they keep undetected breaks from polluting the 'working' reference. A run is only KNOWN `half` rows after
    its start (the window label needs the rows after it), which is what `detect` records."""
    out = []
    failing = np.asarray(lab[:limit]) == -1
    edges = np.flatnonzero(np.diff(np.r_[0, failing.astype(np.int8), 0]))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a >= min_len:
            out.append(BD.Episode(int(min(a + half, limit - 1)), int(a), int(b) if b < limit else None))
    return out


def merge_episodes(det: Sequence[BD.Episode], extra: Sequence[BD.Episode], m: int, horizon: int) -> list:
    """Detector episodes plus label episodes that do not coincide with one (onsets within 2*horizon, or inside its stretch)."""
    out = list(det)
    for e in extra:
        if any(abs(e.onset - d.onset) <= 2 * horizon or d.onset <= e.onset < d.end(m) for d in det):
            continue
        out.append(e)
    return sorted(out, key=lambda e: e.onset)


def usable_columns(design: BD.Design, P: dict) -> list:
    cols = []
    for c in design.columns(causal=True):
        if design.tags[c].source in P["excluded_sources"] or c in P["excluded_sources"]:
            continue
        x = design.X[c].values.astype(float)
        if not np.isfinite(x).all() or x.std() < 1e-12:
            continue
        cols.append(c)
    return cols


def build_frame(item: BD.ItemSeries, as_of, cfg=None) -> AnalysisFrame:
    """Validate, run the future-leak canary (FirewallBreach on a leaking column), find the episodes, build the point-in-time design
    and the labels. The discovery/confirmation split is chronological with an embargo, exactly as in the break engine."""
    P = _cfg(cfg)
    B = _bd_cfg(P)
    item.require_valid()
    BD.assert_no_leak(item, as_of, B)
    m = item.n_matured(as_of)
    if m < 3:
        raise BreakResearchError(f"{item.item_id}: only {m} matured rows at {as_of}")
    eps, states = BD.find_episodes(item, as_of, B)
    design = BD.build_design(item, as_of, B, eps)
    v = item.frame["value"].values[:m].astype(float)
    lab = BD.window_labels(v, B["label_half"], B["label_t"], m)
    bk = merge_episodes(eps, label_episodes(lab, m, B["label_half"]), m, P["horizon"])
    return AnalysisFrame(item, anon(item.item_id), as_of, P, design, usable_columns(design, P), design.X, v, m, list(eps), states,
                         int(P["disc_frac"] * m), lab, pre_onset_mask(bk, m, P["horizon"], P["min_lead"]),
                         stretch_mask(bk, m), near_mask(bk, m, P["horizon"]), bk)


def event_severity(v: np.ndarray, onset: int, end: int, base: int = 26) -> float:
    """Drop of the mean outcome from the `base` rows before the onset to the broken stretch, in units of the earlier sd."""
    before = v[max(onset - base, 0):onset]
    during = v[onset:max(end, onset + 1)]
    before, during = before[np.isfinite(before)], during[np.isfinite(during)]
    if len(before) < 4 or len(during) < 2:
        return 0.0
    sd = float(before.std(ddof=1))
    return float((before.mean() - during.mean()) / sd) if sd > 1e-12 else 0.0


def detect_events(item: BD.ItemSeries, as_of, cfg=None) -> list:
    """Every break of `item` knowable at `as_of`, oldest first. A break detected at the `as_of` row is knowable (the detector
    only reads outcomes of earlier rows); the outcome of that row itself is hidden."""
    P = _cfg(cfg)
    eps, _ = BD.find_episodes(item, as_of, _bd_cfg(P))
    m = item.n_matured(as_of)
    v = item.frame["value"].values[:m].astype(float)
    key = anon(item.item_id)
    idx = item.frame.index
    out = []
    for e in eps:
        if e.detect >= max(m, 1):
            continue
        out.append(BreakEvent(key, e.onset, e.detect, e.recover, str(idx[e.onset])[:10], str(idx[e.detect])[:10],
                              event_severity(v, e.onset, e.end(m)), e.length(m), m))
    return out


def health_event(item: BD.ItemSeries, as_of, cfg=None) -> BreakEvent | None:
    """A BROKEN health row for an item with no detected episode still deserves a look: treat the last break_win rows as the
    suspected stretch. Returns None when there is too little history."""
    P = _cfg(cfg)
    B = _bd_cfg(P)
    m = item.n_matured(as_of)
    if m < B["work_win"] + B["break_win"]:
        return None
    v = item.frame["value"].values[:m].astype(float)
    onset = m - B["break_win"]
    idx = item.frame.index
    return BreakEvent(anon(item.item_id), onset, m - 1, None, str(idx[onset])[:10], str(idx[m - 1])[:10],
                      event_severity(v, onset, m), B["break_win"], m, "health")


# ------------------------------------------------------------------------------------------------- labels, selection, thresholds

def training_labels(F: AnalysisFrame, limit: int, eps: Sequence[BD.Episode]) -> tuple:
    """(pre, positive, negative) boolean arrays over 0..m-1 using ONLY information available for rows < `limit`:
    pre = rows shortly before each already-detected onset; positive = pre or inside a broken stretch (what a gate should
    withhold); negative = a clearly working row far from any break. Window labels read no row at or beyond `limit`."""
    P = F.P
    m = F.m
    rows = np.arange(m) < limit
    B = _bd_cfg(P)
    lab = np.zeros(m, np.int8)
    lab[:limit] = BD.window_labels(F.v, B["label_half"], B["label_t"], limit)
    known = merge_episodes([e for e in eps if e.detect < limit], label_episodes(lab, limit, B["label_half"]), m, P["horizon"])
    known = [e for e in known if e.detect < limit]
    stretch = stretch_mask([e for e in known if e.onset < limit], m) & rows
    pre = pre_onset_mask(known, m, P["horizon"], P["min_lead"], upto=limit - P["embargo"]) & rows & ~stretch
    neg = (lab == 1) & ~near_mask(known, m, P["horizon"]) & rows
    pos = pre | stretch
    return pre, pos & ~neg, neg & ~pos, known


def select_column(X: np.ndarray, pre_idx: np.ndarray, ref_idx: np.ndarray) -> tuple:
    """(best column position, signed Welch t of ref minus pre, |t| vector): the column that best separates the rows before a
    break from working rows. The same function selects in the real run and in every random-explanation run (Q6), so both
    pay the same price for searching."""
    if len(pre_idx) < 2 or len(ref_idx) < 2 or X.shape[1] == 0:
        return -1, 0.0, np.zeros(X.shape[1])
    t = BD.welch_columns(X[ref_idx], X[pre_idx])
    j = int(np.argmax(np.abs(t)))
    return j, float(t[j]), np.abs(t)


def fit_threshold(x: np.ndarray, direction: int, pos: np.ndarray, neg: np.ndarray, rows: np.ndarray, P: dict) -> tuple:
    """Youden-J threshold for `direction * x >= thr` on the fit rows, with the gate covering between cover_lo and cover_hi of
    all fit rows (a gate that withholds nothing or everything is not a gate). Returns (thr, J, tpr, fpr) or None."""
    s = direction * x[rows]
    p, n = pos[rows], neg[rows]
    if p.sum() < 3 or n.sum() < 3:
        return None
    best = None
    for q in P["cut_quantiles"]:
        thr = float(np.quantile(s, q))
        g = s >= thr
        cover = g.mean()
        if not (P["cover_lo"] <= cover <= P["cover_hi"]):
            continue
        tpr, fpr = float(g[p].mean()), float(g[n].mean())
        j = tpr - fpr
        if best is None or j > best[1]:
            best = (thr, j, tpr, fpr)
    if best is None or best[1] <= 0:
        return None
    return best


def fit_rule(F: AnalysisFrame, limit: int, eps: Sequence[BD.Episode], X: np.ndarray | None = None, cols: Sequence[str] | None = None):
    """Column + direction + threshold fitted on rows < limit only. Returns (rule or None, info dict)."""
    cols = list(cols) if cols is not None else F.cols
    Xm = F.X[cols].values.astype(float) if X is None else X
    pre, pos, neg, _ = training_labels(F, limit, eps)
    pre_idx, ref_idx = np.flatnonzero(pre), np.flatnonzero(neg)
    info = {"n_pre": int(len(pre_idx)), "n_ref": int(len(ref_idx)), "limit": int(limit)}
    if len(pre_idx) < F.P["min_pre_rows"] or len(ref_idx) < F.P["min_ref_rows"]:
        return None, {**info, "why": f"only {len(pre_idx)} pre-break rows / {len(ref_idx)} working rows to fit on"}
    j, t, abs_t = select_column(Xm, pre_idx, ref_idx)
    direction = -1 if t > 0 else 1                  # ref higher than pre -> withhold when the column is LOW
    rows = np.arange(limit)
    fit = fit_threshold(Xm[:, j], direction, pos, neg, rows, F.P)
    if fit is None:
        return None, {**info, "column": cols[j], "why": "no threshold gives a gate with usable coverage and positive J"}
    thr, jj, tpr, fpr = fit
    sc = direction * Xm[np.r_[pre_idx, ref_idx]][:, [j]]
    auc = float(fast_auc_columns(sc, np.r_[np.ones(len(pre_idx), bool), np.zeros(len(ref_idx), bool)])[0])
    return ColumnRule(cols[j], direction, float(thr), auc, jj), {**info, "column": cols[j], "abs_t": float(abs_t[j]), "tpr": tpr, "fpr": fpr}


def rule_score(rule: ColumnRule, X: pd.DataFrame) -> np.ndarray:
    """Signed distance of each row past the rule's threshold: positive = withhold."""
    return rule.direction * X[rule.column].values.astype(float) - rule.threshold


def min_detectable_auc(n1: int, n0: int, deff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest AUC an out-of-sample test of this size could have confirmed: it decides whether a miss is FAILED or NOT_TESTABLE."""
    if n1 < 2 or n0 < 2:
        return 1.0
    sd = math.sqrt((n1 + n0 + 1.0) / (12.0 * n1 * n0) * max(deff, 1.0))
    return float(min(0.5 + (sps.norm.isf(alpha) + sps.norm.ppf(power)) * sd, 1.0))


def _res(qid, outcome, stat, p, n, why, **detail) -> QuestionResult:
    return QuestionResult(qid, outcome, float(stat), float(p), int(n), detail, why)


# ------------------------------------------------------------------------------------------------- Q1: a pre-existing predictor

def lead_check(F: AnalysisFrame, col: str, eps: Sequence[BD.Episode], ref_idx: np.ndarray, bar: float = 0.4) -> tuple:
    """(verdict, pre, post): displacement of the column from the WORKING rows' level, in working-row sd, over the `horizon` rows
    before the onsets (pre) and over the rows from the onset on (post). 'leads' = already displaced by >= bar before the onset, in the
    direction it is displaced after it (or the post-onset displacement is itself small). A column that only moves with or after the
    onset is a symptom, however significantly it differs in the broken rows. Displacement is measured from the working level, not
    from the whole-history mean, because broken stretches drag the whole-history mean towards themselves."""
    h = F.P["horizon"]
    x = F.X[col].values.astype(float)
    if len(ref_idx) < 4 or not eps:
        return "flat", float("nan"), float("nan")
    mu, sd = float(x[ref_idx].mean()), float(x[ref_idx].std()) or 1.0
    pre_rows = np.concatenate([np.arange(max(e.onset - h, 0), max(e.onset - F.P["min_lead"] + 1, 0)) for e in eps])
    post_rows = np.concatenate([np.arange(e.onset, min(e.onset + h, F.m)) for e in eps])
    if not len(pre_rows) or not len(post_rows):
        return "flat", float("nan"), float("nan")
    pre, post = (float(x[pre_rows].mean()) - mu) / sd, (float(x[post_rows].mean()) - mu) / sd
    if abs(pre) >= bar and (pre * post >= 0 or abs(post) < bar):
        return "leads", pre, post
    return ("coincident" if abs(post) >= bar else "flat"), pre, post


def q1_preexisting_predictor(F: AnalysisFrame, seed: int = 0) -> tuple:
    """Is there a variable already displaced BEFORE the onset? Populations: rows shortly before each detected onset (failed)
    against working rows far from any break (success), compared on every causal column with the family-wise max-|t| bar of
    the break engine (circularly shifted labels). The winner must also LEAD in an event study (already moved before the onset,
    not merely with it) and be worth something (AUC bar). Returns (QuestionResult, rule or None)."""
    P = F.P
    qid = QuestionId.PREEXISTING_PREDICTOR
    pre, pos, neg, eps_d = training_labels(F, F.n_disc, F.eps)
    if len(eps_d) < P["min_episodes"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(eps_d),
                    f"only {len(eps_d)} break(s) knowable in the discovery window (need {P['min_episodes']})"), None
    pre_idx, ref_idx = np.flatnonzero(pre), np.flatnonzero(neg)
    if len(pre_idx) < P["min_pre_rows"] or len(ref_idx) < P["min_ref_rows"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(pre_idx),
                    f"only {len(pre_idx)} pre-break rows and {len(ref_idx)} working rows (need {P['min_pre_rows']}/{P['min_ref_rows']})"), None
    if not F.cols:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, 0, "the item supplies no usable causal column"), None
    pops = BD.Populations(ref_idx, pre_idx, "preonset", F.m, int(F.n_disc - len(pre_idx) - len(ref_idx)))
    table = BD.contrast_table(F.X, pops, F.design.tags, F.cols)
    obs, p_fam = BD.family_wise_p(F.X, pops, F.cols, P["n_perm"], seed)
    j = int(np.argmax(obs))
    col, p_best = F.cols[j], float(p_fam[j])
    row = table.iloc[j]
    auc_ref = float(row["auc"])
    pred_auc = max(auc_ref, 1 - auc_ref)
    power = BD.detectable_smd(len(ref_idx), len(pre_idx), len(F.cols), P["alpha"])
    detail = dict(column=col, dimension=str(row["dimension"]), auc=pred_auc, smd=float(row["smd"]), p_family=p_best,
                  n_tested=len(F.cols), n_pre=len(pre_idx), n_ref=len(ref_idx), n_breaks=len(eps_d), detectable_smd=power)
    if p_best > P["alpha"]:
        outcome = Outcome.FAILED if power <= P["max_detectable_smd"] else Outcome.NOT_TESTABLE
        why = (f"no column separates pre-break rows from working rows at the family-wise bar over {len(F.cols)} columns "
               f"(best {col}, p={p_best:.3f}); smallest standardised difference detectable here {power:.2f}")
        return _res(qid, outcome, pred_auc, p_best, len(pre_idx), why, **detail), None
    lead, pre_z, post_z = lead_check(F, col, eps_d, ref_idx)
    detail.update(lead=lead, pre_displacement=pre_z, post_displacement=post_z)
    if lead != "leads":
        return _res(qid, Outcome.FAILED, pred_auc, p_best, len(pre_idx),
                    f"{col} differs but only {lead} the onset: a symptom of the break, not a predictor of it", **detail), None
    if pred_auc < P["auc_bar"]:
        return _res(qid, Outcome.FAILED, pred_auc, p_best, len(pre_idx),
                    f"{col} is significant but weak (AUC {pred_auc:.2f} < {P['auc_bar']:.2f}): not worth gating on", **detail), None
    rule, info = fit_rule(F, F.n_disc, eps_d)
    if rule is None:
        return _res(qid, Outcome.FAILED, pred_auc, p_best, len(pre_idx), f"predictor found but no usable gate: {info.get('why')}", **detail), None
    detail.update(rule=rule.describe(), fit_j=rule.fit_j)
    return _res(qid, Outcome.PASSED, pred_auc, p_best, len(pre_idx),
                f"{col} leads the onset (AUC {pred_auc:.2f}, family-wise p={p_best:.3f} over {len(F.cols)} columns); {rule.describe()}", **detail), rule


# ------------------------------------------------------------------------------------------------- Q2: transfers out of sample

def eval_labels(F: AnalysisFrame, rows: np.ndarray) -> tuple:
    """Hindsight labels on the evaluation rows: positive = pre-onset or inside a broken stretch or a failing window; negative =
    a clearly working row far from any break. Rows in neither class are ambiguous and dropped."""
    pos = (F.pre | F.stretch | (F.lab == -1))
    neg = (F.lab == 1) & ~F.near & ~pos
    sel = np.zeros(F.m, bool)
    sel[rows] = True
    return pos & sel, neg & sel


def q2_transfers_oos(F: AnalysisFrame, rule: ColumnRule | None, seed: int = 0) -> QuestionResult:
    """The frozen rule scored on rows strictly after the discovery window (and the embargo). AUC with a design-effect-adjusted
    one-sided p, a block bootstrap interval, and the gate's precision against the base rate."""
    qid = QuestionId.TRANSFERS_OOS
    P = F.P
    if rule is None:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, 0, "no rule to transfer (Q1 produced none)")
    rows = F.conf_rows
    if len(rows) < P["min_confirm"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(rows), f"only {len(rows)} rows after the discovery window (need {P['min_confirm']})")
    pos, neg = eval_labels(F, rows)
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 < P["min_oos_pos"] or n0 < P["min_oos_pos"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, n1 + n0, f"only {n1} positive / {n0} negative rows out of sample (need {P['min_oos_pos']} each)")
    sel = np.flatnonzero(pos | neg)
    y = pos[sel]
    s = rule_score(rule, F.X)[sel]
    auc = PR.auc(s, y)
    deff = deff_of(y.astype(float), P["max_deff"])
    p = auc_p(auc, n1, n0, deff)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(P["boot"]):
        b = block_resample(len(sel), P["boot_block"], rng)
        a = PR.auc(s[b], y[b])
        if a == a:
            boots.append(a)
    lo = float(np.quantile(boots, 0.05)) if boots else float("nan")
    gate = rule.mask(F.X)[sel]
    precision = float(y[gate].mean()) if gate.any() else float("nan")
    detail = dict(auc=auc, auc_lo90=lo, deff=deff, n_pos=n1, n_neg=n0, precision=precision, base_rate=float(y.mean()),
                  gate_share=float(gate.mean()), recall=float(gate[y].mean()) if y.any() else float("nan"), column=rule.column)
    mdauc = min_detectable_auc(n1, n0, deff, P["oos_alpha"])
    detail["min_detectable_auc"] = mdauc
    if p <= P["oos_alpha"] and auc >= P["auc_bar"] and lo > 0.5:
        return _res(qid, Outcome.PASSED, auc, p, len(sel), f"AUC {auc:.2f} out of sample (90% lower {lo:.2f}, p={p:.3f}, {n1} vs {n0} rows)", **detail)
    outcome = Outcome.FAILED if mdauc <= 0.75 else Outcome.NOT_TESTABLE
    why = f"AUC {auc:.2f} out of sample (p={p:.3f}, 90% lower {lo:.2f}); the test could confirm an AUC of {mdauc:.2f} or better"
    return _res(qid, outcome, auc, p, len(sel), why, **detail)


# ------------------------------------------------------------------------------------------------- Q3: predicts FUTURE breaks

def trailing_weakness(v: np.ndarray, win: int = 8) -> np.ndarray:
    """Naive baseline: minus the mean outcome of the `win` rows before each row (rows < t only). B27: an item's own recent
    hit rate is a symptom, so a real predictor has to beat it."""
    s = pd.Series(v)
    return (-s.shift(1).rolling(win, min_periods=3).mean()).fillna(0.0).values


def time_since_break(eps: Sequence[BD.Episode], m: int) -> np.ndarray:
    """Baseline: rows since the last break KNOWN at each row (detect <= row); rows since the start when none."""
    det = sorted(e.detect for e in eps)
    out = np.zeros(m)
    k, last = 0, None
    for i in range(m):
        while k < len(det) and det[k] <= i:
            last = det[k]
            k += 1
        out[i] = (i - last) if last is not None else i + 1
    return out


def walk_forward_scores(F: AnalysisFrame) -> dict:
    """Refit the predictor at every origin using only what was knowable there, score the next `wf_step` rows. Returns the
    concatenated out-of-sample scores, truth, baselines and the column chosen at each origin."""
    P = F.P
    m = F.m
    step = P["wf_step"]
    start = min(F.n_disc + P["embargo"], m)
    trail = trailing_weakness(F.v)
    tsb = time_since_break(F.eps, m)
    clean = (F.lab == 1) & ~F.near
    S, Y, B1, B2, chosen, used = [], [], [], [], [], []
    for o in range(start, m, step):
        limit = o - P["embargo"]
        if limit < P["wf_min_train"]:
            continue
        rule, info = fit_rule(F, limit, F.eps)
        if rule is None:
            used.append({"origin": o, "fitted": False, "why": info.get("why", "")})
            continue
        rows = np.arange(o, min(o + step, m))
        rows = rows[clean[rows] | F.pre[rows]]           # a new break can only be predicted on rows not already inside one
        if not len(rows):
            continue
        sd = float(F.X[rule.column].values[:limit].std()) or 1.0
        S.append(rule_score(rule, F.X)[rows] / sd)
        Y.append(F.pre[rows])
        B1.append(tsb[rows])
        B2.append(trail[rows])
        chosen.append(rule.column)
        used.append({"origin": o, "fitted": True, "column": rule.column, "direction": rule.direction})
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return {"score": cat(S), "truth": cat(Y).astype(bool), "tsb": cat(B1), "trail": cat(B2), "chosen": chosen, "origins": used}


def q3_predicts_future_breaks(F: AnalysisFrame, seed: int = 0) -> QuestionResult:
    """Walk-forward: at each origin the rule is refitted on breaks already detected, then scores the following rows; the
    truth (a break onset within the next `horizon` rows) is read only afterwards. It must beat chance AND the best naive
    baseline (time since the last break, own recent weakness), and pick the same column at most origins."""
    qid = QuestionId.PREDICTS_FUTURE_BREAKS
    P = F.P
    start = min(F.n_disc + P["embargo"], F.m)
    future = [e for e in F.bk if e.onset >= start]
    if len(future) < P["min_future_breaks"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(future), f"only {len(future)} break(s) after the fitting window (need {P['min_future_breaks']})")
    wf = walk_forward_scores(F)
    y = wf["truth"]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < P["min_oos_pos"] or n0 < P["min_oos_pos"] or not wf["chosen"]:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(y), f"walk-forward left {n1} pre-break and {n0} other rows over {len(wf['chosen'])} fitted origins")
    s = wf["score"]
    auc = PR.auc(s, y)
    deff = deff_of(y.astype(float), P["max_deff"])
    p = auc_p(auc, n1, n0, deff)
    base = {nm: max(PR.auc(wf[nm], y), 1 - PR.auc(wf[nm], y)) for nm in ("tsb", "trail") if np.isfinite(PR.auc(wf[nm], y))}
    best_base = max(base.values()) if base else 0.5
    modal = max(set(wf["chosen"]), key=wf["chosen"].count)
    stability = wf["chosen"].count(modal) / len(wf["chosen"])
    detail = dict(auc=auc, baselines=base, best_baseline=best_base, stability=stability, modal_column=modal, n_origins=len(wf["chosen"]),
                  n_pre=n1, n_other=n0, deff=deff, n_future_breaks=len(future), origins=wf["origins"])
    good = p <= P["oos_alpha"] and auc >= P["auc_bar"] and auc >= best_base + P["baseline_margin"] and stability >= 0.5
    if good:
        return _res(qid, Outcome.PASSED, auc, p, len(y), f"walk-forward AUC {auc:.2f} (p={p:.3f}) vs best naive baseline {best_base:.2f}; {modal} chosen at {stability:.0%} of {len(wf['chosen'])} origins", **detail)
    reasons = []
    if p > P["oos_alpha"] or auc < P["auc_bar"]:
        reasons.append(f"AUC {auc:.2f} (p={p:.3f}) does not beat chance")
    if auc < best_base + P["baseline_margin"]:
        reasons.append(f"it does not beat the naive baseline ({best_base:.2f})")
    if stability < 0.5:
        reasons.append(f"the chosen column changes from origin to origin (modal share {stability:.0%})")
    mdauc = min_detectable_auc(n1, n0, deff, P["oos_alpha"])
    detail["min_detectable_auc"] = mdauc
    outcome = Outcome.FAILED if mdauc <= 0.75 else Outcome.NOT_TESTABLE
    return _res(qid, outcome, auc, p, len(y), "; ".join(reasons) + f" (test could confirm AUC {mdauc:.2f}+)", **detail)


# ------------------------------------------------------------------------------------------------- Q4 / Q5: does gating pay

def gate_frame(F: AnalysisFrame, rule: ColumnRule) -> tuple:
    """(outcomes, gate) over the confirmation rows: gate True = the pattern is withheld on that row."""
    rows = F.conf_rows
    return F.v[rows], rule.mask(F.X)[rows]


def gate_usable(v: np.ndarray, g: np.ndarray, P: dict) -> str:
    """Empty string when the gate can be assessed; otherwise why it cannot."""
    ok = np.isfinite(v)
    if len(v) < P["min_confirm"]:
        return f"only {len(v)} confirmation rows (need {P['min_confirm']})"
    if (g & ok).sum() < 5:
        return f"the gate fires on only {int((g & ok).sum())} confirmation rows"
    if (~g & ok).sum() < 10:
        return f"the gate leaves only {int((~g & ok).sum())} rows to trade"
    return ""


def expectancy_gain(v: np.ndarray, g: np.ndarray) -> float:
    """Mean outcome per exposure taken minus the ungated mean: what a gate is supposed to lift."""
    ok = np.isfinite(v)
    taken = ok & ~g
    if not taken.any() or not ok.any():
        return 0.0
    return float(v[taken].mean() - v[ok].mean())


def net_saved(v: np.ndarray, g: np.ndarray) -> float:
    """Total outcome avoided by withholding: loss avoided minus gain forgone (positive = the gate saved money). For a profitable
    pattern almost any gate has a negative net_saved, so skill is judged by excess_loss_avoided instead."""
    ok = np.isfinite(v)
    return float(-v[g & ok].sum())


def excess_loss_avoided(v: np.ndarray, g: np.ndarray) -> float:
    """Loss (sum of negative outcomes) inside the gate beyond what a gate of the same coverage would catch by chance: the loss
    the gate avoided that withholding random rows would not have avoided. Positive = the gate concentrates on losing rows."""
    ok = np.isfinite(v)
    if not ok.any():
        return 0.0
    neg = np.where(ok, np.minimum(v, 0.0), 0.0)
    cover = float(g[ok].mean())
    return float(-neg[g & ok].sum() + cover * neg.sum())


def q4_gating_improves(F: AnalysisFrame, rule: ColumnRule | None, seed: int = 0) -> QuestionResult:
    """Expectancy per exposure with and without the frozen gate on the confirmation rows. Null: the same gate circularly
    shifted against the outcomes (same coverage, same run structure, no link). Also a block-bootstrap interval of the lift."""
    qid = QuestionId.GATING_IMPROVES
    P = F.P
    if rule is None:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, 0, "no rule to gate with")
    v, g = gate_frame(F, rule)
    why = gate_usable(v, g, P)
    if why:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(v), why)
    rng = np.random.default_rng(seed + 401)
    obs = expectancy_gain(v, g)
    null = shift_null(lambda gg: expectancy_gain(v, gg), g, P["n_perm"], rng)
    p = plus_one_p(obs, null)
    boots = []
    for _ in range(P["boot"]):
        b = block_resample(len(v), P["boot_block"], rng)
        boots.append(expectancy_gain(v[b], g[b]))
    lo, hi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
    ok = np.isfinite(v)
    detail = dict(gain=obs, ci90=(lo, hi), mean_all=float(v[ok].mean()), mean_taken=float(v[ok & ~g].mean()),
                  mean_withheld=float(v[ok & g].mean()), coverage=float(g[ok].mean()), n_taken=int((ok & ~g).sum()),
                  n_withheld=int((ok & g).sum()), null_mean=float(np.mean(null)), null_sd=float(np.std(null)))
    if obs > 0 and p <= P["oos_alpha"] and lo > 0:
        return _res(qid, Outcome.PASSED, obs, p, len(v), f"gating lifts the mean outcome per exposure by {obs:+.4f} (90% interval {lo:+.4f}..{hi:+.4f}, shift-null p={p:.3f})", **detail)
    sd = float(np.nanstd(v))
    resolvable = 2.0 * sd / math.sqrt(max(int((ok & g).sum()), 1)) if sd > 0 else float("inf")
    detail["resolvable_lift"] = resolvable
    outcome = Outcome.FAILED if resolvable < max(abs(obs), 0.0) * 4 + 1e-12 or int((ok & g).sum()) >= 10 else Outcome.NOT_TESTABLE
    return _res(qid, outcome, obs, p, len(v), f"gating changes the mean outcome per exposure by {obs:+.4f} (90% interval {lo:+.4f}..{hi:+.4f}, shift-null p={p:.3f}): no reliable improvement", **detail)


def q5_gating_reduces_losses(F: AnalysisFrame, rule: ColumnRule | None, seed: int = 0) -> QuestionResult:
    """Improved expectancy can hide a gate that merely moves risk around. The gate must reduce the DOWNSIDE: the loss it avoids
    beyond a random gate of the same coverage must be positive against a shift null and a bootstrap, and the taken series must
    have a milder CVaR(10%) and no deeper drawdown than the ungated one. (Absolute net_saved is reported but not the criterion:
    withholding any rows of a profitable pattern forgoes profit.)"""
    qid = QuestionId.GATING_REDUCES_LOSSES
    P = F.P
    if rule is None:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, 0, "no rule to gate with")
    v, g = gate_frame(F, rule)
    why = gate_usable(v, g, P)
    if why:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(v), why)
    rng = np.random.default_rng(seed + 503)
    ok = np.isfinite(v)
    obs = excess_loss_avoided(v, g)
    null = shift_null(lambda gg: excess_loss_avoided(v, gg), g, P["n_perm"], rng)
    p = plus_one_p(obs, null)
    boots = []
    for _ in range(P["boot"]):
        b = block_resample(len(v), P["boot_block"], rng)
        boots.append(excess_loss_avoided(v[b], g[b]))
    lo = float(np.quantile(boots, 0.05))
    taken = np.where(g, 0.0, np.nan_to_num(v))
    cv_all, cv_taken = cvar(v[ok]), cvar(v[ok & ~g])
    dd_all, dd_taken = max_drawdown(v), max_drawdown(taken)
    loss = ok & (v < 0)
    loss_avoided = float(-v[g & loss].sum())
    gain_forgone = float(v[g & ok & (v > 0)].sum())
    detail = dict(excess_loss_avoided=obs, net_saved=net_saved(v, g), ci_lo90=lo, loss_avoided=loss_avoided, gain_forgone=gain_forgone, cvar_all=cv_all, cvar_taken=cv_taken,
                  max_dd_all=dd_all, max_dd_taken=dd_taken, worst_all=float(np.nanmin(v)), worst_taken=float(np.nanmin(np.where(g, np.nan, v))) if (~g & ok).any() else float("nan"),
                  loss_capture=float(g[loss].mean()) if loss.any() else float("nan"), coverage=float(g[ok].mean()))
    milder = cv_taken >= cv_all
    shallower = dd_taken <= dd_all + P["drawdown_slack"]
    detail.update(cvar_milder=bool(milder), drawdown_shallower=bool(shallower))
    if obs > 0 and p <= P["oos_alpha"] and lo > 0 and milder and shallower:
        return _res(qid, Outcome.PASSED, obs, p, len(v), f"gate avoids {obs:+.4f} more loss than a same-coverage random gate (loss avoided {loss_avoided:.4f}, gain forgone {gain_forgone:.4f}); CVaR10 {cv_all:+.4f} -> {cv_taken:+.4f}, drawdown {dd_all:.4f} -> {dd_taken:.4f}", **detail)
    bad = []
    if not (obs > 0 and p <= P["oos_alpha"] and lo > 0):
        bad.append(f"loss avoided beyond a random gate {obs:+.4f} is not reliably positive (p={p:.3f}, 90% lower {lo:+.4f})")
    if not milder:
        bad.append(f"CVaR10 got worse ({cv_all:+.4f} -> {cv_taken:+.4f})")
    if not shallower:
        bad.append(f"drawdown got deeper ({dd_all:.4f} -> {dd_taken:.4f})")
    outcome = Outcome.FAILED if int((g & ok).sum()) >= 10 else Outcome.NOT_TESTABLE
    return _res(qid, outcome, obs, p, len(v), "; ".join(bad), **detail)


# ------------------------------------------------------------------------------------------------- Q6: beats random explanations

def random_explanation_gains(F: AnalysisFrame, rule: ColumnRule, seed: int = 0) -> dict:
    """Gain (Q4's statistic) of random explanations produced by the SAME search the real one paid for.
    shifted: every candidate column circularly shifted by its own random offset (its distribution and autocorrelation are kept,
      its link to the outcomes is destroyed), then the column and threshold are selected on the discovery rows exactly as in Q1
      and the gate scored on the confirmation rows.
    markov: a random gate with the real gate's coverage and mean run length, no column at all."""
    P = F.P
    rng = np.random.default_rng(seed + 601)
    m, k = F.m, len(F.cols)
    Xm = F.X[F.cols].values.astype(float)
    pre, pos, neg, _ = training_labels(F, F.n_disc, F.eps)
    pre_idx, ref_idx = np.flatnonzero(pre), np.flatnonzero(neg)
    rows = F.conf_rows
    v = F.v[rows]
    real_gate = rule.mask(F.X)[rows]
    fit_rows = np.arange(F.n_disc)
    lo, hi = max(int(0.05 * m), 1), max(int(0.95 * m), 2)
    shifted, markov = [], []
    rl = runs(real_gate)
    mean_run = float(np.mean(rl)) if rl else 2.0
    cover = float(real_gate.mean())
    for _ in range(P["n_random"]):
        offs = rng.integers(lo, hi, size=k)
        Xr = np.column_stack([np.roll(Xm[:, j], int(offs[j])) for j in range(k)])
        gm = markov_gate(cover, mean_run, len(rows), rng)
        markov.append(expectancy_gain(v, gm) if gate_usable(v, gm, P) == "" else 0.0)
        j, t, _ = select_column(Xr, pre_idx, ref_idx)
        if j < 0:
            shifted.append(0.0)
            continue
        direction = -1 if t > 0 else 1
        fit = fit_threshold(Xr[:, j], direction, pos, neg, fit_rows, P)
        if fit is None:
            shifted.append(0.0)
            continue
        g = (direction * Xr[rows, j]) >= fit[0]
        shifted.append(expectancy_gain(v, g) if gate_usable(v, g, P) == "" else 0.0)
    return {"shifted": np.array(shifted), "markov": np.array(markov), "mean_run": mean_run, "cover": cover}


def q6_beats_random(F: AnalysisFrame, rule: ColumnRule | None, seed: int = 0) -> QuestionResult:
    """The real explanation's gain must exceed what random explanations reach when they get the same search: p is the LARGER of
    the shifted-column p and the persistence-matched-gate p, so it must beat both kinds of randomness."""
    qid = QuestionId.BEATS_RANDOM
    P = F.P
    if rule is None:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, 0, "no explanation to compare with random ones")
    v, g = gate_frame(F, rule)
    why = gate_usable(v, g, P)
    if why:
        return _res(qid, Outcome.NOT_TESTABLE, float("nan"), 1.0, len(v), why)
    obs = expectancy_gain(v, g)
    rnd = random_explanation_gains(F, rule, seed)
    p_sh, p_mk = plus_one_p(obs, rnd["shifted"]), plus_one_p(obs, rnd["markov"])
    p = max(p_sh, p_mk)
    detail = dict(gain=obs, p_shifted=p_sh, p_markov=p_mk, null_shifted_mean=float(rnd["shifted"].mean()),
                  null_shifted_q95=float(np.quantile(rnd["shifted"], 0.95)), null_markov_q95=float(np.quantile(rnd["markov"], 0.95)),
                  n_random=len(rnd["shifted"]), mean_run=rnd["mean_run"], coverage=rnd["cover"])
    if obs > 0 and p <= P["oos_alpha"]:
        return _res(qid, Outcome.PASSED, obs, p, len(rnd["shifted"]), f"gain {obs:+.4f} beats {len(rnd['shifted'])} random explanations from the same search (shifted-column p={p_sh:.3f}, matched-gate p={p_mk:.3f})", **detail)
    resolution = 1.0 / (len(rnd["shifted"]) + 1.0)
    outcome = Outcome.FAILED if resolution < P["oos_alpha"] else Outcome.NOT_TESTABLE
    return _res(qid, outcome, obs, p, len(rnd["shifted"]), f"random explanations from the same search reach a gain of {obs:+.4f} or more too often (shifted-column p={p_sh:.3f}, matched-gate p={p_mk:.3f})", **detail)


# ------------------------------------------------------------------------------------------------- adjudication

def adjudicate(results: Sequence[QuestionResult]) -> tuple:
    """(verdict, knowability, gate verdict, statement) from the six results. EXPLAINED needs all six PASSED. A FAILED question
    makes the answer UNKNOWN; a NOT_TESTABLE one with no failure makes it INSUFFICIENT_EVIDENCE. Both are honest answers."""
    by = {r.qid: r for r in results}
    missing = [q for q in QUESTION_ORDER if q not in by]
    if len(missing) == len(QUESTION_ORDER):
        return Verdict.INSUFFICIENT_EVIDENCE, Knowability.UNKNOWN, GateVerdict.NEEDS_MORE_EVIDENCE, "no question has been run yet"
    passed = [q for q in QUESTION_ORDER if q in by and by[q].passed]
    failed = [q for q in QUESTION_ORDER if q in by and by[q].outcome == Outcome.FAILED]
    untestable = [q for q in QUESTION_ORDER if q in by and by[q].outcome == Outcome.NOT_TESTABLE]
    if len(passed) == len(QUESTION_ORDER):
        return Verdict.EXPLAINED, Knowability.PREDICTABLE, GateVerdict.PROMOTE, "all six section-14 questions passed"
    if failed:
        weak = len(passed) >= 3 and QuestionId.PREEXISTING_PREDICTOR in passed
        know = Knowability.WEAKLY_PREDICTABLE if weak else Knowability.UNKNOWN
        first = by[failed[0]]
        return Verdict.UNKNOWN, know, GateVerdict.UNKNOWN, f"{first.qid.value} failed: {first.why}"
    first = by[untestable[0]] if untestable else None
    stmt = f"{first.qid.value} could not be answered: {first.why}" if first else "questions still pending"
    return Verdict.INSUFFICIENT_EVIDENCE, Knowability.UNKNOWN, GateVerdict.NEEDS_MORE_EVIDENCE, stmt


def run_ladder(F: AnalysisFrame, seed: int = 0, stop_early: bool = True) -> tuple:
    """The escalation ladder (section 18): cheap screen first, then stronger tests, then cross-checks, then the random-explanation
    control. With stop_early a failed or untestable stage ends the run, so compute is spent only where the previous stage earned it.
    Returns (results, rule)."""
    results: list[QuestionResult] = []
    r1, rule = q1_preexisting_predictor(F, seed)
    results.append(r1)
    if stop_early and not r1.passed:
        return results, None
    for fn in (q2_transfers_oos, q3_predicts_future_breaks):
        r = fn(F, rule, seed) if fn is q2_transfers_oos else fn(F, seed)
        results.append(r)
        if stop_early and not r.passed:
            return results, rule
    for fn in (q4_gating_improves, q5_gating_reduces_losses):
        r = fn(F, rule, seed)
        results.append(r)
        if stop_early and not r.passed:
            return results, rule
    results.append(q6_beats_random(F, rule, seed))
    return results, rule


# ------------------------------------------------------------------------------------------------- hypotheses: the section-14 cause list

def _pops(F: AnalysisFrame) -> BD.Populations:
    """Working vs failed rows of the discovery window (window labels bounded by the window, as in the break engine)."""
    return BD.split_populations(F.item, F.eps, F.states, F.as_of, _bd_cfg(F.P), rows=F.disc_rows)


def _ev(cause, supported, strength, p, n, detail, cols=()) -> HypothesisEvidence:
    return HypothesisEvidence(cause, supported, float(min(max(strength, 0.0), 1.0)), float(min(max(p, 0.0), 1.0)), int(n), detail, tuple(cols))


def dimension_support(F: AnalysisFrame, cause: BreakCause, pops: BD.Populations, seed: int = 0) -> HypothesisEvidence:
    """Do the working and failed rows differ on the dimensions this cause lives on? Family-wise over just those columns (the
    family is what the cause claims, so the bar is not diluted by unrelated columns; multiplicity across the 13 causes is paid
    later with Holm)."""
    dims = DIMENSIONS_OF[cause]
    cols = [c for c in F.design.columns(causal=True) if F.design.tags[c].dimension in dims and c in F.X.columns]
    if not cols:
        return _ev(cause, None, 0.0, 1.0, 0, f"no column measures {', '.join(dims)}: not testable, not ruled out")
    short = pops.enough(_bd_cfg(F.P))
    if short:
        return _ev(cause, None, 0.0, 1.0, len(pops.failed), f"discovery window: {short}")
    table = BD.contrast_table(F.X, pops, F.design.tags, cols)
    obs, p = BD.family_wise_p(F.X, pops, cols, F.P["n_perm"], seed)
    j = int(np.argmax(obs))
    smd = float(table.iloc[j]["smd"])
    strength = min(abs(smd) / 1.5, 1.0)
    ok = float(p[j]) <= F.P["alpha"] and strength >= 0.2
    return _ev(cause, ok, strength if ok else 0.0, float(p[j]), len(pops.failed) + len(pops.success),
               f"{cols[j]} differs by {smd:+.2f} sd between working and failed rows (family-wise p={float(p[j]):.3f} over {len(cols)} columns)", (cols[j],))


def test_sector_shift(F: AnalysisFrame, pops: BD.Populations, seed: int = 0) -> HypothesisEvidence:
    """Composition groups are compared as wholes (total variation of the mix) before falling back to per-column tests."""
    groups = [g for g, (d, _) in F.item.compositions.items() if d in ("sector_composition", "stock_type")]
    if not groups:
        return dimension_support(F, BreakCause.SECTOR_SHIFT, pops, seed)
    if len(pops.success) < 2 or len(pops.failed) < 2:
        return _ev(BreakCause.SECTOR_SHIFT, None, 0.0, 1.0, 0, "too few working/failed rows to compare the mix")
    shifts = [BD.composition_shift(F.design, g, pops, F.P["n_perm"], seed + i) for i, g in enumerate(groups)]
    best = min(shifts, key=lambda s: s["p"])
    ok = best["p"] <= F.P["alpha"] and best["tv"] >= 0.05
    return _ev(BreakCause.SECTOR_SHIFT, ok, min(best["tv"] / 0.3, 1.0) if ok else 0.0, best["p"], len(pops.failed),
               f"the mix moved by total variation {best['tv']:.2f} (p={best['p']:.3f}); largest mover {best['largest_mover']}", (str(best["largest_mover"]),))


def _welch_p(a: np.ndarray, b: np.ndarray, deff: float = 1.0) -> tuple:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 4 or len(b) < 4:
        return 0.0, 1.0
    va, vb = a.var(ddof=1) * deff / len(a), b.var(ddof=1) * deff / len(b)
    if va + vb <= 1e-18:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    return float(t), float(2 * sps.norm.sf(abs(t)))


def test_market_structure(F: AnalysisFrame) -> HypothesisEvidence:
    """A structural change is a level shift in a market column that STARTS with the break and does not revert: a large
    before/after shift, a stable second half of the after-period, and a break that is still open or long."""
    cause = BreakCause.MARKET_STRUCTURE_CHANGE
    cols = [c for c in F.design.columns(causal=True) if F.design.tags[c].dimension in DIMENSIONS_OF[cause] and c in F.X.columns and "*" not in c]
    eps = [e for e in F.eps if e.detect < F.m]
    if not cols or not eps:
        return _ev(cause, None, 0.0, 1.0, 0, "no market column or no break to test")
    W = 26
    best = None
    ps = []
    for c in cols:
        x = F.X[c].values.astype(float)
        sd = float(x.std()) or 1.0
        for e in eps:
            end = e.end(F.m)
            before, after = x[max(e.onset - W, 0):e.onset], x[e.onset:min(end, e.onset + 2 * W)]
            if len(before) < 8 or len(after) < 12:
                continue
            half = len(after) // 2
            shift = abs(after.mean() - before.mean()) / sd
            revert = abs(after[half:].mean() - after[:half].mean()) / sd
            t, p = _welch_p(after, before, deff=3.0)
            ps.append(p)
            persistent = e.recover is None or e.length(F.m) >= W
            score = shift - revert
            if persistent and shift >= 1.0 and revert < 0.5 and (best is None or score > best[0]):
                best = (score, c, p, shift, revert)
    if best is None:
        return _ev(cause, False, 0.0, min(ps) if ps else 1.0, len(ps), "no market column shows a persistent level shift that starts with the break")
    n_family = max(len(ps), 1)
    padj = min(best[2] * n_family, 1.0)
    ok = padj <= F.P["alpha"]
    return _ev(cause, ok, min(best[3] / 2.0, 1.0) if ok else 0.0, padj, n_family, f"{best[1]} shifted by {best[3]:.2f} sd at the onset and stayed there (reversion {best[4]:.2f} sd)", (best[1],))


def test_event_environment(F: AnalysisFrame) -> HypothesisEvidence:
    """Do onsets recur at the same point of the calendar? Rayleigh test on the phase of each onset (day of year when the index is
    dated, position modulo 13 otherwise). Needs at least three onsets to say anything."""
    cause = BreakCause.EVENT_ENVIRONMENT
    onsets = [e.onset for e in F.eps]
    if len(onsets) < 3:
        return _ev(cause, None, 0.0, 1.0, len(onsets), f"only {len(onsets)} onset(s): recurrence cannot be tested")
    idx = F.item.frame.index
    if isinstance(idx, pd.DatetimeIndex):
        ph = 2 * math.pi * (idx[onsets].dayofyear.values - 1) / 365.25
    else:
        ph = 2 * math.pi * (np.asarray(onsets) % 13) / 13.0
    n = len(ph)
    R = math.hypot(np.cos(ph).sum(), np.sin(ph).sum()) / n
    Z = n * R * R
    p = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n * n))
    p = min(max(p, 0.0), 1.0)
    ok = p <= 0.05
    return _ev(cause, ok, R if ok else 0.0, p, n, f"onset phases have resultant length {R:.2f} over {n} onsets (Rayleigh p={p:.3f})")


def test_pattern_crowding(F: AnalysisFrame) -> HypothesisEvidence:
    """Crowding is a GRADUAL decline before the break (edge competed away), often with neighbouring patterns becoming more
    correlated. For every onset: OLS slope of the outcome over the 3*horizon rows before it; slopes are combined across breaks
    (Stouffer). Rising neighbour correlation, when measured, adds strength."""
    cause = BreakCause.PATTERN_CROWDING
    H = 3 * F.P["horizon"]
    zs, slopes = [], []
    for e in F.eps:
        seg = F.v[max(e.onset - H, 0):e.onset + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 12:
            continue
        r = sps.linregress(np.arange(len(seg)), seg)
        slopes.append(r.slope)
        zs.append(float(sps.norm.isf(r.pvalue / 2) * (-1 if r.slope > 0 else 1)) if r.pvalue > 0 else 0.0)
    if not zs:
        return _ev(cause, None, 0.0, 1.0, 0, "no break with a long enough run-up")
    z = sum(zs) / math.sqrt(len(zs))
    p = float(sps.norm.sf(z))
    share_neg = float(np.mean(np.asarray(slopes) < 0))
    rising = 0.0
    if "nb_corr_trail" in F.X.columns:
        c = F.X["nb_corr_trail"].values.astype(float)
        deltas = [c[max(e.onset - 4, 0):e.onset + 1].mean() - c[max(e.onset - H, 0):max(e.onset - 2 * F.P["horizon"], 1)].mean() for e in F.eps if e.onset > 8]
        rising = float(np.mean(deltas)) if deltas else 0.0
    ok = p <= 0.05 and share_neg >= 0.5
    strength = min(0.5 * share_neg + max(rising, 0.0), 1.0) if ok else 0.0
    return _ev(cause, ok, strength, p, len(zs), f"outcomes decline into {share_neg:.0%} of onsets (combined z={z:.2f}); neighbour correlation change {rising:+.2f}")


def test_feature_relationship(F: AnalysisFrame, pops: BD.Populations) -> HypothesisEvidence:
    """Did the relation between a feature and the outcome change while the feature itself did not? Spearman correlation of every
    plain context column with the outcome, working vs failed rows, compared with a Fisher z test (rows deflated by a design
    effect); Holm across columns. A regime shift moves the column; a relationship change moves only its link to the outcome."""
    cause = BreakCause.FEATURE_RELATIONSHIP_CHANGED
    cols = [c for c in F.X.columns if F.design.tags[c].source == c and "*" not in c and "__" not in c and F.design.tags[c].causal]
    if not cols or len(pops.success) < 20 or len(pops.failed) < 20:
        return _ev(cause, None, 0.0, 1.0, len(pops.failed), "too few rows or no plain feature column")
    ps, out = [], []
    for c in cols:
        x = F.X[c].values.astype(float)
        a, b = pops.success, pops.failed
        ra, _ = sps.spearmanr(x[a], F.v[a])
        rb, _ = sps.spearmanr(x[b], F.v[b])
        if not (math.isfinite(ra) and math.isfinite(rb)):
            continue
        sd = float(np.std(x)) or 1.0
        level = abs(x[a].mean() - x[b].mean()) / sd
        za, zb = math.atanh(min(max(ra, -0.999), 0.999)), math.atanh(min(max(rb, -0.999), 0.999))
        se = math.sqrt(3.0 / max(len(a) - 3, 1) + 3.0 / max(len(b) - 3, 1))
        z = (za - zb) / se
        ps.append(float(2 * sps.norm.sf(abs(z))))
        out.append((c, ra, rb, level))
    if not ps:
        return _ev(cause, None, 0.0, 1.0, 0, "no column with a defined correlation")
    padj = PR.holm(ps)
    j = int(np.argmin(padj))
    c, ra, rb, level = out[j]
    ok = padj[j] <= F.P["alpha"] and level < 0.5 and abs(ra - rb) >= 0.2
    return _ev(cause, ok, min(abs(ra - rb), 1.0) if ok else 0.0, float(padj[j]), len(pops.failed),
               f"{c}: correlation with the outcome {ra:+.2f} while working, {rb:+.2f} while failing (level moved {level:.2f} sd)", (c,))


def winsorised_t(v: np.ndarray, a: np.ndarray, b: np.ndarray, q: float = 0.05) -> tuple:
    """Welch t of (working - failed) outcomes raw and after clipping to the [q, 1-q] quantiles of all outcomes."""
    lo, hi = np.nanquantile(v, [q, 1 - q])
    raw, _ = _welch_p(v[a], v[b], 3.0)
    win, _ = _welch_p(np.clip(v[a], lo, hi), np.clip(v[b], lo, hi), 3.0)
    return raw, win


def test_measurement_error(F: AnalysisFrame, pops: BD.Populations, seed: int = 0) -> HypothesisEvidence:
    """Bad data shows as missingness/data-quality shifts, or as a break carried by a few extreme outcomes: if the working-vs-
    failed difference is significant raw but vanishes once outcomes are winsorised, outliers (not the pattern) made the break."""
    base = dimension_support(F, BreakCause.MEASUREMENT_ERROR, pops, seed)
    if len(pops.success) < 8 or len(pops.failed) < 8:
        return base
    raw, win = winsorised_t(F.v, pops.success, pops.failed)
    outlier = raw >= 2.0 and win < 1.0
    if base.supported:
        return base
    if outlier:
        return _ev(BreakCause.MEASUREMENT_ERROR, True, 0.6, float(2 * sps.norm.sf(raw)), len(pops.failed),
                   f"the working-vs-failed difference (t={raw:.1f}) disappears when outcomes are winsorised (t={win:.1f}): a few extreme values carry it")
    return _ev(BreakCause.MEASUREMENT_ERROR, False if base.supported is not None or raw == raw else None, 0.0, base.p, base.n,
               base.detail + f"; winsorising leaves t at {win:.1f} (raw {raw:.1f}), so outliers do not carry the break")


def test_sampling_artifact(F: AnalysisFrame, pops: BD.Populations, seed: int = 0) -> HypothesisEvidence:
    """Thin or concentrated samples produce brief spurious breaks: sample-size / concentration shifts, or every break being too
    short to be more than a few observations."""
    cause = BreakCause.SAMPLING_ARTIFACT
    base = dimension_support(F, cause, pops, seed)
    if base.supported:
        return base
    lens = [e.length(F.m) for e in F.eps]
    if lens and max(lens) <= 6:
        return _ev(cause, True, 0.4, 0.5, len(lens), f"every break lasted at most {max(lens)} rows: too brief to distinguish from a thin sample")
    return _ev(cause, False if lens else None, 0.0, base.p, base.n, (base.detail + "; ") + (f"breaks last {min(lens)}..{max(lens)} rows" if lens else "no break to measure"))


def scan_min_t(v: np.ndarray, w: int) -> float:
    """Most negative t of the mean of any `w` consecutive outcomes: the strongest drop a scan of the history would report."""
    x = pd.Series(v)
    mu = x.rolling(w, min_periods=w).mean()
    sd = x.rolling(w, min_periods=w).std()
    t = (mu / (sd / math.sqrt(w))).replace([np.inf, -np.inf], np.nan).dropna()
    return float(t.min()) if len(t) else 0.0


def test_random_variation(F: AnalysisFrame, seed: int = 0) -> HypothesisEvidence:
    """Would a stable pattern show a drop this deep somewhere in a history this long? The outcome series is cut into blocks
    of the break window and the blocks are permuted (dependence inside a block is kept); the observed strongest drop is
    compared with the strongest drop of each permutation. A large p means chance is a live explanation, not a proven one."""
    cause = BreakCause.RANDOM_VARIATION
    w = _bd_cfg(F.P)["break_win"]
    v = F.v[np.isfinite(F.v)]
    if len(v) < 4 * w:
        return _ev(cause, None, 0.0, 1.0, len(v), f"history of {len(v)} rows is too short for a scan test")
    obs = scan_min_t(v, w)
    rng = np.random.default_rng(seed + 211)
    nb = len(v) // w
    blocks = [v[i * w:(i + 1) * w] for i in range(nb)]
    null = np.empty(F.P["n_perm"])
    for i in range(len(null)):
        order = rng.permutation(nb)
        null[i] = scan_min_t(np.concatenate([blocks[j] for j in order]), w)
    p = float((1.0 + (null <= obs).sum()) / (len(null) + 1.0))
    ok = p > 0.10
    return _ev(cause, ok, min(p, 1.0) if ok else 0.0, p, len(v),
               f"the deepest {w}-row drop (t={obs:.2f}) is {'not unusual' if ok else 'unusually deep'} among block-permuted histories (p={p:.3f})")


def test_hidden_interaction(F: AnalysisFrame) -> HypothesisEvidence:
    """Does the outcome depend on two causal columns TOGETHER (a product term with weak main effects)? Uses the break engine's
    interaction scan on the discovery rows; the pair count is paid with Holm inside the scan."""
    cause = BreakCause.HIDDEN_INTERACTION
    base_cols = [c for c in F.cols if "*" not in c and "__" not in c and F.design.tags[c].source == c]
    if len(base_cols) < 2:
        return _ev(cause, None, 0.0, 1.0, 0, "fewer than two plain columns to interact")
    df = BD.interaction_scan(F.design, F.v, F.disc_rows, base_cols, max_cols=8)
    if not len(df):
        return _ev(cause, None, 0.0, 1.0, 0, "no pair had enough variation to test")
    top = df.iloc[0]
    ok = bool(df["pure_interaction"].any())
    row = df[df["pure_interaction"]].iloc[0] if ok else top
    return _ev(cause, ok, min(abs(float(row["beta_product"])) / (abs(float(row["beta_a"])) + abs(float(row["beta_b"])) + 1e-9), 1.0) if ok else 0.0,
               float(row["p_holm"]), int(row["n"]), f"{row['a']} x {row['b']}: product term p={float(row['p_holm']):.3f} (Holm over {len(df)} pairs), main effects weaker", (str(row["a"]), str(row["b"])))


def test_external_shock(F: AnalysisFrame, shared: Sequence[Mapping] = ()) -> HypothesisEvidence:
    """An outside shock is abrupt, has no observable precursor and often breaks other patterns at the same time. Abrupt =
    the mean outcome steps by at least 2 sd within a few rows of the onset."""
    cause = BreakCause.EXTERNAL_SHOCK
    if not F.eps:
        return _ev(cause, None, 0.0, 1.0, 0, "no break")
    steps = []
    for e in F.eps:
        b, a = F.v[max(e.onset - 6, 0):e.onset], F.v[e.onset:e.onset + 6]
        b, a = b[np.isfinite(b)], a[np.isfinite(a)]
        if len(b) < 4 or len(a) < 3:
            continue
        sd = float(np.std(np.r_[b, a])) or 1.0
        steps.append((b.mean() - a.mean()) / sd)
    if not steps:
        return _ev(cause, None, 0.0, 1.0, 0, "not enough rows around any onset")
    abrupt = float(np.mean(np.asarray(steps) >= 1.0))
    sc = [s for s in shared if any(F.item_key == anon(k) or k == F.item.item_id for k in s["items"])]
    try:
        cols = F.cols[:20]
        prof = BD.event_study(F.design, F.eps, cols, window=F.P["lead_window"])
        leaders = [c for c in cols if BD.lead_lag_verdict(prof.loc[c], window=F.P["lead_window"]) == "leads"]
    except (KeyError, ValueError):
        leaders = []
    ok = abrupt >= 0.5 and (bool(sc) or not leaders)
    strength = min(0.5 * abrupt + (0.4 if sc else 0.0) + (0.2 if not leaders else 0.0), 1.0) if ok else 0.0
    return _ev(cause, ok, strength, 0.5 if ok else 1.0, len(steps),
               f"{abrupt:.0%} of onsets are abrupt steps; {len(leaders)} column(s) lead; shared with other patterns: {'yes' if sc else 'no'}")


def hypothesis_priors(F: AnalysisFrame, event: BreakEvent, shared: Sequence[Mapping] = ()) -> dict:
    """Base rates adjusted by what is visible before any test: a shared break points to outside causes, repeated onsets to a
    recurring environment, a mild drop to chance, no neighbours means no crowding evidence. UNKNOWN keeps its floor."""
    pr = dict(BASE_PRIOR)
    if any(s["share"] >= 0.3 for s in shared):
        pr[BreakCause.EXTERNAL_SHOCK] *= 3.0
        pr[BreakCause.REGIME_CHANGE] *= 1.5
    if len(F.eps) >= 3:
        pr[BreakCause.EVENT_ENVIRONMENT] *= 3.0
    if event.severity < 1.5:
        pr[BreakCause.RANDOM_VARIATION] *= 2.0
        pr[BreakCause.SAMPLING_ARTIFACT] *= 1.5
    if F.item.neighbours is None:
        pr[BreakCause.PATTERN_CROWDING] *= 0.5
    if event.open and event.length >= 26:
        pr[BreakCause.MARKET_STRUCTURE_CHANGE] *= 2.0
    tot = sum(pr.values())
    pr = {k: v / tot for k, v in pr.items()}
    return floor_unknown(pr)


def floor_unknown(pr: Mapping) -> dict:
    """UNKNOWN_CAUSE never drops below UNKNOWN_FLOOR: an explanation cannot be certain by having priced out ignorance."""
    pr = dict(pr)
    u = pr.get(BreakCause.UNKNOWN_CAUSE, 0.0)
    if u < UNKNOWN_FLOOR:
        rest = 1.0 - u
        scale = (1.0 - UNKNOWN_FLOOR) / rest if rest > 0 else 0.0
        pr = {k: v * scale for k, v in pr.items()}
        pr[BreakCause.UNKNOWN_CAUSE] = UNKNOWN_FLOOR
    return pr


def assess_hypotheses(F: AnalysisFrame, event: BreakEvent, shared: Sequence[Mapping] = (), seed: int = 0) -> tuple:
    """Run every cause test, pay for multiplicity across them with Holm, and turn the evidence into a posterior. Returns a tuple
    of BreakHypothesis ordered by posterior. A cause can be 'supported' here without being CLAIMED: claiming needs Q1-Q6."""
    pops = _pops(F)
    ev = {
        BreakCause.REGIME_CHANGE: dimension_support(F, BreakCause.REGIME_CHANGE, pops, seed),
        BreakCause.MARKET_STRUCTURE_CHANGE: test_market_structure(F),
        BreakCause.LIQUIDITY_CHANGE: dimension_support(F, BreakCause.LIQUIDITY_CHANGE, pops, seed + 1),
        BreakCause.SECTOR_SHIFT: test_sector_shift(F, pops, seed + 2),
        BreakCause.EVENT_ENVIRONMENT: test_event_environment(F),
        BreakCause.PATTERN_CROWDING: test_pattern_crowding(F),
        BreakCause.FEATURE_RELATIONSHIP_CHANGED: test_feature_relationship(F, pops),
        BreakCause.MEASUREMENT_ERROR: test_measurement_error(F, pops, seed + 3),
        BreakCause.SAMPLING_ARTIFACT: test_sampling_artifact(F, pops, seed + 4),
        BreakCause.RANDOM_VARIATION: test_random_variation(F, seed),
        BreakCause.HIDDEN_INTERACTION: test_hidden_interaction(F),
        BreakCause.EXTERNAL_SHOCK: test_external_shock(F, shared),
    }
    tested = [c for c, e in ev.items() if e.supported is not None and c != BreakCause.RANDOM_VARIATION]
    adj = dict(zip(tested, PR.holm([ev[c].p for c in tested]))) if tested else {}
    for c in tested:
        e = ev[c]
        if e.supported and adj[c] > F.P["alpha"] and c not in (BreakCause.EXTERNAL_SHOCK, BreakCause.SAMPLING_ARTIFACT):
            ev[c] = dataclasses.replace(e, supported=False, strength=0.0, p=float(adj[c]),
                                        detail=e.detail + f"; does not survive Holm across {len(tested)} causes (adjusted p={adj[c]:.3f})")
    others_supported = any(e.supported for e in ev.values())
    ev[BreakCause.UNKNOWN_CAUSE] = _ev(BreakCause.UNKNOWN_CAUSE, not others_supported, 0.5 if not others_supported else 0.0, 1.0, 0,
                                       "no other cause is supported by the data" if not others_supported else "another cause has support")
    priors = hypothesis_priors(F, event, shared)
    post = {}
    for c in BreakCause:
        e = ev[c]
        lr = 1.0 + 4.0 * e.strength if e.supported else (0.6 if e.supported is False else 1.0)
        post[c] = priors[c] * lr
    z = sum(post.values())
    post = floor_unknown({c: v / z for c, v in post.items()})
    out = [BreakHypothesis(c, safe_text(f"the break of a pattern is due to: {CAUSE_LABEL[c.value]}"), float(priors[c]), float(post[c]), ev[c],
                           DISCRIMINATOR[c]) for c in BreakCause]
    return tuple(sorted(out, key=lambda h: -h.posterior))


DISCRIMINATOR = {
    BreakCause.REGIME_CHANGE: "a regime/trend/breadth/macro column is displaced in failed rows and leads the onset",
    BreakCause.MARKET_STRUCTURE_CHANGE: "a market column shifts at the onset and stays shifted",
    BreakCause.LIQUIDITY_CHANGE: "a liquidity column differs between working and failed rows",
    BreakCause.SECTOR_SHIFT: "the sector / stock-type mix differs between working and failed rows",
    BreakCause.EVENT_ENVIRONMENT: "onsets recur at the same point of the calendar",
    BreakCause.PATTERN_CROWDING: "the outcome declines gradually into the break while neighbouring patterns converge",
    BreakCause.FEATURE_RELATIONSHIP_CHANGED: "a feature's link to the outcome changes while the feature's level does not",
    BreakCause.MEASUREMENT_ERROR: "missingness/quality shifts, or winsorising removes the break",
    BreakCause.SAMPLING_ARTIFACT: "sample size or concentration shifts, or the break is only a few rows long",
    BreakCause.RANDOM_VARIATION: "a stable pattern would show a drop this deep somewhere in this history",
    BreakCause.HIDDEN_INTERACTION: "a product term of two columns explains the outcome beyond their main effects",
    BreakCause.EXTERNAL_SHOCK: "the step is abrupt, has no precursor, and other patterns break with it",
    BreakCause.UNKNOWN_CAUSE: "no other discriminator is supported"}


def peer_shared_events(peers: Mapping[str, BD.ItemSeries], as_of, item: BD.ItemSeries | None = None, cfg=None, gap: int = 4, min_items: int = 3) -> list:
    """Breaks of different items that start within `gap` rows of each other (a market-wide cause rather than an item-specific one)."""
    B = _bd_cfg(_cfg(cfg))
    pool = dict(peers)
    if item is not None:
        pool[item.item_id] = item
    eps = {}
    for k, it in pool.items():
        e, _ = BD.find_episodes(it, as_of, B)
        eps[k] = e
    return BD.shared_break_events(eps, gap=gap, min_items=min_items)


def peer_transfer(F: AnalysisFrame, rule: ColumnRule, peers: Mapping[str, BD.ItemSeries], cfg=None) -> dict:
    """Does the rule generalise to other patterns' breaks? Each peer that has the same column gets the rule with its threshold
    mapped by percentile (scales differ), scored by AUC against that peer's own hindsight labels. Informational: it never turns
    a failed question into a pass, but an explanation that fails on every peer is flagged item-specific."""
    P = _cfg(cfg)
    B = _bd_cfg(P)
    pct = float((F.X[rule.column].values.astype(float) * rule.direction <= rule.threshold).mean())
    rows = []
    for k, it in sorted(peers.items()):
        if rule.column not in {c.name for c in it.columns}:
            continue
        try:
            PF = build_frame(it, F.as_of, P)
        except (BreakResearchError, FirewallBreach, BD.BreakInputError):
            continue
        if rule.column not in PF.X.columns:
            continue
        s = rule.direction * PF.X[rule.column].values.astype(float)
        y_pos, y_neg = eval_labels(PF, np.arange(PF.m))
        sel = np.flatnonzero(y_pos | y_neg)
        if y_pos.sum() < 8 or y_neg.sum() < 8:
            continue
        rows.append({"peer": anon(k), "auc": PR.auc(s[sel], y_pos[sel]), "n_pos": int(y_pos.sum()), "n_neg": int(y_neg.sum())})
    if not rows:
        return {"n_peers": 0, "median_auc": float("nan"), "item_specific": None, "percentile": pct}
    aucs = np.array([r["auc"] for r in rows])
    return {"n_peers": len(rows), "median_auc": float(np.median(aucs)), "share_above_chance": float((aucs > 0.55).mean()),
            "item_specific": bool((aucs <= 0.55).all()), "percentile": pct, "peers": rows}


# ------------------------------------------------------------------------------------------------- claiming a cause

DIMENSION_CAUSE = {"regime": BreakCause.REGIME_CHANGE, "trend": BreakCause.REGIME_CHANGE, "breadth": BreakCause.REGIME_CHANGE,
                   "macro": BreakCause.REGIME_CHANGE, "volatility": BreakCause.MARKET_STRUCTURE_CHANGE,
                   "liquidity": BreakCause.LIQUIDITY_CHANGE, "sector_composition": BreakCause.SECTOR_SHIFT,
                   "stock_type": BreakCause.SECTOR_SHIFT, "timing": BreakCause.EVENT_ENVIRONMENT,
                   "missingness": BreakCause.MEASUREMENT_ERROR, "data_quality": BreakCause.MEASUREMENT_ERROR,
                   "sample_size": BreakCause.SAMPLING_ARTIFACT, "concentration": BreakCause.SAMPLING_ARTIFACT,
                   "interaction_structure": BreakCause.HIDDEN_INTERACTION, "feature_distribution": BreakCause.FEATURE_RELATIONSHIP_CHANGED,
                   "neighbouring_patterns": BreakCause.PATTERN_CROWDING}


def claim_cause(hyps: Sequence[BreakHypothesis], rule: ColumnRule | None, design: BD.Design) -> BreakCause:
    """Only called for an EXPLAINED investigation. The cause named is the supported hypothesis that agrees with the dimension of
    the validated predictor; failing that, the best-supported hypothesis; failing that, UNKNOWN_CAUSE (a validated predictor with
    no supported mechanism is 'predictive but not understood', which is a legitimate and useful state)."""
    supported = [h for h in hyps if h.evidence is not None and h.evidence.supported and h.cause not in (BreakCause.UNKNOWN_CAUSE, BreakCause.RANDOM_VARIATION)]
    if rule is not None and rule.column in design.tags:
        want = DIMENSION_CAUSE.get(design.tags[rule.column].dimension)
        for h in supported:
            if h.cause == want:
                return h.cause
    return supported[0].cause if supported else BreakCause.UNKNOWN_CAUSE


# ------------------------------------------------------------------------------------------------- costs, values, priority

def estimate_minutes(m: int, k: int, P: dict, stage: Stage) -> float:
    """CPU-minute estimate for running up to `stage` on an item of m rows and k candidate columns. The permutation and random-explanation
    counts dominate; the numbers are estimates used to schedule, replaced by measurements in the value accounting."""
    cells = float(max(m, 1) * max(k, 1))
    mult = {Stage.CHEAP_SCREEN: P["n_perm"] * 1.5, Stage.STRONGER_TESTS: P["n_perm"] * 1.5 + 25 * P["wf_step"],
            Stage.CROSS_YEAR: P["n_perm"] * 3.0 + 25 * P["wf_step"], Stage.FRESH_HOLDOUT: P["n_perm"] * 3.0 + 25 * P["wf_step"] + P["n_random"] * 4.0,
            Stage.INTEGRATION: P["n_perm"] * 3.0 + 25 * P["wf_step"] + P["n_random"] * 4.0}[stage]
    return float(cells * mult * P["cpu_min_per_cell"])


def entropy_share(post: Mapping) -> float:
    """Normalised Shannon entropy of a posterior over causes (1 = maximally undecided): how much there is to learn."""
    p = np.array([v for v in post.values() if v > 0], float)
    return float(-(p * np.log(p)).sum() / math.log(len(BreakCause))) if len(p) else 0.0


def value_vector(event: BreakEvent, stake: float, priors: Mapping, n_peers: int, minutes: float, similar_open: int,
                 n_cols: int, n_rows: int) -> ExperimentValue:
    """The section-19 value estimate of investigating this break BEFORE doing it. loss_reduction_value is the severity-weighted
    stake (what an early warning could have saved); redundancy rises with investigations of the same item already open;
    overfit risk with the number of columns searched per row of evidence."""
    sev = min(abs(event.severity) / 4.0, 1.0)
    info = entropy_share(priors)
    return ExperimentValue(information_gain=info, decision_value=float(stake * (0.4 + 0.6 * sev)), uncertainty_reduction=info * 0.7,
                           transfer_potential=float(min(n_peers / 10.0, 1.0)), failure_reduction_value=float(sev),
                           loss_reduction_value=float(stake * sev), compute_cost=float(minutes),
                           overfit_risk=float(min(n_cols / max(n_rows, 1), 1.0)), redundancy=float(min(0.25 * similar_open, 1.0)))


def priority_of(val: ExperimentValue, budget_minutes: float) -> float:
    """Benefit per unit of compute, discounted by redundancy and overfit risk. Unestimated fields count as 0, never as a guess."""
    g = lambda x: 0.0 if x is None else float(x)
    benefit = 0.30 * g(val.decision_value) + 0.25 * g(val.information_gain) + 0.20 * g(val.loss_reduction_value) \
        + 0.10 * g(val.transfer_potential) + 0.10 * g(val.failure_reduction_value) + 0.05 * g(val.uncertainty_reduction)
    disc = (1.0 - 0.5 * g(val.redundancy)) * (1.0 - 0.5 * g(val.overfit_risk))
    return float(benefit * disc / (1.0 + g(val.compute_cost) / max(budget_minutes, 1e-9)))


# ------------------------------------------------------------------------------------------------- outputs handed to the research queue

@dataclasses.dataclass(frozen=True)
class GateProposal:
    """A candidate gate for the research queue. It is a proposal, not a decision: it needs the fresh-holdout replication, the
    champion/challenger process, and then the curator, before any trader sees an effect of it."""
    proposal_id: str
    inv_id: str
    item_key: str
    rule: ColumnRule
    cause: BreakCause
    evidence: tuple                      # brief() of the six results
    matured_at: str
    replicated: bool = False
    effect: str = "ABSTENTION"
    status: str = "PROPOSED"

    def validate(self) -> list[str]:
        errs = list(self.rule.validate())
        if len(self.evidence) != len(QUESTION_ORDER):
            errs.append("a gate proposal needs all six results")
        if self.status not in ("PROPOSED", "REPLICATED", "WITHDRAWN"):
            errs.append(f"unknown proposal status {self.status!r}")
        return errs


@dataclasses.dataclass(frozen=True)
class StepReport:
    now: str
    opened: tuple = ()
    reopened: tuple = ()
    ran: tuple = ()
    deferred: tuple = ()
    refused: tuple = ()
    holdouts: tuple = ()
    questions: tuple = ()
    signals: tuple = ()
    records: tuple = ()
    proposals: tuple = ()
    minutes: float = 0.0
    unknown_share: float = float("nan")
    counts: Mapping = dataclasses.field(default_factory=dict)
    withheld: tuple = ()


def main_question(inv: Investigation, created_real: str) -> ResearchQuestion:
    """The research question every break becomes: why did it stop, and can the stop be seen coming? Identity-free by construction
    (the pattern is named by a hash, no date or year is in the text)."""
    text = safe_text(f"Why did pattern {inv.label} stop working, and can a variable known beforehand predict such a stop, "
                     f"transfer to unseen data, and reduce losses when used as a gate?")
    return ResearchQuestion.make(text, "break", Problem.LOSS_AVOIDANCE, created_real, inv.event.detect_at,
                                 "all six section-14 questions pass, including the random-explanation control and a fresh holdout",
                                 "any question fails with adequate power, or the cause stays UNKNOWN after the evidence stops growing",
                                 expected=inv.value)


def follow_up_questions(inv: Investigation, created_real: str) -> list:
    """What to do next, from what failed. A not-testable question asks for more evidence, a failed one for a different
    angle, a supported-but-unproven cause for a cross-pattern replication. UNKNOWN stays UNKNOWN: no follow-up invents a cause."""
    parent = main_question(inv, created_real).question_id
    out = []
    for r in inv.results:
        if r.passed:
            continue
        if r.outcome == Outcome.NOT_TESTABLE:
            text = f"Pattern {inv.label}: what additional matured evidence would let the question '{QUESTION_TEXT[r.qid]}' be answered? ({r.why})"
            src, succ, fail = "break_evidence", "the question becomes testable and is answered", "still untestable after the next reopen threshold"
        else:
            text = f"Pattern {inv.label}: is the negative answer to '{QUESTION_TEXT[r.qid]}' stable, or does another predictor family answer it?"
            src, succ, fail = "break_negative", "an independent test agrees with the negative", "a different predictor family passes the question"
        out.append(ResearchQuestion.make(safe_text(text), src, Problem.LOSS_AVOIDANCE, created_real, inv.event.detect_at, succ, fail, parents=(parent,)))
        break                                 # the first unmet question is the next thing to learn
    for h in inv.hypotheses[:3]:
        e = h.evidence
        if e is not None and e.supported and h.cause not in (BreakCause.UNKNOWN_CAUSE,) and inv.verdict != Verdict.EXPLAINED:
            text = f"Pattern {inv.label}: {CAUSE_LABEL[h.cause.value]} is consistent with the data; does the same cause explain breaks of other patterns?"
            out.append(ResearchQuestion.make(safe_text(text), "break_hypothesis", Problem.CONSISTENCY, created_real, inv.event.detect_at,
                                             "the cause is supported in at least two more independent patterns", "it is supported in none of them", parents=(parent,)))
    return out


def break_signal(inv: Investigation, stake: float = 0.5) -> RP.Signal:
    """The Signal that lets the existing research-priority queue schedule this break. An EXPLAINED break is marked explained so
    it stops generating work; UNKNOWN stays open (an unexplained failure is still a failure)."""
    cause = FAILURE_CAUSE_OF[inv.cause].value if inv.verdict == Verdict.EXPLAINED else ""
    return RP.make_signal(RP.SignalKind.FAILURE, inv.event.detect_at, f"break_{inv.item_key}", min(abs(inv.event.severity) / 4.0, 1.0),
                          subsystem="SELECTION", cause=cause, explained=inv.verdict == Verdict.EXPLAINED, stake=float(stake), n_obs=inv.event.n_matured,
                          evidence=tuple(r.qid.value for r in inv.results if r.passed))


def _years_of(item: BD.ItemSeries, m: int) -> tuple:
    idx = item.frame.index[:m]
    return tuple(sorted({int(y) for y in idx.year})) if isinstance(idx, pd.DatetimeIndex) else ()


def matured_record(inv: Investigation, item: BD.ItemSeries, now, created_real: str, seed: int) -> tuple:
    """(MaturedRecord, filed_years) for a conclusion. The payload is identity-free (hash id, column names, numbers); the years the
    evidence came from are kept beside it, on the trusted side, for the same-year release guard."""
    m = inv.evidence_rows
    matured_at = str(item.frame.index[max(m - 1, 0)])[:10]
    require_past(matured_at, now, f"conclusion for {inv.label}")
    payload = {"item": inv.item_key, "verdict": inv.verdict.value if inv.verdict else "PENDING", "cause": inv.cause.value,
               "rule": inv.rule.describe() if inv.rule else None, "results": [r.brief() for r in inv.results],
               "posterior": {h.cause.value: round(h.posterior, 4) for h in inv.hypotheses}, "state": inv.state.value, "stage": inv.stage.value,
               "n_rows": m, "code_hash": current_code_hash()}
    for r in payload["results"]:
        safe_text(r["why"])
    safe_text(str(payload["rule"] or ""))
    prov = Provenance(created_real=created_real, learned_at=matured_at, code_hash=current_code_hash(), experiment_id=inv.inv_id,
                      run_id=stable_hash([inv.inv_id, m], 8), seed=seed, outcomes_seen_through=matured_at)
    rec = MaturedRecord(stable_hash([inv.inv_id, m, payload["verdict"]], 14), matured_at, payload, prov)
    return rec, _years_of(item, m)


def release_guard(filed_years: Iterable[int], replaying_years: Iterable[int]) -> None:
    """The same-year rerun leak (mapping rule 27): research filed under a real year may not be released while that year is being
    replayed in disguise, because the replay would then be trading against knowledge of its own future."""
    clash = sorted(set(filed_years) & set(replaying_years))
    if clash:
        raise FirewallBreach(f"same-year rerun leak: research built on year(s) {clash} may not be released while those year(s) are replayed")


# ------------------------------------------------------------------------------------------------- state and the ledger

@dataclasses.dataclass
class BreakResearchState:
    investigations: dict = dataclasses.field(default_factory=dict)
    proposals: dict = dataclasses.field(default_factory=dict)
    records: dict = dataclasses.field(default_factory=dict)
    filed_years: dict = dataclasses.field(default_factory=dict)
    ledger: list = dataclasses.field(default_factory=list)
    refused: list = dataclasses.field(default_factory=list)
    steps: list = dataclasses.field(default_factory=list)
    minutes_spent: float = 0.0
    last_now: str | None = None

    def log(self, when, kind: str, payload: Mapping) -> dict:
        """Append-only, hash-chained: an entry cannot be rewritten without breaking every later hash."""
        prev = self.ledger[-1]["chain"] if self.ledger else "genesis"
        row = {"when": str(when), "kind": kind, "payload": dict(payload), "prev": prev}
        row["chain"] = stable_hash([prev, row["when"], kind, row["payload"]], 20)
        self.ledger.append(row)
        return row

    def verify(self) -> list[str]:
        errs, prev = [], "genesis"
        for i, r in enumerate(self.ledger):
            if r["prev"] != prev or r["chain"] != stable_hash([prev, r["when"], r["kind"], r["payload"]], 20):
                errs.append(f"ledger row {i}: chain broken")
            prev = r["chain"]
        return errs

    def unknown_share(self) -> float:
        """Share of concluded investigations whose answer is UNKNOWN or INSUFFICIENT_EVIDENCE (section 33: track how often we admit it)."""
        done = [i for i in self.investigations.values() if i.verdict in (Verdict.EXPLAINED, Verdict.UNKNOWN, Verdict.INSUFFICIENT_EVIDENCE)]
        return sum(i.verdict != Verdict.EXPLAINED for i in done) / len(done) if done else float("nan")

    def counts(self) -> dict:
        c: dict = {}
        for i in self.investigations.values():
            c[i.state.value] = c.get(i.state.value, 0) + 1
        return c


def new_state() -> BreakResearchState:
    return BreakResearchState()


def _record(state: BreakResearchState, when, inv: Investigation, action: str) -> None:
    state.log(when, action, {"inv": inv.inv_id, "item": inv.item_key, "state": inv.state.value, "stage": inv.stage.value,
                             "verdict": inv.verdict.value if inv.verdict else None, "passed": [r.qid.value for r in inv.results if r.passed],
                             "rows": inv.evidence_rows})


def open_investigation(event: BreakEvent, item_id: str, now, trigger: Trigger, value: ExperimentValue, P: dict) -> Investigation:
    errs = event.validate()
    if errs:
        raise BreakResearchError("; ".join(errs))
    return Investigation("INV" + event.event_id, item_id, event.item_key, event, ResearchState.QUEUED, Stage.CHEAP_SCREEN, trigger,
                         priority_of(value, P["minutes_budget"]), value, created_at=str(now), updated_at=str(now))


# ------------------------------------------------------------------------------------------------- investigating one break

def state_after(results: Sequence[QuestionResult]) -> ResearchState:
    """Research-branch state from the answers so far: all passed -> VALIDATING (fresh holdout pending); any adequately powered
    failure -> FAILED; otherwise the data could not answer and the branch sleeps (DORMANT) until it grows."""
    if len(results) == len(QUESTION_ORDER) and all(r.passed for r in results):
        return ResearchState.VALIDATING
    if any(r.outcome == Outcome.FAILED for r in results):
        return ResearchState.FAILED
    if results and results[0].passed:
        return ResearchState.PROMISING
    return ResearchState.DORMANT


def investigate(inv: Investigation, item: BD.ItemSeries, now, cfg=None, peers: Mapping[str, BD.ItemSeries] | None = None, seed: int = 0) -> Investigation:
    """Run one investigation to the end of what the data support: hypotheses, then the six questions up the escalation ladder."""
    P = _cfg(cfg)
    F = build_frame(item, now, P)
    shared = peer_shared_events(peers, now, item, P) if peers else []
    hyps = assess_hypotheses(F, inv.event, shared, seed)
    results, rule = run_ladder(F, seed, stop_early=True)
    verdict, know, gate, stmt = adjudicate(results)
    if not F.eps and inv.event.source != "health":
        verdict = Verdict.NO_BREAK
    cause = claim_cause(hyps, rule, F.design) if verdict == Verdict.EXPLAINED else BreakCause.UNKNOWN_CAUSE
    stage = STAGE_OF[results[-1].qid] if results else Stage.CHEAP_SCREEN
    if verdict == Verdict.EXPLAINED:
        stage = Stage.INTEGRATION
    extras = {"knowability": know.value, "gate_verdict": gate.value, "statement": stmt, "n_hypotheses_supported": sum(1 for h in hyps if h.evidence and h.evidence.supported)}
    if rule is not None and peers:
        extras["peer_transfer"] = peer_transfer(F, rule, peers, P)
    minutes = estimate_minutes(F.m, len(F.cols), P, stage)
    return dataclasses.replace(inv, state=state_after(results), stage=stage, hypotheses=hyps, results=tuple(results), verdict=verdict, cause=cause,
                               rule=rule if verdict == Verdict.EXPLAINED else None, updated_at=str(now), evidence_rows=F.m,
                               frozen_row=F.m if verdict == Verdict.EXPLAINED else -1, attempts=inv.attempts + 1, cost_minutes=inv.cost_minutes + minutes,
                               note=stmt, extras={**dict(inv.extras), **extras})


def fresh_holdout(inv: Investigation, item: BD.ItemSeries, now, cfg=None, seed: int = 0) -> Investigation:
    """Replicate an EXPLAINED investigation on rows that arrived AFTER the rule was frozen: Q2, Q4 and Q5 with the frozen rule and
    nothing refitted. Too few new rows leaves it VALIDATING; a pass replicates it; a failure withdraws the explanation."""
    P = _cfg(cfg)
    if inv.verdict != Verdict.EXPLAINED or inv.rule is None or inv.frozen_row < 0:
        raise BreakResearchError(f"{inv.label}: only an EXPLAINED investigation with a frozen rule can be held out")
    F = build_frame(item, now, P)
    new_rows = F.m - inv.frozen_row - P["embargo"]
    if new_rows < P["holdout_min_rows"]:
        return dataclasses.replace(inv, updated_at=str(now), note=f"waiting for fresh data: {max(new_rows, 0)} of {P['holdout_min_rows']} rows")
    H = dataclasses.replace(F, n_disc=inv.frozen_row, P={**P, "min_confirm": P["holdout_min_rows"], "min_oos_pos": 4})
    res = (q2_transfers_oos(H, inv.rule, seed), q4_gating_improves(H, inv.rule, seed), q5_gating_reduces_losses(H, inv.rule, seed))
    if all(r.passed for r in res):
        return dataclasses.replace(inv, state=ResearchState.ESCALATED, stage=Stage.INTEGRATION, updated_at=str(now), holdout=tuple(res),
                                   note=f"replicated on {new_rows} fresh rows", evidence_rows=F.m)
    if any(r.outcome == Outcome.FAILED for r in res):
        return dataclasses.replace(inv, state=ResearchState.FAILED, verdict=Verdict.UNKNOWN, cause=BreakCause.UNKNOWN_CAUSE, rule=None, updated_at=str(now),
                                   holdout=tuple(res), note="failed on fresh data: explanation withdrawn", evidence_rows=F.m)
    return dataclasses.replace(inv, updated_at=str(now), holdout=tuple(res), note="fresh data could not settle it yet", evidence_rows=F.m)


# ------------------------------------------------------------------------------------------------- the scheduler

def refuse(state: BreakResearchState, now, item_id: str, why: str) -> dict:
    row = {"item": anon(item_id), "when": str(now), "why": why}
    state.refused.append(row)
    state.log(now, "refused", row)
    return row


def health_triggers(rows: Iterable[Mapping] | None, now) -> dict:
    """item id -> weight for items a health monitor calls BROKEN/CONTRADICTED/DEGRADING. Uses research_priority.signals_from_health,
    so a health row dated at or after `now` raises FirewallBreach there."""
    if not rows:
        return {}
    out = {}
    for s in RP.signals_from_health(list(rows), now):
        if s.kind in (RP.SignalKind.FAILURE, RP.SignalKind.CONTRADICTION, RP.SignalKind.WEAK_PATTERN):
            out[s.subject] = max(out.get(s.subject, 0.0), s.magnitude)
    return out


def step(state: BreakResearchState, now, items: Mapping[str, BD.ItemSeries], *, health: Iterable[Mapping] | None = None,
         peers: Mapping[str, BD.ItemSeries] | None = None, stakes: Mapping[str, float] | None = None, cfg=None, seed: int = 0,
         created_real: str | None = None, replaying_years: Iterable[int] = ()) -> StepReport:
    """One scheduling pass at `now` (the research loop calls this daily/weekly; it is idempotent for the same evidence):
      1 refuse leaking or invalid items (fail closed, logged)      2 open an investigation per newly detected break / BROKEN health row
      3 reopen sleeping ones that gained enough new evidence       4 run the best within the compute budget (escalation ladder)
      5 fresh-holdout any explained one that gained enough rows    6 emit questions, queue signals, matured records, gate proposals
    Nothing is returned to the trader: records leave only through `release`."""
    P = _cfg(cfg)
    B = _bd_cfg(P)
    created = created_real or str(now)
    if state.last_now is not None and as_date(now) < as_date(state.last_now):
        raise FirewallBreach(f"step at {now} is earlier than the previous step at {state.last_now}: research time only moves forward")
    stakes = dict(stakes or {})
    hmap = health_triggers(health, now)
    opened, reopened, deferred, refused, holdouts = [], [], [], [], []
    live = {}
    for iid, item in sorted(items.items()):
        try:
            item.require_valid()
            BD.assert_no_leak(item, now, B)
        except (FirewallBreach, BD.BreakInputError) as err:
            refused.append(refuse(state, now, iid, str(err)))
            for inv in [i for i in state.investigations.values() if i.item_id == iid]:
                state.investigations[inv.inv_id] = dataclasses.replace(inv, state=ResearchState.CANCELLED, verdict=Verdict.DATA_FAILURE, updated_at=str(now), note="input failed a firewall")
                state.log(now, "cancelled", {"inv": inv.inv_id, "why": "input failed a firewall"})
            continue
        live[iid] = item
    shared = peer_shared_events(peers, now, None, P) if peers else []
    for iid, item in live.items():
        events = detect_events(item, now, P)
        if not events and hmap.get(iid, 0.0) > 0:
            he = health_event(item, now, P)
            events = [he] if he else []
        n_open = sum(1 for i in state.investigations.values() if i.item_id == iid and i.state in (ResearchState.QUEUED, ResearchState.EXPLORING))
        for ev in events:
            inv_id = "INV" + ev.event_id
            if inv_id in state.investigations:
                continue
            trig = Trigger("health" if ev.source == "health" else "episode", float(hmap.get(iid, 0.6)), f"severity {ev.severity:.2f}")
            k = len(item.columns)
            val = value_vector(ev, float(stakes.get(iid, 0.5)), BASE_PRIOR, len(peers or {}), estimate_minutes(ev.n_matured, k, P, Stage.INTEGRATION), n_open, k, ev.n_matured)
            inv = open_investigation(ev, iid, now, trig, val, P)
            state.investigations[inv.inv_id] = inv
            _record(state, now, inv, "opened")
            opened.append(inv.inv_id)
            n_open += 1
    for inv in list(state.investigations.values()):
        item = live.get(inv.item_id)
        if item is None or inv.state != ResearchState.DORMANT and inv.state != ResearchState.PROMISING:
            continue
        gained = item.n_matured(now) - inv.evidence_rows
        if gained >= P["reopen_rows"] and inv.attempts < P["max_attempts"]:
            state.investigations[inv.inv_id] = dataclasses.replace(inv, state=ResearchState.QUEUED, updated_at=str(now))
            _record(state, now, state.investigations[inv.inv_id], "reopened")
            reopened.append(inv.inv_id)
    queue = sorted((i for i in state.investigations.values() if i.state == ResearchState.QUEUED and i.item_id in live),
                   key=lambda i: (-i.priority, i.inv_id))
    ran, spent = [], 0.0
    for inv in queue:
        est = inv.value.compute_cost or 0.0
        if len(ran) >= P["max_per_step"] or (ran and spent + est > P["minutes_budget"]) or inv.priority < P["min_priority"]:
            deferred.append(inv.inv_id)
            continue
        run = dataclasses.replace(inv, state=ResearchState.EXPLORING)
        try:
            done = investigate(run, live[inv.item_id], now, P, peers, seed + len(ran))
        except (FirewallBreach, BreakResearchError, BD.BreakInputError) as err:
            refused.append(refuse(state, now, inv.item_id, str(err)))
            state.investigations[inv.inv_id] = dataclasses.replace(inv, state=ResearchState.CANCELLED, verdict=Verdict.DATA_FAILURE, note=str(err), updated_at=str(now))
            continue
        errs = done.validate()
        if errs:
            raise BreakResearchError(f"{done.label}: internal inconsistency: {'; '.join(errs)}")
        state.investigations[inv.inv_id] = done
        spent += done.cost_minutes - inv.cost_minutes
        _record(state, now, done, "investigated")
        ran.append(inv.inv_id)
    for inv in list(state.investigations.values()):
        if inv.state == ResearchState.VALIDATING and inv.item_id in live and inv.inv_id not in ran:
            new = fresh_holdout(inv, live[inv.item_id], now, P, seed)
            if new is not inv and (new.state != inv.state or new.holdout):
                state.investigations[inv.inv_id] = new
                _record(state, now, new, "holdout")
                holdouts.append(inv.inv_id)
    touched = set(ran) | set(holdouts) | set(opened)
    replaying_years = tuple(replaying_years)
    questions, signals, records, proposals, withheld = [], [], [], [], []
    for iid_ in sorted(touched):
        inv = state.investigations[iid_]
        questions.append(main_question(inv, created))
        questions.extend(follow_up_questions(inv, created))
        signals.append(break_signal(inv, float(stakes.get(inv.item_id, 0.5))))
        if inv.results and inv.item_id in live:
            rec, years = matured_record(inv, live[inv.item_id], now, created, seed)
            state.records[rec.record_id] = rec
            state.filed_years[rec.record_id] = years
            if set(years) & set(replaying_years):
                withheld.append(rec.record_id)                # kept on the trusted side, not offered while its own years are replayed
            else:
                records.append(rec)
        if inv.verdict == Verdict.EXPLAINED and inv.rule is not None:
            pid = stable_hash([inv.inv_id, inv.rule.key], 12)
            replicated = inv.state == ResearchState.ESCALATED
            prop = GateProposal(pid, inv.inv_id, inv.item_key, inv.rule, inv.cause, tuple(r.brief() for r in inv.results),
                                str(live[inv.item_id].frame.index[max(inv.evidence_rows - 1, 0)])[:10], replicated,
                                status="REPLICATED" if replicated else "PROPOSED")
            state.proposals[pid] = prop
            proposals.append(prop)
        elif inv.state == ResearchState.FAILED:
            for p in [p for p in state.proposals.values() if p.inv_id == inv.inv_id and p.status != "WITHDRAWN"]:
                state.proposals[p.proposal_id] = dataclasses.replace(p, status="WITHDRAWN")
    state.minutes_spent += spent
    state.last_now = str(now)
    rep = StepReport(str(now), tuple(opened), tuple(reopened), tuple(ran), tuple(deferred), tuple(r["item"] for r in refused), tuple(holdouts),
                     tuple(questions), tuple(signals), tuple(records), tuple(proposals), spent, state.unknown_share(), state.counts(), tuple(withheld))
    state.steps.append({"now": str(now), "opened": len(opened), "ran": len(ran), "minutes": spent})
    return rep


def release(state: BreakResearchState, record_id: str, now, replaying_years: Iterable[int] = ()) -> Mapping:
    """The only exit of a conclusion: the same-year guard first, then MaturedRecord.gate(now), which fails closed unless the evidence
    matured strictly before now."""
    if record_id not in state.records:
        raise BreakResearchError(f"unknown record {record_id}")
    release_guard(state.filed_years.get(record_id, ()), replaying_years)
    return state.records[record_id].gate(now)


# ------------------------------------------------------------------------------------------------- diagnostics and audits

def events_prefix_stable(item: BD.ItemSeries, early, late, cfg=None) -> list[str]:
    """No-look-ahead audit: every break knowable at `early` must appear unchanged (same onset and detection) when the history is
    extended to `late`. A detector that revises the past with the future fails this. Returns the violations."""
    a = {(e.onset_at, e.detect_at) for e in detect_events(item, early, cfg)}
    b = {(e.onset_at, e.detect_at) for e in detect_events(item, late, cfg)}
    return sorted(f"break {o}/{d} known at {early} changed or vanished by {late}" for o, d in a - b)


def agreement_with_engine(inv: Investigation, item: BD.ItemSeries, now, cfg=None, seed: int = 0) -> dict:
    """Cross-check against the one break engine (`explain_break`): does its validated condition use the same column or the same
    dimension as this module's rule? Agreement from two different searches is evidence; disagreement is reported, not hidden."""
    P = _cfg(cfg)
    ex = BD.explain_break(item, now, _bd_cfg(P), seed)
    engine_cols = tuple(t.column for t in ex.condition.terms) if ex.condition is not None else ()
    engine_dims = tuple(ex.condition.dimensions) if ex.condition is not None else ()
    mine = inv.rule.column if inv.rule else None
    dim_mine = None
    if mine is not None:
        F = build_frame(item, now, P)
        dim_mine = F.design.tags[mine].dimension if mine in F.design.tags else None
    return {"engine_status": ex.status, "engine_columns": engine_cols, "engine_dimensions": engine_dims, "my_column": mine, "my_dimension": dim_mine,
            "same_column": bool(mine and mine in engine_cols), "same_dimension": bool(dim_mine and dim_mine in engine_dims),
            "both_explain": bool(ex.explained and inv.verdict == Verdict.EXPLAINED), "both_unknown": bool(not ex.explained and inv.verdict != Verdict.EXPLAINED)}


def recurrence_profile(events: Sequence[BreakEvent], m: int) -> dict:
    """How breaks arrive: gaps between onsets, their coefficient of variation (regular < 1 < clustered), the share of history spent
    broken, and a dispersion test of onset counts in equal blocks (Poisson index of dispersion against chi-square)."""
    ons = sorted(e.onset for e in events)
    if len(ons) < 2:
        return {"n": len(ons), "gaps": [], "cv": float("nan"), "share_broken": float(sum(e.length for e in events) / max(m, 1)), "dispersion_p": float("nan")}
    gaps = np.diff(ons)
    cv = float(gaps.std(ddof=1) / gaps.mean()) if len(gaps) > 1 and gaps.mean() > 0 else float("nan")
    nb = max(min(len(ons), 8), 2)
    counts, _ = np.histogram(ons, bins=nb, range=(0, max(m, 1)))
    mean = counts.mean()
    disp = float(((counts - mean) ** 2).sum() / mean) if mean > 0 else 0.0
    p = float(sps.chi2.sf(disp, nb - 1)) if mean > 0 else float("nan")
    return {"n": len(ons), "gaps": [int(g) for g in gaps], "cv": cv, "share_broken": float(min(sum(e.length for e in events) / max(m, 1), 1.0)),
            "dispersion_index": disp, "dispersion_p": p, "pattern": "clustered" if cv == cv and cv > 1.2 else ("regular" if cv == cv and cv < 0.6 else "irregular")}


def hazard_by_age(eps: Sequence[BD.Episode], m: int, bins: Sequence[int] = (0, 13, 26, 52, 104, 10_000)) -> pd.DataFrame:
    """Empirical hazard of a new break by rows since the previous one ended (or the start). A flat hazard means breaks are memoryless."""
    tsb = time_since_break(eps, m)
    onsets = {e.onset for e in eps}
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (tsb >= lo) & (tsb < hi)
        n = int(sel.sum())
        k = int(sum(1 for o in onsets if 0 <= o < m and sel[o]))
        rows.append({"age_from": lo, "age_to": hi, "rows_at_risk": n, "onsets": k, "hazard": (k / n) if n else float("nan")})
    return pd.DataFrame(rows)


def power_summary(F: AnalysisFrame) -> dict:
    """What this item's history could and could not have shown: the numbers behind every UNKNOWN."""
    pre, pos, neg, eps_d = training_labels(F, F.n_disc, F.eps)
    cpos, cneg = eval_labels(F, F.conf_rows)
    return {"n_matured": F.m, "n_columns": len(F.cols), "breaks_total": len(F.bk), "breaks_discovery": len(eps_d),
            "breaks_after": sum(e.onset >= F.n_disc + F.P["embargo"] for e in F.bk), "pre_rows": int(pre.sum()), "ref_rows": int(neg.sum()),
            "detectable_smd_q1": BD.detectable_smd(int(neg.sum()), int(pre.sum()), len(F.cols), F.P["alpha"]),
            "oos_pos": int(cpos.sum()), "oos_neg": int(cneg.sum()),
            "min_detectable_auc_q2": min_detectable_auc(int(cpos.sum()), int(cneg.sum()), 2.0, F.P["oos_alpha"])}


def null_calibration(items: Sequence[BD.ItemSeries], as_of, cfg=None, seed: int = 0, shifts: Sequence[int] = (37, 83)) -> dict:
    """False-EXPLAINED rate: run the whole ladder on placebo copies of real items (context columns circularly shifted against
    the outcomes, so any link is destroyed while every outcome property is kept). This is the empirical cost of the search; a
    healthy pipeline explains almost none. Also reports how far the ladder got and where placebos die."""
    P = _cfg(cfg)
    n = ex = 0
    reached = {q: 0 for q in QUESTION_ORDER}
    for i, item in enumerate(items):
        for sh in shifts:
            pl = BD.shifted_placebo(item, sh)
            try:
                F = build_frame(pl, as_of, P)
            except (BreakResearchError, FirewallBreach):
                continue
            res, _ = run_ladder(F, seed + 31 * i + sh, stop_early=True)
            n += 1
            for r in res:
                reached[r.qid] += 1
            ex += adjudicate(res)[0] == Verdict.EXPLAINED
    return {"placebos": n, "explained": ex, "false_explained_rate": (ex / n) if n else float("nan"), "reached": {q.value: c for q, c in reached.items()}}


def planted_recall(items: Sequence[BD.ItemSeries], as_of, cfg=None, seed: int = 0) -> dict:
    """The mirror of the null calibration: on items where a real predictor was planted, how often does the ladder EXPLAIN it?"""
    P = _cfg(cfg)
    n = ex = unk = 0
    for i, item in enumerate(items):
        try:
            F = build_frame(item, as_of, P)
        except (BreakResearchError, FirewallBreach):
            continue
        res, _ = run_ladder(F, seed + i, stop_early=True)
        v = adjudicate(res)[0]
        n += 1
        ex += v == Verdict.EXPLAINED
        unk += v in (Verdict.UNKNOWN, Verdict.INSUFFICIENT_EVIDENCE)
    return {"items": n, "explained": ex, "unknown": unk, "recall": (ex / n) if n else float("nan")}


def question_funnel(state: BreakResearchState) -> pd.DataFrame:
    """Where investigations die: for each of the six questions, how many were asked and how they came out. The bottleneck is the
    first question whose PASSED share collapses, which tells the scheduler whether to collect more breaks or better columns."""
    rows = []
    for q in QUESTION_ORDER:
        rs = [i.result(q) for i in state.investigations.values() if i.result(q) is not None]
        c = {o: sum(r.outcome == o for r in rs) for o in Outcome}
        rows.append({"question": q.value, "asked": len(rs), "passed": c[Outcome.PASSED], "failed": c[Outcome.FAILED], "not_testable": c[Outcome.NOT_TESTABLE],
                     "pass_share": (c[Outcome.PASSED] / len(rs)) if rs else float("nan")})
    return pd.DataFrame(rows)


def cause_table(state: BreakResearchState) -> pd.DataFrame:
    """Per cause: how often it was the leading posterior, how often supported, how often actually CLAIMED (only after six passes)."""
    rows = {c: {"leading": 0, "supported": 0, "claimed": 0, "untestable": 0} for c in BreakCause}
    for inv in state.investigations.values():
        if not inv.hypotheses:
            continue
        rows[inv.hypotheses[0].cause]["leading"] += 1
        for h in inv.hypotheses:
            if h.evidence is not None and h.evidence.supported:
                rows[h.cause]["supported"] += 1
            if h.evidence is not None and h.evidence.supported is None:
                rows[h.cause]["untestable"] += 1
        if inv.verdict == Verdict.EXPLAINED:
            rows[inv.cause]["claimed"] += 1
    return pd.DataFrame([{"cause": c.value, **v} for c, v in rows.items()])


def compute_accounting(state: BreakResearchState) -> dict:
    """Section 19/20 in miniature: minutes spent against questions answered, and the minutes spent on branches that ended
    FAILED or DORMANT without an explanation (candidates for the stop-wasting mechanism)."""
    invs = list(state.investigations.values())
    answered = sum(1 for i in invs for r in i.results if r.outcome != Outcome.NOT_TESTABLE)
    total = sum(i.cost_minutes for i in invs)
    wasted = sum(i.cost_minutes for i in invs if i.state in (ResearchState.FAILED, ResearchState.DORMANT, ResearchState.RETIRED) and i.attempts >= 2)
    return {"investigations": len(invs), "minutes": total, "questions_answered": answered, "minutes_per_answer": (total / answered) if answered else float("nan"),
            "minutes_on_stalled_branches": wasted, "unknown_share": state.unknown_share()}


def retire_exhausted(state: BreakResearchState, now, cfg=None) -> list:
    """Stop wasting compute: a branch that has used its attempts without new evidence is RETIRED (kept, never deleted, and
    reopened only by a fresh break). Returns the retired investigation ids."""
    P = _cfg(cfg)
    out = []
    for inv in list(state.investigations.values()):
        if inv.state in (ResearchState.DORMANT, ResearchState.FAILED) and inv.attempts >= P["max_attempts"]:
            state.investigations[inv.inv_id] = dataclasses.replace(inv, state=ResearchState.RETIRED, updated_at=str(now),
                                                                   note=inv.note + f" | retired after {inv.attempts} attempts")
            state.log(now, "retired", {"inv": inv.inv_id, "attempts": inv.attempts})
            out.append(inv.inv_id)
    return out


def investigation_table(state: BreakResearchState) -> pd.DataFrame:
    rows = []
    for i in sorted(state.investigations.values(), key=lambda x: x.inv_id):
        rows.append({"inv": i.inv_id, "item": i.item_key, "state": i.state.value, "stage": i.stage.value, "verdict": i.verdict.value if i.verdict else None,
                     "cause": i.cause.value, "passed": i.n_passed(), "priority": i.priority, "minutes": i.cost_minutes, "attempts": i.attempts,
                     "severity": i.event.severity, "rows": i.evidence_rows, "rule": i.rule.describe() if i.rule else None})
    return pd.DataFrame(rows)


def render_report(state: BreakResearchState, top: int = 10) -> str:
    """Plain-text account, honest about what is unknown."""
    lines = [f"Pattern-break research [{LABEL}]", f"investigations: {len(state.investigations)}  states: {state.counts()}",
             f"unknown share: {state.unknown_share():.0%}" if state.unknown_share() == state.unknown_share() else "unknown share: n/a (nothing concluded)"]
    for _, r in question_funnel(state).iterrows():
        lines.append(f"  {r['question']:<34} asked {int(r['asked']):>3}  passed {int(r['passed']):>3}  failed {int(r['failed']):>3}  not testable {int(r['not_testable']):>3}")
    for i in sorted(state.investigations.values(), key=lambda x: -x.priority)[:top]:
        lines.append(f"- {i.label} {i.state.value}/{i.verdict.value if i.verdict else 'PENDING'}: {i.note[:160]}")
    for p in state.proposals.values():
        lines.append(f"* proposal {p.proposal_id} [{p.status}] {p.rule.describe()} (cause: {p.cause.value})")
    acc = compute_accounting(state)
    lines.append(f"compute: {acc['minutes']:.2f} min, {acc['questions_answered']} questions answered, {acc['minutes_on_stalled_branches']:.2f} min on stalled branches")
    if state.verify():
        lines.append("LEDGER INTEGRITY FAILURE: " + "; ".join(state.verify()))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- run helpers

def run_item(item: BD.ItemSeries, as_of, cfg=None, seed: int = 0, peers: Mapping[str, BD.ItemSeries] | None = None) -> Investigation:
    """One-shot: detect the item's most recent break and investigate it without a scheduler (for scripts and tests)."""
    P = _cfg(cfg)
    events = detect_events(item, as_of, P) or ([e] if (e := health_event(item, as_of, P)) else [])
    if not events:
        raise BreakResearchError(f"{item.item_id}: no break and too little history at {as_of}")
    ev = events[-1]
    m = item.n_matured(as_of)
    val = value_vector(ev, 0.5, BASE_PRIOR, len(peers or {}), estimate_minutes(m, len(item.columns), P, Stage.INTEGRATION), 0, len(item.columns), m)
    inv = open_investigation(ev, item.item_id, as_of, Trigger("episode", 0.6), val, P)
    return investigate(inv, item, as_of, P, peers, seed)


def run_history(items: Mapping[str, BD.ItemSeries], nows: Sequence[Any], cfg=None, seed: int = 0, peers=None) -> tuple:
    """Replay the scheduler over successive `nows` (oldest first). Returns (state, reports). The state after each step depends only on
    items visible at that `now`: this is what the reachability wave calls."""
    state = new_state()
    reports = []
    for k, now in enumerate(nows):
        reports.append(step(state, now, items, peers=peers, cfg=cfg, seed=seed + k))
    return state, reports
