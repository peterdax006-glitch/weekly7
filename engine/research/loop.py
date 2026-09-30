"""The permanent autonomous research loop (contract C66 sections 3, 47 and 50; also 2, 15, 18-20, 29-31, 37-41; canon C63, C64,
C66, C67). IMPLEMENTED - NOT VALIDATED (C63: code, unit and integration tests on planted worlds only; no real-data run).

This is the TRUSTED-SIDE research brain (RESEARCH_MAPPING rule 27: research that uses hindsight never runs inside the blind learner).
One cycle walks the section-3 loop, each stage calling the REAL module's public entry:

  OBSERVE            feed panel checks; observer.step; autopsy.step
  UPDATE KNOWLEDGE   experiments.step (stale / unfollowed); firewall.run_day (matured knowledge -> live state for the trader)
  EVALUATE           two_stage.run_day + evaluate (section 35); volatility_lab.step; direction_lab.step; frontier.step; symmetry.step
  SURPRISES          feature screen (volatility_lab.oriented_scan); cross_section.step; regimes.step; multiscale.step
  FAILURES           decision losses; winners_losers.step; loss_pipeline.step
  MISSED WINNERS     missed.step (missed winners and missed losers); knowability.step; unknown_cause.step; counterfactual.step
  PATTERN BREAKS     released-knowledge break check; break_research.step
  QUESTIONS          discovery / interactions / precursors (R21 episodes); targets.run_day; research_graph.step; questions.generate
  HYPOTHESES         hypothesis_tree.step (a tree per question; results move beliefs, dead branches redirect)
  INFORMATION GAIN   priority.step (value per compute, learned and validated priority model, controller objective weights)
  ALLOCATE COMPUTE   controller.step (section 51); diversity.step; compute_manager.step (escalation ladder, RAM >= 2.5 GB rule)
  RUN EXPERIMENT     executor: engine.learning.compute workers (inline / threads / isolated processes); a long job never blocks
  VALIDATE           compute.reconcile (hash, seed, spec, code); fail-closed maturity check of every result
  CONTROLS           shuffled-label control per result; leakage audit of suspiciously strong results
  UPDATE KNOWLEDGE   ladder record_result; experiment memory; tree results; value_accounting.step; science_memory.step; research
                     graph; replication.step; quality_gate.step; decision_bridge.step; knowledge filed for the firewall
  UPDATE PRIORITIES  waste.step; brain_health.step; meta_research.step; failed_lab.step; scorecard.step; stale-hypothesis and
                     answered-question cancellation
  SWEEPS             C67 always-on sweeps (R21 episodes/precursors, volatility lab, discovery) as low-priority background units
  REPORT             cycle report and the section-47 chain audit

Honesty rules made mechanical: a stage whose module is absent is SKIPPED_MISSING_MODULE and a stage whose inputs the feed does not
supply is SKIPPED_NO_INPUT (both reported, never passed); a FirewallBreach is REFUSED_LEAK at the stage that caught it; any other
exception is FAILED with its message and the cycle continues (a crash in one module must not stop research).

Operation: checkpoints after every stage (engine.learning.checkpoints: atomic, self-verifying, code-hash stamped; resume fails closed
on stale code unless allowed), crash recovery (in-flight jobs are reconciled from the compute ledger or re-queued), experiment
isolation (engine.learning.compute: private attempt folders, reconciliation, duplicate refusal), cancellation / escalation / demotion
(compute_manager ladder + answered-question cancellation), duplicate detection (question merge, compute-ledger keys, a design key per
feature x rung x data), stale-hypothesis detection, lineage (event -> question -> item -> branch -> job -> result -> memory -> new
question -> knowledge -> decision) and reports. Public entry: `step(state, runtime)` runs one cycle; `run(...)` runs many."""
from __future__ import annotations

import concurrent.futures as cf
import copyreg
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import re
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from engine.learning import checkpoints as CK
from engine.learning import compute as C
from engine.learning.core import current_code_hash
from engine.research import controller as CT
from engine.research import two_stage as TS
from engine.research.core import (ExperimentValue, FirewallBreach, MaturedRecord, Namespace, Problem, Provenance, ResearchQuestion,
                                  Stage, _StrEnum, as_date, require_past, stable_hash)

LABEL = "IMPLEMENTED - NOT VALIDATED"
NAMESPACE = Namespace.MATURED_RESEARCH
TASK_DIR_ENV = "W7_RESEARCH_TASK_DIR"
_FEATURE_RE = re.compile(r"feature ([a-z][a-z0-9_]*)")


# ================================================================================================================ vocabulary
class StageStatus(_StrEnum):
    OK = "OK"
    SKIPPED_MISSING_MODULE = "SKIPPED_MISSING_MODULE"
    SKIPPED_NO_INPUT = "SKIPPED_NO_INPUT"
    SKIPPED_DISABLED = "SKIPPED_DISABLED"
    SKIPPED_CADENCE = "SKIPPED_CADENCE"
    REFUSED_LEAK = "REFUSED_LEAK"
    FAILED = "FAILED"


class LoopPhase(_StrEnum):                  # the section-3 loop, in order
    OBSERVE = "OBSERVE"
    UPDATE_KNOWLEDGE = "UPDATE_KNOWLEDGE"
    EVALUATE = "EVALUATE"
    SURPRISES = "IDENTIFY_SURPRISES"
    FAILURES = "IDENTIFY_FAILURES"
    MISSED = "IDENTIFY_MISSED_WINNERS_AND_LOSERS"
    BREAKS = "IDENTIFY_PATTERN_BREAKS"
    QUESTIONS = "GENERATE_QUESTIONS"
    HYPOTHESES = "GENERATE_HYPOTHESES"
    GAIN = "ESTIMATE_INFORMATION_GAIN"
    ALLOCATE = "ALLOCATE_COMPUTE"
    RUN = "RUN_EXPERIMENT"
    VALIDATE = "VALIDATE"
    CONTROLS = "COMPARE_AGAINST_CONTROLS"
    LEARN = "UPDATE_KNOWLEDGE_FROM_RESULTS"
    PRIORITIES = "UPDATE_RESEARCH_PRIORITIES"
    SWEEPS = "ALWAYS_ON_SWEEPS"
    REPORT = "REPORT"


class JobState(_StrEnum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    HARVESTED = "HARVESTED"
    FAILED = "FAILED"
    REFUSED = "REFUSED"
    CONTROL_FAILURE = "CONTROL_FAILURE"


class MissingModule(ImportError):
    """A research module the stage needs is not (yet) in the tree."""


class NoInput(Exception):
    """The stage has nothing to work on this cycle (the reason is reported)."""


# ================================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """How the loop's own experiments are run (RUN EXPERIMENT). Units: per-date rank AUC for volatility and direction; loss share for
    loss questions."""
    orient_frac: float = 0.3              # earliest share of dates used only to orient the feature (its sign)
    n_blocks: int = 8                     # rung-1 tests = contiguous date blocks
    min_dates: int = 8
    min_names: int = 8                    # per date, both classes present and at least this many names
    mover_frac: float = 0.20              # predicted-mover universe for direction and loss experiments
    fresh_min_dates: int = 4              # dates never used by the branch that a fresh holdout needs
    control_z: float = 3.0                # a shuffled-label control beyond this |t| means the design leaks (~0.3% by chance per rung)
    horizon_days: int = 5
    base_features: tuple = ("lv20", "shock1")
    min_effect: float = 0.05              # smallest useful effect in these units (per-date AUC 0.55): the ladder's power target

    def validate(self) -> list[str]:
        errs = []
        if not 0.1 <= self.orient_frac <= 0.6 or self.n_blocks < 2 or self.min_dates < 3 or self.min_names < 4:
            errs.append("orient_frac in [0.1,0.6], n_blocks >= 2, min_dates >= 3, min_names >= 4 required")
        if not 0 < self.mover_frac < 0.5 or self.fresh_min_dates < 2 or self.control_z <= 0 or not 0 < self.min_effect < 0.5:
            errs.append("mover_frac in (0,0.5), fresh_min_dates >= 2, control_z > 0 and min_effect in (0,0.5) required")
        return errs


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    run_id: str = "research"
    seed: int = 0
    budget_cpu_min: float = 240.0
    period_cpu_min: float = 1200.0
    mode: str = "inline"                  # inline | thread | process
    max_workers: int = 2
    free_gb: float | None = None          # None = measure (psutil); tests pass a number
    min_free_gb: float = 2.5
    checkpoint: str = "stage"             # stage | cycle | off
    keep_checkpoints: int = 6
    code_hash: str = ""                   # "" = engine.learning.core.current_code_hash()
    allow_code_change: bool = False
    max_new_questions: int = 12
    screen_top: int = 6
    screen_t: float = 2.0
    stale_days: int = 90
    job_timeout_s: float = 900.0
    sweep_units: int = 1
    report_keep: int = 60
    disabled: tuple = ()
    cadence: Mapping[str, int] = dataclasses.field(default_factory=lambda: {
        "evaluate.volatility_lab": 13, "evaluate.direction_lab": 13, "surprises.feature_screen": 4, "questions.discovery": 4,
        "questions.interactions": 8, "priorities.meta_research": 4, "priorities.scorecard": 4})
    experiment: ExperimentConfig = ExperimentConfig()
    two_stage: TS.TwoStageConfig = TS.TwoStageConfig()
    firewall_secret: str = "research-loop"

    def validate(self) -> list[str]:
        errs = list(self.experiment.validate()) + list(self.two_stage.validate())
        if not self.run_id or Path(self.run_id).name != self.run_id:
            errs.append("run_id must be a plain identifier")
        if self.mode not in ("inline", "thread", "process"):
            errs.append("mode must be inline, thread or process")
        if self.checkpoint not in ("stage", "cycle", "off"):
            errs.append("checkpoint must be stage, cycle or off")
        if self.budget_cpu_min <= 0 or self.period_cpu_min <= 0 or self.max_workers < 1:
            errs.append("budgets and max_workers must be positive")
        unknown = sorted(set(self.disabled) - {s.name for s in STAGES})
        if unknown:
            errs.append(f"unknown stages disabled: {unknown}")
        return errs


# ================================================================================================================ records
@dataclasses.dataclass(frozen=True)
class StageRecord:
    cycle: int
    stage: str
    phase: str
    status: StageStatus
    reason: str
    n_in: int
    n_out: int
    seconds: float
    module: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d


@dataclasses.dataclass
class JobRecord:
    """One experiment job, persisted with the loop state so a restart can reconcile it with the compute ledger."""
    key: str
    branch_id: str
    question_id: str
    item_id: str
    problem: str
    feature: str
    stage: str
    cutoff: str
    seen_through: str
    submitted_at: str
    cycle: int
    task: dict
    cost: float = 0.0
    state: JobState = JobState.SUBMITTED
    result: dict | None = None
    error: str = ""
    attempts: int = 0
    harvested_at: str = ""


@dataclasses.dataclass
class Observation:
    """What the feed offers at `now`: the matured research panel (outcomes ended strictly before now; hindsight allowed, trusted
    side), the point-in-time decision day, and optional per-stage inputs for modules whose data the panel cannot supply."""
    now: str
    matured: pd.DataFrame
    today: pd.DataFrame
    extras: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    data_hash: str = ""

    @property
    def evidence_through(self) -> str:
        if len(self.matured) and "end" in self.matured:
            return str(pd.to_datetime(self.matured["end"]).max().date())
        return ""


class Feed(Protocol):
    def dates(self) -> Sequence[str]: ...

    def observe(self, now) -> Observation: ...


class FrameFeed:
    """A feed over one research frame in the volatility-lab layout (index (date, ticker); features; outcome columns touch/up/absmove/
    close/tday/end). The clock is the frame's own decision dates. Deterministic: observe(now) is a pure function of (frame, now), so a
    resumed loop sees exactly what the killed one saw."""

    def __init__(self, frame: pd.DataFrame, dates: Sequence | None = None, extras: Callable[[str, pd.DataFrame], Mapping] | None = None):
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
            raise ValueError("FrameFeed needs a (date, ticker) MultiIndex")
        if "end" not in frame:
            raise ValueError("FrameFeed needs the 'end' column (when each outcome matured)")
        self.frame = frame.sort_index()
        all_dates = sorted(pd.unique(self.frame.index.get_level_values(0)))
        self._dates = [str(pd.Timestamp(d).date()) for d in (dates if dates is not None else all_dates)]
        self._extras = extras

    def dates(self) -> Sequence[str]:
        return list(self._dates)

    def observe(self, now) -> Observation:
        n = pd.Timestamp(as_date(now))
        ends = pd.to_datetime(self.frame["end"])
        mat = self.frame[np.asarray(ends < n)]
        pit = TS.point_in_time_state(self.frame, now)
        dh = stable_hash({"n": len(mat), "through": str(ends[ends < n].max()) if len(mat) else "", "cols": list(mat.columns)}, 16)
        ex = dict(self._extras(str(as_date(now)), mat)) if self._extras else {}
        return Observation(str(as_date(now)), mat, pit.frame, ex, dh)


def world_feed(source: str = "planted", years: Sequence[int] = (), sample: int | None = 300, plant: Mapping | None = None,
               feed: Mapping | None = None):
    """The W02 data feed (engine.research.feeds.WorldFeed): bars (+ events / insider / macro / sectors) -> EVERY stage's inputs per
    simulated day, streamed year by year. 'planted' = the rich planted world with known mechanisms; 'real' = the real-cache adapter
    (C63: code only, not yet run on the real files). Use `feed.sweeps()` as the loop's always-on sweeps."""
    from engine.research import feeds as FD
    fc = FD.FeedConfig(**dict(feed or {}))
    if source == "planted":
        return FD.planted_feed(FD.PlantConfig(**dict(plant or {})), fc)
    if source == "real":
        if not years:
            raise ValueError("the real-cache feed needs the years to stream")
        return FD.real_cache_feed(years, fc, sample=sample)
    raise ValueError(f"unknown feed source {source!r} (planted | real)")


# ================================================================================================================ lineage
@dataclasses.dataclass
class Lineage:
    """Research lineage as a small typed graph. Node ids are namespaced by kind; edges carry the relation. It answers the section-47
    question 'did a question lead to a priority, an experiment, a result, memory and a new question?' by walking edges, not by
    trusting counters."""
    nodes: dict = dataclasses.field(default_factory=dict)       # nid -> {"kind", "cycle", "at", **attrs}
    edges: list = dataclasses.field(default_factory=list)       # (src, dst, relation)

    def add(self, kind: str, ident: str, cycle: int, at: str, **attrs) -> str:
        nid = f"{kind}:{ident}"
        if nid not in self.nodes:
            self.nodes[nid] = {"kind": kind, "cycle": cycle, "at": at, **attrs}
        else:
            self.nodes[nid].update({k: v for k, v in attrs.items() if v is not None})
        return nid

    def link(self, src: str, dst: str, relation: str) -> None:
        if src in self.nodes and dst in self.nodes and (src, dst, relation) not in self._edge_set():
            self.edges.append((src, dst, relation))

    def _edge_set(self) -> set:
        return set(self.edges)

    def out(self, nid: str, relation: str | None = None) -> list[str]:
        return [d for s, d, r in self.edges if s == nid and (relation is None or r == relation)]

    def kinds(self) -> dict:
        out: dict = {}
        for v in self.nodes.values():
            out[v["kind"]] = out.get(v["kind"], 0) + 1
        return dict(sorted(out.items()))

    def section47_chains(self) -> list[list[str]]:
        """Complete chains QUESTION -> ITEM -> BRANCH -> JOB -> RESULT -> MEMORY and RESULT -> EVENT -> QUESTION (a new one)."""
        chains = []
        for q in (n for n, v in self.nodes.items() if v["kind"] == "QUESTION"):
            for it in self.out(q, "prioritised_as"):
                for br in self.out(it, "allocated_to"):
                    for job in self.out(br, "ran"):
                        for res in self.out(job, "produced"):
                            mem = self.out(res, "remembered_as")
                            for ev in self.out(res, "raised"):
                                for q2 in self.out(ev, "asked_as"):
                                    if q2 != q and mem:
                                        chains.append([q, it, br, job, res, mem[0], ev, q2])
        return chains

    def knowledge_chains(self) -> list[list[str]]:
        """RESULT -> KNOWLEDGE -> RELEASE -> DECISION: knowledge that reached a decision through the firewall."""
        out = []
        for k in (n for n, v in self.nodes.items() if v["kind"] == "KNOWLEDGE"):
            for rel in self.out(k, "released_in"):
                for dec in self.out(rel, "used_by"):
                    out.append([k, rel, dec])
        return out


# ================================================================================================================ experiment science
@dataclasses.dataclass(frozen=True)
class ExperimentTask:
    """One rung of one branch, fully specified (JSON-serialisable: it travels to isolated workers)."""
    key: str
    branch_id: str
    problem: str
    feature: str
    stage: str
    cutoff: str                       # newest outcome date the task may use (strictly before the launch date)
    seen_through: str                 # newest decision date earlier rungs of this branch used ('' = none): fresh holdouts start after it
    seed: int
    cfg: dict
    repeat: int = 0                   # how many times this rung was already repeated (a repeat is run LARGER)
    n_planned: int = 0                # sample the ladder asked for after an underpowered run (0 = the rung's default)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def resolve_feature(text: str, problem: Problem, columns: Iterable[str]) -> str:
    """The derived feature a question is about: named in its subject/text ('feature lv20 ...'), else the problem's default probe.
    Only features derivable from the panel qualify; '' = none (the task is then refused, not guessed)."""
    from engine.research import vol_hypotheses as VH
    cols = set(columns)
    for m in _FEATURE_RE.finditer(text or ""):
        f = m.group(1)
        if f in VH.DERIVED and not VH.missing_columns((f,), cols):
            return f
    default = {Problem.DIRECTION: ("rel_r5", "rel_r20"), Problem.LOSS_AVOIDANCE: ("gap_ratio", "shock1"),
               Problem.COVERAGE: ("rel_r5",)}.get(Problem.parse(problem), ("lv20", "vol_ratio_short"))
    for f in default:
        if f in VH.DERIVED and not VH.missing_columns((f,), cols):
            return f
    return ""


def _per_date_auc(score: np.ndarray, y: np.ndarray, codes: np.ndarray, min_names: int) -> tuple[np.ndarray, np.ndarray]:
    """(date codes, AUC per date) over dates with both classes and at least min_names finite rows."""
    from engine.pattern_movers import auc as rank_auc
    ds, vals = [], []
    for c in np.unique(codes):
        m = (codes == c) & np.isfinite(score) & np.isfinite(y)
        if m.sum() < min_names:
            continue
        yy = y[m] >= 0.5
        if yy.all() or not yy.any():
            continue
        a = rank_auc(score[m], yy)
        if np.isfinite(a):
            ds.append(c)
            vals.append(a - 0.5)
    return np.array(ds, int), np.array(vals, float)


def design_kind(problem) -> str:
    """Which experiment design a problem uses: two questions of different problems that map to the same design, feature, rung and
    data are ONE experiment (the duplicate detector keys on this, not on the problem label)."""
    prob = Problem.parse(problem)
    if prob in (Problem.DIRECTION, Problem.COVERAGE):
        return "direction_auc"
    if prob == Problem.LOSS_AVOIDANCE:
        return "loss_auc"
    return "volatility_auc"


def _mover_mask(F: pd.DataFrame, frac: float) -> np.ndarray:
    """Predicted movers for direction/loss experiments: a two-stage walk-forward score when the panel carries one (p_TS), else the
    rank of own-volatility (lv20) per date - a point-in-time proxy, never the realised movers."""
    from engine.research import vol_hypotheses as VH
    s = F["p_TS"] if "p_TS" in F else VH.derive(F, ("lv20",))["lv20"]
    r = s.groupby(level=0).rank(ascending=False, pct=True)
    return np.asarray(r <= frac)


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    if len(x) == 1:
        return float(x[0]), float(abs(x[0]) + 1e-3)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def _design(F: pd.DataFrame, task: ExperimentTask, ec: ExperimentConfig) -> tuple:
    """Rows, score, label and per-row date codes for the task's problem. Returns (G, score, y, codes, dates, kind)."""
    from engine.research import vol_hypotheses as VH
    prob = Problem.parse(task.problem)
    if prob in (Problem.DIRECTION, Problem.COVERAGE):
        G = F[_mover_mask(F, ec.mover_frac)]
        y, kind = G["up"].to_numpy(float), "direction_auc"
    elif prob == Problem.LOSS_AVOIDANCE:
        G = F[_mover_mask(F, ec.mover_frac)]
        y, kind = (G["close"].to_numpy(float) < 0).astype(float), "loss_auc"
    else:
        G = F
        y, kind = G["touch"].to_numpy(float), "volatility_auc"
    s = VH.derive(G, (task.feature,))[task.feature].to_numpy(float)
    dates = G.index.get_level_values(0)
    codes, uniq = pd.factorize(dates, sort=True)
    return G, s, y, codes, np.asarray(uniq), kind


def run_task(task: ExperimentTask, frame: pd.DataFrame) -> dict:
    """Execute one rung. Everything is computed on rows whose outcome ended on/before task.cutoff; the feature's sign is fixed on the
    earliest `orient_frac` of dates and never re-chosen on the evaluation dates. Returns a JSON-serialisable result with a shuffled-
    label control computed by the SAME code path."""
    ec = ExperimentConfig(**task.cfg)
    cut = pd.Timestamp(as_date(task.cutoff))
    F = frame[np.asarray(pd.to_datetime(frame["end"]) <= cut)]
    base = {"key": task.key, "branch_id": task.branch_id, "stage": task.stage, "feature": task.feature, "problem": task.problem,
            "cutoff": task.cutoff}
    if len(F) == 0 or not task.feature:
        return {**base, "ok": False, "why": "no matured rows or no feature", "n_obs": 0}
    G, s, y, codes, uniq, kind = _design(F, task, ec)
    nd = len(uniq)
    if nd < ec.min_dates:
        return {**base, "ok": False, "why": f"{nd} dates < {ec.min_dates}", "n_obs": 0}
    k_or = max(2, int(nd * ec.orient_frac))
    rng = np.random.default_rng(task.seed)
    o = codes < k_or
    d0, a0 = _per_date_auc(s[o], y[o], codes[o], max(4, ec.min_names // 2))
    sign = 1.0 if (len(a0) == 0 or a0.mean() >= 0) else -1.0
    reserve = nd - ec.fresh_min_dates                  # rungs 1-3 never see the newest fresh_min_dates dates: rung 4's holdout
    ev_mask = (codes >= k_or) & (codes < reserve)
    seen_code = -1
    if task.seen_through:
        seen_code = int(np.searchsorted(uniq, np.datetime64(pd.Timestamp(as_date(task.seen_through))), side="right")) - 1
    stage = Stage(task.stage)
    fresh = False
    if stage == Stage.FRESH_HOLDOUT or stage == Stage.INTEGRATION:
        ev_mask = codes > max(seen_code, k_or - 1)
        fresh = bool(task.seen_through) and int(len(np.unique(codes[ev_mask]))) >= ec.fresh_min_dates
    elif stage == Stage.CHEAP_SCREEN:
        base_n = max(ec.min_dates, (nd - k_or) // 2)
        want = max(base_n * (2 ** task.repeat), task.n_planned)
        ev_mask = (codes >= k_or) & (codes < min(k_or + want, reserve))
    mn = ec.min_names if kind == "volatility_auc" else max(4, ec.min_names // 2)
    ds, a = _per_date_auc(sign * s[ev_mask], y[ev_mask], codes[ev_mask], mn)
    y_sh = y.copy()
    for c in np.unique(codes[ev_mask]):
        ix = np.flatnonzero(codes == c)
        y_sh[ix] = rng.permutation(y[ix])
    _, ac = _per_date_auc(sign * s[ev_mask], y_sh[ev_mask], codes[ev_mask], mn)
    eff, se = _mean_se(a)
    ceff, cse = _mean_se(ac)
    out = {**base, "ok": len(a) >= ec.min_dates or (stage in (Stage.FRESH_HOLDOUT, Stage.INTEGRATION) and len(a) >= 2), "kind": kind,
           "sign": sign, "effect": eff, "se": se, "n_obs": int(len(a)), "control_effect": ceff, "control_se": cse,
           "n_tests": 1, "n_positive": int(eff > 0), "n_units": 1, "units_positive": int(eff > 0), "replications": 0, "fresh": fresh,
           "data_through": str(pd.to_datetime(G["end"]).max().date()),
           "seen_through": str(pd.Timestamp(uniq[int(ds.max())]).date()) if len(ds) else task.seen_through, "n_rows": int(ev_mask.sum())}
    if not out["ok"]:
        out["why"] = f"{len(a)} evaluable dates"
    if stage == Stage.CHEAP_SCREEN and len(a):
        blocks = np.array_split(a, min(ec.n_blocks, len(a)))
        out["n_tests"] = len(blocks)
        out["n_positive"] = int(sum(b.mean() > 0 for b in blocks))
    elif stage == Stage.STRONGER_TESTS:
        out.update(_units(G, sign * s, y, codes, ev_mask, ec, "sector"))
    elif stage == Stage.CROSS_YEAR:
        out.update(_units(G, sign * s, y, codes, ev_mask, ec, "year"))
    elif stage == Stage.INTEGRATION:
        out.update(_integration(F, task, ec, seen_code, uniq))
    return out


def _units(G: pd.DataFrame, s: np.ndarray, y: np.ndarray, codes: np.ndarray, mask: np.ndarray, ec: ExperimentConfig, by: str) -> dict:
    """Per-context (sector) or per-period (year; quarters when fewer than three years) consistency. `replications` counts the later
    periods whose effect agrees with the first, which is what 'independent replication' means for a time-ordered unit."""
    if by == "sector" and "sector" in G:
        lab = G["sector"].astype(str).to_numpy()
    else:
        d = pd.to_datetime(G.index.get_level_values(0))
        lab = d.year.astype(str).to_numpy()
        if len(np.unique(lab[mask])) < 3:
            lab = (d.year.astype(str) + "Q" + d.quarter.astype(str)).to_numpy()
    effs = []
    for u in sorted(np.unique(lab[mask])):
        m = mask & (lab == u)
        _, a = _per_date_auc(s[m], y[m], codes[m], max(4, ec.min_names // 2))
        if len(a) >= 2:
            effs.append(float(a.mean()))
    if not effs:
        return {"n_units": 1, "units_positive": 0, "replications": 0}
    first = np.sign(effs[0])
    return {"n_units": len(effs), "units_positive": int(sum(e > 0 for e in effs)),
            "replications": int(sum(np.sign(e) == first and first > 0 for e in effs[1:])), "unit_effects": effs}


def _integration(F: pd.DataFrame, task: ExperimentTask, ec: ExperimentConfig, seen_code: int, uniq: np.ndarray) -> dict:
    """Rung 5: does adding the feature to the base volatility model improve fresh-date ranking? Base and base+feature logistic models
    (engine.research.vol_hypotheses) are fitted on dates the branch already used and scored on the fresh dates."""
    from engine.research import vol_hypotheses as VH
    feats = tuple(f for f in ec.base_features if f != task.feature and not VH.missing_columns((f,), F.columns))
    if not feats or seen_code < 2:
        return {"integration_delta": None, "integration_se": None, "why_integration": "no base features or no prior rung"}
    d = F.index.get_level_values(0)
    split = pd.Timestamp(uniq[seen_code])
    tr = F[np.asarray(pd.to_datetime(F["end"]) <= split)]
    te = F[np.asarray(d > split)]
    if len(tr) < 200 or len(te) == 0:
        return {"integration_delta": None, "integration_se": None, "why_integration": "too few rows"}
    fc = VH.FitConfig(min_rows=100, min_events=10)
    hb = VH.Hypothesis("IB", "base", "own volatility and last shock", feats, origin="loop")
    hx = VH.Hypothesis("IX", "base+feature", "base plus the branch's feature", feats + (task.feature,), origin="loop")
    fb, fx = VH.fit_hypothesis(hb, tr, split + pd.Timedelta(days=1), fc), VH.fit_hypothesis(hx, tr, split + pd.Timedelta(days=1), fc)
    if not (fb.ok and fx.ok):
        return {"integration_delta": None, "integration_se": None, "why_integration": f"fit failed: {fb.reason or fx.reason}"}
    y = te["touch"].to_numpy(float)
    codes = pd.factorize(te.index.get_level_values(0), sort=True)[0]
    _, ab = _per_date_auc(fb.predict(te)[0], y, codes, ec.min_names)
    _, ax = _per_date_auc(fx.predict(te)[0], y, codes, ec.min_names)
    n = min(len(ab), len(ax))
    if n < 2:
        return {"integration_delta": None, "integration_se": None, "why_integration": "too few fresh dates"}
    diff = ax[:n] - ab[:n]
    m, se = _mean_se(diff)
    return {"integration_delta": m, "integration_se": se}


def evidence_from_result(res: Mapping[str, Any], cost: float):
    """The compute_manager StageEvidence of a finished rung."""
    from engine.research import compute_manager as CM
    return CM.StageEvidence(Stage(res["stage"]), int(res.get("n_obs", 0)), float(res.get("effect", 0.0)), float(res.get("se", 0.0)),
                            max(1, int(res.get("n_tests", 1))), int(res.get("n_positive", 0)), max(1, int(res.get("n_units", 1))),
                            int(res.get("units_positive", 0)), int(res.get("replications", 0)), bool(res.get("fresh", False)),
                            str(res.get("data_through", "")), float(cost),
                            res.get("integration_delta"), res.get("integration_se"), 0.0)


# ================================================================================================================ executor
_FRAMES: dict[str, pd.DataFrame] = {}            # transient: data_hash -> matured frame, for inline/thread workers


def _worker_fn(task: ExperimentTask, frame: pd.DataFrame) -> Callable:
    def fn(spec: C.ExperimentSpec, ctx: C.WorkerContext) -> dict:
        ctx.beat()
        return run_task(task, frame)
    return fn


def task_worker(spec: C.ExperimentSpec, ctx: C.WorkerContext) -> dict:
    """The isolated-process worker: the task file and the matured panel are read from the task directory the loop wrote."""
    d = Path(os.environ[TASK_DIR_ENV])
    t = json.loads((d / f"{spec.name}.json").read_text(encoding="utf-8"))
    frame = pd.read_pickle(d / f"{t['data']}.pkl")
    ctx.beat()
    return run_task(ExperimentTask(**t["task"]), frame)


def register_task_workers() -> int:
    """In a worker process, register `task_worker` under every task name the loop wrote (compute.WORKERS is keyed by spec name).
    Called at import only when the task-directory variable is set, i.e. only inside a worker the loop launched."""
    d = os.environ.get(TASK_DIR_ENV)
    if not d or not Path(d).is_dir():
        return 0
    n = 0
    for p in Path(d).glob("*.json"):
        if p.stem not in C.WORKERS:
            C.register_worker(p.stem)(task_worker)
            n += 1
    return n


class Executor:
    """Runs ExperimentTasks through engine.learning.compute (claim, private attempt folder, result envelope, ledger). inline: in this
    process, now. thread: a thread pool - the cycle continues while jobs run and harvests them later. process: each job in its own
    Python process (compute.run_experiment_process) driven from a thread, so the loop never blocks on it."""

    def __init__(self, root: Path, cfg: LoopConfig, code_hash: str):
        self.root, self.cfg, self.code_hash = Path(root), cfg, code_hash
        self.ledger = C.ExperimentLedger(self.root / "compute" / "ledger.json")
        self.out_root = self.root / "compute" / "out"
        self.task_dir = self.root / "compute" / "tasks"
        self.pool: cf.ThreadPoolExecutor | None = None
        self.futures: dict[str, cf.Future] = {}

    def _pool(self) -> cf.ThreadPoolExecutor:
        if self.pool is None:
            self.pool = cf.ThreadPoolExecutor(max_workers=self.cfg.max_workers, thread_name_prefix="research-job")
        return self.pool

    def submit(self, spec: C.ExperimentSpec, task: ExperimentTask, frame: pd.DataFrame, data_hash: str, now) -> str:
        if self.cfg.mode == "inline":
            return self._run_inline(spec, task, frame, now)
        if self.cfg.mode == "thread":
            self.futures[spec.key] = self._pool().submit(self._run_inline, spec, task, frame, now)
            return "RUNNING"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        pk = self.task_dir / f"{data_hash}.pkl"
        if not pk.exists():
            frame.to_pickle(pk)
        C.atomic_write_json(self.task_dir / f"{spec.name}.json", {"task": task.to_dict(), "data": data_hash})
        os.environ[TASK_DIR_ENV] = str(self.task_dir)
        free = self.cfg.free_gb
        self.futures[spec.key] = self._pool().submit(
            C.run_experiment_process, spec, self.ledger, self.out_root, self.root / "compute" / "work", _epoch(now),
            self.cfg.job_timeout_s, ("engine.research.loop",), None, None, (lambda: free) if free is not None else None)
        return "RUNNING"

    def _run_inline(self, spec: C.ExperimentSpec, task: ExperimentTask, frame: pd.DataFrame, now) -> str:
        r = C.run_worker(spec, _worker_fn(task, frame), self.ledger, None, self.out_root, f"w{os.getpid()}", _epoch(now),
                         code_hash=self.code_hash)
        return r.get("state", "FAILED")

    def finished(self) -> list[str]:
        """Keys whose job has returned (thread/process); inline jobs are finished on return."""
        done = [k for k, f in self.futures.items() if f.done()]
        for k in done:
            f = self.futures.pop(k)
            exc = f.exception()
            if exc is not None and isinstance(exc, FirewallBreach):
                raise exc
        return done

    def running(self) -> int:
        return sum(1 for f in self.futures.values() if not f.done())

    def wait(self, timeout: float | None = None) -> None:
        if self.futures:
            cf.wait(list(self.futures.values()), timeout=timeout)

    def results(self) -> dict[str, dict]:
        """Accepted results by experiment key (compute.reconcile: hash, spec, seed and code all verified)."""
        rec = C.reconcile(self.ledger, self.out_root, self.code_hash)
        return {k: r for k, r in rec.accepted}, {k: (why, det) for k, why, det in rec.rejected}

    def ledger_state(self, key: str) -> str | None:
        e = self.ledger.get(key)
        return None if e is None else e.get("state")

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=False)
            self.pool = None


def _epoch(now) -> float:
    return dt.datetime.combine(as_date(now), dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp()


# ================================================================================================================ C67 sweeps
@dataclasses.dataclass(frozen=True)
class SweepSpec:
    """A permanent low-priority job: `factory(root)` returns a callable run(n_units, now) -> int (units done). It must resume where it
    stopped on its own (its module's checkpoint / coverage book); the loop only decides WHEN it may run."""
    name: str
    area: str
    factory: Callable[[Path], Callable[[int, Any], int]]


@dataclasses.dataclass
class SweepProgress:
    units: int = 0
    runs: int = 0
    yielded: int = 0
    last_error: str = ""
    last_at: str = ""


def precursor_sweep(years: Sequence[int], loader, directory_name: str = "precursors") -> SweepSpec:
    """R21's mover-episode precursor sweep (engine.research.precursors.step, one coverage unit at a time, committed per unit)."""
    def factory(root: Path):
        from engine.research import precursors as PC
        st, store = PC.open_state(root / "sweeps" / directory_name, years)

        def run(n: int, now) -> int:
            return len(PC.step(st, now, loader, max_units=n, store=store).done)
        return run
    return SweepSpec("sweep.precursors", "NEW_REPRESENTATION", factory)


def volatility_sweep(loader: Callable[[int, int], pd.DataFrame], years: Sequence[int], now_for: Callable[[int], Any]) -> SweepSpec:
    """The volatility lab's resumable (pass, year) sweep (engine.research.volatility_lab.sweep with its own checkpoint)."""
    def factory(root: Path):
        from engine.research import volatility_lab as VL
        st = VL.LabState()
        gen = VL.sweep(loader, years, st, now_for, checkpoint_path=root / "sweeps" / "volatility_lab.json", max_passes=None)

        def run(n: int, now) -> int:
            k = 0
            for _ in range(n):
                if next(gen, None) is None:
                    break
                k += 1
            return k
        return run
    return SweepSpec("sweep.volatility_lab", "VOLATILITY", factory)


def discovery_sweep(cfg, engine_factory: Callable[[], Any], loader, last_date_for: Callable[[Any], Any]) -> SweepSpec:
    """Discovery's always-on sweep (engine.research.discovery.DiscoverySweep, resumed from its coverage book)."""
    def factory(root: Path):
        from engine.research import discovery as DI
        sw, st = DI.DiscoverySweep.resume(cfg, engine_factory(), root / "sweeps" / "discovery")

        def run(n: int, now) -> int:
            before = sw.units_run
            sw.run(st, loader, last_date_for(now), max_units=n)
            return sw.units_run - before
        return run
    return SweepSpec("sweep.discovery", "NEW_REPRESENTATION", factory)


# ================================================================================================================ state
@dataclasses.dataclass
class LoopState:
    """Everything the loop remembers (picklable; transient handles live in Runtime). `bus` holds the current cycle's intermediate
    outputs so a resumed cycle continues from the next stage with exactly the inputs the killed one had."""
    cfg: LoopConfig
    cycle: int = 0
    now: str = ""
    done_stages: dict = dataclasses.field(default_factory=dict)        # cycle -> [stage names completed]
    exec_count: dict = dataclasses.field(default_factory=dict)         # "cycle|stage" -> times executed (tests: exactly once)
    modules: dict = dataclasses.field(default_factory=dict)            # module name -> its state object
    controller: Any = None
    bus: dict = dataclasses.field(default_factory=dict)
    carry_events: list = dataclasses.field(default_factory=list)       # QuestionEvents raised by results, asked next cycle
    questions: dict = dataclasses.field(default_factory=dict)          # qid -> QuestionObject
    q_nodes: dict = dataclasses.field(default_factory=dict)            # question key (source|subject|evidence) -> lineage node
    items: dict = dataclasses.field(default_factory=dict)              # item_id -> ResearchItem
    branch_of: dict = dataclasses.field(default_factory=dict)          # question_id -> branch_id
    question_of_branch: dict = dataclasses.field(default_factory=dict)
    jobs: dict = dataclasses.field(default_factory=dict)               # experiment key -> JobRecord
    designs: dict = dataclasses.field(default_factory=dict)            # design key -> experiment key (duplicate detection)
    seen_through: dict = dataclasses.field(default_factory=dict)       # branch -> newest decision date used by its rungs
    pending_realised: list = dataclasses.field(default_factory=list)   # priority.RealisedValue, absorbed once matured
    pending_yields: list = dataclasses.field(default_factory=list)     # controller.YieldRecord
    outcomes: list = dataclasses.field(default_factory=list)           # brain_health.Outcome (diversity reads the same)
    knowledge: dict = dataclasses.field(default_factory=dict)          # knowledge id -> dict (feature, sign, problem, filed_at, ...)
    decisions: list = dataclasses.field(default_factory=list)          # recent two_stage.DayDecision
    evaluations: list = dataclasses.field(default_factory=list)        # two_stage.Evaluation dicts
    readings: list = dataclasses.field(default_factory=list)           # controller.CapabilityReading not yet ingested
    lineage: Lineage = dataclasses.field(default_factory=Lineage)
    reports: list = dataclasses.field(default_factory=list)
    statements: list = dataclasses.field(default_factory=list)
    counters: dict = dataclasses.field(default_factory=dict)
    sweeps: dict = dataclasses.field(default_factory=dict)             # sweep name -> SweepProgress
    screened: dict = dataclasses.field(default_factory=dict)           # feature|problem -> date an event was raised
    memo: dict = dataclasses.field(default_factory=dict)               # per-module bookkeeping (what was already fed to whom)
    namespace: Namespace = NAMESPACE

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n


class Runtime:
    """Transient per-process handles: the feed, the executor, the current observation, the live store and sweep runners. Rebuilt on
    resume; never pickled."""

    def __init__(self, feed: Feed, root: str | Path, cfg: LoopConfig, sweeps: Sequence[SweepSpec] = (), clock: Callable[[], float] = time.time,
                 kill_after: str | None = None):
        self.feed, self.root, self.cfg = feed, Path(root), cfg
        self.code_hash = cfg.code_hash or current_code_hash()
        self.executor = Executor(self.root, cfg, self.code_hash)
        self.obs: Observation | None = None
        self.live = None
        self.release = None
        self.sweep_specs = {s.name: s for s in sweeps}
        self.sweep_runs: dict[str, Callable] = {}
        self.clock = clock
        self.kill_after = kill_after                     # tests: raise KeyboardInterrupt right after this stage completes
        self.pipe: TS.TwoStage | None = None
        self.checkpointer = Checkpointer(self.root, cfg) if cfg.checkpoint != "off" else None

    def free_gb(self) -> float | None:
        if self.cfg.free_gb is not None:
            return float(self.cfg.free_gb)
        try:
            from engine import resources as R
            return R.memory_gb()[0]
        except Exception:                                   # noqa: BLE001 - unreadable memory must refuse, not guess
            return None


# ================================================================================================================ checkpoints
def make_mappingproxy(d: dict):
    return types.MappingProxyType(d)


def _reduce_mappingproxy(m):
    return (make_mappingproxy, (dict(m),))


# W02: filed knowledge is deep-frozen (engine.research.namespaces.deep_freeze -> MappingProxyType), which pickle refuses, so the first
# promoted finding made every later checkpoint fail. A read-only proxy is saved as a read-only proxy of a copy of its contents.
copyreg.pickle(types.MappingProxyType, _reduce_mappingproxy)


class Checkpointer:
    """engine.learning.checkpoints.CheckpointStore holds the execution record (cycle, completed stages, next action, current experiment,
    code hash, failures) and names the pickled loop state as an artifact with its sha256, so a torn or edited state file is refused and
    the previous verified one is used instead."""

    def __init__(self, root: Path, cfg: LoopConfig):
        self.dir = Path(root) / "checkpoints"
        self.store = CK.CheckpointStore(self.dir, cfg.run_id, keep=max(3, cfg.keep_checkpoints))

    def save(self, state: LoopState, code_hash: str, phase: str, next_action: str, current_experiment: str = "",
             failures: Sequence[CK.FailureNote] = ()) -> CK.ExecutionState:
        seqs = self.store.sequences()
        seq = (seqs[-1] + 1) if seqs else 0
        blob = self.store.dir / f"state_{seq:06d}.pkl"
        blob.parent.mkdir(parents=True, exist_ok=True)
        tmp = blob.with_name(blob.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, blob)
        done = {f"{state.cycle}|{s}": code_hash for s in state.done_stages.get(state.cycle, [])}
        st = self.store.save(state.now or "1970-01-01", phase, next_action, code_hash, current_experiment, failures, done, (),
                             {"cycle": str(state.cycle)}, [blob])
        for p in self.store.dir.glob("state_*.pkl"):
            if int(p.stem[6:]) < seq - self.store.keep:
                try:
                    p.unlink()
                except OSError:
                    pass
        return st

    def load(self, code_hash: str, allow_code_change: bool = False) -> tuple[LoopState | None, CK.ResumePlan]:
        plan = CK.resume_verified(self.store, code_hash, allow_code_change)
        st = plan.state
        while st is not None:
            ok = True
            for path, sha in st.artifacts.items():
                p = Path(path)
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                    ok = False
            if ok:
                with open(next(iter(st.artifacts)), "rb") as f:
                    return pickle.load(f), plan
            older = [s for s in self.store.sequences() if s < st.sequence]
            st = self.store.read(older[-1]) if older else None
        return None, plan

    def interruption(self, state: LoopState, code_hash: str, reason: str) -> Path | None:
        st, _ = self.store.latest_valid()
        if st is None:
            return None
        rec = CK.InterruptionRecord.from_state(st, reason)
        return CK.write_interruption(self.dir / "INTERRUPTED.json", rec)


# ================================================================================================================ stage plumbing
@dataclasses.dataclass(frozen=True)
class StageSpec:
    name: str
    phase: LoopPhase
    fn: Callable[["Ctx"], tuple]
    module: str = ""


@dataclasses.dataclass
class Ctx:
    state: LoopState
    rt: Runtime
    now: str
    cycle: int

    @property
    def bus(self) -> dict:
        return self.state.bus

    @property
    def obs(self) -> Observation:
        if self.rt.obs is None:
            raise NoInput("no observation this cycle (the OBSERVE stage refused or the feed returned nothing)")
        return self.rt.obs

    def extra(self, stage: str, what: str) -> dict:
        """The feed's input for `stage`. A callable input (engine.research.feeds builders) is built HERE, inside the stage, so it can
        read what earlier stages of this cycle produced and so a future leak it carries is refused at this stage (REFUSED_LEAK)."""
        ex = (self.rt.obs.extras if self.rt.obs is not None else {}).get(stage)
        if ex is not None and callable(ex):
            ex = ex(self)
        if not ex:
            raise NoInput(f"the feed supplies no {what}")
        return dict(ex)

    def namespace(self, ns: str) -> dict:
        """Extra per-day inputs the feed carries under a namespaced key (e.g. 'c68': per-prediction expectations and realised paths),
        for registered stages. Built lazily like any stage input; absent or empty -> NoInput (the stage is SKIPPED_NO_INPUT)."""
        return self.extra(ns, f"inputs under the {ns!r} namespace")

    def handle(self, name: str, factory: Callable[[], Any]) -> Any:
        """A transient per-process object a stage reuses across cycles (never checkpointed: engines, stores)."""
        h = self.rt.__dict__.setdefault("handles", {})
        if name not in h:
            h[name] = factory()
        return h[name]

    def mod_state(self, name: str, factory: Callable[[], Any]) -> Any:
        if name not in self.state.modules:
            self.state.modules[name] = factory()
        return self.state.modules[name]

    def created_real(self) -> str:
        return dt.datetime.fromtimestamp(self.rt.clock(), dt.timezone.utc).isoformat(timespec="seconds")

    def evidence_date(self) -> str:
        """Newest matured evidence date strictly before now (the date research events rest on)."""
        e = self.obs.evidence_through
        if not e:
            raise NoInput("nothing has matured yet")
        return e


def _prov(ctx: Ctx, learned_at: str, **kw) -> Provenance:
    return Provenance(ctx.created_real(), learned_at, ctx.rt.code_hash, data_hash=ctx.obs.data_hash if ctx.rt.obs else "",
                      run_id=ctx.state.cfg.run_id, seed=ctx.state.cfg.seed, outcomes_seen_through=learned_at, **kw)


def _events(ctx: Ctx) -> list:
    return ctx.bus.setdefault("events", [])


def _rquestions(ctx: Ctx) -> list:
    return ctx.bus.setdefault("research_questions", [])


def _records(ctx: Ctx) -> list:
    return ctx.bus.setdefault("records", [])


# ================================================================================================================ OBSERVE
def st_panel(ctx: Ctx) -> tuple:
    """Take the feed's observation and check it fail-closed: every matured row ended strictly before now, the decision day carries no
    outcome columns and nothing dated after now."""
    obs = ctx.rt.feed.observe(ctx.now)
    if len(obs.matured) and "end" in obs.matured:
        late = pd.to_datetime(obs.matured["end"]) >= pd.Timestamp(as_date(ctx.now))
        if late.any():
            raise FirewallBreach(f"feed offered {int(late.sum())} 'matured' rows whose outcome ends on/after now={ctx.now}")
    TS.assert_point_in_time(obs.today, ctx.now)
    ctx.rt.obs = obs
    ctx.bus["data_hash"] = obs.data_hash
    ctx.bus["evidence_through"] = obs.evidence_through
    ctx.state.lineage.add("OBSERVATION", f"{ctx.now}", ctx.cycle, ctx.now, rows=len(obs.matured), data_hash=obs.data_hash)
    return len(obs.matured), len(obs.today), ""


def st_observer(ctx: Ctx) -> tuple:
    try:
        from engine.research import observer as OB
    except ImportError as e:
        raise MissingModule("observer") from e
    ex = ctx.extra("observe.observer", "DecisionSnapshot/DayOutcome pair (observer inputs)")
    st = ctx.mod_state("observer", OB.ObserverState)
    rec = OB.step(st, ex["snap"], ex["out"], ctx.now)
    ctx.bus["observer_day"] = rec
    return 1, 1, ""


def st_autopsy(ctx: Ctx) -> tuple:
    try:
        from engine.research import autopsy as AU
    except ImportError as e:
        raise MissingModule("autopsy") from e
    ex = ctx.extra("observe.autopsy", "DecisionSnapshot/DayOutcome pair (autopsy inputs)")
    st = ctx.mod_state("autopsy", AU.AutopsyState)
    a = AU.step(st, ex["snap"], ex["out"], ctx.now, ctx.created_real(), ex.get("ctx"))
    qs = a.questions() if callable(a.questions) else a.questions
    _rquestions(ctx).extend(qs)
    return 1, len(qs), ""


# ================================================================================================================ UPDATE KNOWLEDGE
def st_experiments_sweep(ctx: Ctx) -> tuple:
    """experiments.step: the experiment memory's maintenance sweep (stale answers, follow-ups nobody ran, contradictions)."""
    try:
        from engine.research import experiments as EX
    except ImportError as e:
        raise MissingModule("experiments") from e
    mem = _experiment_memory(ctx)
    rep = EX.step(mem, ctx.now, current_code=ctx.rt.code_hash, current_data=ctx.bus.get("data_hash", ""))
    ctx.bus["memory_report"] = {"stale": len(rep.stale), "unfollowed": len(rep.unfollowed_actions), "contradictions": len(rep.contradictions),
                                "needs_replication": len(rep.needs_replication)}
    for c in rep.contradictions[:3]:
        _events(ctx).append(_q_event("contradiction", f"experiment memory contradiction {stable_hash(c, 6)}", ctx, 0.6, Problem.RESEARCH_PROCESS))
    return rep.n_experiments, len(rep.contradictions), ""


def _experiment_memory(ctx: Ctx):
    from engine.learning.experiment_memory import ExperimentLedger
    from engine.research import experiments as EX
    return ctx.mod_state("experiments", lambda: EX.ResearchExperimentMemory(ExperimentLedger()))


def _firewall(ctx: Ctx):
    from engine.research import firewall as FWL
    from engine.research.namespaces import ResearchStore
    return ctx.mod_state("firewall", lambda: FWL.ResearchTraderFirewall(ResearchStore("research"), ctx.state.cfg.firewall_secret))


def st_release(ctx: Ctx) -> tuple:
    """firewall.run_day: every filed knowledge object that passes the research/trader firewall at `now` crosses into a fresh live
    store; the TraderRelease is the ONLY knowledge the decision chain receives this cycle."""
    try:
        from engine.research import firewall as FWL
    except ImportError as e:
        raise MissingModule("firewall") from e
    fw = _firewall(ctx)
    ctx.rt.release, ctx.rt.live = None, None
    ids = list(fw.store.ids())
    if not ids:
        raise NoInput("no knowledge has been filed for release yet")
    live = fw.live_store("live")
    res = FWL.run_day(fw, ctx.now, ids, live=live)
    ctx.rt.release, ctx.rt.live = res.release, live
    rid = ctx.state.lineage.add("RELEASE", res.release_id, ctx.cycle, ctx.now, items=len(res.release))
    for oid in ids:
        kn = ctx.state.knowledge.get(oid)
        if kn is not None and any(d.object_id == oid and d.admitted for d in res.decisions):
            ctx.state.lineage.link(f"KNOWLEDGE:{oid}", rid, "released_in")
    ctx.bus["release_id"] = rid
    return len(ids), len(res.release), f"{len(res.refused)} refused"


# ================================================================================================================ EVALUATE
def st_two_stage(ctx: Ctx) -> tuple:
    """Section 35 on today's point-in-time state with the firewall's release; then score every earlier decision whose outcome has
    matured. The evaluation becomes the controller's capability readings."""
    obs = ctx.obs
    if len(obs.matured) == 0:
        raise NoInput("no matured history to fit the two-stage chain on")
    if ctx.rt.pipe is None:
        ctx.rt.pipe = TS.TwoStage(ctx.state.cfg.two_stage)
    dec = TS.run_day(ctx.rt.pipe, obs.matured, obs.today, ctx.now, ctx.rt.release)
    ctx.state.decisions = (ctx.state.decisions + [dec])[-60:]
    did = ctx.state.lineage.add("DECISION", dec.decided_at, ctx.cycle, ctx.now, positions=int((dec.table["side"] != 0).sum()),
                                knowledge=dec.knowledge_digest)
    if ctx.bus.get("release_id") and ctx.rt.release is not None and len(ctx.rt.release):
        ctx.state.lineage.link(ctx.bus["release_id"], did, "used_by")
    ev = TS.evaluate(ctx.state.decisions, obs.matured, ctx.now)
    ctx.state.evaluations = (ctx.state.evaluations + [ev.to_dict()])[-60:]
    ctx.bus["evaluation"] = ev.to_dict()
    if ev.matured_through:
        ctx.state.readings.extend(TS.capability_readings(ev, ev.matured_through))
    rep = ctx.rt.pipe.report
    ctx.bus["two_stage"] = {"gate_open": dec.gate.get("open"), "positions": int((dec.table["side"] != 0).sum()),
                            "vol_ok": bool(rep.vol_ok) if rep else False, "dir_ok": bool(rep.dir_ok) if rep else False}
    return len(obs.today), int((dec.table["side"] != 0).sum()), f"gate {'open' if dec.gate.get('open') else 'closed'}"


def st_volatility_lab(ctx: Ctx) -> tuple:
    try:
        from engine.research import volatility_lab as VL
    except ImportError as e:
        raise MissingModule("volatility_lab") from e
    F = ctx.obs.matured
    if len(F) == 0 or "touch" not in F:
        raise NoInput("no matured volatility frame")
    st = ctx.mod_state("volatility_lab", VL.LabState)
    res = VL.step(st, ctx.now, F, max_tasks=1)
    if getattr(res, "record", None) is not None:
        _records(ctx).append(res.record)
    return len(F), 1 if getattr(res, "record", None) is not None else 0, ""


def st_direction_lab(ctx: Ctx) -> tuple:
    """direction_lab.step on the two-stage chain's walk-forward predicted-mover block (score = the champion's walk-forward P(vol))."""
    try:
        from engine.research import direction_lab as DL
    except ImportError as e:
        raise MissingModule("direction_lab") from e
    from engine.research import vol_hypotheses as VH
    pipe = ctx.rt.pipe
    oos = getattr(pipe, "last_oos", None) if pipe is not None else None
    if oos is None or len(oos) < 200:
        raise NoInput("no walk-forward volatility block from the two-stage chain yet")
    feats = tuple(f for f in ctx.state.cfg.two_stage.direction_features if not VH.missing_columns((f,), oos.columns))
    X = VH.derive(oos, feats)
    d = pd.to_datetime(oos.index.get_level_values(0))
    labels = pd.DataFrame({"entry_date": d + pd.Timedelta(days=1), "end_date": pd.to_datetime(oos["end"]).to_numpy(),
                           "fwd": oos["close"].to_numpy(float), "mover": oos["touch"].to_numpy(float), "up_sign": oos["up"].to_numpy(float),
                           "up_first": oos["up"].to_numpy(float)}, index=oos.index)
    inputs = DL.LabInputs(X, oos["p_TS"], labels, sector=oos["sector"] if "sector" in oos else None)
    st, rep = DL.step(ctx.state.modules.get("direction_lab"), ctx.now, inputs)
    ctx.state.modules["direction_lab"] = st
    return len(oos), 1, str(getattr(rep, "outcome", ""))


def st_frontier(ctx: Ctx) -> tuple:
    try:
        from engine.research import frontier as FR
    except ImportError as e:
        raise MissingModule("frontier") from e
    ex = ctx.extra("evaluate.frontier", "P(up | predicted mover) rows or a populated FrontierState")
    st = ctx.state.modules.get("frontier")
    if st is None:
        st = ctx.state.modules["frontier"] = ex["state"] if "state" in ex else ex["make"]()
    if ex.get("rows") is not None and len(ex["rows"]):
        st.add(ex["rows"])
    rep = FR.step(st, ctx.now, ctx.state.cfg.seed, code_hash=ctx.rt.code_hash)
    _records(ctx).append(rep.to_matured_record())
    return 1, 1, ""


def st_symmetry(ctx: Ctx) -> tuple:
    try:
        from engine.research import symmetry as SY
    except ImportError as e:
        raise MissingModule("symmetry") from e
    ex = ctx.extra("evaluate.symmetry", "winner/loser pattern rows or a populated SymmetryState")
    st = ctx.state.modules.get("symmetry")
    if st is None:
        st = ctx.state.modules["symmetry"] = ex["state"] if "state" in ex else ex["make"]()
    n = 0
    if ex.get("rows") is not None and len(ex["rows"]):
        st.add(ex["rows"])
        n = len(ex["rows"])
    rep = SY.step(st, ctx.now, ctx.state.cfg.seed, code_hash=ctx.rt.code_hash)
    _records(ctx).append(rep.to_matured_record(getattr(st, "bank", None)))
    ctx.bus["symmetry_report"] = rep
    return n, 1, ""


# ================================================================================================================ SURPRISES
def st_feature_screen(ctx: Ctx) -> tuple:
    """volatility_lab.oriented_scan over the matured panel: a feature whose walk-forward AUC is significant and that no open question
    covers becomes a 'new_discovery' event (the generalisation question). One event per feature per window (no re-asking)."""
    from engine.research import questions as Q
    from engine.research import vol_hypotheses as VH
    from engine.research import volatility_lab as VL
    F = ctx.obs.matured
    if len(F) == 0 or "touch" not in F:
        raise NoInput("no matured frame to screen")
    feats = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in TS.HISTORY_DEPENDENT
             and not VH.missing_columns((f,), F.columns)]
    n_dates = F.index.get_level_values(0).nunique()
    lab = VL.LabConfig(min_train_dates=max(8, n_dates // 3), test_step_dates=max(2, n_dates // 8), n_boot=100)
    tab = VL.oriented_scan(F, feats, ctx.now, lab)
    ok = tab[tab["auc"].notna() & tab["lo"].notna()]
    raised = 0
    when = ctx.evidence_date()
    for _, r in ok.sort_values("auc", ascending=False).iterrows():
        if raised >= ctx.state.cfg.screen_top:
            break
        se = max((r["hi"] - r["lo"]) / (2 * 1.645), 1e-6)
        t = (r["auc"] - 0.5) / se
        key = f"{r['feature']}|VOLATILITY"
        if t < ctx.state.cfg.screen_t or key in ctx.state.screened:
            continue
        ev = Q.event_from_discovery(f"feature {r['feature']} volatility", float(t), int(r["n_dates"]), when, problem=Problem.VOLATILITY)
        if ev is not None:
            _events(ctx).append(ev)
            ctx.state.screened[key] = when
            raised += 1
    if not any(k.endswith("|DIRECTION") for k in ctx.state.screened):
        _events(ctx).append(_q_event("coverage_gap", "feature rel_r5 direction among predicted movers", ctx, 0.5, Problem.DIRECTION))
        ctx.state.screened["rel_r5|DIRECTION"] = when
    return len(feats), raised, ""


def _q_event(source: str, subject: str, ctx: Ctx, magnitude: float, problem: Problem, **kw):
    from engine.research import questions as Q
    return Q.QuestionEvent(source, subject, ctx.evidence_date(), float(min(1.0, max(0.0, magnitude))), problem=problem, **kw)


def st_cross_section(ctx: Ctx) -> tuple:
    try:
        from engine.research import cross_section as CS
    except ImportError as e:
        raise MissingModule("cross_section") from e
    ex = ctx.extra("surprises.cross_section", "day frames of returns through each session's close (cross-section inputs)")
    lab = ctx.mod_state("cross_section", ex.get("make", CS.CrossSectionLab))
    days = ex["days"] if "days" in ex else [(ctx.now, ex["day_frame"], ex.get("settle", ()))]
    res, n = None, 0
    for i, (day, frame, settle) in enumerate(days):         # the shuffled-null control runs on the newest session of the batch
        res = CS.step(lab, day, frame, settle, ex.get("null_seed") if i == len(days) - 1 else None, n_shuffles=3)
        n += int(res.n_names)
    if res is None:
        raise NoInput("every offered session was already processed")
    outs = getattr(lab, "outcomes", lab)
    _records(ctx).extend(outs.matured_records(ctx.now))
    return n, len(days), str(res.status)


def st_regimes(ctx: Ctx) -> tuple:
    """regimes.step on today's market row (the m_ columns of the point-in-time day); persisted regime changes become questions."""
    try:
        from engine.research import regimes as RG
    except ImportError as e:
        raise MissingModule("regimes") from e
    today = ctx.obs.today
    mcols = [c for c in today.columns if str(c).startswith("m_")]
    if not len(today) or not mcols:
        raise NoInput("no market (m_) columns on the decision day")
    row = {c: float(np.nanmean(today[c].to_numpy(float))) for c in mcols}
    ret_col = next((c for c in ("m_ret", "m_r1", "m_r5") if c in today), None)
    if ret_col is None:
        raise NoInput("no market-return column (m_ret / m_r1 / m_r5) on the decision day")
    row["ret"] = row[ret_col]
    if "r5" in today:
        row["dispersion"] = float(np.nanstd(today["r5"].to_numpy(float)))
    if "m_breadth" in row:
        row["breadth"] = row["m_breadth"]
    mon = ctx.mod_state("regimes", RG.RegimeMonitor)
    book = ctx.mod_state("regimes_book", RG.PatternRegimeBook)
    res = RG.step(mon, book, ctx.now, row, ctx.bus.get("pattern_effects", ()))
    changes = tuple(getattr(res, "changes", ()) or ())
    if changes:
        _rquestions(ctx).extend(RG.regime_questions(changes, ctx.created_real(), ctx.evidence_date()))
    return len(mcols), len(changes), ""


def st_multiscale(ctx: Ctx) -> tuple:
    try:
        from engine.research import multiscale as MS
    except ImportError as e:
        raise MissingModule("multiscale") from e
    ex = ctx.extra("surprises.multiscale", "pattern flags and a close-price panel (multiscale inputs)")
    led = ctx.mod_state("multiscale", MS.ScaleLedger)
    res = MS.step(led, ctx.now, ex["flags"], ex["close"], ex.get("open"), declared=ex.get("declared"))
    _records(ctx).extend(led.matured_records(ctx.now))
    return len(ex["flags"]), 1, str(type(res).__name__)


# ================================================================================================================ FAILURES
def _matured_decisions(ctx: Ctx) -> pd.DataFrame:
    """Decision rows joined with outcomes that matured strictly before now (research side: hindsight allowed)."""
    frames = [d.table.assign(_day=d.decided_at) for d in ctx.state.decisions if len(d.table)]
    M = ctx.obs.matured
    if not frames or len(M) == 0:
        return pd.DataFrame()
    J = pd.concat(frames).join(M[["touch", "up", "close", "end", "absmove"] + [c for c in ("sector",) if c in M]], how="inner")
    return J[J["touch"].notna()]


def st_decision_losses(ctx: Ctx) -> tuple:
    """Section 5/34: where did realised losses of the decision chain concentrate? For each derived feature, the share of predicted-
    mover losses in its top quintile; the largest concentrations become loss events (loss share = expected loss avoidable)."""
    from engine.research import questions as Q
    from engine.research import vol_hypotheses as VH
    J = _matured_decisions(ctx)
    if len(J) == 0:
        raise NoInput("no matured decisions yet")
    mv = J[J["mover"].to_numpy(bool)]
    loss = np.clip(-mv["close"].to_numpy(float), 0, None)
    if len(mv) < 30 or loss.sum() <= 0:
        raise NoInput(f"{len(mv)} matured predicted movers, no loss to attribute")
    M = ctx.obs.matured.loc[mv.index.intersection(ctx.obs.matured.index)]
    feats = [f for f in ("gap_ratio", "shock1", "vol_ratio_short", "dv_rel", "price_low", "near_hi", "near_lo")
             if not VH.missing_columns((f,), M.columns)]
    shares = {}
    if feats and len(M) == len(mv):
        D = VH.derive(M, feats)
        for f in feats:
            v = D[f].to_numpy(float)
            q = np.nanquantile(v, 0.8)
            shares[f] = float(loss[v >= q].sum() / loss.sum())
    ctx.bus["loss_sources"] = {f"feature {k}": float(min(1.0, max(0.0, (v - 0.2) / 0.8))) for k, v in shares.items()}
    raised = 0
    for f, s in sorted(shares.items(), key=lambda kv: -kv[1])[:2]:
        if s >= 0.3 and f"{f}|LOSS" not in ctx.state.screened:
            _events(ctx).append(Q.event_from_loss(f"feature {f} loss", s, ctx.evidence_date()))
            ctx.state.screened[f"{f}|LOSS"] = ctx.now
            raised += 1
    return len(mv), raised, ""


def move_records(ctx: Ctx) -> list:
    """winners_losers.MoveRecord per matured decision row (identity-free: the id is a hash, no ticker or date text inside)."""
    from engine.research import winners_losers as WL
    J = _matured_decisions(ctx)
    out = []
    for (d, tk), r in J.iterrows():
        rid = "M" + stable_hash([str(d), str(tk)], 14)
        side = int(r["side"])
        out.append(WL.MoveRecord(rid, str(pd.Timestamp(d).date()), str(pd.Timestamp(r["end"]).date()), ctx.state.cfg.experiment.horizon_days,
                                 float(r["close"]), considered=bool(r["mover"]), held=side != 0, side=side,
                                 prob_move=None if not np.isfinite(r["p_move"]) else float(r["p_move"]),
                                 dir_prob=None if not np.isfinite(r["p_up"]) else float(r["p_up"]),
                                 pnl=float(side * r["close"]) if side else None))
    return out


def st_winners_losers(ctx: Ctx) -> tuple:
    try:
        from engine.research import winners_losers as WL
    except ImportError as e:
        raise MissingModule("winners_losers") from e
    recs = move_records(ctx)
    if not recs:
        raise NoInput("no matured decision rows")
    st = ctx.mod_state("winners_losers", WL.WinnerState)
    rep = WL.step(st, recs, ctx.now)
    _records(ctx).extend(rep.matured)
    return len(recs), len(rep.findings), ""


def st_loss_pipeline(ctx: Ctx) -> tuple:
    try:
        from engine.research import loss_pipeline as LP
    except ImportError as e:
        raise MissingModule("loss_pipeline") from e
    recs = move_records(ctx)
    if not recs:
        raise NoInput("no matured decision rows")
    st = ctx.mod_state("loss_pipeline", LP.LossState)
    rep = LP.step(st, recs, ctx.now, created_real=ctx.created_real())
    _rquestions(ctx).extend(rep.questions)
    _records(ctx).extend(rep.matured)
    return len(recs), len(rep.questions), f"unknown rate {rep.unknown_rate:.2f}" if np.isfinite(rep.unknown_rate) else ""


# ================================================================================================================ MISSED WINNERS / LOSERS
def st_missed(ctx: Ctx) -> tuple:
    """missed.submit + step: every matured decision day becomes a DayBook (ranks of the decision features), so missed winners AND
    missed losers (section 6) are researched; its research targets become questions."""
    try:
        from engine.research import missed as MI
    except ImportError as e:
        raise MissingModule("missed") from e
    from engine.research import vol_hypotheses as VH
    J = _matured_decisions(ctx)
    if len(J) == 0:
        raise NoInput("no matured decision days")
    M = ctx.obs.matured
    names = tuple(f for f in ("lv20", "shock1", "vol_ratio_short", "rel_r5") if not VH.missing_columns((f,), M.columns))
    st = ctx.mod_state("missed", lambda: MI.new_state(names))
    reports = []
    for day, g in J.groupby("_day"):
        if day in st.done or any(p.decided_at == day for p in st.pending):
            continue
        rows = M.loc[g.index.intersection(M.index)]
        if len(rows) < 10:
            continue
        R = VH.derive(rows, names).groupby(level=0).rank(pct=True)
        obs = []
        for ix, r in g.loc[rows.index].iterrows():
            obs.append(MI.MoveObs("C" + stable_hash([str(ix)], 12), day, str(pd.Timestamp(r["end"]).date()), float(r["close"]),
                                  {n: float(R.loc[ix, n]) for n in names}, picked=int(r["side"]) != 0,
                                  score=None if not np.isfinite(r["p_move"]) else float(r["p_move"]),
                                  pred_prob=None if not np.isfinite(r["p_up"]) else float(r["p_up"])))
        MI.submit(st, MI.DayBook(day, str(pd.to_datetime(g["end"]).max().date()), tuple(obs), names))
    reports = MI.step(st, ctx.now)
    for rep in reports:
        _records(ctx).extend(MI.matured_records(rep, ctx.created_real(), ctx.state.cfg.seed))
    _rquestions(ctx).extend(MI.stream_questions(st, ctx.created_real()))
    return len(J), len(reports), ""


def st_knowability(ctx: Ctx) -> tuple:
    try:
        from engine.research import knowability as KB
    except ImportError as e:
        raise MissingModule("knowability") from e
    ex = ctx.extra("missed.knowability", "MoveInputs (bars and information items per move)")
    st = ctx.mod_state("knowability", KB.KnowabilityState)
    st, rep = KB.step(st, ctx.now, ex["inputs"], ex.get("calendar"), code_hash=ctx.rt.code_hash)
    ctx.state.modules["knowability"] = st
    ids = {i.move.move_id for i in ex["inputs"]}
    ass = tuple(getattr(rep, "assessments", ()) or ()) or tuple(a for a in st.ledger.rows() if a.move_id in ids)
    ctx.bus["knowability_assessments"] = ass
    return len(ex["inputs"]), len(ass), f"{rep.n_classified} classified, {rep.n_immature} immature"


def st_unknown_cause(ctx: Ctx) -> tuple:
    try:
        from engine.research import unknown_cause as UC
    except ImportError as e:
        raise MissingModule("unknown_cause") from e
    ass = ctx.bus.get("knowability_assessments") or ()
    if not ass:
        raise NoInput("no knowability assessments this cycle")
    st = ctx.mod_state("unknown_cause", UC.UnknownCauseState)
    st, rep = UC.step(st, ctx.now, ass)
    ctx.state.modules["unknown_cause"] = st
    return len(ass), 1, ""


def st_counterfactual(ctx: Ctx) -> tuple:
    try:
        from engine.research import counterfactual as CFM
    except ImportError as e:
        raise MissingModule("counterfactual") from e
    ex = ctx.extra("missed.counterfactual", "a point-in-time store and matured event specs (counterfactual inputs)")
    reps = CFM.step(ex["store"], ex["events"], ctx.now, ex.get("providers"), seed=ctx.state.cfg.seed)
    _records(ctx).extend(r.matured_record() for r in reps)
    return len(ex["events"]), len(reps), ""


# ================================================================================================================ PATTERN BREAKS
def st_knowledge_breaks(ctx: Ctx) -> tuple:
    """Released knowledge is re-measured on the dates after it was filed; a knowledge item whose effect has collapsed raises a
    'pattern_break' question (section 14) instead of silently going on influencing decisions."""
    from engine.research import questions as Q
    from engine.research import vol_hypotheses as VH
    if not ctx.state.knowledge:
        raise NoInput("no filed knowledge to watch")
    F = ctx.obs.matured
    raised, n = 0, 0
    for kid, k in sorted(ctx.state.knowledge.items()):
        after = F[np.asarray(pd.to_datetime(F.index.get_level_values(0)) > pd.Timestamp(as_date(k["filed_at"])))]
        if len(after) == 0 or VH.missing_columns((k["feature"],), after.columns):
            continue
        s = k["sign"] * VH.derive(after, (k["feature"],))[k["feature"]].to_numpy(float)
        codes = pd.factorize(after.index.get_level_values(0), sort=True)[0]
        _, a = _per_date_auc(s, after["touch"].to_numpy(float), codes, ctx.state.cfg.experiment.min_names)
        n += 1
        if len(a) < 4:
            continue
        k["post_effect"], k["post_n"] = float(a.mean()), int(len(a))
        before, now_rel = 0.5 + float(k["effect"]), 0.5 + float(a.mean())
        ev = Q.event_from_break(f"feature {k['feature']} knowledge", before, now_rel, int(len(a)) * 10, ctx.evidence_date())
        if ev is not None and not k.get("break_raised"):
            _events(ctx).append(ev)
            k["break_raised"] = ctx.now
            raised += 1
    return n, raised, ""


def st_break_research(ctx: Ctx) -> tuple:
    try:
        from engine.research import break_research as BR
    except ImportError as e:
        raise MissingModule("break_research") from e
    ex = ctx.extra("breaks.break_research", "pattern item series (break-research inputs)")
    st = ctx.mod_state("break_research", BR.new_state)
    rep = BR.step(st, ctx.now, ex["items"], health=ex.get("health"), created_real=ctx.created_real(), seed=ctx.state.cfg.seed)
    _rquestions(ctx).extend(q for q in getattr(rep, "questions", ()) if isinstance(q, ResearchQuestion))
    return len(ex["items"]), len(getattr(rep, "questions", ())), ""


# ================================================================================================================ QUESTIONS
def st_discovery(ctx: Ctx) -> tuple:
    try:
        from engine.research import discovery as DI
    except ImportError as e:
        raise MissingModule("discovery") from e
    ex = ctx.extra("questions.discovery", "discovery SourceInputs")
    st = ctx.mod_state("discovery", DI.DiscoveryState)
    eng = ctx.handle("discovery_engine", ex["engine_factory"]) if ex.get("engine_factory") else None
    rep = DI.step(st, ctx.now, ex["inputs"], engine=eng)
    new = getattr(rep, "new", ()) or ()
    return 1, new if isinstance(new, int) else len(new), ""


def st_interactions(ctx: Ctx) -> tuple:
    try:
        from engine.research import interactions as IA
    except ImportError as e:
        raise MissingModule("interactions") from e
    ex = ctx.extra("questions.interactions", "InteractionInputs")
    st = ctx.mod_state("interactions", IA.InteractionState)
    rep = IA.step(st, ex["inputs"], ctx.now)
    _records(ctx).extend(IA.to_matured_records(rep, ctx.created_real(), seed=ctx.state.cfg.seed))
    return 1, len(getattr(rep, "findings", ()) or ()), ""


def st_precursors(ctx: Ctx) -> tuple:
    """R21 mover-episode precursors, one coverage unit per cycle (the rest runs as the always-on sweep)."""
    try:
        from engine.research import precursors as PC
    except ImportError as e:
        raise MissingModule("precursors") from e
    ex = ctx.extra("questions.precursors", "a bar loader and sweep years (R21 precursor inputs)")
    kw = {k: ex[k] for k in ("registry",) if ex.get(k) is not None}
    args = (ex["cfg"],) if ex.get("cfg") is not None else ()
    if "precursors" not in ctx.state.modules:
        st, store = PC.open_state(ctx.rt.root / "sweeps" / "precursors_stage", ex["years"], *args, **kw)
        ctx.state.modules["precursors"] = st
        ctx.rt.__dict__.setdefault("handles", {})["precursor_store"] = store
    store = ctx.handle("precursor_store", lambda: PC.open_state(ctx.rt.root / "sweeps" / "precursors_stage", ex["years"], *args, **kw)[1])
    ctx.state.modules["precursors"].book.extend_years(ex["years"])
    rep = PC.step(ctx.state.modules["precursors"], ctx.now, ex["loader"], max_units=1, store=store)
    ev = rep.evaluation
    if ev is not None:
        for c in list(getattr(ev, "promoted", ()) or ())[:3]:
            cand = ctx.state.modules["precursors"].candidates.get(c) if isinstance(c, str) else c
            if cand is not None:
                _rquestions(ctx).append(PC.to_question(cand, ctx.created_real()))
    return len(rep.done), len(rep.done), ""


def st_targets(ctx: Ctx) -> tuple:
    try:
        from engine.research import targets as TG
    except ImportError as e:
        raise MissingModule("targets") from e
    ex = ctx.extra("questions.targets", "a matured DayInput (daily research target sources)")
    led = ctx.mod_state("targets", TG.TargetLedger)
    rep = TG.run_day(ex["day"], ctx.now, led, ctx.state.modules.get("priority"), ctx.state.cfg.seed)
    got = getattr(rep, "targets", None)
    if got is None:
        got = getattr(rep, "ranked", ())
    return len(ex["day"].predictions), len(got or ()), ""


def _graph(ctx: Ctx):
    from engine.research import research_graph as RGR
    return ctx.mod_state("research_graph", RGR.ResearchGraph)


def st_research_graph(ctx: Ctx) -> tuple:
    """research_graph.step: unanswered relationships in what the brain knows become questions (section 27 gap discovery)."""
    try:
        from engine.research import research_graph as RGR
    except ImportError as e:
        raise MissingModule("research_graph") from e
    g = _graph(ctx)
    rep = RGR.step(g, ctx.now, ctx.created_real())
    _rquestions(ctx).extend(rep.new_questions)
    return int(rep.nodes), len(rep.new_questions), f"{len(rep.gaps)} gaps, {rep.skipped_existing} already asked"


_SOURCE_MAP = {"surprise": "surprise", "contradiction": "contradiction", "loss": "loss", "missed_winner": "missed_winner",
               "missed_loser": "loss", "break": "pattern_break", "pattern_break": "pattern_break", "regime": "regime_change",
               "regime_change": "regime_change", "discovery": "new_discovery", "new_discovery": "new_discovery", "gap": "coverage_gap",
               "coverage": "coverage_gap", "autopsy": "surprise", "data": "data_anomaly", "failure": "research_failure"}


def event_from_research_question(rq: ResearchQuestion, now):
    """A module's ResearchQuestion as a QuestionEvent for the ONE question generator (so every question gets hypotheses, a test plan,
    a value vector and a priority, and duplicates merge). The subject is the module's own identity-free text."""
    from engine.research import questions as Q
    src = next((v for k, v in _SOURCE_MAP.items() if k in str(rq.source).lower()), "surprise")
    ev = str(rq.evidence_through)
    require_past(ev, now, f"research question {rq.question_id}")
    mag = rq.expected.decision_value if rq.expected.decision_value is not None else 0.5
    return Q.QuestionEvent(src, rq.text[:160], ev, float(min(1.0, max(0.0, mag))), problem=rq.problem,
                           loss_share=float(min(1.0, max(0.0, rq.expected.loss_reduction_value or 0.0))))


def st_generate(ctx: Ctx) -> tuple:
    """questions.generate over this cycle's events, the questions other modules raised, and the events last cycle's results raised."""
    from engine.research import questions as Q
    from engine.research import priority as PRI
    events = list(ctx.state.carry_events) + list(ctx.bus.get("events", []))
    future = [e for e in events if as_date(e.evidence_through) >= as_date(ctx.now)]
    if future:
        ctx.state.memo.setdefault("quarantined_events", []).extend((ctx.now, _event_key(e)) for e in future)
        ctx.state.carry_events = [e for e in ctx.state.carry_events if e not in future]
        raise FirewallBreach(f"{len(future)} question event(s) rest on evidence dated on/after now={ctx.now} (quarantined): "
                             + "; ".join(_event_key(e) for e in future[:3]))
    converted, refused = [], []
    for rq in ctx.bus.get("research_questions", []):
        try:
            converted.append(event_from_research_question(rq, ctx.now))
        except FirewallBreach:
            raise
        except Exception as e:                            # noqa: BLE001 - one malformed module question must not stop the rest
            refused.append((getattr(rq, "question_id", "?"), str(e)[:120]))
    events += converted
    if not events:
        ctx.bus["new_questions"] = []
        raise NoInput("no events or module questions this cycle")
    led = ctx.mod_state("question_ledger", Q.QuestionLedger)
    pst = ctx.mod_state("priority", PRI.new_state)
    rep = Q.generate(events, ctx.now, led, pst, max_new=ctx.state.cfg.max_new_questions)
    new = []
    L = ctx.state.lineage
    for qo in rep.questions:
        is_new = qo.qid not in ctx.state.questions
        ctx.state.questions[qo.qid] = qo
        qn = L.add("QUESTION", qo.qid, ctx.cycle, ctx.now, source=qo.source, problem=qo.question.problem.value, text=qo.question.text[:100])
        for e in events:
            ek = _event_key(e)
            en = ctx.state.q_nodes.get(ek)
            if en and e.subject == qo.subject:
                L.link(en, qn, "asked_as")
        if is_new:
            new.append(qo.qid)
    ctx.state.carry_events = []
    ctx.bus["new_questions"] = new
    ctx.state.count("questions_generated", len(new))
    ctx.state.count("questions_merged", rep.merged)
    return len(events), len(new), f"{rep.merged} merged, {len(rep.refused) + len(refused)} refused, {len(rep.skipped_answered)} already answered"


def _event_key(e) -> str:
    return f"{e.source}|{e.subject}|{e.evidence_through}"


# ================================================================================================================ HYPOTHESES
def st_trees(ctx: Ctx) -> tuple:
    """A hypothesis tree per new question (a duplicate question joins the existing tree), then hypothesis_tree.step with the results
    queued since the last step."""
    try:
        from engine.research import hypothesis_tree as HT
    except ImportError as e:
        raise MissingModule("hypothesis_tree") from e
    forest = ctx.mod_state("hypothesis_tree", HT.TreeForest)
    added = 0
    for qid in ctx.bus.get("new_questions", []):
        qo = ctx.state.questions[qid]
        tid = "t_" + qid
        if tid in forest.trees or forest.find_similar(qo.question.text):
            continue
        try:
            forest.add(qo.to_tree(ctx.now))
            added += 1
        except HT.TreeError as e:
            ctx.state.count("trees_refused")
            ctx.bus.setdefault("notes", []).append(f"tree for {qid} refused: {e}")
    results = [r for r in ctx.bus.get("tree_results", []) if as_date(r.evidence_through) < as_date(ctx.now)]
    fs = HT.step(forest, results, ctx.now)
    ctx.bus["tree_results"] = []
    ctx.bus["tree_actions"] = [(a.tree_id, a.nid, round(a.eig_bits, 4)) for a in fs.actions]
    for tid in fs.resolved:
        ctx.state.count("trees_resolved")
    return len(ctx.bus.get("new_questions", [])), added, f"{len(fs.killed)} branches killed, {len(fs.resolved)} resolved"


# ================================================================================================================ INFORMATION GAIN
def _apply_objective_weights(pst, weights) -> None:
    """Rebuild the priority model on its own history with the controller's objective weights (the same rebuild priority.refine_weights
    does), so the learned model and its prior stay consistent."""
    from engine.research import priority as PRI
    old = pst.model
    m = PRI.PriorityModel(pst.cfg, weights)
    m.label = old.label
    for rv in pst.history:
        m.observe(rv)
    pst.model = m


def st_priority(ctx: Ctx) -> tuple:
    """priority.step over every open question not yet on the ladder; realised values of finished experiments (matured before now)
    teach the priority model first. The plan's selection is what ALLOCATE may fund."""
    try:
        from engine.research import priority as PRI
    except ImportError as e:
        raise MissingModule("priority") from e
    from engine.learning.research_policy import ComputeBudget
    pst = ctx.mod_state("priority", PRI.new_state)
    dec = ctx.bus.get("controller_prev")
    if dec is not None:
        _apply_objective_weights(pst, CT.objective_weights(dec, pst.weights()))
    matured = [r for r in ctx.state.pending_realised if as_date(r.matured_at) < as_date(ctx.now)]
    ctx.state.pending_realised = [r for r in ctx.state.pending_realised if r not in matured]
    items = []
    for qid, qo in sorted(ctx.state.questions.items()):
        if qid in ctx.state.branch_of:
            continue
        led = ctx.state.modules.get("question_ledger")
        if led is not None and led.fate(qid) not in ("OPEN", "NONE"):
            continue
        it = qo.to_item(qo.question.evidence_through)
        ctx.state.items[it.item_id] = it
        items.append(it)
    if not items and not matured:
        raise NoInput("no open questions to prioritise and no realised values to learn from")
    free = ctx.rt.free_gb()
    plan = PRI.step(pst, ctx.now, items, matured, ComputeBudget(ctx.state.cfg.budget_cpu_min, free or 0.0), ctx.state.cfg.seed)
    ctx.bus["plan_selected"] = list(plan.selected)
    ctx.bus["plan_share_by_problem"] = dict(plan.share_by_problem)
    for iid in plan.selected:
        it = ctx.state.items[iid]
        ctx.state.lineage.add("ITEM", iid, ctx.cycle, ctx.now, problem=it.problem.value)
        ctx.state.lineage.link(f"QUESTION:{it.question_id}", f"ITEM:{iid}", "prioritised_as")
    return len(items), len(plan.selected), f"learned from {len(matured)} results; trust {plan.model_trust:.2f}; {plan.validation}"


# ================================================================================================================ ALLOCATE
def st_controller(ctx: Ctx) -> tuple:
    """Section 51: capability readings (two-stage evaluation, matured strictly before now) and realised yields -> phase shares and
    the section-50 statements. The decision re-weights priority (next cycle), the ladder's problem weights and diversity (this cycle)."""
    st = ctx.state
    if st.controller is None:
        st.controller = CT.new_state()
    reads = [r for r in st.readings if as_date(r.as_of) < as_date(ctx.now)]
    st.readings = [r for r in st.readings if r not in reads]
    ylds = [y for y in st.pending_yields if as_date(y.as_of) < as_date(ctx.now)]
    st.pending_yields = [y for y in st.pending_yields if y not in ylds]
    dec = CT.step(st.controller, ctx.now, reads, ylds, ctx.bus.get("loss_sources"))
    ctx.bus["controller"] = dec
    ctx.bus["controller_prev"] = dec
    st.statements = (st.statements + [s.to_dict() | {"now": ctx.now} for s in dec.statements])[-200:]
    return len(reads) + len(ylds), len(dec.statements), dec.statements[0].text if dec.statements else ""


def st_diversity(ctx: Ctx) -> tuple:
    try:
        from engine.research import diversity as DV
    except ImportError as e:
        raise MissingModule("diversity") from e
    ctrl = ctx.mod_state("diversity", DV.DiversityController)
    outs = [o for o in ctx.state.outcomes if as_date(o.when) < as_date(ctx.now)]
    dec = ctx.bus.get("controller")
    dirs = CT.directives(dec) if dec is not None else None
    if dirs is not None and ctx.bus.get("brain_directives") is not None:
        dirs = CT.merge_directives(dirs, ctx.bus["brain_directives"])
    seen = ctx.state.memo.setdefault("diversity_seen", set())
    fresh = [o for o in outs if o.exp_id not in seen]
    plan = DV.step(ctrl, ctx.now, fresh, ctx.state.cfg.budget_cpu_min, ctx.state.cfg.seed, directives=dirs)
    seen.update(o.exp_id for o in fresh)
    ctx.bus["diversity_shares"] = {str(k): float(v) for k, v in plan.shares.items()}
    return len(fresh), len(plan.shares), ""


def _ladder_policy(ctx: Ctx):
    from engine.research import compute_manager as CM
    pol = CM.LadderPolicy(min_effect=ctx.state.cfg.experiment.min_effect)
    dec = ctx.bus.get("controller")
    if dec is not None:
        pol = dataclasses.replace(pol, problem_weight=CT.problem_weight(dec))
    return pol


def st_compute(ctx: Ctx) -> tuple:
    """compute_manager: a branch per selected question, then one scheduling tick (period roll, explore floor, rung share caps, RAM
    admission that fails closed, one real-data job at a time). Launched allocations become jobs; a design already run on the same
    data is refused as a duplicate."""
    try:
        from engine.research import compute_manager as CM
    except ImportError as e:
        raise MissingModule("compute_manager") from e
    from engine.learning.research_policy import ComputeBudget
    ms = ctx.mod_state("compute_manager", CM.ManagerState)
    pol = _ladder_policy(ctx)
    made = 0
    for iid in ctx.bus.get("plan_selected", []):
        it = ctx.state.items[iid]
        qo = ctx.state.questions.get(it.question_id)
        if qo is None or it.question_id in ctx.state.branch_of:
            continue
        b = CM.make_branch(ms, qo.question, ctx.now, it.family)
        if b is None:
            ctx.state.count("duplicate_branches")
            continue
        ctx.state.branch_of[it.question_id] = b.branch_id
        ctx.state.question_of_branch[b.branch_id] = it.question_id
        ctx.state.lineage.add("BRANCH", b.branch_id, ctx.cycle, ctx.now, problem=b.problem.value)
        ctx.state.lineage.link(f"ITEM:{iid}", f"BRANCH:{b.branch_id}", "allocated_to")
        made += 1
    free = ctx.rt.free_gb()
    if free is None:
        raise NoInput("free memory unreadable: nothing may launch (fail closed)")
    ex = ctx.rt.executor
    res = CM.step(ms, ctx.now, pol, (), ComputeBudget(ctx.state.cfg.budget_cpu_min, free), free, ctx.state.cfg.period_cpu_min,
                  ex.ledger, ctx.rt.code_hash, ctx.state.cfg.seed)
    ctx.bus["launched"] = list(res.launched)
    ctx.bus["launch_cost"] = {a.branch_id: float(a.cpu_min) for a in res.selection.allocations}
    ctx.bus["deferred"] = [list(d) for d in res.selection.deferred[:20]]
    return made, len(res.launched), f"{len(res.selection.deferred)} deferred"


# ================================================================================================================ RUN
def st_run(ctx: Ctx) -> tuple:
    """Turn every launched ladder allocation into an ExperimentTask and hand it to the executor. The task may only use outcomes
    matured strictly before now (cutoff = the observation's evidence date)."""
    from engine.research import compute_manager as CM
    ms = ctx.state.modules.get("compute_manager")
    launched = ctx.bus.get("launched", [])
    held = ctx.state.memo.setdefault("held_for_fresh_data", {})
    if (not launched and not held) or ms is None:
        raise NoInput("nothing was launched this cycle and nothing is waiting for fresh data")
    obs = ctx.obs
    cutoff = ctx.evidence_date()
    ec = ctx.state.cfg.experiment
    n = _release_held(ctx, ms, held, cutoff)
    for bid in launched:
        b = ms.branches[bid]
        key = b.in_flight
        spec = ctx.rt.executor.ledger.spec(key)
        qid = ctx.state.question_of_branch.get(bid, b.question_id)
        qo = ctx.state.questions.get(qid)
        text = f"{qo.subject} {qo.question.text}" if qo is not None else b.text
        feat = resolve_feature(text, b.problem, obs.matured.columns)
        dkey = stable_hash([design_kind(b.problem), feat, b.frontier.value, b.repeats.get(b.frontier.value, 0), obs.data_hash], 16)
        task = ExperimentTask(key, bid, b.problem.value, feat, b.frontier.value, cutoff, ctx.state.seen_through.get(bid, ""),
                              int(spec.seed) + ctx.state.cfg.seed, dataclasses.asdict(ec), int(b.repeats.get(b.frontier.value, 0)),
                              int(b.n_obs_planned or 0))
        rec = JobRecord(key, bid, qid, f"r_{qid}", b.problem.value, feat, b.frontier.value, cutoff, task.seen_through, ctx.now, ctx.cycle,
                        task.to_dict(), float(ctx.bus.get("launch_cost", {}).get(bid, 0.0)))
        ctx.state.jobs[key] = rec
        L = ctx.state.lineage
        L.add("JOB", key, ctx.cycle, ctx.now, stage=b.frontier.value, feature=feat)
        L.link(f"BRANCH:{bid}", f"JOB:{key}", "ran")
        other = ctx.state.designs.get(dkey)
        if other is not None and other != key:
            _refuse_job(ctx, ms, rec, f"duplicate design of {other[:10]} on the same data")
            ctx.state.count("duplicate_designs")
            led = ctx.state.modules.get("question_ledger")
            if led is not None and led.latest(qid) is not None and led.fate(qid) == "OPEN":
                led.set_fate(qid, "MERGED", ctx.now)
            ctx.state.lineage.link(f"JOB:{key}", f"JOB:{other}", "duplicate_of")
            continue
        ctx.state.designs[dkey] = key
        if not feat:
            _refuse_job(ctx, ms, rec, "no derivable feature for the question: nothing is guessed")
            continue
        if b.frontier.value in (Stage.FRESH_HOLDOUT.value, Stage.INTEGRATION.value):
            k, need = fresh_dates(obs.matured, task.seen_through), fresh_needed(task, ec, b, _ladder_policy(ctx))
            if k < need:                                  # W02: a holdout needs data the branch never saw; wait for it, never park
                held[key] = task.to_dict()
                rec.error = f"waiting for fresh data: {k} of {need} matured dates after {task.seen_through or 'the start'}"
                ctx.state.count("held_for_fresh_data")
                continue
        st = ctx.rt.executor.submit(spec, task, obs.matured, obs.data_hash, ctx.now)
        rec.state = JobState.DONE if st == C.DONE else JobState.RUNNING if st == "RUNNING" else JobState.FAILED
        rec.attempts += 1
        n += 1
    return len(launched), n, f"mode {ctx.state.cfg.mode}; {ctx.rt.executor.running()} still running"


def fresh_dates(matured: pd.DataFrame, seen_through: str) -> int:
    """Matured decision dates strictly after the newest date a branch's earlier rungs used: the fresh holdout's sample."""
    if len(matured) == 0:
        return 0
    d = pd.to_datetime(pd.unique(matured.index.get_level_values(0)))
    return int((d > pd.Timestamp(as_date(seen_through))).sum()) if seen_through else int(len(d))


def fresh_needed(task: ExperimentTask, ec: ExperimentConfig, branch=None, policy=None) -> int:
    """Fresh dates a holdout / integration rung waits for: the configured minimum; the sample the ladder planned after an underpowered
    run (a REPEAT's `n_planned`); and the sample that gives the rung its required power at the smallest useful effect, with the
    per-date spread the branch measured on its earlier rungs. Waiting costs no compute; running earlier only produces an
    underpowered result that the ladder must park as INFEASIBLE (observed on the planted world: 5 fresh dates, power 0.13)."""
    need = max(int(ec.fresh_min_dates), int(task.n_planned or 0))
    if branch is not None and policy is not None:
        from engine.learning.research_policy import required_n
        sds = [float(r["se"]) * math.sqrt(float(r["n_obs"])) for r in getattr(branch, "runs", [])
               if r.get("se") not in (None, "") and r.get("n_obs") and float(r["n_obs"]) >= 2 and float(r["se"]) > 0]
        if sds:
            rule = policy.rule(Stage(task.stage))
            need = max(need, int(required_n(policy.min_effect, max(sds[-3:]), policy.alpha, rule.min_power)))
    return need


def _release_held(ctx: Ctx, ms, held: dict, cutoff: str) -> int:
    """Holdout / integration rungs launched before enough fresh data existed run as soon as it does, on the data of THIS cycle (the
    cutoff moves forward; the rung, its seed and its seen_through do not). A branch that moved on or was cancelled drops its hold."""
    n = 0
    ec = ctx.state.cfg.experiment
    for key in sorted(held):
        t = held[key]
        rec = ctx.state.jobs.get(key)
        b = ms.branches.get(t["branch_id"])
        if rec is None or b is None or b.in_flight != key or rec.state != JobState.SUBMITTED:
            held.pop(key)
            continue
        if fresh_dates(ctx.obs.matured, t["seen_through"]) < fresh_needed(ExperimentTask(**t), ec, b, _ladder_policy(ctx)):
            continue
        task = ExperimentTask(**{**t, "cutoff": cutoff})
        rec.cutoff, rec.task, rec.error = cutoff, task.to_dict(), ""
        st = ctx.rt.executor.submit(ctx.rt.executor.ledger.spec(key), task, ctx.obs.matured, ctx.obs.data_hash, ctx.now)
        rec.state = JobState.DONE if st == C.DONE else JobState.RUNNING if st == "RUNNING" else JobState.FAILED
        rec.attempts += 1
        held.pop(key)
        ctx.state.count("released_from_hold")
        n += 1
    return n


def _refuse_job(ctx: Ctx, ms, rec: JobRecord, why: str) -> None:
    """A launched job the loop will not run: the compute-ledger entry is closed as FAILED and the branch parked with the reason (a
    parked branch is DORMANT, revivable by the waste controller; never deleted)."""
    from engine.research import compute_manager as CM
    rec.state, rec.error = JobState.REFUSED, why
    led = ctx.rt.executor.ledger
    e = led.get(rec.key)
    if e is not None and e.get("state") == C.PENDING:
        led.claim(rec.key, "research-loop-refusal", _epoch(ctx.now), ctx.rt.code_hash)
        led.fail(rec.key, C.FAILED, "refused by the research loop: " + why, _epoch(ctx.now))
    b = ms.branches[rec.branch_id]
    b.in_flight = ""
    if b.state.value not in ("DORMANT", "FAILED", "RETIRED", "CANCELLED"):
        CM.park(ms, rec.branch_id, ctx.now, why)


# ================================================================================================================ VALIDATE
def st_harvest(ctx: Ctx) -> tuple:
    """Collect finished jobs through compute.reconcile (result hash, spec, seed and code must all verify); every result's data must end
    strictly before now (fail closed) and must carry the fields a rung needs."""
    ex = ctx.rt.executor
    ex.finished()
    accepted, rejected = ex.results()
    todo = [r for r in ctx.state.jobs.values() if r.state in (JobState.RUNNING, JobState.DONE, JobState.SUBMITTED)]
    got = []
    for rec in todo:
        if rec.key in accepted:
            res = dict(accepted[rec.key])
            if res.get("data_through"):
                require_past(res["data_through"], ctx.now, f"experiment {rec.key[:10]} result")
            rec.result, rec.state, rec.harvested_at = res, JobState.HARVESTED, ctx.now
            got.append(rec.key)
        elif rec.key in rejected:
            rec.state, rec.error = JobState.FAILED, f"reconcile rejected: {rejected[rec.key][0]}"
        elif ex.ledger_state(rec.key) in (C.FAILED, C.GAVE_UP, C.OOM, C.CRASHED):
            rec.state, rec.error = JobState.FAILED, f"compute ledger: {ex.ledger_state(rec.key)}"
    ctx.bus["harvested"] = got
    return len(todo), len(got), f"{ex.running()} still running"


def st_controls(ctx: Ctx) -> tuple:
    """Every result is compared with its shuffled-label control (same code path, same rows). A control that 'finds' an effect means
    the design leaks: the result is refused as evidence, the branch is parked with the reason and a data question is raised. A result
    that could not be evaluated is refused the same way (never read as a null). Suspiciously strong results with a clean control go
    to the ladder, whose AUDIT is then resolved by `_leakage_audit`."""
    ms = ctx.state.modules.get("compute_manager")
    ec = ctx.state.cfg.experiment
    ok, bad = [], []
    for key in ctx.bus.get("harvested", []):
        rec = ctx.state.jobs[key]
        r = rec.result or {}
        cse = r.get("control_se") or 0.0
        ct = (r.get("control_effect", 0.0) / cse) if cse > 0 else 0.0
        r["control_t"] = float(ct)
        if not r.get("ok", False):
            bad.append(key)
            if ms is not None and rec.branch_id in ms.branches:
                _refuse_job(ctx, ms, rec, "not evaluable: " + str(r.get("why", "?")))
            rec.state = JobState.FAILED
            continue
        if abs(ct) > ec.control_z:
            bad.append(key)
            ctx.state.count("control_failures")
            if ms is not None and rec.branch_id in ms.branches:
                _refuse_job(ctx, ms, rec, f"shuffled-label control t={ct:.2f}: the design finds structure in noise")
            rec.state = JobState.CONTROL_FAILURE
            _carry(ctx, "data_anomaly", f"feature {rec.feature} control failure", 0.8, Problem.DATA_QUALITY, key)
            continue
        ok.append(key)
    ctx.bus["validated"] = ok
    return len(ctx.bus.get("harvested", [])), len(ok), f"{len(bad)} refused"


def _carry(ctx: Ctx, source: str, subject: str, magnitude: float, problem: Problem, job_key: str, **kw) -> None:
    """Queue an event raised by a result for next cycle's question generation, with lineage RESULT -> EVENT."""
    from engine.research import questions as Q
    rec = ctx.state.jobs.get(job_key)
    when = (rec.result or {}).get("data_through") if rec else None
    when = when or ctx.evidence_date()
    ev = Q.QuestionEvent(source, subject, when, float(min(1.0, max(0.0, magnitude))), problem=problem, **kw)
    ctx.state.carry_events.append(ev)
    ek = _event_key(ev)
    en = ctx.state.lineage.add("EVENT", ek, ctx.cycle, ctx.now, source=source)
    ctx.state.q_nodes[ek] = en
    ctx.state.lineage.link(f"RESULT:{job_key}", en, "raised")


# ================================================================================================================ LEARN
def st_ladder(ctx: Ctx) -> tuple:
    """compute_manager.record_result per validated result: ADVANCE / COMPLETE / REPEAT / DEMOTE / FAIL / PARK / AUDIT."""
    from engine.research import compute_manager as CM
    ms = ctx.state.modules.get("compute_manager")
    val = ctx.bus.get("validated", [])
    if ms is None or not val:
        raise NoInput("no validated results")
    pol = _ladder_policy(ctx)
    acts = {}
    L = ctx.state.lineage
    for key in sorted(val):
        rec = ctx.state.jobs[key]
        r = rec.result
        b = ms.branches.get(rec.branch_id)
        if b is None or b.frontier.value != rec.stage or b.in_flight != key:
            rec.error = "branch moved on before the result arrived"
            continue
        cost = rec.cost or CM.planned_cost(ms, b, pol)
        dec = CM.record_result(ms, rec.branch_id, evidence_from_result(r, cost), ctx.now, pol)
        r["action"], r["cost"] = dec.action.value, cost
        if dec.action == CM.Act.AUDIT:
            _leakage_audit(ctx, ms, rec, r)
        acts[dec.action.value] = acts.get(dec.action.value, 0) + 1
        if r.get("seen_through") and rec.stage in (Stage.CHEAP_SCREEN.value, Stage.STRONGER_TESTS.value, Stage.CROSS_YEAR.value):
            prev = ctx.state.seen_through.get(rec.branch_id, "")
            ctx.state.seen_through[rec.branch_id] = max(prev, r["seen_through"])
        L.add("RESULT", key, ctx.cycle, ctx.now, action=dec.action.value, effect=round(float(r.get("effect", 0.0)), 6),
              t=round(float(r["effect"] / r["se"]), 3) if r.get("se") else None)
        L.link(f"JOB:{key}", f"RESULT:{key}", "produced")
    ctx.bus["ladder_actions"] = acts
    return len(val), sum(acts.values()), ", ".join(f"{k}={v}" for k, v in sorted(acts.items()))


def _leakage_audit(ctx: Ctx, ms, rec: JobRecord, r: Mapping[str, Any]) -> None:
    """The ladder holds a too-good result for a leakage audit. The audit the loop can run is the design's own shuffled-label control
    plus the point-in-time checks already passed (cutoff strictly before now, result data through < now): a clean control clears the
    audit so the rung is re-run as reproduced evidence; a control that finds structure keeps the branch held and raises a data
    question. The audit never promotes anything by itself."""
    from engine.research import compute_manager as CM
    ct = abs(float(r.get("control_t", 0.0)))
    clean = ct <= ctx.state.cfg.experiment.control_z and as_date(r.get("data_through") or rec.cutoff) < as_date(ctx.now)
    if clean:
        CM.clear_audit(ms, rec.branch_id, ctx.now, leak_found=False,
                       note=f"shuffled-label control t={ct:.2f}; data through {r.get('data_through')} < {ctx.now}")
        ctx.state.count("audits_cleared")
    else:
        CM.clear_audit(ms, rec.branch_id, ctx.now, leak_found=True, note=f"control t={ct:.2f}: suspected leak")
        ctx.state.count("audits_failed")
        _carry(ctx, "data_anomaly", f"feature {rec.feature} leakage audit", 0.9, Problem.DATA_QUALITY, rec.key)


def _tree_outcome(tree, nid: str, favoured: str) -> str | None:
    """The outcome label most expected under the hypothesis the evidence favours (argmax of that hypothesis's likelihood row)."""
    try:
        row = tree.get(nid).likelihood_row(favoured)
    except Exception:                                   # noqa: BLE001 - an unknown node is simply not recordable
        return None
    return max(sorted(row), key=lambda o: row[o]) if row else None


def st_memory(ctx: Ctx) -> tuple:
    """Every validated result is written to the ONE experiment ledger (engine.learning.experiment_memory via the section-15 front
    door), the hypothesis tree receives the matching outcome, and the result raises its follow-up question."""
    from engine.research import hypothesis_tree as HT
    val = ctx.bus.get("validated", [])
    if not val:
        raise NoInput("no validated results")
    mem = _experiment_memory(ctx)
    forest = ctx.state.modules.get("hypothesis_tree")
    written, refused = 0, 0
    L = ctx.state.lineage
    for key in sorted(val):
        rec = ctx.state.jobs[key]
        r = rec.result
        qo = ctx.state.questions.get(rec.question_id)
        if qo is None or "action" not in r:
            continue
        t = r["effect"] / r["se"] if r.get("se") else 0.0
        favoured = _favoured(qo, r["action"])
        tree = forest.trees.get("t_" + qo.qid) if forest is not None else None
        pend = tree.pending() if tree is not None else []
        outcome = _tree_outcome(tree, pend[0].nid, favoured) if (pend and favoured) else None
        try:
            eid = _ledger_record(ctx, mem, rec, qo, r, t, outcome)
            written += 1
            L.link(f"RESULT:{key}", L.add("MEMORY", eid, ctx.cycle, ctx.now), "remembered_as")
        except Exception as e:                           # noqa: BLE001 - the ledger refusing a record is a reported finding
            refused += 1
            ctx.bus.setdefault("notes", []).append(f"experiment memory refused {key[:10]}: {type(e).__name__}: {str(e)[:160]}")
        if outcome is not None:
            HT.step(forest, [HT.TestResult(tree.tree_id, pend[0].nid, outcome, r["data_through"])], ctx.now)
            ctx.state.count("tree_results")
        _raise_followups(ctx, rec, r, t)
    return len(val), written, f"{refused} refused by the ledger"


def _favoured(qo, action: str) -> str | None:
    """The hypothesis a rung's verdict points to: a pass favours the leading non-chance explanation, a powered null the chance
    hypothesis; a repeat / park / audit favours nothing (an underpowered test is not evidence either way)."""
    hyps = qo.hypotheses
    if action in ("ADVANCE", "COMPLETE"):
        return max((h for h in hyps if h.kind != "noise"), key=lambda h: (h.prior, h.hid), default=hyps[0]).hid
    if action == "FAIL":
        return next((h.hid for h in hyps if h.kind == "noise"), None)
    return None


def _ledger_record(ctx: Ctx, mem, rec: JobRecord, qo, r: Mapping[str, Any], t: float, outcome: str | None) -> str:
    """Register (through the section-15 launch gate) and close one rung in the experiment ledger. Each rung is its own experiment
    question; a repeat of a rung must justify itself (the gate VERIFIES the claimed fresh period / changed data against the design)."""
    from engine.learning import experiment_memory as EM
    from engine.research import experiments as EX
    start = str(ctx.obs.matured.index.get_level_values(0).min().date())
    design = EM.DesignSpec(config={"feature": rec.feature, "stage": rec.stage, "problem": rec.problem}, windows=((start, rec.cutoff),),
                           seed=int(rec.task["seed"]), controls=("shuffled_labels",), cost_minutes=float(r.get("cost", 0.0)),
                           target=rec.problem, data_hash=ctx.obs.data_hash)
    question = f"{qo.question.text} [rung {rec.stage.lower()}]"
    erec = EM.ExperimentRecord(f"E{rec.key[:16]}", 1, EM.ExperimentStatus.PROPOSED, question, "unknown until tested", qo.hypotheses,
                               EM.Prediction(f"{rec.feature} ranks outcomes at {rec.stage.lower()}", metric=str(r.get("kind", "")), direction="up"),
                               design, qo.expected, ctx.now, ctx.now, tags=(rec.stage, rec.problem))
    ext = EX.DesignExtension(dataset="research_panel", time_period=EX.TimePeriod(start, rec.cutoff), information_cutoff=rec.cutoff,
                             features=(rec.feature,), representation="derived_feature_rank", algorithm=str(r.get("kind", "rank_auc")),
                             hyperparameters={"stage": rec.stage}, complexity=EX.Complexity(1, 1, 1, int(r.get("n_rows", 0))),
                             expected_value=qo.value)
    claims = (EX.ReplicationReason.DATA_CHANGED, EX.ReplicationReason.FRESH_PERIOD)
    stamped, _ = mem.register(erec, ext, ctx.now, claims=claims, difference_statement=f"rung {rec.stage} on data through {rec.cutoff}",
                              current_code=ctx.rt.code_hash, current_data=ctx.obs.data_hash)
    act = r["action"]
    kind = EM.ResultKind.CONFIRMED if act in ("ADVANCE", "COMPLETE") else EM.ResultKind.REFUTED if act == "FAIL" else EM.ResultKind.NULL
    se = float(r["se"]) if r.get("se") else 0.0
    res = EM.ExperimentResult(kind, outcome or "inconclusive", {"effect": float(r["effect"]), "se": se, "t": float(t)}, int(r.get("n_obs", 0)),
                              (float(r["effect"]) - 1.96 * se, float(r["effect"]) + 1.96 * se),
                              power_note=f"n={r.get('n_obs', 0)} dates; ladder action {act}", observed_at=ctx.now,
                              summary=f"{rec.feature} {rec.stage}: effect {float(r['effect']):.4f} (t={t:.2f})")
    from engine.research.core import FailureCause
    outc = EX.OutcomeExtension(FailureCause.UNKNOWN if kind == EM.ResultKind.REFUTED else None, compute_cost_minutes=float(r.get("cost", 0.0)))
    mem.close(stamped.experiment_id, res, outc, ctx.now, [f"{rec.stage}: {act} (effect {float(r['effect']):.4f}, t={t:.2f})"],
              ["transfer to unseen stocks and regimes is not settled by one rung"], _next_action(act))
    return stamped.experiment_id


def _next_action(action: str) -> str:
    return {"ADVANCE": "run the next rung", "COMPLETE": "quality gate and replication before any promotion",
            "REPEAT": "repeat the rung with a larger sample", "FAIL": "record the failure; redirect compute",
            "PARK": "park as DORMANT until new data or a new representation arrives", "DEMOTE": "re-earn the earlier rung",
            "AUDIT": "leakage audit before anything else is spent"}.get(action, "review")


def _raise_followups(ctx: Ctx, rec: JobRecord, r: Mapping[str, Any], t: float) -> None:
    """Section 47 'new research questions generated': a passed rung asks whether the finding generalises (and, for volatility, whether
    it carries direction); a powered null asks why the idea failed; a park records the unresolved question."""
    act = r["action"]
    if act in ("ADVANCE", "COMPLETE") and t >= 2.0:
        _carry(ctx, "new_discovery", f"feature {rec.feature} {rec.problem.lower()} rung {rec.stage.lower()}", min(1.0, t / 6.0),
               Problem.parse(rec.problem), rec.key, n_obs=int(r.get("n_obs", 0)))
        if Problem.parse(rec.problem) == Problem.VOLATILITY and f"{rec.feature}|DIRECTION" not in ctx.state.screened:
            _carry(ctx, "coverage_gap", f"feature {rec.feature} direction among predicted movers", 0.4, Problem.DIRECTION, rec.key)
            ctx.state.screened[f"{rec.feature}|DIRECTION"] = ctx.now
    elif act == "FAIL":
        _carry(ctx, "research_failure", f"feature {rec.feature} {rec.problem.lower()} null", 0.3, Problem.RESEARCH_PROCESS, rec.key)
    elif act == "PARK":
        _carry(ctx, "coverage_gap", f"feature {rec.feature} {rec.problem.lower()} unresolved at {rec.stage.lower()}", 0.3,
               Problem.parse(rec.problem), rec.key)


def st_value(ctx: Ctx) -> tuple:
    """value_accounting.step per result; its realised value feeds priority (next cycle), the controller's yield and diversity /
    brain-health outcomes (all dated `now`, so they are read only by later cycles)."""
    try:
        from engine.research import value_accounting as VA
    except ImportError as e:
        raise MissingModule("value_accounting") from e
    from engine.research import brain_health as BH
    from engine.research import priority as PRI
    val = [k for k in ctx.bus.get("validated", []) if "action" in (ctx.state.jobs[k].result or {})]
    if not val:
        raise NoInput("no validated results")
    ledger = ctx.mod_state("value_accounting", VA.ValueLedger)
    jobs = []
    for key in val:
        rec = ctx.state.jobs[key]
        r = rec.result
        qo = ctx.state.questions.get(rec.question_id)
        jobs.append(VA.JobMeasurement(key, rec.branch_id, f"question:{qo.source if qo else 'unknown'}", Problem.parse(rec.problem),
                                      Stage(rec.stage), ctx.now, r["data_through"], float(r.get("cost", 1.0)), seed=int(rec.task["seed"]),
                                      code_hash=ctx.rt.code_hash, data_hash=ctx.obs.data_hash, prior_se=None, post_se=float(r["se"]) or None,
                                      powered_null=r["action"] == "FAIL", replications=int(r.get("replications", 0)),
                                      estimate=qo.value if qo else None))
    out = VA.step(ledger, jobs, ctx.now)
    it_of = {k: ctx.state.items.get(ctx.state.jobs[k].item_id) for k in val}
    for key in val:
        rec = ctx.state.jobs[key]
        r = rec.result
        t = r["effect"] / r["se"] if r.get("se") else 0.0
        passed = r["action"] in ("ADVANCE", "COMPLETE")
        gain = float(max(0.0, r["effect"])) if passed else 0.0
        it = it_of.get(key)
        if it is not None:
            realised = ExperimentValue(information_gain=float(min(3.0, abs(t) / 2.0)), decision_value=float(min(1.0, 10 * gain)),
                                       compute_cost=float(r.get("cost", 1.0)), volatility_value=float(min(1.0, 10 * gain)) if rec.problem == "VOLATILITY" else None,
                                       direction_value=float(min(1.0, 10 * gain)) if rec.problem == "DIRECTION" else None)
            ctx.state.pending_realised.append(PRI.RealisedValue(it, realised, max(float(r.get("cost", 1.0)), 1e-3), ctx.now,
                                                                survived_oos=None if rec.stage != Stage.FRESH_HOLDOUT.value else passed))
        phase = {"VOLATILITY": CT.Phase.VOLATILITY, "DIRECTION": CT.Phase.DIRECTION, "LOSS_AVOIDANCE": CT.Phase.CATASTROPHIC_LOSS,
                 "COVERAGE": CT.Phase.DIRECTION_AT_COVERAGE, "CONSISTENCY": CT.Phase.TRANSFER}.get(rec.problem, CT.Phase.DISCOVERY)
        ctx.state.pending_yields.append(CT.YieldRecord(phase, gain, max(float(r.get("cost", 1.0)), 1e-3), ctx.now,
                                                       transferable=passed if rec.stage == Stage.FRESH_HOLDOUT.value else None))
        area = {"VOLATILITY": "VOLATILITY", "DIRECTION": "DIRECTION", "LOSS_AVOIDANCE": "RISK", "DATA_QUALITY": "DATA_QUALITY"}.get(rec.problem, "UNCERTAIN")
        ctx.state.outcomes.append(BH.Outcome(key, ctx.now, area, f"feature:{rec.feature}", max(float(r.get("cost", 1.0)), 1e-3), passed,
                                             gain_bits=float(min(3.0, abs(t) / 2.0)), question_id=rec.question_id,
                                             claimed_discovery=r["action"] == "COMPLETE", exploratory=rec.stage == Stage.CHEAP_SCREEN.value))
    ctx.state.outcomes = ctx.state.outcomes[-2000:]
    return len(jobs), len(out.get("accounted", [])), f"net value {out.get('net_value', 0.0):.4f}"


def st_science(ctx: Ctx) -> tuple:
    """science_memory + research graph from the SAME result, then the fold: every tested feature is a graph pattern node (its id is
    the memory item), each rung an experiment node answering its question; the memory gets the proposal (with its falsifier) and a
    TESTED entry per rung that cites the experiment node. SM.step then FOLDS the graph into the memory (W02: the fold is idempotent
    under changing node versions and skips experiments the memory already cites, so nothing is counted twice) and audits it."""
    try:
        from engine.research import science_memory as SM
    except ImportError as e:
        raise MissingModule("science_memory") from e
    mem = ctx.mod_state("science_memory", SM.ScienceMemory)
    g = _graph(ctx)
    n = 0
    nodes = ctx.state.memo.setdefault("graph_patterns", {})
    for key in ctx.bus.get("validated", []):
        rec = ctx.state.jobs[key]
        r = rec.result
        if "action" not in r:
            continue
        item = f"{rec.problem.lower()}:{rec.feature}"
        try:
            if item not in nodes:
                nodes[item] = g.add_pattern(item, rec.cutoff, label=rec.feature, effect=float(r["effect"])).node_id
            pnode = nodes[item]
            enode = g.add_test(key[:16].lower(), pnode, r["action"] in ("ADVANCE", "COMPLETE"), rec.cutoff, float(r["effect"]),
                               r.get("kind", "")).node_id
            qo = ctx.state.questions.get(rec.question_id)
            if qo is not None:
                qnode = g.add_question(qo.question, (pnode,), rec.cutoff).node_id
                g.answer_question(qnode, key[:16].lower(), rec.cutoff, r["action"].lower())
            ctx.state.count("graph_tests")
        except Exception as e:                           # noqa: BLE001 - graph vocabulary refusals are reported
            ctx.bus.setdefault("notes", []).append(f"research graph refused: {str(e)[:120]}")
            continue
        proposed = ctx.state.memo.setdefault("science_items", [])
        if pnode not in proposed:
            fz = SM.Falsifier("effect_below", 0.0, f"out-of-sample per-date AUC of {rec.feature} falls to chance")
            try:
                mem.propose(pnode, rec.cutoff, f"{rec.feature} ranks {rec.problem.lower()} outcomes", "research_loop", fz)
                proposed.append(pnode)
            except Exception as e:                       # noqa: BLE001 - a refused proposal is reported, not fatal
                ctx.bus.setdefault("notes", []).append(f"science memory proposal refused: {str(e)[:120]}")
        try:
            mem.tested(pnode, rec.cutoff, f"{r.get('kind', 'rank_auc')} {rec.stage}", r["action"], int(r.get("n_obs", 0)),
                       float(r["effect"]), float(r["se"]), controls=("shuffled_labels",), holdout=rec.stage == Stage.FRESH_HOLDOUT.value,
                       evidence=(enode,))
            n += 1
        except Exception as e:                           # noqa: BLE001
            ctx.bus.setdefault("notes", []).append(f"science memory test entry refused: {str(e)[:120]}")
    before = len(mem)
    rep = SM.step(mem, ctx.now, graph=g, previous=ctx.state.memo.get("science_prev"))
    ctx.state.memo["science_prev"] = ctx.now
    folded = len(mem) - before
    ctx.state.count("science_folded", folded)
    return len(ctx.bus.get("validated", [])), n, (f"{rep.items} items; folded {folded} graph entries {dict(rep.ingested.get('graph', {}))}; "
                                                  f"{len(rep.findings)} audit findings ({len(rep.errors)} errors)")


def st_replication(ctx: Ctx) -> tuple:
    """replication.step: a branch that has passed the cross-year rung registers its discovery; later rungs (fresh holdout) are
    replication runs. Only REPLICATED discoveries clear of the same-year replay may change anything (section 32)."""
    try:
        from engine.research import replication as RP
    except ImportError as e:
        raise MissingModule("replication") from e
    led = ctx.mod_state("replication", _replication_ledger)
    known = set(led.discoveries())
    added = 0
    for key in ctx.bus.get("validated", []):
        rec = ctx.state.jobs[key]
        r = rec.result
        if r.get("action") not in ("ADVANCE", "COMPLETE"):
            continue
        did = f"D{rec.branch_id}"
        effs = tuple(float(x) for x in r.get("unit_effects", ()) or (r["effect"],))
        if rec.stage == Stage.CROSS_YEAR.value and did not in known:
            try:
                led.add_discovery(RP.Discovery(did, float(np.mean(effs)), float(np.std(effs) or r["se"]), len(effs), (rec.cutoff, rec.cutoff),
                                               ctx.state.cfg.experiment.horizon_days, frozenset({"panel"}), (int(rec.task["seed"]),),
                                               frozenset({"all"}), ctx.rt.code_hash, ctx.obs.data_hash, r["data_through"]))
                added += 1
                known.add(did)
            except Exception as e:                       # noqa: BLE001
                ctx.bus.setdefault("notes", []).append(f"replication discovery refused: {str(e)[:120]}")
        elif rec.stage in (Stage.FRESH_HOLDOUT.value, Stage.INTEGRATION.value) and did in known:
            try:
                led.add_run(RP.ReplicationRun(f"R{key[:12]}", did, (rec.seen_through or rec.cutoff, r["data_through"]), frozenset({"panel"}),
                                              int(rec.task["seed"]), "all", (float(r["effect"]),), (float(r.get("control_effect", 0.0)),),
                                              ctx.rt.code_hash, ctx.obs.data_hash, r["data_through"]), ctx.now)
            except Exception as e:                       # noqa: BLE001
                ctx.bus.setdefault("notes", []).append(f"replication run refused: {str(e)[:120]}")
    rep = RP.step(led, ctx.now, seed=ctx.state.cfg.seed)
    ctx.bus["replication_authorized"] = list(rep.authorized)
    ctx.bus["replication_assessments"] = dict(rep.assessments)
    return added, len(rep.authorized), f"{len(rep.status_changes)} status changes"


def st_quality_and_knowledge(ctx: Ctx) -> tuple:
    """A branch that COMPLETED the ladder goes through quality_gate.step with the FULL evidence bundle (engine.research.evidence:
    statistics, out-of-sample, identity, transfer, risk, calibration, complexity, failure behaviour, reproducibility, replication in
    the ONE ledger, the future-information audit, provenance). Only PROMOTE files it as knowledge for the firewall; anything else
    stays research-only with its verdict and blocking gates recorded (never promoted on a missing input)."""
    try:
        from engine.research import quality_gate as QG
    except ImportError as e:
        raise MissingModule("quality_gate") from e
    from engine.research import evidence as EV
    done = [k for k in ctx.bus.get("validated", []) if (ctx.state.jobs[k].result or {}).get("action") == "COMPLETE"]
    regate = ctx.state.memo.setdefault("regate", {})
    year = as_date(ctx.evidence_date()).year
    due = [v["key"] for d, v in sorted(regate.items()) if year > v["year"] and v["looks"] < REGATE_MAX_LOOKS and v["key"] not in done]
    done = done + due
    if not done:
        raise NoInput("no branch completed the ladder this cycle and no earlier finding gained a new year of evidence")
    store = ctx.mod_state("quarantine", QG.QuarantineStore)
    led = ctx.mod_state("replication", _replication_ledger)
    sci = set(ctx.state.memo.get("science_items", []))
    graph_nodes = ctx.state.memo.get("graph_patterns", {})
    bundles, by_key = [], {}
    for key in done:
        rec = ctx.state.jobs[key]
        r = rec.result
        spec = EV.FindingSpec(f"D{rec.branch_id}", rec.feature, 1.0 if float(r.get("sign", 1.0)) >= 0 else -1.0, rec.problem,
                              n_tests_searched=max(1, len(ctx.state.screened)),
                              has_falsifier=graph_nodes.get(f"{rec.problem.lower()}:{rec.feature}") in sci,
                              experiment_id=key[:16], run_id=ctx.state.cfg.run_id, seed=int(rec.task.get("seed", 0)))
        b = EV.assemble(ctx.obs.matured, spec, ctx.now, code_hash=ctx.rt.code_hash, data_hash=ctx.obs.data_hash,
                        created_real=ctx.created_real(), ledger=led)
        bundles.append(b)
        by_key[key] = b
    rep = EV.gate(bundles, ctx.now, ctx.rt.code_hash, store=store)
    promoted = set(rep.promoted)
    verdicts = EV.verdicts(rep)
    filed = 0
    for key in done:
        rec = ctx.state.jobs[key]
        did = f"D{rec.branch_id}"
        b = by_key[key]
        ctx.state.lineage.add("GATE", did, ctx.cycle, ctx.now, verdict=verdicts.get(did, "UNKNOWN"), feature=rec.feature,
                              problem=rec.problem, blocking=EV.blocking(rep, did), missing=dict(b.missing),
                              effect_test=b.parts.get("effect_test"), effect_train=b.parts.get("effect_train"))
        ctx.state.lineage.link(f"RESULT:{key}", f"GATE:{did}", "gated")
        ctx.bus.setdefault("gate_verdicts", {})[did] = {"verdict": verdicts.get(did, "UNKNOWN"), "blocking": EV.blocking(rep, did),
                                                         "missing": dict(b.missing), "feature": rec.feature}
        v = verdicts.get(did, "UNKNOWN")
        if did in promoted:
            filed += _file_knowledge(ctx, rec)
            regate.pop(did, None)
        elif v != "QUARANTINED":                          # integrity failures are never retried; evidence-limited ones are
            look = regate.get(did, {"looks": 0})["looks"] + 1
            regate[did] = {"key": key, "year": year, "looks": look}
            if look >= REGATE_MAX_LOOKS:
                regate.pop(did, None)
                ctx.state.count("gate_retired_after_looks")
        else:
            regate.pop(did, None)
    return len(done), filed, f"verdicts {verdicts}; {len(due)} re-gated on a new year of evidence"


# W02: a finding that completed the ladder before enough unseen calendar years existed (the gate needs two) FAILS for lack of time,
# not for lack of effect. It is looked at again once per NEW calendar year of matured evidence, at most this many times in all (every
# look is recorded on its GATE lineage node; bounded looks keep repeated testing from manufacturing a pass).
REGATE_MAX_LOOKS = 3


def _replication_ledger():
    from engine.research import replication as RP
    return RP.ReplicationLedger()


def _file_knowledge(ctx: Ctx, rec: JobRecord) -> int:
    """File a promoted finding in MATURED_RESEARCH_STATE, trader-shaped (features/lean/horizon), for the firewall to release."""
    from engine.research.namespaces import InfoKind, InfoObject
    r = rec.result
    fw = _firewall(ctx)
    lean = 0.0
    if rec.problem == Problem.DIRECTION.value:
        lean = float(np.clip(r.get("sign", 1.0) * min(1.0, 2 * abs(r["effect"])), -1, 1))
    payload = {"features": {rec.feature: float(r.get("sign", 1.0))}, "lean": lean, "horizon": ctx.state.cfg.experiment.horizon_days,
               "trader_kind": "pattern"}
    oid = "K" + stable_hash([rec.branch_id, rec.feature, rec.problem], 14)
    obj = InfoObject(oid, InfoKind.RESEARCH_RESULT, NAMESPACE, r["data_through"], payload, _prov(ctx, r["data_through"]),
                     origin="engine.research.loop")
    if oid in fw.store:
        return 0
    fw.store.put(obj)
    ctx.state.knowledge[oid] = {"feature": rec.feature, "sign": float(r.get("sign", 1.0)), "problem": rec.problem, "filed_at": ctx.now,
                                "effect": float(r["effect"]), "branch": rec.branch_id}
    ctx.state.lineage.add("KNOWLEDGE", oid, ctx.cycle, ctx.now, feature=rec.feature)
    ctx.state.lineage.link(f"RESULT:{rec.key}", f"KNOWLEDGE:{oid}", "filed_as")
    ctx.state.count("knowledge_filed")
    return 1


def st_bridge(ctx: Ctx) -> tuple:
    """decision_bridge.step: this cycle's promoted knowledge (and the priority requests it implies) is routed and audited."""
    try:
        from engine.research import decision_bridge as DB
    except ImportError as e:
        raise MissingModule("decision_bridge") from e
    new = [(k, v) for k, v in ctx.state.knowledge.items() if v["filed_at"] == ctx.now]
    if not new:
        raise NoInput("no new knowledge to route")
    b = ctx.mod_state("decision_bridge", DB.Bridge)
    ds = []
    for kid, k in new:
        ds.append(DB.Discovery(kid, f"{k['feature']} ranks {k['problem'].lower()} outcomes (effect {k['effect']:.4f})", "research_loop",
                               ctx.evidence_date(), _prov(ctx, ctx.evidence_date()), problem=Problem.parse(k["problem"]),
                               informational_note="research-only until released by the firewall"))
    rep = DB.step(b, ds, ctx.now)
    return len(ds), len(rep.entries), f"refused {len(rep.refused)}"


# ================================================================================================================ UPDATE PRIORITIES
def st_waste(ctx: Ctx) -> tuple:
    try:
        from engine.research import waste as WA
    except ImportError as e:
        raise MissingModule("waste") from e
    ms, vl = ctx.state.modules.get("compute_manager"), ctx.state.modules.get("value_accounting")
    if ms is None or vl is None:
        raise NoInput("no ladder or value ledger yet")
    book = ctx.mod_state("waste_book", WA.DormantBook)
    wc = WA.WorldContext(ctx.now, ctx.obs.data_hash, len(ctx.obs.matured))
    res = WA.step(ms, vl, book, wc, ctx.now, ladder=_ladder_policy(ctx), rng_seed=ctx.state.cfg.seed)
    return len(ms.branches), len(getattr(res, "parked", ()) or ()), ""


def st_brain_health(ctx: Ctx) -> tuple:
    try:
        from engine.research import brain_health as BH
    except ImportError as e:
        raise MissingModule("brain_health") from e
    outs = [o for o in ctx.state.outcomes if as_date(o.when) < as_date(ctx.now)]
    rep = BH.step(outs, ctx.now)
    ctx.bus["brain_directives"] = rep.directives
    ctx.bus["brain_level"] = str(rep.level)
    return len(outs), len(rep.findings), f"level {rep.level}"


def st_meta_research(ctx: Ctx) -> tuple:
    try:
        from engine.research import meta_research as MR
    except ImportError as e:
        raise MissingModule("meta_research") from e
    st = ctx.mod_state("meta_research", MR.MetaResearchState)
    seen = ctx.state.memo.setdefault("meta_seen", set())
    rows = []
    for rec in ctx.state.jobs.values():
        r = rec.result or {}
        if rec.key in seen or "action" not in r or not rec.harvested_at or as_date(rec.harvested_at) >= as_date(ctx.now):
            continue
        rows.append(MR.ResearchOutcome(rec.key[:16], "ladder_rung", f"feature:{rec.feature}", "derived_feature_rank", "research_panel",
                                       "walk_forward_dates", "loop", rec.problem, rec.cutoff, rec.harvested_at,
                                       float(r.get("cost", 1.0)), failed=r["action"] in ("FAIL",), n_tests=int(r.get("n_tests", 1)),
                                       decision_changed=r["action"] == "COMPLETE"))
    if rows:
        st.store.extend(rows)
        ids = {r.run_id for r in rows}
        seen.update(k for k in ctx.state.jobs if k[:16] in ids)
    res = MR.step(st, ctx.now, ctx.state.cfg.seed)
    return len(rows), 1, str(getattr(res, "label", ""))


def st_failed_lab(ctx: Ctx) -> tuple:
    try:
        from engine.research import failed_lab as FL
    except ImportError as e:
        raise MissingModule("failed_lab") from e
    lab = ctx.mod_state("failed_lab", FL.FailedLearnerLab)
    rep = FL.step(lab, ctx.now)
    return 1, len(getattr(rep, "dead_classes", ()) or ()), ""


def st_scorecard(ctx: Ctx) -> tuple:
    try:
        from engine.research import scorecard as SC
    except ImportError as e:
        raise MissingModule("scorecard") from e
    J = _matured_decisions(ctx)
    if len(J) == 0:
        raise NoInput("no matured decisions to score")
    per = pd.to_datetime(J["_day"]).dt.to_period("M").astype(str).to_numpy()
    mat = pd.to_datetime(J["end"]).dt.date.astype(str).to_numpy()
    vol = pd.DataFrame({"matured_at": mat, "p_move": J["p_move"].to_numpy(float), "moved": J["touch"].to_numpy(float),
                        "abs_move": J["absmove"].to_numpy(float), "period": per}).dropna(subset=["p_move"])
    dm = J[J["mover"].to_numpy(bool) & np.isfinite(J["p_up"].to_numpy(float))]
    direc = pd.DataFrame({"matured_at": pd.to_datetime(dm["end"]).dt.date.astype(str).to_numpy(), "p_up": dm["p_up"].to_numpy(float),
                          "up": dm["up"].to_numpy(float), "period": pd.to_datetime(dm["_day"]).dt.to_period("M").astype(str).to_numpy()})
    pos = J[J["side"].to_numpy(int) != 0]
    losses = pd.DataFrame({"matured_at": pd.to_datetime(pos["end"]).dt.date.astype(str).to_numpy(),
                           "ret": pos["side"].to_numpy(float) * pos["close"].to_numpy(float),
                           "period": pd.to_datetime(pos["_day"]).dt.to_period("M").astype(str).to_numpy()})
    a = SC.Artefacts(vol=vol if len(vol) else None, direction=direc if len(direc) else None, losses=losses if len(losses) else None)
    pst = ctx.state.modules.get("priority")
    w = dict(pst.weights().w) if pst is not None else {p: 1.0 for p in Problem}
    rep = SC.step(a, ctx.now, ctx.rt.code_hash, w, ctx.mod_state("scorecard_log", SC.ScorecardLog))
    return len(J), 1, ""


def st_stale(ctx: Ctx) -> tuple:
    """Stale-hypothesis detection and cancellation: questions open too long are abandoned (their queued/dormant branches cancelled);
    trees that can no longer move their belief are reported; the ladder's stalled branches are parked with the reason."""
    from engine.research import compute_manager as CM
    from engine.research import questions as Q
    led = ctx.state.modules.get("question_ledger")
    ms = ctx.state.modules.get("compute_manager")
    forest = ctx.state.modules.get("hypothesis_tree")
    if led is None:
        raise NoInput("no questions yet")
    abandoned = Q.abandon_stale(led, ctx.now, ctx.state.cfg.stale_days)
    cancelled = parked = 0
    if ms is not None:
        for qid in abandoned:
            bid = ctx.state.branch_of.get(qid)
            b = ms.branches.get(bid) if bid else None
            if b is not None and b.state.value in ("QUEUED", "DORMANT"):
                CM.cancel(ms, bid, ctx.now, "question abandoned as stale")
                cancelled += 1
        for bid in CM.stalled_branches(ms, ctx.now):
            b = ms.branches[bid]
            if b.state.value not in ("DORMANT", "FAILED", "RETIRED", "CANCELLED") and not b.in_flight:
                CM.park(ms, bid, ctx.now, "stalled: no progress on the ladder")
                parked += 1
    stalled = forest.stalled(ctx.now) if forest is not None else []
    ctx.bus["stale"] = {"abandoned": list(abandoned), "cancelled": cancelled, "parked": parked, "stalled_trees": list(stalled)}
    ctx.state.count("stale_abandoned", len(abandoned))
    return len(led.rows), len(abandoned) + parked, f"{len(stalled)} stalled trees"


# ================================================================================================================ SWEEPS + REPORT
def st_sweeps(ctx: Ctx) -> tuple:
    """C67 always-on sweeps. They run only when higher-value work leaves room: never while ranked experiments are still waiting for a
    worker, and with units scaled by the controller's continuous-discovery share. Each sweep resumes from its own checkpoint."""
    specs = ctx.rt.sweep_specs
    if not specs:
        raise NoInput("no always-on sweeps configured")
    busy = ctx.rt.executor.running() >= ctx.state.cfg.max_workers
    dec = ctx.bus.get("controller")
    share = dec.share(CT.Phase.DISCOVERY) if dec is not None else 0.05
    units = 0 if busy else max(1, int(round(ctx.state.cfg.sweep_units * share / 0.05)))
    ran = 0
    for name, spec in sorted(specs.items()):
        pr = ctx.state.sweeps.setdefault(name, SweepProgress())
        if units == 0:
            pr.yielded += 1
            continue
        try:
            if name not in ctx.rt.sweep_runs:
                ctx.rt.sweep_runs[name] = spec.factory(ctx.rt.root)
            k = int(ctx.rt.sweep_runs[name](units, ctx.now))
            pr.units += k
            pr.runs += 1
            pr.last_at = ctx.now
            ran += k
        except FirewallBreach:
            raise
        except Exception as e:                           # noqa: BLE001 - a broken sweep is reported and retried later
            pr.last_error = f"{type(e).__name__}: {str(e)[:160]}"
    return len(specs), ran, "yielded to higher-value work" if units == 0 else ""


def st_report(ctx: Ctx) -> tuple:
    return 0, 0, ""


# ================================================================================================================ the stage table
STAGES: tuple[StageSpec, ...] = (
    StageSpec("observe.panel", LoopPhase.OBSERVE, st_panel, "feed"),
    StageSpec("observe.observer", LoopPhase.OBSERVE, st_observer, "observer"),
    StageSpec("observe.autopsy", LoopPhase.OBSERVE, st_autopsy, "autopsy"),
    StageSpec("update.experiment_memory", LoopPhase.UPDATE_KNOWLEDGE, st_experiments_sweep, "experiments"),
    StageSpec("update.firewall_release", LoopPhase.UPDATE_KNOWLEDGE, st_release, "firewall"),
    StageSpec("evaluate.two_stage", LoopPhase.EVALUATE, st_two_stage, "two_stage"),
    StageSpec("evaluate.volatility_lab", LoopPhase.EVALUATE, st_volatility_lab, "volatility_lab"),
    StageSpec("evaluate.direction_lab", LoopPhase.EVALUATE, st_direction_lab, "direction_lab"),
    StageSpec("evaluate.frontier", LoopPhase.EVALUATE, st_frontier, "frontier"),
    StageSpec("evaluate.symmetry", LoopPhase.EVALUATE, st_symmetry, "symmetry"),
    StageSpec("surprises.feature_screen", LoopPhase.SURPRISES, st_feature_screen, "volatility_lab"),
    StageSpec("surprises.cross_section", LoopPhase.SURPRISES, st_cross_section, "cross_section"),
    StageSpec("surprises.regimes", LoopPhase.SURPRISES, st_regimes, "regimes"),
    StageSpec("surprises.multiscale", LoopPhase.SURPRISES, st_multiscale, "multiscale"),
    StageSpec("failures.decision_losses", LoopPhase.FAILURES, st_decision_losses, "two_stage"),
    StageSpec("failures.winners_losers", LoopPhase.FAILURES, st_winners_losers, "winners_losers"),
    StageSpec("failures.loss_pipeline", LoopPhase.FAILURES, st_loss_pipeline, "loss_pipeline"),
    StageSpec("missed.missed", LoopPhase.MISSED, st_missed, "missed"),
    StageSpec("missed.knowability", LoopPhase.MISSED, st_knowability, "knowability"),
    StageSpec("missed.unknown_cause", LoopPhase.MISSED, st_unknown_cause, "unknown_cause"),
    StageSpec("missed.counterfactual", LoopPhase.MISSED, st_counterfactual, "counterfactual"),
    StageSpec("breaks.knowledge", LoopPhase.BREAKS, st_knowledge_breaks, "loop"),
    StageSpec("breaks.break_research", LoopPhase.BREAKS, st_break_research, "break_research"),
    StageSpec("questions.discovery", LoopPhase.QUESTIONS, st_discovery, "discovery"),
    StageSpec("questions.interactions", LoopPhase.QUESTIONS, st_interactions, "interactions"),
    StageSpec("questions.precursors", LoopPhase.QUESTIONS, st_precursors, "precursors"),
    StageSpec("questions.targets", LoopPhase.QUESTIONS, st_targets, "targets"),
    StageSpec("questions.research_graph", LoopPhase.QUESTIONS, st_research_graph, "research_graph"),
    StageSpec("questions.generate", LoopPhase.QUESTIONS, st_generate, "questions"),
    StageSpec("hypotheses.trees", LoopPhase.HYPOTHESES, st_trees, "hypothesis_tree"),
    StageSpec("gain.priority", LoopPhase.GAIN, st_priority, "priority"),
    StageSpec("allocate.controller", LoopPhase.ALLOCATE, st_controller, "controller"),
    StageSpec("allocate.diversity", LoopPhase.ALLOCATE, st_diversity, "diversity"),
    StageSpec("allocate.compute", LoopPhase.ALLOCATE, st_compute, "compute_manager"),
    StageSpec("run.experiments", LoopPhase.RUN, st_run, "loop"),
    StageSpec("validate.harvest", LoopPhase.VALIDATE, st_harvest, "loop"),
    StageSpec("controls.compare", LoopPhase.CONTROLS, st_controls, "compute_manager"),
    StageSpec("learn.ladder", LoopPhase.LEARN, st_ladder, "compute_manager"),
    StageSpec("learn.memory", LoopPhase.LEARN, st_memory, "experiments"),
    StageSpec("learn.value", LoopPhase.LEARN, st_value, "value_accounting"),
    StageSpec("learn.science_memory", LoopPhase.LEARN, st_science, "science_memory"),
    StageSpec("learn.replication", LoopPhase.LEARN, st_replication, "replication"),
    StageSpec("learn.quality_gate", LoopPhase.LEARN, st_quality_and_knowledge, "quality_gate"),
    StageSpec("learn.decision_bridge", LoopPhase.LEARN, st_bridge, "decision_bridge"),
    StageSpec("priorities.waste", LoopPhase.PRIORITIES, st_waste, "waste"),
    StageSpec("priorities.brain_health", LoopPhase.PRIORITIES, st_brain_health, "brain_health"),
    StageSpec("priorities.meta_research", LoopPhase.PRIORITIES, st_meta_research, "meta_research"),
    StageSpec("priorities.failed_lab", LoopPhase.PRIORITIES, st_failed_lab, "failed_lab"),
    StageSpec("priorities.scorecard", LoopPhase.PRIORITIES, st_scorecard, "scorecard"),
    StageSpec("priorities.stale", LoopPhase.PRIORITIES, st_stale, "questions"),
    StageSpec("sweeps.background", LoopPhase.SWEEPS, st_sweeps, "loop"),
    StageSpec("report.cycle", LoopPhase.REPORT, st_report, "loop"),
)
STAGE_NAMES = tuple(s.name for s in STAGES)
BUILTIN_STAGES = STAGE_NAMES
_REGISTERED: dict[str, StageSpec] = {}


def register_stage(name: str, fn: Callable[["Ctx"], tuple], after: str, phase: LoopPhase | None = None, module: str = "") -> StageSpec:
    """PUBLIC. Insert an extra stage into every cycle right after the stage `after` (e.g. the C68 prediction-error stages of
    engine/research/error_loop.py), with the built-in semantics: run_stage classifies it (OK / SKIPPED_NO_INPUT via NoInput /
    SKIPPED_MISSING_MODULE via MissingModule / REFUSED_LEAK via FirewallBreach / FAILED), it is checkpointed after it runs, a resumed
    cycle never re-runs it, and it can be disabled or given a cadence by name. `fn(ctx)` returns (n_in, n_out, note); feed inputs
    reach it through ctx.extra(key) or ctx.namespace(ns). Registering the same name with the same function again is a no-op (module
    re-import); a different function under a registered name, an unknown `after` or a slot after the report is refused."""
    global STAGES, STAGE_NAMES
    if not name or not isinstance(name, str) or not callable(fn):
        raise ValueError("register_stage needs a name and a callable fn(ctx) -> (n_in, n_out, note)")
    if name in _REGISTERED:
        if _REGISTERED[name].fn is fn or getattr(_REGISTERED[name].fn, "__qualname__", 1) == getattr(fn, "__qualname__", 2):
            return _REGISTERED[name]
        raise ValueError(f"stage {name!r} is already registered with a different function")
    if name in STAGE_NAMES:
        raise ValueError(f"{name!r} is a built-in stage")
    if after not in STAGE_NAMES or after == "report.cycle":
        raise ValueError(f"cannot insert after {after!r}: it must be an existing stage before report.cycle")
    i = STAGE_NAMES.index(after)
    spec = StageSpec(name, phase or STAGES[i].phase, fn, module or getattr(fn, "__module__", "registered"))
    STAGES = STAGES[:i + 1] + (spec,) + STAGES[i + 1:]
    STAGE_NAMES = tuple(x.name for x in STAGES)
    _REGISTERED[name] = spec
    return spec


def unregister_stage(name: str) -> None:
    """Remove a registered (never a built-in) stage; tests use it to leave the stage table as they found it."""
    global STAGES, STAGE_NAMES
    if name not in _REGISTERED:
        raise ValueError(f"{name!r} is not a registered stage")
    del _REGISTERED[name]
    STAGES = tuple(x for x in STAGES if x.name != name)
    STAGE_NAMES = tuple(x.name for x in STAGES)


def registered_stages() -> tuple[str, ...]:
    return tuple(n for n in STAGE_NAMES if n in _REGISTERED)


# ================================================================================================================ running
def new_state(cfg: LoopConfig | None = None) -> LoopState:
    cfg = cfg or LoopConfig()
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid LoopConfig: " + "; ".join(errs))
    return LoopState(cfg=cfg)


def _due(spec: StageSpec, cfg: LoopConfig, cycle: int) -> bool:
    k = int(cfg.cadence.get(spec.name, 1))
    return k <= 1 or cycle % k == 0


def run_stage(spec: StageSpec, ctx: Ctx) -> StageRecord:
    """Run one stage, classifying the outcome. FirewallBreach -> REFUSED_LEAK at this stage; MissingModule -> SKIPPED_MISSING_MODULE;
    NoInput -> SKIPPED_NO_INPUT; anything else -> FAILED with the message (the cycle continues)."""
    st = ctx.state
    key = f"{ctx.cycle}|{spec.name}"
    st.exec_count[key] = st.exec_count.get(key, 0) + 1
    t0 = time.perf_counter()
    if spec.name in st.cfg.disabled:
        return StageRecord(ctx.cycle, spec.name, spec.phase.value, StageStatus.SKIPPED_DISABLED, "disabled by configuration", 0, 0, 0.0, spec.module)
    if not _due(spec, st.cfg, ctx.cycle):
        return StageRecord(ctx.cycle, spec.name, spec.phase.value, StageStatus.SKIPPED_CADENCE,
                           f"runs every {st.cfg.cadence.get(spec.name)} cycles", 0, 0, 0.0, spec.module)
    try:
        n_in, n_out, why = spec.fn(ctx)
        status = StageStatus.OK
    except FirewallBreach as e:
        n_in, n_out, why, status = 0, 0, f"{type(e).__name__}: {str(e)[:300]}", StageStatus.REFUSED_LEAK
        st.count("leaks_refused")
    except MissingModule as e:
        n_in, n_out, why, status = 0, 0, f"module engine.research.{e.args[0] if e.args else '?'} is not present", StageStatus.SKIPPED_MISSING_MODULE
    except ImportError as e:                             # a research module a stage imports without its own guard
        if "engine.research" not in f"{e} {getattr(e, 'name', '') or ''}":
            n_in, n_out, why, status = 0, 0, f"{type(e).__name__}: {str(e)[:260]}", StageStatus.FAILED
            st.count("stage_failures")
        else:
            n_in, n_out, why, status = 0, 0, f"{str(e)[:200]}", StageStatus.SKIPPED_MISSING_MODULE
    except NoInput as e:
        n_in, n_out, why, status = 0, 0, str(e), StageStatus.SKIPPED_NO_INPUT
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:                               # noqa: BLE001 - a crash in one module must not stop research
        n_in, n_out, status = 0, 0, StageStatus.FAILED
        tb = traceback.extract_tb(e.__traceback__)
        where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        why = f"{type(e).__name__}: {str(e)[:260]}{where}"
        st.count("stage_failures")
    return StageRecord(ctx.cycle, spec.name, spec.phase.value, status, why, int(n_in), int(n_out), round(time.perf_counter() - t0, 4), spec.module)


def _cycle_now(state: LoopState, rt: Runtime) -> str | None:
    """The next decision date strictly after the last cycle's (the feed's own clock)."""
    dates = [str(as_date(d)) for d in rt.feed.dates()]
    if state.now and state.done_stages.get(state.cycle) and len(state.done_stages[state.cycle]) < len(STAGES):
        return state.now                                                 # resume the unfinished cycle at its own date
    later = [d for d in dates if not state.now or d > state.now]
    return later[0] if later else None


def recover_jobs(state: LoopState, rt: Runtime) -> dict:
    """Crash recovery. A job the killed loop had in flight is either DONE in the compute ledger (its result is harvested normally) or
    not: then it is re-queued (ledger requeue after marking the dead attempt) and re-run on the same inputs, never skipped."""
    ex = rt.executor
    out = {"reconciled": 0, "requeued": 0, "lost": 0}
    held = state.memo.get("held_for_fresh_data", {})
    for rec in state.jobs.values():
        if rec.state not in (JobState.RUNNING, JobState.SUBMITTED) or rec.key in held:
            continue                                     # a held rung is waiting for data, not lost: st_run releases it
        ls = ex.ledger_state(rec.key)
        if ls == C.DONE:
            rec.state = JobState.DONE
            out["reconciled"] += 1
            continue
        if ls == C.RUNNING:
            ex.ledger.fail(rec.key, C.CRASHED, "research loop restarted while the job ran", _epoch(state.now or rec.submitted_at))
            ls = ex.ledger_state(rec.key)
        if ls in (C.PENDING, C.CRASHED, C.FAILED):
            try:
                spec = ex.ledger.spec(rec.key)
                if ls != C.PENDING:
                    ex.ledger.submit(spec, _epoch(rec.submitted_at), rt.code_hash)
                obs = rt.feed.observe(rec.submitted_at)
                st = ex.submit(spec, ExperimentTask(**rec.task), obs.matured, obs.data_hash, rec.submitted_at)
                rec.state = JobState.DONE if st == C.DONE else JobState.RUNNING
                rec.attempts += 1
                out["requeued"] += 1
            except C.ComputeError as e:
                rec.state, rec.error = JobState.FAILED, f"recovery refused: {e}"
                out["lost"] += 1
    return out


def step(state: LoopState, rt: Runtime) -> dict | None:
    """PUBLIC ENTRY. Run (or finish) one cycle at the feed's next decision date. Stages already completed in this cycle (a resumed
    cycle) are not run again; a checkpoint is written after every stage. Returns the cycle report, or None when the feed has no date."""
    now = _cycle_now(state, rt)
    if now is None:
        return None
    resumed = bool(state.done_stages.get(state.cycle))
    if not resumed:
        carry = {k: state.bus[k] for k in ("controller_prev",) if state.bus.get(k) is not None}
        state.bus = carry
        state.done_stages[state.cycle] = []
    state.now = now
    ctx = Ctx(state, rt, now, state.cycle)
    if resumed and rt.obs is None:
        try:
            rt.obs = rt.feed.observe(now)
        except Exception:                                 # noqa: BLE001 - OBSERVE had refused; later stages will say so
            rt.obs = None
    records = [StageRecord(**{**r, "status": StageStatus(r["status"])}) for r in state.bus.get("_records", [])]
    for spec in STAGES:
        if spec.name in state.done_stages[state.cycle]:
            continue
        if spec.name == "report.cycle":
            k = f"{state.cycle}|{spec.name}"
            state.exec_count[k] = state.exec_count.get(k, 0) + 1
            rep = cycle_report(state, rt, records)
            records.append(StageRecord(state.cycle, spec.name, spec.phase.value, StageStatus.OK, "", len(records), 1, 0.0, "loop"))
            rep["stages"] = [r.to_dict() for r in records]
            _write_report(rt.root, rep)
            state.reports = (state.reports + [rep])[-state.cfg.report_keep:]
        else:
            records.append(run_stage(spec, ctx))
        state.done_stages[state.cycle].append(spec.name)
        state.bus["_records"] = [dataclasses.asdict(r) for r in records]
        if rt.checkpointer is not None and state.cfg.checkpoint == "stage":
            rt.checkpointer.save(state, rt.code_hash, f"cycle {state.cycle}", _next_name(spec.name), _in_flight(state))
        if rt.kill_after == f"{state.cycle}|{spec.name}":
            raise KeyboardInterrupt(f"planted kill after {spec.name}")
    report = state.reports[-1]
    state.cycle += 1
    state.bus = {"controller_prev": ctx.bus.get("controller")} if ctx.bus.get("controller") is not None else {}
    rt.obs = None
    if rt.checkpointer is not None and state.cfg.checkpoint in ("stage", "cycle"):
        rt.checkpointer.save(state, rt.code_hash, f"cycle {state.cycle}", "start the next cycle", _in_flight(state))
    return report


def _next_name(name: str) -> str:
    i = STAGE_NAMES.index(name)
    return f"run stage {STAGE_NAMES[i + 1]}" if i + 1 < len(STAGE_NAMES) else "start the next cycle"


def _in_flight(state: LoopState) -> str:
    live = sorted(k for k, r in state.jobs.items() if r.state in (JobState.RUNNING, JobState.SUBMITTED))
    return live[0] if live else ""


def open_loop(feed: Feed, root: str | Path, cfg: LoopConfig | None = None, sweeps: Sequence[SweepSpec] = (), fresh: bool = False,
              clock: Callable[[], float] = time.time, kill_after: str | None = None) -> tuple[LoopState, Runtime, dict]:
    """Build the runtime and either resume from the newest verified checkpoint (fail closed on stale code unless the config allows a
    code change) or start fresh. Returns (state, runtime, resume info)."""
    cfg = cfg or LoopConfig()
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid LoopConfig: " + "; ".join(errs))
    rt = Runtime(feed, root, cfg, sweeps, clock, kill_after)
    info = {"action": "START_FRESH", "recovered": {}}
    state = None
    if rt.checkpointer is not None and not fresh:
        state, plan = rt.checkpointer.load(rt.code_hash, cfg.allow_code_change)
        info["action"] = plan.action if state is not None else "START_FRESH"
        info["reasons"] = list(plan.reasons)
    if state is None:
        state = new_state(cfg)
    else:
        lost = [n for n in state.memo.get("stage_table", ()) if n not in STAGE_NAMES]
        if lost:
            raise RuntimeError(f"the checkpoint was written with stages {lost} that are not registered in this process: register them "
                               "(import their module) before resuming, or the resumed cycle would silently skip them")
        state = dataclasses.replace(state, cfg=dataclasses.replace(cfg, run_id=state.cfg.run_id))
        info["recovered"] = recover_jobs(state, rt)
    state.memo["stage_table"] = list(STAGE_NAMES)
    return state, rt, info


def run(feed: Feed, root: str | Path, cfg: LoopConfig | None = None, max_cycles: int | None = 1, sweeps: Sequence[SweepSpec] = (),
        fresh: bool = False, wall_budget_s: float | None = None, clock: Callable[[], float] = time.time,
        kill_after: str | None = None) -> tuple[LoopState, list[dict]]:
    """Run cycles until the feed runs out, `max_cycles` (None = forever) or the wall budget is spent. An interruption writes the
    section-58 interruption record next to the checkpoints before it propagates."""
    state, rt, info = open_loop(feed, root, cfg, sweeps, fresh, clock, kill_after)
    reports = []
    t0 = time.monotonic()
    try:
        while max_cycles is None or len(reports) < max_cycles:
            rep = step(state, rt)
            if rep is None:
                break
            rep["resume"] = info
            reports.append(rep)
            if wall_budget_s is not None and time.monotonic() - t0 > wall_budget_s:
                break
        if state.cfg.mode != "inline":
            rt.executor.wait(state.cfg.job_timeout_s)
    except KeyboardInterrupt:
        if rt.checkpointer is not None:
            rt.checkpointer.interruption(state, rt.code_hash, "interrupted")
        rt.executor.shutdown()
        raise
    rt.executor.shutdown()
    return state, reports


# ================================================================================================================ reports
def cycle_report(state: LoopState, rt: Runtime, records: Sequence[StageRecord]) -> dict:
    by = {}
    for r in records:
        by[r.status.value] = by.get(r.status.value, 0) + 1
    dec = state.bus.get("controller")
    chains = state.lineage.section47_chains()
    kchains = state.lineage.knowledge_chains()
    return {"cycle": state.cycle, "now": state.now, "label": LABEL, "status_counts": by,
            "missing_modules": sorted({r.stage for r in records if r.status == StageStatus.SKIPPED_MISSING_MODULE}),
            "no_input": sorted({r.stage for r in records if r.status == StageStatus.SKIPPED_NO_INPUT}),
            "failed": {r.stage: r.reason for r in records if r.status == StageStatus.FAILED},
            "refused": {r.stage: r.reason for r in records if r.status == StageStatus.REFUSED_LEAK},
            "questions": len(state.questions), "new_questions": len(state.bus.get("new_questions", [])),
            "selected": len(state.bus.get("plan_selected", [])), "launched": len(state.bus.get("launched", [])),
            "harvested": len(state.bus.get("harvested", [])), "validated": len(state.bus.get("validated", [])),
            "ladder": dict(state.bus.get("ladder_actions", {})), "knowledge": len(state.knowledge),
            "controller": dec.to_dict() if dec is not None else None,
            "statements": [s.text for s in dec.statements] if dec is not None else [],
            "evaluation": state.bus.get("evaluation"), "two_stage": state.bus.get("two_stage"),
            "section47_chains": len(chains), "knowledge_chains": len(kchains), "lineage": state.lineage.kinds(),
            "counters": {k: v for k, v in state.counters.items() if not k.startswith("_")}, "notes": list(state.bus.get("notes", []))[:20],
            "stale": state.bus.get("stale"), "sweeps": {k: dataclasses.asdict(v) for k, v in state.sweeps.items()},
            "code_hash": rt.code_hash}


def _write_report(root: Path, rep: Mapping[str, Any]) -> Path:
    d = Path(root) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"cycle_{int(rep['cycle']):05d}.json"
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(json.dumps(rep, indent=1, sort_keys=True, default=str).encode("utf-8"))
    os.replace(tmp, p)
    return p


def render_report(rep: Mapping[str, Any]) -> str:
    lines = [f"RESEARCH LOOP cycle {rep['cycle']} at {rep['now']}  [{rep['label']}]",
             f"  stages: {rep['status_counts']}",
             f"  questions {rep['questions']} (+{rep['new_questions']}), selected {rep['selected']}, launched {rep['launched']}, "
             f"harvested {rep['harvested']}, validated {rep['validated']}, ladder {rep['ladder']}, knowledge {rep['knowledge']}",
             f"  section-47 chains {rep['section47_chains']}, knowledge->decision chains {rep['knowledge_chains']}"]
    if rep.get("missing_modules"):
        lines.append("  SKIPPED_MISSING_MODULE: " + ", ".join(rep["missing_modules"]))
    for k, v in (rep.get("refused") or {}).items():
        lines.append(f"  REFUSED_LEAK {k}: {v[:160]}")
    for k, v in (rep.get("failed") or {}).items():
        lines.append(f"  FAILED {k}: {v[:160]}")
    lines += ["  > " + s for s in rep.get("statements", [])[:4]]
    return "\n".join(lines)


def stage_status_table(reports: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [dict(s, cycle=r["cycle"]) for r in reports for s in r.get("stages", [])]
    return pd.DataFrame(rows)


if os.environ.get(TASK_DIR_ENV):                     # only inside a worker process the loop launched (see register_task_workers)
    register_task_workers()
