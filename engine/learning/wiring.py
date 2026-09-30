"""Integration wiring of the learning brain into the existing engine (contract C62 sections 55, 61, 85; canon C62-C65).

IMPLEMENTED - NOT VALIDATED. This module is the ONE place where the old engine calls the new learning package. Each old
call site (engine/lessons.py, memory.py, missed_winners.py, improve.py, registry.py, patterns.py) makes a single call into a
function here; the function turns the old structure into the new vocabulary and hands it to the facade that owns it:

  lessons / memory lessons          -> failure.hypotheses_from_lessons                  (on_lessons)
  lessons.post_mortem losses        -> failure.records_from_lessons_frame + LossClassifier + FailureLedger   (on_post_mortem)
  MissedLedger weeks                -> learning.missed_winners.MissedLearningLedger     (on_missed_week)
  improve.log_experiment            -> experiment_memory.ExperimentLedger (one facade)  (on_experiment, repro_dict)
  challenger launch                 -> ExperimentLedger.already_tested                  (pre_launch, blocks_launch)
  registry.audit                    -> firewalls.registry_findings                      (registry_audit)
  miner prune_redundant             -> knowledge_graph REDUNDANT_WITH edges             (on_redundancy)
  production readers of a weight    -> champion.KnowledgeBoard.weight                   (effective_weight)
  promotion of a learning claim     -> scorecard.gate_improvement_claim + LearningFirewallGate + IdentityHarness
                                                                                        (promotion_gate / require_promotion)
S21b (29 Sep: the audit found five of the hooks above in code production never runs; scripts/reachability.py now checks the call
graph, not the text):
  adaptive.Adapter closed weeks     -> MissedLedger.observe -> on_missed_week (week_of: vectorised), on_post_mortem one decision later
  adaptive.Session.result           -> on_session_end -> Memory.export_lessons -> on_lessons
  improve.log_experiment (decided)  -> on_memory_entry(version_existing=True): the Phase-30 answers as a new ledger version
  improve.weekly / loop2 worker     -> configure_production(lane): sinks persist under state/learning/hub/<lane>/, board weights apply
  PromotionGate.evaluate            -> on_knowledge_promotion: every learned promotion is a learning claim (record | enforce)
  LessonBook.adjust (decision run)  -> effective_weight(run=) + end_decision_run -> champion.audit_decision_sources

Design rules
  * SINKS OBSERVE, GATES DECIDE. A sink hook (everything except promotion_gate/require_promotion) can never break the caller:
    any exception is recorded in Hub.errors (and warned once per hook) and the old code path continues unchanged. A gate hook
    fails closed: a learning claim with a missing piece of evidence is refused, never waved through.
  * NO NEW BEHAVIOUR WHERE UNTESTED. effective_weight() multiplies by the board only for knowledge the board knows about;
    unknown knowledge keeps its legacy weight unless the hub is switched to strict mode. A challenger that makes no learning
    claim is judged by the old live-shadow z-test alone. Every such adapter says so in its docstring.
  * NOTHING READS THE HUB'S OUTPUT IN PRODUCTION. Hypotheses, classifications and graph edges are research records; the
    decision path does not consume them, so a sink can not leak a future outcome into a decision.
  * NO WALL CLOCK IN DECISIONS. Every hook that has a time takes it as an argument."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import threading
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd

from . import experiment_memory as EM
from . import failure as F
from . import missed_winners as MW
from .champion import KnowledgeBoard
from .contradiction import ContradictionLedger
from .contradiction_monitor import ContradictionMonitor, MonitorReport
from .core import FirewallBreach, Subsystem, ValidationLabel, as_date, canonical_json, current_code_hash, stable_hash
from .firewalls import Finding, GateContext, LearningFirewallGate, registry_findings
from .identity_firewall import IdentityHarness, IdentityReport
from .knowledge_graph import KnowledgeGraph, NodeType
from .postmortem import make_hypothesis
from .reproducibility import ReproRecord, WorkerConfig, capture_worker, make_record
from .scorecard import LearningScorecard, gate_improvement_claim

T = TypeVar("T")
NL = chr(10)
MAX_CLASSIFY = 200                       # per post-mortem call: the worst losses only (a classifier pass costs ~1 ms per loss)
BLOCKING_LAUNCH = (EM.DuplicateStatus.SAME_CONFIG_REPEAT, EM.DuplicateStatus.NEAR_DUPLICATE, EM.DuplicateStatus.KNOWN_FAILURE_NEARBY)
UNRECORDED = "unrecorded"                # memory_hash of a legacy registry record: honest, not a fabricated snapshot


# ================================================================================================================== hub
@dataclasses.dataclass(frozen=True)
class SinkError:
    """A sink hook failed. Recorded (never raised into the caller) so the failure is visible in hook_report()."""
    hook: str
    error_type: str
    message: str


@dataclasses.dataclass(frozen=True)
class IdentityJob:
    """A deferred identity-firewall run: the harness executes when the promotion gate asks, on the panels given here."""
    learner: Callable[..., pd.Series]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_eval: pd.DataFrame
    y_eval: pd.Series
    seed: int = 0
    attacks: tuple[str, ...] | None = None
    attack_kwargs: Mapping[str, Mapping[str, Any]] | None = None       # e.g. {"episode_substitution": {"block": 8, "frac": 1.0}}

    def run(self) -> IdentityReport:
        kw: dict[str, Any] = {"seed": self.seed, "attack_kwargs": self.attack_kwargs}
        if self.attacks is not None:
            kw["attacks"] = self.attacks
        return IdentityHarness(self.learner, **kw).run(self.X_train, self.y_train, self.X_eval, self.y_eval)


@dataclasses.dataclass(frozen=True)
class LearningEvidence:
    """Everything the promotion gate needs for one learning claim. Any None is a failed check, not a skipped one."""
    card: LearningScorecard | None = None
    gate_ctx: GateContext | None = None
    identity: IdentityReport | IdentityJob | None = None


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclasses.dataclass(frozen=True)
class PromotionVerdict:
    """Outcome of promotion_gate. `allowed` for a learning claim needs all three of scorecard, firewalls and identity; the label
    is never VALIDATED here (a sealed blind window is what validates, not this gate)."""
    subject: str
    now: str
    claims_learning: bool
    allowed: bool
    checks: tuple[Check, ...]
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if not c.ok)

    def digest(self) -> str:
        return stable_hash([self.subject, self.now, self.claims_learning, self.allowed, [(c.name, c.ok) for c in self.checks]])


class Hub:
    """Process-wide sink state. `configure()` changes where it persists; `reset()` empties it (tests, and a new blind window)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self, root: str | Path | None = None, enabled: bool = True) -> None:
        with self._lock:
            self.root: Path | None = Path(root) if root else None
            self.enabled = enabled
            self.calls: Counter[str] = Counter()
            self.delivered: Counter[str] = Counter()
            self.errors: list[SinkError] = []
            self._warned: set[str] = set()
            self.hypotheses: dict[str, F.Hypothesis] = {}
            self.failure_ledger = F.FailureLedger()
            self.classifier = F.LossClassifier()
            self._classified: set[str] = set()
            self.missed = MW.MissedLearningLedger()
            self._ledgers: dict[str, EM.ExperimentLedger] = {}
            self.graph = KnowledgeGraph(None)
            self.board: KnowledgeBoard | None = None
            self.monitor: ContradictionMonitor | None = None
            self.strict = False
            self.evidence: dict[str, LearningEvidence] = {}
            self.decisions: list[PromotionVerdict] = []
            self._worker: WorkerConfig | None = None
            self._persisted: dict[str, set[str]] = {}
            self._backfilled: set[str] = set()
            self.lane: str = ""
            self.used: dict[str, set[str]] = {}               # decision run -> board member ids that carried weight in it
            self.source_violations: list[dict[str, Any]] = []
            self.scorecards: dict[str, LearningScorecard] = {}
            self.identity_reports: dict[str, IdentityReport | IdentityJob] = {}
            self.knowledge_verdicts: dict[str, PromotionVerdict] = {}

    # ---- bookkeeping
    def record_error(self, hook: str, exc: BaseException) -> None:
        err = SinkError(hook, type(exc).__name__, str(exc)[:300])
        with self._lock:
            self.errors.append(err)
            first = hook not in self._warned
            self._warned.add(hook)
        if first:
            warnings.warn(f"[learning.wiring] sink {hook} failed ({err.error_type}: {err.message}); the old path continued", stacklevel=3)

    def persist(self, name: str, key: str, obj: Any) -> bool:
        """Append one record to root/<name>.jsonl once per key (also across processes: existing keys are read on first use)."""
        if self.root is None:
            return False
        with self._lock:
            path = self.root / f"{name}.jsonl"
            seen = self._persisted.get(name)
            if seen is None:
                seen = set()
                if path.exists():
                    for ln in path.read_bytes().decode("utf-8").split("\n"):
                        if ln.strip():
                            seen.add(str(json.loads(ln).get("key")))
                self._persisted[name] = seen
            if key in seen:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as fh:                       # binary: text mode would write CRLF on Windows
                fh.write((json.dumps({"key": key, "body": json.loads(canonical_json(obj))}, sort_keys=True) + "\n").encode("utf-8"))
            seen.add(key)
            return True

    def ledger_for(self, registry_path: str | Path, backfill: bool = False) -> EM.ExperimentLedger:
        """The ONE facade ledger of a state directory: experiment_ledger.jsonl beside whichever legacy file (experiments.jsonl,
        the experiment-memory file) is being mirrored, so every old writer lands in the same place. With `backfill`, the first
        call per process also imports every row already in `registry_path` (ids dedupe), so 'have we tested this' knows the whole
        history and not only what this process wrote - including rows written by processes running older code."""
        path = Path(registry_path).with_name("experiment_ledger.jsonl")
        key = str(path)
        with self._lock:
            if key not in self._ledgers:
                self._ledgers[key] = EM.ExperimentLedger(path)
            led = self._ledgers[key]
            if backfill and key not in self._backfilled:
                self._backfilled.add(key)
                rows, bad = read_jsonl(Path(registry_path))
                self.delivered["backfill_bad_lines"] += bad
                if rows:
                    self.delivered["ledger_rows"] += EM.import_legacy(led, [legacy_row(r) for r in rows])["added"]
            return led


HUB = Hub()


def configure(root: str | Path | None = None, board: KnowledgeBoard | None = None, strict: bool | None = None,
              enabled: bool | None = None) -> Hub:
    """Point the hub at a persistence root and/or a KnowledgeBoard. `strict=True` makes effective_weight() return 0 for knowledge
    the board has never seen (the production end state); the default keeps legacy weights for unregistered knowledge."""
    if root is not None:
        HUB.root = Path(root)
        HUB._persisted.clear()
    if board is not None:
        HUB.board = board
    if strict is not None:
        HUB.strict = bool(strict)
    if enabled is not None:
        HUB.enabled = bool(enabled)
    return HUB


_LANE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def production_root() -> Path:
    """Where the production entry points persist the sinks: state/learning/hub/ (trusted side; nothing on the decision path reads it)."""
    from engine import config as K
    return Path(K.STATE) / "learning" / "hub"


def configure_production(lane: str, root: str | Path | None = None, board_path: str | Path | None = None,
                         strict: bool = False) -> Hub:
    """Called by a production entry point (improve.weekly; livesim_loop2's worker when --learner legit) - never on import, never
    by a test unless it passes its own tmp root. Sinks then append to <root>/<lane>/<sink>.jsonl: one lane per process, so parallel
    workers never interleave lines in one file. The KnowledgeBoard at `board_path` (default <root>/board.jsonl, shared by every
    lane and read-only here) makes effective_weight() board-scaled. NEUTRAL WITH NO BOARD: an empty board registers nothing, so every
    legacy weight passes through unchanged until knowledge is registered and promoted on it (strict=False)."""
    if not _LANE.match(lane):
        raise ValueError(f"lane {lane!r} must be 1-64 characters of [A-Za-z0-9_.-] (it names a directory)")
    base = Path(root) if root is not None else production_root()
    board = KnowledgeBoard(Path(board_path) if board_path is not None else base / "board.jsonl")
    configure(root=base / lane, board=board, strict=strict, enabled=True)
    HUB.lane = lane
    HUB.calls["production_config"] += 1
    return HUB


ENV_HUB = "WEEKLY7_HUB"                  # "off" disables the sinks for the process
ENV_HUB_ROOT = "WEEKLY7_HUB_ROOT"        # persistence root (default state/learning/hub)
ENV_HUB_LANE = "WEEKLY7_HUB_LANE"        # lane name when the entry point does not pass one
ENV_HUB_BOARD = "WEEKLY7_HUB_BOARD"      # board file (default <root>/board.jsonl)
ENV_HUB_STRICT = "WEEKLY7_HUB_STRICT"    # "1" -> effective_weight() is 0 for knowledge the board never saw


def configure_from_env(lane: str | None = None, root: str | Path | None = None, environ: Mapping[str, str] | None = None) -> Hub | None:
    """The one line an ENTRY SCRIPT calls to make the sinks persist: scripts/research_loop.py -> `wiring.configure_from_env(run_id)`.
    Arguments beat the environment; the environment beats the defaults (state/learning/hub/<lane>/). WEEKLY7_HUB=off returns None and
    leaves the hub untouched (an operator kill switch, never a silent default). Returns the configured Hub, whose root the caller can
    print. Deterministic: no clock, no randomness."""
    env = os.environ if environ is None else environ
    if str(env.get(ENV_HUB, "")).strip().lower() in ("off", "0", "false", "no"):
        return None
    name = lane or env.get(ENV_HUB_LANE) or "research_loop"
    base = root if root is not None else (env.get(ENV_HUB_ROOT) or None)
    return configure_production(name, root=base, board_path=env.get(ENV_HUB_BOARD) or None,
                                strict=str(env.get(ENV_HUB_STRICT, "")).strip() in ("1", "true", "yes"))


def sink(name: str) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator for observer hooks: counts the call, swallows-and-records any failure, and is inert when the hub is disabled."""
    def deco(fn: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(fn)
        def run(*a: Any, **k: Any) -> T | None:
            if not HUB.enabled:
                return None
            HUB.calls[name] += 1
            try:
                return fn(*a, **k)
            except Exception as e:                                # noqa: BLE001 - a sink must never break the caller
                HUB.record_error(name, e)
                return None
        return run
    return deco


# ================================================================================================================== failure
def _worst_taken_losses(frame: pd.DataFrame, now: Any, cost: float, limit: int) -> pd.DataFrame:
    """Taken rows that lost and resolved strictly before `now`, worst first, at most `limit`. Vectorised: a post-mortem frame can hold
    every candidate of every week."""
    if frame.empty or not {"score", "y", "resolved"} <= set(frame.columns):
        return frame.iloc[:0]
    taken = frame["taken"].astype(bool) if "taken" in frame else pd.Series(True, index=frame.index)
    side = frame["side"].astype(float) if "side" in frame else np.sign(frame["score"]).replace(0, 1)
    pnl = side * frame["y"].astype(float) - cost
    ok = taken & (frame["resolved"] < pd.Timestamp(now)) & (pnl < 0)
    sub = frame[ok]
    if sub.empty:
        return sub
    order = pnl[ok].sort_values(kind="mergesort").index[:limit]
    return sub.loc[order]


@sink("post_mortem")
def on_post_mortem(frame: pd.DataFrame, X: pd.DataFrame, now: Any, max_classify: int = MAX_CLASSIFY) -> dict[str, int]:
    """lessons.post_mortem -> failure ledger + hypotheses. The classifier runs on the worst losses only; each rank_pct is taken from the
    FULL cross-section of its date, not from the subset, so the selection detector sees the rank the decision really had.
    Returns counts. A loss with no explanation is recorded as UNKNOWN / INSUFFICIENT_EVIDENCE: honest, and it still teaches what to collect."""
    hub = HUB
    if frame is None or frame.empty:
        return {"seen": 0, "classified": 0, "named": 0, "hypotheses": 0}
    cost = 0.0005
    worst = _worst_taken_losses(frame, now, cost, max_classify)
    if worst.empty:
        return {"seen": len(frame), "classified": 0, "named": 0, "hypotheses": 0}
    rank_full = frame["score"].abs().groupby(level=0).rank(pct=True)
    records = F.records_from_lessons_frame(worst, X, now, cost=cost)
    kept = worst[worst["resolved"] < pd.Timestamp(now)]
    if len(records) != len(kept):
        raise ValueError(f"record/row misalignment ({len(records)} vs {len(kept)}): refusing to attach ranks by position")
    classified = named = n_hyp = 0
    for t, idx in zip(records, kept.index):
        if t.rid in hub._classified:
            continue
        t = dataclasses.replace(t, rank_pct=float(rank_full.at[idx]))
        cls = hub.classifier.classify(t, None, now)
        hub._classified.add(t.rid)
        hub.failure_ledger.add(cls)
        hub.persist("failures", t.rid, {"rid": t.rid, "cause": cls.cause.value, "named": cls.named, "confidence": cls.confidence,
                                        "subsystem": (cls.top_subsystem() or Subsystem.SELECTION).value, "decided_at": t.decided_at,
                                        "resolved_at": t.resolved_at, "pnl": t.pnl, "rank_pct": t.rank_pct,
                                        "unknown_state": cls.unknown_state.value if cls.unknown_state is not None else None})
        classified += 1
        named += cls.named
        hyp = make_hypothesis(cls.cause, cls.top_subsystem() or Subsystem.SELECTION, t, cls)
        if hyp is not None and _store_hypothesis(hyp):
            n_hyp += 1
    hub.delivered["failure_classifications"] += classified
    return {"seen": len(frame), "classified": classified, "named": named, "hypotheses": n_hyp}


def _store_hypothesis(h: F.Hypothesis) -> bool:
    errs = h.validate()
    if errs:
        raise ValueError("; ".join(errs))
    if h.hid in HUB.hypotheses:
        return False
    HUB.hypotheses[h.hid] = h
    HUB.delivered["hypotheses"] += 1
    HUB.persist("hypotheses", h.hid, h)
    return True


@sink("lessons")
def on_lessons(lessons: Iterable[Any] | Mapping[str, Iterable[Any]]) -> list[str]:
    """LessonBook.learn / learn_by_kind and Memory.export_lessons -> failure hypotheses. Accepts a list or the {kind: [lessons]} that
    learn_by_kind returns. Returns the ids of hypotheses that were NEW to the hub. A hypothesis never has a production effect."""
    flat: list[Any] = []
    if isinstance(lessons, Mapping):
        for v in lessons.values():
            flat.extend(v)
    else:
        flat.extend(lessons)
    fresh = []
    for h in F.hypotheses_from_lessons(flat):
        if _store_hypothesis(h):
            fresh.append(h.hid)
    return fresh


def post_mortem_frame(decided: Any, closed: Any, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence[Any],
                      score: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One closed adapter week as the (frame, X) pair lessons.post_mortem / on_post_mortem read: index (decided, ticker), columns
    score (the pick score, 0 where unscored), y (the week's realised return), resolved (the close the outcome matured on), taken.
    X carries p0's market-context columns so the classifier sees the regime the decision was made in. Rows without an outcome
    are dropped (a name that stopped trading has no loss to explain yet)."""
    if p0 is None or p0.empty:
        empty = pd.DataFrame(columns=["score", "y", "resolved", "taken"])
        return empty, pd.DataFrame()
    y = fwd.reindex(p0.index).astype(float)
    sc = (score.reindex(p0.index).astype(float) if score is not None else pd.Series(0.0, index=p0.index)).fillna(0.0)
    keep = y.notna().to_numpy()
    names = p0.index[keep]
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([pd.Timestamp(as_date(decided))] * len(names)), names], names=["date", "ticker"])
    pk = set(picked)
    frame = pd.DataFrame({"score": sc[keep].to_numpy(), "y": y[keep].to_numpy(),
                          "resolved": pd.Timestamp(as_date(closed)), "taken": [t in pk for t in names]}, index=idx)
    ctx = [c for c in p0.columns if str(c).startswith("m_")]
    X = pd.DataFrame(p0.loc[names, ctx].to_numpy(), index=idx, columns=ctx) if ctx else pd.DataFrame(index=idx)
    return frame, X


@sink("adapter_post_mortem")
def adapter_week_frame(decided: Any, closed: Any, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence[Any],
                       score: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """post_mortem_frame with sink semantics, for the adapter: a failure to build the frame is recorded, never raised into a decision."""
    return post_mortem_frame(decided, closed, p0, fwd, picked, score)


@sink("session_lessons")
def on_session_end(memory: Any) -> int:
    """End of an adaptive session (the path the Test loop really runs): the adapter's memory exports its sanitised lesson bank, which
    reaches on_lessons through memory.export_lessons (the memory.py hook). Returns the number of lessons exported."""
    if memory is None or not len(memory):
        return 0
    return int(len(memory.export_lessons()))


# ================================================================================================================== missed winners
def week_of(decided: Any, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence[Any], score: pd.Series | None = None, k: int = 10,
            thr: float = MW.WINNER, resolved_at: str | None = None, era: str = "", salt: str = "mw") -> MW.Week:
    """learning.missed_winners.week_from_base, vectorised over the cross-section (the Test loop hands it ~3,000 names a week; the
    per-name version costs seconds per week). Same candidates, ids, features, ranks and kinds - tests hold the two equal - for the
    case the adapter uses (every name eligible, no extra decision trail)."""
    from engine import missed_winners as base
    feats = [c for c in base.DET_FEATS if c in p0.columns]
    R = MW.cross_rank(p0, feats)[feats].to_numpy(dtype=float) if feats else np.empty((len(p0), 0))
    f = fwd.reindex(p0.index).astype(float).to_numpy() if fwd is not None else np.full(len(p0), np.nan)
    sc = score.reindex(p0.index).astype(float).to_numpy() if score is not None else None
    rk = score.rank(ascending=False, method="first").reindex(p0.index).to_numpy(dtype=float) if score is not None else None
    kinds = base.winner_types(p0).to_numpy()
    pk = set(picked)
    ds = str(decided)
    cands = []
    for i, tkr in enumerate(p0.index):
        cands.append(MW.Candidate(
            cid=stable_hash({"s": salt, "d": ds, "t": str(tkr)}, 12), features={c: float(R[i, j]) for j, c in enumerate(feats)},
            fwd=float(f[i]) if np.isfinite(f[i]) else None, picked=tkr in pk,
            score=float(sc[i]) if sc is not None and np.isfinite(sc[i]) else None,
            rank=int(rk[i]) if rk is not None and np.isfinite(rk[i]) else None, eligible=True, filters_hit=(), kind=str(kinds[i])))
    d0 = as_date(decided)
    return MW.Week(label=stable_hash({"s": salt, "d": ds}, 8), decided_at=str(d0),
                   resolved_at=str(resolved_at or d0 + pd.Timedelta(days=7).to_pytimedelta()), candidates=tuple(cands), k=k, thr=thr, era=era)


@sink("missed_week")
def on_missed_week(decided: Any, closed: Any, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence[Any], score: pd.Series | None = None,
                   era: str = "", thr: float = MW.WINNER, k: int = 10) -> int:
    """A closed adapter week -> learning MissedLearningLedger.add_week (WHY each winner was rejected). `decided` is the decision date,
    `closed` the date the outcome matured. Returns the number of rejections explained."""
    week = week_of(decided, p0, fwd, list(picked), score=score, k=k, thr=thr, resolved_at=str(as_date(closed)), era=era)
    errs = week.validate()
    if errs:
        raise ValueError("; ".join(errs[:3]))
    rej = HUB.missed.add_week(week)
    HUB.delivered["rejections"] += len(rej)
    kinds = {c.cid: c for c in week.candidates}
    for r in rej:                                  # the ticker never leaves: cid is a salted hash of (date, ticker)
        c = kinds[r.cid]
        HUB.persist("missed_winners", stable_hash([week.label, r.cid, r.primary.value]),
                    {"period": week.label, "decided_at": week.decided_at, "resolved_at": week.resolved_at, "era": week.era,
                     "reason": r.primary.value, "named": r.named, "kind": c.kind, "fwd": c.fwd, "rank": c.rank})
    return len(rej)


# ================================================================================================================== experiments
def _question_of(rec: Mapping[str, Any]) -> str:
    return str(rec.get("question") or rec.get("desc") or rec.get("reason") or rec.get("event") or "experiment")


def legacy_row(rec: Mapping[str, Any]) -> dict[str, Any]:
    """A registry record as the row experiment_memory.legacy_to_record reads: the design config is `model_params` (what the writer
    passed as cfg) and the question is the challenger description when there is one, else the event or reason."""
    row = dict(rec)
    params = rec.get("model_params")
    if isinstance(params, Mapping):
        row["config"] = dict(params)
    row["question"] = _question_of(rec)
    return row


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(rows, bad_line_count) of a JSON-lines file; a missing file is ([], 0) and a malformed line is counted, never fatal."""
    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    bad = 0
    for ln in path.read_bytes().decode("utf-8", errors="replace").split("\n"):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


@sink("experiment")
def on_experiment(rec: Mapping[str, Any], registry_path: str | Path) -> str:
    """improve.log_experiment -> the ExperimentLedger facade (a legacy row is imported honestly: fields the old writer never captured
    are marked NOT RECORDED). Returns the ledger status of the row ('added' or 'skipped')."""
    led = HUB.ledger_for(registry_path, backfill=True)
    out = EM.import_legacy(led, [legacy_row(rec)])
    HUB.delivered["ledger_rows"] += out["added"]
    return "added" if out["added"] else "skipped"


@sink("memory_entry")
def on_memory_entry(experiment_id: str, change: Mapping[str, Any], answers: Mapping[str, Any], now: Any, registry_path: str | Path,
                    version_existing: bool = False) -> str:
    """registry.ExperimentMemory.record -> the same facade, so the two experiment stores agree (adapter: no old behaviour changes).
    version_existing (S21b, improve.log_experiment - the path production really runs): the decided experiment is already in the
    ledger, so its Phase-30 answers are written as a NEW VERSION of that same record (append-only; never a second experiment)."""
    led = HUB.ledger_for(registry_path)
    hist = led.history(experiment_id) if version_existing else []
    if version_existing and not hist:
        return "absent"                          # the record never reached the ledger (its own sink failed): nothing to version
    if hist:
        cur = hist[-1]
        if "phase30" in cur.tags:
            return "skipped"
        verdict = "adopted" if answers.get("adopted") else "rejected"
        why = answers.get("why_changed") if answers.get("adopted") else answers.get("if_rejected_why")
        learned = tuple(cur.learned) + (f"{verdict}: {answers.get('what_changed') or cur.question}" + (f" - {why}" if why else ""),)
        tags = tuple(cur.tags) + ("phase30",) + tuple(f"{k}={answers[k]}" for k in ("data_used", "data_unseen") if k in answers)
        missing = tuple(m for m in cur.legacy_missing if m != "learned")
        led._append(cur.with_(version=cur.version + 1, recorded_at=cur.recorded_at, learned=learned, tags=tags, legacy_missing=missing))
        HUB.delivered["memory_versions"] += 1
        return "versioned"
    row = {"experiment_id": f"{experiment_id}:memory", "config": dict(change), "question": str(answers.get("what_changed") or answers.get("why_changed") or "experiment"),
           "outcome": "adopt" if answers.get("adopted") else "reject", "reason": str(answers.get("if_rejected_why") or answers.get("why_changed") or ""),
           "t": str(now)}
    out = EM.import_legacy(HUB.ledger_for(registry_path), [row])
    return "added" if out["added"] else "skipped"


def memory_answers(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The Phase-30 answers a decided registry record really carries (adopted, what/why changed, the rejection reason, the windows and
    metrics it was judged on). Questions the record does not answer are ABSENT, never filled with a placeholder: the ledger marks
    them NOT RECORDED rather than pretending they were asked."""
    oc = str(rec.get("outcome") or "").lower()
    if oc not in ("adopt", "reject"):
        raise ValueError(f"only a decided record (adopt/reject) is a memory entry, not outcome {oc!r}")
    out: dict[str, Any] = {"adopted": oc == "adopt", "what_changed": _question_of(rec)}
    reason = rec.get("reason")
    if reason:
        out["why_changed" if oc == "adopt" else "if_rejected_why"] = str(reason)
    for src, dst in (("window_ids", "data_used"), ("test_range", "data_unseen")):
        v = rec.get(src)
        if v not in (None, "", [], {}):
            out[dst] = v
    return out


def pre_launch(question: str, config: Mapping[str, Any], now: Any, registry_path: str | Path, seed: int | None = 7,
               data_hash: str = "", code_hash: str = "") -> EM.DuplicateVerdict | None:
    """'Have we already tested this?' for a candidate about to launch. None when the hub is disabled or the ledger can not answer
    (never blocks on its own failure). `code_hash`/`data_hash` default to the running code and empty data; a changed hash on an
    otherwise identical design downgrades an exact repeat to RETEST_JUSTIFIED, which does not block."""
    if not HUB.enabled:
        return None
    HUB.calls["pre_launch"] += 1
    try:
        led = HUB.ledger_for(registry_path, backfill=True)
        design = EM.DesignSpec(config=dict(config), seed=seed, code_hash=code_hash or current_code_hash(), data_hash=data_hash)
        return led.already_tested(question, design, now)
    except Exception as e:                                        # noqa: BLE001
        HUB.record_error("pre_launch", e)
        return None


def blocks_launch(verdict: EM.DuplicateVerdict | None) -> bool:
    """Only verdicts decided by the CONFIGURATION block a launch (same config, near duplicate, known failure nearby). 'Same question
    answered' is shown to the caller but does not block: two challenges worded alike can differ in what they change."""
    return verdict is not None and verdict.status in BLOCKING_LAUNCH


def repro_dict(rec: Mapping[str, Any]) -> dict[str, Any] | None:
    """A ReproRecord for one registry record, built from the stamp log_experiment already computed (no second code hash). The memory
    snapshot of a legacy record is the explicit marker 'unrecorded'. None when the record can not be described (sink semantics)."""
    if not HUB.enabled:
        return None
    HUB.calls["repro"] += 1
    try:
        if HUB._worker is None:
            HUB._worker = capture_worker()
        seed = rec.get("seed")
        params = rec.get("model_params")
        r = make_record(params if isinstance(params, Mapping) else {},
                        int(seed) if isinstance(seed, (int, float, str)) and str(seed).lstrip("-").isdigit() else None,
                        data=str(rec.get("data_snapshot") or ""), memory_hash=str(rec.get("memory_hash") or UNRECORDED),
                        experiment_id=str(rec.get("experiment_id") or "") or None, code_hash=str(rec.get("code_hash") or ""),
                        worker=HUB._worker, code_files=list(rec.get("code_files") or []))
        HUB.delivered["repro_records"] += 1
        return r.to_dict()
    except Exception as e:                                        # noqa: BLE001
        HUB.record_error("repro", e)
        return None


def repro_from_registry(rec: Mapping[str, Any]) -> ReproRecord | None:
    """Read back the ReproRecord stored on a registry record (None if it has none or it is malformed)."""
    d = rec.get("repro")
    if not isinstance(d, Mapping):
        return None
    try:
        return ReproRecord.from_dict(d)
    except (KeyError, TypeError, ValueError):
        return None


def registry_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Registry.audit() -> provenance findings from the firewall layer, summarised. Adds information only; Registry.audit()['ok'] is
    unchanged. `repro_records` counts how many records carry a well-formed ReproRecord."""
    HUB.calls["registry_audit"] += 1
    try:
        fs: list[Finding] = registry_findings(list(records)) if len(records) else []
    except Exception as e:                                        # noqa: BLE001
        HUB.record_error("registry_audit", e)
        return {"error": str(e)}
    by = Counter(f.check for f in fs)
    with_repro = sum(1 for r in records if (rr := repro_from_registry(r)) is not None and not rr.validate())
    return {"n_findings": len(fs), "by_check": dict(sorted(by.items())), "repro_records": with_repro, "n_records": len(records)}


# ================================================================================================================== redundancy graph
@sink("redundancy")
def on_redundancy(names: Sequence[str], dup: Mapping[int, int], masks: Any, known_at: Any, max_overlap: float = 0.8) -> int:
    """PatternMiner's prune_redundant result -> REDUNDANT_WITH edges (dropped pattern -> the keeper that displaced it, weighted by the
    overlap that condemned it). The pruning decision stays with the miner; the graph only remembers the relationship. `masks` is
    the same indexable the miner pruned with, so nothing is recomputed. Returns the number of edges written."""
    from engine import pattern_stats as ps
    from .core import Edge
    g = HUB.graph
    n = 0
    for dropped, keeper in sorted(dup.items()):
        a, b = str(names[dropped]), str(names[keeper])
        for nm in (a, b):
            g.add_node(nm, NodeType.PATTERN, known_at, nm)
        ov = float(ps.jaccard(masks[dropped], masks[keeper]))
        g.add_edge(a, b, Edge.REDUNDANT_WITH, known_at, weight=min(1.0, ov), attrs={"overlap": round(ov, 6), "metric": "jaccard",
                                                                                   "max_overlap": max_overlap})
        n += 1
    HUB.delivered["redundancy_edges"] += n
    return n


# ================================================================================================================== contradiction period loop
@sink("contradiction_period")
def on_period(now: Any, ledger: ContradictionLedger | None = None) -> MonitorReport:
    """Once per period (S20): ContradictionMonitor(graph, ledger).run_period(now) over the hub's knowledge graph. The monitor is kept
    so its run order is enforced across calls (a period not after the previous one is a FirewallBreach, recorded as a sink error).
    Returns the report; None when the run failed."""
    if HUB.monitor is None:
        HUB.monitor = ContradictionMonitor(HUB.graph, ledger)
    rep = HUB.monitor.run_period(now)
    HUB.delivered["contradictions_tracked"] += len(rep.items)
    return rep


def research_step(report: MonitorReport, engine: Any, ledger: EM.ExperimentLedger, budget: Any, now: Any, seed: int) -> Any:
    """Feed the period's contradiction signals into ResearchPriorityEngine.step (an open contradiction becomes a research question).
    Signals are built by the monitor; the engine and budget are the caller's. Returns the EngineStep."""
    if HUB.monitor is None:
        raise FirewallBreach("research_step before on_period: there is no monitor to take signals from")
    HUB.calls["research_step"] += 1
    signals = HUB.monitor.signals(report)
    HUB.delivered["contradiction_signals"] += len(signals)
    return engine.step(signals, ledger, budget, now, seed)


def write_dashboard_inputs(report: MonitorReport, root: str | Path) -> dict[str, Path]:
    """Write the monitor's rows where reports.health_dashboard reads them: knowledge.jsonl (one row per knowledge id touched by a
    contradiction, from dashboard_rows) and contradictions.jsonl (one row per tracked pair; open ones carry verdict UNRESOLVED, which
    is what makes assess_knowledge mark both ids CONTRADICTED). Rows are dated the report's own day, so ReportContext(now=that day)
    excludes them, exactly like any record made on `now`; pass the NEXT period as its `now`."""
    if HUB.monitor is None:
        raise FirewallBreach("write_dashboard_inputs before on_period: there is no monitor")
    HUB.calls["dashboard"] += 1
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    rows = HUB.monitor.dashboard_rows(report)["rows"]
    know = [{**r, "version": 1, "at": report.now, "lifecycle": "ACTIVE", "confidence": {},
             "evidence": {"note": r["evidence"], "sample_size": r["sample_size"]}} for r in rows]     # shape assess_knowledge reads
    con = [{"pair": list(t.pair), "verdict": "UNRESOLVED" if t.is_open else t.phase.value, "at": report.now} for t in report.items]
    out = {"knowledge": base / "knowledge.jsonl", "contradictions": base / "contradictions.jsonl"}
    for name, data in (("knowledge", know), ("contradictions", con)):
        out[name].write_bytes((NL.join(json.dumps(r, sort_keys=True, default=str) for r in data) + (NL if data else "")).encode("utf-8"))
    HUB.delivered["dashboard_rows"] += len(know)
    return out


# ================================================================================================================== production weight
def _scope_chain(scope: str) -> list[str]:
    return [scope] if scope == "global" else [scope, "global"]


def deciding_member(kid: str) -> str | None:
    """The board member of `kid` that carries decision weight (role CHAMPION; KnowledgeBoard.weight is the authority). None when no
    member of `kid` is a champion, or no board is configured."""
    b = HUB.board
    if b is None:
        return None
    for mid, m in sorted(b.members.items()):
        if m.knowledge_id == kid and b.weight(mid) > 0:
            return mid
    return None


def champion_of(effect: Any, subsystem: Any, scope: str) -> str | None:
    """champion.effective_champion for a production request: the most specific champion along (scope, global). None = no champion
    anywhere, and the caller uses its general rule (neutral with no board)."""
    from .champion import effective_champion
    if HUB.board is None:
        return None
    found: str | None = effective_champion(HUB.board, effect, subsystem, _scope_chain(scope))
    return found


def board_weight(kid: str) -> float | None:
    """The board's decision weight for a knowledge id: 1 if any of its members is a champion, 0 if it is registered but not one,
    None if the board has never heard of it (or no board is configured)."""
    b = HUB.board
    if b is None:
        return None
    mine = [mid for mid, m in b.members.items() if m.knowledge_id == kid]
    if not mine:
        return None
    return max(b.weight(mid) for mid in mine)


def effective_weight(kid: str, legacy_weight: float, run: str | None = None) -> float:
    """What a production reader should use for knowledge `kid`. ADAPTER: with no board configured, or knowledge the board never
    registered (and strict off), this returns `legacy_weight` unchanged, so untested paths do not move. Registered knowledge is
    scaled by the board: a shadow or challenger has weight exactly 0 and can never influence a decision; a champion keeps its weight.
    With `run`, every registered member that carried non-zero weight is noted for end_decision_run's source audit."""
    HUB.calls["weight"] += 1
    bw = board_weight(kid)
    if bw is None:
        # visible, not silent: a pass-through means the board never heard of this knowledge (hook_report shows the count)
        HUB.delivered["weight_unregistered_passthrough" if not (HUB.strict and HUB.board is not None) else "weight_unregistered_zeroed"] += 1
        return 0.0 if (HUB.strict and HUB.board is not None) else float(legacy_weight)
    HUB.delivered["weight_board_scaled"] += 1
    w = float(legacy_weight) * bw
    if run is not None and w != 0.0:
        mid = deciding_member(kid)
        if mid is not None:
            HUB.used.setdefault(run, set()).add(mid)
    return w


def note_use(run: str, mids: Iterable[str]) -> None:
    """A decision path that reads the board directly (e.g. a learner scoring with champion knowledge) declares what it used."""
    HUB.used.setdefault(run, set()).update(str(m) for m in mids)


def end_decision_run(run: str) -> list[dict[str, Any]]:
    """After each decision run that consulted the board: champion.audit_decision_sources over every member that carried weight.
    A non-champion that influenced a decision is a violation; each is kept in HUB.source_violations and persisted. With no board
    this is neutral (nothing could have been used) and returns []. With strict on, a violation raises FirewallBreach."""
    from .champion import audit_decision_sources
    HUB.calls["decision_sources"] += 1
    used = HUB.used.pop(run, set())
    if HUB.board is None:
        return []
    from .champion import Slot
    bad: list[dict[str, Any]] = audit_decision_sources(HUB.board, {run: sorted(used)})
    flagged = {str(v["mid"]) for v in bad}
    for mid in sorted(used - flagged):             # a champion that is not the EFFECTIVE champion of its slot decided out of turn
        slot = Slot.from_key(HUB.board.members[mid].slot)
        eff = champion_of(slot.effect, slot.subsystem, slot.scope)
        if eff != mid:
            bad.append({"decision": run, "mid": mid, "problem": f"not the effective champion of {slot.key} (that is {eff})"})
    HUB.delivered["decision_runs_audited"] += 1
    HUB.source_violations.extend(bad)
    for v in bad:
        HUB.persist("decision_source_violations", stable_hash([run, v]), v)
    if bad and HUB.strict:
        raise FirewallBreach(f"decision run {run} used non-champion knowledge: {bad[:3]}")
    return bad


# ================================================================================================================== promotion gate
def register_evidence(claim_id: str, evidence: LearningEvidence) -> None:
    """The Test loop registers the evidence for a learning claim under the challenger's id before promotion is attempted."""
    HUB.evidence[claim_id] = evidence


def _guarded(name: str, fn: Callable[[], Check]) -> Check:
    """A check that raises is a FAILED check (fail closed), never an exception into the promotion path."""
    try:
        return fn()
    except Exception as e:                                        # noqa: BLE001
        return Check(name, False, f"check could not run: {type(e).__name__}: {str(e)[:200]}")


def _check_scorecard(ev: LearningEvidence) -> Check:
    if ev.card is None:
        return Check("scorecard", False, "no learning scorecard supplied")
    dec = gate_improvement_claim(ev.card)
    return Check("scorecard", dec.allowed, "ok" if dec.allowed else "; ".join(dec.blockers[:3]))


def _check_firewalls(ev: LearningEvidence, now: Any) -> Check:
    if ev.gate_ctx is None:
        return Check("firewalls", False, "no firewall context supplied")
    if as_date(ev.gate_ctx.now) != as_date(now):
        return Check("firewalls", False, f"firewall context is for {ev.gate_ctx.now}, promotion is at {now}")
    gv = LearningFirewallGate().evaluate(ev.gate_ctx)
    failed = ",".join(sorted(str(x) for x in gv.failed_layers))
    return Check("firewalls", gv.passed, "ok" if gv.passed else ("failed layers " + failed if failed else "no layer was relevant"))


def _check_identity(ev: LearningEvidence) -> Check:
    if ev.identity is None:
        return Check("identity", False, "no identity-harness report supplied")
    rep = ev.identity.run() if isinstance(ev.identity, IdentityJob) else ev.identity
    return Check("identity", rep.passed, "ok" if rep.passed else identity_failure_reason(rep))


def identity_failure_reason(rep: IdentityReport) -> str:
    """Why an identity report did not pass, in the words that are true of it (F10; F07 found the old text said 'collapsed' when no
    attack had collapsed: a no-op attack had raised a harness finding).  In order: nothing was attacked; attacks that collapsed;
    attacks that could not be judged (no skill to lose, too few dates, nondeterministic); the run itself was nondeterministic;
    failing harness findings (e.g. an attack that changed no identity tested nothing)."""
    parts = []
    if not rep.verdicts:
        parts.append("no attack was run")
    if rep.collapsed:
        parts.append("collapsed under " + ", ".join(rep.collapsed))
    odd = sorted({f"{v.kind}[{v.mode}]={v.status}" for v in rep.verdicts if v.status not in ("OK", "COLLAPSE")})
    if odd:
        parts.append("not judged: " + ", ".join(odd[:4]) + (f" (+{len(odd) - 4})" if len(odd) > 4 else ""))
    if not rep.deterministic:
        parts.append("nondeterministic: two identical runs disagreed")
    fails = [f for f in rep.findings if f.is_fail]
    if fails:
        parts.append("harness finding: " + "; ".join(f"{f.check} ({f.subject}): {f.message}"[:120] for f in fails[:2]))
    return "; ".join(parts) or "failed without a recorded reason"


def promotion_gate(subject: str, now: Any, claims_learning: bool = False, evidence: LearningEvidence | None = None) -> PromotionVerdict:
    """The composite gate in front of promotion.
      claims_learning=False  (a config / weight change judged by the live-shadow z-test): the old test governs; recorded as such.
      claims_learning=True   scorecard (gate_improvement_claim) AND all eight firewall layers (LearningFirewallGate) AND the identity
                             harness must all pass; a missing piece is a failed check. The gate context must be for the same `now`.
    Never returns VALIDATED: the label stays IMPLEMENTED - NOT VALIDATED until a sealed blind window says otherwise."""
    HUB.calls["promotion_gate"] += 1
    checks: list[Check] = []
    if not claims_learning:
        checks.append(Check("no_learning_claim", True, "no learning claim made; the live-shadow z-test governs (scorecard, firewall and identity gates apply to learning claims only)"))
    else:
        ev = evidence or LearningEvidence()
        checks.append(_guarded("scorecard", lambda: _check_scorecard(ev)))
        checks.append(_guarded("firewalls", lambda: _check_firewalls(ev, now)))
        checks.append(_guarded("identity", lambda: _check_identity(ev)))
    v = PromotionVerdict(subject, str(as_date(now)), claims_learning, all(c.ok for c in checks), tuple(checks))
    HUB.decisions.append(v)
    HUB.persist("promotion_decisions", v.digest(), {"subject": v.subject, "now": v.now, "allowed": v.allowed, "blockers": list(v.blockers)})
    return v


def require_promotion(subject: str, now: Any, claims_learning: bool = False, evidence: LearningEvidence | None = None) -> PromotionVerdict:
    """promotion_gate that raises FirewallBreach instead of returning a refusal - for callers that must not catch-and-continue."""
    v = promotion_gate(subject, now, claims_learning, evidence)
    if not v.allowed:
        raise FirewallBreach(f"promotion of {subject} refused at {v.now}: " + "; ".join(v.blockers))
    return v


LEGACY_CHALLENGER_KINDS = frozenset({"config", "meta", "evidence_weights", "recalibrate"})   # improve.spawn_challengers' vocabulary


def challenger_claims_learning(challenger: Mapping[str, Any]) -> bool:
    """Does this challenger make a LEARNING claim? A config / meta-weight / evidence-weight / recalibration change is judged by the
    live-shadow z-test. Anything else - an unknown or missing kind, evidence already registered under its id, or an explicit
    claims_learning=True - is a learning claim. A challenger cannot opt OUT by writing claims_learning=False: the kind decides
    (the gate used to be bypassed because nothing ever set the flag)."""
    if bool(challenger.get("claims_learning")):
        return True
    if str(challenger.get("id", "")) in HUB.evidence:
        return True
    return str(challenger.get("kind") or "") not in LEGACY_CHALLENGER_KINDS


def promotion_allowed(challenger: Mapping[str, Any], now: Any) -> PromotionVerdict:
    """improve.test_and_promote hook. A challenger of a legacy kind (config / meta / evidence_weights / recalibrate) is judged by its
    live-shadow test alone; every other challenger is a learning claim (challenger_claims_learning) and needs registered evidence, else
    the gate FAILS CLOSED on scorecard, firewalls and identity."""
    cid = str(challenger.get("id", ""))
    return promotion_gate(cid or "challenger", now, challenger_claims_learning(challenger), HUB.evidence.get(cid))


def register_scorecard(kid: str, card: LearningScorecard) -> None:
    """The learner (or its runner) states the scorecard its learned knowledge `kid` is promoted on; the card must be valid."""
    errs = card.check()
    if errs:
        raise ValueError(f"scorecard for {kid} is invalid: {'; '.join(errs[:3])}")
    HUB.scorecards[kid] = card


def register_identity(kid: str, report: IdentityReport | IdentityJob) -> None:
    """The identity-harness result (or a deferred job) for learned knowledge `kid`."""
    HUB.identity_reports[kid] = report


def store_scorecard(card: LearningScorecard) -> str | None:
    """scorecard -> ScorecardStore.append under the configured root (append-only, hash-chained). None with no root. A learner
    version already stored is not an error here: the store refuses to overwrite history, and the refusal is counted."""
    from .scorecard import ScorecardStore
    if HUB.root is None:
        return None
    try:
        return ScorecardStore(HUB.root / "scorecards.jsonl").append(card)
    except FileExistsError:
        HUB.delivered["scorecard_already_stored"] += 1
        return None


def learning_evidence_for(k: Any, now: Any) -> LearningEvidence:
    """The LearningEvidence of a knowledge promotion: the registered scorecard and identity report (None when nobody registered
    one - a failed check at the gate, never a pass), and a firewall context over the knowledge item itself, checked by the memory
    and provenance layers (could this item exist at `now`, and is its lineage complete)."""
    from .firewalls import LayerName
    kid = str(getattr(k, "knowledge_id", ""))
    ctx = GateContext(now=now, subject=kid, items=[k], relevant=frozenset({LayerName.MEMORY, LayerName.PROVENANCE}))
    return LearningEvidence(card=HUB.scorecards.get(kid), gate_ctx=ctx, identity=HUB.identity_reports.get(kid))


def on_knowledge_promotion(k: Any, now: Any) -> PromotionVerdict:
    """Every promotion attempt of LEARNED knowledge (promotion.PromotionGate.evaluate) is a learning claim: its evidence is
    registered under the knowledge id and the composite gate runs with claims_learning=True, so the scorecard, firewall and identity
    checks genuinely execute. Whether the verdict blocks is the PromotionGate's choice (learning_claim='enforce'); it is always
    recorded and persisted with the other promotion decisions."""
    HUB.calls["knowledge_promotion"] += 1
    kid = str(getattr(k, "knowledge_id", "")) or "knowledge"
    ev = learning_evidence_for(k, now)
    register_evidence(kid, ev)
    v = promotion_gate(kid, now, claims_learning=True, evidence=ev)
    HUB.knowledge_verdicts[kid] = v
    HUB.delivered["learning_claims_evaluated"] += 1
    return v


# ================================================================================================================== health of the wiring itself
@dataclasses.dataclass(frozen=True)
class Hook:
    """One declared seam: which wiring function an old file must call, where, and what it delivers."""
    function: str
    sites: tuple[str, ...]                # repo-relative files that must contain a call to wiring.<function>
    delivers: str


HOOKS: dict[str, Hook] = {
    "post_mortem": Hook("on_post_mortem", ("engine/lessons.py",), "failure classifications + hypotheses"),
    "lessons": Hook("on_lessons", ("engine/lessons.py", "engine/memory.py"), "failure hypotheses"),
    "missed_week": Hook("on_missed_week", ("engine/missed_winners.py",), "why-rejected ledger"),
    "experiment": Hook("on_experiment", ("engine/improve.py",), "ExperimentLedger facade row"),
    "memory_entry": Hook("on_memory_entry", ("engine/registry.py", "engine/improve.py"), "ExperimentLedger facade row"),
    "adapter_post_mortem": Hook("on_post_mortem", ("engine/adaptive.py",), "the Test loop's closed weeks -> failure ledger"),
    "session_lessons": Hook("on_session_end", ("engine/adaptive.py",), "adapter memory lessons -> hypotheses"),
    "knowledge_promotion": Hook("on_knowledge_promotion", ("engine/learning/promotion.py",), "learning-claim verdict per promotion"),
    "decision_sources": Hook("end_decision_run", ("engine/lessons.py",), "champion-only source audit per decision run"),
    "production_config": Hook("configure_production", ("engine/improve.py",), "persisting sinks + board in production"),
    "pre_launch": Hook("pre_launch", ("engine/improve.py",), "already-tested verdict"),
    "repro": Hook("repro_dict", ("engine/improve.py",), "ReproRecord on the registry record"),
    "registry_audit": Hook("registry_audit", ("engine/registry.py",), "provenance findings"),
    "redundancy": Hook("on_redundancy", ("engine/patterns.py",), "REDUNDANT_WITH edges"),
    "weight": Hook("effective_weight", ("engine/lessons.py",), "board-scaled weight"),
    "promotion_gate": Hook("promotion_allowed", ("engine/improve.py",), "composite promotion verdict"),
}
# Period-level seams are called by the period runner (S17b / the nightly loop), not by an old engine file, so they have no `sites` here;
# the integration tests prove the data flows. Listed for the report: on_period -> research_step and write_dashboard_inputs.
PERIOD_HOOKS: tuple[str, ...] = ("contradiction_period", "research_step", "dashboard")


def unwired_hooks(root: str | Path | None = None, hooks: Mapping[str, Hook] | None = None) -> list[str]:
    """Static check: every declared hook must be CALLED (`wiring.<function>(` or `_wiring().<function>(`) in each of its site files. Returns 'name @ file' for each
    missing call. An empty list means every declared seam is really connected; a hook that is declared but never called is exactly
    the silent integration gap this module exists to prevent."""
    base = Path(root) if root else Path(__file__).resolve().parents[2]
    missing = []
    for name, h in (hooks if hooks is not None else HOOKS).items():
        for site in h.sites:
            f = base / site
            text = f.read_text(encoding="utf-8") if f.exists() else ""
            if not re.search(r"wiring(?:\(\))?\." + re.escape(h.function) + r"\(", text):
                missing.append(f"{name} @ {site}")
    return missing


def hook_report() -> dict[str, Any]:
    """Which hooks fired, what they delivered, and what failed. A hook with calls == 0 in a full run is a wiring gap."""
    return {"calls": dict(HUB.calls), "delivered": dict(HUB.delivered), "errors": [dataclasses.asdict(e) for e in HUB.errors],
            "silent_hooks": sorted(h for h in HOOKS if HUB.calls[h] == 0), "hypotheses": len(HUB.hypotheses),
            "failures_classified": len(HUB.failure_ledger), "failure_unknown_rate": HUB.failure_ledger.unknown_rate(),
            "rejections": len(HUB.missed.rows), "graph_edges": len(HUB.graph.edges("2100-01-01")), "decisions": len(HUB.decisions),
            "root": str(HUB.root) if HUB.root is not None else None, "board": HUB.board is not None,
            "source_violations": len(HUB.source_violations), "learning_claims": len(HUB.knowledge_verdicts)}


def state_digest() -> str:
    """Deterministic fingerprint of everything the sinks hold (hypothesis ids, classification counts by cause, rejection reasons,
    graph size, ledger sizes). Two runs on the same inputs must give the same digest; a digest that moves between identical runs means
    a sink is reading a clock, an unseeded draw or dictionary order."""
    with HUB._lock:
        rej = sorted((str(r["period"]), str(r["reason"]), str(r["kind"])) for r in HUB.missed.rows)
        return stable_hash({"hypotheses": sorted(HUB.hypotheses), "failures": HUB.failure_ledger.cause_counts(),
                            "n_failures": len(HUB.failure_ledger), "rejections": rej,
                            "edges": sorted((e.src, e.dst, e.rel) for e in HUB.graph.edges("2100-01-01")),
                            "ledgers": sorted(len(led) for led in HUB._ledgers.values())})


def audit_hub(now: Any) -> list[str]:
    """Cross-check the sinks' own invariants; [] means clean. Every hypothesis validates and names no date; the failure ledger holds
    exactly the losses that were classified (once each); the redundancy graph passes its integrity audit at `now`; no facade ledger
    dropped a line it could not parse. This is the read-back that proves the sinks are consistent, not merely non-empty."""
    from . import archive as AR
    issues: list[str] = []
    for hid, h in sorted(HUB.hypotheses.items()):
        issues += [f"hypothesis {hid}: {e}" for e in h.validate()]
        text = f"{h.statement} {h.target}"
        if AR._ISO.search(text) or AR._has_year(text):
            issues.append(f"hypothesis {hid}: names a date or year")
    issues += [f"graph {i.code} @ {i.where}" for i in HUB.graph.audit(now) if i.severity == "error"]
    if len(HUB.failure_ledger) != len(HUB._classified):
        issues.append(f"failure ledger holds {len(HUB.failure_ledger)} classifications for {len(HUB._classified)} classified losses")
    for path, led in sorted(HUB._ledgers.items()):
        if led.unparseable:
            issues.append(f"ledger {path}: {led.unparseable} unparseable line(s)")
    return issues


def assert_hooks_healthy(required: Iterable[str] = ()) -> None:
    """Raise FirewallBreach if any sink recorded an error or a required hook never fired: the wiring may not fail silently."""
    if HUB.errors:
        e = HUB.errors[0]
        raise FirewallBreach(f"{len(HUB.errors)} sink error(s); first: {e.hook} {e.error_type}: {e.message}")
    silent = [h for h in required if HUB.calls[h] == 0]
    if silent:
        raise FirewallBreach(f"hooks never fired: {silent}")
