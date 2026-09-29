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
from .core import FirewallBreach, Subsystem, ValidationLabel, as_date, canonical_json, current_code_hash, stable_hash
from .firewalls import Finding, GateContext, LearningFirewallGate, registry_findings
from .identity_firewall import IdentityHarness, IdentityReport
from .knowledge_graph import KnowledgeGraph, NodeType
from .postmortem import make_hypothesis
from .reproducibility import ReproRecord, WorkerConfig, capture_worker, make_record
from .scorecard import LearningScorecard, gate_improvement_claim

T = TypeVar("T")
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
            self.strict = False
            self.evidence: dict[str, LearningEvidence] = {}
            self.decisions: list[PromotionVerdict] = []
            self._worker: WorkerConfig | None = None
            self._persisted: dict[str, set[str]] = {}

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

    def ledger_for(self, registry_path: str | Path) -> EM.ExperimentLedger:
        """The ONE facade ledger of a state directory: experiment_ledger.jsonl beside whichever legacy file (experiments.jsonl,
        the experiment-memory file) is being mirrored, so every old writer lands in the same place."""
        path = Path(registry_path).with_name("experiment_ledger.jsonl")
        key = str(path)
        with self._lock:
            if key not in self._ledgers:
                self._ledgers[key] = EM.ExperimentLedger(path)
            return self._ledgers[key]


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


# ================================================================================================================== missed winners
@sink("missed_week")
def on_missed_week(decided: Any, closed: Any, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence[Any], score: pd.Series | None = None,
                   era: str = "", thr: float = MW.WINNER, k: int = 10) -> int:
    """A closed adapter week -> learning MissedLearningLedger.add_week (WHY each winner was rejected). `decided` is the decision date,
    `closed` the date the outcome matured. Returns the number of rejections explained."""
    week = MW.week_from_base(decided, p0, fwd, list(picked), score=score, k=k, thr=thr, resolved_at=str(as_date(closed)), era=era)
    errs = week.validate()
    if errs:
        raise ValueError("; ".join(errs[:3]))
    rej = HUB.missed.add_week(week)
    HUB.delivered["rejections"] += len(rej)
    return len(rej)


# ================================================================================================================== experiments
def _question_of(rec: Mapping[str, Any]) -> str:
    return str(rec.get("question") or rec.get("desc") or rec.get("reason") or rec.get("event") or "experiment")


@sink("experiment")
def on_experiment(rec: Mapping[str, Any], registry_path: str | Path) -> str:
    """improve.log_experiment -> the ExperimentLedger facade (a legacy row is imported honestly: fields the old writer never captured
    are marked NOT RECORDED). Returns the ledger status of the row ('added' or 'skipped')."""
    cfg = rec.get("model_params") if isinstance(rec.get("model_params"), Mapping) else None
    row = dict(rec)
    if cfg is not None:
        row["config"] = dict(cfg)
    row["question"] = _question_of(rec)
    led = HUB.ledger_for(registry_path)
    out = EM.import_legacy(led, [row])
    HUB.delivered["ledger_rows"] += out["added"]
    return "added" if out["added"] else "skipped"


@sink("memory_entry")
def on_memory_entry(experiment_id: str, change: Mapping[str, Any], answers: Mapping[str, Any], now: Any, registry_path: str | Path) -> str:
    """registry.ExperimentMemory.record -> the same facade, so the two experiment stores agree (adapter: no old behaviour changes)."""
    row = {"experiment_id": f"{experiment_id}:memory", "config": dict(change), "question": str(answers.get("what_changed") or answers.get("why_changed") or "experiment"),
           "outcome": "adopt" if answers.get("adopted") else "reject", "reason": str(answers.get("if_rejected_why") or answers.get("why_changed") or ""),
           "t": str(now)}
    out = EM.import_legacy(HUB.ledger_for(registry_path), [row])
    return "added" if out["added"] else "skipped"


def pre_launch(question: str, config: Mapping[str, Any], now: Any, registry_path: str | Path, seed: int | None = 7,
               data_hash: str = "", code_hash: str = "") -> EM.DuplicateVerdict | None:
    """'Have we already tested this?' for a candidate about to launch. None when the hub is disabled or the ledger can not answer
    (never blocks on its own failure). `code_hash`/`data_hash` default to the running code and empty data; a changed hash on an
    otherwise identical design downgrades an exact repeat to RETEST_JUSTIFIED, which does not block."""
    if not HUB.enabled:
        return None
    HUB.calls["pre_launch"] += 1
    try:
        led = HUB.ledger_for(registry_path)
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
        r = make_record(rec.get("model_params") if isinstance(rec.get("model_params"), Mapping) else {},
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


# ================================================================================================================== production weight
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


def effective_weight(kid: str, legacy_weight: float) -> float:
    """What a production reader should use for knowledge `kid`. ADAPTER: with no board configured, or knowledge the board never
    registered (and strict off), this returns `legacy_weight` unchanged, so untested paths do not move. Registered knowledge is
    scaled by the board: a shadow or challenger has weight exactly 0 and can never influence a decision; a champion keeps its weight."""
    HUB.calls["weight"] += 1
    bw = board_weight(kid)
    if bw is None:
        return 0.0 if (HUB.strict and HUB.board is not None) else float(legacy_weight)
    return float(legacy_weight) * bw


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
    return Check("identity", rep.passed, "ok" if rep.passed else "collapsed under " + (", ".join(rep.collapsed) or "an attack (or was nondeterministic)"))


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


def promotion_allowed(challenger: Mapping[str, Any], now: Any) -> PromotionVerdict:
    """improve.test_and_promote hook. A challenger dict with claims_learning=True is a learning claim and needs registered evidence;
    any other challenger is judged by its live-shadow test alone."""
    cid = str(challenger.get("id", ""))
    return promotion_gate(cid or "challenger", now, bool(challenger.get("claims_learning")), HUB.evidence.get(cid))


# ================================================================================================================== health of the wiring itself
HOOKS: dict[str, tuple[str, str]] = {
    "post_mortem": ("engine/lessons.py post_mortem", "failure classifications + hypotheses"),
    "lessons": ("engine/lessons.py LessonBook.learn / learn_by_kind; engine/memory.py export_lessons", "failure hypotheses"),
    "missed_week": ("engine/missed_winners.py MissedLedger.observe", "why-rejected ledger"),
    "experiment": ("engine/improve.py log_experiment", "ExperimentLedger facade row"),
    "memory_entry": ("engine/registry.py ExperimentMemory.record", "ExperimentLedger facade row"),
    "pre_launch": ("engine/improve.py spawn_challengers", "already-tested verdict"),
    "repro": ("engine/improve.py log_experiment", "ReproRecord on the registry record"),
    "registry_audit": ("engine/registry.py Registry.audit", "provenance findings"),
    "redundancy": ("engine/patterns.py PatternMiner.fit", "REDUNDANT_WITH edges"),
    "weight": ("engine/lessons.py LessonBook.factor / advice", "board-scaled weight"),
    "promotion_gate": ("engine/improve.py test_and_promote", "composite promotion verdict"),
}


def hook_report() -> dict[str, Any]:
    """Which hooks fired, what they delivered, and what failed. A hook with calls == 0 in a full run is a wiring gap."""
    return {"calls": dict(HUB.calls), "delivered": dict(HUB.delivered), "errors": [dataclasses.asdict(e) for e in HUB.errors],
            "silent_hooks": sorted(h for h in HOOKS if HUB.calls[h] == 0), "hypotheses": len(HUB.hypotheses),
            "failures_classified": len(HUB.failure_ledger), "failure_unknown_rate": HUB.failure_ledger.unknown_rate(),
            "rejections": len(HUB.missed.rows), "graph_edges": len(HUB.graph.edges("2100-01-01")), "decisions": len(HUB.decisions)}


def assert_hooks_healthy(required: Iterable[str] = ()) -> None:
    """Raise FirewallBreach if any sink recorded an error or a required hook never fired: the wiring may not fail silently."""
    if HUB.errors:
        e = HUB.errors[0]
        raise FirewallBreach(f"{len(HUB.errors)} sink error(s); first: {e.hook} {e.error_type}: {e.message}")
    silent = [h for h in required if HUB.calls[h] == 0]
    if silent:
        raise FirewallBreach(f"hooks never fired: {silent}")
