"""Creator constraint loop (theory of constraints) - Nupen finds what limits its own improvement rate, works on it, re-measures.

Owner, 2 Oct 2026: "find all the limiting factors within its own system and then it improves it until something else is the limiting
factor" and "the limiting factor is how fast it can improve its ability to improve itself". So the HEADLINE is the META rate:

    improvement rate   adopted, measured improvements per hour (and per cycle-hour) in the current window
    machinery rate     those that changed the improvement machinery itself (creator/ code, not tests) + recursion adoptions
    meta rate          change of the improvement rate against the previous window (per day)

Every candidate limiting factor is MEASURED from Nupen's own records (no model call, read-only): kernel cycles (state/creator/cycles/*/
cycle.json, evaluation.json, footprint.json; the kernel_log.jsonl summary), swarm_log.jsonl, nupen_service.log, lessons.jsonl via
creator.curriculum, AUDIT.json, the ledger. Each Metric has value, unit, window, previous-window value, trend and a LOSS in [0, 1]:

    loss = the fraction of the window's improvement capacity this factor costs, by a documented formula (see each metric function),
           x META_WEIGHT[kind]: 1.0 for levers on the improvement MACHINERY (learning signal, student skill, teacher dependence,
           feedback-loop time, recursion), 0.7 for pure throughput waste, 0.5 where part of the "waste" is still information
           (a REJECTED cycle taught something; evaluation cost buys trust).
    score = loss x weight; constraints are ranked by score and the top one IS the constraint.
Metrics flagged `parent` are DRIVERS of another metric (model timeouts of the learning signal, ...): shown under it, never ranked.

ACT: the top constraint maps to a remedy through existing machinery (goal proposal with source 'constraint' under the same approval
gate; a recursion.design_change step for process levers; an efficiency work item). Snapshots go to state/creator/constraints.jsonl;
the next measure re-reads them and says whether the constraint MOVED; a change of the top logs 'constraint shifted from X to Y'.
A remedy rejected twice is not proposed again unless the loss grew by >= 25% since the last rejection (new evidence)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

CONSTRAINTS_FILE = "constraints.jsonl"
RUN_FILE = "constraints_run.json"
META_WEIGHT = {"meta": 1.0, "throughput": 0.7, "information": 0.5}
RANK_TOP = 3
REJECT_LIMIT = 2
NEW_EVIDENCE_GROWTH = 1.25
MACHINERY_PREFIXES = ("creator/",)
RAM_WORDS = ("ram tight", "ram", "memory")


@dataclasses.dataclass
class Metric:
    name: str
    value: float
    unit: str
    window: str
    prev: Optional[float]
    trend: str                                   # "worse" | "better" | "flat" | "new"
    loss: float
    kind: str                                    # key of META_WEIGHT
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    remedy: str = "work"                         # "goal" | "process" | "work"
    parent: str = ""

    @property
    def score(self) -> float:
        return round(self.loss * META_WEIGHT[self.kind], 4)

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "score": self.score}


# ------------------------------------------------------------------------------------------------ reading the records

def _naive(text: str) -> Optional[dt.datetime]:
    try:
        d = dt.datetime.fromisoformat(str(text))
    except (ValueError, TypeError):
        return None
    return d.astimezone().replace(tzinfo=None) if d.tzinfo else d


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                o = json.loads(ln)
            except ValueError:
                continue
            if isinstance(o, dict):
                out.append(o)
    except OSError:
        pass
    return out


def _json(path: Path) -> dict[str, Any]:
    try:
        o = json.loads(path.read_text(encoding="utf-8"))
        return o if isinstance(o, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclasses.dataclass
class Cycle:
    package: str
    outcome: str
    reason: str
    seconds: float
    at: dt.datetime
    by: str
    requirement: str
    eval_seconds: float = 0.0
    changed: tuple[str, ...] = ()
    stages: dict[str, float] = dataclasses.field(default_factory=dict)   # kernel per-stage wall seconds (CycleReport.details['stages'])


def _stage_seconds(cj: dict[str, Any]) -> dict[str, float]:
    raw = (cj.get("details") or {}).get("stages") or {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def read_cycles(state: Path) -> list[Cycle]:
    out: list[Cycle] = []
    root = Path(state) / "cycles"
    try:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return out
    for d in dirs:
        cj = _json(d / "cycle.json")
        if not cj:
            continue
        try:
            at = dt.datetime.fromtimestamp((d / "cycle.json").stat().st_mtime)
        except OSError:
            continue
        ev = _json(d / "evaluation.json")
        steps = (ev.get("build") or {}).get("steps") or []
        es = sum(float(s.get("seconds") or 0.0) for s in steps if isinstance(s, dict))
        worker = (cj.get("details") or {}).get("worker") or {}
        out.append(Cycle(str(cj.get("package", d.name)), str(cj.get("outcome", "")), str(cj.get("reason", "")),
                         float(cj.get("seconds") or 0.0), at, str(worker.get("by") or ""), str(cj.get("requirement", "")), es,
                         tuple(sorted((ev.get("changed") or {}).keys())), _stage_seconds(cj)))
    return out


def _in(at: Optional[dt.datetime], lo: dt.datetime, hi: dt.datetime) -> bool:
    return at is not None and lo < at <= hi


def _trend(cur: float, prev: Optional[float], higher_is_worse: bool = True, tol: float = 0.05) -> str:
    if prev is None:
        return "new"
    d = cur - prev
    if abs(d) <= tol * max(abs(prev), 1e-9) and abs(d) < 0.02:
        return "flat"
    return "worse" if (d > 0) == higher_is_worse else "better"


def _classify_reason(reason: str) -> str:
    low = reason.lower()
    if "timeoutexpired" in low or "timed out" in low:
        return "timeout"
    if any(w in low for w in RAM_WORDS) and "pulled back" in low:
        return "ram_pullback"
    if "pulled back" in low or "cancel" in low:
        return "pulled_back_other"
    if "changed nothing" in low or "no edit" in low or "no change" in low:
        return "no_change"
    return "other"


def _reject_class(reason: str) -> str:
    low = reason.lower()
    for key in ("regression", "no_improvement", "no improvement", "inconclusive", "weaken", "build", "audit"):
        if key in low:
            return key.replace(" ", "_")
    return "other"


def _hours(lo: dt.datetime, hi: dt.datetime) -> float:
    return max((hi - lo).total_seconds() / 3600.0, 1e-9)


# ------------------------------------------------------------------------------------------------ the headline

def _is_machinery(c: Cycle) -> bool:
    return any(p.startswith(MACHINERY_PREFIXES) for p in c.changed)


def _recursion_adoptions(state: Path, lo: dt.datetime, hi: dt.datetime) -> int:
    n = 0
    for r in _jsonl(Path(state) / "ledger.jsonl"):
        if r.get("rtype") != "Decision":
            continue
        d = r.get("data") or {}
        if str(d.get("reason", "")).startswith("RECURSION_STEP ") and str(d.get("verdict", "")).lower().endswith("adopt"):
            at = _naive(str((r.get("provenance") or {}).get("at", "")) or str(r.get("at", "")))
            if _in(at, lo, hi):
                n += 1
    return n


def headline(state: Path, now: dt.datetime, window_h: float) -> dict[str, Any]:
    """improvement_rate = ADOPTED cycles / window hours; per_cycle_hour = ADOPTED / (sum of all cycle wall-hours);
    machinery_rate = adopted cycles whose evaluated change touches creator/ + recursion adoptions, per hour;
    meta_rate_per_day = (improvement_rate - previous window's) x 24 / window_h (change of the rate per day)."""
    cyc = read_cycles(state)
    w = dt.timedelta(hours=window_h)
    out: dict[str, Any] = {"window_h": window_h}
    rates = []
    for lo, hi in ((now - w, now), (now - 2 * w, now - w)):
        sel = [c for c in cyc if _in(c.at, lo, hi)]
        ad = [c for c in sel if c.outcome == "ADOPTED"]
        ch = sum(c.seconds for c in sel) / 3600.0
        mach = sum(1 for c in ad if _is_machinery(c)) + _recursion_adoptions(state, lo, hi)
        rates.append({"cycles": len(sel), "adopted": len(ad), "per_hour": len(ad) / window_h,
                      "per_cycle_hour": (len(ad) / ch) if ch else 0.0, "machinery_per_hour": mach / window_h})
    out["current"], out["previous"] = rates
    out["meta_rate_per_day"] = round((rates[0]["per_hour"] - rates[1]["per_hour"]) * 24.0 / window_h, 4)
    return out


# ------------------------------------------------------------------------------------------------ the metrics

def _cycle_window(cyc: list[Cycle], now: dt.datetime, window_h: float) -> tuple[list[Cycle], list[Cycle]]:
    w = dt.timedelta(hours=window_h)
    return [c for c in cyc if _in(c.at, now - w, now)], [c for c in cyc if _in(c.at, now - 2 * w, now - w)]


def _share(sel: list[Cycle], pred: Callable[[Cycle], bool]) -> float:
    tot = sum(c.seconds for c in sel)
    return sum(c.seconds for c in sel if pred(c)) / tot if tot else 0.0


def waste_metrics(state: Path, now: dt.datetime, window_h: float) -> list[Metric]:
    """Share of the window's cycle wall-seconds spent in cycles that produced nothing, split by cause. loss = that share
    (timeouts/errors, RAM pull-backs and other cancellations: weight throughput; rejected cycles: weight information)."""
    cur, prev = _cycle_window(read_cycles(state), now, window_h)
    label = f"last {window_h:g}h"
    specs: list[tuple[str, str, Callable[[Cycle], bool], str, str]] = [
        ("infra_timeouts", "throughput", lambda c: c.outcome == "ERROR" and _classify_reason(c.reason) == "timeout", "work", "ERROR cycles killed by a command timeout (git/sandbox under load)"),
        ("errors_other", "throughput", lambda c: c.outcome == "ERROR" and _classify_reason(c.reason) != "timeout", "work", "ERROR cycles not caused by a timeout"),
        ("ram_pullbacks", "throughput", lambda c: c.outcome == "CANCELLED" and _classify_reason(c.reason) == "ram_pullback", "work", "cycles pulled back because RAM was tight"),
        ("cancelled_other", "throughput", lambda c: c.outcome == "CANCELLED" and _classify_reason(c.reason) != "ram_pullback", "work", "cycles cancelled for another reason (pause, owner)"),
        ("rejected_cycles", "information", lambda c: c.outcome == "REJECTED", "process", "REJECTED cycles (a rejection still teaches; weight 0.5)"),
    ]
    out = []
    for name, kind, pred, remedy, what in specs:
        s, p = _share(cur, pred), (_share(prev, pred) if prev else None)
        det: dict[str, Any] = {"what": what, "cycles": sum(1 for c in cur if pred(c)), "of": len(cur),
                               "wasted_cycle_hours": round(sum(c.seconds for c in cur if pred(c)) / 3600.0, 3)}
        if name == "rejected_cycles":
            cls: dict[str, int] = {}
            for c in cur:
                if pred(c):
                    cls[_reject_class(c.reason)] = cls.get(_reject_class(c.reason), 0) + 1
            det["by_class"] = cls
            det["no_change"] = sum(1 for c in cur if _classify_reason(c.reason) == "no_change")
        out.append(Metric(name, round(s, 4), "share of cycle-seconds", label, None if p is None else round(p, 4), _trend(s, p), round(s, 4), kind, det, remedy))
    return out


def planning_metrics(state: Path, now: dt.datetime, window_h: float, grace_s: float = 1800.0) -> list[Metric]:
    """Work the planner spends and never gets an answer for, read from the ledger.
    wasted_work  = share of packages planned in the window that never reached a verdict (interrupted/cancelled, or still unjudged
                   once older than `grace_s`; a package younger than that may simply be running).
    repeat_rate  = share of the window's efficiency packages whose target was already planned by an efficiency package in the
                   24 h before it (the same file planned again and again). loss = value for both."""
    from creator import model as M
    from creator import planner as P
    from creator.ledger import Ledger
    path = Path(state) / "ledger.jsonl"
    lo, label = now - dt.timedelta(hours=window_h), f"last {window_h:g}h"
    planned: list[tuple[dt.datetime, str, bool]] = []                   # (planned at, efficiency target or "", reached a verdict)
    if path.is_file():
        led = Ledger(path)
        open_states = (M.Status.NOT_STARTED, M.Status.IN_PROGRESS, M.Status.IMPLEMENTED)
        for e in led.of_type("WorkPackage"):
            at = dt.datetime.fromtimestamp(P._at(e))
            eff = getattr(e.record, "objective", "").startswith(P.EFFICIENCY_OBJECTIVES) and getattr(e.record, "outputs")
            judged = led.view.status.get(e.id) not in open_states and not P.was_interrupted(led, e.id)
            planned.append((at, getattr(e.record, "outputs")[0] if eff else "", judged))
    cur = [x for x in planned if lo <= x[0] <= now and (x[2] or (now - x[0]).total_seconds() >= grace_s)]
    wasted = [x for x in cur if not x[2]]
    effs = [x for x in cur if x[1]]
    rep = [x for x in effs if any(y[1] == x[1] and x[0] - dt.timedelta(hours=24) <= y[0] < x[0] for y in planned)]
    w, r = len(wasted) / len(cur) if cur else 0.0, len(rep) / len(effs) if effs else 0.0
    return [Metric("wasted_work", round(w, 4), "share of planned packages never judged", label, None, "new", round(w, 4), "throughput",
                   {"planned": len(cur), "never_judged": len(wasted), "what": "packages interrupted, cancelled or never judged"}, "goal"),
            Metric("repeat_rate", round(r, 4), "share of efficiency packages on an already-planned target", label, None, "new", round(r, 4),
                   "throughput", {"efficiency_packages": len(effs), "repeats": len(rep),
                                  "repeated_targets": sorted({x[1] for x in rep})[:8]}, "goal")]


def cycle_time_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Feedback-loop time: median wall-minutes of a completed (ADOPTED/REJECTED) cycle. loss = 1 - p25/median, the slack between a
    typical cycle and the fast quartile (0 when every cycle is as fast as the fast ones). Stage split: evaluation build steps
    (evaluation.json) vs everything else (sandbox open + worker + tests + merge); `dominant_stage` names the larger."""
    cyc = read_cycles(state)
    cur, prev = _cycle_window(cyc, now, window_h)

    def med(sel: list[Cycle]) -> Optional[float]:
        v = [c.seconds for c in sel if c.outcome in ("ADOPTED", "REJECTED") and c.seconds > 0]
        return statistics.median(v) if v else None

    done = sorted(c.seconds for c in cur if c.outcome in ("ADOPTED", "REJECTED") and c.seconds > 0)
    m, pm = med(cur), med(prev)
    if not done or m is None:
        return Metric("cycle_time", 0.0, "median minutes", f"last {window_h:g}h", None if pm is None else round(pm / 60, 2), "new", 0.0, "meta", {"note": "no completed cycles"}, "process")
    p25 = done[max(0, len(done) // 4)]
    ev = sum(c.eval_seconds for c in cur if c.outcome in ("ADOPTED", "REJECTED"))
    tot = sum(done)
    stages = {"evaluation_build": round(ev, 1), "worker_sandbox_tests_merge": round(max(tot - ev, 0.0), 1)}
    fine: dict[str, float] = {}
    for c in cur:
        if c.outcome in ("ADOPTED", "REJECTED"):
            for k, v in c.stages.items():
                fine[k] = fine.get(k, 0.0) + v
    loss = max(0.0, 1.0 - p25 / m)
    return Metric("cycle_time", round(m / 60.0, 2), "median minutes", f"last {window_h:g}h", None if pm is None else round(pm / 60.0, 2),
                  _trend(m, pm), round(loss, 4), "meta",
                  {"p25_minutes": round(p25 / 60.0, 2), "stage_seconds": stages, "dominant_stage": max(stages, key=lambda k: stages[k]), "n": len(done),
                   **({"stage_breakdown_seconds": {k: round(v, 1) for k, v in sorted(fine.items(), key=lambda kv: -kv[1])},
                       "dominant_stage": max(fine, key=lambda k: fine[k])} if fine else {})}, "process")


def eval_cost_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Evaluation cost per package: build-step seconds from evaluation.json / package. loss = evaluation seconds / cycle seconds."""
    cur, prev = _cycle_window(read_cycles(state), now, window_h)
    ev = [c for c in cur if c.eval_seconds > 0]
    per = statistics.mean(c.eval_seconds for c in ev) if ev else 0.0
    pev = [c for c in prev if c.eval_seconds > 0]
    pper = statistics.mean(c.eval_seconds for c in pev) if pev else None
    tot = sum(c.seconds for c in cur)
    loss = (sum(c.eval_seconds for c in cur) / tot) if tot else 0.0
    fp = len(list((Path(state) / "cycles").glob("*/footprint.json")))
    return Metric("evaluation_cost", round(per, 2), "build-step seconds per package", f"last {window_h:g}h", None if pper is None else round(pper, 2), _trend(per, pper), round(loss, 4), "information", {"packages_with_footprint": fp}, "work")


def supply_metric(state: Path, now: dt.datetime, window_h: float, min_packages: int = 2) -> Metric:
    """Work supply: share of swarm rounds (swarm_log.jsonl) that planned fewer than `min_packages` packages. loss = that share."""
    rows = _jsonl(Path(state) / "swarm_log.jsonl")
    w = dt.timedelta(hours=window_h)

    def sh(lo: dt.datetime, hi: dt.datetime) -> tuple[Optional[float], list[dict[str, Any]]]:
        sel = [r for r in rows if _in(_naive(str(r.get("at", ""))), lo, hi)]
        return ((sum(1 for r in sel if int(r.get("packages", 0)) < min_packages) / len(sel)) if sel else None), sel
    cur, sel = sh(now - w, now)
    prev, _ = sh(now - 2 * w, now - w)
    c = cur or 0.0
    return Metric("work_supply", round(c, 4), f"share of rounds with < {min_packages} packages", f"last {window_h:g}h", prev, _trend(c, prev), round(c, 4), "throughput",
                  {"rounds": len(sel), "peak_parallel_max": max((int(r.get("peak_parallel", 0)) for r in sel), default=0),
                   "pulled_back": sum(int(r.get("pulled_back", 0)) for r in sel)}, "goal")


def availability_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Service availability from nupen_service.log ('supervisor down' ... 'supervisor up'). loss = downtime hours / window hours."""
    ev: list[tuple[dt.datetime, str]] = []
    try:
        for ln in (Path(state) / "nupen_service.log").read_text(encoding="utf-8", errors="replace").splitlines():
            at = _naive(ln[:19])
            if at and "supervisor down" in ln:
                ev.append((at, "down"))
            elif at and "supervisor up" in ln:
                ev.append((at, "up"))
    except OSError:
        pass

    def down(lo: dt.datetime, hi: dt.datetime) -> float:
        tot, start = 0.0, None
        for at, k in sorted(ev):
            if k == "down":
                start = at
            elif start is not None:
                a, b = max(start, lo), min(at, hi)
                tot += max((b - a).total_seconds(), 0.0)
                start = None
        if start is not None:
            tot += max((hi - max(start, lo)).total_seconds(), 0.0)
        return tot / 3600.0
    w = dt.timedelta(hours=window_h)
    c, p = down(now - w, now) / window_h, down(now - 2 * w, now - w) / window_h
    return Metric("service_downtime", round(c, 4), "share of window", f"last {window_h:g}h", round(p, 4), _trend(c, p), round(min(c, 1.0), 4), "throughput", {"downtime_hours": round(c * window_h, 3)}, "work")


# ---- lessons: learning signal, student skill, model dependence, teacher dependence

def _lessons(state: Path, now: dt.datetime, window_h: float) -> tuple[list[Any], list[Any]]:
    from creator import curriculum as CUR
    allv = CUR.LessonLog(Path(state) / "lessons.jsonl").lessons()
    w = dt.timedelta(hours=window_h)
    return ([x for x in allv if _in(_naive(x.at), now - w, now)], [x for x in allv if _in(_naive(x.at), now - 2 * w, now - w)])


def _student(les: list[Any]) -> list[Any]:
    from creator import curriculum as CUR
    return [x for x in les if x.solver not in CUR.TEACHER]


def _measured(x: Any) -> bool:
    """A real kernel verdict: the cycle's outcome was ADOPTED or REJECTED ("not claimed done", "cancelled" and a still-pending lesson are not)."""
    return str(x.verdict).upper().startswith(("ADOPTED", "REJECTED"))


def _unmeasured_class(x: Any) -> str:
    v = str(x.verdict).lower()
    if "timeout" in v or "model call failed" in v:
        return "model_timeouts"
    if "no usable" in v or "held no usable" in v:
        return "unusable_model_reply"
    if v.startswith(("cancelled", "error", "interrupted")):
        return "cancelled_unrecorded"
    if v == "":
        return "pending_no_verdict"
    if "no edit applied" in v or "no learned template" in v or "no unmeasured teacher" in v:
        return "no_candidate_made"
    return "other_unmeasured"


def practice_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Offline practice measurements (state/creator/practice_rows.jsonl, scripts/practice.py) counted SEPARATELY from real kernel verdicts:
    value = practice rows per hour in the window, detail carries the chooser's held-out accuracy from practice_log.jsonl (before/after
    the latest retrain). Informational (loss 0): the headline learning_signal stays the share of REAL attempts with a kernel verdict."""
    w = dt.timedelta(hours=window_h)
    rows = [r for r in _jsonl(Path(state) / "practice_rows.jsonl") if r.get("source") == "prescreen"]
    cur = sum(1 for r in rows if _in(_naive(str(r.get("at", ""))), now - w, now))
    prev = sum(1 for r in rows if _in(_naive(str(r.get("at", ""))), now - 2 * w, now - w))
    logs = _jsonl(Path(state) / "practice_log.jsonl")
    last = logs[-1] if logs else {}
    detail = {"practice_rows_total": len(rows), "practice_rows_in_window": cur, "real_verdicts_separate": True,
              "chooser_holdout_before": (last.get("before") or {}).get("chooser_top1"), "chooser_holdout_after": (last.get("after") or {}).get("chooser_top1"),
              "random_baseline": (last.get("after") or {}).get("random_top1"), "holdout_decisions": (last.get("after") or {}).get("decisions"),
              "retrains": len(logs)}
    return Metric("practice_signal", round(cur / window_h, 2), "practice measurements per hour (not real verdicts)", f"last {window_h:g}h",
                  round(prev / window_h, 2), _trend(cur / window_h, prev / window_h, higher_is_worse=False), 0.0, "meta", detail, "goal", parent="learning_signal")


def learning_metrics(state: Path, now: dt.datetime, window_h: float) -> list[Metric]:
    """Student attempts (non-teacher lessons) and what became of them. The three shares partition the attempts:
        learning_signal   loss = unmeasured / attempts   (no real kernel verdict: timed out, unusable reply, cancelled, nothing made)
        student_skill     loss = (measured - adopted) / attempts   (a real verdict that was not an adoption)
        (adopted / attempts is the part that works)
    Drivers of the signal loss (parent=learning_signal): model_timeouts, unusable_model_reply, cancelled_unrecorded, no_candidate_made,
    each loss = its attempts / attempts. measured_per_hour is the learning-signal throughput."""
    cur, prev = _lessons(state, now, window_h)
    sc, sp = _student(cur), _student(prev)
    label = f"last {window_h:g}h"

    def parts(s: list[Any]) -> tuple[float, float, float]:
        n = len(s)
        if not n:
            return 0.0, 0.0, 0.0
        m = sum(1 for x in s if _measured(x))
        a = sum(1 for x in s if _measured(x) and x.adopted)
        return (n - m) / n, (m - a) / n, a / n
    un, sk, ad = parts(sc)
    pun, psk, _pad = parts(sp) if sp else (None, None, None)
    cls: dict[str, int] = {}
    for x in sc:
        if not _measured(x):
            cls[_unmeasured_class(x)] = cls.get(_unmeasured_class(x), 0) + 1
    n = len(sc)
    where: dict[str, dict[str, int]] = {}                 # per student|kind: where the signal is lost
    for x in sc:
        w = where.setdefault(f"{x.solver}|{x.task_kind}", {"attempts": 0, "with_verdict": 0, "adopted": 0})
        w["attempts"] += 1
        w["with_verdict"] += 1 if _measured(x) else 0
        w["adopted"] += 1 if (_measured(x) and x.adopted) else 0
    out = [Metric("learning_signal", round(un, 4), "share of student attempts never measured", label, None if pun is None else round(pun, 4), _trend(un, pun), round(un, 4), "meta",
                  {"attempts": n, "measured": sum(1 for x in sc if _measured(x)), "measured_per_hour": round(sum(1 for x in sc if _measured(x)) / window_h, 3),
                   "unmeasured_by_class": cls, "per_student_kind": dict(sorted(where.items()))}, "goal")]
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        out.append(Metric(k, round(v / n, 4), "share of student attempts", label, None, "new", round(v / n, 4), "meta", {"attempts": v}, "goal", parent="learning_signal"))
    out.append(practice_metric(state, now, window_h))
    from creator import curriculum as CUR
    cells = CUR.student_scores(sc)["cells"]
    kinds: dict[str, list[int]] = {}
    for c in cells.values():
        t = kinds.setdefault(c["task_kind"], [0, 0])
        t[0] += c["attempts"]
        t[1] += c["adopted"]
    out.append(Metric("student_skill", round(ad, 4), "student adopted share of attempts", label, None, "new", round(sk, 4), "meta",
                      {"measured_not_adopted_share": round(sk, 4), "per_kind": {k: {"attempts": a, "adopted": d} for k, (a, d) in sorted(kinds.items())},
                       "kinds_every_student_fails": sorted(k for k, (a, d) in kinds.items() if a >= 3 and d == 0)}, "goal"))
    from creator import reasoning as RE
    cal = RE.calibration(sc)
    claimed = sum(c["claimed"] for c in cal.values())
    if claimed:                                                          # a student that cannot predict its own change is a constraint
        well = sum(c["well_calibrated"] for c in cal.values())
        out.append(Metric("calibration", round(well / claimed, 4), "student claimed changes whose predicted effect matched the measured one", label, None, "new",
                          round(1.0 - well / claimed, 4), "information", {"per_student": cal, "claimed": claimed, "well_calibrated": well,
                                                                          "formula": "loss = 1 - well calibrated / claimed (no prediction counts as a miss)"}, "goal"))
    # teacher dependence: improvements only the teacher can make
    sc_all = CUR.student_scores(cur)
    ts = sc_all.get("teacher_share")
    handoffs = [x for x in cur if x.solver in CUR.TEACHER]
    waits = [c.seconds for c in read_cycles(state) if c.by in ("claude-session",) and _in(c.at, now - dt.timedelta(hours=window_h), now)]
    student_rate = ad
    loss = (ts if ts is not None else 0.0) * (1.0 - student_rate)
    pts = CUR.student_scores(prev).get("teacher_share") if prev else None
    out.append(Metric("teacher_dependence", round(ts, 4) if ts is not None else 0.0, "teacher share of adopted changes", label, pts, _trend(ts or 0.0, pts), round(loss, 4), "meta",
                      {"formula": "teacher_share x (1 - student adopted share of attempts)", "handoffs": len(handoffs),
                       "teacher_cycles": len(waits), "mean_teacher_cycle_hours": round(statistics.mean(waits) / 3600.0, 3) if waits else 0.0}, "goal"))
    return out


def recursion_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Recursion adoptions (changes of the development PROCESS adopted via creator.recursion) in the window. loss = 0.25 when none
    (the process lever is idle), else 0. A documented estimate: one idle lever of four process parameters."""
    w = dt.timedelta(hours=window_h)
    c, p = _recursion_adoptions(state, now - w, now), _recursion_adoptions(state, now - 2 * w, now - w)
    return Metric("recursion_idle", float(c), "process changes adopted", f"last {window_h:g}h", float(p), _trend(float(c), float(p), higher_is_worse=False), 0.25 if c == 0 else 0.0, "meta", {}, "process")


def audit_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Audit/validation gaps: AUDIT.json counts. loss = min(1, 0.5 CRITICAL + 0.2 HIGH + 0.05 MEDIUM) (a red audit stops the kernel)."""
    a = (_json(Path(state) / "AUDIT.json").get("audit") or {})
    cnt = a.get("counts") or {}
    loss = min(1.0, 0.5 * cnt.get("CRITICAL", 0) + 0.2 * cnt.get("HIGH", 0) + 0.05 * cnt.get("MEDIUM", 0))
    return Metric("audit_gaps", float(sum(cnt.values())), "open findings", "current", None, "new", round(loss, 4), "throughput", {"counts": cnt, "errors": a.get("errors", {})}, "work")


def fundamentals_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """Advisory engineering-fundamentals violations (creator.fundamentals) per candidate in the window. loss = min(0.5, 0.05 x the
    mean violations per candidate): a documented estimate; it ranks as 'information' (weight 0.5) so it can never outrank a
    blocking constraint until the owner makes the checks blocking."""
    from creator import fundamentals as FU
    w = dt.timedelta(hours=window_h)

    def rate(lo: dt.datetime, hi: dt.datetime) -> tuple[float, int, dict[str, int]]:
        n, tot, by = 0, 0, dict[str, int]()
        for name, rep in FU.read_reports(state):
            try:
                at = dt.datetime.fromtimestamp((Path(state) / "cycles" / name / "fundamentals.json").stat().st_mtime)
            except OSError:
                continue
            if _in(at, lo, hi):
                n += 1
                tot += int(rep.get("total", 0))
                for k, v in (rep.get("counts") or {}).items():
                    by[k] = by.get(k, 0) + int(v)
        return (tot / n if n else 0.0), n, by
    cur, n, by = rate(now - w, now)
    prev, pn, _ = rate(now - 2 * w, now - w)
    return Metric("fundamentals_violations", round(cur, 4), "violations per candidate", f"last {window_h:g}h", round(prev, 4) if pn else None,
                  _trend(cur, prev if pn else None), round(min(0.5, 0.05 * cur), 4), "information", {"candidates": n, "by_principle": by}, "work")


def resource_balance_metric(state: Path, now: dt.datetime, window_h: float) -> Metric:
    """CPU and RAM used together (owner, 2 Oct 2026: when CPU is out look for RAM work and the reverse). Over the last hour of
    resource samples (creator.resources): a = share with CPU saturated (>= 95%) while RAM is idle (< 60% used), b = share with RAM
    > 85% used while CPU < 60%. loss = min(1, a + b) x 0.6: a documented estimate - an idle half of the machine is not the whole
    improvement capacity, but it is the cheapest to reclaim. No samples -> 0 (nothing to judge)."""
    from creator import resources as RS
    ts = now.timestamp()
    cur, prev = RS.balance_shares(state, 3600.0, ts), RS.balance_shares(state, 3600.0, ts - 3600.0)
    s = cur["cpu_full_ram_idle"] + cur["ram_full_cpu_idle"]
    p = prev["cpu_full_ram_idle"] + prev["ram_full_cpu_idle"] if prev["samples"] else None
    return Metric("resource_balance", round(s, 4), "share of samples with one resource full and the other idle", "last 1h", None if p is None else round(p, 4),
                  _trend(s, p), round(min(1.0, s) * 0.6, 4) if cur["samples"] else 0.0, "throughput", {**cur, "previous": prev}, "goal")


METRICS: tuple[Callable[[Path, dt.datetime, float], Any], ...] = (
    waste_metrics, planning_metrics, cycle_time_metric, eval_cost_metric, supply_metric, availability_metric, learning_metrics, recursion_metric, audit_metric, fundamentals_metric,
    resource_balance_metric)


def measure_all(state: Path, now: Optional[dt.datetime] = None, window_h: float = 24.0) -> dict[str, Any]:
    """Every metric (a failing metric becomes an error note, never an exception), the ranking and the headline."""
    state, now = Path(state), now or dt.datetime.now()
    metrics: list[Metric] = []
    errors: dict[str, str] = {}
    for fn in METRICS:
        try:
            r = fn(state, now, window_h)
            metrics.extend(r if isinstance(r, list) else [r])
        except Exception as e:                                          # noqa: BLE001 - a broken record never stops the loop
            errors[fn.__name__] = f"{type(e).__name__}: {e}"
    try:
        head = headline(state, now, window_h)
    except Exception as e:                                              # noqa: BLE001
        head, errors["headline"] = {}, f"{type(e).__name__}: {e}"
    ranked = sorted((m for m in metrics if not m.parent), key=lambda m: (-m.score, m.name))
    return {"at": now.isoformat(timespec="seconds"), "window_h": window_h, "headline": head, "ranked": [m.to_dict() for m in ranked],
            "drivers": [m.to_dict() for m in metrics if m.parent], "errors": errors,
            "top": ranked[0].name if ranked and ranked[0].score > 0 else ""}


# ------------------------------------------------------------------------------------------------ memory, shift, act

def history(state: Path) -> list[dict[str, Any]]:
    return _jsonl(Path(state) / CONSTRAINTS_FILE)


def _append(state: Path, rec: dict[str, Any]) -> None:
    Path(state).mkdir(parents=True, exist_ok=True)
    with open(Path(state) / CONSTRAINTS_FILE, "ab") as f:
        f.write((json.dumps(rec, sort_keys=True, default=str) + "\n").encode("utf-8"))


def _remedy_text(m: dict[str, Any]) -> str:
    return {"learning_signal": "make every student attempt reach a measured kernel verdict (fewer model timeouts/unusable replies, record cancelled attempts)",
            "wasted_work": "stop planning work that is never judged: finish or release in-flight packages, rotate targets, plan only what fits the slots",
            "repeat_rate": "plan efficiency work on targets not planned in the last 24 h (rotate; skip targets with an unjudged package)",
            "student_skill": "raise the student adopted rate on task kinds that every student fails",
            "teacher_dependence": "move task kinds from the teacher to students (handoff-free adoption)",
            "work_supply": "plan more packages per round (more open gaps, wider scheduling)",
            "cycle_time": "shorten the evaluation feedback loop (the dominant stage)",
            "resource_balance": "when CPU is saturated and RAM idle run RAM-specific work (model-cache warming, hot caches, data preload); when RAM is full and CPU idle start CPU-light work (creator.resources)",
            "recursion_idle": "run a recursion step on the development process",
            "fundamentals_violations": "feed principles_for(kind) into package text and filter candidates on the worst-violated principle"}.get(m["name"], f"reduce {m['name']}")


def rejected_count(state: Path, name: str) -> tuple[int, float]:
    """(times a remedy for constraint `name` was rejected, the loss when it was last rejected)."""
    n, last = 0, 0.0
    try:
        from creator import goals as GO
        for p in GO.listing(Path(state)):
            if p["key"].startswith(f"constraint:{name}:") and p["status"] == "REJECTED":
                n += 1
                last = max(last, float(p.get("loss_at_proposal", 0.0)))
    except Exception:                                                   # noqa: BLE001
        pass
    for r in history(state):
        if r.get("event") == "remedy_rejected" and r.get("name") == name:
            n += 1
            last = max(last, float(r.get("loss", 0.0)))
    return n, last


def reject_remedy(state: Path, name: str, loss: float, reason: str) -> None:
    """Record that a process/work remedy for `name` was rejected (goal proposals are rejected through creator.goals)."""
    _append(Path(state), {"event": "remedy_rejected", "name": name, "loss": loss, "reason": reason, "at": dt.datetime.now().isoformat(timespec="seconds")})


def suppressed(state: Path, m: dict[str, Any]) -> str:
    n, last = rejected_count(state, m["name"])
    if n >= REJECT_LIMIT and m["loss"] < last * NEW_EVIDENCE_GROWTH:
        return f"remedy rejected {n}x and the loss ({m['loss']}) has not grown 25% over {last}: no new evidence"
    return ""


def act(state: Path, report: dict[str, Any], now: Optional[str] = None) -> dict[str, Any]:
    """Map the highest-ranked NOT-suppressed constraint to work: a goal proposal (source 'constraint', pending approval), a
    recursion.design_change step proposal for process levers, or a recorded work item. Returns what was done."""
    state = Path(state)
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    skipped = []
    for m in report["ranked"]:
        if m["score"] <= 0:
            break
        why = suppressed(state, m)
        if why:
            skipped.append({"name": m["name"], "why": why})
            continue
        res: dict[str, Any] = {"constraint": m["name"], "remedy": m["remedy"], "text": _remedy_text(m), "skipped": skipped}
        if m["remedy"] == "goal":
            from creator import goals as GO
            key_subject = f"{m['name']}:{int(m['loss'] * 100)}"
            p = GO._make("constraint", m["name"], f"remove limiting factor {m['name']}: {_remedy_text(m)}",
                         f"constraint loop: {m['name']} costs {m['loss']:.0%} of improvement capacity (score {m['score']})", [m["name"]], 10, m["loss"], now)
            known = {x["key"] for x in GO.listing(state)}
            key = f"constraint:{key_subject}"
            d = {**p.to_dict(), "key": key, "id": GO._pid(key), "loss_at_proposal": m["loss"]}
            pend = [x for x in GO.listing(state) if x["key"].startswith(f"constraint:{m['name']}:") and x["status"] in ("PENDING", "APPROVED")]
            if key in known or pend:
                res["proposal"] = pend[0]["id"] if pend else GO._pid(key)
                res["note"] = "a proposal for this constraint is already open"
            else:
                GO._append(state, d)
                res["proposal"] = d["id"]
        elif m["remedy"] == "process":
            try:
                from creator import recursion as R
                ch = R.design_change(R.Weakness("rejection", m["name"], 1, float(m["loss"]), ()), R.ProcessConfig(), set())
                res["process_change"] = dataclasses.asdict(ch) if ch else None
            except Exception as e:                                      # noqa: BLE001
                res["process_change"] = f"unavailable: {e}"
        _append(state, {"event": "act", "at": now, **res, "loss": m["loss"]})
        return res
    return {"constraint": "", "skipped": skipped}


def snapshot(state: Path, report: dict[str, Any]) -> list[str]:
    """Append the snapshot; returns messages: 'constraint shifted from X to Y' and whether the previous top constraint MOVED."""
    state = Path(state)
    msgs: list[str] = []
    snaps = [r for r in history(state) if r.get("event") == "snapshot"]
    top = report["top"]
    if snaps:
        prev = snaps[-1]
        pt = prev.get("top", "")
        if pt and top and pt != top:
            msgs.append(f"constraint shifted from {pt} to {top}")
        if pt:
            before = next((x["loss"] for x in prev["ranked"] if x["name"] == pt), None)
            after = next((x["loss"] for x in report["ranked"] if x["name"] == pt), None)
            if before is not None and after is not None and before > 0:
                moved = "moved" if after < before * 0.9 else "did not move"
                msgs.append(f"previous constraint {pt}: loss {before} -> {after} ({moved})")
    _append(state, {"event": "snapshot", "at": report["at"], "window_h": report["window_h"], "top": top, "headline": report["headline"],
                    "ranked": [{k: x[k] for k in ("name", "loss", "score", "value", "unit", "remedy")} for x in report["ranked"]],
                    "messages": msgs})
    for s in msgs:
        _append(state, {"event": "note", "at": report["at"], "text": s})
    return msgs


def run(state: Path, now: Optional[dt.datetime] = None, window_h: float = 24.0, do_act: bool = True) -> dict[str, Any]:
    state = Path(state)
    rep = measure_all(state, now, window_h)
    rep["messages"] = snapshot(state, rep)
    rep["act"] = act(state, rep) if do_act else {}
    if do_act:
        try:
            rep["engine"] = engine_cycle(state, now.timestamp() if now else None, claim=False)   # record only: the engine loop (engineloop.engine_step) claims the WIP slot
        except Exception as e:                                                          # noqa: BLE001
            rep["engine"] = {"error": f"{type(e).__name__}: {e}"}
    return rep


def maybe_run(state: Path, now: Optional[dt.datetime] = None, every_s: float = 3600.0) -> Optional[dict[str, Any]]:
    """At most once per `every_s` (the swarm calls it at every round start; cheap; never raises)."""
    now = now or dt.datetime.now()
    run_f = Path(state) / RUN_FILE
    try:
        last = dt.datetime.fromisoformat(json.loads(run_f.read_text(encoding="utf-8"))["at"])
        if (now - last).total_seconds() < every_s:
            return None
    except (OSError, ValueError, KeyError):
        pass
    try:
        rep = run(state, now)
        Path(state).mkdir(parents=True, exist_ok=True)
        run_f.write_text(json.dumps({"at": now.isoformat(timespec="seconds"), "top": rep["top"]}), encoding="utf-8")
        return rep
    except Exception:                                                   # noqa: BLE001 - the constraint loop never costs a round
        return None


def top_constraints(state: Path, n: int = RANK_TOP) -> list[dict[str, Any]]:
    """The last snapshot's top n (name, loss, score, value, unit) - what creator_status prints; [] when never measured."""
    snaps = [r for r in history(state) if r.get("event") == "snapshot"]
    return list(snaps[-1]["ranked"][:n]) if snaps else []


def format_report(rep: dict[str, Any], n: int = 8) -> str:
    h = rep.get("headline") or {}
    cur, prev = h.get("current", {}), h.get("previous", {})
    lines = [f"META (window {rep['window_h']:g}h): improvement rate {cur.get('per_hour', 0):.3f}/h (prev {prev.get('per_hour', 0):.3f}/h), "
             f"{cur.get('per_cycle_hour', 0):.3f} per cycle-hour, machinery {cur.get('machinery_per_hour', 0):.3f}/h, "
             f"meta rate {h.get('meta_rate_per_day', 0):+.3f}/h per day; adopted {cur.get('adopted', 0)} of {cur.get('cycles', 0)} cycles",
             "RANKED CONSTRAINTS (score = loss x weight):"]
    for i, m in enumerate(rep["ranked"][:n], 1):
        lines.append(f"  {i}. {m['name']}: score {m['score']} (loss {m['loss']}) value {m['value']} {m['unit']} [{m['window']}; prev {m['prev']}, {m['trend']}] remedy={m['remedy']}")
    for d in rep.get("drivers", []):
        lines.append(f"     driver of {d['parent']}: {d['name']} {d['value']:.0%} of student attempts")
    for s in rep.get("messages", []):
        lines.append("  " + s)
    return "\n".join(lines)


# ================================================================================================ improvement engine (blueprint 7.2-7.4, 7.11)
# DETECT (opportunity detectors over AGGREGATED metrics-bus events) -> VALUE (payback, one currency) -> DECIDE (top ROI, WIP 1,
# 10% exploration) and the TARGET REGISTRY (a ratchet: targets only get stricter). Extra fields on bus events the detectors read:
# sig (envelope signature), form_in/form_out (form-minimum tokens), cls (task class), err (error text), progress, queue,
# idle_frac, cores. Shadow results: metrics/shadow.jsonl {cls, rung, pass_rate, cost, n, current}; skill scores: metrics/skills.jsonl.
import random as _random
import time as _time

ENGINE_DIR = "engine"
REPEAT_N = 5                    # same envelope signature this many times per week -> cache / code tool
FAIL_N = 3                      # same error signature this many times in the window
STUCK_K = 5                     # cycles of one goal without progress
HOT_SHARE = 0.15                # a non-model actor above this share of total cost is a hot path
TOKEN_WASTE_X = 2.0             # mean prompt+output tokens this many times the form minimum
SHADOW_TOL = 0.02               # a smaller rung "passes" at current pass_rate minus this
SHADOW_MIN_N = 20
PAYBACK_MAX_DAYS = 3.0
ROI_HORIZON_DAYS = 30.0
EXPLORE_SHARE = 0.10
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
RISK_B_FACTOR = {"low": 1.0, "medium": 2.0, "high": 6.0}      # risk penalty on build cost (protected/core code = high)
CALIB_KEEP = 10
CALIB_CLAMP = (0.1, 2.0)
DAY_S = 86400.0


def cost_units(r: dict[str, Any], rate: Optional[dict[str, float]] = None) -> float:
    """ONE currency, CPU-seconds on this PC: cpu_s (else wall_s); + RAM-seconds (GB x wall) when RAM binds; + GPU-$ x cpu_s_per_gpu_usd."""
    rate = rate or {}
    c = float(r.get("cpu_s") if r.get("cpu_s") is not None else r.get("wall_s") or 0)
    if rate.get("ram_binds"):
        c += float(r.get("ram_peak_mb") or 0) / 1024.0 * float(r.get("wall_s") or 0) * float(rate.get("ram_weight", 1.0))
    return c + float(r.get("gpu_usd") or 0) * float(rate.get("cpu_s_per_gpu_usd", 0.0))


def load_events(state: Path, days: float, now: Optional[float] = None) -> list[dict[str, Any]]:
    now = _time.time() if now is None else now
    d = Path(state) / "metrics"
    out: list[dict[str, Any]] = []
    for f in sorted(d.glob("events-*.jsonl")) if d.is_dir() else []:
        for r in _jsonl(f):
            if now - float(r.get("t") or 0) <= days * DAY_S:
                out.append(r)
    return out


def _cand(kind: str, key: str, evidence: dict[str, Any], f_day: float, c0: float, c1: float, p: float, fix: str, risk: str = "low",
          build_s: float = 1800.0) -> dict[str, Any]:
    return {"kind": kind, "key": f"{kind}:{key}", "evidence": evidence, "f_per_day": round(f_day, 4), "c0": round(c0, 6), "c1": round(max(0.0, c1), 6),
            "p": p, "fix": fix, "risk": risk, "build_s": build_s}


def detect_hot_path(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    tot, by = 0.0, {}
    for r in rows:
        c = cost_units(r, rate)
        tot += c
        if not r.get("model"):
            a = by.setdefault(str(r.get("actor")), [0.0, 0])
            a[0] += c
            a[1] += 1
    return [_cand("hot_path", a, {"share": round(c / tot, 3), "total_cost": round(c, 2), "events": n}, n / days, c / n, c / n * 0.5, 0.6,
                  "optimize / cache / vectorize / move to idle") for a, (c, n) in by.items() if tot and c / tot >= HOT_SHARE]


def detect_repeated_call(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None, n_week: int = REPEAT_N) -> list[dict[str, Any]]:
    g: dict[str, list[float]] = {}
    for r in rows:
        if r.get("model") and r.get("sig") and not r.get("cache_hit"):
            g.setdefault(str(r["sig"]), []).append(cost_units(r, rate))
    out = []
    for s, cs in g.items():
        per_week = len(cs) * 7.0 / days
        if per_week >= n_week:
            c0 = sum(cs) / len(cs)
            out.append(_cand("repeated_call", s, {"calls": len(cs), "per_week": round(per_week, 1)}, (len(cs) - 1) / days, c0, c0 * 0.01, 0.9,
                             "result cache keyed on the envelope signature, or a code tool"))
    return out


def detect_shadow(state: Path, rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """model_too_big (a cheaper rung passes the class in shadow) and qwen_replaceable (the Nupen-own rung does)."""
    freq: dict[str, int] = {}
    for r in rows:
        if r.get("model") and r.get("cls"):
            freq[str(r["cls"])] = freq.get(str(r["cls"]), 0) + 1
    by: dict[str, list[dict[str, Any]]] = {}
    for s in _jsonl(Path(state) / "metrics" / "shadow.jsonl"):
        by.setdefault(str(s.get("cls")), []).append(s)
    out = []
    for cls, ss in by.items():
        cur = next((s for s in reversed(ss) if s.get("current")), None)
        if not cur or cls not in freq or cur.get("cost") is None:         # cost None = not measured: nothing is claimed
            continue
        ok = [s for s in ss if not s.get("current") and s.get("cost") is not None and int(s.get("n", 0)) >= SHADOW_MIN_N and float(s["cost"]) < float(cur["cost"])
              and float(s["pass_rate"]) >= float(cur["pass_rate"]) - SHADOW_TOL]
        if not ok:
            continue
        best = min(ok, key=lambda s: float(s["cost"]))
        own = str(best.get("rung")) == "own"
        out.append(_cand("qwen_replaceable" if own else "model_too_big", f"{cls}->{best['rung']}",
                         {"cls": cls, "from": cur.get("rung"), "to": best["rung"], "pass_from": cur["pass_rate"], "pass_to": best["pass_rate"], "n": best.get("n")},
                         freq[cls] / days, float(cur["cost"]), float(best["cost"]), 0.7 if own else 0.85,
                         "unload Qwen for this class" if own else "move the class down a rung"))
    return out


def detect_token_waste(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    g: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("model") and r.get("form_in") is not None:
            g.setdefault(str(r.get("actor")), []).append(r)
    out = []
    for a, rs in g.items():
        tok = sum(int(r.get("in_tok") or 0) + int(r.get("out_tok") or 0) for r in rs) / len(rs)
        mn = sum(int(r.get("form_in") or 0) + int(r.get("form_out") or 0) for r in rs) / len(rs)
        if mn > 0 and tok >= TOKEN_WASTE_X * mn:
            c0 = sum(cost_units(r, rate) for r in rs) / len(rs)
            out.append(_cand("token_waste", a, {"mean_tok": round(tok, 1), "form_min_tok": round(mn, 1), "ratio": round(tok / mn, 2)}, len(rs) / days, c0,
                             c0 * mn / tok, 0.7, "tighter context packer rule / grammar / max_tokens"))
    return out


def detect_recurring_failure(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None, n: int = FAIL_N) -> list[dict[str, Any]]:
    g: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("outcome") not in (None, "ok"):
            g.setdefault(f"{r.get('actor')}|{str(r.get('err') or r.get('outcome'))[:80]}", []).append(r)
    try:
        from creator import knownfix as KF
    except Exception:                                                   # noqa: BLE001
        KF = None
    out = []
    for sig, rs in g.items():
        if len(rs) >= n:
            text = str(rs[0].get("err") or rs[0].get("outcome"))
            kf = KF.match_line(text) if KF else []
            c0 = sum(cost_units(r, rate) for r in rs) / len(rs)
            out.append(_cand("recurring_failure", sig, {"count": len(rs), "known_fix": [k.id for k in kf]}, len(rs) / days, c0, 0.0, 0.85 if kf else 0.5,
                              "apply the known-fix entry" if kf else "add a known-fix entry or a code fix", "low" if kf else "medium"))
    return out


def detect_idle_resource(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    s = [r for r in rows if r.get("idle_frac") is not None]
    busy = [r for r in s if float(r["idle_frac"]) >= 0.3 and int(r.get("queue") or 0) > 0]
    if len(s) < 5 or len(busy) / len(s) < 0.5:
        return []
    c0 = sum(float(r["idle_frac"]) * float(r.get("cores") or 1) * float(r.get("wall_s") or 1) for r in busy) / len(busy)
    return [_cand("idle_resource", "cpu", {"samples": len(s), "idle_while_queued": len(busy)}, len(busy) / days, c0, c0 * 0.3, 0.7,
                  "scheduler rule: start queued work when CPU idle", "medium")]


def detect_weak_skill(state: Path) -> list[dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for s in _jsonl(Path(state) / "metrics" / "skills.jsonl"):
        last[str(s.get("role"))] = s
    out = []
    for role, s in last.items():
        if float(s["score"]) < float(s["target"]):
            fc = float(s.get("fail_cost", 0))
            out.append(_cand("weak_skill", role, {"score": s["score"], "target": s["target"], "gap": round(float(s["target"]) - float(s["score"]), 3)},
                             float(s.get("uses_per_day", 0)), fc * (1 - float(s["score"])), fc * (1 - float(s["target"])), 0.6,
                             "training module (GPU queue)", "low", float(s.get("train_cost_s", 7200))))
    return out


def detect_stuck(rows: list[dict[str, Any]], days: float, rate: Optional[dict[str, float]] = None, k: int = STUCK_K) -> list[dict[str, Any]]:
    g: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(rows, key=lambda r: float(r.get("t") or 0)):
        if r.get("goal_id") and r.get("step") == "cycle":
            g.setdefault(str(r["goal_id"]), []).append(r)
    out = []
    for gid, rs in g.items():
        tail = 0
        for r in reversed(rs):
            if r.get("progress"):
                break
            tail += 1
        if tail >= k:
            c0 = sum(cost_units(r, rate) for r in rs[-tail:]) / tail
            out.append(_cand("stuck", gid, {"cycles_without_progress": tail}, tail / days, c0, 0.0, 0.5, "fallback ladder: park, split, or escalate"))
    return out


def detect_missed_targets(state: Path) -> list[dict[str, Any]]:
    out = []
    for m, e in registry_load(state).items():
        cur = e.get("last_measured")
        if cur is not None and not _meets(e, cur):
            out.append(_cand("missed_target", m, {"last": cur, "target": e["target"], "floor": e["floor"]}, float(e.get("f_per_day", 0)),
                             float(e.get("c0", 0)), float(e.get("c1", 0)), 0.5, f"reach target {e['target']} for {m}"))
    return out


def detect_all(state: Path, days: float = 7.0, now: Optional[float] = None, rate: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """All detectors over the aggregated window (one read); a failing detector is skipped, never raises."""
    state = Path(state)
    rows = load_events(state, days, now)
    cands: list[dict[str, Any]] = []
    for fn in (lambda: detect_hot_path(rows, days, rate), lambda: detect_repeated_call(rows, days, rate), lambda: detect_shadow(state, rows, days, rate),
               lambda: detect_token_waste(rows, days, rate), lambda: detect_recurring_failure(rows, days, rate),
               lambda: detect_idle_resource(rows, days, rate), lambda: detect_weak_skill(state),
               lambda: detect_stuck(rows, days, rate), lambda: detect_missed_targets(state)):
        try:
            cands.extend(fn())
        except Exception:                                               # noqa: BLE001
            continue
    return cands


# ------------------------------------------------------------------------------------------------ VALUE (7.3) and DECIDE (7.4)
def _hist_file(state: Path) -> Path:
    return Path(state) / ENGINE_DIR / "predicted_vs_actual.jsonl"


def record_outcome(state: Path, cand: dict[str, Any], actual_saving_day: float) -> None:
    """After an attempt: predicted vs actual saving/day, the history that recalibrates the estimates."""
    v = cand.get("value") or value(cand)
    p = _hist_file(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "ab") as f:
        f.write((json.dumps({"kind": cand["kind"], "key": cand["key"], "pred_saving_day": v["raw_saving_day"],
                             "actual_saving_day": actual_saving_day, "at": _time.time()}) + "\n").encode())


def calibration(state: Optional[Path]) -> dict[str, float]:
    """Per kind: median(actual / predicted) over the last CALIB_KEEP attempts, clamped. Kinds without history are absent (= 1.0)."""
    by: dict[str, list[float]] = {}
    for r in _jsonl(_hist_file(state)) if state is not None else []:
        if float(r.get("pred_saving_day") or 0) > 0:
            by.setdefault(r["kind"], []).append(float(r["actual_saving_day"]) / float(r["pred_saving_day"]))
    lo, hi = CALIB_CLAMP
    return {k: min(hi, max(lo, statistics.median(v[-CALIB_KEEP:]))) for k, v in by.items()}


def value(cand: dict[str, Any], calib: Optional[dict[str, float]] = None, allowed_risk: str = "medium") -> dict[str, Any]:
    """saving/day = f x (c0 - c1) x p x calibration; B = build seconds x risk factor; payback = B / saving; worth = payback <= 3 d
    (equivalently ROI over 30 d >= 10) AND the risk class is allowed. Saving is in CPU-seconds/day, B in seconds: same currency."""
    raw = cand["f_per_day"] * max(0.0, cand["c0"] - cand["c1"]) * cand["p"]
    sav = raw * (calib or {}).get(cand["kind"], 1.0)
    b = cand["build_s"] * RISK_B_FACTOR[cand["risk"]]
    payback = b / sav if sav > 0 else float("inf")
    roi = sav * ROI_HORIZON_DAYS / b if b > 0 else 0.0
    allowed = RISK_ORDER[cand["risk"]] <= RISK_ORDER[allowed_risk]
    return {"raw_saving_day": raw, "saving_day": sav, "build_cost": b, "payback_days": payback, "roi_30d": roi, "risk_allowed": allowed,
            "worth": bool(allowed and (payback <= PAYBACK_MAX_DAYS or roi >= 10.0))}


def rank(cands: list[dict[str, Any]], state: Optional[Path] = None, allowed_risk: str = "medium") -> list[dict[str, Any]]:
    cal = calibration(state)
    out = [{**c, "value": value(c, cal, allowed_risk)} for c in cands]
    return sorted(out, key=lambda c: (-c["value"]["roi_30d"], c["key"]))


def decide(state: Path, cands: list[dict[str, Any]], in_flight: int = 0, rng: Optional[_random.Random] = None,
           allowed_risk: str = "medium") -> Optional[dict[str, Any]]:
    """Top ROI among the worth-it candidates; WIP 1 (nothing while a build is in flight); with probability EXPLORE_SHARE take a
    lower-ranked, allowed, still-uncertain (p < 0.7) candidate instead so estimates do not lock in. Logged to engine/decisions.jsonl."""
    if in_flight >= 1:
        return None
    rk = rank(cands, state, allowed_risk)
    worth = [c for c in rk if c["value"]["worth"]]
    if not worth:
        return None
    rng = rng or _random.Random()
    pick, mode = worth[0], "exploit"
    pool = [c for c in rk if c is not worth[0] and c["value"]["risk_allowed"] and c["p"] < 0.7 and c["value"]["saving_day"] > 0]
    if pool and rng.random() < EXPLORE_SHARE:
        pick, mode = pool[0], "explore"
    d = Path(state) / ENGINE_DIR / "decisions.jsonl"
    d.parent.mkdir(parents=True, exist_ok=True)
    with open(d, "ab") as f:
        f.write((json.dumps({"at": _time.time(), "mode": mode, "key": pick["key"], "value": pick["value"]}, default=str) + "\n").encode())
    return {**pick, "mode": mode}


# ------------------------------------------------------------------------------------------------ TARGET REGISTRY (7.11)
TARGETS_FILE = "targets.json"
TARGET_X = 1.5                  # target = TARGET_X x floor (lower-is-better); the mirror image for higher-is-better


def registry_load(state: Path) -> dict[str, dict[str, Any]]:
    return _json(Path(state) / ENGINE_DIR / TARGETS_FILE)


def _registry_save(state: Path, reg: dict[str, dict[str, Any]]) -> None:
    p = Path(state) / ENGINE_DIR / TARGETS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=1, sort_keys=True), encoding="utf-8")


def _meets(e: dict[str, Any], v: float) -> bool:
    return v >= e["target"] if e.get("higher_is_better") else v <= e["target"]


def target_set(state: Path, metric: str, *, floor: float, target: float, higher_is_better: bool = False, best_measured: Optional[float] = None,
               evidence: str = "", **extra: Any) -> dict[str, Any]:
    """Create a metric record, or re-set an existing one (an existing target is only ever replaced by a stricter one)."""
    reg = registry_load(state)
    e = reg.get(metric)
    if e:
        hib = bool(e.get("higher_is_better"))
        target = max(target, e["target"]) if hib else min(target, e["target"])
        e.update({"floor": floor, "target": target, "evidence": evidence or e["evidence"], **extra})
    else:
        e = {"floor": floor, "target": target, "higher_is_better": higher_is_better, "best_measured": best_measured, "last_measured": None,
             "evidence": evidence, "history": [], **extra}
    e["history"].append({"event": "set", "target": e["target"], "floor": floor, "evidence": evidence, "at": _time.time()})
    reg[metric] = e
    _registry_save(state, reg)
    return e


def target_measure(state: Path, metric: str, value_: float, evidence: str = "") -> dict[str, Any]:
    """Record a measurement. When the best measured beats the target, the target moves toward ~1.5x the (re-estimated) floor, in the
    strict direction only, with the evidence logged. Higher-is-better metrics are handled on the reciprocal scale."""
    reg = registry_load(state)
    e = reg[metric]
    hib = bool(e.get("higher_is_better"))
    e["last_measured"] = value_
    b = e.get("best_measured")
    if b is None or (value_ > b if hib else value_ < b):
        e["best_measured"] = b = value_
    if (b >= e["target"]) if hib else (b <= e["target"]):
        t = (lambda x: 1.0 / x) if hib else (lambda x: x)               # work on a lower-is-better scale
        floor, old, best = t(e["floor"]), t(e["target"]), t(b)
        floor = min(floor, best * 0.75)                                 # beating the target is evidence the floor is lower than thought
        new = min(old, max(TARGET_X * floor, min(best * 1.1, old * 0.9)))     # never less strict
        if new < old * (1 - 1e-9):
            e["history"].append({"event": "raised", "from": e["target"], "to": t(new), "best": b, "floor": t(floor), "evidence": evidence, "at": _time.time()})
            e["target"], e["floor"] = t(new), t(floor)
    reg[metric] = e
    _registry_save(state, reg)
    return e


# Seeded from the PHASE0_WORK.md ratchet log (floor, target as of 4 Oct 23:45; best_measured where one was measured).
RATCHET_SEED: dict[str, dict[str, Any]] = {
    "p0.2_event_us": dict(floor=4.0, target=10.0, best_measured=7.4, evidence="ratchet 4 Oct 23:05/23:30: 940->38->8 us/event"),
    "p0.3_locate_p50_ms": dict(floor=2.5, target=5.0, best_measured=7.0, evidence="ratchet 23:05: one indexed query; measured p50 7-18 ms"),
    "p0.3_locate_p95_ms": dict(floor=3.0, target=15.0, best_measured=20.0, evidence="ratchet 23:05; measured p95 20-67 ms"),
    "p0.3_incremental_update_s": dict(floor=0.04, target=0.1, evidence="ratchet 23:05: stat 1730 files"),
    "p0.8_envelope_median_tok": dict(floor=27.0, target=35.0, best_measured=27.5, evidence="ratchet 23:20: at floor"),
    "p0.8_envelope_max_tok": dict(floor=39.0, target=60.0, best_measured=39.0, evidence="ratchet 23:20: at floor"),
    "p0.4_tool_overhead_s": dict(floor=0.06, target=0.1, evidence="ratchet 23:10"),
    "p0.6_diff_check_ms": dict(floor=5.0, target=5.0, best_measured=5.0, evidence="ratchet 23:05: at floor, kept"),
    "p0.5_locate_top1": dict(floor=0.9, target=0.6, higher_is_better=True, evidence="ratchet 23:05: published methods reach 0.6 (floor = limit)"),
    "p0.5_locate_top5": dict(floor=0.97, target=0.8, higher_is_better=True, evidence="ratchet 23:05"),
}


def seed_registry(state: Path) -> int:
    """Add the ratchet-log metrics not yet in the registry; returns how many were added."""
    reg, n = registry_load(state), 0
    for m, s in RATCHET_SEED.items():
        if m not in reg:
            target_set(state, m, **s)
            if s.get("best_measured") is not None:
                target_measure(state, m, s["best_measured"], s["evidence"])
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ one engine cycle (OBSERVE-DETECT-VALUE-DECIDE)
RECORD_FILE = "decision_record.json"
IN_FLIGHT_FILE = "in_flight.json"
IN_FLIGHT_TTL_S = 86400.0       # a build nobody reported on for a day no longer holds the WIP slot


def in_flight(state: Path, now: Optional[float] = None) -> int:
    now = _time.time() if now is None else now
    d = _json(Path(state) / ENGINE_DIR / IN_FLIGHT_FILE)
    return sum(1 for t in d.values() if now - float(t) < IN_FLIGHT_TTL_S)


def mark_done(state: Path, key: str, actual_saving_day: Optional[float] = None, cand: Optional[dict[str, Any]] = None) -> None:
    """The builder/adopter reports the end of an attempt: frees the WIP slot and (with a measured saving) feeds the calibration."""
    p = Path(state) / ENGINE_DIR / IN_FLIGHT_FILE
    d = _json(p)
    d.pop(key, None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    if actual_saving_day is not None and cand is not None:
        record_outcome(state, cand, actual_saving_day)


def engine_cycle(state: Path, now: Optional[float] = None, days: float = 7.0, rng: Optional[_random.Random] = None, claim: bool = True) -> dict[str, Any]:
    """Detect over the aggregated bus, rank, decide (WIP 1, 10% exploration); writes engine/decision_record.json (the ranked top 10, the
    decision or why none) and, on a decision, takes the WIP slot. Read-only on everything else; never touches kernel adoption."""
    state = Path(state)
    now = _time.time() if now is None else now
    t0 = _time.perf_counter()
    if not registry_load(state):
        seed_registry(state)
    cands = detect_all(state, days, now)
    busy = in_flight(state, now)
    pick = decide(state, cands, in_flight=busy, rng=rng)
    rk = rank(cands, state)
    rec = {"at": now, "window_days": days, "candidates": len(cands), "in_flight": busy,
           "decision": ({k: pick[k] for k in ("key", "kind", "fix", "risk", "p", "f_per_day", "c0", "c1", "build_s", "evidence", "value", "mode")} if pick else None),
           "why_none": "" if pick else ("WIP 1: a build is in flight" if busy else ("no candidates" if not cands else "no candidate passes payback <= 3 d with an allowed risk")),
           "top": [{"key": c["key"], "roi_30d": round(c["value"]["roi_30d"], 3), "payback_days": round(c["value"]["payback_days"], 3), "worth": c["value"]["worth"]} for c in rk[:10]],
           "seconds": round(_time.perf_counter() - t0, 3)}
    d = state / ENGINE_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / RECORD_FILE).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    if pick and claim:
        f = d / IN_FLIGHT_FILE
        cur = _json(f)
        cur[pick["key"]] = now
        f.write_text(json.dumps(cur), encoding="utf-8")
    return rec
