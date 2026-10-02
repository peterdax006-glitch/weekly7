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
                         tuple(sorted((ev.get("changed") or {}).keys()))))
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
    loss = max(0.0, 1.0 - p25 / m)
    return Metric("cycle_time", round(m / 60.0, 2), "median minutes", f"last {window_h:g}h", None if pm is None else round(pm / 60.0, 2),
                  _trend(m, pm), round(loss, 4), "meta",
                  {"p25_minutes": round(p25 / 60.0, 2), "stage_seconds": stages, "dominant_stage": max(stages, key=lambda k: stages[k]), "n": len(done)}, "process")


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


METRICS: tuple[Callable[[Path, dt.datetime, float], Any], ...] = (
    waste_metrics, cycle_time_metric, eval_cost_metric, supply_metric, availability_metric, learning_metrics, recursion_metric, audit_metric, fundamentals_metric)


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
            "student_skill": "raise the student adopted rate on task kinds that every student fails",
            "teacher_dependence": "move task kinds from the teacher to students (handoff-free adoption)",
            "work_supply": "plan more packages per round (more open gaps, wider scheduling)",
            "cycle_time": "shorten the evaluation feedback loop (the dominant stage)",
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
