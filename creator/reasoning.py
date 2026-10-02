"""Nupen's THINKING PROCEDURE (owner, 2 Oct 2026: "exactly how to know what to do ... find all the fundamentals of how an ai should think ...
speed up time by letting it know so we dont have to wait until millions of trial and error").

The procedure is fixed and versioned (STEPS). It is applied, not just stated, in three places:
  * the work package (planner.build_package) opens with `how_to_approach()`: the procedure, what worked / failed before for this
    component and kind (recalled from the lessons, the failure memory and the diagnoses), the engineering principles for the kind
    (creator.fundamentals, optional) and the sparse-signal rule (creator.registry, optional);
  * predictions: a student records a predicted effect with each candidate (WorkResult.predicted); curriculum stores it next to the
    measured effect of the same change; `calibration()` scores every student and creator.constraints ranks a poorly calibrated one;
  * stop rules: `stalled()` (the planner stops an approach that failed twice with the same reason and no new evidence) and
    `identical_failed()` (a student's change identical to an already rejected one is not evaluated again): both route to the teacher.
Optional modules are imported by name behind a try: absent or broken = skipped, planning never breaks on them."""
from __future__ import annotations

import ast
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

VERSION = 1
MAX_FAILS_NO_EVIDENCE = 2            # stop rule: this many failures with the same reason and nothing new learned -> stop, escalate
EXPLORE_SHARE = 0.2                  # explore vs exploit: at most this share of attempts on approaches not yet known to work
EVIDENCE_TYPES = ("Finding", "Diagnosis", "ResearchQuestion")
WELL_CALIBRATED_ERR = 0.5
NEEDS_TEACHER = "escalated to the teacher"

STEPS: tuple[tuple[str, str, str], ...] = (
    ("goal", "State the goal and its measurable done-criterion.",
     "Write the package's check (check:<step>:<component>) in your own words before touching code; done = that computed check passes, "
     "the audit is clean and no test regresses. 'The change looks right' is not done."),
    ("constraint", "Find the constraint and check the task serves it.",
     "creator.constraints ranks what limits the improvement rate; say which constraint this change relieves, or that it serves the "
     "requirement directly. A change that serves neither is not worth making."),
    ("recall", "RECALL before trying.",
     "Read what worked and what failed before for this component and kind (below). Never repeat a failed change; start from the "
     "approach that worked."),
    ("hypothesis", "Form a hypothesis with a predicted, measurable effect.",
     "'If I change X, check Y moves from A to B' - a number or a pass/fail, not a feeling."),
    ("experiment", "Choose the smallest experiment that can falsify it.",
     "The smallest edit and the one test that fails if the hypothesis is wrong; no bundled changes, so a failure says which one."),
    ("predict", "Predict before measuring and record the prediction.",
     "Return the predicted effect (e.g. size_delta in AST nodes) with the candidate; the system stores it next to the measured effect."),
    ("measure", "Measure, then compare with the prediction.",
     "The kernel measures. A miss is information: the gap between predicted and measured is your calibration, and it is scored."),
    ("stop", "Apply the stop rules.",
     f"After {MAX_FAILS_NO_EVIDENCE} failures with the same reason and no new evidence stop that approach; an identical change to a "
     "rejected one is never retried; when stuck, escalate to the teacher with the reason."),
    ("record", "Record the lesson.",
     "State what was learned in general terms (which kind of change, which signal), not only what happened this once."),
)

HEURISTICS: tuple[str, ...] = (
    "Expected value over cost: prefer the change with the best (probability of adoption x benefit) per unit of work.",
    f"Explore versus exploit: try the best-known approach first; give at most {EXPLORE_SHARE:.0%} of attempts to a new one.",
    "Prefer reversible actions: small, sandboxed, easily reverted changes over sweeping ones.",
    "Prefer known-good patterns over invention: reuse what the lessons show was adopted before.",
)


def procedure_text(compact: bool = True) -> str:
    """The procedure as text: compact = one line per step (for a package); else with what each step means in Nupen."""
    lines = [f"{i}. {rule}" + ("" if compact else f" {means}") for i, (_n, rule, means) in enumerate(STEPS, 1)]
    return f"Thinking procedure v{VERSION}: " + (" ".join(lines) if compact else "\n".join(lines)) + \
        " Heuristics: " + " ".join(HEURISTICS)


def _optional(module: str, func: str, *args: Any) -> list[str]:
    """Call creator.<module>.<func>(*args) if the module exists; any failure or absence yields []."""
    try:
        out = getattr(importlib.import_module(module), func)(*args)
    except Exception:                                                   # noqa: BLE001 - a missing helper never breaks planning
        return []
    if not out:
        return []
    return [str(x) for x in out] if isinstance(out, (list, tuple)) else [str(out)]


# ------------------------------------------------------------------------------------------------ 3. recall

def _lessons_of(ledger: Any, lessons: Optional[Sequence[Any]]) -> list[Any]:
    if lessons is not None:
        return list(lessons)
    try:
        from creator import curriculum as CUR
        return CUR.LessonLog(Path(ledger.path).parent / "lessons.jsonl").lessons()
    except Exception:                                                   # noqa: BLE001
        return []


def recall_lessons(lessons: Iterable[Any], component: str, kind: str, limit: int = 4) -> list[str]:
    """What worked (adopted) first, then what failed (rejected), for this component and kind; teacher and student alike."""
    worked: list[tuple[bool, str]] = []
    failed: list[tuple[bool, str]] = []
    for les in lessons:
        same_c, same_k = getattr(les, "component", "") == component, getattr(les, "task_kind", "") == kind
        if getattr(les, "adopted", None) is None or not (same_c or same_k):
            continue
        same = same_c and same_k
        verdict = str(getattr(les, "verdict", ""))
        if verdict.lower().startswith(("cancelled", "error")):
            continue
        files = ", ".join(sorted(getattr(les, "files_after", {}))[:3])
        line = (f"{getattr(les, 'solver', '?')} on {getattr(les, 'component', '?')} ({getattr(les, 'package_id', '')}"
                f"{', ' + files if files else ''})")
        (worked if les.adopted else failed).append((same, line + ("" if les.adopted else f": {verdict[:140]}")))
    pick = lambda xs, n: [t for _s, t in sorted(xs, key=lambda p: not p[0])][:n]     # noqa: E731 - same component first
    return ["what worked before: " + t for t in pick(worked, max(1, limit // 2))] + \
           ["what failed before: " + t for t in pick(failed, limit - min(len(worked), max(1, limit // 2)))]


def recall(ledger: Any, component: str, kind: str, objective: str, prior: Sequence[str] = (),
           lessons: Optional[Sequence[Any]] = None) -> list[str]:
    """Lessons + the ledger's failure memory, diagnosed root causes and findings, worked-before first."""
    out = recall_lessons(_lessons_of(ledger, lessons), component, kind)
    try:
        from creator import planner as PL
        out += PL.seen_before(ledger, objective, prior)
    except Exception:                                                   # noqa: BLE001
        pass
    return out


def constraint_line(ledger: Any) -> str:
    try:
        from creator import constraints as CO
        top = CO.top_constraints(Path(ledger.path).parent, 1)
    except Exception:                                                   # noqa: BLE001
        return ""
    return f"current top constraint: {top[0]['name']} (score {top[0].get('score')})" if top else ""


def how_to_approach(ledger: Any, component: str, kind: str, objective: str, prior: Sequence[str] = (),
                    lessons: Optional[Sequence[Any]] = None) -> tuple[str, ...]:
    """The 'How to approach this' section of a work package: one string per line group, compact."""
    out = ["How to approach this: " + procedure_text(compact=True)]
    cl = constraint_line(ledger)
    if cl:
        out.append("How to approach this - constraint: " + cl + "; say how this task serves it.")
    rec = recall(ledger, component, kind, objective, prior, lessons)
    out.append("How to approach this - recalled: " + (" | ".join(rec) if rec else "nothing recorded yet for this component/kind; "
                                                      "say so and keep the experiment small"))
    pr = _optional("creator.fundamentals", "principles_for", kind)
    if pr:
        out.append("How to approach this - principles: " + " ".join(pr))
    sp = _optional("creator.registry", "sparse_rule_text")
    if sp:
        out.append("How to approach this - sparse rule: " + " ".join(sp))
    return tuple(s[:1800] for s in out)


# ------------------------------------------------------------------------------------------------ 4-7. prediction and calibration

def nodes(src: str) -> int:
    """AST node count of python source (a snippet that does not parse falls back to a word-count estimate)."""
    try:
        return sum(1 for _ in ast.walk(ast.parse(src)))
    except (SyntaxError, ValueError):
        for cand in (re.sub(r"^", "    ", src, flags=re.M),):
            try:
                return sum(1 for _ in ast.walk(ast.parse("if 1:\n" + cand))) - 3
            except (SyntaxError, ValueError):
                pass
        return len(re.findall(r"\w+|[^\s\w]", src)) // 2


def size_delta(before: Mapping[str, str], after: Mapping[str, str]) -> float:
    """Measured effect of a change: AST nodes after minus before, over the changed python files."""
    return float(sum(nodes(t) for p, t in after.items() if p.endswith(".py")) - sum(nodes(t) for p, t in before.items() if p.endswith(".py")))


def calibration(lessons: Iterable[Any], solvers_exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Per solver, over claimed-done attempts: how many recorded a prediction, how many predictions were well calibrated (sign right and
    relative error <= WELL_CALIBRATED_ERR), the mean relative error, and score = well calibrated / claimed attempts, so a student that
    never predicts is poorly calibrated too."""
    acc: dict[str, dict[str, Any]] = {}
    for les in lessons:
        solver = str(getattr(les, "solver", ""))
        if solver in solvers_exclude or not getattr(les, "claimed_done", False):
            continue
        a = acc.setdefault(solver, {"claimed": 0, "predicted": 0, "well": 0, "errs": []})
        a["claimed"] += 1
        pred, meas = getattr(les, "predicted", None) or {}, getattr(les, "measured", None) or {}
        for k, p in pred.items():
            if k not in meas:
                continue
            m = float(meas[k])
            p = float(p)
            err = abs(p - m) / max(abs(p), abs(m), 1.0)
            a["predicted"] += 1
            a["errs"].append(err)
            if err <= WELL_CALIBRATED_ERR and (p * m > 0 or p == m):
                a["well"] += 1
            break
    return {s: {"claimed": a["claimed"], "predicted": a["predicted"], "well_calibrated": a["well"],
                "coverage": round(a["predicted"] / a["claimed"], 3),
                "mean_error": round(sum(a["errs"]) / len(a["errs"]), 3) if a["errs"] else None,
                "score": round(a["well"] / a["claimed"], 3)} for s, a in sorted(acc.items())}


# ------------------------------------------------------------------------------------------------ 8. stop rules

def normalize_reason(reason: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", re.sub(r"^\S+ failed:\s*", "", reason.lower()))).strip()[:100]


def stalled(ledger: Any, prior: Sequence[str], limit: int = MAX_FAILS_NO_EVIDENCE) -> str:
    """Non-empty reason when the last `limit` real failures of a gap's packages share one reason and nothing was learned since the first
    of them (no Finding, Diagnosis or ResearchQuestion, no unblock): the approach is exhausted, not the gap."""
    fails: list[tuple[int, str]] = []
    for w in prior:
        for t in ledger.about(w):
            if t.rtype == "Transition" and getattr(t.record, "to_state").name in ("FAILED", "ROLLED_BACK", "REJECTED"):
                fails.append((t.seq, str(getattr(t.record, "reason", ""))))
    fails.sort()
    if len(fails) < limit:
        return ""
    last = fails[-limit:]
    if len({normalize_reason(r) for _s, r in last}) != 1:
        return ""
    since = last[0][0]
    if any(e.seq > since for rt in EVIDENCE_TYPES for e in ledger.of_type(rt)):
        return ""
    for g in {p for w in prior for p in getattr(ledger.get(w), "parents", ())}:
        for t in ledger.about(g):
            if t.seq > since and t.rtype == "Transition" and str(getattr(t.record, "reason", "")).startswith("unblocked"):
                return ""
    return f"{limit} failures with the same reason without new evidence ({last[-1][1][:120]}); {NEEDS_TEACHER}"


def diff_hash(files_before: Mapping[str, str], files_after: Mapping[str, str]) -> str:
    """Identity of a change: the post-change text of every touched file, whitespace-insensitive at line ends."""
    h = hashlib.sha256()
    for p in sorted(files_after):
        h.update(p.encode() + b"\0")
        h.update("\n".join(ln.rstrip() for ln in files_after[p].splitlines()).encode() + b"\0")
        h.update("\n".join(ln.rstrip() for ln in files_before.get(p, "").splitlines()).encode() + b"\1")
    return h.hexdigest()[:16]


def identical_failed(lessons: Iterable[Any], component: str, files_before: Mapping[str, str],
                     files_after: Mapping[str, str]) -> Optional[Any]:
    """The earlier lesson (rejected, a real verdict) whose change is identical to this one on the same component, if any."""
    if not files_after:
        return None
    h = diff_hash(files_before, files_after)
    for les in lessons:
        if (getattr(les, "component", "") == component and getattr(les, "adopted", None) is False
                and str(getattr(les, "verdict", "")).upper().startswith(("REJECTED", "ROLLED_BACK"))
                and getattr(les, "files_after", None) and diff_hash(les.files_before, les.files_after) == h):
            return les
    return None
