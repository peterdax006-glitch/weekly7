"""K29 - remove the limiting factor learning_signal: every student attempt should reach a measured kernel verdict.

A student only learns from attempts the kernel JUDGED (ADOPTED / REJECTED / ROLLED_BACK). This module measures how many attempts
reach a verdict, classifies WHY the others did not (model timeout, unusable reply, no candidate, interrupted, cancelled, error,
claimed but never judged), names the lever that removes each cause, and finds cancelled/errored cycles that have no lesson at all
(attempts that left no record cannot teach anything). Pure functions over lessons.jsonl rows and kernel_log.jsonl rows: no I/O
except the small loaders, nothing is written.

Measured on 2 Oct 2026 (302 lessons): 11 judged; 53 interrupted, 76 claimed-but-unjudged, 24 model timeouts, 13 no candidate,
11 unusable replies, 12 cancelled - the learning signal was under 5%.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

JUDGED = "judged"
INTERRUPTED = "interrupted"
CANCELLED = "cancelled"
ERROR = "error"
MODEL_TIMEOUT = "model_timeout"
UNUSABLE_REPLY = "unusable_reply"
NO_CANDIDATE = "no_candidate"
DEFERRED = "deferred"
UNJUDGED = "claimed_unjudged"
NOT_CLAIMED = "not_claimed_other"
NOT_AN_ATTEMPT = "not_an_attempt"                  # teacher lessons written ahead of any cycle

VERDICTS = ("ADOPTED", "REJECTED", "ROLLED_BACK")
_REASON_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (MODEL_TIMEOUT, ("model call failed", "did not become healthy", "timed out")),
    (UNUSABLE_REPLY, ("no usable", "search text not found", "no edit applied", "unusable")),
    (NO_CANDIDATE, ("prescreen: no candidate", "no valid action candidates", "no learned template", "no target python file",
                    "no unmeasured teacher solution", "cannot attempt")),
    (DEFERRED, ("teacher absent", "deferred")),
)

# What removes each cause (the lever names the part of Nupen that owns it).
REMEDIES: dict[str, dict[str, str]] = {
    MODEL_TIMEOUT: {"lever": "creator/generator.py LocalModel (server slots, start-up timeout, threads)",
                    "action": "start model servers before they are needed and scale the start-up wait with servers starting"},
    UNUSABLE_REPLY: {"lever": "creator/model_student.py reply format / creator/action_student.py actions",
                     "action": "ask for edits in a format that is validated before it counts (actions over free text)"},
    NO_CANDIDATE: {"lever": "creator/curriculum.py routing",
                   "action": "route the task kind to a student that has a candidate for it; hand it to the teacher otherwise"},
    INTERRUPTED: {"lever": "stopping and deploying (drain mode, creator/swarmops.py release_orphans)",
                  "action": "drain in-flight cycles before a stop so their attempts reach a verdict"},
    CANCELLED: {"lever": "creator/swarm.py Governor admission",
                "action": "admit only work whose evaluation fits in memory, so it is not pulled back before the verdict"},
    ERROR: {"lever": "creator/kernel.py / creator/sandbox.py",
            "action": "fix the cycle error (sandbox, git, evaluation) named in the verdict"},
    UNJUDGED: {"lever": "creator/curriculum.py reconcile",
               "action": "settle every claimed attempt with the kernel's outcome for its package"},
    DEFERRED: {"lever": "the teacher presence rule (scripts/creator_swarm.py --teacher-presence)",
               "action": "keep the teacher heartbeat fresh or let a student take the deferred kind"},
    NOT_CLAIMED: {"lever": "the student that gave up",
                  "action": "read the student's reason and add the missing capability"},
}


def classify(lesson: Mapping[str, Any]) -> str:
    """Why this lesson did or did not produce a learning signal: one of the cause constants above.
    A lesson with a kernel verdict (ADOPTED / REJECTED / ROLLED_BACK) is JUDGED; a teacher lesson is NOT_AN_ATTEMPT."""
    verdict = str(lesson.get("verdict") or "")
    reason = str(lesson.get("reasoning") or "").lower()
    if verdict.startswith(VERDICTS):
        return JUDGED
    if verdict.startswith("teacher lesson"):
        return NOT_AN_ATTEMPT
    for prefix, cause in (("interrupted", INTERRUPTED), ("cancelled", CANCELLED), ("error", ERROR)):
        if verdict.lower().startswith(prefix):
            return cause
    for cause, needles in _REASON_RULES:
        if any(n in reason for n in needles):
            return cause
    if lesson.get("claimed_done"):
        return UNJUDGED
    return NOT_CLAIMED


def signal_report(lessons: Iterable[Mapping[str, Any]], since: Optional[str] = None) -> dict[str, Any]:
    """Counts per cause, the learning-signal rate (judged / attempts) overall and per solver, and the causes ranked by how many
    attempts they cost. `since` (ISO time prefix-comparable with the lessons' 'at') limits the window. Teacher lessons are not
    attempts. An empty input gives rate None and no causes."""
    causes: collections.Counter[str] = collections.Counter()
    per_solver: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for les in lessons:
        if since is not None and str(les.get("at") or "") < since:
            continue
        cause = classify(les)
        if cause == NOT_AN_ATTEMPT:
            continue
        causes[cause] += 1
        per_solver[str(les.get("solver") or "?")][cause] += 1
    attempts = sum(causes.values())
    lost = sorted(((c, n) for c, n in causes.items() if c != JUDGED), key=lambda x: -x[1])
    return {
        "attempts": attempts,
        "judged": causes[JUDGED],
        "signal_rate": round(causes[JUDGED] / attempts, 4) if attempts else None,
        "causes": dict(causes),
        "lost_by_cause": lost,
        "per_solver": {s: {"attempts": sum(c.values()), "judged": c[JUDGED],
                           "signal_rate": round(c[JUDGED] / sum(c.values()), 4)} for s, c in sorted(per_solver.items())},
    }


def remedies(report: Mapping[str, Any], top: int = 3) -> list[dict[str, Any]]:
    """The levers for the `top` causes that cost the most attempts, most costly first: {cause, lost, lever, action}."""
    out = []
    for cause, n in list(report.get("lost_by_cause") or [])[:top]:
        fix = REMEDIES.get(cause, REMEDIES[NOT_CLAIMED])
        out.append({"cause": cause, "lost": n, "lever": fix["lever"], "action": fix["action"]})
    return out


def unrecorded_cycles(lessons: Iterable[Mapping[str, Any]], kernel_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Packages the kernel CANCELLED or ERRORed that have NO lesson: attempts that left no record and so teach nothing.
    Kernel rows carry 'package' and 'outcome' (kernel_log.jsonl)."""
    taught = {str(les.get("package_id")) for les in lessons if les.get("package_id")}
    missing = []
    for row in kernel_rows:
        pkg, outcome = str(row.get("package") or ""), str(row.get("outcome") or "")
        if pkg and outcome in ("CANCELLED", "ERROR") and pkg not in taught and pkg not in missing:
            missing.append(pkg)
    return missing


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file; a missing file or a broken line is skipped, never raised."""
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def assess(state: Path, since: Optional[str] = None) -> dict[str, Any]:
    """The whole K29 picture for a state directory: the signal report, the top remedies and the unrecorded cycles."""
    lessons = read_jsonl(Path(state) / "lessons.jsonl")
    report = signal_report(lessons, since)
    return {**report, "remedies": remedies(report),
            "unrecorded_cycles": unrecorded_cycles(lessons, read_jsonl(Path(state) / "kernel_log.jsonl"))}


def summary_lines(result: Mapping[str, Any]) -> Sequence[str]:
    """A short human-readable report of `assess` output (for the constraint report and the terminal)."""
    rate = result.get("signal_rate")
    lines = [f"learning signal: {result.get('judged', 0)} of {result.get('attempts', 0)} attempts judged"
             + (f" ({rate:.1%})" if rate is not None else " (no attempts)")]
    for r in result.get("remedies") or []:
        lines.append(f"- {r['cause']}: {r['lost']} attempts lost -> {r['action']} [{r['lever']}]")
    if result.get("unrecorded_cycles"):
        lines.append(f"- {len(result['unrecorded_cycles'])} cancelled/errored cycles left no lesson")
    return lines
