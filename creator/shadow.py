"""Shadow evaluation of Nupen's own trained chooser against the default policy, and the PRE-REGISTERED SWITCH RULE.

Every real ActionStudent decision also records what the chooser would have picked (state/creator/shadow_choices.jsonl). The lesson
log supplies the kernel's measured outcome later (adopted / rejected; cancelled and errored cycles are NO SIGNAL and ignored).
The shadow code never changes a decision and never raises into the student.

PRE-REGISTERED SWITCH RULE (fixed 2 Oct 2026, before any data; do not edit the thresholds to fit results):
  Evidence is REAL ONLY: lessons in state/creator/lessons.jsonl whose outcome is a real kernel judgement (never synthetic /
  contrast / teacher-generated lessons, never cancelled/error). Each adopted real lesson is one decision whose good answer is the
  action set that lesson took (creator.chooser.lesson_rows). The chooser is scored by LEAVE-ONE-OUT (trained on the other real rows
  only); the default policy (lexical prior, the model is not available offline) is scored on the same rows. A hit = the 1-based
  pick lands in the adopted action set. n = number of such real decisions. Wilson 95% intervals (z = 1.96) everywhere.
  FLIP TO THE CHOOSER when the policy is currently the default AND n >= 30 AND either
     (A) wilson_lower(chooser hits, n) >= default hits / n, or
     (B) among resolved SHADOW decisions where the two picks differed and the applied pick was ADOPTED, tested disagreements
         t = (chooser-applied adopted) + (default-applied adopted) >= 10 and wilson_lower(chooser-applied adopted, t) > 0.5.
  FLIP BACK TO THE DEFAULT when the policy is currently the chooser AND n >= 30 AND chooser hits / n < wilson_lower(default hits, n).
  Each flip is appended as a JSON event (with the numbers) to state/creator/policy_events.jsonl and the decision persisted in
  state/creator/policy.json {"use_chooser": bool}; ActionStudent reads it when use_chooser is not given explicitly.
The kernel's measurement still decides adoption; nothing here judges itself."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "creator"
MIN_N = 30
MIN_TESTED = 10
Z = 1.96
STUDENT = "nupen-model-v2"
log = logging.getLogger("creator.shadow")


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------------------------------ policy file


def read_policy(state: Path = STATE) -> Optional[bool]:
    """The persisted decision, or None when absent/unreadable (the module default then applies)."""
    try:
        v = json.loads((Path(state) / "policy.json").read_text(encoding="utf-8")).get("use_chooser")
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, bool) else None


def _write_policy(use: bool, evidence: dict[str, Any], state: Path) -> None:
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    (state / "policy.json").write_text(json.dumps({"use_chooser": use, "at": now}), encoding="utf-8")
    with (state / "policy_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now, "event": "flip", "to": "chooser" if use else "default", "evidence": evidence}, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------------ shadow rows


def record(state: Path, package_id: str, cands: list[Any], default_pick: Optional[int], chooser_pick: Optional[int],
           model_pick: Optional[int], applied: str, lesson_id: str = "") -> None:
    """Append one shadow row. Never raises."""
    try:
        state = Path(state)
        state.mkdir(parents=True, exist_ok=True)
        row = {"package_id": package_id, "lesson_id": lesson_id, "candidates": [a.short() for a in cands],
               "default_pick": default_pick, "chooser_pick": chooser_pick, "model_pick": model_pick, "applied": applied,
               "at": dt.datetime.now().isoformat(timespec="seconds")}
        with (state / "shadow_choices.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as e:                                   # noqa: BLE001 - shadowing must never hurt the student
        log.warning("shadow record failed: %s", e)


def _rows(state: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in (Path(state) / "shadow_choices.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def resolve(state: Path, lessons: list[Any]) -> list[dict[str, Any]]:
    """Shadow rows joined to the student's lessons (by lesson id, else the latest lesson of that package by this student);
    only rows with a real outcome (adopted True/False, not no-signal) are returned, with 'adopted' filled in."""
    from creator.curriculum import is_skill_signal
    by_id = {les.lesson_id: les for les in lessons}
    by_pkg: dict[str, list[Any]] = {}
    for les in lessons:
        if les.solver == STUDENT:
            by_pkg.setdefault(les.package_id, []).append(les)
    for lst in by_pkg.values():
        lst.sort(key=lambda x: x.at or "")                              # stable: lessons without a stamp keep their log order
    used: set[str] = set()
    out = []
    for r in _rows(state):
        les = by_id.get(r.get("lesson_id") or "")
        if les is None:
            # A package retried several times has several rows and several lessons: a row belongs to the FIRST lesson written at or
            # after the decision and not yet claimed, never to the package's latest lesson (that fed one attempt's outcome to all).
            for x in by_pkg.get(str(r.get("package_id")), []):
                if x.lesson_id in used or (x.at and r.get("at") and x.at < str(r["at"])):
                    continue
                les = x
                if x.at:
                    used.add(x.lesson_id)
                break
        if les is None or les.adopted is None or not is_skill_signal(les):
            continue
        out.append({**r, "adopted": bool(les.adopted)})
    return out


# ------------------------------------------------------------------------------------------------ offline real evidence


def loo(lessons: list[Any], max_rows: int = 200) -> dict[str, Any]:
    """Leave-one-out on REAL rows only: chooser (trained on the other real rows) vs the lexical default, on adopted decisions."""
    from creator import action_student as A
    from creator import chooser as CH
    rows = CH.lesson_rows(lessons)[-max_rows:]
    pos = [i for i, r in enumerate(rows) if r.sign == 1]
    kc = kd = 0
    for i in pos:
        r = rows[i]
        rest = rows[:i] + rows[i + 1:]
        try:
            ch = CH.Chooser().fit(rest) if rest else CH.Chooser()
            cp = ch.pick(r.objective, r.cands, {c.path: r.src for c in r.cands})
        except Exception as e:                               # noqa: BLE001
            log.warning("loo chooser failed: %s", e)
            cp = None
        dp = A.lexical_pick(r.objective, r.cands)
        kc += cp is not None and (cp - 1) in r.chosen
        kd += dp is not None and (dp - 1) in r.chosen
    return {"n": len(pos), "chooser_hits": int(kc), "default_hits": int(kd)}


def stats(state: Path, lessons: list[Any]) -> dict[str, Any]:
    st = loo(lessons)
    res = resolve(state, lessons)
    n = st["n"]
    cl, cu = wilson(st["chooser_hits"], n)
    dl, du = wilson(st["default_hits"], n)
    diff = [r for r in res if r.get("default_pick") != r.get("chooser_pick") and r.get("chooser_pick") is not None]
    c_adopt = sum(1 for r in diff if r["adopted"] and r.get("applied") == "chooser")
    d_adopt = sum(1 for r in diff if r["adopted"] and r.get("applied") != "chooser")
    both = [r for r in res if r.get("chooser_pick") is not None]
    return {**st, "chooser_acc": st["chooser_hits"] / n if n else None, "default_acc": st["default_hits"] / n if n else None,
            "chooser_ci": [round(cl, 3), round(cu, 3)], "default_ci": [round(dl, 3), round(du, 3)],
            "shadow_resolved": len(res), "agreement": (sum(1 for r in both if r["default_pick"] == r["chooser_pick"]) / len(both)) if both else None,
            "disagree_tested": c_adopt + d_adopt, "disagree_chooser_adopted": c_adopt, "disagree_default_adopted": d_adopt}


def decide(s: dict[str, Any], use_chooser: bool) -> Optional[bool]:
    """The pre-registered rule: the new value of use_chooser, or None for no change."""
    n = int(s["n"])
    if n < MIN_N:
        return None
    if not use_chooser:
        a = wilson(int(s["chooser_hits"]), n)[0] >= int(s["default_hits"]) / n
        t = int(s["disagree_tested"])
        b = t >= MIN_TESTED and wilson(int(s["disagree_chooser_adopted"]), t)[0] > 0.5
        return True if (a or b) else None
    return False if int(s["chooser_hits"]) / n < wilson(int(s["default_hits"]), n)[0] else None


def update_policy(state: Path = STATE, lessons_path: Optional[Path] = None, current_default: bool = False) -> dict[str, Any]:
    """Recompute the evidence, apply the rule, persist a flip. Returns the stats plus the current policy."""
    from creator.curriculum import LessonLog
    state = Path(state)
    s = stats(state, LessonLog(lessons_path or state / "lessons.jsonl").lessons())
    cur = read_policy(state)
    cur = current_default if cur is None else cur
    new = decide(s, cur)
    if new is not None and new != cur:
        _write_policy(new, s, state)
        cur = new
    return {**s, "use_chooser": cur}
