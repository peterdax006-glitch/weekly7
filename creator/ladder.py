"""P1.4 stuck detector, fallback ladder and parking (split out of creator.gaps, which re-exports these names lazily so the start-time
load stays small)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

# ------------------------------------------------------------------------------------------------ P1.4 stuck detector + fallback ladder + parking
# MASTER_BLUEPRINT 7.8: "never stuck". A goal with no measured progress for STUCK_CYCLES cycles (or RUNG_MAX_S seconds on one rung) climbs
# the ladder: smaller step -> different approach -> bigger rung -> PARK with its reason (re-tried when a dependency changes) -> a GPU/data
# job is queued when the gap is a skill/data gap -> an owner digest entry. A hard stall limit (STALL_MAX_S) parks the goal whatever rung it
# is on, so no goal stalls an hour. State is one JSON file (state/ladder.json) next to the parking lot; standard library only.
STUCK_CYCLES = 3
RUNG_MAX_S = 800.0                    # 4 phases (start + 3 rungs) of this stay under STALL_MAX_S
STALL_MAX_S = 3600.0
RUNGS = ("start", "smaller_step", "different_approach", "bigger_rung")
LADDER_FILE = "ladder.json"
JOBS_FILE = "jobs_queue.jsonl"        # GPU / data jobs wanted by parked goals (read by the GPU operator and the placement wishlist)
DIGEST_FILE = "digest_queue.jsonl"    # owner digest entries (scripts/nupen_digest.py may list them)
DATA_KINDS = ("skill", "data")        # a parked gap of these kinds also queues a job


def _lad_load(state: Path) -> dict[str, Any]:
    try:
        x = json.loads((Path(state) / LADDER_FILE).read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except (OSError, ValueError):
        return {}


def _lad_save(state: Path, d: Mapping[str, Any]) -> None:
    f = Path(state) / LADDER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(f)


def _lad_append(state: Path, name: str, row: Mapping[str, Any]) -> None:
    f = Path(state) / name
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def ladder_note(state: Path, goal: str, value: Optional[float], now: float, *, kind: str = "", deps: Optional[Mapping[str, str]] = None,
                eps: float = 1e-9) -> dict[str, Any]:
    """One finished cycle of `goal`: `value` is its measured progress metric (higher = better; None = no measurement), `deps` the
    fingerprints of what a parked goal waits on. Returns {'action': continue | smaller_step | different_approach | bigger_rung | park |
    parked, 'rung', 'reason'}. Progress = value above the best so far; it resets the rung counters (the ladder is for the stuck only)."""
    st = _lad_load(state)
    g = st.setdefault(goal, {"rung": 0, "rung_since": now, "no_progress": 0, "best": None, "progress_at": now, "status": "active"})
    if g["status"] == "parked":
        return {"action": "parked", "rung": g["rung"], "reason": g.get("park", {}).get("reason", "")}
    if value is not None and (g["best"] is None or value > g["best"] + eps):
        g.update(best=value, no_progress=0, rung=0, rung_since=now, progress_at=now)
        _lad_save(state, st)
        return {"action": "continue", "rung": 0, "reason": "progress"}
    g["no_progress"] += 1
    stalled = now - g["progress_at"] >= STALL_MAX_S
    stuck = g["no_progress"] >= STUCK_CYCLES or now - g["rung_since"] >= RUNG_MAX_S
    if not stuck and not stalled:
        _lad_save(state, st)
        return {"action": "continue", "rung": g["rung"], "reason": f"{g['no_progress']}/{STUCK_CYCLES} cycles without progress"}
    if g["rung"] < len(RUNGS) - 1 and not stalled:
        g.update(rung=g["rung"] + 1, rung_since=now, no_progress=0)
        _lad_save(state, st)
        return {"action": RUNGS[g["rung"]], "rung": g["rung"], "reason": f"no measured progress; climbing to {RUNGS[g['rung']]}"}
    why = (f"no measured progress for {(now - g['progress_at']) / 60:.0f} min and the ladder ({', '.join(RUNGS[1:])}) "
           f"{'was cut short by the stall limit' if stalled and g['rung'] < len(RUNGS) - 1 else 'is used up'}")
    g.update(status="parked", park={"reason": why, "at": now, "deps": dict(deps or {}), "kind": kind})
    _lad_save(state, st)
    jobs = None
    if kind in DATA_KINDS:
        jobs = {"at": now, "goal": goal, "job": "gpu_module" if kind == "skill" else "data_job", "why": why}
        _lad_append(state, JOBS_FILE, jobs)
    _lad_append(state, DIGEST_FILE, {"at": now, "goal": goal, "kind": "parked", "reason": why, "retry": "when a dependency changes",
                                     "job_queued": bool(jobs)})
    return {"action": "park", "rung": g["rung"], "reason": why, "job": jobs}


def ladder_due(state: Path, fingerprints: Mapping[str, str], now: float, retry_after_s: float = 86400.0) -> list[str]:
    """Parked goals to re-try now: a dependency fingerprint changed (or, with no dependency recorded, `retry_after_s` passed). Each is
    unparked with fresh counters (rung 0)."""
    st = _lad_load(state)
    due: list[str] = []
    for goal, g in st.items():
        if g.get("status") != "parked":
            continue
        p = g.get("park", {})
        deps = p.get("deps") or {}
        changed = [k for k, v in deps.items() if fingerprints.get(k, v) != v]
        if changed or (not deps and now - float(p.get("at", now)) >= retry_after_s):
            g.update(status="active", rung=0, rung_since=now, no_progress=0, progress_at=now, retried=changed or ["timer"])
            due.append(goal)
    if due:
        _lad_save(state, st)
        for goal in due:
            _lad_append(state, DIGEST_FILE, {"at": now, "goal": goal, "kind": "retried", "reason": f"dependency changed: {st[goal]['retried']}"})
    return due


def ladder_rung(state: Path, goal: str) -> str:
    return RUNGS[int(_lad_load(state).get(goal, {}).get("rung", 0))]
