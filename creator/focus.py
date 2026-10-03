"""FOCUS MODE (owner, 2 Oct 2026: "we want to prioritize thinking right now, have Nupen work solely on thinking until its opinion is trust
worthy"). state/creator/focus.json {"focus": "thinking"} is owner/teacher-owned: it is a PROTECTED path (creator/sandbox.py), so no work
package of Nupen's can write it. While the focus is "thinking", efficiency work (shrink / coverage / start-load / activation) is NOT planned
and compute goes to the thinking drills (creator.thinking) and to development gaps of THINKING capabilities only.

HOOK POINT for the planner and the schedule ranking (owned by F2): filter candidates with `allowed(item, state)` before ranking, e.g.
    focus = registry.get("focus"); gaps = [g for g in gaps if focus.allowed(g, state_dir)]
`item` may be a mapping (gap / package / plan row) or any object; the fields read are kind / task_kind, requirement / req / key, component,
modules / outputs and text (objective / why / description). With no focus.json (or any other focus value) everything is allowed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

THINKING = "thinking"
EFFICIENCY_KINDS = frozenset({"shrink", "efficiency", "coverage", "tests", "start_load", "startload", "activation", "size"})
EFFICIENCY_REQ_PREFIXES = ("EFF.",)
DRILL_KINDS = frozenset({"drill", "thinking", "thinking_drill"})
# capabilities (creator/capabilities.json ids) and modules whose development IS thinking: self-model, planner, meta-learning, autotune,
# oversight, goal generation; plus the modules named by the owner's list (reasoning, constraints, recon, chooser, shadow) and the drills.
THINKING_COMPONENTS = frozenset({"K02", "K09", "K13", "K19", "K23", "K27", "K29"})   # K29: every attempt reaches a verdict - the drills' ground truth
THINKING_MODULES = frozenset({"creator/reasoning.py", "creator/goals.py", "creator/constraints.py", "creator/selfmodel.py", "creator/recon.py",
                              "creator/oversight.py", "creator/autotune.py", "creator/chooser.py", "creator/shadow.py", "creator/thinking.py",
                              "creator/focus.py", "creator/planner.py", "creator/meta.py", "scripts/nupen_blueprint.py"})


THINKING_MODULE_PREFIX = "creator/goal_think_"     # goals proposed by the learning loop (creator.learnloop) for a diagnosed thinking weakness
APPROVED_FILE = Path(__file__).resolve().parent / "capabilities_approved.json"


def approved_thinking_components(path: Path = APPROVED_FILE) -> frozenset[str]:
    """Approved capabilities (K-ids) whose module is a learn-loop thinking goal: their gaps are thinking work like THINKING_COMPONENTS."""
    try:
        caps = json.loads(path.read_text(encoding="utf-8")).get("capabilities", [])
    except (OSError, ValueError, AttributeError):
        return frozenset()
    return frozenset(str(c.get("id")) for c in caps if isinstance(c, dict)
                     and any(str(m).replace("\\", "/").startswith(THINKING_MODULE_PREFIX) for m in c.get("modules", [])))


def focus_path(state: Path) -> Path:
    return Path(state) / "focus.json"


def current(state: Union[str, Path]) -> str:
    """The focus word in state/creator/focus.json, or "" (no focus) when the file is absent, unreadable or not an object with a string."""
    try:
        d = json.loads(focus_path(Path(state)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    f = d.get("focus") if isinstance(d, dict) else None
    return f.strip().lower() if isinstance(f, str) else ""


def _get(item: Any, *names: str) -> Any:
    for n in names:
        v = item.get(n) if isinstance(item, Mapping) else getattr(item, n, None)
        if v not in (None, "", [], ()):
            return v
    return None


def is_efficiency(item: Any) -> bool:
    kind = str(_get(item, "kind", "task_kind", "step") or "").lower()
    req = str(_get(item, "requirement_key", "requirement", "req", "key") or "")   # kernel plans carry requirement_key
    return kind in EFFICIENCY_KINDS or req.startswith(EFFICIENCY_REQ_PREFIXES) or req.lower().endswith((".size", ".shrink"))


def is_thinking_work(item: Any) -> bool:
    if str(_get(item, "kind", "task_kind") or "").lower() in DRILL_KINDS:
        return True
    comp = str(_get(item, "component") or "").split(".")[0]
    mods = _get(item, "modules", "outputs", "files") or []
    mods = [mods] if isinstance(mods, str) else list(mods)
    req = str(_get(item, "requirement_key", "requirement", "req", "key") or "").split(".")[0]
    if comp in THINKING_COMPONENTS or req in THINKING_COMPONENTS or comp.startswith(THINKING_MODULE_PREFIX):
        return True
    if any(str(m).replace("\\", "/") in THINKING_MODULES or str(m).replace("\\", "/").startswith(THINKING_MODULE_PREFIX) for m in mods):
        return True
    learned = approved_thinking_components()
    return bool(learned) and (comp in learned or req in learned)


DEFAULT_STATE = Path(__file__).resolve().parents[1] / "state" / "creator"


def allowed(item: Any, state: Union[str, Path, None] = None) -> bool:
    """May this plan / gap be planned now? Always True outside thinking focus. In thinking focus: never efficiency work; development work only
    when it targets a thinking capability; drill work always."""
    if current(DEFAULT_STATE if state is None else state) != THINKING:   # swarm hook calls allowed(plan)
        return True
    if is_efficiency(item):
        return False
    return is_thinking_work(item)


def burst_gb(state: Union[str, Path]) -> Optional[float]:
    """3 Oct 2026: in thinking focus the fillers are short drills and judgment batches, and the 15 s start spacing kept ONE running
    (CPU 29-65%, 19 GB free). They may burst inside the spacing window while free RAM covers each start at 3 GB (above a thinking
    model server, ~2.6 GB). Any other filler (the heavy self-bench of 1 Oct run9) keeps the plain spacing (None)."""
    return 3.0 if current(state) == THINKING else None
