"""Creator sparse activation (owner, 2 Oct 2026: "format it in a way so you only ever run the code you are using to be RAM efficient so you
can have billions of lines of code one day"; 1 Oct: "it can have a billion lines of code but it isnt using its entire capablity 24/7").

ARCHITECTURE RULE: only the kernel core is imported at module level. Every other capability is found BY NAME here and imported on first
use, so what a process loads is what it uses, and the code base can grow without the start-time load growing. The registry is data (a name
-> module path table) and imports nothing from the Creator itself. The kernel's start-load guard (efficiency.start_load, kernel.cycle)
rejects an adopted change that grows what the kernel or the swarm loads at start unless it declares and justifies an eager dependency."""
from __future__ import annotations

import importlib
from typing import Any

# capability name -> "module" (the module object) or "module:attribute" (one object from it)
CAPABILITIES: dict[str, str] = {
    "students": "creator.student",
    "lesson_student": "creator.student:LessonStudent",
    "replay_student": "creator.replay_student:ReplayStudent",
    "resume_student": "creator.pending:ResumeStudent",
    "model_student": "creator.model_student:ModelStudent",
    "action_student": "creator.action_student:ActionStudent",
    "curriculum": "creator.curriculum",
    "goals": "creator.goals",
    "constraints": "creator.constraints",
    "doctrine": "creator.doctrine",
    "lm": "creator.lm",
    "process_levers": "creator.process_levers",
    "selfworkers": "creator.selfworkers",
    "recursion": "creator.recursion",
    "oversight": "creator.oversight",
    "debug": "creator.debug",
    "efficiency": "creator.efficiency",
    "swarm": "creator.swarm",
    "schedule": "creator.schedule",
    "reasoning": "creator.reasoning",
    "device": "creator.device",
    "tools": "creator.tools.toolbox",
}

_loaded: dict[str, Any] = {}


class CapabilityError(ImportError):
    """The name is not registered, or its module is missing: callers that treat the capability as optional catch ImportError."""


def register(name: str, target: str) -> None:
    """Add (or repoint) a capability; a new module is reachable by name without any module importing it."""
    CAPABILITIES[name] = target
    _loaded.pop(name, None)


def names() -> list[str]:
    return sorted(CAPABILITIES)


def loaded() -> list[str]:
    """Capabilities this process has actually imported (the activation record)."""
    return sorted(_loaded)


def get(name: str) -> Any:
    """The capability's module or attribute, imported on first use and cached."""
    if name in _loaded:
        return _loaded[name]
    target = CAPABILITIES.get(name)
    if target is None:
        raise CapabilityError(f"unknown capability {name!r}")
    mod, _, attr = target.partition(":")
    obj: Any = importlib.import_module(mod)
    if attr:
        try:
            obj = getattr(obj, attr)
        except AttributeError as e:
            raise CapabilityError(f"{target}: {e}") from e
    _loaded[name] = obj
    return obj


def optional(name: str) -> Any:
    """Like get, but None when the capability is unregistered or its module does not exist (an absent optional never crashes)."""
    try:
        return get(name)
    except ImportError:
        return None


def sparse_rule_text() -> str:
    """The rule every work package carries (planner fundamentals)."""
    return ("Sparse activation: new modules and optional capabilities are loaded on demand through creator.registry (registry.get(name) "
            "imports on first use); module-level imports only for the core. A change that raises the kernel's or the swarm's start-time "
            "load (what `import creator.kernel` and scripts/creator_swarm.py load eagerly) is rejected unless the package's objective "
            "is an eager dependency with a recorded justification.")
