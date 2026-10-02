"""Creator K03 - objectives and the requirement compiler (C77 secs 2, 12, 13; package CR07) - IMPLEMENTED, NOT VALIDATED.

An objective is a structured problem (Objective record). The compiler turns it into Requirements, each of which names a COMPUTED
acceptance check - `check:<name>:<capability>` - that creator.gaps evaluates against the evidence-derived self-model. No requirement
can be satisfied by a statement; only by its check returning True on the current source.

The first autonomous target is the system itself (C77 sec 2): `self_objective()` + `compile_capabilities()` register one
Capability per declared component (creator/capabilities.json) and, for each, the ladder of requirements

    <K>.exists      every implementing module exists and parses                              (CAPABILITY gap when unmet)
    <K>.tested      every declared test file passed on the current source                    (TESTING gap)
    <K>.no_stubs    no public stub (pass-only / NotImplementedError / ellipsis bodies)       (CAPABILITY gap)
    <K>.depth       meaningful lines >= the declared floor (the repository's one ruler)       (ARCHITECTURE gap)
    <K>.integrated  imported by something other than its own tests (reachable, not orphaned)  (INTEGRATION gap)
    <K>.validated   moved to VALIDATED in the ledger by an independent role with evidence     (EVIDENCE gap)

with dependencies (tested <- exists; validated <- tested, depth, no_stubs, integrated) and across components the build order of
creator/ARCHITECTURE.md. Owners may add objectives in a small structured text format (`parse_objective`). Registration is
idempotent: a requirement key is unique in the ledger, so compiling twice changes nothing."""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import model as M
from creator import selfmodel as SM
from creator.ledger import Ledger

SELF_STATEMENT = ("Develop the Creator itself (C77 sec 2): every declared component implemented, tested on its current source, "
                  "free of stubs, at its depth floor, integrated, and validated by an independent role.")

LADDER: tuple[tuple[str, M.GapKind, float, str], ...] = (
    ("exists", M.GapKind.CAPABILITY, 1.00, "every implementing module exists and parses"),
    ("tested", M.GapKind.TESTING, 0.90, "every declared test file passed on the current source"),
    ("no_stubs", M.GapKind.CAPABILITY, 0.60, "no public stub bodies remain"),
    ("integrated", M.GapKind.INTEGRATION, 0.55, "imported by production code, not only by its tests"),
    ("validated", M.GapKind.EVIDENCE, 0.40, "VALIDATED in the ledger by an independent role with evidence"),
)
STEP_DEPENDS = {"tested": ("exists",), "no_stubs": ("exists",), "integrated": ("exists",),
                "validated": ("tested", "no_stubs", "integrated")}
# Owner, 1 Oct 2026 ("constantly develop itself to shrink its own code"; ruling "capability, not lines"): there is no line-depth
# step any more - depth is behaviour proven by tests, and size is pushed DOWN by the standing efficiency objective (K17).
SUPERSEDED_STEPS = ("depth",)
# ARCHITECTURE build order: a component's `exists` depends on the components it is built on
COMPONENT_DEPENDS = {"K02": ("K01",), "K03": ("K01", "K02"), "K04": ("K02", "K03"), "K05": ("K01",), "K06": ("K01",),
                     "K07": ("K05",), "K08": ("K05", "K07"), "K09": ("K04",), "K10": ("K06", "K15"), "K11": ("K06", "K05"),
                     "K12": ("K01",), "K13": ("K12", "K10"), "K14": ("K09", "K10", "K06", "K05"), "K15": ("K06",),
                     "K16": ("K01", "K10"),
                     # K17-K26, from what each module imports
                     "K17": ("K06",), "K18": ("K15", "K25"), "K19": ("K03", "K10", "K15", "K18"),
                     "K20": ("K09", "K14", "K18"), "K21": ("K09", "K14", "K18", "K20"),
                     "K22": ("K01", "K06", "K14", "K15", "K18"), "K23": ("K01", "K03", "K04"), "K24": ("K01", "K13"),
                     "K25": ("K09", "K14"), "K26": ("K01",), "K27": ("K01", "K03", "K24")}
PRIORITY_WEIGHT = {M.Priority.CRITICAL: 1.0, M.Priority.HIGH: 0.75, M.Priority.MEDIUM: 0.5, M.Priority.LOW: 0.25}
CRITICAL_COMPONENTS = frozenset({"K01", "K06", "K10", "K14", "K15", "K16"})       # integrity, measurement and the loop itself


class ObjectiveError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ computed acceptance checks

CheckFn = Callable[[SM.SelfModel, SM.CapabilityState, Optional[Ledger], Optional[str]], tuple[bool, str]]


def _exists(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    bad = [m for m in c.present_modules if model.components[m].parse_error]
    if c.missing_modules:
        return False, f"missing: {', '.join(c.missing_modules)}"
    if bad:
        return False, f"unparsable: {', '.join(bad)}"
    return True, f"{len(c.present_modules)} module(s) present"


def _tested(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    if c.state == "TESTED":
        return True, "every test file passed on the current source"
    return False, f"state {c.state}: " + (", ".join(f"{k}={v}" for k, v in c.last_results.items()) or "no test files")


def _no_stubs(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    if not c.present_modules:
        return False, "nothing to inspect"
    return (not c.stubs), (f"{len(c.stubs)} stub(s): {'; '.join(c.stubs[:3])}" if c.stubs else "no stubs")


def _depth(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    return (bool(c.present_modules) and not c.below_floor), f"{c.meaningful} meaningful lines, floor {c.floor}"


def _integrated(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    if not c.present_modules:
        return False, "nothing to integrate"
    orphans = [m for m in c.present_modules
               if not any(not d.startswith("tests/") and d not in c.present_modules for d in model.dependents.get(m, ()))]
    return (not orphans), (f"not imported by production code: {', '.join(orphans)}" if orphans else "imported by production code")


def _validated(model: SM.SelfModel, c: SM.CapabilityState, led: Optional[Ledger], cap_id: Optional[str]) -> tuple[bool, str]:
    if led is None or cap_id is None:
        return False, "no ledger"
    st = led.view.status.get(cap_id)
    return st is M.Status.VALIDATED, f"ledger state {st.value if st else 'none'}"


CHECKS: dict[str, CheckFn] = {"exists": _exists, "tested": _tested, "no_stubs": _no_stubs, "depth": _depth,
                              "integrated": _integrated, "validated": _validated}
CHECK_RE = re.compile(r"^check:(?P<name>[a-z_]+):(?P<cap>K\d{2})$")


def parse_check(acceptance_test: str) -> tuple[str, str]:
    m = CHECK_RE.match(acceptance_test)
    if not m or m.group("name") not in CHECKS:
        raise ObjectiveError(f"not a computed acceptance check: {acceptance_test!r}")
    return m.group("name"), m.group("cap")


# ------------------------------------------------------------------------------------------------ objectives

@dataclasses.dataclass(frozen=True)
class ObjectiveSpec:
    statement: str
    outcomes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()


SECTIONS = {"statement", "outcomes", "constraints", "non_goals", "acceptance_criteria", "metrics", "risks", "unknowns"}


def parse_objective(text: str) -> ObjectiveSpec:
    """A small structured format: `Statement: ...` then sections `Outcomes:` etc. with `- item` lines. Unknown sections and
    items outside a section are refused (an objective that cannot be parsed exactly is not guessed at)."""
    fields: dict[str, list[str]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z _]+):\s*(.*)$", line)
        if m and m.group(1).strip().lower().replace(" ", "_") in SECTIONS and not line.startswith("-"):
            current = m.group(1).strip().lower().replace(" ", "_")
            fields.setdefault(current, [])
            if m.group(2):
                fields[current].append(m.group(2).strip())
        elif line.startswith("- ") and current is not None:
            fields[current].append(line[2:].strip())
        else:
            raise ObjectiveError(f"cannot place line: {line!r}")
    stmt = " ".join(fields.get("statement", [])).strip()
    if not stmt:
        raise ObjectiveError("an objective needs a Statement")
    if not fields.get("acceptance_criteria"):
        raise ObjectiveError("an objective needs at least one acceptance criterion (C77 sec 12)")
    return ObjectiveSpec(stmt, *(tuple(fields.get(k, ())) for k in ("outcomes", "constraints", "non_goals",
                                                                     "acceptance_criteria", "metrics", "risks", "unknowns")))


def register_objective(ledger: Ledger, spec: ObjectiveSpec, created_by: M.Role = M.Role.OWNER) -> str:
    """Append the objective unless an identical statement is already recorded (idempotent)."""
    for e in ledger.of_type("Objective"):
        if getattr(e.record, "statement", None) == spec.statement:
            return e.id
    return ledger.append(M.Objective(created_by=created_by, **dataclasses.asdict(spec)))


OWNER_GOAL = (
    "the whole idea is to create an ai that can think for itself, the goal is to make you, claude, to not be neccessarry at all "
    "as soon as possible where you hardly even need to moniter it, you just need to design the role you are taking so then it can "
    "use you less and less to become its own ai essentially / the goal is to make it so it is fully independent of any other ai "
    "where it just permanently develops itself forever one day / it should constantly develop itself to shrink its own code down "
    "so that it can run more code at a time and run on less memory")                     # owner, 1 Oct 2026, verbatim


def goal_objective(ledger: Ledger) -> str:
    """The TOP objective, in the owner's own words: everything else (the component ladder, shrinking) serves it."""
    return register_objective(ledger, ObjectiveSpec(
        statement=OWNER_GOAL,
        outcomes=("the Creator's own worker writes the code that develops the Creator",
                  "no other AI is called: the worker runs locally on weights the Creator owns and can retrain",
                  "Claude only directs, then is not needed at all", "it keeps developing itself, permanently"),
        constraints=("every change measured and reversible", "the sealed holdout is never learned from",
                     "no fake autonomy: a change counts as the Creator's only if its own worker wrote it"),
        non_goals=("looking autonomous without being measured as autonomous",),
        acceptance_criteria=("own-worker solve rate on the sealed devbench holdout rises over time",
                             "share of adopted changes written by claude-session falls to 0"),
        metrics=("self_share = adopted changes made by the system's own workers / all adopted (owner: it works on itself "
                 "more than a third party works on it)", "own_worker_holdout_solve_rate",
                 "claude_dependence = adopted changes by claude-session / all adopted",
                 "creator package AST nodes", "peak memory"),
        risks=("a small local model may plateau; the benchmark must grow to show real gains",),
        unknowns=("how far a laptop-sized model can be improved by the Creator's own learning",)), created_by=M.Role.OWNER)


def claude_dependence(ledger: Ledger) -> dict[str, Any]:
    """Share of ADOPTED changes whose worker was the Claude session (or a Claude agent) - the number that must fall to 0."""
    by: dict[str, int] = {}
    for e in ledger.of_type("StrategyOutcome"):
        if getattr(e.record, "success"):
            sid = str(getattr(e.record, "strategy_id"))
            by[sid] = by.get(sid, 0) + 1
    claude = sum(v for k, v in by.items() if k.startswith(("claude", "agent-")))
    own = sum(v for k, v in by.items() if k.startswith("self-"))
    total = sum(by.values())
    return {"adopted_by_worker": by, "claude_share": round(claude / total, 3) if total else None,
            "self_share": round(own / total, 3) if total else None, "adopted": total}


def self_objective(ledger: Ledger) -> str:
    return register_objective(ledger, ObjectiveSpec(
        statement=SELF_STATEMENT,
        outcomes=("the Creator can develop itself measurably", "every claim about its state is computed from evidence"),
        constraints=("no fake autonomy (sec 68)", "no hard-coded demonstrations (sec 50)", "protected paths never modified",
                     "agent calls budgeted"),
        non_goals=("trading performance", "manual solutions by the foundation builder (sec 67)"),
        acceptance_criteria=("every requirement's computed check passes on the current source",),
        metrics=("requirements met", "devbench solve rate", "open gaps"),
        risks=("measurement on a small task suite is noisy", "contamination of sealed answer keys"),
        unknowns=("whether LLM workers can improve the Creator's own code measurably",)), created_by=M.Role.OWNER)


# ------------------------------------------------------------------------------------------------ compiling capabilities

@dataclasses.dataclass(frozen=True)
class Compiled:
    objective_id: str
    capability_ids: Mapping[str, str]                   # K-id -> Capability record id
    requirement_ids: Mapping[str, str]                  # requirement key -> record id
    created: int


def _find_capability(ledger: Ledger, component: str) -> Optional[str]:
    for e in ledger.of_type("Capability"):
        if getattr(e.record, "component", None) == component:
            return e.id
    return None


def _build_order(specs: Sequence[SM.CapabilitySpec]) -> list[SM.CapabilitySpec]:
    """Specs with every component after the components it is built on (COMPONENT_DEPENDS), ties by id. K10 is built on K15, so id
    order alone silently dropped that dependency."""
    by_id = {s.id: s for s in specs}
    out: list[SM.CapabilitySpec] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen:
            return
        seen.add(cid)
        for d in COMPONENT_DEPENDS.get(cid, ()):
            if d in by_id:
                visit(d)
        out.append(by_id[cid])
    for cid in sorted(by_id):
        visit(cid)
    return out


def compile_capabilities(ledger: Ledger, objective_id: str, specs: Sequence[SM.CapabilitySpec],
                         created_by: M.Role = M.Role.KERNEL) -> Compiled:
    """Register a Capability and the requirement ladder for every declared component. Idempotent and dependency-ordered."""
    created = 0
    caps: dict[str, str] = {}
    reqs: dict[str, str] = {}
    order = sorted(specs, key=lambda s: s.id)
    for spec in order:
        cid = _find_capability(ledger, spec.id)
        if cid is None:
            cid = ledger.append(M.Capability(created_by=created_by, name=spec.name, component=spec.id,
                                             description=f"{spec.name}: modules {', '.join(spec.modules)}; tests "
                                                         f"{', '.join(spec.tests) or 'none'}; floor {spec.floor}"))
            created += 1
        caps[spec.id] = cid
    for spec in _build_order(order):
        prio = M.Priority.CRITICAL if spec.id in CRITICAL_COMPONENTS else M.Priority.HIGH
        for step, _kind, _w, desc in LADDER:
            key = f"{spec.id}.{step}"
            existing = ledger.view.unique.get(("Requirement", key))
            if existing:
                reqs[key] = existing
                continue
            deps = [reqs[f"{spec.id}.{d}"] for d in STEP_DEPENDS.get(step, ())]
            if step == "exists":
                deps += [reqs[f"{k}.tested"] for k in COMPONENT_DEPENDS.get(spec.id, ()) if f"{k}.tested" in reqs]
            reqs[key] = ledger.append(M.Requirement(
                created_by=created_by, parents=(objective_id,), key=key, description=f"{spec.name}: {desc}", priority=prio,
                acceptance_test=f"check:{step}:{spec.id}", measurement_method="creator.selfmodel.build + creator.objective.CHECKS",
                failure_condition=f"check:{step}:{spec.id} returns False on the current source",
                validation_method="recomputed by creator.gaps.assess on every kernel cycle",
                evidence_location="state/creator/selfmodel.json", depends_on=tuple(deps)))
            created += 1
    return Compiled(objective_id, caps, reqs, created)


def importance(priority: M.Priority, step: str, component: str) -> float:
    """Gap importance in [0, 1]: requirement priority x ladder weight x an earlier-in-build-order bonus."""
    w = next(w for s, _k, w, _d in LADDER if s == step)
    order = int(component[1:]) if component[1:].isdigit() else 16
    return round(PRIORITY_WEIGHT[priority] * w * (1.0 - 0.02 * (order - 1)), 4)


def requirement_view(ledger: Ledger) -> list[dict[str, Any]]:
    out = []
    for e in ledger.of_type("Requirement"):
        r = e.record
        out.append({"id": e.id, "key": getattr(r, "key"), "status": ledger.view.status.get(e.id, M.Status.NOT_STARTED).value,
                    "acceptance_test": getattr(r, "acceptance_test"), "depends_on": list(getattr(r, "depends_on"))})
    return out
