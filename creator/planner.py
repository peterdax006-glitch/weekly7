"""Creator K09 - the planner: gaps -> work packages -> change proposals -> experiments (C77 secs 17-19, 63; package CR09)
- IMPLEMENTED, NOT VALIDATED.

`plan_next()` takes the highest-ranked UNBLOCKED open gap (creator.gaps.ranked) and writes, in one dependency chain:

    WorkPackage    every C77 sec 63 field, derived from the gap's requirement step (exists / tested / no_stubs / depth /
                   integrated / validated) and the component's declaration; completion criteria are the COMPUTED check plus a
                   clean audit and no regressions - never "the worker says done"
    ChangeProposal what will change, why, where, and the expected effect (the subject an ADOPT decision will rest on)
    Experiment     hypothesis, design, metrics, baseline (the current main commit) and candidate (the sandbox branch)

and moves the gap to IN_PROGRESS. Some gaps are not implementer work: a `validated` gap is routed to the VALIDATOR role (C77 sec 48),
never to a worker that could self-certify. A gap whose packages failed `max_attempts` times is moved to BLOCKED with the failure
reasons, so the loop does not grind on one problem forever (C77 sec 64 anti-stall); `unblock()` reopens it when something it
depends on changes. Package ids are sequential (CP0001...), so a plan's order is reproducible."""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

from creator import design as DS
from creator import gaps as G
from creator import memory as MEM
from creator import model as M
from creator import objective as O
from creator import selfmodel as SM
from creator.ledger import Ledger

PREFIX = "CP"
MAX_ATTEMPTS = 3
DEFERRED_PREFIX = "deferred:"
WORKER_STEPS = ("exists", "tested", "no_stubs", "integrated")
EFFICIENCY_KEY = "EFF.size"
ACTIVATION_KEY = "EFF.activation"
VALIDATOR_STEPS = ("validated",)


class PlanningError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Plan:
    gap_id: str
    requirement_key: str
    component: str
    step: str
    role: M.Role
    work_package_id: str
    package_id: str
    change_proposal_id: str
    experiment_id: str
    attempt: int


# ------------------------------------------------------------------------------------------------ package content per step

STEP_TEXT: dict[str, dict[str, Any]] = {
    "exists": {
        "do": ("create {modules} implementing {name} as declared in creator/ARCHITECTURE.md",
               "public functions and classes carry docstrings stating their contract",
               "no stub bodies: every public function does real work"),
        "tests": ("create {tests} exercising the new module's contract, including its empty and failure cases",),
        "effect": "the module(s) of {cid} exist, parse, and are covered by passing tests",
        "fail": ("module created but empty or stub-only", "tests that assert nothing", "imports that only work from the repo root"),
    },
    "tested": {
        "do": ("make every test in {tests} pass on the current source by fixing the code under test",
               "if a test is genuinely wrong, change it only with a written reason; never delete or weaken assertions"),
        "tests": ("{tests} passes in full on the current source",),
        "effect": "{cid}'s declared tests pass on the current source",
        "fail": ("tests weakened or skipped to pass", "flaky pass hiding a real failure", "fix in the test instead of the code"),
    },
    "no_stubs": {
        "do": ("replace each public stub in {modules} with a real implementation: {detail}",),
        "tests": ("add tests that would fail against the stub version of every replaced function",),
        "effect": "{cid} has no public stub bodies",
        "fail": ("stub replaced by a trivial return that tests do not check", "NotImplementedError swapped for a silent pass"),
    },
    "depth": {
        "do": ("deepen {modules} where the architecture asks for capability it lacks ({detail}); every new line must be "
               "exercised by a test that fails without it",),
        "tests": ("new behaviour is covered: each added function has at least one test that fails when it is removed",),
        "effect": "{cid} reaches its depth floor with tested behaviour, not padding",
        "fail": ("padding: duplicated code, dead branches or verbose rewrites that add lines without behaviour",
                 "new code with no test that depends on it"),
    },
    "integrated": {
        "do": ("wire {modules} into the production code that should use it ({detail}) so the capability is reachable",),
        "tests": ("an integration test drives the caller and observes the integrated module's effect",),
        "effect": "{cid} is imported and used by production code",
        "fail": ("an unused import added only to satisfy the reachability check", "integration that is never executed"),
    },
    "validated": {
        "do": ("independently validate {cid}: re-run its tests, run the auditor and the adversary, inspect behaviour against "
               "creator/ARCHITECTURE.md, and either transition the Capability to VALIDATED with evidence or record a Failure",),
        "tests": ("the validator's own checks, recorded as evidence",),
        "effect": "{cid} is VALIDATED by an independent role, or its defects are recorded",
        "fail": ("the implementer validating its own work", "validation with no evidence attached"),
    },
}


# ------------------------------------------------------------------------------------------------ K08 / K12 / K07 in the loop

def _approach_options(prior_n: int, fails: Sequence[str]) -> list[DS.DesignOption]:
    """K08: the ways a worker could attack this package. Risk of the plain approach grows with every failed attempt, and
    diagnosing first only pays once there is a failure to diagnose - so the choice changes with the history."""
    return DS.generate_options("approach", [
        {"name": "direct", "summary": "do exactly what the package says, in one pass", "assumptions": ("the instructions suffice",),
         "failure_modes": tuple(fails[:1]) or ("the same mistake as last time",), "cost": 0.3, "risk": min(0.9, 0.25 + 0.25 * prior_n),
         "benefit": 0.5},
        {"name": "test-first", "summary": "write the failing test that proves the requirement first, then the smallest change that passes it",
         "assumptions": ("the requirement can be expressed as a test",), "failure_modes": ("a test that passes without the change",),
         "cost": 0.45, "risk": min(0.9, 0.3 + 0.1 * prior_n), "benefit": 0.65},
        {"name": "diagnose-first", "summary": "reproduce the earlier failure, find its root cause, fix that and nothing else",
         "assumptions": ("the failure reproduces",), "failure_modes": ("a root cause that is a symptom",), "cost": 0.6, "risk": 0.2,
         "benefit": 0.8 if prior_n else 0.2}])


def _history(ledger: Ledger) -> MEM.Memory:
    """K12: what the ledger remembers of past development, by kind - failed packages (failure), diagnosed root causes (repair),
    research findings (research). Rebuilt from the ledger, so it can never disagree with it."""
    mem = MEM.Memory()
    for e in ledger.of_type("WorkPackage"):
        wp = e.record
        for t in ledger.about(e.id):
            if t.rtype == "Transition" and getattr(t.record, "to_state") in (M.Status.FAILED, M.Status.ROLLED_BACK, M.Status.REJECTED):
                mem.remember("failure", getattr(wp, "objective"), getattr(t.record, "reason"), (getattr(wp, "package_id"),))
    for e in ledger.of_type("Diagnosis"):
        fail = ledger.get(getattr(e.record, "failure_id"))
        sid = getattr(fail, "subject_id", "")
        if sid in ledger.view.by_id and ledger.view.by_id[sid].rtype == "WorkPackage":
            wp = ledger.get(sid)
            mem.remember("repair", getattr(wp, "objective"), getattr(e.record, "root_cause"), (getattr(wp, "package_id"),))
    for e in ledger.of_type("Finding"):
        mem.remember("research", getattr(e.record, "statement")[:120], getattr(e.record, "statement"), ("finding",))
    return mem


def seen_before(ledger: Ledger, query: str, skip_packages: Sequence[str] = ()) -> list[str]:
    """Lines for the package text: earlier failures / root causes / findings that resemble this package's objective."""
    skip = {str(getattr(ledger.get(w), "package_id")) for w in skip_packages}
    out = []
    for _, en in _history(ledger).seen_before(query, limit=6, min_score=0.15):
        if skip & set(en.tags):
            continue                                                    # this gap's own attempts are already in `why`
        label = {"failure": "failed before", "repair": "root cause found before", "research": "research found"}.get(en.kind, en.kind)
        out.append(f"{label} ({', '.join(en.tags)}): {en.detail}"[:300])
    return out[:3]


def research_dossier(ledger: Ledger, gap_id: str) -> list[str]:
    """Findings of the research questions opened for this gap when it was blocked (K07 -> the next package)."""
    out = []
    for q in ledger.view.children.get(gap_id, []):
        if ledger.view.by_id[q].rtype == "ResearchQuestion":
            out += [getattr(ledger.get(f), "statement") for f in ledger.view.children.get(q, []) if ledger.view.by_id[f].rtype == "Finding"]
    return out


def research_blocked(ledger: Ledger, gap_id: str, cid: str, step: str, modules: Sequence[str], reasons: Sequence[str]) -> str:
    """K07: a gap that exhausted its attempts becomes a research question; repo evidence is gathered and weighed, and the question,
    the confidence and the findings are recorded (ResearchQuestion + Finding) for the next package. Returns the question id."""
    from pathlib import Path as _P
    from creator import research as RS
    desc = f"{cid} step '{step}' stays unmet after {len(reasons)} failed packages ({'; '.join(reasons)[:300]})"
    q = RS.prioritise(RS.generate_questions([{"id": gap_id, "kind": "KNOWLEDGE", "description": desc, "importance": 0.8}]), budget=1)[0]
    root = getattr(ledger, "evidence_root", None)
    stems = [_P(m).stem for m in modules]
    terms = [t for s in stems for t in (f"import {s}", f"creator.{s}", f"{s}.py")]
    ev = RS.collect_repo_evidence(q.text, _P(root) / "creator", terms, max_hits=6) if root and (_P(root) / "creator").is_dir() else []
    di = RS.to_design_input(q.text, ev)
    rq = ledger.append(M.ResearchQuestion(created_by=M.Role.KERNEL, parents=(gap_id,), question=q.text[:500], priority=M.Priority.HIGH))
    where = "; ".join(di.findings[:3]) or "no repo evidence found for the module names"
    ledger.append(M.Finding(created_by=M.Role.KERNEL, parents=(rq,), uncertainty=M.Uncertainty.UNCERTAIN if di.open else M.Uncertainty.LIKELY,
                            statement=f"{cid} {step}: research confidence {di.confidence:.2f} ({'open' if di.open else 'settled'}); "
                                      f"where it is referenced: {where}", sources=tuple("creator/" + e.source for e in ev[:6])))
    return rq


def _fmt(items: Sequence[str], **kw: str) -> tuple[str, ...]:
    return tuple(s.format(**kw) for s in items)


def next_package_id(ledger: Ledger) -> str:
    n = 0
    for e in ledger.of_type("WorkPackage"):
        pid = str(getattr(e.record, "package_id"))
        if pid.startswith(PREFIX) and pid[len(PREFIX):].isdigit():
            n = max(n, int(pid[len(PREFIX):]))
    return f"{PREFIX}{n + 1:04d}"


def attempts_for(ledger: Ledger, gap_id: str) -> list[str]:
    """Work packages already planned for this gap, oldest first. A package that was only DEFERRED (handed over to a student that
    owns its kind, never attempted) is not an attempt: counting it let three deferrals BLOCK the gap and drop the task."""
    wps = [c for c in ledger.view.children.get(gap_id, []) if ledger.view.by_id[c].rtype == "WorkPackage"]
    return [w for w in wps if not any(t.rtype == "Transition" and str(getattr(t.record, "reason", "")).startswith(DEFERRED_PREFIX)
                                      for t in ledger.about(w))]


def failure_reasons(ledger: Ledger, wp_ids: Sequence[str]) -> list[str]:
    out = []
    for w in wp_ids:
        for t in ledger.about(w):
            if t.rtype == "Transition" and getattr(t.record, "to_state") in (M.Status.FAILED, M.Status.ROLLED_BACK,
                                                                               M.Status.REJECTED):
                out.append(f"{getattr(ledger.get(w), 'package_id')}: {getattr(t.record, 'reason')}")
    return out


def _spec_for(model: SM.SelfModel, cid: str) -> Optional[SM.CapabilityState]:
    return next((c for c in model.capabilities if c.id == cid), None)


def build_package(ledger: Ledger, gap_id: str, model: SM.SelfModel,
                  specs: Mapping[str, SM.CapabilitySpec]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The WorkPackage fields for one gap and its planning metadata (pure: nothing is written)."""
    gap = ledger.get(gap_id)
    req_id = next(p for p in gap.parents if ledger.view.by_id[p].rtype == "Requirement")
    req = ledger.get(req_id)
    step, cid = O.parse_check(getattr(req, "acceptance_test"))
    spec = specs.get(cid)
    state = _spec_for(model, cid)
    detail = getattr(gap, "description").split("unmet: ", 1)[-1]
    if state is not None:                   # the CURRENT check, not the text frozen into the gap when it was opened (1 Oct: a
        cap_rec = next((e.id for e in ledger.of_type("Capability") if getattr(e.record, "component", None) == cid), None)
        detail = O.CHECKS[step](model, state, ledger, cap_rec)[1]        # stale 'nothing to integrate' reached the planner)
    kw = {"cid": cid, "name": spec.name if spec else cid, "modules": ", ".join(spec.modules) if spec else cid,
          "tests": ", ".join(spec.tests) if spec else "its tests", "detail": detail}
    t = STEP_TEXT[step]
    prior = attempts_for(ledger, gap_id)
    why = f"gap {gap_id} ({getattr(req, 'key')}): {detail}"
    if prior:
        why += f"; attempt {len(prior) + 1} after: " + "; ".join(failure_reasons(ledger, prior)[-2:])
    deps = tuple(getattr(ledger.get(d), "package_id") for d in prior[-1:]) if prior else ()
    objective = t["effect"].format(**kw)
    dec = DS.select(f"{cid}:{step}:attempt{len(prior) + 1}", _approach_options(len(prior), failure_reasons(ledger, prior)[-1:]))
    sel = dec.selected
    approach = (f"Approach chosen by design.select ({', '.join(f'{k} {v:.2f}' for k, v in dec.scores.items())}): {sel.name} - "
                f"{sel.summary}; rejected: {'; '.join(f'{o.name} ({r})' for o, r in dec.rejected)}",) if sel else ()
    lessons = tuple(f"Memory - {x}" for x in seen_before(ledger, objective, prior))
    dossier = tuple(f"Research - {x}" for x in research_dossier(ledger, gap_id))
    return dict(
        package_id=next_package_id(ledger), objective=objective, why_it_exists=why,
        prerequisites=tuple(f"open gap {g} resolved" for g in getattr(gap, "depends_on")) or ("none",),
        inputs=(f"self-model {model.digest()}", f"requirement {getattr(req, 'key')}", *(spec.modules if spec else ())),
        outputs=(spec.modules if spec else ()) + (spec.tests if spec else ()),
        implementation_requirements=_fmt(t["do"], **kw) + approach + dossier,
        interfaces=tuple(f"{i.kind} {i.signature}" for m in (state.present_modules if state else ())
                         for i in model.components[m].interfaces[:12]) or ("as declared in creator/ARCHITECTURE.md",),
        data_flow=f"sandbox worktree -> build -> affected tests -> check:{step}:{cid} -> evaluate -> decision",
        dependencies=deps, test_requirements=_fmt(t["tests"], **kw),
        validation_requirements=(f"check:{step}:{cid} passes on the candidate", "audit clean on the candidate",
                                 "no regression in the affected tests versus the base commit"),
        expected_failure_modes=_fmt(t["fail"], **kw) + lessons,
        evidence_requirements=("sandbox build log", "junit of base and candidate", "self-model snapshot of the candidate",
                               "audit report"),
        failure_conditions=(f"check:{step}:{cid} still fails", "any regression", "any audit finding", "a protected path touched"),
        rollback_requirements=("discard the sandbox; main is untouched until an ADOPT decision",
                               "after adoption: revert the merge commit and record ROLLED_BACK"),
        completion_criteria=(f"check:{step}:{cid} computed True on main after adoption", "audit clean",
                             "requirement transitioned to TESTED by creator.gaps.sync"),
        anti_premature_completion=("a worker's STATUS: DONE is a claim, not a result",
                                   "code that exists is not tested; tested is not validated (C77 sec 7)",
                                   "lines added without behaviour do not count"),
        meaningful_code_depth=max(0, spec.floor - (state.meaningful if state else 0)) if spec and step == "depth" else 0,
    ), {"req_id": req_id, "step": step, "cid": cid, "key": getattr(req, "key"), "attempt": len(prior) + 1,
                                                 "decision": dec, "modules": spec.modules if spec else ()}


def record_decision(ledger: Ledger, wp: str, dec: DS.Decision) -> None:
    """K08: every option stays in the ledger (DesignOption under the package); each rejected one gets a REJECT decision with its score."""
    for o, why in [(dec.selected, "")] * (dec.selected is not None) + list(dec.rejected):
        d = ledger.append(M.DesignOption(created_by=M.Role.KERNEL, parents=(wp,), decision_key=dec.question,
                                         summary=f"{o.name}: {o.summary}", assumptions=tuple(o.assumptions),
                                         failure_modes=tuple(o.failure_modes), cost_estimate=o.cost, risk_estimate=o.risk))
        if why:
            ledger.append(M.Decision(created_by=M.Role.KERNEL, subject_id=d, verdict=M.DecisionVerdict.REJECT, reason=why))


def plan_next(ledger: Ledger, model: SM.SelfModel, base_ref: str, specs: Optional[Sequence[SM.CapabilitySpec]] = None,
              max_attempts: int = MAX_ATTEMPTS, steps: Sequence[str] = WORKER_STEPS + VALIDATOR_STEPS,
              exclude_components: Sequence[str] = (), prefer: Sequence[str] = ()) -> Optional[Plan]:
    """Plan the top unblocked gap whose step is in `steps`; gaps named in `prefer` come first (creator.schedule's order).
    Returns None when nothing is plannable."""
    spec_map = {s.id: s for s in (specs if specs is not None else SM.load_capabilities())}
    objective_id = next((e.id for e in ledger.of_type("Objective") if getattr(e.record, "statement") == O.SELF_STATEMENT), None)
    rank = {gid: i for i, gid in enumerate(prefer)}
    for g in sorted(G.ranked(ledger), key=lambda x: rank.get(x.gap_id, len(rank))):     # stable: the rest keep gaps.ranked order
        if g.blocked_by or ledger.view.status.get(g.gap_id) not in (M.Status.NOT_STARTED, M.Status.FAILED):
            continue
        gap = ledger.get(g.gap_id)
        req_id = next((p for p in gap.parents if ledger.view.by_id[p].rtype == "Requirement"), None)
        if req_id is None:
            continue
        try:
            step, cid = O.parse_check(getattr(ledger.get(req_id), "acceptance_test"))
        except O.ObjectiveError:
            continue
        if step not in steps or cid in exclude_components:
            continue                                                    # another worker is on this component right now
        prior = attempts_for(ledger, g.gap_id)
        if len(prior) >= max_attempts:
            reasons = failure_reasons(ledger, prior)[-3:]
            rq = research_blocked(ledger, g.gap_id, cid, step, getattr(spec_map.get(cid), "modules", ()), reasons)
            ledger.transition(g.gap_id, M.Status.BLOCKED,
                              f"{len(prior)} packages failed: " + "; ".join(reasons) + f"; research question {rq} opened", M.Role.KERNEL)
            continue
        fields, meta = build_package(ledger, g.gap_id, model, spec_map)
        wp = ledger.append(M.WorkPackage(created_by=M.Role.KERNEL, parents=(g.gap_id,), **fields))
        record_decision(ledger, wp, meta["decision"])
        cp = ledger.append(M.ChangeProposal(
            created_by=M.Role.KERNEL, parents=(wp,), reason=fields["why_it_exists"],
            parent_objective=objective_id or ledger.of_type("Objective")[0].id, originating_task=wp,
            affected_components=tuple(fields["outputs"]) or (cid,), expected_effect=fields["objective"]))
        ex = ledger.append(M.Experiment(
            created_by=M.Role.KERNEL, parents=(cp,), hypothesis=f"the candidate makes check:{step}:{cid} pass without regressions",
            design="sandbox from the base commit; build; affected tests on base and candidate; recompute the check on the "
                   "candidate's self-model; audit the candidate", metrics=(f"check:{step}:{cid}", "regressions", "audit_findings"),
            seed=0, baseline_ref=base_ref, candidate_ref=f"sandbox:{fields['package_id']}"))
        if ledger.view.status[g.gap_id] is M.Status.FAILED:
            ledger.transition(g.gap_id, M.Status.IN_PROGRESS, f"re-planned as {fields['package_id']}", M.Role.KERNEL)
        else:
            ledger.transition(g.gap_id, M.Status.IN_PROGRESS, f"planned as {fields['package_id']}", M.Role.KERNEL)
        role = M.Role.VALIDATOR if step in VALIDATOR_STEPS else M.Role.IMPLEMENTER
        return Plan(g.gap_id, meta["key"], cid, step, role, wp, fields["package_id"], cp, ex, meta["attempt"])
    return None


EFFICIENCY_TEXT = {
    "do": ("make {target} smaller in AST nodes ({nodes} now) while keeping every behaviour: remove duplication, dead code, "
           "needless indirection and over-general code; prefer simpler data flow",
           "do not move code into other modules (the whole package must shrink, not just this file)",
           "do not reformat to fewer lines - size is counted in AST nodes, so formatting changes count for nothing"),
    "tests": ("every existing test keeps passing unchanged; the number of tests must not drop",),
    "fail": ("behaviour removed that no test caught - every requirement is re-checked", "code moved to another module",
             "tests deleted or weakened to make the shrink pass", "memory use made worse"),
}


def efficiency_requirement(ledger: Ledger, objective_id: str) -> str:
    rid = ledger.view.unique.get(("Requirement", EFFICIENCY_KEY))
    if rid:
        return rid
    return ledger.append(M.Requirement(
        created_by=M.Role.OWNER, parents=(objective_id,), key=EFFICIENCY_KEY, priority=M.Priority.HIGH,
        description="the Creator constantly shrinks its own code and memory (owner, 1 Oct 2026) without losing capability",
        acceptance_test="efficiency:size (a standing objective: never permanently met)",
        measurement_method="creator.efficiency AST-node size of the target and the whole package; peak memory of a fixed workload",
        failure_condition="a shrink that loses a requirement, a test, or makes memory worse",
        validation_method="creator.kernel efficiency claim computed by creator.model:improvement_verdict",
        evidence_location="state/creator/cycles/<package>/"))


def recent_failed_targets(ledger: Ledger, window: int = 6) -> list[str]:
    """Targets whose last efficiency packages failed - skipped for a while so the loop does not grind on one file."""
    out = []
    for e in ledger.of_type("WorkPackage")[-window:]:
        if ledger.view.status.get(e.id) is M.Status.FAILED and getattr(e.record, "outputs") \
                and getattr(e.record, "objective", "").startswith("shrink "):
            out.append(getattr(e.record, "outputs")[0])
    return out


def plan_efficiency(ledger: Ledger, root: Any, base_ref: str, avoid: Sequence[str] = (), kind: Optional[str] = None) -> Optional[Plan]:
    """Plan one efficiency package: SIZE (shrink the largest module the Creator may edit) or ACTIVATION (load less code at start -
    owner: 'never run any more code than absolutely necessarry'). Kinds alternate unless `kind` is given."""
    past = [e for e in ledger.of_type("WorkPackage") if getattr(e.record, "objective", "").startswith(("shrink ", "load less"))]
    kind = kind or ("activation" if len(past) % 2 else "size")
    if kind == "activation":
        return _plan_activation(ledger, root, base_ref, avoid)
    from pathlib import Path as _P
    from creator import efficiency as E
    objective_id = next((e.id for e in ledger.of_type("Objective") if getattr(e.record, "statement") == O.SELF_STATEMENT), None)
    if objective_id is None:
        return None
    target = E.pick_target(_P(root), avoid=tuple(avoid) + tuple(recent_failed_targets(ledger)))
    if target is None:
        return None
    path, nodes = target
    req = efficiency_requirement(ledger, objective_id)
    gap = ledger.append(M.Gap(created_by=M.Role.KERNEL, parents=(req,), kind=M.GapKind.ARCHITECTURE,
                              description=f"shrink {path}: {nodes} AST nodes", importance=0.3))
    kw = {"target": path, "nodes": str(nodes)}
    pid = next_package_id(ledger)
    wp = ledger.append(M.WorkPackage(
        created_by=M.Role.KERNEL, parents=(gap,), package_id=pid, objective=f"shrink {path} without losing capability",
        why_it_exists=f"owner directive 1 Oct 2026: the Creator constantly shrinks its own code; {path} is the largest module "
                      f"it may edit ({nodes} AST nodes)",
        prerequisites=("none",), inputs=(path,), outputs=(path,), implementation_requirements=_fmt(EFFICIENCY_TEXT["do"], **kw),
        interfaces=("every public interface of the module stays as it is unless no caller uses it",),
        data_flow="sandbox -> build -> affected tests -> every requirement re-checked -> size + memory measured -> decision",
        dependencies=(), test_requirements=_fmt(EFFICIENCY_TEXT["tests"], **kw),
        validation_requirements=(f"{path} and the whole creator package smaller in AST nodes", "every base requirement kept",
                                 "test count not lower", "peak memory not significantly higher"),
        expected_failure_modes=_fmt(EFFICIENCY_TEXT["fail"], **kw),
        evidence_requirements=("size and memory footprints of base and candidate", "junit of base and candidate"),
        failure_conditions=("no shrink of the module or of the package", "any requirement lost", "any regression"),
        rollback_requirements=("discard the sandbox; after adoption revert the merge commit",),
        completion_criteria=("efficiency claim IMPROVEMENT computed by creator.model:improvement_verdict", "audit clean"),
        anti_premature_completion=("a smaller file is not a smaller package", "fewer lines is not fewer AST nodes"),
        meaningful_code_depth=0))
    cp = ledger.append(M.ChangeProposal(created_by=M.Role.KERNEL, parents=(wp,), reason=f"shrink {path}",
                                        parent_objective=objective_id, originating_task=wp, affected_components=(path,),
                                        expected_effect=f"{path} and the creator package smaller with nothing lost"))
    ex = ledger.append(M.Experiment(created_by=M.Role.KERNEL, parents=(cp,),
                                    hypothesis=f"{path} can be made smaller without losing any requirement, test or memory",
                                    design="sandbox; base vs candidate AST size, package size, peak memory, tests, requirements",
                                    metrics=(f"ast_nodes:{path}", "package_ast_nodes", "peak_memory_mb"), seed=0,
                                    baseline_ref=base_ref, candidate_ref=f"sandbox:{pid}"))
    ledger.transition(gap, M.Status.IN_PROGRESS, f"planned as {pid}", M.Role.KERNEL)
    return Plan(gap, EFFICIENCY_KEY, path, "efficiency", M.Role.IMPLEMENTER, wp, pid, cp, ex, 1)


def _plan_activation(ledger: Ledger, root: Any, base_ref: str, avoid: Sequence[str] = ()) -> Optional[Plan]:
    from pathlib import Path as _P
    from creator import efficiency as E
    objective_id = next((e.id for e in ledger.of_type("Objective") if getattr(e.record, "statement") == O.SELF_STATEMENT), None)
    if objective_id is None:
        return None
    act = E.activation(_P(root))
    skip = set(avoid) | set(recent_failed_targets(ledger))
    gains = E.activation_gains(_P(root))
    cands = [m for m, g in gains.items() if g > 0 and E.editable(m) and m not in skip]
    if not cands:
        return None                                                     # nothing could load less: no package is planned
    path = max(cands, key=lambda m: (gains[m], m))
    eager = {path: gains[path]}
    rid = ledger.view.unique.get(("Requirement", ACTIVATION_KEY)) or ledger.append(M.Requirement(
        created_by=M.Role.OWNER, parents=(objective_id,), key=ACTIVATION_KEY, priority=M.Priority.HIGH,
        description="never run more code than necessary: starting the Creator loads as little of it as possible (owner 1 Oct 2026)",
        acceptance_test="efficiency:activation (a standing objective: never permanently met)",
        measurement_method="creator.efficiency.activation (kernel start) + static_load (every module's eager imports)",
        failure_condition="a change that loads more code, or loses a requirement or a test",
        validation_method="creator.kernel efficiency claim computed by creator.model:improvement_verdict",
        evidence_location="state/creator/cycles/<package>/"))
    gap = ledger.append(M.Gap(created_by=M.Role.KERNEL, parents=(rid,), kind=M.GapKind.ARCHITECTURE,
                              description=f"load less: making {path}'s Creator imports lazy would unload {eager[path]} "
                                          f"AST nodes at start "
                                          f"(kernel start loads {act['fraction']:.0%} of the package)", importance=0.3))
    pid = next_package_id(ledger)
    wp = ledger.append(M.WorkPackage(
        created_by=M.Role.KERNEL, parents=(gap,), package_id=pid, objective=f"load less code at start: {path}",
        why_it_exists=f"owner directive 1 Oct 2026 (sparse activation); kernel start loads {act['active_nodes']} of "
                      f"{act['package_nodes']} AST nodes",
        prerequisites=("none",), inputs=(path,), outputs=(path,),
        implementation_requirements=(f"in {path}, import Creator components inside the functions that use them instead of at "
                                     "module level, and drop imports nothing uses",),
        interfaces=("public interfaces unchanged",),
        data_flow="sandbox -> tests -> activation + static eager load measured -> decision", dependencies=(),
        test_requirements=("every existing test keeps passing; the number of tests must not drop",),
        validation_requirements=("kernel-start activation lower", "static eager load over all modules lower",
                                 "every base requirement kept"),
        expected_failure_modes=("an import moved into a function that runs at import time anyway", "a type-hint import moved"),
        evidence_requirements=("activation + static load of base and candidate",),
        failure_conditions=("no lower activation", "any requirement lost"),
        rollback_requirements=("discard the sandbox; after adoption revert the merge commit",),
        completion_criteria=("efficiency claim IMPROVEMENT computed by creator.model:improvement_verdict",),
        anti_premature_completion=("moving code between modules is not loading less",), meaningful_code_depth=0))
    cp = ledger.append(M.ChangeProposal(created_by=M.Role.KERNEL, parents=(wp,), reason=f"load less: {path}",
                                        parent_objective=objective_id, originating_task=wp, affected_components=(path,),
                                        expected_effect="the Creator loads less code to start"))
    ex = ledger.append(M.Experiment(created_by=M.Role.KERNEL, parents=(cp,), hypothesis=f"{path} can load its components lazily",
                                    design="sandbox; base vs candidate kernel-start activation and static eager load",
                                    metrics=("active_ast_nodes:kernel_start", "static_eager_load"), seed=0,
                                    baseline_ref=base_ref, candidate_ref=f"sandbox:{pid}"))
    ledger.transition(gap, M.Status.IN_PROGRESS, f"planned as {pid}", M.Role.KERNEL)
    return Plan(gap, ACTIVATION_KEY, path, "efficiency", M.Role.IMPLEMENTER, wp, pid, cp, ex, 1)


def record_outcome(ledger: Ledger, plan: Plan, success: bool, reason: str) -> None:
    """Close the plan's chain: on failure the work package FAILS and the gap returns to FAILED (re-plannable, counted toward
    max_attempts); on success the gap stays IN_PROGRESS until creator.gaps.sync proves the check on main."""
    wp_state = ledger.view.status[plan.work_package_id]
    if wp_state is M.Status.NOT_STARTED:
        ledger.transition(plan.work_package_id, M.Status.IN_PROGRESS, "executing", M.Role.KERNEL)
    if success:
        ledger.transition(plan.work_package_id, M.Status.IMPLEMENTED, reason, M.Role.KERNEL)
        return
    ledger.transition(plan.work_package_id, M.Status.FAILED, reason, M.Role.KERNEL)
    if ledger.view.status[plan.gap_id] is M.Status.IN_PROGRESS:
        ledger.transition(plan.gap_id, M.Status.FAILED, f"{plan.package_id} failed: {reason}", M.Role.KERNEL)


def unblock(ledger: Ledger, gap_id: str, reason: str) -> None:
    """Reopen a BLOCKED gap (e.g. after a dependency or strategy changed); its old attempts stay in the record."""
    if ledger.view.status.get(gap_id) is not M.Status.BLOCKED:
        raise PlanningError(f"{gap_id} is not BLOCKED")
    ledger.transition(gap_id, M.Status.IN_PROGRESS, f"unblocked: {reason}", M.Role.KERNEL)
    ledger.transition(gap_id, M.Status.FAILED, "reopened for re-planning", M.Role.KERNEL)
