"""CR082/CR083: critical path, parallel-safe batches, learned costs, explanations (creator.schedule)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import schedule as S
from creator import selfmodel as SM
from creator.ledger import Ledger


def node(i: str, comp: str, cost: float = 100.0, deps: tuple[str, ...] = (), files: tuple[str, ...] = (), status: str = "ready",
         step: str = "exists", value: float = 0.5) -> S.Node:
    return S.Node(i, comp, step, value, cost, frozenset(files or (f"{comp}.py",)), deps, status)


def test_critical_path_is_the_longest_cost_weighted_chain() -> None:
    # a -> b -> c is 3 nodes of 100 (300); d -> e is 2 nodes but 500 + 400 = 900 and so is the longest
    nodes = [node("a", "A"), node("b", "B", deps=("a",)), node("c", "C", deps=("b",)),
             node("d", "D", cost=500), node("e", "E", cost=400, deps=("d",))]
    a = S.analyse(nodes)
    assert a.length == 900 and a.path == ("d", "e")
    assert a.slack["d"] == a.slack["e"] == 0 and a.slack["a"] == 600
    assert S.schedule(nodes, 1).picks[0].id == "d"                      # the critical node goes first, not the cheap one


def test_batch_never_holds_dependants_or_shared_files() -> None:
    nodes = [node("a", "A"), node("b", "B", deps=("a",)), node("c", "C", files=("shared.py",)),
             node("d", "D", files=("shared.py", "d.py")), node("e", "E")]
    b = S.schedule(nodes, 5)
    ids = {n.id for n in b.picks}
    assert "b" not in ids and not ({"c", "d"} <= ids) and {"a", "e"} <= ids and len(ids & {"c", "d"}) == 1
    files = [f for n in b.picks for f in n.files]
    assert len(files) == len(set(files))
    assert len(S.schedule(nodes, 2).picks) == 2


def test_running_work_holds_its_files_and_components() -> None:
    nodes = [node("a", "A", status="running"), node("b", "B", files=("A.py",)), node("c", "C")]
    assert [n.id for n in S.schedule(nodes, 3).picks] == ["c"]
    assert [n.id for n in S.schedule([node("c", "C")], 3, held_components=["C"]).picks] == []
    assert [n.id for n in S.schedule([node("c", "C")], 3, held_files=["C.py"]).picks] == []


def test_blocked_dependency_delays_its_dependants_and_says_so() -> None:
    nodes = [node("a", "A", status="blocked"), node("b", "B", deps=("a",)), node("c", "C")]
    b = S.schedule(nodes, 3)
    assert [n.id for n in b.picks] == ["c"]
    why = {r["node"]: r["why"] for r in b.reasons}
    assert why["b"].startswith("blocked by a") and "blocked" in why["a"] and why["c"].startswith(("on the critical", "off the critical"))


def test_non_worker_steps_are_not_picked() -> None:
    nodes = [node("v", "V", step="validated"), node("w", "W")]
    assert [n.id for n in S.schedule(nodes, 2, steps=("exists",)).picks] == ["w"]


def test_dependency_cycle_does_not_hang() -> None:
    a = S.analyse([node("a", "A", deps=("b",)), node("b", "B", deps=("a",))])
    assert a.length >= 0


# ------------------------------------------------------------------------------------------------ with a real (synthetic) ledger

SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 0),
         SM.CapabilitySpec("K05", "mid", ("pkg/mid.py",), ("tests/test_mid.py",), 0),
         SM.CapabilitySpec("K06", "other", ("pkg/other.py",), ("tests/test_other.py",), 0)]


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Ledger, SM.SelfModel]:
    r = tmp_path / "proj"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "tests" / "__init__.py").write_text("", encoding="utf-8")
    led = Ledger(r / "dev.jsonl", evidence_root=r)
    O.compile_capabilities(led, O.self_objective(led), SPECS)
    model = SM.build(r, scope=("pkg", "tests"), capabilities=SPECS, test_evidence={}, include_versions=False)
    G.sync(led, model)
    return led, model


def test_nodes_carry_graph_files_and_cost(world) -> None:
    led, _ = world
    nodes = {(n.component, n.step): n for n in S.build_nodes(led, SPECS)}
    k5 = nodes[("K05", "exists")]
    assert k5.files == {"pkg/mid.py", "tests/test_mid.py"} and k5.cost == S.DEFAULT_COST_S["exists"]
    assert any(d for d in k5.deps)                                      # K05 is built on K01: it waits for K01's tests


def test_learned_costs_change_the_order() -> None:
    def chains(h: S.History) -> list[S.Node]:
        c = lambda i, comp, step, deps=(): S.Node(i, comp, step, 0.5, h.cost(step), frozenset({comp}), deps)  # noqa: E731
        return [c("x1", "X", "exists"), c("x2", "X2", "exists", ("x1",)), c("y1", "Y", "tested"), c("y2", "Y2", "tested", ("y1",))]
    slow_exists = S.History.from_samples([("exists", 9000.0, True)] * 3 + [("tested", 1.0, True)] * 3)
    slow_tested = S.History.from_samples([("tested", 9000.0, True)] * 3 + [("exists", 1.0, True)] * 3)
    assert S.schedule(chains(slow_exists), 1).picks[0].id == "x1"
    assert S.schedule(chains(slow_tested), 1).picks[0].id == "y1"
    assert S.History().cost("exists") == S.DEFAULT_COST_S["exists"]


def test_failed_history_lowers_value() -> None:
    bad = S.History.from_samples([("exists", 5.0, False)] * 6)
    good = S.History.from_samples([("exists", 5.0, True)] * 6)
    assert bad.success_rate("exists") < S.History().success_rate("exists") < good.success_rate("exists")


def test_samples_from_a_ledger_and_explanations_file(world, tmp_path: Path) -> None:
    led, model = world
    plan = P.plan_next(led, model, "b", SPECS)
    assert plan is not None
    led.transition(plan.work_package_id, M.Status.IN_PROGRESS, "cycle started", M.Role.KERNEL)
    led.transition(plan.work_package_id, M.Status.FAILED, "claim REGRESSION", M.Role.KERNEL)
    got = S.samples_from_ledger(led)
    assert len(got) == 1 and got[0][0] == plan.step and got[0][1] >= 0 and got[0][2] is False
    out = tmp_path / "state" / S.EXPLAIN_FILE
    b = S.next_batch(led, 2, SPECS, explain_path=out)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["why"] for r in rows) and {r["node"] for r in rows if r["chosen"]} == {n.id for n in b.picks}


def test_plan_next_prefer_never_overrides_blocking_and_default_unchanged(world) -> None:
    led, model = world
    ranked = G.ranked(led)
    top = next(g for g in ranked if not g.blocked_by)
    blocked = next(g for g in ranked if g.blocked_by)
    plan = P.plan_next(led, model, "b", SPECS, prefer=(blocked.gap_id,))       # a blocked gap cannot jump the queue
    assert plan is not None and plan.gap_id == top.gap_id and plan.package_id == "CP0001"


def test_plan_batch_never_orphans_a_package_it_planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A picked gap that plan_next gives up on (3 failed attempts -> BLOCKED) makes plan_next plan the NEXT gap; that plan was
    written to the ledger (gap IN_PROGRESS) and must be returned, not dropped - a dropped plan strands its gap forever."""
    from types import SimpleNamespace
    monkeypatch.setattr(O, "COMPONENT_DEPENDS", {})                      # two independent components
    specs = [SM.CapabilitySpec("K01", "a", ("pkg/a.py",), ("tests/test_a.py",), 0),
             SM.CapabilitySpec("K05", "b", ("pkg/b.py",), ("tests/test_b.py",), 0)]
    r = tmp_path / "proj"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "tests" / "__init__.py").write_text("", encoding="utf-8")
    led = Ledger(r / "dev.jsonl", evidence_root=r)
    O.compile_capabilities(led, O.self_objective(led), specs)
    model = SM.build(r, scope=("pkg", "tests"), capabilities=specs, test_evidence={}, include_versions=False)
    G.sync(led, model)
    first = next(g for g in G.ranked(led) if not g.blocked_by)
    causes = ("claim REGRESSION in the parser", "claim REGRESSION in the writer", "claim REGRESSION in the loader")   # distinct: identical failures stall (reasoning.stalled)
    for cause in causes[:P.MAX_ATTEMPTS]:
        plan = P.plan_next(led, model, "b", specs, prefer=(first.gap_id,))
        assert plan is not None and plan.gap_id == first.gap_id
        led.transition(plan.work_package_id, M.Status.IN_PROGRESS, "cycle started", M.Role.KERNEL)
        led.transition(plan.work_package_id, M.Status.FAILED, cause, M.Role.KERNEL)
        led.transition(plan.gap_id, M.Status.FAILED, cause, M.Role.KERNEL)
    cfg = SimpleNamespace(specs=lambda: specs, steps=P.WORKER_STEPS, mode="gaps", ledger_path=led.path, repo=tmp_path)
    plans = S.plan_batch(cfg, led, SimpleNamespace(model=model), "b", 3)
    in_progress = {i for i, s in led.view.status.items() if s is M.Status.IN_PROGRESS and led.view.by_id[i].rtype == "Gap"}
    assert in_progress <= {p.gap_id for p in plans}, "a gap was marked IN_PROGRESS by a plan that plan_batch dropped"


def test_a_gap_left_in_progress_by_a_dead_process_is_released_and_planned_again(world, tmp_path: Path) -> None:
    """2 Oct 2026 (new PC): K28.exists stayed IN_PROGRESS from the old machine; the scheduler skipped it as 'worker already on
    it' forever and no development work was planned. A starting swarm releases such orphans so they are planned again."""
    from creator import swarmops as K
    led, model = world
    plan = P.plan_next(led, model, "b", SPECS)
    assert plan is not None
    led.transition(plan.work_package_id, M.Status.IN_PROGRESS, "cycle started", M.Role.KERNEL)
    assert led.view.status[plan.gap_id] is M.Status.IN_PROGRESS                       # the state the move left behind
    stuck = S.next_batch(led, 4, SPECS, explain_path=tmp_path / "e.jsonl")
    assert plan.gap_id not in {n.id for n in stuck.picks}                              # 'worker already on it'
    freed = K.release_orphans(led, "swarm restarted, no worker survives")
    assert set(freed) == {plan.work_package_id, plan.gap_id}
    assert led.view.status[plan.work_package_id] is M.Status.FAILED and led.view.status[plan.gap_id] is M.Status.FAILED
    again = S.next_batch(led, 4, SPECS, explain_path=tmp_path / "e2.jsonl")
    assert plan.gap_id in {n.id for n in again.picks}                                  # planned again
    assert K.release_orphans(led, "again") == []                                       # nothing left: idempotent
