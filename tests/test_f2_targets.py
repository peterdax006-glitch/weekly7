"""Fix engineer 2: the efficiency planner stops re-planning the same target after every restart (state read from the ledger),
rotates to the least recently attempted target, and the waste is measured (wasted_work, repeat_rate)."""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pytest

from creator import constraints as CON
from creator import efficiency as E
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "proj"
    for name, n in (("kernel", 40), ("generator", 30), ("planner", 20), ("tiny", 2)):
        put(root, f"creator/{name}.py", "x = 1\n" * n)
    return root, tmp_path / "state"


def ledger_of(state: Path, root: Path) -> Ledger:
    led = Ledger(state / "ledger.jsonl", evidence_root=root)
    O.self_objective(led)
    return led


def plan_targets(led: Ledger, root: Path, n: int) -> list[str | None]:
    out: list[str | None] = []
    for _ in range(n):
        p = P.plan_efficiency(led, root, "sha", kind="size")
        out.append(p.component if p else None)
    return out


def test_the_largest_module_is_not_planned_18_times_over(world) -> None:
    """Reproduces the real ledger: every restart re-planned the same largest file (kernel.py 18x) because nothing it planned was judged."""
    root, state = world
    led = ledger_of(state, root)
    got = plan_targets(led, root, 18)
    assert got[:4] == ["creator/kernel.py", "creator/generator.py", "creator/planner.py", "creator/tiny.py"]
    assert got[4:] == [None] * 14                                       # every target has an unjudged package: nothing is planned twice
    assert sorted(P.recent_unjudged_targets(led)) == sorted(f"creator/{n}.py" for n in ("kernel", "generator", "planner", "tiny"))
    reopened = Ledger(state / "ledger.jsonl", evidence_root=root)       # a swarm restart: the memory is the ledger
    assert P.plan_efficiency(reopened, root, "sha", kind="size") is None


def test_the_window_passes_and_the_oldest_attempt_goes_first(world, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state = world
    led = ledger_of(state, root)
    plan_targets(led, root, 4)
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 7 * 3600)          # past the 6 h window
    assert P.recent_unjudged_targets(led) == []
    assert plan_targets(led, root, 2) == ["creator/kernel.py", "creator/generator.py"]     # rotation = attempt order


def test_pick_target_rotates_by_last_attempt_then_size(tmp_path: Path) -> None:
    put(tmp_path, "creator/big.py", "x = 1\n" * 30)
    put(tmp_path, "creator/mid.py", "x = 1\n" * 20)
    put(tmp_path, "creator/small.py", "x = 1\n")
    pick = E.pick_target(tmp_path)
    assert pick is not None and pick[0] == "creator/big.py"
    pick = E.pick_target(tmp_path, last={"creator/big.py": 5.0})
    assert pick is not None and pick[0] == "creator/mid.py"              # never attempted beats attempted; larger breaks the tie
    pick = E.pick_target(tmp_path, last={"creator/big.py": 5.0, "creator/mid.py": 9.0, "creator/small.py": 7.0})
    assert pick is not None and pick[0] == "creator/big.py"              # all attempted: the oldest first


def test_an_interrupted_package_keeps_its_target_off_a_judged_one_does_not(world) -> None:
    root, state = world
    led = ledger_of(state, root)
    p1 = P.plan_efficiency(led, root, "sha", kind="size")
    assert p1 is not None
    led.transition(p1.work_package_id, M.Status.IN_PROGRESS, "worker started", M.Role.KERNEL)
    led.transition(p1.work_package_id, M.Status.FAILED, "interrupted: host stop", M.Role.KERNEL)
    assert P.recent_unjudged_targets(led) == [p1.component]             # released as interrupted: nothing was decided
    p2 = P.plan_efficiency(led, root, "sha", kind="size")
    assert p2 is not None
    led.transition(p2.work_package_id, M.Status.IN_PROGRESS, "worker started", M.Role.KERNEL)
    led.transition(p2.work_package_id, M.Status.FAILED, "no shrink: the candidate is not smaller", M.Role.KERNEL)
    assert p2.component not in P.recent_unjudged_targets(led)           # a real verdict frees the target


def test_wasted_work_and_repeat_rate_are_measured_from_the_ledger(world) -> None:
    root, state = world
    led = ledger_of(state, root)
    first = P.plan_efficiency(led, root, "sha", kind="size")
    assert first is not None
    led.transition(first.work_package_id, M.Status.IN_PROGRESS, "worker started", M.Role.KERNEL)
    led.transition(first.work_package_id, M.Status.FAILED, "interrupted: host stop", M.Role.KERNEL)
    second = P.plan_efficiency(led, root, "sha", kind="size")
    assert second is not None and second.component != first.component
    led.transition(second.work_package_id, M.Status.IN_PROGRESS, "worker started", M.Role.KERNEL)
    led.transition(second.work_package_id, M.Status.FAILED, "no shrink: the candidate is not smaller", M.Role.KERNEL)
    now = dt.datetime.now() + dt.timedelta(hours=2)                     # both are older than the running-work grace
    m = {x.name: x for x in CON.planning_metrics(state, now, 24.0)}
    assert m["wasted_work"].value == 0.5 and m["wasted_work"].detail["never_judged"] == 1     # one interrupted, one judged
    assert m["repeat_rate"].value == 0.0                                # rotation: no repeats
    led.append(M.WorkPackage(                                           # what the old planner did: the same target planned again
        created_by=M.Role.KERNEL, parents=led.get(first.work_package_id).parents, package_id="CP9999",
        objective=f"shrink {first.component} without losing capability", why_it_exists="r", prerequisites=("none",),
        inputs=(first.component,), outputs=(first.component,), implementation_requirements=("x",), interfaces=("x",),
        data_flow="x", dependencies=(), test_requirements=("x",), validation_requirements=("x",), expected_failure_modes=("x",),
        evidence_requirements=("x",), failure_conditions=("x",), rollback_requirements=("x",), completion_criteria=("x",),
        anti_premature_completion=("x",), meaningful_code_depth=0))
    m = {x.name: x for x in CON.planning_metrics(state, now, 24.0)}
    assert abs(m["repeat_rate"].value - 1 / 3) < 1e-3 and m["repeat_rate"].detail["repeated_targets"] == [first.component]
    rep = CON.measure_all(state, now, 24.0)
    assert {"wasted_work", "repeat_rate"} <= {r["name"] for r in rep["ranked"]} and not rep["errors"]
    assert "wasted_work" in CON.format_report(rep)


def test_no_ledger_means_zero_waste(tmp_path: Path) -> None:
    m = {x.name: x.value for x in CON.planning_metrics(tmp_path, dt.datetime.now(), 24.0)}
    assert m == {"wasted_work": 0.0, "repeat_rate": 0.0}
