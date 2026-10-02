"""CR07: objectives and the requirement compiler (C77 secs 2, 12, 13)."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import model as M
from creator import objective as O
from creator import selfmodel as SM
from creator.ledger import Ledger

SPECS = [SM.CapabilitySpec("K01", "ledger", ("pkg/a.py",), ("tests/test_a.py",), 10),
         SM.CapabilitySpec("K02", "selfmodel", ("pkg/b.py",), ("tests/test_b.py",), 10)]


@pytest.fixture()
def led(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)


def test_parse_objective_strictly() -> None:
    spec = O.parse_objective("Statement: make the planner faster\nOutcomes:\n- plans in under a second\n"
                             "Acceptance criteria:\n- median plan time < 1 s on devbench\nRisks:\n- noisy timing\n")
    assert spec.statement == "make the planner faster" and spec.acceptance_criteria == ("median plan time < 1 s on devbench",)
    assert spec.risks == ("noisy timing",)
    with pytest.raises(O.ObjectiveError, match="Statement"):
        O.parse_objective("Outcomes:\n- x\nAcceptance criteria:\n- y\n")
    with pytest.raises(O.ObjectiveError, match="acceptance criterion"):
        O.parse_objective("Statement: x\n")
    with pytest.raises(O.ObjectiveError, match="cannot place"):
        O.parse_objective("Statement: x\nAcceptance criteria:\n- y\nsome loose prose\n")


def test_self_objective_and_ladder_are_idempotent(led: Ledger) -> None:
    oid = O.self_objective(led)
    assert O.self_objective(led) == oid
    first = O.compile_capabilities(led, oid, SPECS)
    assert first.created == 2 + 2 * len(O.LADDER)
    again = O.compile_capabilities(led, oid, SPECS)
    assert again.created == 0 and again.requirement_ids == first.requirement_ids
    keys = {r["key"] for r in O.requirement_view(led)}
    assert {"K01.exists", "K01.validated", "K02.tested", "K02.integrated"} <= keys


def test_dependencies_follow_ladder_and_build_order(led: Ledger) -> None:
    oid = O.self_objective(led)
    c = O.compile_capabilities(led, oid, SPECS)
    req = c.requirement_ids
    assert set(led.get(req["K01.validated"]).depends_on) == {req[f"K01.{s}"] for s in ("tested", "no_stubs", "integrated")}
    assert "K01.depth" not in req                    # owner ruling 1 Oct 2026: capability, not lines - no line-depth step
    assert req["K01.tested"] in led.get(req["K02.exists"]).depends_on            # K02 is built on a tested K01
    assert led.get(req["K01.exists"]).acceptance_test == "check:exists:K01"


def test_only_computed_checks_are_accepted() -> None:
    assert O.parse_check("check:depth:K06") == ("depth", "K06")
    for bad in ("it works", "check:vibes:K01", "check:depth:X1"):
        with pytest.raises(O.ObjectiveError):
            O.parse_check(bad)


def test_importance_orders_priority_step_and_build_order() -> None:
    assert O.importance(M.Priority.CRITICAL, "exists", "K01") > O.importance(M.Priority.HIGH, "exists", "K01")
    assert O.importance(M.Priority.HIGH, "exists", "K02") > O.importance(M.Priority.HIGH, "validated", "K02")
    assert O.importance(M.Priority.HIGH, "tested", "K02") > O.importance(M.Priority.HIGH, "tested", "K09")


# ---- owner ruling 2 Oct: entry-point integration and retired components -------------------------------------------------

def _mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "creator").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "creator" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("swarm", "ordinary", "agents"):
        (tmp_path / "creator" / f"{name}.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "creator_swarm.py").write_text("from creator import swarm as W\n", encoding="utf-8")
    (tmp_path / "scripts" / "other.py").write_text("from creator import ordinary\n", encoding="utf-8")
    return tmp_path


def _states(tmp_path: Path) -> SM.SelfModel:
    specs = [SM.CapabilitySpec("K22", "swarm", ("creator/swarm.py",), (), 1),
             SM.CapabilitySpec("K99", "ordinary", ("creator/ordinary.py",), (), 1),
             SM.CapabilitySpec("K05", "agents", ("creator/agents.py",), (), 1)]
    return SM.build(_mini_repo(tmp_path), scope=("creator",), capabilities=specs, include_versions=False)


def test_entry_point_script_integrates_listed_components_only(tmp_path: Path) -> None:
    model = _states(tmp_path)
    ok, detail = O._integrated(model, model.capability("K22"), None, None)
    assert ok and "entry point" in detail
    ok, detail = O._integrated(model, model.capability("K99"), None, None)       # imported only by a script, not listed
    assert not ok and "creator/ordinary.py" in detail


def test_entry_point_must_actually_import_the_module(tmp_path: Path) -> None:
    model = _states(tmp_path)
    (tmp_path / "scripts" / "creator_swarm.py").write_text("import os\n", encoding="utf-8")
    assert not O._integrated(model, model.capability("K22"), None, None)[0]


def test_retired_component_closes_integrated_and_validated(tmp_path: Path) -> None:
    model = _states(tmp_path)
    c = model.capability("K05")
    for fn in (O._integrated, O._validated):
        ok, detail = fn(model, c, None, None)
        assert ok and detail.startswith("retired: ")
    assert O.retired_reason("K05") and O.retired_reason("K22") is None
    assert not O._validated(model, model.capability("K22"), None, None)[0]
