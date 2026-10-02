"""Swarm rounds plan through creator.schedule: batches sized by slots and governor, never two plans on one component, efficiency
fallback, the flag, and visible explanations."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import pytest

from creator import kernel as K
from creator import model as M
from creator import planner as P
from creator import schedule as SCH
from creator import swarm as W
from tests.test_creator_swarm import cfg, size_only  # noqa: F401 - fixtures


def mk(n: int, comp: str, step: str = "exists") -> P.Plan:
    return P.Plan(f"g{n}", "K", comp, step, M.Role.KERNEL, f"w{n}", f"CP{n}", "x", "e", 1)


def gov(max_workers: int = 8, free: float = 10.0) -> W.Governor:
    return W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=1.0, max_workers=max_workers,
                      free=lambda: free, total=lambda: 16.0, observe=lambda n: None)


def rig(monkeypatch: pytest.MonkeyPatch, batches: list[list[P.Plan]], asked: list[int], live: dict[str, int]) -> None:
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    it = iter(batches)

    def plan_batch(c: Any, led: Any, main: Any, base: str, slots: int, held: Any = (), held_files: Any = ()) -> list[P.Plan]:
        asked.append(slots)
        return list(next(it, []))
    monkeypatch.setattr(SCH, "plan_batch", plan_batch)

    def fake_execute(cfg: Any, worker: Any, plan: P.Plan, *a: Any, **k: Any) -> K.CycleReport:
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        time.sleep(0.3)
        live["now"] -= 1
        return K.CycleReport(1, "ADOPTED", plan.package_id, "K")
    monkeypatch.setattr(K, "execute", fake_execute)


def test_a_batch_fills_the_free_slots_and_starts_one_worker_per_plan(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [[mk(1, "A"), mk(2, "B"), mk(3, "C")]], asked, live)
    rnd = W.run_round(dataclasses.replace(cfg, mode="gaps"), lambda: None, gov(max_workers=3), max_packages=8, poll_s=0.02)
    assert asked[0] == 3                                                # asked for exactly the free slots
    assert sorted(r.package for r in rnd.reports) == ["CP1", "CP2", "CP3"] and live["max"] == 3


def test_the_governor_limits_the_batch(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [[mk(1, "A"), mk(2, "B")], []], asked, live)
    g = gov(max_workers=8, free=2.5)                                    # 1.0 GB per worker, floor 0: two fit
    rnd = W.run_round(dataclasses.replace(cfg, mode="gaps"), lambda: None, g, max_packages=8, poll_s=0.02)
    assert asked[0] == 2 and live["max"] <= 2 and len(rnd.reports) == 2


def test_no_two_plans_of_one_component_run_together(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    held_seen: list[Any] = []
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [], [], live)
    pending = [[mk(1, "A")], [mk(2, "A"), mk(3, "B")]]

    def plan_batch(c: Any, led: Any, main: Any, base: str, slots: int, held: Any = (), held_files: Any = ()) -> list[P.Plan]:
        held_seen.append(list(held))
        out = [p for p in (pending.pop(0) if pending else []) if p.component not in held]   # the real scheduler honours held
        return out
    monkeypatch.setattr(SCH, "plan_batch", plan_batch)
    rnd = W.run_round(dataclasses.replace(cfg, mode="gaps"), lambda: None, gov(), max_packages=8, poll_s=0.02)
    assert held_seen[0] == [] and "A" in held_seen[1]                   # the second ask was told A is busy
    assert sorted(r.package for r in rnd.reports) == ["CP1", "CP3"]


def test_an_empty_batch_still_plans_shrink_work_in_auto_mode(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [[], []], asked, live)
    eff = iter([mk(9, "creator/m1.py", "size")])
    monkeypatch.setattr(K, "plan_one", lambda c, *a, **k: next(eff, None) if c.mode == "efficiency" else None)
    rnd = W.run_round(dataclasses.replace(cfg, mode="auto"), lambda: None, gov(), max_packages=4, poll_s=0.02)
    assert [r.package for r in rnd.reports] == ["CP9"] and asked


def test_flag_off_uses_the_old_planner_only(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [[mk(1, "A")]], asked, live)
    old = iter([mk(5, "Z")])
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: next(old, None))
    rnd = W.run_round(dataclasses.replace(cfg, mode="gaps"), lambda: None, gov(), max_packages=4, poll_s=0.02, scheduled=False)
    assert asked == [] and [r.package for r in rnd.reports] == ["CP5"]


def test_a_scheduler_failure_falls_back_and_is_recorded(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    live = {"now": 0, "max": 0}
    rig(monkeypatch, [], [], live)

    def boom(*a: Any, **k: Any) -> list[P.Plan]:
        raise RuntimeError("graph exploded")
    monkeypatch.setattr(SCH, "plan_batch", boom)
    old = iter([mk(5, "Z")])
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: next(old, None))
    c = dataclasses.replace(cfg, mode="gaps")
    rnd = W.run_round(c, lambda: None, gov(), max_packages=4, poll_s=0.02)
    assert [r.package for r in rnd.reports] == ["CP5"]
    assert "graph exploded" in SCH.explain_path_for(c.ledger_path).read_text(encoding="utf-8")


def test_last_plan_reads_the_newest_batch(tmp_path: Path) -> None:
    path = SCH.explain_path_for(tmp_path / "ledger.jsonl")
    rows = [{"at": "t1", "critical_path_s": 5, "node": "old", "component": "O", "step": "exists", "chosen": True, "slack_s": 0, "why": "w"},
            {"at": "t2", "critical_path_s": 9, "node": "a", "component": "A", "step": "exists", "chosen": True, "slack_s": 0, "why": "critical"},
            {"at": "t2", "critical_path_s": 9, "node": "b", "component": "B", "step": "exists", "chosen": False, "slack_s": 3, "why": "later"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    lp = SCH.last_plan(tmp_path / "ledger.jsonl")
    assert lp["at"] == "t2" and lp["critical_path"] == ["a"] and lp["chosen"] == ["a"] and lp["critical_path_s"] == 9
    assert SCH.last_plan(tmp_path / "sub" / "none.jsonl")["at"] is None
