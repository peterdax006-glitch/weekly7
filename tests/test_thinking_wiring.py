"""Thinking focus is wired end to end (2 Oct 2026): the swarm's focus hook really filters, and in thinking focus the swarm's filler
is the thinking drills + live predictions + blueprint. Before this, the hook called focus.allowed(plan) with one argument while
focus.allowed required two - the TypeError was swallowed and NOTHING was filtered, silently."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import focus as F
from creator import swarm as W

ROOT = Path(__file__).resolve().parents[1]


def plan(req: str, step: str, component: str) -> SimpleNamespace:
    return SimpleNamespace(requirement_key=req, step=step, component=component, package_id="CPX")


def test_the_real_swarm_hook_filters_in_thinking_focus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    monkeypatch.setattr(F, "DEFAULT_STATE", tmp_path)
    shrink = plan("EFF.size", "size", "creator/kernel.py")
    coverage = plan("EFF.coverage", "coverage", "creator/build.py")
    k29 = plan("K29.exists", "exists", "K29")
    k28 = plan("K28.exists", "exists", "K28")
    assert W._focus_allows(shrink) is False                         # through the swarm's own hook, one argument
    assert W._focus_allows(coverage) is False
    assert W._focus_allows(k29) is True                             # thinking work (its own learning-signal goal)
    assert W._focus_allows(k28) is False                            # not thinking work
    (tmp_path / "focus.json").unlink()
    assert W._focus_allows(shrink) is True                          # no focus: nothing filtered


def _swarm_script():                                                 # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("creator_swarm_t", ROOT / "scripts" / "creator_swarm.py")
    mod = importlib.util.module_from_spec(spec)                      # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                     # type: ignore[union-attr]
    return mod


def test_thinking_focus_makes_the_filler_think(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cs = _swarm_script()
    monkeypatch.setattr(cs, "STATE", tmp_path)
    calls: list[str] = []
    fake_thinking = SimpleNamespace(run=lambda state: calls.append("run"))
    fake_drills = SimpleNamespace(drill_filler=lambda *a, **k: (lambda: (lambda: calls.append("drill"))))
    modules = {"thinking": fake_thinking, "drillsources": fake_drills}
    monkeypatch.setattr(cs.REG, "get", lambda name: modules[name])
    monkeypatch.setattr(cs.REG, "optional", lambda name: F if name == "focus" else None)
    monkeypatch.setattr(cs, "_THINK_LAST", {"run": 0.0, "blueprint": 1e18})           # blueprint not due in this test
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    nxt = cs.make_filler()
    nxt()()                                                          # first: the live predictions (due)
    nxt()()                                                          # then drills
    assert calls == ["run", "drill"]
    (tmp_path / "focus.json").unlink()
    monkeypatch.setattr(cs.W, "self_bench_filler", lambda store: "SELF_BENCH")
    assert cs.make_filler() == "SELF_BENCH"                          # no focus: the old filler


def test_thinking_focus_holds_back_whole_components_before_planning(tmp_path: Path) -> None:
    # 3 Oct: every empty round planned packages the focus then released (6 plans/h, goal_student_shrink 15x in 24 h)
    from creator import schedule as SCHED
    nodes = [SCHED.Node("g1", "K28", "exists", 1.0, 1.0), SCHED.Node("g2", "K29", "exists", 1.0, 1.0),
             SCHED.Node("g3", "K29", "size", 1.0, 1.0), SCHED.Node("g4", "creator/goal_student_shrink.py", "size", 1.0, 1.0)]
    fake = SimpleNamespace(build_nodes=lambda led, specs=None, history=None: nodes)
    cfg = SimpleNamespace(ledger_path=tmp_path / "ledger.jsonl", specs=lambda: [])
    assert W.focus_off_components(fake, None, cfg) == []                       # no focus: nothing held back
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    assert W.focus_off_components(fake, None, cfg) == ["K28", "creator/goal_student_shrink.py"]   # K29 keeps its allowed gap
    broken = SimpleNamespace(build_nodes=lambda *a, **k: 1 / 0)
    assert W.focus_off_components(broken, None, cfg) == []                     # a failure filters nothing here (the release still runs)
