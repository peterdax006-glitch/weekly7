"""F7: focus mode. In thinking focus efficiency work is not planned; only drills and thinking-capability development are."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from creator import focus as F


def setf(st: Path, text: str) -> None:
    (st / "focus.json").write_text(text, encoding="utf-8")


def test_no_focus_allows_everything(tmp_path: Path) -> None:
    assert F.current(tmp_path) == ""
    assert F.allowed({"kind": "shrink", "requirement": "EFF.size"}, tmp_path)
    for bad in ("not json", "[]", json.dumps({"focus": 3}), json.dumps({"focus": "other"})):
        setf(tmp_path, bad)
        assert F.allowed({"kind": "shrink"}, tmp_path), bad


def test_thinking_focus_blocks_efficiency_work(tmp_path: Path) -> None:
    setf(tmp_path, json.dumps({"focus": "Thinking"}))
    assert F.current(tmp_path) == "thinking"
    for item in ({"kind": "shrink"}, {"task_kind": "tests", "component": "K27"}, {"requirement": "EFF.size", "component": "K02"},
                 {"requirement": "K17.size"}, {"kind": "activation"}, {"kind": "coverage", "modules": ["creator/goals.py"]}):
        assert not F.allowed(item, tmp_path), item


def test_thinking_focus_allows_drills_and_thinking_capability_gaps_only(tmp_path: Path) -> None:
    setf(tmp_path, json.dumps({"focus": "thinking"}))
    assert F.allowed({"kind": "drill"}, tmp_path)
    assert F.allowed({"kind": "gap", "component": "K27"}, tmp_path)                       # goal generation
    assert F.allowed({"kind": "gap", "requirement": "K23.integrated"}, tmp_path)          # oversight
    assert F.allowed({"kind": "gap", "outputs": ["creator/reasoning.py", "tests/x.py"]}, tmp_path)
    assert F.allowed(SimpleNamespace(kind="gap", component="K09"), tmp_path)             # objects work too
    assert not F.allowed({"kind": "gap", "component": "K22"}, tmp_path)                   # swarm: not thinking
    assert not F.allowed({"kind": "gap", "component": "K14", "outputs": ["creator/kernel.py"]}, tmp_path)
    assert not F.allowed({}, tmp_path)


def test_the_focus_file_is_not_writable_by_a_work_package() -> None:
    from creator import sandbox as SB
    assert SB.is_protected("state/creator/focus.json")
