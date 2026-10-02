"""Regressions for the second independent validation round (each failed before its fix)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from creator import agents as A
from creator import autotune as AT
from creator import design as D
from creator import generator as G
from creator import localworker as LW
from creator import memory as MEM
from creator import objective as O
from creator import research as R


def test_run_tests_timeout_is_a_failed_attempt_not_an_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)
    monkeypatch.setattr(subprocess, "run", boom)
    ok, out = LW.run_tests(tmp_path, ["tests/test_x.py"], timeout=1)
    assert ok is False and "timed out" in out


def test_a_corrupt_history_line_is_skipped_and_counted(tmp_path: Path) -> None:
    hist = tmp_path / "h.jsonl"
    cfg = G.WorkerConfig()
    prop = AT.propose(cfg, hist)
    assert prop is not None
    _, cand = prop
    hist.write_text(json.dumps({"candidate": cand.digest()}) + "\n{not json\n[1]\n", encoding="utf-8")
    assert AT.tried(hist) == {cand.digest()}
    assert AT.read_history(hist)[1] == 2
    nxt_prop = AT.propose(cfg, hist)
    assert nxt_prop is not None
    nxt = nxt_prop[1]
    assert nxt.digest() != cand.digest()


def test_budget_error_does_not_point_at_the_disabled_env_switch(tmp_path: Path) -> None:
    src = Path(A.__file__).read_text(encoding="utf-8")
    assert "set CREATOR_AGENT_CALLS=1" not in src
    assert "--setting-sources" in src and "no user/project settings" not in src


def test_select_keeps_same_named_options_distinct() -> None:
    good = D.DesignOption("X", assumptions=("a",), failure_modes=("f",), cost=0.1, risk=0.1, benefit=0.9)
    bad = D.DesignOption("X", cost=0.9, risk=0.9, benefit=0.1)
    dec = D.select("q", [bad, good])
    assert dec.selected == good
    (rej, reason), = dec.rejected
    assert rej == bad and D.score(bad) != D.score(good) and f"{D.score(bad):.3f}" in reason


def test_generate_questions_tolerates_non_numeric_importance() -> None:
    qs = R.generate_questions([{"description": "a", "importance": "high"}, {"description": "b", "importance": None},
                               {"description": "c", "importance": float("nan")}])
    assert [q.importance for q in qs] == [0.5, 0.5, 0.5]


def test_memory_load_keeps_saved_seq_across_corrupt_lines(tmp_path: Path) -> None:
    m = MEM.Memory()
    for i in range(3):
        m.remember(MEM.KINDS[0] if isinstance(MEM.KINDS, (list, tuple)) else next(iter(MEM.KINDS)), f"s{i}")
    p = tmp_path / "m.jsonl"
    m.save(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join([lines[0], "garbage", lines[1], lines[2]]) + "\n", encoding="utf-8")
    assert [e.seq for e in MEM.load(p).entries] == [1, 2, 3]
    p.write_text("\n".join([lines[0], "garbage", lines[2]]) + "\n", encoding="utf-8")
    assert [e.seq for e in MEM.load(p).entries] == [1, 3]


def test_component_depends_covers_k17_to_k26_acyclic() -> None:
    for n in range(17, 27):
        assert f"K{n:02d}" in O.COMPONENT_DEPENDS
    seen: dict[str, int] = {}

    def visit(k: str) -> None:
        assert seen.get(k) != 1, f"cycle at {k}"
        if k in seen:
            return
        seen[k] = 1
        for d in O.COMPONENT_DEPENDS.get(k, ()):
            visit(d)
        seen[k] = 2
    for k in O.COMPONENT_DEPENDS:
        visit(k)


def test_recursion_docstring_names_k24() -> None:
    from creator import recursion
    assert recursion.__doc__.startswith("Creator K24")
