"""creator.slowpath: operation timer + slow-path detector (owner, 3 Oct 2026: a 44-minute thought must be detected and made ~100x faster)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import slowpath as SP


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_judge_spike_budget_long_and_fine() -> None:
    assert SP.judge("x", 5.0, [0.1] * 10) is None                                   # under MIN_FLAG_S: never worth a work item
    f = SP.judge("x", 400.0, [30.0] * 10, budget=1e9)
    assert f is not None and f["kind"] == "spike" and f["factor"] == pytest.approx(13.3)
    assert SP.judge("x", 200.0, [30.0] * 10, budget=1e9) is None                    # 6.7x its median: not a spike
    assert SP.judge("x", 400.0, [30.0] * 3, budget=1e9) is None                     # too few earlier runs to judge a spike
    f = SP.judge("x", 100.0, [], budget=60.0)
    assert f is not None and f["kind"] == "over_budget" and f["factor"] == pytest.approx(1.7)
    f = SP.judge("y", 2600.0, [980.0, 1060.0])                                       # unbudgeted, median >= LONG_S: minutes nobody asked for
    assert f is not None and f["kind"] == "long"
    assert SP.judge("y", 300.0, [200.0, 250.0]) is None


def test_record_flags_once_and_proposes_one_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for _ in range(6):
        assert SP.record(tmp_path, "op.a", 40.0, 39.0) is None
    flag = SP.record(tmp_path, "op.a", 2640.0, 2600.0)                               # the 44-minute thought
    assert flag is not None and flag["kind"] == "spike" and flag["proposal"].startswith("GP-")
    assert SP.record(tmp_path, "op.a", 2640.0, 2600.0) is None                       # one flag per op and kind per day
    assert len(_rows(tmp_path / SP.OPS_FILE)) == 8
    assert [r["op"] for r in _rows(tmp_path / SP.FLAGS_FILE)] == ["op.a"]
    from creator import goals as GO
    props = [p for p in GO.listing(tmp_path) if p["source"] == "slowpath"]
    assert len(props) == 1 and props[0]["key"] == "slowpath:op.a" and props[0]["status"] == "PENDING"
    assert props[0]["metric"]["before"]["wall_s"] == 2640.0
    monkeypatch.setattr(SP, "_recent_flag", lambda *a: False)                        # even a later flag ...
    again = SP.record(tmp_path, "op.a", 2640.0)
    assert again is not None and again.get("proposal") is None                       # ... never duplicates the proposal


def test_timed_records_wall_cpu_and_errors_and_never_raises_itself(tmp_path: Path) -> None:
    with SP.timed(tmp_path, "op.b"):
        sum(range(10000))
    with pytest.raises(ValueError):
        with SP.timed(tmp_path, "op.b"):
            raise ValueError("work failed")                                          # the work's error propagates, timing is kept
    rows = _rows(tmp_path / SP.OPS_FILE)
    assert [r["op"] for r in rows] == ["op.b", "op.b"] and "cpu_s" in rows[0] and rows[1]["error"] == "ValueError"
    assert SP.record(tmp_path / "nul\0bad", "op.c", 1.0) is None                      # an unwritable state never raises
    assert SP.wrap(tmp_path, "op.d", lambda: 7)() == 7


def test_history_reads_only_the_tail_and_report_ranks_by_cpu_hours(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(30):
        SP.record(tmp_path, "cheap", 1.0, 1.0)
        SP.record(tmp_path, "dear", 100.0, 90.0)
    assert SP.history(tmp_path, "dear")[-1] == 100.0 and len(SP.history(tmp_path, "dear")) == 30
    monkeypatch.setattr(SP, "TAIL_BYTES", 400)
    assert 0 < len(SP.history(tmp_path, "dear")) < 30                                 # bounded read, first partial line dropped
    rep = SP.report(tmp_path)
    assert [r["op"] for r in rep] == ["dear", "cheap"] and rep[0]["cpu_h_per_day"] == pytest.approx(0.75)


def test_thinking_filler_jobs_are_timed(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from creator import thinkfill as TF
    calls: list[str] = []
    reg = SimpleNamespace(get=lambda name: {
        "drillsources": SimpleNamespace(drill_filler=lambda *a, **k: (lambda: (lambda: calls.append("drill")))),
        "thinking": SimpleNamespace(run=lambda st: calls.append("run")),
        "device": SimpleNamespace(settings=lambda: {}),
    }[name], optional=lambda name: None)
    last = {"run": 0.0, "blueprint": 1e18}
    nxt = TF.make(reg, tmp_path, tmp_path, last, 600.0, 3600.0, 1800.0, lambda: calls.append("live"))
    nxt()()
    nxt()()
    assert calls == ["live", "run", "drill"]
    assert [r["op"] for r in _rows(tmp_path / SP.OPS_FILE)] == ["think.run", "think.drills"]
