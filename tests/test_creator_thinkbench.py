"""The thinking benchmark: statistics, parsing, the frozen item set (hash), scoring and the paired comparison, and one end-to-end run on a synthetic
state with a scripted model (no server, no real state)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from creator import thinkbench as TB


def test_spearman_handles_ties_and_constants() -> None:
    assert TB.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert TB.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert TB.spearman([1, 2, 3], [5, 5, 5]) is None
    assert TB.ranks([3, 1, 1, 2]) == [4.0, 1.5, 1.5, 3.0]


def test_intervals() -> None:
    w = TB.wilson(5, 10)
    assert w[0] == 0.5 and w[1] is not None and w[2] is not None and w[1] < 0.5 < w[2]
    assert TB.wilson(0, 0) == [None, None, None]
    m = TB.mean_ci([1.0, 1.0, 1.0, 1.0])
    assert m == [1.0, 1.0, 1.0]
    assert TB.mean_ci([2.0])[1] is None
    assert TB.tcrit(1000) == 1.96 and TB.tcrit(2) == pytest.approx(4.30)


def test_parsers() -> None:
    assert TB.parse_choice("thinking...\nANSWER: C") == 2
    assert TB.parse_choice("answer = (b)") == 1
    assert TB.parse_choice("no idea") is None
    assert TB.parse_rank("RANK: 3, 1", 4) == [3, 1, 2, 4]
    assert TB.parse_rank("RANK: 9, 9", 4) is None
    assert TB.parse_rank("nothing", 3) is None


def _write_state(state: Path, n: int = 40) -> None:
    state.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    klog = []
    for i in range(n):
        pid = f"CP{i:04d}"
        ts = f"2026-10-01T{i // 6:02d}:{(i % 6) * 10:02d}:00+00:00"
        rows.append({"id": f"WP-{i}", "rtype": "WorkPackage", "provenance": {"timestamp": ts},
                     "data": {"package_id": pid, "why_it_exists": f"gap K{i % 5}.exists missing module m{i}", "objective": f"module m{i} exists",
                              "outputs": [f"creator/m{i}.py", f"tests/test_creator_m{i}.py"]}})
        rows.append({"id": f"SO-{i}", "rtype": "StrategyOutcome", "provenance": {"timestamp": f"2026-10-01T{i // 6:02d}:{(i % 6) * 10 + 5:02d}:00+00:00"},
                     "data": {"subject_id": f"WP-{i}"}})
        klog.append({"package": pid, "requirement": f"K{i % 5}.exists", "outcome": "ADOPTED" if i % 3 == 0 else "REJECTED", "seconds": 120.0, "usd": 0.0})
    for c in range(6):
        rows.append({"id": f"CAP-{c}", "rtype": "Capability", "provenance": {"timestamp": "2026-10-01T00:00:00+00:00"},
                     "data": {"component": f"K{c:02d}", "name": f"capability number {c}",
                              "description": f"thing {c}: modules creator/m{c}.py; tests tests/test_creator_m{c}.py; floor 10"}})
    (state / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (state / "kernel_log.jsonl").write_text("\n".join(json.dumps(r) for r in klog), encoding="utf-8")
    snaps = [{"at": f"2026-10-01T{h:02d}:00:00", "event": "snapshot", "top": f"c{h % 3}",
              "ranked": [{"name": f"c{j}", "value": j + h} for j in range(5)]} for h in range(1, 7)]
    (state / "constraints.jsonl").write_text("\n".join(json.dumps(r) for r in snaps), encoding="utf-8")


class Scripted:
    """A model that answers every multiple-choice question 'A', ranks candidates as shown, and gives probability 0.3."""

    def __init__(self, i: int) -> None:
        self.i = i

    def __enter__(self) -> "Scripted":
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def chat(self, msgs: Any, max_tokens: int = 0, temperature: float = 0.0, seed: int = 0, timeout: float = 0.0) -> str:
        u = msgs[-1]["content"]
        if "Candidates:" in u:
            return "RANK: 1, 2, 3, 4, 5, 6"
        if "Which" in u or "which" in u:
            return "ANSWER: A"
        return "PROBABILITY: 0.30"


def test_freeze_is_hashed_and_tamper_evident(tmp_path: Path) -> None:
    st = tmp_path / "state"
    _write_state(st)
    f1 = TB.freeze(st, tmp_path / "norepo", tmp_path / "owner")
    assert f1["a"] and f1["c"] and f1["d"] and f1["hash"]
    assert {q["kind"] for q in f1["c"]} >= {"test_file", "capability", "adopted", "constraint"}
    assert all(0 <= q["answer"] < 4 for q in f1["c"])
    f2 = TB.freeze(st, tmp_path / "norepo", tmp_path / "owner")          # a second freeze returns the stored set, it never re-chooses
    assert f2["hash"] == f1["hash"]
    p = TB.bench_dir(st) / "items.json"
    body = json.loads(p.read_text(encoding="utf-8"))
    body["c"][0]["answer"] = (body["c"][0]["answer"] + 1) % 4
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        TB.load_frozen(st)


def test_prompts_carry_no_answer_and_no_future(tmp_path: Path) -> None:
    st = tmp_path / "state"
    _write_state(st)
    f = TB.freeze(st, tmp_path / "norepo", tmp_path / "owner")
    for q in f["c"]:
        if q["kind"] == "constraint":
            assert q["id"].split(":", 2)[2] in q["prompt"]               # the snapshot's own time is asked about, earlier tops are listed
    for w in f["d"]:
        t0 = min(c["text"] for c in w["candidates"])
        assert t0 and w["context"].startswith("Time:")
        assert len(w["relevance"]) == TB.WINDOW


def test_end_to_end_with_a_scripted_model_and_comparison(tmp_path: Path) -> None:
    st = tmp_path / "state"
    _write_state(st)
    f = TB.freeze(st, tmp_path / "norepo", tmp_path / "owner")
    res = TB.run(st, tmp_path / "norepo", tmp_path / "owner", f, llm_factory=Scripted, workers=2)
    parts = res["parts"]
    assert set(parts) == {"a_prediction", "b_judgment", "c_reasoning", "d_planning"}
    c = parts["c_reasoning"]
    assert c["n"] == len(f["c"]) and c["baseline"] == 0.25
    assert c["score"] == pytest.approx(sum(1 for q in f["c"] if q["answer"] == 0) / len(f["c"]), abs=1e-3)
    assert parts["d_planning"]["n"] >= 1 and parts["d_planning"]["baseline"] == 0.0
    assert res["items_hash"] == f["hash"] and res["versions"]["local_model"]
    same = TB.compare(res, res)                                          # the same run against itself: zero change, never "significant"
    assert same["same_items"] and all(v["change"] == 0.0 and "SIGNIFICANTLY" not in v["verdict"] for v in same["parts"].values())
    better = json.loads(json.dumps(res))
    for k in better["parts"]["c_reasoning"]["per_item"]:
        better["parts"]["c_reasoning"]["per_item"][k] = 1.0
    cmp = TB.compare(res, better)
    assert cmp["parts"]["c_reasoning"]["change"] is not None and cmp["parts"]["c_reasoning"]["change"] >= 0
    p = TB.save(st, res, "baseline")
    assert p.parent == TB.bench_dir(st) and json.loads(p.read_text(encoding="utf-8"))["items_hash"] == f["hash"]


def test_worse_is_flagged_for_lower_is_better_parts() -> None:
    base = {"higher_is_better": False, "per_item": {str(i): 0.2 for i in range(20)}, "score": 0.2, "baseline": 0.25, "n": 20}
    now = {"higher_is_better": False, "per_item": {str(i): 0.2 + 0.1 * (1 + (i % 2) * 0.1) for i in range(20)}, "score": 0.3, "baseline": 0.25, "n": 20}
    r = TB.compare_part(base, now)
    assert r["verdict"] == "SIGNIFICANTLY WORSE" and r["change"] is not None and r["change"] < 0
    assert not math.isnan(r["change"])
