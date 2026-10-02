"""Constraint loop: every metric from planted synthetic records, the ranking picks the planted dominant loss, a shift is detected
after a planted improvement, a twice-rejected remedy is not proposed again, nothing raises on missing files."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Optional

from creator import constraints as CON
from creator import curriculum as CUR
from creator import goals as GO

NOW = dt.datetime(2026, 10, 2, 12, 0, 0)


def cycle(state: Path, pkg: str, outcome: str, seconds: float, hours_ago: float, reason: str = "", by: str = "", changed: Optional[list[str]] = None,
          eval_s: float = 0.0) -> None:
    d = state / "cycles" / pkg
    d.mkdir(parents=True, exist_ok=True)
    (d / "cycle.json").write_text(json.dumps({"package": pkg, "outcome": outcome, "reason": reason, "seconds": seconds, "requirement": "EFF.size",
                                              "details": {"worker": {"by": by}}}), encoding="utf-8")
    if changed is not None or eval_s:
        (d / "evaluation.json").write_text(json.dumps({"changed": {p: "M" for p in (changed or [])},
                                                       "build": {"steps": [{"name": "x", "seconds": eval_s}]}}), encoding="utf-8")
    t = (NOW - dt.timedelta(hours=hours_ago)).timestamp()
    os.utime(d / "cycle.json", (t, t))


def lesson(state: Path, i: int, solver: str, verdict: str, adopted: Optional[bool], hours_ago: float = 1.0, kind: str = "shrink") -> None:
    les = CUR.Lesson(f"L{i}", f"CP{i}", "K01", kind, "obj", solver=solver, adopted=adopted, verdict=verdict,
                     at=(NOW - dt.timedelta(hours=hours_ago)).isoformat(timespec="seconds"))
    CUR.LessonLog(state / "lessons.jsonl").add(les)


def by_name(rep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in rep["ranked"] + rep["drivers"]}


def test_every_metric_is_computed_from_planted_records(tmp_path: Path) -> None:
    s = tmp_path
    cycle(s, "CP1", "ADOPTED", 1800, 2, changed=["creator/kernel.py"], eval_s=120)
    cycle(s, "CP2", "REJECTED", 1800, 3, reason="claim REGRESSION: guard")
    cycle(s, "CP3", "CANCELLED", 1000, 4, reason="pulled back before evaluation (RAM tight)")
    cycle(s, "CP4", "ERROR", 1000, 5, reason="TimeoutExpired: Command ['git'] timed out after 120.0 seconds")
    cycle(s, "CP5", "ADOPTED", 3600, 30, changed=["tests/t.py"])                    # previous window
    for i in range(10):
        lesson(s, i, "nupen-model-v1", "not claimed done: model call failed: TimeoutError" if i < 6 else "", None if i >= 6 else False)
    lesson(s, 20, "nupen-model-v1", "REJECTED: claim REGRESSION", False)
    lesson(s, 21, "nupen-model-v1", "ADOPTED", True)
    (s / "swarm_log.jsonl").write_text("\n".join(json.dumps({"round": i, "packages": p, "peak_parallel": 3, "pulled_back": 1, "at": (NOW - dt.timedelta(hours=i + 1)).isoformat()})
                                                for i, p in enumerate([0, 1, 4, 5])), encoding="utf-8")
    (s / "nupen_service.log").write_text("2026-10-02T10:00:00 supervisor down\n2026-10-02T11:00:00 supervisor up pid=1\n", encoding="utf-8")
    (s / "AUDIT.json").write_text(json.dumps({"audit": {"counts": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 0}}}), encoding="utf-8")
    rep = CON.measure_all(s, NOW, 24.0)
    m = by_name(rep)
    total = 1800 + 1800 + 1000 + 1000 + 1800 * 0 + 0
    assert rep["headline"]["current"]["adopted"] == 1 and abs(rep["headline"]["current"]["per_hour"] - 1 / 24) < 1e-9
    assert rep["headline"]["current"]["machinery_per_hour"] > 0 and rep["headline"]["previous"]["machinery_per_hour"] == 0
    assert rep["headline"]["meta_rate_per_day"] == 0.0                                 # 1 adopted in both windows
    secs = 1800 + 1800 + 1000 + 1000
    assert abs(m["rejected_cycles"]["value"] - 1800 / secs) < 1e-3
    assert abs(m["ram_pullbacks"]["value"] - 1000 / secs) < 1e-3
    assert abs(m["infra_timeouts"]["value"] - 1000 / secs) < 1e-3 and total == secs
    assert m["rejected_cycles"]["detail"]["by_class"] == {"regression": 1}
    assert m["cycle_time"]["value"] == 30.0 and m["cycle_time"]["detail"]["dominant_stage"] == "worker_sandbox_tests_merge"
    assert m["evaluation_cost"]["value"] == 120.0
    assert m["work_supply"]["value"] == 0.5 and m["work_supply"]["detail"]["rounds"] == 4
    assert abs(m["service_downtime"]["value"] - 1 / 24) < 1e-3
    assert m["audit_gaps"]["loss"] == 0.3
    assert m["learning_signal"]["detail"]["attempts"] == 12 and m["learning_signal"]["detail"]["measured"] == 2
    assert abs(m["learning_signal"]["value"] - 10 / 12) < 1e-3
    assert abs(m["student_skill"]["loss"] - 1 / 12) < 1e-3
    assert abs(m["model_timeouts"]["value"] - 6 / 12) < 1e-3 and m["model_timeouts"]["parent"] == "learning_signal"
    assert abs(m["pending_no_verdict"]["value"] - 4 / 12) < 1e-3
    assert m["recursion_idle"]["loss"] == 0.25


def test_ranking_picks_the_planted_dominant_loss(tmp_path: Path) -> None:
    s = tmp_path
    for i in range(20):                                                               # 95% of student attempts never measured
        lesson(s, i, "nupen-model-v1", "not claimed done: model call failed: TimeoutError", False)
    lesson(s, 99, "nupen-model-v1", "ADOPTED", True)
    cycle(s, "CP1", "ADOPTED", 1000, 1)
    cycle(s, "CP2", "CANCELLED", 100, 1, reason="pulled back before evaluation (RAM tight)")
    rep = CON.measure_all(s, NOW, 24.0)
    assert rep["top"] == "learning_signal" and rep["ranked"][0]["remedy"] == "goal"
    assert all(rep["ranked"][i]["score"] >= rep["ranked"][i + 1]["score"] for i in range(len(rep["ranked"]) - 1))
    assert "model_timeouts" not in [m["name"] for m in rep["ranked"]]                 # drivers are shown, not ranked
    assert rep["drivers"][0]["name"] == "model_timeouts"
    other = s / "other"                                                               # a different planted dominant loss: RAM pull-backs
    cycle(other, "CP1", "ADOPTED", 100, 1)
    cycle(other, "CP2", "CANCELLED", 100000, 1, reason="pulled back before evaluation (RAM tight)")
    assert CON.measure_all(other, NOW, 24.0)["top"] == "ram_pullbacks"


def test_a_shift_is_detected_after_a_planted_improvement(tmp_path: Path) -> None:
    s = tmp_path
    for i in range(10):
        lesson(s, i, "nupen-model-v1", "not claimed done: model call failed: TimeoutError", False)
    first = CON.run(s, NOW, 24.0)
    assert first["top"] == "learning_signal" and first["messages"] == []
    assert first["act"]["proposal"].startswith("GP-")
    for i in range(10, 40):                                                           # the remedy is adopted: attempts are now measured
        lesson(s, i, "nupen-model-v1", "REJECTED: claim REGRESSION", False)
    second = CON.run(s, NOW + dt.timedelta(minutes=5), 24.0)
    assert second["top"] != "learning_signal"
    assert f"constraint shifted from learning_signal to {second['top']}" in second["messages"]
    assert any("previous constraint learning_signal" in x and "(moved)" in x for x in second["messages"])
    assert [r["text"] for r in CON.history(s) if r.get("event") == "note" and "shifted" in r["text"]]


def test_a_twice_rejected_remedy_is_not_proposed_again_without_new_evidence(tmp_path: Path) -> None:
    s = tmp_path
    for i in range(10):
        lesson(s, i, "nupen-model-v1", "not claimed done: model call failed: TimeoutError", False)
    cycle(s, "CP1", "ADOPTED", 1000, 1)
    rep = CON.measure_all(s, NOW, 24.0)
    loss = rep["ranked"][0]["loss"]
    for _ in range(2):
        res = CON.act(s, rep)
        assert res["constraint"] == "learning_signal"
        GO.reject(s, res["proposal"], "not what is limiting us")
        rep["ranked"][0]["loss"] = round(rep["ranked"][0]["loss"] - 0.0001, 4)       # a different key each round, same evidence
    assert CON.rejected_count(s, "learning_signal")[0] == 2
    res = CON.act(s, rep)
    assert res["constraint"] != "learning_signal" and res["skipped"][0]["name"] == "learning_signal" and "no new evidence" in res["skipped"][0]["why"]
    assert len([p for p in GO.listing(s) if p["source"] == "constraint" and p["status"] == "PENDING"]) <= 1
    rep["ranked"][0]["loss"] = round(loss * 1.5, 4)                                  # new evidence: the loss grew by half
    assert CON.suppressed(s, rep["ranked"][0]) == ""
    CON.reject_remedy(s, "cycle_time", 0.3, "r1")
    CON.reject_remedy(s, "cycle_time", 0.3, "r2")
    assert CON.suppressed(s, {"name": "cycle_time", "loss": 0.3}) != ""


def test_it_never_raises_on_missing_or_broken_files(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    rep = CON.measure_all(missing, NOW, 24.0)
    assert rep["top"] in ("", "recursion_idle") and not rep["errors"]
    assert CON.run(missing, NOW)["act"] is not None
    (tmp_path / "lessons.jsonl").write_text("not json\n{\n", encoding="utf-8")
    (tmp_path / "swarm_log.jsonl").write_text("garbage", encoding="utf-8")
    (tmp_path / "cycles" / "CPX").mkdir(parents=True)
    (tmp_path / "cycles" / "CPX" / "cycle.json").write_text("{bad", encoding="utf-8")
    assert isinstance(CON.measure_all(tmp_path, NOW, 24.0)["ranked"], list)
    assert CON.maybe_run(tmp_path / "unwritable\0name", NOW) is None
    first = CON.maybe_run(tmp_path, NOW)
    assert first is not None and CON.maybe_run(tmp_path, NOW + dt.timedelta(minutes=10)) is None      # at most hourly
    assert CON.maybe_run(tmp_path, NOW + dt.timedelta(hours=2)) is not None
    assert CON.top_constraints(tmp_path / "never_measured") == []
