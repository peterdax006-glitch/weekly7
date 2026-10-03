"""scripts/nupen_monitor.py on planted logs: each WASTE alert fires when its condition is planted and stays quiet otherwise."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nupen_monitor", ROOT / "scripts" / "nupen_monitor.py")
MON = importlib.util.module_from_spec(spec)                                  # type: ignore[arg-type]
spec.loader.exec_module(MON)                                                 # type: ignore[union-attr]

NOW = dt.datetime(2026, 10, 2, 12, 0, 0).timestamp()


def iso_utc(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec="seconds")


def local(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


class Plant:
    def __init__(self, state: Path) -> None:
        self.state = state
        self.n = 0
        self.rows: list[dict[str, Any]] = []

    def plan(self, at: float, target: str, objective: str = "fix a gap") -> str:
        self.n += 1
        rid = f"WP-{self.n:04d}"
        self.rows.append({"rtype": "WorkPackage", "id": rid, "data": {"outputs": [target], "objective": objective},
                          "provenance": {"timestamp": iso_utc(at)}})
        return rid

    def verdict(self, at: float, rid: str, to: str = "IMPLEMENTED", reason: str = "adopted") -> None:
        self.n += 1
        self.rows.append({"rtype": "Transition", "id": f"TRN-{self.n:04d}", "data": {"subject_id": rid, "to_state": to, "reason": reason},
                          "provenance": {"timestamp": iso_utc(at)}})

    def write(self) -> None:
        (self.state / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in self.rows) + "\n", encoding="utf-8")

    def svc(self, *lines: str) -> None:
        (self.state / "nupen_service.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def status(self, **kw: Any) -> None:
        base = {"at": "11:59:00", "running": 3, "active": 3, "waiting_for_teacher": 0, "queued": 0, "fillers": 0, "planned": 3,
                "free_gb": 10.0, "floor_gb": 1.0, "limit": "none (planning next)"}
        base.update(kw)
        with (self.state / "swarm_service.log").open("a", encoding="utf-8") as f:
            f.write("STATUS " + json.dumps(base) + "\n")


@pytest.fixture()
def pl(tmp_path: Path) -> Plant:
    p = Plant(tmp_path)
    up = NOW - 3 * 3600
    p.svc(f"{local(up)} supervisor up pid=1", f"{local(up)} swarm started pid=2")
    return p


def run(pl: Plant, cpu: float | None = 90.0, hist: Any = None) -> tuple[str, dict[str, str]]:
    pl.write()
    health, alerts = MON.evaluate(pl.state, NOW, cpu, (10.0, 32.0), hist, MON.LedgerReader(pl.state / "ledger.jsonl"))
    return health, {c: m for c, m in alerts}


def test_a_healthy_run_has_a_health_line_and_no_alerts(pl: Plant) -> None:
    for i in range(4):
        rid = pl.plan(NOW - 3000 + i, f"creator/m{i}.py")
        pl.verdict(NOW - 1500 + i, rid)
    pl.plan(NOW - 100, "creator/x.py", "shrink creator/x.py without losing capability")
    pl.status()
    (pl.state / "lm_train.log").write_text("step 1 loss 2 lr 1 tok/s 1000 elapsed 1m\nstep 2 loss 2 lr 1 tok/s 2000 elapsed 2m\n", encoding="utf-8")
    health, alerts = run(pl)
    assert alerts == {}
    assert "verdicts 4/h 4/24h" in health and "plans 5/h" in health and "dev 80% eff 20%" in health and "lm 1500 tok/s" in health
    assert "run 3" in health and "alerts 0" in health


def test_an_interrupted_release_is_not_a_verdict(pl: Plant) -> None:
    rid = pl.plan(NOW - 7200, "creator/a.py")
    pl.verdict(NOW - 60, rid, "FAILED", "interrupted: swarm restarted, no worker survives")
    health, _ = run(pl)
    assert "verdicts 0/h 0/24h" in health


def test_no_verdict_in_90_minutes_alerts_only_while_the_swarm_is_up(pl: Plant) -> None:
    rid = pl.plan(NOW - 3 * 3600, "creator/a.py")
    pl.verdict(NOW - 2.5 * 3600, rid)                                        # last verdict 150 min ago, swarm up for 180 min
    _, alerts = run(pl)
    assert "NO_VERDICT" in alerts and "150 min" in alerts["NO_VERDICT"]
    pl.svc(f"{local(NOW - 3 * 3600)} supervisor up pid=1", f"{local(NOW - 3 * 3600)} swarm started pid=2", f"{local(NOW - 60)} swarm exited code=1")
    _, alerts = run(pl)
    assert "NO_VERDICT" not in alerts and "SWARM_DOWN" in alerts            # down is its own alert; a drain/stop silences that
    (pl.state / "NUPEN_DRAIN").write_text("x", encoding="utf-8")
    assert "SWARM_DOWN" not in run(pl)[1]


def test_the_same_target_planned_twice_in_24_hours(pl: Plant) -> None:
    pl.plan(NOW - 5 * 3600, "creator/kernel.py")
    pl.plan(NOW - 3600, "creator/kernel.py")
    pl.plan(NOW - 3600, "creator/other.py")
    pl.plan(NOW - 30 * 3600, "creator/other.py")                             # the old one is outside the window
    _, alerts = run(pl)
    assert "REPEAT_TARGET" in alerts and "creator/kernel.py 2x" in alerts["REPEAT_TARGET"] and "1 targets" in alerts["REPEAT_TARGET"]


def test_a_student_skipped_without_judged_evidence(pl: Plant) -> None:
    rows = [{"solver": "model-student", "task_kind": "shrink", "why": "model-student skipped on shrink: 5/6 attempts produced no measurable work, 0 adopted",
             "at": local(NOW - 600)},
            {"solver": "rule-student", "task_kind": "cover", "why": "gave up", "at": local(NOW - 600)},
            {"solver": "old", "task_kind": "x", "why": "gave up", "at": local(NOW - 3 * DAY)}]
    (pl.state / "skips.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _, alerts = run(pl)
    assert "rule-student on cover" in alerts["SKIP_UNJUDGED"] and "model-student" not in alerts["SKIP_UNJUDGED"]


DAY = 86400


def test_practice_never_run_alerts_after_six_hours_of_uptime(pl: Plant) -> None:
    pl.svc(f"{local(NOW - 7 * 3600)} supervisor up pid=1", f"{local(NOW - 7 * 3600)} swarm started pid=2")
    assert "PRACTICE_IDLE" in run(pl)[1]
    (pl.state / "practice_service.log").write_text(f"{local(NOW - 3600)} practice started (idle 700s)\n", encoding="utf-8")
    assert "PRACTICE_IDLE" not in run(pl)[1]


def test_a_traceback_in_the_current_run_alerts_but_one_before_the_last_start_does_not(pl: Plant) -> None:
    (pl.state / "swarm_service.log").write_text(
        "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: old run\n"
        "device: Windows/x64\nSTATUS {\"running\": 1}\n"
        "Traceback (most recent call last):\n  File \"y.py\", line 2\nKeyError: 'now'\n", encoding="utf-8")
    _, alerts = run(pl)
    assert "KeyError: 'now'" in alerts["TRACEBACK"] and "old run" not in alerts["TRACEBACK"]


def test_low_cpu_for_ten_minutes_while_work_exists_alerts(pl: Plant, tmp_path: Path) -> None:
    hist = MON.CpuHistory(tmp_path / "hist.json")
    for k in range(11):
        hist.add(NOW - 600 + k * 60, 40.0)
    pl.status()                                                              # running 3: work exists
    _, alerts = run(pl, cpu=40.0, hist=MON.CpuHistory(tmp_path / "hist.json"))
    assert "CPU_LOW" in alerts and "< 85%" in alerts["CPU_LOW"]
    # one busy sample in the window, or nothing to do: no alert
    h2 = MON.CpuHistory(tmp_path / "hist2.json")
    for k in range(11):
        h2.add(NOW - 600 + k * 60, 95.0 if k == 5 else 40.0)
    assert "CPU_LOW" not in run(pl, cpu=40.0, hist=h2)[1]
    h3 = MON.CpuHistory(tmp_path / "hist3.json")
    for k in range(11):
        h3.add(NOW - 600 + k * 60, 40.0)
    pl.status(running=0, queued=0, limit="no more work planned this round; filler budget used")
    assert "CPU_LOW" not in run(pl, cpu=40.0, hist=h3)[1]                    # idle because there is nothing to do is not waste


def test_a_short_cpu_history_is_not_enough_to_alert(pl: Plant, tmp_path: Path) -> None:
    hist = MON.CpuHistory(tmp_path / "h.json")
    hist.add(NOW - 120, 10.0)
    pl.status()
    assert "CPU_LOW" not in run(pl, cpu=10.0, hist=hist)[1]


def test_the_ledger_is_read_incrementally_and_a_half_written_line_waits(tmp_path: Path) -> None:
    f = tmp_path / "ledger.jsonl"
    row = json.dumps({"rtype": "WorkPackage", "id": "WP-1", "data": {"outputs": ["a.py"], "objective": "x"}, "provenance": {"timestamp": iso_utc(NOW)}})
    f.write_text(row + "\n" + row[:20], encoding="utf-8")
    r = MON.LedgerReader(f)
    r.refresh()
    assert len(r.plans) == 1
    f.write_text(row + "\n" + row.replace("WP-1", "WP-2") + "\n", encoding="utf-8")
    r.refresh()
    assert [p["id"] for p in r.plans] == ["WP-1", "WP-2"]


def test_main_once_prints_health_and_waste_lines(pl: Plant, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pl.plan(NOW - 100, "creator/k.py")
    pl.plan(NOW - 50, "creator/k.py")
    pl.write()
    rc = MON.main(["--once", "--state", str(pl.state), "--history", str(tmp_path / "h.json")])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("HEALTH ") and any(ln.startswith("WASTE REPEAT_TARGET") for ln in out) and rc == 1
