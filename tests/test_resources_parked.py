"""h38 (3 Oct 2026): saturation is judged on the cores the OS schedules on. This PC's two low-power E-cores stay idle while the other 12
are pegged, so the machine-wide figure stopped at 85.7% and "CPU saturated" never fired."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import resources as RS

TODAY = [100.0] * 12 + [0.1, 0.1]


@pytest.fixture(autouse=True)
def _fresh_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RS, "_CPU_BUSY_AT", {})


def test_cores_idle_next_to_saturated_ones_are_parked_after_the_window() -> None:
    assert RS.usable_cpu(TODAY, 0.0) == (pytest.approx(85.73, abs=0.01), 0)       # first sight: every core counts
    assert RS.usable_cpu(TODAY, 30.0)[1] == 0                                    # 30 s idle is not yet parked
    pct, parked = RS.usable_cpu(TODAY, 61.0)
    assert parked == 2 and pct == 100.0 and pct >= RS.SATURATED_PCT


def test_idle_cores_on_a_machine_with_room_are_not_parked() -> None:
    half = [50.0] * 12 + [0.0, 0.0]                                              # idle cores because there is no work: real capacity
    RS.usable_cpu(half, 0.0)
    assert RS.usable_cpu(half, 120.0) == (pytest.approx(600 / 14), 0)


def test_a_core_that_wakes_up_counts_again() -> None:
    RS.usable_cpu(TODAY, 0.0)
    assert RS.usable_cpu(TODAY, 61.0)[1] == 2
    woke = [100.0] * 13 + [0.1]
    assert RS.usable_cpu(woke, 62.0)[1] == 1
    assert RS.usable_cpu([100.0] * 14, 63.0) == (100.0, 0)


def test_admitter_defers_cpu_heavy_work_on_todays_machine(tmp_path: Path) -> None:
    t = [0.0]
    m = RS.Machine(14, 33.8, 16.0, 85.7, 85.7, 100.0, 2)
    adm = RS.Admitter(tmp_path, machine=lambda: m, clock=lambda: t[0], sample_every_s=10)
    for _ in range(8):
        t[0] += 10
        d = adm.admit("cycle:EFF", 4, 8.0)
    assert not d.allow and d.prefer == "ram"
    assert adm.busy_cores() == pytest.approx(12.0)                              # 100% of the 12 scheduled cores, not 14
    assert RS._rows(tmp_path / RS.SAMPLES_FILE)[-1]["cpu_usable"] == 100.0
    plain = RS.Admitter(tmp_path / "b", machine=lambda: RS.Machine(14, 33.8, 16.0, 85.7, 85.7), clock=lambda: t[0], sample_every_s=10)
    for _ in range(8):
        t[0] += 10
        d2 = plain.admit("cycle:EFF", 4, 8.0)
    assert d2.allow                                                              # without per-core figures: the old machine-wide rule
