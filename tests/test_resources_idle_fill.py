"""h43 (3 Oct 2026): with saturation judged on the scheduled cores (h38), CPU is "full" most of the day. The thinking fillers (IDLE class)
must still start then - bounded only by oversubscription and RAM - while BELOW_NORMAL CPU-heavy work (cycles) is deferred."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import resources as RS

FULL = RS.Machine(14, 33.8, 16.0, 100.0, 100.0, 100.0, 2)                      # 12 scheduled cores pegged, half the RAM free


def test_saturated_cpu_still_admits_fillers_but_defers_cycles() -> None:
    fil, cyc = RS.profile("filler"), RS.profile("cycle:EFF")
    assert fil.cpu_heavy and RS.priority_class("filler") == "IDLE"              # the case the rule is about
    assert RS.decide(fil, FULL, running_cores=10.0, running=4).allow
    d = RS.decide(cyc, FULL, running_cores=10.0, running=4)
    assert not d.allow and d.prefer == "ram"


def test_fillers_stay_bounded_by_oversubscription_and_ram() -> None:
    fil = RS.profile("filler")
    assert not RS.decide(fil, FULL, running_cores=27.5, running=20, oversub=2.0).allow          # 28.5 > 2 x 14
    tight = RS.Machine(14, 33.8, 0.5, 100.0, 100.0, 100.0, 2)
    assert not RS.decide(RS.Profile("filler", 1.0, 0.5, 60.0), tight, 4.0, 2).allow             # below the RAM floor


def test_admitter_on_todays_machine_keeps_starting_fillers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RS, "_CPU_BUSY_AT", {})
    t = [0.0]
    adm = RS.Admitter(tmp_path, machine=lambda: FULL, clock=lambda: t[0], sample_every_s=10)
    for _ in range(8):
        t[0] += 10
        f, c = adm.admit("filler", 4, 8.0), adm.admit("cycle:EFF", 4, 8.0)
    assert f.allow and not c.allow and adm.wants_ram_work(4)                    # thinking continues, RAM work fills beside it
