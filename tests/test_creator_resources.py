"""creator.resources: profiles, CPU/RAM-aware admission (fake machines), RAM work, and the resource_balance constraint."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from creator import constraints as C
from creator import registry as REG
from creator import resources as RS
from creator import swarm as W

BOX = RS.Machine(cores=14, total_gb=33.8, free_gb=16.0, cpu_pct=100.0)          # this PC today: CPU full, RAM half free


def test_registered_and_priority_classes() -> None:
    assert REG.get("resources") is RS
    assert RS.priority_class("cycle:EFF") == "BELOW_NORMAL" and RS.priority_class("practice") == "BELOW_NORMAL"
    assert RS.priority_class("filler") == "IDLE" and RS.priority_class("ram:warm_model_cache") == "IDLE"


def test_profiles_prior_then_measured(tmp_path: Path) -> None:
    assert RS.profile("cycle:EFF", tmp_path).source == "prior"
    for s in (100, 200, 300):
        RS.record_task(tmp_path, "cycle:EFF", s, s * 3.0, 0.2)
    p = RS.profiles(tmp_path)["cycle:EFF"]
    assert p.source == "measured" and p.cores == 3.0 and p.peak_gb == 0.2 and p.seconds == 200
    (tmp_path / "test_proc_mb.json").write_text(json.dumps({"avg_mb": 38.0}))
    assert abs(RS.profile("test_proc", tmp_path).peak_gb - 38 / 1024) < 1e-4


def test_cycle_json_seconds_feed_profiles(tmp_path: Path) -> None:
    for i, sec in enumerate((90, 110, 100)):
        d = tmp_path / "cycles" / f"c{i}"
        d.mkdir(parents=True)
        (d / "cycle.json").write_text(json.dumps({"outcome": "ADOPTED", "requirement": "EFF.coverage", "seconds": sec}))
    assert RS.profiles(tmp_path)["cycle:EFF"].seconds == 100


def test_cpu_saturated_defers_cpu_heavy_and_allows_ram_work() -> None:
    cyc, ram = RS.profile("cycle:EFF"), RS.profile("ram:warm_model_cache")
    d = RS.decide(cyc, BOX, running_cores=10.0, running=4)
    assert not d.allow and d.prefer == "ram"
    assert RS.decide(ram, BOX, 10.0, 4).allow


def test_oversubscription_is_bounded_by_factor() -> None:
    m = RS.Machine(14, 33.8, 16.0, 60.0)
    cyc = RS.profile("cycle:EFF")                                               # 2.5 cores
    assert RS.decide(cyc, m, 24.0, 10, oversub=2.0).allow                       # 26.5 <= 28
    assert not RS.decide(cyc, m, 26.0, 10, oversub=2.0).allow                   # 28.5 > 28
    assert RS.decide(cyc, m, 26.0, 10, oversub=3.0).allow


def test_ram_floor_defers_ram_heavy_prefers_cpu_light() -> None:
    m = RS.Machine(14, 33.8, 1.5, 40.0)
    d = RS.decide(RS.profile("lm_train"), m, 4.0, 2)
    assert not d.allow and d.prefer == "cpu"
    assert RS.decide(RS.profile("ram"), m, 4.0, 2).allow is False               # 2 GB task on 1.5 GB free
    assert RS.decide(RS.profile("test_proc"), m, 4.0, 2).allow                  # 40 MB task still fits


def test_never_defers_when_nothing_is_running() -> None:
    assert RS.decide(RS.profile("lm_train"), RS.Machine(14, 33.8, 0.1, 100.0), 0.0, 0).allow


def test_admitter_needs_sustained_saturation_and_samples(tmp_path: Path) -> None:
    t = [0.0]
    adm = RS.Admitter(tmp_path, machine=lambda: RS.Machine(14, 33.8, 16.0, 100.0), clock=lambda: t[0], sample_every_s=10)
    assert adm.admit("cycle:EFF", 4, 8.0).allow                                 # one instant of 100% is not "sustained"
    for _ in range(8):
        t[0] += 10
        d = adm.admit("cycle:EFF", 4, 8.0)
    assert not d.allow and d.prefer == "ram" and adm.wants_ram_work(4)
    assert len(RS._rows(tmp_path / RS.SAMPLES_FILE)) >= 5


def test_warm_model_cache_end_to_end(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * (3 << 20))
    r = RS.warm_model_cache(tmp_path, model, free_gb=20.0)
    assert r["bytes"] == 3 << 20 and r["saved_s_estimate"] >= 0 and r["warm_mb_s"] > 0
    assert RS.warm_model_cache(tmp_path, model, free_gb=20.0)["skipped"] == "warmed within 6 h"
    assert RS.ram_job(tmp_path) is None                                         # nothing left to do for six hours
    assert "ram:warm_model_cache" in RS.profiles(tmp_path)
    assert "RAM" in RS.warm_model_cache(tmp_path / "x", model, free_gb=1.0)["skipped"]
    assert RS.warm_model_cache(tmp_path, tmp_path / "none", free_gb=20.0)["skipped"] == "model file missing"


def test_ram_job_runs_the_warm(tmp_path: Path, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(RS, "warm_model_cache", lambda state: called.append(state) or {})
    job = RS.ram_job(tmp_path)
    assert job is not None
    job()
    assert called == [tmp_path]


def _samples(state: Path, rows: list[tuple[float, float]], at: float) -> None:
    for cpu, free in rows:
        RS._append(state / RS.SAMPLES_FILE, {"at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(at - 60)), "cpu": cpu, "free_gb": free,
                                             "total_gb": 33.8, "cores": 14})


def test_resource_balance_metric_and_goal_proposal(tmp_path: Path) -> None:
    now = dt.datetime.now()
    _samples(tmp_path, [(100, 17.0)] * 8 + [(50, 17.0)] * 2, now.timestamp())
    m = C.resource_balance_metric(tmp_path, now, 24.0)
    assert m.name == "resource_balance" and m.value == 0.8 and m.loss == 0.48 and m.remedy == "goal"
    assert C.measure_all(tmp_path, now)["top"] == "resource_balance"
    out = C.act(tmp_path, C.measure_all(tmp_path, now))
    assert out["constraint"] == "resource_balance" and out.get("proposal")
    from creator import goals as GO
    assert any(p["key"].startswith("constraint:resource_balance:") for p in GO.listing(tmp_path))


def test_resource_balance_reverse_and_empty(tmp_path: Path) -> None:
    now = dt.datetime.now()
    assert C.resource_balance_metric(tmp_path, now, 24.0).loss == 0.0
    _samples(tmp_path, [(30, 2.0)] * 4, now.timestamp())
    assert RS.balance_shares(tmp_path, 3600.0)["ram_full_cpu_idle"] == 1.0


class _Adm:
    def __init__(self, allow: bool) -> None:
        self.allow, self.seen = allow, []

    def admit(self, kind, running, cores=None):
        self.seen.append(kind)
        return RS.Decision(self.allow, "t")


def test_governor_admits_through_the_admitter_and_defaults_to_ram_only() -> None:
    g = W.Governor(free=lambda: 20.0, total=lambda: 32.0)
    assert RS.ok(g, "cycle:EFF", 3) and not RS.ram_thread(g, Path("."), 3, True, [])
    g.admit = _Adm(False)
    assert not RS.ok(g, "cycle:EFF", 3) and g.admit.seen == ["cycle:EFF"]


def test_cap_is_a_safety_valve_and_work_continues_past_64(tmp_path: Path) -> None:
    g = W.Governor(free=lambda: 20.0, total=lambda: 32.0)
    RS.attach(g, tmp_path, W.free_ram_gb)                                       # fake machine: untouched
    assert g.admit is None and g.max_workers == 32
    real = W.Governor(max_workers=64)
    RS.attach(real, tmp_path, W.free_ram_gb)                                    # the real machine: cap becomes 10 x logical cores
    assert real.admit is not None and real.max_workers >= 10 * real.admit.machine().cores
    light = RS.simulate(kinds=("test_proc",))                                   # 0.04 GB, 1 core each: stops where CPU is full (14 cores), not at a count
    assert 14 <= light["workers"] <= 20 and light["cpu_pct"] == 100.0
    import creator.resources as R
    R.PRIORS["tiny"] = (0.1, 0.05, 10.0)
    try:
        tiny = RS.simulate(kinds=("tiny",), steps=2000)
    finally:
        R.PRIORS.pop("tiny")
    assert 64 < tiny["workers"] <= 140                                          # past the old fixed cap, below the 10x safety valve
    assert tiny["workers"] == 140 or tiny["demand_cores"] <= 28.0


def test_simulation_fills_both_resources_and_beats_the_old_rule() -> None:
    new, old = RS.simulate(), RS.simulate(admission=False)
    assert new["cpu_pct"] >= 95.0 and new["ram_used_pct"] > old["ram_used_pct"]
    assert new["by_kind"].get("ram", 0) > 0                                     # RAM-specific work took the idle RAM
    assert new["demand_cores"] <= 2.0 * 14                                      # CPU oversubscription stays within the factor
    assert old["demand_cores"] > new["demand_cores"]                            # the old rule oversubscribed without bound until 64
