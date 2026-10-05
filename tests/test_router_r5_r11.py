"""R5 router + R11 resource-aware auto-config: fake resource readings, no servers, no models."""
from __future__ import annotations

from pathlib import Path

from creator import resources as R
from creator import router as RT
from creator.modelpool import SlotSpec, slot_pool

SPEEDS = {"C06": 58.0, "C17": 22.0, "MOE": 9.0}
EASY = {"id": "fn.easy", "family": "fn", "request": "Write f", "stub": 'def f(x):\n    """Double x."""\n', "n_visible": 2}
HARD = {"id": "fn.hard", "family": "fn", "request": "Write g " + "detail " * 200,
        "stub": 'def g(a, b, c):\n    """' + "long spec " * 60 + '"""\n', "n_visible": 1}
APP = {"id": "app.x", "family": "app", "request": "Add Catalog.search to minishop/catalog.py " * 3}


def mach(free=20.0, cpu=10.0, cores=14, total=32.0):
    return R.Machine(cores, total, free, cpu)


def test_classes_and_start_rung():
    r = RT.Router(SPEEDS)
    assert r.route(EASY)["class"] == "fn_s" and r.route(EASY)["start"] == "C06"
    assert r.route(HARD)["class"] in ("fn_m", "fn_l")
    a = r.route(APP)
    assert a["class"].startswith("app") and a["start"] == "MOE"


def test_pending_moe_not_routable_and_escalation():
    r = RT.Router({"C06": 58.0, "C17": 22.0, "MOE": None})
    x = r.route(EASY)
    assert x["ladder"] == ["C06", "C17"]
    assert r.next_coder(x, "C06") == "C17" and r.next_coder(x, "C17") is None
    assert r.route(EASY, available={"C17"})["start"] == "C17"


def test_policy_learns_from_counts(tmp_path):
    st = tmp_path / "r.json"
    r = RT.Router(SPEEDS, st)
    assert r.route(EASY)["start"] == "C06"
    for _ in range(30):
        r.observe("fn_s", "C06", False, 80)
    assert r.route(EASY)["start"] != "C06"
    r2 = RT.Router(SPEEDS, st)                                   # persisted
    assert r2.counts["fn_s"]["C06"] == [0, 30] and r2.route(EASY)["start"] != "C06"
    for _ in range(60):
        r2.observe("fn_s", "C06", True)
    assert r2.route(EASY)["start"] == "C06"


def test_reuse_hit_lowers_class():
    hard = dict(HARD, reuse_score=0.9)
    base = RT.task_class(RT.features(HARD))
    assert base != "fn_s" and RT.task_class(RT.features(hard)) == ("fn_s" if base == "fn_m" else "fn_m")


def test_floor_never_violated():
    want = ["MOE", "C17", "C06"]
    for free in (2.0, 3.0, 5.0, 12.0, 17.0, 25.0):
        plans = R.plan_models(mach(free=free), want)
        used = sum(p.resident_gb for p in plans.values() if p.loadable)
        assert free - used >= R.FLOOR_FREE_GB - 1e-9, (free, plans)


def test_moe_quant_follows_ram():
    assert R.plan_models(mach(free=30), ["MOE"])["MOE"].quant == "Q4_K_M"
    assert R.plan_models(mach(free=20), ["MOE"])["MOE"].quant == "Q3_K_M"
    assert R.plan_models(mach(free=18), ["MOE"])["MOE"].quant == "Q2_K"
    p = R.plan_models(mach(free=8), ["MOE", "C17"])
    assert not p["MOE"].loadable and "budget" in p["MOE"].reason and p["C17"].loadable
    q = R.plan_models(mach(free=30), ["MOE"], exists=lambda f: "Q3_K_M" in f)["MOE"]
    assert q.quant == "Q3_K_M"
    assert not R.plan_models(mach(free=30), ["MOE"], exists=lambda f: False)["MOE"].loadable


def test_threads_follow_free_cpu():
    busy = R.plan_models(mach(cpu=95.0), ["C17"])["C17"]
    idle = R.plan_models(mach(cpu=5.0), ["C17"])["C17"]
    assert busy.threads == 2 and idle.threads == 6 and idle.parallel == 2 and busy.parallel == 1


def test_gpu_profile_same_code():
    p = R.plan_models(mach(free=2.0), ["MOE", "C06"], profile="gpu", gpu_vram_gb=48.0)
    assert p["MOE"].loadable and p["MOE"].quant == "Q4_K_M" and p["MOE"].parallel == 8 and p["C06"].parallel == 4
    assert not R.plan_models(mach(), ["MOE"], profile="gpu", gpu_vram_gb=8.0)["MOE"].loadable


def test_keep_warm_by_reuse():
    plans = R.plan_models(mach(free=30), ["C06", "C17", "MOE"])
    w = R.keep_warm_plan({"C06": 70, "C17": 25, "MOE": 2}, plans, mach(free=30))
    assert w == {"C06": 600.0, "C17": 600.0, "MOE": 0.0}
    w = R.keep_warm_plan({"MOE": 10}, plans, mach(free=1.5))
    assert w["MOE"] == 0.0


def test_slot_specs_and_pool(tmp_path):
    plans = R.plan_models(mach(cpu=5.0), ["C06"])
    spec = R.slot_specs(plans, tmp_path, {"C06": 120.0})["C06"]
    assert isinstance(spec, SlotSpec) and spec.parallel == 2 and spec.keep_warm_s == 120.0 and spec.model.name.endswith("0.6B-Q4_K_M.gguf")
    assert slot_pool(spec, spawn=lambda cmd: None, health=lambda p: True).share == 2
    assert SlotSpec("x", Path("m")).parallel == 1                  # old callers unchanged
