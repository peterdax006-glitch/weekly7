"""Improvement engine (P1.1 detectors, P1.2 value/decide, 7.11 target registry): each detector finds its seeded case in a synthetic
metrics fixture, the estimator ranks a seeded set in the right order and applies the payback rule, the registry raises and never lowers."""
from __future__ import annotations

import json
import random
from pathlib import Path

from creator import constraints as CON

NOW = 1_800_000_000.0
DAYS = 7.0


def write_events(state: Path, rows: list[dict]) -> None:
    d = state / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events-20270101.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"t": NOW - 3600, "outcome": "ok", "in_tok": 0, "out_tok": 0, "wall_s": 1.0, "cache_hit": False, **r}) + "\n")


def jl(state: Path, name: str, rows: list[dict]) -> None:
    (state / "metrics").mkdir(parents=True, exist_ok=True)
    (state / "metrics" / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def fixture(state: Path) -> None:
    rows: list[dict] = []
    rows += [{"actor": "indexer", "cpu_s": 40.0} for _ in range(50)]                                   # hot path: 2000 s
    rows += [{"actor": "filler", "cpu_s": 1.0} for _ in range(100)]
    rows += [{"actor": "model:q", "model": True, "sig": "sigA", "cpu_s": 20.0} for _ in range(14)]       # repeated call
    rows += [{"actor": "model:q", "model": True, "sig": f"u{i}", "cpu_s": 2.0, "cls": "classify"} for i in range(30)]   # too big (shadow)
    rows += [{"actor": "model:fat", "model": True, "in_tok": 900, "out_tok": 300, "form_in": 80, "form_out": 40, "cpu_s": 6.0} for _ in range(10)]
    rows += [{"actor": "model:q", "model": True, "outcome": "Timeout", "err": "CUDA out of memory", "cpu_s": 30.0} for _ in range(4)]  # failure
    rows += [{"actor": "sampler", "idle_frac": 0.6, "queue": 3, "cores": 8, "wall_s": 60.0, "cpu_s": 0.1} for _ in range(10)]   # idle resource
    rows += [{"actor": "cyc", "goal_id": "G-stuck", "step": "cycle", "t": NOW - 7200 + i, "cpu_s": 10.0} for i in range(6)]   # stuck
    rows += [{"actor": "cyc", "goal_id": "G-ok", "step": "cycle", "t": NOW - 7200 + i, "progress": i == 5, "cpu_s": 10.0} for i in range(6)]
    rows += [{"actor": "model:q", "model": True, "cls": "summarize", "cpu_s": 9.0} for _ in range(20)]    # qwen replaceable
    write_events(state, rows)
    jl(state, "shadow.jsonl", [
        {"cls": "classify", "rung": "1.7b", "pass_rate": 0.90, "cost": 2.0, "n": 50, "current": True},
        {"cls": "classify", "rung": "0.6b", "pass_rate": 0.89, "cost": 0.5, "n": 50},
        {"cls": "summarize", "rung": "1.7b", "pass_rate": 0.80, "cost": 9.0, "n": 50, "current": True},
        {"cls": "summarize", "rung": "own", "pass_rate": 0.81, "cost": 1.0, "n": 50}])
    jl(state, "skills.jsonl", [{"role": "CODE", "score": 0.55, "target": 0.8, "uses_per_day": 20, "fail_cost": 200.0},
                               {"role": "CHECK", "score": 0.95, "target": 0.9, "uses_per_day": 50, "fail_cost": 10.0}])


def kinds(cands: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in cands:
        out.setdefault(c["kind"], []).append(c["key"])
    return out


def test_every_detector_finds_its_seeded_case(tmp_path: Path) -> None:
    fixture(tmp_path)
    CON.target_set(tmp_path, "locate_p50_ms", floor=2.5, target=5.0, evidence="t")
    CON.target_measure(tmp_path, "locate_p50_ms", 12.0)                                                  # missed target
    k = kinds(CON.detect_all(tmp_path, DAYS, NOW))
    assert k["hot_path"] == ["hot_path:indexer"]                                                         # filler (100 s) is below the share
    assert k["repeated_call"] == ["repeated_call:sigA"]                                                  # u0..u29 are single calls
    assert k["model_too_big"] == ["model_too_big:classify->0.6b"]
    assert k["qwen_replaceable"] == ["qwen_replaceable:summarize->own"]
    assert k["token_waste"] == ["token_waste:model:fat"]
    assert len(k["recurring_failure"]) == 1
    assert k["idle_resource"] == ["idle_resource:cpu"]
    assert k["weak_skill"] == ["weak_skill:CODE"]                                                        # CHECK is above its target
    assert k["stuck"] == ["stuck:G-stuck"]                                                               # G-ok made progress
    assert k["missed_target"] == ["missed_target:locate_p50_ms"]


def test_recurring_failure_links_the_known_fix(tmp_path: Path) -> None:
    from creator import knownfix as KF
    sample = next((k.signature[0] for k in KF.TABLE if k.signature and KF.match_line(k.signature[0])), None)
    if sample is None:
        return
    write_events(tmp_path, [{"actor": "x", "outcome": "Err", "err": sample, "cpu_s": 5.0} for _ in range(3)])
    c = CON.detect_recurring_failure(CON.load_events(tmp_path, DAYS, NOW), DAYS)
    assert c and c[0]["evidence"]["known_fix"]


def test_thresholds_do_not_fire_on_quiet_data(tmp_path: Path) -> None:
    write_events(tmp_path, [{"actor": "a", "cpu_s": 1.0}, {"actor": "b", "cpu_s": 1.0}, {"actor": "m", "model": True, "sig": "s", "cpu_s": 1.0}])
    assert [c for c in CON.detect_all(tmp_path, DAYS, NOW) if c["kind"] != "hot_path"] == []


def cand(kind: str, f: float, c0: float, c1: float, p: float, build: float, risk: str = "low") -> dict:
    return CON._cand(kind, f"{kind}{f}{c0}", {}, f, c0, c1, p, "x", risk, build)


def test_estimator_ranks_and_applies_payback_rule(tmp_path: Path) -> None:
    a = cand("hot_path", 100, 10, 5, 0.8, 1000)           # saving 400/d, B 1000 -> payback 2.5 d, ROI 12
    b = cand("repeated_call", 100, 10, 0.1, 0.9, 1000)    # saving 891/d -> payback 1.1 d, ROI 26.7  (best)
    c = cand("token_waste", 10, 10, 5, 0.5, 1000)         # saving 25/d -> payback 40 d: not worth
    d = cand("stuck", 1000, 10, 0, 0.9, 1000, "high")     # saving 9000/d but B 6000 -> payback 0.67 d, risk high: not allowed
    e = cand("weak_skill", 100, 10, 5, 0.8, 1500)         # payback 3.75 d: not worth (just over)
    rk = CON.rank([c, a, d, e, b], tmp_path)
    assert [x["kind"] for x in rk] == ["stuck", "repeated_call", "hot_path", "weak_skill", "token_waste"]    # raw ROI order
    worth = {x["kind"]: x["value"]["worth"] for x in rk}
    assert worth == {"stuck": False, "repeated_call": True, "hot_path": True, "weak_skill": False, "token_waste": False}
    assert abs(rk[1]["value"]["payback_days"] - 1000 / 891) < 1e-6
    top = CON.decide(tmp_path, [c, a, d, e, b], rng=random.Random(1))
    assert top and top["kind"] == "repeated_call"                                    # top worth-it ROI, the unallowed high-risk one skipped
    assert CON.decide(tmp_path, [a, b], in_flight=1) is None                         # WIP 1
    assert CON.decide(tmp_path, [c, e]) is None                                      # nothing worth it


def test_exploration_share_is_about_ten_percent(tmp_path: Path) -> None:
    best = cand("repeated_call", 100, 10, 0.1, 0.9, 1000)
    unsure = cand("idle_resource", 100, 10, 5, 0.5, 4000, "low")                      # allowed, uncertain, lower ranked
    rng = random.Random(7)
    modes = [CON.decide(tmp_path, [best, unsure], rng=rng)["mode"] for _ in range(1000)]
    assert 60 <= modes.count("explore") <= 140


def test_history_recalibrates_the_estimate(tmp_path: Path) -> None:
    a = cand("hot_path", 100, 10, 5, 0.8, 1000)
    base = CON.value(a)["saving_day"]
    for _ in range(3):
        CON.record_outcome(tmp_path, a, base * 0.25)                                  # the fix delivers a quarter of the prediction
    cal = CON.calibration(tmp_path)
    assert abs(cal["hot_path"] - 0.25) < 1e-9
    v = CON.rank([a], tmp_path)[0]["value"]
    assert abs(v["saving_day"] - base * 0.25) < 1e-9 and v["payback_days"] > 3 and not v["worth"]      # no longer worth it


def test_one_currency_ram_and_gpu() -> None:
    r = {"cpu_s": 10.0, "wall_s": 100.0, "ram_peak_mb": 2048.0, "gpu_usd": 0.5}
    assert CON.cost_units(r) == 10.0
    assert CON.cost_units(r, {"ram_binds": True, "ram_weight": 1.0}) == 10.0 + 2.0 * 100.0
    assert CON.cost_units(r, {"cpu_s_per_gpu_usd": 3600.0}) == 10.0 + 1800.0


def test_registry_seed_raise_and_never_lower(tmp_path: Path) -> None:
    n = CON.seed_registry(tmp_path)
    assert n == len(CON.RATCHET_SEED) and CON.seed_registry(tmp_path) == 0
    reg = CON.registry_load(tmp_path)
    assert reg["p0.2_event_us"]["target"] < 10.0                                      # best 7.4 beat 10 -> raised (stricter)
    assert reg["p0.8_envelope_median_tok"]["target"] < 35.0                           # measured 27.5 beats 35: stricter, never looser
    assert reg["p0.3_locate_p50_ms"]["target"] == 5.0                                 # best 7.0 does not beat 5 ms
    # seeded raise: floor 4, target 10, now measure 5.0 -> target moves to ~1.5x floor
    CON.target_set(tmp_path, "m", floor=4.0, target=10.0, evidence="seed")
    e = CON.target_measure(tmp_path, "m", 5.0, "bench")
    assert abs(e["target"] - 1.5 * e["floor"]) < 1e-9 and 5.0 < e["target"] < 10.0 and e["history"][-1]["event"] == "raised" and e["history"][-1]["evidence"] == "bench"
    t1 = e["target"]
    e = CON.target_measure(tmp_path, "m", 40.0)                                       # a bad measurement never lowers the target
    assert e["target"] == t1 and e["best_measured"] == 5.0
    e = CON.target_set(tmp_path, "m", floor=4.0, target=20.0)                         # a looser re-set is ignored
    assert e["target"] == t1
    # floor estimate wrong: measurement far below the floor re-estimates it
    e = CON.target_measure(tmp_path, "m", 2.0, "new technique")
    assert e["floor"] == 1.5 and e["target"] < t1 and e["target"] >= 1.5 * e["floor"] - 1e-9


def test_registry_higher_is_better(tmp_path: Path) -> None:
    CON.target_set(tmp_path, "top1", floor=0.9, target=0.6, higher_is_better=True, evidence="seed")
    e = CON.target_measure(tmp_path, "top1", 0.75, "run")
    assert e["target"] > 0.6 and e["history"][-1]["event"] == "raised"
    t = e["target"]
    assert CON.target_measure(tmp_path, "top1", 0.3)["target"] == t
    assert CON.detect_missed_targets(tmp_path)[0]["key"] == "missed_target:top1"
