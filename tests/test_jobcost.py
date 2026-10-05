"""creator.jobcost: cost model, pilot, leave-one-out, placement (blueprint 7.10), wishlist proposal, predicted-vs-actual recalibration."""
from __future__ import annotations

import json
from pathlib import Path

from creator import jobcost as JC
from creator import resources as R


def _recs(kind: str, pairs: list[tuple[float, float]]) -> list[dict]:
    return [JC.record(kind, s, u) for u, s in pairs]


def test_linear_fit_recovers_fixed_and_rate() -> None:
    recs = _recs("train", [(u, 30 + 0.002 * u) for u in (1000, 5000, 20000, 80000)])
    m = JC.CostModel(recs).predict("train", 50000)
    assert m["basis"] == "history:linear" and abs(m["seconds"] - 130) < 1


def test_constant_kind_and_unknown_kind_and_pilot() -> None:
    cm = JC.CostModel(_recs("reg", [(0, 6.0), (0, 5.0), (0, 7.0)]) and [JC.record("reg", s) for s in (6.0, 5.0, 7.0)])
    assert cm.predict("reg")["seconds"] == 6.0
    assert cm.predict("nope") is None
    p = cm.predict("nope", pilot_seconds=12.0, pilot_fixed_s=2.0)           # 2% pilot: 2 + (12 - 2) / 0.02
    assert p["basis"] == "pilot" and p["seconds"] == 502.0 and p["high"] == 2 * p["seconds"]


def test_leave_one_out_judges_kinds_and_flags_a_bad_one() -> None:
    good = [JC.record("a", s) for s in (10.0, 11.0, 9.5, 10.5, 10.2)]
    bad = [JC.record("b", s) for s in (10.0, 100.0, 12.0, 300.0)]
    r = JC.leave_one_out(good + bad + [JC.record("single", 5.0)])
    assert r["kinds"]["a"]["pass"] and not r["kinds"]["b"]["pass"] and not r["kinds"]["single"]["judged"]
    assert r["judged"] == 2 and r["passing"] == 1


def test_place_short_runs_on_the_pc() -> None:
    d = JC.place("j", "k", 15 * 60, t_gpu_s=60)
    assert d["placement"] == "pc" and "wishlist_entry" not in d


def test_place_long_with_speedup_goes_to_gpu_wishlist() -> None:
    d = JC.place("train", "ft", 8 * 3600, t_gpu_s=1200, value_usd=10, deps=["data"], deadline="2026-10-09")
    e = d["wishlist_entry"]
    assert d["placement"] == "gpu_wishlist" and e["speedup"] == 24.0 and e["deps"] == ["data"] and e["deadline"] == "2026-10-09"
    assert abs(e["usd"] - JC.gpu_usd(1200)) < 1e-9 and e["usd"] > 1200 / 3600 * JC.DEFAULT_GPU_USD_H      # setup is charged


def test_place_prefers_make_it_faster_when_it_gets_under_30_minutes_for_less_effort() -> None:
    d = JC.place("embed", "ft", 6 * 3600, t_gpu_s=1800, faster={"name": "cache the embeddings", "t_pc_s": 20 * 60, "effort_h": 0.5})
    assert d["placement"] == "faster" and d["improvement"]["name"] == "cache the embeddings"
    assert d["wishlist_entry"]["status"] == "alternative"


def test_place_below_the_rule_of_thumb_stays_on_the_pc() -> None:
    d = JC.place("mid", "x", 2 * 3600, t_gpu_s=1500)                       # S = 4.8 < 5 and T_pc < 5 h
    assert d["placement"] in ("pc_long", "gpu_wishlist") and d["options"]["pc"]["hours"] > 0
    d2 = JC.place("mid2", "x", 3 * 3600, t_gpu_s=7000)                     # S = 1.5: barely faster, not worth a rental
    assert d2["placement"] == "pc_long" and "gpu" not in d2["options"]


def test_wishlist_proposal_threshold_dependencies_and_never_rents(tmp_path: Path) -> None:
    w = tmp_path / "w.json"
    JC.add_wish(w, {"id": "data", "value_usd": 1.0, "t_gpu_s": 1800, "usd": 0.3, "deps": [], "status": "wishlist"})
    JC.add_wish(w, {"id": "train", "value_usd": 30.0, "t_gpu_s": 3600, "usd": 0.5, "deps": ["data"], "status": "wishlist"})
    p = JC.proposal(JC.load_wishlist(w))
    assert p and p["rents"] is False and p["jobs"] == ["data", "train"] and p["value_per_usd"] > JC.PROPOSE_VALUE_PER_USD
    assert JC.proposal({"x": {"id": "x", "value_usd": 0.1, "t_gpu_s": 3600, "usd": 0.5, "status": "wishlist"}}) is None      # not worth a digest line
    due = JC.proposal({"x": {"id": "x", "value_usd": 0.1, "t_gpu_s": 3600, "usd": 0.5, "status": "wishlist", "deadline": "soon"}}, deadline_days=2)
    assert due and due["trigger"] == "deadline"
    assert JC.proposal({}) is None


def test_recalibration_log_gives_a_factor_after_persistent_error(tmp_path: Path) -> None:
    log = tmp_path / "out.jsonl"
    for actual in (200.0, 210.0, 190.0):
        JC.record_outcome(log, {"job": "j", "kind": "k", "t_pc_s": 100.0}, actual)
    JC.record_outcome(log, {"job": "j2", "kind": "ok", "t_pc_s": 100.0}, 105.0)
    c = JC.calibration(log)
    assert c["k"]["factor"] == 2.0 and c["ok"]["factor"] == 1.0 and c["ok"]["within_30pct"] == 1.0
    cm = JC.CostModel([JC.record("k", 100.0)], JC.factors(log))
    assert cm.predict("k")["seconds"] == 200.0


def test_pc_history_bridge_reads_resources_rows(tmp_path: Path) -> None:
    for u, s in ((1000, 40.0), (2000, 60.0), (4000, 100.0)):
        R.record_task(tmp_path, "embed", s, None, 0.5, units=u)
    recs = JC.pc_history(tmp_path)
    assert len(recs) == 3 and recs[0]["units"] == 1000 and recs[0]["where"] == "pc"
    assert abs(JC.CostModel(recs).predict("embed", 3000)["seconds"] - 80.0) < 1.0
    json.dumps(recs)
