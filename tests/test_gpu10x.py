"""The 5 Oct '10x faster GPU modules' work: data curation (creator.gpucurate), the smoke-first decision and safe-mode retry (creator.gpucompile),
speed candidates + plateau evidence (creator.gpueff), inference / measurement jobs in the schedule. Synthetic data only (no runtime files, no network)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creator import gpucompile as GC
from creator import gpucurate as CU
from creator import gpueff as GE
from creator import jobcost as JC
from tests.test_gpucompile import _built, _eff, _mix


def _row(i: int, user: str, asst: str, **meta: Any) -> dict[str, Any]:
    return {"id": f"r{i}", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user}, {"role": "assistant", "content": asst}], "meta": meta}


def test_exact_variant_and_length_steps() -> None:
    rows = [_row(1, "question one about the airflow repo " * 3, "ANSWER: A"), _row(2, "question one about the airflow repo " * 3, "a long worked trace " * 30),
            _row(3, "question one about the airflow repo " * 3, "ANSWER: A"), _row(4, "x" * 9000, "ANSWER: B")]
    pol = CU.Policy("t", variant_key=CU.user_text, variant_pref=CU.is_plain_answer, near_dup=False, len_cap=1000.0)
    c = CU.curate(rows, pol)
    assert [r["id"] for r in c["kept"]] == ["r1"]                      # the plain variant survives; the duplicate and the 9000-char row are gone
    steps = {s["step"]: s for s in c["report"]}
    assert steps["length"]["rows_out"] == 3 and steps["exact"]["rows_out"] == 2 and steps["variant"]["rows_out"] == 1
    assert c["tokens_after"] < c["tokens_before"]


def test_template_cap_keeps_all_hard_rows_and_reserves_the_easy_ones() -> None:
    rows = []
    for i in range(40):                                                 # 10 bugs x 4 restatements; label '1' = the code tools' top candidate (easy)
        rows.append(_row(i, f"bug {i % 10} failing test {i} lines unique text {i}", "1" if i % 5 else "3", file="f.py", line=i % 10, mutation="m"))
    pol = CU.Policy("p", template_key=lambda r: f"{r['meta']['file']}:{r['meta']['line']}:{r['meta']['mutation']}", max_per_template=2, near_dup=False,
                    is_easy=lambda r: r["messages"][-1]["content"] == "1", easy_keep=0.25)
    c = CU.curate(rows, pol)
    assert len({(r["meta"]["line"]) for r in c["kept"]}) == 10          # every distinct bug is still represented
    hard_in = [r for r in rows if r["messages"][-1]["content"] == "3"]
    kept_ids = {r["id"] for r in c["kept"]}
    capped = {s["step"]: s for s in c["report"]}["template"]
    assert capped["rows_out"] <= 20
    assert all(r["id"] in kept_ids or r["id"] not in {x["id"] for x in c["reserve"]} for r in hard_in)   # a hard row is never moved to the reserve
    assert all(r["messages"][-1]["content"] == "1" for r in c["reserve"])
    again = CU.curate(rows, pol)
    assert [r["id"] for r in again["kept"]] == [r["id"] for r in c["kept"]]       # deterministic


def test_round_robin_balances_groups_and_reserves_the_rest() -> None:
    rows = [_row(i, f"text number {i} " * 5, "x", g="big") for i in range(30)] + [_row(100 + i, f"other thing {i} " * 5, "x", g="small") for i in range(3)]
    pol = CU.Policy("b", group_key=lambda r: r["meta"]["g"], row_cap=9, near_dup=False)
    c = CU.curate(rows, pol)
    assert len(c["kept"]) == 9 and sum(1 for r in c["kept"] if r["meta"]["g"] == "small") == 3
    assert len(c["reserve"]) == 24


def test_exclusions_drop_rows_and_calibration_policy_never_rebalances() -> None:
    class Ex:
        def __bool__(self) -> bool:
            return True

        def violation(self, text: str, item_id: str = "") -> str | None:
            return "suite" if "SECRET" in text else None
    rows = [_row(1, "plain question number one", "0.9"), _row(2, "SECRET suite request text", "0.1"), _row(3, "plain question number two", "0.9")]
    c = CU.curate(rows, CU.phase2_policies()["calib"], Ex())
    assert [r["id"] for r in c["kept"]] == ["r1", "r3"] and c["reserve"] == []


def test_write_mix_is_trainmix_shaped(tmp_path: Path) -> None:
    rows = [_row(i, f"user text {i} " * 4, "a") for i in range(5)]
    man = CU.write_mix(tmp_path, "m.c", rows[:3], rows[3:], [], {"report": []})
    assert man["rows"] == {"train": 3, "dev": 2, "pref": 0, "reserve": 0}
    assert json.loads((tmp_path / "m.c" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]).keys() == {"messages"}
    mixrow = GC._mix("m.c", tmp_path)                                    # the compiler reads it like any mix
    assert mixrow["rows"] == 3 and mixrow["len"]["p99"] > 0


def test_plateau_fraction_from_measured_curves() -> None:
    fast = {"name": "coderonly_a_17b", "steps": 100, "epochs": 1.0, "stopped_early": False, "eval_loss": [[25, 0.30], [50, 0.205], [75, 0.2], [100, 0.2]]}
    assert GE.plateau_fraction([fast], "code", 0.05) == 0.5
    slow = {"name": "calib_a_17b", "steps": 100, "epochs": 1.0, "stopped_early": False, "eval_loss": [[25, 0.5], [50, 0.3], [75, 0.2], [100, 0.1]]}
    assert GE.plateau_fraction([slow], "calib", 0.05) == 1.0            # still improving at the end of epoch 1: no evidence for cutting data
    assert GE.plateau_fraction([], "code") is None


def test_candidates_keep_the_effective_batch_and_respect_vram() -> None:
    cands = GE.candidate_settings("1.7b", 10.0, 2000.0, 1000.0)
    assert cands[0]["name"] == "safe" and all(c["batch"] * c["accum"] == 16 for c in cands)
    assert {"group", "group_nockpt_b8"} <= {c["name"] for c in cands}
    tight = GE.candidate_settings("1.7b", 28.5, 3000.0, 3000.0)       # a long-sequence module: no-checkpoint variants do not fit and are not tried
    assert [c["name"] for c in tight if not c["grad_ckpt"]] == [] and tight[0]["name"] == "safe"
    assert all(c["unverified"] for c in cands[1:]) and not cands[0]["unverified"]


def test_pick_setting_takes_the_fastest_non_regressing_candidate() -> None:
    res = [{"name": "safe", "ok": True, "tok_s": 8000, "peak_gb": 14, "dev_loss": 0.50},
           {"name": "group", "ok": True, "tok_s": 10000, "peak_gb": 14, "dev_loss": 0.505},
           {"name": "group_nockpt", "ok": True, "tok_s": 13000, "peak_gb": 26, "dev_loss": 0.50},
           {"name": "group_b8", "ok": False, "tok_s": 0, "peak_gb": 0, "dev_loss": None},
           {"name": "packing", "ok": True, "tok_s": 15000, "peak_gb": 14, "dev_loss": 0.56},
           {"name": "group_nockpt_b8", "ok": True, "tok_s": 16000, "peak_gb": 31, "dev_loss": 0.50}]
    p = GC.pick_setting(res)
    assert p["choice"] == "group_nockpt" and abs(p["speedup"] - 1.625) < 1e-6
    why = dict(p["rejected"])
    assert "regressed" in why["packing"] and "failed" in why["group_b8"] and "peak" in why["group_nockpt_b8"]
    assert GC.pick_setting(res[:1])["choice"] == "safe"
    assert GC.pick_setting([{"name": "group", "ok": True, "tok_s": 9, "dev_loss": 1}])["choice"] == "safe"       # no safe reference: never guess
    slow = [res[0], {"name": "group", "ok": True, "tok_s": 8100, "peak_gb": 14, "dev_loss": 0.5}]
    assert GC.pick_setting(slow)["choice"] == "safe"


def test_safe_mode_retry_restarts_once_with_the_safe_setting() -> None:
    calls: list[str] = []

    def run(setting: dict[str, Any]) -> dict[str, Any]:
        calls.append(setting["name"])
        return {"ok": setting["name"] == "safe", "reason": "" if setting["name"] == "safe" else "CUDA out of memory"}
    r = GC.execute_with_safe_retry(run, {"name": "packing"}, {"name": "safe"})
    assert r["ok"] and r["retried"] and r["used"] == "safe" and calls == ["packing", "safe"]
    calls.clear()
    bad = GC.execute_with_safe_retry(lambda s: {"ok": False, "reason": "x"}, {"name": "safe"}, {"name": "safe"})
    assert not bad["ok"] and not bad["retried"]


def test_small_curated_mix_gets_a_smaller_effective_batch_only_when_allowed(tmp_path: Path) -> None:
    _mix(tmp_path, "small_06b", 1700, 300)
    base = {"id": "s", "name": "small", "size": "0.6b", "mixes": ["small_06b"], "families": {"judge": 1.0}, "value_usd": 1.0}
    c = GC.compile_wishlist([base], _built(_eff()), tmp_path, hashing=False)
    assert c["jobs"][0]["thin_mix"]["thin"] and c["jobs"][0]["settings"]["eff_batch"] == 16        # default: the old thin-mix rule, unchanged
    c2 = GC.compile_wishlist([dict(base, min_eff_batch=8)], _built(_eff()), tmp_path, hashing=False)
    j = c2["jobs"][0]
    assert not j["thin_mix"]["thin"] and j["settings"]["eff_batch"] == 8 and j["settings"]["steps"] >= GE.THIN_STEPS


def test_smoke_first_orders_infer_jobs_and_expected_scenario(tmp_path: Path) -> None:
    eff = _eff()
    for r in eff.runs:
        r["peak_gb"] = 12.0
    _mix(tmp_path, "a_17b", 6000, 400)
    wl = [{"id": "a", "name": "a", "size": "1.7b", "mixes": ["a_17b"], "families": {"code": 1.0}, "value_usd": 5.0, "deps": ["t"]},
          {"id": "t", "name": "teach", "kind": "infer", "servers": [{"label": "moe", "size": "30b-a3b", "tokens": 600000.0}], "verify_attempts": 100,
           "yield_rows": 500, "yields": "rows", "time_box_s": 900.0, "value_usd": 3.0}]
    c = GC.compile_wishlist(wl, _built(eff), tmp_path, hashing=False, smoke_first=True)
    names = [j["name"] for j in c["jobs"]]
    assert "smoke_1.7b" in names and "probe_30b-a3b" in names
    tl = c["schedule"]["timeline"]
    order = c["schedule"]["order"]
    assert order.index("p2.smoke_17b") < order.index("a") and order.index("p2.probe_30ba3b") < order.index("t") and order.index("t") < order.index("a")
    assert tl["a"]["train_start"] >= tl["t"]["train_end"] - 1e-6        # the training waits for the data job that feeds it
    assert tl["t"]["train_start"] >= GC.BOOT_S + c["jobs"][0]["predicted"]["base_download_s"]      # the 30B download gates the teacher, not the smoke run
    e = c["expected"]["schedule"]
    assert e["makespan_s"] < c["schedule"]["makespan_s"] and c["expected"]["jobs"]["a"]["speedup"] > 1.0
    assert c["schedule"]["gpu_busy_after_first_start_pct"] >= c["schedule"]["gpu_busy_pct"]
    smoke = next(j for j in c["jobs"] if j["name"] == "smoke_1.7b")
    assert smoke["smoke"]["candidates"][0]["name"] == "safe" and smoke["kind"] == "smoke"
    spec = GC.write_package(c, tmp_path / "pkg", wl, None)
    assert "specs/a.json" in [f.replace("\\", "/") for f in spec] and "specs/teach.json" in [f.replace("\\", "/") for f in spec]
    txt = (tmp_path / "pkg" / "TIMELINE.md").read_text(encoding="utf-8")
    assert "EXPECTED scenario" in txt and "teach (infer)" in txt


def test_time_boxed_rows_scale_with_the_throughput() -> None:
    a = GC.rows_in_box(1500.0, "30b-a3b", 1300.0)
    assert a == int((1500.0 - GC.INFER_LOAD_S["30b-a3b"]) * GC.INFER_TOK_S["30b-a3b"] / 1300.0) and a > 500
    j = GC.compile_infer_job({"id": "i", "name": "i", "servers": [{"label": "x", "size": "30b-a3b", "tokens": a * 1300.0}], "verify_attempts": 10, "yield_rows": a}, 0.47)
    assert abs(j["predicted"]["train_s"] - 1500.0) < 2.0
    four = GC.compile_infer_job({"id": "e", "name": "e", "servers": [{"label": s, "size": z, "tokens": 1e5} for s, z in
                                                                      (("a", "1.7b"), ("b", "1.7b"), ("c", "0.6b"), ("d", "30b-a3b"))], "yield_rows": 0}, 0.47)
    assert four["peak_vram"]["gb"] <= GE.SAFE_VRAM_GB + 3.0              # the four parallel suite servers fit one card (GGUF weights + slots)
    _ = JC
