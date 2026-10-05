"""creator.gpueff + creator.gpucompile on synthetic runs/mixes (no runtime files, no network): efficiency model, thin-mix rule, epochs from the
measured gain, VRAM pairing, cached evals, timeline + $."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from creator import gpucompile as GC
from creator import gpueff as GE
from creator import jobcost as JC


def _curve(first: float, mid: float, last: float, steps: int) -> list[list[float]]:
    return [[steps // 4, first], [steps // 2, mid], [steps, last]]


def _run(name: str, size: str, steps: int, seconds: float, p80: float, gain: float, epochs: float = 2.0, peak: float = 10.0) -> dict[str, Any]:
    rows = int(steps * 16 / epochs)
    half = 0.5
    return {"name": name, "size": size, "rows": rows, "steps": steps, "epochs": epochs, "seconds": seconds, "peak_gb": peak, "stopped_early": False,
            "tokens_train": rows * p80 * 0.8, "trained_tokens": steps * 16 * p80 * 0.8, "padded_tokens": steps * 16 * p80, "len": {"max": 2000.0, "p80": p80},
            "eval_loss": _curve(0.9, half, half * (1 - gain), steps), "lora_gguf_s": 5.0}


def _eff() -> GE.Eff:
    rate = 8000.0
    runs = [_run(f"coder_{i}_17b", "1.7b", s, 100 + s * 16 * 500 / rate, 500.0, 0.02) for i, s in enumerate((100, 300, 600, 900))]
    runs += [_run("calib_17b", "1.7b", 800, 100 + 800 * 16 * 600 / rate, 600.0, 0.4)]
    runs += [_run("judge_06b", "0.6b", 300, 60 + 300 * 16 * 700 / 9000.0, 700.0, 0.06)]
    runs += [_run(f"calib_{i}_4b", "4b", s, 50 + s * 16 * 600 / 4300.0, 600.0, 0.4, 1.0) for i, s in enumerate((200, 400, 500))]
    return GE.Eff(runs)


def test_rates_and_epoch_gain_from_runs() -> None:
    eff = _eff()
    assert abs(eff.rate["1.7b"] - 8000) / 8000 < 0.05 and 3500 < eff.rate["4b"] < 5200
    g = eff.gains()
    assert g["coder_0_17b"] < 0.10 <= g["calib_17b"]                    # code mix: 2nd epoch not worth it; calibration mix: worth it
    assert abs(eff.train_s("1.7b", 400 * 16 * 500) - (100 + 400 * 16 * 500 / 8000)) < 40
    v = eff.validate()
    assert v["median_abs_err"] < 0.1 and v["n"] >= 8


def test_peak_model_uses_the_worst_measured_run_at_least_that_long() -> None:
    eff = _eff()
    assert eff.peak_gb("1.7b", 1500)["gb"] == 10.0 and "measured" in eff.peak_gb("1.7b", 1500)["basis"]
    est = eff.peak_gb("0.6b", 3000)                                    # nothing measured that long for 0.6B: scaled estimate, named as such
    assert est["gb"] > 0 and "estimate" in est["basis"] or "lower bound" in est["basis"]


def test_mix_epochs_weighted_and_prior() -> None:
    eff = _eff()
    assert GC.mix_epochs(eff, {"code": 1.0}, {})[0] == 1
    assert GC.mix_epochs(eff, {"calib": 0.8, "code": 0.2}, {})[0] == 2
    ep, why = GC.mix_epochs(eff, {"thinker": 1.0}, {"thinker": 1})
    assert ep == 1 and "prior" in why


def _mix(root: Path, name: str, rows: int, toks: int = 400) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text(json.dumps({"rows": {"train": rows, "dev": 10, "pref": 0}, "tokens_train": rows * toks}), encoding="utf-8")
    (d / "train.jsonl").write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": "x" * int(toks * 3.6)}]}) for _ in range(min(rows, 50))),
                                   encoding="utf-8")


def _built(eff: GE.Eff) -> dict[str, Any]:
    jobs = [{"job": "roleeval_calib_17b_base", "rc": 0, "wall_s": 40.0, "usd_total": 1.0, "skipped": False, "log": "a"}]
    return {"eff": eff, "jobs": jobs, "history": [JC.record("roleeval_base", 40.0)], "usd_h": 0.5, "runs": eff.runs}


def test_compile_applies_thin_mix_rule_and_epochs(tmp_path: Path) -> None:
    _mix(tmp_path, "tiny_17b", 800)                                    # 800 rows, code family: 1 epoch -> 50 steps < 200
    _mix(tmp_path, "ok_17b", 5000)
    wl = [{"id": "t", "name": "tiny", "size": "1.7b", "mixes": ["tiny_17b"], "families": {"code": 1.0}, "value_usd": 1.0},
          {"id": "o", "name": "ok", "size": "1.7b", "mixes": ["ok_17b"], "families": {"code": 1.0}, "value_usd": 1.0}]
    c = GC.compile_wishlist(wl, _built(_eff()), tmp_path)
    by = {j["id"]: j for j in c["jobs"]}
    assert by["t"]["thin_mix"]["thin"] and by["t"]["thin_mix"]["rows_needed"] == 3200 - 800
    assert not by["o"]["thin_mix"]["thin"] and by["o"]["settings"]["epochs"] == 1 and by["o"]["settings"]["steps"] == 313
    assert by["o"]["data"]["mixes"][0]["sha256"]


def test_generated_rows_size_the_data_job_and_gate_the_schedule(tmp_path: Path) -> None:
    _mix(tmp_path, "a_06b", 3000, 300)
    gen = {"name": "pin", "rows": 0, "rows_have": 500, "rows_target": 4000, "mean": 400.0, "p80": 450.0, "p90": 500.0, "p99": 600.0, "max": 700.0, "gen": True}
    wl = [{"id": "g", "name": "gen", "size": "0.6b", "mixes": [], "extra": gen, "families": {"judge": 1.0}, "value_usd": 5.0},
          {"id": "f", "name": "fast", "size": "0.6b", "mixes": ["a_06b"], "families": {"judge": 1.0}, "value_usd": 1.0, "deps": []}]
    c = GC.compile_wishlist(wl, _built(_eff()), tmp_path, hashing=False)
    g = next(j for j in c["jobs"] if j["id"] == "g")
    assert g["data"]["gen_rows"] == 3500 and g["data"]["gen_s"] == 3500 * GC.GEN_S_PER_ROW / GC.GEN_WORKERS
    tl = c["schedule"]["timeline"]
    assert tl["g"]["train_start"] >= GC.BOOT_S + g["data"]["gen_s"] - 1                      # waits for its planted rows
    assert c["schedule"]["order"][0] == "f"                                                  # the data-ready job fills the GPU meanwhile


def test_pairing_only_when_both_peaks_and_the_eval_server_fit() -> None:
    a = {"peak_vram": {"gb": 6.0}}
    b = {"peak_vram": {"gb": 10.0}}
    big = {"peak_vram": {"gb": 25.0}}
    assert GC.can_pair(a, b) and not GC.can_pair(a, big) and not GC.can_pair(big, big)


def test_schedule_pairs_small_jobs_and_saves_time(tmp_path: Path) -> None:
    eff = _eff()
    for r in eff.runs:
        r["peak_gb"] = 6.0
    _mix(tmp_path, "s1_17b", 40000)
    _mix(tmp_path, "s2_17b", 40000)
    wl = [{"id": f"s{i}", "name": f"s{i}", "size": "1.7b", "mixes": [f"s{i}_17b"], "families": {"code": 1.0}, "value_usd": 1.0} for i in (1, 2)]
    c = GC.compile_wishlist(wl, _built(eff), tmp_path, hashing=False)
    assert [sorted(x) for x in c["schedule"]["slots"]] == [["s1", "s2"]] and c["schedule"]["makespan_s"] < c["unpaired_makespan_s"]
    short = GC.compile_wishlist(wl, _built(eff), tmp_path, hashing=False)
    assert short["schedule"]["makespan_s"] <= min(short["paired_makespan_s"], short["unpaired_makespan_s"])


def test_cached_base_evals_are_not_repeated(tmp_path: Path) -> None:
    _mix(tmp_path, "calib_17b", 4000)
    wl = [{"id": "c", "name": "calib_17b", "size": "1.7b", "mixes": ["calib_17b"], "families": {"code": 1.0}, "value_usd": 1.0}]
    b = _built(_eff())
    base_only = GC.compile_wishlist(wl, b, tmp_path, hashing=False)["schedule"]["timeline"]["c"]["pre_evals_s"]
    b["jobs"] = b["jobs"] + [{"job": "roleeval_calib_17b_generalist", "rc": 0, "wall_s": 50.0, "usd_total": 1.0, "skipped": False, "log": "a"}]
    both = GC.compile_wishlist(wl, b, tmp_path, hashing=False)["schedule"]["timeline"]["c"]["pre_evals_s"]
    assert both < base_only and both == 0.0


def test_package_written_with_timeline_specs_and_placement(tmp_path: Path) -> None:
    _mix(tmp_path / "mixes", "m_17b", 5000)
    wl = [{"id": "m", "name": "m", "size": "1.7b", "mixes": ["m_17b"], "families": {"code": 1.0}, "value_usd": 9.0}]
    c = GC.compile_wishlist(wl, _built(_eff()), tmp_path / "mixes", hashing=False)
    files = GC.write_package(c, tmp_path / "pkg", wl, {"ok": True})
    assert {"PACKAGE.json", "TIMELINE.md", "wishlist.json", "jobcost_validation.json"} <= set(files) and any("m.json" in f for f in files)
    assert "| m | 1.7b |" in (tmp_path / "pkg" / "TIMELINE.md").read_text(encoding="utf-8")
    pl = GC.place_all(c, state_dir=tmp_path / "pkg")
    assert pl["decisions"][0]["placement"] == "gpu_wishlist" and (tmp_path / "pkg" / "wishlist_placed.json").exists()
    assert pl["proposal"] is None or pl["proposal"]["rents"] is False


REAL = GE.outputs_root()


@pytest.mark.skipif(not any(REAL.glob("*/ft_*/gpuday/runs/ft_*/result.json")), reason="the 3-4 Oct GPU run records are not on this machine")
def test_acceptance_on_the_real_3_4_oct_records() -> None:
    b = GE.build()
    loo = JC.leave_one_out(b["history"])
    assert loo["passing"] >= 10                                        # P1.5: within 30% (leave-one-out) on >= 10 job kinds
    v = b["eff"].validate(b["jobs"])
    assert v["median_abs_err"] < 0.3 and v["ft_total_median_abs_err"] < 0.3
