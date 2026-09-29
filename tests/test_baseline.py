import os
import stat
import pytest
from engine import baseline as B, checkpoint as C
from engine.registry import Registry
import json


def log(tmp_path, extra=()):
    recs = [{"event": "historical_test", "t": "2026-01-01", "variant": "v10", "mean_week": 0.0017, "median_week": 0.0033,
             "pct_ge_7": 0.0256, "pct_le_m7": 0.0296, "win_weeks": 0.55, "worst_week": -0.21, "best_week": 0.14,
             "max_dd": -0.667, "turnover": 730.0, "cost_total": 689.0, "weeks": 507},
            {"event": "livesim_cycle", "run_id": "r1", "mean_week": 0.001, "weeks_ge_7": 1, "year_return": 0.03},
            {"event": "livesim_cycle", "run_id": "r2", "mean_week": 0.003, "weeks_ge_7": 3, "year_return": 0.07},
            *extra]
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return Registry(p)


def frozen(tmp_path):
    return B.freeze({"k": 4, "exit_q": 0.9}, "2026-09-28", root=tmp_path / "bl", registry=log(tmp_path))


def test_extract_and_unmeasured_are_explicit(tmp_path):
    m, src, un = B.extract_metrics(log(tmp_path).records)
    assert m["max_dd"] == -0.667 and m["blind_mean_week"] == pytest.approx(0.002) and m["blind_cycles"] == 2
    assert {"mover_accuracy", "pattern_results", "analog_results", "missed_winners", "direction_accuracy", "model_results"} <= set(un)
    assert "max_drawdown" not in un and "blind_window_results" not in un


def test_freeze_writes_verified_readonly_bundle(tmp_path):
    d, summ = frozen(tmp_path)
    assert B.verify(d)["ok"] and B.load(d)["config"] == {"k": 4, "exit_q": 0.9}
    assert B.load(d)["provenance"]["code_hash"] and "unmeasured" in summ
    with pytest.raises(PermissionError):
        (d / "metrics.json").write_text("{}")
    assert B.latest(tmp_path / "bl") == d


def test_freeze_refuses_overwrite_empty_cfg_and_empty_log(tmp_path):
    frozen(tmp_path)
    with pytest.raises(C.CheckpointError):
        frozen(tmp_path)
    with pytest.raises(ValueError):
        B.freeze({}, "t", root=tmp_path / "b2", registry=log(tmp_path))
    with pytest.raises(ValueError):
        B.freeze({"a": 1}, "t", root=tmp_path / "b3", registry=Registry(tmp_path / "none.jsonl"))


def test_tampering_with_baseline_is_caught_and_blocks_diff(tmp_path):
    d, _ = frozen(tmp_path)
    m = d / "metrics.json"
    os.chmod(m, stat.S_IWRITE)
    m.write_text(m.read_text().replace("-0.667", "-0.1"))
    assert not B.verify(d)["ok"]
    with pytest.raises(C.CheckpointError):
        B.diff_vs_baseline({"max_dd": -0.1}, d)


def test_diff_verdicts_use_metric_polarity(tmp_path):
    d, _ = frozen(tmp_path)
    r = B.diff_vs_baseline({"mean_week": 0.07, "max_dd": -0.9, "pct_le_m7": 0.0296, "pct_ge_7": 0.0, "mystery": 1.0,
                            "worst_week": float("nan")}, d)
    v = {k: x["verdict"] for k, x in r["rows"].items()}
    assert v == {"mean_week": "better", "max_dd": "worse", "pct_le_m7": "unchanged", "pct_ge_7": "worse"}
    assert r["no_baseline"] == ["mystery"] and r["better"] == 1 and r["worse"] == 2
    assert "mover_accuracy" in r["unmeasured"]


def test_no_baseline_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ROOT", tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        B.diff_vs_baseline({"mean_week": 1.0})


def test_code_drift_flag(tmp_path, monkeypatch):
    d, _ = frozen(tmp_path)
    assert B.verify(d)["code_drift"] is False
    monkeypatch.setattr(B.P, "code_hash", lambda *a, **k: "different")
    assert B.verify(d)["code_drift"] is True
