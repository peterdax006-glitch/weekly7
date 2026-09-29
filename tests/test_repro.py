"""Bible Phase 33: reproducibility - identical artefacts by exact hash, and a diagnosis (not a tolerance) when not."""
import random
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from engine import repro


def good(cfg, seed):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=5), ["A", "B", "C"]], names=["date", "ticker"])
    pred = pd.Series(rng.standard_normal(len(idx)), index=idx)
    return {"candidate_order": ["p3", "p1", "p2"], "model_config": dict(cfg, seed=seed), "predictions": pred,
            "trades": pd.DataFrame({"t": ["A", "B"], "qty": [1.0, np.nan]}), "metrics": {"ic": float(pred.mean())}}


CFG = {"lr": 0.1, "depth": 3}


def test_deterministic_experiment_is_reproducible():
    rep = repro.run_twice(good, CFG, seed=5)
    assert rep["reproducible"], rep["problems"]
    assert all(v["same"] for v in rep["artifacts"].values())
    assert set(repro.REQUIRED) <= set(rep["artifacts"])
    assert rep["global_rng_use"] == []
    repro.assert_reproducible(good, CFG, 5)


def test_global_numpy_rng_is_caught_and_diagnosed():
    def leaky(cfg, seed):
        a = good(cfg, seed)
        a["predictions"] = a["predictions"] + np.random.standard_normal(len(a["predictions"])) * 1e-3   # unseeded
        return a
    rep = repro.run_twice(leaky, CFG, 5)
    assert not rep["reproducible"]
    d = next(x for x in rep["diagnoses"] if x["name"] == "predictions")
    assert any("value_change" in c for c in d["causes"]) and d["max_abs_diff"] > 0
    assert "numpy.random global state" in rep["global_rng_use"]
    with pytest.raises(repro.ReproError, match="predictions"):
        repro.assert_reproducible(leaky, CFG, 5)


def test_python_random_use_is_caught():
    def leaky(cfg, seed):
        a = good(cfg, seed); a["metrics"] = {"ic": random.random()}
        return a
    rep = repro.run_twice(leaky, CFG, 1)
    assert not rep["reproducible"] and "python random global state" in rep["global_rng_use"]


def test_ordering_only_difference_is_named():
    state = {"n": 0}

    def flip(cfg, seed):
        a = good(cfg, seed); state["n"] += 1
        a["candidate_order"] = ["p1", "p2", "p3"] if state["n"] % 2 else ["p3", "p1", "p2"]
        return a
    rep = repro.run_twice(flip, CFG, 1)
    d = next(x for x in rep["diagnoses"] if x["name"] == "candidate_order")
    assert any(c.startswith("ordering_only") for c in d["causes"]) and d["first_difference"] == "root[0]"


def test_float_noise_is_a_failure_not_a_tolerance():
    state = {"n": 0}

    def ulp(cfg, seed):
        a = good(cfg, seed); state["n"] += 1
        if state["n"] % 2 == 0:
            a["metrics"] = {"ic": np.nextafter(a["metrics"]["ic"], 1.0)}
        return a
    rep = repro.run_twice(ulp, CFG, 1)
    assert not rep["reproducible"]                                   # a single-ulp gap still fails
    d = next(x for x in rep["diagnoses"] if x["name"] == "metrics")
    assert any(c.startswith("float_noise") for c in d["causes"]) and 0 < d["max_rel_diff"] < 1e-9


def test_cfg_mutation_and_missing_artifact_are_reported():
    def mutating(cfg, seed):
        cfg["depth"] += 1
        return good(cfg, seed)
    rep = repro.run_twice(mutating, CFG, 1)
    assert any("mutated its own cfg" in p for p in rep["problems"]) and CFG == {"lr": 0.1, "depth": 3}

    def missing(cfg, seed):
        a = good(cfg, seed); del a["trades"]
        return a
    rep = repro.run_twice(missing, CFG, 1)
    assert rep["artifacts"]["trades"]["missing"] and not rep["reproducible"]


def test_hash_semantics():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})
    assert repro.artifact_hash(df) == repro.artifact_hash(df.copy())
    assert repro.artifact_hash(df) != repro.artifact_hash(df.iloc[::-1])          # row order is part of the artefact
    assert repro.artifact_hash({"a": 1, "b": 2}) == repro.artifact_hash({"b": 2, "a": 1})   # dict order is not
    assert repro.artifact_hash([1, 2]) != repro.artifact_hash([2, 1])
    assert repro.artifact_hash({1, 2, 3}) == repro.artifact_hash({3, 2, 1})
    assert repro.artifact_hash(np.float32(1)) == repro.artifact_hash(1.0)
    assert repro.artifact_hash(1) != repro.artifact_hash(1.0) and repro.artifact_hash("1") != repro.artifact_hash(1)
    assert repro.artifact_hash(np.array([np.nan])) == repro.artifact_hash(np.array([float("nan")]))
    with pytest.raises(TypeError):
        repro.artifact_hash(object())


def test_diagnose_shapes_and_nan_pattern():
    a = pd.DataFrame({"x": [1.0, 2.0]}); b = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert "size_change" in repro.diagnose("d", a, b)["causes"]
    assert "nan_pattern" in repro.diagnose("d", np.array([1.0, np.nan]), np.array([1.0, 2.0]))["causes"]
    assert repro.diagnose("d", 1, 1) == {"name": "d", "same": True}
    assert "structure_change" in repro.diagnose("d", {"a": 1}, {"b": 1})["causes"]
    s = pd.Series([1, 2, 3], index=[0, 1, 2]); s2 = s.iloc[[2, 0, 1]]
    assert any("ordering_only" in c for c in repro.diagnose("s", s, s2)["causes"])


def test_manifest_roundtrip_and_absent_manifest(tmp_path):
    art = good(CFG, 3)
    p = tmp_path / "m" / "manifest.json"
    repro.write_manifest(art, p, CFG, 3)
    assert repro.check_manifest(good(CFG, 3), p) == []
    assert repro.check_manifest(good(CFG, 4), p) == ["metrics", "model_config", "predictions"]
    with pytest.raises(FileNotFoundError):
        repro.check_manifest(art, tmp_path / "none.json")


def test_hash_seed_dependence_needs_a_fresh_process(tmp_path):
    (tmp_path / "exp_mod.py").write_text(textwrap.dedent('''
        def ordered(cfg, seed):
            return {"candidate_order": sorted({"aa", "bb", "cc", "dd"}), "metrics": {"m": 1.0}}
        def set_order(cfg, seed):
            return {"candidate_order": list({"aa", "bb", "cc", "dd", "ee", "ff"}), "metrics": {"m": 1.0}}
    '''))
    ok = repro.run_in_subprocess("exp_mod:ordered", {}, 1, pythonpath=[tmp_path])
    assert ok["reproducible"]
    bad = repro.run_in_subprocess("exp_mod:set_order", {}, 1, hashseeds=("0", "1", "2"), pythonpath=[tmp_path])
    assert bad["differs"] == ["candidate_order"] and not bad["reproducible"]
    with pytest.raises(repro.ReproError, match="child failed"):
        repro.run_in_subprocess("exp_mod:nope", {}, 1, pythonpath=[tmp_path])


def test_empty_artifacts():
    rep = repro.run_twice(lambda c, s: {}, {}, 0)
    assert not rep["reproducible"] and sum("required artifact" in p for p in rep["problems"]) == len(repro.REQUIRED) - 2
    rep = repro.run_twice(lambda c, s: {"metrics": pd.DataFrame(), "trades": [], "predictions": pd.Series(dtype=float),
                                        "candidate_order": [], "model_config": {}}, {}, 0)
    assert rep["reproducible"]
