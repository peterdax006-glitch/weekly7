import pytest
from engine.experiment_memory import Space, TriedIndex
from engine.registry import Registry
import json

SP = Space({"k": (2, 12, 2), "exit_q": (0.5, 1.0, 0.05), "pick": ["top", "rank", "random"]})


def idx(tmp_path):
    return TriedIndex(tmp_path / "t.jsonl", SP)


def test_snap_makes_tiny_differences_identical():
    assert SP.key({"k": 4, "exit_q": 0.9}) == SP.key({"k": 4.0, "exit_q": 0.9004})
    assert SP.key({"k": 4, "exit_q": 0.9}) != SP.key({"k": 6, "exit_q": 0.9})


def test_distance_properties():
    a = {"k": 4, "exit_q": 0.9, "pick": "top"}
    assert SP.distance(a, a) == 0
    assert SP.distance(a, {**a, "pick": "rank"}) == pytest.approx(1 / 3)
    assert SP.distance(a, {"k": 4}) > SP.distance(a, {**a, "k": 6})       # a missing parameter is a full difference
    assert SP.distance({}, {}) == 0


def test_planted_repeat_and_known_failure_are_blocked(tmp_path):
    t = idx(tmp_path)
    t.add("E1", {"k": 4, "exit_q": 0.9, "pick": "top"}, "reject", -0.5, "drawdown 60%")
    same = t.check({"k": 4, "exit_q": 0.9002, "pick": "top"})
    near = t.check({"k": 4, "exit_q": 0.95, "pick": "top"}, near=0.1)
    far = t.check({"k": 12, "exit_q": 0.5, "pick": "random"})
    assert same["verdict"] == "repeat" and same["block"] and same["reasons"] == ["drawdown 60%"]
    assert near["verdict"] == "known_failure_nearby" and near["block"]
    assert far["verdict"] == "novel" and not far["block"]


def test_near_duplicate_of_success_is_not_blocked(tmp_path):
    t = idx(tmp_path)
    t.add("E1", {"k": 4, "exit_q": 0.9}, "adopt", 1.0)
    c = t.check({"k": 4, "exit_q": 0.95}, near=0.1)
    assert c["verdict"] == "near_duplicate" and not c["block"]


def test_add_validation(tmp_path):
    t = idx(tmp_path)
    with pytest.raises(ValueError):
        t.add("E1", {"k": 4}, "reject")                       # no reason
    with pytest.raises(ValueError):
        t.add("E1", {"k": 4}, "maybe")
    with pytest.raises(ValueError):
        t.add("E1", {"k": 4}, "adopt", float("nan"))
    t.add("E1", {"k": 4}, "adopt", 1.0)
    with pytest.raises(ValueError):
        t.add("E1", {"k": 6}, "adopt", 1.0)


def test_untried_neighbours_skip_tried_and_lean_to_good(tmp_path):
    t = idx(tmp_path)
    base = {"k": 4, "exit_q": 0.9, "pick": "top"}
    t.add("E1", base, "adopt", 1.0)
    t.add("E2", {**base, "k": 6}, "reject", -1.0, "bad")
    n = t.untried_neighbours(base, n=50, near=0.01)
    cfgs = [x["cfg"] for x in n]
    assert base not in cfgs and {**base, "k": 6} not in cfgs
    assert {**base, "k": 2} in cfgs and {**base, "pick": "rank"} in cfgs
    assert all(x["param"] in SP.spec for x in n)
    assert len({SP.key(c) for c in cfgs}) == len(cfgs)


def test_negative_results_table_sorted(tmp_path):
    t = idx(tmp_path)
    t.add("A", {"k": 2}, "reject", -0.1, "r1")
    t.add("B", {"k": 4}, "reject", -0.9, "r2")
    t.add("C", {"k": 6}, "reject", None, "r3")
    t.add("D", {"k": 8}, "adopt", 1.0)
    assert [r["experiment_id"] for r in t.negative_results()] == ["B", "A", "C"]


def test_persistence_coverage_and_empty(tmp_path):
    t = idx(tmp_path)
    assert t.check({"k": 4})["verdict"] == "novel" and t.negative_results() == [] and t.coverage()["k"] == 0
    t.add("A", {"k": 2, "pick": "top"}, "adopt", 0.1)
    t2 = idx(tmp_path)
    assert len(t2.rows) == 1 and t2.check({"k": 2, "pick": "top"})["verdict"] == "repeat"
    assert t2.coverage()["pick"] == pytest.approx(1 / 3) and t2.coverage()["k"] == pytest.approx(1 / 6)


def test_import_registry_skips_unusable(tmp_path):
    p = tmp_path / "e.jsonl"
    recs = [{"experiment_id": "A", "config": {"k": 2}, "outcome": "adopt", "metrics": {"score": 0.3}},
            {"experiment_id": "B", "config": {"k": 4}, "outcome": "reject"},              # no reason
            {"experiment_id": "C", "outcome": "adopt"},                                   # no config
            {"experiment_id": "D", "config": {"k": 6}, "outcome": "reject", "reason": "x"}]
    p.write_text("\n".join(json.dumps(r) for r in recs))
    t = idx(tmp_path)
    assert t.import_registry(Registry(p)) == {"added": 2, "skipped": 2}
    assert t.import_registry(Registry(p))["added"] == 0
