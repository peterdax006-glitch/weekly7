"""CR K13: the meta component."""
from __future__ import annotations

import pytest

from creator import meta as M


def _learner():
    m = M.MetaLearner()
    m.register(M.Strategy("a"))
    m.register(M.Strategy("b", test_mode="code_first"))
    return m


def test_validate_ok_and_bad():
    assert M.Strategy("x").validate() == []
    bad = M.Strategy("", design_breadth=0, test_mode="nope", agent=" ")
    assert len(bad.validate()) == 4


def test_register_rejects_invalid_and_duplicate():
    m = _learner()
    with pytest.raises(ValueError):
        m.register(M.Strategy("a"))
    with pytest.raises(ValueError):
        m.register(M.Strategy("c", max_retries=99))


def test_record_failures():
    m = _learner()
    with pytest.raises(ValueError):
        m.record("zzz", "bug", True)
    with pytest.raises(ValueError):
        m.record("a", " ", True)
    with pytest.raises(ValueError):
        m.record("a", "bug", True, -1)


def test_stats_empty_and_filled():
    m = _learner()
    assert m.stats("a")["success_rate"] is None
    m.record("a", "bug", True, 2)
    m.record("a", "bug", False, 4)
    m.record("a", "feat", True, 0)
    s = m.stats("a", "bug")
    assert (s["n"], s["successes"], s["success_rate"], s["mean_cost"]) == (2, 1, 0.5, 3.0)
    assert m.stats("a")["n"] == 3


def test_select_empty_untried_and_best():
    with pytest.raises(ValueError):
        M.MetaLearner().select("bug")
    m = _learner()
    assert m.select("bug").name == "a"
    m.record("a", "bug", True)
    assert m.select("bug").name == "b"
    for _ in range(10):
        m.record("a", "bug", True)
        m.record("b", "bug", False)
    assert m.select("bug", explore=0.1).name == "a"


def test_propose_improvements():
    m = _learner()
    assert m.propose_improvements() == []
    for _ in range(5):
        m.record("a", "bug", False)
        m.record("b", "bug", False)
    m.record("a", "feat", True)
    props = {p.target: p for p in m.propose_improvements()}
    assert set(props) == {"strategy:a", "strategy:b"}
    assert "reviewer_depth: 1 -> 2" in props["strategy:a"].change
    assert "test_first" in props["strategy:b"].change
    assert props["strategy:a"].evidence["n"] == 5


def test_apply_creates_variant():
    m = _learner()
    v = m.apply("strategy:a", reviewer_depth=2)
    assert v.name == "a+v1" and v.reviewer_depth == 2
    assert m.apply("a", max_retries=3).name == "a+v2"
    with pytest.raises(ValueError):
        m.apply("nope")
    with pytest.raises(ValueError):
        m.apply("a", reviewer_depth=99)
