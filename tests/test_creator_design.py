"""CR K08: the design component."""
from __future__ import annotations

import json

from creator import design as D


def _cands():
    return [
        {"name": "A", "assumptions": ["x"], "failure_modes": ["y"], "cost": 0.2, "risk": 0.2, "benefit": 0.8},
        {"name": "B", "cost": 0.9, "risk": 0.9, "benefit": 0.1},
        {"name": "a", "cost": 0.0},
        {"name": " "},
    ]


def test_generate_options_empty_and_limits():
    assert D.generate_options("q", []) == []
    assert D.generate_options("q", _cands(), n=0) == []
    opts = D.generate_options("q", _cands(), n=5)
    assert [o.name for o in opts] == ["A", "B"]
    assert len(D.generate_options("q", _cands(), n=1)) == 1


def test_generate_options_clamps():
    o = D.generate_options("q", [{"name": "Z", "cost": 5, "risk": -1}])[0]
    assert o.cost == 1.0 and o.risk == 0.0


def test_critique():
    good, bad = D.generate_options("q", _cands())
    assert D.critique(good) == []
    issues = D.critique(bad)
    assert "no assumptions stated" in issues and "risk is high" in issues and "benefit is low" in issues


def test_score_and_bad_criteria():
    good, bad = D.generate_options("q", _cands())
    assert D.score(good) > D.score(bad)
    assert D.score(good, {"bogus": 1}) == 0.0
    assert D.score(good, {"cost": 0}) == 0.0
    assert 0.0 <= D.score(bad) <= 1.0


def test_select_keeps_rejected():
    opts = D.generate_options("q", _cands())
    d = D.select("q", opts)
    assert d.selected.name == "A"
    assert [o.name for o, _ in d.rejected] == ["B"]
    assert d.rejected[0][1]
    json.dumps(d.to_dict())
    assert d.to_dict()["rejected"][0]["option"]["name"] == "B"


def test_select_empty():
    d = D.select("q", [])
    assert d.selected is None and d.rejected == [] and d.to_dict()["selected"] is None


def test_select_criteria_change_winner():
    cheap = D.DesignOption("cheap", cost=0.0, risk=0.5, benefit=0.2, assumptions=("a",), failure_modes=("f",))
    great = D.DesignOption("great", cost=0.9, risk=0.5, benefit=1.0, assumptions=("a",), failure_modes=("f",))
    assert D.select("q", [cheap, great], {"cost": 1}).selected.name == "cheap"
    assert D.select("q", [cheap, great], {"benefit": 1}).selected.name == "great"
