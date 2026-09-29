"""engine.learning.core: shared vocabulary, canonical hashing and the time firewall (contract C62 A05, A08, A12)."""
import numpy as np
import pytest

from engine.learning import core as C


def test_stable_hash_is_order_free_and_numpy_safe():
    a = {"b": np.float64(0.1) + np.float64(0.2), "a": [np.int64(3), 1.0], "c": np.float32(2.5)}
    b = {"c": 2.5, "a": [3, 1.0], "b": 0.30000000000000004}
    assert C.stable_hash(a) == C.stable_hash(b)                 # numpy 2 scalars used to crash here


def test_nan_is_hashed_visibly_not_as_zero():
    assert C.stable_hash({"x": float("nan")}) != C.stable_hash({"x": 0.0})
    assert C.stable_hash({"x": np.float64("nan")}) == C.stable_hash({"x": float("nan")})


def test_require_past_fails_closed_on_same_day_and_missing():
    C.require_past("2001-01-01", "2001-01-02")
    for when, now in (("2001-01-02", "2001-01-02"), ("2001-01-03", "2001-01-02"), (None, "2001-01-02")):
        with pytest.raises(C.FirewallBreach):
            C.require_past(when, now)


def test_provenance_could_exist_uses_newest_outcome_seen():
    p = C.Provenance("2026-09-29T10:00", "2001-03-02", "abc", outcomes_seen_through="2001-03-09")
    assert p.could_exist_at("2001-03-10") and not p.could_exist_at("2001-03-09")
    assert C.Provenance("", "2001-01-01", "").check()             # missing fields are reported
    assert C.Provenance("t", "2001-03-09", "c", outcomes_seen_through="2001-03-01").check()


def test_confidence_dimensions_stay_separate_and_bounded():
    c = C.Confidence(truth=0.9, current_reliability=0.1)
    assert c.check() == [] and "transfer" in c.untested() and "truth" not in c.untested()
    assert C.Confidence(failure_risk=1.5).check()
    with pytest.raises(ValueError):
        C.clip01(float("nan"))


def test_enums_round_trip_and_knowledge_like_conformance():
    assert C.Epistemic.parse("GATED") is C.Epistemic.GATED and str(C.Health.BROKEN) == "BROKEN"
    assert C.KnowledgeLike.conforms(object()) and len(C.KnowledgeLike.conforms(object())) == len(C.KnowledgeLike.REQUIRED)
