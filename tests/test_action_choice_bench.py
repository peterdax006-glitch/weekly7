"""The choice bench must expose position bias: a model that always answers 1 scores exactly the 'always 1' baseline."""
from __future__ import annotations

from scripts import action_choice_bench as B


class Always1:
    def chat(self, messages, **kw) -> str:                 # type: ignore[no-untyped-def]
        return "CHOICE: 1\nWHY: first."


def test_bench_catches_position_bias() -> None:
    res = B.run_bench(Always1(), n=24, seed=3, with_lessons=False)
    r = res["no_lessons"]
    assert r["accuracy"] == res["baselines"]["always_choose_1"]
    by = r["accuracy_by_correct_position"]
    assert by["1"]["accuracy"] == 1.0
    assert all(v["accuracy"] == 0.0 for k, v in by.items() if k != "1")
    assert len(by) >= 2                                    # the correct index really varies
    assert r["none_rate"] == 0.0 and r["invalid_rate"] == 0.0


def test_tasks_have_one_correct_and_are_deterministic() -> None:
    a, b = B.generate_tasks(10, 5), B.generate_tasks(10, 5)
    assert [t.correct for t in a] == [t.correct for t in b]
    assert all(2 <= len(t.candidates) and 1 <= t.correct <= len(t.candidates) for t in a)


def test_hard_objectives_never_name_the_target_and_wilson_is_sane() -> None:
    for t in B.generate_tasks(24, 5, hard=True):
        name = t.candidates[t.correct - 1].short().split("(")[1].rstrip(")").split(",")[0]
        assert name not in t.objective
    lo, hi = B._wilson(45, 90)
    assert 0.39 < lo < 0.5 < hi < 0.61
    assert B._wilson(0, 0) == (0.0, 0.0)
