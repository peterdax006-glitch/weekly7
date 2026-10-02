"""creator.power sample-size arithmetic and the DevWorkload result cache."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from creator import power as P
from creator import process_levers as PL
from creator import recursion as R


def test_required_n_matches_hand_arithmetic() -> None:
    # (2 + 0.84)^2 * (0.4*0.6 + 0.5*0.5) / 0.1^2 = 8.0656 * 0.49 / 0.01 = 395.2 -> 396
    assert P.required_n(0.4, 0.1) == 396


def test_detectable_effect_shrinks_with_n_and_is_self_consistent() -> None:
    e40, e160 = P.detectable_effect(0.4, 40), P.detectable_effect(0.4, 160)
    assert e160 < e40
    assert P.required_n(0.4, e160) <= 161       # the effect found at n=160 needs about 160 tasks


def test_paired_needs_fewer_tasks_than_unpaired() -> None:
    assert P.paired_required_n(0.15, 0.1) < P.required_n(0.4, 0.1)
    with pytest.raises(ValueError):
        P.paired_required_n(0.05, 0.1)


def test_pilot_counts() -> None:
    base = [True] * 4 + [False] * 6
    cand = [True] * 4 + [True] * 2 + [False] * 4
    r = P.pilot(base, cand)
    assert r["p_base"] == 0.4 and r["p_cand"] == 0.6 and math.isclose(r["gain"] or 0, 0.2)
    assert r["discordant"] == 0.2
    with pytest.raises(ValueError):
        P.pilot([True], [True, False])


def test_workload_cache_resumes_without_rerunning(tmp_path: Path) -> None:
    calls: list[str] = []

    def ev(p: R.ProcessConfig, t: str) -> tuple[bool, bool]:
        calls.append(t)
        return (len(t) % 2 == 0, True)
    ids = [f"t{i:02d}" for i in range(12)]
    cache = tmp_path / "c.jsonl"
    w1 = PL.DevWorkload(seed=1, chunk=2, reps=2, confirm=2, evaluate=ev, dev_ids=ids, cache_path=cache)
    s1 = w1(R.ProcessConfig())
    n1 = len(calls)
    cache.write_text(cache.read_text(encoding="utf-8") + '{"torn', encoding="utf-8")        # a killed run leaves a torn last line
    w2 = PL.DevWorkload(seed=1, chunk=2, reps=2, confirm=2, evaluate=ev, dev_ids=ids, cache_path=cache)
    assert w2(R.ProcessConfig()) == s1 and len(calls) == n1 and w2.runs == 0
