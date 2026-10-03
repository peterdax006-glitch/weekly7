"""The compiled walk-forward (creator.fastwalk) must give the reference's predictions, prediction by prediction (3 Oct 2026: the speed-up must
never change what Nupen learns)."""
from __future__ import annotations

import random

import pytest

from creator import drillsources as D
from creator import fastwalk as FW


def _items(n: int, seed: int, keys: int = 30, ties: bool = False, open_frac: float = 0.1) -> list[D.BItem]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        t = float(i // 3 if ties else i) + (0.0 if ties else rnd.random())
        ks = tuple(f"k{rnd.randrange(keys)}:{j}" for j in range(rnd.randint(1, 4)))
        res = None if rnd.random() < open_frac else t + rnd.choice([0.0, 0.5, 2.0, 7.5, 40.0])
        out.append(D.BItem(ks, t, res, int(rnd.random() < 0.35), f"s{i}"))
    return out


def _same(a: list, b: list) -> None:                                    # type: ignore[type-arg]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.subject, x.made_at, x.outcome) == (y.subject, y.made_at, y.outcome)
        assert x.p == pytest.approx(y.p, abs=1e-9) and x.base == pytest.approx(y.base, abs=1e-12) and x.last == y.last


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("agg", ["mean", "logit"])
def test_same_predictions_as_the_reference(seed: int, agg: str) -> None:
    items = _items(600, seed, ties=seed % 2 == 1)
    for decay, k, cap in ((0.97, 3.0, 0.02), (1.0, 0.5, 0.05), (0.8, 12.0, 0.1)):
        _same(FW.walk_forward(items, "t", decay, k, agg, cap), D.walk_forward_reference(items, "t", decay, k, agg, cap))


def test_renormalisation_path_matches() -> None:
    items = _items(3000, 9, keys=8, open_frac=0.0)                     # decay 0.8 underflows 1e-100 after ~1000 resolutions
    _same(FW.walk_forward(items, "t", 0.8, 3.0), D.walk_forward_reference(items, "t", 0.8, 3.0))


def test_empty_and_all_open() -> None:
    assert FW.walk_forward([], "t") == []
    assert FW.walk_forward(_items(20, 1, open_frac=1.0), "t") == []


def test_drillsources_uses_it_and_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    items = _items(200, 3)
    if FW.HAVE_NUMBA:
        assert FW.enabled()
    monkeypatch.setenv("NUPEN_FASTWALK", "0")
    assert not FW.enabled()
    _same(D.walk_forward(items, "t"), D.walk_forward_reference(items, "t"))


def test_small_walks_use_the_reference_until_the_compiled_loop_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed h50: below SMALL_WALK items a process that has not imported fastwalk walks in Python (numba start-up ~2.5 s); same predictions."""
    import sys
    items = _items(200, 5)
    called: list[int] = []
    monkeypatch.setattr(FW, "walk_forward", lambda *a, **k: called.append(1) or [])
    monkeypatch.setitem(sys.modules, "creator.fastwalk", None)
    monkeypatch.delitem(sys.modules, "creator.fastwalk")
    _same(D.walk_forward(items, "t"), D.walk_forward_reference(items, "t"))
    assert not called
    monkeypatch.setattr(D, "SMALL_WALK", 0)
    monkeypatch.setitem(sys.modules, "creator.fastwalk", FW)
    D.walk_forward(items, "t")
    assert called
