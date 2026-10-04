"""Prospective drill predictions (3 Oct 2026): the git drills beat their baselines on held-out history and were calibrated, yet could never be
trusted - no prediction had ever been made before its outcome existed. live_pass records the best variant's prediction for every commit whose
outcome window is still open, resolves it when the window closes, and the trust gate counts only those as prospective."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import thinking as T

VARIANT = {"decay": 1.0, "k": 2.0}


def planted(n: int, open_last: int, seed: int = 2) -> list[D.BItem]:
    """Key 'a' -> event 90%, key 'b' -> 10%; the last `open_last` items have no outcome yet."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        a = i % 2 == 0
        y = int(rnd.random() < (0.9 if a else 0.1))
        done = i < n - open_last
        out.append(D.BItem((f"k:{'a' if a else 'b'}",), 1000.0 + i, 1000.0 + i + 2.5 if done else None, y if done else 0, f"c{i:04d}"))
    return out


def test_a_live_prediction_is_what_the_replay_says_and_the_open_outcome_never_reaches_it() -> None:
    items = planted(120, open_last=6)
    live = {p.subject: p.p for p in D.predict_open(items, "git_fixed", VARIANT)}
    assert len(live) == 6
    closed = [D.BItem(it.keys, it.created, it.resolved if it.resolved is not None else 5000.0, it.y, it.subject) for it in items]
    replay = {p.subject: p.p for p in D.walk_forward(closed, "git_fixed", 1.0, 2.0)}
    for s, p in live.items():
        assert p == pytest.approx(replay[s])                         # resolved only after every creation: the same history
    flipped = [D.BItem(it.keys, it.created, it.resolved, 1 - it.y if it.resolved is None else it.y, it.subject) for it in items]
    assert {p.subject: p.p for p in D.predict_open(flipped, "git_fixed", VARIANT)} == live    # placeholder outcomes change nothing
    assert live["c0114"] > 0.7 and live["c0115"] < 0.3              # it learned the planted keys


def _patch(monkeypatch: pytest.MonkeyPatch, items: list[D.BItem], heldout: dict[str, Any]) -> dict[str, list[D.BItem]]:
    box = {"items": items}
    monkeypatch.setattr(D, "load", lambda s, *a, **k: box["items"])
    monkeypatch.setattr(D, "best_variant", lambda state, s: {"source": s, "variant": VARIANT, "items": 0, "resolved": 0, "heldout": heldout,
                                                              "variants_tried": 1} if s in D.LIVE_SOURCES else None)
    monkeypatch.setattr(D, "search_report", lambda state: {})
    return box


GOOD_HELDOUT = {"n": 300, "n_live": 0, "gain_ci95": [0.01, 0.02], "ece": 0.02, "best_baseline": "base", "brier": 0.16}


def test_live_pass_records_once_resolves_later_and_counts_as_prospective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = planted(200, open_last=30)
    box = _patch(monkeypatch, items, GOOD_HELDOUT)
    r1 = D.live_pass(tmp_path, tmp_path, tmp_path / "J.md", now=9999.0)
    assert r1 == {"new": 60, "resolved": 0}                          # 30 open commits x 2 git sources
    assert D.live_pass(tmp_path, tmp_path, tmp_path / "J.md", now=10000.0) == {"new": 0, "resolved": 0}   # idempotent
    rows = [json.loads(ln) for ln in D.live_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert all("y" not in r and "resolved" not in r and r["made_at"] == 9999.0 for r in rows)       # no outcome is ever stored with a prediction
    sec = D.trust_section(tmp_path)
    assert sec["git_fixed"]["heldout"]["n_live"] == 0 and not sec["git_fixed"]["trusted"]           # nothing resolved yet: not trusted
    box["items"] = planted(200, open_last=0)                         # the windows closed
    assert D.live_pass(tmp_path, tmp_path, tmp_path / "J.md", now=20000.0) == {"new": 0, "resolved": 60}
    lp = D.live_preds(tmp_path)
    assert len(lp["git_fixed"]) == 30 and all(p.mode == "live" for p in lp["git_fixed"])
    sec = D.trust_section(tmp_path)
    assert sec["git_fixed"]["heldout"]["n_live"] == 30 and sec["git_fixed"]["live"]["n"] == 30
    assert sec["git_fixed"]["trusted"], sec["git_fixed"]["why_not"]


def test_prospective_predictions_that_do_worse_block_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [], GOOD_HELDOUT)
    path = D.live_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(40):                                           # confident and wrong every time; the base rate is right
            y = i % 2
            f.write(json.dumps({"id": f"git_fixed:c{i}", "source": "git_fixed", "subject": f"c{i}", "made_at": float(i), "p": 0.95 if y == 0 else 0.05,
                                "base": 0.5, "last": 0.5, "mode": "live"}) + "\n")
            f.write(json.dumps({"id": f"git_fixed:c{i}", "resolved": y, "at": float(i) + 1}) + "\n")
    sec = D.trust_section(tmp_path)["git_fixed"]
    assert sec["heldout"]["n_live"] == 40 and not sec["trusted"]
    assert any("do WORSE" in w for w in sec["why_not"])
    assert T.trust_of(GOOD_HELDOUT | {"n_live": 40})[0]              # the replay alone would have passed: the live check is what refused


def test_drills_in_worker_processes_give_the_same_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    monkeypatch.setattr(D, "DEFAULT_STATE", D.DEFAULT_STATE)           # load() sets this module global: restored after the test
    j = tmp_path / "JOURNAL.md"
    topics = ["nupen kernel swarm", "spanish app tico", "weekly7 stock backtest"]
    rnd = random.Random(5)
    j.write_text("\n".join(f"- 2026-0{1 + i // 28}-{1 + i % 28:02d} {topics[rnd.randrange(3)]} entry {i}" for i in range(150)) + "\n", encoding="utf-8")
    v = {"decay": 0.97, "k": 3.0}
    a = D.run_job("journal_persist", v, tmp_path, tmp_path, j, tmp_path, processes=False)
    b = D.run_job("journal_persist", v, tmp_path, tmp_path, j, tmp_path, processes=True)
    for k in ("at", "cpu_s", "wall_s"):                                   # timings (h58 cost fields) differ by nature
        a.pop(k), b.pop(k)
    assert a == b and a["resolved"] > 100
    assert D._pool().submit(os.getpid).result() != os.getpid()       # the arithmetic really ran in another process
    rows = [json.loads(ln) for ln in D.runs_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2                                             # the parent wrote both rows


def test_the_latest_data_digest_wins_not_the_largest_string(tmp_path: Path) -> None:
    # 3 Oct: best_variant and the search report picked max(digest); git digests are commit hashes, so after a commit hashing below the
    # previous one every drill topic would have reported an OLD run
    p = D.runs_path(tmp_path)
    p.parent.mkdir(parents=True)
    sc = {"n": 100, "brier": 0.2}
    rows = [{"source": "git_fixed", "variant": {"decay": 1.0, "k": 2.0}, "digest": "f00000000000", "items": 1, "resolved": 1,
             "select": sc, "heldout": {**sc, "tag": "old"}},
            {"source": "git_fixed", "variant": {"decay": 1.0, "k": 3.0}, "digest": "a00000000000", "items": 2, "resolved": 2,
             "select": sc, "heldout": {**sc, "tag": "new"}}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert D.best_variant(tmp_path, "git_fixed")["heldout"]["tag"] == "new"      # type: ignore[index]
    assert D.search_report(tmp_path)["git_fixed"]["tried"] == 1
