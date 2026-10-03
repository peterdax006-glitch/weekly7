"""Reasoning methods: retrieval is time-ordered (nothing known at/after the case), teacher traces, structured prompt, self-consistency, kNN control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creator import judgment as J
from creator import reasonmethods as RM
from creator import thinking as T


def cases(n: int = 60) -> list[J.Case]:
    return [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, int(i % 3 == 0), "K",
                   f"package p{i} requirement exists module alpha{i % 4} widget") for i in range(n)]


def corpus(cs: list[J.Case]) -> list[RM.Rec]:
    recs = RM.build_corpus(Path("/nonexistent"), cs)
    recs.append(RM.Rec(1100.0, "lesson", "verdict", "lesson widget alpha1 teacher package", "ADOPTED", 1, "claude", "The widget needs tests; I wrote them first."))
    recs.append(RM.Rec(5000.0, "lesson", "verdict", "lesson widget alpha1 FUTURE", "ADOPTED", 1, "claude", "future reasoning"))
    return recs


def test_search_is_strictly_time_ordered() -> None:
    cs = cases()
    idx = RM.Index(corpus(cs))
    c = cs[30]
    got = idx.search(c.text, c.created, 50, topic="verdict")
    assert got and all(r.known < c.created for r in got)
    assert not any(r.text.startswith("package p30 ") or "FUTURE" in r.text for r in got)          # the case itself resolves later: invisible


def test_prompt_with_retrieval_traces_and_template_has_no_future() -> None:
    cs = cases()
    idx = RM.Index(corpus(cs))
    c = cs[30]
    msgs = J.build_prompt({"shots": 0, "hint": 1, "retrieve": 4, "traces": 2, "structured": 1}, cs, c, idx)
    blob = json.dumps(msgs)
    assert "Similar past cases" in blob and "QUESTION" in blob and "The widget needs tests" in blob
    assert "FUTURE" not in blob and "future reasoning" not in blob and not any(f"package p{i} " in blob for i in range(31, 60))
    early = RM.Index(corpus(cs)).search(cs[2].text, cs[2].created, 4)
    assert all(r.known < cs[2].created for r in early)


def test_knn_and_self_consistency_and_paired_gain() -> None:
    cs = cases()
    idx = RM.Index(corpus(cs))
    p = RM.knn_probability(idx, cs[30], 8, 0.33)
    assert p is not None and 0.03 <= p <= 0.97
    calls: list[float] = []

    def chat(msgs: Any, max_tokens: int, temperature: float, seed: int, timeout: float) -> str:
        calls.append(temperature)
        return f"x\nPROBABILITY: {0.2 + 0.2 * seed:.2f}"
    ps, first, toks = RM.sample(chat, [{"role": "user", "content": "q"}], J.parse, 3, 50)
    assert ps == [0.2, 0.4, 0.6] and calls == [0.7] * 3 and toks > 0
    assert abs((RM.aggregate(ps) or 0) - 0.4) < 1e-9 and (RM.majority([0.1, 0.9, 0.8]) or 0) > 0.5
    g = RM.paired_gain([0.5] * 10, [0.9, 0.1] * 5, [1, 0] * 5)
    assert g[0] > 0 and g[1] > 0


class Fake:
    def __enter__(self) -> "Fake":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def chat(self, messages: Any, **kw: Any) -> str:
        return "ok\nPROBABILITY: 0.30"


def test_all_strategies_run_and_are_scored_with_cost_and_default(tmp_path: Path, monkeypatch: Any) -> None:
    cs = cases(40)
    st = tmp_path / "st"
    st.mkdir()
    idx = RM.Index(RM.build_corpus(st, cs))
    batch = [(cs[35], s) for s in J.ALL_STRATEGIES]
    assert J.run_batch(st, batch, cs, lambda: Fake(), idx) == len(batch)
    rows = T._jsonl(J.path(st))
    assert all("seconds" in r and "tokens" in r for r in rows)
    assert any(len(r.get("ps", [])) == 3 for r in rows) and any(r["reply"] == "knn" for r in rows)
    for s in J.ALL_STRATEGIES:                                 # the second subject so a paired comparison exists
        J.run_batch(st, [(cs[36], s)], cs, lambda: Fake(), idx)
    monkeypatch.setattr(J, "_stat_preds", lambda t, s, r: {f"p{i}": T.Pred("verdict", f"p{i}", 1.0, 0.5, 0.5, 0.5) for i in range(40)})
    sc = J.score_topic("verdict", st, tmp_path)
    one = next(v for v in sc.values() if "gain_vs_default" in v)
    assert one["gain_vs_default"][-1] == 2 and "accuracy" in one and "tokens_per_item" in one
    assert any("by_samples" in v for v in sc.values())


def test_alive_over_the_wide_pool_keeps_the_best_method() -> None:
    recs = []
    best = J.EXTRA_STRATEGIES[2]
    for i in range(70):
        for s in J.ALL_STRATEGIES:
            recs.append({"subject": f"s{i}", "strategy": s, "y": i % 2, "p": (0.9 if i % 2 else 0.1) if s == best else 0.5})
    assert J.alive(recs, J.ALL_STRATEGIES, J.HALVING_ALL) == [best]
