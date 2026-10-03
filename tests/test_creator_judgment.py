"""Judgment drills with the local model: no future in the prompt, calibration only from earlier resolved records, successive halving, filler."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creator import judgment as J
from creator import thinking as T


def cases(n: int = 60) -> list[J.Case]:
    return [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, int(i % 2 == 0), "K" if i % 2 else "E", f"case {i} text") for i in range(n)]


def test_prompt_shows_only_cases_resolved_before_the_case_was_created() -> None:
    cs = cases()
    c = cs[30]
    msgs = J.build_prompt({"shots": 5, "hint": 1}, cs, c)
    blob = json.dumps(msgs)
    assert "case 29 text" not in blob and "case 31 text" not in blob and "case 28 text" in blob or "case 27 text" in blob
    assert "PROBABILITY" in blob and f"{len([x for x in cs if x.resolved < c.created])} resolved cases" in blob
    assert c.text in msgs[-1]["content"]


def test_parse_and_platt_is_fitted_on_the_past_only() -> None:
    assert J.parse("it looks risky.\nPROBABILITY: 0.30") == 0.3 and J.parse("no number") is None
    recs = [{"topic": "verdict", "subject": f"s{i}", "created": 100.0 + i, "resolved": 100.0 + i + 0.5, "y": i % 2, "p": 0.9 if i % 2 else 0.1} for i in range(40)]
    base = J.calibrated(recs)
    flipped = [dict(r, y=1 - r["y"]) if i >= 30 else r for i, r in enumerate(recs)]
    again = J.calibrated(flipped)
    assert [round(p, 9) for _r, p, _c in base[:30]] == [round(p, 9) for _r, p, _c in again[:30]]       # later outcomes never touch earlier calibration
    assert not base[0][2] and base[-1][2]


def test_halving_keeps_the_better_strategies() -> None:
    recs = []
    for i in range(45):
        for s in J.STRATEGIES:
            good = s["shots"] == 5 and s["hint"] == 1
            recs.append({"subject": f"s{i}", "strategy": s, "y": i % 2, "p": (0.9 if i % 2 else 0.1) if good else 0.5})
    live = J.alive(recs)
    assert live == [{"shots": 5, "hint": 1}]


class FakeLLM:
    def __init__(self, log: list[Any]) -> None:
        self.log = log

    def __enter__(self) -> "FakeLLM":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def chat(self, messages: Any, **kw: Any) -> str:
        self.log.append(messages)
        return "Looks fine.\nPROBABILITY: 0.60"


def test_filler_runs_batches_newest_first_one_in_flight_and_scores(tmp_path: Path, monkeypatch: Any) -> None:
    st = tmp_path / "st"
    st.mkdir()
    monkeypatch.setattr(J, "load_cases", lambda topic, state, repo: cases(30) if topic == "verdict" else [])
    log: list[Any] = []
    nj = J.judgment_filler(st, tmp_path, llm_factory=lambda: FakeLLM(log))
    j1 = nj()
    assert j1 is not None and nj() is None                       # a second batch waits for the first (one server slot)
    j1()
    rows = T._jsonl(J.path(st))
    assert len(rows) == J.BATCH and rows[0]["subject"] == "p29" and all(r["p"] == 0.6 for r in rows)
    assert nj() is not None and len(log) == J.BATCH
    monkeypatch.setattr(J, "_stat_preds", lambda t, s, r: {f"p{i}": T.Pred("verdict", f"p{i}", 1.0, 0.5, 0.5, 0.5) for i in range(30)})
    sec = J.trust_section(st, tmp_path)
    assert sec["verdict"]["n"] > 0 and sec["verdict"]["trusted"] is False and sec["verdict"]["why_not"]


def test_prompt_never_carries_the_items_outcome_or_post_creation_fields() -> None:
    """Rendered prompt for an item: no outcome words, no 'resolved' wording, no resolution time, no example that is the item itself."""
    cs = [J.Case("verdict", f"PKG{i:03d}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, int(i % 2 == 0), "K" if i % 2 else "E",
                 f"package PKG{i:03d} spec {i}") for i in range(60)]
    c = cs[40]
    for s in J.STRATEGIES:
        blob = json.dumps(J.build_prompt(s, cs, c))
        assert not any("esolved" in m["content"] for m in J.build_prompt(s, cs, c) if m["role"] == "assistant") and "ADOPTED" not in blob.replace("will the kernel ADOPT", "")
        assert c.subject not in blob.replace(c.text, "") and str(c.resolved) not in blob
        assert not any(f"PKG{i:03d}" in blob for i in range(40, 60) if i != 40)         # nothing created or resolved after / at the item


def test_git_topic_without_the_history_cache_has_no_cases_yet(tmp_path: Path) -> None:
    # 3 Oct: the git history is read only by the git_cache job; before it has run, git_fixed has no cases (the benchmark and the
    # judgment filler raised GitCacheMissing instead)
    assert J.load_cases("git_fixed", tmp_path / "state", tmp_path / "norepo") == []
