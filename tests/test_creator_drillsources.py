"""F7 extension: more ground-truth sources for the thinking drills, strict time order, hard exclusions, and the parallel job factory."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

from creator import drillsources as D
from creator import thinking as T


def planted(n: int = 200, seed: int = 2) -> list[D.BItem]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        a = i % 2 == 0
        y = int(rnd.random() < (0.9 if a else 0.1))
        out.append(D.BItem((f"k:{'a' if a else 'b'}",), 1000.0 + i, 1000.0 + i + 2.5, y, f"s{i}"))
    return out


def test_walk_forward_learns_and_beats_baselines() -> None:
    sc = T.score(D.walk_forward(planted(), "x"))
    assert sc["brier"] < sc["brier_base_rate"] - 0.1 and sc["gain_ci95"][0] > 0


def test_walk_forward_never_sees_the_future() -> None:
    items = planted()
    base = D.walk_forward(items, "x")
    flipped = [D.BItem(i.keys, i.created, i.resolved, 1 - i.y if k >= 100 else i.y, i.subject) for k, i in enumerate(items)]
    again = D.walk_forward(flipped, "x")
    # a prediction made at creation of item k depends only on items resolved before: outcomes flipped from item 100 on change nothing before ~102
    assert [p.p for p in base[:100]] == [p.p for p in again[:100]]
    # and an item whose window has not closed yet is not learned from
    window = [D.BItem(("k",), 0.0, 100.0, 1, "w1"), D.BItem(("k",), 10.0, 20.0, 0, "w2"), D.BItem(("k",), 50.0, 60.0, 0, "w3")]
    pr = {p.subject: p for p in D.walk_forward(window, "x")}
    assert pr["w1"].p == 0.5 and pr["w2"].base == 0.5 and pr["w3"].base != 0.5    # w1 resolves at 100: invisible to w3, w2 resolved at 20 is visible


def test_split_selection_reports_the_heldout_part() -> None:
    sp = D.split_score(D.walk_forward(planted(), "x"))
    assert sp["select"]["n"] + sp["heldout"]["n"] == sp["all"]["n"] == 200 and sp["select"]["n"] == int(200 * D.SELECT_SPLIT)


def test_exclusions() -> None:
    for bad in ("state/livesim/x.json", ".env", "a/credentials.json", "secret_x.json", "api_key.txt", "W1.key.json", "x.answers.json", "creator/devbench/sealed/a"):
        assert D.excluded(bad), bad
    assert not D.excluded("state/research/algorithm/summary.json")


def test_research_skips_secret_and_answer_files(tmp_path: Path) -> None:
    (tmp_path / "fam").mkdir()
    (tmp_path / "fam" / "r1.json").write_text(json.dumps({"passed": True, "n": 3}))
    (tmp_path / "fam" / "r2.json").write_text(json.dumps({"passed": False}))
    (tmp_path / "fam" / "W1.key.json").write_text(json.dumps({"shift": True}))
    (tmp_path / "fam" / "tokens.json").write_text(json.dumps({"ok": True}))
    its = D.research_items(tmp_path)
    assert sorted(i.subject for i in its) == ["r1.json:passed", "r2.json:passed"]


def test_journal_and_plan_sources(tmp_path: Path) -> None:
    j = tmp_path / "J.md"
    j.write_text("## Journal\n\n- 2026-09-28, nupen kernel work\n  - swarm ledger\n- 2026-09-29: weekly7 backtest of the stock pattern\n- 2026-09-30: weekly7 trader blind\n", encoding="utf-8")
    its = D.journal_items(j)
    assert [i.y for i in its] == [0, 1, 0] and its[-1].resolved is None and its[0].resolved == its[1].created
    assert D.topic_of("nupen kernel swarm") == "creator" and D.topic_of("lunch") == "other"
    (tmp_path / "plan_explanations.jsonl").write_text(
        "\n".join(json.dumps({"at": f"2026-10-02T06:48:{i:02d}Z", "chosen": i % 2 == 0, "step": "s", "component": "K1", "value": 0.2, "node": f"n{i}"}) for i in range(6)), encoding="utf-8")
    pl = D.plan_items(tmp_path)
    assert len(pl) == 6 and [p.y for p in pl] == [1, 0, 1, 0, 1, 0]


def test_git_drills_use_windows_and_ignore_the_future(tmp_path: Path) -> None:
    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)
    git("init", "-q")
    for i, msg in enumerate(["feat: a", "feat: b", "fix: a again", "feat: c", "feat: d"]):
        (tmp_path / ("a.py" if i in (0, 2) else f"f{i}.py")).write_text(f"v{i}\n")
        git("add", "-A")
        git("commit", "-qm", msg, "--date", f"2026-01-0{i + 1}T00:00:00")
    fixed = D.git_items(tmp_path, 3, "fixed")
    assert [i.y for i in fixed[:2]] == [1, 0] and fixed[0].resolved is not None and fixed[-1].resolved is None   # the last 3 windows are still open
    churn = D.git_items(tmp_path, 3, "churn")
    assert churn[0].y == 1


def test_filler_hands_out_independent_jobs_and_is_idempotent(tmp_path: Path) -> None:
    st = tmp_path / "st"
    st.mkdir()
    (st / "plan_explanations.jsonl").write_text(
        "\n".join(json.dumps({"at": f"2026-10-02T06:48:{i:02d}Z", "chosen": i % 3 == 0, "step": "s", "component": "K1", "value": 0.2, "node": f"n{i}"}) for i in range(40)) + "
", encoding="utf-8")
    nj = D.drill_filler(st, tmp_path, tmp_path / "none.md", tmp_path / "res", sources=["plan_choice"])
    jobs = []
    while (j := nj()) is not None:
        jobs.append(j)
    assert len(jobs) == len(D.VARIANTS)
    for j in jobs:
        j()
    rows = T._jsonl(D.runs_path(st))
    assert len(rows) == len(D.VARIANTS) and all(r["source"] == "plan_choice" and r["resolved"] == 40 for r in rows)
    assert D.drill_filler(st, tmp_path, tmp_path / "none.md", tmp_path / "res", sources=["plan_choice"])() is None       # same data: nothing to redo
    b = D.best_variant(st, "plan_choice")
    assert b and b["variants_tried"] == len(D.VARIANTS) and "brier" in b["heldout"]
    with (st / "plan_explanations.jsonl").open("a") as f:
        f.write(json.dumps({"at": "2026-10-02T07:00:00Z", "chosen": True, "step": "s", "component": "K1", "value": 0.2, "node": "new"}) + "\n")
    assert D.drill_filler(st, tmp_path, tmp_path / "none.md", tmp_path / "res", sources=["plan_choice"])() is not None   # new data re-queues
