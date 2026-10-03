"""Reasoning drills (3 Oct 2026): verifiable multiple-choice questions from public git histories, a self-renewing strategy search over
reasoning methods, and the report. The frozen benchmark (creator.thinkbench part c) must never become training data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from creator import reasondrills as R

ROOT = Path(__file__).resolve().parents[1]


def commits(n: int = 120) -> list[R.Commit]:
    """A synthetic history: modules a0..a9 under pkg/, docs, tests; commit i touches pkg/a{i%10}.py (+ its test every 3rd commit)."""
    cs = []
    for i in range(n):
        ch = [("A" if i < 10 else "M", f"pkg/a{i % 10}.py")]
        if i < 30:
            ch.append(("A" if i < 20 else "M", f"docs/d{i % 20}.rst"))
        if i % 3 == 0:
            ch.append(("A" if i < 30 else "M", f"tests/test_a{i % 10}.py"))
        cs.append(R.Commit(hashlib.sha1(str(i).encode()).hexdigest(), 1_600_000_000.0 + 3600 * i, f"Improve behaviour number {i} of the engine", ch))
    return cs


def test_files_changed_has_one_right_answer_from_files_that_existed_then() -> None:
    cs = commits()
    qs = R.gen_files_changed("demo", cs)
    assert len(qs) > 50
    for q in qs:
        c = next(x for x in cs if x.h.startswith(q.subject))
        i = cs.index(c)
        changed = {p for _s, p in c.changes}
        before = {p for x in cs[:i] for _s, p in x.changes}
        assert len(q.options) == 4 and len(set(q.options)) == 4
        assert [o in changed for o in q.options].count(True) == 1 and q.options[q.answer] in changed          # exactly one right answer
        assert all(o in before for k, o in enumerate(q.options) if k != q.answer)                              # distractors existed then
        assert not any(R._names(c.s, o) for o in q.options)                                                    # the message names no option
    assert len(Counter(q.answer for q in qs)) == 4                                                             # shuffled, not always one letter
    assert {q.difficulty for q in qs} == {"easy", "hard"}


def test_a_subject_naming_its_file_is_dropped_and_private_subjects_never_used() -> None:
    cs = commits()
    cs[60].s = "Fix a0 handling in a0.py"
    cs[61].s = "Thanks someone@example.com for the patch"
    cs[62].s = "Merge branch 'feature' of someone/fork"
    hs = {c.h[:12] for c in cs[60:63]}
    assert not [q for q in R.gen_files_changed("demo", cs) if q.subject in hs]
    private = {c.h[:12] for c in cs[61:63]}
    assert not [q for q in R.gen_which_first("demo", cs, every=1) if any(k in private for k in q.keys)]


def test_which_first_answer_is_the_earliest_and_reverts_hide_their_own_subject() -> None:
    cs = commits()
    by = {c.h[:12]: c for c in cs}
    qs = R.gen_which_first("demo", cs, every=1)
    assert qs
    for q in qs:
        group = [by[k] for k in q.keys]
        first = min(group, key=lambda c: c.t)
        assert q.options[q.answer] == R._short(first.s, 120) and len(set(q.options)) == 4
    cs2 = commits()
    cs2.append(R.Commit("f" * 40, cs2[-1].t + 60, f'Revert "{cs2[100].s}"', [("M", "pkg/a0.py")]))
    rv = R.gen_revert_of("demo", cs2)
    assert len(rv) == 1 and rv[0].options[rv[0].answer] == cs2[100].s
    assert cs2[100].s not in rv[0].stem and "Revert" not in rv[0].stem.split("REVERTED")[0]


def test_co_change_counts_only_the_past_and_never_on_nupens_own_repo() -> None:
    cs = commits(400)
    qs = R.gen_co_change("demo", cs)
    assert qs
    for q in qs:
        tf = q.subject.split("@")[0]
        assert q.options[q.answer] == "pkg/" + tf.split("test_")[1]
    assert R.gen_co_change("nupen", cs) == []


def test_parse_log_reads_no_author_and_history_never_asks_for_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = "@@abc\t1600000000\tAdd x\nM\tpkg/x.py\nR100\told.py\tnew.py\n\n@@def\t1599990000\tFirst\nA\tpkg/x.py\n"
    cs = R.parse_log(text)
    assert [c.s for c in cs] == ["First", "Add x"] and cs[1].changes == [("M", "pkg/x.py"), ("D", "old.py"), ("A", "new.py")]
    seen: list[tuple[str, ...]] = []

    def fake_git(repo: Path, *args: str) -> str:
        seen.append(args)
        return "h1\n" if args[0] == "rev-parse" else text
    monkeypatch.setattr(R, "_git", fake_git)
    R._LOGS.clear()
    R.history(tmp_path)
    fmt = next(a for a in seen if a[0] == "log")[-1]
    assert not [x for x in ("%an", "%ae", "%aN", "%aE", "%cn", "%ce", "%cN", "%cE", "%b", "%B", "%G") if x in fmt]   # no name, e-mail or body


def test_repos_are_public_clones_plus_nupen_and_skip_half_clones(tmp_path: Path) -> None:
    pub = tmp_path / "public_repos"
    for n in ("flask", "django.tmp", "notgit"):
        (pub / n).mkdir(parents=True)
    (pub / "flask" / ".git").mkdir()
    (pub / "django.tmp" / ".git").mkdir()
    own = tmp_path / "own"
    (own / ".git").mkdir(parents=True)
    assert [n for n, _p in R.repos(own, pub)] == ["flask", "nupen"]


def frozen_state(tmp_path: Path, items: list[dict[str, Any]]) -> Path:
    d = tmp_path / "state"
    (d / "thinkbench").mkdir(parents=True)
    (d / "thinkbench" / "items.json").write_text(json.dumps({"c": items, "hash": "x"}), encoding="utf-8")
    return d


def test_generated_questions_never_collide_with_the_frozen_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cs = commits()
    q0 = R.gen_files_changed("nupen", cs)[0]
    other = R.gen_files_changed("nupen", cs)[1]
    items = [{"id": "c:test_file:creator/goals.py", "prompt": "Which test file covers the module creator/goals.py?\nA. x\nB. y\nC. z\nD. w", "answer": 0},
             {"id": f"c:adopted:{q0.subject}", "prompt": "Four packages...\nA. p\nB. q\nC. r\nD. s", "answer": 1},
             {"id": "c:capability:creator/x.py", "prompt": "Which capability?\nA. " + other.options[other.answer] + "\nB. y\nC. z\nD. w", "answer": 0}]
    state = frozen_state(tmp_path, items)
    monkeypatch.setattr(R, "history", lambda p: cs)
    got = R.generate([("nupen", tmp_path)], state)
    ids = {q.qid for q in got}
    assert q0.qid not in ids                                       # same subject as a frozen item
    assert other.qid not in ids                                    # its answer is a frozen answer (Nupen's own repo)
    fz = R.Frozen(items)
    for q in got:
        assert not fz.collides(q) and q.qid not in fz.ids and q.subject not in fz.subjects
        assert " ".join(q.prompt().split()) not in fz.texts
        assert q.kind in R.KINDS and q.kind not in ("test_file", "capability", "adopted", "constraint")


def test_real_frozen_benchmark_is_not_reproduced() -> None:
    """Against the live frozen benchmark when this machine has it: no generated question of Nupen's own repo shares id/subject/text/answer."""
    items_p = ROOT / "state" / "creator" / "thinkbench" / "items.json"
    alt = Path.home() / "weekly7" / "state" / "creator" / "thinkbench" / "items.json"
    p = items_p if items_p.exists() else alt
    if not p.exists():
        pytest.skip("no frozen benchmark on this machine")
    items = json.loads(p.read_text(encoding="utf-8"))["c"]
    fz = R.Frozen(items)
    assert len(fz.ids) == len(items) > 0
    qs = R.generate([("nupen", ROOT)], p.parent.parent)
    assert qs and not [q for q in qs if fz.collides(q)]


# ------------------------------------------------------------------------------------------------ strategies, filler, halving, report
def test_parse_vote_and_solved_examples_only_from_the_past_or_other_repos() -> None:
    assert R.parse_choice("drop A\nkeep C\nANSWER: C") == 2 and R.parse_choice("**ANSWER: (b)**") == 1 and R.parse_choice("no idea") is None
    assert R.vote([1, 2, 2, None]) == 2 and R.vote([3, 1]) == 3 and R.vote([None]) is None
    cs = commits()
    qs = R.gen_files_changed("demo", cs)
    q = qs[40]
    later = [x for x in qs if x.t > q.t][:3]
    earlier = [x for x in qs if x.t < q.t][-3:]
    other = R.Question("r:files_changed:else:zz", "files_changed", "else", "zz", q.t + 1e6, "Repository 'else'. stem", ["a", "b", "c", "d"], 0, "easy", ("zz",))
    same = R.Question(q.qid + "x", "files_changed", "demo", "s", q.t - 10, q.stem, q.options, q.answer, "easy", q.keys)
    solved = [(x, 0, 1.0) for x in later + earlier + [other, same]]
    got = [s.qid for s, _p in R.solved_examples(q, solved, 10, now=100.0)]
    assert set(got) == {x.qid for x in earlier} | {other.qid}         # never a later question of the same repo, never the same commit
    assert R.solved_examples(q, solved, 10, now=0.5) == []             # only questions answered before now
    msgs = R.build_messages(R.BY_NAME["retrieve4"], q, [(earlier[0], (earlier[0].answer + 1) % 4)])
    assert f"Correct answer: {R.LETTERS[earlier[0].answer]}" in msgs[-1]["content"] and "that was wrong" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith(q.prompt())


class FakeLLM:
    """Right only when asked to eliminate (or, for the control, on every 4th question): eliminate must win the search."""
    model = "fake-model.gguf"

    def __init__(self, truth: dict[str, int], log: list[str]) -> None:
        self.truth, self.log = truth, log

    def __enter__(self) -> "FakeLLM":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def chat(self, messages: Any, **kw: Any) -> str:
        sysm, user = messages[0]["content"], messages[-1]["content"]
        q = user.split("Now the question:\n")[-1]
        a = self.truth[q]
        self.log.append(sysm[:40])
        right = "one by one" in sysm or ("exactly one line" in sysm and hash(q) % 4 == 0)
        return f"ANSWER: {R.LETTERS[a if right else (a + 1) % 4]}"


def run_filler(tmp_path: Path, qs: list[R.Question], jobs: int) -> tuple[Path, list[str]]:
    state = tmp_path / "state"
    truth = {q.prompt(): q.answer for q in qs}
    log: list[str] = []
    nxt = R.reasoning_filler(state, tmp_path, max_servers=1, llm_factory=lambda: FakeLLM(truth, log), questions=lambda: list(qs), tag="fake-model.gguf")
    for _ in range(jobs):
        j = nxt()
        if j is None:
            break
        assert nxt() is None                                       # max_servers: one batch in flight at a time
        j()
    return state, log


def test_filler_hands_out_epochs_halves_and_retests_winners_on_fresh_questions(tmp_path: Path) -> None:
    qs = R.gen_files_changed("demo", commits(400))
    assert len(qs) > 2 * R.EPOCH_Q + 10
    state, _log = run_filler(tmp_path, qs, 400)
    rows = [json.loads(x) for x in R.path(state).read_text(encoding="utf-8").splitlines()]
    assert not [r for r in rows if r.get("error")]
    eps = R.epochs([r for r in rows if r.get("qid")])
    assert len(eps) >= 2 and eps[0]["complete"]
    assert eps[0]["pool"] == [s.name for s in R.STRATEGIES]
    assert eps[0]["alive"] == ["plain", "eliminate"] or set(eps[0]["alive"]) == {"plain", "eliminate", "retrieve4_elim"}
    assert "eliminate" in eps[1]["pool"] and "plain" in eps[1]["pool"] and len(eps[1]["pool"]) > len(eps[0]["alive"])   # challengers back in
    assert not set(eps[0]["questions"]) & set(eps[1]["questions"])                            # the winners are re-tested on FRESH questions
    per = Counter((r["epoch"], r["qid"]) for r in rows if r.get("strategy") == "plain")
    assert max(per.values()) == 1                                                             # nothing answered twice by one strategy
    vote = [r for r in rows if r.get("strategy") == "vote3"]
    assert vote and all(len(r["picks"]) == 3 for r in vote)


def test_report_has_accuracy_cis_vs_chance_and_plain_and_the_supply(tmp_path: Path) -> None:
    qs = R.gen_files_changed("demo", commits(400))
    state, _log = run_filler(tmp_path, qs, 400)
    rep = R.report_section(state, tag="fake-model.gguf")
    el, pl = rep["strategies"]["eliminate"], rep["strategies"]["plain"]
    assert el["accuracy"][0] == 1.0 and el["beats_chance"] and el["n"] >= R.EPOCH_Q
    assert el["gain_vs_plain"][1] > 0                                       # CI lower bound above zero vs the control
    assert pl["accuracy"][0] < 0.5 and pl["gain_vs_plain"] is None
    assert rep["supply"]["available"] == len(qs) and rep["supply"]["fresh"] == len(qs) - len({r for r in _qids(state)})
    assert rep["epochs"][1]["out_of_sample"] and rep["curve"][0]["best"][0] in ("eliminate", "retrieve4_elim")


def _qids(state: Path) -> set[str]:
    return {json.loads(x).get("qid") for x in R.path(state).read_text(encoding="utf-8").splitlines()} - {None}


def test_supply_running_low_is_reported_not_silent(tmp_path: Path) -> None:
    qs = R.gen_files_changed("demo", commits(80))
    state, _log = run_filler(tmp_path, qs, 3)
    st = json.loads(R.status_path(state).read_text(encoding="utf-8"))
    assert st["need_acquisition"] is True and st["low"] is True and st["available"] == len(qs) < R.LOW_FRESH


def _swarm_script():                                                 # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("creator_swarm_rd", ROOT / "scripts" / "creator_swarm.py")
    mod = importlib.util.module_from_spec(spec)                      # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                     # type: ignore[union-attr]
    return mod


def test_swarm_thinking_filler_rotates_reasoning_with_judgment_and_drills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import focus as F
    cs = _swarm_script()
    monkeypatch.setattr(cs, "STATE", tmp_path)
    calls: list[str] = []
    mods = {"thinking": SimpleNamespace(run=lambda state: calls.append("run")),
            "drillsources": SimpleNamespace(drill_filler=lambda *a, **k: (lambda: (lambda: calls.append("drill")))),
            "judgment": SimpleNamespace(judgment_filler=lambda *a, **k: (lambda: (lambda: calls.append("judge")))),
            "reasondrills": SimpleNamespace(reasoning_filler=lambda *a, **k: (lambda: (lambda: calls.append("reason")))),
            "device": SimpleNamespace(settings=lambda: {"llama_servers": 1})}
    monkeypatch.setattr(cs.REG, "get", lambda name: mods[name])
    monkeypatch.setattr(cs.REG, "optional", lambda name: F if name == "focus" else None)
    monkeypatch.setattr(cs, "_THINK_LAST", {"run": 1e18, "blueprint": 1e18})
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    nxt = cs.make_filler()
    for _ in range(12):
        nxt()()
    c = Counter(calls)
    assert c["reason"] == 2 and c["judge"] == 2 and c["drill"] == 8          # model batches take turns; drills keep two thirds


def test_trust_report_carries_the_reasoning_section_and_the_gate_is_untouched(tmp_path: Path) -> None:
    from creator import thinking as T
    qs = R.gen_files_changed("demo", commits(80))
    state, _log = run_filler(tmp_path, qs, 2)
    sec = T.reasoning_section(state)
    assert "answers" in sec and "supply" in sec
    sc = {"n": 10, "brier": 0.2, "brier_base": 0.25, "brier_last": 0.3, "gain_ci95": [0.01, 0.05], "ece": 0.01}
    assert T.trust_of(sc) == T.trust_of(dict(sc))                             # the gate is a pure function of the score, unchanged


def test_registry_exposes_reasondrills() -> None:
    from creator import registry as REG
    assert REG.get("reasondrills") is R


def test_real_git_repo_round_trip(tmp_path: Path) -> None:
    repo = tmp_path / "pub" / "tiny"
    repo.mkdir(parents=True)

    def git(*a: str, env: dict[str, str] | None = None) -> None:
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, env=env)
    import os
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    for i in range(40):
        (repo / f"m{i % 7}.py").write_text(f"x = {i}\n", encoding="utf-8")
        e = dict(os.environ, GIT_AUTHOR_DATE=f"2020-0{1 + i // 20}-{i % 20 + 1:02d}T12:00:00", GIT_COMMITTER_DATE=f"2020-0{1 + i // 20}-{i % 20 + 1:02d}T12:00:00")
        git("add", "-A")
        git("commit", "-q", "-m", f"Tune the parser step {i}", env=e)
    R._LOGS.clear()
    cs = R.history(repo)
    assert len(cs) == 40 and cs[0].s == "Tune the parser step 0"
    qs = R.gen_which_first("tiny", cs, every=1)
    assert qs and all("example.invalid" not in q.prompt() for q in qs)
