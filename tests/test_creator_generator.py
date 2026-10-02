"""K18: the Creator's own worker - local model + test-guided search, learning from its own experience (owner, 1 Oct 2026)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from creator import devbench as D
from creator import generator as G


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "w"
    put(r, "app/__init__.py", "")
    put(r, "app/dates.py", "def parse(s):\n    y, m, d = s.split('-')\n    return int(y), int(m) - 1, int(d)\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_dates.py", "from app.dates import parse\n\n\ndef test_parse():\n    assert parse('2024-03-15') == (2024, 3, 15)\n")
    return r


class FakeLLM:
    """Scripted replies standing in for the local model (the real one is exercised by scripts/generator_bench.py)."""
    model = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, **kw) -> str:
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "no idea"


GOOD = "FILE: app/dates.py\n```python\ndef parse(s):\n    y, m, d = s.split('-')\n    return int(y), int(m), int(d)\n```\n"
BAD = "FILE: app/dates.py\n```python\ndef parse(s):\n    return 0\n```\n"


def test_parse_files_refuses_paths_outside_the_task() -> None:
    reply = GOOD + "FILE: ../evil.py\n```python\nx = 1\n```\nFILE: /abs.py\n```python\nx = 1\n```\n"
    assert list(G.parse_files(reply)) == ["app/dates.py"]


def test_model_strategy_retries_with_the_failure(repo: Path) -> None:
    llm = FakeLLM([BAD, GOOD])
    ok, files, calls = G.solve_with_model({"id": "t", "objective": "fix parse", "category": "bugfix"}, repo, llm, None)
    assert ok and calls == 2 and "app/dates.py" in files
    assert "The tests fail" in llm.seen[1][-1]["content"]                 # the second attempt saw the failure


def test_model_strategy_gives_up_honestly(repo: Path) -> None:
    ok, _, calls = G.solve_with_model({"id": "t", "objective": "fix", "category": "bugfix"}, repo, FakeLLM([BAD] * 3), None)
    assert not ok and calls == 3


def test_search_repairs_an_off_by_one_without_any_model(repo: Path) -> None:
    ok, files, tried = G.solve_with_search({"id": "t", "objective": "fix"}, repo, budget=60)
    assert ok and tried <= 60
    assert G.visible_tests(repo).ok and "int(m)" in files["app/dates.py"] and "- 1" not in files["app/dates.py"]


def test_search_restores_the_file_when_it_fails(tmp_path: Path) -> None:
    put(tmp_path, "app/x.py", "def f():\n    return 'a'\n")
    put(tmp_path, "tests/test_x.py", "from app.x import f\n\n\ndef test_f():\n    assert f() == 'zzz'\n")
    before = (tmp_path / "app/x.py").read_text(encoding="utf-8")
    ok, _, _ = G.solve_with_search({"id": "t", "objective": "x"}, tmp_path, budget=20)
    assert not ok and (tmp_path / "app/x.py").read_text(encoding="utf-8") == before


def test_experience_orders_strategies_and_supplies_examples(tmp_path: Path) -> None:
    mem = G.ExperienceMemory(tmp_path / "mem.jsonl")
    assert mem.strategy_order("bugfix") == ["model", "search"]
    for _ in range(3):
        mem.add(G.Experience("bugfix", "fix the date parser", "search", True, {"app/a.py": "x"}, 1.0))
        mem.add(G.Experience("bugfix", "fix the date parser", "model", False, {}, 1.0))
    assert mem.strategy_order("bugfix") == ["search", "model"] and mem.strategy_order("feature") == ["model", "search"]
    mem.add(G.Experience("feature", "add a slugify function to text utils", "model", True, {"app/t.py": "def slugify(): ..."}, 2))
    ex = mem.examples("add a title function to text utils")
    assert ex and ex[0].objective.startswith("add a slugify") and mem.examples("completely unrelated words") == []


def test_holdout_runs_never_write_experience(repo: Path, tmp_path: Path) -> None:
    mem = G.ExperienceMemory(tmp_path / "mem.jsonl")
    G.GeneratorSolver(FakeLLM([GOOD]), memory=mem, learn=False)({"id": "H", "objective": "fix", "category": "bugfix"}, repo)
    assert mem.load() == []
    G.GeneratorSolver(FakeLLM([GOOD]), memory=mem, learn=True)({"id": "D", "objective": "fix", "category": "bugfix"}, repo)
    assert len(mem.load()) == 1 and mem.load()[0].solved_visible


def test_the_solver_is_scored_by_hidden_tests(tmp_path: Path) -> None:
    """Passing visible tests is only a claim: the sealed devbench decides (an overfit fix is FALSE_COMPLETION)."""
    tasks, sealed = tmp_path / "tasks", tmp_path / "sealed"
    repo = tasks / "T1" / "repo"
    for rel, text in {"app/__init__.py": "", "app/core.py": "def double(x):\n    return x + 1\n", "tests/__init__.py": "",
                      "tests/test_core.py": "from app.core import double\n\n\ndef test_two():\n    assert double(2) == 4\n"}.items():
        put(repo, rel, text)
    put(tasks / "T1", "task.json", json.dumps({"id": "T1", "category": "bugfix", "split": "dev", "objective": "double"}))
    put(sealed / "T1/hidden", "__init__.py", "")
    put(sealed / "T1/hidden", "test_h.py", "from app.core import double\n\n\ndef test_h():\n    assert double(5) == 10\n")
    D.seal(tasks, sealed, sealed / "M.json")
    task, m = D.load_tasks(tasks, sealed)[0], D.load_manifest(sealed / "M.json")
    overfit = "FILE: app/core.py\n```python\ndef double(x):\n    return x + 2\n```\n"                # passes visible only
    real = "FILE: app/core.py\n```python\ndef double(x):\n    return 2 * x\n```\n"
    assert D.run_task(task, G.GeneratorSolver(FakeLLM([overfit]), search_budget=0), m).outcome == "FALSE_COMPLETION"
    assert D.run_task(task, G.GeneratorSolver(FakeLLM([real]), search_budget=0), m).outcome == "SOLVED"


@pytest.mark.skipif(not (G.SERVER_EXE.is_file() and G.DEFAULT_MODEL.is_file()), reason="local model runtime not installed")
def test_the_real_local_model_answers_offline() -> None:
    with G.LocalModel() as llm:
        reply = llm.chat([{"role": "user", "content": "Reply with exactly: FILE: a.py then a python block defining x = 1"}],
                         max_tokens=60, temperature=0.0)
    assert reply.strip() and llm.calls == 1


# ---- targeted mutation families (one test per injected-bug class)

def _fixes(buggy: str, fixed: str, per_family: int = 40) -> bool:
    import ast
    want = ast.unparse(ast.parse(fixed))
    return any(ast.unparse(m) == want for m in G.targeted_mutations(ast.parse(buggy), per_family))


def test_family_wrong_variable() -> None:
    assert _fixes("def f(s):\n    total = 0\n    for ch in s:\n        total += 1\n    return ch\n",
                  "def f(s):\n    total = 0\n    for ch in s:\n        total += 1\n    return total\n")


def test_family_dropped_guard() -> None:
    assert _fixes("def f(xs):\n    return xs[0]\n", "def f(xs):\n    if not xs:\n        return []\n    return xs[0]\n")
    assert _fixes("def f(xs):\n    return xs[0]\n", "def f(xs):\n    if not xs:\n        return None\n    return xs[0]\n")


def test_family_guard_not_duplicated() -> None:
    import ast
    src = "def f(xs):\n    if not xs:\n        return 0\n    return xs[0]\n"
    assert not any(ast.unparse(m).count("if not xs") > 1 for m in G.targeted_mutations(ast.parse(src)))


def test_family_swapped_branches() -> None:
    assert _fixes("def f(x):\n    if x > 0:\n        return 'neg'\n    else:\n        return 'pos'\n",
                  "def f(x):\n    if x > 0:\n        return 'pos'\n    else:\n        return 'neg'\n")


def test_family_range_end() -> None:
    assert _fixes("def f(n):\n    t = 0\n    for i in range(1, n):\n        t += i\n    return t\n",
                  "def f(n):\n    t = 0\n    for i in range(1, n + 1):\n        t += i\n    return t\n")


@pytest.mark.parametrize("bad,good", [("max", "min"), ("sorted", "list"), ("sum", "len"), ("list", "sorted")])
def test_family_builtin_swaps(bad: str, good: str) -> None:
    assert _fixes(f"def f(xs):\n    return {bad}(xs)\n", f"def f(xs):\n    return {good}(xs)\n")


def test_family_method_swap() -> None:
    assert _fixes("def f(a, b):\n    a.append(b)\n    return a\n", "def f(a, b):\n    a.extend(b)\n    return a\n")


def test_targeted_families_are_bounded() -> None:
    import ast
    big = "def f(a, b, c, d):\n" + "".join(f"    t{i} = max(a, b) + sum(c)\n" for i in range(60)) + "    return a\n"
    assert len(list(G.targeted_mutations(ast.parse(big), 40))) <= 5 * 40


OVERFIT_SRC = "def f(n: int) -> int:\n    if n > 3:\n        return n * 2\n    else:\n        return n\n"


def _fast_visible(workdir: Path) -> D.TestCounts:
    """Stands in for pytest: the single visible test is f(3) == 6 (fast, deterministic)."""
    ns: dict = {}
    try:
        exec((workdir / "app/f.py").read_text(encoding="utf-8"), ns)
        ok = ns["f"](3) == 6
    except Exception:
        ok = False
    return D.TestCounts(1 if ok else 0, 0 if ok else 1, 0, 0 if ok else 1)


def test_ranking_prefers_the_minimal_behaviour_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the first visible-passing mutant swaps the branches (overfits f(3)); the right fix is the boundary, which changes nothing else
    monkeypatch.setattr(G, "visible_tests", _fast_visible)
    put(tmp_path, "app/f.py", OVERFIT_SRC)
    first = G.solve_with_search({"id": "t"}, tmp_path, budget=200, rank=False)
    put(tmp_path, "app/f.py", OVERFIT_SRC)
    ranked = G.solve_with_search({"id": "t"}, tmp_path, budget=200, rank=True)
    assert first[0] and ranked[0]
    assert G.behavioural_divergence(tmp_path, OVERFIT_SRC, first[1]["app/f.py"]) > 0
    assert G.behavioural_divergence(tmp_path, OVERFIT_SRC, ranked[1]["app/f.py"]) == 0
    assert "n >= 3" in ranked[1]["app/f.py"]


def test_divergence_counts_changed_calls_and_survives_broken_candidates(tmp_path: Path) -> None:
    assert G.behavioural_divergence(tmp_path, OVERFIT_SRC, OVERFIT_SRC) == 0
    assert G.behavioural_divergence(tmp_path, OVERFIT_SRC, "def f(n: int) -> int:\n    return 0\n") > 0
    assert G.behavioural_divergence(tmp_path, OVERFIT_SRC, "def f(:\n") > 0


def test_experience_memory_concurrent_adds_never_interleave(tmp_path: Path) -> None:
    """Regression (validator open issue 2): swarm threads share one experience file; a large record is written to the OS in several
    pieces, so unlocked appends interleave. The file here writes every record in small pieces, yielding between them."""
    import threading
    import time
    real = tmp_path / "mem.jsonl"

    class Piecewise:
        def __init__(self, fh: Any) -> None:
            self.fh = fh

        def __enter__(self) -> "Piecewise":
            self.fh.__enter__()
            return self

        def __exit__(self, *a: Any) -> None:
            self.fh.__exit__(*a)

        def write(self, data: Any) -> None:
            for k in range(0, len(data), 64):
                self.fh.write(data[k:k + 64])
                self.fh.flush()
                time.sleep(0.0005)

        def flush(self) -> None:
            self.fh.flush()

    class SlowPath:
        parent = tmp_path

        def open(self, mode: str = "r", **kw: Any) -> Piecewise:
            return Piecewise(real.open(mode, **kw))

    mem = G.ExperienceMemory(SlowPath())  # type: ignore[arg-type]

    def work(w: int) -> None:
        for i in range(4):
            mem.add(G.Experience("bugfix", f"obj {w} {i}", "search", True, {"app/a.py": "x = 1\n" * 300}, 1.0))
    ts = [threading.Thread(target=work, args=(w,)) for w in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = real.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 24 and len(G.ExperienceMemory(real).load()) == 24


def test_search_restores_the_file_when_visible_tests_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (validator open issue 3): an exception from visible_tests used to leave the last mutated candidate on disk."""
    put(tmp_path, "app/x.py", "def f(n):\n    return n + 1\n")
    put(tmp_path, "tests/test_x.py", "from app.x import f\n\n\ndef test_f():\n    assert f(1) == 5\n")
    before = (tmp_path / "app/x.py").read_text(encoding="utf-8")
    calls = {"n": 0}

    def boom(workdir: Path) -> object:
        calls["n"] += 1
        assert (tmp_path / "app/x.py").read_text(encoding="utf-8") != before        # a mutant is on disk when it raises
        raise RuntimeError("test runner died")
    monkeypatch.setattr(G, "visible_tests", boom)
    with pytest.raises(RuntimeError):
        G.solve_with_search({"id": "t", "objective": "x"}, tmp_path, budget=20)
    assert calls["n"] == 1 and (tmp_path / "app/x.py").read_text(encoding="utf-8") == before
