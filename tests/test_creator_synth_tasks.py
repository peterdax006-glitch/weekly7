"""creator.synth_tasks - the cheap in-process workload: purity, determinism, disjoint seeds, lever sensitivity, no devbench access."""
from __future__ import annotations

import ast
from pathlib import Path

from creator import process_levers as PL
from creator import recursion as R
from creator import synth_tasks as S

SRC = '''
def clamp(x: int, lo: int = 0, hi: int = 10) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x + 0

def spread(xs: list, k: int = 2) -> int:
    best = 0
    for i in range(len(xs)):
        if xs[i] > best:
            best = xs[i]
    return best * k + len(xs)

def evil(p):
    return open(p).read() + str(p) + str(p) + str(p) + str(p)
'''


def bench(tmp_path: Path) -> S.SynthBench:
    (tmp_path / "creator").mkdir()
    (tmp_path / "creator" / "m.py").write_text(SRC, encoding="utf-8")
    return S.SynthBench(root=tmp_path)


def test_purity_rejects_io_and_unknown_names() -> None:
    fns = {n.name: n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef)}
    assert S.pure_function(fns["clamp"]) and S.pure_function(fns["spread"]) and not S.pure_function(fns["evil"])


def test_corpus_skips_devbench_and_sealed(tmp_path: Path) -> None:
    b = bench(tmp_path)
    (tmp_path / "creator" / "devbench").mkdir()
    (tmp_path / "creator" / "devbench" / "secret.py").write_text(SRC.replace("clamp", "hidden_fn"), encoding="utf-8")
    assert all("devbench" not in name and "hidden_fn" not in name for name, _ in S.corpus(tmp_path)) and b.funcs


def test_tasks_deterministic_and_seeds_differ(tmp_path: Path) -> None:
    b1, b2 = bench(tmp_path), S.SynthBench(root=tmp_path)
    a, a2, c = b1.ids(1, 6), b2.ids(1, 6), b1.ids(2, 6)
    assert a == a2 and set(a) != set(c)
    t = b1.task(a[0])
    assert t is not None and not t.check(t.buggy, t.visible)           # the bug breaks a visible call
    assert t.check(ast.Module(body=[t.fn], type_ignores=[]), t.visible + t.hidden)   # the original passes all


def test_search_solves_some_and_levers_matter(tmp_path: Path) -> None:
    b = bench(tmp_path)
    ids = b.ids(3, 12)
    rich = R.ProcessConfig(research_budget=8, design_breadth=4, reviewer_depth=3, max_retries=4)
    poor = R.ProcessConfig(research_budget=0, design_breadth=1, reviewer_depth=0, max_retries=0)
    solved = lambda p: sum(1 for i in ids if b.evaluate(p, i)[0])      # noqa: E731
    assert solved(poor) == 0                                           # a zero budget tries nothing
    assert solved(rich) >= 3 and solved(rich) >= solved(R.ProcessConfig())


def test_workload_is_a_devworkload_with_disjoint_arms(tmp_path: Path) -> None:
    wl = S.workload(seed=1, chunk=3, reps=2, confirm=3, bench=bench(tmp_path))
    assert isinstance(wl, PL.DevWorkload)
    flat = [t for c in wl.chunks for t in c]
    assert not set(flat) & set(wl.confirm_ids) and len(set(flat)) == 6
    s = wl(R.ProcessConfig())
    assert len(s.dev) == 2 and s.n_holdout == 3
