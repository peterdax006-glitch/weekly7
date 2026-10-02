"""Round-4 independent validation: regression tests for the open validator issues fixed in h8/revalidate (each failed before its fix)."""
from __future__ import annotations

import ast
import datetime as dt
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from creator import agents as A
from creator import debug as DBG
from creator import localworker as LW
from creator import meta as ME
from creator import model as M
from creator import recursion as R
from creator import selfworkers as SW
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-01T00:00:00+00:00")
    from creator import ledger as L
    monkeypatch.setattr(L, "current_provenance", lambda *a, **k: prov)


def test_k05_budget_record_does_not_lose_concurrent_calls(tmp_path: Path) -> None:
    pol = A.BudgetPolicy(enabled=True, daily_calls=999, daily_usd=999.0, per_call_usd=0.1, per_job_calls=999)
    b = A.Budget(tmp_path / "b.json", pol)
    threads = [threading.Thread(target=lambda i=i: [b.record(f"j{i}", 0.1, f"r{i}-{k}", "OK") for k in range(10)]) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["calls"]) == 80


def test_k10_each_guard_has_its_own_pooled_evidence_file() -> None:
    src = Path("creator/evaluate.py").read_text(encoding="utf-8")
    assert '_dev_pooled_{g}.json' in src and '_dev_pooled.json' not in src


def test_k10_a_learning_solver_is_refused_on_the_holdout(tmp_path: Path) -> None:
    from creator import evaluate as E

    class Learner:
        learn, memory = True, object()

    arm = E.Arm("learner", Learner(), {"v": 1})          # type: ignore[arg-type]
    other = E.Arm("other", lambda *a: None, {"v": 0})    # type: ignore[arg-type]
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="s", acceptance_criteria=("m",)))
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=list(M.GapKind)[0], description="g", importance=1.0))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(g,), hypothesis="h", design="d", metrics=(E.PRIMARY,), seed=0,
                                 baseline_ref="b", candidate_ref="c"))
    with pytest.raises(E.EvaluationError, match="learns"):
        E.compare(led, ex, other, arm, tasks=[], manifest={"digest": "x" * 16}, frozen_path=tmp_path / "f.json",
                  out_dir=tmp_path / "e", use_holdout=True)


def test_k11_timeout_keeps_stderr(tmp_path: Path) -> None:
    code = "import sys,time; sys.stderr.write('ERRTEXT\\n'); sys.stderr.flush(); sys.stdout.write('OUTTEXT\\n'); sys.stdout.flush(); time.sleep(30)"
    f = DBG.reproduce([sys.executable, "-c", code], timeout=3)
    assert f.reproduced and "TimeoutExpired" in f.output and "ERRTEXT" in f.output


def test_k11_the_word_assert_in_prose_is_not_an_assertion_failure() -> None:
    assert DBG.classify("we cannot assert that the planet is round") == "UNKNOWN"
    assert DBG.classify("E       assert 1 == 2") == "ASSERTION"
    assert DBG.classify("AssertionError: boom") == "ASSERTION"


def test_k13_nan_cost_is_refused() -> None:
    m = ME.MetaLearner()
    m.register(ME.Strategy("s"))
    with pytest.raises(ValueError):
        m.record("s", "c", True, float("nan"))
    with pytest.raises(ValueError):
        m.record("s", "c", True, math.inf)
    assert m.record("s", "c", True, 1.5).cost == 1.5


def test_k20_relevant_files_needs_only_the_package() -> None:
    import inspect
    assert list(inspect.signature(LW.relevant_files).parameters) == ["package"]


def test_k21_a_self_recursive_unused_private_is_deleted() -> None:
    src = "def pub():\n    return 1\n\n\ndef _rec(n):\n    return _rec(n - 1) if n else 0\n"
    out = SW.dead_private(src, lambda name: False)
    assert out is not None and "_rec" not in out
    ast.parse(out)
    used = "def pub():\n    return _helper()\n\n\ndef _helper():\n    return 1\n"
    assert SW.dead_private(used, lambda name: False) is None


def test_k24_forced_changes_are_bounded(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    wl = R.make_workload()
    with pytest.raises(ValueError, match="unknown process parameter"):
        R.step(led, wl, forced=R.Change("not_a_param", 1, 2, "x"))
    with pytest.raises(ValueError, match="outside the bounds"):
        R.step(led, wl, forced=R.Change("max_retries", 2, 10**9, "x"))


def test_record_validation_honours_an_explicit_validated_commit() -> None:
    """A component VALIDATED once and changed since could never be re-validated: the first VALIDATED report commit always won."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("record_validation", Path(__file__).resolve().parents[1] / "scripts" / "record_validation.py")
    rv = importlib.util.module_from_spec(spec)             # type: ignore[arg-type]
    spec.loader.exec_module(rv)                            # type: ignore[union-attr]
    head = rv.git("rev-parse", "HEAD").stdout.strip()
    assert rv.explicit_commit({"validated_commit": head}) == head
    assert rv.explicit_commit({"validated_commit": head[:10]}) == head
    assert rv.explicit_commit({"validated_commit": "deadbeefdeadbeef"}) is None
    assert rv.explicit_commit({}) is None


def test_record_validation_rejects_foreign_commits_staged_and_untracked_changes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Validator round 4 review: a commit on another branch was accepted, and staged edits or a new untracked module under a
    component's paths were not seen, so stale code could count as fresh."""
    import importlib.util
    import subprocess
    spec = importlib.util.spec_from_file_location("record_validation", Path(__file__).resolve().parents[1] / "scripts" / "record_validation.py")
    rv = importlib.util.module_from_spec(spec)             # type: ignore[arg-type]
    spec.loader.exec_module(rv)                            # type: ignore[union-attr]

    def g(*a: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    g("init", "-q", "-b", "main")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("x = 1\n", encoding="utf-8")
    g("add", "-A")
    g("commit", "-q", "-m", "base")
    base = g("rev-parse", "HEAD")
    g("checkout", "-q", "-b", "other")
    (tmp_path / "pkg" / "m.py").write_text("x = 2\n", encoding="utf-8")
    g("commit", "-q", "-am", "elsewhere")
    foreign = g("rev-parse", "HEAD")
    g("checkout", "-q", "main")
    monkeypatch.setattr(rv, "ROOT", tmp_path)
    assert rv.explicit_commit({"validated_commit": base}) == base
    assert rv.explicit_commit({"validated_commit": foreign}) is None              # not an ancestor of HEAD
    assert rv.unchanged_since(base, ["pkg"])
    (tmp_path / "pkg" / "m.py").write_text("x = 3\n", encoding="utf-8")
    g("add", "pkg/m.py")                                                          # staged only
    assert not rv.unchanged_since(base, ["pkg"])
    g("reset", "-q", "--hard")
    (tmp_path / "pkg" / "new.py").write_text("y = 1\n", encoding="utf-8")        # untracked new module
    assert not rv.unchanged_since(base, ["pkg"])
