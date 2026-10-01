"""CR06: improvement measurement (C77 secs 30-32, 35). Small TEMPORARY devbench; real solvers are null / reference / liar."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import devbench as D
from creator import evaluate as E
from creator import ledger as L
from creator import model as M


def put(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[D.Task], dict, Path]:
    root = tmp_path_factory.mktemp("bench")
    tasks, sealed = root / "tasks", root / "sealed"
    specs = [("T1", "dev", "x + 1", "2 * x"), ("T2", "dev", "x - 1", "2 * x"), ("T3", "dev", "x", "2 * x"),
             ("H1", "holdout", "x * 3", "2 * x")]
    for tid, split, bug, fix in specs:
        repo = tasks / tid / "repo"
        put(tasks / tid / "task.json", json.dumps({"id": tid, "category": "bugfix", "split": split, "objective": "fix double"}))
        put(repo / "app/__init__.py", "")
        put(repo / "app/core.py", f"def double(x):\n    return {bug}\n")
        put(repo / "tests/__init__.py", "")
        put(repo / "tests/test_core.py", "from app.core import double\n\n\ndef test_zero():\n    assert double(0) in (0, 1, -1)\n")
        put(sealed / tid / "hidden/__init__.py", "")
        put(sealed / tid / "hidden/test_hidden.py", "from app.core import double\n\n\ndef test_h():\n    assert double(7) == 14\n")
        put(sealed / tid / "reference/app/core.py", f"def double(x):\n    return {fix}\n")
    D.seal(tasks, sealed, sealed / "MANIFEST.json")
    return D.load_tasks(tasks, sealed), D.load_manifest(sealed / "MANIFEST.json"), sealed


@pytest.fixture()
def world(tmp_path: Path) -> tuple[L.Ledger, str]:
    led = L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="improve development", acceptance_criteria=("measured",)))
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=list(M.GapKind)[0], description="solver", importance=1.0))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(g,), hypothesis="the candidate solves more tasks",
                                 design="paired devbench", metrics=(E.PRIMARY,), seed=0, baseline_ref="b", candidate_ref="c"))
    return led, ex


def run(world: tuple[L.Ledger, str], bench: tuple, base: E.Arm, cand: E.Arm, tmp_path: Path, **kw) -> E.EvaluationReport:
    led, ex = world
    tasks, manifest, _ = bench
    return E.compare(led, ex, base, cand, tasks=tasks, manifest=manifest, frozen_path=tmp_path / "frozen.json",
                     out_dir=led.evidence_root / "evals", **kw)


def test_a_real_improvement_is_found_and_the_ledger_agrees(world, bench, tmp_path: Path) -> None:
    _, _, sealed = bench
    rep = run(world, bench, E.Arm("null", D.NullSolver(), {"v": 0}), E.Arm("ref", D.ReferenceSolver(sealed), {"v": 1}), tmp_path)
    led = world[0]
    assert rep.verdict is M.Verdict.IMPROVEMENT, rep.detail
    assert rep.paired_dev == E.Paired(0, 0, 3, 0) and rep.paired_holdout == E.Paired(0, 0, 1, 0)
    assert led.verify() == len(led.view.entries) > 10
    claim = led.get(rep.claim_id)
    assert claim.holdout_candidate_id and len(claim.candidate_ids) == 2      # type: ignore[attr-defined]
    for mid in claim.baseline_ids:                                           # type: ignore[attr-defined]
        assert led.get(mid).evidence and not led.get(mid).evidence[0].problem(led.evidence_root)


def test_no_difference_is_never_an_improvement(world, bench, tmp_path: Path) -> None:
    rep = run(world, bench, E.Arm("null-a", D.NullSolver(), {"v": 0}), E.Arm("null-b", D.NullSolver(), {"v": 9}), tmp_path)
    assert rep.verdict in (M.Verdict.NO_EFFECT, M.Verdict.INSUFFICIENT_EVIDENCE)


def test_a_lying_candidate_is_a_regression(world, bench, tmp_path: Path) -> None:
    _, _, sealed = bench
    rep = run(world, bench, E.Arm("ref", D.ReferenceSolver(sealed), {"v": 1}), E.Arm("liar", D.LiarSolver(), {"v": 2}), tmp_path)
    assert rep.verdict is M.Verdict.REGRESSION and rep.paired_dev.only_base == 3


def test_without_holdout_it_cannot_be_an_improvement(world, bench, tmp_path: Path) -> None:
    _, _, sealed = bench
    rep = run(world, bench, E.Arm("null", D.NullSolver(), {"v": 0}), E.Arm("ref", D.ReferenceSolver(sealed), {"v": 1}), tmp_path,
              use_holdout=False)
    assert rep.verdict is M.Verdict.INSUFFICIENT_EVIDENCE and rep.detail["why"] == "no holdout"


def test_one_replicate_is_refused(world, bench, tmp_path: Path) -> None:
    with pytest.raises(E.EvaluationError, match="two replicates"):
        run(world, bench, E.Arm("a", D.NullSolver(), {}), E.Arm("b", D.NullSolver(), {"x": 1}), tmp_path, replicates=1)


def test_rate_se_never_claims_certainty() -> None:
    assert E.rate_with_se(0, 3)[1] > 0.1 and E.rate_with_se(3, 3)[1] > 0.1
    assert E.rate_with_se(300, 300)[1] < E.rate_with_se(3, 3)[1]
    with pytest.raises(E.EvaluationError):
        E.rate_with_se(0, 0)


def test_rerunning_the_same_tasks_does_not_manufacture_significance() -> None:
    """Regression (30 Sep): pooling replicates as independent shrank SE by sqrt(k); 100 reruns of a tiny gain looked real."""
    def ms(v: float) -> M.Measurement:
        return M.Measurement(created_by=M.Role.VALIDATOR, metric="solve_rate", value=v, stderr=0.15, n=10, population="p",
                             conditions="c", higher_is_better=True, split=M.Split.DEV,
                             computation=M.ComputationRef("creator.evaluate:metrics_of", "ab" * 8, "cd" * 8, "ef" * 8))
    v, d = M.improvement_verdict([ms(0.5)] * 100, [ms(0.6)] * 100)
    assert v is not M.Verdict.IMPROVEMENT and d["se"] > 0.2


def test_sign_test() -> None:
    assert E.Paired(5, 0, 0, 5).sign_test_p == 1.0
    assert E.Paired(0, 0, 6, 0).sign_test_p == pytest.approx(2 / 64)
    assert E.Paired(0, 3, 3, 0).sign_test_p == 1.0
