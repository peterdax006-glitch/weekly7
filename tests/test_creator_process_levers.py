"""K20b: the recursion's process is consumed by the worker and measured on a (here: fake) dev workload. No real devbench run."""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from typing import Any

import pytest

from creator import model as M
from creator import process_levers as PL
from creator import recursion as R
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-01T00:00:00+00:00")
    monkeypatch.setattr("creator.ledger.current_provenance", lambda *a, **k: prov)


def fake_eval(p: R.ProcessConfig, tid: str) -> tuple[bool, bool]:
    """A task needs research_budget >= need (0 or 4, fixed by the task id). Retries never help; harmless always."""
    need = 0 if int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 3 == 0 else 4
    return p.research_budget >= need, True


IDS = [f"D{i:02d}" for i in range(120)]


def wl(seed: int) -> PL.DevWorkload:
    return PL.DevWorkload(seed=seed, chunk=12, reps=3, confirm=12, evaluate=fake_eval, dev_ids=IDS)


def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("creator_swarm_script", Path(__file__).resolve().parents[1] / "scripts" / "creator_swarm.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_levers_default_equal_old_worker_settings() -> None:
    lv = PL.levers(R.ProcessConfig())
    assert (lv.search_budget, lv.per_family, lv.pair_width, lv.rule_rounds) == (120, 40, 12, 25)


def test_adopted_change_is_read_by_worker_construction(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    rep = R.step(led, wl(1), forced=R.Change("research_budget", 3, 4, "t"))
    assert rep.adopted, rep.detail
    pf = tmp_path / "process.json"
    PL.save_process(R.current_process(led), pf)
    script = load_script()
    w = script.make_process_worker(None, pf)
    search = [x for x in w.own if x.name.endswith("search-v1")][0]
    assert search.budget == 160                                       # was 120 before the adoption
    default = script.make_process_worker(None, tmp_path / "missing.json")
    assert [x for x in default.own if x.name.endswith("search-v1")][0].budget == 120


def test_second_step_other_seed_decides_by_measurement(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    r1 = R.step(led, wl(1), forced=R.Change("research_budget", 3, 4, "t"))
    r2 = R.step(led, wl(2), forced=R.Change("research_budget", 4, 5, "t"))
    assert wl(1).chunks != wl(2).chunks                              # different conditions
    for r in (r1, r2):
        assert r.verdict in {v.value for v in M.Verdict}
        assert r.claim_id is not None
        claim: Any = led.get(r.claim_id)
        assert claim.verdict.value == r.verdict
        assert r.adopted == (r.verdict == "IMPROVEMENT")
    assert R.current_process(led).research_budget == (5 if r2.adopted else 4 if r1.adopted else 3)
    # a lever that cannot matter on the fake tasks is not adopted
    r3 = R.step(led, wl(3), forced=R.Change("max_retries", 2, 3, "t"))
    assert not r3.adopted and r3.verdict != "IMPROVEMENT"


def test_workload_is_dev_only_and_cached() -> None:
    w = wl(1)
    w(R.ProcessConfig())
    n = w.runs
    w(R.ProcessConfig())
    assert w.runs == n == 3 * 12 + 12
    assert set(sum(w.chunks, [])).isdisjoint(w.confirm_ids)


def test_real_workload_filters_to_dev_split() -> None:
    real = PL.DevWorkload(seed=1, chunk=2, reps=2, confirm=2)
    assert all(t.split == "dev" for t in real._tasks.values())
    assert set(sum(real.chunks, []) + real.confirm_ids) <= set(real._tasks)


def test_process_solver_passes_levers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[Any, ...]] = []

    def fake(task: Any, wd: Path, budget: int, pairs: bool, per_family: int, pair_width: int) -> tuple[bool, dict[str, str], int]:
        seen.append((budget, pairs, per_family, pair_width))
        return len(seen) == 2, {}, 5
    monkeypatch.setattr("creator.generator.solve_with_search", fake)
    res = PL.ProcessSolver(R.ProcessConfig()).__call__({}, tmp_path)
    assert res.claimed_done and seen == [(120, True, 40, 12), (120, True, 80, 12)]


def test_load_process_rejects_out_of_bounds_values(tmp_path: Path) -> None:
    """Regression (validator round 3): load_process trusted any integer, so max_retries=10**9 or research_budget=-5 reached the workers."""
    pf = tmp_path / "process.json"
    for bad in ({"max_retries": 10**9}, {"research_budget": -5}, {"design_breadth": 0}):
        pf.write_text(__import__("json").dumps(bad), encoding="utf-8")
        assert PL.load_process(pf) == R.ProcessConfig()
    pf.write_text('{"research_budget": 4}', encoding="utf-8")
    assert PL.load_process(pf).research_budget == 4


def test_process_solver_passes_the_levers_to_the_right_search_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (validator round 3): the levers were passed positionally into the wrong slots of solve_with_search
    (per_family landed in `rank`, pair_width in `extra_passes`), so design_breadth and reviewer_depth never acted."""
    from creator import generator as G
    seen: list[dict[str, Any]] = []

    def fake(task: Any, workdir: Path, budget: int = 0, pairs: bool = True, rank: bool = True, extra_passes: int = 5,
             extra_budget: int = 40, per_family: int = 40, pair_width: int = 12) -> tuple[bool, dict[str, str], int]:
        seen.append(dict(budget=budget, pairs=pairs, rank=rank, extra_passes=extra_passes, per_family=per_family,
                         pair_width=pair_width))
        return False, {}, 1
    monkeypatch.setattr(G, "solve_with_search", fake)
    PL.ProcessSolver(R.ProcessConfig(research_budget=2, design_breadth=3, reviewer_depth=2, max_retries=1))({}, tmp_path)
    assert seen[0] == dict(budget=80, pairs=True, rank=True, extra_passes=5, per_family=60, pair_width=24)
    assert seen[1]["per_family"] == 120


def test_devworkload_with_too_few_dev_tasks_raises_clearly() -> None:
    """Regression (validator open issue 6): too few dev tasks silently produced empty chunks scored 0.0."""
    with pytest.raises(ValueError, match="needs"):
        PL.DevWorkload(seed=1, chunk=4, reps=3, confirm=4, evaluate=fake_eval, dev_ids=[f"t{i}" for i in range(10)])



def test_an_adopted_worker_configuration_reaches_the_worker_the_swarm_builds(tmp_path: Path) -> None:
    """K19 -> process: autotune's adopted WorkerConfig (worker_config.json) sets the search budget of the built worker."""
    from creator import autotune as AT
    from creator import generator as G
    p = R.ProcessConfig()
    absent = tmp_path / "none.json"
    default_budget = PL.levers(p).search_budget
    search = lambda w: next(x for x in w.own if hasattr(x, "budget"))     # noqa: E731
    assert search(PL.build_worker(p, None, absent)).budget == default_budget
    cfg = tmp_path / "worker_config.json"
    AT.save_active(G.WorkerConfig(search_budget=480), "adopted by measured trial", cfg)
    assert PL.tuned_search_budget(p, cfg) == 480 != default_budget
    assert search(PL.build_worker(p, None, cfg)).budget == 480
