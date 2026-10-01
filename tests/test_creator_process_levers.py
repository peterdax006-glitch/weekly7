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
