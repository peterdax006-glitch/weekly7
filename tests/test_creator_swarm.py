"""K22: several of the system's own workers in parallel on disjoint parts, sized by free RAM (owner, 1 Oct 2026)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest

from creator import build as B
from creator import kernel as K
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import selfmodel as SM
from creator import selfworkers as SW
from creator import swarm as W
from creator import ledger as LG
from creator.ledger import Ledger


@pytest.fixture(scope="module", autouse=True)
def cached_provenance() -> Iterator[None]:
    """Ledger.append recomputes provenance (tree hashes of creator/ and engine/, git state of the real checkout) for every record,
    ~0.17 s x hundreds of appends. No test here edits that checkout, so the real value is computed once per module and reused
    with a fresh seed, config hash and timestamp; the ledger's own computation is covered by tests/test_creator_ledger.py."""
    base = LG.current_provenance()

    def cached(seed: Optional[int] = None, config: Optional[Mapping[str, Any]] = None, *a: Any, **k: Any) -> M.Provenance:
        cfg_hash = LG.sha256_text(LG.canonical(dict(config)))[:16] if config is not None else None
        return dataclasses.replace(base, seed=seed, config_hash=cfg_hash,
                                   timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    mp = pytest.MonkeyPatch()
    mp.setattr(LG, "current_provenance", cached)
    try:
        yield
    finally:
        mp.undo()


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def module(i: int) -> str:
    pad = "".join(f"\n\ndef f{i}_{k}(x):\n    return x + {k}\n" for k in range(i + 3))
    return f"import os\nimport json\n\n\ndef _dead_{i}():\n    return {i}\n\n\ndef use_{i}(x):\n    return x * {i}\n{pad}"


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "creator/__init__.py", "")
    put(r, "tests/__init__.py", "")
    for i in (1, 2, 3):
        put(r, f"creator/m{i}.py", module(i))
        put(r, f"tests/test_m{i}.py", f"from creator.m{i} import use_{i}\n\n\ndef test_use():\n    assert use_{i}(2) == {2 * i}\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    specs = [SM.CapabilitySpec(f"K0{i}", f"m{i}", (f"creator/m{i}.py",), (f"tests/test_m{i}.py",), 0) for i in (1, 2, 3)]
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=specs,
                          scope=("creator", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False),
                          mode="efficiency", measure_memory=False)


@pytest.fixture(autouse=True)
def size_only(monkeypatch: pytest.MonkeyPatch) -> None:
    real = P.plan_efficiency
    monkeypatch.setattr(P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: real(led, root, base, avoid, "size"))


def own() -> SW.SelfFirst:
    return SW.SelfFirst([SW.RuleWorker(), SW.SearchWorker()], None)


def test_parallel_workers_on_disjoint_modules_all_get_adopted(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    real_execute = K.execute
    gate = threading.Barrier(3, timeout=120)                                 # every worker waits until all three are running

    def gated_execute(*a: Any, **k: Any) -> Any:
        gate.wait()                      # BrokenBarrierError (test fails) if the swarm never runs three at once
        return real_execute(*a, **k)
    monkeypatch.setattr(K, "execute", gated_execute)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=3, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.2)
    assert rnd.peak_parallel == 3 and rnd.pulled_back == 0
    assert sorted(r.outcome for r in rnd.reports) == ["ADOPTED"] * 3, [(r.package, r.outcome, r.reason) for r in rnd.reports]
    targets = {Ledger(cfg.ledger_path, evidence_root=cfg.repo).get(
        next(e.id for e in Ledger(cfg.ledger_path, evidence_root=cfg.repo).of_type("WorkPackage")
             if e.record.package_id == r.package)).outputs[0] for r in rnd.reports}
    assert targets == {"creator/m1.py", "creator/m2.py", "creator/m3.py"}           # never two on the same module
    for i in (1, 2, 3):
        text = (cfg.repo / f"creator/m{i}.py").read_text(encoding="utf-8")
        assert "import os" not in text and f"_dead_{i}" not in text
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert led.verify() and O.claude_dependence(led)["self_share"] == 1.0


def test_little_ram_means_one_worker_at_a_time(cfg: K.KernelConfig) -> None:
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.5, floor_fraction=0.0, per_worker_gb=1.0, max_workers=3, free=lambda: 3.0,
                     total=lambda: 16.0,
                     observe=lambda running: 5.0)                         # each worker measured at 5 GB: only one fits
    rnd = W.run_round(cfg, own, gov, max_packages=2, poll_s=0.2)
    assert rnd.peak_parallel == 1 and len(rnd.reports) == 2


def test_tight_ram_pulls_the_youngest_back_and_adopts_nothing_of_it(cfg: K.KernelConfig) -> None:
    state = {"n": 0}

    def free() -> float:
        state["n"] += 1
        return 10.0 if state["n"] < 6 else 0.1                          # plenty at first, then tight

    gov = W.Governor(ramp_s=0.0, floor_min_gb=1.0, floor_fraction=0.0, pull_fraction=1.0, per_worker_gb=0.0, max_workers=3, free=free,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.2)
    assert rnd.pulled_back >= 1
    cancelled = [r for r in rnd.reports if r.outcome == "CANCELLED"]
    assert cancelled and all("pulled back" in r.reason and not r.merge_commit for r in cancelled)
    assert Ledger(cfg.ledger_path, evidence_root=cfg.repo).verify()


def test_hard_pull_back_kills_only_that_workers_processes(tmp_path: Path) -> None:
    """Regression (1 Oct): a cooperative pull-back came too late and the host stopped the whole swarm at critical RAM."""
    import json
    import sys
    import time
    box, other = tmp_path / "scratch" / "sb1", tmp_path / "elsewhere"
    box.mkdir(parents=True)
    other.mkdir()
    (box / ".creator_sandbox.json").write_text(json.dumps({"label": "CP9"}), encoding="utf-8")
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    inside = subprocess.Popen(sleeper, cwd=box)
    outside = subprocess.Popen(sleeper, cwd=other)
    try:
        deadline = time.monotonic() + 60.0                  # the sleepers may be slow to appear in the process table
        stopped = W.stop_worker_processes(tmp_path / "scratch", "CP9")
        while stopped < 1 and time.monotonic() < deadline:
            time.sleep(0.2)
            stopped = W.stop_worker_processes(tmp_path / "scratch", "CP9")
        assert stopped >= 1
        inside.wait(timeout=20)
        assert inside.returncode is not None and outside.poll() is None
        assert W.stop_worker_processes(tmp_path / "scratch", "NOPE") == 0
    finally:
        for p in (inside, outside):
            if p.poll() is None:
                p.kill()



def test_the_governor_fills_the_machine_with_measured_workers() -> None:
    """Owner, 1 Oct (asked 4+ times): use the memory. The reserve is 7% of THIS machine's RAM, not fixed GB that ate most of
    the ~3-4 GB actually available; workers are added while available - measured worker >= reserve."""
    gov = W.Governor(free=lambda: 3.0, total=lambda: 15.6, observe=lambda n: 0.3, max_workers=32)
    assert abs(gov.floor() - 1.092) < 1e-6                                   # 7% of 15.6 GB
    assert gov.can_start(0) and gov.can_start(20)                             # 3.0 - 0.375 >= 1.09: keep adding
    assert not W.Governor(free=lambda: 1.3, total=lambda: 15.6, observe=lambda n: 0.3).can_start(3)
    assert W.Governor(free=lambda: 0.7, total=lambda: 15.6).too_tight()       # < 0.75 x floor: pull back hard
    assert not W.Governor(free=lambda: 1.0, total=lambda: 15.6).too_tight()
    assert W.Governor(total=lambda: 4.0).floor() == 0.8                       # never below 0.8 GB


def test_a_worker_waiting_for_the_session_holds_no_slot() -> None:
    with W.waiting_on_thinker("CP7"):
        assert "CP7" in W.WAITING
    assert "CP7" not in W.WAITING


def test_leftover_memory_runs_filler_jobs(cfg: K.KernelConfig) -> None:
    done = []

    def filler():
        if len(done) >= 3:
            return None
        return lambda: done.append(1)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=4, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=1, poll_s=0.1, filler=filler, filler_budget=5)
    assert len(rnd.reports) == 1 and len(done) == 3                           # gap work done, then filler while memory allows


def test_pull_back_spares_finished_claude_work_when_it_can(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 Oct: the youngest worker was pulled back although it carried finished Claude work; it is now pulled back last."""
    import threading
    import time as _t
    from creator import planner as P2
    pulled: list[str] = []
    monkeypatch.setattr(W, "stop_worker_processes", lambda scratch, pid: pulled.append(pid) or 0)
    plans = iter([P2.Plan("g1", "K", "c1", "efficiency", None, "w1", "CPA", "x", "e", 1),
                  P2.Plan("g2", "K", "c2", "efficiency", None, "w2", "CPB", "x", "e", 1)])
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: next(plans, None))
    release = threading.Event()

    def fake_execute(cfg, worker, plan, *a, cancel=None, **k):
        release.wait(5)
        return K.CycleReport(1, "CANCELLED" if cancel.is_set() else "ADOPTED", plan.package_id, "K")
    monkeypatch.setattr(K, "execute", fake_execute)
    state = {"n": 0}

    def free() -> float:
        state["n"] += 1
        if state["n"] > 8:
            release.set()
        return 10.0 if state["n"] < 5 else 0.1
    gov = W.Governor(ramp_s=0.0, floor_min_gb=1.0, floor_fraction=0.0, pull_fraction=1.0, per_worker_gb=0.0, max_workers=2, free=free,
                     total=lambda: 16.0, observe=lambda n: None)
    W.HANDED_BACK.add("CPB")                                              # the youngest carries finished Claude work
    try:
        W.run_round(cfg, lambda: None, gov, max_packages=2, poll_s=0.05)
    finally:
        W.HANDED_BACK.discard("CPB")
    assert pulled and pulled[0] == "CPA"                                   # the older worker without finished work goes first


def test_a_worker_that_crashes_before_the_kernel_runs_is_reported_not_lost(cfg: K.KernelConfig) -> None:
    """Regression (validator round 3): an exception in a job thread (here: make_worker) left no report - the package vanished from
    the round and its ledger chain stayed IN_PROGRESS. It must surface as an ERROR report."""
    def broken() -> SW.SelfFirst:
        raise RuntimeError("worker could not be built")
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, broken, gov, max_packages=1, poll_s=0.1)
    assert [r.outcome for r in rnd.reports] == ["ERROR"] and "could not be built" in rnd.reports[0].reason
    assert rnd.outcome == "WORKED"


def test_finished_work_by_any_worker_is_pulled_back_last() -> None:
    """1 Oct run8: a student's finished CP0049/CP0050 were pulled back unmeasured; only the teacher's work was protected."""
    from types import SimpleNamespace
    from creator import swarm as SWM
    done = SWM.protect_finished(lambda plan, package, wd: SimpleNamespace(claimed_done=True))
    idle = SWM.protect_finished(lambda plan, package, wd: SimpleNamespace(claimed_done=False))
    SWM.HANDED_BACK.discard("PKG-done")
    SWM.HANDED_BACK.discard("PKG-idle")
    done(SimpleNamespace(package_id="PKG-done"), None, None)
    idle(SimpleNamespace(package_id="PKG-idle"), None, None)
    assert "PKG-done" in SWM.HANDED_BACK and "PKG-idle" not in SWM.HANDED_BACK
    SWM.HANDED_BACK.discard("PKG-done")


def _two_plan_round(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch, free_tight: float) -> list[str]:
    import threading
    from creator import planner as P2
    pulled: list[str] = []
    monkeypatch.setattr(W, "stop_worker_processes", lambda scratch, pid: pulled.append(pid) or 0)
    plans = iter([P2.Plan("g1", "K", "c1", "efficiency", None, "w1", "CPA", "x", "e", 1),
                  P2.Plan("g2", "K", "c2", "efficiency", None, "w2", "CPB", "x", "e", 1)])
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: next(plans, None))
    release = threading.Event()

    def fake_execute(cfg, worker, plan, *a, cancel=None, **k):
        release.wait(5)
        return K.CycleReport(1, "CANCELLED" if cancel.is_set() else "ADOPTED", plan.package_id, "K")
    monkeypatch.setattr(K, "execute", fake_execute)
    state = {"n": 0}

    def free() -> float:
        state["n"] += 1
        if state["n"] > 12:
            release.set()
        return 10.0 if state["n"] < 5 else free_tight
    gov = W.Governor(ramp_s=0.0, floor_min_gb=1.0, floor_fraction=0.0, pull_fraction=1.0, per_worker_gb=0.0, max_workers=2,
                     free=free, total=lambda: 16.0, observe=lambda n: None)
    W.HANDED_BACK.update({"CPA", "CPB"})                                    # both carry finished work
    try:
        W.run_round(cfg, lambda: None, gov, max_packages=2, poll_s=0.05)
    finally:
        W.HANDED_BACK.difference_update({"CPA", "CPB"})
    return pulled


def test_finished_work_is_not_pulled_back_when_ram_is_tight_but_not_critical(cfg: K.KernelConfig,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """1 Oct run9: with every running package protected, the old fallback pulled a protected one anyway (CP0051/52 by Nupen's
    student, unmeasured). Tight-but-safe now waits; only critical RAM may take finished work."""
    assert _two_plan_round(cfg, monkeypatch, free_tight=0.6) == []                  # 0.6 < floor 1.0, > critical 0.35
    assert _two_plan_round(cfg, monkeypatch, free_tight=0.1)                        # critical: the host comes first


def test_fillers_count_toward_the_worker_cap_and_start_after_a_ramp(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """1 Oct run9: fillers were not counted (peak 42 > max 32) and started back to back, so RAM fell to 9 MB before any reading
    could show them."""
    import threading
    import time as _t
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: None)
    hold = threading.Event()
    starts: list[float] = []
    live = {"n": 0, "max": 0}
    lock = threading.Lock()

    def filler():
        def job() -> None:
            with lock:
                starts.append(_t.monotonic())
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])
            hold.wait(3)
            with lock:
                live["n"] -= 1
        return job
    gov = W.Governor(ramp_s=0.3, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=2, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    threading.Timer(2.0, hold.set).start()
    W.run_round(cfg, lambda: None, gov, max_packages=0, poll_s=0.05, filler=filler, filler_budget=4)
    assert live["max"] <= 2                                                         # the cap holds with fillers alone
    assert all(b - a >= 0.29 for a, b in zip(starts, starts[1:]))                   # spaced by the ramp


def test_a_target_held_by_a_running_worker_pauses_planning_instead_of_ending_the_round(cfg: K.KernelConfig,
                                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """1 Oct: plan_one returned None while one package held the only free target, so the round ended with 2 packages although
    more work became plannable seconds later. Planning waits for the running set to change, and does not spin meanwhile."""
    import time
    from creator import planner as P2
    seen: list[tuple[str, ...]] = []
    state = {"calls": 0}

    def plan_one(c, led, main, base, exclude_components=(), exclude_paths=()):
        state["calls"] += 1
        seen.append(tuple(exclude_components))
        return P2.Plan("g1", "K", "c1", "efficiency", None, "w1", "CPA", "x", "e", 1) if state["calls"] == 1 else \
            (P2.Plan("g2", "K", "c2", "efficiency", None, "w2", "CPB", "x", "e", 1) if state["calls"] == 3 else None)
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", plan_one)

    def fake_execute(cfg, worker, plan, *a, **k):
        time.sleep(0.6)
        return K.CycleReport(1, "ADOPTED", plan.package_id, "K")
    monkeypatch.setattr(K, "execute", fake_execute)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=2, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, lambda: None, gov, max_packages=5, poll_s=0.05)
    assert sorted(r.package for r in rnd.reports) == ["CPA", "CPB"]
    assert state["calls"] == 5                  # A; none while A runs (asked once, not every poll); B; none while B runs; none, idle: end


def test_coverage_package_is_planned_worked_by_testgen_measured_and_adopted(cfg: K.KernelConfig,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The test-gap work source end to end: the kernel plans the module with the most untested public names, the system's own
    TestGenWorker writes tests, and the claim is the COMPUTED fall of the untested-name count (target and package)."""
    from creator import efficiency as E
    from creator import testgen as TG
    monkeypatch.setattr(P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: P._plan_coverage(led, root, base, avoid))
    before = E.uncovered_total(cfg.repo)
    assert before > 0 and "f3_0" in E.uncovered_public(cfg.repo, "creator/m3.py")
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, lambda: SW.SelfFirst([TG.TestGenWorker()], None), gov, max_packages=1, poll_s=0.2)
    assert [r.outcome for r in rnd.reports] == ["ADOPTED"], [(r.package, r.outcome, r.reason) for r in rnd.reports]
    assert rnd.reports[0].requirement == P.COVERAGE_KEY
    assert rnd.reports[0].details["detail"]["uncovered"][0] > rnd.reports[0].details["detail"]["uncovered"][1]
    assert E.uncovered_total(cfg.repo) < before
    assert any((cfg.repo / "tests").glob("test_gen_m*.py"))                  # tests were added, production untouched
    assert (cfg.repo / "creator/m3.py").read_text(encoding="utf-8") == module(3)


def test_a_coverage_package_that_edits_production_code_is_rejected(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import selfworkers as SW2
    monkeypatch.setattr(P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: P._plan_coverage(led, root, base, avoid))

    class Sneaky:
        name = "sneaky"

        def __call__(self, plan, package, workdir):
            put(workdir, "creator/m3.py", module(3) + "\n\ndef extra():\n    return 1\n")
            put(workdir, "tests/test_extra.py", "from creator.m3 import extra, f3_0\n\n\ndef test_x():\n    assert extra() == 1 and f3_0(1) == 1\n")
            return K.WorkResult(True, "edited production", by="sneaky")
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, lambda: SW2.SelfFirst([Sneaky()], None), gov, max_packages=1, poll_s=0.2)
    assert [r.outcome for r in rnd.reports] == ["REJECTED"] and "may only add tests" in rnd.reports[0].reason


def test_drain_lets_in_flight_work_finish_and_plans_nothing_new(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deploy pause must not kill a 35-45 min cycle: once draining, the running package finishes, nothing else is planned."""
    real_execute = K.execute
    draining = {"on": False}

    def execute(*a: Any, **k: Any) -> Any:
        draining["on"] = True                                               # the pause arrives while this package runs
        return real_execute(*a, **k)
    monkeypatch.setattr(K, "execute", execute)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.1, drain=lambda: draining["on"], drain_max_s=600.0)
    assert rnd.outcome == "DRAINED"
    assert [r.outcome for r in rnd.reports] == ["ADOPTED"]                   # it finished; the other two were never planned


def test_drain_before_anything_starts_runs_nothing(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("nothing may start while draining")
    monkeypatch.setattr(K, "execute", boom)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.1, drain=lambda: True, drain_max_s=0.0)
    assert rnd.outcome == "DRAINED" and rnd.reports == []


def test_drain_cancels_work_that_outlasts_the_wait(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    draining = {"on": False}

    def hang(*a: Any, **k: Any) -> Any:
        draining["on"] = True
        k["cancel"].wait(60)                                                # only the cancel from the drain timeout ends it
        return K.CycleReport(1, "CANCELLED", a[2].package_id, a[2].requirement_key, reason="drain timeout")
    monkeypatch.setattr(K, "execute", hang)
    gov = W.Governor(ramp_s=0.0, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=1, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.1, drain=lambda: draining["on"], drain_max_s=0.5)
    assert rnd.outcome == "DRAINED" and [r.outcome for r in rnd.reports] == ["CANCELLED"]
