"""The machine-wide test-process budget (creator.testslots): bounded concurrency, crash-safe slots, identical verdicts, and a
Governor that reserves the evaluation's memory. Fast: fake test commands only."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from creator import swarm as W
from creator import testrun as T
from creator import testslots as TS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUPEN_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("NUPEN_TEST_MB_FILE", str(tmp_path / "mb.json"))


def _roomy() -> tuple[float, float]:
    return 64.0, 64.0


def test_concurrent_holders_never_exceed_the_slot_count() -> None:
    now = [0]
    peak = [0]
    lock = threading.Lock()

    def worker() -> None:
        with TS.Slot(cap=3, mem=_roomy, poll_s=0.01):
            with lock:
                now[0] += 1
                peak[0] = max(peak[0], now[0])
            time.sleep(0.05)
            with lock:
                now[0] -= 1

    ts = [threading.Thread(target=worker) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert 1 <= peak[0] <= 3 and now[0] == 0


def test_run_launches_a_fake_command_under_a_slot_and_measures_it() -> None:
    cp = TS.run([sys.executable, "-c", "import time; time.sleep(0.5); print('hi')"], timeout=240, text=True)
    assert cp.returncode == 0 and cp.stdout.strip() == "hi"
    assert TS.per_test_mb() >= TS.MIN_MB and (Path(TS.mb_file())).is_file()


def test_run_timeout_raises_like_subprocess_run() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        TS.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5, text=True)


def test_a_killed_holder_frees_its_slot(tmp_path: Path) -> None:
    code = ("import sys,time; from pathlib import Path\n"
            "from creator import testslots as TS\n"
            "s = TS.Slot(cap=1, mem=lambda: (64.0, 64.0)); s.__enter__(); print('held', s.index, flush=True); time.sleep(30)\n")
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "held 0"
        blocked = TS.Slot(cap=1, mem=_roomy, wait_s=0.3, poll_s=0.05)
        blocked.__enter__()
        assert blocked.index is None                       # the holder is alive: no slot (the wait gave up, fail open)
        TS._kill_tree(p.pid)                               # the venv launcher's child is the real holder: kill the tree
        p.wait(timeout=10)
        freed = TS.Slot(cap=1, mem=_roomy, wait_s=5, poll_s=0.05)
        with freed:
            assert freed.index == 0                        # the OS dropped the dead holder's lock
    finally:
        TS._kill_tree(p.pid)
        if p.stdout:
            p.stdout.close()


def test_extra_slots_wait_while_free_ram_cannot_hold_one_more_test() -> None:
    with TS.Slot(cap=4, mem=_roomy):                       # takes slot 0
        tight = TS.Slot(cap=4, mem=lambda: (1.0, 15.6), mb=lambda: 500.0, wait_s=0.3, poll_s=0.05)
        tight.__enter__()
        assert tight.index is None                         # 1.0 GB free < 1.09 GB reserve + one test: queued, then fail open
        roomy = TS.Slot(cap=4, mem=_roomy, mb=lambda: 500.0)
        with roomy:
            assert roomy.index == 1


def _tree(root: Path) -> None:
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_ok.py").write_text("def test_a():\n    assert 1 + 1 == 2\n\ndef test_b():\n    assert True\n")
    (root / "tests" / "test_bad.py").write_text("def test_c():\n    assert False\n")
    (root / "tests" / "test_third.py").write_text("import pytest\n\n@pytest.mark.skip\ndef test_d():\n    pass\n\ndef test_e():\n    pass\n")


def _verdicts(root: Path, out: Path, cap: int, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    monkeypatch.setenv("NUPEN_TEST_SLOTS_MAX", str(cap))
    got: dict[str, dict[str, str]] = {}

    def one(name: str) -> None:
        r = T.run_pytest(root, [f"tests/{name}"], out / f"{name}.xml", label="c")
        got[name] = {"status": r.status.value, **{k: c.outcome.value for k, c in sorted(r.cases.items())}}

    ts = [threading.Thread(target=one, args=(n,)) for n in ("test_ok.py", "test_bad.py", "test_third.py")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return got


def test_verdicts_are_identical_with_budget_1_and_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tree(tmp_path / "t")
    one = _verdicts(tmp_path / "t", tmp_path / "o1", 1, monkeypatch)
    four = _verdicts(tmp_path / "t", tmp_path / "o4", 4, monkeypatch)
    assert one == four and one["test_bad.py"]["status"] != one["test_ok.py"]["status"]


def test_governor_refuses_a_worker_whose_evaluation_reservation_does_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    mb = {"v": 1000.0}
    monkeypatch.setattr(TS, "per_test_mb", lambda: mb["v"])
    free = [3.0]
    kw = dict(free=lambda: free[0], total=lambda: 15.6, per_worker_gb=0.4, floor_min_gb=0.8)
    plain = W.Governor(**kw)                                           # type: ignore[arg-type]
    budgeted = W.Governor(test_parallel=4, **kw)                       # type: ignore[arg-type]
    assert plain.can_start(0)                                          # 3.0 - 0.4 >= 1.09: the old rule admits it
    assert not budgeted.can_start(0)                                   # + 4 x 1000 MB of evaluation: 3.0 - 0.4 - 3.9 < floor
    free[0] = 8.0
    assert budgeted.can_start(0)
    # the second worker shares the slots (cap 4): its marginal reservation is zero, only its own memory counts
    monkeypatch.setenv("NUPEN_TEST_SLOTS_MAX", "4")
    assert budgeted.reservation(1) == 0.0 and budgeted.reservation(0) > 3.0


def test_a_nested_test_launch_runs_under_its_ancestors_slot_instead_of_deadlocking(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """2 Oct 2026, new PC: 8 outer test files held all 8 machine slots while the test runs they started internally waited
    for a slot - nothing moved for 30 minutes. A launch now marks its children; a nested launch on the machine pool takes
    nothing and returns at once, so the cap still counts every top-level run."""
    import subprocess
    import sys
    import time

    from creator import testslots as TS
    pool = tmp_path / "pool"
    monkeypatch.setattr(TS, "slot_dir", lambda: pool)
    monkeypatch.setattr(TS, "slot_cap", lambda: 1)
    monkeypatch.delenv(TS.HELD_ENV, raising=False)
    child = ("import os, sys, time; sys.path.insert(0, %r); from creator import testslots as TS; "
             "TS.slot_dir = lambda: __import__('pathlib').Path(%r); TS.slot_cap = lambda: 1; "
             "t = time.monotonic(); s = TS.Slot(wait_s=20); s.__enter__(); "
             "print(os.environ.get(TS.HELD_ENV), s.nested, s.index, round(time.monotonic() - t, 1))") % (str(TS.ROOT), str(pool))
    t0 = time.monotonic()
    out = TS.run([sys.executable, "-c", child], timeout=60)                  # holds the ONLY slot while the child asks for one
    text = out.stdout.decode() if isinstance(out.stdout, bytes) else out.stdout
    held, nested, index, waited = text.split()
    assert (held, nested, index) == ("1", "True", "None"), text
    assert float(waited) < 5 and time.monotonic() - t0 < 30                 # before the fix: waited the full wait_s
    with TS.Slot(wait_s=5) as s:                                              # the parent's slot was released afterwards
        assert s.index == 0 and not s.nested
    monkeypatch.setenv(TS.HELD_ENV, "1")                                      # a private pool is never skipped
    with TS.Slot(wait_s=5, directory=tmp_path / "private", cap=1) as s:
        assert s.index == 0 and not s.nested
