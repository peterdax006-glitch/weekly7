"""A machine-wide budget for test processes (owner, 2 Oct 2026: "increase RAM efficiency").

Measured 2 Oct, live Nupen: 178 pytest processes = 5.0 GB. Every kernel evaluation, devbench acceptance run and temp-repo test
launches pytest, and nothing bounded how many ran at once on the whole machine, so the Governor admitted workers whose
evaluations then pushed free RAM to 0.4 GB and work was pulled back unfinished.

Every test launch in creator/ goes through `run()`, which holds one of N slots (N lock files, OS byte-range locks, so a killed
holder frees its slot at once - same idea as generator.MachineLock) for the life of the child. A launch only waits; it never
changes WHICH tests run or what they report. Slot i beyond the first is only taken while free RAM minus a reserve still covers
the MEASURED memory of one more test process (state/creator/test_proc_mb.json, updated from every real launch); slot 0 is always
available, so progress never stops. If the wait ever exceeds `wait_s` the launch proceeds unbudgeted (fail open: a budget must
never turn into a hang or a changed verdict)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
MIN_MB = 20.0


def _device() -> Any:
    """creator/device.py is the one source of truth for the cap and the per-test MB; loaded on demand (sparse activation)."""
    from creator import device
    return device


RESERVE_FRACTION, RESERVE_MIN_GB = 0.07, 0.8          # same reserve the Governor keeps free


HELD_ENV = "NUPEN_TEST_SLOT_HELD"                       # set in every budgeted test launch's environment


def slot_dir() -> Path:
    return Path(os.environ.get("NUPEN_TEST_SLOTS_DIR") or Path(tempfile.gettempdir()) / "nupen_test_slots")


def mb_file() -> Path:
    return Path(os.environ.get("NUPEN_TEST_MB_FILE") or ROOT / "state" / "creator" / "test_proc_mb.json")


def slot_cap() -> int:
    try:
        return max(1, int(os.environ.get("NUPEN_TEST_SLOTS_MAX", _device().settings()["test_slots"])))
    except ValueError:
        return int(_device().settings()["test_slots"])


def _mem() -> tuple[float, float]:
    """(available GB, total GB); (99, 99) without psutil, which makes the budget purely the cap."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return float(vm.available) / 1e9, float(vm.total) / 1e9
    except ImportError:
        return 99.0, 99.0


def per_test_mb() -> float:
    try:
        d = json.loads(mb_file().read_text(encoding="utf-8"))
        return max(MIN_MB, float(d["mb"]))
    except (OSError, ValueError, KeyError, TypeError):
        return float(_device().TEST_MB)       # until a real launch has been measured


def record_mb(peak_mb: float) -> None:
    """Fold one measured launch (peak resident MB of the whole child tree) into a running figure: mostly the larger of the
    average and a decaying maximum, so one small run cannot talk the budget into overcommitting."""
    if peak_mb <= 0:
        return
    p = mb_file()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        n, avg, mx = int(d.get("samples", 0)), float(d["avg_mb"]), float(d["max_mb"])
    except (OSError, ValueError, KeyError, TypeError):
        n, avg, mx = 0, peak_mb, peak_mb
    n += 1
    avg += (peak_mb - avg) / min(n, 50)
    mx = max(peak_mb, mx * 0.98)
    out = {"mb": round(max(avg, 0.7 * mx), 1), "avg_mb": round(avg, 1), "max_mb": round(mx, 1), "samples": n,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def reserve_gb(total_gb: float) -> float:
    """The RAM kept free: the Governor's own floor from device settings (one source of truth - 2 Oct: a separate 7% here kept
    2.4 GB free on the 33.8 GB machine when the owner asked for 1 GB). Falls back to the defaults if settings are unreadable."""
    try:
        s = _device().settings()
        return max(float(s["governor_floor_min_gb"]), float(s["governor_floor_fraction"]) * total_gb)
    except Exception:                                   # noqa: BLE001 - the budget never breaks a test launch
        return max(RESERVE_MIN_GB, RESERVE_FRACTION * total_gb)


def affordable(free_gb: float, total_gb: float, mb: float) -> bool:
    """May one MORE test process start? True while free RAM minus the reserve still holds one measured test process."""
    return free_gb - reserve_gb(total_gb) >= mb / 1024.0


def eval_reserve_gb(running: int, test_parallel: int, extra: int = 1) -> float:
    """Memory to set aside when admitting `extra` more workers beyond `running`: their evaluations' test slots, bounded by the
    machine-wide cap (workers share the slots, so ten workers do not need ten times the memory)."""
    if test_parallel <= 0:
        return 0.0
    cap = slot_cap()
    more = min(cap, (running + extra) * test_parallel) - min(cap, running * test_parallel)
    return more * per_test_mb() / 1024.0


class _SlotFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh: Optional[Any] = None

    def try_lock(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        fh, self.fh = self.fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


class Slot:
    """Context manager holding one test slot. `.index` is the slot taken (None if the wait timed out and the launch goes
    unbudgeted)."""

    def __init__(self, wait_s: float = 1800.0, poll_s: float = 0.25, cap: Optional[int] = None,
                 mem: Callable[[], tuple[float, float]] = _mem, mb: Callable[[], float] = per_test_mb,
                 directory: Optional[Path] = None) -> None:
        self.wait_s, self.poll_s, self.cap, self.mem, self.mb, self.dir = wait_s, poll_s, cap, mem, mb, directory
        self.index: Optional[int] = None
        self.nested = False                             # True: covered by an ancestor's slot (HELD_ENV), nothing taken
        self.waited = 0.0
        self._file: Optional[_SlotFile] = None

    def _attempt(self) -> bool:
        d = self.dir or slot_dir()
        cap = self.cap or slot_cap()
        for i in range(cap):
            f = _SlotFile(d / f"slot{i}.lock")
            if not f.try_lock():
                continue
            if i > 0:                                   # slot 0 is always allowed: progress never stops
                free, total = self.mem()
                if not affordable(free, total, self.mb()):
                    f.release()
                    return False                        # higher slots are no cheaper: wait for a running test to end
            self._file, self.index = f, i
            return True
        return False

    def __enter__(self) -> "Slot":
        t0 = time.monotonic()
        if self.dir is None and os.environ.get(HELD_ENV) == "1":
            # An ancestor test launch already holds a slot in the machine pool: run under it. Waiting here deadlocked the
            # new PC on 2 Oct 2026 - 8 outer test files held all 8 slots while their nested test runs queued behind them.
            self.nested = True
            return self
        while not self._attempt():
            if time.monotonic() - t0 > self.wait_s:
                break
            time.sleep(self.poll_s)
        self.waited = time.monotonic() - t0
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._file is not None:
            self._file.release()
            self._file = None


def _tree_rss_mb(pid: int) -> float:
    try:
        import psutil
        p = psutil.Process(pid)
        procs = [p, *p.children(recursive=True)]
        total = 0
        for q in procs:
            try:
                total += q.memory_info().rss
            except psutil.Error:
                pass
        return total / 1048576.0
    except Exception:                                   # noqa: BLE001 - measurement never breaks a test run
        return 0.0


def _kill_tree(pid: int) -> None:
    """Stop the child this call started and ITS descendants (the venv launcher's real interpreter is a child of it)."""
    try:
        import psutil
        p = psutil.Process(pid)
        for q in [*p.children(recursive=True), p]:
            try:
                q.kill()
            except psutil.Error:
                pass
    except Exception:                                   # noqa: BLE001
        pass


def run(argv: list[str], *, timeout: float, input: Optional[str] = None, slot: Optional[Slot] = None,
        **kw: Any) -> "subprocess.CompletedProcess[str]":
    """subprocess.run(capture_output=True) for a TEST launch: waits for a slot, measures the child tree's peak memory.
    Raises subprocess.TimeoutExpired like subprocess.run (the child tree is killed first); the slot wait is not part of the
    timeout."""
    with (slot or Slot()):
        t0 = time.monotonic()
        peak = [0.0]
        done = threading.Event()
        kw["env"] = {**(kw.get("env") or os.environ), HELD_ENV: "1"}      # nested test launches run under this slot
        with subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else None, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, **kw) as p:
            def sample() -> None:
                while not done.is_set():
                    peak[0] = max(peak[0], _tree_rss_mb(p.pid))
                    done.wait(0.15)
            th = threading.Thread(target=sample, daemon=True)
            th.start()
            try:
                try:
                    out, err = p.communicate(input=input, timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_tree(p.pid)
                    out, err = p.communicate()
                    raise subprocess.TimeoutExpired(argv, timeout, out, err) from None
            finally:
                done.set()
                th.join(1.0)
            if peak[0] > 0 and time.monotonic() - t0 > 0.2:
                record_mb(peak[0])
            return subprocess.CompletedProcess(argv, p.returncode, out, err)
