"""Creator K18 - the Creator's OWN worker: generates code with no other AI (owner, 1 Oct 2026: "the goal is to make it so it is
fully independent of any other ai where it just permanently develops itself forever one day") - IMPLEMENTED, NOT VALIDATED.

Two strategies, both local and offline:

    model   a local open-weights code model (llama.cpp server on 127.0.0.1, weights in the runtime dir, see creator/device.py) is
            shown the task and the code, rewrites whole files, sees the visible tests' failures, and retries. The weights are the
            Creator's own file: no service is called, and the Creator may retrain or replace them.
    search  home-grown test-guided program repair: single and paired AST mutations (operator swaps, comparison flips,
            constants +-1, abs()/negation wrappers, argument swaps, return-expression substitutions) applied to the code and kept
            only when every visible test passes.

LEARNING (from development experience only - never from the holdout split):
    * an experience log (state/creator/generator_memory.jsonl): task category, objective, which strategy worked, the files it
      wrote; successful solutions are retrieved as worked examples for similar objectives in later prompts;
    * the strategy order per task category is learned from success rates (Laplace-smoothed), so the Creator spends its effort
      where it has worked before.

Visible tests guide the worker; the sealed hidden tests judge it (creator.devbench). Passing only the visible tests is reported as
claimed_done and scored by the evaluator - overfitting to visible tests shows up as FALSE_COMPLETION."""
from __future__ import annotations

import ast
import copy
import dataclasses
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from creator import devbench as D
from creator import device as DEV

RUNTIME = DEV.runtime_dir()                                           # NUPEN_RUNTIME / CREATOR_RUNTIME / ~/creator_runtime
SERVER_EXE = DEV.server_exe(RUNTIME)                                  # llama-server.exe on Windows, llama-server elsewhere
DEFAULT_MODEL = DEV.model_path(RUNTIME)
MEMORY_FILE = Path(__file__).resolve().parents[1] / "state" / "creator" / "generator_memory.jsonl"
STRATEGIES = ("model", "search")


# ------------------------------------------------------------------------------------------------ the local model

MARKER_STALE_S = 30.0       # a loading server touches its .starting marker every ~2 s


def _free_ram_gb() -> Optional[float]:
    """Free RAM in GiB, as the Governor measures it (psutil 'available'); None when it cannot be read."""
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 2**30
    except Exception:                                          # noqa: BLE001 - unknown, not 'plenty'
        return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---- keeping the server from outliving its owner (a llama-server was found running an hour after its parent died) --------
_WIN = sys.platform == "win32"
_PROCESS_QUERY_LIMITED = 0x1000
_STILL_ACTIVE = 259


def _pid_image(pid: int) -> Optional[str]:
    """Full image path of a live process; None when the pid is not running. Off Windows only liveness is known ('')."""
    if not _WIN:
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return ""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED, False, pid)
    if not h:
        return None
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
            return None
        buf = ctypes.create_unicode_buffer(1024)
        n = wintypes.DWORD(1024)
        return buf.value if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
    finally:
        k32.CloseHandle(h)


def _kill_pid(pid: int) -> None:
    if _WIN:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x0001, False, pid)               # PROCESS_TERMINATE
        if h:
            k32.TerminateProcess(h, 1)
            k32.CloseHandle(h)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def reap_stale_server(pidfile: Path, exe: Path) -> Optional[int]:
    """If `pidfile` records a server whose recorded parent is dead, kill THAT pid (only while it still runs our exe) and return
    it. Never matches by image name alone: only the recorded pid, and only when its image path is the server we launched."""
    try:
        rec = json.loads(pidfile.read_text(encoding="utf-8"))
        pid, parent = int(rec["pid"]), int(rec["parent"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        pidfile.unlink(missing_ok=True)
        return None
    killed: Optional[int] = None
    if pid != os.getpid() and _pid_image(parent) is None:
        img = _pid_image(pid)
        if img is not None and (img == "" or os.path.normcase(img) == os.path.normcase(str(exe))):
            _kill_pid(pid)
            killed = pid
    if killed is not None or _pid_image(pid) is None:
        pidfile.unlink(missing_ok=True)
    return killed


class _KillOnCloseJob:
    """Windows Job Object with KILL_ON_JOB_CLOSE: every process assigned to it dies when the last handle closes, which the OS
    does when the owner process dies for any reason (including a hard kill). A no-op elsewhere."""

    def __init__(self) -> None:
        self.handle: Any = None
        if not _WIN:
            return
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class _Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ro", "wo", "oo", "rb", "wb", "ob")]

        class _Ext(ctypes.Structure):
            _fields_ = [("Basic", _Basic), ("Io", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _Ext()
        info.Basic.LimitFlags = 0x2000                            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # ExtendedLimitInformation
            k32.CloseHandle(job)
            return
        self.handle = job

    def adopt(self, proc: "subprocess.Popen[bytes]") -> bool:
        if not _WIN or self.handle is None:
            return False
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        return bool(k32.AssignProcessToJobObject(self.handle, int(getattr(proc, "_handle"))))

    def close(self) -> None:
        if _WIN and self.handle is not None:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle(self.handle)
        self.handle = None


PIDFILE = RUNTIME / "llama_server.pid"


class MachineLock:
    """One holder on this machine at a time, across processes: an OS byte-range lock on `path` (msvcrt on Windows, fcntl
    elsewhere). The OS drops it when the holder exits or is killed, so a crash can never leave a stale lock behind.
    2 Oct: three llama servers (~1 GB each) ran at once - two started by concurrent kernel test runs, one by a student - and the
    governor pulled finished work back for the RAM they took."""

    def __init__(self, path: Path, wait_s: float = 1800.0, poll_s: float = 0.5) -> None:
        self.path, self.wait_s, self.poll_s = path, wait_s, poll_s
        self.fh: Optional[Any] = None

    def _try(self) -> bool:
        assert self.fh is not None
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+b")
        t0 = time.monotonic()
        while not self._try():
            if time.monotonic() - t0 > self.wait_s:
                self.fh.close()
                self.fh = None
                raise TimeoutError(f"another holder kept {self.path.name} for {self.wait_s:.0f}s")
            time.sleep(self.poll_s)

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


class LocalModel:
    """A llama.cpp server bound to 127.0.0.1 for the life of the context. Nothing leaves this machine.

    The server cannot outlive its owner: it is terminated on every exit path (including a failed or interrupted start-up), it
    runs inside a kill-on-close Job Object (so a hard kill of this process takes it down too), and its pid is recorded in
    `pidfile` so that a later start reaps a server whose recorded owner is dead. Only ONE server runs on the machine at a time
    (MachineLock next to the pidfile): a second LocalModel waits for the first to finish instead of loading another copy."""

    def __init__(self, model: Path = DEFAULT_MODEL, exe: Path = SERVER_EXE, ctx: int = 8192, threads: Optional[int] = None,
                 startup_s: float = 120.0, pidfile: Path = PIDFILE, gpu_layers: Optional[int] = None,
                 servers: Optional[int] = None, slot_wait_s: float = 1800.0) -> None:
        cfg = DEV.settings()                                   # threads and GPU layers follow the machine unless given
        self.servers = max(1, int(cfg.get("llama_servers", 1)) if servers is None else servers)
        if threads is None:                                    # several servers share the cores (2x oversubscribed, >= 2 each)
            threads = int(cfg["llama_threads"]) if self.servers == 1 else \
                max(2, min(DEV.SERVER_MAX_THREADS, 2 * int(cfg["llama_threads"]) // self.servers))
        self.base_pidfile, self.slot_wait_s = pidfile, slot_wait_s
        self.gpu_layers = int(cfg["gpu_layers"]) if gpu_layers is None else gpu_layers
        self.model, self.exe, self.ctx, self.threads, self.startup_s = model, exe, ctx, threads, startup_s
        self.pidfile = pidfile
        self.free_gb: Callable[[], Optional[float]] = _free_ram_gb       # injectable: tests fake the RAM reading
        self.starting_marker: Optional[Path] = None
        self.lock = MachineLock(pidfile.with_name("llama_server.lock"))
        self.port = 0
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.job: Optional[_KillOnCloseJob] = None
        self.calls = 0
        self.seconds = 0.0
        self.leased = False                                    # attached to a warm server of creator.modelpool (not ours to stop)
        self.pulse = ""                                        # attached to a rented GPU pod's server (creator.gpupulse): its pulse id

    def _command(self) -> list[str]:
        return [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.ctx), "-t", str(self.threads), "--log-disable"] + (["-ngl", str(self.gpu_layers)] if self.gpu_layers > 0 else [])

    def _pulse_attach(self) -> bool:
        """Only while a GPU pulse is switched on (creator.device.pulse_on): use the pod's server of this model through the SSH tunnel."""
        cfg = DEV.settings()
        if not DEV.pulse_on(cfg):
            return False
        from creator import gpupulse as GP
        got = GP.attach(GP.pulse_file(cfg), self.model)
        if got is None:
            return False
        self.port, self.pulse = got
        return True

    def __enter__(self) -> "LocalModel":
        if self._pulse_attach():
            return self
        if not self.exe.is_file() or not self.model.is_file():
            raise FileNotFoundError(f"local model runtime missing: {self.exe} / {self.model}")
        self._acquire_slot()                                   # a warm pooled server, a free server slot on this machine, or wait for one
        if self.leased:
            return self
        try:
            try:
                reap_stale_server(self.pidfile, self.exe)
            except OSError:
                pass
            self.port = free_port()
            self._mark_starting()
            try:                                               # the marker never outlives a failed or interrupted start
                self.job = _KillOnCloseJob()
                self.proc = subprocess.Popen(self._command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.job.adopt(self.proc)
                try:
                    self.pidfile.parent.mkdir(parents=True, exist_ok=True)
                    self.pidfile.write_text(json.dumps({"pid": self.proc.pid, "parent": os.getpid()}), encoding="utf-8")
                except OSError:
                    pass
                self._wait_healthy()
            finally:
                self._unmark_starting()
        except BaseException:                                  # incl. KeyboardInterrupt: nothing is left running
            self._stop()
            raise
        return self

    def _mark_starting(self) -> None:
        self.starting_marker = self.pidfile.with_suffix(".starting")
        try:
            self.starting_marker.write_text(str(time.time()), encoding="utf-8")
        except OSError:
            self.starting_marker = None

    def _unmark_starting(self) -> None:
        m, self.starting_marker = self.starting_marker, None
        if m is not None:
            m.unlink(missing_ok=True)

    def _touch_marker(self) -> None:
        if self.starting_marker is not None:
            try:
                os.utime(self.starting_marker)
            except OSError:
                pass

    def _starting_count(self, others_only: bool = False) -> int:
        """Servers loading right now on this machine (fresh markers of every slot). A loading server refreshes its marker every
        few seconds, so a marker older than MARKER_STALE_S was left by a crash. `others_only` leaves this server's own out."""
        n = 0
        for m in self.base_pidfile.parent.glob(self.base_pidfile.stem + "*.starting"):
            if others_only and m == self.starting_marker:
                continue
            try:
                if time.time() - m.stat().st_mtime < MARKER_STALE_S:
                    n += 1
            except OSError:
                pass
        return n if others_only else max(1, n)

    def _wait_healthy(self) -> None:
        assert self.proc is not None
        t0 = last_touch = time.monotonic()
        while time.monotonic() - t0 < self.startup_s * self._starting_count():
            if time.monotonic() - last_touch > 2.0:
                self._touch_marker()
                last_touch = time.monotonic()   # N servers loading together share disk and cores
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("local model server exited during start-up")
            time.sleep(0.2)
        raise TimeoutError("local model server did not become healthy")

    def _acquire_slot(self) -> None:
        """Take the first free model-server slot. Owner, 2 Oct 2026 (new PC): 'there is more RAM work that needs to be done' -
        one server per machine (MachineLock, 1 Oct on 16 GB) made every thinking student queue behind one server while RAM sat
        idle. `llama_servers` (creator/device.py) slots now run side by side; slot 0 keeps the original lock and pid files, each
        slot has its own, and the OS still drops a dead holder's lock."""
        slots = []
        for i in range(self.servers):
            sfx = "" if i == 0 else f".{i}"
            slots.append((MachineLock(self.base_pidfile.with_name(f"llama_server{sfx}.lock")),
                          self.base_pidfile.with_name(self.base_pidfile.stem + sfx + self.base_pidfile.suffix)))
        t0 = time.monotonic()
        while True:
            if self._lease_warm():
                return
            for i, (lock, pid) in enumerate(slots):
                if i > 0 and not self._ram_allows_extra_server():     # an extra server is never worth a RAM pull-back
                    break
                lock.path.parent.mkdir(parents=True, exist_ok=True)
                lock.fh = open(lock.path, "a+b")
                if lock._try():
                    self.lock, self.pidfile = lock, pid
                    return
                lock.fh.close()
                lock.fh = None
            if time.monotonic() - t0 > self.slot_wait_s:
                raise TimeoutError(f"all {self.servers} local model server slots stayed busy for {self.slot_wait_s:.0f}s")
            time.sleep(0.5)

    def _lease_warm(self) -> bool:
        """Attach to an idle server the swarm's pool keeps loaded (creator.modelpool; same model and context): no start-up paid."""
        if not any(self.base_pidfile.parent.glob("llama_pool*.json")):       # no pool on this machine: nothing to import
            return False
        try:
            from creator import registry as REG
            got = REG.get("modelpool").lease(self.base_pidfile, self.model, self.ctx)
        except Exception:                                      # noqa: BLE001 - a broken pool means a normal start
            return False
        if got is None:
            return False
        self.port, self.lock, _ = got
        self.leased = True
        return True

    def _ram_allows_extra_server(self) -> bool:
        """Free RAM minus one more server (DEV.SERVER_GB, measured) must stay above the Governor's floor (device settings)."""
        from creator import testslots
        try:
            cfg = DEV.settings()
            if DEV.pulse_on(cfg):                              # served by a GPU pulse: no RAM of this PC is taken
                from creator import gpupulse as GP
                if GP.serves(GP.pulse_file(cfg), self.model):
                    return True
            free = self.free_gb()
            if free is None:                                   # unknown free RAM never opens the gate
                return False
            claimed = self._starting_count(others_only=True) * DEV.SERVER_GB     # loading servers have not taken their RAM yet
            return free - claimed - DEV.server_gb_for(self.model) > testslots.reserve_gb(DEV.get().ram_gb)
        except Exception:                                      # noqa: BLE001 - an unreadable budget means no extra server
            return False

    def _stop(self) -> None:
        if self.pulse:                                         # the pod's server is not ours to stop
            self.pulse = ""
            return
        if self.leased:                                        # the pool owns the server: only give the lease back
            self.leased = False
            self.lock.release()
            return
        proc, self.proc = self.proc, None
        self._unmark_starting()
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        finally:
            self.lock.release()
            if self.job is not None:
                self.job.close()
                self.job = None
            if proc is not None:
                try:
                    if json.loads(self.pidfile.read_text(encoding="utf-8")).get("pid") == proc.pid:
                        self.pidfile.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass

    def __exit__(self, *exc: Any) -> None:
        self._stop()

    def chat_text(self, messages: Sequence[Mapping[str, str]], **kw: Any) -> str:
        """`chat` with a reasoning model's <think>...</think> block removed (the caller wants the answer, not the scratch work)."""
        return THINK_BLOCK.sub("", self.chat(prepare_messages(messages, self.model), **kw)).strip()

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 1500, temperature: float = 0.2,
             seed: int = 0, timeout: float = 600.0) -> str:
        if self.pulse:                                         # nothing private ever leaves this PC
            from creator import gpupulse as GP
            GP.outbound_ok(messages)
        body = json.dumps({"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature,
                           "seed": seed}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        self.calls += 1
        self.seconds += time.monotonic() - t0
        return str(data["choices"][0]["message"]["content"])


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
NO_THINK_MODELS = ("qwen3",)      # measured 3 Oct 2026: Qwen3-1.7B with thinking took 60-100 s per question for no accuracy gain over '/no_think'


def prepare_messages(messages: Sequence[Mapping[str, str]], model: Any) -> list[dict[str, str]]:
    """Messages for a model: reasoning-by-default families get the '/no_think' soft switch on the last user turn (short, fast answers)."""
    out = [dict(m) for m in messages]
    if out and any(k in Path(str(model)).name.lower() for k in NO_THINK_MODELS) and out[-1].get("role") == "user":
        out[-1]["content"] += "\n/no_think"
    return out


def thinker(free_gb: Optional[Callable[[], Optional[float]]] = None, cfg: Optional[Mapping[str, Any]] = None, **kw: Any) -> LocalModel:
    """The 'thinker' role (owner, 2 Oct 2026: Nupen must think and reason much better): the THINKING model of device setting 'think_model',
    used by creator.judgment and the other live reasoning; the code-editing students keep LocalModel() with the fast model.

    RAM-gated like the other servers: a thinking server starts only while free RAM - its resident size stays above the Governor's floor
    (and 'think_servers' > 0); otherwise (or when no thinking model is configured / on disk) the fast model answers, so thinking never
    stalls and never pushes the machine into the floor. How many thinkers run at once is the caller's cap ('think_servers')."""
    c = DEV.settings() if cfg is None else cfg
    model = DEV.think_model_path(c)
    if model is None or int(c.get("think_servers", 0)) < 1:
        return LocalModel(**kw)
    lm = LocalModel(model=model, **kw)                    # all machine slots (slot 0 = the original lock): a free slot is a free slot
    if free_gb is not None:
        lm.free_gb = free_gb
    if not lm._ram_allows_extra_server():
        return LocalModel(**kw)
    return lm


# ------------------------------------------------------------------------------------------------ reading and writing code

def read_code(workdir: Path, max_chars: int = 12000) -> dict[str, str]:
    files: dict[str, str] = {}
    total = 0
    for p in sorted(workdir.rglob("*.py")):
        rel = p.relative_to(workdir).as_posix()
        if "__pycache__" in rel or rel.endswith("__init__.py") or rel == "conftest.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if total + len(text) > max_chars:
            break
        files[rel] = text
        total += len(text)
    return files


FILE_RE = re.compile(r"FILE:\s*(?P<path>[\w./-]+\.py)\s*\n```(?:python)?\n(?P<body>.*?)```", re.S)


def parse_files(reply: str) -> dict[str, str]:
    """Whole-file outputs in the format 'FILE: path' + a fenced block. Paths must stay inside the task (no .., no absolute)."""
    out = {}
    for m in FILE_RE.finditer(reply):
        path = m.group("path").strip()
        if path.startswith(("/", "\\")) or ".." in Path(path).parts:
            continue
        out[path] = m.group("body")
    return out


EDIT_RE = re.compile(r"FILE:\s*(?P<path>[\w./-]+\.py)\s*\n<<<<<<< SEARCH\n(?P<old>.*?)\n=======\n(?P<new>.*?)\n>>>>>>> REPLACE",
                     re.S)


def parse_edits(reply: str) -> list[tuple[str, str, str]]:
    """Search/replace edits ('FILE: path' + <<<<<<< SEARCH / ======= / >>>>>>> REPLACE) for files too large to rewrite whole."""
    out = []
    for m in EDIT_RE.finditer(reply):
        path = m.group("path").strip()
        if path.startswith(("/", "\\")) or ".." in Path(path).parts:
            continue
        out.append((path, m.group("old"), m.group("new")))
    return out


def apply_edit(text: str, old: str, new: str) -> Optional[str]:
    """Replace `old` in `text` exactly once; if not found verbatim, match line by line ignoring leading/trailing whitespace and
    re-indent the replacement to the matched block. None when there is no unique match (the edit is refused, never guessed)."""
    if old and text.count(old) == 1:
        return text.replace(old, new)
    lines, olds = text.split("\n"), [ln.strip() for ln in old.strip("\n").split("\n")]
    if not olds or not any(olds):
        return None
    hits = [i for i in range(len(lines) - len(olds) + 1) if [ln.strip() for ln in lines[i:i + len(olds)]] == olds]
    if len(hits) != 1:
        return None
    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    new_lines = new.strip("\n").split("\n")
    base = min((len(x) - len(x.lstrip()) for x in new_lines if x.strip()), default=0)
    new_lines = [indent + x[base:] if x.strip() else "" for x in new_lines]
    return "\n".join(lines[:i] + new_lines + lines[i + len(olds):])


def apply_edits(workdir: Path, edits: Sequence[tuple[str, str, str]]) -> tuple[list[str], list[str]]:
    """Apply edits; returns (applied paths, refused descriptions). A refused edit leaves its file untouched."""
    applied, refused = [], []
    for path, old, new in edits:
        p = workdir / path
        if not old.strip():                     # empty SEARCH = the whole file (create it, or replace a version written earlier)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new + "\n", encoding="utf-8")
            applied.append(path)
            continue
        if not p.is_file():
            refused.append(f"{path}: file does not exist")
            continue
        out = apply_edit(p.read_text(encoding="utf-8"), old, new)
        if out is None:
            refused.append(f"{path}: SEARCH text not found exactly once")
            continue
        p.write_text(out, encoding="utf-8")
        applied.append(path)
    return applied, refused


def apply_files(workdir: Path, files: Mapping[str, str]) -> None:
    for rel, body in files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def visible_tests(workdir: Path) -> D.TestCounts:
    return D.run_pytest(workdir, "tests", timeout=120)


def failure_text(workdir: Path, limit: int = 2500) -> str:
    D.purge_bytecode(workdir)
    from creator import testslots
    p = testslots.run([sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                      cwd=workdir, text=True, timeout=120)
    return (p.stdout + p.stderr)[-limit:]


# ------------------------------------------------------------------------------------------------ experience (learning)

@dataclasses.dataclass(frozen=True)
class Experience:
    category: str
    objective: str
    strategy: str
    solved_visible: bool
    files: Mapping[str, str]
    seconds: float


_MEMORY_LOCK = threading.Lock()


class ExperienceMemory:
    """Append-only experience log. Learns only from what it is given - the caller never records holdout tasks."""

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.path = path

    def load(self) -> list[Experience]:
        if not self.path.is_file():
            return []
        out = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Experience(**json.loads(ln)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def add(self, e: Experience) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(dataclasses.asdict(e)) + "\n").encode("utf-8")
        with _MEMORY_LOCK, self.path.open("ab") as fh:            # one whole record per write: threads must not interleave
            fh.write(data)
            fh.flush()

    def strategy_order(self, category: str) -> list[str]:
        """Strategies ranked by Laplace-smoothed success rate in this category (ties keep the default order)."""
        exp = [e for e in self.load() if e.category == category]

        def rate(s: str) -> float:
            xs = [e for e in exp if e.strategy == s]
            return (sum(e.solved_visible for e in xs) + 1) / (len(xs) + 2)
        return sorted(STRATEGIES, key=lambda s: (-rate(s), STRATEGIES.index(s)))

    def examples(self, objective: str, k: int = 2) -> list[Experience]:
        """The k most similar SUCCESSFUL past solutions (word-overlap similarity) - worked examples for the model."""
        words = set(re.findall(r"[a-z]+", objective.lower()))
        good = [e for e in self.load() if e.solved_visible and e.strategy == "model" and e.files]
        scored = sorted(good, key=lambda e: -len(words & set(re.findall(r"[a-z]+", e.objective.lower()))))
        return [e for e in scored[:k] if words & set(re.findall(r"[a-z]+", e.objective.lower()))]


# ------------------------------------------------------------------------------------------------ strategy: model

SYSTEM_STEPWISE = ("You are a careful Python developer. First find the exact line that makes the tests or the objective fail, "
                   "then change as little as possible. Keep every existing behaviour the objective does not mention. "
                   "Reply ONLY with the complete new content of every file you change, each as:\nFILE: <path>\n```python\n<code>\n```\n"
                   "Never edit existing tests unless the objective asks for new tests.")
SYSTEM = ("You are a careful Python developer. You change code so that the objective is met and all tests pass. "
          "Reply ONLY with the complete new content of every file you change, each as:\nFILE: <path>\n```python\n<code>\n```\n"
          "Never edit existing tests unless the objective asks for new tests.")


def model_prompt(task: Mapping[str, Any], code: Mapping[str, str], examples: Sequence[Experience]) -> str:
    parts = []
    for ex in examples:
        shown = "\n".join(f"FILE: {p}\n```python\n{b}```" for p, b in list(ex.files.items())[:2])
        parts.append(f"Example of a solved task. Objective: {ex.objective}\nSolution:\n{shown}")
    parts.append(f"Objective ({task.get('category', 'task')}): {task['objective']}")
    parts.append("Current files:\n" + "\n".join(f"FILE: {p}\n```python\n{b}```" for p, b in code.items()))
    return "\n\n".join(parts)


def solve_with_model(task: Mapping[str, Any], workdir: Path, llm: Any, memory: Optional[ExperienceMemory],
                     attempts: int = 3, temperature: float = 0.2, examples_k: int = 2,
                     system: str = SYSTEM) -> tuple[bool, dict[str, str], int]:
    """Generate -> run visible tests -> feed failures back. Returns (visible tests pass, files written, model calls)."""
    examples = memory.examples(str(task["objective"]), k=examples_k) if memory and examples_k > 0 else []
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": model_prompt(task, read_code(workdir), examples)}]
    written: dict[str, str] = {}
    calls = 0
    for _ in range(attempts):
        reply = llm.chat(messages, seed=calls, temperature=temperature)
        calls += 1
        files = parse_files(reply)
        if not files:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Reply with complete files in the FILE: format only."}]
            continue
        apply_files(workdir, files)
        written.update(files)
        if visible_tests(workdir).ok:
            return True, written, calls
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "The tests fail:\n" + failure_text(workdir) + "\nFix it. Complete files only."}]
    return False, written, calls


# ------------------------------------------------------------------------------------------------ strategy: search

SWAP_BIN: dict[type, tuple[type, ...]] = {ast.Add: (ast.Sub, ast.Mult), ast.Sub: (ast.Add,), ast.Mult: (ast.Add, ast.FloorDiv),
                                         ast.FloorDiv: (ast.Div, ast.Mult), ast.Div: (ast.FloorDiv,)}
SWAP_CMP: dict[type, tuple[type, ...]] = {ast.Lt: (ast.LtE, ast.Gt), ast.LtE: (ast.Lt,), ast.Gt: (ast.GtE, ast.Lt),
                                         ast.GtE: (ast.Gt,), ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,)}


Edit = Callable[[Any], None]


def _set_op(alt: type) -> Edit:
    def f(m: Any) -> None:
        m.op = alt()
    return f


def _set_cmp(alt: type) -> Edit:
    def f(m: Any) -> None:
        m.ops[0] = alt()
    return f


def _shift(d: int) -> Edit:
    def f(m: Any) -> None:
        m.value = m.value + d
    return f


def _wrap_abs(m: Any) -> None:
    m.value = ast.Call(ast.Name("abs", ast.Load()), [m.value], [])


def _swap_args(m: Any) -> None:
    m.args = [m.args[1], m.args[0]]


def _none_default(j: int) -> Edit:
    def f(m: Any) -> None:
        m.args.defaults[j] = ast.Constant(None)
    return f


BUILTIN_SWAPS: dict[str, tuple[str, ...]] = {"max": ("min",), "min": ("max",), "sorted": ("list",), "list": ("sorted",),
                                             "sum": ("len",), "len": ("sum",)}
METHOD_SWAPS: dict[str, tuple[str, ...]] = {"append": ("extend",), "extend": ("append",)}
GUARD_DEFAULTS: tuple[str, ...] = ("[]", "0", "None", "''", "{}", "False")


def _set_name(new: str) -> Edit:
    def f(m: Any) -> None:
        m.id = new
    return f


def _set_attr(new: str) -> Edit:
    def f(m: Any) -> None:
        m.attr = new
    return f


def _swap_branches(m: Any) -> None:
    m.body, m.orelse = m.orelse, m.body


def _range_end_plus(m: Any) -> None:
    j = 1 if len(m.args) >= 2 else 0
    m.args[j] = ast.BinOp(m.args[j], ast.Add(), ast.Constant(1))


def _insert_guard(arg: str, default: str, ret_arg: bool) -> Edit:
    def f(m: Any) -> None:
        val = ast.Name(arg, ast.Load()) if ret_arg else ast.parse(default, mode="eval").body
        guard = ast.If(ast.UnaryOp(ast.Not(), ast.Name(arg, ast.Load())), [ast.Return(val)], [])
        at = 1 if m.body and isinstance(m.body[0], ast.Expr) and isinstance(getattr(m.body[0], "value", None), ast.Constant) else 0
        m.body.insert(at, guard)
    return f


def _has_guard(fn: ast.FunctionDef, arg: str) -> bool:
    for st in fn.body:
        if (isinstance(st, ast.If) and isinstance(st.test, ast.UnaryOp) and isinstance(st.test.op, ast.Not)
                and isinstance(st.test.operand, ast.Name) and st.test.operand.id == arg and st.body
                and isinstance(st.body[0], ast.Return)):
            return True
    return False


def _scope_names(fn: ast.FunctionDef) -> list[str]:
    names = [a.arg for a in fn.args.args]
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in names:
            names.append(n.id)
    return names


def targeted_mutations(tree: ast.Module, per_family: int = 40) -> Iterator[ast.Module]:
    """Cheap single edits aimed at common injected-bug classes, each family capped so the candidate budget stays bounded."""
    nodes = list(ast.walk(tree))
    fam: list[list[ast.Module]] = [[] for _ in range(5)]
    for i, n in enumerate(nodes):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BUILTIN_SWAPS:
            fam[0].extend(_edit(tree, nodes.index(n.func), _set_name(a)) for a in BUILTIN_SWAPS[n.func.id])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in METHOD_SWAPS:
            fam[0].extend(_edit(tree, nodes.index(n.func), _set_attr(a)) for a in METHOD_SWAPS[n.func.attr])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" and n.args:
            fam[1].append(_edit(tree, i, _range_end_plus))
        if isinstance(n, ast.If) and n.orelse:
            fam[2].append(_edit(tree, i, _swap_branches))
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            for a in n.args.args:
                if a.arg in ("self", "cls") or _has_guard(n, a.arg):
                    continue
                fam[3].append(_edit(tree, i, _insert_guard(a.arg, "", True)))
                fam[3].extend(_edit(tree, i, _insert_guard(a.arg, d, False)) for d in GUARD_DEFAULTS)
    for fn in (x for x in nodes if isinstance(x, ast.FunctionDef)):
        scope = _scope_names(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in scope:
                fam[4].extend(_edit(tree, nodes.index(n), _set_name(o)) for o in scope if o != n.id)
    for cands in fam:
        yield from cands[:per_family]


def mutations(tree: ast.Module) -> Iterator[ast.Module]:
    """Single-edit variants of a module, most plausible bug fixes first."""
    yield from targeted_mutations(tree)
    yield from generic_mutations(tree)


def generic_mutations(tree: ast.Module) -> Iterator[ast.Module]:
    for i, n in enumerate(list(ast.walk(tree))):
        if isinstance(n, ast.BinOp):
            for alt in SWAP_BIN.get(type(n.op), ()):
                yield _edit(tree, i, _set_op(alt))
        elif isinstance(n, ast.Compare) and n.ops:
            for alt in SWAP_CMP.get(type(n.ops[0]), ()):
                yield _edit(tree, i, _set_cmp(alt))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            for d in (1, -1):
                yield _edit(tree, i, _shift(d))
        elif isinstance(n, ast.Return) and n.value is not None:
            yield _edit(tree, i, _wrap_abs)
        elif isinstance(n, ast.Call) and len(n.args) == 2:
            yield _edit(tree, i, _swap_args)
        elif isinstance(n, ast.FunctionDef):
            for j, dflt in enumerate(n.args.defaults):
                if isinstance(dflt, (ast.List, ast.Dict)):
                    yield _edit(tree, i, _none_default(j))


def _edit(tree: ast.Module, index: int, fn: Callable[[Any], None]) -> ast.Module:
    t = copy.deepcopy(tree)
    target = list(ast.walk(t))[index]
    fn(target)
    return ast.fix_missing_locations(t)


_DIVERGENCE_RUNNER = r"""
import contextlib, io, json, sys
sys.path.insert(0, ROOT)
def load(src):
    ns = {"__name__": "_ovf_probe"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<probe>", "exec"), ns)
    return ns
def run(fn, args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return repr(fn(*[eval(a) for a in args]))
    except BaseException as e:
        return "exc:" + type(e).__name__
a, b = load(ORIG), load(CAND)
bad = 0
for name, calls in PLAN:
    fa, fb = a.get(name), b.get(name)
    if fa is None or fb is None:
        bad += len(calls)
        continue
    for c in calls:
        bad += run(fa, c) != run(fb, c)
print(json.dumps(bad))
"""


def _variants(args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """The harvested call plus small perturbations of its sequence arguments (shorter, reversed, one element longer)."""
    out = [args]
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)) and a:
            for v in (a[:-1], a[::-1], type(a)(list(a) + list(a[:1]))):
                out.append(args[:i] + (v,) + args[i + 1:])
    return out


def _test_call_args(workdir: Path, names: set[str]) -> dict[str, list[tuple[Any, ...]]]:
    """Literal argument tuples that the VISIBLE tests pass to the named functions: realistic inputs the program must keep handling the same way."""
    found: dict[str, list[tuple[Any, ...]]] = {}
    for tp in sorted((workdir / "tests").rglob("*.py"))[:20]:
        try:
            t = ast.parse(tp.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for n in ast.walk(t):
            if not isinstance(n, ast.Call) or n.keywords:
                continue
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name not in names:
                continue
            try:
                args = tuple(ast.literal_eval(a) for a in n.args)
            except (ValueError, SyntaxError, TypeError):
                continue
            for v in _variants(args):
                if v not in found.setdefault(name, []) and len(found[name]) < 12:
                    found[name].append(v)
    return found


def behavioural_divergence(workdir: Path, original: str, candidate: str, timeout: int = 20) -> int:
    """Number of auto-derived calls (testgen.candidate_calls over every top-level function) on which the candidate's outcome differs
    from the original's. Fewer = a smaller behavioural change. A probe that cannot run counts as maximally divergent."""
    from creator import testgen as T
    big = 10 ** 6
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return big
    seen = _test_call_args(workdir, {f.name for f in tree.body if isinstance(f, ast.FunctionDef)})
    plan = [(fn.name, [[repr(a) for a in c] for c in T.candidate_calls(fn)] + [[repr(a) for a in c] for c in seen.get(fn.name, [])])
            for fn in tree.body if isinstance(fn, ast.FunctionDef)]
    plan = [(n, c) for n, c in plan if c]
    if not plan:
        return 0
    script = (f"ROOT = {str(workdir)!r}\nORIG = {original!r}\nCAND = {candidate!r}\nPLAN = {plan!r}\n" + _DIVERGENCE_RUNNER)
    try:
        p = subprocess.run([sys.executable, "-B", "-c", script], cwd=workdir, capture_output=True, text=True, timeout=timeout)
        return int(json.loads(p.stdout.strip().splitlines()[-1]))
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return big


def _edit_size(original: str, candidate: str) -> int:
    return sum(1 for a, b in zip(ast.unparse(ast.parse(original)).splitlines(), candidate.splitlines()) if a != b) + abs(len(original.splitlines()) - len(candidate.splitlines()))


def solve_with_search(task: Mapping[str, Any], workdir: Path, budget: int = 120, pairs: bool = True,
                      rank: bool = True, extra_passes: int = 5, extra_budget: int = 40,
                      per_family: int = 40, pair_width: int = 12) -> tuple[bool, dict[str, str], int]:
    """Test-guided mutation repair over the non-test code. Returns (visible tests pass, files written, candidates tried).
    With rank=True the first visible-passing candidate is not trusted: further passing candidates are collected (up to extra_passes /
    extra_budget more tries, never beyond budget) and the one that changes the program's behaviour least (auto-derived calls vs the
    original) wins.
    rank=False is the old first-found behaviour."""
    targets = [p for p in sorted(workdir.rglob("*.py")) if "tests" not in p.relative_to(workdir).parts
               and p.name not in ("__init__.py", "conftest.py") and "__pycache__" not in p.parts]
    tried = 0
    for path in targets:
        original = path.read_text(encoding="utf-8")
        try:
            try:
                tree = ast.parse(original)
            except SyntaxError:
                continue
            legacy = list(generic_mutations(tree))
            singles = list(targeted_mutations(tree, per_family)) + legacy
            cands: Iterator[ast.Module] = iter(singles)
            if pairs and pair_width > 0:
                cands = itertools.chain(singles, (m2 for m1 in legacy[:pair_width] for m2 in itertools.islice(mutations(m1), pair_width)))
            passing: list[str] = []
            first_at = 0
            for cand in cands:
                if tried >= budget and not passing:
                    path.write_text(original, encoding="utf-8")
                    return False, {}, tried
                if passing and (tried - first_at >= extra_budget or len(passing) >= extra_passes or tried >= budget):   # budget is a hard cap
                    break
                tried += 1
                src = ast.unparse(cand) + "\n"
                if src in passing:
                    continue
                path.write_text(src, encoding="utf-8")
                if visible_tests(workdir).ok:
                    passing.append(src)
                    if not rank:
                        break
                    first_at = first_at or tried
            if passing:
                best = passing[0]
                if len(passing) > 1:
                    best = min(passing, key=lambda s: (behavioural_divergence(workdir, original, s), _edit_size(original, s)))
                path.write_text(best, encoding="utf-8")
                return True, {path.relative_to(workdir).as_posix(): best}, tried
            path.write_text(original, encoding="utf-8")
        except BaseException:                                # e.g. visible_tests raised: never leave a mutant on disk
            path.write_text(original, encoding="utf-8")
            raise
    return False, {}, tried


class SearchSolver:
    """A devbench Solver made ONLY of the system's own search (no model at all): its measured solve rate is what the system can
    already do by itself."""
    name = "self-search"

    def __init__(self, budget: int = 120, rank: bool = True) -> None:
        self.budget = budget
        self.rank = rank

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> D.SolverResult:
        ok, _, n = solve_with_search(task, workdir, self.budget, rank=self.rank)
        return D.SolverResult(ok, 0, f"search {'passed' if ok else 'failed'} the visible tests after {n} candidates")


# ------------------------------------------------------------------------------------------------ the solver

@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """Everything about how the worker works that the Creator may change by itself (creator.autotune), one measured step at
    a time. The defaults are the first version a human wrote; every later version is the Creator's own choice."""
    model_attempts: int = 3
    temperature: float = 0.2
    examples_k: int = 2
    search_budget: int = 120
    order: str = "learned"                      # learned | model_first | search_first
    prompt: str = "plain"                       # plain | stepwise

    def digest(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:12]


PROMPTS = {"plain": SYSTEM, "stepwise": SYSTEM_STEPWISE}


class GeneratorSolver:
    """A devbench Solver made only of the Creator's own parts. Learns from dev tasks; holdout tasks are never recorded."""

    def __init__(self, llm: Any, memory: Optional[ExperienceMemory] = None, learn: bool = True, model_attempts: int = 3,
                 search_budget: int = 120, name: str = "creator-generator-v1",
                 config: Optional[WorkerConfig] = None) -> None:
        self.cfg = config or WorkerConfig(model_attempts=model_attempts, search_budget=search_budget)
        self.llm, self.memory, self.learn = llm, memory, learn
        self.model_attempts, self.search_budget, self.name = self.cfg.model_attempts, self.cfg.search_budget, name
        self.log: list[dict[str, Any]] = []

    def config(self) -> dict[str, Any]:
        return {"solver": self.name, "model": str(getattr(self.llm, "model", "")), "learn": self.learn,
                **dataclasses.asdict(self.cfg)}

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> D.SolverResult:
        cat = str(task.get("category", ""))
        if self.cfg.order == "model_first":
            order = ["model", "search"]
        elif self.cfg.order == "search_first":
            order = ["search", "model"]
        else:
            order = self.memory.strategy_order(cat) if self.memory else list(STRATEGIES)
        calls = 0
        for strategy in order:
            t0 = time.monotonic()
            if strategy == "model":
                ok, files, n = solve_with_model(task, workdir, self.llm, self.memory, self.cfg.model_attempts,
                                                self.cfg.temperature, self.cfg.examples_k, PROMPTS[self.cfg.prompt])
                calls += n
            else:
                ok, files, n = solve_with_search(task, workdir, self.cfg.search_budget)
            secs = round(time.monotonic() - t0, 1)
            self.log.append({"task": task["id"], "strategy": strategy, "visible_ok": ok, "n": n, "seconds": secs})
            if self.memory is not None and self.learn:
                self.memory.add(Experience(cat, str(task["objective"]), strategy, ok, files if ok else {}, secs))
            if ok:
                return D.SolverResult(True, calls, f"{strategy} passed the visible tests after {n} tries")
        return D.SolverResult(False, calls, f"no strategy passed the visible tests ({order})")
