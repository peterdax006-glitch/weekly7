"""Nupen tools - the shell: a per-package session whose working directory persists between calls (Claude Code's Bash tool).

Each call runs `bash -c` (Git Bash at C:\\Program Files\\Git\\bin\\bash.exe when present, else `cmd /c`) in the session's cwd after the
command passed policy.check_command; the cwd the command ended in is recorded for the next call (and re-validated). Output goes to
files and is read back bounded. Every process started is put in a Windows job object (KILL_ON_JOB_CLOSE) and tracked by PID, so on a
timeout - and when the package ends - the whole tree is ended by PID (`taskkill /PID n /T /F` as the fallback), orphans included."""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from creator.tools import policy as PO

GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)
_SCRUB = ("NUPEN_REAL_MODEL_TEST", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "VIRTUAL_ENV", "BASH_ENV", "ENV")


def _taskkill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


class _JobObject:
    """A Windows job object: every descendant of the started process joins it; closing or terminating it ends them all."""

    def __init__(self) -> None:
        self.h: Any = None
        if sys.platform != "win32":
            return
        from ctypes import wintypes as W

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", W.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

        class IoC(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("a", "b", "c", "d", "e", "f")]

        class Ext(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IoC), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k = ctypes.windll.kernel32                                     # type: ignore[attr-defined]
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        k.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
        k.OpenProcess.restype = ctypes.c_void_p
        k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.k = k
        h = k.CreateJobObjectW(None, None)
        if not h:
            return
        info = Ext()
        info.Basic.LimitFlags = 0x2000                                  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k.CloseHandle(h)
            return
        self.h = h

    def assign(self, pid: int) -> bool:
        if not self.h:
            return False
        ph = self.k.OpenProcess(0x1F0FFF, False, pid)
        if not ph:
            return False
        try:
            return bool(self.k.AssignProcessToJobObject(self.h, ph))
        finally:
            self.k.CloseHandle(ph)

    def pids(self) -> list[int]:
        if not self.h:
            return []

        class IdList(ctypes.Structure):
            _fields_ = [("Assigned", ctypes.c_uint32), ("InList", ctypes.c_uint32), ("Ids", ctypes.c_size_t * 512)]

        buf = IdList()
        if not self.k.QueryInformationJobObject(self.h, 3, ctypes.byref(buf), ctypes.sizeof(buf), None):
            return []
        return [int(buf.Ids[i]) for i in range(min(buf.InList, 512))]

    def terminate(self) -> None:
        if self.h:
            self.k.TerminateJobObject(self.h, 1)

    def close(self) -> None:
        if self.h:
            self.k.CloseHandle(self.h)
            self.h = None


class Tree:
    """One started process and everything it starts."""

    def __init__(self, popen: "subprocess.Popen[Any]") -> None:
        self.p, self.root_pid = popen, popen.pid
        self.job = _JobObject()
        self.job_ok = self.job.assign(popen.pid)
        self.seen: set[int] = {popen.pid}

    def pids(self) -> list[int]:
        if self.job_ok:
            live = self.job.pids()
        else:
            try:
                import psutil
                live = [self.root_pid] + [c.pid for c in psutil.Process(self.root_pid).children(recursive=True)]
            except Exception:                                           # noqa: BLE001
                live = [self.root_pid] if self.p.poll() is None else []
        self.seen.update(live)
        return live

    def alive(self) -> bool:
        return self.p.poll() is None or bool(self.pids())

    def kill(self) -> list[int]:
        """End the whole tree by PID; returns every PID that belonged to it."""
        known = set(self.pids()) | self.seen
        if self.job_ok:
            self.job.terminate()
        _taskkill(self.root_pid)
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return sorted(known)

    def close(self) -> None:
        self.job.close()


class Procs:
    """Every tree one package started; the policy asks it which PIDs are 'own'."""

    def __init__(self) -> None:
        self.trees: list[Tree] = []
        self.lock = threading.Lock()

    def add(self, t: Tree) -> None:
        with self.lock:
            self.trees = [x for x in self.trees if x.alive()] + [t]

    def pids(self) -> list[int]:
        with self.lock:
            return sorted({p for t in self.trees for p in t.pids()})

    def kill_all(self) -> list[int]:
        with self.lock:
            out = sorted({p for t in self.trees for p in t.kill()})
            for t in self.trees:
                t.close()
            self.trees = []
            return out


def bash_path() -> Optional[Path]:
    return GIT_BASH if GIT_BASH.is_file() else None


def to_posix(p: str) -> str:
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    return f"/{m.group(1).lower()}/{m.group(2).replace(os.sep, '/')}" if m else p.replace("\\", "/")


def child_env(sandbox: Path, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k) and k not in _SCRUB}
    env["PYTHONPATH"] = str(sandbox)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


def bounded(path: Path, cap: int) -> tuple[str, bool]:
    try:
        data = path.read_bytes()
    except OSError:
        return "", False
    if len(data) <= cap:
        return data.decode("utf-8", errors="replace"), False
    head, tail = data[: cap * 2 // 3], data[-(cap // 3):]
    return head.decode("utf-8", errors="replace") + f"\n... [{len(data) - cap} bytes cut] ...\n" + tail.decode("utf-8", errors="replace"), True


def start_process(command: str, cwd: str, sandbox: Path, out_path: Path, err_path: Optional[Path], cwd_file: Optional[Path],
                  shell: Optional[str] = None) -> tuple["subprocess.Popen[Any]", Tree, list[Any]]:
    """Launch `command` (already gated) in the shell, stdout/stderr to files; returns (popen, tree, open handles)."""
    kind = shell or ("bash" if bash_path() else "cmd")
    if kind == "bash":
        tail = f"\n__rc=$?\npwd -W > '{to_posix(str(cwd_file))}' 2>/dev/null || pwd > '{to_posix(str(cwd_file))}'\nexit $__rc" if cwd_file else ""
        script = f"cd '{to_posix(cwd)}' || exit 97\n{command}{tail}"
        argv: list[str] = [str(bash_path()), "--noprofile", "--norc", "-c", script]
    else:
        # one command-line STRING: list2cmdline would escape the inner quotes with backslashes, which cmd does not understand
        tail = f' && (cd > "{cwd_file}") || (cd > "{cwd_file}" & exit /b 1)' if cwd_file else ""
        argv = f'cmd /d /s /c "{command}{tail}"'  # type: ignore[assignment]
    fo = open(out_path, "wb")
    fe = open(err_path, "wb") if err_path else None
    flags = (0x00000200 | 0x08000000) if sys.platform == "win32" else 0                    # new process group, no console window
    p = subprocess.Popen(argv, cwd=cwd, env=child_env(sandbox), stdout=fo, stderr=fe if fe else subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags, start_new_session=(sys.platform != "win32"))
    tree = Tree(p)
    return p, tree, [h for h in (fo, fe) if h]


class ShellSession:
    """    sh = ShellSession(policy, procs, scratch_dir)
           sh.run("cd creator && ls", timeout_s=30)      -> {"exit", "stdout", "stderr", "truncated", "cwd", "decision"...}
    A denied command returns exit -1 and the reason; nothing is started."""

    def __init__(self, policy: PO.Policy, procs: Procs, scratch: Path, shell: Optional[str] = None) -> None:
        self.policy, self.procs, self.scratch = policy, procs, Path(scratch)
        self.shell = shell or ("bash" if bash_path() else "cmd")
        self.cwd = str(Path(policy.sandbox).resolve())
        self.n = 0
        self.scratch.mkdir(parents=True, exist_ok=True)

    def run(self, command: str, timeout_s: Optional[float] = None, max_output: Optional[int] = None) -> dict[str, Any]:
        t0 = time.monotonic()
        d = self.policy.check_command(command, self.cwd, "cmd" if self.shell == "cmd" else "bash", "shell")
        if not d:
            self.policy.log(d, "bash", command, 0.0)
            return {"exit": -1, "stdout": "", "stderr": "", "truncated": False, "denied": d.reason, "rule": d.rule, "cwd": self.cwd}
        timeout = self.policy.clamp_timeout(timeout_s)
        cap = min(int(max_output or PO.MAX_OUTPUT), PO.MAX_OUTPUT * 5)
        self.n += 1
        out, err, cwdf = (self.scratch / f".run{self.n}.{x}" for x in ("out", "err", "cwd"))
        slot: Any = self.policy.test_slot() if re.search(r"\bpytest\b", command) else None
        timed_out, pids = False, []
        try:
            if slot is not None:
                slot.__enter__()
            p, tree, handles = start_process(command, self.cwd, Path(self.policy.sandbox), out, err, cwdf, self.shell)
            self.procs.add(tree)
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                pids = tree.kill()
            if not timed_out:
                tree.pids()
                if tree.alive():                                     # background children left behind: not ours to keep
                    pids = tree.kill()
            for h in handles:
                h.close()
        finally:
            if slot is not None:
                slot.__exit__(None, None, None)
        so, t1 = bounded(out, cap)
        se, t2 = bounded(err, cap // 2)
        rc = p.returncode if p.returncode is not None else -9
        self._take_cwd(cwdf)
        for f in (out, err, cwdf):
            f.unlink(missing_ok=True)
        res = {"exit": -9 if timed_out else rc, "stdout": so, "stderr": se, "truncated": t1 or t2, "cwd": self.cwd, "timed_out": timed_out,
               "pids": pids, "rule": d.rule}
        self.policy.log(d, "bash", command, time.monotonic() - t0)
        return res

    def _take_cwd(self, f: Path) -> None:
        try:
            new = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return
        if not new:
            return
        try:
            ab = self.policy.resolve(new, self.cwd)
        except PO._Deny:
            return
        if self.policy.check_abs(ab, False):
            self.cwd = os.path.normpath(new if os.path.isabs(new) else ab)
