"""Nupen tools - background jobs and the monitor (Claude Code's background Bash + Monitor tools).

A job is a gated command started with its output in a file under the package scratchpad; the package may list, read, wait for and stop
its OWN jobs only (ids are per package). Monitor(job or command, pattern, deadline) yields the lines of the job's output that match a
regex as events; wait_until polls a condition to a deadline. At most `policy.max_jobs` jobs run at once per package."""
from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from creator.tools import policy as PO
from creator.tools import shell as SH


@dataclasses.dataclass
class Job:
    id: str
    command: str
    pid: int
    tree: SH.Tree
    log: Path
    started: float
    handles: list[Any]
    stopped: bool = False


class JobTable:
    def __init__(self, policy: PO.Policy, procs: SH.Procs, scratch: Path) -> None:
        self.policy, self.procs = policy, procs
        self.dir = Path(scratch) / "jobs"
        self.jobs: dict[str, Job] = {}
        self.n = 0

    # -- helpers
    def _gate(self, tool: str, args: str, cls: str = "jobs") -> Optional[dict[str, Any]]:
        d = self.policy.check_tool(cls)
        self.policy.log(d, tool, args)
        return None if d else {"ok": False, "denied": d.reason, "rule": d.rule}

    def _get(self, jid: str) -> Optional[Job]:
        return self.jobs.get(str(jid))

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.tree.p.poll() is None]

    # -- start / status / output / stop
    def start(self, command: str, cwd: str, shell: Optional[str] = None) -> dict[str, Any]:
        den = self._gate("job_start", command)
        if den:
            return den
        d = self.policy.check_command(command, cwd, "cmd" if shell == "cmd" else "bash", "shell")
        if not d:
            self.policy.log(d, "job_start", command)
            return {"ok": False, "denied": d.reason, "rule": d.rule}
        if len(self.running()) >= self.policy.max_jobs:
            return {"ok": False, "denied": f"at most {self.policy.max_jobs} background jobs may run at once for a package", "rule": "builtin:max-jobs"}
        self.n += 1
        jid = f"j{self.n}"
        self.dir.mkdir(parents=True, exist_ok=True)
        log = self.dir / f"{jid}.log"
        slot: Any = self.policy.test_slot() if re.search(r"\bpytest\b", command) else None
        if slot is not None:
            slot.__enter__()
        p, tree, handles = SH.start_process(command, cwd, Path(self.policy.sandbox), log, None, None, shell)
        self.procs.add(tree)
        job = Job(jid, command, p.pid, tree, log, time.time(), handles + ([slot] if slot is not None else []))
        self.jobs[jid] = job
        return {"ok": True, "job": jid, "pid": p.pid, "log": str(log)}

    def status(self, jid: str) -> dict[str, Any]:
        den = self._gate("job_status", jid)
        if den:
            return den
        j = self._get(jid)
        if j is None:
            return {"ok": False, "denied": f"no job {jid!r} of this package", "rule": "builtin:own-jobs"}
        rc = j.tree.p.poll()
        if rc is not None:
            self._release(j)
        return {"ok": True, "job": j.id, "state": "running" if rc is None else ("stopped" if j.stopped else "exited"), "exit": rc,
                "pid": j.pid, "seconds": round(time.time() - j.started, 1), "command": j.command[:200]}

    def list(self) -> list[dict[str, Any]]:
        return [self.status(j) for j in list(self.jobs)]

    def output(self, jid: str, offset: int = 0, max_bytes: int = 20000) -> dict[str, Any]:
        den = self._gate("job_output", jid)
        if den:
            return den
        j = self._get(jid)
        if j is None:
            return {"ok": False, "denied": f"no job {jid!r} of this package", "rule": "builtin:own-jobs"}
        try:
            with j.log.open("rb") as fh:
                fh.seek(max(0, int(offset)))
                data = fh.read(max_bytes)
        except OSError:
            data = b""
        return {"ok": True, "text": data.decode("utf-8", errors="replace"), "next_offset": int(offset) + len(data)}

    def stop(self, jid: str) -> dict[str, Any]:
        den = self._gate("job_stop", jid)
        if den:
            return den
        j = self._get(jid)
        if j is None:
            return {"ok": False, "denied": f"no job {jid!r} of this package (only own jobs can be stopped)", "rule": "builtin:own-jobs"}
        pids = j.tree.kill()
        j.stopped = True
        self._release(j)
        return {"ok": True, "job": j.id, "pids": pids}

    def _release(self, j: Job) -> None:
        for h in j.handles:
            try:
                h.close() if hasattr(h, "close") else h.__exit__(None, None, None)
            except Exception:                                  # noqa: BLE001
                pass
        j.handles = []

    def stop_all(self) -> None:
        for j in list(self.jobs.values()):
            if j.tree.p.poll() is None or j.tree.alive():
                j.tree.kill()
            self._release(j)

    # -- monitor
    def monitor(self, source: str, pattern: str, deadline_s: float = 60.0, poll_s: float = 0.2, command: bool = False,
                cwd: str = "") -> Iterator[dict[str, Any]]:
        """Yield {'event': 'line', 'line': ...} for each output line matching `pattern`, then one {'event': 'end', 'reason': ...}.
        `source` is a job id of this package, or (command=True) a command started here and stopped when the monitor ends."""
        den = self._gate("monitor", f"{source} /{pattern}/", "monitor")
        if den:
            yield {"event": "end", "reason": "denied", **den}
            return
        if len(pattern) > 300:
            yield {"event": "end", "reason": "pattern too long"}
            return
        try:
            rx = re.compile(pattern)
        except re.error as e:
            yield {"event": "end", "reason": f"bad pattern: {e}"}
            return
        started: Optional[str] = None
        jid = source
        if command:
            r = self.start(source, cwd or str(self.policy.sandbox))
            if not r.get("ok"):
                yield {"event": "end", "reason": "denied", **r}
                return
            jid = started = r["job"]
        j = self._get(jid)
        if j is None:
            yield {"event": "end", "reason": f"no job {jid!r} of this package"}
            return
        end = time.monotonic() + PO.Policy.clamp_timeout(deadline_s)
        off, partial, reason = 0, "", "deadline"
        try:
            while True:
                done = j.tree.p.poll() is not None
                chunk = self.output(jid, off, 65536)
                text = partial + chunk.get("text", "")
                off = chunk.get("next_offset", off)
                lines = text.split("\n")
                partial = "" if done and not chunk.get("text") else lines.pop()
                for ln in lines:
                    if rx.search(ln):
                        yield {"event": "line", "line": ln.rstrip("\r"), "job": jid}
                if done and not chunk.get("text"):
                    if partial and rx.search(partial):
                        yield {"event": "line", "line": partial, "job": jid}
                    reason = "exited"
                    break
                if time.monotonic() >= end:
                    break
                time.sleep(poll_s)
        finally:
            if started:
                self.stop(started)
        yield {"event": "end", "reason": reason}


def wait_until(condition: Callable[[], bool], deadline_s: float = 60.0, poll_s: float = 0.2) -> bool:
    """Poll `condition` until it is true (returns True) or the deadline passes (False)."""
    end = time.monotonic() + PO.Policy.clamp_timeout(deadline_s)
    while True:
        if condition():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(poll_s)
