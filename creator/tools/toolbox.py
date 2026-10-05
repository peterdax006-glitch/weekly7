"""Nupen tools - the ToolBox one package gets, and the text protocol workers use to call it.

    box = ToolBox(sandbox, state, package, project="creator")
    box.call("bash", {"command": "ls creator", "timeout_s": 30})  -> dict      (every call is gated and logged by creator.tools.policy)
    ask_with_tools(ask, messages, box)                              -> the model's final reply, after answering its TOOL: requests

A worker requests a tool with one line `TOOL: {"tool": "<name>", "args": {...}}` in its reply; the results come back as text in the next
user message. Tools change only what a worker can DO while working: the kernel's sandbox evaluation, verdicts and adoption rules
are untouched. Loaded on demand (creator.registry capability "tools"); the kernel and the swarm never import this eagerly."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from creator.tools import files as F
from creator.tools import jobs as J
from creator.tools import policy as PO
from creator.tools import scratch as SC
from creator.tools import shell as SH

TOOL_LINE = re.compile(r"^\s*TOOL:\s*(\{.*\})\s*$", re.M)
MAX_CALLS_PER_ROUND, MAX_ROUNDS, MAX_RESULT_CHARS = 4, 4, 6000

TOOL_HELP = [
    ("bash", '{"command": str, "timeout_s": 120 (max 600), "background": false}', "run a shell command in your persistent session (cwd persists); background=true starts a job and returns its id"),
    ("read", '{"path": str, "offset": 1, "limit": 2000}', "read a file with line numbers"),
    ("write", '{"path": str, "content": str}', "write a whole file (LF)"),
    ("edit", '{"path": str, "old": str, "new": str, "replace_all": false}', "exact unique string replace; refuses an ambiguous match"),
    ("patch", '{"patch": str, "path": ""}', "SEARCH/REPLACE blocks or a unified diff; whitespace-tolerant, all-or-nothing, returns an undo record"),
    ("grep", '{"pattern": regex, "path": ".", "glob": "*.py", "ignore_case": false}', "regex search over files"),
    ("glob", '{"pattern": "creator/*.py", "path": "."}', "find files by pattern"),
    ("jobs", "{}", "list your background jobs"),
    ("job_status", '{"job": id}', "state and exit code of one of your jobs"),
    ("job_output", '{"job": id, "offset": 0}', "output of one of your jobs from a byte offset"),
    ("job_stop", '{"job": id}', "end one of your jobs (own jobs only)"),
    ("monitor", '{"job": id | "command": str, "pattern": regex, "deadline_s": 60}', "wait for output lines matching a regex; returns them as events"),
    ("scratch", "{}", "the path of your private scratchpad (also reachable as @scratch/<file> in file tools)"),
]


def policy_text(scratch: str = "") -> str:
    """The tools and the policy in plain words (put into every work package)."""
    tools = "\n".join(f"  - {n} {a}: {d}" for n, a, d in TOOL_HELP)
    return ("Tools you may call (reply with one line per call: TOOL: {\"tool\": \"<name>\", \"args\": {...}}; results come back as text):\n" + tools +
            (f"\nYour scratchpad: {scratch}" if scratch else "") +
            "\nPolicy (one gate checks every call and logs it): you may work only inside your sandbox and your own scratchpad; protected paths, "
            "secret files (.env, credentials, keys, tokens) and state/livesim are refused; no network or package installs; no git push/fetch/pull/"
            "remote/clone/stash/rebase/reset --hard; processes may only be ended by the PID of your own jobs, never by name; no registry, "
            "scheduler, services, permissions or model servers; commands run by name from an allowed list, with no $(...), backticks, variables "
            "or here-documents; commands time out at 120 s by default (600 s max), output is capped, and at most 2 background jobs run at once. "
            "Run tests as `python -m pytest -q <files>`. Denied calls tell you why. Tools never change how your result is measured.")


class ToolBox:
    def __init__(self, sandbox: Path | str, state: Path | str, package: str, project: str = "creator", shell: Optional[str] = None,
                 **policy_kw: Any) -> None:
        self.sandbox, self.state, self.package, self.project = Path(sandbox), Path(state), package, project
        self.scratch = SC.ensure(self.state, package)
        self.procs = SH.Procs()
        self.policy = PO.Policy(self.sandbox, self.scratch, self.state, package, project, tracked_pids=self.procs.pids, **policy_kw)
        self.shell = SH.ShellSession(self.policy, self.procs, self.scratch, shell)
        self.jobs = J.JobTable(self.policy, self.procs, self.scratch)
        self.files = F.FileTools(self.policy, lambda: self.shell.cwd)
        self.closed = False

    def describe(self) -> str:
        return policy_text(str(self.scratch))

    # -- dispatcher
    def call(self, tool: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        a = args or {}
        t0 = time.monotonic()
        try:
            return self._call(tool, a)
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments for {tool}: {e}"}
        except Exception as e:                                             # noqa: BLE001 - a tool never crashes the worker
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "seconds": round(time.monotonic() - t0, 2)}

    def _call(self, tool: str, a: dict[str, Any]) -> dict[str, Any]:
        if tool == "bash":
            if a.get("background"):
                return self.jobs.start(str(a["command"]), self.shell.cwd, self.shell.shell)
            r = self.shell.run(str(a["command"]), a.get("timeout_s"), a.get("max_output"))
            r["ok"] = r["exit"] == 0
            return r
        if tool == "read":
            return self.files.read(str(a["path"]), int(a.get("offset", 1)), int(a.get("limit", 2000)))
        if tool == "write":
            return self.files.write(str(a["path"]), str(a["content"]))
        if tool == "edit":
            return self.files.edit(str(a["path"]), str(a["old"]), str(a["new"]), bool(a.get("replace_all")))
        if tool == "patch":
            r = self.files.apply_patch(str(a["patch"]), str(a.get("path", "")))
            r.pop("undo", None) if not r.get("ok") else None
            return r
        if tool == "grep":
            return self.files.grep(str(a["pattern"]), str(a.get("path", ".")), a.get("glob"), bool(a.get("ignore_case")))
        if tool == "glob":
            return self.files.glob(str(a["pattern"]), str(a.get("path", ".")))
        if tool == "jobs":
            return {"ok": True, "jobs": self.jobs.list()}
        if tool == "job_status":
            return self.jobs.status(str(a["job"]))
        if tool == "job_output":
            return self.jobs.output(str(a["job"]), int(a.get("offset", 0)))
        if tool == "job_stop":
            return self.jobs.stop(str(a["job"]))
        if tool == "monitor":
            src, is_cmd = (str(a["command"]), True) if "command" in a else (str(a["job"]), False)
            ev = []
            for e in self.jobs.monitor(src, str(a["pattern"]), float(a.get("deadline_s", 60)), command=is_cmd, cwd=self.shell.cwd):
                ev.append(e)
                if len(ev) >= int(a.get("max_events", 50)):
                    break
            return {"ok": True, "events": ev}
        if tool == "scratch":
            d = self.policy.check_tool("scratch")
            self.policy.log(d, "scratch", "")
            return {"ok": True, "path": str(self.scratch)} if d else {"ok": False, "denied": d.reason, "rule": d.rule}
        return {"ok": False, "error": f"unknown tool {tool!r}"}

    def close(self, lesson_recorded: bool = False, keep_scratch: bool = False) -> list[int]:
        """End the package: stop every job, end every process tree by PID, and (lesson recorded) clean the scratchpad unless kept."""
        pids: list[int] = []
        if not self.closed:
            self.closed = True
            self.jobs.stop_all()
            pids = self.procs.kill_all()
            if lesson_recorded and not keep_scratch:
                SC.cleanup(self.state, self.package)
        return pids

    def __enter__(self) -> "ToolBox":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ------------------------------------------------------------------------------------------------ the text protocol

def parse_tool_calls(reply: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in TOOL_LINE.finditer(reply):
        try:
            d = json.loads(m.group(1))
        except ValueError:
            out.append({"tool": "", "args": {}, "error": "unparseable TOOL line"})
            continue
        if isinstance(d, dict) and isinstance(d.get("tool"), str):
            args = d.get("args")
            out.append({"tool": d["tool"], "args": args if isinstance(args, dict) else {}})
    return out


def render_result(tool: str, r: dict[str, Any]) -> str:
    if r.get("denied"):
        return f"{tool}: DENIED - {r['denied']}"
    if tool == "bash" and "stdout" in r:
        return f"exit {r['exit']}{' (timed out)' if r.get('timed_out') else ''}\n{r['stdout']}{('[stderr] ' + r['stderr']) if r.get('stderr') else ''}"[:MAX_RESULT_CHARS]
    return json.dumps(r, ensure_ascii=False)[:MAX_RESULT_CHARS]


def ask_with_tools(ask: Callable[[list[dict[str, str]]], str], messages: list[dict[str, str]], box: ToolBox,
                   max_rounds: int = MAX_ROUNDS) -> str:
    """Ask; while the reply holds TOOL: lines, run them (at most MAX_CALLS_PER_ROUND) and ask again with the results. Returns the last reply."""
    reply = ask(messages)
    for _ in range(max_rounds):
        calls = parse_tool_calls(reply)
        if not calls:
            break
        res = []
        for c in calls[:MAX_CALLS_PER_ROUND]:
            r = box.call(c["tool"], c["args"]) if c["tool"] else {"ok": False, "denied": c.get("error", "bad call")}
            res.append(f"RESULT of {c['tool'] or '?'}:\n{render_result(c['tool'], r)}")
        messages = messages + [{"role": "assistant", "content": reply},
                               {"role": "user", "content": "\n\n".join(res) + "\n\nContinue. Call more tools, or give your final answer in the required format."}]
        reply = ask(messages)
    return reply


def open_for(workdir: Path | str, state: Path | str, package: str, project: str = "creator") -> ToolBox:
    return ToolBox(workdir, state, package, project)


def register() -> None:
    """Make the tools reachable by name through creator.registry (a no-op where the registry module does not exist yet)."""
    try:
        import importlib
        importlib.import_module("creator.registry").register("tools", "creator.tools.toolbox")
    except ImportError:
        pass
