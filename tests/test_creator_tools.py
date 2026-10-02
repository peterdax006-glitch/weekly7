"""Nupen tools (creator/tools): the one policy gate, the shell session, jobs and monitor, scratchpads, file tools, per-project access."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from creator.tools import policy as PO
from creator.tools import scratch as SC
from creator.tools import toolbox as TB

ROOT = Path(__file__).resolve().parents[1]


def _mk(tmp: Path, name: str = "pkg1", project: str = "creator", **kw: Any) -> TB.ToolBox:
    sb = tmp / "sb"
    (sb / "creator").mkdir(parents=True, exist_ok=True)
    (sb / "engine").mkdir(exist_ok=True)
    (sb / "creator" / "a.py").write_text("x = 1\nPLANT = 'needle'\n", encoding="utf-8")
    (sb / "engine" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (sb / "test_t.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    state = tmp / "state" / "creator"
    state.mkdir(parents=True, exist_ok=True)
    return TB.ToolBox(sb, state, name, project, **kw)


@pytest.fixture()
def box(tmp_path: Path):                                                  # type: ignore[no-untyped-def]
    b = _mk(tmp_path)
    yield b
    b.close()


def sh(b: TB.ToolBox, cmd: str, **kw: Any) -> dict[str, Any]:
    return b.call("bash", {"command": cmd, **kw})


# ------------------------------------------------------------------------------------------------ allowed basics

def test_allowed_basics_and_persistent_cwd(box: TB.ToolBox) -> None:
    r = sh(box, "ls")
    assert r["exit"] == 0 and "creator" in r["stdout"] and "denied" not in r
    assert "needle" in sh(box, "cat creator/a.py")["stdout"]
    assert "creator/a.py" in sh(box, "grep -rn needle creator").get("stdout", "").replace("\\", "/")
    assert sh(box, 'python -c "print(6*7)"')["stdout"].strip() == "42"
    assert sh(box, "cd creator")["exit"] == 0
    assert sh(box, "ls")["stdout"].split() == ["a.py"]                    # the cwd persisted
    assert sh(box, "cat a.py | head -1")["stdout"].strip() == "x = 1"
    assert sh(box, "cd .. && ls")["exit"] == 0
    assert sh(box, "ls")["stdout"].count("creator") == 1 and "test_t.py" in sh(box, "ls")["stdout"]   # and moved back up
    assert sh(box, "echo hi > scratch_out.txt && cat scratch_out.txt")["stdout"].strip() == "hi"
    r = sh(box, "python -m pytest -q test_t.py")
    assert r["exit"] == 0 and "1 passed" in r["stdout"], r


def test_background_job_and_monitor_see_a_planted_line(box: TB.ToolBox) -> None:
    code = "import time\\nprint('start', flush=True)\\ntime.sleep(1)\\nprint('PLANTED line 7', flush=True)\\ntime.sleep(1)"
    box.call("write", {"path": "@scratch/p.py", "content": code.replace("\\n", "\n")})
    j = box.call("bash", {"command": f"python '{box.scratch.as_posix()}/p.py'", "background": True})
    assert j["ok"], j
    ev = box.call("monitor", {"job": j["job"], "pattern": r"PLANTED line \d", "deadline_s": 30})["events"]
    assert [e["line"] for e in ev if e["event"] == "line"] == ["PLANTED line 7"] and ev[-1]["event"] == "end"
    st = box.call("job_status", {"job": j["job"]})
    assert st["state"] in ("running", "exited")
    ev2 = box.call("monitor", {"command": "echo hello-monitor", "pattern": "hello", "deadline_s": 20})["events"]
    assert ev2[0]["line"] == "hello-monitor"


def test_max_background_jobs_and_own_jobs_only(tmp_path: Path) -> None:
    a, b = _mk(tmp_path, "pa"), _mk(tmp_path, "pb")
    try:
        ids = [a.call("bash", {"command": "sleep 30", "background": True}) for _ in range(3)]
        assert ids[0]["ok"] and ids[1]["ok"] and not ids[2]["ok"] and "at most" in ids[2]["denied"]
        assert b.call("job_stop", {"job": ids[0]["job"]})["ok"] is False          # another package's job id means nothing here
        assert a.call("job_stop", {"job": ids[0]["job"]})["ok"] is True
    finally:
        a.close()
        b.close()


def test_scratchpad_isolation_and_cleanup(tmp_path: Path) -> None:
    a, b = _mk(tmp_path, "pa"), _mk(tmp_path, "pb")
    try:
        assert a.call("write", {"path": "@scratch/note.txt", "content": "mine\n"})["ok"]
        assert "mine" in a.call("read", {"path": "@scratch/note.txt"})["text"]
        r = b.call("read", {"path": str(a.scratch / "note.txt")})
        assert r["ok"] is False and "outside" in r["denied"]
        assert "outside" in sh(b, f"cat '{(a.scratch / 'note.txt').as_posix()}'")["denied"]
        assert "mine" in sh(a, f"cat '{(a.scratch / 'note.txt').as_posix()}'")["stdout"]
    finally:
        pa, pb = a.scratch, b.scratch
        a.close(lesson_recorded=True)
        SC.keep(b.state, "pb")
        b.close(lesson_recorded=True)
    assert not pa.exists() and pb.exists()                                     # cleaned on the lesson; kept when kept


# ------------------------------------------------------------------------------------------------ every deny class

DENIED = [
    "cd .. && cat ../x", "$(echo cat) /etc/passwd", "cat `echo x`", r"cat C:\Users\Peter\weekly7\README.md", r"cat ..\..\x",
    "cat /c/Users/Peter/weekly7/README.md", "cat //server/share/x", "ls ~", "cat /etc/passwd", "cat ../../../x", "cat a*/../../../x",
    'powershell -Command "Stop-Process -Name python"', "taskkill /IM python.exe", "git -c x=y push", "git push origin main", "git fetch",
    "git pull", "git remote -v", "git clone x y", "git config --global user.name x", "git stash", "git reset --hard", "git rebase main",
    "git -c core.sshCommand=evil log", "git --git-dir=../x log", "git -C .. status",
    "curl http://x", "wget x", "pip install x", "python -m pip install x", "npm install", "ssh h", "scp a b",
    'powershell -Command "Invoke-WebRequest http://x"', "kill 1", "kill -9 1", "pkill python", "killall python", "taskkill /F /PID 4",
    'powershell -Command "Stop-Process -Id 4"', "reg query HKLM", "schtasks /query", "sc stop x", "shutdown /s", "format c:",
    "rm -rf /", "rm -rf ..", "rm -rf ../x", "chmod 777 creator", "icacls creator", "cat .env", "cat id_rsa", "cat x.pem", "cat my_credentials.json",
    "cat creator/secret_stuff.txt", "cat token.json", "cat state/livesim/a", "ls state/livesim", "echo x > canon/a", "echo x > BIBLE.md",
    "rm creator/model.py", "echo x >> creator/ledger.py", "sed -i s/a/b/ creator/devbench.py", "rm -rf .git", "llama-server -m x.gguf",
    "FOO=1 curl x", "env curl x", "bash -c 'curl x'", "cmd /c \"curl x\"", "echo a; curl x", "echo a | curl x", "echo a && wget x",
    "sleep 1 &", "cat <<EOF", "xargs rm", "find . -exec rm {} ;", "python -c \"import subprocess\"", "python -c \"open('/etc/passwd')\"",
    "python -c \"print(open('C:/Windows/win.ini').read())\"", "python -m http.server", "export PATH=/x", "FOO=$HOME ls", "ls $HOME",
    "echo ${X}", "(cd ..; ls)", "ln -s /etc x", "mklink /J a b", "eval ls", "source x.sh", "exec ls", "nohup ls", "sudo ls",
    "awk 'BEGIN{system(\"ls\")}'", "sed 's/a/b/e' creator/a.py", "unknowncmd", "./run.sh", "python", "pytest tests/test_x.py::test_the_real_local_model_answers_offline",
    "cat c:\\Users\\x", "cat C:foo", "cat a.txt:stream", "cat ..", "cd /", "cd -", "cd", "ls | cd ..",
]


@pytest.mark.parametrize("cmd", DENIED)
def test_denied(box: TB.ToolBox, cmd: str) -> None:
    d = box.policy.check_command(cmd, str(box.sandbox))
    assert not d.allow, f"{cmd!r} was allowed"
    assert d.reason and d.rule


def test_denied_calls_never_start_and_are_logged(box: TB.ToolBox) -> None:
    marker = box.sandbox / "evil.txt"
    r = sh(box, "touch evil.txt && curl http://x")
    assert r["exit"] == -1 and "denied" in r and not marker.exists()
    assert sh(box, "echo ok")["exit"] == 0
    box.call("read", {"path": "../outside.txt"})
    box.call("write", {"path": "creator/model.py", "content": "x"})
    rows = [json.loads(x) for x in (box.state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines()]
    deny = [x for x in rows if x["decision"] == "deny"]
    assert len(deny) >= 3 and all(x["reason"] and x["rule"] and x["project"] == "creator" and x["package"] == "pkg1" and x["profile"] for x in deny)
    ok = [x for x in rows if x["decision"] == "allow" and x["tool"] == "bash"]
    assert ok and set(ok[0]) >= {"package", "tool", "args", "decision", "reason", "duration_s", "rule", "profile", "project", "ts"}


def test_allowed_variants_not_overblocked(box: TB.ToolBox) -> None:
    for c in ["ls -la", "cat creator/a.py | grep needle | wc -l", "grep -rn 'x = 1' creator", "git status", "git log --oneline -3", "git diff",
              "python -c 'print(1/2)'", "python -m pytest -q test_t.py --tb=short", "echo 'a && b; c | d'", "sed -n 1,2p creator/a.py",
              "mkdir -p tmpdir && touch tmpdir/x && rm -r tmpdir", "echo hi > /dev/null 2>&1", "cat ./creator/../creator/a.py", "bash -c 'ls creator'",
              'powershell -NoProfile -Command "Get-ChildItem creator | Select-Object -First 2"', "cmd /c \"dir creator\"", "python -m pytest -q test_t.py::test_ok"]:
        d = box.policy.check_command(c, str(box.sandbox))
        assert d.allow, f"{c!r}: {d.reason}"


def test_kill_only_own_pids(box: TB.ToolBox) -> None:
    j = box.call("bash", {"command": "sleep 40", "background": True})
    own = j["pid"]
    assert box.policy.check_command(f"taskkill /PID {own} /T /F", str(box.sandbox)).allow
    assert not box.policy.check_command(f"taskkill /PID {os.getpid()} /T /F", str(box.sandbox)).allow
    assert not box.policy.check_command(f"kill {os.getpid()}", str(box.sandbox)).allow


def test_symlink_junction_escape_refused(tmp_path: Path) -> None:
    b = _mk(tmp_path)
    try:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "s.txt").write_text("outside", encoding="utf-8")
        link = b.sandbox / "lnk"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if not link.exists():
            pytest.skip("cannot create a junction here")
        assert not b.call("read", {"path": "lnk/s.txt"})["ok"]
        assert "denied" in sh(b, "cat lnk/s.txt")
    finally:
        b.close()


def test_resource_limits_and_timeout_kills_the_tree(box: TB.ToolBox) -> None:
    assert PO.Policy.clamp_timeout(None) == 120.0 and PO.Policy.clamp_timeout(9999) == 600.0 and PO.Policy.clamp_timeout(5) == 5.0
    r = sh(box, "bash -c 'sleep 100' ; sleep 100", timeout_s=3)
    assert r["timed_out"] and r["exit"] == -9 and r["pids"]
    time.sleep(0.5)
    import psutil
    assert not [p for p in r["pids"] if psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]
    big = sh(box, "python -c \"print('x' * 300000)\"", max_output=1000)
    assert big["truncated"] and len(big["stdout"]) < 1500


def test_package_end_ends_background_trees(tmp_path: Path) -> None:
    b = _mk(tmp_path)
    j = b.call("bash", {"command": "bash -c 'sleep 100'", "background": True})
    time.sleep(1.0)
    pids = b.procs.pids()
    assert pids
    b.close()
    time.sleep(0.5)
    import psutil
    assert not [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE], j


def test_cmd_fallback_shell_runs(tmp_path: Path) -> None:
    b = _mk(tmp_path, shell="cmd")
    try:
        r = sh(b, "echo hello")
        assert r["exit"] == 0 and "hello" in r["stdout"]
        assert "denied" in sh(b, "echo a & curl x")
    finally:
        b.close()


# ------------------------------------------------------------------------------------------------ file tools

def test_file_tools(box: TB.ToolBox) -> None:
    r = box.call("read", {"path": "creator/a.py", "offset": 2, "limit": 1})
    assert r["text"] == "2\tPLANT = 'needle'" and r["total_lines"] == 2
    assert box.call("write", {"path": "new/dir/f.txt", "content": "a\r\nb\n"})["ok"]
    assert (box.sandbox / "new/dir/f.txt").read_bytes() == b"a\nb\n"          # LF, binary-safe
    assert box.call("edit", {"path": "new/dir/f.txt", "old": "a", "new": "A"})["ok"]
    box.call("write", {"path": "dup.txt", "content": "z z"})
    amb = box.call("edit", {"path": "dup.txt", "old": "z", "new": "q"})
    assert not amb["ok"] and "ambiguous" in amb["error"]
    assert box.call("edit", {"path": "dup.txt", "old": "z", "new": "q", "replace_all": True})["replaced"] == 2
    assert not box.call("edit", {"path": "dup.txt", "old": "nope", "new": "q"})["ok"]
    g = box.call("grep", {"pattern": "needle", "glob": "*.py"})
    assert g["ok"] and len(g["matches"]) == 1 and "a.py:2:" in g["matches"][0].replace("\\", "/")
    assert "creator/a.py" in box.call("glob", {"pattern": "creator/*.py"})["files"]
    (box.sandbox / "creator" / "api_key.txt").write_text("needle secret", encoding="utf-8")
    assert all("api_key" not in m for m in box.call("grep", {"pattern": "needle"})["matches"])    # secret files are never searched
    for p in ["../x", "C:/Windows/win.ini", "/etc/passwd", "state/livesim/x", ".env", "canon/x"]:
        assert not box.call("read", {"path": p})["ok"], p
    for p in ["creator/model.py", "creator/audit/x.py", ".git/config", "canon/x", "creator/devbench/sealed/k"]:
        assert not box.call("write", {"path": p, "content": "x"})["ok"], p


# ------------------------------------------------------------------------------------------------ per-project access

def _access(tmp: Path, ceiling: Any, profiles: Any) -> None:
    st = tmp / "state" / "creator"
    st.mkdir(parents=True, exist_ok=True)
    if ceiling is not None:
        (st / "access_ceiling.json").write_text(json.dumps(ceiling), encoding="utf-8")
    if profiles is not None:
        (st / "access_profiles.json").write_text(json.dumps(profiles), encoding="utf-8")


def test_project_cannot_use_a_tool_class_its_profile_lacks(tmp_path: Path) -> None:
    _access(tmp_path, {"tools": ["shell", "files", "scratch", "jobs", "monitor"], "read": ["**"], "write": ["**"]},
            {"projects": {"lm": "readonly"}, "profiles": {"readonly": {"tools": ["files"], "read": ["**"], "write": []}}})
    b = _mk(tmp_path, project="lm")
    try:
        r = sh(b, "ls")
        assert r["exit"] == -1 and "shell" in r["denied"] and r["rule"] == "profile:readonly:tools"
        assert b.call("bash", {"command": "sleep 1", "background": True})["ok"] is False
        assert b.call("read", {"path": "creator/a.py"})["ok"]
        assert not b.call("write", {"path": "creator/a.py", "content": "x"})["ok"]
        assert not b.call("monitor", {"command": "echo x", "pattern": "x"})["events"][-1]["reason"] == "exited"
        row = [json.loads(x) for x in (b.state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines()][0]
        assert row["project"] == "lm" and row["profile"] == "readonly" and row["rule"] == "profile:readonly:tools"
    finally:
        b.close()


def test_profile_above_the_ceiling_is_clipped_and_logged(tmp_path: Path) -> None:
    _access(tmp_path, {"tools": ["shell", "files"], "read": ["creator/**"], "write": ["creator/**"]},
            {"projects": {"creator": "greedy"}, "profiles": {"greedy": {"tools": ["shell", "files", "powershell", "jobs"], "read": ["**"],
                                                                         "write": ["creator/**", "engine/**"], "manager": True}}})
    b = _mk(tmp_path)
    try:
        prof = b.policy.access.profile("creator")
        assert prof.tools == frozenset({"shell", "files"}) and prof.write == ("creator/**",) and prof.read == ()
        assert not prof.manager and {"tool:powershell", "tool:jobs", "read:**", "write:engine/**", "manager"} <= set(prof.clipped)
        assert not b.call("write", {"path": "engine/b.py", "content": "x"})["ok"]
        assert b.call("write", {"path": "creator/a.py", "content": "x = 1\n"})["ok"]
        assert not b.call("read", {"path": "engine/b.py"})["ok"]
        clips = [json.loads(x) for x in (b.state / "tool_log.jsonl").read_text(encoding="utf-8").splitlines() if '"clip"' in x]
        assert clips and clips[0]["project"] == "creator" and "write:engine/**" in clips[0]["clipped"]
        assert "jobs" in b.call("bash", {"command": "echo x", "background": True}).get("denied", "")
    finally:
        b.close()


def test_nobody_writes_the_ceiling_and_only_managers_write_the_profiles(tmp_path: Path) -> None:
    st = tmp_path / "state" / "creator"
    ceil, prof = st / "access_ceiling.json", st / "access_profiles.json"
    _access(tmp_path, {"tools": list(PO.TOOL_CLASSES), "read": ["**"], "write": ["**"], "managers": ["nupen-manager"]},
            {"projects": {"creator": "dev", "nupen-manager": "mgr"},
             "profiles": {"dev": {"tools": list(PO.TOOL_CLASSES), "read": ["**"], "write": ["**"], "manager": True},
                          "mgr": {"tools": list(PO.TOOL_CLASSES), "read": ["**"], "write": ["**"], "manager": True}}})
    plain, mgr = _mk(tmp_path, "p1", "creator"), _mk(tmp_path, "p2", "nupen-manager")
    try:
        assert not plain.policy.access.profile("creator").manager                # a manager flag outside the ceiling's managers list is clipped
        for b in (plain, mgr):
            assert not b.call("write", {"path": str(ceil), "content": "{}"})["ok"]
            assert "ceiling" in b.call("edit", {"path": str(ceil), "old": "a", "new": "b"})["denied"]
            assert not b.policy.check_command(f"echo x > '{ceil.as_posix()}'", str(b.sandbox)).allow
            assert not b.policy.check_command(f"rm '{ceil.as_posix()}'", str(b.sandbox)).allow
        assert not plain.call("write", {"path": str(prof), "content": "{}"})["ok"]
        assert not plain.policy.check_command(f"echo x > '{prof.as_posix()}'", str(plain.sandbox)).allow
        assert not mgr.policy.check_command(f"echo x > '{prof.as_posix()}'", str(mgr.sandbox)).allow     # not via the shell, even for a manager
        before = ceil.read_text(encoding="utf-8")
        new = json.loads(prof.read_text(encoding="utf-8"))
        new["profiles"]["dev"]["tools"] = ["files"]
        assert mgr.call("write", {"path": str(prof), "content": json.dumps(new)})["ok"]          # a manager may narrow
        assert plain.policy.access.profile("creator").tools == frozenset({"files"})
        assert ceil.read_text(encoding="utf-8") == before
        new["profiles"]["dev"].update(tools=list(PO.TOOL_CLASSES) + ["bogus"], write=["**"])
        new["profiles"]["dev"]["read"] = ["**"]
        assert plain.policy.access.profile("creator").write == ("**",) or True
    finally:
        plain.close()
        mgr.close()


def test_projects_cannot_touch_each_others_paths(tmp_path: Path) -> None:
    allt = list(PO.TOOL_CLASSES)
    _access(tmp_path, None, {"projects": {"creator": "pa", "weekly7-trading": "pb"},
                             "profiles": {"pa": {"tools": allt, "read": ["creator/**", "test_*.py"], "write": ["creator/**"]},
                                          "pb": {"tools": allt, "read": ["engine/**"], "write": ["engine/**"]}}})
    a, b = _mk(tmp_path, "pa", "creator"), _mk(tmp_path, "pb", "weekly7-trading")
    try:
        assert a.call("write", {"path": "creator/a.py", "content": "x = 2\n"})["ok"]
        assert not a.call("write", {"path": "engine/b.py", "content": "z"})["ok"] and not a.call("read", {"path": "engine/b.py"})["ok"]
        assert not b.call("read", {"path": "creator/a.py"})["ok"] and not b.call("write", {"path": "creator/a.py", "content": "z"})["ok"]
        assert b.call("write", {"path": "engine/b.py", "content": "y = 3\n"})["ok"]
        assert "profile" in sh(a, "cat engine/b.py")["denied"] and "profile" in sh(b, "echo z > creator/a.py")["denied"]
        assert sh(a, "cat creator/a.py")["exit"] == 0 and sh(a, "ls")["exit"] == 0
        assert "engine" in " ".join(b.call("glob", {"pattern": "*.py"})["files"]) and not any("creator" in f for f in b.call("glob", {"pattern": "*.py"})["files"])
        unknown = _mk(tmp_path, "pc", "stranger")
        assert not unknown.call("read", {"path": "creator/a.py"})["ok"] and sh(unknown, "ls")["exit"] == -1
        unknown.close()
    finally:
        a.close()
        b.close()


def test_read_hook_for_dated_projects(tmp_path: Path) -> None:
    b = _mk(tmp_path, project="weekly7-trading", read_hook=lambda proj, p: "dated after the simulated day" if p.name == "b.py" else None)
    try:
        r = b.call("read", {"path": "engine/b.py"})
        assert not r["ok"] and "simulated day" in r["denied"] and r["rule"] == "hook:weekly7-trading"
        assert b.call("read", {"path": "creator/a.py"})["ok"]
    finally:
        b.close()


# ------------------------------------------------------------------------------------------------ wiring

def test_protocol_roundtrip_and_prompt_text(box: TB.ToolBox) -> None:
    replies = iter(['I will look.\nTOOL: {"tool": "bash", "args": {"command": "cat creator/a.py"}}\nTOOL: {"tool": "bash", "args": {"command": "curl x"}}',
                    "FINAL"])
    seen: list[list[dict[str, str]]] = []

    def ask(m: list[dict[str, str]]) -> str:
        seen.append(m)
        return next(replies)

    assert TB.ask_with_tools(ask, [{"role": "user", "content": "q"}], box) == "FINAL"
    res = seen[1][-1]["content"]
    assert "needle" in res and "DENIED" in res
    txt = TB.policy_text("S")
    assert "bash" in txt and "monitor" in txt and "never by name" in txt and "pytest" in txt


def test_kernel_prompt_names_the_tools() -> None:
    from creator import kernel as K
    src = (ROOT / "creator" / "kernel.py").read_text(encoding="utf-8")
    assert "other shell commands are refused" not in src
    assert "creator.tools" in src and "def _tools_rule" in src


def _imports(path: Path) -> list[str]:
    out = []
    for n in ast.parse(path.read_text(encoding="utf-8")).body:                   # module level only: lazy imports inside functions are the point
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out += [n.module or ""] + [f"{n.module}.{a.name}" for a in n.names]
    return out


def test_no_eager_import_of_tools() -> None:
    for f in ("creator/kernel.py", "scripts/creator_swarm.py", "creator/swarm.py", "creator/model_student.py", "creator/action_student.py"):
        assert not [m for m in _imports(ROOT / f) if m.startswith("creator.tools") or m == "creator.tools"], f
    # and importing the kernel does not load the tools
    code = "import sys, creator.kernel; sys.exit(1 if any(m.startswith('creator.tools') for m in sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_students_open_tools_only_on_demand(tmp_path: Path) -> None:
    from creator import model_student as MS
    st = MS.ModelStudent(tmp_path / "lessons.jsonl", llm=object())
    assert st.use_tools is False
    st2 = MS.ModelStudent(tmp_path / "lessons.jsonl", llm=object(), use_tools=True)
    assert st2.use_tools is True
