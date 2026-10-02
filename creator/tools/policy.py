"""Nupen tools - THE gate every tool call passes (owner, 2 Oct 2026: "all the commands you are capable of running like that, this ai should
be able to run"; "it should be able to manage what projects have access to what").

`Policy.check_command / check_path / check_tool` return a Decision(allow, reason, rule); `Policy.log` appends it to
state/creator/tool_log.jsonl with the project, the profile and the rule that decided. The policy is the security boundary of the
tools: commands are LEXED (quotes, chains, redirects, substitutions), every command word must be on an allow-list, every path
argument is resolved (symlinks, `..`, drive letters, UNC, MSYS `/c/...`, `~`, globs expanded) and must stay inside the package
sandbox or its own scratchpad; anything that cannot be understood is denied. Per-project access (profiles clipped to an owner-owned
ceiling) is applied on top of the built-in rules. Honest limits: a Python or test process the policy lets run can do what Python can
do - `python -c`/`python file.py` are scanned for process, network and path-escape idioms, `pytest` is not (the kernel's sandbox
firewall and the measured verdict remain the authority over what is adopted)."""
from __future__ import annotations

import dataclasses
import fnmatch
import glob as _glob
import hashlib
import json
import ntpath
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

TOOL_CLASSES = ("shell", "powershell", "files", "jobs", "monitor", "scratch")
MAX_TIMEOUT_S, DEFAULT_TIMEOUT_S, MAX_OUTPUT = 600.0, 120.0, 20000
MAX_JOBS = 2
_DEPTH = 3


@dataclasses.dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str = ""
    rule: str = ""

    def __bool__(self) -> bool:
        return self.allow


class _Deny(Exception):
    def __init__(self, reason: str, rule: str = "builtin") -> None:
        super().__init__(reason)
        self.reason, self.rule = reason, rule


# ------------------------------------------------------------------------------------------------ command tables

DENY_WORDS: dict[str, tuple[str, str]] = {}


def _deny(words: str, cls: str, why: str) -> None:
    for w in words.split():
        DENY_WORDS[w] = (cls, why)


_deny("curl wget invoke-webrequest iwr invoke-restmethod irm ssh scp sftp ftp telnet nc ncat netcat nslookup ping tracert "
      "pip pip3 pipx npm npx yarn pnpm conda choco winget scoop apt apt-get brew gem cargo rsync", "network", "network and package-install tools are denied")
_deny("pkill killall", "kill", "processes are never ended by name; kill a PID of your own job tree")
_deny("reg regedit regini schtasks sc service systemctl net netsh shutdown restart-computer stop-computer format diskpart bcdedit vssadmin "
      "icacls takeown chmod chown chgrp setx runas sudo doas su wmic mshta rundll32 cipher crontab at mklink ln attrib start explorer "
      "start-process set-itemproperty new-itemproperty new-service set-service invoke-expression iex invoke-command",
      "system", "registry, scheduling, services, permissions, privilege and system-wide commands are denied")
_deny("xargs eval exec source . alias trap function declare typeset set unset-all nohup disown", "indirect", "indirect execution is denied (use plain commands)")
_deny("llama-server llama-cli llama-cpp ollama llamafile koboldcpp", "model", "starting a model is denied (the machine-wide model lock owns that)")

READERS = set("""ls dir cat type head tail wc grep egrep fgrep rg sort uniq cut tr diff cmp echo printf pwd true false test [ sleep date
basename dirname realpath readlink stat file du which where tasklist ps env printenv nl tac rev fold paste comm join sha256sum md5sum
cksum od xxd hexdump strings tree more less column jq whoami hostname uname findstr fc expr seq tput wait jobs mypy""".split())
WRITERS = set("rm rmdir mv cp touch mkdir tee install truncate patch dd del erase copy move ren rename md rd xcopy robocopy ruff".split())
PY_WORDS = {"python", "python3", "py", "python3.11", "pythonw"}
PY_MODULES = set("""pytest mypy ruff unittest json.tool py_compile compileall doctest ast tokenize timeit trace pyflakes black isort dis
pyclbr pydoc calendar this""".split())
SAFE_ENV_PREFIX = ("NUPEN_TEST", "PYTEST", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING", "LC_", "LANG", "TZ", "COLUMNS", "NO_COLOR", "FORCE_COLOR")
FORBIDDEN_ENV = re.compile(r"^(PATH|PATHEXT|PYTHONPATH|PYTHONSTARTUP|PYTHONHOME|BASH_ENV|ENV|IFS|LD_.*|DYLD_.*|GIT_.*|NUPEN_REAL_MODEL_TEST|"
                           r"NUPEN_TEST_SLOTS_DIR|NUPEN_STOP|COMSPEC|SHELL|HOME|USERPROFILE|TEMP|TMP)$", re.I)
GIT_ALLOW = set("""status diff log show add commit checkout switch branch restore rev-parse ls-files grep blame apply mv rm merge-base
describe cat-file show-ref reset diff-tree shortlog rev-list ls-tree config tag cherry-pick merge notes count-objects whatchanged
diff-files diff-index for-each-ref name-rev help version clean stage""".split())
GIT_BAD_C = re.compile(r"^(alias\.|core\.(ssh|fsmonitor|pager|editor|hookspath|askpass|sshcommand)|credential|http|url\.|remote\.|protocol|"
                       r"include|filter\.|diff\..*\.(command|textconv)|gpg|sequence\.editor|user\.signingkey)", re.I)
SECRET_NAME = re.compile(r"(^\.env($|\.)|\.env$|credential|secret|^id_(rsa|dsa|ecdsa|ed25519)|\.pem$|\.key$|\.p12$|\.pfx$|\.kdbx$|^\.netrc$|"
                         r"^\.npmrc$|^\.pypirc$|^\.git-credentials$|(^|[._-])tokens?([._-]|$)|api[_-]?key)", re.I)
PY_BAD = re.compile(r"\b(subprocess|os\.system|os\.popen|os\.exec\w*|os\.spawn\w*|os\.kill|os\.startfile|socket|urllib|http\.client|http\.server|"
                    r"requests|httpx|aiohttp|ftplib|smtplib|telnetlib|ctypes|winreg|_winapi|shutil\.rmtree|signal\.|__import__|importlib|"
                    r"\beval\s*\(|\bexec\s*\(|\bpty\b|multiprocessing|webbrowser|paramiko|llama|gguf|LocalModel)\b")
ESCAPE_RE = re.compile(r"""(?:^|[\s'"=(,\[{])((?:/[^\s'"),\]}]*)|(?:[A-Za-z]:[\\/][^\s'"),\]}]*)|(?:\\\\[^\s'"),\]}]*)|(?:~[\\/][^\s'"),\]}]*)|"""
                       r"""(?:\.\.[\\/][^\s'"),\]}]*)|(?:\.\.))(?=$|[\s'"),\]}])""")
PATHY = re.compile(r"^[\w.\-/\\: ~@+,%=#\[\]*?]+$")


# ------------------------------------------------------------------------------------------------ the lexer

@dataclasses.dataclass
class Word:
    text: str                       # quotes removed, backslashes kept literally (Windows paths)
    alt: str = ""                   # the same word as bash would unescape it
    glob: bool = False              # an unquoted * ? [
    quoted: bool = False


@dataclasses.dataclass
class Seg:
    words: list[Word] = dataclasses.field(default_factory=list)
    redirs: list[tuple[str, Word]] = dataclasses.field(default_factory=list)
    before: str = ""
    after: str = ""


def lex(cmd: str, mode: str = "sh") -> list[Seg]:
    """Split a command line into simple commands. Raises _Deny for anything whose meaning cannot be established statically."""
    if len(cmd) > 20000 or "\x00" in cmd:
        raise _Deny("command too long or holds a NUL byte")
    segs: list[Seg] = []
    cur = Seg()
    lit: list[str] = []
    alt: list[str] = []
    st = {"in": False, "glob": False, "quoted": False, "pending": ""}
    conn = ""

    def endword() -> None:
        if not st["in"]:
            return
        w = Word("".join(lit), "".join(alt), bool(st["glob"]), bool(st["quoted"]))
        lit.clear()
        alt.clear()
        st["in"], st["glob"], st["quoted"] = False, False, False
        if st["pending"]:
            cur.redirs.append((str(st["pending"]), w))
            st["pending"] = ""
        else:
            cur.words.append(w)

    def endseg(c: str) -> None:
        nonlocal cur, conn
        endword()
        if st["pending"]:
            raise _Deny("redirect without a target")
        if cur.words or cur.redirs:
            cur.after = c
            segs.append(cur)
            cur = Seg(before=c)
        else:
            if c in ("&&", "||", "|") and segs:
                raise _Deny("empty command around an operator")
            cur.before = c or cur.before
        conn = c

    def add(c: str, a: Optional[str] = None) -> None:
        lit.append(c)
        alt.append(c if a is None else a)
        st["in"] = True

    i, n = 0, len(cmd)
    while i < n:
        c = cmd[i]
        nx = cmd[i + 1] if i + 1 < n else ""
        if mode == "sh" and c == "'":
            j = cmd.find("'", i + 1)
            if j < 0:
                raise _Deny("unterminated quote")
            for ch in cmd[i + 1:j]:
                add(ch)
            st["in"], st["quoted"] = True, True
            i = j + 1
            continue
        if c == '"':
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise _Deny("unterminated quote")
                d = cmd[j]
                if d == '"':
                    break
                if mode == "sh":
                    if d in "$`":
                        raise _Deny("variable or command substitution is denied")
                    if d == "\\" and j + 1 < n and cmd[j + 1] in '"\\$`':
                        buf.append(cmd[j + 1])
                        j += 2
                        continue
                elif d in "%^":
                    raise _Deny("cmd expansion characters (% ^) are denied")
                buf.append(d)
                j += 1
            for ch in buf:
                add(ch)
            st["in"], st["quoted"] = True, True
            i = j + 1
            continue
        if mode == "sh" and c == "\\":
            if nx == "\n":
                i += 2
                continue
            lit.append("\\")
            alt.append(nx)
            st["in"] = True
            i += 2
            continue
        if c in "$`" and mode == "sh":
            raise _Deny("variable or command substitution is denied")
        if c in "%^" and mode == "cmd":
            raise _Deny("cmd expansion characters (% ^) are denied")
        if c in "(){}":
            raise _Deny("subshells, groups and brace expansion are denied")
        if c in " \t\r":
            endword()
        elif c == "\n":
            endseg(";")
        elif c == ";" and mode == "sh":
            endseg(";")
        elif c == "#" and mode == "sh" and not st["in"]:
            while i < n and cmd[i] != "\n":
                i += 1
            continue
        elif c == "&":
            if nx == "&":
                endseg("&&")
                i += 1
            elif nx == ">":
                endword()
                st["pending"] = ">"
                i += 1 + (1 if cmd[i + 2:i + 3] == ">" else 0)
            elif mode == "cmd":
                endseg("&")
            else:
                raise _Deny("a lone & (background) is denied; start a background job through the jobs tool")
        elif c == "|":
            if nx == "|":
                endseg("||")
                i += 1
            elif nx == "&":
                raise _Deny("|& is denied")
            else:
                endseg("|")
        elif c in "<>":
            if st["in"] and "".join(lit).isdigit():
                lit.clear()
                alt.clear()
                st["in"] = False
            else:
                endword()
            if c == "<" and nx == "<":
                raise _Deny("here-documents are denied (use the write tool)")
            if c == "<" and nx == ">":
                raise _Deny("<> redirect is denied")
            op = c
            if c == ">" and nx == ">":
                op, i = ">>", i + 1
            if cmd[i + 1:i + 2] == "&" and re.match(r"[0-9-]", cmd[i + 2:i + 3] or "x"):
                i += 2
                while i < n and cmd[i] in "0123456789-":
                    i += 1
                continue
            st["pending"] = op
        else:
            if c in "*?[":
                st["glob"] = True
            add(c)
        i += 1
    endseg("")
    if conn in ("&&", "||", "|") and not cur.words and segs:
        raise _Deny("a command ends with an operator")
    return segs


# ------------------------------------------------------------------------------------------------ access profiles

def _glob_re(g: str) -> "re.Pattern[str]":
    out, i = [], 0
    while i < len(g):
        if g[i:i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif g[i:i + 2] == "**":
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        elif g[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$", re.I)


def glob_match(rel: str, globs: Iterable[str]) -> Optional[str]:
    for g in globs:
        if _glob_re(g).match(rel):
            return g
    return None


def covered(g: str, ceiling: Sequence[str]) -> bool:
    for c in ceiling:
        if c in ("**", "*", g):
            return True
        if c.endswith("/**") and (g == c[:-3] or g.startswith(c[:-3] + "/")):
            return True
        if "*" not in g and _glob_re(c).match(g):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class Profile:
    project: str
    name: str
    tools: frozenset[str]
    read: tuple[str, ...]
    write: tuple[str, ...]
    manager: bool = False
    known: bool = True
    clipped: tuple[str, ...] = ()


DEFAULT_CEILING: dict[str, Any] = {"tools": list(TOOL_CLASSES), "read": ["**"], "write": ["**"], "managers": []}
_CEIL_KEYS = ("tools", "read", "write")


class Access:
    """state/creator/access_ceiling.json (OWNER-OWNED maximum) and state/creator/access_profiles.json (NUPEN-MANAGED assignment
    {"projects": {project: profile_name}, "profiles": {profile_name: {tools, read, write, manager}}}). The effective profile of a
    project is its assigned profile CLIPPED to the ceiling at every load, so no edit of the managed file can exceed the ceiling.
    Without a profiles file every project gets the ceiling itself; with one, an unlisted project gets nothing."""

    def __init__(self, ceiling_path: Optional[Path], profiles_path: Optional[Path], log: Optional[Callable[[dict[str, Any]], None]] = None):
        self.ceiling_path = Path(ceiling_path) if ceiling_path else None
        self.profiles_path = Path(profiles_path) if profiles_path else None
        self._log = log
        self._seen: set[tuple[str, str]] = set()
        self._cache: dict[Any, Any] = {}

    def _load(self, p: Optional[Path]) -> Optional[dict[str, Any]]:
        if p is None or not p.is_file():
            return None
        try:
            st = p.stat()
            key = (st.st_mtime_ns, st.st_size)
            hit = self._cache.get(str(p))
            if hit is not None and hit[0] == key:
                return hit[1]  # type: ignore[no-any-return]
            d = json.loads(p.read_text(encoding="utf-8"))
            out = d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {"__broken__": True}
        self._cache[str(p)] = (key, out)
        return out

    def ceiling(self) -> dict[str, Any]:
        d = self._load(self.ceiling_path)
        if d is None:
            return dict(DEFAULT_CEILING)
        if d.get("__broken__"):
            return {"tools": [], "read": [], "write": [], "managers": []}          # an unreadable ceiling grants nothing
        return {"tools": [t for t in d.get("tools", []) if t in TOOL_CLASSES], "read": [str(x) for x in d.get("read", [])],
                "write": [str(x) for x in d.get("write", [])], "managers": [str(x) for x in d.get("managers", [])]}

    def profile(self, project: str) -> Profile:
        ceil, prof = self.ceiling(), self._load(self.profiles_path)
        if prof is None:
            return Profile(project, "ceiling", frozenset(ceil["tools"]), tuple(ceil["read"]), tuple(ceil["write"]), project in ceil["managers"])
        pname = (prof.get("projects") or {}).get(project)
        raw = (prof.get("profiles") or {}).get(pname) if pname else None
        if not isinstance(raw, dict):
            return Profile(project, str(pname or "(unassigned)"), frozenset(), (), (), False, known=False)
        clipped: list[str] = []
        tools = frozenset(str(t) for t in raw.get("tools", []) if t in TOOL_CLASSES and t in ceil["tools"])
        clipped += [f"tool:{t}" for t in raw.get("tools", []) if t not in tools]
        paths: dict[str, tuple[str, ...]] = {}
        for k in ("read", "write"):
            keep = [str(g) for g in raw.get(k, []) if covered(str(g), ceil[k])]
            clipped += [f"{k}:{g}" for g in raw.get(k, []) if str(g) not in keep]
            paths[k] = tuple(keep)
        mgr = bool(raw.get("manager")) and project in ceil["managers"]
        if raw.get("manager") and not mgr:
            clipped.append("manager")
        out = Profile(project, str(pname), tools, paths["read"], paths["write"], mgr, True, tuple(clipped))
        if clipped and self._log:
            key = (project, hashlib.sha256(json.dumps(clipped).encode()).hexdigest())
            if key not in self._seen:
                self._seen.add(key)
                self._log({"event": "clip", "project": project, "profile": str(pname), "clipped": clipped})
        return out


# ------------------------------------------------------------------------------------------------ the policy

def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


def _under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("\\/") + os.sep)


class Policy:
    """One per package. `sandbox` = the worker's worktree, `scratch` = its scratchpad, `state` = state/creator (tool log, access files)."""

    def __init__(self, sandbox: Path, scratch: Path, state: Path, package: str = "", project: str = "creator",
                 protected: Optional[Sequence[str]] = None, tracked_pids: Optional[Callable[[], Iterable[int]]] = None,
                 ceiling_path: Optional[Path] = None, profiles_path: Optional[Path] = None,
                 read_hook: Optional[Callable[[str, Path], Optional[str]]] = None,
                 test_slot: Optional[Callable[[], Any]] = None, max_jobs: int = MAX_JOBS):
        if protected is None:
            from creator import sandbox as S
            protected = S.PROTECTED
        self.sandbox, self.scratch, self.state = Path(sandbox), Path(scratch), Path(state)
        self.package, self.project, self.protected = package, project, tuple(protected)
        self.tracked_pids = tracked_pids or (lambda: ())
        self.read_hook, self.max_jobs = read_hook, max_jobs
        self.test_slot = test_slot or machine_test_slot                       # hook: returns a context manager held while a test process runs (machine-wide test budget)
        self.root, self.scratch_root = _norm(str(sandbox)), _norm(str(scratch))
        self.log_path = self.state / "tool_log.jsonl"
        self.ceiling_path = Path(ceiling_path) if ceiling_path else self.state / "access_ceiling.json"
        self.profiles_path = Path(profiles_path) if profiles_path else self.state / "access_profiles.json"
        self.access = Access(self.ceiling_path, self.profiles_path, self._append)
        self._ceil_n, self._prof_n = _norm(str(self.ceiling_path)), _norm(str(self.profiles_path))
        self.livesim = _norm(str(self.state.parent / "livesim"))

    # -------------------------------------------------------------- logging
    def _append(self, rec: dict[str, Any]) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "package": self.package, "project": self.project, **rec}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def log(self, d: Decision, tool: str, args: Any, duration_s: float = 0.0) -> None:
        prof = self.access.profile(self.project)
        self._append({"tool": tool, "args": str(args)[:300], "decision": "allow" if d.allow else "deny", "reason": d.reason[:300],
                      "rule": d.rule, "profile": prof.name, "duration_s": round(duration_s, 3)})

    # -------------------------------------------------------------- tool classes
    def check_tool(self, tool_class: str) -> Decision:
        prof = self.access.profile(self.project)
        if tool_class not in TOOL_CLASSES:
            return Decision(False, f"unknown tool class {tool_class!r}", "builtin:tool-class")
        if tool_class not in prof.tools:
            return Decision(False, f"project {self.project!r} (profile {prof.name!r}) may not use the {tool_class} tool class",
                            f"profile:{prof.name}:tools")
        return Decision(True, "", f"profile:{prof.name}:tools[{tool_class}]")

    # -------------------------------------------------------------- paths
    def resolve(self, text: str, cwd: str) -> str:
        """The absolute, normalized, symlink-resolved path `text` names from `cwd`; raises _Deny for forms that cannot be trusted."""
        t = text
        if t in ("/dev/null", "nul", "NUL", "/dev/stdin", "/dev/stdout", "/dev/stderr"):
            return "<null>"
        if t.startswith("~"):
            raise _Deny(f"{text!r}: home-directory paths are outside the sandbox")
        if t.startswith(("\\\\", "//")) and not re.match(r"^//[a-zA-Z](/|$)", t):
            raise _Deny(f"{text!r}: UNC and device paths are denied")
        m = re.match(r"^/(?:cygdrive/)?([a-zA-Z])(?:/(.*))?$", t.replace("\\", "/")) if t.startswith(("/", "\\")) else None
        if m and not re.match(r"^[/\\](tmp|usr|etc|bin|dev|proc|var|home|opt|mnt)\b", t):
            t = f"{m.group(1)}:/" + (m.group(2) or "")
        elif t.startswith(("/", "\\")):
            drv = ntpath.splitdrive(cwd)[0]
            if not re.match(r"^[/\\](tmp|usr|etc|bin|dev|proc|var|home|opt|mnt)\b", t):
                t = drv + t.replace("\\", "/")
            else:
                raise _Deny(f"{text!r}: absolute paths outside the sandbox are denied")
        drv, rest = ntpath.splitdrive(t)
        if drv and not re.match(r"^[A-Za-z]:[\\/]", t):
            raise _Deny(f"{text!r}: drive-relative or device path is denied")
        if ":" in rest or "\\\\?\\" in t or "\\\\.\\" in t:
            raise _Deny(f"{text!r}: alternate data streams and device paths are denied")
        if re.search(r"[. ]+(?:$|[\\/])", rest.rstrip("\\/") + "/") and re.search(r"[^\\/.][. ]+(?:[\\/]|$)", rest):
            raise _Deny(f"{text!r}: names with trailing dots or spaces are denied")
        full = ntpath.normpath(ntpath.join(cwd, t))
        return _norm(full)

    def _inside(self, abs_path: str, allow_scratch: bool = True) -> bool:
        return abs_path == "<null>" or _under(abs_path, self.root) or (allow_scratch and _under(abs_path, self.scratch_root))

    def _rel(self, abs_path: str) -> str:
        if _under(abs_path, self.root):
            return abs_path[len(self.root):].lstrip("\\/").replace("\\", "/")
        return "@scratch/" + abs_path[len(self.scratch_root):].lstrip("\\/").replace("\\", "/")

    def check_abs(self, abs_path: str, write: bool, tool_class: str = "shell", delete: bool = False) -> Decision:
        """The path rules for ONE resolved absolute path."""
        prof = self.access.profile(self.project)
        if abs_path == "<null>":
            return Decision(True, "", "builtin:null-device")
        if write and abs_path in (self._ceil_n, self._prof_n):
            if abs_path == self._ceil_n:
                return Decision(False, "the access ceiling is owner-owned: no tool may write it", "builtin:ceiling-file")
            if not (prof.manager and tool_class == "files"):
                return Decision(False, "the access profiles file may only be written by a manager project through the files tool",
                                "builtin:profiles-file")
            return Decision(True, "", f"profile:{prof.name}:manager")
        if not prof.manager and abs_path in (self._ceil_n, self._prof_n):
            return Decision(False, "access files are readable only by manager projects", "builtin:access-files")
        if abs_path in (self._ceil_n, self._prof_n):
            return Decision(True, "", f"profile:{prof.name}:manager")
        if _under(abs_path, self.livesim) or re.search(r"[\\/]state[\\/]livesim([\\/]|$)", abs_path):
            return Decision(False, "state/livesim is never read or written by tools", "builtin:livesim")
        if not self._inside(abs_path):
            return Decision(False, "path is outside the package sandbox and its scratchpad", "builtin:outside-sandbox")
        in_sb = _under(abs_path, self.root)
        if not in_sb and not _under(abs_path, self.scratch_root):
            return Decision(False, "path is outside the package sandbox and its scratchpad", "builtin:outside-sandbox")
        base = os.path.basename(abs_path)
        if SECRET_NAME.search(base) or any(SECRET_NAME.search(p) for p in abs_path.split(os.sep)[len(self.root.split(os.sep)):] if in_sb):
            return Decision(False, f"{base!r} looks like a secret file (.env, credentials, keys, tokens)", "builtin:secret")
        rel = self._rel(abs_path)
        if in_sb and write:
            from creator import sandbox as S
            if S.is_protected(rel, self.protected):
                return Decision(False, f"{rel} is a protected path (the sandbox firewall owns it)", "builtin:protected")
            if rel == ".git" or rel.startswith(".git/") or rel == ".creator_sandbox.json":
                return Decision(False, "the worktree's git metadata is not edited directly", "builtin:git-metadata")
            if delete and abs_path == self.root:
                return Decision(False, "the sandbox root itself is never removed", "builtin:sandbox-root")
        if in_sb and not write and re.match(r"^(\.git)(/|$)", rel):
            return Decision(False, "read the repository through git commands, not .git", "builtin:git-metadata")
        kind = "write" if write else "read"
        globs = getattr(prof, kind)
        hit = glob_match(rel, globs)
        if hit is None and not write and os.path.isdir(abs_path):
            hit = "@dir"                                       # listing a directory; the files in it are gated one by one
        if hit is None and not (not in_sb):                  # the package's own scratchpad is always its own
            return Decision(False, f"project {self.project!r} (profile {prof.name!r}) has no {kind} rule for {rel}", f"profile:{prof.name}:{kind}")
        if hit is None:
            hit = "@scratch"
        if not write and self.read_hook is not None:
            why = self.read_hook(self.project, Path(abs_path))
            if why:
                return Decision(False, why, f"hook:{self.project}")
        return Decision(True, "", f"profile:{prof.name}:{kind}[{hit}]")

    def check_path(self, path: str, write: bool = False, cwd: Optional[str] = None, tool_class: str = "files", delete: bool = False) -> Decision:
        """Gate for the files tools (and anything with a single path). Relative paths resolve against `cwd` (default: the sandbox)."""
        t = self.check_tool(tool_class)
        if not t:
            return t
        try:
            ab = self.resolve(path, cwd or str(self.sandbox))
        except _Deny as e:
            return Decision(False, e.reason, "builtin:path-form")
        return self.check_abs(ab, write, tool_class, delete)

    def _expand(self, w: Word, cwd: str) -> list[str]:
        """The literal paths a glob word names: its fixed prefix (checked as a path) plus what the shell would expand it to (bounded)."""
        res: list[str] = []
        for t in dict.fromkeys([w.text, w.alt]):
            pre = re.split(r"[*?\[]", t, 1)[0]
            if pre:
                res.append(pre)
            res.append(re.sub(r"[*?\[\]]", "x", t))                  # the pattern's own shape (`a*/../../x`) must stay inside too
            try:
                base = ntpath.join(cwd, t.replace("\\", "/")) if not (t.startswith(("/", "\\")) or ntpath.splitdrive(t)[0]) else t
                if pre and not self._inside(self.resolve(pre, cwd)):
                    continue
                res += sorted(_glob.glob(base))[:5000]
            except _Deny:
                continue
        return res

    # -------------------------------------------------------------- the command checker
    def check_command(self, command: str, cwd: Optional[str] = None, shell: str = "bash", tool_class: str = "shell") -> Decision:
        """Gate for one shell command line. Returns the deciding rule; any command that cannot be understood is denied."""
        d = self.check_tool(tool_class)
        if not d:
            return d
        try:
            self._cmd(command, [os.path.normpath(cwd or str(self.sandbox))], "cmd" if shell == "cmd" else "sh", 0)
        except _Deny as e:
            return Decision(False, e.reason, e.rule if e.rule != "builtin" else "builtin:command")
        return Decision(True, "", d.rule)

    def _paths(self, cwds: list[str], w: Word, write: bool, delete: bool = False, tc: str = "shell") -> None:
        for cwd in cwds:
            for cand in self._candidates(w, cwd):
                try:
                    ab = self.resolve(cand, cwd)
                except _Deny as e:
                    raise _Deny(e.reason, "builtin:path-form")
                d = self.check_abs(ab, write, tc, delete)
                if not d:
                    raise _Deny(f"{cand!r}: {d.reason}", d.rule)

    def _candidates(self, w: Word, cwd: str) -> list[str]:
        out: list[str] = []
        for t in dict.fromkeys([w.text, w.alt]):
            if not t or t == "-":
                continue
            if t.startswith("-"):
                if "=" in t:
                    v = t.split("=", 1)[1]
                    if v and (v[0] in "/\\~" or re.match(r"[A-Za-z]:", v) or ".." in v):
                        out.append(v)
                continue
            t = t.split("::", 1)[0]                                      # pytest node ids
            if w.glob:
                for hit in self._expand(Word(t, t, True), cwd):
                    out.append(hit)
                continue
            if PATHY.match(t) and not (" " in t and w.quoted and not re.match(r"^([A-Za-z]:)?[\\/]", t)):
                out.append(t)
            else:
                out += [m.group(1) for m in ESCAPE_RE.finditer(t)]
        return [o for o in out if o]

    def _cmd(self, command: str, cwds0: list[str], mode: str, depth: int) -> list[str]:
        if depth > _DEPTH:
            raise _Deny("commands nested too deeply")
        cwds = list(cwds0)
        segs = lex(command, mode)
        for si, seg in enumerate(segs):
            cwds = self._segment(seg, cwds, mode, depth, first=(si == 0 or seg.before not in ("|",)), piped=seg.before == "|" or seg.after == "|")
        return cwds

    def _segment(self, seg: Seg, cwds: list[str], mode: str, depth: int, first: bool, piped: bool) -> list[str]:
        words = list(seg.words)
        for op, tgt in seg.redirs:
            if op in (">", ">>") :
                self._paths(cwds, tgt, True)
            else:
                self._paths(cwds, tgt, False)
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0].text) and mode == "sh":
            name = words[0].text.split("=", 1)[0]
            if FORBIDDEN_ENV.match(name):
                raise _Deny(f"setting {name} is denied")
            words.pop(0)
        while words:                                                        # wrappers that only run the next command
            c = _cname(words[0].text)
            if c in ("time", "command", "builtin"):
                words.pop(0)
            elif c == "env":
                words.pop(0)
                while words and (words[0].text.startswith("-") or "=" in words[0].text):
                    nm = words.pop(0).text.split("=", 1)[0].lstrip("-")
                    if FORBIDDEN_ENV.match(nm) or nm in ("i", "u", "S", "C", "chdir"):
                        raise _Deny("env options and sensitive variables are denied")
            elif c == "nice":
                words.pop(0)
                while words and (words[0].text.startswith("-") or words[0].text.isdigit()):
                    words.pop(0)
            elif c == "timeout":
                words.pop(0)
                while words and words[0].text.startswith("-"):
                    words.pop(0)
                if words:
                    words.pop(0)
            else:
                break
        if not words:
            return cwds
        head = words[0]
        if head.text != head.alt or head.glob or head.text.startswith(("~", "-")):
            raise _Deny("the command word is escaped, globbed or malformed: denied when unsure")
        if re.search(r"[\\/]", head.text):
            raise _Deny("run commands by name (no path to an executable); scripts go through python or bash")
        name, args = _cname(head.text), words[1:]
        if name in DENY_WORDS:
            cls, why = DENY_WORDS[name]
            if cls == "kill":
                raise _Deny(why, "builtin:kill")
            raise _Deny(f"{name}: {why}", f"builtin:{cls}")
        if re.search(r"llama|gguf", name):
            raise _Deny("starting a model is denied (the machine-wide model lock owns that)", "builtin:model")
        if name in ("cd", "chdir", "pushd", "popd"):
            return self._cd(name, args, seg, cwds)
        if name in ("export", "unset"):
            for a in args:
                nm = a.text.split("=", 1)[0]
                if FORBIDDEN_ENV.match(nm):
                    raise _Deny(f"setting {nm} is denied")
            return cwds
        if name in ("kill", "taskkill", "stop-process"):
            self._kill(name, args, first and not piped)
            return cwds
        if name in ("powershell", "pwsh"):
            d = self.check_tool("powershell")
            if not d:
                raise _Deny(d.reason, d.rule)
            self._powershell(args, cwds, depth)
            return cwds
        if name in ("bash", "sh", "cmd", "dash", "zsh"):
            self._nested_shell(name, args, cwds, depth)
            return cwds
        if name in PY_WORDS:
            self._python(args, cwds)
            return cwds
        if name == "pytest" or name == "py.test":
            self._pytest(args, cwds)
            return cwds
        if name == "git":
            self._git(args, cwds)
            return cwds
        if name == "find":
            for a in args:
                if re.match(r"^-(exec|execdir|ok|okdir|fprint0?|fprintf|fls|delete)$", a.text):
                    raise _Deny(f"find {a.text} is denied (delete and run through plain commands)")
        if name == "awk" or name == "gawk":
            prog = " ".join(a.text for a in args if not a.text.startswith("-"))
            if re.search(r"system|getline|popen|\|\s*\"|>\s*\"|\bprint\s*>|@include", prog):
                raise _Deny("awk programs that run commands or write files are denied")
        writer = name in WRITERS
        if name in ("sed", "gsed"):
            script = " ".join(a.text for a in args if not a.text.startswith("-"))
            if re.search(r"(^|[;{}\n]\s*)[ew]\s|\bs(?P<d>.).*?(?P=d).*?(?P=d)[a-z0-9]*[ew]", script):
                raise _Deny("sed scripts that run commands or write files are denied")
            writer = any(re.match(r"^-[a-zA-Z]*i|^--in-place", a.text) for a in args)
        if name == "ruff":
            writer = any(a.text in ("--fix", "format") for a in args)
        if not (name in READERS or name in WRITERS or name in ("sed", "gsed", "find", "awk", "gawk", "ruff")):
            raise _Deny(f"{name!r} is not on the allowed command list", "builtin:allowlist")
        for a in args:
            self._paths(cwds, a, writer, delete=name in ("rm", "rmdir", "del", "erase", "rd", "mv"))
        return cwds

    def _cd(self, name: str, args: list[Word], seg: Seg, cwds: list[str]) -> list[str]:
        if name in ("pushd", "popd") or len(args) != 1 or args[0].text in ("-", "") or args[0].glob:
            raise _Deny("cd needs exactly one literal directory")
        if seg.before == "|" or seg.after == "|":
            raise _Deny("cd inside a pipeline is denied")
        new = []
        for cwd in cwds:
            t = args[0].text
            if t.lower().startswith("/d "):
                t = t[3:]
            ab = self.resolve(t, cwd)
            d = self.check_abs(ab, False, "shell")
            if not d:
                raise _Deny(f"cd {t!r}: {d.reason}", d.rule)
            new.append(ab)
        return new if seg.after == "&&" else list(dict.fromkeys(cwds + new))

    def _pids(self) -> set[int]:
        return {int(p) for p in self.tracked_pids()}

    def _kill(self, name: str, args: list[Word], first: bool) -> None:
        if not first:
            raise _Deny("a kill that takes its targets from a pipeline is denied")
        mine, pids = self._pids(), []
        txt = [a.text for a in args]
        if name == "kill":
            for k, t in enumerate(txt):
                if t.isdigit():
                    pids.append(int(t))
                elif t.startswith("-") and k > 0 and t[1:].isdigit():
                    raise _Deny("negative pids (process groups) are denied", "builtin:kill")
                elif t.startswith("-") and k == 0:
                    continue
                elif t.startswith("-s") or t in ("-l", "-L"):
                    continue
                elif not t.isdigit():
                    raise _Deny("kill takes numeric PIDs of your own jobs", "builtin:kill")
        elif name == "taskkill":
            i = 0
            while i < len(txt):
                t = txt[i].lower()
                if t in ("/pid", "-pid") and i + 1 < len(txt) and txt[i + 1].isdigit():
                    pids.append(int(txt[i + 1]))
                    i += 1
                elif t in ("/t", "/f", "-t", "-f"):
                    pass
                else:
                    raise _Deny("taskkill is allowed only as /PID <own pid> [/T] [/F]; never by image name or filter", "builtin:kill")
                i += 1
        else:
            i = 0
            while i < len(txt):
                t = txt[i].lower()
                if t in ("-id", "-processid") and i + 1 < len(txt) and txt[i + 1].isdigit():
                    pids.append(int(txt[i + 1]))
                    i += 1
                elif t in ("-force", "-confirm:$false", "-passthru"):
                    pass
                else:
                    raise _Deny("Stop-Process is allowed only with -Id <own pid>; never by name or pipeline", "builtin:kill")
                i += 1
        if not pids:
            raise _Deny("kill needs at least one PID", "builtin:kill")
        for p in pids:
            if p not in mine:
                raise _Deny(f"pid {p} is not in this package's own tracked process tree", "builtin:kill")

    def _nested_shell(self, name: str, args: list[Word], cwds: list[str], depth: int) -> None:
        txt = [a.text for a in args]
        if name == "cmd":
            for k, t in enumerate(txt):
                if t.lower() in ("/c", "/k", "/s"):
                    inner = " ".join(txt[k + 1:])
                    self._cmd(inner, cwds, "cmd", depth + 1)
                    return
            raise _Deny("cmd is allowed only as cmd /c \"<command>\"")
        if "-c" in txt:
            k = txt.index("-c")
            if k + 1 >= len(txt):
                raise _Deny("bash -c needs a command string")
            self._cmd(txt[k + 1], cwds, "sh", depth + 1)
            return
        files = [a for a in args if not a.text.startswith("-")]
        if len(files) != 1:
            raise _Deny("bash is allowed as `bash -c '<cmd>'` or `bash <script file in the sandbox>`")
        self._paths(cwds, files[0], False)
        for cwd in cwds:
            ab = self.resolve(files[0].text, cwd)
            try:
                src = Path(ab).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self._cmd(src, cwds, "sh", depth + 1)

    def _python(self, args: list[Word], cwds: list[str]) -> None:
        i = 0
        txt = [a.text for a in args]
        while i < len(txt) and txt[i].startswith("-") and txt[i] not in ("-c", "-m"):
            i += 1
        if i >= len(txt):
            raise _Deny("an interactive python session is denied; use python -c, -m or a script")
        if txt[i] == "-c":
            code = " ".join(txt[i + 1:i + 2])
            self._scan_py(code, cwds)
            return
        if txt[i] == "-m":
            mod = txt[i + 1] if i + 1 < len(txt) else ""
            if mod not in PY_MODULES:
                raise _Deny(f"python -m {mod}: only {', '.join(sorted(PY_MODULES)[:6])}... style test and lint modules are allowed", "builtin:python-module")
            if mod == "pytest":
                self._pytest(args[i + 2:], cwds)
            else:
                for a in args[i + 2:]:
                    self._paths(cwds, a, False)
            return
        script = args[i]
        self._paths(cwds, script, False)
        for a in args[i + 1:]:
            self._paths(cwds, a, False)
        for cwd in cwds:
            try:
                src = Path(self.resolve(script.text, cwd)).read_text(encoding="utf-8", errors="replace")[:300000]
            except (OSError, _Deny):
                continue
            self._scan_py(src, cwds)

    def _scan_py(self, code: str, cwds: list[str]) -> None:
        m = PY_BAD.search(code)
        if m:
            raise _Deny(f"python code using {m.group(1)!r} is denied (processes, network, dynamic import, models)", "builtin:python-code")
        for hit in ESCAPE_RE.finditer(code):
            for cwd in cwds:
                try:
                    ab = self.resolve(hit.group(1), cwd)
                except _Deny as e:
                    raise _Deny(f"python code names {hit.group(1)!r}: {e.reason}", "builtin:python-path")
                d = self.check_abs(ab, False)
                if not d:
                    raise _Deny(f"python code names {hit.group(1)!r}: {d.reason}", d.rule)

    def _pytest(self, args: list[Word], cwds: list[str]) -> None:
        for a in args:
            if "real_local_model" in a.text or a.text.startswith("--rootdir") or a.text in ("-p", "--basetemp", "--confcutdir"):
                raise _Deny("that pytest selection or option is denied (a real model test, or a path redirect)", "builtin:pytest")
            if a.text.startswith("--") and "=" in a.text and not a.text.startswith(("--tb=", "--maxfail=", "--durations=", "--timeout=")) \
                    and not re.search(r"^--(deselect|ignore|ignore-glob|junitxml|junit-xml)=", a.text):
                continue
            self._paths(cwds, a, False)

    def _git(self, args: list[Word], cwds: list[str]) -> None:
        txt = [a.text for a in args]
        i = 0
        while i < len(txt) and txt[i].startswith("-"):
            t = txt[i]
            if t == "-c" and i + 1 < len(txt):
                if GIT_BAD_C.match(txt[i + 1]):
                    raise _Deny(f"git -c {txt[i + 1].split('=')[0]} is denied (it can run programs or reach remotes)", "builtin:git")
                i += 2
            elif t == "-C" and i + 1 < len(txt):
                self._paths(cwds, args[i + 1], False)
                i += 2
            elif t in ("-p", "--no-pager", "--no-optional-locks", "--literal-pathspecs", "--paginate", "-P"):
                i += 1
            else:
                raise _Deny(f"git option {t} is denied", "builtin:git")
        if i >= len(txt):
            raise _Deny("git needs a subcommand", "builtin:git")
        sub, rest = txt[i], args[i + 1:]
        if sub in ("push", "fetch", "pull", "remote", "clone", "stash", "rebase", "submodule", "worktree", "filter-branch", "gc", "prune",
                   "ls-remote", "send-pack", "credential", "daemon", "update-ref", "symbolic-ref", "reflog", "fast-import", "bundle", "archive"):
            raise _Deny(f"git {sub} is denied (no remotes, no stash/rebase, no history rewriting by tools)", "builtin:git")
        if sub not in GIT_ALLOW:
            raise _Deny(f"git {sub} is not on the allowed list", "builtin:git")
        rt = [a.text for a in rest]
        if sub == "reset" and any(t in ("--hard", "--merge", "--keep") for t in rt):
            raise _Deny("git reset --hard is denied", "builtin:git")
        if sub == "config" and any(t in ("--global", "--system", "--file", "-f", "--worktree", "--add", "--unset", "--replace-all", "--edit", "-e")
                                   or t.startswith("--file=") for t in rt):
            raise _Deny("git config may only read the local configuration", "builtin:git")
        if sub == "config" and not any(t in ("--get", "--list", "-l", "--get-all", "--get-regexp") for t in rt):
            raise _Deny("git config may only read", "builtin:git")
        if sub in ("clean",) and not any(t in ("-n", "--dry-run") for t in rt):
            raise _Deny("git clean without -n is denied", "builtin:git")
        if sub == "commit" and any(t in ("--no-verify", "-n") for t in rt):
            raise _Deny("git commit --no-verify is denied", "builtin:git")
        for a in rest:
            if a.text.startswith(("--git-dir", "--work-tree", "--exec-path", "--upload-pack", "--receive-pack", "--output=")):
                raise _Deny("git options that redirect the repository or run programs are denied", "builtin:git")
            self._paths(cwds, a, sub in ("add", "mv", "rm", "checkout", "restore", "commit", "apply", "clean", "switch"))

    def _powershell(self, args: list[Word], cwds: list[str], depth: int) -> None:
        txt = [a.text for a in args]
        i, script = 0, None
        while i < len(txt):
            t = txt[i].lower()
            if t in ("-command", "-c") and i + 1 < len(txt):
                script = " ".join(txt[i + 1:])
                break
            if t in ("-noprofile", "-noninteractive", "-nologo", "-nop", "-noni"):
                i += 1
            elif t in ("-executionpolicy", "-ep") and i + 1 < len(txt):
                i += 2
            else:
                raise _Deny("powershell is allowed only as powershell -NoProfile -Command \"<allowed cmdlets>\"")
        if script is None:
            raise _Deny("powershell needs -Command")
        if re.search(r"[$`{}()@&<>]|\.\.\.", script):
            raise _Deny("powershell scripts may not use variables, sub-expressions, script blocks, call or redirect operators")
        allowed = set("""get-childitem gci ls dir get-content gc cat type select-string sls get-location pwd test-path measure-object measure
select-object select sort-object sort write-output echo get-date get-item gi resolve-path get-filehash split-path get-process stop-process""".split())
        pipes = [p for p in re.split(r"\|", script)]
        for k, part in enumerate(pipes):
            for sub in part.split(";"):
                toks = _ps_split(sub)
                if not toks:
                    continue
                cmd = toks[0].lower()
                if cmd not in allowed:
                    raise _Deny(f"powershell cmdlet {toks[0]!r} is not on the allowed list", "builtin:powershell")
                if cmd == "stop-process":
                    self._kill("stop-process", [Word(t) for t in toks[1:]], k == 0)
                    continue
                for t in toks[1:]:
                    if t.startswith("-"):
                        continue
                    self._paths(cwds, Word(t, t, bool(re.search(r"[*?\[]", t)), True), False, tc="powershell")

    # -------------------------------------------------------------- resource limits
    @staticmethod
    def clamp_timeout(t: Optional[float]) -> float:
        return DEFAULT_TIMEOUT_S if not t or t <= 0 else min(float(t), MAX_TIMEOUT_S)


def _ps_split(s: str) -> list[str]:
    return [t[1:-1] if len(t) > 1 and t[0] in "'\"" and t[-1] == t[0] else t for t in re.findall(r"""'[^']*'|"[^"]*"|\S+""", s)]


def _cname(t: str) -> str:
    t = t.lower()
    return re.sub(r"\.(exe|cmd|bat|com)$", "", t)


def machine_test_slot(command_text: str = "") -> Any:
    """The machine-wide test budget (creator.testslots): a test launch holds one slot while its process runs. The slot only ever makes
    a launch WAIT, never changes which tests run; if the budget cannot be built the launch proceeds unbudgeted (fail open)."""
    import contextlib
    try:
        from creator import testslots
        return testslots.Slot()
    except Exception:                                   # noqa: BLE001 - the budget is optional: fail open
        return contextlib.nullcontext()
