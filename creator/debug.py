"""K11 debug: failure reproduction, localisation, classification, hypotheses, evidence,
root cause, repair proposals, repair experiments.

Pure standard library; works on captured test/command output text.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict

# Failure taxonomy: class -> (regex over the failure text, remediation hint). First match wins.
TAXONOMY = {
    "SYNTAX": (r"SyntaxError|IndentationError|TabError", "fix the syntax at the reported line"),
    "IMPORT": (r"ModuleNotFoundError|ImportError|cannot import name", "fix the import path or add the missing module/name"),
    "TYPE": (r"TypeError", "check argument types, counts and None handling"),
    "NAME": (r"NameError|UnboundLocalError", "define or import the missing name"),
    "ATTRIBUTE": (r"AttributeError", "check the object's actual type and attribute names"),
    "KEY": (r"KeyError|IndexError", "guard the lookup or fix the key/index"),
    "VALUE": (r"ValueError|ZeroDivisionError|OverflowError", "validate the input range and edge cases"),
    "IO": (r"FileNotFoundError|PermissionError|OSError|IOError", "check paths, permissions and existence"),
    "TIMEOUT": (r"TimeoutExpired|Timeout|timed out", "find the slow or blocking step"),
    "ASSERTION": (r"AssertionError|(?m:^\s*(?:E\s+)?assert\s)", "the behaviour differs from the expectation; compare actual and expected"),
}

_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
_SHORT_FRAME = re.compile(r"^([\w./\\:-]+\.py):(\d+):", re.M)
_EXC_LINE = re.compile(r"^(?:E\s+)?([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)):?\s*(.*)$", re.M)


@dataclass
class Failure:
    """A reproduced failure: the command run, its exit code, and captured output."""
    command: list = field(default_factory=list)
    returncode: int = 0
    output: str = ""
    reproduced: bool = False


@dataclass
class Location:
    """A source location (file, line, optional function) implicated in a failure."""
    file: str
    line: int
    function: str = ""


@dataclass
class Diagnosis:
    """Result of diagnosing a failure: class, locations, hypotheses, evidence, root cause."""
    classification: str
    exception: str = ""
    message: str = ""
    locations: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    root_cause: str = ""

    def to_dict(self) -> dict:
        """Plain-dict (JSON-serialisable) form of the diagnosis."""
        return asdict(self)


def reproduce(command: list, cwd: str | None = None, timeout: float = 120.0) -> Failure:
    """Run command (argv list) and capture a Failure; reproduced is True iff it exits non-zero
    or times out. An empty command or an unlaunchable executable yields reproduced=True with
    returncode -1 and the error as output; it never raises.
    """
    if not command:
        return Failure([], -1, "empty command", True)
    try:
        p = subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        def text(x: object) -> str:
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x or "")
        out, err = text(e.stdout), text(e.stderr)
        return Failure(list(command), -1, out + err + "\nTimeoutExpired: timed out", True)
    except OSError as e:
        return Failure(list(command), -1, f"{type(e).__name__}: {e}", True)
    return Failure(list(command), p.returncode, (p.stdout or "") + (p.stderr or ""), p.returncode != 0)


def localise(output: str, prefer: str = "") -> list[Location]:
    """Extract source locations from traceback / pytest-short output, innermost (last) first.

    If prefer is non-empty, locations whose file contains it are ranked before others
    (order otherwise preserved). Empty or unparseable output yields [].
    """
    locs: list[Location] = []
    for m in _FRAME.finditer(output or ""):
        locs.append(Location(m.group(1), int(m.group(2)), m.group(3) or ""))
    if not locs:
        for m in _SHORT_FRAME.finditer(output or ""):
            locs.append(Location(m.group(1), int(m.group(2))))
    locs.reverse()
    if prefer:
        locs.sort(key=lambda l: prefer not in l.file)  # stable
    return locs


def classify(output: str) -> str:
    """Classify failure text into a TAXONOMY key, or 'UNKNOWN' if nothing matches/empty."""
    text = output or ""
    exc = _EXC_LINE.findall(text)
    probe = exc[-1][0] if exc else ""
    for name, (rx, _) in TAXONOMY.items():
        if probe and re.search(rx, probe):
            return name
    for name, (rx, _) in TAXONOMY.items():
        if re.search(rx, text):
            return name
    return "UNKNOWN"


def hypothesise(classification: str, locations: list | None = None, message: str = "") -> list[str]:
    """Ranked, testable hypotheses for a failure class; always at least one."""
    where = ""
    if locations:
        l = locations[0]
        where = f" at {l.file}:{l.line}"
    hints = TAXONOMY.get(classification)
    out = []
    if hints:
        out.append(f"{classification}{where}: {hints[1]}")
    if message:
        out.append(f"the condition behind '{message[:120]}' is not guaranteed by callers{where}")
    out.append(f"an earlier change broke an assumption{where}; compare against the last passing revision")
    return out


def diagnose(output: str, prefer: str = "") -> Diagnosis:
    """Diagnose failure output into a Diagnosis (classification, locations, hypotheses,
    evidence, root cause). Empty output gives classification 'UNKNOWN' with no locations and
    an explicit root_cause stating that there is no evidence; it never raises.
    """
    text = output or ""
    if not text.strip():
        return Diagnosis("UNKNOWN", hypotheses=hypothesise("UNKNOWN"),
                         root_cause="undetermined: no failure output to analyse")
    cls = classify(text)
    excs = _EXC_LINE.findall(text)
    exc, msg = excs[-1] if excs else ("", "")
    locs = localise(text, prefer)
    evidence = [f"exception: {exc}: {msg}".strip()] if exc else []
    evidence += [f"frame: {l.file}:{l.line}" + (f" in {l.function}" if l.function else "") for l in locs[:3]]
    if cls == "UNKNOWN":
        root = "undetermined: failure does not match a known class"
    elif locs:
        root = f"{cls} failure originating at {locs[0].file}:{locs[0].line}" + (f" ({exc})" if exc else "")
    else:
        root = f"{cls} failure (location unknown)"
    return Diagnosis(cls, exc, msg.strip(), locs, hypothesise(cls, locs, msg.strip()), evidence, root)


@dataclass
class RepairProposal:
    """A proposed repair: target location, description, and the hypothesis it tests."""
    target: str
    description: str
    hypothesis: str = ""
    rank: int = 0


def propose_repairs(diagnosis: Diagnosis) -> list[RepairProposal]:
    """One RepairProposal per hypothesis, ranked from 0 (most likely). A diagnosis without
    hypotheses yields []. Targets are 'file:line' of the primary location, or '' if unknown.
    """
    target = ""
    if diagnosis.locations:
        l = diagnosis.locations[0]
        target = f"{l.file}:{l.line}"
    hint = TAXONOMY.get(diagnosis.classification, ("", "investigate the failure"))[1]
    return [RepairProposal(target, hint if i == 0 else h, h, i) for i, h in enumerate(diagnosis.hypotheses)]


def repair_experiment(command: list, apply_fn, revert_fn=None, cwd: str | None = None,
                      timeout: float = 120.0) -> dict:
    """Run a repair experiment: reproduce before, apply_fn(), reproduce after.

    Returns {'before','after','fixed','error'}. fixed is True only if the failure reproduced
    before and not after. If apply_fn raises, the error is recorded, fixed is False and
    revert_fn (if given) is still called. If the repair did not fix it, revert_fn is called.
    """
    before = reproduce(command, cwd, timeout)
    result: dict = {"before": before, "after": None, "fixed": False, "error": ""}
    try:
        apply_fn()
        result["after"] = reproduce(command, cwd, timeout)
        result["fixed"] = before.reproduced and not result["after"].reproduced
    except Exception as e:  # recorded as evidence, not swallowed silently
        result["error"] = f"{type(e).__name__}: {e}"
    if not result["fixed"] and revert_fn is not None:
        revert_fn()
    return result


def python_cmd(*args: str) -> list:
    """argv list running the current interpreter with args (portable helper for reproduce)."""
    return [sys.executable, *args]
