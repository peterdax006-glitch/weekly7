"""P0.9 real slot actors for creator.team: the team's slots answered by merged GGUFs on llama.cpp, the tools by the code tools.

    runner = SlotRunner(default_specs(), workdir)           # ONE llama-server at a time (BelowNormal), swapped by use()
    team = MeasuredTeam(state, actors(runner), goal_id="g1", sink=collect)
    team.ctx = ToolContext(...)                              # the workspace the tool actors work in

Model steps (STEP profile = slot, output form -> grammar, max_tokens, max_s):
    SPEC, PLAN   THINKER  base Qwen3-1.7B until a THINKER slot exists      forms spec / plan
    CODE         CODER    CODER-1.7B                                       form diff (SEARCH/REPLACE blocks)
    DEBUG_FIX    CODER                                                     form debugfix (CAUSE / FIX blocks)
    CONFIDENCE   CHECKER  CHECKER-1.7B-calib                               form confidence ('CONFIDENCE: 0.xx')
Code tools stay code: LOCATE / PACK = creator.tools.index, edits = creator.tools.patch (through creator.quickloop), VALIDATE =
creator.quickloop.quick_cycle, DEBUG_PINPOINT = creator.tools.pinpoint, SAFETY = creator.tools.policy.check_diff.
Every envelope's budget {max_tok, max_s} caps the model call: n_predict = min(profile, budget) and the HTTP timeout = min(profile, budget).
Grammars live in creator/grammars/<form>.gbnf. Light imports at module level; the heavy tools load inside the actors."""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import time
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from creator import modelpool as MP
from creator import team as TM

GRAMMAR_DIR = Path(__file__).resolve().parent / "grammars"
FORMS = ("diff", "diff1", "fnbody", "lineedit", "debugfix", "confidence", "verdict", "fileline", "spec", "plan")
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000


# ------------------------------------------------------------------------------------------------------ profiles and slot specs
@dataclass(frozen=True)
class StepProfile:
    slot: str            # which slot's model answers the step
    form: str            # output form = grammar file
    role: str            # role tag in the system prompt (creator.pipelinemix.tag)
    max_tokens: int      # per-role output cap
    max_s: float         # wall cap of the call (also capped by the envelope budget)


PROFILES: dict[str, StepProfile] = {
    "SPEC": StepProfile("THINKER", "spec", "SPEC", 256, 150.0),
    "PLAN": StepProfile("THINKER", "plan", "PLAN", 192, 120.0),
    "CODE": StepProfile("CODER", "diff", "CODE", 512, 420.0),
    "DEBUG_FIX": StepProfile("CODER", "debugfix", "DEBUG", 512, 420.0),
    "CONFIDENCE": StepProfile("CHECKER", "confidence", "CALIB", 12, 150.0),
}
CALIB_SYSTEM = "[ROLE: CALIB] You are Nupen's self-check: estimate honestly how likely a proposed answer is right."


def grammars() -> dict[str, Path]:
    return {f: GRAMMAR_DIR / f"{f}.gbnf" for f in FORMS}


def default_specs(runtime: Optional[Path] = None, threads: int = 6) -> dict[str, MP.SlotSpec]:
    """Slot specs of this PC: the merged slot GGUFs (slots/manifest.json) and the base model as the THINKER until a THINKER slot exists."""
    from creator import generator as G
    rt = Path(runtime) if runtime else G.RUNTIME
    slots = rt / "models" / "slots"
    g = grammars()
    mk = lambda name, f, ctx, mt: MP.SlotSpec(name, slots / f, threads=threads, ctx=ctx, max_tokens=mt, grammars=g)      # noqa: E731
    return {"CODER": mk("CODER", "CODER-1.7B.gguf", 8192, 512), "CHECKER": mk("CHECKER", "CHECKER-1.7B-calib.gguf", 4096, 16),
            "CHECKER_SMALL": mk("CHECKER_SMALL", "CHECKER-0.6B-judge.gguf", 2048, 16),
            "CODER06": MP.SlotSpec("CODER06", rt / "models" / "Qwen3-0.6B-Q4_K_M.gguf", threads=threads, ctx=4096, max_tokens=512, grammars=g),
            "THINKER": MP.SlotSpec("THINKER", rt / "models" / "Qwen3-1.7B-Q4_K_M.gguf", threads=threads, ctx=4096, max_tokens=224, grammars=g)}


MOE_FILE = "Qwen3-30B-A3B-Q3_K_M.gguf"


def moe_specs(specs: dict[str, MP.SlotSpec], runtime: Optional[Path] = None, threads: int = 6) -> dict[str, MP.SlotSpec]:
    """R10: the same slots with CODER served by the Qwen3-30B-A3B mixture-of-experts GGUF (~3B parameters active per token). Same prompts,
    grammars and prefills; the model is the stock instruct model (no adapter), the prompt keeps the empty think block (= /no_think)."""
    from creator import generator as G
    rt = Path(runtime) if runtime else G.RUNTIME
    out = dict(specs)
    out["CODER"] = MP.SlotSpec("CODER", rt / "models" / MOE_FILE, threads=threads, ctx=8192, max_tokens=512, grammars=grammars())
    return out


def chatml(system: str, user: str) -> str:
    """Qwen3 chat prompt with the empty think block (what the adapters were trained on: the last assistant turn renders <think></think>)."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ------------------------------------------------------------------------------------------------------ one server at a time
def _spawn_below_normal(cmd: list[str]) -> Any:
    flags = (BELOW_NORMAL | NO_WINDOW) if sys.platform == "win32" else 0
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)


_JOB: Any = None


def _job_cpu_s() -> Optional[float]:
    """CPU seconds of every process of this run (this process, the llama-server, pytest children - also the finished ones) from a Windows Job
    Object this process is assigned to on first use. None where that is unavailable."""
    global _JOB
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32                                                  # type: ignore[attr-defined]
    if _JOB is None:
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        h = k32.CreateJobObjectW(None, None)
        _JOB = h if (h and k32.AssignProcessToJobObject(h, k32.GetCurrentProcess())) else False
    if not _JOB:
        return None

    class Acc(ctypes.Structure):
        _fields_ = [("user", ctypes.c_int64), ("kernel", ctypes.c_int64), ("pu", ctypes.c_int64), ("pk", ctypes.c_int64),
                    ("faults", wintypes.DWORD), ("total", wintypes.DWORD), ("active", wintypes.DWORD), ("term", wintypes.DWORD)]
    acc = Acc()
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    if not k32.QueryInformationJobObject(_JOB, 1, ctypes.byref(acc), ctypes.sizeof(acc), None):
        return None
    return (acc.user + acc.kernel) / 1e7


def tree_cpu_s() -> float:
    """CPU seconds consumed by this run so far (job accounting; else this process + its live children). The delta around a step is the CPU
    the step cost, independent of what else the machine is doing."""
    v = _job_cpu_s()
    if v is not None:
        return v
    import psutil
    me = psutil.Process()
    t = me.cpu_times()
    tot = t.user + t.system
    for c in me.children(recursive=True):
        try:
            ct = c.cpu_times()
            tot += ct.user + ct.system
        except psutil.Error:
            pass
    return float(tot)


def busy_cpu_s() -> float:
    """Machine-wide busy CPU seconds so far (all cores, user + system), noise from other programs included."""
    import psutil
    t = psutil.cpu_times()
    return float(t.user + t.system + getattr(t, "interrupt", 0.0) + getattr(t, "dpc", 0.0))


class SlotRunner:
    """Serves the slot models ONE llama-server at a time through creator.modelpool.slot_pool; use() swaps the server when the slot changes.
    complete() returns the llama-server answer with its token counts and prompt / generation seconds. Timeouts and server errors come back
    as an empty content with `error` set (never raised): the caller decides."""

    def __init__(self, specs: dict[str, MP.SlotSpec], workdir: Path, spawn: Callable[[list[str]], Any] = _spawn_below_normal) -> None:
        self.specs, self.work, self.spawn = specs, Path(workdir), spawn
        self.work.mkdir(parents=True, exist_ok=True)
        self.cur: Optional[str] = None
        self.pool: Optional[MP.ModelPool] = None
        self.port = 0
        self.pid = 0
        self.loads: list[dict[str, Any]] = []

    def use(self, slot: str, load_timeout: float = 240.0) -> int:
        if self.cur == slot and self.pool is not None and self.pool.servers:
            return self.port
        self.stop()
        spec = self.specs[slot]
        t0, c0 = time.perf_counter(), tree_cpu_s()
        pool = MP.slot_pool(spec, base_pidfile=self.work / "llama_server.pid", slots=1, max_servers=1, floor_gb=0.5, headroom_gb=0.5,
                            spawn=self.spawn)
        pool.tick()
        while not pool.servers and time.perf_counter() - t0 < load_timeout:
            if pool.loading is None and not pool.servers:
                if pool.log and pool.log[-1].get("event") == "failed":
                    break
                pool.last_tick = -1e9
                pool.tick()
            time.sleep(0.2)
        if not pool.servers:
            err = pool.log[-1] if pool.log else {}
            pool.close()
            raise RuntimeError(f"slot {slot} server did not start: {err}")
        v = next(iter(pool.servers.values()))
        self.pool, self.cur, self.port, self.pid = pool, slot, int(v["port"]), int(v["proc"].pid)
        self.loads.append({"slot": slot, "load_s": round(time.perf_counter() - t0, 2), "cpu_s": round(tree_cpu_s() - c0, 2)})
        return self.port

    def stop(self) -> None:
        if self.pool is not None:
            self.pool.close()
        self.pool, self.cur, self.port, self.pid = None, None, 0, 0

    def complete(self, slot: str, prompt: str, form: Optional[str], max_tokens: int, timeout: float, guard: Optional[Callable[[str], bool]] = None,
                 extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        port = self.use(slot)
        spec = self.specs[slot]
        t0 = time.perf_counter()
        out: dict[str, Any] = {"content": "", "slot": slot, "form": form, "max_tokens": max_tokens}
        try:
            raw = MP.slot_call(port, spec, prompt, form, max_tokens, timeout, guard=guard, extra=extra)
            tm = raw.get("timings") or {}
            out.update(content=str(raw.get("content", "")), in_tok=int(raw.get("tokens_evaluated") or 0), out_tok=int(raw.get("tokens_predicted") or 0),
                       cached_tok=int(raw.get("tokens_cached") or 0), prompt_n=int(tm.get("prompt_n") or 0), prompt_s=float(tm.get("prompt_ms") or 0) / 1000,
                       gen_s=float(tm.get("predicted_ms") or 0) / 1000, stop=str(raw.get("stop_type") or ""),
                       truncated=bool(raw.get("truncated")) or str(raw.get("stop_type")) in ("limit", "loop"), looped=str(raw.get("stop_type")) == "loop")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            out["error"] = type(e).__name__
        out["wall_s"] = time.perf_counter() - t0
        return out


# ------------------------------------------------------------------------------------------------------ the workspace the tools work in
_FILE_LINE = re.compile(r"^\s*(?:FILE|PATH):\s*`?([^\s`]+)`?\s*$", re.I)
_OPEN = re.compile(r"^<{5,9} SEARCH\s*$")
_CLOSE = re.compile(r"^>{5,9} REPLACE\s*$")


def _is_test_path(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    return low.startswith("tests/") or "/tests/" in low or low.rsplit("/", 1)[-1].startswith("test_")


def canonical_patch(text: str, default_path: str = "", drop_tests: bool = True) -> tuple[str, int]:
    """The model's edit text as the plain SEARCH/REPLACE form creator.tools.patch reads (path line directly above the marker): 'FILE: x' lines
    become the path; CAUSE / EVIDENCE / FIX / EXPECT lines and code fences outside blocks are dropped; blocks aimed at tests are dropped
    (never weaken a test; the second value counts them). The patch tool itself stays the strict applier."""
    out: list[str] = []
    lines = text.replace("\r\n", "\n").split("\n")
    path, i, dropped = default_path, 0, 0
    while i < len(lines):
        m = _FILE_LINE.match(lines[i])
        if m:
            path = m.group(1)
        elif _OPEN.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _CLOSE.match(lines[j]):
                j += 1
            if drop_tests and _is_test_path(path):
                dropped += 1
            else:
                out += ([path] if path else []) + lines[i:j + 1]        # an unterminated block is passed on: the patch tool reports it
            i = j
        i += 1
    return "\n".join(out) + "\n", dropped



class WsFiles:
    """FileTools-shaped patch applier confined to one workspace; tests/ is read-only (never weaken a test)."""

    def __init__(self, ws: Path) -> None:
        self.ws = Path(ws).resolve()
        self.journal = self.ws.parent / (".journal_" + self.ws.name)

    def resolve(self, rel: str) -> tuple[Optional[dict[str, Any]], str]:
        p = (self.ws / rel).resolve()
        try:
            r = p.relative_to(self.ws).as_posix()
        except ValueError:
            return {"ok": False, "denied": "outside the workspace", "rule": "sandbox"}, ""
        if r.startswith("tests/") or r.split("/")[-1].startswith("test_"):
            return {"ok": False, "denied": "tests are read-only", "rule": "tests"}, ""
        return None, str(p)

    def apply_patch(self, patch: str, default_path: str = "") -> dict[str, Any]:
        from creator.tools import patch as PT
        try:
            hunks = PT.parse(patch, default_path)
        except PT.PatchError as e:
            return {"ok": False, "error": str(e)}
        return PT.apply_to_files(hunks, self.resolve, self.journal)


@dataclass
class ToolContext:
    ws: Path
    ix: Any = None
    default_path: str = "solution.py"
    orig: dict[str, str] = field(default_factory=dict)          # rel -> text before any change (for unified diffs)
    python: str = sys.executable
    test_timeout: float = 120.0
    k: int = 3
    pack_tokens: int = 1200
    files: Any = None

    def __post_init__(self) -> None:
        self.ws = Path(self.ws).resolve()
        self.files = self.files or WsFiles(self.ws)

    def snapshot(self) -> None:
        self.orig = {p.relative_to(self.ws).as_posix(): p.read_text(encoding="utf-8", errors="replace")
                     for p in self.ws.rglob("*.py") if "__pycache__" not in p.parts and ".git" not in p.parts}

    def unified_diff(self) -> str:
        out: list[str] = []
        for p in sorted(self.ws.rglob("*.py")):
            rel = p.relative_to(self.ws).as_posix()
            if "__pycache__" in p.parts or rel.startswith("tests/"):
                continue
            new = p.read_text(encoding="utf-8", errors="replace")
            old = self.orig.get(rel, "")
            if new != old:
                out.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
        return "".join(out)


# ------------------------------------------------------------------------------------------------------ prompts
EDIT_FORMAT = ("Edit format: one or more blocks exactly like\n<<<<<<< SEARCH\n<exact lines of {f}>\n=======\n<new lines>\n>>>>>>> REPLACE")
EDIT_FORMAT_FILES = ("Edit format: for each change one block exactly like\nFILE: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n"
                     "<new lines>\n>>>>>>> REPLACE")


def _of_kind(team: Any, env: TM.Envelope, kind: str) -> str:
    for ref in reversed(env.inputs):
        if TM.input_kind(ref) == "board":
            row = team.board.get(ref)
            if row and row["kind"] == kind:
                return str(row["body"])
    return ""


def _task(team: Any, env: TM.Envelope) -> dict[str, Any]:
    t = _of_kind(team, env, "task")
    return json.loads(t) if t else {}


def _request(t: dict[str, Any]) -> str:
    ex = f"\n\nExamples:\n{t['examples']}" if t.get("examples") else ""
    return f"{t.get('request', '')}{ex}"


def code_user(t: dict[str, Any], plan: str, pack: str) -> str:
    """The CODE-role user prompt (the format the CODER adapter was trained on)."""
    fn = t.get("family") == "fn"
    cur = (f"\n\nCurrent solution.py:\n```python\n{t.get('stub', '')}\n```\n" + EDIT_FORMAT.format(f="solution.py")) if fn \
        else (f"\n\nRelevant code:\n{pack}\n" + EDIT_FORMAT_FILES)
    return f"ROLE: CODE\nPlan:\n{plan.strip()}\n\nRequest:\n{_request(t)}{cur}"


NL = chr(10)
FENCE = chr(96) * 3
EX_LINES, EX_CHARS = 4, 160
LEAN = False                                      # R4 variant: the stub once in the prompt + a one-line SEARCH prefill (instead of the stub twice in the prefill)
DUP_STUB = False                                  # the prefilled SEARCH half already holds the stub: do not send it a second time
FAIL_CHARS = 500
FAST_PLAN ="Implement the function body below the signature; keep the signature and docstring."
CODE_CAP = {"fn": 140, "app": 420}               # per-family output caps from the measured lengths (baseline: 4 of 33 CODE calls hit 512)
MAX_FAST_DEBUG = 0                                  # measured 0 fixes in 13 gated attempts (2 fast runs); the gate code stays (debug_worthwhile), raise to re-enable
STUB_LAST = "    raise NotImplementedError"
CODE_CAP_CLASS: dict[str, int] = {}               # router class (fn_s ... app_l) -> output cap measured from the class's completed-output distribution (h94)
PREFIX_FIRST = False                              # h94 R4: the fixed text (role, plan, edit format) first, everything task-specific last -> longer KV-cache prefix shared by tasks
COUPLED_SLOT = "slot:"                            # an envelope constraint "slot:CODER06" sends a CODE step to another slot (router, R5); it is also part of the cache key


def _fast_request(t: dict[str, Any]) -> str:
    """Function task: the stub already holds the signature + docstring, so the request is one line plus the examples."""
    exs = [x[:EX_CHARS] for x in (t.get("examples") or "").splitlines()[:EX_LINES]]
    ex = (NL * 2 + "Examples:" + NL + NL.join(exs)) if exs else ""
    return f"Write the Python function `{t.get('name', '')}` described in the docstring (standard library only).{ex}"


def fast_prefill(t: dict[str, Any]) -> str:
    """The SEARCH half of a function task's edit block and the start of its REPLACE half (signature + docstring, unchanged) are known from the
    stub: they are fed as the start of the answer (prompt tokens read ~4x faster than tokens are generated), so the model generates only the
    new body. One-line SEARCH prefills made the model copy the stub line, so the whole function is the SEARCH."""
    stub = (t.get("stub") or "").rstrip()
    m = re.search(r"^(?:async )?def .*$", stub, re.M)
    if t.get("family") != "fn" or not m or not stub.endswith(STUB_LAST.strip()):
        return ""
    if LEAN:
        return f"<<<<<<< SEARCH{NL}{STUB_LAST}{NL}======={NL}"
    func = stub[m.start():]
    head = func[:len(func) - len(STUB_LAST.lstrip(NL))].rstrip() if func.endswith(STUB_LAST.strip()) else func
    head = func[:func.rindex(STUB_LAST.strip())].rstrip(" ").rstrip(NL)
    return f"<<<<<<< SEARCH{NL}{func}{NL}======={NL}{head}{NL}"


def fast_code_user(t: dict[str, Any], pack: str, hint: str = "") -> str:
    fn = t.get("family") == "fn"
    if fn:
        cur = (f"{NL}{NL}Current solution.py:{NL}{FENCE}python{NL}{t.get('stub', '')}{NL}{FENCE}{NL}" if (DUP_STUB or LEAN) else NL * 2)             + EDIT_FORMAT.format(f="solution.py")
    else:
        cur = f"{NL}{NL}Relevant code:{NL}{pack}{NL}" + EDIT_FORMAT_FILES
    if hint:                                                                  # R2 regenerate: the failing visible example rides along
        cur = f"{NL}{NL}{hint}" + cur
    if PREFIX_FIRST and fn and not hint:
        return f"ROLE: CODE{NL}Plan:{NL}{FAST_PLAN}{NL}{NL}{EDIT_FORMAT.format(f='solution.py')}{NL}{NL}Request:{NL}{_fast_request(t)}"
    return f"ROLE: CODE{NL}Plan:{NL}{FAST_PLAN}{NL}{NL}Request:{NL}{_fast_request(t) if fn else _request(t)}{cur}"


def diff_stats(diff: str) -> tuple[int, int]:
    return (sum(1 for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++")),
            sum(1 for x in diff.splitlines() if x.startswith("-") and not x.startswith("---")))


def confidence_summary(t: dict[str, Any], diff: str, outcome: str, pin: str, rounds: int) -> str:
    """<= 120 tokens: what the checker needs (diff stats, test outcome, top suspect), never the code or the prompt."""
    a, d = diff_stats(diff)
    return (f"task {t.get('name') or t.get('id', '')}; edit +{a}/-{d} lines; visible tests {outcome[:60]}; fix rounds {rounds}"
            + (f"; suspect {pin.splitlines()[0][:60]}" if pin else ""))


def build_prompt(step: str, env: TM.Envelope, team: Any) -> tuple[str, str]:
    """(system, user) of a model step, from board facts only."""
    from creator import pipelinemix as PM
    t = _task(team, env)
    prof = PROFILES[step]
    system = CALIB_SYSTEM if prof.role == "CALIB" else PM.tag(prof.role)
    if step == "SPEC":
        stub = f"\n\nStarting code (solution.py):\n```python\n{t['stub']}\n```" if t.get("stub") else ""
        return system, (f"ROLE: SPEC\nRequest:\n{_request(t)}{stub}\n"
                        "Write the acceptance tests as pytest functions that import nothing but the function under test (it is already imported).")
    if step == "PLAN":
        spec, pack = _of_kind(team, env, "spec"), _of_kind(team, env, "pack")
        ctx = f"\n\nCurrent solution.py:\n```python\n{t['stub']}\n```" if t.get("stub") else f"\n\nRelevant code:\n{pack}"
        return system, f"ROLE: PLAN\nSPEC:\n{spec.strip()}\n\nRequest:\n{_request(t)}{ctx}"
    if step == "CODE" and getattr(team, "fast", False):
        return system, fast_code_user(t, _of_kind(team, env, "pack"), _of_kind(team, env, "hint"))
    if step == "CODE":
        return system, code_user(t, _of_kind(team, env, "plan"), _of_kind(team, env, "pack"))
    if step == "DEBUG_FIX" and _of_kind(team, env, "target"):                  # R2 line repair: only the pinpointed line is edited
        from creator import repair as RP
        tg = json.loads(_of_kind(team, env, "target"))
        return system, RP.line_prompt(team.ctx.ws, tg["file"], tg["line"], tg["text"], tg["fail"], t.get("family") == "fn")
    if step == "DEBUG_FIX" and getattr(team, "fast", False):
        fail, pin, cur = _of_kind(team, env, "failure"), _of_kind(team, env, "pinpoint"), _of_kind(team, env, "code")
        fn = t.get("family") == "fn"
        return system, (f"ROLE: DEBUG{NL}" + (f"Current solution.py:{NL}{FENCE}python{NL}{cur}{NL}{FENCE}{NL}" if fn else f"Current code:{NL}{cur}{NL}")
                        + (f"{NL}Suspect lines:{NL}{pin}{NL}" if pin else "") + f"{NL}Exact failing output:{NL}{fail[:FAIL_CHARS]}{NL}"
                        + (EDIT_FORMAT.format(f="solution.py") if fn else EDIT_FORMAT_FILES))
    if step == "DEBUG_FIX":
        fail, pin, cur = _of_kind(team, env, "failure"), _of_kind(team, env, "pinpoint"), _of_kind(team, env, "code")
        fn = t.get("family") == "fn"
        return system, (f"ROLE: DEBUG\nRequest (expected behaviour):\n{_request(t)}\n\n"
                        + (f"Current solution.py:\n```python\n{cur}\n```\n" if fn else f"Current code:\n{cur}\n")
                        + (f"\nSuspect lines:\n{pin}\n" if pin else "") + f"\nExact failing output:\n{fail[:1500]}\n"
                        + (EDIT_FORMAT.format(f="solution.py") if fn else EDIT_FORMAT_FILES))
    if step == "CONFIDENCE" and getattr(team, "fast", False):
        from creator import trainmods as TMODS
        brief = (t.get("name") and f"Function `{t['name']}`.") or str(t.get("request", ""))[:160]
        return system, f"ROLE OF THE ANSWER: CODE\n{brief}" + TMODS.CALIB_ASK.format(answer=_of_kind(team, env, "answer").strip())
    if step == "CONFIDENCE":
        from creator import trainmods as TMODS
        user = code_user(t, _of_kind(team, env, "plan"), _of_kind(team, env, "pack"))
        return system, f"ROLE OF THE ANSWER: CODE\n{user}" + TMODS.CALIB_ASK.format(answer=_of_kind(team, env, "answer").strip())
    raise TM.Refused(f"no model profile for step {step!r}")


# ------------------------------------------------------------------------------------------------------ actors
LOOP_STEPS = ("CODE", "DEBUG_FIX")


def slot_actor(name: str, runner: SlotRunner) -> TM.Actor:
    def fn(env: TM.Envelope, team: Any) -> str:
        prof = PROFILES.get(env.step)
        if prof is None:
            raise TM.Refused(f"no model profile for step {env.step!r}")
        system, user = build_prompt(env.step, env, team)
        prompt, form, pre, cap = chatml(system, user), prof.form, "", min(prof.max_tokens, int(env.budget["max_tok"]))
        if getattr(team, "fast", False) and env.step == "CODE":
            pre = fast_prefill(_task(team, env))
            form = "fnbody" if pre else "diff1"
            prompt += pre
        tg = _of_kind(team, env, "target") if env.step == "DEBUG_FIX" else ""
        if tg:                                                                # R2 line repair: tiny prefilled, grammar-bound answer
            from creator import repair as RP
            t_ = json.loads(tg)
            pre = RP.line_prefill(team.ctx.ws, t_["file"], t_["line"], t_.get("with_file", False))[0]
            form, cap = "lineedit", min(cap, RP.LINE_CAP)
            prompt += pre
        guard = None
        if getattr(team, "fast", False) and env.step in LOOP_STEPS:
            from creator import repair as RP
            guard = RP.looping
        wall = min(prof.max_s, float(env.budget["max_s"]))
        kw: dict[str, Any] = {"guard": guard} if guard is not None else {}
        if env.step == "CODE":
            from creator import repair as RP
            ex = RP.attempt_extra(_of_kind(team, env, "hint"))
            if ex:
                kw["extra"] = ex
        slot = next((c[len(COUPLED_SLOT):] for c in env.constraints if c.startswith(COUPLED_SLOT)), prof.slot)
        res = runner.complete(slot, prompt, form, cap, wall, **kw)
        if guard is not None and res.get("looped") and not res.get("error"):
            # R4 retry policy: a greedy loop would loop again, so ONE retry with a repeat penalty and a tighter cap; a second loop gives up
            extra, cap2 = RP.retry_extra(cap)
            first = res
            res = runner.complete(slot, prompt, form, cap2, wall, guard=guard, extra=extra)
            for k in ("in_tok", "out_tok", "prompt_s", "gen_s", "wall_s"):
                res[k] = res.get(k, 0) + first.get(k, 0)
            res["retried"], res["first_out_tok"], res["first_looped"] = 1, first.get("out_tok", 0), bool(first.get("looped"))
        team.last_call = res
        return pre + str(res["content"])
    return TM.actor(name, fn)


def _query(team: Any, env: TM.Envelope) -> str:
    return str(_task(team, env).get("query") or _task(team, env).get("request") or "")


def locate_actor() -> TM.Actor:
    """LOCATE = creator.tools.index: file:a-b ranges, one per line. DEBUG_PINPOINT = creator.tools.pinpoint (spectrum + traceback)."""
    def fn(env: TM.Envelope, team: Any) -> str:
        c: ToolContext = team.ctx
        if env.step == "DEBUG_PINPOINT":
            return _pinpoint(team, env, c)
        t0 = time.perf_counter()
        c.ix.update()
        team.last_call = {"index_update_s": round(time.perf_counter() - t0, 4)}
        hits = c.ix.locate(_query(team, env), k=c.k)
        c.hits = hits                                                       # type: ignore[attr-defined]
        return "\n".join(f"{h.path}:{h.line}-{h.end}" for h in hits) or "none"
    return TM.actor("locate", fn, accepts=("LOCATE", "DEBUG_PINPOINT"))


_HEAD = re.compile(r"^### (\S+?):(\d+)-(\d+) \(.*\)$", re.M)


def _fenced(pack: str) -> str:
    """index.pack blocks as the 'FILE: path (lines a-b)' + fenced code the CODER was trained on (a bare '###' header got copied into edits)."""
    parts = _HEAD.split(pack)
    out = []
    for i in range(1, len(parts), 4):
        path, a, b, body = parts[i], parts[i + 1], parts[i + 2], parts[i + 3].strip(NL)
        out.append(f"FILE: {path} (lines {a}-{b}){NL}{FENCE}{NL}{body}{NL}{FENCE}")
    return (NL * 2).join(out)


def pack_actor() -> TM.Actor:
    def fn(env: TM.Envelope, team: Any) -> str:
        c: ToolContext = team.ctx
        if getattr(c, "hits", None) is None:
            c.hits = c.ix.locate(_query(team, env), k=c.k)                 # type: ignore[attr-defined]
        return _fenced(str(c.ix.pack(c.hits, max_tokens=c.pack_tokens))) or "(nothing located)"
    return TM.actor("pack", fn)


def _preview_diff(hunks: list[Any], ctx: ToolContext) -> str:
    """Unified diff of a SEARCH/REPLACE patch computed in memory (nothing written): what SAFETY judges before the patch is applied."""
    from creator.tools import patch as PT
    out: list[str] = []
    by: dict[str, list[Any]] = {}
    for h in hunks:
        by.setdefault(h[0], []).append(h)
    for rel, hs in by.items():
        p = ctx.ws / rel
        old = p.read_text(encoding="utf-8") if p.is_file() else ""
        lines = old.split("\n") if p.is_file() else []
        for i, h in enumerate(hs, 1):
            lines, _ = PT.apply_hunk(lines, h, i)
        out.extend(difflib.unified_diff(old.splitlines(True), "\n".join(lines).splitlines(True), f"a/{rel}", f"b/{rel}"))
    return "".join(out)


def safety_actor() -> TM.Actor:
    """SAFETY = creator.tools.policy.check_diff on the previewed diff of the patch. Returns 'OK' or 'BLOCK: <reasons>'."""
    def fn(env: TM.Envelope, team: Any) -> str:
        from creator.tools import patch as PT
        from creator.tools import policy as PO
        c: ToolContext = team.ctx
        try:
            hunks = PT.parse(canonical_patch(_of_kind(team, env, "diff"), c.default_path)[0], c.default_path)
            for h in hunks:
                den, _ = c.files.resolve(h[0])
                if den:
                    return f"BLOCK: {den['denied']} ({h[0]})"
            v = PO.check_diff(_preview_diff(hunks, c))
        except Exception as e:                                                # noqa: BLE001 - an unpreviewable patch is reported by VALIDATE
            return f"OK (not previewed: {type(e).__name__})"
        return ("BLOCK: " if v.verdict == "block" else "OK") + ("; ".join(v.reasons)[:200] if v.verdict != "ok" else "")
    return TM.actor("safety", fn)


def _failure_text(junit: Path, limit: int = 1800) -> str:
    try:
        root = ET.parse(str(junit)).getroot()
    except (OSError, ET.ParseError):
        return ""
    out = []
    for tc in root.iter("testcase"):
        for kind in ("failure", "error"):
            f = tc.find(kind)
            if f is not None:
                out.append(f"FAIL {tc.get('classname', '')}.{tc.get('name', '')}: {(f.get('message') or '')[:300]}\n{(f.text or '')[-500:]}")
    return "\n".join(out)[:limit]


def tests_actor() -> TM.Actor:
    """VALIDATE = creator.quickloop.quick_cycle: apply the patch (creator.tools.patch, atomic), static checks, affected tests. Returns
    'PASS' or 'FAIL' + the failure text; the stage seconds ride along in team.last_call."""
    def fn(env: TM.Envelope, team: Any) -> str:
        from creator import quickloop as QL
        c: ToolContext = team.ctx
        junit = c.ws.parent / f".junit_{c.ws.name}.xml"
        smoke = [p.relative_to(c.ws).as_posix() for p in (c.ws / "tests").glob("test_*.py")] if (c.ws / "tests").is_dir() else []
        r = QL.quick_cycle(c.files, c.ws, canonical_patch(_of_kind(team, env, "diff"), c.default_path)[0], junit, default_path=c.default_path, smoke=smoke, warm=False,
                           timeout=c.test_timeout)
        team.last_call = {"stage_s": r.get("stage_s"), "tests_run": (r.get("selection") or {}).get("tests") and len(r["selection"]["tests"])}
        if r.get("ok"):
            return "PASS"
        if not r.get("apply", {}).get("ok", True):
            ap = r["apply"]
            return "FAIL apply: " + str(ap.get("error") or ap.get("denied") or ap)[:400]
        if r.get("why"):
            st = r.get("static") or {}
            return f"FAIL static: {json.dumps(st)[:500]}"
        run = r.get("run") or {}
        txt = _failure_text(junit) or "; ".join(run.get("problems") or []) or f"run status {run.get('status')}"
        return "FAIL " + (run.get("status") or "") + ": " + txt
    return TM.actor("tests", fn)


def _pinpoint(team: Any, env: TM.Envelope, c: ToolContext) -> str:
    from creator.tools import pinpoint as PP
    tests = [p.relative_to(c.ws).as_posix() for p in (c.ws / "tests").glob("test_*.py")]
    fail = _of_kind(team, env, "failure")
    tb = PP.parse_traceback(fail, c.ws)
    lines: list[tuple[str, int, float]] = []
    try:
        lines = PP.pinpoint(c.ws, tests, diff=c.unified_diff(), top=5, python=c.python, timeout=c.test_timeout)
    except Exception as e:                                                      # noqa: BLE001 - fall back to the traceback frames
        team.last_call = {"pinpoint_error": type(e).__name__}
    if not lines:
        lines = [(f.file, f.line, 0.0) for f in tb.repo_frames()[:5]]
    c.pin_scores = [float(x[2]) for x in lines[:2]]                           # type: ignore[attr-defined]
    # a raised error inside the repo pins the line; a plain failed assert does not (0 of 7 such fixes worked in the first fast run)
    c.tb_lines = [(f.file, f.line) for f in tb.repo_frames()] if tb.exc_type and "Assertion" not in tb.exc_type else []   # type: ignore[attr-defined]
    out = []
    for f, ln, _s in lines[:5]:
        try:
            txt = (c.ws / f).read_text(encoding="utf-8", errors="replace").splitlines()[ln - 1].strip()
        except (OSError, IndexError):
            txt = ""
        out.append(f"{f}:{ln}: {txt}")
    return "\n".join(out)


class MeasuredTeam(TM.Team):
    """Team that emits ONE event per dispatched step to `sink(actor, **fields)` with the machine-wide busy CPU seconds of the step and, for a
    model step, the real token counts and prompt / generation seconds from the llama-server (team.last_call)."""

    def __init__(self, state: Any, actors: Any, goal_id: Optional[str] = None, sink: Optional[Callable[..., Any]] = None) -> None:
        super().__init__(state, actors, goal_id, log=self._emit)
        self.sink = sink
        self.ctx: Any = None
        self.last_call: dict[str, Any] = {}
        self.fast = False
        self._cpu0 = 0.0

    def dispatch(self, env: TM.Envelope, actor_name: Optional[str] = None) -> str:
        self.last_call = {}
        self._cpu0 = tree_cpu_s()
        return super().dispatch(env, actor_name)

    def _emit(self, actor_name: str, **kw: Any) -> None:
        if self.sink is None or kw.get("step") == "refused":
            return
        lc = dict(self.last_call)
        kw["cpu_s"] = round(tree_cpu_s() - self._cpu0, 4)
        if kw.get("model") and not kw.get("cache_hit") and lc:
            kw["in_tok"], kw["out_tok"] = lc.get("in_tok", kw["in_tok"]), lc.get("out_tok", kw["out_tok"])
            kw["outcome"] = lc.get("error") or ("truncated" if lc.get("truncated") else "ok")
        kw.update({k: v for k, v in lc.items() if k in ("prompt_s", "gen_s", "prompt_n", "cached_tok", "stop", "slot", "form", "stage_s", "looped", "retried",
                                                       "first_out_tok", "first_looped",
                                                       "index_update_s", "tests_run", "pinpoint_error")})
        self.sink(actor_name, **kw)


def actors(runner: SlotRunner) -> list[TM.Actor]:
    return [slot_actor("CHECKER", runner), slot_actor("THINKER", runner), slot_actor("CODER", runner), locate_actor(), pack_actor(),
            safety_actor(), tests_actor()]


def debug_worthwhile(ctx: Any) -> bool:
    """One debug attempt only when the failure is pinned: the traceback names a repo (non-test) line, or the top pinpoint line is clearly ahead."""
    if getattr(ctx, "tb_lines", None):
        return True
    sc = getattr(ctx, "pin_scores", None) or []
    return bool(sc) and sc[0] >= 0.8 and (len(sc) < 2 or sc[0] - sc[1] >= 0.15)


def tests_decided(ctx: Any, val: str) -> bool:
    """True when the visible tests ran (or could not apply the edit) and said PASS / FAIL: the outcome is known, a CONFIDENCE call adds nothing.
    Only a task with no runnable test file still needs the checker."""
    tdir = Path(ctx.ws) / "tests"
    has_tests = tdir.is_dir() and any(tdir.glob("test_*.py"))
    return bool(has_tests and val.startswith(("PASS", "FAIL")))
