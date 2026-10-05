"""Operation timer + slow-path detector (owner, 3 Oct 2026: "if something takes like 44 minutes on the PC developing a thought or anything
like that it should be detected and we should make it take like 100x faster").

Every named operation Nupen runs can be timed (`timed(state, name)` / `record(state, name, wall_s, cpu_s)`): one JSON line per run in
state/creator/op_timings.jsonl (wall seconds, CPU seconds of the running thread, pid). After each run the operation is checked against
  - its own history: the rolling median of its last WINDOW runs; a run >= FACTOR x that median (and >= MIN_FLAG_S) is a 'spike';
  - its budget (BUDGETS: what it SHOULD cost on this PC, documented per operation); a run over budget (and >= MIN_FLAG_S) is 'over_budget';
  - for an operation with no budget: a median >= LONG_S is 'long' - minutes where nobody has said minutes are needed.
A flag is appended to state/creator/slowpaths.jsonl (at most once per operation and kind per FLAG_EVERY_S) and becomes ONE goal proposal
(creator.goals, source 'slowpath', key 'slowpath:<operation>', pending the usual approval) carrying the measured cost, the expected cost
and the factor to win - so Nupen's own improvement queue keeps receiving its slowest paths. `report(state)` ranks every operation by
CPU-hours per day (python -m creator.slowpath [state]).

Overhead: two clock reads per operation and one appended line; the history check reads only the tail of the timing file. Never raises
into the timed operation (timing is measurement, not work). Lightweight imports only: the goals module is loaded when a flag is proposed."""
from __future__ import annotations

import atexit
import contextvars
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

OPS_FILE = "op_timings.jsonl"
FLAGS_FILE = "slowpaths.jsonl"
FACTOR = 10.0                  # a spike: this many times the operation's own rolling median
WINDOW = 50                    # runs in the rolling median
MIN_SAMPLES = 5                # earlier runs needed before a spike can be judged
MIN_FLAG_S = 30.0              # nothing shorter than this is worth a work item (seconds of wall time)
LONG_S = 600.0                 # an unbudgeted operation whose median is ten minutes or more is flagged as 'long'
FLAG_EVERY_S = 86400.0         # one flag per (operation, kind) per day
TAIL_BYTES = 1 << 20           # the history check reads at most this much of the timing file
# What an operation SHOULD cost on the home PC (wall seconds, uncontended). Measured on 3 Oct 2026 (h62 slow-path hunt) after the fixes,
# x3 for contention; an operation over budget is flagged. Unlisted operations are judged by their own history and LONG_S only.
BUDGETS: dict[str, float] = {
    "chooser.retrain": 120.0,          # 7 listwise fits on ~450 rows (was 977-2593 s before the h62 feature/row caches)
    "think.run": 120.0,                # live drill predictions + the trust report
    "think.learnloop": 600.0,          # measure, curve, retention, diagnosis, improve
    "think.blueprint": 300.0,          # scripts/nupen_blueprint.py
}

_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _append(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
    with _lock, open(path, "ab") as f:                  # one write per line: concurrent appenders interleave whole lines
        f.write(line)


def _tail(path: Path, nbytes: Optional[int] = None) -> list[dict[str, Any]]:
    nbytes = TAIL_BYTES if nbytes is None else nbytes
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if size > nbytes:
        lines = lines[1:]                                # the first line may be cut
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def history(state: Path, name: str, rows: Optional[list[dict[str, Any]]] = None) -> list[float]:
    """Wall seconds of the operation's recent runs, oldest first (at most WINDOW)."""
    rows = _tail(Path(state) / OPS_FILE) if rows is None else rows
    return [float(r["wall_s"]) for r in rows if r.get("op") == name and isinstance(r.get("wall_s"), (int, float))][-WINDOW:]


def judge(name: str, wall_s: float, prior: list[float], budget: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The flag for one run (None = fine). `prior` = earlier runs' wall seconds (not including this one)."""
    budget = BUDGETS.get(name) if budget is None else budget
    med = statistics.median(prior) if prior else None
    if wall_s < MIN_FLAG_S:
        return None
    if len(prior) >= MIN_SAMPLES and med and wall_s >= FACTOR * med:
        return {"kind": "spike", "wall_s": round(wall_s, 3), "median_s": round(med, 3), "factor": round(wall_s / med, 1), "expected_s": round(med, 3)}
    if budget is not None and wall_s > budget:
        return {"kind": "over_budget", "wall_s": round(wall_s, 3), "budget_s": budget, "factor": round(wall_s / budget, 1), "expected_s": budget,
                "median_s": round(med, 3) if med else None}
    if budget is None:
        runs = prior + [wall_s]
        m = statistics.median(runs)
        if m >= LONG_S:
            return {"kind": "long", "wall_s": round(wall_s, 3), "median_s": round(m, 3), "runs": len(runs), "expected_s": None,
                    "factor": None}
    return None


def _recent_flag(state: Path, name: str, kind: str, now: float) -> bool:
    for r in reversed(_tail(Path(state) / FLAGS_FILE, 256 * 1024)):
        if r.get("op") == name and r.get("kind") == kind and now - float(r.get("t") or 0) < FLAG_EVERY_S:
            return True
    return False


def propose(state: Path, flag: dict[str, Any]) -> Optional[str]:
    """One goal proposal per slow operation (key 'slowpath:<op>'): never a duplicate of an existing one, whatever its status."""
    from creator import goals as GO                     # lazy: only when something is flagged
    name = str(flag["op"])
    key = f"slowpath:{name}"
    if any(x.get("key") == key for x in GO.listing(Path(state))):
        return None
    exp = flag.get("expected_s")
    want = f"cost <= {exp:.0f} s per run" if exp else "seconds, not minutes, per run"
    title = f"make {name} 10-100x faster ({flag['kind']}: {flag['wall_s']:.0f} s per run)"
    rationale = (f"slow-path detector: {name} took {flag['wall_s']:.1f} s wall / {flag.get('cpu_s')} s CPU (median {flag.get('median_s')} s); "
                 f"{want}. Keep its outputs identical (prove with equality/hashes): cache on inputs + source hash, incremental work, no "
                 f"repeated rescans, vectorise, batch.")
    p = GO._make("slowpath", name, title, rationale, [name, flag["kind"]], 10, 1.0, time.strftime("%Y-%m-%dT%H:%M:%S"))
    d = {**p.to_dict(), "key": key, "id": GO._pid(key), "kind": "efficiency", "metric": {"op": name, "before": flag}}
    GO._append(Path(state), d)
    return str(d["id"])


def record(state: Path, name: str, wall_s: float, cpu_s: Optional[float] = None, budget: Optional[float] = None,
           propose_goal: bool = True, **extra: Any) -> Optional[dict[str, Any]]:
    """Record one run and judge it. Returns the flag written (None when the run was fine). Never raises."""
    try:
        state = Path(state)
        now = _now()
        prior = history(state, name)
        row = {"op": name, "t": round(now, 3), "wall_s": round(float(wall_s), 3), "pid": os.getpid(), **extra}
        if cpu_s is not None:
            row["cpu_s"] = round(float(cpu_s), 3)
        _append(state / OPS_FILE, row)
        flag = judge(name, float(wall_s), prior, budget)
        if flag is None or _recent_flag(state, name, flag["kind"], now):
            return None
        flag = {"event": "flag", "op": name, "t": round(now, 3), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpu_s": row.get("cpu_s"), **flag}
        if propose_goal:
            try:
                flag["proposal"] = propose(state, flag)
            except Exception as e:                        # noqa: BLE001 - a broken goals module never loses the flag
                flag["proposal_error"] = f"{type(e).__name__}: {e}"
        _append(state / FLAGS_FILE, flag)
        return flag
    except Exception:                                     # noqa: BLE001 - timing never breaks the timed work
        return None


class timed:
    """`with timed(state, "think.run"): ...` - records wall and this thread's CPU seconds of the block (also when it raises)."""

    def __init__(self, state: Path, name: str, budget: Optional[float] = None, **extra: Any) -> None:
        self.state, self.name, self.budget, self.extra = state, name, budget, extra
        self.flag: Optional[dict[str, Any]] = None

    def __enter__(self) -> "timed":
        self.w0, self.c0 = time.perf_counter(), time.thread_time()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.flag = record(self.state, self.name, time.perf_counter() - self.w0, time.thread_time() - self.c0, self.budget,
                           **({"error": exc[0].__name__} if exc[0] else {}), **self.extra)


def wrap(state: Path, name: str, fn: Any) -> Any:
    """`fn` timed under `name` (for job callables handed to threads)."""
    def run() -> Any:
        with timed(state, name):
            return fn()
    return run


def report(state: Path, days: float = 1.0, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Operations ranked by CPU-hours per day over the last `days` (wall-hours where CPU was not measured)."""
    now = _now() if now is None else now
    rows = [r for r in _tail(Path(state) / OPS_FILE, 64 << 20) if now - float(r.get("t") or 0) <= days * 86400]
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("op")), []).append(r)
    out = []
    for name, rs in by.items():
        walls = [float(r["wall_s"]) for r in rs]
        cpus = [float(r["cpu_s"]) for r in rs if r.get("cpu_s") is not None]
        out.append({"op": name, "runs_per_day": round(len(rs) / days, 1), "median_wall_s": round(statistics.median(walls), 3),
                    "median_cpu_s": round(statistics.median(cpus), 3) if cpus else None,
                    "cpu_h_per_day": round((sum(cpus) if cpus else sum(walls)) / 3600 / days, 3), "budget_s": BUDGETS.get(name)})
    return sorted(out, key=lambda x: -x["cpu_h_per_day"])


# ---- the metrics bus (P0.2): one compact event per step, ONE function for every local-model call ------------------------------------------
# {t, goal_id, step, actor, in_tok, out_tok, cpu_s, wall_s, ram_peak_mb, cache_hit, outcome} in state/creator/metrics/events-YYYYMMDD.jsonl
# (rotated daily). `event()` is the only writer; `model_call()` is the only accounting path of a model call (generator.LocalModel._post,
# effladder.Endpoint.call, gpuday.chat_http all go through it). `step_context()` sets goal_id/step/actor for the calls inside it.
METRICS_DIR = "metrics"
_state_override: Optional[Path] = None
_prefix: tuple[Optional[Path], str] = (None, "")
_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("metrics_ctx", default={})


def set_state(state: Optional[Path]) -> None:
    """Where events go (None = the default). Tests and the kernel set this once."""
    global _state_override, _prefix
    _state_override = Path(state) if state is not None else None
    _prefix = (_state_override, str(_state_override / METRICS_DIR / "events-") if _state_override is not None else "")
    close_handles()


def ensure_state(state: Path) -> None:
    """Point the bus at `state` unless it already is (the kernel calls this where it starts)."""
    if _state_override != Path(state):
        set_state(state)


def close_handles() -> None:
    """Close the kept-open event files (set_state does; call before deleting a state directory)."""
    flush()
    with _lock:
        for f in list(_fh.values()):
            try:
                f.close()
            except OSError:
                pass
        _fh.clear()


def _state() -> Optional[Path]:
    if _state_override is not None:
        return _state_override
    if os.environ.get("PYTEST_CURRENT_TEST"):          # a test that did not ask for a bus never writes to the live state
        return None
    env = os.environ.get("NUPEN_STATE")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "state" / "creator"


class step_context:
    """`with step_context(goal_id="g1", step="code", actor="coder"): ...` - defaults for every event logged inside (thread/task local)."""

    def __init__(self, **fields: Any) -> None:
        self.fields = fields

    def __enter__(self) -> "step_context":
        self._tok = _ctx.set({**_ctx.get(), **self.fields})
        return self

    def __exit__(self, *exc: Any) -> None:
        _ctx.reset(self._tok)


_peak_fn: Any = None


def _make_peak_fn() -> Any:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        h = k32.GetCurrentProcess()

        def peak() -> Optional[float]:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            return round(pmc.PeakWorkingSetSize / 1048576, 1) if ps.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb) else None
        return peak
    import resource

    def peak_unix() -> Optional[float]:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024), 1)
    return peak_unix


def ram_peak_mb() -> Optional[float]:
    """Peak resident memory of this process in MB (one syscall)."""
    global _peak_fn
    try:
        if _peak_fn is None:
            _peak_fn = _make_peak_fn()
        return _peak_fn()
    except Exception:                                    # noqa: BLE001
        return None


# Buffered writer. Durability: lines are queued in memory and written when the buffer reaches FLUSH_BYTES, when FLUSH_S seconds have passed
# since the last write (checked at the next event), on flush()/close_handles()/set_state(), and at interpreter exit. A hard kill loses at
# most the last buffer (< FLUSH_BYTES or < FLUSH_S of events); nothing already written is ever rewritten.
FLUSH_BYTES = 32768
FLUSH_S = 2.0
_buf: list[bytes] = []
_buf_n = 0
_buf_path: Optional[str] = None
_last_flush = 0.0
_fh: dict[str, Any] = {}
_jstr: dict[str, str] = {}                            # json-encoded strings (goal ids, actors, steps repeat endlessly)
_day = ("", 0)                                        # (YYYYMMDD, minute stamp it was computed for)
_ram = (0.0, None)                                    # (monotonic stamp, MB): peak RSS is sampled at most once a second


_jkeys: dict[str, str] = {}


def _jk(k: str) -> str:
    v = _jkeys.get(k)
    if v is None:
        v = _jkeys[k] = json.dumps(k) if len(_jkeys) < 512 else json.dumps(k)
    return v


def _enc_other(v: Any) -> str:
    if v is True or v is False:
        return "true" if v else "false"
    if type(v) is float and v == v and abs(v) != float("inf"):
        return repr(v)
    return json.dumps(v, default=str)


def _js(s: Any) -> str:
    k = str(s)
    v = _jstr.get(k)
    if v is None:
        v = json.dumps(k)
        if len(_jstr) < 4096:
            _jstr[k] = v
    return v


def _flush_locked() -> None:
    global _buf_n, _last_flush
    _last_flush = time.monotonic()
    if not _buf or _buf_path is None:
        return
    data = b"".join(_buf)
    _buf.clear()
    _buf_n = 0
    key = _buf_path
    f = _fh.get(key)
    if f is None or f.closed:
        for k in list(_fh):
            try:
                _fh.pop(k).close()                    # yesterday's file
            except OSError:
                pass
        os.makedirs(os.path.dirname(_buf_path), exist_ok=True)
        f = _fh[key] = open(_buf_path, "ab", buffering=0)
    f.write(data)


def flush() -> None:
    """Write the buffered events now (the report does; so does exit)."""
    try:
        with _lock:
            _flush_locked()
    except Exception:                                 # noqa: BLE001
        pass


atexit.register(flush)


def _queue(path: str, line: bytes) -> None:
    global _buf_n, _buf_path
    with _lock:
        if _buf_path is not None and path != _buf_path:   # a new day or a new state dir: write the old buffer first
            _flush_locked()
        _buf_path = path
        _buf.append(line)
        _buf_n += len(line)
        if _buf_n >= FLUSH_BYTES or time.monotonic() - _last_flush >= FLUSH_S:
            _flush_locked()


def event(actor: str, *, goal_id: Optional[str] = None, step: Optional[str] = None, in_tok: int = 0, out_tok: int = 0,
          cpu_s: Optional[float] = None, wall_s: float = 0.0, cache_hit: bool = False, outcome: str = "ok", model: bool = False,
          state: Optional[Path] = None, **extra: Any) -> bool:
    """Queue one step event (written within FLUSH_S / FLUSH_BYTES). `model` marks a step that used a model (the Qwen share counts them).
    Never raises."""
    global _day, _ram
    try:
        if state is None and _state_override is not None:
            pre = _prefix[1]                                  # fast path: the configured bus, no Path arithmetic
        else:
            st = Path(state) if state is not None else _state()
            if st is None:
                return False
            pre = str(st / METRICS_DIR / "events-")
        c = _ctx.get()
        if goal_id is None:
            goal_id = c.get("goal_id")
        if step is None:
            step = c.get("step")
        now = _now()
        mn = int(now // 60)
        if _day[1] != mn:
            _day = (time.strftime("%Y%m%d", time.localtime(now)), mn)
        mono = time.monotonic()
        if mono - _ram[0] >= 1.0:
            _ram = (mono, ram_peak_mb())
        parts = [f'{{"t":{now:.3f},"actor":{_js(actor)},"in_tok":{in_tok},"out_tok":{out_tok},"wall_s":{wall_s:.4f},"cache_hit":{"true" if cache_hit else "false"},'
                 f'"outcome":{_js(outcome)},"model":{"true" if model else "false"}']
        if goal_id is not None:
            parts.append(f',"goal_id":{_js(goal_id)}')
        if step is not None:
            parts.append(f',"step":{_js(step)}')
        if cpu_s is not None:
            parts.append(f',"cpu_s":{cpu_s:.4f}')
        if _ram[1] is not None:
            parts.append(f',"ram_peak_mb":{_ram[1]}')
        if extra:
            for k, v in extra.items():                       # scalars encoded inline (json.dumps of the dict costs ~50 us)
                t = type(v)
                parts.append(f',{_jk(k)}:{_js(v) if t is str else (repr(v) if t is int else _enc_other(v))}')
        parts.append("}\n")
        _queue(pre + _day[0] + ".jsonl", "".join(parts).encode())
        return True
    except Exception:                                    # noqa: BLE001 - measurement never breaks the work
        return False


def est_tokens(text: Any) -> int:
    """Rough token count (4 chars per token) for backends that report no usage."""
    return (len(str(text)) + 3) // 4


_NORM = re.compile(r"\d+|\s+")


def sig_of(text: Any) -> str:
    """Normalized signature (12 hex): case, digit runs and whitespace folded, so the same envelope/prompt shape repeats to ONE signature."""
    return hashlib.blake2b(_NORM.sub(" ", str(text).lower()).encode("utf-8", "replace"), digest_size=6).hexdigest()


def err_sig(exc: Any, text: Any = "") -> str:
    """Error signature: exception class + the first line of its message, digits folded (stable across runs)."""
    return (type(exc).__name__ + ": " + _NORM.sub(" ", str(text or exc).strip().split(chr(10))[0].lower())[:80]).strip()


class model_call:
    """THE accounting path of a local-model call: `with model_call("qwen3-1.7b") as mc: ...; mc.tokens(in_, out)`.
    Logs one event (wall, thread CPU, tokens, outcome 'ok' / the exception name) when the block ends, also when it raises."""

    def __init__(self, model: Any = "", *, actor: Optional[str] = None, prompt: Any = None, sig: Optional[str] = None, cls: Optional[str] = None,
                 form_in: Optional[int] = None, form_out: Optional[int] = None, **extra: Any) -> None:
        name = Path(str(model)).name
        self.actor = actor or str(_ctx.get().get("actor") or "model:" + name)
        self.extra = {"model": True, "backend": name, **extra}
        sig = sig or (sig_of(prompt) if prompt is not None else None)         # fields the opportunity detectors read (constraints.detect_*)
        for k, v in (("sig", sig), ("cls", cls), ("form_in", form_in), ("form_out", form_out)):
            if v is not None:
                self.extra[k] = v
        self.in_tok = self.out_tok = 0
        self.cache_hit = False

    def tokens(self, in_tok: int, out_tok: int, cache_hit: bool = False) -> None:
        self.in_tok, self.out_tok, self.cache_hit = int(in_tok), int(out_tok), bool(cache_hit)

    def __enter__(self) -> "model_call":
        self.w0, self.c0 = time.perf_counter(), time.thread_time()
        return self

    def __exit__(self, *exc: Any) -> None:
        event(self.actor, in_tok=self.in_tok, out_tok=self.out_tok, cpu_s=time.thread_time() - self.c0, wall_s=time.perf_counter() - self.w0,
              cache_hit=self.cache_hit, outcome=exc[0].__name__ if exc[0] else "ok",
              **({**self.extra, "err": err_sig(exc[1])} if exc[0] else self.extra))


def metrics_report(state: Path, days: float = 1.0, now: Optional[float] = None) -> dict[str, Any]:
    """Qwen share (fraction of steps that used a model) and per-actor totals over the last `days`."""
    flush()
    now = _now() if now is None else now
    d = Path(state) / METRICS_DIR
    rows: list[dict[str, Any]] = []
    for f in sorted(d.glob("events-*.jsonl")) if d.is_dir() else []:
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if now - float(r.get("t") or 0) <= days * 86400:
                rows.append(r)
    actors: dict[str, dict[str, Any]] = {}
    for r in rows:
        a = actors.setdefault(str(r.get("actor")), {"steps": 0, "model_steps": 0, "in_tok": 0, "out_tok": 0, "cpu_s": 0.0, "wall_s": 0.0,
                                                    "cache_hits": 0, "errors": 0, "ram_peak_mb": 0.0})
        a["steps"] += 1
        a["model_steps"] += 1 if r.get("model") else 0
        a["in_tok"] += int(r.get("in_tok") or 0)
        a["out_tok"] += int(r.get("out_tok") or 0)
        a["cpu_s"] = round(a["cpu_s"] + float(r.get("cpu_s") or 0), 4)
        a["wall_s"] = round(a["wall_s"] + float(r.get("wall_s") or 0), 4)
        a["cache_hits"] += 1 if r.get("cache_hit") else 0
        a["errors"] += 1 if r.get("outcome") not in (None, "ok") else 0
        a["ram_peak_mb"] = max(a["ram_peak_mb"], float(r.get("ram_peak_mb") or 0))
    n = len(rows)
    m = sum(1 for r in rows if r.get("model"))
    return {"steps": n, "model_steps": m, "qwen_share": round(m / n, 4) if n else 0.0, "actors": actors}


def _main() -> None:
    st = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(__file__).resolve().parents[1] / "state" / "creator"
    if "--metrics" in sys.argv:
        k = sys.argv.index("--metrics")
        rep = metrics_report(st, float(sys.argv[k + 1]) if len(sys.argv) > k + 1 else 1.0)
        print(f"steps={rep['steps']} model_steps={rep['model_steps']} qwen_share={rep['qwen_share']:.1%}")
        for a, v in sorted(rep["actors"].items(), key=lambda x: -x[1]["wall_s"]):
            print(a, json.dumps(v))
    else:
        for r in report(st):
            print(json.dumps(r))


if __name__ == "__main__":                                # python -m creator.slowpath [state] [--metrics [days]]
    _main()
