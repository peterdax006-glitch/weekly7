"""Creator resources - CPU and RAM as two budgets that Nupen manages together.

Owner, 2 Oct 2026: "if CPU is out, it should look for tasks that dont use CPU that are RAM specific, it should also learn how to
structure itself so it can have a bunch of tasks going so if they all run at the same time its like 200% CPU power but it should be
structured so they manage CPU well and then if RAM is full it should see if there are other RAM specific goals". Measured the same
day on this PC (14 threads, 33.8 GB): CPU 100% while RAM ~50%; Nupen's dev work is CPU-bound (test processes ~38 MB each).

1. PROFILES   what a kind of work costs: average cores busy, peak GB, seconds. Priors until measured; then the median of what was
              recorded (resource_tasks.jsonl), cycle seconds (cycle.json) and the measured test-process memory (test_proc_mb.json).
2. ADMISSION  decide(): CPU may be oversubscribed up to `oversub` x logical cores of runnable work, but managed - dev/learning work
              runs BELOW_NORMAL, filler/measurement IDLE (priority_class) so the important work wins; with CPU saturated (sustained)
              new CPU-heavy work is deferred and RAM-heavy/low-CPU work preferred; near the RAM floor RAM-heavy work is deferred and
              CPU-light work preferred. Work is never deferred when nothing of ours is running (progress is guaranteed).
3. RAM WORK    useful low-CPU/high-RAM tasks (RAM_CANDIDATES); implemented end to end: warm_model_cache reads the local model file
              into the OS file cache so the next server start does not pay the disk read.
4. SAMPLES    resource_samples.jsonl (append-only) feeds the 'resource_balance' constraint metric (creator.constraints).
Loaded on demand through creator.registry ('resources'); only psutil is imported (lazily)."""
from __future__ import annotations

import collections
import dataclasses
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Optional

TASKS_FILE = "resource_tasks.jsonl"
SAMPLES_FILE = "resource_samples.jsonl"
RAM_WORK_FILE = "ram_work.jsonl"
SATURATED_PCT = 95.0
MIN_MEASURED = 3                                   # measurements before a profile stops leaning on its prior
CPU_HEAVY = 0.75                                   # average busy cores from which work counts as CPU-heavy
CPU_LIGHT = 0.25
RAM_HEAVY_GB = 1.0
RAM_JOB_GB = 1.0                                   # headroom a RAM job is assumed to need
SAFETY_CEILING_X = 10                              # worker ceiling = 10 x logical cores: a runaway-bug valve, NOT the operating limit


@dataclasses.dataclass(frozen=True)
class Profile:
    kind: str
    cores: float                                   # average logical cores busy while it runs
    peak_gb: float
    seconds: float
    source: str = "prior"                          # "prior" | "measured" | "blend"

    @property
    def cpu_heavy(self) -> bool:
        return self.cores >= CPU_HEAVY

    @property
    def ram_heavy(self) -> bool:
        return self.peak_gb >= RAM_HEAVY_GB

    @property
    def cpu_light(self) -> bool:
        return self.cores <= CPU_LIGHT


# sensible priors (cores, peak GB, seconds) until something is measured
PRIORS: dict[str, tuple[float, float, float]] = {
    "cycle": (2.5, 0.5, 600.0),                    # a worker + its test processes (~38 MB each, test_proc_mb.json)
    "filler": (1.0, 0.3, 120.0),                   # own-worker self-benchmark
    "practice": (1.5, 0.5, 300.0),
    "lm_train": (6.0, 4.0, 1800.0),
    "model_think": (8.0, 6.0, 120.0),              # a local llama server's thinking time
    "test_proc": (1.0, 0.04, 30.0),
    "ram": (0.1, 2.0, 60.0),                       # any RAM-specific task
}
PRIORITY = {"cycle": "BELOW_NORMAL", "practice": "BELOW_NORMAL", "lm_train": "BELOW_NORMAL", "model_think": "BELOW_NORMAL",
            "test_proc": "BELOW_NORMAL", "filler": "IDLE", "ram": "IDLE"}


def family(kind: str) -> str:
    return kind.split(":", 1)[0]


def priority_class(kind: str) -> str:
    """Importance -> OS priority class: development/learning BELOW_NORMAL (the owner's own programs stay NORMAL), filler and
    measurement IDLE, so when CPU is oversubscribed the important work wins."""
    return PRIORITY.get(family(kind), "BELOW_NORMAL")


def _rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict):
                out.append(r)
    except OSError:
        pass
    return out


def _append(path: Path, rec: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            f.write((json.dumps(rec, sort_keys=True, default=str) + "\n").encode("utf-8"))
    except OSError:
        pass


def record_task(state: Path, kind: str, seconds: float, cpu_seconds: Optional[float], peak_gb: float, units: Optional[float] = None) -> None:
    """One measured run of a kind of work (cores = cpu_seconds / seconds; omitted when the CPU cost was not measured). `units` (tokens, rows,
    steps ...) lets creator.jobcost price the kind as fixed + rate x units for the PC-vs-GPU placement decision."""
    row: dict[str, Any] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "seconds": round(seconds, 2), "peak_gb": round(peak_gb, 4)}
    if units is not None:
        row["units"] = units
    if cpu_seconds is not None and seconds > 0:
        row["cores"] = round(cpu_seconds / seconds, 3)
    _append(Path(state) / TASKS_FILE, row)


def _cycle_seconds(state: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    root = Path(state) / "cycles"
    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return out
    for d in dirs[-200:]:
        try:
            cj = json.loads((d / "cycle.json").read_text(encoding="utf-8"))
            sec = float(cj.get("seconds") or 0.0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        req = str(cj.get("requirement") or "").split(".", 1)[0]
        if sec > 0 and cj.get("outcome") in ("ADOPTED", "REJECTED"):
            out.setdefault(f"cycle:{req}" if req else "cycle", []).append(sec)
    return out


def profiles(state: Path) -> dict[str, Profile]:
    """Every kind with measurements, as Profiles; kinds with fewer than MIN_MEASURED rows blend with their prior."""
    state = Path(state)
    rows: dict[str, list[dict[str, Any]]] = {}
    for r in _rows(state / TASKS_FILE):
        rows.setdefault(str(r.get("kind", "")), []).append(r)
    secs = _cycle_seconds(state)
    out: dict[str, Profile] = {}
    for kind in set(rows) | set(secs):
        pc, pg, ps = PRIORS.get(family(kind), PRIORS["cycle"])
        rs = rows.get(kind, [])
        cores = [float(r["cores"]) for r in rs if "cores" in r]
        gbs = [float(r["peak_gb"]) for r in rs if "peak_gb" in r]
        wall = [float(r["seconds"]) for r in rs if "seconds" in r] + secs.get(kind, [])
        n = max(len(cores), len(gbs), len(wall))
        med = lambda xs, prior: statistics.median(xs) if xs else prior          # noqa: E731
        w = min(1.0, n / MIN_MEASURED)
        out[kind] = Profile(kind, round(w * med(cores, pc) + (1 - w) * pc, 3), round(w * med(gbs, pg) + (1 - w) * pg, 4),
                            round(w * med(wall, ps) + (1 - w) * ps, 1), "measured" if n >= MIN_MEASURED else "blend")
    return out


def profile(kind: str, state: Optional[Path] = None, cache: Optional[dict[str, Profile]] = None) -> Profile:
    known = cache if cache is not None else (profiles(state) if state is not None else {})
    if kind in known:
        return known[kind]
    c, g, s = PRIORS.get(family(kind), PRIORS["cycle"])
    if family(kind) == "test_proc" and state is not None:                       # measured test-process memory
        try:
            g = float(json.loads((Path(state) / "test_proc_mb.json").read_text(encoding="utf-8"))["avg_mb"]) / 1024.0
            return Profile(kind, c, round(g, 4), s, "measured")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return Profile(kind, c, g, s, "prior")


# ------------------------------------------------------------------------------------------------ the machine

@dataclasses.dataclass(frozen=True)
class Machine:
    cores: int
    total_gb: float
    free_gb: float
    cpu_pct: float                                  # sustained CPU load, percent of all logical cores
    cpu_now: float = 0.0
    cpu_usable: float = -1.0                        # load of the cores the OS actually schedules on (-1 = not measured); see usable_cpu
    parked: int = 0                                 # logical cores left idle while the others are saturated


# 3 Oct 2026 (h38): the CPU sample sat at 85.7-86.3% all day although the machine was saturated (processor queue 30). Not a stuck sampler:
# this Core Ultra 7 255U has 14 logical processors, of which the 2 low-power E-cores (CPU 12-13) get no work from the Windows scheduler
# (0.1% busy while the other 12 run at 100%; they do run work pinned to them). 12/14 = 85.7% < SATURATED_PCT, so "CPU saturated" never fired
# and CPU-heavy work kept being admitted into a full machine. A core that stays idle for PARK_S while the others are saturated is not
# capacity the scheduler will use: saturation is judged on the cores that are.
PARK_S = 60.0
PARK_IDLE_PCT = 5.0
_CPU_BUSY_AT: dict[int, float] = {}


def usable_cpu(percpu: list[float], now: float, park_s: float = PARK_S) -> tuple[float, int]:
    """(busy percent of the scheduled cores, how many cores are parked). A core is parked when it has been under PARK_IDLE_PCT for park_s
    (since it was first seen) AND the remaining cores average at least SATURATED_PCT: idle capacity next to waiting work. Otherwise every
    core counts and the result equals the machine-wide mean."""
    if not percpu:
        return 0.0, 0
    for i, v in enumerate(percpu):
        if v >= PARK_IDLE_PCT or i not in _CPU_BUSY_AT:
            _CPU_BUSY_AT[i] = now
    idle = [i for i in range(len(percpu)) if now - _CPU_BUSY_AT[i] >= park_s]
    rest = [v for i, v in enumerate(percpu) if i not in idle]
    if idle and rest and sum(rest) / len(rest) >= SATURATED_PCT:
        return sum(rest) / len(rest), len(idle)
    return sum(percpu) / len(percpu), 0


def read_machine(cpu_pct: Optional[float] = None) -> Machine:
    try:
        import psutil
        vm = psutil.virtual_memory()
        now = float(psutil.cpu_percent(interval=None))
        try:
            usable, parked = usable_cpu([float(x) for x in psutil.cpu_percent(interval=None, percpu=True)], time.monotonic())
        except Exception:                                                       # noqa: BLE001 - per-core figures are a refinement only
            usable, parked = -1.0, 0
        return Machine(psutil.cpu_count(logical=True) or 1, vm.total / 1e9, vm.available / 1e9, now if cpu_pct is None else cpu_pct, now,
                       usable, parked)
    except ImportError:
        return Machine(1, 16.0, 8.0, 0.0 if cpu_pct is None else cpu_pct)


@dataclasses.dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str
    priority: str = "BELOW_NORMAL"
    prefer: str = ""                                # "ram" | "cpu": what to start instead when deferred


def decide(p: Profile, m: Machine, running_cores: float, running: int, oversub: float = 2.0, floor_gb: float = 0.8,
           reserved_gb: float = 0.0) -> Decision:
    """Admit or defer one piece of work on this machine (pure; tests use fake machines)."""
    pri = priority_class(p.kind)
    if running <= 0:
        return Decision(True, "nothing of ours is running: always progress", pri)
    saturated = m.cpu_pct >= SATURATED_PCT
    # h43 (3 Oct 2026): the lowest class, IDLE (the thinking fillers: IDLE threads; the drill/model processes they start run BELOW_NORMAL),
    # is not deferred by saturation - "~200% CPU managed by priority", the oversubscription bound below still applies, and a filler always
    # loses the CPU to cycles and to the owner's programs. Replaying the live samples of 2-3 Oct through
    # the Admitter: once h38 made saturation visible, a new filler was admitted in 31.8% of samples instead of 100%; now 100% again, while
    # BELOW_NORMAL CPU-heavy work (cycles) is still deferred as h38 intended.
    if saturated and p.cpu_heavy and pri != "IDLE":
        return Decision(False, f"CPU saturated ({m.cpu_pct:.0f}%): CPU-heavy work deferred, RAM work preferred", pri, "ram")
    if running_cores + p.cores > oversub * m.cores:
        return Decision(False, f"oversubscribed: {running_cores + p.cores:.1f} runnable cores > {oversub:g} x {m.cores}", pri, "ram")
    left = m.free_gb - p.peak_gb - reserved_gb
    if left < floor_gb and p.ram_heavy:
        return Decision(False, f"RAM near the floor ({m.free_gb:.1f} GB free): RAM-heavy work deferred, CPU-light work preferred", pri, "cpu")
    if left < floor_gb and not p.cpu_light:
        return Decision(False, f"RAM below the floor after this task ({left:.2f} GB)", pri, "cpu")
    return Decision(True, "fits", pri)


def current_rule(p: Profile, m: Machine, running: int, max_workers: int = 32, floor_gb: float = 0.8) -> bool:
    """The previous behaviour for comparison (simulation only): RAM alone decided, CPU was never looked at."""
    return running < max_workers and m.free_gb - p.peak_gb >= floor_gb


# ------------------------------------------------------------------------------------------------ priorities

def set_process_priority(pid: int, cls: str) -> bool:
    """Lower one process's priority class (Windows: BELOW_NORMAL / IDLE; elsewhere nice 10 / 19). Never raises."""
    try:
        import psutil
        proc = psutil.Process(pid)
        if sys.platform == "win32":
            proc.nice(psutil.IDLE_PRIORITY_CLASS if cls == "IDLE" else psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(19 if cls == "IDLE" else 10)
        return True
    except Exception:                                                           # noqa: BLE001 - priority is a hint, never a failure
        return False


def lower_sandbox_priority(scratch: Any, cls: str = "BELOW_NORMAL") -> int:
    """Apply `cls` to every process whose cwd is inside the swarm's sandboxes (workers' test processes); returns how many."""
    try:
        import psutil
    except ImportError:
        return 0
    root = str(Path(scratch).resolve()).lower() if scratch else ""
    n = 0
    for proc in (psutil.process_iter(["pid"]) if root else []):
        try:
            if str(Path(proc.cwd()).resolve()).lower().startswith(root) and set_process_priority(proc.pid, cls):
                n += 1
        except (psutil.Error, OSError):
            continue
    return n


def idle_thread(fn: Callable[[], None]) -> Callable[[], None]:
    """Wrap a filler job so its thread runs at idle priority on Windows (Win32 SetThreadPriority); elsewhere unchanged."""
    def run() -> None:
        if sys.platform == "win32":
            try:
                import ctypes
                k32 = ctypes.windll.kernel32                                    # type: ignore[attr-defined]
                k32.SetThreadPriority(k32.GetCurrentThread(), -15)              # THREAD_PRIORITY_IDLE
            except Exception:                                                   # noqa: BLE001
                pass
        fn()
    return run


# ------------------------------------------------------------------------------------------------ samples + admitter

def record_sample(state: Path, m: Machine) -> None:
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpu": round(m.cpu_now or m.cpu_pct, 1), "free_gb": round(m.free_gb, 2),
           "total_gb": round(m.total_gb, 2), "cores": m.cores}
    if m.cpu_usable >= 0:
        row.update(cpu_usable=round(m.cpu_usable, 1), parked=m.parked)
    _append(Path(state) / SAMPLES_FILE, row)


class Admitter:
    """What the Governor consults: live machine, sustained-CPU watch, per-kind profiles, running demand."""

    def __init__(self, state: Path, oversub: float = 2.0, sustain_s: float = 60.0, machine: Optional[Callable[[], Machine]] = None,
                 clock: Callable[[], float] = time.monotonic, sample_every_s: float = 30.0) -> None:
        self.state, self.oversub, self.sustain_s = Path(state), oversub, sustain_s
        self._machine, self._clock, self.sample_every_s = machine or read_machine, clock, sample_every_s
        self._cpu: Deque[tuple[float, float]] = collections.deque()
        self._last_sample = -1e9
        self._profiles: dict[str, Profile] = {}
        self._loaded = -1e9
        self.demand: dict[str, float] = {}                                      # kind -> cores of work we have running
        self.lock = threading.Lock()
        self.log: list[Decision] = []

    def machine(self) -> Machine:
        m = self._machine()
        t = self._clock()
        with self.lock:
            self._cpu.append((t, m.cpu_usable if m.cpu_usable >= 0 else (m.cpu_now or m.cpu_pct)))
            while self._cpu and t - self._cpu[0][0] > self.sustain_s:
                self._cpu.popleft()
            enough = len(self._cpu) >= 3 and t - self._cpu[0][0] >= self.sustain_s / 2
            sustained = statistics.fmean(v for _, v in self._cpu) if enough else min(m.cpu_pct, SATURATED_PCT - 1)
        if t - self._last_sample >= self.sample_every_s:
            self._last_sample = t
            record_sample(self.state, m)
        return dataclasses.replace(m, cpu_pct=sustained, cpu_now=m.cpu_now or m.cpu_pct)

    def profile(self, kind: str) -> Profile:
        if self._clock() - self._loaded > 300:
            self._profiles, self._loaded = profiles(self.state), self._clock()
        return profile(kind, self.state, self._profiles)

    def admit(self, kind: str, running: int, running_cores: Optional[float] = None) -> Decision:
        p = self.profile(kind)
        d = decide(p, self.machine(), self._cores(running) if running_cores is None else running_cores, running, self.oversub)
        self.log.append(d)
        del self.log[:-50]
        return d

    def _cores(self, running: int) -> float:
        return sum(self.demand.values()) or running * self.profile("cycle").cores

    def busy_cores(self) -> float:
        """Average logical cores busy over the sustain window (machine-wide)."""
        m = self._machine()
        with self.lock:
            pct = statistics.fmean(v for _, v in self._cpu) if self._cpu else m.cpu_now
        return pct / 100.0 * (m.cores - m.parked)                              # the window holds the scheduled cores' load

    def wants_ram_work(self, running: int) -> bool:
        """CPU has no room (saturated, or runnable work at the oversubscription limit) while RAM does: fill the other resource."""
        m = self.machine()
        cpu_full = m.cpu_pct >= SATURATED_PCT or self._cores(running) + self.profile("cycle").cores > self.oversub * m.cores
        return running > 0 and cpu_full and m.free_gb > 2.0 + RAM_JOB_GB

    def next_kind(self, running: int, candidates: tuple[str, ...] = ("cycle", "filler", "ram")) -> str:
        """Which kind of work to add next: the first candidate admitted, CPU-heavy kinds first while CPU has room, RAM-heavy/low-CPU
        kinds when only RAM has room; '' when both resources are full."""
        m = self.machine()
        ram_room = (m.free_gb - 0.8) / m.total_gb > 1.0 - m.cpu_pct / 100.0              # more headroom in RAM than in CPU
        key = (lambda k: -self.profile(k).peak_gb / (self.profile(k).cores + 0.1)) if ram_room else               (lambda k: -self.profile(k).cores / (self.profile(k).peak_gb + 0.1))
        for kind in sorted(candidates, key=key):
            if self.admit(kind, running).allow:
                return kind
        return ""


# ------------------------------------------------------------------------------------------------ RAM work

RAM_CANDIDATES: dict[str, str] = {
    "warm_model_cache": "IMPLEMENTED: read the local model file into the OS file cache so the next llama-server start skips the disk read",
    "warm_tree_cache": "IMPLEMENTED as warm_git_objects: read the git pack indexes/packs into the OS file cache so sandbox creation and assess do not hit a cold disk",
    "hot_test_evidence": "keep recent test evidence and cycle records parsed in memory for constraints and curriculum scans",
    "lm_data_in_memory": "load the language-model training corpus into RAM once so a training run starts without tokenising from disk",
    "price_data_preload": "load research/price panels for later analysis so a probe starts with the data resident",
    "extra_model_servers": "start another llama server slot when RAM is free (the dev-bench thinking time is wall time, not CPU)",
}


def warm_file(path: Path, max_gb: float = 8.0, chunk: int = 8 << 20, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Read a file sequentially so the OS keeps it in its file cache. Returns bytes and the throughput of the first (cold) pass and,
    for the first chunks again, the warm re-read: the difference estimates the start-up seconds saved for the next reader."""
    path = Path(path)
    limit = int(max_gb * 2**30)
    done, t0 = 0, clock()
    with open(path, "rb") as f:
        while done < limit:
            b = f.read(min(chunk, limit - done))
            if not b:
                break
            done += len(b)
    cold = max(clock() - t0, 1e-9)
    probe = min(done, 256 << 20)
    t1, got = clock(), 0
    with open(path, "rb") as f:
        while got < probe:
            b = f.read(min(chunk, probe - got))
            if not b:
                break
            got += len(b)
    warm_rate = got / max(clock() - t1, 1e-9)
    saved = max(0.0, done / (done / cold) - done / warm_rate) if done and got else 0.0
    return {"bytes": done, "cold_s": round(cold, 3), "warm_mb_s": round(warm_rate / 1e6, 1), "saved_s_estimate": round(saved, 3)}


def warm_model_cache(state: Path, model: Optional[Path] = None, free_gb: Optional[float] = None, floor_gb: float = 2.0,
                     clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """RAM-specific, low-CPU task: pre-warm the local model file (the file the next student attempt's server loads). Skipped when
    the file is missing, already warmed by us within 6 h, or free RAM minus the file would drop under `floor_gb`."""
    state = Path(state)
    if model is None:
        try:
            from creator import registry as REG
            model = Path(REG.get("device").model_path())
        except Exception:                                                       # noqa: BLE001
            return {"task": "warm_model_cache", "skipped": "model path unknown"}
    if not Path(model).is_file():
        return {"task": "warm_model_cache", "skipped": "model file missing"}
    size = Path(model).stat().st_size / 1e9
    free = read_machine().free_gb if free_gb is None else free_gb
    if free - size < floor_gb:
        return {"task": "warm_model_cache", "skipped": f"RAM: {free:.1f} GB free, file {size:.1f} GB"}
    for r in reversed(_rows(state / RAM_WORK_FILE)):
        if r.get("task") == "warm_model_cache" and r.get("file") == str(model) and time.time() - float(r.get("ts", 0)) < 6 * 3600:
            return {"task": "warm_model_cache", "skipped": "warmed within 6 h"}
    res = {"task": "warm_model_cache", "file": str(model), "ts": time.time(), **warm_file(Path(model), max_gb=max(0.1, free - floor_gb), clock=clock)}
    _append(state / RAM_WORK_FILE, res)
    record_task(state, "ram:warm_model_cache", res["cold_s"], res["cold_s"] * 0.1, size)
    return res


def warm_git_objects(state: Path, repo: Path, free_gb: Optional[float] = None, floor_gb: float = 4.0, ttl_s: float = 3 * 3600,
                     clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """RAM-specific, low-CPU task: keep the repository's git object store (pack indexes, multi-pack-index, packs) in the OS file cache.
    Every sandbox (`git worktree add`, checkout, diff, rev-parse) and every assess/plan cycle reads these files; on a repo whose packs are
    GBs the first reader of a cold cache pays the disk. Packs are read smallest first while free RAM minus what is read stays above
    `floor_gb`; re-done after `ttl_s`. Returns the files read, bytes and the cold-vs-warm throughput (estimated seconds saved)."""
    state, repo = Path(state), Path(repo)
    pack = repo / ".git" / "objects" / "pack"
    if not pack.is_dir():
        return {"task": "warm_git_objects", "skipped": "no pack directory"}
    for r in reversed(_rows(state / RAM_WORK_FILE)):
        if r.get("task") == "warm_git_objects" and r.get("repo") == str(repo) and time.time() - float(r.get("ts", 0)) < ttl_s:
            return {"task": "warm_git_objects", "skipped": "warmed recently"}
    free = read_machine().free_gb if free_gb is None else free_gb
    files = sorted((f for f in pack.iterdir() if f.is_file()), key=lambda f: (f.suffix == ".pack", f.stat().st_size))   # indexes first
    read = cold = 0.0
    saved = 0.0
    names: list[str] = []
    for f in files:
        size = f.stat().st_size / 1e9
        if free - read - size < floor_gb:
            continue
        w = warm_file(f, max_gb=size + 0.01, clock=clock)
        read += w["bytes"] / 1e9
        cold += w["cold_s"]
        saved += w["saved_s_estimate"]
        names.append(f.name[:12])
    res = {"task": "warm_git_objects", "repo": str(repo), "ts": time.time(), "files": len(names), "gb": round(read, 3),
           "cold_s": round(cold, 3), "saved_s_estimate": round(saved, 3)}
    if names:
        _append(state / RAM_WORK_FILE, res)
    record_task(state, "ram:warm_git_objects", cold, cold * 0.1, read)
    return res


_STARTED: dict[str, float] = {}


def attach(gov: Any, state: Path, real_free: Any) -> None:
    """Switch a REAL-machine governor to resource-governed admission (a test's fake governor stays RAM-only): workers are added
    while CPU (oversubscribed under priority classes) or RAM has room; max_workers becomes a 10x-cores safety valve (owner, 2 Oct)."""
    if gov.admit is None and gov.free is real_free:
        gov.admit = Admitter(state)
        gov.max_workers = max(gov.max_workers, SAFETY_CEILING_X * (gov.admit.machine().cores))


def ok(gov: Any, kind: str, running: int) -> bool:
    return gov.admit is None or bool(gov.admit.admit(kind, running).allow)


def ram_thread(gov: Any, state: Path, running: int, ramped: bool, pool: list[Any]) -> bool:
    """Start a RAM-work thread (appended to `pool`) when CPU has no room and RAM does; True when one started."""
    job = ram_job(state) if (gov.admit is not None and ramped and not gov.too_tight() and gov.admit.wants_ram_work(running)) else None
    if job is None:
        return False
    pool.append(threading.Thread(target=job, name="swarm-ram", daemon=True))
    pool[-1].start()
    return True


def learn(gov: Any, state: Path, step: str, started: float, others: int) -> None:
    """Record what a finished cycle really cost (wall seconds, machine busy cores / workers, per-worker GB) for later admission."""
    if gov.admit is None:
        return
    try:
        sec = time.monotonic() - started
        record_task(state, f"cycle:{step}", sec, sec * gov.admit.busy_cores() / max(1, others + 1), (gov.observe(1) or 0.0) if gov.observe else 0.0)
    except Exception:                                                           # noqa: BLE001 - measuring never stops the swarm
        pass


def ram_job(state: Path) -> Optional[Callable[[], None]]:
    """The next RAM-specific job for run_round's saturated-CPU branch (None when nothing is worth doing)."""
    done = {r.get("task") for r in _rows(Path(state) / RAM_WORK_FILE) if time.time() - float(r.get("ts", 0)) < 6 * 3600}
    if time.time() - _STARTED.get(str(state), 0.0) < 3600:
        return None
    if "warm_model_cache" not in done:
        _STARTED[str(state)] = time.time()
        return idle_thread(lambda: None if warm_model_cache(Path(state)) is None else None)
    if "warm_git_objects" not in done:
        _STARTED[str(state)] = time.time()
        repo = Path(state).resolve().parents[1]                                  # state/creator -> the repository
        return idle_thread(lambda: None if warm_git_objects(Path(state), repo) is None else None)
    return None


# ------------------------------------------------------------------------------------------------ the constraint metric

def balance_shares(state: Path, since_s: float, now_ts: Optional[float] = None) -> dict[str, Any]:
    """Shares of samples in the window with CPU saturated while RAM idles (used < 60%) and the reverse (RAM > 85% used, CPU < 60%)."""
    now_ts = now_ts if now_ts is not None else time.time()
    n = a = b = 0
    for r in _rows(Path(state) / SAMPLES_FILE):
        try:
            ts = time.mktime(time.strptime(str(r["at"]), "%Y-%m-%dT%H:%M:%S"))
            cpu, free, total = float(r["cpu"]), float(r["free_gb"]), float(r["total_gb"])
        except (KeyError, ValueError, TypeError):
            continue
        if now_ts - since_s <= ts <= now_ts and total > 0:
            n += 1
            used = 1.0 - free / total
            a += cpu >= SATURATED_PCT and used < 0.60
            b += used > 0.85 and cpu < 60.0
    return {"samples": n, "cpu_full_ram_idle": round(a / n, 4) if n else 0.0, "ram_full_cpu_idle": round(b / n, 4) if n else 0.0}


# ------------------------------------------------------------------------------------------------ simulation (fake machine)

def simulate(cores: int = 14, total_gb: float = 33.8, base_gb: float = 8.0, kinds: tuple[str, ...] = ("cycle", "filler", "ram", "model_think"),
             oversub: float = 2.0, ceiling: Optional[int] = None, admission: bool = True, steps: int = 600, state: Optional[Path] = None) -> dict[str, Any]:
    """Add work one at a time on a fake machine until nothing more is admitted. Machine load = what the work would use (CPU % capped
    at 100, GB on top of `base_gb` the owner/OS use). admission=False is the previous rule: RAM only, a fixed worker cap (64)."""
    import tempfile
    st = Path(state) if state else Path(tempfile.mkdtemp())
    running: list[Profile] = []
    t = [0.0]

    def machine() -> Machine:
        busy = sum(p.cores for p in running)
        return Machine(cores, total_gb, total_gb - base_gb - sum(p.peak_gb for p in running), min(100.0, 100.0 * busy / cores))
    adm = Admitter(st, oversub, machine=machine, clock=lambda: t[0], sample_every_s=1e9)
    cap = ceiling if ceiling is not None else SAFETY_CEILING_X * cores
    for i in range(steps):
        t[0] += 10.0
        if admission:
            adm.demand = {"all": sum(p.cores for p in running)}
            kind = adm.next_kind(len(running), kinds)
            if not kind or len(running) >= cap:
                if i > 12:
                    break
                continue
        else:
            kind = kinds[i % len(kinds)] if kinds[i % len(kinds)] != "ram" else kinds[0]
            if len(running) >= 64 or machine().free_gb - profile(kind).peak_gb < 0.07 * total_gb:
                break
        running.append(profile(kind))
    m = machine()
    by: dict[str, int] = {}
    for p in running:
        by[p.kind] = by.get(p.kind, 0) + 1
    return {"workers": len(running), "by_kind": by, "cpu_pct": round(m.cpu_pct, 1), "demand_cores": round(sum(p.cores for p in running), 1),
            "ram_used_pct": round(100.0 * (1 - m.free_gb / total_gb), 1)}
