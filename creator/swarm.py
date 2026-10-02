"""Creator K22 - the swarm: several of the system's OWN workers at once, sized by free RAM (owner, 1 Oct 2026: "my computer sometimes
has available RAM so we might as well use that RAM and have the system have multiple agents running, working on different parts so
they arent working around each other" / "but then when RAM gets tight just pull agents back in") - IMPLEMENTED, NOT VALIDATED.

A ROUND: assess main once (kernel.prepare); then, while the governor allows, plan one more package that touches no component or
file a running worker holds (kernel.plan_one with exclusions) and run it (kernel.execute) in its own thread with its own sandbox and
ledger handle. Everything that changes the repository itself - opening sandboxes, merging, post-merge verification, rollback - is
serialised by one lock, so workers never trip over each other in git; their work and evaluation run in parallel.

GOVERNOR: a new worker starts only if free RAM >= start_gb + per_worker_gb x (running workers), and never more than max_workers.
If free RAM falls below low_gb, nothing new starts and the YOUNGEST running worker is cancelled at its next checkpoint (before
work / evaluation / adoption): its sandbox is discarded and the cancellation is recorded - nothing half-done is ever adopted.
These are the system's own workers (rules, search) - never Claude agents."""
from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from creator import goals as GOALS
from creator import kernel as K
from creator import model as M
from creator import sandbox as S
from creator import schedule as SCHED
from creator.ledger import Ledger


def free_ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1e9
    except ImportError:
        return 99.0


def sandbox_memory_gb(scratch: Any) -> float:
    """Resident memory of every process running inside the swarm's sandboxes (their cwd is under `scratch`)."""
    from pathlib import Path
    try:
        import psutil
    except ImportError:
        return 0.0
    root = str(Path(scratch).resolve()).lower() if scratch else ""
    if not root:
        return 0.0
    total = 0
    for proc in psutil.process_iter(["pid"]):
        try:
            if str(Path(proc.cwd()).resolve()).lower().startswith(root):
                total += proc.memory_info().rss
        except (psutil.Error, OSError):
            continue
    return total / 1e9


def stop_worker_processes(scratch: Any, package_id: str) -> int:
    """HARD pull-back: kill the processes running inside a worker's sandbox (found by their working directory), so memory is
    released now - not at the worker's next checkpoint (1 Oct: memory spiked faster than workers reached one and the host
    stopped the whole swarm). Only processes whose cwd is inside that sandbox are touched."""
    import json
    from pathlib import Path
    try:
        import psutil
    except ImportError:
        return 0
    root = Path(scratch) if scratch else None
    if root is None or not root.is_dir():
        return 0
    boxes = []
    for d in root.iterdir():
        marker = d / ".creator_sandbox.json"
        try:
            if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("label") == package_id:
                boxes.append(str(d.resolve()).lower())
        except (OSError, ValueError):
            continue
    if not boxes:
        return 0
    killed = 0
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            cwd = str(Path(proc.cwd()).resolve()).lower()
        except (psutil.Error, OSError):
            continue
        if any(cwd == b or cwd.startswith(b + "\\") or cwd.startswith(b + "/") for b in boxes):
            try:
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                killed += 1
            except psutil.Error:
                continue
    return killed


def total_ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().total) / 1e9
    except ImportError:
        return 16.0


WAITING: set[str] = set()                           # packages whose worker is waiting for the Claude session (uses no RAM)
HANDED_BACK: set[str] = set()                       # packages carrying finished work (any worker): pulled back LAST (2 Oct: a
                                                    # pull-back discarded a finished 6.5k-node activation win)


class protect_finished:
    """Wraps a kernel worker: once it returns finished work (claimed_done), its package is pulled back LAST. 1 Oct run8: Nupen's
    own student (nupen-model-v2) finished CP0049 and CP0050 and both were pulled back before measurement - only the teacher's
    finished work was protected, so a student's first real attempts were thrown away unmeasured."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.name = getattr(worker, "name", type(worker).__name__)

    def __call__(self, plan: Any, package: Any, workdir: Any) -> Any:
        res = self.worker(plan, package, workdir)
        if getattr(res, "claimed_done", False):
            with _WAITING_LOCK:
                HANDED_BACK.add(str(getattr(plan, "package_id", "")))
        return res
_WAITING_LOCK = threading.Lock()


class waiting_on_thinker:
    """Mark a package as waiting for the session while it waits: it does not hold a worker slot (1 Oct: waiting workers held
    slots while using no memory, so few agents ever ran)."""

    def __init__(self, package_id: str) -> None:
        self.package_id = package_id

    def __enter__(self) -> "waiting_on_thinker":
        with _WAITING_LOCK:
            WAITING.add(self.package_id)
        return self

    def __exit__(self, *exc: Any) -> None:
        with _WAITING_LOCK:
            WAITING.discard(self.package_id)


def user_idle_seconds() -> float:
    """Seconds since the owner last touched keyboard or mouse (Windows GetLastInputInfo). Unknown = 0 (assume the owner is
    there: the cautious answer)."""
    try:
        import ctypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32           # type: ignore[attr-defined]
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        kernel32.GetTickCount.restype = ctypes.c_uint
        return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:                                                   # noqa: BLE001 - non-Windows or no desktop
        return 0.0


@dataclasses.dataclass
class Governor:
    """Use the machine's memory (owner, 1 Oct 2026, asked 4+ times; diagnosis in memory 'use-the-memory-means-change-the-rule').
    The reserve is a FRACTION of this machine's RAM (default 7%, at least 0.8 GB) instead of fixed GB numbers that ate most of
    the ~3-4 GB actually available. Another worker starts while  available - measured_worker >= floor ; below pull_fraction x floor
    the youngest worker is pulled back HARD (its processes killed), which is what keeps the host's reaper away."""
    floor_fraction: float = 0.07
    floor_min_gb: float = 0.8
    pull_fraction: float = 0.75
    critical_fraction: float = 0.35                 # below this even protected (finished) work is pulled back
    ramp_s: float = 15.0                            # a new worker/filler only after the last one's memory can show (1 Oct run9:
                                                    # 40 fillers started back to back before RAM fell, then free RAM hit 9 MB)
    per_worker_gb: float = 0.4                      # only until real worker memory has been measured
    max_workers: int = 32
    free: Callable[[], float] = free_ram_gb
    total: Callable[[], float] = total_ram_gb
    observe: Optional[Callable[[int], Optional[float]]] = None
    user_active_floor_fraction: Optional[float] = None  # owner, 1 Oct: "when i start doing things it adjusts how much RAM it can
    idle_after_s: float = 300.0                         # use" - while the owner is at the keyboard keep this larger share free
    idle: Optional[Callable[[], float]] = None          # seconds since the owner's last input (None = user_idle_seconds)

    def floor(self) -> float:
        frac = self.floor_fraction
        if self.user_active_floor_fraction is not None and (self.idle or user_idle_seconds)() < self.idle_after_s:
            frac = max(frac, self.user_active_floor_fraction)
        return max(self.floor_min_gb, frac * self.total())

    def estimate(self, running: int) -> float:
        seen = self.observe(running) if (self.observe is not None and running) else None
        return max(0.1, 1.25 * seen) if seen else self.per_worker_gb

    def can_start(self, running: int) -> bool:
        return running < self.max_workers and self.free() - self.estimate(running) >= self.floor()

    def too_tight(self) -> bool:
        return self.free() < self.pull_fraction * self.floor()

    def critical(self) -> bool:
        """So tight that even finished work may be pulled back (the host's reaper is near)."""
        return self.free() < self.critical_fraction * self.floor()


@dataclasses.dataclass
class _Running:
    plan: Any
    thread: threading.Thread
    cancel: threading.Event
    started: float
    result: list[K.CycleReport] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RoundReport:
    outcome: str                                    # WORKED / NOTHING_TO_DO / AUDIT_RED / RAM_TIGHT
    reports: list[K.CycleReport]
    peak_parallel: int
    pulled_back: int
    reason: str = ""


def plan_scheduled(cfg: K.KernelConfig, led: Ledger, main: Any, base_sha: str, gov: Governor, load: int, cap: int,
                   held: list[str], held_paths: list[str], planned: int) -> Optional[list[Any]]:
    """One scheduling decision for run_round: as many plans as there are free slots AND the governor allows, chosen together by
    creator.schedule so no two share a component or file. Gap work comes from schedule.plan_batch; shrink/coverage work
    (plan_efficiency) fills the rest in 'auto' mode and is the fallback when the batch is empty. Returns None only for
    'use the old path' (efficiency-only mode has nothing to schedule; a scheduler failure is recorded, not hidden)."""
    if cfg.mode == "efficiency":
        return None
    slots = 0
    est, free, floor = gov.estimate(load), gov.free(), gov.floor()
    while slots < cap and gov.can_start(load + slots) and free - (slots + 1) * est >= floor:   # each planned worker eats its share
        slots += 1
    if slots == 0:
        return []
    held_files = list(held_paths)
    plans: list[Any] = []
    gap_slots = slots - 1 if (cfg.mode == "auto" and slots >= 2) else slots    # keep one slot for shrink work alongside gaps
    try:
        plans = list(SCHED.plan_batch(dataclasses.replace(cfg, mode="gaps"), led, main, base_sha, gap_slots, held, held_files))
    except Exception as e:                                              # noqa: BLE001 - never lose a round to the scheduler
        SCHED.note(cfg.ledger_path, {"error": f"{type(e).__name__}: {e}"[:300]})
        return None
    seen = {p.component for p in plans}
    if cfg.mode == "auto" and len(plans) < slots:
        eff = K.plan_one(dataclasses.replace(cfg, mode="efficiency"), led, main, base_sha,
                         exclude_components=held + sorted(seen), exclude_paths=held_paths)
        if eff is not None:
            plans.append(eff)
    return plans[:slots]


def run_round(cfg: K.KernelConfig, make_worker: Callable[[], Any], governor: Optional[Governor] = None,
              max_packages: int = 8, poll_s: float = 2.0, on_report: Optional[Callable[[K.CycleReport], None]] = None,
              filler: Optional[Callable[[], Optional[Callable[[], None]]]] = None, filler_budget: int = 0,
              scheduled: bool = True) -> RoundReport:
    """`scheduled` (default): gap work is chosen as a batch by creator.schedule (dependency graph, critical path, no two plans on
    one component/file); False keeps the old one-package-at-a-time K.plan_one path."""
    gov = governor or Governor()
    scratch_dir = cfg.scratch or cfg.repo.parent / f".{cfg.repo.name}_creator_sandboxes"
    if gov.observe is None:                                             # measure what workers really use
        mem_peak = {"gb": 0.0}

        def observe(running: int) -> Optional[float]:
            per = sandbox_memory_gb(scratch_dir) / max(1, running)
            mem_peak["gb"] = max(mem_peak["gb"], per)
            return mem_peak["gb"] or None
        gov.observe = observe
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    GOALS.maybe_propose(led, Path(cfg.ledger_path).parent, cfg.repo)      # at most daily; proposals are not work until approved
    main, recovered, stop = K.prepare(cfg, led)
    if stop is not None or main is None:
        return RoundReport("AUDIT_RED", [], 0, 0, stop or "")
    base_sha = S.head(cfg.repo)
    lock = threading.Lock()
    running: list[_Running] = []
    reports: list[K.CycleReport] = []
    peak = pulled = planned = 0
    exhausted = False
    fillers: list[threading.Thread] = []
    fill_left = filler_budget
    starved = False
    last_start = -1e9
    nothing_while: Optional[tuple[str, ...]] = None                     # running set for which planning found nothing

    queue: list[Any] = []                                               # planned, not yet started (the ramp spaces the starts)

    def start(plan: Any) -> None:
        nonlocal last_start, peak
        ev = threading.Event()
        box: list[K.CycleReport] = []

        def job() -> None:
            own = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
            try:
                box.append(K.execute(cfg, protect_finished(make_worker()), plan, main, base_sha, len(reports) + 1, own, recovered,
                                     lock=lock, cancel=ev, checkpoint=False))
            except Exception as e:                                      # noqa: BLE001 - a dead job thread must leave a report, not a hole
                why = f"{type(e).__name__}: {e}"
                try:
                    if own.view.status[plan.work_package_id] not in (M.Status.FAILED, M.Status.IMPLEMENTED):
                        K.P.record_outcome(own, plan, False, f"swarm job crashed: {why}"[:500])
                except Exception:                                       # noqa: BLE001 - the report below is what matters
                    pass
                box.append(K.CycleReport(len(reports) + 1, "ERROR", plan.package_id, plan.requirement_key, reason=why))
        t = threading.Thread(target=job, name=f"swarm-{plan.package_id}", daemon=True)
        running.append(_Running(plan, t, ev, time.monotonic(), box))
        t.start()
        last_start = time.monotonic()
        peak = max(peak, len(running) + sum(1 for f in fillers if f.is_alive()))

    def finish(r: _Running) -> None:
        if r.result:
            reports.append(r.result[0])
            if on_report:
                on_report(r.result[0])

    while True:
        for r in [r for r in running if not r.thread.is_alive()]:
            running.remove(r)
            finish(r)
        key = tuple(sorted(r.plan.package_id for r in running))
        active = [r for r in running if r.plan.package_id not in WAITING]
        fillers[:] = [f for f in fillers if f.is_alive()]
        load = len(active) + len(fillers)                               # fillers use memory too (1 Oct run9: uncounted, peak 42 > 32)
        ramped = time.monotonic() - last_start >= gov.ramp_s
        if gov.too_tight() and active:
            pool = [r for r in active if not r.cancel.is_set()]
            cheap = [r for r in pool if r.plan.package_id not in HANDED_BACK]
            if not cheap and gov.critical():
                cheap = pool                                            # finished work goes only when the host is in danger
            youngest = max(cheap, key=lambda r: r.started, default=None)
            if youngest is not None:
                youngest.cancel.set()                               # pull it back: stop its processes now, not at a checkpoint
                stop_worker_processes(cfg.scratch or cfg.repo.parent / f".{cfg.repo.name}_creator_sandboxes",
                                      youngest.plan.package_id)
                pulled += 1
        elif queue and ramped and gov.can_start(load):                  # a planned package starts once the ramp allows
            start(queue.pop(0))
            continue
        elif (not queue and not exhausted and planned < max_packages and ramped and gov.can_start(load)
              and nothing_while != key):
            with lock:
                held = [r.plan.component for r in running]
                held_paths = [r.plan.component for r in running if r.plan.step in K.P.EFFICIENCY_STEPS]
                plans = plan_scheduled(cfg, led, main, base_sha, gov, load, min(max_packages - planned, gov.max_workers),
                                       held, held_paths, planned) if scheduled else None
                if plans is None:                                       # flag off, or the scheduler failed: the old path
                    plans = []
                    order = [cfg]
                    if cfg.mode == "auto":                              # shrink work runs ALONGSIDE gap work, not only after it
                        order = [dataclasses.replace(cfg, mode="efficiency"), dataclasses.replace(cfg, mode="gaps")]
                        if planned % 2 == 0:
                            order.reverse()
                    for c in order:
                        plan = K.plan_one(c, led, main, base_sha, exclude_components=held, exclude_paths=held_paths)
                        if plan is not None:
                            plans = [plan]
                            break
            if not plans:                                               # a target held by a running worker may free up: wait for
                if running:                                             # the running set to change before planning again (1 Oct:
                    nothing_while = key   # one held target ended the round at 2)
                else:
                    exhausted = True
            else:
                planned += len(plans)
                queue.extend(plans)
                continue
        if ((exhausted or planned >= max_packages or nothing_while == key) and filler is not None and fill_left > 0 and ramped and not gov.too_tight()
                and gov.can_start(load)):
            job_fn = filler()                                           # leftover memory: useful measurement work
            if job_fn is not None:
                fill_left -= 1
                ft = threading.Thread(target=job_fn, name="swarm-filler", daemon=True)
                fillers.append(ft)
                ft.start()
                last_start = time.monotonic()
                peak = max(peak, len(running) + sum(1 for f in fillers if f.is_alive()))
                continue
        fillers[:] = [f for f in fillers if f.is_alive()]
        if not running and not fillers and not queue and (exhausted or planned >= max_packages):
            break
        if not running and not fillers and not queue and not gov.can_start(0):        # too tight to start anything: end the round so the
            starved = True                                              # runner records it and retries later (never a silent
            break                                                       # wait forever - found by the tight-RAM test, 2 Oct)
        time.sleep(poll_s)
    Ledger(cfg.ledger_path, evidence_root=cfg.repo).checkpoint(f"swarm round: {len(reports)} packages, peak {peak} parallel")
    outcome = "WORKED" if reports else ("RAM_TIGHT" if starved else "NOTHING_TO_DO")
    return RoundReport(outcome, reports, peak, pulled, "free RAM below the reserve; nothing could start" if starved else "")



def self_bench_filler(store: Any, tasks: Optional[list[Any]] = None) -> Callable[[], Optional[Callable[[], None]]]:
    """Filler work for leftover memory: benchmark the system's OWN search worker on dev tasks it has not measured for the current
    code (holdout never touched). Results append to `store` - real data on what the system can already do by itself."""
    import json
    from pathlib import Path
    from creator import devbench as D
    from creator import generator as G
    store = Path(store)
    code = hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:12]
    done = set()
    if store.is_file():
        for ln in store.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(ln)
                if row.get("code") == code:
                    done.add(row["task"])
            except ValueError:
                continue
    todo = [t for t in (tasks or D.load_tasks()) if t.split == "dev" and t.id not in done]
    manifest = D.load_manifest()
    write_lock = threading.Lock()

    def next_job() -> Optional[Callable[[], None]]:
        if not todo:
            return None
        task = todo.pop(0)

        def job() -> None:
            try:
                s = D.run_task(task, G.SearchSolver(), manifest, solver_name="self-search")
                row = {"task": task.id, "category": task.category, "outcome": s.outcome, "seconds": s.seconds, "code": code}
            except Exception as e:                                      # noqa: BLE001 - a filler never stops the swarm
                row = {"task": task.id, "outcome": "ERROR", "error": f"{type(e).__name__}: {e}", "code": code}
            with write_lock:
                store.parent.mkdir(parents=True, exist_ok=True)
                with store.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        return job
    return next_job
