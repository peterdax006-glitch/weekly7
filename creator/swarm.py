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
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from creator import registry as REG                                   # goals/constraints load on demand (sparse activation)
from creator import kernel as K
from creator import model as M
from creator import planner as P
from creator import sandbox as S
from creator import testslots as TS
from creator.ledger import Ledger


def _free_disk_gb() -> float:
    import shutil
    return shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 2**30


def free_ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1e9
    except ImportError:
        return 99.0


def sandbox_memory_gb(scratch: Any) -> float:
    """Resident memory of every process running inside the swarm's sandboxes (their cwd is under `scratch`)."""
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
    there: the cautious answer). Linux: xprintidle, macOS: ioreg (creator/device.py)."""
    if sys.platform != "win32":
        return REG.get("device").idle_seconds_unix()
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
    burst_gb: Optional[float] = None                # filler_ramped: None = plain spacing; else what each start inside one ramp window is assumed to take
    max_workers: int = 32
    free: Callable[[], float] = free_ram_gb
    total: Callable[[], float] = total_ram_gb
    observe: Optional[Callable[[int], Optional[float]]] = None
    user_active_floor_fraction: Optional[float] = None  # owner, 1 Oct: "when i start doing things it adjusts how much RAM it can
    idle_after_s: float = 300.0                         # use" - while the owner is at the keyboard keep this larger share free
    test_parallel: int = 0                              # test processes ONE worker's evaluation runs side by side (0 = not counted)
    eval_reserve: Callable[[int, int, int], float] = TS.eval_reserve_gb   # (running, test_parallel, extra) -> GB set aside
    idle: Optional[Callable[[], float]] = None          # seconds since the owner's last input (None = user_idle_seconds)
    disk_floor_gb: float = 1.0                          # owner, 2 Oct 2026: use the storage too, "except a single GB"
    free_disk: Optional[Callable[[], float]] = None     # free GB on the repo's drive (None = shutil.disk_usage)
    admit: Any = None                                   # creator.resources.Admitter or None

    def disk_ok(self) -> bool:
        try:
            free = self.free_disk() if self.free_disk is not None else _free_disk_gb()
        except OSError:
            return True                                 # unknown: never block work on a failed measurement
        return free >= self.disk_floor_gb

    def floor(self) -> float:
        frac = self.floor_fraction
        if self.user_active_floor_fraction is not None and (self.idle or user_idle_seconds)() < self.idle_after_s:
            frac = max(frac, self.user_active_floor_fraction)
        return max(self.floor_min_gb, frac * self.total())

    def estimate(self, running: int) -> float:
        seen = self.observe(running) if (self.observe is not None and running) else None
        return max(0.1, 1.25 * seen) if seen else self.per_worker_gb

    def reservation(self, running: int, extra: int = 1) -> float:
        """Memory the evaluations of `extra` more workers will need: a worker is not started if the work it would do cannot
        finish (2 Oct: workers were admitted by their own memory, then pulled back mid-evaluation for the test processes)."""
        return self.eval_reserve(running, self.test_parallel, extra)

    def can_start(self, running: int) -> bool:
        return (running < self.max_workers and self.disk_ok()
                and self.free() - self.estimate(running) - self.reservation(running) >= self.floor())

    def filler_ramped(self, since_last: float, recent_starts: int, running: int) -> bool:
        """A filler may start before ramp_s has passed when free RAM above the floor covers every start of the last ramp window plus
        this one at a pessimistic burst_gb (or the measured per-worker estimate, if larger) each: then waiting for their memory to show
        buys nothing. 3 Oct 2026: with the 15 s spacing a drill job finished before the next could start - one thinking job at a time,
        CPU 29-65%, 19 GB free. When RAM is short the plain spacing holds (1 Oct run9: 40 heavy starts back to back, free RAM 9 MB)."""
        if since_last >= self.ramp_s:
            return True
        if self.burst_gb is None:                       # opt-in (the live swarm, thinking focus only): heavy fillers keep the spacing
            return False
        return self.free() - self.floor() >= (recent_starts + 1) * max(self.burst_gb, self.estimate(running))

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


def _focus_allows(item: Any, state: Optional[Path] = None) -> bool:
    """FOCUS HOOK (owner 2 Oct: work only on thinking until its opinion is trustworthy). When the optional `focus` capability is
    registered and present, `focus.allowed(plan, state)` decides - with THIS run's state directory (2 Oct: reading the live repo's
    focus.json made every test run in thinking focus); an absent or broken module filters nothing."""
    try:
        focus = REG.optional("focus")
        if focus is None:
            return True
        if state is not None:
            try:
                return bool(focus.allowed(item, state))
            except TypeError:                                           # a one-argument allowed(plan): ask it without the state
                pass
        return bool(focus.allowed(item))
    except Exception:                                                   # noqa: BLE001 - no focus module: no filter
        return True


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
    while slots < cap and gov.can_start(load + slots) and free - (slots + 1) * est - gov.reservation(load, slots + 1) >= floor:   # each planned worker eats its share
        slots += 1
    if slots == 0:
        return []
    from creator import schedule as SCHED                              # lazy: only a scheduled round needs it (start-load guard)
    held_files = list(held_paths)
    plans: list[Any] = []
    try:                                                                # development gaps (capacity-raising ones first) take EVERY slot
        plans = list(SCHED.plan_batch(dataclasses.replace(cfg, mode="gaps"), led, main, base_sha, slots, held, held_files))
    except Exception as e:                                              # noqa: BLE001 - never lose a round to the scheduler
        try:
            REG.get("schedule").note(cfg.ledger_path, {"error": f"{type(e).__name__}: {e}"[:300]})
        except Exception:                                               # noqa: BLE001 - a broken scheduler module is only noted
            print(f"SCHEDULER error: {type(e).__name__}: {e}", flush=True)
        return None
    kept = [p for p in plans if _focus_allows(p, Path(cfg.ledger_path).parent)]                       # FOCUS HOOK: release what the focus does not allow (not an attempt)
    for p in plans:
        if p not in kept:
            P.record_outcome(led, p, False, "interrupted: not allowed by the current focus")
    plans = kept
    seen = {p.component for p in plans}
    probe = P.Plan("", "EFF.size", "", "size", M.Role.KERNEL, "", "", "", "", 1)
    while cfg.mode == "auto" and len(plans) < slots and _focus_allows(probe, Path(cfg.ledger_path).parent):    # FOCUS HOOK: no efficiency work in thinking focus                    # efficiency work only fills what gap work could not (the old
        eff = K.plan_one(dataclasses.replace(cfg, mode="efficiency"), led, main, base_sha,   # reserved shrink slot, now earned by a shortfall)
                         exclude_components=held + sorted(seen), exclude_paths=held_paths)
        if eff is None:
            break
        plans.append(eff)
        seen.add(eff.component)
    final = plans[:slots]
    try:                                                                # FAST-PREDICTION HOOK (creator.fastpred, on demand): which of these reaches a verdict first
        fp = REG.optional("fastpred")
        if fp is not None and fp.enabled() and len(final) > 1:
            fp.plan_begin([(p.package_id, str(p.requirement_key)) for p in final])
    except Exception:                                                   # noqa: BLE001
        pass
    return final




def run_round(cfg: K.KernelConfig, make_worker: Callable[[], Any], governor: Optional[Governor] = None,
              max_packages: int = 8, poll_s: float = 2.0, on_report: Optional[Callable[[K.CycleReport], None]] = None,
              filler: Optional[Callable[[], Optional[Callable[[], None]]]] = None, filler_budget: int = 0,
              scheduled: bool = True, drain: Optional[Callable[[], bool]] = None, drain_max_s: float = 3600.0) -> RoundReport:
    """`scheduled` (default): gap work is chosen as a batch by creator.schedule (dependency graph, critical path, no two plans on
    one component/file); False keeps the old one-package-at-a-time K.plan_one path.
    `drain` (a deploy pause, 2 Oct: a force-kill lost every in-flight 35-45 min cycle): once it returns True nothing new is planned,
    started or filled; running packages finish; after `drain_max_s` they are cancelled; the round ends 'DRAINED'."""
    gov = governor or Governor()
    scratch_dir = cfg.scratch or REG.get("device").sandbox_root(cfg.repo)
    if gov.observe is None:                                             # measure what workers really use
        mem_peak = {"gb": 0.0}

        def observe(running: int) -> Optional[float]:
            per = sandbox_memory_gb(scratch_dir) / max(1, running)
            mem_peak["gb"] = max(mem_peak["gb"], per)
            return mem_peak["gb"] or None
        gov.observe = observe
    state_dir, RS = Path(cfg.ledger_path).parent, REG.get("resources")
    RS.attach(gov, state_dir, free_ram_gb)
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    REG.get("goals").maybe_propose(led, Path(cfg.ledger_path).parent, cfg.repo)      # at most daily; proposals are not work until approved
    REG.get("constraints").maybe_run(Path(cfg.ledger_path).parent)                    # at most hourly; measures what limits improvement, never raises
    lock = threading.Lock()
    running: list[_Running] = []
    reports: list[K.CycleReport] = []
    peak = pulled = planned = 0
    exhausted = False
    fillers: list[threading.Thread] = []
    fill_left = filler_budget
    last_start = -1e9
    fill_starts: list[float] = []                                       # filler start times inside the current ramp window

    def fill_ramped(load: int) -> bool:
        now = time.monotonic()
        fill_starts[:] = [t for t in fill_starts if now - t < gov.ramp_s]
        return gov.filler_ramped(now - last_start, len(fill_starts), load)

    def try_fill() -> bool:
        """Start one filler job when the governor and the resource admission allow it (leftover memory/CPU: useful measurement work)."""
        nonlocal fill_left, last_start, peak
        load = len([r for r in running if r.plan.package_id not in WAITING]) + len(fillers)
        if not (filler is not None and fill_left > 0 and not gov.too_tight() and fill_ramped(load)
                and gov.can_start(load) and RS.ok(gov, "filler", load)):
            return False
        job_fn = filler()
        if job_fn is None:
            return False
        fill_left -= 1
        job_fn = RS.idle_thread(job_fn) if gov.admit else job_fn
        ft = threading.Thread(target=job_fn, name="swarm-filler", daemon=True)
        fillers.append(ft)
        ft.start()
        last_start = time.monotonic()
        fill_starts.append(last_start)
        peak = max(peak, len(running) + sum(1 for f in fillers if f.is_alive()))
        return True

    if filler is None or fill_left <= 0:
        main, recovered, stop = K.prepare(cfg, led)
    else:                                                               # 2 Oct: prepare (recover + assess + audit) takes minutes; CPU must not
        box: list[Any] = []                                             # taper meanwhile - fillers run while it does, and nothing is PLANNED
        def prep() -> None:                                             # until it has finished (the loop below starts after the join)
            try:
                box.append(K.prepare(cfg, led))
            except BaseException as e:                                  # noqa: BLE001 - re-raised in the round's own thread
                box.append(e)
        pt = threading.Thread(target=prep, name="swarm-prepare", daemon=True)
        pt.start()
        while pt.is_alive():
            fillers[:] = [f for f in fillers if f.is_alive()]
            if gov.admit is not None:
                REG.get("modelpool").tick(gov, free_ram_gb)
            if not try_fill():
                pt.join(poll_s)
        if isinstance(box[0], BaseException):
            raise box[0]
        main, recovered, stop = box[0]
    if stop is not None or main is None:
        while any(f.is_alive() for f in fillers):                       # a red audit: let the started drills finish, start no more
            time.sleep(poll_s)
        return RoundReport("AUDIT_RED", [], 0, 0, stop or "")
    base_sha = S.head(cfg.repo)
    starved = False
    nothing_while: Optional[tuple[str, ...]] = None                     # running set for which planning found nothing

    queue: list[Any] = []                                               # planned, not yet started (the ramp spaces the starts)
    status_at = [time.monotonic()]                                     # first STATUS after a minute: never disturbs the start
    drain_t0: Optional[float] = None

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
        fill_starts.append(last_start)
        peak = max(peak, len(running) + sum(1 for f in fillers if f.is_alive()))

    def finish(r: _Running) -> None:
        RS.learn(gov, state_dir, r.plan.step, r.started, len(running))
        if r.result:
            reports.append(r.result[0])
            if on_report:
                on_report(r.result[0])

    while True:
        for r in [r for r in running if not r.thread.is_alive()]:
            running.remove(r)
            finish(r)
        if drain is not None and drain():
            if drain_t0 is None:
                drain_t0 = time.monotonic()
                dropped, queue[:] = len(queue), []                      # planned but not started: never started
                print(f"DRAIN started: nothing new; {len(running)} running may finish for {drain_max_s:.0f}s; {dropped} queued dropped", flush=True)
            exhausted = True
            if time.monotonic() - drain_t0 > drain_max_s:
                for r in running:
                    if not r.cancel.is_set():
                        r.cancel.set()
                        stop_worker_processes(cfg.scratch or REG.get("device").sandbox_root(cfg.repo), r.plan.package_id)
                if time.monotonic() - drain_t0 > drain_max_s + 120.0:
                    break                                               # cancelled workers and fillers had two minutes to end
        key = tuple(sorted(r.plan.package_id for r in running))
        active = [r for r in running if r.plan.package_id not in WAITING]
        fillers[:] = [f for f in fillers if f.is_alive()]
        load = len(active) + len(fillers)                               # fillers use memory too (1 Oct run9: uncounted, peak 42 > 32)
        ramped = time.monotonic() - last_start >= gov.ramp_s
        if gov.admit is not None:                                      # RAM has room while CPU is full: keep model servers loaded for students
            REG.get("modelpool").tick(gov, free_ram_gb)
        if time.monotonic() - status_at[0] >= 60.0:                     # once a minute: what runs, and what stops more starting
            status_at[0] = time.monotonic()
            RS.lower_sandbox_priority(scratch_dir) if gov.admit else 0
            try:
                REG.get("swarmops").status_line(gov, load, running, active, queue, fillers, planned, max_packages, exhausted,
                                                nothing_while == key, fill_left, ramped or fill_ramped(load))
            except Exception as e:                                      # noqa: BLE001 - a status line never breaks a round
                print(f"STATUS error: {type(e).__name__}: {e}", flush=True)
        if gov.too_tight() and active:
            pool = [r for r in active if not r.cancel.is_set()]
            cheap = [r for r in pool if r.plan.package_id not in HANDED_BACK]
            if not cheap and gov.critical():
                cheap = pool                                            # finished work goes only when the host is in danger
            youngest = max(cheap, key=lambda r: r.started, default=None)
            if youngest is not None:
                youngest.cancel.set()                               # pull it back: stop its processes now, not at a checkpoint
                stop_worker_processes(cfg.scratch or REG.get("device").sandbox_root(cfg.repo),
                                      youngest.plan.package_id)
                pulled += 1
        elif queue and ramped and gov.can_start(load) and RS.ok(gov, f"cycle:{queue[0].step}", load):
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
        if RS.ram_thread(gov, state_dir, load, ramped, fillers):
            last_start = time.monotonic()
            fill_starts.append(last_start)
            continue
        if drain_t0 is None and (exhausted or planned >= max_packages or nothing_while == key) and try_fill():
            continue
        fillers[:] = [f for f in fillers if f.is_alive()]
        if not running and not fillers and not queue and (exhausted or planned >= max_packages):
            break
        if not running and not fillers and not gov.can_start(0):        # too tight to start anything: end the round so the
            for q in queue:                                             # runner records it and retries later (never a silent wait
                try:                                                    # forever - 2 Oct: a QUEUED plan under tight RAM looped
                    P.record_outcome(led, q, False, "interrupted: RAM too tight to start it")   # here forever; released, not an attempt
                except Exception:                                       # noqa: BLE001 - the round still ends
                    pass
            queue.clear()
            starved = True
            break
        time.sleep(poll_s)
    Ledger(cfg.ledger_path, evidence_root=cfg.repo).checkpoint(f"swarm round: {len(reports)} packages, peak {peak} parallel")
    outcome = "DRAINED" if drain_t0 is not None else "WORKED" if reports else ("RAM_TIGHT" if starved else "NOTHING_TO_DO")
    if outcome != "WORKED" and gov.admit is not None:                  # idle: give the pool's RAM back until work returns
        REG.get("modelpool").close_if_idle()
    return RoundReport(outcome, reports, peak, pulled, "free RAM below the reserve; nothing could start" if starved else "")



def self_bench_filler(store: Any, tasks: Optional[list[Any]] = None) -> Callable[[], Optional[Callable[[], None]]]:
    """Filler work for leftover memory: benchmark the system's OWN search worker on dev tasks it has not measured for the current
    code (holdout never touched). Results append to `store` - real data on what the system can already do by itself."""
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
