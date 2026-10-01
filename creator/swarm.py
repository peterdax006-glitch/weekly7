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
import threading
import time
from typing import Any, Callable, Optional

from creator import kernel as K
from creator import sandbox as S
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


@dataclasses.dataclass
class Governor:
    """The first worker starts when free RAM >= start_gb. Each further worker needs free RAM >= low_gb + headroom_gb + the
    MEASURED memory of an average running worker (`observe`; `per_worker_gb` until something is measured) - free RAM already
    reflects the running workers, so they are not subtracted again (1 Oct: doing so kept the swarm at one worker while 5 workers
    really used ~0.3 GB each). Below low_gb the youngest worker is pulled back hard."""
    start_gb: float = 2.5
    per_worker_gb: float = 0.5
    low_gb: float = 2.0
    max_workers: int = 4
    free: Callable[[], float] = free_ram_gb
    headroom_gb: float = 0.5
    observe: Optional[Callable[[int], Optional[float]]] = None

    def estimate(self, running: int) -> float:
        seen = self.observe(running) if (self.observe is not None and running) else None
        return max(0.15, 1.25 * seen) if seen else self.per_worker_gb

    def can_start(self, running: int) -> bool:
        if running >= self.max_workers:
            return False
        free = self.free()
        if running == 0:
            return free >= self.start_gb
        return free >= self.low_gb + self.headroom_gb + self.estimate(running)

    def too_tight(self) -> bool:
        return self.free() < self.low_gb


@dataclasses.dataclass
class _Running:
    plan: Any
    thread: threading.Thread
    cancel: threading.Event
    started: float
    result: list[K.CycleReport] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RoundReport:
    outcome: str                                    # WORKED / NOTHING_TO_DO / AUDIT_RED
    reports: list[K.CycleReport]
    peak_parallel: int
    pulled_back: int
    reason: str = ""


def run_round(cfg: K.KernelConfig, make_worker: Callable[[], Any], governor: Optional[Governor] = None,
              max_packages: int = 8, poll_s: float = 2.0, on_report: Optional[Callable[[K.CycleReport], None]] = None) -> RoundReport:
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
    main, recovered, stop = K.prepare(cfg, led)
    if stop is not None or main is None:
        return RoundReport("AUDIT_RED", [], 0, 0, stop or "")
    base_sha = S.head(cfg.repo)
    lock = threading.Lock()
    running: list[_Running] = []
    reports: list[K.CycleReport] = []
    peak = pulled = planned = 0
    exhausted = False

    def finish(r: _Running) -> None:
        if r.result:
            reports.append(r.result[0])
            if on_report:
                on_report(r.result[0])

    while True:
        for r in [r for r in running if not r.thread.is_alive()]:
            running.remove(r)
            finish(r)
        if gov.too_tight() and running:
            youngest = max((r for r in running if not r.cancel.is_set()), key=lambda r: r.started, default=None)
            if youngest is not None:
                youngest.cancel.set()                               # pull it back: stop its processes now, not at a checkpoint
                stop_worker_processes(cfg.scratch or cfg.repo.parent / f".{cfg.repo.name}_creator_sandboxes",
                                      youngest.plan.package_id)
                pulled += 1
        elif not exhausted and planned < max_packages and gov.can_start(len(running)):
            with lock:
                plan = K.plan_one(cfg, led, main, base_sha,
                                  exclude_components=[r.plan.component for r in running],
                                  exclude_paths=[r.plan.component for r in running if r.plan.step == "efficiency"])
            if plan is None:
                exhausted = True
            else:
                planned += 1
                ev = threading.Event()
                box: list[K.CycleReport] = []

                def job(plan: Any = plan, ev: threading.Event = ev, box: list[K.CycleReport] = box) -> None:
                    own = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
                    box.append(K.execute(cfg, make_worker(), plan, main, base_sha, len(reports) + 1, own, recovered,
                                         lock=lock, cancel=ev, checkpoint=False))
                t = threading.Thread(target=job, name=f"swarm-{plan.package_id}", daemon=True)
                running.append(_Running(plan, t, ev, time.monotonic(), box))
                t.start()
                peak = max(peak, len(running))
                continue
        if not running and (exhausted or planned >= max_packages):
            break
        time.sleep(poll_s)
    Ledger(cfg.ledger_path, evidence_root=cfg.repo).checkpoint(f"swarm round: {len(reports)} packages, peak {peak} parallel")
    return RoundReport("WORKED" if reports else "NOTHING_TO_DO", reports, peak, pulled)
