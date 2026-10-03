"""Swarm housekeeping, loaded on demand (sparse activation: not part of the kernel's or the swarm's start-time load).

release_orphans: work left IN_PROGRESS by a dead process is released at swarm start so it is planned again.
status_line: once a minute, what runs and the first reason no more work starts (machine under-use is read, not guessed)."""
from __future__ import annotations

import json
import time
from typing import Any

from creator import model as M
from creator.ledger import Ledger


def release_orphans(led: Ledger, why: str) -> list[str]:
    """Work packages and gaps still IN_PROGRESS when a swarm STARTS have no worker: the kernel lock guarantees no other swarm
    runs, so each is the leftover of a stopped process. Record them FAILED (the legal IN_PROGRESS -> FAILED transition, as for
    any failed attempt) so their gaps are re-planned. 2 Oct 2026: after the move to the new PC, gap K28.exists stayed IN_PROGRESS
    from the old machine - the scheduler skipped it as 'worker already on it' forever and planned no development work at all."""
    released: list[str] = []
    for kind in ("WorkPackage", "Gap"):                                 # packages first: a gap's state follows its package's
        for rid, st in list(led.view.status.items()):
            rec = led.view.by_id.get(rid)
            if st is M.Status.IN_PROGRESS and rec is not None and rec.rtype == kind:
                led.transition(rid, M.Status.FAILED, f"interrupted: {why}", M.Role.KERNEL)
                released.append(rid)
    return released


def status_line(gov: Any, load: int, running: list[Any], active: list[Any], queue: list[Any], fillers: list[Any],
                 planned: int, max_packages: int, exhausted: bool, nothing_new: bool, fill_left: int, ramped: bool) -> None:
    """One STATUS line a minute in the swarm log (owner, 2 Oct 2026: 'diagnose ... confirm its fixed and is set up to stay
    fixed'): what is running and the FIRST reason no further worker starts - so under-use of the machine is read, not guessed."""
    try:
        free, floor = gov.free(), gov.floor()
        est, res = gov.estimate(load), gov.reservation(load)
        if load >= gov.max_workers:
            limit = f"worker cap {gov.max_workers}"
        elif not gov.disk_ok():
            limit = "disk floor"
        elif free - est - res < floor:
            limit = f"RAM: free {free:.1f} - worker {est:.1f} - eval reserve {res:.1f} < floor {floor:.1f} GB"
        elif not ramped:
            limit = "ramp (spacing starts)"
        elif queue:
            limit = "none (a queued package starts next)"
        elif exhausted or (planned >= max_packages):
            limit = "no more work planned this round" + ("" if fill_left > 0 else "; filler budget used")
        elif nothing_new:
            limit = "planning found nothing new while these run"
        else:
            limit = "none (planning next)"
        print("STATUS " + json.dumps({"at": time.strftime("%H:%M:%S"), "running": len(running), "active": len(active),
                                      "waiting_for_teacher": len(running) - len(active), "queued": len(queue),
                                      "fillers": len(fillers), "planned": planned, "free_gb": round(free, 2),
                                      "floor_gb": round(floor, 2), "limit": limit}), flush=True)
    except Exception as e:                                              # noqa: BLE001 - a status line never breaks a round
        print(f"STATUS error: {type(e).__name__}: {e}", flush=True)
