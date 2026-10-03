"""Swarm housekeeping, loaded on demand (sparse activation: not part of the kernel's or the swarm's start-time load).

release_orphans: work left IN_PROGRESS by a dead process is released at swarm start so it is planned again.
status_line: once a minute, what runs and the first reason no more work starts (machine under-use is read, not guessed)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from creator import model as M
from creator.ledger import Ledger


def release_orphans(led: Ledger, why: str) -> list[str]:
    """Work packages and gaps still IN_PROGRESS when a swarm STARTS have no worker: the kernel lock guarantees no other swarm
    runs, so each is the leftover of a stopped process. Record them FAILED (the legal IN_PROGRESS -> FAILED transition, as for
    any failed attempt) so their gaps are re-planned. 2 Oct 2026: after the move to the new PC, gap K28.exists stayed IN_PROGRESS
    from the old machine - the scheduler skipped it as 'worker already on it' forever and planned no development work at all."""
    released: list[str] = []
    judged = {M.Status.IMPLEMENTED, M.Status.TESTED, M.Status.INTENDED_BEHAVIOR_VERIFIED, M.Status.VALIDATED}
    st = led.view.status

    def pkgs(gap: str) -> list[str]:
        return [c for c in led.view.children.get(gap, []) if led.view.by_id[c].rtype == "WorkPackage"]

    def fail(rid: str) -> None:
        if st.get(rid) is M.Status.NOT_STARTED:                           # never started: the legal path is via IN_PROGRESS
            led.transition(rid, M.Status.IN_PROGRESS, f"interrupted: {why}", M.Role.KERNEL)
        led.transition(rid, M.Status.FAILED, f"interrupted: {why}", M.Role.KERNEL)
        released.append(rid)

    for rid, s_ in list(st.items()):                                    # packages first: a gap's state follows its package's
        rec = led.view.by_id.get(rid)
        if s_ is M.Status.IN_PROGRESS and rec is not None and rec.rtype == "WorkPackage":
            fail(rid)
    for rid, s_ in list(st.items()):
        rec = led.view.by_id.get(rid)
        if s_ is not M.Status.IN_PROGRESS or rec is None or rec.rtype != "Gap":
            continue
        kids = pkgs(rid)
        if any(st.get(k) in judged for k in kids):                      # real work awaits judgement: gaps.sync closes the gap
            continue
        for k in kids:
            if st.get(k) is M.Status.NOT_STARTED:                       # CP0172: IN_PROGRESS gap, package never started
                fail(k)
        fail(rid)
    return released


def revalidate(repo: Path, timeout_s: float = 1800.0) -> str:
    """The independent validator's verdicts must reach the ledger or every '<K>.validated' gap stays open for ever (8 of ~13 unmet
    requirements, 2 Oct 2026). Run at swarm start, under the kernel lock (no other writer), the EXISTING reconciliation
    scripts/record_validation.py --due-only: it advances only a component whose independent report says VALIDATED, whose code is
    byte-identical to the verdict commit and whose tests pass now. It creates no verdict: a stale or missing verdict stays open
    until an independent validator re-reads the code. Returns the script's last output line (or why it did not run)."""
    try:
        p = subprocess.run([sys.executable, "scripts/record_validation.py", "--due-only"], cwd=repo, capture_output=True, text=True,
                           timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as e:
        return f"not run: {type(e).__name__}: {e}"
    return f"exit {p.returncode}: " + ((p.stdout + p.stderr).strip().splitlines() or [""])[-1][:300]


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
