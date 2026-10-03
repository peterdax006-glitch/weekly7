"""Efficiency monitor for the teacher: is Nupen busy, producing verdicts, and not wasting its time? READ-ONLY on the run's state.

    python scripts/nupen_monitor.py --once      # one health line + WASTE alerts, then exit (exit code 1 when any alert)
    python scripts/nupen_monitor.py --watch     # for a Monitor: prints only NEW alerts / cleared alerts (and one health line at start)
    python scripts/nupen_monitor.py --state D   # another state/creator directory (tests, a worktree)

Reads: swarm_service.log ('STATUS {json}' once a minute from creator/swarmops.py status_line), ledger.jsonl (plans = WorkPackage
records, verdicts = a WorkPackage's final transition that is not an 'interrupted' release), lm_train.log (tok/s), skips.jsonl,
practice_service.log, nupen_service.log, and psutil for system-wide CPU and RAM. The 10-minute CPU window lives in a small
history file OUTSIDE the state directory (--history, default in the temp dir), so --once can judge it across calls.

WASTE alerts (owner, 2 Oct 2026: never waste time, work hard, CPU busy):
  CPU_LOW         system CPU < 85% for 10 min while work exists
  NO_VERDICT      no verdict for 90 min while the swarm is up
  REPEAT_TARGET   the same target planned twice within 24 h
  SKIP_UNJUDGED   a student skipped without judged evidence (fewer than 3 judged attempts named)
  PRACTICE_IDLE   practice never run in 6 h of supervisor uptime
  TRACEBACK       a traceback in the current run's logs
  SWARM_DOWN      the supervisor log says the swarm is not running (and no NUPEN_STOP / NUPEN_DRAIN)"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
CPU_FLOOR = 85.0
CPU_WINDOW_S = 600.0
NO_VERDICT_S = 90 * 60.0
DAY_S = 24 * 3600.0
PRACTICE_S = 6 * 3600.0
MIN_JUDGED = 3
EFFICIENCY_PREFIXES = ("shrink ", "cover the untested", "load less code")
NO_WORK_LIMITS = ("no more work planned", "planning found nothing new")


def _ts(text: str) -> Optional[float]:
    """Epoch seconds of an ISO timestamp; naive means local time (the service logs), an offset is honoured (the ledger's UTC)."""
    try:
        d = dt.datetime.fromisoformat(text)
        return d.timestamp()
    except (ValueError, TypeError):
        return None


def _lines(path: Path, tail_bytes: Optional[int] = None) -> list[str]:
    try:
        with path.open("rb") as f:
            if tail_bytes is not None:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


class LedgerReader:
    """Incremental reader of ledger.jsonl: only the new tail is parsed on each refresh (a partial last line waits for the next one)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.kind: dict[str, str] = {}                                  # id -> rtype
        self.plans: list[dict[str, Any]] = []                           # WorkPackage: id, at, target, efficiency
        self.verdicts: list[dict[str, Any]] = []                        # id, at, to
        self.pkg_ids: set[str] = set()

    def refresh(self) -> None:
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.__init__(self.path)                                # type: ignore[misc]  (rotated / rewritten: start over)
            with self.path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for raw in chunk[:end].decode("utf-8", "replace").splitlines():
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            self._take(d)

    def _take(self, d: dict[str, Any]) -> None:
        rt, data, rid = d.get("rtype"), d.get("data") or {}, d.get("id", "")
        at = _ts((d.get("provenance") or {}).get("timestamp", "")) or 0.0
        self.kind[rid] = rt
        if rt == "WorkPackage":
            outs = [o for o in data.get("outputs", []) if o and o != "none"]
            obj = str(data.get("objective", ""))
            self.pkg_ids.add(rid)
            self.plans.append({"id": rid, "at": at, "target": outs[0] if outs else obj,
                               "efficiency": obj.startswith(EFFICIENCY_PREFIXES)})
        elif rt == "Transition" and data.get("subject_id") in self.pkg_ids:
            to = data.get("to_state")
            if to in ("IMPLEMENTED", "FAILED", "REJECTED", "ROLLED_BACK", "TESTED", "VALIDATED") \
                    and not str(data.get("reason", "")).startswith("interrupted"):
                self.verdicts.append({"id": data["subject_id"], "at": at, "to": to})


def read_statuses(log: Path) -> list[dict[str, Any]]:
    out = []
    for ln in _lines(log, 2_000_000):
        if ln.startswith("STATUS {"):
            try:
                out.append(json.loads(ln[7:]))
            except ValueError:
                pass
    return out


def lm_toks(log: Path) -> Optional[float]:
    vals = [float(m.group(1)) for ln in _lines(log, 100_000) for m in [re.search(r"tok/s ([\d.]+)", ln)] if m]
    last = vals[-5:]
    return sum(last) / len(last) if last else None


def service_state(svc_log: Path) -> dict[str, Any]:
    """From nupen_service.log: when the supervisor and swarm last started, and whether the swarm is currently up."""
    up_at = swarm_at = None
    swarm_up = False
    for ln in _lines(svc_log, 400_000):
        t = _ts(ln[:19])
        if " supervisor up " in ln:
            up_at, swarm_up = t, False
        elif " swarm started " in ln:
            swarm_at, swarm_up = t, True
        elif " swarm exited " in ln or " supervisor down" in ln:
            swarm_up = False
    return {"supervisor_up_at": up_at, "swarm_at": swarm_at, "swarm_up": swarm_up}


def tracebacks(state: Path) -> list[str]:
    """Tracebacks in the CURRENT run: swarm_service.log after its last 'device:' line (printed at every swarm start), and the tails of
    the helper logs. Each is described by its final exception line."""
    found = []
    sw = _lines(state / "swarm_service.log", 3_000_000)
    start = max((i for i, ln in enumerate(sw) if ln.startswith("device:")), default=0)
    svc = _lines(state / "nupen_service.log", 400_000)
    up = max((i for i, ln in enumerate(svc) if " supervisor up " in ln), default=0)
    groups = [("swarm", sw[start:]), ("service", svc[up:])] + [(n, _lines(state / f, 60_000)) for n, f in
                                                              (("lm_train", "lm_train.log"), ("practice", "practice_run.log"))]
    for name, lines in groups:
        for i, ln in enumerate(lines):
            if ln.startswith("Traceback") or " Traceback" in ln[:40]:
                j = i + 1
                while j < len(lines) and (lines[j].startswith((" ", "\t")) or lines[j].startswith("Traceback")):
                    j += 1
                found.append(f"{name}: {(lines[j] if j < len(lines) else lines[-1]).strip()[:160]}")
    return found


def skips(state: Path, now: float) -> list[str]:
    """Skips of a student recorded in the last 24 h whose reason does not name enough JUDGED attempts (N/M attempts, M >= 3)."""
    bad = []
    for ln in _lines(state / "skips.jsonl"):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        at = _ts(d.get("at", ""))
        if at is not None and now - at > DAY_S:
            continue
        m = re.search(r"(\d+)/(\d+) attempts", str(d.get("why", "")))
        if not m or int(m.group(2)) < MIN_JUDGED:
            bad.append(f"{d.get('solver')} on {d.get('task_kind')}: {str(d.get('why', ''))[:100]!r}")
    return bad


class CpuHistory:
    """(epoch, percent) samples kept in a small JSON file so separate --once calls see the last 10 minutes."""

    def __init__(self, path: Path, window_s: float = CPU_WINDOW_S) -> None:
        self.path, self.window_s = path, window_s
        self.samples: list[tuple[float, float]] = []
        try:
            self.samples = [(float(a), float(b)) for a, b in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            pass

    def add(self, now: float, pct: float) -> None:
        self.samples = [(t, p) for t, p in self.samples if now - t <= self.window_s * 2] + [(now, pct)]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.samples), encoding="utf-8")
        except OSError:
            pass

    def low_for_window(self, now: float) -> Optional[float]:
        """Mean CPU over the window when every sample in it is below the floor and the samples span (almost) all of it."""
        win = [(t, p) for t, p in self.samples if now - t <= self.window_s]
        if len(win) < 2 or win[-1][0] - win[0][0] < self.window_s * 0.9 or any(p >= CPU_FLOOR for _, p in win):
            return None
        return sum(p for _, p in win) / len(win)


def work_exists(status: Optional[dict[str, Any]]) -> bool:
    if not status:
        return False
    if status.get("running", 0) or status.get("queued", 0):
        return True
    return not str(status.get("limit", "")).startswith(NO_WORK_LIMITS)


def evaluate(state: Path, now: float, cpu_pct: Optional[float], ram: tuple[float, float], hist: Optional[CpuHistory],
             ledger: LedgerReader) -> tuple[str, list[tuple[str, str]]]:
    """(health line, [(alert code, message)])."""
    ledger.refresh()
    statuses = read_statuses(state / "swarm_service.log")
    st = statuses[-1] if statuses else None
    svc = service_state(state / "nupen_service.log")
    alerts: list[tuple[str, str]] = []
    h1, h24 = now - 3600, now - DAY_S
    verd1 = sum(1 for v in ledger.verdicts if v["at"] >= h1)
    verd24 = sum(1 for v in ledger.verdicts if v["at"] >= h24)
    plans24 = [p for p in ledger.plans if p["at"] >= h24]
    plans1 = sum(1 for p in plans24 if p["at"] >= h1)
    eff = sum(1 for p in plans24 if p["efficiency"])
    dev_share = f"dev {round(100 * (len(plans24) - eff) / len(plans24))}% eff {round(100 * eff / len(plans24))}%" if plans24 else "dev -- eff --"
    toks = lm_toks(state / "lm_train.log")
    stopped = (state / "NUPEN_STOP").exists() or (state / "NUPEN_DRAIN").exists()

    if cpu_pct is not None and hist is not None:
        hist.add(now, cpu_pct)
        low = hist.low_for_window(now)
        if low is not None and work_exists(st):
            alerts.append(("CPU_LOW", f"system CPU averaged {low:.0f}% (< {CPU_FLOOR:.0f}%) for {CPU_WINDOW_S / 60:.0f} min while work exists"
                                      f" (limit: {st.get('limit') if st else '?'})"))
    last_v = max((v["at"] for v in ledger.verdicts), default=0.0)
    since = max(last_v, svc["swarm_at"] or 0.0) if svc["swarm_up"] else None
    if since is not None and since and now - since > NO_VERDICT_S:
        alerts.append(("NO_VERDICT", f"no verdict for {(now - since) / 60:.0f} min (last at "
                                     f"{dt.datetime.fromtimestamp(last_v).strftime('%d %H:%M') if last_v else 'never'}, swarm up since "
                                     f"{dt.datetime.fromtimestamp(svc['swarm_at']).strftime('%H:%M') if svc['swarm_at'] else '?'})"))
    by_target: dict[str, list[dict[str, Any]]] = {}
    for p in plans24:
        by_target.setdefault(p["target"], []).append(p)
    rep = sorted(((len(ps), t) for t, ps in by_target.items() if len(ps) >= 2), reverse=True)
    if rep:                                                             # one line, worst first (25 lines of it hid everything else)
        alerts.append(("REPEAT_TARGET", f"{len(rep)} targets planned 2+ times in 24 h ({sum(n for n, _ in rep)} plans); worst: "
                                        + ", ".join(f"{t} {n}x" for n, t in rep[:4])))
    for s in skips(state, now):
        alerts.append(("SKIP_UNJUDGED", f"student skipped without judged evidence: {s}"))
    plog = state / "practice_service.log"
    ran = any("practice started" in ln for ln in _lines(plog, 500_000))
    up_at = svc["supervisor_up_at"]
    if not ran and up_at and now - up_at > PRACTICE_S:
        alerts.append(("PRACTICE_IDLE", f"practice has not run in {(now - up_at) / 3600:.1f} h of supervisor uptime"))
    for t in tracebacks(state):
        alerts.append(("TRACEBACK", t))
    if not svc["swarm_up"] and svc["supervisor_up_at"] is not None and not stopped:
        alerts.append(("SWARM_DOWN", "the supervisor log says no swarm is running (and no NUPEN_STOP / NUPEN_DRAIN)"))

    free, total = ram
    run = f"run {st.get('running')} q {st.get('queued')} wait {st.get('waiting_for_teacher')} limit \"{str(st.get('limit'))[:60]}\"" if st \
        else "no STATUS yet"
    health = (f"HEALTH {dt.datetime.fromtimestamp(now).strftime('%H:%M')} cpu {'--' if cpu_pct is None else f'{cpu_pct:.0f}%'} "
              f"ram {total - free:.1f}/{total:.1f}G | {run} | verdicts {verd1}/h {verd24}/24h plans {plans1}/h {dev_share} | "
              f"lm {'--' if toks is None else f'{toks:.0f}'} tok/s | {'STOPPED ' if stopped else ''}alerts {len(alerts)}")
    return health, alerts


def sample_machine() -> tuple[float, tuple[float, float]]:
    import psutil
    vm = psutil.virtual_memory()
    return psutil.cpu_percent(interval=1.0), (vm.available / 2**30, vm.total / 2**30)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--watch", action="store_true")
    ap.add_argument("--state", type=Path, default=ROOT / "state" / "creator")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--history", type=Path, default=Path(tempfile.gettempdir()) / "nupen_monitor_cpu.json")
    a = ap.parse_args(list(argv))
    hist = CpuHistory(a.history)
    ledger = LedgerReader(a.state / "ledger.jsonl")
    if a.once:
        cpu, ram = sample_machine()
        health, alerts = evaluate(a.state, time.time(), cpu, ram, hist, ledger)
        print(health)
        for code, msg in alerts:
            print(f"WASTE {code}: {msg}")
        return 1 if alerts else 0
    seen: set[tuple[str, str]] = set()
    first = True
    while True:
        cpu, ram = sample_machine()
        health, alerts = evaluate(a.state, time.time(), cpu, ram, hist, ledger)
        if first:
            print(health, flush=True)
            first = False
        now_set = {(c, m if c in ("SKIP_UNJUDGED", "TRACEBACK") else "") for c, m in alerts}   # numbers that drift are not news
        for (code, msg), full in zip(((c, m if c in ("SKIP_UNJUDGED", "TRACEBACK") else "") for c, m in alerts), alerts):
            if (code, msg) not in seen:
                print(f"WASTE {full[0]}: {full[1]}", flush=True)
        for code in sorted({c for c, _ in seen} - {c for c, _ in now_set}):
            print(f"OK {code} cleared", flush=True)
        seen = now_set
        time.sleep(max(1.0, a.interval - 1.0))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
