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

import json
import os
import statistics
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


if __name__ == "__main__":                                # python -m creator.slowpath [state]
    import sys
    st = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "state" / "creator"
    for r in report(st):
        print(json.dumps(r))
