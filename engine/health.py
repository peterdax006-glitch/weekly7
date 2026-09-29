"""Bible Phase 24 (canons C11, C19): worker health for parallel research.

A worker writes an append-only JSONL log of events: start, config, window, seed, memory, heartbeat, complete, crash,
timeout, oom, invalid. `classify()` turns a log (plus the result file, if any) into exactly one status. A worker that
is not OK is EXCLUDED - never silently replaced, never reused from a stale run, never marked successful.

  supervise()        run a command with a wall-clock timeout, a memory ceiling and a heartbeat requirement
  classify()         one status per worker, fail closed (anything unproven is not OK)
  validate_result()  finite numbers, required keys, and the result must be for the config/window/seed it claims
  exclusion_report() the list of excluded workers and why, plus duplicate-result and stale-code detection"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

OK, CRASHED, TIMEOUT, OOM, INVALID, HUNG, INCOMPLETE, STALE = (
    "OK", "CRASHED", "TIMEOUT", "OOM", "INVALID", "HUNG", "INCOMPLETE", "STALE_CODE")
REQUIRED_RESULT = ("window", "seed", "weekly_returns")
EVENTS = ("start", "config", "window", "seed", "memory", "heartbeat", "complete", "crash", "timeout", "oom", "invalid")


# ------------------------------------------------------------------ worker side
class WorkerLog:
    """Append-only event log. Used by the worker itself; each line is flushed so a hard kill loses nothing."""

    def __init__(self, path, worker_id):
        self.path, self.id = Path(path), worker_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **kw):
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}")
        rec = {"t": time.time(), "worker": self.id, "event": event, **kw}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def begin(self, config, window, seed, code_hash=None, code=None):
        """`code` is engine.provenance.code_stamp(): code_hash, code_files, code_mixed of what this process loaded."""
        self.emit("start", **{"pid": os.getpid(), "code_hash": code_hash, **(code or {})})
        self.emit("config", config=config)
        self.emit("window", window=window)
        self.emit("seed", seed=seed)
        self.beat()

    def beat(self, note=""):
        self.emit("heartbeat", note=note)
        self.emit("memory", rss_mb=rss_mb(os.getpid()))

    def done(self, result_path, code=None):
        """Completion, with the code stamp re-taken NOW so a file edited during the run shows up as code_mixed."""
        self.emit("complete", result=str(result_path), **(code or {}))


def read_log(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"event": "_corrupt", "raw": line[:80], "t": float("nan")})   # a torn last line is evidence
    return rows


# ------------------------------------------------------------------ memory
def rss_mb(pid):
    """Resident set size of `pid` in MB, or None when it cannot be read (never a fake 0)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000 | 0x0400, False, int(pid))   # QUERY_LIMITED_INFORMATION | QUERY_INFORMATION
            if not h:
                h = k32.OpenProcess(0x1000, False, int(pid))
            if not h:
                return None
            try:
                c = PMC()
                c.cb = ctypes.sizeof(c)
                ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
                if not ps.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                    return None
                return c.WorkingSetSize / 2 ** 20
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/{int(pid)}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:   # noqa: BLE001 - unreadable memory is reported as None, not guessed
        return None
    return None


# ------------------------------------------------------------------ supervisor
def supervise(cmd, log_path, worker_id, timeout_s, mem_limit_mb, heartbeat_s=None, poll_s=0.2, env=None, cwd=None,
              stdout=None):
    """Run `cmd` as a child. Kills it and records the reason when it exceeds `timeout_s`, its RSS exceeds
    `mem_limit_mb`, or (when `heartbeat_s` is set) it goes that long without a heartbeat line. The supervisor writes
    its verdict events into the same log with worker=supervisor so the worker cannot overwrite them.
    Output goes to files beside the log (<log>.out, <log>.err), never to a pipe nobody drains: a chatty worker on a
    full pipe would block and look hung. `stdout="inherit"` lets the worker print to the caller's console (stderr
    still goes to the .err file so a crash records its tail).
    Returns (returncode or None, reason) where reason in {exit, timeout, oom, hung}."""
    sup = WorkerLog(log_path, "supervisor")
    lp = Path(log_path)
    t0 = time.time()
    fout = None if stdout == "inherit" else open(lp.with_suffix(".out"), "wb")
    ferr = open(lp.with_suffix(".err"), "wb")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fout, stderr=ferr)
    peak, reason = 0.0, "exit"
    last_beat = t0
    while p.poll() is None:
        time.sleep(poll_s)
        now = time.time()
        m = rss_mb(p.pid)
        if m is not None:
            peak = max(peak, m)
        if now - t0 > timeout_s:
            reason = "timeout"
        elif m is not None and m > mem_limit_mb:
            reason = "oom"
        elif heartbeat_s:
            beats = [r["t"] for r in read_log(log_path) if r.get("worker") == worker_id and r.get("event") == "heartbeat"]
            last_beat = max([last_beat] + beats)
            if now - last_beat > heartbeat_s:
                reason = "hung"
        if reason != "exit":
            p.kill()
            p.wait()
            break
    for f in (fout, ferr):
        if f is not None:
            f.close()
    err = lp.with_suffix(".err").read_bytes()
    if reason == "timeout":
        sup.emit("timeout", waited_s=round(time.time() - t0, 2), limit_s=timeout_s)
    elif reason == "oom":
        sup.emit("oom", rss_mb=peak, limit_mb=mem_limit_mb)
    elif reason == "hung":
        sup.emit("timeout", waited_s=round(time.time() - t0, 2), limit_s=heartbeat_s, kind="heartbeat")
    elif p.returncode != 0:
        sup.emit("crash", returncode=p.returncode, stderr_tail=err.decode("utf-8", "replace")[-400:])
    sup.emit("memory", rss_mb=peak, peak=True)
    return (p.returncode if reason == "exit" else None), reason


# ------------------------------------------------------------------ validation
def _walk_numbers(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk_numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _walk_numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield x


def validate_result(result, claimed_window=None, claimed_seed=None, min_weeks=1, max_abs_weekly=1.5):
    """Returns a list of problems (empty = valid). Catches: not a dict, missing keys, NaN/inf anywhere, a result for a
    different window or seed than the worker was given, no weeks, a weekly return that is impossible (< -100% or far
    above +150%), and an all-identical return vector (a stuck or copied result)."""
    if not isinstance(result, dict):
        return [f"result is {type(result).__name__}, not a dict"]
    bad = [f"missing key {k!r}" for k in REQUIRED_RESULT if k not in result]
    if bad:
        return bad
    bad = []
    nonfinite = [x for x in _walk_numbers(result) if not math.isfinite(x)]
    if nonfinite:
        bad.append(f"{len(nonfinite)} non-finite numbers in result")
    if claimed_window is not None and result["window"] != claimed_window:
        bad.append(f"result is for window {result['window']!r}, worker was given {claimed_window!r}")
    if claimed_seed is not None and result["seed"] != claimed_seed:
        bad.append(f"result carries seed {result['seed']!r}, worker was given {claimed_seed!r}")
    wr = result["weekly_returns"]
    vals = list(wr.values()) if isinstance(wr, dict) else list(wr)
    if len(vals) < min_weeks:
        bad.append(f"only {len(vals)} weekly returns (need {min_weeks})")
    fin = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if any(v < -1 or v > max_abs_weekly for v in fin):
        bad.append("weekly return outside [-100%, +150%]")
    if len(fin) >= 8 and len(set(round(v, 12) for v in fin)) == 1:
        bad.append("every weekly return identical: stuck or copied result")
    return bad


# ------------------------------------------------------------------ classification
def classify(log_rows, result=None, expected_config=None, expected_window=None, expected_seed=None,
             current_code_hash=None, heartbeat_s=None, now=None, stale_check=None):
    """One status per worker log. Precedence, most damning first: OOM, TIMEOUT, CRASHED, HUNG, INCOMPLETE, INVALID,
    STALE_CODE, OK. OK requires: start+config+window+seed all reported, a complete event, a result that validates for
    that window and seed, a config that matches what was asked, and (when given) the code that is on disk."""
    ev = {}
    for r in log_rows:
        ev.setdefault(r.get("event"), []).append(r)
    if "oom" in ev:
        return OOM, f"rss {ev['oom'][-1].get('rss_mb', 0):.0f} MB over limit"
    if "timeout" in ev:
        return TIMEOUT, "exceeded wall-clock or heartbeat limit"
    if "crash" in ev:
        return CRASHED, f"return code {ev['crash'][-1].get('returncode')}"
    if "start" not in ev:
        return INCOMPLETE, "no start event: the worker never reported in"
    missing = [e for e in ("config", "window", "seed") if e not in ev]
    if missing:
        return INCOMPLETE, f"never reported {missing}"
    if "invalid" in ev:
        return INVALID, ev["invalid"][-1].get("why", "worker flagged its own result invalid")
    if heartbeat_s and "complete" not in ev:
        beats = [r["t"] for r in ev.get("heartbeat", []) if isinstance(r.get("t"), float) and r["t"] == r["t"]]
        last = max(beats) if beats else ev["start"][-1]["t"]
        if (now if now is not None else time.time()) - last > heartbeat_s:
            return HUNG, f"no heartbeat for over {heartbeat_s}s and no completion"
    if "complete" not in ev:
        return INCOMPLETE, "no complete event (still running or died silently)"
    w, s = ev["window"][-1].get("window"), ev["seed"][-1].get("seed")
    if expected_window is not None and w != expected_window:
        return INVALID, f"worker ran window {w!r}, was assigned {expected_window!r}"
    if expected_seed is not None and s != expected_seed:
        return INVALID, f"worker ran seed {s!r}, was assigned {expected_seed!r}"
    if expected_config is not None and ev["config"][-1].get("config") != expected_config:
        return INVALID, "worker config differs from the assigned config"
    problems = validate_result(result, w, s)
    if problems:
        return INVALID, "; ".join(problems)
    ch = ev["start"][-1].get("code_hash")
    if current_code_hash is not None and ch != current_code_hash:
        return STALE, f"worker ran code {ch}, disk has {current_code_hash}"
    if stale_check is not None:
        stamp_row = ev["complete"][-1] if "code_files" in ev["complete"][-1] else ev["start"][-1]
        if stale_check(stamp_row):
            return STALE, "the code this worker loaded is no longer what is on disk (or was edited during the run)"
    if any(r.get("event") == "_corrupt" for r in log_rows):
        return INVALID, "log has a torn line"
    return OK, ""


def peak_memory(log_rows):
    v = [r["rss_mb"] for r in log_rows if r.get("event") == "memory" and r.get("rss_mb") is not None]
    return max(v) if v else None


def result_fingerprint(result):
    body = {k: v for k, v in result.items() if k not in ("seed", "window", "worker")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exclusion_report(workers, current_code_hash=None, heartbeat_s=None, now=None, stale_check=None):
    """workers: {id: {"log": rows, "result": dict|None, "config", "window", "seed"}}. Returns
    {"included": [...], "excluded": [{"worker","status","why"}], "counts": {...}}. Beyond per-worker status it
    excludes DUPLICATES: two workers with different seeds or windows but the same result body is a copied/stale
    result; both are excluded (we cannot tell which is the original)."""
    rows, fp = {}, {}
    for wid, w in workers.items():
        st, why = classify(w.get("log", []), w.get("result"), w.get("config"), w.get("window"), w.get("seed"),
                           current_code_hash, heartbeat_s, now, stale_check)
        rows[wid] = [st, why]
        if st == OK:
            fp.setdefault(result_fingerprint(w["result"]), []).append(wid)
    for ids in fp.values():
        if len(ids) > 1:
            for wid in ids:
                rows[wid] = [INVALID, f"result identical to {sorted(set(ids) - {wid})}: copied or stale"]
    inc = sorted(w for w, (s, _) in rows.items() if s == OK)
    exc = [{"worker": w, "status": s, "why": y, "peak_mb": peak_memory(workers[w].get("log", []))}
           for w, (s, y) in sorted(rows.items()) if s != OK]
    counts = {}
    for s, _ in rows.values():
        counts[s] = counts.get(s, 0) + 1
    return {"included": inc, "excluded": exc, "counts": counts, "n": len(rows)}


def aggregate_ready(report, min_ok_fraction=0.0, require_all=False):
    """May the batch be aggregated? Never fills a gap: excluded workers stay missing. `require_all` demands zero
    exclusions; otherwise at least `min_ok_fraction` of workers must be OK and at least one."""
    if not report["n"]:
        return False, "no workers"
    if require_all and report["excluded"]:
        return False, f"{len(report['excluded'])} excluded"
    frac = len(report["included"]) / report["n"]
    if not report["included"] or frac < min_ok_fraction:
        return False, f"only {frac:.0%} of workers OK"
    return True, f"{len(report['included'])}/{report['n']} OK; excluded workers stay excluded"


def write_report(report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=1, default=str))


# ------------------------------------------------------------------ resource trends and batch planning
def memory_series(log_rows):
    """[(t, rss_mb)] from memory events, in time order, ignoring unreadable samples."""
    return sorted((r["t"], r["rss_mb"]) for r in log_rows
                  if r.get("event") == "memory" and r.get("rss_mb") is not None and not r.get("peak"))


def memory_growth_mb_per_min(log_rows):
    """Least-squares slope of RSS over time. A worker that grows steadily will hit the ceiling eventually; the slope
    lets the supervisor warn before the OOM. None with fewer than 3 samples or zero time span."""
    s = memory_series(log_rows)
    if len(s) < 3:
        return None
    t = [x[0] for x in s]
    m = [x[1] for x in s]
    t0 = sum(t) / len(t)
    den = sum((a - t0) ** 2 for a in t)
    if den == 0:
        return None
    return sum((a - t0) * (b - sum(m) / len(m)) for a, b in zip(t, m)) / den * 60.0


def projected_oom_minutes(log_rows, mem_limit_mb):
    """Minutes until the ceiling at the current growth, or None when memory is flat/shrinking or unknown."""
    g, s = memory_growth_mb_per_min(log_rows), memory_series(log_rows)
    if g is None or g <= 0 or not s:
        return None
    return max(0.0, (mem_limit_mb - s[-1][1]) / g)


def heartbeat_gaps(log_rows):
    """Seconds between consecutive heartbeats: (max gap, median gap, n). Used to size the heartbeat timeout from
    what workers actually do rather than a guess."""
    t = sorted(r["t"] for r in log_rows if r.get("event") == "heartbeat" and r.get("t") == r.get("t"))
    if len(t) < 2:
        return None
    g = sorted(b - a for a, b in zip(t, t[1:]))
    return {"max": g[-1], "median": g[len(g) // 2], "n": len(t)}


def suggested_heartbeat_s(all_logs, safety=3.0, floor=5.0):
    """Timeout = safety x the largest gap seen among workers that COMPLETED, never below `floor`. Workers that died
    are excluded from the sample because their gap is what the timeout is meant to catch."""
    worst = 0.0
    for rows in all_logs:
        if any(r.get("event") == "complete" for r in rows) and not any(r.get("event") in ("crash", "timeout", "oom") for r in rows):
            g = heartbeat_gaps(rows)
            worst = max(worst, g["max"]) if g else worst
    return max(floor, safety * worst)


def scan_dir(dir_path, pattern="*.jsonl"):
    """{worker_id: log rows} for every log in a directory (the supervisor's own log lines are folded into the worker
    whose name the file carries)."""
    out = {}
    for p in sorted(Path(dir_path).glob(pattern)):
        out[p.stem] = read_log(p)
    return out


def plan_rerun(report, seeds_in_use, next_seed):
    """Reruns for excluded workers: NEW worker ids and NEW seeds, so the replacement can never be mistaken for, or
    merged into, the excluded one. Returns [{"replaces", "new_id", "seed"}]. The excluded workers stay in the report;
    this only schedules extra work."""
    used, out = set(seeds_in_use), []
    s = next_seed
    for e in report["excluded"]:
        while s in used:
            s += 1
        out.append({"replaces": e["worker"], "new_id": f"{e['worker']}_rerun{s}", "seed": s, "reason": e["status"]})
        used.add(s)
    return out


def failure_rates(reports_by_batch):
    """{status: share} across batches; the crash rate an operator watches for creeping up."""
    tot, n = {}, 0
    for r in reports_by_batch:
        for k, v in r["counts"].items():
            tot[k] = tot.get(k, 0) + v
        n += r["n"]
    return {k: v / n for k, v in tot.items()} if n else {}


def window_breakdown(workers, report):
    """{window: {"assigned", "ok"}}: a window where every worker was excluded has no result at all and must show as
    a hole, not as an average of nothing."""
    ok = set(report["included"])
    out = {}
    for wid, w in workers.items():
        r = out.setdefault(w.get("window"), {"assigned": 0, "ok": 0})
        r["assigned"] += 1
        r["ok"] += wid in ok
    return out


def check_no_silent_replacement(report, results_used):
    """`results_used`: worker ids whose results were fed to the aggregate. Fails when an excluded worker's result was
    used, or when a used id is unknown to the report."""
    excluded = {e["worker"] for e in report["excluded"]}
    known = set(report["included"]) | excluded
    problems = [f"excluded worker {w} was used" for w in results_used if w in excluded]
    problems += [f"unknown worker {w} was used" for w in results_used if w not in known]
    return problems
