"""Bible Phase 44 (resource management), Phase 43 (priority when compute is limited): a memory-aware job runner.

The target is a Windows PC with 8 cores and 16 GB shared by long experiments and many builders. Three real incidents
shape this module: an OOM cascade when too many workers started at once; a `taskkill /IM python.exe` that killed every
experiment on the machine; and two copies of one worker writing the same output folder. So:

* `worker_count` turns FREE memory (not total) into 3-7 workers, and returns fewer than 3 when memory says so -
  memory outranks the floor, because thrashing is worse than idling;
* `ProcRegistry` (state/procs.json) records every job we start with its PID AND the process create-time. `stop` refuses
  any PID that is not in the registry and any PID whose create-time no longer matches (Windows reuses PIDs). There is
  deliberately no stop-by-name function in this module;
* `stale_report` classifies registered jobs: dead, orphaned (parent gone), silent (heartbeat too old), duplicate job id,
  shared output folder. Duplicates are reported, never auto-killed;
* `JobQueue` launches a job only when a worker slot is free AND free memory minus the not-yet-materialised footprint of
  recently launched jobs covers the job's estimate; it refuses a duplicate job id or output folder up front;
* `cleanup` deletes old temp/scratch files under allow-listed roots only, dry-run by default, never in a sealed or
  cache tree and never inside a folder a live job owns.
Deterministic: worker seeds derive from (base seed, job id); every time-dependent function takes `now` (epoch seconds)."""
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as K

MIN_WORKERS, MAX_WORKERS = 3, 7
RESERVE_GB = 2.5            # left for the OS, the browser and whatever long experiment is already running
DEFAULT_JOB_GB = 2.0
HEARTBEAT_S = 300.0
RAMP_S = 120.0              # a launched job is assumed to still be growing toward its estimate for this long
PROTECTED = ("livesim", "cache")      # path parts that no cleanup may ever touch (sealed windows, data cache)
TEMP_PATTERNS = ("*.tmp", "*.partial", "*.part", "*.lock", "scratch*", "tmp_*", "*.swp", "*.pyc")


# ------------------------------------------------------------------------------------------------ machine facts
def memory_gb():
    """(free_gb, total_gb) of physical memory, or (None, None) when unreadable - never a fake number."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 1e9, vm.total / 1e9
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.avail / 1e9, m.total / 1e9
        except Exception:
            pass
    return None, None


def cpu_cores():
    return os.cpu_count() or 1


def worker_count(free_gb, per_worker_gb=DEFAULT_JOB_GB, cores=None, reserve_gb=RESERVE_GB, lo=MIN_WORKERS, hi=MAX_WORKERS):
    """How many workers to run right now. Memory decides: floor((free - reserve) / per_worker), capped at `hi` and at
    cores-1 (one core stays for the operator). Below `lo` the memory answer still wins (0 is a legitimate answer) and
    the caller is told why. Unknown free memory is treated as zero headroom: fail closed."""
    if free_gb is None or not math.isfinite(free_gb) or per_worker_gb <= 0:
        return {"workers": 0, "limit": "memory unreadable", "by_memory": 0, "by_cores": 0}
    cores = cores or cpu_cores()
    by_mem = max(0, int(math.floor((free_gb - reserve_gb) / per_worker_gb)))
    by_cores = max(1, cores - 1)
    n = min(by_mem, by_cores, hi)
    limit = "memory" if by_mem <= min(by_cores, hi) else ("cores" if by_cores < hi else "ceiling")
    if n < lo:
        limit = "memory (below the floor of %d workers; memory wins)" % lo if by_mem < lo else limit
    return {"workers": n, "limit": limit, "by_memory": by_mem, "by_cores": by_cores}


def worker_seed(base_seed, job_id):
    """Deterministic per-job seed: the same (base, job) always gets the same stream, different jobs never share one."""
    h = hashlib.sha256(f"{int(base_seed)}|{job_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ------------------------------------------------------------------------------------------------ process facts
def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def process_info(pid):
    """{'alive', 'create_time', 'ppid', 'rss_gb'} for a PID, or alive False. create_time identifies the process
    instance so a recycled PID is not mistaken for the job we started."""
    ps = _psutil()
    if ps is None or pid is None:
        return {"alive": False, "create_time": None, "ppid": None, "rss_gb": None}
    try:
        p = ps.Process(int(pid))
        if p.status() == ps.STATUS_ZOMBIE:
            return {"alive": False, "create_time": None, "ppid": None, "rss_gb": None}
        return {"alive": True, "create_time": p.create_time(), "ppid": p.ppid(), "rss_gb": p.memory_info().rss / 1e9}
    except Exception:
        return {"alive": False, "create_time": None, "ppid": None, "rss_gb": None}


def is_same_process(pid, create_time, tol=2.0):
    """True only when `pid` is alive AND started when the registry says (within tol seconds)."""
    info = process_info(pid)
    if not info["alive"]:
        return False
    return create_time is None or abs(info["create_time"] - float(create_time)) <= tol


# ------------------------------------------------------------------------------------------------ registry
@contextlib.contextmanager
def _lock(path, wait_s=10.0, stale_s=30.0):
    """Cross-process lock via an exclusively created file; a lock older than stale_s is a crashed writer's and is broken."""
    lp = Path(str(path) + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lp.stat().st_mtime > stale_s:
                    lp.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"could not lock {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lp.unlink()


class ProcRegistry:
    """state/procs.json: {job_id: {pid, create_time, ppid, cmd, out_dir, started, heartbeat, est_gb, state}}.
    Every mutation is load-modify-write under a file lock and lands atomically (temp file + rename), so two managers
    cannot interleave and a crash cannot leave half a file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else K.STATE / "procs.json"

    def load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, d):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp%d" % os.getpid())
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def register(self, job_id, pid, cmd="", out_dir=None, est_gb=DEFAULT_JOB_GB, now=None, ppid=None, create_time="auto"):
        """Record a job we started. A live job with the same id or the same output folder is refused (the incident:
        two copies of one worker writing one folder)."""
        now = time.time() if now is None else now
        info = process_info(pid)
        ct = info["create_time"] if create_time == "auto" else create_time
        with _lock(self.path):
            d = self.load()
            for jid, e in d.items():
                if e.get("state") != "running":
                    continue
                live = is_same_process(e["pid"], e.get("create_time"))
                if live and jid == job_id:
                    raise ValueError(f"job {job_id} is already running as pid {e['pid']}")
                if live and out_dir and e.get("out_dir") and _same_path(e["out_dir"], out_dir):
                    raise ValueError(f"output folder {out_dir} is already owned by live job {jid} (pid {e['pid']})")
            d[job_id] = {"pid": int(pid), "create_time": ct, "ppid": ppid if ppid is not None else info["ppid"],
                         "cmd": cmd, "out_dir": str(out_dir) if out_dir else None, "started": now, "heartbeat": now,
                         "est_gb": est_gb, "state": "running"}
            self._write(d)
        return d[job_id]

    def heartbeat(self, job_id, now=None):
        now = time.time() if now is None else now
        with _lock(self.path):
            d = self.load()
            if job_id not in d:
                raise KeyError(job_id)
            d[job_id]["heartbeat"] = now
            self._write(d)

    def finish(self, job_id, state="done", now=None):
        """Mark a job done/failed/stopped. The row stays (a silent disappearance is exactly what Phase 44 forbids)."""
        if state not in ("done", "failed", "stopped"):
            raise ValueError(state)
        with _lock(self.path):
            d = self.load()
            if job_id not in d:
                raise KeyError(job_id)
            d[job_id]["state"] = state
            d[job_id]["ended"] = time.time() if now is None else now
            self._write(d)

    def running(self):
        return {j: e for j, e in self.load().items() if e.get("state") == "running"}

    def prune(self, keep_finished=50):
        """Drop the oldest finished rows beyond keep_finished. Running rows are never pruned."""
        with _lock(self.path):
            d = self.load()
            fin = sorted((j for j, e in d.items() if e.get("state") != "running"), key=lambda j: d[j].get("ended") or 0)
            drop = fin[:max(0, len(fin) - keep_finished)]
            for j in drop:
                del d[j]
            self._write(d)
        return drop

    def stop(self, job_id, grace_s=5.0, tree=True, now=None):
        """Terminate a job by ITS RECORDED PID, after checking that pid still is the process we started. Returns a dict
        saying what happened; refuses (and touches nothing) for an unknown job or a recycled PID."""
        e = self.load().get(job_id)
        if e is None:
            return {"stopped": False, "reason": "unknown job id: only registered jobs can be stopped"}
        if e.get("state") != "running":
            return {"stopped": False, "reason": f"job is already {e.get('state')}"}
        if not is_same_process(e["pid"], e.get("create_time")):
            self.finish(job_id, "failed", now)
            return {"stopped": False, "reason": "pid is gone or was reused by another process; nothing was signalled"}
        ps = _psutil()
        killed = []
        try:
            p = ps.Process(e["pid"])
            procs = p.children(recursive=True) + [p] if tree else [p]
            for q in procs:
                with contextlib.suppress(Exception):
                    q.terminate()
            gone, alive = ps.wait_procs(procs, timeout=grace_s)
            for q in alive:
                with contextlib.suppress(Exception):
                    q.kill()
            killed = [q.pid for q in procs]
        except Exception as ex:
            return {"stopped": False, "reason": f"{type(ex).__name__}: {ex}"}
        self.finish(job_id, "stopped", now)
        return {"stopped": True, "pids": killed}


def _same_path(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


# ------------------------------------------------------------------------------------------------ staleness
def stale_report(registry, now, heartbeat_s=HEARTBEAT_S, info=process_info):
    """Classify every running registry row. `info` is injectable so tests need no real processes.
    kinds: dead (registered running but pid gone / recycled), orphaned (parent gone: nobody will collect it),
    silent (alive but heartbeat older than heartbeat_s), duplicate_job (two live rows with one job id in the cmd line
    are impossible by key, so this flags one *command* running twice), shared_output (two live jobs, one folder)."""
    rows = registry.running() if hasattr(registry, "running") else dict(registry)
    findings = []
    live = {}
    for jid, e in sorted(rows.items()):
        pi = info(e["pid"])
        same = pi["alive"] and (e.get("create_time") is None or pi["create_time"] is None
                                or abs(pi["create_time"] - float(e["create_time"])) <= 2.0)
        age = now - float(e.get("heartbeat") or e.get("started") or now)
        if not same:
            findings.append({"job": jid, "kind": "dead", "detail": "process gone or PID reused", "heartbeat_age_s": age})
            continue
        live[jid] = e
        ppid = e.get("ppid")
        if ppid and not info(ppid)["alive"]:
            findings.append({"job": jid, "kind": "orphaned", "detail": f"parent pid {ppid} is gone", "heartbeat_age_s": age})
        if age > heartbeat_s:
            findings.append({"job": jid, "kind": "silent", "detail": f"no heartbeat for {age:.0f}s (limit {heartbeat_s:.0f}s)",
                             "heartbeat_age_s": age})
    by_cmd, by_dir = {}, {}
    for jid, e in live.items():
        if e.get("cmd"):
            by_cmd.setdefault(_norm_cmd(e["cmd"]), []).append(jid)
        if e.get("out_dir"):
            by_dir.setdefault(str(Path(e["out_dir"]).resolve()), []).append(jid)
    for cmd, jids in by_cmd.items():
        if len(jids) > 1:
            findings += [{"job": j, "kind": "duplicate_job", "detail": f"same command also running as {[x for x in jids if x != j]}",
                          "heartbeat_age_s": now - float(live[j].get("heartbeat") or now)} for j in jids]
    for d, jids in by_dir.items():
        if len(jids) > 1:
            findings += [{"job": j, "kind": "shared_output", "detail": f"{d} is written by {jids}",
                          "heartbeat_age_s": now - float(live[j].get("heartbeat") or now)} for j in jids]
    kinds = {}
    for f in findings:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    return {"n_running": len(rows), "n_live": len(live), "findings": findings, "by_kind": kinds,
            "healthy": not findings}


def _norm_cmd(cmd):
    """Command line with whitespace collapsed, so `a  b` and `a b` are the same command."""
    return " ".join(cmd.split()) if isinstance(cmd, str) else " ".join(map(str, cmd))


def reap_dead(registry, now, info=process_info):
    """Mark registry rows whose process is gone as failed (never as done: a job that vanished did not report success).
    Returns the job ids reaped. This edits the registry only; it signals nothing."""
    dead = [f["job"] for f in stale_report(registry, now, info=info)["findings"] if f["kind"] == "dead"]
    for j in dead:
        registry.finish(j, "failed", now)
    return dead


# ------------------------------------------------------------------------------------------------ queue
class Job:
    def __init__(self, job_id, cmd, est_gb=DEFAULT_JOB_GB, out_dir=None, priority=5, cwd=None, env=None):
        if not job_id or Path(str(job_id)).name != str(job_id):
            raise ValueError(f"unusable job id {job_id!r}")
        if not cmd:
            raise ValueError("a job needs a command")
        if not (est_gb and est_gb > 0 and math.isfinite(est_gb)):
            raise ValueError("est_gb must be a positive number")
        self.job_id, self.cmd, self.est_gb, self.out_dir = job_id, cmd, float(est_gb), out_dir
        self.priority, self.cwd, self.env = priority, cwd, env

    def as_dict(self):
        return {"job_id": self.job_id, "cmd": self.cmd, "est_gb": self.est_gb, "out_dir": self.out_dir, "priority": self.priority}


def default_launcher(job, log_path=None):
    """Start the job as its own process (no shell, detached from our stdin). Returns the Popen."""
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    cmd = job.cmd if isinstance(job.cmd, (list, tuple)) else job.cmd.split()
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
    return subprocess.Popen(cmd, cwd=job.cwd, env=job.env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags)


class JobQueue:
    """Priority queue (lower number first, then FIFO). `tick(now)` launches what memory and the worker cap allow and
    returns a log of decisions, including every reason a job was held - a silent hold is a silently dead job."""

    def __init__(self, registry, free_fn=None, launcher=None, rss_fn=None, per_worker_gb=DEFAULT_JOB_GB, reserve_gb=RESERVE_GB,
                 max_workers=MAX_WORKERS, ramp_s=RAMP_S, cores=None, log_dir=None, info=process_info):
        self.registry = registry
        self.free_fn = free_fn or (lambda: memory_gb()[0])
        self.launcher = launcher or (lambda job: default_launcher(job, (Path(log_dir) / f"{job.job_id}.log") if log_dir else None))
        self.rss_fn = rss_fn or (lambda pid: process_info(pid)["rss_gb"])
        self.per_worker_gb, self.reserve_gb, self.max_workers, self.ramp_s, self.cores = per_worker_gb, reserve_gb, max_workers, ramp_s, cores
        self.info = info
        self.pending = []
        self._seq = 0

    def submit(self, job):
        """Queue a job. Refuses a duplicate job id or an output folder already claimed by a queued or running job."""
        for j in self.pending:
            if j[2].job_id == job.job_id:
                raise ValueError(f"job {job.job_id} is already queued")
            if job.out_dir and j[2].out_dir and _same_path(j[2].out_dir, job.out_dir):
                raise ValueError(f"output folder {job.out_dir} is already claimed by queued job {j[2].job_id}")
        for jid, e in self.registry.running().items():
            if jid == job.job_id:
                raise ValueError(f"job {job.job_id} is already running")
            if job.out_dir and e.get("out_dir") and _same_path(e["out_dir"], job.out_dir):
                raise ValueError(f"output folder {job.out_dir} is already owned by running job {jid}")
        self._seq += 1
        self.pending.append((job.priority, self._seq, job))
        self.pending.sort(key=lambda x: (x[0], x[1]))

    def headroom_gb(self, now):
        """Free memory minus the part of recently launched jobs' estimates that has not yet shown up as RSS."""
        free = self.free_fn()
        if free is None:
            return None
        owed = 0.0
        for jid, e in self.registry.running().items():
            if now - float(e.get("started") or 0) <= self.ramp_s:
                rss = self.rss_fn(e["pid"])
                owed += max(0.0, float(e.get("est_gb") or self.per_worker_gb) - (rss or 0.0))
        return free - owed

    def tick(self, now):
        log = []
        reap_dead(self.registry, now, info=self.info)
        while self.pending:
            running = self.registry.running()
            room = self.headroom_gb(now)
            if room is None:
                log.append({"event": "hold", "reason": "free memory unreadable"})
                break
            slots = max(1, min(self.max_workers, (self.cores or cpu_cores()) - 1))
            if len(running) >= slots:
                log.append({"event": "hold", "reason": f"{len(running)} workers running (ceiling {slots})"})
                break
            job = self.pending[0][2]
            if room - self.reserve_gb < job.est_gb:
                log.append({"event": "hold", "job": job.job_id,
                            "reason": f"needs {job.est_gb:.1f} GB, headroom {max(0.0, room - self.reserve_gb):.1f} GB after reserve"})
                break
            self.pending.pop(0)
            try:
                proc = self.launcher(job)
                pid = proc.pid if hasattr(proc, "pid") else int(proc)
                self.registry.register(job.job_id, pid, job.cmd if isinstance(job.cmd, str) else " ".join(job.cmd),
                                       job.out_dir, job.est_gb, now, ppid=os.getpid())
                log.append({"event": "launch", "job": job.job_id, "pid": pid, "est_gb": job.est_gb})
            except Exception as ex:
                log.append({"event": "launch_failed", "job": job.job_id, "reason": f"{type(ex).__name__}: {ex}"})
        return log

    def snapshot(self):
        return {"pending": [j[2].as_dict() for j in self.pending], "running": self.registry.running()}


# ------------------------------------------------------------------------------------------------ cleanup
def _protected(path, live_dirs):
    parts = {p.lower() for p in Path(path).resolve().parts}
    if parts & set(PROTECTED):
        return "sealed or cache tree"
    rp = Path(path).resolve()
    for d in live_dirs:
        try:
            dd = Path(d).resolve()
        except Exception:
            continue
        if dd == rp or dd in rp.parents:
            return f"inside live job folder {dd.name}"
    return None


def cleanup(roots, older_than_days, now, patterns=TEMP_PATTERNS, dry_run=True, registry=None, max_delete=10_000):
    """Remove temp/scratch FILES older than N days under `roots`. Dry-run by default: the same plan is returned either
    way and only `dry_run=False` deletes. Skips sealed/cache trees, live job folders and anything not matching a temp
    pattern; refuses a root that is a drive root or the repo root, and refuses more than max_delete files in one go.
    Age is `now` minus mtime, so a file from the future is never old."""
    if older_than_days is None or older_than_days < 0:
        raise ValueError("older_than_days must be >= 0")
    live = [e["out_dir"] for e in (registry.running().values() if registry else []) if e.get("out_dir")]
    plan, skipped = [], []
    cutoff = now - older_than_days * 86400.0
    for root in roots:
        r = Path(root).resolve()
        if r == Path(r.anchor) or r == K.ROOT.resolve():
            raise ValueError(f"refusing to clean {r}: too broad")
        if not r.exists():
            continue
        for p in sorted(r.rglob("*")):
            if not p.is_file():
                continue
            if not any(p.match(pat) for pat in patterns):
                continue
            why = _protected(p, live)
            if why:
                skipped.append({"path": str(p), "reason": why})
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime <= cutoff:
                plan.append({"path": str(p), "bytes": st.st_size, "age_days": (now - st.st_mtime) / 86400.0})
    if len(plan) > max_delete:
        raise ValueError(f"{len(plan)} files matched (> {max_delete}); narrow the roots or patterns")
    deleted, failed = [], []
    if not dry_run:
        for e in plan:
            try:
                Path(e["path"]).unlink()
                deleted.append(e["path"])
            except OSError as ex:
                failed.append({"path": e["path"], "reason": str(ex)})
    return {"dry_run": dry_run, "would_delete": len(plan), "bytes": sum(e["bytes"] for e in plan), "plan": plan,
            "skipped": skipped, "deleted": deleted, "failed": failed}


# ------------------------------------------------------------------------------------------------ health report
def worker_health(registry, now, heartbeat_s=HEARTBEAT_S, per_worker_gb=DEFAULT_JOB_GB, free_fn=None, info=process_info):
    """One dict for the dashboard and the run report: memory, recommended workers, every running job with its age,
    RSS and heartbeat age, and the stale findings. Nothing is rounded into a reassuring zero: unreadable is None."""
    free, total = (free_fn(), None) if free_fn else memory_gb()
    rows = []
    for jid, e in sorted(registry.running().items()):
        pi = info(e["pid"])
        rows.append({"job": jid, "pid": e["pid"], "age_s": now - float(e.get("started") or now),
                     "heartbeat_age_s": now - float(e.get("heartbeat") or e.get("started") or now),
                     "rss_gb": pi["rss_gb"], "est_gb": e.get("est_gb"), "alive": pi["alive"], "out_dir": e.get("out_dir")})
    stale = stale_report(registry, now, heartbeat_s, info)
    return {"free_gb": free, "total_gb": total, "recommended": worker_count(free, per_worker_gb),
            "running": rows, "stale": stale, "n_running": len(rows),
            "over_estimate": [r["job"] for r in rows if r["rss_gb"] and r["est_gb"] and r["rss_gb"] > 1.5 * r["est_gb"]]}


def wait_for_memory(need_gb, free_fn=None, poll_s=60.0, give_up_s=1200.0, sleep=time.sleep, clock=time.monotonic):
    """The builders' rule 10 as code: poll until `need_gb` is free; give up after give_up_s. Returns (ok, free_gb)."""
    free_fn = free_fn or (lambda: memory_gb()[0])
    t0 = clock()
    while True:
        f = free_fn()
        if f is not None and f >= need_gb:
            return True, f
        if clock() - t0 >= give_up_s:
            return False, f
        sleep(poll_s)
