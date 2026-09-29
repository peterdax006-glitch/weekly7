"""Compute resource manager for parallel research (contract C62 section 57; checklist J04-J06). IMPLEMENTED - NOT VALIDATED.

Parallel research must never corrupt learning state, so this module gives the learning engine:
  * deterministic workers   - a worker's seed derives from (base seed, experiment key) via engine.resources.worker_seed, so the
                              same experiment gives the same numbers whichever worker, order or machine runs it;
  * isolated experiments    - every attempt writes ONLY inside its own directory; nothing a worker does touches shared
                              learning state. Results reach the shared state only through `reconcile` + a deterministic merge;
  * memory snapshots        - workers read an immutable, hash-verified snapshot of learning state dated strictly before the
                              experiment's as_of (a snapshot from the future is a FirewallBreach);
  * checkpointing           - attempts leave a `progress.json` the next attempt can resume from;
  * crash / OOM recovery    - a dead or silent worker is detected, classified (OOM raises the memory estimate and shrinks the
                              batch; a crash just retries) and re-queued up to a cap, after which it is GAVE_UP and escalated;
  * stale-code detection    - a result produced by other code than the engine now loaded is stale and is never reconciled;
  * result reconciliation   - results are verified against the ledger (hash, seed, spec, code); duplicates that disagree expose
                              non-determinism;
  * duplicate prevention    - one experiment key can be active once, and a finished experiment is not silently rerun.
Builds on engine.resources (worker_count, worker_seed, Job, ProcRegistry locking, process_info, stale_report) - it adds the
learning-state semantics on top; it does not start or stop processes by name (there is deliberately no such function)."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import shutil
import stat
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np

from engine import resources as R

from .core import FirewallBreach, as_date, canonical_json, current_code_hash, require_past, stable_hash

PENDING, RUNNING, DONE, FAILED, OOM, CRASHED, STALE, SUPERSEDED, GAVE_UP = (
    "PENDING", "RUNNING", "DONE", "FAILED", "OOM", "CRASHED", "STALE", "SUPERSEDED", "GAVE_UP")
ACTIVE = (PENDING, RUNNING)
RETRYABLE = (FAILED, OOM, CRASHED, STALE)
OOM_EXIT_CODES = frozenset({137, -9, 3221225495, 3221226505, -1073741801})   # SIGKILL, STATUS_NO_MEMORY, STATUS_STACK_BUFFER_OVERRUN
OOM_TEXT = ("memoryerror", "out of memory", "unable to allocate", "cannot allocate memory", "std::bad_alloc", "bad allocation")


class ComputeError(RuntimeError):
    pass


class DuplicateExperiment(ComputeError):
    pass


class ClaimError(ComputeError):
    pass


class SnapshotError(ComputeError):
    pass


# ------------------------------------------------------------------------------------------------ specification
@dataclass(frozen=True)
class ExperimentSpec:
    """One unit of research. The experiment key ignores the code hash (same question = same experiment) while the run key
    includes it (same question under different code = a different, incomparable run)."""
    name: str
    params: Mapping[str, Any]
    seed: int
    as_of: str                       # the decision date the experiment reasons at: nothing dated at/after it may be used
    snapshot_id: str = ""
    data_hash: str = ""
    est_gb: float = R.DEFAULT_JOB_GB
    priority: int = 5
    after: tuple[str, ...] = ()      # experiment keys that must be DONE first; scheduling only, so not part of the identity

    def validate(self) -> list[str]:
        errs = []
        if not self.name or Path(self.name).name != self.name:
            errs.append("name must be a plain identifier")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            errs.append("seed must be a non-negative int")
        try:
            as_date(self.as_of)
        except ValueError:
            errs.append("as_of is not a date")
        if not (math.isfinite(self.est_gb) and self.est_gb > 0):
            errs.append("est_gb must be positive")
        if any(not isinstance(a, str) or not a for a in self.after):
            errs.append("after must be a tuple of experiment keys")
        try:
            canonical_json(self.params)
        except TypeError as e:
            errs.append(f"params not serialisable: {e}")
        return errs

    @property
    def key(self) -> str:
        return stable_hash([self.name, self.params, self.seed, self.as_of, self.snapshot_id, self.data_hash], 20)

    def run_key(self, code_hash: str) -> str:
        return stable_hash([self.key, code_hash], 20)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentSpec":
        kw = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d}
        kw["after"] = tuple(kw.get("after") or ())
        return cls(**kw)


def worker_seed_for(spec: ExperimentSpec) -> int:
    """Deterministic per-experiment stream (engine.resources.worker_seed keyed by the experiment key)."""
    return R.worker_seed(spec.seed, spec.key)


# ------------------------------------------------------------------------------------------------ failure classification
def classify_failure(exc: BaseException | None = None, returncode: int | None = None, stderr: str = "",
                     timed_out: bool = False) -> str:
    """OOM, TIMEOUT, CRASH (process died without a Python error) or ERROR (an ordinary exception). OOM is recognised from
    MemoryError, from the known kill/allocation exit codes and from allocator messages in stderr."""
    if isinstance(exc, MemoryError):
        return OOM
    if isinstance(exc, TimeoutError):
        return CRASHED
    text = (stderr or "").lower() + (str(exc).lower() if exc is not None else "")
    if any(t in text for t in OOM_TEXT):
        return OOM
    if returncode is not None and returncode in OOM_EXIT_CODES:
        return OOM
    if timed_out:
        return CRASHED
    if exc is not None:
        return FAILED
    return CRASHED if returncode not in (None, 0) else FAILED


# ------------------------------------------------------------------------------------------------ atomic files
def atomic_write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def read_json(path: str | Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except ValueError:
        return default


def _readonly_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)


def _rmtree(d: Path) -> None:
    def unlock(fn, path, _):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    shutil.rmtree(d, onerror=unlock)


# ------------------------------------------------------------------------------------------------ memory snapshots
class SnapshotStore:
    """Immutable copies of learning state for workers. Content-addressed: identical state gives an identical id, so a snapshot
    can be shared by many workers and can never be edited after the fact (files are read-only and re-hashed on every load)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create(self, state: Mapping[str, Any], as_of, code_hash: str = "") -> str:
        """`as_of` is the last date the state contains information from. Returns the snapshot id."""
        body = json.loads(canonical_json({"as_of": str(as_date(as_of)), "state": state, "code_hash": code_hash}))
        sid = stable_hash(body, 20)
        d = self.root / sid
        if (d / "snapshot.json").exists():
            return sid
        d.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"id": sid, "body": json.loads(canonical_json(body))}, sort_keys=True, allow_nan=False)
        f = d / "snapshot.json"
        f.write_text(text, encoding="utf-8")
        os.chmod(f, stat.S_IREAD)
        return sid

    def load(self, sid: str) -> dict:
        f = self.root / sid / "snapshot.json"
        if not f.is_file():
            raise SnapshotError(f"snapshot {sid} not found")
        doc = json.loads(f.read_text(encoding="utf-8"))
        if stable_hash(doc["body"], 20) != sid or doc.get("id") != sid:
            raise SnapshotError(f"snapshot {sid} was altered after creation")
        return doc["body"]

    def for_experiment(self, spec: ExperimentSpec) -> dict:
        """The snapshot a worker may read: it must be dated strictly before the experiment's as_of."""
        if not spec.snapshot_id:
            return {"as_of": None, "state": {}, "code_hash": ""}
        body = self.load(spec.snapshot_id)
        require_past(body["as_of"], spec.as_of, f"snapshot {spec.snapshot_id}")
        return body

    def materialize(self, sid: str, dest: str | Path) -> Path:
        """Copy the verified snapshot into a worker's private directory so it cannot even read a sibling worker's files."""
        self.load(sid)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "snapshot.json"
        shutil.copyfile(self.root / sid / "snapshot.json", out)
        return out

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "snapshot.json").exists()) if self.root.is_dir() else []


# ------------------------------------------------------------------------------------------------ experiment ledger
@dataclass(frozen=True)
class SubmitResult:
    key: str
    action: str                      # queued | requeued | superseding
    attempts: int
    note: str = ""


class ExperimentLedger:
    """state/.../experiments.json: {key: entry}. All mutations are load-modify-write under a cross-process file lock
    (engine.resources._lock) and land atomically, so two managers cannot interleave and a crash cannot leave half a file."""

    def __init__(self, path: str | Path, max_attempts: int = 3, oom_growth: float = 1.5, max_est_gb: float = 8.0,
                 backoff: "BackoffPolicy | None" = None):
        if max_attempts < 1 or oom_growth <= 1.0:
            raise ValueError("max_attempts >= 1 and oom_growth > 1 required")
        if backoff is not None and backoff.validate():
            raise ValueError(f"invalid backoff: {backoff.validate()}")
        self.backoff = backoff
        self.path = Path(path)
        self.max_attempts, self.oom_growth, self.max_est_gb = max_attempts, oom_growth, max_est_gb

    # ---- storage ----
    def load(self) -> dict[str, dict]:
        d = read_json(self.path, {})
        return d if isinstance(d, dict) else {}

    @contextlib.contextmanager
    def _txn(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with R._lock(self.path):
            data = self.load()
            yield data
            atomic_write_json(self.path, data)

    def get(self, key: str) -> dict | None:
        return self.load().get(key)

    @staticmethod
    def _note(e: dict, now: float, state: str, note: str = "") -> None:
        e["state"] = state
        e["history"].append([float(now), state, note])

    # ---- submission and duplicate prevention ----
    def submit(self, spec: ExperimentSpec, now: float, code_hash: str) -> SubmitResult:
        errs = spec.validate()
        if errs:
            raise ComputeError(f"invalid spec {spec.name}: {errs}")
        key = spec.key
        with self._txn() as d:
            e = d.get(key)
            if e is None:
                d[key] = {"spec": spec.to_dict(), "state": PENDING, "attempts": 0, "code_hash": code_hash, "est_gb": spec.est_gb,
                          "batch_scale": 1.0, "history": [[float(now), PENDING, "submitted"]], "result_hash": None,
                          "worker": None, "heartbeat": None, "runs": []}
                return SubmitResult(key, "queued", 0)
            st = e["state"]
            if st in ACTIVE:
                raise DuplicateExperiment(f"{spec.name} ({key}) is already {st}")
            if st == DONE and e["code_hash"] == code_hash:
                raise DuplicateExperiment(f"{spec.name} ({key}) already finished under this code; its result is {e['result_hash']}")
            if st == GAVE_UP:
                raise DuplicateExperiment(f"{spec.name} ({key}) exhausted {e['attempts']} attempts; escalate, do not resubmit blindly")
            if st == DONE:                                     # same question, different code: a new, incomparable run
                e["runs"].append({"code_hash": e["code_hash"], "result_hash": e["result_hash"], "attempts": e["attempts"]})
                e.update(code_hash=code_hash, attempts=0, result_hash=None, worker=None, heartbeat=None)
                self._note(e, now, PENDING, f"rerun under new code (superseding {e['runs'][-1]['code_hash']})")
                return SubmitResult(key, "superseding", 0, "previous DONE result is stale under the new code")
            if e["attempts"] >= self.max_attempts:
                self._note(e, now, GAVE_UP, "attempt cap reached at resubmission")
                raise DuplicateExperiment(f"{spec.name} exceeded {self.max_attempts} attempts")
            e["code_hash"] = code_hash
            self._note(e, now, PENDING, f"requeued after {st}")
            return SubmitResult(key, "requeued", e["attempts"])

    # ---- claiming and lifecycle ----
    def claim(self, key: str, worker_id: str, now: float, code_hash: str) -> dict:
        """PENDING -> RUNNING for exactly one worker. A second claimant gets ClaimError (atomic under the lock)."""
        with self._txn() as d:
            e = d.get(key)
            if e is None:
                raise ClaimError(f"unknown experiment {key}")
            if e["state"] != PENDING:
                raise ClaimError(f"{key} is {e['state']}, not PENDING (claimed by {e.get('worker')})")
            if e["code_hash"] != code_hash:
                raise ClaimError(f"{key} was submitted under code {e['code_hash']} but the worker runs {code_hash}")
            e["attempts"] += 1
            if e.get("not_before") is not None and float(now) < float(e["not_before"]):
                raise ClaimError(f"{key} is backing off until {e['not_before']:.0f}")
            e.update(worker=worker_id, heartbeat=float(now), launched=None, not_before=None)
            self._note(e, now, RUNNING, f"attempt {e['attempts']} by {worker_id}")
            return json.loads(json.dumps(e))

    def mark_launched(self, key: str, now: float) -> None:
        """The supervisor started a process for this PENDING experiment; do not start another until it claims or the grace passes."""
        with self._txn() as d:
            if key not in d or d[key]["state"] != PENDING:
                raise ComputeError(f"{key} is not PENDING")
            d[key]["launched"] = float(now)

    def heartbeat(self, key: str, now: float) -> None:
        with self._txn() as d:
            if key not in d or d[key]["state"] != RUNNING:
                raise ComputeError(f"{key} is not running")
            d[key]["heartbeat"] = float(now)

    def complete(self, key: str, result_hash: str, now: float) -> None:
        with self._txn() as d:
            e = d.get(key)
            if e is None or e["state"] != RUNNING:
                raise ComputeError(f"{key} cannot complete from {None if e is None else e['state']}")
            e["result_hash"] = result_hash
            self._note(e, now, DONE, f"result {result_hash}")

    def fail(self, key: str, kind: str, note: str, now: float) -> str:
        """Record a failed attempt and decide what happens next. Returns the resulting state (PENDING again for a retry,
        GAVE_UP once the attempt cap is reached). OOM grows the memory estimate and halves the batch scale for the retry."""
        if kind not in (FAILED, OOM, CRASHED, STALE):
            raise ValueError(kind)
        with self._txn() as d:
            e = d.get(key)
            if e is None or e["state"] != RUNNING:
                raise ComputeError(f"{key} cannot fail from {None if e is None else e['state']}")
            self._note(e, now, kind, note)
            e["worker"] = None
            if e["attempts"] >= self.max_attempts:
                self._note(e, now, GAVE_UP, f"{e['attempts']} attempts exhausted; last failure {kind}: {note[:120]}")
                return GAVE_UP
            if kind == OOM:
                e["est_gb"] = min(self.max_est_gb, e["est_gb"] * self.oom_growth)
                e["batch_scale"] = max(0.125, e["batch_scale"] * 0.5)
            if self.backoff is not None:
                e["not_before"] = float(now) + self.backoff.delay(key, e["attempts"])
            self._note(e, now, PENDING, f"retry after {kind}")
            return PENDING

    # ---- recovery ----
    def recover(self, now: float, alive: Callable[[str], bool], heartbeat_s: float = R.HEARTBEAT_S) -> list[dict]:
        """Find RUNNING experiments whose worker is gone (alive(worker) False) or silent past heartbeat_s and fail them as
        CRASHED so they retry or give up. `alive` is injectable so tests need no real processes."""
        found = []
        for key, e in sorted(self.load().items()):
            if e["state"] != RUNNING:
                continue
            silent = now - float(e.get("heartbeat") or 0.0) > heartbeat_s
            gone = not alive(cast(str, e.get("worker")))
            if gone or silent:
                why = "worker gone" if gone else f"silent {now - float(e['heartbeat']):.0f}s"
                found.append({"key": key, "why": why, "next": self.fail(key, CRASHED, why, now)})
        return found

    def stale(self, current_code_hash: str) -> list[str]:
        """Keys whose recorded code differs from the code now loaded (DONE results included: they cannot be reconciled)."""
        return sorted(k for k, e in self.load().items() if e["state"] not in (SUPERSEDED, GAVE_UP) and e["code_hash"] != current_code_hash)

    def mark_stale(self, current_code_hash: str, now: float) -> list[str]:
        marked = []
        with self._txn() as d:
            for k, e in sorted(d.items()):
                if e["state"] in (PENDING, DONE) and e["code_hash"] != current_code_hash:
                    self._note(e, now, STALE, f"code {e['code_hash']} != {current_code_hash}")
                    marked.append(k)
        return marked

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.load().values():
            out[e["state"]] = out.get(e["state"], 0) + 1
        return out

    def spec(self, key: str) -> ExperimentSpec:
        e = self.get(key)
        if e is None:
            raise ComputeError(f"unknown experiment {key}")
        return ExperimentSpec.from_dict(e["spec"])


# ------------------------------------------------------------------------------------------------ worker side
@dataclass
class WorkerContext:
    """Everything an experiment function may touch. `out_dir` is private to this attempt; `rng` is deterministic."""
    spec: ExperimentSpec
    out_dir: Path
    snapshot: dict
    code_hash: str
    worker_id: str
    attempt: int
    batch_scale: float = 1.0
    deadline: "Deadline | None" = None
    _ledger: "ExperimentLedger | None" = None
    _clock: Callable[[], float] = time.time

    @property
    def seed(self) -> int:
        return worker_seed_for(self.spec)

    def rng(self, stream: str = "") -> np.random.Generator:
        """A fresh generator per (experiment, stream): independent of worker, order and other streams."""
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(stable_hash(stream, 8), 16)]))

    def beat(self) -> None:
        """Heartbeat to the ledger and enforce the wall-clock budget, if any: call this inside long loops."""
        if self.deadline is not None:
            self.deadline.check(self.spec.name)
        if self._ledger is not None:
            self._ledger.heartbeat(self.spec.key, self._clock())

    def save_progress(self, state: Mapping[str, Any]) -> None:
        """Checkpoint inside the experiment; the next attempt can resume from it (`load_progress`)."""
        atomic_write_json(self.out_dir.parent / "progress.json", {"attempt": self.attempt, "code_hash": self.code_hash,
                                                                   "state": state})

    def load_progress(self) -> dict | None:
        """Progress from an earlier attempt, only if it was made with the same code (stale progress is discarded)."""
        p = read_json(self.out_dir.parent / "progress.json")
        if p and p.get("code_hash") == self.code_hash:
            return p["state"]
        return None


def _envelope(spec: ExperimentSpec, result: Any, code_hash: str, worker_id: str, attempt: int, started: float, ended: float) -> dict:
    return {"experiment_key": spec.key, "run_key": spec.run_key(code_hash), "spec": spec.to_dict(), "seed": worker_seed_for(spec),
            "code_hash": code_hash, "data_hash": spec.data_hash, "snapshot_id": spec.snapshot_id, "worker": worker_id,
            "attempt": attempt, "started": started, "ended": ended, "result": json.loads(canonical_json(result)), "result_hash": stable_hash(result, 20)}


def attempt_dir(out_root: str | Path, key: str, attempt: int) -> Path:
    return Path(out_root) / key / f"attempt_{attempt:02d}"


def run_worker(spec: ExperimentSpec, fn: Callable[[ExperimentSpec, WorkerContext], Any], ledger: ExperimentLedger,
               store: SnapshotStore | None, out_root: str | Path, worker_id: str, now: float,
               code_hash: str | None = None, clock: Callable[[], float] = time.time, deadline_s: float | None = None,
               calibrator: "MemoryCalibrator | None" = None) -> dict:
    """Claim, isolate, run, record. Returns {'state', 'result_hash'|'error'}. Every failure path leaves the ledger in a
    consistent state and the attempt directory intact for inspection; nothing is written outside `attempt_dir`."""
    code_hash = code_hash if code_hash is not None else current_code_hash()
    claim = ledger.claim(spec.key, worker_id, now, code_hash)
    attempt = claim["attempts"]
    adir = attempt_dir(out_root, spec.key, attempt)
    if adir.exists():
        raise ComputeError(f"attempt directory {adir} exists: two attempts must never share a folder")
    adir.mkdir(parents=True)
    started = clock()
    try:
        snap = store.for_experiment(spec) if store is not None else {"as_of": None, "state": {}, "code_hash": ""}
        if store is not None and spec.snapshot_id:
            store.materialize(spec.snapshot_id, adir / "in")
        ctx = WorkerContext(spec, adir, snap, code_hash, worker_id, attempt, claim.get("batch_scale", 1.0),
                            Deadline(deadline_s) if deadline_s else None, ledger, clock)
        result = fn(spec, ctx)
        json.loads(canonical_json(result))                 # a non-serialisable result fails here, not at reconcile
        env = _envelope(spec, result, code_hash, worker_id, attempt, started, clock())
        atomic_write_json(adir / "result.json", env)
        (adir / "DONE").write_text(env["result_hash"], encoding="utf-8")     # written last: its presence means result.json is whole
        ledger.complete(spec.key, env["result_hash"], clock())
        seal_attempt(adir)
        if calibrator is not None:
            calibrator.observe(spec.name, claim["est_gb"], R.process_info(os.getpid())["rss_gb"])
        return {"state": DONE, "result_hash": env["result_hash"], "dir": str(adir)}
    except FirewallBreach:
        ledger.fail(spec.key, FAILED, "FirewallBreach: " + traceback.format_exc(limit=1).strip()[-200:], clock())
        raise                                                      # a firewall breach must reach the caller (fail closed)
    except BaseException as e:
        kind = classify_failure(e)
        if calibrator is not None and kind == OOM:
            calibrator.observe(spec.name, claim["est_gb"], oom=True)
        (adir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        nxt = ledger.fail(spec.key, kind, f"{type(e).__name__}: {e}"[:300], clock())
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return {"state": kind, "next": nxt, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------ scheduling
@dataclass(frozen=True)
class Plan:
    launch: tuple[str, ...]
    held: tuple[tuple[str, str], ...]        # (key, reason)
    workers: int
    limit: str


def plan_launches(ledger: ExperimentLedger, free_gb: float | None, running: int = 0, cores: int | None = None,
                  reserve_gb: float = R.RESERVE_GB, per_worker_gb: float = R.DEFAULT_JOB_GB, now: float | None = None,
                  grace_s: float = R.RAMP_S) -> Plan:
    """Which PENDING experiments may start now. Deterministic: ordered by (priority, key); memory decides how many run (a
    started experiment's estimate is reserved before the next is considered, so 5 launches cannot each see the same free GB)."""
    led = ledger.load()
    pend = [(e["spec"]["priority"], k, e) for k, e in led.items() if e["state"] == PENDING]
    pend.sort(key=lambda t: (t[0], t[1]))
    not_ready: list[tuple[str, str]] = []
    ready = []
    for pr, k, e in pend:
        waiting = [d for d in e["spec"].get("after", []) if led.get(d, {}).get("state") != DONE]
        if waiting:
            not_ready.append((k, f"waiting for prerequisites {[w[:8] for w in waiting]}"))
        elif now is not None and e.get("not_before") is not None and now < float(e["not_before"]):
            not_ready.append((k, f"backing off for {float(e['not_before']) - now:.0f}s more"))
        else:
            ready.append((pr, k, e))
    pend = ready
    if now is not None:                                # launched but not yet claimed: its slot and memory are already spoken for
        starting = {k: e for _, k, e in pend if e.get("launched") is not None and now - float(e["launched"]) < grace_s}
        running += len(starting)
        free_gb = None if free_gb is None else free_gb - sum(e["est_gb"] for e in starting.values())
        pend = [t for t in pend if t[1] not in starting]
    wc = R.worker_count(free_gb, per_worker_gb, cores, reserve_gb)
    # free memory already excludes the RSS of workers that are running, so memory bounds the NEW launches while the core
    # count bounds the total
    slots = max(0, min(wc["by_memory"], min(wc["by_cores"], R.MAX_WORKERS) - running))
    launch: list[str] = []
    held: list[tuple[str, str]] = []
    budget = (free_gb or 0.0) - reserve_gb
    for _, key, e in pend:
        if len(launch) >= slots:
            held.append((key, f"no free worker slot ({wc['limit']}; {running} running)"))
        elif e["est_gb"] > budget:
            held.append((key, f"needs {e['est_gb']:.1f} GB, {max(0.0, budget):.1f} GB left after reserve"))
        else:
            launch.append(key)
            budget -= e["est_gb"]
    return Plan(tuple(launch), tuple(not_ready + held), wc["workers"], wc["limit"])


def job_for(spec: ExperimentSpec, spec_dir: str | Path, out_root: str | Path, ledger_path: str | Path,
            python: str | None = None) -> R.Job:
    """A resources.Job that runs this experiment in its own process through `python -m engine.learning.compute worker`."""
    sp = Path(spec_dir) / f"{spec.key}.json"
    atomic_write_json(sp, spec.to_dict())
    cmd = [python or sys.executable, "-m", "engine.learning.compute", "worker", str(sp), str(out_root), str(ledger_path)]
    return R.Job(spec.key, cmd, est_gb=spec.est_gb, out_dir=str(Path(out_root) / spec.key), priority=spec.priority)


WORKERS: dict[str, Callable[[ExperimentSpec, WorkerContext], Any]] = {}


def register_worker(name: str):
    """Decorator: make an experiment function reachable by name from the subprocess entry point."""
    def deco(fn):
        if name in WORKERS and WORKERS[name] is not fn:
            raise ComputeError(f"worker {name} already registered")
        WORKERS[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ reconciliation
@dataclass(frozen=True)
class Reconciliation:
    accepted: tuple[tuple[str, Any], ...]          # (experiment key, result) in deterministic key order
    rejected: tuple[tuple[str, str, str], ...]     # (key, reason kind, detail)
    orphans: tuple[str, ...]                       # result folders with no ledger entry

    @property
    def clean(self) -> bool:
        return not self.rejected and not self.orphans

    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, kind, _ in self.rejected:
            out[kind] = out.get(kind, 0) + 1
        return out


def _read_attempts(out_root: Path, key: str) -> list[dict]:
    envs = []
    kd = out_root / key
    for a in sorted(kd.glob("attempt_*")) if kd.is_dir() else []:
        if (a / "DONE").is_file() and (a / "result.json").is_file():
            env = read_json(a / "result.json")
            if env:
                env["_dir"] = str(a)
                envs.append(env)
    return envs


def reconcile(ledger: ExperimentLedger, out_root: str | Path, current_code_hash_: str) -> Reconciliation:
    """Cross-check the ledger against what is on disk before any result reaches shared learning state. A result is accepted
    only if: the ledger says DONE; a whole result.json exists (DONE marker); its result hash re-computes; the recorded hash,
    seed, spec and code all match; the code is the code now loaded; and every finished attempt of the same experiment
    agrees (disagreeing duplicates prove non-determinism and reject the experiment)."""
    root = Path(out_root)
    led = ledger.load()
    accepted, rejected = [], []
    for key, e in sorted(led.items()):
        if e["state"] != DONE:
            continue
        envs = _read_attempts(root, key)
        if not envs:
            rejected.append((key, "missing_result", "ledger says DONE but no whole result exists"))
            continue
        spec = ExperimentSpec.from_dict(e["spec"])
        bad = None
        for env in envs:
            if stable_hash(env["result"], 20) != env["result_hash"]:
                bad = ("hash_mismatch", f"{env['_dir']}: result altered after writing")
            elif env["spec"] != spec.to_dict():
                bad = ("spec_mismatch", f"{env['_dir']}: spec differs from the ledger")
            elif env["seed"] != worker_seed_for(spec):
                bad = ("seed_mismatch", f"{env['_dir']}: worker used a different seed")
            elif env["code_hash"] != current_code_hash_ or e["code_hash"] != current_code_hash_:
                bad = ("stale_code", f"produced by {env['code_hash']}, engine is {current_code_hash_}")
            if bad:
                break
        if bad:
            rejected.append((key, *bad))
            continue
        hashes = {env["result_hash"] for env in envs}
        if len(hashes) > 1:
            rejected.append((key, "nondeterministic", f"{len(envs)} attempts gave {len(hashes)} different results"))
            continue
        if e["result_hash"] not in hashes:
            rejected.append((key, "ledger_mismatch", "ledger result hash not among the attempts on disk"))
            continue
        accepted.append((key, envs[-1]["result"]))
    orphans = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in led) if root.is_dir() else []
    return Reconciliation(tuple(accepted), tuple(rejected), tuple(orphans))


def merge_results(rec: Reconciliation, reducer: Callable[[Any, str, Any], Any], initial: Any) -> Any:
    """Fold accepted results in KEY order, never completion order, so the merged learning state does not depend on which
    worker finished first (a property test in the suite plants a shuffled completion order)."""
    acc = initial
    for key, result in sorted(rec.accepted, key=lambda kv: kv[0]):
        acc = reducer(acc, key, result)
    return acc


def compute_report(ledger: ExperimentLedger, rec: Reconciliation | None = None, now: float | None = None) -> str:
    counts = ledger.counts()
    lines = ["# Compute manager report", "Status: IMPLEMENTED - NOT VALIDATED", "",
             "states: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no experiments")]
    for key, e in sorted(ledger.load().items()):
        if e["state"] in (GAVE_UP, STALE, OOM, CRASHED, FAILED):
            lines.append(f"- {e['state']} {e['spec']['name']} ({key}) attempts={e['attempts']} est_gb={e['est_gb']:.1f}: {e['history'][-1][2]}")
    if rec is not None:
        lines += ["", f"reconciliation: {len(rec.accepted)} accepted, {len(rec.rejected)} rejected, {len(rec.orphans)} orphan folders"]
        lines += [f"- rejected {k} [{kind}] {detail}" for k, kind, detail in rec.rejected]
        lines += [f"- orphan {o}" for o in rec.orphans]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ memory calibration
class MemoryCalibrator:
    """Learns how much memory each kind of experiment REALLY needs, so the scheduler stops relying on a guess that has already
    cost an OOM. Per experiment name it keeps the observed peaks and every OOM; the estimate is the largest of: the caller's
    default, the biggest peak seen times a safety margin, and (after an OOM) the estimate that died times the growth factor.
    Persisted as JSON; a corrupt file is treated as empty (calibration is advice, never a gate)."""

    def __init__(self, path: str | Path, margin: float = 1.25, min_obs: int = 3, oom_growth: float = 1.5, cap_gb: float = 8.0):
        if margin < 1.0 or oom_growth <= 1.0 or min_obs < 1:
            raise ValueError("margin >= 1, oom_growth > 1, min_obs >= 1 required")
        self.path, self.margin, self.min_obs, self.oom_growth, self.cap_gb = Path(path), margin, min_obs, oom_growth, cap_gb

    def _load(self) -> dict:
        d = read_json(self.path, {})
        return d if isinstance(d, dict) else {}

    def observe(self, name: str, est_gb: float, peak_gb: float | None = None, oom: bool = False) -> None:
        if not (math.isfinite(est_gb) and est_gb > 0):
            raise ValueError("est_gb must be positive")
        if peak_gb is not None and not (math.isfinite(peak_gb) and peak_gb >= 0):
            raise ValueError("peak_gb must be a non-negative number")
        with R._lock(self.path):
            d = self._load()
            e = d.setdefault(name, {"peaks": [], "ooms": []})
            if peak_gb is not None:
                e["peaks"] = (e["peaks"] + [float(peak_gb)])[-50:]
            if oom:
                e["ooms"] = (e["ooms"] + [float(est_gb)])[-20:]
            atomic_write_json(self.path, d)

    def estimate(self, name: str, default_gb: float) -> float:
        e = self._load().get(name)
        if not e:
            return default_gb
        est = default_gb
        if len(e["peaks"]) >= self.min_obs:
            est = max(est, max(e["peaks"]) * self.margin)
        if e["ooms"]:
            est = max(est, max(e["ooms"]) * self.oom_growth)
        return float(min(self.cap_gb, est))

    def report(self) -> dict[str, dict]:
        return {n: {"n_peaks": len(e["peaks"]), "max_peak_gb": max(e["peaks"], default=None), "n_oom": len(e["ooms"])}
                for n, e in sorted(self._load().items())}


def calibrated(spec: ExperimentSpec, cal: MemoryCalibrator) -> ExperimentSpec:
    """The spec with its memory estimate replaced by what this kind of experiment has really needed. est_gb is deliberately not
    part of the experiment key, so recalibrating never makes a finished experiment look new."""
    return dataclasses.replace(spec, est_gb=cal.estimate(spec.name, spec.est_gb))


# ------------------------------------------------------------------------------------------------ building batches
def expand_grid(name: str, base_params: Mapping[str, Any], grid: Mapping[str, Sequence[Any]], seeds: Sequence[int], as_of: str,
                **kw: Any) -> list[ExperimentSpec]:
    """Every combination of `grid` values x `seeds`, in a deterministic order, de-duplicated by experiment key. The same call
    always returns the same specs (so a restarted run resubmits exactly what it submitted before and the duplicate guard, not
    luck, prevents rework)."""
    if not seeds:
        raise ValueError("at least one seed is required")
    keys = sorted(grid)
    combos: list[dict] = [{}]
    for k in keys:
        if not len(grid[k]):
            raise ValueError(f"grid axis {k} is empty")
        combos = [{**c, k: v} for c in combos for v in grid[k]]
    out, seen = [], set()
    for c in combos:
        for sd in sorted(set(int(s) for s in seeds)):
            sp = ExperimentSpec(name, {**dict(base_params), **c}, sd, as_of, **kw)
            errs = sp.validate()
            if errs:
                raise ValueError(f"invalid spec in grid: {errs}")
            if sp.key not in seen:
                seen.add(sp.key)
                out.append(sp)
    return out


def shard(specs: Sequence[ExperimentSpec], n_workers: int) -> list[list[ExperimentSpec]]:
    """Longest-processing-time-first split into `n_workers` bins balanced by memory estimate. Ties break by key, so the same
    input always gives the same partition regardless of the order specs arrive in."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    bins: list[list[ExperimentSpec]] = [[] for _ in range(n_workers)]
    load = [0.0] * n_workers
    for sp in sorted(specs, key=lambda s: (-s.est_gb, s.key)):
        i = min(range(n_workers), key=lambda j: (load[j], j))
        bins[i].append(sp)
        load[i] += sp.est_gb
    return bins


# ------------------------------------------------------------------------------------------------ deadlines
class Deadline:
    """Wall-clock budget for one experiment. Cooperative: the experiment calls `check()` in its loop; an overrun raises
    TimeoutError, which the worker records as CRASHED (retryable) rather than hanging the queue forever."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        if not (seconds > 0 and math.isfinite(seconds)):
            raise ValueError("deadline must be a positive number of seconds")
        self.seconds, self.clock = float(seconds), clock
        self.start = clock()

    def remaining(self) -> float:
        return self.seconds - (self.clock() - self.start)

    def check(self, what: str = "experiment") -> None:
        if self.remaining() < 0:
            raise TimeoutError(f"{what} exceeded its {self.seconds:.0f}s budget")


# ------------------------------------------------------------------------------------------------ isolation audit
def seal_attempt(adir: str | Path) -> int:
    """Make a finished attempt directory read-only so a later process cannot quietly edit a result that was reconciled.
    Returns the number of files sealed."""
    n = 0
    for p in Path(adir).rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
            n += 1
    return n


def verify_isolation(ledger: ExperimentLedger, out_root: str | Path) -> list[dict]:
    """Evidence that experiments did not touch each other or shared state: every entry under an experiment folder must be an
    attempt directory (or the resumable progress file), attempt numbers must not exceed the ledger's count, and a DONE
    experiment's winning attempt must contain both result and marker. Returns findings; empty means isolated."""
    root = Path(out_root)
    led = ledger.load()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for kd in sorted(p for p in root.iterdir() if p.is_dir()):
        e = led.get(kd.name)
        if e is None:
            continue                                                      # orphans are reconcile()'s finding
        for child in sorted(kd.iterdir()):
            if child.is_dir() and child.name.startswith("attempt_"):
                try:
                    num = int(child.name.split("_")[1])
                except ValueError:
                    out.append({"key": kd.name, "kind": "bad_attempt_name", "detail": child.name})
                    continue
                if num > e["attempts"]:
                    out.append({"key": kd.name, "kind": "phantom_attempt", "detail": f"{child.name} but ledger counts {e['attempts']}"})
            elif child.name != "progress.json":
                out.append({"key": kd.name, "kind": "stray_file", "detail": child.name})
        if e["state"] == DONE:
            done_dirs = [a for a in kd.glob("attempt_*") if (a / "DONE").is_file()]
            if not done_dirs:
                out.append({"key": kd.name, "kind": "done_without_marker", "detail": "ledger DONE but no attempt has a DONE marker"})
    return out


# ------------------------------------------------------------------------------------------------ merge accounting
class MergeLog:
    """Which results have already been folded into learning state, and with which result hash. Merging is two-phase so a crash
    can neither lose nor double-count a result: `plan` says what is new, the caller applies it to ITS state and persists that
    state, then `commit` records it. A result that was merged and later re-appears with a DIFFERENT hash is a conflict, not an
    update: the learning state already used the old numbers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def merged(self) -> dict[str, str]:
        d = read_json(self.path, {})
        return d.get("merged", {}) if isinstance(d, dict) else {}

    def plan(self, rec: Reconciliation) -> dict:
        done = self.merged()
        new, same, conflict = [], [], []
        for key, result in sorted(rec.accepted, key=lambda kv: kv[0]):
            h = stable_hash(result, 20)
            if key not in done:
                new.append((key, result, h))
            elif done[key] == h:
                same.append(key)
            else:
                conflict.append({"key": key, "merged_hash": done[key], "now_hash": h})
        return {"new": new, "already_merged": same, "conflicts": conflict}

    def commit(self, plan: dict) -> int:
        if plan["conflicts"]:
            raise ComputeError(f"refusing to commit with unresolved merge conflicts: {[c['key'] for c in plan['conflicts']]}")
        with R._lock(self.path):
            d = read_json(self.path, {}) or {}
            merged = d.get("merged", {})
            for key, _, h in plan["new"]:
                merged[key] = h
            atomic_write_json(self.path, {"merged": merged, "n": len(merged), "digest": stable_hash(sorted(merged.items()), 20)})
        return len(plan["new"])


def merge_new(rec: Reconciliation, log: MergeLog, reducer: Callable[[Any, str, Any], Any], state: Any) -> tuple[Any, dict]:
    """Fold only not-yet-merged results into `state`, in key order. Returns (new state, plan). The caller persists the state and
    THEN calls log.commit(plan); if it crashes in between, the next run sees the same plan and repeats the merge into the
    state it reloaded from disk, which is exactly right."""
    plan = log.plan(rec)
    for key, result, _ in plan["new"]:
        state = reducer(state, key, result)
    return state, plan


# ------------------------------------------------------------------------------------------------ supervision
class Supervisor:
    """One `tick` of the whole compute cycle: mark stale work, recover dead workers, decide what may start, start it. All
    side effects go through injected callables (`free_fn`, `launcher`, `alive_fn`), so the loop is unit-testable with no
    processes and, in production, is wired to engine.resources (JobQueue / ProcRegistry). A launched experiment is not
    launched again until `grace_s` has passed or it has been claimed, so a slow start is not mistaken for a free slot."""

    def __init__(self, ledger: ExperimentLedger, free_fn: Callable[[], float | None], launcher: Callable[[ExperimentSpec], Any],
                 alive_fn: Callable[[str], bool], code_hash: str, cores: int | None = None, heartbeat_s: float = R.HEARTBEAT_S, grace_s: float = R.RAMP_S,
                 reserve_gb: float = R.RESERVE_GB):
        self.ledger, self.free_fn, self.launcher, self.alive_fn = ledger, free_fn, launcher, alive_fn
        self.code_hash, self.cores = code_hash, cores
        self.heartbeat_s, self.grace_s, self.reserve_gb = heartbeat_s, grace_s, reserve_gb

    def tick(self, now: float) -> dict:
        stale = self.ledger.mark_stale(self.code_hash, now)
        recovered = self.ledger.recover(now, self.alive_fn, self.heartbeat_s)
        counts = self.ledger.counts()
        plan = plan_launches(self.ledger, self.free_fn(), counts.get(RUNNING, 0), self.cores, self.reserve_gb, now=now,
                             grace_s=self.grace_s)
        launched, failed = [], []
        for key in plan.launch:
            spec = self.ledger.spec(key)
            try:
                self.launcher(spec)
                self.ledger.mark_launched(key, now)
                launched.append(key)
            except Exception as ex:                                       # a launch that fails is held, never silently dropped
                failed.append({"key": key, "error": f"{type(ex).__name__}: {ex}"})
        return {"now": now, "stale": stale, "recovered": recovered, "launched": launched, "launch_failed": failed,
                "held": list(plan.held), "workers": plan.workers, "limit": plan.limit, "counts": self.ledger.counts()}


def experiment_stats(ledger: ExperimentLedger) -> dict[str, dict]:
    """Per experiment name: runs, success share, mean wall time of successful attempts (from the ledger history), and how
    often it needed a retry - the numbers the research planner needs to price a proposed experiment."""
    out: dict[str, dict] = {}
    for key, e in ledger.load().items():
        s = out.setdefault(e["spec"]["name"], {"runs": 0, "done": 0, "failed": 0, "gave_up": 0, "retries": 0, "_dur": []})
        s["runs"] += 1
        s["done"] += e["state"] == DONE
        s["gave_up"] += e["state"] == GAVE_UP
        s["failed"] += e["state"] in (FAILED, OOM, CRASHED, STALE)
        s["retries"] += max(0, e["attempts"] - 1)
        start = None
        for t, st, _ in e["history"]:
            if st == RUNNING:
                start = t
            elif st == DONE and start is not None:
                s["_dur"].append(t - start)
    for s in out.values():
        d = s.pop("_dur")
        s["success_share"] = s["done"] / s["runs"] if s["runs"] else float("nan")
        s["mean_seconds"] = float(np.mean(d)) if d else None
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------------ retry backoff
@dataclass(frozen=True)
class BackoffPolicy:
    """Deterministic exponential backoff for retries. The jitter comes from a hash of (experiment, attempt), not a clock or an
    RNG, so a replay of the same failure history schedules the same retry times."""
    base_s: float = 30.0
    factor: float = 2.0
    cap_s: float = 1800.0
    jitter: float = 0.25

    def validate(self) -> list[str]:
        errs = []
        if not (self.base_s >= 0 and self.factor >= 1.0 and self.cap_s >= self.base_s and 0 <= self.jitter < 1):
            errs.append("need base_s >= 0, factor >= 1, cap_s >= base_s, 0 <= jitter < 1")
        return errs

    def delay(self, key: str, attempt: int) -> float:
        raw = min(self.cap_s, self.base_s * self.factor ** max(0, attempt - 1))
        u = int(stable_hash([key, attempt], 8), 16) / 16 ** 8               # deterministic in [0, 1)
        return float(raw * (1.0 - self.jitter + 2.0 * self.jitter * u))


# ------------------------------------------------------------------------------------------------ dependencies between experiments
def dependency_order(specs: Sequence[ExperimentSpec]) -> list[ExperimentSpec]:
    """Order a batch so every experiment follows the ones it depends on (`spec.after`, a tuple of experiment keys). Ties break
    by key. A dependency that is neither in the batch nor already known, or a cycle, is an error - a graph that can never
    finish must be refused up front, not discovered after hours of compute."""
    by = {s.key: s for s in specs}
    out: list[ExperimentSpec] = []
    state: dict[str, int] = {}

    def visit(k: str, stack: tuple[str, ...]) -> None:
        if state.get(k) == 2:
            return
        if state.get(k) == 1:
            raise ComputeError(f"dependency cycle: {' -> '.join(stack + (k,))}")
        state[k] = 1
        for d in sorted(by[k].after):
            if d in by:
                visit(d, stack + (k,))
        state[k] = 2
        out.append(by[k])

    for k in sorted(by):
        visit(k, ())
    return out


def submit_batch(ledger: ExperimentLedger, specs: Sequence[ExperimentSpec], now: float, code_hash: str) -> dict:
    """Submit a dependency-ordered batch. Prerequisites must be in the batch or already in the ledger. Experiments that are
    duplicates of finished or active ones are reported, not raised, so a restarted run can resubmit its whole plan safely."""
    known = set(ledger.load())
    batch = dependency_order(specs)
    keys = {s.key for s in batch}
    for s in batch:
        missing = [d for d in s.after if d not in keys and d not in known]
        if missing:
            raise ComputeError(f"{s.name} ({s.key}) depends on unknown experiments {missing}")
    queued, duplicates = [], []
    for s in batch:
        try:
            ledger.submit(s, now, code_hash)
            queued.append(s.key)
        except DuplicateExperiment as e:
            duplicates.append({"key": s.key, "why": str(e)})
    return {"queued": queued, "duplicates": duplicates}


def blocked_by_failure(ledger: ExperimentLedger) -> list[dict]:
    """PENDING experiments that can never run because a prerequisite has GAVE_UP (transitively). They are reported for
    escalation instead of waiting forever."""
    led = ledger.load()
    dead = {k for k, e in led.items() if e["state"] == GAVE_UP}
    changed = True
    blocked: dict[str, str] = {}
    while changed:
        changed = False
        for k, e in sorted(led.items()):
            if k in blocked or e["state"] != PENDING:
                continue
            bad = [d for d in e["spec"].get("after", []) if d in dead or d in blocked]
            if bad:
                blocked[k] = bad[0]
                changed = True
    return [{"key": k, "name": led[k]["spec"]["name"], "blocked_by": v} for k, v in sorted(blocked.items())]


# ------------------------------------------------------------------------------------------------ run manifest
def write_run_manifest(path: str | Path, ledger: ExperimentLedger, rec: Reconciliation, code_hash: str,
                       snapshot_ids: Iterable[str] = ()) -> dict:
    """Freeze what a batch produced: every experiment with its state and result hash, what reconciliation accepted and
    rejected, the code and the snapshots. The manifest hash reuses engine.checkpoint._body_hash, so a batch can later be
    compared with a rerun of itself (`compare_manifests`) - that comparison IS the reproducibility check of section 56."""
    from engine import checkpoint as bundle
    led = ledger.load()
    body = {"code_hash": code_hash,
            "experiments": {k: {"name": e["spec"]["name"], "state": e["state"], "attempts": e["attempts"],
                                "result_hash": e["result_hash"], "seed": e["spec"]["seed"]} for k, e in sorted(led.items())},
            "accepted": [k for k, _ in rec.accepted], "rejected": [list(r) for r in rec.rejected], "orphans": list(rec.orphans),
            "snapshots": sorted(set(snapshot_ids))}
    doc = {"body": json.loads(canonical_json(body)), "sha256": None}
    doc["sha256"] = bundle._body_hash(doc["body"])
    atomic_write_json(path, doc)
    return doc


def verify_manifest(path: str | Path) -> list[str]:
    from engine import checkpoint as bundle
    doc = read_json(path)
    if not isinstance(doc, dict) or "body" not in doc:
        return ["manifest missing or unreadable"]
    return [] if bundle._body_hash(doc["body"]) == doc.get("sha256") else ["manifest body altered after writing"]


def compare_manifests(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    """Two runs of the same plan: which experiments gave different results (a determinism failure), which exist in only one,
    and whether the code differed (in which case a difference is expected and proves nothing)."""
    ea, eb = a["body"]["experiments"], b["body"]["experiments"]
    common = sorted(set(ea) & set(eb))
    differ = [k for k in common if ea[k]["result_hash"] != eb[k]["result_hash"] and ea[k]["state"] == eb[k]["state"] == DONE]
    return {"differ": differ, "only_a": sorted(set(ea) - set(eb)), "only_b": sorted(set(eb) - set(ea)),
            "same_code": a["body"]["code_hash"] == b["body"]["code_hash"], "compared": len(common),
            "reproducible": not differ and a["body"]["code_hash"] == b["body"]["code_hash"]}


# ------------------------------------------------------------------------------------------------ admission control
MIN_FREE_GB = 2.5            # CONTEXT rule 10: no real run starts with less than this much RAM free


def admit(est_gb: float, free_fn: Callable[[], float | None] | None = None, reserve_gb: float = R.RESERVE_GB,
          min_free_gb: float = MIN_FREE_GB) -> tuple[bool, str]:
    """RAM-aware admission (psutil via engine.resources.memory_gb). Fails closed: unreadable memory refuses. Two rules must
    both hold - the machine has at least `min_free_gb` free right now, and after this job's estimate is taken there is still
    `reserve_gb` left for the OS and the long experiments already running."""
    free = (free_fn or (lambda: R.memory_gb()[0]))()
    if free is None or not math.isfinite(free):
        return False, "free memory unreadable (fail closed)"
    if free < min_free_gb:
        return False, f"only {free:.1f} GB free (< {min_free_gb:.1f} GB rule)"
    if free - est_gb < reserve_gb:
        return False, f"needs {est_gb:.1f} GB but only {max(0.0, free - reserve_gb):.1f} GB is available after the {reserve_gb:.1f} GB reserve"
    return True, "ok"


# ------------------------------------------------------------------------------------------------ real subprocess workers
@dataclass(frozen=True)
class SubprocessResult:
    returncode: int | None
    kind: str                # DONE | OOM | CRASHED | FAILED
    elapsed_s: float
    peak_gb: float | None
    stderr_tail: str
    timed_out: bool
    killed_for_memory: bool


def run_subprocess(cmd: Sequence[str], out_dir: str | Path, timeout_s: float, mem_limit_gb: float | None = None,
                   poll_s: float = 0.05, cwd: str | None = None, env: Mapping[str, str] | None = None,
                   rss_fn: Callable[[int], float | None] | None = None) -> SubprocessResult:
    """Run one worker as a real child process with its own stdout/stderr files. The wall-clock budget and an optional RSS
    ceiling are enforced by polling; only the child THIS call started is ever killed (never by image name). The outcome is
    classified with `classify_failure`: a MemoryError message, an allocation-failure exit code or an over-limit RSS is OOM,
    a timeout or a signal death is CRASHED, an ordinary non-zero exit is FAILED."""
    import subprocess
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rss = rss_fn or (lambda pid: R.process_info(pid)["rss_gb"])
    t0 = time.monotonic()
    peak, timed_out, mem_kill = 0.0, False, False
    with open(out / "stdout.txt", "wb") as fo, open(out / "stderr.txt", "wb") as fe:
        proc = subprocess.Popen(list(cmd), stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, cwd=cwd, env=None if env is None else dict(env))
        while proc.poll() is None:
            cur = rss(proc.pid)
            peak = max(peak, cur or 0.0)
            if mem_limit_gb is not None and cur is not None and cur > mem_limit_gb:
                mem_kill = True
                proc.kill()
                break
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                proc.kill()
                break
            time.sleep(poll_s)
        proc.wait()
    tail = (out / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-500:]
    if mem_kill:
        kind = OOM
    elif proc.returncode == 0 and not timed_out:
        kind = DONE
    else:
        kind = classify_failure(None, proc.returncode, tail, timed_out)
        if kind == CRASHED and not timed_out and proc.returncode is not None and proc.returncode > 0:
            kind = FAILED
    return SubprocessResult(proc.returncode, kind, time.monotonic() - t0, peak or None, tail, timed_out, mem_kill)


def settle(ledger: ExperimentLedger, key: str, res: SubprocessResult, now: float) -> str:
    """Make the ledger agree with what actually happened to the child. A child that recorded its own result has already moved
    the entry out of RUNNING and nothing is touched; a child that died (or exited 0 without recording anything) leaves the
    entry RUNNING, so it is failed here with the classified kind and retried or given up by the ledger's normal rules."""
    e = ledger.get(key)
    if e is None or e["state"] != RUNNING:
        return e["state"] if e else "UNKNOWN"
    kind = res.kind if res.kind != DONE else CRASHED
    note = "exited 0 without recording a result" if res.kind == DONE else f"exit {res.returncode}: {res.stderr_tail[-160:]}"
    return ledger.fail(key, kind, note, now)


def run_experiment_process(spec: ExperimentSpec, ledger: ExperimentLedger, out_root: str | Path, work_dir: str | Path, now: float,
                           timeout_s: float = 600.0, modules: Sequence[str] = (), snapshot_root: str | Path | None = None,
                           mem_limit_gb: float | None = None, free_fn: Callable[[], float | None] | None = None,
                           python: str | None = None) -> dict:
    """The whole real-process cycle for one experiment: admission (the 2.5 GB rule), write the spec, run the worker as a child
    (`python -m engine.learning.compute worker`, with `modules` imported first so its @register_worker functions exist), then
    settle the ledger with whatever happened. Logs go under `work_dir`, never inside the experiment's own result folder."""
    ok, why = admit(spec.est_gb, free_fn)
    if not ok:
        return {"launched": False, "why": why, "state": (ledger.get(spec.key) or {}).get("state")}
    work = Path(work_dir)
    sp = work / "specs" / f"{spec.key}.json"
    atomic_write_json(sp, spec.to_dict())
    cmd = [python or sys.executable, "-m", "engine.learning.compute", "worker", str(sp), str(out_root), str(ledger.path)]
    if snapshot_root is not None:
        cmd.append(str(snapshot_root))
    env = {**os.environ, "W7_WORKER_MODULES": ",".join(modules)}
    res = run_subprocess(cmd, work / "logs" / spec.key, timeout_s, mem_limit_gb, cwd=str(Path(__file__).resolve().parents[2]), env=env)
    return {"launched": True, "result": res, "state": settle(ledger, spec.key, res, now)}


# ------------------------------------------------------------------------------------------------ adjudicating disagreement
def adjudicate(ledger: ExperimentLedger, out_root: str | Path, key: str) -> dict:
    """Two workers returned different results for one key: say why, using engine.repro.diagnose for the numeric difference.
    Attempts made with different code or seeds explain themselves; identical code and seed with different numbers is
    non-determinism (a bug). Nothing is auto-accepted: with three or more attempts a strict majority is REPORTED as the
    candidate, but the experiment stays flagged until someone fixes the source of the disagreement."""
    from engine import repro
    envs = _read_attempts(Path(out_root), key)
    if len(envs) < 2:
        return {"key": key, "agree": True, "attempts": len(envs), "action": "nothing to adjudicate"}
    groups: dict[str, list[dict]] = {}
    for env in envs:
        groups.setdefault(env["result_hash"], []).append(env)
    summary = [{"attempt": e["attempt"], "worker": e["worker"], "result_hash": e["result_hash"], "code_hash": e["code_hash"],
                "seed": e["seed"]} for e in envs]
    if len(groups) == 1:
        return {"key": key, "agree": True, "attempts": len(envs), "summary": summary, "action": "none"}
    first, second = [g[0] for g in list(groups.values())[:2]]
    diag = repro.diagnose(key, first["result"], second["result"])
    if len({e["code_hash"] for e in envs}) > 1:
        cause = "code_difference"
    elif len({e["seed"] for e in envs}) > 1:
        cause = "seed_difference"
    else:
        cause = "nondeterministic"
    top = max(groups.values(), key=len)
    majority = top[0]["result_hash"] if len(envs) >= 3 and len(top) * 2 > len(envs) else None
    action = ("rerun under the current code" if cause == "code_difference" else "rerun with one fixed seed" if cause == "seed_difference"
              else "run a third attempt" if len(envs) < 3 else "escalate: fix the source of non-determinism" if majority is None
              else "majority reported; still flagged until the non-determinism is fixed")
    return {"key": key, "agree": False, "attempts": len(envs), "cause": cause, "n_distinct": len(groups), "majority_hash": majority,
            "difference": {k: diag.get(k) for k in ("first_difference", "detail", "causes", "max_abs_diff", "max_rel_diff")},
            "summary": summary, "action": action}


# ------------------------------------------------------------------------------------------------ subprocess entry
def main(argv: Sequence[str] | None = None) -> int:
    """python -m engine.learning.compute worker <spec.json> <out_root> <ledger.json> [snapshot_root]"""
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 4 or a[0] != "worker":
        print(main.__doc__)
        return 2
    spec = ExperimentSpec.from_dict(read_json(a[1]))
    import importlib
    for mod in filter(None, os.environ.get("W7_WORKER_MODULES", "").split(",")):
        importlib.import_module(mod)
    fn = WORKERS.get(spec.name)
    if fn is None:
        print(f"no registered worker named {spec.name}", file=sys.stderr)
        return 3
    store = SnapshotStore(a[4]) if len(a) > 4 else None
    res = run_worker(spec, fn, ExperimentLedger(a[3]), store, a[2], f"pid{os.getpid()}", time.time())
    return 0 if res["state"] == DONE else 1


if __name__ == "__main__":
    # run the canonical module, not this __main__ copy: worker modules do `from engine.learning import compute`, and their
    # @register_worker calls must land in the same WORKERS table this entry point reads
    from engine.learning.compute import main as _main
    raise SystemExit(_main())
