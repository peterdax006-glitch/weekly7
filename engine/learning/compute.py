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
from typing import Any, Callable, Iterable, Mapping, Sequence

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
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})


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
        body = {"as_of": str(as_date(as_of)), "state": state, "code_hash": code_hash}
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

    def __init__(self, path: str | Path, max_attempts: int = 3, oom_growth: float = 1.5, max_est_gb: float = 8.0):
        if max_attempts < 1 or oom_growth <= 1.0:
            raise ValueError("max_attempts >= 1 and oom_growth > 1 required")
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
            e.update(worker=worker_id, heartbeat=float(now))
            self._note(e, now, RUNNING, f"attempt {e['attempts']} by {worker_id}")
            return json.loads(json.dumps(e))

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
            gone = not alive(e.get("worker"))
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
    _ledger: "ExperimentLedger | None" = None
    _clock: Callable[[], float] = time.time

    @property
    def seed(self) -> int:
        return worker_seed_for(self.spec)

    def rng(self, stream: str = "") -> np.random.Generator:
        """A fresh generator per (experiment, stream): independent of worker, order and other streams."""
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(stable_hash(stream, 8), 16)]))

    def beat(self) -> None:
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
            "attempt": attempt, "started": started, "ended": ended, "result": result, "result_hash": stable_hash(result, 20)}


def attempt_dir(out_root: str | Path, key: str, attempt: int) -> Path:
    return Path(out_root) / key / f"attempt_{attempt:02d}"


def run_worker(spec: ExperimentSpec, fn: Callable[[ExperimentSpec, WorkerContext], Any], ledger: ExperimentLedger,
               store: SnapshotStore | None, out_root: str | Path, worker_id: str, now: float,
               code_hash: str | None = None, clock: Callable[[], float] = time.time) -> dict:
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
        ctx = WorkerContext(spec, adir, snap, code_hash, worker_id, attempt, claim.get("batch_scale", 1.0), ledger, clock)
        result = fn(spec, ctx)
        json.loads(canonical_json(result))                       # a non-serialisable result fails here, not at reconcile
        env = _envelope(spec, result, code_hash, worker_id, attempt, started, clock())
        atomic_write_json(adir / "result.json", env)
        (adir / "DONE").write_text(env["result_hash"], encoding="utf-8")     # written last: its presence means result.json is whole
        ledger.complete(spec.key, env["result_hash"], clock())
        return {"state": DONE, "result_hash": env["result_hash"], "dir": str(adir)}
    except FirewallBreach:
        ledger.fail(spec.key, FAILED, "FirewallBreach: " + traceback.format_exc(limit=1).strip()[-200:], clock())
        raise                                                      # a firewall breach must reach the caller (fail closed)
    except BaseException as e:
        kind = classify_failure(e)
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
                  reserve_gb: float = R.RESERVE_GB, per_worker_gb: float = R.DEFAULT_JOB_GB) -> Plan:
    """Which PENDING experiments may start now. Deterministic: ordered by (priority, key); memory decides how many run (a
    started experiment's estimate is reserved before the next is considered, so 5 launches cannot each see the same free GB)."""
    wc = R.worker_count(free_gb, per_worker_gb, cores, reserve_gb)
    slots = max(0, wc["workers"] - running)
    pend = [(e["spec"]["priority"], k, e) for k, e in ledger.load().items() if e["state"] == PENDING]
    pend.sort(key=lambda t: (t[0], t[1]))
    launch, held = [], []
    budget = (free_gb or 0.0) - reserve_gb
    for _, key, e in pend:
        if len(launch) >= slots:
            held.append((key, f"no free worker slot ({wc['limit']})"))
        elif e["est_gb"] > budget:
            held.append((key, f"needs {e['est_gb']:.1f} GB, {max(0.0, budget):.1f} GB left after reserve"))
        else:
            launch.append(key)
            budget -= e["est_gb"]
    return Plan(tuple(launch), tuple(held), wc["workers"], wc["limit"])


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


# ------------------------------------------------------------------------------------------------ subprocess entry
def main(argv: Sequence[str] | None = None) -> int:
    """python -m engine.learning.compute worker <spec.json> <out_root> <ledger.json> [snapshot_root]"""
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 4 or a[0] != "worker":
        print(main.__doc__)
        return 2
    spec = ExperimentSpec.from_dict(read_json(a[1]))
    fn = WORKERS.get(spec.name)
    if fn is None:
        print(f"no registered worker named {spec.name}", file=sys.stderr)
        return 3
    store = SnapshotStore(a[4]) if len(a) > 4 else None
    res = run_worker(spec, fn, ExperimentLedger(a[3]), store, a[2], f"pid{os.getpid()}", time.time())
    return 0 if res["state"] == DONE else 1


if __name__ == "__main__":
    raise SystemExit(main())
