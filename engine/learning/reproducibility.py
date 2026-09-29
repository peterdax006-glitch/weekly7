"""Reproducibility of learning experiments (contract C62 section 56; checklist A13 support, H12, L14; section 62 test 12).

Every learning experiment must be reproducible from SEVEN things: code hash, data hash, configuration hash, random seed,
worker configuration, memory snapshot and experiment id. If any one differs from what the result recorded, the result is
marked with the exact reason - it is never silently treated as the same experiment.

  ReproRecord         frozen record of the seven components (+ artefact hashes and the derived run key)
  WorkerConfig        interpreter / library versions / threads / hash seed - captured, compared, classified hard vs soft
  compare()           recorded vs current -> ReproVerdict with one ReproLabel per differing component
  ReproVerdict.mark   stamps a result dict: status NOT_REPRODUCIBLE plus the labels
  CodeWatcher         detects source edited DURING a run (section 62 test 12): a stale result is rejected, not averaged in
  reproduce()         rerun the experiment and compare artefact hashes, naming the first differing element
  ReproLedger         append-only record of runs; `conflicts` finds equal run keys with unequal results (a hidden input)

Reuses engine.repro (artifact_hash, hash_artifacts, config_hash, diagnose, run_twice), engine.provenance (code_hash, stale) and
core.stable_hash: no new hash function is defined here. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core import FirewallBreach, _StrEnum, stable_hash


class ReproLabel(_StrEnum):
    REPRODUCIBLE = "REPRODUCIBLE"
    CODE_CHANGED = "CODE_CHANGED"
    CODE_EDITED_MID_RUN = "CODE_EDITED_MID_RUN"
    DATA_CHANGED = "DATA_CHANGED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    SEED_CHANGED = "SEED_CHANGED"
    WORKER_CHANGED = "WORKER_CHANGED"
    WORKER_CHANGED_SOFT = "WORKER_CHANGED_SOFT"
    MEMORY_CHANGED = "MEMORY_CHANGED"
    EXPERIMENT_MISMATCH = "EXPERIMENT_MISMATCH"
    RECORD_INCOMPLETE = "RECORD_INCOMPLETE"
    RESULT_DIFFERS = "RESULT_DIFFERS"


# labels that make a result not reproducible from its record; the soft worker label only warns
HARD_LABELS = frozenset({ReproLabel.CODE_CHANGED, ReproLabel.CODE_EDITED_MID_RUN, ReproLabel.DATA_CHANGED, ReproLabel.CONFIG_CHANGED,
                         ReproLabel.SEED_CHANGED, ReproLabel.WORKER_CHANGED, ReproLabel.MEMORY_CHANGED,
                         ReproLabel.EXPERIMENT_MISMATCH, ReproLabel.RECORD_INCOMPLETE, ReproLabel.RESULT_DIFFERS})
COMPONENTS = ("code_hash", "data_hash", "config_hash", "seed", "worker", "memory_hash", "experiment_id")
_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# ---------------------------------------------------------------- worker configuration
@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """The execution environment. Float summation order (threads, BLAS build) and library versions can change numeric
    results, so they are part of the identity of a run."""
    python: str = ""
    platform: str = ""
    packages: Mapping[str, str] = dataclasses.field(default_factory=dict)
    n_workers: int = 1
    threads: Mapping[str, str] = dataclasses.field(default_factory=dict)
    hashseed: str = ""

    # fields whose change can alter numeric output for the same code/data/seed (hard) vs bookkeeping only (soft)
    HARD = ("python", "platform", "packages", "hashseed")
    SOFT = ("n_workers", "threads")

    def fingerprint(self) -> str:
        return stable_hash(self)

    def diff(self, other: "WorkerConfig") -> dict[str, tuple[Any, Any]]:
        out = {}
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if dict(a) != dict(b) if isinstance(a, Mapping) else a != b:
                out[f.name] = (a, b)
        return out

    def hard_diff(self, other: "WorkerConfig") -> dict[str, tuple[Any, Any]]:
        return {k: v for k, v in self.diff(other).items() if k in self.HARD}


def capture_worker(n_workers: int = 1, packages: Sequence[str] = _PACKAGES) -> WorkerConfig:
    """Read the running environment. A missing package is recorded as 'absent', never skipped (absence is information)."""
    pk = {}
    for name in packages:
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = "absent"
    return WorkerConfig(f"{sys.version_info.major}.{sys.version_info.minor}", f"{platform.system()}-{platform.machine()}", pk,
                        int(n_workers), {k: os.environ[k] for k in _THREAD_ENV if k in os.environ},
                        os.environ.get("PYTHONHASHSEED", ""))


# ---------------------------------------------------------------- the record
@dataclasses.dataclass(frozen=True)
class ReproRecord:
    experiment_id: str
    code_hash: str
    data_hash: str
    config_hash: str
    seed: int | None
    worker: WorkerConfig
    memory_hash: str
    artifact_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_real: str = ""
    code_files: tuple[str, ...] = ()
    code_mixed: tuple[str, ...] = ()

    def missing(self) -> tuple[str, ...]:
        out = []
        for c in COMPONENTS:
            v = getattr(self, c)
            if v is None or v == "" or (c == "worker" and not v.python):
                out.append(c)
        return tuple(out)

    def validate(self) -> list[str]:
        errs = [f"component {c} missing" for c in self.missing()]
        if self.seed is not None and not isinstance(self.seed, int):
            errs.append("seed must be an int")
        return errs

    @property
    def run_key(self) -> str:
        """Identity of the run: equal keys must give equal artefacts. created_real and artefact hashes are outputs, not inputs."""
        return stable_hash({c: (getattr(self, c).fingerprint() if c == "worker" else getattr(self, c)) for c in COMPONENTS}, 24)

    def to_dict(self) -> dict:
        d = json.loads(json.dumps(dataclasses.asdict(self), default=str))
        d["run_key"] = self.run_key
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "ReproRecord":
        w = dict(d.get("worker") or {})
        worker = WorkerConfig(w.get("python", ""), w.get("platform", ""), dict(w.get("packages") or {}), int(w.get("n_workers", 1)),
                              dict(w.get("threads") or {}), w.get("hashseed", ""))
        return cls(d["experiment_id"], d["code_hash"], d["data_hash"], d["config_hash"], d.get("seed"), worker, d["memory_hash"],
                   dict(d.get("artifact_hashes") or {}), d.get("created_real", ""), tuple(d.get("code_files") or ()),
                   tuple(d.get("code_mixed") or ()))


def _hash_data(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    from engine.repro import artifact_hash
    return artifact_hash(data)[:24]


def make_record(cfg: Mapping, seed: int | None, data: Any = None, memory_hash: str = "", experiment_id: str | None = None,
                code_hash: str | None = None, worker: WorkerConfig | None = None, artifacts: Mapping[str, Any] | None = None,
                created_real: str | None = None, code_files: Sequence[str] | None = None) -> ReproRecord:
    """Assemble a record from a running experiment. `data` may be the object (hashed with engine.repro.artifact_hash) or a
    ready hash string. The experiment id is DERIVED from the other six components unless given, so it cannot disagree with them."""
    from engine import provenance
    from engine.repro import config_hash, hash_artifacts
    files = tuple(code_files) if code_files is not None else tuple(provenance.loaded_code())
    ch = code_hash if code_hash is not None else provenance.code_hash(list(files))
    mixed = tuple(provenance.code_mixed(list(files))) if code_files is None else ()
    w = worker or capture_worker()
    d = _hash_data(data)
    cfgh = config_hash(dict(cfg))
    base = {"code_hash": ch, "data_hash": d, "config_hash": cfgh, "seed": seed, "worker": w.fingerprint(), "memory_hash": memory_hash}
    eid = experiment_id or ("exp-" + stable_hash(base, 16))
    return ReproRecord(eid, ch, d, cfgh, seed, w, memory_hash, hash_artifacts(dict(artifacts)) if artifacts else {},
                       created_real or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), files, mixed)


def memory_snapshot_hash(items: Iterable, taken_at) -> str:
    """Root hash of exactly which items a run could use (memory_firewall.snapshot_memory)."""
    from .memory_firewall import snapshot_memory
    return snapshot_memory(items, taken_at).root


# ---------------------------------------------------------------- comparing
@dataclasses.dataclass(frozen=True)
class ReproVerdict:
    labels: tuple[ReproLabel, ...]
    diffs: Mapping[str, Any]
    run_key_recorded: str = ""
    run_key_current: str = ""

    @property
    def reproducible(self) -> bool:
        return not any(lab in HARD_LABELS for lab in self.labels)

    @property
    def status(self) -> str:
        if self.reproducible and not self.labels:
            return "REPRODUCIBLE"
        if self.reproducible:
            return "REPRODUCIBLE_WITH_NOTES: " + ",".join(map(str, self.labels))
        return "NOT_REPRODUCIBLE: " + ",".join(str(lab) for lab in self.labels if lab in HARD_LABELS)

    def mark(self, result: Mapping) -> dict:
        """Copy of a result carrying its reproducibility label. A result that cannot be reproduced from its record is
        marked, never dropped and never passed off as clean."""
        out = dict(result)
        out["repro"] = {"status": self.status, "labels": [str(lab) for lab in self.labels], "reproducible": self.reproducible,
                        "run_key_recorded": self.run_key_recorded, "run_key_current": self.run_key_current}
        return out

    def require(self) -> "ReproVerdict":
        if not self.reproducible:
            raise FirewallBreach(f"result is {self.status}")
        return self


def compare(recorded: ReproRecord, current: ReproRecord, check_artifacts: bool = False) -> ReproVerdict:
    """Component-by-component comparison. Every differing component gets its own label; an incomplete record can never be
    called reproducible (RECORD_INCOMPLETE) because a missing component cannot be shown to be equal."""
    labels: list[ReproLabel] = []
    diffs: dict[str, Any] = {}
    if recorded.missing() or current.missing():
        labels.append(ReproLabel.RECORD_INCOMPLETE)
        diffs["missing"] = {"recorded": recorded.missing(), "current": current.missing()}
    pairs = (("code_hash", ReproLabel.CODE_CHANGED), ("data_hash", ReproLabel.DATA_CHANGED), ("config_hash", ReproLabel.CONFIG_CHANGED),
             ("seed", ReproLabel.SEED_CHANGED), ("memory_hash", ReproLabel.MEMORY_CHANGED), ("experiment_id", ReproLabel.EXPERIMENT_MISMATCH))
    for name, lab in pairs:
        a, b = getattr(recorded, name), getattr(current, name)
        if a != b:
            labels.append(lab)
            diffs[name] = (a, b)
    if recorded.code_mixed or current.code_mixed:
        labels.append(ReproLabel.CODE_EDITED_MID_RUN)
        diffs["code_mixed"] = tuple(sorted(set(recorded.code_mixed) | set(current.code_mixed)))
    wd = recorded.worker.diff(current.worker)
    if wd:
        hard = recorded.worker.hard_diff(current.worker)
        labels.append(ReproLabel.WORKER_CHANGED if hard else ReproLabel.WORKER_CHANGED_SOFT)
        diffs["worker"] = wd
    if check_artifacts and recorded.artifact_hashes and current.artifact_hashes:
        bad = sorted(k for k in set(recorded.artifact_hashes) | set(current.artifact_hashes)
                     if recorded.artifact_hashes.get(k) != current.artifact_hashes.get(k))
        if bad:
            labels.append(ReproLabel.RESULT_DIFFERS)
            diffs["artifacts"] = bad
    return ReproVerdict(tuple(labels), diffs, recorded.run_key, current.run_key)


# ---------------------------------------------------------------- code edited during a run (section 62 test 12)
def _norm_bytes(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


class CodeWatcher:
    """Fingerprint source files when a run starts; `check()` later says which changed. A worker that finishes after its code
    was edited holds a result nobody can reproduce - `assert_fresh` refuses it. Files are hashed with line endings
    normalised (a CRLF/LF round trip is not an edit)."""

    def __init__(self, files: Iterable[str | Path], root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        self.files = tuple(str(f) for f in files)
        if not self.files:
            raise ValueError("CodeWatcher needs at least one file to watch")
        self.started = self._snapshot()

    def _path(self, f: str) -> Path:
        p = Path(f)
        return p if p.is_absolute() or self.root is None else self.root / p

    def _snapshot(self) -> dict[str, str]:
        from engine.repro import artifact_hash
        out = {}
        for f in self.files:
            p = self._path(f)
            out[f] = artifact_hash(_norm_bytes(p))[:24] if p.exists() else "<missing>"
        return out

    def check(self) -> list[str]:
        now = self._snapshot()
        return sorted(f for f in self.files if now[f] != self.started[f])

    def digest(self) -> str:
        return stable_hash(self.started)

    def assert_fresh(self, what: str = "result") -> None:
        changed = self.check()
        if changed:
            raise FirewallBreach(f"{what} is stale: source changed during the run ({changed[:5]})")

    def accept(self, result: Mapping) -> tuple[bool, dict]:
        """(accepted, marked_result). A stale result is returned marked CODE_EDITED_MID_RUN and NOT accepted."""
        changed = self.check()
        verdict = ReproVerdict((ReproLabel.CODE_EDITED_MID_RUN,) if changed else (), {"changed": tuple(changed)} if changed else {})
        return (not changed), verdict.mark(result)


def assert_fresh_record(rec: ReproRecord, current_code_hash: str | None = None) -> None:
    """Refuse a result whose recorded code no longer matches the code on disk (engine.provenance.stale for recorded file
    lists, or a direct hash comparison when the caller supplies the current hash)."""
    if rec.code_mixed:
        raise FirewallBreach(f"stale result {rec.experiment_id}: files edited after the process started {list(rec.code_mixed)[:5]}")
    if current_code_hash is not None:
        if rec.code_hash != current_code_hash:
            raise FirewallBreach(f"stale result {rec.experiment_id}: code {rec.code_hash} != current {current_code_hash}")
        return
    from engine import provenance
    if not rec.code_files or provenance.stale({"code_files": list(rec.code_files), "code_hash": rec.code_hash, "code_mixed": []}):
        raise FirewallBreach(f"stale result {rec.experiment_id}: recorded code no longer matches the code on disk")


# ---------------------------------------------------------------- reproducing
@dataclasses.dataclass(frozen=True)
class ReproductionReport:
    verdict: ReproVerdict
    differing_artifacts: tuple[str, ...]
    diagnoses: tuple[Mapping, ...]
    deterministic: bool

    @property
    def reproduced(self) -> bool:
        return self.verdict.reproducible and not self.differing_artifacts and self.deterministic


def reproduce(fn: Callable[[dict, int], Mapping], cfg: Mapping, seed: int, recorded: ReproRecord, data: Any = None,
              memory_hash: str = "", check_determinism: bool = True, worker: WorkerConfig | None = None) -> tuple[ReproductionReport, dict]:
    """Run `fn(cfg, seed)` again and compare its artefact hashes with the recorded ones. Returns the report and the new
    artefacts. Differences are diagnosed to the first differing element (engine.repro.diagnose)."""
    from engine.repro import diagnose, hash_artifacts, run_twice
    import copy
    art = dict(fn(copy.deepcopy(dict(cfg)), seed))
    cur = make_record(cfg, seed, data if data is not None else recorded.data_hash, memory_hash or recorded.memory_hash,
                      code_hash=recorded.code_hash, artifacts=art, experiment_id=recorded.experiment_id, code_files=recorded.code_files,
                      worker=worker)
    hashes = hash_artifacts(art)
    bad = tuple(sorted(k for k in recorded.artifact_hashes if hashes.get(k) != recorded.artifact_hashes[k]))
    diagnoses = []
    for k in bad:
        if k in art:
            diagnoses.append({"artifact": k, "hash_recorded": recorded.artifact_hashes[k][:16], "hash_now": hashes[k][:16]})
    det = True
    if check_determinism:
        rep = run_twice(fn, dict(cfg), seed, required=())
        det = bool(rep["reproducible"])
        if not det:
            diagnoses.extend(rep["diagnoses"])
    v = compare(recorded, cur)
    if bad and ReproLabel.RESULT_DIFFERS not in v.labels:
        v = ReproVerdict(v.labels + (ReproLabel.RESULT_DIFFERS,), {**v.diffs, "artifacts": bad}, v.run_key_recorded, v.run_key_current)
    return ReproductionReport(v, bad, tuple(diagnoses), det), art


# ---------------------------------------------------------------- ledger
class ReproLedger:
    """Append-only JSON-lines record of runs. The same run key must always give the same artefact hashes; when it does
    not, an input is missing from the record."""

    def __init__(self, path):
        self.path = Path(path)

    def records(self) -> list[ReproRecord]:
        if not self.path.exists():
            return []
        return [ReproRecord.from_dict(json.loads(ln)) for ln in self.path.read_bytes().decode("utf-8").split("\n") if ln.strip()]

    def append(self, rec: ReproRecord) -> None:
        errs = rec.validate()
        if errs:
            raise FirewallBreach(f"refusing to ledger an incomplete record: {errs}")
        if any(r.experiment_id == rec.experiment_id and r.run_key != rec.run_key for r in self.records()):
            raise FirewallBreach(f"experiment id {rec.experiment_id!r} already used by a different run")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:                        # binary: text mode would add CRLF on Windows
            fh.write((json.dumps(rec.to_dict(), sort_keys=True) + "\n").encode("utf-8"))

    def find(self, experiment_id: str) -> ReproRecord | None:
        got = [r for r in self.records() if r.experiment_id == experiment_id]
        return got[-1] if got else None

    def conflicts(self) -> list[dict]:
        """Records sharing a run key but disagreeing on an artefact hash: the run is not determined by its recorded inputs."""
        by: dict[str, list[ReproRecord]] = {}
        for r in self.records():
            by.setdefault(r.run_key, []).append(r)
        out = []
        for key, rs in by.items():
            names = sorted({k for r in rs for k in r.artifact_hashes})
            bad = [n for n in names if len({r.artifact_hashes.get(n) for r in rs}) > 1]
            if len(rs) > 1 and bad:
                out.append({"run_key": key, "records": [r.experiment_id for r in rs], "artifacts": bad})
        return out


def label_counts(verdicts: Iterable[ReproVerdict]) -> dict[str, int]:
    """How many results carry each label: a run of results dominated by CODE_CHANGED means the pipeline was edited under them."""
    out: dict[str, int] = {}
    n = 0
    for v in verdicts:
        n += 1
        for lab in (v.labels or (ReproLabel.REPRODUCIBLE,)):
            out[str(lab)] = out.get(str(lab), 0) + 1
    out["_total"] = n
    return out


def markdown(rec: ReproRecord, verdict: ReproVerdict | None = None) -> str:
    """One-screen account of a run's identity for a report."""
    lines = [f"# Run {rec.experiment_id}", f"run key {rec.run_key}", "",
             f"- code {rec.code_hash}  data {rec.data_hash}  config {rec.config_hash}",
             f"- seed {rec.seed}  memory {rec.memory_hash or 'NONE'}",
             f"- worker python {rec.worker.python} on {rec.worker.platform}, {rec.worker.n_workers} worker(s)"]
    if rec.missing():
        lines.append(f"- INCOMPLETE: missing {', '.join(rec.missing())}")
    if verdict is not None:
        lines.append(f"- status: {verdict.status}")
        for k, v in verdict.diffs.items():
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------- multi-worker runs
def derive_worker_seeds(base_seed: int, n_workers: int) -> list[int]:
    """Child seeds derived deterministically from the base seed and the worker index (core.stable_hash, no clock, no global
    RNG). Distinct by construction; a collision is checked and raised rather than tolerated."""
    seeds = [int(stable_hash(["worker", int(base_seed), i], 15), 16) for i in range(int(n_workers))]
    if len(set(seeds)) != len(seeds):
        raise FirewallBreach("derived worker seeds collide")
    return seeds


def check_manifest_consistency(records: Sequence[ReproRecord], shared: Sequence[str] = ("code_hash", "data_hash", "config_hash", "memory_hash"),
                               base_seed: int | None = None) -> list[str]:
    """One experiment run by several workers: the workers must share code, data, config and memory (else the pooled result
    mixes different experiments), use DISTINCT seeds, and - when a base seed is given - exactly the seeds it derives."""
    problems: list[str] = []
    if not records:
        return ["no worker records: nothing shows the workers agreed"]
    for c in shared:
        vals = {getattr(r, c) for r in records}
        if len(vals) > 1:
            problems.append(f"workers disagree on {c}: {sorted(map(str, vals))[:4]}")
    seeds = [r.seed for r in records]
    if len(set(seeds)) != len(seeds):
        problems.append("two workers share a seed (their draws are identical, not independent)")
    if base_seed is not None and set(seeds) != set(derive_worker_seeds(base_seed, len(records))):
        problems.append("worker seeds are not the ones derived from the base seed")
    wf = {r.worker.fingerprint() for r in records}
    if len(wf) > 1:
        problems.append(f"{len(wf)} different worker environments in one run")
    for r in records:
        problems.extend(f"worker {r.experiment_id}: {e}" for e in r.validate())
    return problems


def stale_results(records: Iterable[ReproRecord], current_code_hash: str) -> list[str]:
    """Experiment ids whose recorded code differs from the code now (or that were edited mid-run): results to re-run, not to trust."""
    return sorted(r.experiment_id for r in records if r.code_hash != current_code_hash or r.code_mixed)


def record_from_stamp(stamp: Mapping, experiment_id: str, memory_hash: str = "", worker: WorkerConfig | None = None,
                      artifacts: Mapping[str, Any] | None = None) -> ReproRecord:
    """Adapt an engine.provenance.stamp() dict (code_hash, config_hash, data_snapshot, seed) to a ReproRecord."""
    from engine.repro import hash_artifacts
    return ReproRecord(experiment_id, str(stamp.get("code_hash") or ""), str(stamp.get("data_snapshot") or ""),
                       str(stamp.get("config_hash") or ""), stamp.get("seed"), worker or capture_worker(), memory_hash,
                       hash_artifacts(dict(artifacts)) if artifacts else {}, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       tuple(stamp.get("code_files") or ()), tuple(stamp.get("code_mixed") or ()))


def global_rng_use(fn: Callable, cfg: Mapping, seed: int) -> list[str]:
    """Which global generators the experiment consumed (engine.repro.uses_global_rng): any use makes the seed a lie."""
    from engine.repro import uses_global_rng
    return uses_global_rng(fn, dict(cfg), seed)


# ---------------------------------------------------------------- hidden inputs
class Perturbation:
    """Something a reproducible experiment must be indifferent to: an environment variable, the working directory, a
    global RNG state, the order of dict keys in its config. `apply()` is a context manager."""

    def __init__(self, name: str, enter: Callable[[], Any], leave: Callable[[Any], None]):
        self.name, self._enter, self._leave = name, enter, leave

    def __enter__(self):
        self._tok = self._enter()
        return self

    def __exit__(self, *exc):
        self._leave(self._tok)
        return False


def env_perturbation(key: str, value: str) -> Perturbation:
    def enter():
        old = os.environ.get(key)
        os.environ[key] = value
        return old

    def leave(old):
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
    return Perturbation(f"env:{key}={value}", enter, leave)


def cwd_perturbation(path: str | Path) -> Perturbation:
    def enter():
        old = os.getcwd()
        os.chdir(str(path))
        return old

    def leave(old):
        os.chdir(old)
    return Perturbation(f"cwd:{path}", enter, leave)


def global_rng_perturbation(seed: int) -> Perturbation:
    import random
    import numpy as np

    def enter():
        st = (random.getstate(), np.random.get_state())
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        return st

    def leave(st):
        random.setstate(st[0])
        np.random.set_state(st[1])
    return Perturbation(f"global-rng:{seed}", enter, leave)


def hidden_input_probe(fn: Callable[[dict, int], Mapping], cfg: Mapping, seed: int, perturbations: Sequence[Perturbation]) -> dict:
    """Run the experiment once clean and once under each perturbation; any artefact that moves reveals an input that the
    record does not capture (a working directory, an environment variable, global RNG state). Returns {perturbation: [names]}."""
    import copy
    from engine.repro import hash_artifacts
    base = hash_artifacts(dict(fn(copy.deepcopy(dict(cfg)), seed)))
    out: dict[str, list[str]] = {}
    for p in perturbations:
        with p:
            got = hash_artifacts(dict(fn(copy.deepcopy(dict(cfg)), seed)))
        moved = sorted(k for k in set(base) | set(got) if base.get(k) != got.get(k))
        if moved:
            out[p.name] = moved
    return out


def config_order_invariant(fn: Callable[[dict, int], Mapping], cfg: Mapping, seed: int) -> bool:
    """The result must not depend on the insertion order of config keys (json-sorted hashing hides such a dependence)."""
    from engine.repro import hash_artifacts
    a = hash_artifacts(dict(fn(dict(cfg), seed)))
    b = hash_artifacts(dict(fn(dict(reversed(list(cfg.items()))), seed)))
    return a == b


def diff_records(a: ReproRecord, b: ReproRecord) -> dict[str, tuple[Any, Any]]:
    """Field-level difference between two records (worker shown as its own diff), for the report."""
    out: dict[str, tuple[Any, Any]] = {}
    for c in COMPONENTS:
        if c == "worker":
            wd = a.worker.diff(b.worker)
            if wd:
                out["worker"] = wd
        elif getattr(a, c) != getattr(b, c):
            out[c] = (getattr(a, c), getattr(b, c))
    return out


# ---------------------------------------------------------------- manifests and drift
@dataclasses.dataclass(frozen=True)
class ExperimentManifest:
    """The full reproducibility statement of one experiment: every worker's record plus the pooled result hashes, sealed with
    a digest so the manifest itself cannot be edited unnoticed."""
    experiment_id: str
    records: tuple[ReproRecord, ...]
    pooled: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def digest(self) -> str:
        return stable_hash([self.experiment_id, [r.to_dict() for r in self.records], dict(self.pooled)], 32)

    def problems(self, base_seed: int | None = None) -> list[str]:
        out = check_manifest_consistency(self.records, base_seed=base_seed)
        if any(r.experiment_id != self.experiment_id and not r.experiment_id.startswith(self.experiment_id) for r in self.records):
            out.append("a worker record belongs to a different experiment")
        return out

    def save(self, path) -> str:
        d = {"experiment_id": self.experiment_id, "records": [r.to_dict() for r in self.records], "pooled": dict(self.pooled), "digest": self.digest()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(json.dumps(d, sort_keys=True, indent=1).encode("utf-8"))            # binary: no CRLF rewriting on Windows
        return d["digest"]

    @classmethod
    def load(cls, path) -> "ExperimentManifest":
        d = json.loads(Path(path).read_bytes().decode("utf-8"))
        m = cls(d["experiment_id"], tuple(ReproRecord.from_dict(r) for r in d["records"]), dict(d.get("pooled") or {}))
        if m.digest() != d.get("digest"):
            raise FirewallBreach(f"manifest {path} was edited: its digest no longer matches")
        return m


def compare_many(recorded: ReproRecord, currents: Iterable[ReproRecord]) -> dict[str, ReproVerdict]:
    """One recorded run against several later ones (different machines, days, branches): a verdict per candidate, keyed by its id."""
    return {c.experiment_id: compare(recorded, c) for c in currents}


def environment_drift(records: Iterable[ReproRecord]) -> list[dict]:
    """Walk records in creation order and report each change of worker environment (interpreter, library or hash seed).
    Results straddling a drift may differ numerically for no reason in the learning logic."""
    recs = sorted(records, key=lambda r: r.created_real)
    out, prev = [], None
    for r in recs:
        if prev is not None:
            d = prev.worker.diff(r.worker)
            if d:
                out.append({"from": prev.experiment_id, "to": r.experiment_id, "at": r.created_real, "changed": sorted(d), "hard": bool(prev.worker.hard_diff(r.worker))})
        prev = r
    return out


def worker_summary(w: WorkerConfig) -> str:
    """One-line description of a worker environment for logs."""
    pk = ",".join(f"{k}={v}" for k, v in sorted(w.packages.items()))
    return f"py{w.python} {w.platform} x{w.n_workers} [{pk}] hashseed={w.hashseed or 'random'}"


# ---------------------------------------------------------------- explaining a difference, saving the environment
def explain_difference(name: str, a: Any, b: Any) -> dict:
    """Why do two artefacts differ? Names the first differing element and classifies the cause (ordering, float noise, shape,
    NaN pattern, dtype, missing key) via engine.repro.diagnose. Equal artefacts return {'same': True}."""
    from engine.repro import artifact_hash, diagnose
    if artifact_hash(a) == artifact_hash(b):
        return {"artifact": name, "same": True}
    return {"artifact": name, "same": False, **diagnose(name, a, b)}


def save_environment(path, worker: WorkerConfig | None = None) -> str:
    """Write the worker configuration next to a result so the environment travels with it. Returns the fingerprint."""
    w = worker or capture_worker()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps({"fingerprint": w.fingerprint(), **dataclasses.asdict(w)}, sort_keys=True, indent=1).encode("utf-8"))
    return w.fingerprint()


def load_environment(path) -> WorkerConfig:
    """Read a saved worker configuration; a file whose content no longer matches its fingerprint was edited and is refused."""
    d = json.loads(Path(path).read_bytes().decode("utf-8"))
    w = WorkerConfig(d["python"], d["platform"], dict(d["packages"]), int(d["n_workers"]), dict(d["threads"]), d["hashseed"])
    if w.fingerprint() != d.get("fingerprint"):
        raise FirewallBreach(f"environment file {path} does not match its fingerprint (edited)")
    return w


# ---------------------------------------------------------------- sealing a result to its record
def seal_result(result: Mapping, rec: ReproRecord) -> dict:
    """Attach the reproducibility record to a result and seal both with a digest. The digest covers the result content and the
    run key, so a result cannot be re-labelled with a different run's record."""
    body = {k: v for k, v in dict(result).items() if k != "_seal"}
    from engine.repro import artifact_hash
    seal = stable_hash({"result": artifact_hash(body), "run_key": rec.run_key}, 32)
    return {**body, "_repro": rec.to_dict(), "_seal": seal}


def verify_sealed(sealed: Mapping) -> ReproRecord:
    """Check a sealed result: its content still hashes to the seal and its record's run key matches. Returns the record;
    raises FirewallBreach if the result or its record was altered."""
    from engine.repro import artifact_hash
    if "_seal" not in sealed or "_repro" not in sealed:
        raise FirewallBreach("result is not sealed: it carries no reproducibility record")
    rec = ReproRecord.from_dict(sealed["_repro"])
    body = {k: v for k, v in sealed.items() if k not in ("_seal", "_repro")}
    if stable_hash({"result": artifact_hash(body), "run_key": rec.run_key}, 32) != sealed["_seal"]:
        raise FirewallBreach("sealed result was altered after it was sealed")
    return rec


def by_label(verdicts: Mapping[str, ReproVerdict]) -> dict[str, list[str]]:
    """Invert a {experiment_id: verdict} map: label -> experiment ids carrying it (for triage after a code or data change)."""
    out: dict[str, list[str]] = {}
    for eid, v in verdicts.items():
        for lab in (v.labels or (ReproLabel.REPRODUCIBLE,)):
            out.setdefault(str(lab), []).append(eid)
    return {k: sorted(v) for k, v in sorted(out.items())}


def assert_bitwise_reproducible(fn: Callable[[dict, int], Mapping], cfg: Mapping, seed: int) -> None:
    """Run the experiment twice and require bit-identical artefacts. On a difference raise FirewallBreach naming the artefact and
    the first differing element (engine.repro.run_twice + diagnose), so a nondeterministic learner is stopped where it starts."""
    from engine.repro import run_twice
    rep = run_twice(fn, dict(cfg), seed, required=())
    if not rep["reproducible"]:
        detail = "; ".join(f"{d['name']} at {d['first_difference']}" for d in rep["diagnoses"]) or "; ".join(rep["problems"])
        raise FirewallBreach(f"experiment is not bitwise reproducible: {detail}")


def require_labelled(result: Mapping) -> str:
    """Refuse a result that carries no reproducibility label: every learning result must say whether it can be reproduced from
    its record. Returns the status string; NOT_REPRODUCIBLE results are allowed through (they are marked, not dropped)."""
    block = result.get("repro") if isinstance(result, Mapping) else None
    if not isinstance(block, Mapping) or "status" not in block:
        raise FirewallBreach("result carries no reproducibility label (mark it with ReproVerdict.mark)")
    return str(block["status"])


def hard_labels(verdict: ReproVerdict) -> tuple[ReproLabel, ...]:
    """The labels that make a result not reproducible (soft notes excluded), in the order found."""
    return tuple(lab for lab in verdict.labels if lab in HARD_LABELS)


def is_reproduced(verdict: ReproVerdict) -> bool:
    return not hard_labels(verdict)
